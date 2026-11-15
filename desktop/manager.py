"""RVC Voice desktop app for Windows: everything on one PC.

The .exe built from this file (see build.ps1) opens one window that
  1. asks for the microphone and headphones,
  2. installs what the voice engine needs into %LOCALAPPDATA%\\RVCVoice: uv, Python 3.12.3, PyTorch for the
     PC's NVIDIA card, the voice-conversion packages, the VB-Cable virtual audio cable and a voice,
  3. starts the engine (server/server.py, which also reads the mic and plays into the cable) and shows its
     /app page.
Every step checks what is already there, so later launches go straight to step 3.

The window is a web view on a tiny local HTTP server (ui/setup.html + /api/*); if the web view cannot start,
the same page opens in the browser and a small control window keeps the app alive.

Voices: if the .exe sits inside a checkout of the repo, that checkout's models/ folder is used; otherwise
voices live in %LOCALAPPDATA%\\RVCVoice\\models.
"""
import argparse
import collections
import contextlib
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

APP_NAME = "RVC Voice"
VERSION = "1.0.0"
LINKEDIN_URL = "https://in.linkedin.com/in/sapandeep-singh"
VBCABLE_SITE = "https://vb-audio.com/Cable/"

IS_WINDOWS = sys.platform == "win32"
FROZEN = getattr(sys, "frozen", False)
BUNDLE = Path(sys._MEIPASS) if FROZEN else Path(__file__).resolve().parent.parent   # server/, setup/, config.json
EXE_DIR = Path(sys.executable).resolve().parent if FROZEN else BUNDLE
HOME = Path(os.environ["LOCALAPPDATA"]) / "RVCVoice" if IS_WINDOWS else Path.home() / ".rvcvoice"
NO_WINDOW = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0

sys.path[:0] = [str(BUNDLE / "server"), str(BUNDLE / "setup")]
import audio_devices  # noqa: E402  (server/audio_devices.py: the same device names the engine uses)
import download_models  # noqa: E402  (setup/download_models.py)

# Pinned downloads, each checked against its SHA-256
UV_VERSION = "0.13.0"
UV_URL = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-x86_64-pc-windows-msvc.zip"
UV_SHA256 = "088962f9e7b7bd9ea740c04c650b2a21c8928c345bd99ac24350dc924dba656c"
VBCABLE_URL = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack46.zip"
VBCABLE_SHA256 = "bc44e8974191a5a2427d6c979e2dab08dbb09925f3f435dfa2f0ba3e45b6eb90"

PYTHON_VERSION = "3.12.3"   # voiceconversion 0.0.8 installs only on exactly this version
PACKAGES = ["voiceconversion==0.0.8", "fastapi", "uvicorn[standard]", "websockets", "numpy<2", "pydantic"]
MIN_FREE_GB = 12
BASE_FILES_GB = 1.8         # pitch and content models the engine downloads on its first start

STEPS = [
    ("pc", "Check this PC"),
    ("runtime", "Get Python"),
    ("packages", "Download the AI software"),
    ("cable", "Install the virtual audio cable (VB-Cable)"),
    ("voices", "Download the voice"),
    ("engine", "Start the voice engine"),
]

log = logging.getLogger("RVC-Desktop")


class SetupError(RuntimeError):
    """A setup step failed; the message is shown to the user as is."""


def detect_gpu() -> dict | None:
    """The first NVIDIA card and its driver version, or None when there is none."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, creationflags=NO_WINDOW,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    line = out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else ""
    if "," not in line:
        return None
    name, driver = (part.strip() for part in line.split(",", 1))
    return {"name": name, "driver": driver}


def torch_backend(gpu: dict | None) -> str:
    """Which PyTorch build to install. RTX 50-series cards need the CUDA 12.8 build; that build dropped
    GTX 10-series and older, which the CUDA 12.6 build still runs. No NVIDIA card: the small CPU build."""
    if os.environ.get("RVC_TORCH_BACKEND"):
        return os.environ["RVC_TORCH_BACKEND"]
    if gpu is None:
        return "cpu"
    return "cu128" if re.search(r"RTX\s*50\d\d|Blackwell", gpu["name"]) else "cu126"


def find_models_dir() -> Path:
    """The models/ folder of the repo checkout the .exe sits in (up to two levels up), else the app's own."""
    for base in (EXE_DIR, EXE_DIR.parent, EXE_DIR.parent.parent):
        if (base / "models").is_dir() and (base / "server" / "server.py").exists():
            return base / "models"
    return HOME / "models"


def folder_gb(path: Path) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            with contextlib.suppress(OSError):
                total += os.path.getsize(os.path.join(root, name))
    return total / 1e9


class App:
    def __init__(self):
        self.lock = threading.Lock()
        self.phase = "welcome"            # welcome -> working -> ready, or error
        self.steps = [{"id": i, "label": label, "status": "pending", "detail": ""} for i, label in STEPS]
        self.error = None
        self.engine_url = None
        self.log_lines = collections.deque(maxlen=400)
        self.gpu = None
        self.gpu_checked = False
        self.engine = None
        self.quit = threading.Event()
        self.models_dir = find_models_dir()
        self.venv_python = HOME / "venv" / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
        if os.environ.get("RVC_DEV_VENV"):   # development: reuse an existing environment, skip the installs
            self.venv_python = Path(os.environ["RVC_DEV_VENV"]) / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
        self.audio_settings = HOME / "audio_settings.json"
        self.state_file = HOME / "state.json"

    # ---- state shown in the window ----------------------------------------------------------------
    def note(self, line: str):
        log.info(line)
        with self.lock:
            self.log_lines.append(line)

    def set_step(self, step_id: str, status: str | None = None, detail: str | None = None):
        with self.lock:
            step = next(s for s in self.steps if s["id"] == step_id)
            if status is not None:
                step["status"] = status
            if detail is not None:
                step["detail"] = detail

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "version": VERSION,
                "phase": self.phase,
                "steps": [dict(s) for s in self.steps],
                "error": self.error,
                "engine_url": self.engine_url,
                "log": list(self.log_lines)[-120:],
                "gpu": self.gpu,
                "gpu_checked": self.gpu_checked,
                "models_dir": str(self.models_dir),
                "home": str(HOME),
            }

    def check_gpu(self):
        self.gpu = detect_gpu()
        self.gpu_checked = True

    def is_set_up(self) -> bool:
        return self.state_file.exists() and self.audio_settings.exists()

    # ---- setup ------------------------------------------------------------------------------------
    def start_setup(self, devices: dict | None = None):
        with self.lock:
            if self.phase == "working":
                return
            self.phase, self.error = "working", None
            for step in self.steps:
                step["status"], step["detail"] = "pending", ""
        if devices is not None:
            HOME.mkdir(parents=True, exist_ok=True)
            with open(self.audio_settings, "w", encoding="utf-8") as f:
                # "output" stays unset: the engine sends the voice to the virtual cable it finds
                json.dump({"input": devices["input"], "output": None, "monitor": devices["monitor"], "gain": 1.0}, f, indent=2)
        threading.Thread(target=self._run_setup, name="setup", daemon=True).start()

    def _run_setup(self):
        current = None
        try:
            for step_id, _label in STEPS:
                current = step_id
                self.set_step(step_id, "running")
                getattr(self, f"_step_{step_id}")()
                with self.lock:
                    step = next(s for s in self.steps if s["id"] == step_id)
                    if step["status"] == "running":
                        step["status"] = "done"
            self.state_file.write_text(json.dumps({"version": VERSION}), encoding="utf-8")
            with self.lock:
                self.phase = "ready"
        except Exception as e:
            if not isinstance(e, SetupError):
                log.exception("Setup step failed")
            message = str(e) if isinstance(e, SetupError) else f"Something unexpected went wrong: {e}"
            self.note(f"FAILED ({current}): {message}")
            self.set_step(current, "error")
            with self.lock:
                self.phase, self.error = "error", message

    def run_logged(self, cmd: list, env: dict | None = None, timeout: float | None = None) -> None:
        """Run a command, copying its output into the details log; SetupError if it fails."""
        self.note("> " + " ".join(str(c) for c in cmd))
        proc = subprocess.Popen(
            [str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace", env=env, creationflags=NO_WINDOW,
        )
        timer = threading.Timer(timeout, proc.kill) if timeout else None
        if timer:
            timer.start()
        tail = collections.deque(maxlen=12)
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                tail.append(line)
                self.note(line)
        code = proc.wait()
        if timer:
            timer.cancel()
        if code != 0:
            raise SetupError(f"\"{Path(str(cmd[0])).name}\" failed (code {code}):\n" + "\n".join(tail))

    def _step_pc(self):
        HOME.mkdir(parents=True, exist_ok=True)
        if not self.gpu_checked:
            self.check_gpu()
        free_gb = shutil.disk_usage(HOME).free / 1e9
        if not self.venv_python.exists() and free_gb < MIN_FREE_GB:
            raise SetupError(f"Not enough free disk space: {free_gb:.0f} GB free on the drive with {HOME}, "
                             f"about {MIN_FREE_GB} GB needed. Free up some space and try again.")
        if self.gpu is None:
            self.set_step("pc", "warn", "No NVIDIA graphics card found. The voice will run on the processor, "
                                        "which is usually too slow for live use.")
        else:
            self.set_step("pc", detail=f"{self.gpu['name']} (driver {self.gpu['driver']})")
        self.note(f"GPU: {self.gpu}; PyTorch build: {torch_backend(self.gpu)}; free disk {free_gb:.0f} GB; voices in {self.models_dir}")

    def uv_env(self) -> dict:
        # Keep uv's Python downloads and cache inside the app folder, so deleting it removes everything
        return {**os.environ, "UV_PYTHON_INSTALL_DIR": str(HOME / "python"), "UV_CACHE_DIR": str(HOME / "uv-cache"),
                "UV_NO_PROGRESS": "1"}

    def _step_runtime(self):
        if os.environ.get("RVC_DEV_VENV"):
            self.set_step("runtime", detail="Using the development environment")
            return
        if not IS_WINDOWS:
            raise SetupError("The installer runs on Windows only.")
        uv = HOME / "bin" / "uv.exe"
        if not uv.exists():
            uv.parent.mkdir(parents=True, exist_ok=True)
            archive = HOME / "bin" / "uv.zip"
            download_models.fetch(UV_URL, archive, UV_SHA256, lambda line: self.set_step("runtime", detail=line.strip()))
            with zipfile.ZipFile(archive) as z:
                z.extract("uv.exe", uv.parent)
            archive.unlink()
        if not self.venv_python.exists():
            self.set_step("runtime", detail=f"Downloading Python {PYTHON_VERSION}…")
            self.run_logged([uv, "venv", "--python", PYTHON_VERSION, HOME / "venv"], env=self.uv_env())
        self.set_step("runtime", detail=f"Python {PYTHON_VERSION}")

    def _step_packages(self):
        if os.environ.get("RVC_DEV_VENV"):
            self.set_step("packages", detail="Using the development environment")
            return
        backend = torch_backend(self.gpu)
        marker = HOME / "venv" / "rvc-packages.txt"
        wanted = " ".join(PACKAGES) + f" | torch {backend}"
        if not (marker.exists() and marker.read_text(encoding="utf-8") == wanted):
            # One resolution with the right PyTorch index: installing torch separately lets the next
            # install replace it with the CPU build from PyPI.
            done = threading.Event()

            def show_size():
                while not done.wait(3):
                    self.set_step("packages", detail=f"{folder_gb(HOME / 'uv-cache') + folder_gb(HOME / 'venv'):.1f} GB so far "
                                                     "(about 3 to 6 GB in total). This is the slow step.")
            threading.Thread(target=show_size, daemon=True).start()
            try:
                self.run_logged([HOME / "bin" / "uv.exe", "pip", "install", "--python", self.venv_python,
                                 "--torch-backend", backend, *PACKAGES], env=self.uv_env())
            finally:
                done.set()
            marker.write_text(wanted, encoding="utf-8")
        check = subprocess.run(
            [str(self.venv_python), "-c",
             "import json, torch; ok = torch.cuda.is_available(); "
             "print(json.dumps({'cuda': ok, 'name': torch.cuda.get_device_name(0) if ok else None, 'torch': torch.__version__}))"],
            capture_output=True, text=True, creationflags=NO_WINDOW,
        )
        if check.returncode != 0:
            raise SetupError("PyTorch was installed but does not start:\n" + check.stderr[-800:])
        result = json.loads(check.stdout.strip().splitlines()[-1])
        self.note(f"PyTorch check: {result}")
        if result["cuda"]:
            self.set_step("packages", detail=f"PyTorch {result['torch']} on {result['name']}")
        elif self.gpu is not None:
            self.set_step("packages", "warn", f"PyTorch cannot use the {self.gpu['name']} (driver {self.gpu['driver']}). "
                                              "Update the NVIDIA driver from nvidia.com/drivers, then reopen the app.")
        else:
            self.set_step("packages", detail=f"PyTorch {result['torch']} (processor only)")

    def _cable_name(self) -> str | None:
        audio_devices.refresh()
        return audio_devices.list_devices()["cable"]

    def _step_cable(self):
        if self._cable_name():
            self.set_step("cable", detail="Already installed")
            return
        if not IS_WINDOWS:
            self.set_step("cable", "warn", "Skipped: VB-Cable is installed on Windows only")
            return
        folder = HOME / "vbcable"
        folder.mkdir(parents=True, exist_ok=True)
        archive = folder / "pack.zip"
        manual = f"Install VB-Cable yourself from {VBCABLE_SITE} (run VBCABLE_Setup_x64.exe as administrator), then press Try again."
        try:
            download_models.fetch(VBCABLE_URL, archive, VBCABLE_SHA256, lambda line: self.set_step("cable", detail=line.strip()))
        except (OSError, download_models.ChecksumError) as e:
            raise SetupError(f"Could not download VB-Cable ({e}). {manual}") from e
        with zipfile.ZipFile(archive) as z:
            z.extractall(folder)
        self.set_step("cable", detail="Installing… if Windows asks to install the driver, click Install.")
        # -i install, -h without VB-Audio's own window; Windows may still ask to trust the driver once
        self.run_logged([folder / "VBCABLE_Setup_x64.exe", "-i", "-h"], timeout=300)
        for _ in range(30):
            if self._cable_name():
                self.set_step("cable", detail="Installed")
                return
            time.sleep(1)
        self.set_step("cable", "warn", "Installed, but Windows needs a restart before the cable shows up. "
                                       "Restart the PC, then open the app again.")

    def _step_voices(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        try:
            download_models.download(self.models_dir, lambda line: (self.note(line), self.set_step("voices", detail=line.strip())))
        except (OSError, download_models.ChecksumError) as e:
            raise SetupError(f"Could not download the voice: {e}. Check your internet connection and try again.") from e
        voices = sorted(d.name for d in self.models_dir.iterdir() if d.is_dir())
        self.set_step("voices", detail=", ".join(voices))

    def _step_engine(self):
        self.stop_engine()
        code_dir = HOME / "app"
        shutil.copytree(BUNDLE / "server", code_dir / "server", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.sh", "requirements*"))
        with open(BUNDLE / "config.json", "r", encoding="utf-8") as f:
            server_cfg = json.load(f)["server"]
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        pretrain = HOME / "pretrain"
        server_cfg.update(host="127.0.0.1", port=port, model_dir=str(self.models_dir), pretrain_dir=str(pretrain))
        config_path = HOME / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"server": server_cfg}, f, indent=2)

        (HOME / "logs").mkdir(exist_ok=True)
        engine_log = open(HOME / "logs" / "engine.log", "w", encoding="utf-8")
        env = {**os.environ, "RVC_CONFIG": str(config_path), "RVC_EXIT_WITH_PARENT": "1", "PYTHONUNBUFFERED": "1",
               "PYTHONIOENCODING": "utf-8"}
        self.engine = subprocess.Popen(
            [str(self.venv_python), "-m", "uvicorn", "--app-dir", str(code_dir / "server"), "server:app",
             "--host", "127.0.0.1", "--port", str(port)],
            stdin=subprocess.PIPE, stdout=engine_log, stderr=subprocess.STDOUT, env=env, cwd=str(code_dir),
            creationflags=NO_WINDOW,
        )
        url = f"http://127.0.0.1:{port}"
        self.note(f"Engine starting on {url} (log: {HOME / 'logs' / 'engine.log'})")
        deadline = time.monotonic() + 45 * 60   # the first start downloads the base files
        while time.monotonic() < deadline:
            if self.engine.poll() is not None:
                raise SetupError("The voice engine stopped while starting:\n" + self._engine_log_tail())
            with contextlib.suppress(OSError, ValueError):
                with urllib.request.urlopen(url + "/health", timeout=2) as r:
                    health = json.load(r)
                if health.get("init_error"):
                    raise SetupError(f"The voice engine could not load the voice: {health['init_error']}")
                if health.get("model"):
                    self.set_step("engine", detail=f"{health['model']} on {'the graphics card' if health['device'] == 'cuda' else health['device']}")
                    with self.lock:
                        self.engine_url = url
                    return
            have = folder_gb(pretrain) if pretrain.exists() else 0.0
            self.set_step("engine", detail="Loading…" if have >= BASE_FILES_GB else
                          f"First start: downloading base files, {have:.1f} of about {BASE_FILES_GB} GB")
            time.sleep(1.5)
        raise SetupError("The voice engine did not start in time:\n" + self._engine_log_tail())

    def _engine_log_tail(self) -> str:
        with contextlib.suppress(OSError):
            lines = (HOME / "logs" / "engine.log").read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-15:])
        return ""

    def stop_engine(self):
        proc, self.engine = self.engine, None
        if proc is None or proc.poll() is not None:
            return
        with contextlib.suppress(OSError):
            proc.stdin.close()      # the engine exits when this pipe closes
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()


class Handler(BaseHTTPRequestHandler):
    app: App = None

    def log_message(self, *args):
        pass

    def _send(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _local(self) -> bool:
        # Only this PC's own pages may talk to us (blocks other sites reaching 127.0.0.1 by a DNS trick)
        return self.headers.get("Host", "").split(":")[0] in ("127.0.0.1", "localhost")

    def do_GET(self):
        if not self._local():
            return self._json({"detail": "forbidden"}, 403)
        if self.path == "/":
            return self._send(200, (BUNDLE / "desktop" / "ui" / "setup.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/state":
            return self._json({**self.app.snapshot(), "set_up": self.app.is_set_up()})
        if self.path == "/api/devices":
            audio_devices.refresh()
            return self._json(audio_devices.list_devices())
        self._json({"detail": "not found"}, 404)

    def do_POST(self):
        if not self._local() or "application/json" not in self.headers.get("Content-Type", ""):
            return self._json({"detail": "forbidden"}, 403)
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/install":
            if not body.get("input") or not body.get("monitor"):
                return self._json({"detail": "Choose a microphone and headphones or speakers first."}, 400)
            self.app.start_setup({"input": body["input"], "monitor": body["monitor"]})
        elif self.path == "/api/retry":
            self.app.start_setup()
        elif self.path == "/api/open/linkedin":
            webbrowser.open(LINKEDIN_URL)
        elif self.path == "/api/open/vbcable":
            webbrowser.open(VBCABLE_SITE)
        elif self.path == "/api/quit":
            self.app.quit.set()
        else:
            return self._json({"detail": "not found"}, 404)
        self._json({"status": "ok"})


def show_window(url: str):
    """The app window. Blocks until the user closes it."""
    try:
        import webview
        webview.create_window(APP_NAME, url, width=660, height=980, min_size=(440, 640), background_color="#111418")
        # edgechromium = the WebView2 runtime that ships with Windows 10/11; never the old IE engine
        webview.start(gui="edgechromium" if IS_WINDOWS else None, storage_path=str(HOME / "webview"), private_mode=False)
        return
    except Exception:
        log.exception("The app window could not open; falling back to the browser")
    edge = Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft" / "Edge" / "Application" / "msedge.exe"
    if IS_WINDOWS and edge.exists():
        subprocess.Popen([str(edge), f"--app={url}"])
    else:
        webbrowser.open(url)
    import tkinter as tk
    root = tk.Tk()
    root.title(APP_NAME)
    tk.Label(root, text=f"{APP_NAME} is running in your browser.\nKeep this window open; closing it stops the voice changer.",
             padx=24, pady=16).pack()
    tk.Button(root, text="Open the app page", command=lambda: webbrowser.open(url)).pack(pady=4)
    tk.Button(root, text="Quit", command=root.destroy).pack(pady=(4, 16))
    root.mainloop()


def already_running() -> bool:
    if not IS_WINDOWS:
        return False
    import ctypes
    ctypes.windll.kernel32.CreateMutexW(None, False, "Local\\RVCVoiceDesktopApp")
    return ctypes.windll.kernel32.GetLastError() == 183   # ERROR_ALREADY_EXISTS


def selftest(out_path: str) -> int:
    """Checks a built .exe can find what it bundles and see audio devices (used by CI)."""
    result = {"version": VERSION, "frozen": FROZEN, "home": str(HOME), "models_dir": str(find_models_dir())}
    result["bundled"] = {name: (BUNDLE / name).exists() for name in (
        "server/server.py", "server/local_audio.py", "server/audio_devices.py", "server/app.html",
        "server/panel.html", "config.json", "desktop/ui/setup.html")}
    result["devices"] = audio_devices.list_devices()
    result["gpu"] = detect_gpu()
    try:
        import webview  # noqa: F401
        result["webview"] = True
    except Exception as e:
        result["webview"] = repr(e)
    Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if all(result["bundled"].values()) and result["webview"] is True else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--no-window", action="store_true", help="serve the setup API without a window (automated tests)")
    parser.add_argument("--port", type=int, default=0, help="port for the setup API (default: any free port)")
    parser.add_argument("--selftest", metavar="OUT_JSON", help="write a self-check report and exit")
    args = parser.parse_args()

    HOME.mkdir(parents=True, exist_ok=True)
    (HOME / "logs").mkdir(exist_ok=True)
    logging.basicConfig(filename=HOME / "logs" / "app.log", level=logging.INFO, encoding="utf-8",
                        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s")
    if args.selftest:
        return selftest(args.selftest)
    if already_running():
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, f"{APP_NAME} is already open.", APP_NAME, 0x40)
        return 0

    app = App()
    Handler.app = app
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, name="setup-api", daemon=True).start()
    threading.Thread(target=app.check_gpu, name="gpu-check", daemon=True).start()
    log.info(f"{APP_NAME} {VERSION} at {url}; data in {HOME}; voices in {app.models_dir}")
    if app.is_set_up():
        app.start_setup()   # already installed: every step is a quick check, then the engine starts
    try:
        if args.no_window:
            app.quit.wait()
        else:
            show_window(url)
    finally:
        app.stop_engine()
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
