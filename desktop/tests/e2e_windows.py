"""End-to-end test of the built desktop app on Windows (run by .github/workflows/windows-app.yml).

Drives dist\\RVCVoice.exe the way its window does, through the setup API:
  1. full install (uv, Python, PyTorch, packages, VB-Cable, voice, base files) with a WAV file as the microphone,
  2. the engine comes up and loads the voice,
  3. with VB-Cable present: switch on, and record "CABLE Output" to prove converted audio arrives,
  4. the window version of the app opens and reaches the app page (screenshots are kept),
  5. quitting stops the engine.
Needs: pip install sounddevice numpy pillow. Exit code 0 = all checks passed.
"""

import json
import math
import os
import struct
import subprocess
import sys
import time
import traceback
import urllib.request
import wave
from pathlib import Path

EXE = Path("dist/RVCVoice.exe").resolve()
OUT = Path("e2e-out").resolve()
API = "http://127.0.0.1:8765"
HOME = Path(os.environ["LOCALAPPDATA"]) / "RVCVoice"
failures = []


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.load(r)


def post(url, body=None):
    req = urllib.request.Request(
        url, data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def check(name, ok, info=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {info}", flush=True)
    if not ok:
        failures.append(name)


def make_voice_like_wav(path: Path, seconds=6.0):
    """A sung-vowel-like tone (harmonics, vibrato, syllable rhythm): enough for the model to produce voice."""
    rate, frames = 48000, []
    for n in range(int(seconds * rate)):
        t = n / rate
        f0 = 150 * (1 + 0.03 * math.sin(2 * math.pi * 5 * t))
        s = sum(math.sin(2 * math.pi * f0 * k * t) / k for k in range(1, 9))
        env = 0.5 * (1 - math.cos(2 * math.pi * 2.5 * t))  # 2.5 "syllables" a second
        frames.append(struct.pack("<h", int(max(-1, min(1, 0.25 * s * env)) * 32767)))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes(b"".join(frames))


def wait_api(seconds=60):
    for _ in range(seconds):
        try:
            return get(API + "/api/state")
        except OSError:
            time.sleep(1)
    raise SystemExit("The app's setup API did not come up")


def wait_setup(minutes):
    seen, deadline = {}, time.time() + minutes * 60
    while time.time() < deadline:
        state = get(API + "/api/state")
        for step in state["steps"]:
            key = (step["status"], step["detail"][:70])
            if seen.get(step["id"]) != key:
                seen[step["id"]] = key
                print(f"  [{step['status']:8}] {step['label']}: {step['detail']}", flush=True)
        if state["phase"] in ("ready", "error"):
            return state
        time.sleep(2)
    return get(API + "/api/state")


def screenshot(name):
    from PIL import ImageGrab

    ImageGrab.grab().save(OUT / name)


def main():
    OUT.mkdir(exist_ok=True)
    wav = OUT / "mic.wav"
    make_voice_like_wav(wav)

    selftest = OUT / "selftest.json"
    code = subprocess.run([str(EXE), "--selftest", str(selftest)], timeout=120).returncode
    print(selftest.read_text() if selftest.exists() else "(no selftest output)")
    check("exe self-test (bundled files, web view import)", code == 0)

    env = {**os.environ, "RVC_AUDIO_TEST": "1"}  # lets the engine accept "wav:<file>" as the microphone
    app = subprocess.Popen([str(EXE), "--no-window", "--port", "8765"], env=env)
    try:
        state = wait_api()
        check("starts on the welcome screen", state["phase"] == "welcome" and not state["set_up"], state["phase"])
        devices = get(API + "/api/devices")
        print("devices before setup:", devices)
        monitor = next((d["name"] for d in devices["outputs"] if d["name"] != devices["cable"]), "none (CI)")
        status, _ = post(API + "/api/install", {"input": f"wav:{wav}", "monitor": monitor})
        check("install accepted", status == 200)

        state = wait_setup(minutes=30)
        print("\n".join(state["log"][-40:]))
        if state["phase"] != "ready" and (HOME / "logs" / "engine.log").exists():
            print("---- engine.log ----")
            print((HOME / "logs" / "engine.log").read_text(encoding="utf-8", errors="replace")[-6000:])
        check("setup finished", state["phase"] == "ready", state.get("error") or "")
        if state["phase"] != "ready":
            return
        steps = {s["id"]: s for s in state["steps"]}
        check("VB-Cable installed by the app", steps["cable"]["status"] == "done", steps["cable"]["detail"])

        engine = state["engine_url"]
        health = get(engine + "/health")
        print("health:", health)
        check("voice loaded", bool(health["model"]) and not health["init_error"], str(health["model"]))
        check("app page served", urllib.request.urlopen(engine + "/app").status == 200)
        audio_devices = get(engine + "/audio/devices")
        print("engine sees:", audio_devices)

        try:
            audio_phase(engine, audio_devices, wav)
        except Exception:
            check("audio phase ran to the end", False, traceback.format_exc())

        post(API + "/api/quit")
        app.wait(timeout=30)
        time.sleep(3)
        try:
            get(engine + "/health")
            check("engine stops with the app", False, "still answering")
        except OSError:
            check("engine stops with the app", True)
        window_phase(env)
    finally:
        if app.poll() is None:
            app.kill()
        for name in ("app.log", "engine.log"):
            src = HOME / "logs" / name
            if src.exists():
                (OUT / name).write_text(src.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")


def audio_phase(engine, audio_devices, wav):
    if audio_devices["cable"]:
        import numpy as np
        import sounddevice as sd

        status, body = post(engine + "/audio/power", {"on": True})
        check("switch on", status == 200 and body.get("on") is True, str(body)[:300])
        time.sleep(6)  # first frames: buffers allocate, the model warms up
        # Record what a game would hear: the cable's other end
        api = next(i for i, a in enumerate(sd.query_hostapis()) if a["name"] == "Windows WASAPI")
        rec_dev = next(
            i
            for i, d in enumerate(sd.query_devices())
            if d["hostapi"] == api and d["max_input_channels"] > 0 and "CABLE Output" in d["name"]
        )
        rate = int(sd.query_devices(rec_dev)["default_samplerate"])
        rec_channels = min(2, sd.query_devices(rec_dev)["max_input_channels"])

        def record(seconds):
            data = sd.rec(
                int(seconds * rate),
                samplerate=rate,
                channels=rec_channels,
                device=rec_dev,
                dtype="float32",
                extra_settings=sd.WasapiSettings(auto_convert=True),
            )
            sd.wait()
            return data.mean(axis=1)

        rec = record(8)
        rms, peak = float(np.sqrt(np.mean(rec**2))), float(np.abs(rec).max())
        st = get(engine + "/audio/status")
        health = get(engine + "/health")
        print("status while on:", st)
        print(
            f"recorded from CABLE Output: rms {rms:.4f} peak {peak:.3f}; "
            f"processing {health['avg_inference_ms']} ms per {health['frame_samples'] / 48:.0f} ms frame; "
            f"underruns {st['underruns']}"
        )
        check("converted audio reaches CABLE Output", rms > 0.003, f"rms {rms:.4f}")
        check("no audio error while on", st["on"] and not st["error"], str(st["error"]))
        status, body = post(engine + "/audio/power", {"on": False})
        check("switch off", status == 200 and body.get("on") is False)

        # Self check through a real Windows microphone device. The machine's only devices are the
        # cable's two ends, so: play the WAV into the cable, let the engine record the cable's other
        # end as its microphone, and listen for its converted playback coming back through the cable.
        mic_name = next(d["name"] for d in audio_devices["inputs"] if "CABLE Output" in d["name"])
        status, body = post(engine + "/audio/settings", {"input": mic_name, "monitor": audio_devices["cable"]})
        check("choose devices", status == 200, str(body)[:200])
        with wave.open(str(wav)) as w:
            tone = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        out_dev = next(
            i
            for i, d in enumerate(sd.query_devices())
            if d["hostapi"] == api and d["max_output_channels"] > 0 and "CABLE Input" in d["name"]
        )
        sd.play(tone[: int(4.5 * 48000)], 48000, device=out_dev, extra_settings=sd.WasapiSettings(auto_convert=True))
        status, body = post(engine + "/audio/test", {"seconds": 4})
        check("self check starts recording", status == 200 and body.get("test") == "recording", str(body)[:200])
        mic_level, deadline = 0.0, time.time() + 30
        while time.time() < deadline:
            st = get(engine + "/audio/status")
            mic_level = max(mic_level, st["input_level"])
            if st["test"] != "recording":
                break
            time.sleep(0.3)
        sd.stop()
        check("engine hears the Windows microphone device", mic_level > 0.003, f"peak level {mic_level:.4f}")
        check("self check plays back", st["test"] == "playing", st["test"])
        rms = float(np.sqrt(np.mean(record(3) ** 2)))
        check("self-check playback is audible", rms > 0.003, f"rms {rms:.4f}")
        for _ in range(40):
            st = get(engine + "/audio/status")
            if st["test"] == "idle":
                break
            time.sleep(0.5)
        check("self check finishes", st["test"] == "idle" and not st["error"], str(st["error"]))
    else:
        check("VB-Cable visible to the engine", False, "no CABLE Input device")


def window_phase(env):
    # The real window: already set up, so it should go straight to the app page
    win = subprocess.Popen([str(EXE)], env=env)
    try:
        time.sleep(4)
        screenshot("window-1-starting.png")
        deadline, opened = time.time() + 240, False
        log_file = HOME / "logs" / "app.log"
        while time.time() < deadline and not opened:
            time.sleep(3)
            opened = (
                "Engine starting" in log_file.read_text(encoding="utf-8", errors="replace").split("RVC Voice")[-1]
                and (HOME / "logs" / "engine.log").exists()
                and "/app " in (HOME / "logs" / "engine.log").read_text(encoding="utf-8", errors="replace")
            )
        time.sleep(4)
        screenshot("window-2-app.png")
        check("window opens the app page", opened, "the engine served /app to the window")
        fallback = "falling back to the browser" in log_file.read_text(encoding="utf-8", errors="replace")
        check("embedded window (no browser fallback)", not fallback)
    finally:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(win.pid)], capture_output=True)


if __name__ == "__main__":
    main()
    print(f"\n{len(failures)} failed: {failures}" if failures else "\nAll checks passed")
    sys.exit(1 if failures else 0)
