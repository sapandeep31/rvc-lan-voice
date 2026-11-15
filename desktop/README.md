# RVC Voice desktop app (Windows, one PC)

`RVCVoice.exe` runs the whole voice changer on one Windows PC with an NVIDIA graphics card: no second
machine, no command line. Download it from the repo's [Releases](https://github.com/sapandeep31/rvc-lan-voice/releases).

## What it does

1. Asks for the microphone and the headphones or speakers.
2. Installs everything into `%LOCALAPPDATA%\RVCVoice` (nothing system-wide except the VB-Cable driver):
   uv, Python 3.12.3, PyTorch for the PC's NVIDIA card (CUDA 12.8 build for RTX 50-series, CUDA 12.6 build
   for older cards), the voice-conversion packages, VB-Cable, the Narrator voice and the engine's base files.
   About 6 GB of downloads, once.
3. Starts the voice engine and shows the app: an on/off switch, the voice panel, a self check and the
   device choice. While it is on, the new voice goes to **CABLE Output**, which Discord and games use
   as their microphone.

It asks for administrator rights at launch because installing the VB-Cable driver needs them.

Voices: if the .exe sits in a checkout of this repo (in the repo folder, or up to two folders below it),
it uses that checkout's `models/` folder. Otherwise voices live in `%LOCALAPPDATA%\RVCVoice\models`.
To remove everything, delete `%LOCALAPPDATA%\RVCVoice` and uninstall VB-Cable from Windows' Apps list.

## How it is built

| Part | File | Runs in |
|---|---|---|
| Setup window and installer | `desktop/manager.py`, `desktop/ui/setup.html` | the .exe (PyInstaller, pywebview) |
| Voice engine | `server/server.py` | the Python environment the app installs |
| Mic and cable audio | `server/local_audio.py`, `server/audio_devices.py` | the engine |
| App page | `server/app.html` (embeds `server/panel.html`) | the engine, at `/app` |

The .exe bundles `server/`, `config.json` and `setup/download_models.py`, copies the engine code to
`%LOCALAPPDATA%\RVCVoice\app` at launch and starts it with `RVC_CONFIG` pointing at its own `config.json`.
The engine stops when the app closes.

Build on Windows with Python 3.12, from the repo root:

```
powershell -ExecutionPolicy Bypass -File desktop\build.ps1
```

GitHub Actions (`.github/workflows/windows-app.yml`) builds the .exe on every change and runs
`desktop/tests/e2e_windows.py` against it: a full install on a clean Windows machine, VB-Cable, audio
converted into the cable, and the app window. The tested .exe is that run's `RVCVoice` artifact.

Logs: `%LOCALAPPDATA%\RVCVoice\logs\app.log` (setup) and `engine.log` (voice engine).
