"""Names of this computer's microphones and speakers, as the voice engine opens them.

Shared by the engine (local_audio.py) and the desktop app's setup window, so a device picked
during setup is found again by exactly the same name. Needs only sounddevice (no numpy).
"""
import sounddevice as sd

CABLE_KEYWORD = "CABLE Input"   # VB-Audio Virtual Cable: what we play into; apps record "CABLE Output"
PREFERRED_HOST_API = "Windows WASAPI"   # full device names and low latency; other systems use their default API


def refresh():
    """Re-scan devices (PortAudio caches the list; needed after a driver is installed or a device plugged in).
    Must not be called while a stream is open."""
    sd._terminate()
    sd._initialize()


def _host_api_index() -> int:
    apis = sd.query_hostapis()
    for i, api in enumerate(apis):
        if api["name"] == PREFERRED_HOST_API:
            return i
    return sd.default.hostapi


def _devices(kind: str) -> list[dict]:
    """Devices of one kind ("input" or "output") on the preferred host API, default device first."""
    api_index = _host_api_index()
    api = sd.query_hostapis(api_index)
    default_index = api[f"default_{kind}_device"]
    found = []
    for index, dev in enumerate(sd.query_devices()):
        if dev["hostapi"] != api_index or dev[f"max_{kind}_channels"] < 1:
            continue
        found.append({
            "index": index,
            "name": dev["name"],
            "channels": dev[f"max_{kind}_channels"],
            "default": index == default_index,
            "wasapi": api["name"] == PREFERRED_HOST_API,
        })
    found.sort(key=lambda d: not d["default"])
    return found


def list_devices() -> dict:
    inputs, outputs = _devices("input"), _devices("output")
    cable = next((d["name"] for d in outputs if CABLE_KEYWORD.lower() in d["name"].lower()), None)
    return {
        "inputs": [{"name": d["name"], "default": d["default"]} for d in inputs],
        "outputs": [{"name": d["name"], "default": d["default"]} for d in outputs],
        "cable": cable,
    }


def find_device(kind: str, name: str) -> dict | None:
    """The device with exactly this name, or None if it is not connected right now."""
    return next((d for d in _devices(kind) if d["name"] == name), None)
