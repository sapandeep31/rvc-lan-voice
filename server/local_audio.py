"""Local sound for running everything on one PC: microphone -> voice model -> virtual cable (and headphones).

The desktop app uses this instead of a separate client: the mic is read here, each frame is converted
on the server's model thread, and the result is played into VB-Cable's "CABLE Input" (games and Discord
record "CABLE Output") and, for the self check, into the user's headphones.

Each output has a small jitter buffer: frames arrive one whole frame at a time and a little unevenly
(model time varies), while the device drains continuously. Gaps (underruns) are counted; surplus delay
is cut out only during pauses in speech.
"""
import json
import logging
import os
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import audio_devices

logger = logging.getLogger("RVC-LocalAudio")

SAMPLE_RATE = 48000
LOCAL_STREAM_ID = 0                      # the server's WebSocket connections use 1, 2, ...
PREFILL = int(0.060 * SAMPLE_RATE)       # silence queued ahead of audio after a start or a gap
SAFETY = int(0.020 * SAMPLE_RATE)        # spare audio we aim to keep queued at the lowest point
TRIM_MARGIN = SAMPLE_RATE // 100         # tolerate 10 ms above that before trimming
TRIM_WINDOW_S = 5.0                      # the low point is taken over this long
HOLD_AFTER_GAP_S = 30.0                  # after a gap, keep the fresh prefill this long before trimming again
XFADE = SAMPLE_RATE // 200               # 5 ms crossfade when cutting
QUIET_RMS = 10 ** (-50 / 20)             # only cut where the audio is below -50 dBFS
FADE_IN = np.linspace(0.0, 1.0, XFADE, dtype=np.float32)
FADE_OUT = 1.0 - FADE_IN
MAX_BACKLOG_FRAMES = 3                   # mic audio waiting for the model beyond this is dropped
TEST_INPUT_PREFIX = "wav:"               # automated tests: stream a WAV file as the microphone (RVC_AUDIO_TEST=1)


class AudioError(RuntimeError):
    """A problem the user can fix (device missing, cable not installed); the message is shown in the app."""


def _extra_settings(device: dict):
    # Shared-mode WASAPI only opens at the device's own rate; let Windows convert to our 48 kHz
    return sd.WasapiSettings(auto_convert=True) if device["wasapi"] else None


class OutputLine:
    """Plays converted audio on one device through a small jitter buffer."""

    def __init__(self, device: dict):
        self.name = device["name"]
        self.lock = threading.Lock()
        self.buf = np.empty(0, dtype=np.float32)
        self.playing = False
        self.underruns = 0
        self.peak = 0.0
        self.low_point = None
        self.window_start = time.monotonic()
        self.hold_until = 0.0
        self.stream = sd.OutputStream(
            device=device["index"], samplerate=SAMPLE_RATE, channels=min(device["channels"], 2), dtype="float32",
            latency="low", callback=self._callback, extra_settings=_extra_settings(device),
        )
        self.stream.start()

    def push(self, y: np.ndarray):
        now = time.monotonic()
        with self.lock:
            if not self.playing:
                self.buf = np.concatenate([np.zeros(PREFILL, dtype=np.float32), y])
                self.playing = True
                self.low_point, self.window_start = None, now
                return
            level = len(self.buf)
            self.low_point = level if self.low_point is None else min(self.low_point, level)
            self.buf = np.concatenate([self.buf, y])
            ceiling = PREFILL + 4 * len(y)
            if len(self.buf) > ceiling:
                self._cut(len(self.buf) - ceiling)
            elif now - self.window_start >= TRIM_WINDOW_S:
                self._trim_surplus(now)

    def _trim_surplus(self, now: float):
        """If even the lowest queue level of the window was above the safety margin, that surplus is pure
        delay (clock drift between mic and output, or a start-up burst): cut it out, in a pause."""
        excess = self.low_point - SAFETY - TRIM_MARGIN
        k = min(excess, len(self.buf) - XFADE)
        if k > 0 and now >= self.hold_until:
            if np.sqrt(np.mean(self.buf[:k + XFADE] ** 2)) > QUIET_RMS:
                return   # speech: keep the window and retry on the next frame
            self._cut(k)
        self.low_point, self.window_start = None, now

    def _cut(self, k: int):
        buf = self.buf
        if len(buf) < k + XFADE:
            self.buf = buf[k:]
            return
        head = buf[:XFADE] * FADE_OUT + buf[k:k + XFADE] * FADE_IN
        self.buf = np.concatenate([head, buf[k + XFADE:]]).astype(np.float32)

    def _callback(self, outdata, frames, time_info, status):
        with self.lock:
            n = min(frames, len(self.buf))
            outdata[:n] = self.buf[:n, None]
            outdata[n:] = 0.0
            if n:
                self.peak = max(self.peak, float(np.abs(self.buf[:n]).max()))
            self.buf = self.buf[n:]
            if n < frames and self.playing:
                self.playing = False
                self.underruns += 1
                self.hold_until = time.monotonic() + HOLD_AFTER_GAP_S

    def take_peak(self) -> float:
        with self.lock:
            peak, self.peak = self.peak, 0.0
        return peak

    def close(self):
        self.stream.abort()
        self.stream.close()


class MicInput:
    """The microphone, delivered as mono float32 blocks to on_audio; on_stopped fires if the device dies."""

    def __init__(self, device: dict, on_audio, on_stopped):
        self.name = device["name"]
        self.on_audio = on_audio
        self.on_stopped = on_stopped
        self.closing = False
        last_error = None
        # Mono first; some devices only open with their own channel count, then we average
        for channels in dict.fromkeys([1, min(device["channels"], 2)]):
            try:
                self.stream = sd.InputStream(
                    device=device["index"], samplerate=SAMPLE_RATE, channels=channels, dtype="float32",
                    latency="low", callback=self._callback, finished_callback=self._finished,
                    extra_settings=_extra_settings(device),
                )
                break
            except sd.PortAudioError as e:
                last_error = e
        else:
            raise last_error
        self.stream.start()

    def _callback(self, indata, frames, time_info, status):
        self.on_audio(indata.mean(axis=1) if indata.shape[1] > 1 else indata[:, 0].copy())

    def _finished(self):
        if not self.closing:
            self.on_stopped()

    def close(self):
        self.closing = True
        self.stream.abort()
        self.stream.close()


class WavInput:
    """Test microphone: a mono 16-bit 48 kHz WAV file, looped and paced in real time."""

    def __init__(self, path: str, on_audio):
        self.name = TEST_INPUT_PREFIX + path
        with wave.open(path) as w:
            if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise AudioError(f"Test input must be mono 16-bit {SAMPLE_RATE} Hz: {path}")
            self.audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        self.on_audio = on_audio
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="wav-input", daemon=True)
        self.thread.start()

    def _run(self):
        t0, sent = time.perf_counter(), 0
        while not self.stop.wait(0.01):
            due = int((time.perf_counter() - t0) * SAMPLE_RATE) - sent
            if due > 0:
                self.on_audio(self.audio[(sent + np.arange(due)) % len(self.audio)])
                sent += due

    def close(self):
        self.stop.set()
        self.thread.join()


class LocalAudio:
    def __init__(self, engine, model_executor, settings_path: Path):
        self.engine = engine
        self.model_executor = model_executor
        self.settings_path = settings_path
        self.settings = {"input": None, "output": None, "monitor": None, "gain": 1.0}
        if settings_path.exists():
            with open(settings_path, "r", encoding="utf-8") as f:
                self.settings.update(json.load(f))

        self.lock = threading.RLock()   # every state change (power, monitor, devices, test) holds this
        self.cable_on = False
        self.monitor_on = False
        self.error = None
        self.source = None
        self.cable_line = None
        self.monitor_line = None
        self.worker = None
        self.stopping = False
        self.cond = threading.Condition()
        self.pending: list[np.ndarray] = []
        self.pending_len = 0
        self.test_state = "idle"        # idle -> recording -> playing -> idle
        self.test_frames: list[np.ndarray] = []
        self.test_until = 0.0

    # ---- devices ----------------------------------------------------------------------------------
    @property
    def streaming(self) -> bool:
        return self.source is not None

    def devices(self) -> dict:
        with self.lock:
            if not self.streaming and self.test_state == "idle":
                audio_devices.refresh()   # pick up devices plugged in or installed since the last look
            return audio_devices.list_devices()

    def resolved(self) -> dict:
        """The devices that will be used: the saved choice, else a sensible default."""
        listed = audio_devices.list_devices()
        default_in = next((d["name"] for d in listed["inputs"] if d["default"]), None)
        default_out = next((d["name"] for d in listed["outputs"] if d["default"] and d["name"] != listed["cable"]), None)
        return {
            "input": self.settings["input"] or default_in,
            "output": self.settings["output"] or listed["cable"],
            "monitor": self.settings["monitor"] or default_out,
            # Windows often makes a newly installed cable the default speaker, which silences the PC
            "cable_is_default_speaker": any(d["default"] and d["name"] == listed["cable"] for d in listed["outputs"]),
        }

    def _device(self, kind: str, role: str, label: str) -> dict:
        name = self.resolved()[role]
        if name is None:
            if role == "output":
                raise AudioError("The virtual cable (VB-Cable) is not installed. Install it, restart the app, then switch on.")
            raise AudioError(f"No {label} found. Plug one in and pick it under Devices.")
        device = audio_devices.find_device(kind, name)
        if device is None:
            raise AudioError(f"{label.capitalize()} \"{name}\" is not connected. Pick another under Devices.")
        return device

    # ---- state ------------------------------------------------------------------------------------
    def set_power(self, on: bool):
        with self.lock:
            self.error = None
            self.cable_on = on
            self._apply()

    def set_monitor(self, on: bool):
        with self.lock:
            if self.test_state != "idle":
                raise AudioError("Wait for the test recording to finish.")
            self.error = None
            self.monitor_on = on
            self._apply()

    def update_settings(self, changes: dict):
        with self.lock:
            self.error = None
            self.settings.update(changes)
            tmp = self.settings_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
            tmp.replace(self.settings_path)
            if self.streaming and any(k in changes for k in ("input", "output", "monitor")):
                self._close_all()   # reopen on the newly chosen devices
                self._apply()

    def start_test(self, seconds: float):
        with self.lock:
            if self.test_state != "idle":
                raise AudioError("A test is already running.")
            self.error = None
            self._device("output", "monitor", "headphones or speakers")   # fail now, not after recording
            self.monitor_on = False
            self.test_frames = []
            self.test_until = time.monotonic() + seconds
            self.test_state = "recording"
            try:
                self._apply()
            except AudioError:
                self.test_state = "idle"
                raise

    def _apply(self):
        """Open or close streams so they match cable_on / monitor_on / test_state. On a device problem
        everything is switched off and the reason is kept for the app to show."""
        try:
            self._sync()
            self.error = None
        except (AudioError, sd.PortAudioError) as e:
            logger.error(f"Audio device problem: {e}")
            self.cable_on = self.monitor_on = False
            self._close_all()
            self.error = str(e)
            raise AudioError(str(e)) from e

    def _sync(self):
        if (self.cable_on or self.monitor_on or self.test_state == "recording") and self.engine.vc is None:
            raise AudioError("The voice engine is still loading. Try again in a moment.")
        for attr, wanted, role, label in (
            ("cable_line", self.cable_on, "output", "virtual cable"),
            ("monitor_line", self.monitor_on, "monitor", "headphones or speakers"),
        ):
            line = getattr(self, attr)
            if wanted and line is None:
                setattr(self, attr, OutputLine(self._device("output", role, label)))
            elif not wanted and line is not None:
                setattr(self, attr, None)
                line.close()

        want_input = self.cable_on or self.monitor_on or self.test_state == "recording"
        if want_input and self.source is None:
            self._start_input()
        elif not want_input and self.source is not None:
            self._stop_input()

    def _start_input(self):
        name = self.resolved()["input"]
        self.pending, self.pending_len = [], 0
        if name and name.startswith(TEST_INPUT_PREFIX) and os.environ.get("RVC_AUDIO_TEST") == "1":
            self.source = WavInput(name[len(TEST_INPUT_PREFIX):], self._on_audio)
        else:
            self.source = MicInput(self._device("input", "input", "microphone"), self._on_audio, self._on_mic_stopped)
        self.stopping = False
        self.worker = threading.Thread(target=self._run, name="local-audio", daemon=True)
        self.worker.start()
        logger.info(f"Local audio started: mic \"{self.source.name}\"")

    def _stop_input(self):
        self.source.close()
        self.source = None
        with self.cond:
            self.stopping = True
            self.cond.notify()
        self.worker.join()
        self.worker = None
        self.model_executor.submit(self.engine.drop_stream, LOCAL_STREAM_ID)
        logger.info("Local audio stopped")

    def _close_all(self):
        for attr in ("cable_line", "monitor_line"):
            line = getattr(self, attr)
            if line is not None:
                setattr(self, attr, None)
                line.close()
        if self.source is not None:
            self._stop_input()

    def shutdown(self):
        with self.lock:
            self.cable_on = self.monitor_on = False
            self._close_all()

    # ---- audio path -------------------------------------------------------------------------------
    def _on_audio(self, mono: np.ndarray):
        with self.cond:
            self.pending.append(mono)
            self.pending_len += len(mono)
            self.cond.notify()

    def _on_mic_stopped(self):
        # Called on PortAudio's thread when the device goes away: switch off from another thread
        def fail():
            with self.lock:
                if self.source is None:
                    return
                logger.error("Microphone stopped delivering audio (unplugged?)")
                self.cable_on = self.monitor_on = False
                self.test_state = "idle"
                self._close_all()
                self.error = "The microphone stopped (was it unplugged?). Check Devices, then switch on again."
        threading.Thread(target=fail, name="local-audio-fail", daemon=True).start()

    def _take_frame(self) -> np.ndarray | None:
        with self.cond:
            while True:
                if self.stopping:
                    return None
                n = self.engine.chunk_size * 128
                if self.pending_len >= n:
                    break
                self.cond.wait()
            audio = np.concatenate(self.pending)
            if len(audio) > n * MAX_BACKLOG_FRAMES:   # the model fell behind: never let delay build up
                audio = audio[-n * MAX_BACKLOG_FRAMES:]
            frame, rest = audio[:n], audio[n:]
            self.pending, self.pending_len = [rest], len(rest)
            return frame.astype(np.float32)

    def _run(self):
        while True:
            frame = self._take_frame()
            if frame is None:
                return
            try:
                y = self.model_executor.submit(self.engine.process_chunk, frame, LOCAL_STREAM_ID).result()
            except Exception:
                logger.exception("Voice conversion failed for one frame")
                continue
            y = np.nan_to_num(np.asarray(y, dtype=np.float32))
            y = np.clip(y * self.settings["gain"], -1.0, 1.0)
            cable, monitor = self.cable_line, self.monitor_line
            if cable is not None:
                cable.push(y)
            if monitor is not None:
                monitor.push(y)
            if self.test_state == "recording":
                self.test_frames.append(y)
                if time.monotonic() >= self.test_until:
                    self.test_state = "playing"
                    threading.Thread(target=self._play_test, name="local-audio-test", daemon=True).start()

    def _play_test(self):
        audio = np.concatenate(self.test_frames)
        self.test_frames = []
        try:
            with self.lock:
                self._apply()   # the recording is over: release the mic unless the cable is on
                line = OutputLine(self._device("output", "monitor", "headphones or speakers"))
            line.push(audio)
            time.sleep((PREFILL + len(audio)) / SAMPLE_RATE + 0.3)
            line.close()
        except (AudioError, sd.PortAudioError) as e:
            logger.error(f"Could not play the test recording: {e}")
            self.error = str(e)
        finally:
            self.test_state = "idle"

    def status(self) -> dict:
        cable = self.cable_line
        with self.lock:   # not while devices() is re-scanning
            resolved = self.resolved()
        return {
            "on": self.cable_on,
            "monitor_on": self.monitor_on,
            "test": self.test_state,
            "test_seconds_left": round(max(0.0, self.test_until - time.monotonic()), 1) if self.test_state == "recording" else 0,
            "error": self.error,
            "engine_ready": self.engine.vc is not None,
            "devices": resolved,
            "cable_is_default_speaker": resolved.pop("cable_is_default_speaker"),
            "gain": self.settings["gain"],
            "input_level": round(self.engine.last_vol, 4) if self.streaming else 0.0,
            "output_level": round(cable.take_peak(), 4) if cable is not None else 0.0,
            "underruns": cable.underruns if cable is not None else 0,
        }


class PowerRequest(BaseModel):
    on: bool


class TestRequest(BaseModel):
    seconds: float = Field(5.0, ge=2.0, le=15.0)


class AudioSettingsUpdate(BaseModel):
    input: str | None = None
    output: str | None = None
    monitor: str | None = None
    gain: float | None = Field(None, ge=0.2, le=3.0)


def create_router(audio: LocalAudio) -> APIRouter:
    router = APIRouter(prefix="/audio")

    def run(action, *args):
        try:
            action(*args)
        except AudioError as e:
            raise HTTPException(status_code=409, detail=str(e))
        return audio.status()

    @router.get("/devices")
    def devices():
        return audio.devices()

    @router.get("/status")
    def status():
        return audio.status()

    @router.post("/power")
    def power(req: PowerRequest):
        return run(audio.set_power, req.on)

    @router.post("/monitor")
    def monitor(req: PowerRequest):
        return run(audio.set_monitor, req.on)

    @router.post("/test")
    def test(req: TestRequest):
        return run(audio.start_test, req.seconds)

    @router.post("/settings")
    def settings(req: AudioSettingsUpdate):
        return run(audio.update_settings, req.model_dump(exclude_none=True))

    return router
