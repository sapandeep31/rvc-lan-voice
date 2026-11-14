"""RVC LAN voice client for Windows.

Mic (TONOR) -> float32 mono frames -> Mac server /ws/voice -> converted audio -> VB-Cable "CABLE Input"
(Discord / games use "CABLE Output" as their microphone).

Frames are pipelined: the client sends each frame as soon as it fills and receives replies on a
separate task, matched FIFO (the server answers every frame exactly once, in order). Throughput is
therefore bound by server processing time, not by processing + network round trip.

Converted audio goes through a small jitter buffer before the output device, which is fed only as
much as it can take, so audio is never silently dropped by the device. Underruns (silence inserted)
and overflows (old audio dropped) are counted and logged.
"""

import argparse
import asyncio
import collections
import json
import logging
import queue
import statistics
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import websockets
from PySide6.QtCore import QCoreApplication, Qt, QTimer
from PySide6.QtMultimedia import (
    QAudio,
    QAudioDevice,
    QAudioFormat,
    QAudioSink,
    QAudioSource,
    QMediaDevices,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("Windows-Client")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
else:
    config = {}
client_cfg = config.get("client", {})


def parse_args():
    p = argparse.ArgumentParser(description="RVC LAN voice client")
    p.add_argument("--chunk", type=int, default=client_cfg.get("chunk_size", 56),
                   help="frame size in units of 128 samples (56 = 7168 samples = 149 ms)")
    p.add_argument("--seconds", type=float, default=0, help="exit after N seconds (0 = run forever)")
    p.add_argument("--record", type=str, default="",
                   help="write mic input and converted output to <path>_in.wav / <path>_out.wav")
    p.add_argument("--monitor", type=str, default=client_cfg.get("monitor_device_keyword", ""),
                   help="also play converted audio to an output device whose name contains this (e.g. WH-CH720N)")
    p.add_argument("--prefill-ms", type=float, default=client_cfg.get("jitter_prefill_ms", 30))
    p.add_argument("--quiet", action="store_true", help="no periodic stats lines")
    return p.parse_args()


ARGS = parse_args()
SERVER_IP = client_cfg.get("server_ip", "192.168.1.100")
SERVER_PORT = client_cfg.get("server_port", 8000)
SAMPLE_RATE = client_cfg.get("sample_rate", 48000)
CHUNK_SIZE = ARGS.chunk
OUTPUT_GAIN = client_cfg.get("output_gain", 1.5)
INPUT_KEYWORD = client_cfg.get("input_device_keyword", "TONOR")
OUTPUT_KEYWORD = client_cfg.get("output_device_keyword", "CABLE Input")
OUTPUT_BUFFER_MS = client_cfg.get("output_buffer_ms", 40)
INPUT_BUFFER_MS = client_cfg.get("input_buffer_ms", 20)
MAX_IN_FLIGHT = client_cfg.get("max_in_flight", 4)
PASSTHROUGH = client_cfg.get("passthrough_when_disconnected", False)

BLOCK_SAMPLES = CHUNK_SIZE * 128
BLOCK_MS = BLOCK_SAMPLES / SAMPLE_RATE * 1000
PREFILL_SAMPLES = int(ARGS.prefill_ms * SAMPLE_RATE / 1000)
# Jitter buffer ceiling: a full frame on top of the prefill plus one frame of slack.
MAX_OUT_SAMPLES = PREFILL_SAMPLES + 2 * BLOCK_SAMPLES
TRIM_MARGIN = SAMPLE_RATE // 100          # tolerate 10 ms above target before trimming
XFADE = SAMPLE_RATE // 200                # 5 ms crossfade when trimming
FADE_IN = np.linspace(0.0, 1.0, XFADE, dtype=np.float32)
FADE_OUT = 1.0 - FADE_IN


def ms(samples: float) -> float:
    return samples / SAMPLE_RATE * 1000


class AudioNetworkBridge:
    """Background asyncio thread: pipelined send/receive over one WebSocket."""

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.recv_queue: "queue.Queue[bytes]" = queue.Queue()
        self.is_running = True
        self.is_connected = False
        self.loop = None
        self._send_q = None
        self.lock = threading.Lock()
        self.rtts = []          # ms, drained by stats
        self.rtt_history = collections.deque(maxlen=60)   # rolling, for jitter estimate
        self.in_flight = 0
        self.max_in_flight_seen = 0
        self.send_drops = 0     # frames dropped because too many were in flight
        self.reconnects = 0
        self._ready = threading.Event()
        self.net_thread = threading.Thread(target=self._run, daemon=True)
        self.net_thread.start()
        self._ready.wait(5)

    def submit(self, frame: bytes):
        if self.loop is not None and self._send_q is not None:
            self.loop.call_soon_threadsafe(self._send_q.put_nowait, frame)

    def _run(self):
        asyncio.run(self._main())

    async def _main(self):
        self.loop = asyncio.get_running_loop()
        self._send_q = asyncio.Queue()
        self._ready.set()
        while self.is_running:
            try:
                logger.info(f"Connecting to Mac RVC server at {self.server_url}...")
                async with websockets.connect(
                    self.server_url,
                    ping_interval=10,
                    ping_timeout=5,
                    max_size=10 * 1024 * 1024,
                    compression=None,
                ) as ws:
                    while not self._send_q.empty():   # don't send stale audio after reconnect
                        self._send_q.get_nowait()
                    pending = collections.deque()      # send timestamps, FIFO
                    self.is_connected = True
                    logger.info("CONNECTED to Mac server. Audio streaming active.")
                    sender = asyncio.create_task(self._sender(ws, pending))
                    receiver = asyncio.create_task(self._receiver(ws, pending))
                    done, rest = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
                    for t in rest:
                        t.cancel()
                    for t in done:
                        t.result()
            except Exception as e:
                logger.warning(f"Connection lost ({type(e).__name__}: {e}). Retrying in 2 s...")
            self.is_connected = False
            self.in_flight = 0
            self.reconnects += 1
            await asyncio.sleep(2.0)

    async def _sender(self, ws, pending):
        while True:
            frame = await self._send_q.get()
            if len(pending) >= MAX_IN_FLIGHT:
                # Server is falling behind: drop this frame rather than build up latency.
                self.send_drops += 1
                continue
            pending.append(time.perf_counter())
            self.in_flight = len(pending)
            self.max_in_flight_seen = max(self.max_in_flight_seen, self.in_flight)
            await ws.send(frame)

    async def _receiver(self, ws, pending):
        async for msg in ws:
            now = time.perf_counter()
            if pending:
                rtt = (now - pending.popleft()) * 1000
                with self.lock:
                    self.rtts.append(rtt)
                    self.rtt_history.append(rtt)
            self.in_flight = len(pending)
            if isinstance(msg, (bytes, bytearray)):
                self.recv_queue.put_nowait(bytes(msg))


class OutputStream:
    """One output device fed from a mono float buffer, writing only what the device can take."""

    def __init__(self, dev: QAudioDevice, buffer_ms: float):
        fmt = dev.preferredFormat()
        fmt.setSampleRate(SAMPLE_RATE)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Float)
        self.channels = fmt.channelCount()
        self.sink = QAudioSink(dev, fmt)
        self.sink.setBufferSize(int(buffer_ms * SAMPLE_RATE / 1000) * 4 * self.channels)
        self.io = self.sink.start()
        self.buffer_ms = ms(self.sink.bufferSize() / (4 * self.channels))

    def free_samples(self) -> int:
        return self.sink.bytesFree() // (4 * self.channels)

    def queued_samples(self) -> int:
        return (self.sink.bufferSize() - self.sink.bytesFree()) // (4 * self.channels)

    def write(self, mono: np.ndarray):
        if len(mono) == 0 or self.io is None:
            return
        if self.channels == 1:
            data = mono
        else:
            data = np.repeat(mono, self.channels)   # interleave identical channels
        self.io.write(data.astype(np.float32).tobytes())

    def stop(self):
        self.sink.stop()


class WindowsAudioClient:
    def __init__(self, bridge: AudioNetworkBridge):
        self.bridge = bridge
        self.mono_in = np.empty(0, dtype=np.float32)
        self.raw_rem = b""

        in_dev = self._find_device(True, INPUT_KEYWORD)
        out_dev = self._find_device(False, OUTPUT_KEYWORD)
        logger.info(f"Mic:    {in_dev.description()}")
        logger.info(f"Output: {out_dev.description()}")

        fmt_in = in_dev.preferredFormat()
        fmt_in.setSampleRate(SAMPLE_RATE)
        fmt_in.setSampleFormat(QAudioFormat.SampleFormat.Float)
        self.in_channels = fmt_in.channelCount()
        self.audio_source = QAudioSource(in_dev, fmt_in)
        self.audio_source.setBufferSize(int(INPUT_BUFFER_MS * SAMPLE_RATE / 1000) * 4 * self.in_channels)
        self.input_io = self.audio_source.start()

        self.out = OutputStream(out_dev, OUTPUT_BUFFER_MS)
        logger.info(f"Output device buffer: {self.out.buffer_ms:.0f} ms | jitter prefill: {ARGS.prefill_ms:.0f} ms")

        self.monitor = None
        self.mon_buf = np.empty(0, dtype=np.float32)
        if ARGS.monitor:
            mon_dev = self._find_device(False, ARGS.monitor, fallback=False)
            if mon_dev is None:
                logger.warning(f"Monitor device '{ARGS.monitor}' not found; monitor off.")
            else:
                self.monitor = OutputStream(mon_dev, OUTPUT_BUFFER_MS)
                logger.info(f"Monitor: {mon_dev.description()} (you hear what Discord hears)")

        # Jitter buffer state
        self.out_buf = np.empty(0, dtype=np.float32)
        self.playing = False
        self.underruns = 0
        self.overflow_ms = 0.0
        self.frames_out = 0
        self.bad_frames = 0
        self.out_level_samples = []
        self.trimmed_ms = 0.0
        self.target = PREFILL_SAMPLES
        self.win_min = None
        self.win_t0 = time.perf_counter()
        self.win_len = max(1.0, 4 * BLOCK_MS / 1000)

        self.rec_in = [] if ARGS.record else None
        self.rec_out = [] if ARGS.record else None

        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._process)
        self.timer.start(5)
        if self.input_io is not None:
            self.input_io.readyRead.connect(self._process)

        self.t_start = time.perf_counter()
        if not ARGS.quiet:
            self.stats_timer = QTimer()
            self.stats_timer.timeout.connect(self._stats)
            self.stats_timer.start(2000)

    @staticmethod
    def _find_device(is_input: bool, keyword: str, fallback: bool = True):
        devs = QMediaDevices.audioInputs() if is_input else QMediaDevices.audioOutputs()
        for d in devs:
            if keyword.lower() in d.description().lower():
                return d
        if not fallback:
            return None
        d = QMediaDevices.defaultAudioInput() if is_input else QMediaDevices.defaultAudioOutput()
        logger.warning(f"No device matching '{keyword}'; falling back to default: {d.description()}")
        return d

    # ---- main audio tick -------------------------------------------------
    def _process(self):
        if self.input_io is None:
            return
        self._read_mic()
        self._dispatch_frames()
        self._collect_replies()
        self._feed_output()

    def _read_mic(self):
        avail = self.input_io.bytesAvailable()
        if avail <= 0:
            return
        raw = self.raw_rem + bytes(self.input_io.read(avail))
        frame_bytes = 4 * self.in_channels
        usable = len(raw) - len(raw) % frame_bytes   # keep sample frames aligned
        self.raw_rem = raw[usable:]
        if usable == 0:
            return
        x = np.frombuffer(raw[:usable], dtype=np.float32)
        mono = x.reshape(-1, self.in_channels).mean(axis=1) if self.in_channels > 1 else x
        self.mono_in = np.concatenate([self.mono_in, mono.astype(np.float32)])
        if len(self.mono_in) > BLOCK_SAMPLES * 3:   # never let input lag build up
            self.mono_in = self.mono_in[-BLOCK_SAMPLES * 3:]

    def _dispatch_frames(self):
        while len(self.mono_in) >= BLOCK_SAMPLES:
            chunk = self.mono_in[:BLOCK_SAMPLES].copy()
            self.mono_in = self.mono_in[BLOCK_SAMPLES:]
            if self.rec_in is not None:
                self.rec_in.append(chunk)
            if self.bridge.is_connected:
                self.bridge.submit(chunk.tobytes())
            elif PASSTHROUGH:
                self._enqueue_out(chunk)

    def _collect_replies(self):
        while True:
            try:
                data = self.bridge.recv_queue.get_nowait()
            except queue.Empty:
                break
            y = np.frombuffer(data, dtype=np.float32)
            if len(y) != BLOCK_SAMPLES or not np.isfinite(y).all():
                self.bad_frames += 1
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            if self.rec_out is not None:
                self.rec_out.append(y.copy())
            self.frames_out += 1
            self._enqueue_out(y)

    def _enqueue_out(self, y: np.ndarray):
        if OUTPUT_GAIN != 1.0:
            y = np.clip(y * OUTPUT_GAIN, -1.0, 1.0)
        self.out_buf = np.concatenate([self.out_buf, y.astype(np.float32)])
        if len(self.out_buf) > MAX_OUT_SAMPLES:
            drop = len(self.out_buf) - MAX_OUT_SAMPLES
            self.overflow_ms += ms(drop)
            self.out_buf = self.out_buf[drop:]

    def _feed_output(self):
        if not self.playing:
            # Prime: wait for the prefill before starting, so playback doesn't stutter immediately.
            if len(self.out_buf) >= self.target and len(self.out_buf) > 0:
                self.playing = True
            else:
                return
        free = self.out.free_samples()
        n = min(free, len(self.out_buf))
        if n > 0:
            piece = self.out_buf[:n]
            self.out_buf = self.out_buf[n:]
            self.out.write(piece)
            if self.monitor is not None:
                self.mon_buf = np.concatenate([self.mon_buf, piece])
        if len(self.out_buf) == 0 and self.out.queued_samples() < SAMPLE_RATE // 200:
            # Device is about to run dry (<5 ms queued) with nothing to give it.
            if self.bridge.is_connected and self.frames_out > 0:
                self.underruns += 1
            self.playing = False
        level = len(self.out_buf) + self.out.queued_samples()
        self.out_level_samples.append(level)
        self._latency_control(level)
        if self.monitor is not None and len(self.mon_buf):
            m = min(self.monitor.free_samples(), len(self.mon_buf))
            if m > 0:
                self.monitor.write(self.mon_buf[:m])
                self.mon_buf = self.mon_buf[m:]
            if len(self.mon_buf) > SAMPLE_RATE // 5:   # monitor is listen-only; never let it lag
                self.mon_buf = self.mon_buf[-SAMPLE_RATE // 10:]

    def _update_target(self):
        """Prefill target = observed network/processing jitter (p95 - median rtt) + 10 ms, floored at --prefill-ms."""
        with self.bridge.lock:
            h = sorted(self.bridge.rtt_history)
        if len(h) >= 10:
            jitter = h[int(len(h) * 0.95)] - h[len(h) // 2]
            self.target = max(PREFILL_SAMPLES, int((jitter + 10) * SAMPLE_RATE / 1000))

    def _latency_control(self, level: int):
        """Keep queued output near the prefill target.

        Replies arrive in bursts at start-up or after a network hiccup; since audio arrives and plays
        at the same rate, any backlog would otherwise stay forever as extra delay. Track the lowest
        queue level over a window longer than one frame period: if even the low point is well above
        target, that surplus is pure latency, so cut it out with a short crossfade (no click).
        """
        now = time.perf_counter()
        if not self.playing:
            self.win_min = None
            self.win_t0 = now
            return
        self.win_min = level if self.win_min is None else min(self.win_min, level)
        if now - self.win_t0 < self.win_len:
            return
        self._update_target()
        excess = self.win_min - self.target - TRIM_MARGIN
        self.win_min, self.win_t0 = None, now
        k = min(excess, len(self.out_buf) - XFADE)
        if k <= 0:
            return
        buf = self.out_buf
        head = buf[:XFADE] * FADE_OUT + buf[k:k + XFADE] * FADE_IN
        self.out_buf = np.concatenate([head, buf[k + XFADE:]]).astype(np.float32)
        self.trimmed_ms += ms(k)

    # ---- reporting -------------------------------------------------------
    def snapshot(self, reset=True):
        with self.bridge.lock:
            rtts = self.bridge.rtts
            if reset:
                self.bridge.rtts = []
        lvl = self.out_level_samples
        if reset:
            self.out_level_samples = []
        return rtts, lvl

    def _stats(self):
        rtts, lvl = self.snapshot()
        if not self.bridge.is_connected:
            logger.info("waiting for server...")
            return
        if not rtts:
            logger.info(f"connected, no replies in last 2 s | in-flight {self.bridge.in_flight}")
            return
        rtts.sort()
        med = statistics.median(rtts)
        p95 = rtts[min(len(rtts) - 1, int(len(rtts) * 0.95))]
        # A frame's first sample is captured, waits BLOCK_MS for the frame to fill, travels (rtt), then
        # plays after whatever is still queued when it lands - the queue's low point.
        out_ms = ms(min(lvl)) if lvl else 0.0
        e2e = BLOCK_MS + med + out_ms
        logger.info(
            f"rtt med {med:.0f} p95 {p95:.0f} max {rtts[-1]:.0f} ms | in-flight max {self.bridge.max_in_flight_seen} "
            f"| queue low {out_ms:.0f} (target {ms(self.target):.0f}) ms | underruns {self.underruns} | trimmed {self.trimmed_ms:.0f} ms | dropped: out {self.overflow_ms:.0f} ms, "
            f"send {self.bridge.send_drops} | bad {self.bad_frames} | est mouth->Discord {e2e:.0f} ms"
        )
        self.bridge.max_in_flight_seen = self.bridge.in_flight

    def stop(self):
        self.timer.stop()
        self.out.stop()
        if self.monitor is not None:
            self.monitor.stop()
        self.audio_source.stop()
        if ARGS.record:
            for suffix, chunks in (("_in", self.rec_in), ("_out", self.rec_out)):
                path = f"{ARGS.record}{suffix}.wav"
                data = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
                with wave.open(path, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(SAMPLE_RATE)
                    w.writeframes((np.clip(data, -1, 1) * 32767).astype(np.int16).tobytes())
                logger.info(f"wrote {path} ({ms(len(data)) / 1000:.1f} s)")


def main():
    print("=" * 64)
    print("    RVC LAN Voice Client for Windows")
    print("=" * 64)
    print(f"Server:      ws://{SERVER_IP}:{SERVER_PORT}/ws/voice")
    print(f"Mic:         '{INPUT_KEYWORD}'   Output: '{OUTPUT_KEYWORD}'   Monitor: '{ARGS.monitor or 'off'}'")
    print(f"Frame:       chunk {CHUNK_SIZE} = {BLOCK_SAMPLES} samples = {BLOCK_MS:.0f} ms @ {SAMPLE_RATE} Hz")
    print("In Discord/games pick 'CABLE Output (VB-Audio Virtual Cable)' as the microphone.")
    print("=" * 64)

    app = QCoreApplication(sys.argv[:1])
    bridge = AudioNetworkBridge(f"ws://{SERVER_IP}:{SERVER_PORT}/ws/voice")
    client = WindowsAudioClient(bridge)

    if ARGS.seconds > 0:
        QTimer.singleShot(int(ARGS.seconds * 1000), app.quit)

    try:
        app.exec()
    except KeyboardInterrupt:
        logger.info("Exiting...")
    finally:
        bridge.is_running = False
        client.stop()
        if ARGS.seconds > 0:
            summary = (f"SUMMARY chunk {CHUNK_SIZE} ({BLOCK_MS:.0f} ms): frames {client.frames_out}, "
                       f"underruns {client.underruns}, trimmed {client.trimmed_ms:.0f} ms, out dropped {client.overflow_ms:.0f} ms, "
                       f"send drops {bridge.send_drops}, bad {client.bad_frames}, reconnects {bridge.reconnects}")
            logger.info(summary)


if __name__ == "__main__":
    main()
