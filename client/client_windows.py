import asyncio
import json
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import websockets
from PySide6.QtCore import QByteArray, QCoreApplication, QTimer
from PySide6.QtMultimedia import (
    QAudioDevice,
    QAudioFormat,
    QAudioSink,
    QAudioSource,
    QMediaDevices,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("Windows-Client")

# Load configuration
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
else:
    config = {}

client_cfg = config.get("client", {})
SERVER_IP = client_cfg.get("server_ip", "192.168.1.100")
SERVER_PORT = client_cfg.get("server_port", 8000)
SAMPLE_RATE = client_cfg.get("sample_rate", 48000)
CHUNK_SIZE = client_cfg.get("chunk_size", 56)
OUTPUT_GAIN = client_cfg.get("output_gain", 1.5)
INPUT_KEYWORD = client_cfg.get("input_device_keyword", "TONOR")
OUTPUT_KEYWORD = client_cfg.get("output_device_keyword", "CABLE Input")

BLOCK_SAMPLES = CHUNK_SIZE * 128  # 7168 mono samples


class AudioNetworkBridge:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.send_queue = queue.Queue(maxsize=10)
        self.recv_queue = queue.Queue(maxsize=10)
        self.is_running = True
        self.is_connected = False
        self.last_latency_ms = 0.0

        # Start background async network thread
        self.net_thread = threading.Thread(target=self._run_network_loop, daemon=True)
        self.net_thread.start()

    def _run_network_loop(self):
        asyncio.run(self._network_worker())

    async def _network_worker(self):
        while self.is_running:
            try:
                logger.info(f"Connecting to Mac RVC Server at {self.server_url}...")
                async with websockets.connect(
                    self.server_url,
                    ping_interval=10,
                    ping_timeout=5,
                    max_size=10 * 1024 * 1024,
                ) as ws:
                    self.is_connected = True
                    logger.info("Successfully CONNECTED to Mac server! Audio streaming active.")

                    while self.is_running:
                        try:
                            # Non-blocking get from queue
                            data_to_send = self.send_queue.get_nowait()
                        except queue.Empty:
                            await asyncio.sleep(0.002)
                            continue

                        t_start = time.perf_counter()
                        await ws.send(data_to_send)

                        # Receive converted audio bytes
                        resp_bytes = await ws.recv()
                        t_elapsed = (time.perf_counter() - t_start) * 1000
                        self.last_latency_ms = t_elapsed

                        if not self.recv_queue.full():
                            self.recv_queue.put_nowait(resp_bytes)

            except Exception as e:
                self.is_connected = False
                logger.warning(f"Connection lost ({e}). Retrying in 2 seconds...")
                await asyncio.sleep(2.0)


class WindowsAudioClient:
    def __init__(self, bridge: AudioNetworkBridge):
        self.bridge = bridge
        self.mono_in_buffer = np.empty(0, dtype=np.float32)

        # Select devices
        in_dev = self._find_device(is_input=True, keyword=INPUT_KEYWORD)
        out_dev = self._find_device(is_input=False, keyword=OUTPUT_KEYWORD)

        logger.info(f"Selected Mic: {in_dev.description()}")
        logger.info(f"Selected Output: {out_dev.description()}")

        fmt_in = in_dev.preferredFormat()
        fmt_in.setSampleRate(SAMPLE_RATE)
        fmt_in.setSampleFormat(QAudioFormat.SampleFormat.Float)

        fmt_out = out_dev.preferredFormat()
        fmt_out.setSampleRate(SAMPLE_RATE)
        fmt_out.setSampleFormat(QAudioFormat.SampleFormat.Float)

        self.in_channels = fmt_in.channelCount()
        self.out_channels = fmt_out.channelCount()

        self.audio_source = QAudioSource(in_dev, fmt_in)
        self.audio_sink = QAudioSink(out_dev, fmt_out)

        self.input_io = self.audio_source.start()
        self.output_io = self.audio_sink.start()

        self.total_streamed_chunks = 0

        # High-resolution timer for audio processing
        self.timer = QTimer()
        self.timer.timeout.connect(self._process)
        self.timer.start(10)

        if self.input_io is not None:
            self.input_io.readyRead.connect(self._process)

    def _find_device(self, is_input: bool, keyword: str) -> QAudioDevice:
        devs = QMediaDevices.audioInputs() if is_input else QMediaDevices.audioOutputs()
        for d in devs:
            if keyword.lower() in d.description().lower():
                return d
        return QMediaDevices.defaultAudioInput() if is_input else QMediaDevices.defaultAudioOutput()

    def _process(self):
        if self.input_io is None or self.output_io is None:
            return

        # 1. Read available mic bytes from TONOR
        avail = self.input_io.bytesAvailable()
        if avail > 0:
            raw_bytes = self.input_io.read(avail)
            raw_floats = np.frombuffer(raw_bytes, dtype=np.float32)
            if len(raw_floats) > 0:
                if self.in_channels == 2:
                    mono = 0.5 * (raw_floats[0::2] + raw_floats[1::2])
                elif self.in_channels > 2:
                    mono = raw_floats.reshape(-1, self.in_channels).mean(axis=1)
                else:
                    mono = raw_floats
                self.mono_in_buffer = np.append(self.mono_in_buffer, mono)

        # Buffer ceiling guard (max 3 chunks to prevent latency lag)
        max_buf = BLOCK_SAMPLES * 3
        if len(self.mono_in_buffer) > max_buf:
            self.mono_in_buffer = self.mono_in_buffer[-max_buf:]

        # 2. Dispatch ready chunks to Mac over LAN
        while len(self.mono_in_buffer) >= BLOCK_SAMPLES:
            chunk = self.mono_in_buffer[:BLOCK_SAMPLES]
            self.mono_in_buffer = self.mono_in_buffer[BLOCK_SAMPLES:]

            if self.bridge.is_connected:
                try:
                    self.bridge.send_queue.put_nowait(chunk.tobytes())
                except queue.Full:
                    pass
            else:
                # If disconnected, loop original audio as pass-through
                self._write_output(chunk)

        # 3. Write received converted chunks to CABLE Input
        while not self.bridge.recv_queue.empty():
            try:
                converted_bytes = self.bridge.recv_queue.get_nowait()
                out_mono = np.frombuffer(converted_bytes, dtype=np.float32)
                self._write_output(out_mono)
                self.total_streamed_chunks += 1

                if self.total_streamed_chunks % 100 == 0:
                    logger.info(
                        f"[{self.total_streamed_chunks} chunks] Round-Trip Latency: {self.bridge.last_latency_ms:.1f}ms | Windows GPU: 0% (Free for Valorant)"
                    )
            except queue.Empty:
                break

    def _write_output(self, out_mono: np.ndarray):
        if len(out_mono) == 0:
            return

        # Apply digital gain boost
        if OUTPUT_GAIN != 1.0:
            out_mono = np.clip(out_mono * OUTPUT_GAIN, -1.0, 1.0)

        # Upmix to stereo for VB-CABLE
        if self.out_channels == 2:
            out_stereo = np.column_stack([out_mono, out_mono]).flatten()
            self.output_io.write(out_stereo.tobytes())
        elif self.out_channels > 2:
            out_multi = np.tile(out_mono[:, None], (1, self.out_channels)).flatten()
            self.output_io.write(out_multi.tobytes())
        else:
            self.output_io.write(out_mono.tobytes())

    def stop(self):
        self.timer.stop()
        if self.audio_sink:
            self.audio_sink.stop()
        if self.audio_source:
            self.audio_source.stop()


def main():
    print("=" * 60)
    print("    RVC LAN Audio Client for Windows (0% GPU)     ")
    print("=" * 60)
    print(f"Target Mac Server: ws://{SERVER_IP}:{SERVER_PORT}/ws/voice")
    print(f"Microphone:        Searching for '{INPUT_KEYWORD}'")
    print(f"Output Cable:      Searching for '{OUTPUT_KEYWORD}'")
    print(f"Sample Rate:       {SAMPLE_RATE} Hz | Chunk: {CHUNK_SIZE}")
    print("=" * 60)

    app = QCoreApplication(sys.argv)
    ws_url = f"ws://{SERVER_IP}:{SERVER_PORT}/ws/voice"

    bridge = AudioNetworkBridge(ws_url)
    client = WindowsAudioClient(bridge)

    try:
        app.exec()
    except KeyboardInterrupt:
        logger.info("Exiting...")
    finally:
        bridge.is_running = False
        client.stop()


if __name__ == "__main__":
    main()
