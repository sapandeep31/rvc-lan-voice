"""Record the user's raw voice with on-screen prompts (for pitch/voice calibration).

Shows a big window: countdown, then "BOLO" with a line to read, a live mic level bar and time left.
Saves mono 16-bit 48 kHz WAV from the configured mic (TONOR). No server involved.

    client\\.venv\\Scripts\\python.exe client\\voice_recorder.py --out my_voice.wav --seconds 45
"""

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices
from PySide6.QtWidgets import QApplication, QLabel, QProgressBar, QVBoxLayout, QWidget

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("client", {}) if CONFIG_PATH.exists() else {}
SR = 48000

PROMPTS = [
    ("Normal awaaz mein:", "Hello everyone, welcome back to the stream!"),
    ("Pyaar se, mummy ki tarah:", "Beta, khana kha liya? Chalo jaldi aao, thanda ho jayega."),
    ("Sawaal poochho:", "Homework kab karoge? Kitni baar bolna padega?"),
    ("Thoda zor se / daant ke:", "Abhi ke abhi phone rakho aur so jao!"),
    ("Ek baar hasso:", "Haha, arey tum bhi na... bahut natkhat ho!"),
    ("Lamba sentence, aaram se:", "Aaj market gayi thi, sabzi li, doodh liya, aur tumhare liye chocolate bhi laayi."),
    ("Chup raho (pause)...", "  "),
    ("Dheere, soft awaaz mein:", "Shh... so jao beta, kal school hai. Good night."),
    ("Normal, game wali baat:", "Okay team, left side jao, main cover kar rahi hoon."),
]


class Recorder(QWidget):
    def __init__(self, out_path: str, seconds: float):
        super().__init__()
        self.out_path, self.total = out_path, seconds
        self.setWindowTitle("Voice recording")
        self.setStyleSheet("background:#111; color:#eee;")
        self.resize(900, 520)

        self.state_lbl = QLabel("Taiyaar ho jao...")
        self.state_lbl.setFont(QFont("Segoe UI", 54, QFont.Weight.Bold))
        self.hint_lbl = QLabel("")
        self.hint_lbl.setFont(QFont("Segoe UI", 20))
        self.hint_lbl.setStyleSheet("color:#aaa;")
        self.line_lbl = QLabel("")
        self.line_lbl.setFont(QFont("Segoe UI", 30, QFont.Weight.DemiBold))
        self.line_lbl.setWordWrap(True)
        self.level = QProgressBar()
        self.level.setRange(0, 100)
        self.level.setTextVisible(False)
        self.level.setFixedHeight(22)
        self.level.setStyleSheet("QProgressBar{background:#333;border:none} QProgressBar::chunk{background:#3c3}")
        self.time_lbl = QLabel("")
        self.time_lbl.setFont(QFont("Segoe UI", 18))
        for w in (self.state_lbl, self.hint_lbl, self.line_lbl, self.time_lbl):
            w.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 30, 40, 30)
        lay.addWidget(self.state_lbl)
        lay.addWidget(self.hint_lbl)
        lay.addWidget(self.line_lbl, 1)
        lay.addWidget(QLabel("Mic level:"))
        lay.addWidget(self.level)
        lay.addWidget(self.time_lbl)

        kw = cfg.get("input_device_keyword", "TONOR").lower()
        dev = next((d for d in QMediaDevices.audioInputs() if kw in d.description().lower()),
                   QMediaDevices.defaultAudioInput())
        self.hint_lbl.setText(f"Mic: {dev.description()}")
        fmt = dev.preferredFormat()
        fmt.setSampleRate(SR)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Float)
        self.ch = fmt.channelCount()
        self.src = QAudioSource(dev, fmt)
        self.io = self.src.start()
        self.rem = b""
        self.chunks = []
        self.recording = False

        self.t0 = time.perf_counter()
        self.countdown = 3
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(30)

    def read_mic(self):
        raw = self.rem + bytes(self.io.readAll())
        fb = 4 * self.ch
        use = len(raw) - len(raw) % fb
        self.rem = raw[use:]
        if not use:
            return np.zeros(0, np.float32)
        x = np.frombuffer(raw[:use], np.float32).reshape(-1, self.ch).mean(axis=1)
        return x.astype(np.float32)

    def tick(self):
        x = self.read_mic()
        if len(x):
            db = 20 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-6))
            self.level.setValue(int(np.clip((db + 60) / 60 * 100, 0, 100)))
        el = time.perf_counter() - self.t0
        if not self.recording:
            left = 3 - int(el)
            if left > 0:
                self.state_lbl.setText(f"Taiyaar ho jao... {left}")
                self.state_lbl.setStyleSheet("color:#fc3;")
                self.line_lbl.setText(PROMPTS[0][1])
                return
            self.recording, self.t0 = True, time.perf_counter()
            self.state_lbl.setStyleSheet("color:#f44;")
            return
        self.chunks.append(x)
        left = self.total - el
        if left <= 0:
            self.finish()
            return
        i = min(int(el / (self.total / len(PROMPTS))), len(PROMPTS) - 1)
        hint, line = PROMPTS[i]
        self.state_lbl.setText("● BOLO" if line.strip() else "● chup raho")
        self.hint_lbl.setText(hint)
        self.line_lbl.setText(line)
        self.time_lbl.setText(f"{left:.0f} sec baaki")

    def finish(self):
        self.timer.stop()
        self.src.stop()
        data = np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)
        with wave.open(self.out_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((np.clip(data, -1, 1) * 32767).astype(np.int16).tobytes())
        self.state_lbl.setText("✔ Ho gaya!")
        self.state_lbl.setStyleSheet("color:#3c3;")
        self.hint_lbl.setText(f"{len(data) / SR:.0f} sec saved")
        self.line_lbl.setText("Shukriya! Ye window band ho rahi hai...")
        self.time_lbl.setText("")
        print(f"saved {self.out_path} ({len(data) / SR:.1f} s)", flush=True)
        QTimer.singleShot(2500, QApplication.instance().quit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="my_voice.wav")
    ap.add_argument("--seconds", type=float, default=45)
    a = ap.parse_args()
    app = QApplication(sys.argv[:1])
    r = Recorder(a.out, a.seconds)
    r.show()
    r.raise_()
    r.activateWindow()
    app.exec()


if __name__ == "__main__":
    main()
