# 🎙️ RVC LAN Voice Changer (Mac Host + Windows Client)

High-performance, ultra-low latency real-time AI voice conversion system designed to **offload 100% of GPU compute from your Windows gaming PC to your Mac** over local Wi-Fi.

---

## 🚀 Why This Exists

Running RVC models locally alongside heavy competitive games like **Valorant** on a 4GB GPU (like the GTX 1650) causes GPU frame drops, in-game stutter, and audio static.

By offloading the AI to your **Mac** (Apple Silicon M1/M2/M3/M4 with Metal/MPS acceleration):
- **Windows GPU Usage:** **`0%`** (100% of your GTX 1650 is dedicated to Valorant for max FPS).
- **Windows RAM Usage:** **`< 50 MB`** (vs 3 GB previously).
- **LAN Latency:** **`~1–2 ms`** over local 5GHz Wi-Fi.
- **Mac AI Inference:** **`~30–50 ms`** on Apple Silicon Metal GPU.
- **Total Round-Trip Voice Delay:** **`~180–200 ms`** (imperceptible in voice chat).

---

## 📐 Architecture

```
[Windows PC]                                    [Mac (Same Wi-Fi)]
TONOR TC30 Mic                                  Apple Silicon (MPS)
      │                                                │
      ▼                                                ▼
client_windows.py ──(Binary WebSocket)──► server.py (FastAPI)
(0% GPU, <0.1% CPU)   LAN Latency: 1ms     RVC Kafka Inference
      ▲                                                │
      │ ◄──────────(Converted Audio)───────────────────┘
      │
      ▼
CABLE Input ──► CABLE Output ──► Discord / Valorant / OBS
```

---

## 🛠️ Quick Start Guide

### Step 1: Start the Server on your Mac

1. Copy or clone this folder onto your Mac.
2. Open **Terminal** in this folder and make the script executable:
   ```bash
   chmod +x server/run_mac.sh
   ```
3. Run the one-click launcher:
   ```bash
   ./server/run_mac.sh
   ```
4. The script will automatically:
   - Create a Python virtual environment (`.venv`).
   - Install required dependencies (`fastapi`, `uvicorn`, `torch`, `voiceconversion`, etc.).
   - Verify Apple Silicon **Metal (MPS)** hardware acceleration.
   - Display your Mac's local Wi-Fi IP address (e.g. `192.168.1.105`).
   - Start the FastAPI WebSocket server on port `8000`.

---

### Step 2: Configure the Client on Windows

1. Open `config.json` on your Windows PC.
2. In the `"client"` section, set `"server_ip"` to your Mac's IP address:
   ```json
   {
     "client": {
       "server_ip": "192.168.1.105",
       "server_port": 8000,
       "sample_rate": 48000,
       "chunk_size": 56,
       "input_device_keyword": "TONOR",
       "output_device_keyword": "CABLE Input",
       "output_gain": 1.5
     }
   }
   ```

---

### Step 3: Run the Windows Client

1. Double-click **`client/run_client.bat`** (or run `python client/client_windows.py`).
2. You will see:
   ```
   ============================================================
       RVC LAN Audio Client for Windows (0% GPU)
   ============================================================
   Selected Mic:    Microphone (TONOR TC30 Audio Device)
   Selected Output: CABLE Input (VB-Audio Virtual Cable)
   Connected to:    ws://192.168.1.105:8000/ws/voice
   ============================================================
   ```
3. Speak into your TONOR microphone!
4. In **Valorant** / **Discord**:
   - **Input Device (Mic):** `CABLE Output (VB-Audio Virtual Cable)`
   - **Output Device:** Your headphones

---

## 🎛️ Live Voice Adjustments

You can adjust your voice settings (pitch, index, formant) without restarting the server:

### Via Browser or Curl:
Send a quick HTTP POST to your Mac server:
```bash
curl -X POST http://<MAC_IP>:8000/settings \
  -H "Content-Type: application/json" \
  -d '{"pitch": 7, "formant_shift": 0.7, "index_ratio": 0.4}'
```
- **Pitch:** `+7` for mature feminine ("mommy") voice.
- **Index Ratio:** `0.35`–`0.45` for crisp pronunciation without robotic distortion.
- **Output Gain:** Set in `config.json` (`"output_gain": 1.5`–`2.0`) for loud, punchy audio.

---

## 📁 Project Structure

```
.
├── config.json               # Central network & audio configuration
├── README.md                 # Documentation
├── models/
│   └── Kafka/                # Pre-loaded Kafka RVC weights & .index
├── server/
│   ├── server.py             # FastAPI WebSocket audio server (for Mac)
│   ├── requirements_mac.txt  # Dependencies for macOS
│   └── run_mac.sh            # One-click startup script for macOS
└── client/
    ├── client_windows.py     # 0% GPU audio bridge (for Windows)
    └── run_client.bat        # One-click Windows launcher
```
