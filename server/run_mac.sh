#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "    RVC Voice Changer LAN Server (for macOS)     "
echo "=================================================="

# voiceconversion 0.0.8 only installs on exactly Python 3.12.3; uv fetches it without touching system Python
if ! command -v uv &> /dev/null; then
    echo "[!] Error: uv is required (brew install uv). It provides the exact Python 3.12.3 voiceconversion needs."
    exit 1
fi

if [ ! -x ".venv/bin/python" ] || [ "$(.venv/bin/python -c 'import platform;print(platform.python_version())')" != "3.12.3" ]; then
    echo "[*] Creating Python 3.12.3 virtual environment in .venv..."
    rm -rf .venv
    uv venv --python 3.12.3 .venv
fi

source .venv/bin/activate

echo "[*] Checking and installing dependencies..."
uv pip install --quiet -r server/requirements_mac.txt
# --no-deps: voiceconversion requires onnxruntime-gpu, which has no macOS wheel
uv pip install --quiet --no-deps "voiceconversion==0.0.8"

# Check hardware acceleration
echo "[*] Checking hardware acceleration..."
python -c "
import torch
if torch.backends.mps.is_available():
    print('  -> Apple Silicon GPU (MPS) Acceleration: ACTIVE')
else:
    print('  -> CPU Acceleration: ACTIVE')
"

# Find local Wi-Fi IP address
WIFI_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "127.0.0.1")

echo ""
echo "=================================================="
echo "  [SUCCESS] Server is ready to start!"
echo "  Your Mac's Wi-Fi IP Address is: $WIFI_IP"
echo "  Set this IP in your Windows config.json: \"server_ip\": \"$WIFI_IP\""
echo "=================================================="
echo ""

# Start FastAPI server
PORT=$(python -c "import json;print(json.load(open('config.json'))['server']['port'])")
echo "  Port: $PORT (if something else holds it, run: python -m uvicorn --app-dir server server:app --host 0.0.0.0 --port <other>)"
exec python server/server.py
