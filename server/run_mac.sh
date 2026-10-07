#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "    RVC Voice Changer LAN Server (for macOS)     "
echo "=================================================="

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "[!] Error: python3 is not installed. Please install Python 3 (via python.org or 'brew install python@3.11')."
    exit 1
fi

PYTHON_BIN="python3"
echo "[*] Using Python: $($PYTHON_BIN --version)"

# Create virtual environment if missing
if [ ! -d ".venv" ]; then
    echo "[*] Creating Python virtual environment in .venv..."
    $PYTHON_BIN -m venv .venv
fi

source .venv/bin/activate

# Install / update dependencies
echo "[*] Checking and installing dependencies..."
pip install --upgrade pip --quiet
pip install -r server/requirements_mac.txt --quiet

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
exec python server/server.py
