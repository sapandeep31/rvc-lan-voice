#!/usr/bin/env bash
# One-time setup of the voice server on a Mac (Apple Silicon). Safe to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if ! command -v uv &> /dev/null; then
    echo "[!] uv is required. Install it with:  brew install uv"
    echo "    (or: curl -LsSf https://astral.sh/uv/install.sh | sh)"
    exit 1
fi

# voiceconversion 0.0.8 installs only on exactly Python 3.12.3; uv fetches it without touching system Python
if [ ! -x ".venv/bin/python" ] || [ "$(.venv/bin/python -c 'import platform;print(platform.python_version())')" != "3.12.3" ]; then
    echo "[*] Creating Python 3.12.3 environment in .venv ..."
    rm -rf .venv
    uv venv --python 3.12.3 .venv
fi

echo "[*] Installing server dependencies (first run downloads ~1 GB) ..."
uv pip install --quiet --python .venv/bin/python -r server/requirements_mac.txt
# --no-deps: voiceconversion requires onnxruntime-gpu, which has no macOS build
uv pip install --quiet --python .venv/bin/python --no-deps "voiceconversion==0.0.8"

echo "[*] Downloading voice models ..."
.venv/bin/python setup/download_models.py

.venv/bin/python -c "import torch; print('[*] Apple GPU (MPS):', 'available' if torch.backends.mps.is_available() else 'NOT available, will run on CPU (slow)')"
echo ""
echo "[OK] Installed. Start the server with:  ./setup/mac/start.sh"
