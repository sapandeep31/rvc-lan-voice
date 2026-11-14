#!/usr/bin/env bash
# Start the voice server on a Mac. Port comes from config.json ("server" -> "port") unless PORT is set:
#   PORT=8010 ./setup/mac/start.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [ ! -x ".venv/bin/python" ]; then
    echo "[!] Not installed yet. Run:  ./setup/mac/install.sh"
    exit 1
fi

PORT="${PORT:-$(.venv/bin/python -c "import json;print(json.load(open('config.json'))['server']['port'])")}"
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN &> /dev/null; then
    echo "[!] Port $PORT is already in use. Pick another, e.g.:  PORT=$((PORT + 10)) ./setup/mac/start.sh"
    exit 1
fi

IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "127.0.0.1")
echo "=================================================="
echo "  Voice server starting (model loads in ~30 s)"
echo "  On the PC, set in config.json -> client:"
echo "      \"server_ip\": \"$IP\",  \"server_port\": $PORT"
echo "  Control panel (any device on the Wi-Fi):"
echo "      http://$IP:$PORT/panel"
echo "  Stop with Ctrl+C"
echo "=================================================="
exec .venv/bin/python -m uvicorn --app-dir server server:app --host 0.0.0.0 --port "$PORT"
