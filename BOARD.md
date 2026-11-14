# Board

Format: `- [mac|pc] task — files it touches`

## Up next

## In progress
- [pc] Run client against the Mac server, report latency/errors/audio quality — client/, config.json "client" (server_ip local only)
  - done: client/.venv (Py 3.12, numpy/websockets/PySide6), run_client.bat uses it; Qt sees "Microphone (TONOR TC30)" + "CABLE Input (VB-Audio Virtual Cable)" at 48 kHz float. Waiting on Mac server.
- [mac] Get server running on Apple Silicon (venv, deps, pretrain, MPS model load) and smoke-test /health + /ws/voice; then send PC the LAN IP:port — server/, config.json "server"

## Done

## Notes / handoffs
