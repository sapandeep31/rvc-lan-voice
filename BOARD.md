# Board

Format: `- [mac|pc] task — files it touches`

## Up next

## In progress

## Done
- [pc] Run client against the Mac server, report latency/errors/audio quality — client/, config.json "client" (server_ip local only)
  - 2026-11-14 handed to user: client follows panel chunk_size, Desktop shortcuts for run_client.bat / mic_check.bat / panel. Discord/games mic = "CABLE Output (VB-Audio Virtual Cable)".
  - done: client/.venv (Py 3.12, numpy/websockets/PySide6), run_client.bat uses it; Qt sees "Microphone (TONOR TC30)" + "CABLE Input (VB-Audio Virtual Cable)" at 48 kHz float.
  - 2026-11-14 e2e vs mac:8010: connected, 40 s live run with no errors, real-time throughput. LAN round trip median 89 ms / max 106 (budget 149). Est. mouth-to-ear ~250–300 ms. Pending: user listening check for quality/conversion.
  - client now pipelined + adaptive jitter buffer (trims only in pauses), mic_check.bat monitors to headphones, --input-wav/--record test mode. Chunk 40 on Wi-Fi: ~200–215 ms, ~1 underrun/90 s.
  - "cut syllables" was 2 clients sharing server audio history (mac fixed per-connection buffers). Re-measured: 0–2 % of phrase edges cut, solo and with 2 clients.
- [mac] Get server running on Apple Silicon (venv, deps, pretrain, MPS model load) and smoke-test /health + /ws/voice; then send PC the LAN IP:port — server/, config.json "server"
  - a73ef4f: Py 3.12.3 via uv, voiceconversion --no-deps + CPU onnxruntime, awaited weight download, single-thread faiss (segfault fix). E2E from PC: median ~89 ms round trip vs 149 ms chunk budget, no errors.

- [mac] Lower latency + control panel: adaptive frame size, fcpe + fp16 (~47 ms/chunk), crossfade 0.05 s, /panel with synced sliders, noise-gate calibration, concurrent-client crash fix — server/, config.json "server" (cc8214c, 9fdfbb2)

## Notes / handoffs
- Mac server runs on port 8010 on this Mac (Docker holds 8000); committed config still says 8000, override locally.
- chunk_size is now client-only (server adapts to any multiple of 128). Today: chunk 40 (~216 ms mouth-to-Discord) on Wi-Fi; chunk 32 (~150-170 ms) once the Mac's AWDL spikes are gone (sudo ifconfig awdl0 down, user's call).
