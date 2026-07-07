#!/usr/bin/env bash
# Print the manual musical-mode smoke walk (Jan + Vokel should already be running).
set -euo pipefail

PORT="${VOKEL_PORT:-8000}"
LLM_PORT="${JAN_PORT:-6767}"

cat <<EOF

╔══════════════════════════════════════════════════════════════════╗
║           Vokel Musical Mode — Manual Smoke Walk                 ║
╚══════════════════════════════════════════════════════════════════╝

Servers (started by this compound task):
  • Jan LLM     → http://127.0.0.1:${LLM_PORT}/v1
  • Vokel UI    → http://127.0.0.1:${PORT}

Walk:
  1. Open http://127.0.0.1:${PORT}
  2. Settings → Audio route: Local (not Browser)
  3. Settings → Playback: Kokoro (musical mode is local kokoro/spd-say only)
  4. Connect / start session
  5. Settings → Musical Mode ON, set BPM (60–160)
  6. With musical mode ON: you should hear a low kick/hat loop
  7. Ask a short question — TTS phrases should land on the beat grid
  8. Interrupt (barge-in) — speech stops instantly; beat keeps playing
  9. Stop session — beat and clock should tear down cleanly

Quick automated check (no mic needed):
  Run Task → "Vokel: Probe Musical Mode (WS beats)"

EOF