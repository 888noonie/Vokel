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
 10. Style selector: switch Beat Loop ↔ Metronome mid-session; sound changes on the next bar
 11. Tap tempo: tap 5+ times; BPM follows. Pause >2 s and tap again — the buffer resets
     (Slice 8 rider) rather than averaging across the gap
 12. Load track: upload an mp3/wav (requires ffmpeg on PATH — Load Track is greyed out if
     missing). Uploaded track replaces the synth loop and respects the manual BPM. Note:
     uploaded track overrides the style selector by design
 13. Nudge: ±25 ms drift nudge shifts phrase alignment audibly
 14. Clear track: synth loop returns immediately (server-side DELETE, Slice 9)
 15. Panels: collapse a couple of settings panels, reload the page — collapsed state
     persists; the beat dot still pulses on the collapsed Beat panel header

Quick automated check (no mic needed):
  Run Task → "Vokel: Probe Musical Mode (WS beats)"

EOF