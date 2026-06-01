# Vokel Session Handover - Agent Boundary And Voice-Loop Hardening

**Date:** 2026-05-28  
**Session focus:** Make the LM Studio and Hermes paths cleaner, safer, and easier to evolve into a plugin-style intelligence layer.  
**Validation:** `.venv/bin/python -m pytest -q` -> `96 passed`

## Summary

This session tightened the boundary between the Vokel engine and the intelligence backends that sit behind it.

The useful framing for tomorrow:

- **Vokel engine:** owns voice capture, playback, interruption, routing, consent, audit, transcript display, Media Cards, and speech sanitization.
- **LM Studio intelligence plugin:** local OpenAI-compatible model path. Vokel may provide deterministic local tools when enabled.
- **Hermes intelligence plugin:** external agent path. Hermes owns reasoning, memory, and tools; Vokel exposes state, cues, cancellation, consent, and audit.

The implementation is not a full plugin system yet, but the code now points in that direction through backend capabilities, explicit tool activity events, and cleaner routing signals.

## What Changed

### Backend Boundary

- Added `AgentBackendCapabilities` so backends can advertise ownership and behavior.
- Added `ToolActivityEvent` for tool lifecycle signals without leaking fake assistant text into the transcript.
- Extended the shared tool activity contract so backends are told not to print serialized tool calls and not to infer new image/GIF/web searches from praise, thanks, or brief feedback.
- Hermes HTTP and WebSocket clients now surface tool activity as structured events.
- Hermes startup now checks that the configured model can actually stream text, not just that the gateway is reachable.
- Built-in/local mode keeps Vokel-owned deterministic tools behind the existing `ToolRegistry`.

### Voice-Loop Hardening

- Suppressed local-model leaked tool syntax such as:
  - `[tool_call:search_image]`
  - `<|tool_call>call:search_image{...}<tool_call|>`
- Added a TTS sanitizer fallback so leaked tool syntax is stripped even if it reaches speech cleanup.
- Improved phrase chunking so streamed URLs, media paths, and Markdown image links do not get split mid-link.
- Preserved the key rule: tool/media display can be rich, but speech stays calm and caption-like.

### Dashboard Polish

- Added an active action indicator for searches, image fetches, GIF fetches, generation, and speech.
- Added a near-transcript session cockpit with start/stop/pause/resume/barge-in/mute/reset controls.
- Improved transcript media inference so Markdown file links, image URLs, GIF URLs, and external media links promote to cards instead of noisy spoken or visible raw links.

## Current State

The repo is ready for the next architecture pass:

- Tests pass in the project venv.
- The session export that exposed the local-model serialized tool-call leak was read and deleted from `data/`.
- The direction is now clear enough to start extracting a universal backend/plugin vocabulary tomorrow.

## Tomorrow's Thread

Recommended next slice:

1. Rename the mental model from backend mode to **intelligence plugin** in docs and UI copy where it helps.
2. Introduce a small `ConnectionDefinition` or `IntelligencePlugin` descriptor around the existing LM Studio and Hermes paths.
3. Keep `AgentBackend` as the runtime streaming protocol.
4. Move plugin metadata out of scattered UI/backend conditionals:
   - display name
   - route: local/external
   - tool ownership
   - supports cancellation
   - supports session reset
   - emits tool activity
5. Make the UI render those descriptors instead of hardcoding LM Studio/Hermes branching everywhere.

The product test remains unchanged:

> You speak. It answers. You interrupt. It stops. It listens again. No button.

