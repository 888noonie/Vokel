# Implementation Plan

**Status:** active, foundation-aligned (May 2026).

This plan translates the current direction in `ROADMAP.md` and
`docs/foundation-direction.md` into small, measurable implementation slices.

Product invariant:

> You speak. It answers. You interrupt. It stops. It listens again.

The plan below protects that loop while making backend routing and trust state
clear.

## Scope For This Cycle

1. Connection-first UX language (`Connections`, not profiles).
2. First-class connection types: `lm_studio` and `hermes` only.
3. Visible route/tool-ownership/consent state at all times.
4. Stable Media Card primitive for web/image/gif/tool/context/consent output.
5. Pause/resume/hold voice control before larger context-input features.

## Non-Goals (Now)

- No new backend families beyond LM Studio and Hermes.
- No duplication of Hermes-owned tools in Vokel.
- No cross-device card sharing.
- No large memory-v2 or vector retrieval expansion.

## Architecture Boundaries (Must Hold)

1. Vokel owns capture, playback, interruption, routing display, consent, and audit.
2. LM Studio/Hermes own model identity, reasoning, and their own memory/tool stacks.
3. All streams remain cancellable.
4. TTS output always passes through `sanitize_for_speech`.

## Work Slices

### Slice A: Connection Model And Naming Lock

Outcome:
- UI and API consistently expose `Connection` language and explicit route state.

Code areas:
- `frontend/src/App.tsx`
- `frontend/src/components/AgentConsole.tsx`
- `src/vokel/web.py`
- `src/vokel/agent_backend.py`
- `src/vokel/engine.py`

Tasks:
- Replace user-facing "mode/profile" phrasing with "Connection" labels.
- Standardize connection payload shape in websocket messages:
  - `connection_type`: `lm_studio | hermes`
  - `route`: `local | external`
  - `tools_owner`: `vokel | hermes`
  - `interrupt_available`: `bool`
- Keep Hermes WebSocket auto-selection behavior for `ws://` endpoints.
- Ensure transcript/header badges reflect active connection and route.

Exit checks:
- User can switch between LM Studio and Hermes via one selector.
- Switching is visible and reversible.
- No backend/tool ownership ambiguity in UI copy.

### Slice B: Trust State Surface (Calm Default + Expandable Detail)

Outcome:
- Calm status line always visible; deeper routing details collapsible.

Code areas:
- `frontend/src/App.tsx`
- `frontend/src/components/TranscriptStream.tsx`
- `frontend/src/components/AgentConsole.tsx`
- `src/vokel/web.py`

Tasks:
- Add a compact connection state banner:
  - Connected to
  - Route (local/external)
  - Voice path (local TTS)
  - Tools owner
  - Interrupt availability
  - Consent state
- Add expandable details panel for gateway/session diagnostics.
- Keep state updates event-driven from server status messages.

Exit checks:
- User can answer "what am I connected to?" in under 2 seconds.
- Expanded panel shows full trust truth without cluttering default view.

### Slice C: Media Card V1 Primitive

Outcome:
- One internal card shape powers web/image/gif/tool/context/consent outputs.

Code areas:
- `frontend/src/App.tsx`
- `frontend/src/components` (card renderer + actions)
- `src/vokel/media_formatter.py`
- `src/vokel/web.py`

Tasks:
- Define shared `MediaCard` schema in frontend types aligned to foundation note.
- Map existing tool outputs into card objects with route + privacy badges.
- Support card actions (safe first set):
  - Expand
  - Save
  - Hide
  - Tag
  - Use as Context
- Ensure any speechable caption uses sanitized text only.

Exit checks:
- Web/image/gif/tool outputs render as consistent cards.
- Every card clearly shows source + route + privacy state.

### Slice D: Pause/Resume/Hold Voice Controls

Outcome:
- Voice and control path supports pause/resume/hold reliably before bigger context features.

Code areas:
- `src/vokel/web.py`
- `frontend/src/App.tsx`
- `frontend/src/components/WaveformVisualizer.tsx`
- `src/vokel/engine.py`

Tasks:
- Harden existing `pause_session` / `resume_session` command handling.
- Add phrase handling for: `pause`, `hold on`, `wait`, `continue`.
- Ensure pause blocks auto-followups and turn submission safely.
- Keep interruption correctness ahead of any latency tuning.

Exit checks:
- "pause" halts active stream cleanly.
- "continue" resumes with visible state change.
- Barge-in behavior remains correct while paused/resumed.

### Slice E: Consent + Tool Ownership Clarity

Outcome:
- Consent state is explicit; Hermes mode never implies Vokel tool execution.

Code areas:
- `src/vokel/web.py`
- `src/vokel/tools.py`
- `frontend/src/App.tsx`
- `frontend/src/components/AgentConsole.tsx`
- `docs/agent-tools.md`

Tasks:
- Keep execute consent cues/banners visible in dashboard + console.
- In Hermes connection, show tools as Hermes-owned and disable local forced tools.
- In LM Studio connection, keep Vokel ToolRegistry path available.
- Add/refresh docs language for ownership boundary.

Exit checks:
- No UI path suggests Hermes tools run inside Vokel.
- Consent armed/cancel state is visible and auditable.

## Verification Plan

Required before publishing:

- `python3 -m pytest -q`
- Web dashboard smoke run with both connections:
  - LM Studio route
  - Hermes route (HTTP and ws:// where available)

Manual checks (must pass):

1. Start in LM Studio, speak one turn, interrupt mid-response.
2. Switch to Hermes, verify route/tool ownership badges update.
3. Pause during active session, confirm status + scheduler behavior.
4. Resume and complete another turn.
5. Trigger tool/media output and verify Media Card route/privacy labels.
6. Confirm spoken output never reads raw markdown/URLs.

## Suggested Delivery Order

1. Slice A (naming/model lock)
2. Slice B (trust surface)
3. Slice D (pause/resume hardening)
4. Slice C (Media Card primitive)
5. Slice E (consent/tool-ownership polish)

## Risks And Guardrails

- Risk: UX rename churn breaks existing state wiring.
  - Guardrail: keep wire protocol changes additive, then remove old fields after UI migration.
- Risk: pause/resume races with async generation tasks.
  - Guardrail: cancel/guard at server boundary first (`web.py`) before UI actions.
- Risk: Media Card rollout fragments rendering paths.
  - Guardrail: funnel all tool/media rendering through one normalization step.

## Done Definition

This cycle is done when:

- Connection language and state are consistent across dashboard + transcript.
- LM Studio and Hermes are the only first-class connection types surfaced.
- Tool ownership and consent state are always visible.
- Media outputs use one stable card shape with safe first actions.
- Pause/resume/hold works reliably without regressing interruption correctness.
