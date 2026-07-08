# Slice 10 — Phase 2 Kickoff (Latency Mark · ffmpeg Capability · Smoke Refresh)

**Author:** Fable (auditor/architect) · **Implementer:** Composer 2.5 · **Owner:** Richard
**Branch:** `feature/musical-mode` (clean, pushed; continue on it).
**Baseline:** `5dbcee5` — 202 pytest green, npm build green.

---

## Mission

Three small, independent improvements agreed in the Slice 9 retrospective:

- **10a** — measure `phrase_start_to_beat_ms` (how long a phrase waits at the beat gate) and surface it on the Latency Scoreboard.
- **10b** — surface ffmpeg availability as a first-class capability instead of a surprise 503: grey out **Load Track** when ffmpeg is missing and say so in the Platform Capabilities panel.
- **10c** — refresh `scripts/smoke-musical-walk.sh` to cover everything shipped in Slices 7–9b (it currently stops at Slice 4 features).

The three sub-slices are independent; implement in order 10a → 10b → 10c but report once at the end (they're each too small for a full report cycle). Standing rules from the quad-slice plan still apply: **`engine.py` is not modified**, mode off = zero behavior change, no commits without Richard's OK.

---

## 10a — `phrase_start_to_beat_ms` latency mark

**Goal:** the time between a phrase arriving at `QuantizedPlaybackSink.speak()` and the beat gate releasing it. This is the musical mode's core "feel" number — worst case is one beat interval (667 ms at 90 BPM).

**Design (verified against current code):**

1. `src/vokel/audio/quantized_sink.py` — add an optional `trace` parameter:

   ```python
   def __init__(self, inner, clock, quantum="beat", trace: LatencyTrace | None = None):
   ```

   In `speak()`: mark `musical_gate_entered` on entry, and `musical_gate_opened`
   immediately before `await self.inner.speak(phrase)` **on the normal beat-release
   path only**. Do not mark on the stop-requested early return. On the
   `ClockStopped` fallback path, mark `musical_gate_opened` with a field
   `reason="clock_stopped"` so a degenerate reading is distinguishable.
   `trace is None` → no marks, sink behaves byte-identically to today.
   Import `LatencyTrace` under `TYPE_CHECKING` or accept it duck-typed — keep the
   module's zero-heavy-deps character (it already imports from `vokel.playback`,
   so a plain import from `vokel.telemetry` is also acceptable).

2. `src/vokel/telemetry.py` — add to the `summary_ms()` pairs dict
   ([telemetry.py:51](src/vokel/telemetry.py#L51)):

   ```python
   "phrase_start_to_beat": ("musical_gate_entered", "musical_gate_opened"),
   ```

   Note `summary_ms()` uses `first()` and the engine calls `trace.reset()` at each
   turn ([engine.py:231](src/vokel/engine.py#L231)), so this measures the **first
   phrase of each turn** — exactly the one that actually waits for the grid
   (subsequent phrases queue behind playback anyway). No telemetry changes beyond
   the one pair.

3. `src/vokel/web.py` — wiring. The sink is constructed at
   [web.py:1122](src/vokel/web.py#L1122) *before* the engine exists at
   [web.py:1189](src/vokel/web.py#L1189), so the trace must be created first and
   shared: instantiate `session_trace = LatencyTrace()` before the musical-mode
   block, pass `trace=session_trace` into `QuantizedPlaybackSink`, and ensure the
   same object reaches the engine via `local_engine_kwargs["trace"] = session_trace`
   (`ConversationEngine` already accepts `trace` —
   [engine.py:99](src/vokel/engine.py#L99)). **Check whether
   `local_engine_kwargs` already sets a trace or observer**; if the
   `WebSocketTraceObserver` is attached to the engine's trace after construction,
   attaching to the shared trace must preserve that. Simplest safe form: always
   create the session trace and pass it to the engine regardless of musical mode
   (identical behavior to today since the engine would create one itself), and to
   the sink only when musical mode is on.

4. `frontend/src/components/LatencyScoreboard.tsx` — one new entry in the
   `budgets` dict:

   ```ts
   phrase_start_to_beat: { target: 700, label: "Phrase to Beat", desc: "Wait at the beat gate (≤ one beat @ 90 BPM)" },
   ```

   The existing render loop handles the `_ms` suffix and the missing-value `--`
   state — no other frontend change. When musical mode is off the metric never
   appears and the tile shows `--`, which is correct.

5. `docs/latency-budget.md` — add a row to the Primary Checkpoints table:
   `phrase_start_to_beat_ms | <700 | Musical mode only: gate wait before first phrase; worst case is one beat interval, scales with BPM.`

**Tests:** unit-test the sink with a fake trace (record marks) — assert
`musical_gate_entered`/`musical_gate_opened` on the release path, **no**
`musical_gate_opened` on the stop path, and no marks when `trace=None`.
Assert the new pair appears in `summary_ms()` when both events exist.

**Acceptance:** full pytest green; npm build green; with musical mode ON, the
telemetry WS message's `metrics` includes `phrase_start_to_beat` and the tile
lights up on the scoreboard.

---

## 10b — ffmpeg as a surfaced capability

**Goal:** the UI should know at session start whether backing-track upload can
work, instead of the user discovering a 503 at upload time
([web.py:409](src/vokel/web.py#L409), raised from
[beattrack.py:53](src/vokel/audio/beattrack.py#L53)).

**Design:**

1. Backend: `ffmpeg_available = shutil.which("ffmpeg") is not None` — compute
   **per WS session** (cheap; picks up an install without a server restart, which
   matters since Richard hit exactly this mid-session in Slice 7). Ride it on the
   existing `execute_state` message, which already carries session facts like
   `musical_track_slot` ([web.py:582-588](src/vokel/web.py#L582-L588)):

   ```python
   "ffmpeg_available": ffmpeg_available,
   ```

   Do **not** add a new message type or REST endpoint — one capability path.

2. Frontend (`App.tsx`):
   - Store `ffmpegAvailable` in state from the `execute_state` handler
     (default `true` so nothing flickers disabled before the first message).
   - **Load Track** button (~[App.tsx:1856](frontend/src/App.tsx#L1856)): when
     `!ffmpegAvailable`, render disabled with reduced opacity and
     `title="Install ffmpeg to enable backing-track upload"`. Keep the hidden
     file input inert in that state.
   - Platform Capabilities section
     ([App.tsx:1535-1543](frontend/src/App.tsx#L1535-L1543)): append one line
     under the existing copy, e.g.
     `Backing-track upload: ready (ffmpeg found)` /
     `Backing-track upload: unavailable — install ffmpeg`. Match the existing
     muted `text-[11px] text-zinc-500` styling; use amber tint only for the
     unavailable case.

3. Keep the 503 path in `upload_musical_track` untouched — it remains the
   backstop for the race where ffmpeg disappears mid-session.

**Tests:** backend — monkeypatch `shutil.which` and assert the `execute_state`
payload carries `ffmpeg_available` both ways (there are existing web.py WS tests
to pattern-match). Frontend is covered by `npm run build` type-checking; no new
JS test infra.

**Acceptance:** full pytest green; npm build green; renaming ffmpeg off PATH and
starting a session greys the button and flips the capability line (Richard will
verify this in the walk).

---

## 10c — smoke-script refresh

**Goal:** `scripts/smoke-musical-walk.sh` currently documents only the Slice 1–4
walk. Bring it up to the shipped feature set. Script is print-only (heredoc) —
this is a text edit, no logic.

Extend the numbered walk with, in a sensible order after the current step 6:

- **Style selector:** switch Beat Loop ↔ Metronome mid-session; sound changes on
  the next bar.
- **Tap tempo:** tap 5+ times; BPM follows. Pause >2 s and tap again — the
  buffer resets (Slice 8 rider) rather than averaging across the gap.
- **Load track:** upload an mp3/wav (**requires ffmpeg on PATH** — note that
  the button is greyed out if missing, per 10b); track replaces the synth loop,
  respects the manual BPM. Note: uploaded track overrides the style selector by
  design.
- **Nudge:** ±25 ms drift nudge shifts phrase alignment audibly.
- **Clear track:** synth loop returns immediately (server-side DELETE, Slice 9).
- **Panels:** collapse a couple of settings panels, reload the page — collapsed
  state persists; the beat dot still pulses on the collapsed Beat panel header.

Keep the existing steps 1–9 and the WS-probe pointer at the bottom intact.
Update the box title/wording only if needed. The VSCode task
(`.vscode/tasks.json:124`) references the script by path and needs no change.

**Acceptance:** script runs (`bash scripts/smoke-musical-walk.sh`) and prints
the full walk; `set -euo pipefail` still passes.

---

## Out of scope (do not drift into these)

- Frame-counter-derived BeatClock (top Phase 2 item, but its own slice).
- librosa auto beat-detection, sidechain ducking, cadence/voice selection,
  browser-audio quantization.
- Any second "musical mode contract" doc — the README section from Slice 9 is
  the single source of truth.

## Report-back

One combined report using the quad-slice template: files touched, test counts
before/after, the exact `metrics` payload observed with musical mode on, and
anything that contradicted this plan (STOP and report rather than improvising —
especially if `local_engine_kwargs` already wires a trace differently than 10a
assumes).
