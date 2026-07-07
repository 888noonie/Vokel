# BeatSync Musical Mode — Quad-Slice Implementation Plan

**Author:** Fable (auditor/architect) · **Implementer:** Composer 2.5 · **Owner:** Richard
**Branch:** create `feature/musical-mode` off `mcp_adapter` (do not commit to `mcp_adapter` directly — it has uncommitted vision work awaiting Richard's confirmation).

---

## Mission

Add an **optional** "musical mode" to Vokel: a drift-corrected beat clock plus a playback wrapper that releases TTS phrases locked to the beat grid, with a backing beat track and a frontend toggle + beat visualizer. When the toggle is off, **zero** behavior changes anywhere.

## Architecture (decided — do not redesign)

The audit of the real codebase changed the original sketch. The engine has exactly one
playback seam: `ConversationEngine._playback_queue` → `_run_playback()` →
`await self.playback.speak(phrase)` (`src/vokel/engine.py:1203`). `PlaybackSink` is a
Protocol (`src/vokel/playback.py:35`).

Therefore the musical layer is a **sink wrapper**, not a second queue:

```
PhraseChunker → engine._playback_queue → _run_playback()
                                             │
                                             ▼
                              QuantizedPlaybackSink.speak(phrase)   ← NEW (waits for beat)
                                             │
                                             ▼
                              KokoroPlaybackSink / SubprocessSink / WebSocketPlaybackSink
```

Consequences you must preserve:
- `src/vokel/engine.py` is **NOT modified** in any slice.
- `interrupt()` already drains the queue and calls `playback.stop()`
  (`engine.py:395–419`); the wrapper's `stop()` must abort a pending beat-wait AND
  forward to the inner sink, so the existing sub-millisecond kill path keeps working.
- `wait_for_playback()` semantics (queue join) continue to hold because the wrapper's
  `speak()` simply takes longer — it still returns when the phrase finishes.

New modules (all new files, nothing else touched except where a slice says so):

| File | Purpose |
|---|---|
| `src/vokel/audio/__init__.py` | package marker, re-exports |
| `src/vokel/audio/beatclock.py` | drift-corrected clock, no Vokel imports |
| `src/vokel/audio/quantized_sink.py` | `PlaybackSink` wrapper |
| `src/vokel/audio/beattrack.py` | synthesized loop player (Slice 3) |
| `frontend/src/components/BeatIndicator.tsx` | pulse visualizer (Slice 4) |

## Global rules for Composer

1. **Do not touch**: `engine.py`, `playback.py` (except adding an export if truly needed — prefer not), `text_chunker.py`, `inference.py`, anything under `hermes/`, and the existing sanitize/tool-marker logic.
2. Match existing style: `from __future__ import annotations`, type hints, docstrings only where the code can't speak, async-first, no print-driven logic in library code (the clock may log once at start like other components do).
3. Every slice ends with: `python -m pytest tests/ -x -q` fully green (pre-existing failures, if any, noted explicitly), plus the slice's acceptance checks.
4. **Report back after each slice** using the template at the bottom. Do not start the next slice until Fable/Richard ack.
5. If you hit a blocker or the code contradicts this plan, STOP and report — don't improvise around it.

---

## Slice 1 — `BeatClock` (standalone, ~45 min)

**Goal:** A cancellation-safe, drift-corrected clock with awaitable beat/downbeat, importable with zero Vokel dependencies.

**Contract:**

```python
@dataclass(frozen=True)
class BeatInfo:
    bar: int
    beat: int            # 0..beats_per_bar-1
    at: float            # loop.time() the tick fired

class ClockStopped(Exception): ...

class BeatClock:
    def __init__(self, bpm: float = 90.0, beats_per_bar: int = 4): ...
    @property
    def running(self) -> bool: ...
    async def start(self) -> None      # idempotent
    async def stop(self) -> None       # wakes ALL pending waiters, then they raise ClockStopped
    async def wait_for_beat(self) -> BeatInfo
    async def wait_for_downbeat(self) -> BeatInfo
```

**Reference implementation** (audited; implement this, adapt naming to taste):

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class BeatInfo:
    bar: int
    beat: int
    at: float


class ClockStopped(Exception):
    """Raised to waiters when the clock stops while they are waiting."""


class BeatClock:
    """Drift-corrected musical clock. Ticks are scheduled against a single
    monotonic origin (loop.time()), so error never accumulates."""

    def __init__(self, bpm: float = 90.0, beats_per_bar: int = 4):
        if bpm <= 0:
            raise ValueError("bpm must be positive")
        self.bpm = bpm
        self.beats_per_bar = beats_per_bar
        self.beat_interval = 60.0 / bpm
        self._task: asyncio.Task[None] | None = None
        self._running = False
        # Fresh-event-per-tick pattern: set-then-clear on a shared Event is racy
        # (a waiter arriving between set() and clear() waits a full extra tick,
        # and one arriving after clear() saw nothing). Each tick swaps in a new
        # Event and sets the old one exactly once.
        self._beat_gate = asyncio.Event()
        self._downbeat_gate = asyncio.Event()
        self._last_info: BeatInfo | None = None

    @property
    def running(self) -> bool:
        return self._running

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        # Wake every pending waiter so nothing hangs forever; they observe
        # running=False and raise ClockStopped.
        self._beat_gate.set()
        self._downbeat_gate.set()

    async def wait_for_beat(self) -> BeatInfo:
        return await self._wait(self._beat_gate_ref)

    async def wait_for_downbeat(self) -> BeatInfo:
        return await self._wait(self._downbeat_gate_ref)

    # Indirection so waiters always grab the *current* gate object.
    def _beat_gate_ref(self) -> asyncio.Event:
        return self._beat_gate

    def _downbeat_gate_ref(self) -> asyncio.Event:
        return self._downbeat_gate

    async def _wait(self, gate_ref) -> BeatInfo:
        if not self._running:
            raise ClockStopped
        gate = gate_ref()
        await gate.wait()
        if not self._running or self._last_info is None:
            raise ClockStopped
        return self._last_info

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        origin = loop.time()
        tick = 0
        bar = 0
        while self._running:
            beat = tick % self.beats_per_bar
            if beat == 0 and tick > 0:
                bar += 1
            self._last_info = BeatInfo(bar=bar, beat=beat, at=loop.time())

            # Release current waiters, install fresh gates for the next tick.
            old_beat, self._beat_gate = self._beat_gate, asyncio.Event()
            old_beat.set()
            if beat == 0:
                old_down, self._downbeat_gate = self._downbeat_gate, asyncio.Event()
                old_down.set()

            tick += 1
            next_deadline = origin + tick * self.beat_interval
            delay = next_deadline - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            # If we're late, fire immediately; scheduling from `origin` means we
            # re-converge instead of drifting.
```

**Tests** (`tests/test_beatclock.py`, use high BPM like 6000 so tests run in ms; follow the async test style used in `tests/test_engine.py`):
1. Ticks land on the grid: collect 8 `wait_for_beat()` timestamps at bpm=6000, assert inter-tick spacing within ±30% of interval and **cumulative** drift over 8 ticks < 1 interval.
2. `wait_for_downbeat()` fires every `beats_per_bar` beats with correct `BeatInfo.beat == 0` and incrementing `bar`.
3. `stop()` while a waiter is pending → waiter raises `ClockStopped` promptly (guard with `asyncio.wait_for(..., 1.0)`).
4. `start()` twice is a no-op; `wait_for_beat()` on a never-started clock raises `ClockStopped`.
5. `bpm <= 0` raises `ValueError`.

**Acceptance:** tests green; `python -c "from vokel.audio.beatclock import BeatClock"` works; file imports nothing from `vokel.*`.

→ **REPORT BACK** before Slice 2.

---

## Slice 2 — `QuantizedPlaybackSink` (~45 min)

**Goal:** A `PlaybackSink`-conformant wrapper that delays each `speak()` until the next grid point, and whose `stop()` both aborts a pending wait *and* stops the inner sink.

**Contract:**

```python
Quantum = Literal["beat", "bar"]

class QuantizedPlaybackSink:
    def __init__(self, inner: PlaybackSink, clock: BeatClock, quantum: Quantum = "beat"): ...
    async def speak(self, phrase: str) -> None
    async def stop(self) -> None
```

**Critical behaviors (this is where Fable will audit hardest):**

1. `speak()` waits for the next beat (or downbeat if `quantum="bar"`), **then** delegates `await inner.speak(phrase)`. Quantize the *start* of each phrase only — do not try to pace audio within the phrase (that's Phase 2 cadence work).
2. **The stop race:** `engine.interrupt()` drains the queue then calls `playback.stop()`. At that moment a phrase may be parked inside `speak()` waiting up to a whole beat/bar. `stop()` must make that pending `speak()` return immediately *without* speaking. Implement with a race:

```python
async def speak(self, phrase: str) -> None:
    self._stop_requested.clear()
    wait_gate = (
        self.clock.wait_for_downbeat() if self.quantum == "bar"
        else self.clock.wait_for_beat()
    )
    beat_task = asyncio.ensure_future(wait_gate)
    stop_task = asyncio.ensure_future(self._stop_requested.wait())
    try:
        done, pending = await asyncio.wait(
            {beat_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        for t in (beat_task, stop_task):
            if not t.done():
                t.cancel()
    if stop_task in done or self._stop_requested.is_set():
        return                      # interrupted while parked — say nothing
    try:
        beat_task.result()          # re-raise ClockStopped if clock died
    except ClockStopped:
        await self.inner.speak(phrase)   # clock gone → degrade gracefully, still speak
        return
    await self.inner.speak(phrase)

async def stop(self) -> None:
    self._stop_requested.set()
    await self.inner.stop()
```

3. **Graceful degradation:** if the clock is stopped/never started, `speak()` must still speak (unquantized). Musical mode failing must never mute Vokel.
4. No polling loops anywhere. No buffering — the engine's `_playback_queue` is already the buffer.

**Tests** (`tests/test_quantized_sink.py`, use a `FakeSink` recording `(phrase, loop.time())` and a fast clock):
1. Phrases pushed mid-beat get released on grid points (timestamps align to clock ticks within tolerance).
2. `stop()` while a phrase is parked → inner sink never receives it, and inner `stop()` was called.
3. Clock stopped → phrase still speaks immediately (degradation path).
4. Multiple sequential `speak()` calls each wait for their own grid point (simulate `_run_playback`'s serial loop).
5. Protocol conformance: instance satisfies `PlaybackSink` usage (`speak`, `stop` awaitable).

**Acceptance:** tests green; run an integration check against the real engine in a test: build `ConversationEngine` exactly like `tests/test_engine.py` does but with `QuantizedPlaybackSink(ConsolePlaybackSink(), clock)`, submit a canned turn, call `interrupt()`, assert no hang and no stray phrase.

→ **REPORT BACK** before Slice 3.

---

## Slice 3 — Backend wiring: beat track + web toggle + WS beat events (~60 min)

**Goal:** `musical_mode` + `bpm` accepted in the WebSocket `start` message; sink gets wrapped; a backing beat plays; the browser receives beat events for visualization. Engine untouched.

**3a. `src/vokel/audio/beattrack.py` — synthesized loop (no audio assets, no licensing):**

```python
class BeatTrackPlayer:
    def __init__(self, bpm: float, beats_per_bar: int = 4, gain: float = 0.35): ...
    async def start(self) -> None
    async def stop(self) -> None
```

- Pre-render exactly one bar of audio with numpy at 24000 Hz (matches Kokoro output ballpark): kick (decaying 60 Hz sine, ~120 ms) on every beat, closed hat (short filtered noise burst, ~30 ms) on the off-beats. Keep `gain` low — TTS must sit on top.
- Play it looped via a `sounddevice.OutputStream` callback that reads from the pre-rendered buffer with a wrapping cursor. The callback is sample-accurate; do not drive it from `BeatClock`.
- **Known accepted limitation (document in a comment):** the asyncio `BeatClock` and the audio hardware clock will drift apart very slowly; fine for Phase 1. Phase 2 derives the clock from the stream's frame counter.
- **RISK TO VERIFY FIRST (do this before writing the rest of 3a):** `KokoroPlaybackSink` uses `sd.play()` (default stream) while `BeatTrackPlayer` holds its own `OutputStream`. Verify two simultaneous PortAudio streams play on this machine with a 10-line scratch script. If they don't: fall back to looping via a `ffplay -loop 0` subprocess (write the rendered bar to a temp WAV) and note it in your report.

**3b. `src/vokel/web.py` wiring (the ONLY existing backend file you modify):**

- In the `start` message handling (near `web.py:939` where `playback_backend` is read):
  ```python
  musical_mode = bool(data.get("musical_mode", False))
  musical_bpm = min(160.0, max(60.0, float(data.get("musical_bpm", 90.0))))
  ```
- When `musical_mode` and the session is local playback (kokoro/spd-say — **not** the Hermes/browser `WebSocketPlaybackSink` path, leave that unwrapped for Phase 1):
  1. create `BeatClock(bpm=musical_bpm)` + `BeatTrackPlayer(bpm=musical_bpm)`, start both,
  2. wrap: `sink = QuantizedPlaybackSink(sink, clock, quantum="beat")`,
  3. spawn one task that forwards ticks to the browser:
     ```python
     async def forward_beats() -> None:
         try:
             while True:
                 info = await clock.wait_for_beat()
                 await send_json({"type": "beat", "bar": info.bar,
                                  "beat": info.beat, "bpm": musical_bpm})
         except ClockStopped:
             pass
     ```
- Teardown: wherever the session's engine/tasks are cleaned up on stop/disconnect, also `await clock.stop()`, `await beat_track.stop()`, cancel the forwarder. Grep how the session currently tears down (`close()` / disconnect handling around the receive loop) and mirror it exactly.
- Session `interrupt` needs no changes — it flows through `playback.stop()` into the wrapper.

**Tests** (`tests/test_web.py` style — look at how existing WS tests fake the session):
1. `start` with `musical_mode: true` yields a wrapped sink (assert type) and emits at least 2 `beat` messages.
2. `start` without the flag → sink is NOT wrapped (regression guard).
3. `musical_bpm` out of range is clamped.
4. Disconnect/stop tears the clock down (no pending tasks warning; `clock.running is False`).

**Acceptance:** tests green; manual smoke: start backend (`scripts/start-backend.sh`), start a local kokoro session with `musical_mode`, ask a question, hear phrases land on the grid over the beat, press interrupt — speech dies instantly, beat keeps playing.

→ **REPORT BACK** before Slice 4.

---

## Slice 4 — Frontend UX + docs + full regression (~60 min)

**Goal:** Toggle + BPM in settings, a beat visualizer with flair, prefs persisted, docs updated, everything green.

**4a. `frontend/src/App.tsx`:**
- New state: `musicalMode: boolean` (default false), `musicalBpm: number` (default 90, range 60–160 step 5).
- Persist both alongside the existing voice prefs (`vokel.voicePrefs.v1` pattern, `App.tsx:97` — either extend that object or add `vokel.musicalPrefs.v1`; match whichever is cleaner with the existing `loadJson` usage).
- Include `musical_mode` and `musical_bpm` in the `start` WS payload (same block that already sends `voice`, `tts_speed`, `vision_voice_enabled` — around `App.tsx:622`).
- Handle the new `beat` WS message in the existing `switch (data.type)` — store `{bar, beat, bpm, receivedAt}` in state (keep it cheap; this fires up to ~2.7×/sec at 160 BPM).

**4b. `frontend/src/components/BeatIndicator.tsx` (new):**
- Renders only when `musicalMode` && session active.
- Four dots (one per beat in the bar); the active beat pulses (scale + glow via CSS transition, ~120 ms decay), downbeat gets a stronger pulse and an accent color.
- Show `♩ 90 BPM · BAR 12` and a small **IN POCKET** badge that lights while beats are arriving (fade it if no beat event for >1.5× interval — that's the "clock lost" signal).
- Style with the existing CSS conventions in `frontend/src/index.css` / component patterns (look at `LiveVisionPanel.tsx` for how panels are structured). Subtle flair is the brief: it should feel like Vokel, not a DAW.
- Settings UI: add the toggle + BPM slider in the settings workspace tab next to the voice/TTS controls.

**4c. Docs + regression:**
- Add a short "Musical mode (experimental)" section to `README.md`: what it is, the toggle, local-playback-only for now, Phase 2 items (sidechain ducking, cadence TTS, browser-audio path, audio-clock-derived timing).
- Run the FULL suite: `python -m pytest tests/ -q` and the frontend build (`npm run build` in `frontend/` — check `package.json` for the exact script). Both must pass.
- Do NOT commit. Leave the working tree ready for Fable's final audit and Richard's call.

**Acceptance:** toggle off → UI and payloads byte-identical to before (verify the `start` payload includes the new keys with safe defaults, or omits them — backend defaults to off either way); toggle on → beat plays, dots pulse in sync, phrases land on grid, interrupt still feels instant.

→ **FINAL REPORT** for Fable's audit.

---

## Report-back template (use after every slice)

```
SLICE N REPORT — <title>
Status: complete | blocked
Files created: ...
Files modified: ...
Tests: <command run> → <pass/fail counts>
Deviations from plan: <none, or exactly what and why>
Risks verified: <e.g. dual PortAudio streams: OK / fell back to ffplay>
Open questions for Fable/Richard: ...
```

## Out of scope (Phase 2 — do not start)

- Sidechain ducking / mixer layer
- Syllable-level cadence TTS (prosody-to-grid)
- Musical mode for the Hermes / browser-audio (`WebSocketPlaybackSink`) path
- Variable BPM / tempo changes mid-session
- Beat detection from arbitrary audio files
