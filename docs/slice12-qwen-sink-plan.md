# Slice 12: Qwen3-TTS playback backend ("performance voice")

**Implementer:** Composer. **Auditor:** Fable. Report back before any commit; note every
deviation explicitly.

**Branch:** created AFTER Richard's merge consolidation (he wants scattered builds
unified first). Target: `feature/qwen-voice` off the consolidated branch. Do not start
until Richard confirms the base.

## Context — read this before designing

Richard's listen gate PASSED on Qwen3-TTS-12Hz-0.6B-CustomVoice ("must have Qwen —
it has the emotion and timing"). Kokoro stays the snappy conversational default;
Qwen becomes an opt-in performance voice. The bench (`benchmarks/qwen_tts_bench.py`
on `local_tts_bench`, plus `bench-out/*.json`) measured on the RTX 4050 6GB:

- VRAM: 2.29 GB whole-GPU after load (bf16). Load 1.4–2.9 s.
- RTF ≈ 0.93 constant across phrase lengths → synth wall time ≈ 0.95 × audio
  duration. First-audio latency = duration of the first phrase (~2 s for a short
  phrase, ~4 s for a full rap bar).
- **No streaming exists in the offline `qwen-tts` package** (verified in source;
  the 97 ms claim is their vLLM serving stack). Do not go looking for a streaming
  API. Do not attempt to abort an in-flight `generate_custom_voice` call.
- **Ultra-short input hazard:** "Yo." produced 8 s of hallucinated audio. Inputs
  below ~12 chars are dangerous.
- Speaker default: Ryan (English). `instruct` conditioning works (~10% RTF cost).

## 12a — `QwenPlaybackSink` in `src/vokel/playback.py`

Mirror `KokoroPlaybackSink`'s architecture exactly: background synthesis →
asyncio queue → sounddevice playback, same `PlaybackSink` Protocol, same stop
semantics (stop clears the queue and kills playback instantly; the synthesis
call that is already running is left to finish and its result discarded).

Constructor: `model_dir` (required, no default download — error clearly if the
path is missing), `speaker="Ryan"`, `language="English"`, `instruct=None`,
`dtype="bfloat16"`, `device="cuda:0"`.

Constraints:
- **Lazy imports.** `torch` and `qwen_tts` import inside the sink's init/first
  use, never at module top — `playback.py` must keep importing on machines
  without CUDA. Optional extra in pyproject: `qwen = ["qwen-tts>=0.1.1"]`, plus a
  README note that torchaudio must be pinned `2.10.0+cu128` from the pytorch
  cu128 index (qwen-tts otherwise drags in a CUDA-13 torchaudio that breaks import).
- **Pipeline, don't batch.** The existing background-synthesis pattern already
  overlaps synth of phrase N+1 with playback of N; keep that. With RTF 0.93 the
  flow sustains, barely — do not add per-phrase overhead in the hot loop.
- **Short-input guard.** The text chunker's `min_chars: 12` protects mid-stream,
  but its end-of-turn flush can emit tiny fragments. In the sink: hold a fragment
  shorter than 12 chars and prepend it to the next phrase; if playback drains and
  the turn ends with a fragment still held, synthesize it anyway (rare; accept the
  risk) — never drop text silently.
- `close()` releases the model (`del` + `torch.cuda.empty_cache()`).
- Model load happens on first `speak`, not construction (session start must not
  block ~3 s before the mic even opens); alternatively an explicit async
  `warm_up()` called from web.py after `session_started` is sent — implementer's
  choice, state it in the report.

## 12b — web.py wiring

- `playback_backend == "qwen"`: construct `QwenPlaybackSink` with `model_dir`
  from the start payload (`qwen_model_dir`), falling back to env
  `VOKEL_QWEN_MODEL_DIR`. On construction failure (no package, no CUDA, bad
  path): send a clear `error` message and fall back to Kokoro — never a dead session.
- Extend the musical-mode guard tuple (web.py:1122) to
  `("kokoro", "spd-say", "qwen")` so the quantized wrapper accepts it.
- Capability surfacing, same pattern as `ffmpeg_available` (web.py:594): add
  `qwen_tts_available` (importable AND model dir resolves) to the
  `execute_state` message.
- Voice preview: if the existing preview path is Kokoro-specific, return a
  graceful "preview unavailable for Qwen" rather than 500.

## 12c — Frontend

- Playback selector gains "Qwen (performance)" — disabled with a hint line when
  `qwen_tts_available` is false (same pattern as the greyed Load Track button).
- Platform Capabilities panel: one line for Qwen TTS availability.
- Set expectations in the UI: small hint near the selector, e.g. "higher quality,
  ~2–4 s first response".

## Tests

1. Sink unit tests with `qwen_tts`/`torch` mocked at import (see how Kokoro tests
   mock; module may need an import indirection seam) — speak/stop/close lifecycle,
   short-fragment hold-and-merge, fragment flushed at turn end.
2. web.py: qwen backend selected + unavailable → error + Kokoro fallback, session
   still starts; guard tuple accepts qwen (musical mode wraps it); capability flag
   present in `execute_state`.
3. Frontend builds; no test framework there — visual check is Richard's walk.

## Gates

- `.venv/bin/python -m pytest` all green (venv only), `npm run build` green.
- engine.py untouched (`git diff --stat` proof in the report).
- Kokoro/spd-say/console sessions byte-identical in behavior when qwen is not
  selected (zero-behavior-change rule).
- Richard's walk: musical session, Qwen voice, Hermes brain — the full
  "cloud brain + local performance voice" configuration this has all been for.

## Out of scope (do not do)

Streaming synthesis (doesn't exist offline); aborting in-flight synthesis;
auto-downloading models; touching engine.py; the 1.7B variant (RTF > 1 expected
on this GPU — bench will confirm; sink stays 0.6B-configured by default).
