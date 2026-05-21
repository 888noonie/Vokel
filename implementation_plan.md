# Phase 3 Implementation Plan

**Status:** implemented and locked in (see `ROADMAP.md` Phase 3). This document
remains the design record; verification steps still apply when changing the
streaming path.

This plan updates the earlier streaming-ASR sketch so it stays aligned with the
current Voyce architecture, `AGENTS.md`, and the Phase 3 exit criteria in
`ROADMAP.md`.

## Goals

- Reduce the waiting period after speech ends.
- Preserve the existing conversation loop and cancellation behavior.
- Keep microphone, model, and playback details out of `ConversationEngine`.
- Add telemetry before deciding whether the extra complexity is worthwhile.

## Non-Goals For This Slice

- No echo-cancellation pipeline yet.
- No always-open microphone during TTS playback.
- No retrieval or LLM prefill triggered from partial transcripts yet.
- No Android-specific work.

## Architectural Constraints

1. `ConversationEngine` remains unaware of microphone hardware, streaming
   recognizer internals, and audio callback details.
2. ASR continues to sit behind the `AsrEngine` protocol for completed-turn
   transcription compatibility.
3. Streaming capture logic lives in the audio layer, exposed through a
   `TurnProducer`.
4. Every capture path remains cancellable and measurable.
5. Audio capture settings come from named audio profiles, not ad-hoc CLI tuning.

## Proposed Design

### 1. Keep `ConversationEngine.run_turns()` unchanged

Do not add `run_streaming_turns()` or any streaming-specific method to
`engine.py`.

The existing loop already has the right contract:

- `producer.next_turn()` returns an `AudioTurn`
- `asr.transcribe(turn)` produces the final transcript
- `submit_turn()` handles generation and playback

Phase 3 should plug into that loop by swapping the producer and, when useful,
the ASR implementation.

### 2. Add a streaming turn producer in the audio layer

Introduce a `StreamingTurnProducer` in `src/voyce/audio.py` that implements the
existing `TurnProducer` protocol.

Responsibilities:

- open and close the microphone stream for one turn
- feed frames into streaming ASR
- accumulate raw samples for traceability and offline fallback
- emit telemetry for partial and stable transcript milestones
- detect endpoint completion
- return a finalized `AudioTurn`

This keeps streaming mechanics close to the microphone and recognizer code
instead of leaking them into the engine.

### 3. Keep transcript stability heuristics in the producer

The stability timer belongs in `StreamingTurnProducer`, not in
`ConversationEngine`.

Initial heuristic:

- mark `partial_transcript` whenever decoded text changes
- mark `stable_transcript` after the text remains unchanged for about `600 ms`
- if endpoint fires before the stability timer does, mark
  `stable_transcript` with the last partial result before returning

This is explicitly a measurement tool for Phase 3, not a final product policy.

### 4. Use a dual-mode streaming ASR adapter

Add `SherpaOnlineAsr` with two entry points:

- `create_stream()` for incremental decoding
- `transcribe(AudioTurn)` for compatibility with the existing `AsrEngine`
  contract and current tests

That preserves the current turn-based path while enabling the streaming
producer to decode incrementally.

### 5. Wrap recognizer and stream together

If we keep a `SherpaOnlineStream` wrapper, it should hold both:

- the underlying `sherpa_onnx.OnlineRecognizer`
- the created `OnlineStream`

The wrapper should own the combined operations callers actually need:

- `accept_waveform(...)`
- `decode()`
- `get_result()`
- `is_endpoint()`
- `reset()` if required by the recognizer lifecycle

The caller should not have to juggle the recognizer and stream separately.

### 6. Use audio profiles as the source of microphone settings

`AsynchronousMicStream` should be configured from the active `AudioProfile`
through its `MicVadConfig`, not from a parallel set of streaming-only CLI
flags.

At minimum the streaming path should inherit:

- `device`
- `sample_rate`
- `read_seconds`
- `input_gain`
- `remove_dc_offset`

This keeps the desktop baseline consistent with the existing
`laptop-mic-headphones` recommendation.

### 7. Keep playback behavior simple for Phase 3

For this phase, keep the same turn boundary as the current desktop loop:

- microphone capture is active during user speech
- once a turn is finalized and submitted, the mic is closed during playback

This avoids pretending we solved echo handling before we actually have. The
open decision about barge-in while TTS is playing remains in `ROADMAP.md`.

## CLI And Config Shape

### Model selection

Prefer a single directory flag for the streaming model, for example:

- `--streaming-asr-dir models/sherpa-onnx-streaming-zipformer-en-2023-06-26`

Infer these files from that directory:

- `tokens.txt`
- `encoder*.onnx`
- `decoder*.onnx`
- `joiner*.onnx`

Avoid adding four separate path flags for one bundled model archive unless a
later need makes it unavoidable.

### Suggested flags

- `--audio-profile`
- `--streaming-asr-dir`
- `--asr-provider`
- optional `--streaming-asr` boolean to select the producer explicitly during
  evaluation

Keep the flag surface small while Phase 3 is still being measured.

## Model Download Work

Add the streaming Zipformer asset to `scripts/download_models.py` using the
existing safe archive extraction path.

Acceptance notes:

- model extracts under `models/`
- `models/` remains ignored by git
- the downloaded directory layout is compatible with `--streaming-asr-dir`

## Telemetry Work

Add Phase 3 trace events before optimizing:

- `partial_transcript`
- `stable_transcript`

Use them to report:

- `capture_to_first_partial_ms`
- `capture_to_stable_ms`
- `capture_to_first_token_ms`

Do not remove the existing turn-level metrics. Phase 3 success depends on
comparing the streaming path against the Phase 1 and Phase 2 baseline numbers.

## Implementation Steps

1. Extend model download support for the streaming Zipformer archive.
2. Add `SherpaOnlineAsrConfig` and `SherpaOnlineAsr`.
3. Add a thin `SherpaOnlineStream` helper that owns recognizer-plus-stream
   operations.
4. Add `AsynchronousMicStream` as an async context manager that bridges the
   sounddevice callback thread into the event loop with
   `call_soon_threadsafe(...)`.
5. Add `StreamingTurnProducer` that uses `AsynchronousMicStream`,
   `SherpaOnlineAsr`, and `LatencyTrace`.
6. Wire CLI/config setup so the streaming producer consumes the selected
   `AudioProfile`.
7. Keep `ConversationEngine.run_turns()` unchanged and exercise the new
   producer through the existing engine loop.
8. Add or update tests for:
   - online ASR adapter construction and file validation
   - transcript stability telemetry
   - producer endpoint completion
   - backward compatibility of `transcribe(AudioTurn)`
9. Measure latency and compare with the current baseline before deciding
   whether to keep the new path.

## Verification Plan

Required:

- `python3 -m pytest -q`
- one local streaming-ASR smoke run with latency output

Check against the Phase 3 exit criteria in `ROADMAP.md`:

1. transcript appears incrementally
2. `capture_to_first_token_ms` improves relative to the current desktop
   baseline, or we explicitly reject the added complexity

Recommended comparison table for the smoke run:

- baseline `capture_to_first_token_ms`
- streaming `capture_to_first_partial_ms`
- streaming `capture_to_stable_ms`
- streaming `capture_to_first_token_ms`
- notes on recognition quality and endpoint behavior

## Explicit Rejection Criteria

We should be willing to stop or revert the streaming path if any of these are
true after measurement:

- interruption correctness regresses
- endpointing becomes unreliable
- first-token latency does not improve enough to justify the complexity
- the streaming path forces engine-level coupling to audio or model details

## Open Follow-Up After Phase 3

- Whether partial transcripts should drive retrieval or LLM prefill
- Whether the mic should remain open during playback for true barge-in
- Echo handling for playback-over-mic scenarios
- Whether the streaming model quality is acceptable compared with offline ASR
