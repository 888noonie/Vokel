# ROADMAP

Voyce is built in measured stages. Each stage should produce a working loop and fresh latency numbers before the next layer is added.

## Phase 0: Desktop Reference Core

Status: complete.

- Async LM Studio streaming
- Phrase chunking for TTS handoff
- Cancellable generation
- Playback queue boundary
- Latency telemetry
- Optional audio module scaffold

Exit check:

- `python3 -m pytest -q`
- one LM Studio smoke test with latency output

## Phase 1: Real Desktop Audio Turn

Goal: replace passthrough text turns with one real microphone turn.

Status: working. Laptop-open and headset-wired profiles have completed mic benchmarks.

- Maintain `docs/latency-budget.md`
- Maintain `docs/model-matrix.md`
- Prepare ignored local model storage with `scripts/download_models.py`
- Run synthetic and LM Studio benchmarks before changing audio backends
- Install optional audio dependencies
- Download VAD and ASR model files
- Capture VAD-bounded microphone turns
- Transcribe with Sherpa-ONNX offline ASR
- Report `asr_duration_ms`, `asr_to_first_token_ms`, and `turn_to_playback_start_ms`

Exit check:

- speak one turn without typing
- receive one streamed response
- compare latency against text-mode baseline

## Phase 2: Real Playback And Barge-In

Goal: prove the full loop with actual audio output and interruption.

Status: in progress. `spd-say` is available as a bridge backend with measured stop latency.

- Add a real TTS `PlaybackSink`
- Start playback as soon as the first phrase is queued
- Stop playback immediately on user speech
- Cancel active LLM generation on barge-in
- Flush pending phrases on interruption

Exit check:

- assistant is speaking
- user interrupts mid-sentence
- playback stops immediately
- next user turn is accepted without touching the keyboard

## Phase 3: Streaming ASR

Goal: reduce the waiting period after speech ends.

- Add a streaming ASR adapter behind the existing `AsrEngine` boundary
- Track partial transcript timing separately from final transcript timing
- Decide when partial text is stable enough to start retrieval or LLM prefill

Exit check:

- transcript appears incrementally
- first-token latency improves or the added complexity is rejected

## Phase 4: Memory

Goal: add useful local context without slowing the voice loop.

- SQLite conversation store
- Lightweight retrieval path
- Explicit privacy boundary
- Latency budget for memory injection

Exit check:

- memory can be disabled
- memory timing is visible in telemetry
- no blocking disk path sits inside audio callbacks

## Phase 5: Android Port

Goal: move the proven event model to Pixel 8 Pro.

- Android Foreground Service
- Sherpa-ONNX Android VAD/ASR
- Local model runner selected by measured device performance
- Local TTS path
- Persistent notification and permission flow

Exit check:

- airplane mode demo
- screen-off or pocket-safe foreground service behavior
- touchless interrupt and resume

## Open Decisions

- Desktop TTS engine: Piper, Kokoro ONNX, Sherpa-ONNX TTS, or another low-latency sink.
- First Android LLM runner: llama.cpp, MLC, Google AI Edge, or another measured runtime.
- First ASR model family: SenseVoice, Moonshine, Whisper, or streaming Zipformer.
- Echo handling strategy for barge-in while TTS is playing.
