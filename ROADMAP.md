# ROADMAP

Vokel is built in measured stages. Each stage should produce a working loop and fresh latency numbers before the next layer is added.

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
`laptop-mic-headphones` is the recommended dev baseline until a true headset mic route is available.
Route diagnostics now report whether the preferred headset input is actually available.

First audible `laptop-mic-headphones` run completed. Next tuning target:
`turn_to_playback_start_ms < 1000`.

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

Status: complete. Kokoro ONNX is the primary TTS backend with streaming synthesis
via `create_stream`. Measured stop latency: **0.1 ms**. `spd-say` remains available
as a fallback.

- Add a real TTS `PlaybackSink` — done (`KokoroPlaybackSink`)
- Start playback as soon as the first phrase is queued — done
- Stop playback immediately on user speech — done (async `_stop_event`)
- Cancel active LLM generation on barge-in — done
- Flush pending phrases on interruption — done

Exit check:

- assistant is speaking — verified via benchmark
- user interrupts mid-sentence — verified (`stop_latency_ms=0.0`)
- playback stops immediately — verified (`speak_completion_after_stop_ms=0.1`)
- next user turn is accepted without touching the keyboard — TUI scaffolded

## Phase 3: Streaming ASR

Goal: reduce the waiting period after speech ends.

Status: complete (locked). Sherpa-ONNX streaming Zipformer is integrated on the
desktop path: `create_streaming_asr` + `StreamingTurnProducer`, shared `LatencyTrace`
with `ConversationEngine`, CLI `--mic --streaming-asr-dir`, benchmark mode
`mic-streaming-lm-studio`, and `streaming-zipformer-en` in `scripts/download_models.py`.

- Add a streaming ASR adapter behind the existing `AsrEngine` boundary — done
- Track partial transcript timing separately from final transcript timing — baseline
  in telemetry; further tuning as needed
- Decide when partial text is stable enough to start retrieval or LLM prefill — future
  overlap work; Phase 4 memory may drive this

Exit check:

- transcript path works end-to-end with streaming models — verified (CLI + benchmark)
- first-token latency can be compared against offline mic path — benchmark + scoreboard

## Phase 4: Memory

Goal: add useful local context without slowing the voice loop.

Status: complete (locked first slice). Conversation Recall is opt-in, stored
under ignored `data/`, runs behind a small `MemoryStore` interface, and records
retrieval/write timings in the same `LatencyTrace` as the voice loop. Saved
notes can be seeded with `--remember` so high-signal preferences help the local
model without an extra summarization call.

- Local conversation store — first SQLite implementation added
- Lightweight retrieval path — recent-scored recall, bounded snippets, saved-note priority
- Explicit privacy boundary — disabled by default; `--memory` / dashboard toggle
- Latency budget for memory injection — `memory_retrieval_ms`, `memory_write_ms`,
  and `memory_to_first_token_ms`

Exit check:

- memory can be disabled — default behavior, dashboard toggle, and CLI flag
- memory timing is visible in telemetry — done
- no blocking disk path sits inside audio callbacks — SQLite work is offloaded

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

## Desktop Control Surface

Goal: add operator controls without weakening the no-button live conversation loop.

Status: complete (first slice).

- Voice selection through named TTS voices/profiles — done (28 Kokoro voices, per-voice preview, speed slider)
- Pause conversation without discarding the current session — done
- Reset the active conversation state — done
- Export transcript and selected memory notes to `.md` — done
- Edit memory entries locally, with clear audit visibility — done

Exit check:

- controls are available from the desktop UI/TUI — done
- pause and reset do not break barge-in or listen/capture cues — verified
- exported Markdown excludes caches, model files, and generated audio — verified
- memory edits remain behind the `MemoryStore` interface — verified

## Phase 5: Agent Tools

Goal: let the local model use external capabilities without hallucinating results.

Status: complete. Web search, image search, and GIF search all integrated with
deterministic forced execution, hybrid synthesis, and rich transcript display.

### Web Search (SerpApi DuckDuckGo)

- Tool registry with JSON schema definitions — done (`ToolRegistry`, `ToolDefinition`)
- `LocalInferenceClient` (renamed from `LmStudioClient`) — done, portable to llama.cpp/MLC LLM
- Deterministic forced search for web/news/current queries — done
- Hybrid synthesis: search evidence injected into model with strict prompt — done
- Page scraping fallback when API snippets are generic — done
- Raw evidence fallback when model hedges — done
- Speech sanitization layer (Markdown, URLs, formatting stripped before TTS) — done
- Tool-call audio cue (two-tone sine, not white noise) — done
- Voice preview without starting a session — done

### Image Search (Unsplash)

- `search_image` tool with Unsplash API — done
- Deterministic detection for "show me an image/picture/photo of" — done
- Media synthesis: LLM gives brief spoken intro, transcript shows inline photo — done
- TTS only speaks caption, not URLs or attribution — done
- Unsplash attribution in transcript — done

### GIF Search (Giphy)

- `search_gif` tool with Giphy API — done
- Deterministic detection for "gif", "meme", "sticker", "reaction", "make me laugh" — done
- Context-aware follow-ups: short replies continue GIF topic from recent history — done
- ASR-tolerant triggers: "show me a g" works when speech recognition truncates — done
- Query extraction with filler-word stripping and 60-char hard cap — done
- Playful LLM synthesis: model gives one-liner reaction, not metadata dump — done
- Compact GIF card in transcript with purple border and GIF badge — done
- TTS sanitizer strips GIF markdown to "GIF shown in transcript" — done

### Shared media pipeline

- Visual tool results (image/GIF) separated: transcript gets rich media, TTS gets caption only
- Vokel custom favicon (purple pulse icon) with dedicated routes
- Quiet uvicorn access log filter suppresses favicon/asset noise

Exit check:

- "What's the top news in France today?" returns real headlines with sources — verified
- "Weather in France today" returns actual temperature data — verified (via page scrape)
- "Show me an image of the Alps" shows inline photo with brief spoken intro — verified
- "Show me a gif of mind blown" shows animated GIF with playful reaction — verified
- Context follow-up "anything" after GIF conversation triggers new GIF — verified
- Model cannot claim it cannot browse — synthesis prompt enforced
- Fallback raw evidence always includes clickable links — verified
- TTS never reads asterisks, URLs, Markdown, or media metadata aloud — sanitizer verified

## Phase 6: Android Port

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

- First Android LLM runner: llama.cpp, MLC, Google AI Edge, or another measured runtime.
- First ASR model family: SenseVoice, Moonshine, Whisper, or streaming Zipformer.
- Echo cancellation strategy for hands-free barge-in on laptop speakers.
- Richer search depth: full-page scraping vs. paid API tier vs. local cache.

## Resolved Decisions

- Desktop TTS engine: **Kokoro ONNX** — streaming synthesis, 0.1 ms stop latency, no external process dependency.
- Inference client naming: **LocalInferenceClient** — decoupled from LM Studio branding, portable to any OpenAI-compatible endpoint.
- Tool-call audio cue: **two-tone sine** — white noise was too harsh; gentle tonal cue reads as "working" without masking speech.
- Barge-in on laptop speakers: **button-based** — true hands-free interruption requires a headset or echo cancellation, which is deferred.
- Visual media pipeline: **caption-only TTS** — transcript gets rich media (images, GIFs), TTS only speaks the LLM's brief intro. Never read URLs, attribution, or metadata aloud.
- Rebranded from Voyce to **Vokel** — voice + invoke + local.
