# ROADMAP

Vokel is built in measured stages. Each stage should preserve the live
conversation loop:

You speak. It answers. You interrupt. It stops. It listens again.

Vokel is local-first by default, but it can also act as a voice front-end for
external agent stacks. The boundary is simple: models and agents may reason,
but Vokel owns capture, playback, interruption, consent, routing, and audit.

Project stance:

> I build it for myself, with respect for others.

## Phase 0: Desktop Reference Core

Status: complete.

- Async OpenAI-compatible streaming
- Phrase chunking for TTS handoff
- Cancellable generation
- Playback queue boundary
- Latency telemetry
- Optional audio module scaffold

Exit check:

- `python3 -m pytest -q`
- one local model smoke test with latency output

## Phase 1: Real Desktop Audio Turn

Status: complete.

Goal: replace passthrough text turns with one real microphone turn.

- Named desktop audio profiles
- Route diagnostics for Pulse/PipeWire source availability
- VAD-bounded microphone turns
- Sherpa-ONNX offline ASR
- Capture, ASR, first-token, and first-playback timing

Recommended desktop dev baseline:

- `laptop-mic-headphones` until a true headset mic route is available

Exit check:

- speak one turn without typing
- receive one streamed response
- compare latency against text-mode baseline

## Phase 2: Real Playback And Barge-In

Status: complete.

Goal: prove the full loop with actual audio output and interruption.

- Kokoro ONNX streaming TTS (`KokoroPlaybackSink`)
- `spd-say` fallback
- Playback starts as soon as the first phrase is queued
- Generation, queued phrases, and playback are cancellable
- Measured stop latency: **0.1 ms**

Exit check:

- assistant is speaking
- user interrupts mid-sentence
- playback stops immediately
- next user turn is accepted without restarting the session

## Phase 3: Streaming ASR

Status: complete.

Goal: reduce the waiting period after speech ends.

- Sherpa-ONNX streaming Zipformer
- `create_streaming_asr`
- `StreamingTurnProducer`
- CLI `--mic --streaming-asr-dir`
- Benchmark mode `mic-streaming-lm-studio`
- Shared `LatencyTrace` with `ConversationEngine`

Exit check:

- transcript path works end-to-end with streaming models
- first-token latency can be compared against offline mic path

## Phase 4: Memory First Slice

Status: complete.

Goal: add useful local context without slowing the voice loop.

- Opt-in Conversation Recall
- SQLite store under ignored `data/`
- Small `MemoryStore` interface
- Saved notes through `--remember`
- Dashboard toggle and editable memory notes
- Timing marks: `memory_retrieval_ms`, `memory_write_ms`,
  `memory_to_first_token_ms`

Boundary:

- Recall is disabled by default.
- Saved notes are user-visible.
- Memory retrieval must stay outside audio callbacks.

Exit check:

- memory can be disabled
- timing is visible in telemetry
- Android can replace the store behind the same interface

## Completed: Desktop Dashboard And Local Tools

Status: complete first slices.

Dashboard:

- Browser/local audio mode switch
- Start, stop, pause, reset, and barge-in controls
- Output mute and browser playback volume
- Persistent voice selection, speech speed, and playback backend
- Kokoro voice preview
- Transcript export to Markdown
- Agent Console tab
- Previous Chats tab backed by browser-local snapshots

Local tool layer:

- `ToolRegistry` / `ToolDefinition`
- Deterministic forced tool execution for small local models
- Web search with SerpApi DuckDuckGo and page scraping fallback
- Image search with Unsplash
- GIF search with Giphy
- Rich transcript rendering for media
- Caption-only TTS path with `sanitize_for_speech`
- Tool-call audio cue

Exit check:

- local model cannot falsely claim it browsed
- raw evidence fallback includes clickable links
- TTS never reads raw Markdown, URLs, media metadata, or attribution aloud

## Completed: Agent Extension First Slice

Status: complete first slice.

Goal: let Vokel serve as the voice front-end for an external agent stack.

- Hermes API Server support through `HermesAgentClient`
- Hermes owns reasoning, memory, and tools in Hermes mode
- Vokel owns voice capture, playback, interruption, session display, and consent
- Agent Console shows backend, session id, gateway health, lifecycle events,
  and consent state
- Execute consent scaffold: arm, cancel, and 3-second hold; no risky action is
  executable until a concrete capability is registered

Boundary:

- In Built-in mode, Vokel-owned tools are available to the local model.
- In Hermes mode, Hermes-owned tools are not duplicated inside Vokel.
- Agent handoff must be visible and reversible.

Exit check:

- Hermes health check passes
- Vokel can start a Hermes conversation session
- barge-in closes the active Hermes HTTP stream

## Phase 5: Android Companion And Termux Hermes

Status: complete first slice.

Goal: bring the proven voice/control surface to Android, with a first target of
talking to a Hermes agent running in Termux on the same phone.

Why this moves forward:

- The desktop voice loop is measurable.
- Hermes mode proves external agent routing.
- Android is the natural always-near interface for voice-first use.

First Android slice:

- Android Foreground Service for microphone/listening lifecycle
- Persistent notification and permission flow
- Browser/PWA dashboard hardening for phone screens
- Local network or loopback routing to Hermes in Termux
- Configurable Hermes gateway URL, defaulting to local device routes
- Touchless interrupt and resume
- Audio focus, ducking, and notification-safe earcons

Termux Hermes path:

- Document Hermes API Server setup in Termux
- Verify `API_SERVER_ENABLED=true` and `/health`
- Support Vokel-to-Hermes handoff over `127.0.0.1`, LAN, or Android-supported
  loopback strategy
- Keep Hermes tools agent-owned and visible in Vokel's Agent Console

Later Android slices:

- Sherpa-ONNX Android VAD/ASR
- Local TTS path
- Local model runner selected by measured device performance
- Airplane-mode local demo
- Screen-off or pocket-safe foreground service behavior

Exit check:

- Android can talk to Hermes running in Termux
- active backend and privacy state are visible
- interruption remains higher priority than fast output

## Phase 6: Wake Word And Consent Boundary

Status: planned.

Goal: make voice routing and execution consent natural without weakening safety.

Voice phrases:

- `Hey Vokel` wakes the listen loop
- `Vokel Execute` arms execution mode
- high-risk actions require a 3-second hold or equivalent explicit confirmation

Risk tiers:

- Low: conversation, summaries, local read-only status
- Medium: cloud routing, web requests, agent handoff, media sent to an agent
- High: file upload to cloud agent, email actions, repository changes,
  calendar/cron scheduling
- Critical: destructive file operations, smart-home control, external messages,
  irreversible account actions

Boundary:

- Agents may request actions.
- Vokel grants or denies actions.
- The user must be able to see what is about to happen before execution.

Exit check:

- `Vokel Execute` changes state but does not execute by itself
- high-risk actions cannot run without explicit confirmation
- consent state is visible in the dashboard and Agent Console

## Phase 7: Media Inputs And Privacy Routing

Status: planned.

Goal: add images, screenshots, file uploads, and eventually camera input without
blurring local/cloud boundaries.

First media slice:

- Image and screenshot upload
- File upload read-only summarization
- Media preview in transcript
- Per-item routing: local model, Hermes/external agent, or refused by privacy
  mode

Privacy controls:

- `Local Only` mode blocks cloud agent handoff and cloud media routing
- visible header state: local-only, cloud enabled, active agent
- confirmation before sending private media or file contents to an external agent
- transcript/audit note records which backend received the media

Deferred:

- camera input
- screen context
- file write-back
- continuous ambient capture

Exit check:

- uploaded media never leaves local mode without confirmation
- cloud routing is visible before and after the action
- TTS does not read file paths, URLs, or metadata aloud unless requested

## Phase 8: Better Barge-In And Visual Feedback

Status: planned.

Goal: improve hands-free interruption and make the voice state legible at a
glance.

Audio:

- WebRTC capture constraints and echo cancellation tuning
- playback-to-mic echo guard for laptop speakers
- browser output analyser for real assistant-speech visual reactivity
- audio focus and ducking strategy for Android

Visualizer:

- Replace the simple sine-wave display with a Vokel-native resonance visualizer
- Use real analyser data when available
- Use procedural fallback when no audio source is available
- Map states to idle, listening, generating, speaking, paused, agent-active,
  and execute-armed

Exit check:

- laptop-speaker barge-in improves without false interrupts
- visual state matches the live loop
- visualizer does not become a heavy console or text duplicate

## Phase 9: Observability And Long-Term Recall

Status: planned.

Goal: make Vokel useful over time without turning memory into hidden surveillance.

Latency history:

- persist turn-level metrics locally
- dashboard trends and alerts
- compare local model, Hermes, browser audio, and Android paths

Memory v2:

- opt-in local vector store candidate: Chroma, LanceDB, or SQLite vector extension
- visible recall snippets
- per-agent conversation summaries
- project/repository threads
- "what changed since last time?" digests
- user-approved durable facts

Boundary:

- memory is clarity, not creepiness
- every durable fact should be inspectable and deletable
- external agents may own their own memory; Vokel records handoff context and
  user-visible audit notes

Exit check:

- local recall can be disabled globally
- recalled snippets are visible
- latency impact is measured before enabling by default

## Phase 10: Model And Agent Routing

Status: planned.

Goal: make switching intelligence sources simple while preserving the same voice
interface.

Local backends:

- LM Studio presets
- Ollama presets
- llama.cpp server presets
- custom OpenAI-compatible endpoint
- automatic fallback between local endpoints

External agents:

- Hermes capability display
- future agent stack adapters behind `AgentBackend`
- route switching by voice: "connect me to Hermes", "switch back to local"

Public-facing behavior:

- show active backend clearly
- explain when a cloud-backed agent is active
- never silently move a conversation from local to external

Exit check:

- user can switch backend without restarting Vokel
- active backend is visible in transcript/export
- privacy mode can prevent external routing

## Open Decisions

- Android local LLM runner: llama.cpp, MLC, Google AI Edge, or another measured runtime.
- Android ASR model family: SenseVoice, Moonshine, Whisper, or streaming Zipformer.
- Termux-Hermes connection strategy on Android loopback and foreground service boundaries.
- Echo cancellation strategy for laptop speakers and phone speaker/mic paths.
- First media payload contract for Hermes API Server mode.
- Vector memory backend and deletion semantics.
- Whether `Local Only` should be a global lock or per-session mode.

## Resolved Decisions

- Desktop TTS engine: **Kokoro ONNX**.
- Inference client naming: **LocalInferenceClient**.
- Tool-call audio cue: **two-tone sine**.
- Visual media pipeline: **caption-only TTS**.
- Hermes mode: **agent-owned tools, Vokel-owned voice and consent**.
- Execute phrase: **`Vokel Execute` arms consent; it does not execute alone**.
- Wake phrase: **`Hey Vokel` opens the listen loop**.
- Privacy posture: **local-first by default; external agents are explicit**.
- Rebranded from Voyce to **Vokel**: voice + invoke + local.
