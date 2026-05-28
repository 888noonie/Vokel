# ROADMAP

Vokel is built in measured stages. Each stage should preserve the live
conversation loop:

You speak. It answers. You interrupt. It stops. It listens again.

Vokel is local-first by default, but it can also act as a voice front-end for
external agent stacks. The boundary is simple: models and agents may reason,
but Vokel owns capture, playback, interruption, consent, routing, display, and
audit.

Project stance:

> I build it for myself, with respect for others.

## Product Direction Lock

Vokel is not an assistant identity. Vokel is the clean voice and display layer
between a person and the intelligence systems they already run.

User-facing language should prefer **Connections**, not profiles. LM Studio and
Hermes already own their own model, agent, memory, tool, provider, and profile
configuration. Vokel should not duplicate those systems.

A Vokel connection is a remembered route to something the user wants to speak
to:

- LM Studio for local OpenAI-compatible model use.
- Hermes for an external agent stack, including the Android/Termux path.

The user should always be able to see:

- what they are connected to
- whether the route is local or external
- who owns the tools
- what context has been shared
- whether interruption is available
- whether execution consent is armed

Everything powerful should be visible. Everything visible should be
collapsible. Everything external should be explicit.

See `docs/foundation-direction.md` for the current grounding note.

## ✅ v0.1 — Hermes Direct WebSocket Transport (DONE)

- [x] `HermesWebSocketClient` (full `AgentBackend` implementation)
- [x] Termux stub with `start_turn` + delta streaming + `turn_complete`
- [x] End-to-end smoke test passing with real tokens from Pixel Pro 8
- Next: Wire into `ConversationEngine` + UI

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

## Completed: Onboarding & First-Run Experience (simple_setup)

Status: complete.

Goal: reduce time from fresh `git clone` to a working voice turn (especially `vokel --web`) from "tribal knowledge + 30 minutes of pain" to under two minutes for any developer.

- `scripts/install.sh` — the canonical one-command installer (venv via Python, uv for core + dev, interactive optional full audio stack, `--minimal` models, `.env` bootstrap)
- `download_models.py --minimal` — fast first-run path that only pulls the essentials (Silero VAD + Kokoro v1 + voices) while the full models remain available via the existing script
- `scripts/__init__.py` — makes `python -m scripts.download_models` reliable
- `.env.example` — declared surface of truth with practical keys:
  - `LM_STUDIO_URL` / `LM_STUDIO_MODEL`
  - `SERPAPI_API_KEY`, `UNSPLASH_ACCESS_KEY`, `GIPHY_API_KEY`
  - `MEMORY_ENABLED`, `MEMORY_DB_PATH`
  - `DEFAULT_VOICE`, `SPEED`
- README.md Quickstart now leads with the clone + `chmod +x scripts/install.sh` + `./scripts/install.sh` flow, followed by immediate next steps (`vokel --web` or a direct voice prompt)
- Success banner and guidance updated to match the new simple path while preserving the interactive audio choice

This change directly makes the Phase 0 Desktop Reference Core usable by others without requiring deep repo archaeology. It is a small, measurable DX slice that protects the core product test (live conversation loop) by getting people into it faster.

Exit check:
- A person who has never seen the repo before can run the one-liner and reach the web dashboard against LM Studio
- Optional audio path remains opt-in and does not block the text-only happy path
- `python3 -m pytest -q` remains green
- No new runtime dependencies introduced

## Phase 5: Connections And Transparent Routing

Status: next build slice.

Goal: replace muddy backend/mode language with a clean connection layer that lets
the user speak to LM Studio or Hermes without Vokel pretending to be the agent.

First slice:

- User-facing label: **Connections**, not profiles.
- First-class connection type: **LM Studio**.
- First-class connection type: **Hermes**.
- Saved connection settings: name, type, endpoint/gateway URL, optional API key,
  local voice/speed preference if needed.
- Active connection badge in the dashboard and transcript.
- Local/external route badge.
- Tool ownership badge:
  - LM Studio/Built-in route: Vokel tools may be active.
  - Hermes route: Hermes owns tools; Vokel does not duplicate them.
- Privacy state badge: local, external agent, or blocked by local-only mode.

Boundary:

- Vokel connects the user's voice to the selected backend.
- Vokel does not manage LM Studio personas or Hermes agent identity.
- Connection switching must be visible and reversible.
- The user must never silently move from local to external routing.

Exit check:

- user can choose LM Studio or Hermes from a simple connection selector
- active connection is visible during every turn
- transcript/export records which connection received the turn
- Hermes tools remain backend-owned
- local-only mode can block external routing

## Phase 6: Android Companion And Termux Hermes

Status: next product proof after connection layer.

Goal: bring the proven voice/control surface to Android, with a first target of
talking to a Hermes agent running in Termux on the same phone.

Why this moves forward:

- The desktop voice loop is measurable.
- Hermes mode proves external agent routing.
- Android is the natural always-near interface for voice-first use.
- The user should be able to talk to phone-based Hermes without needing a cloud
  voice assistant.

First Android slice:

- Android-friendly web/PWA dashboard layout
- configurable Hermes gateway URL
- documented Termux Hermes API Server setup
- `/health` check from Vokel to Hermes
- visible active connection and privacy state
- voice handoff to Hermes mode
- touchless interrupt and resume where the Android audio path permits it
- audio focus, ducking, and notification-safe earcons

Termux Hermes path:

- Document Hermes API Server setup in Termux
- Verify `API_SERVER_ENABLED=true` and `/health`
- Support Vokel-to-Hermes handoff over `127.0.0.1`, LAN, or Android-supported
  loopback strategy
- Keep Hermes tools agent-owned and visible in Vokel's connection state

Later Android slices:

- Sherpa-ONNX Android VAD/ASR
- Local TTS path
- Local model runner selected by measured device performance
- Airplane-mode local demo
- Screen-off or pocket-safe foreground service behavior
- Cross-device routing between the user's desktop and phone where safe

Exit check:

- Android can talk to Hermes running in Termux
- active connection and privacy state are visible
- interruption remains higher priority than fast output

## Phase 7: Wake Word, Pause, Resume, And Consent Boundary

Status: planned.

Goal: make voice routing, live context control, and execution consent natural
without weakening safety.

Voice phrases:

- `Hey Vokel` wakes the listen loop
- `pause`, `hold on`, or `wait` pauses the active stream
- `continue` resumes when safe
- `switch to Hermes` changes connection only after showing the route change
- `switch back to local` returns to local routing
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
- Pause/hold commands stop the stream; they do not grant action permission.
- Live context injection must show what context is being added and where it will
  be routed.
- The user must be able to see what is about to happen before execution.

Exit check:

- `pause` / `hold on` stops the active stream cleanly
- `continue` resumes with clear state
- `Vokel Execute` changes state but does not execute by itself
- high-risk actions cannot run without explicit confirmation
- consent state is visible in the dashboard and Agent Console

## Phase 8: Media Cards, Media Inputs, And Privacy Routing

Status: planned.

Goal: make Media Cards the single visual grammar for useful things entering or
leaving a conversation, then add images, screenshots, file uploads, and later OCR
without blurring local/cloud boundaries.

Media Card first slice:

- One internal `MediaCard` shape for web, image, GIF, tool, context, and consent
  cards.
- Type/kind badge: web, image, GIF, file, screenshot, OCR, voice, device, tool,
  context, consent.
- Route badge: local, LM Studio, Hermes, external.
- Privacy badge: local-only, external allowed, external sent.
- Actions: expand, save, hide, tag, copy, delete, use as context.
- `speechText` field or equivalent so TTS never reads raw URLs, metadata, or
  unsafe card internals.

First media input slice:

- Image and screenshot upload
- File upload read-only summarization
- Media preview in transcript
- Per-item routing: local model, Hermes/external agent, or refused by privacy
  mode

Privacy controls:

- `Local Only` mode blocks cloud agent handoff and cloud media routing
- visible header state: local-only, cloud enabled, active connection
- confirmation before sending private media or file contents to an external agent
- transcript/audit note records which backend received the media

Deferred:

- shareable Media Cards
- card import/export format
- cross-device card sending
- OCR and embedding model integration
- camera input
- screen context
- file write-back
- continuous ambient capture

Exit check:

- uploaded media never leaves local mode without confirmation
- cloud routing is visible before and after the action
- cards can be expanded, saved, hidden, tagged, and used as context where safe
- TTS does not read file paths, URLs, or metadata aloud unless requested

## Phase 9: Better Barge-In And Visual Feedback

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
  execute-armed, and connection-switching

Exit check:

- laptop-speaker barge-in improves without false interrupts
- visual state matches the live loop
- visualizer does not become a heavy console or text duplicate

## Phase 10: Observability And Long-Term Recall

Status: planned.

Goal: make Vokel useful over time without turning memory into hidden surveillance.

Latency history:

- persist turn-level metrics locally
- dashboard trends and alerts
- compare local model, Hermes, browser audio, and Android paths

Memory v2:

- opt-in local vector store candidate: Chroma, LanceDB, or SQLite vector extension
- visible recall snippets
- per-connection conversation summaries
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

## Phase 11: Additional Backends And Cross-Device Routing

Status: planned.

Goal: make switching intelligence sources simple while preserving the same voice
interface.

Local backends:

- Ollama presets
- llama.cpp server presets
- custom OpenAI-compatible endpoint
- automatic fallback between local endpoints

External agents:

- Hermes capability display
- future agent stack adapters behind `AgentBackend`
- route switching by voice: "connect me to Hermes", "switch back to local"

Cross-device direction:

- phone can speak to desktop LM Studio where safe
- desktop can speak to phone Termux Hermes where safe
- device/route visibility must stay explicit
- media/context sharing between devices must remain opt-in and auditable

Public-facing behavior:

- show active connection clearly
- explain when an external/cloud-backed agent is active
- never silently move a conversation from local to external

Exit check:

- user can switch backend without restarting Vokel
- active connection is visible in transcript/export
- privacy mode can prevent external routing

## Open Decisions

- Android local LLM runner: llama.cpp, MLC, Google AI Edge, or another measured runtime.
- Android ASR model family: SenseVoice, Moonshine, Whisper, or streaming Zipformer.
- Termux-Hermes connection strategy on Android loopback and foreground service boundaries.
- Echo cancellation strategy for laptop speakers and phone speaker/mic paths.
- First media payload contract for Hermes API Server mode.
- First stable Media Card schema and persistence format.
- Whether card sharing is a local export format first or a network/share feature later.
- Vector memory backend and deletion semantics.
- Whether `Local Only` should be a global lock or per-session mode.

## Resolved Decisions

- Desktop TTS engine: **Kokoro ONNX**.
- Inference client naming: **LocalInferenceClient**.
- Tool-call audio cue: **two-tone sine**.
- Visual media pipeline: **caption-only TTS**.
- User-facing backend language: **Connections**, not profiles.
- First-class v1 connections: **LM Studio** and **Hermes**.
- Hermes mode: **agent-owned tools, Vokel-owned voice and consent**.
- Media Cards: **display and trust primitive first; sharing later**.
- Execute phrase: **`Vokel Execute` arms consent; it does not execute alone**.
- Wake phrase: **`Hey Vokel` opens the listen loop**.
- Privacy posture: **local-first by default; external agents are explicit**.
- Rebranded from Voyce to **Vokel**: voice + invoke + local.
