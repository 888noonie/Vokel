# Vokel

**Voice-invoked local intelligence.**

Vokel is a local-first voice interface for talking naturally with local LLMs and
external agent stacks. It preserves the live loop that makes voice feel useful:

You speak. It answers. You interrupt. It stops. It listens again.

By default, Vokel runs fully local: local ASR, local TTS, local memory, and OpenAI-compatible local model servers.

When you connect an external agent such as Hermes (via `ws://` or the HTTP gateway), Vokel becomes the **platform layer** — it handles voice I/O (ASR/TTS), interruption handling, consent, memory, UI, and audit — while the connected Hermes agent takes the **reasoning seat** and owns its own tools and decision-making.

In Hermes mode you are speaking directly to Hermes. Vokel simply provides the natural voice interface and safety rails around it.

Interruption works hands-free with a headset (separate mic and speaker). On
laptop speakers the BARGE IN button in the web dashboard is the reliable path,
because the mic hears its own TTS output without echo cancellation.

## Current Status (June 1, 2026)

✅ **Hermes WebSocket Bridge — Fully Working**
- Direct `ws://` connection to Termux stub on Pixel Pro 8
- Real token-by-token streaming validated
- `HermesWebSocketClient` implements full `AgentBackend` protocol
- First end-to-end streaming session completed

## Why Vokel?

The name comes from three roots: **Vok-** (Latin *vocare* — to call or invoke),
**Vocal** (it's voice-first), and **Local** (it runs on your hardware, private
by default). You invoke tools, evoke responses, and provoke searches — all by
speaking.

## What It Does

Vokel gives you one dependable voice surface for different kinds of
intelligence:

- **Talk naturally** — speak a question, get a spoken answer, interrupt mid-sentence
- **Invoke tools by voice** — "Search for the latest UK news", "Show me an image of the Alps"
- **Stay local by default** — local ASR, local TTS, local memory, and local model endpoints
- **Choose your model** — works with any OpenAI-compatible server (LM Studio, llama.cpp, MLC LLM)
- **Connect an agent** — use Hermes gateway mode when you want to speak directly with an external agent stack
- **Choose your voice** — 28 bundled Kokoro voices with preview and speed control
- **Keep consent visible** — Vokel owns interruption, routing, and future execution confirmation

## Architecture

The voice loop runs through five independent stages, each cancellable and
latency-traced:

1. **Capture** — microphone via Sherpa-ONNX VAD, or typed text
2. **ASR** — offline SenseVoice or streaming Zipformer transcription
3. **Engine** — agent reasoning loop with memory retrieval, capability telemetry, and LLM streaming
4. **Phrase chunking** — split streamed tokens into speakable phrases with TTS sanitization
5. **Playback** — Kokoro ONNX synthesis with barge-in cancellation (0.1 ms stop)

## Modes

Vokel currently has two reasoning modes:

| Mode | Reasoning owner | Tool owner | Best for |
| --- | --- | --- | --- |
| **Built-in** | Local OpenAI-compatible model through `LocalInferenceClient` | LM Studio platform | Private local chat, local vision, and LM Studio-managed capabilities |
| **Hermes** | Hermes API Server | Hermes | Speaking directly with a richer external agent stack while keeping Vokel's voice, interruption, and consent layer |

Vokel does not bundle provider-specific web, image, or GIF APIs. LM Studio owns
the capabilities configured in LM Studio. Hermes owns the capabilities
configured behind its gateway. Vokel keeps the voice loop cancellable, shows
tool activity, and renders media cards when a backend returns a real URL or
artifact. A prose-only claim that an image was fetched is displayed as prose;
it cannot produce an image card.

LM Studio's OpenAI-compatible chat endpoint streams model output but does not
execute LM Studio MCP integrations on Vokel's behalf. Platform-managed MCP
execution requires LM Studio's native chat integration path and configured MCP
servers. See [docs/agent-tools.md](docs/agent-tools.md) for the boundary.

Current vision scope is intentionally narrower: explicitly armed webcam frames
are attached to LM Studio visual turns (compat or native path). Hermes voice turns
receive an explicitly approved camera frame (VisualContext payload, both HTTP and
ws://) with visible consent, audit, and full barge-in cancellation during capture.
Hermes gateway forwarding of the frame remains a required downstream change.
See the [platform adapter build brief](docs/mcp-adapter-build-brief.md).

## Latency Scoreboard

Vokel measures the points that decide whether the loop feels alive:

- `asr_duration_ms`
- `asr_to_first_token_ms`
- `capture_duration_ms`
- `capture_to_playback_start_ms`
- `turn_to_first_token_ms`
- `turn_to_first_phrase_ms`
- `turn_to_playback_start_ms`
- `generation_duration_ms`
- `turn_duration_ms`

## Conversation Recall

Recall is explicit, local, and off by default. When enabled, Vokel can use saved
notes and recent completed turns from this machine. Timing is still visible as
`memory_retrieval_ms` / `memory_write_ms`, so recall does not quietly tax the
live loop.

```bash
python3 -m vokel.cli "What did we decide last time?" --memory
python3 -m vokel.cli --remember "Avoid Bluetooth output for latency benchmarks."
```

The web dashboard exposes the same opt-in toggle for browser and local sessions.
The engine talks to a small memory interface rather than SQLite directly, which
keeps the Android port free to use its native SQLite layer later.

## Product Direction

Vokel is not intended to become a heavy agent dashboard. The long-term direction
is a dependable voice layer that can:

- speak with any local LLM you install
- connect to external agent stacks when explicitly selected
- keep privacy state and active backend visible
- require confirmation before speech becomes high-risk action
- organize interaction history only when it helps the user understand and recall
  their own work

Project stance:

> I build it for myself, with respect for others.

## Project Memory

- `AGENTS.md` keeps contributor and AI-agent operating rules in one place.
- `ROADMAP.md` tracks measured milestones and open technical decisions.
- `build-log.md` is ignored for local experiment notes; start from `build-log.example.md` when useful.
- `docs/latency-budget.md` defines the STST timing targets.
- `docs/memory.md` explains the opt-in recall path and timing budget.
- `docs/agent-tools.md` explains the tool registry, hybrid web search, page scraping, image search, speech sanitization, and voice preview.
- `docs/agent-extensions.md` explains Hermes gateway mode (Vokel as voice front-end).
- `docs/android-termux-hermes.md` tracks the Android companion path and Termux Hermes target.
- `docs/model-matrix.md` tracks backend candidates before deeper integration.
- `docs/audio-setup.md` explains the first real microphone benchmark path.

## One-Command Install (Recommended)

```bash
git clone https://github.com/888noonie/Vokel.git
cd Vokel
chmod +x scripts/install.sh
./scripts/install.sh
```

Then:

```bash
# Beautiful web dashboard (recommended first run)
vokel --web

# Or instant CLI voice turn
vokel "Tell me something interesting about voice interfaces"
```

First time? The installer will:

- Create a clean venv
- Install everything (including optional audio stack)
- Download essential models
- Create `.env` from template
- Give you a perfect success message with next steps

## Run Against LM Studio

Start LM Studio's local server on `http://localhost:1234`, then:

```bash
vokel "Give me a compact assessment of the live voice loop architecture."
```

Or without installing the console script:

```bash
python3 -m vokel.cli "Give me a compact assessment of the live voice loop architecture."
```

## Live Local Vision Experiment

The first camera slice captures a fresh local webcam frame, sends it directly to
the loaded LM Studio vision model, prints the description and timings, then
discards the frame. On the measured Pop!_OS development laptop, `/dev/video4` is
the USB PlayStation Eye and `/dev/video0` is the built-in color webcam.

List the detected V4L2 camera nodes:

```bash
python3 scripts/live_vision.py --list-cameras
```

Analyze one PlayStation Eye frame with the loaded LM Studio model:

```bash
python3 scripts/live_vision.py --device /dev/video4 --count 1
```

Run continuously until `Ctrl-C`:

```bash
python3 scripts/live_vision.py --device /dev/video4
```

The web dashboard also exposes the same guarded path in its **Local Vision
Window**. Use **Look Now** for one frame, or **Start Live** for a visibly armed
loop with an immediate stop control. Arm **Camera Questions in Voice Loop** to
attach one fresh local frame when a spoken turn asks something grounded such as
"what am I holding?" or "what can you see?". Camera questions bypass image
search so the model answers from the captured frame rather than fetching a web
image.

This proof uses GStreamer's `gst-launch-1.0` command and keeps images local by
default. Add `--keep-frame /tmp/vokel-latest.jpg` only when an inspectable
retained frame is useful. A non-loopback `--url` also requires the explicit
`--allow-external` flag before a camera frame can leave the machine.

To capture one microphone turn, install the optional audio dependencies and provide
Sherpa-ONNX model paths:

```bash
python3 -m vokel.cli "" \
  --mic \
  --vad-model models/silero_vad.onnx \
  --asr-tokens models/sense-voice/tokens.txt \
  --sense-voice-model models/sense-voice/model.onnx
```

By default, `--mic` uses VAD-bounded turns with **offline** SenseVoice transcription.
For **streaming** Zipformer ASR (Phase 3 path), download the streaming asset
(`python3 scripts/download_models.py` includes `streaming-zipformer-en`) and pass
the model directory:

```bash
python3 -m vokel.cli "" \
  --mic \
  --vad-model models/silero_vad.onnx \
  --streaming-asr-dir models/sherpa-onnx-streaming-zipformer-en-2023-06-26
```

## Interactive Loop (TUI)

Once LM Studio is running and the Kokoro models are downloaded, start an
interactive multi-turn session:

```bash
python3 -m vokel.tui
```

Type a message and press Enter. Type `/exit` to quit.
The default playback backend is `kokoro` when model files are present.

## Web Dashboard

Build the frontend and start the web server:

```bash
make install
make start
```

Open `http://localhost:8000` in your browser.

## Hermes Gateway Mode

Hermes mode lets Vokel act as a realtime voice front-end for Hermes:

```bash
# ~/.hermes/.env
API_SERVER_ENABLED=true
API_SERVER_PORT=8642
# optional for local development
API_SERVER_KEY=change-me-local-dev
```

Start Hermes separately:

```bash
hermes gateway run
```

Then open the Vokel dashboard, choose **Agent Extension -> HERMES**, and set the
gateway URL, normally `http://127.0.0.1:8642`. Leave the API key blank if the
Hermes gateway is running without one.

See `docs/agent-extensions.md` for details.

## Test

```bash
python3 -m pytest -q
```

## Benchmark

Synthetic STST benchmark:

```bash
PYTHONPATH=src python3 -m benchmarks.stst_latency
```

Live LM Studio text-path benchmark:

```bash
PYTHONPATH=src python3 -m benchmarks.stst_latency --mode lm-studio
```

Playback stop benchmark:

```bash
PYTHONPATH=src python3 -m benchmarks.playback_latency --backend spd-say
PYTHONPATH=src python3 -m benchmarks.playback_latency --backend kokoro
```

Audible LM Studio benchmark:

```bash
PYTHONPATH=src python3 -m benchmarks.stst_latency --mode lm-studio --playback spd-say
```

Real microphone benchmark setup:

```bash
python3 scripts/download_models.py
```

Then see `docs/audio-setup.md`.

Recommended dev profile:

```bash
PYTHONPATH=src python3 -m benchmarks.stst_latency \
  --mode mic-lm-studio \
  --audio-profile laptop-mic-headphones \
  --playback spd-say
```

Streaming Zipformer mic benchmark (same LM Studio server; uses `AsynchronousMicStream`
+ online recognizer):

```bash
PYTHONPATH=src python3 -m benchmarks.stst_latency \
  --mode mic-streaming-lm-studio \
  --audio-profile laptop-mic-headphones \
  --streaming-asr-dir models/sherpa-onnx-streaming-zipformer-en-2023-06-26 \
  --playback spd-say
```

Benchmarks print a cue before capture starts:

```text
[listen] Start speaking or play test clip now...
```

Use `--no-progress` to suppress human-readable cues.

Named audio profiles:

```bash
PYTHONPATH=src python3 scripts/audio_profiles.py
PYTHONPATH=src python3 scripts/audio_routes.py --profile headset-wired
```

Latest local mic benchmark result:

```text
turn_to_first_token_ms=368.7
turn_to_playback_start_ms=674.6
```
