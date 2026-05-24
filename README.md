# Vokel

**Voice-invoked local intelligence.**

Vokel is your private, voice-first AI that runs entirely on your own hardware.
No cloud. No subscription. No data leaving your machine. You speak, it acts.

You speak. It answers. You interrupt. It stops. It listens again.

Interruption works hands-free with a headset (separate mic and speaker). On
laptop speakers the BARGE IN button in the web dashboard is the reliable path,
because the mic hears its own TTS output without echo cancellation.

## Why Vokel?

The name comes from three roots: **Vok-** (Latin *vocare* — to call or invoke),
**Vocal** (it's voice-first), and **Local** (it runs on your hardware, private
by default). You invoke tools, evoke responses, and provoke searches — all by
speaking.

## What It Does

Vokel turns any local LLM into a live conversational agent you control with
your voice:

- **Talk naturally** — speak a question, get a spoken answer, interrupt mid-sentence
- **Invoke tools by voice** — "Search for the latest UK news", "Show me an image of the Alps"
- **Stay private** — everything runs locally: ASR, LLM, TTS, memory, tools
- **Choose your model** — works with any OpenAI-compatible server (LM Studio, llama.cpp, MLC LLM)
- **Choose your voice** — 28 bundled Kokoro voices with preview and speed control

## Architecture

The voice loop runs through five independent stages, each cancellable and
latency-traced:

1. **Capture** — microphone via Sherpa-ONNX VAD, or typed text
2. **ASR** — offline SenseVoice or streaming Zipformer transcription
3. **Engine** — agent reasoning loop with tool registry, memory retrieval, and LLM streaming
4. **Phrase chunking** — split streamed tokens into speakable phrases with TTS sanitization
5. **Playback** — Kokoro ONNX synthesis with barge-in cancellation (0.1 ms stop)

## Agent Tools

Vokel extends small local models with a tool registry that runs external
capabilities without leaking implementation into the model itself.

### Web Search

The `search_web` tool uses SerpApi's DuckDuckGo engine. When you ask for news,
weather, or current information, Vokel runs the search deterministically, then
hands the evidence to the local model with a strict synthesis prompt so it
answers conversationally from real data. If API snippets are too thin, Vokel
scrapes the top result page for actual article content. A fallback path returns
the raw numbered evidence if the model still hedges.

### Image Search

The `search_image` tool uses the Unsplash API. Say "show me an image of a
golden retriever" and a landscape photograph appears inline in the transcript
with proper attribution, while the TTS gives a brief spoken introduction.

### GIF Search

The `search_gif` tool uses the Giphy API. Ask for a GIF, reaction, meme, or
sticker and an animated GIF appears in a compact card in the transcript. The
detection is context-aware: if the conversation is already about GIFs, short
follow-ups like "another one" or "cats" automatically trigger a new search.

The TTS speaks a playful one-liner ("Ha! Check this out!") instead of reading
GIF metadata aloud.

Search evidence stays visible in the transcript with clickable links. The TTS
path receives cleaned speech text that strips Markdown markers, raw URLs, and
formatting symbols before synthesis. The web UI plays a quiet tonal cue while
an external tool call is running.

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

## Project Memory

- `AGENTS.md` keeps contributor and AI-agent operating rules in one place.
- `ROADMAP.md` tracks measured milestones and open technical decisions.
- `build-log.md` is ignored for local experiment notes; start from `build-log.example.md` when useful.
- `docs/latency-budget.md` defines the STST timing targets.
- `docs/memory.md` explains the opt-in recall path and timing budget.
- `docs/agent-tools.md` explains the tool registry, hybrid web search, page scraping, image search, speech sanitization, and voice preview.
- `docs/agent-extensions.md` explains Hermes gateway mode (Vokel as voice front-end).
- `docs/model-matrix.md` tracks backend candidates before deeper integration.
- `docs/audio-setup.md` explains the first real microphone benchmark path.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Or using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```INFO:     127.0.0.1:33950 - "GET /favicon.svg HTTP/1.1" 304 Not Modified

Optional audio dependencies:

```bash
sudo apt install -y portaudio19-dev
pip install -e ".[audio,dev]"
```

## Run Against LM Studio

Start LM Studio's local server on `http://localhost:1234`, then:

```bash
vokel "Give me a compact assessment of the live voice loop architecture."
```

Or without installing the console script:

```bash
python3 -m vokel.cli "Give me a compact assessment of the live voice loop architecture."
```

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