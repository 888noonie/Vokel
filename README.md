# Vokel

Vokel is a local-first voice interface for local language models and external
agent stacks. The loop it is built around is deliberately simple:

```text
You speak. It answers. You interrupt. It stops. It listens again.
```

In its local mode, Vokel runs speech recognition, speech synthesis, memory and
the model connection on the same machine. In external-agent mode it connects to
an agent gateway over HTTP or WebSocket, hands the reasoning seat to that agent,
and keeps ownership of voice capture, playback, interruption, routing, consent,
transcript and audit display.

Vokel is a Python backend with a React/TypeScript dashboard, plus a CLI and a
small terminal UI. It is intended to be a voice and display layer, not a second
agent dashboard: models and connected agents own their own tools, memory and
configuration.

## Status

**Dormant since 8 July 2026.** The repository has 62 commits dated May–July 2026
and declares version 0.2.0 in `pyproject.toml`. There is no CI workflow and no
licence file.

The project's roadmap records phases 0–4 and the first dashboard and
agent-extension slices as complete. The connection layer (phase 5), the Android
companion (phase 6), wake-word and consent work (phase 7), richer media handling
and later phases are listed as next or planned.

## What demonstrably works

This review did not install or run Vokel, so the items below describe what is
implemented and covered by tests in the source tree, not behaviour observed here.

- **A working-shaped core is present.** `src/vokel/` contains the conversation
  engine, streaming local inference client, external-agent HTTP and WebSocket
  clients, phrase chunking, playback sinks, telemetry, memory, tools, vision and
  the FastAPI web application.
- **The voice loop is implemented as separate stages**: capture and VAD, ASR,
  engine streaming, phrase chunking, and playback with cancellation. Audio code
  lives under `src/vokel/audio/`.
- **Local speech components are wired.** The code includes Sherpa-ONNX VAD,
  offline SenseVoice and streaming Zipformer ASR paths, and Kokoro ONNX speech
  synthesis via `kokoro-onnx`.
- **Two reasoning modes exist in code.** One uses any OpenAI-compatible local
  server through `LocalInferenceClient` (LM Studio is the documented example).
  The other connects to an external agent gateway and streams turns back into
  Vokel.
- **Memory is explicit and local.** Conversation recall is off unless enabled
  with `--memory` or the dashboard toggle. It stores notes and completed turns in
  a local SQLite file under `data/` and reports retrieval/write timing.
- **A web dashboard, CLI and TUI are present**, along with consent and audit
  events for camera frames and tool activity.
- **29 test modules are present** under `tests/`, covering the engine, audio
  paths, inference, memory, web layer, benchmark code and musical-mode pieces.

Project notes from 28 May 2026 report `96 passed` for the pytest suite. That is
a project claim; the suite was not run for this review.

## Experimental, incomplete or planned

- **Musical mode is explicitly experimental.** It adds a drift-corrected beat
  grid, a synthesized or loaded backing track, and beat-aligned phrase starts,
  while leaving the normal voice loop intact. It currently targets local playback
  with Kokoro or `spd-say`; browser-audio and external-agent playback paths remain
  unquantized. Drift is corrected with manual ±25 ms nudges. Sidechain ducking,
  syllable-level prosody, browser quantization, automatic beat detection and a
  frame-counter beat clock are listed as ideas, not built.
- **Camera and vision support is partial.** Locally, explicitly armed webcam
  frames can be attached to a local-model turn through a GStreamer capture path.
  For external agents, a consented frame can be sent over HTTP or WebSocket, but
  gateway-side forwarding is described in the project's own notes as a required
  downstream change. The Android camera path does not exist yet.
- **The connection layer is the next slice.** Saved connections and transparent
  routing between local and external backends are roadmap work. Today the mode
  and endpoint are selected through the UI/CLI configuration.
- **The Android companion is planned, not implemented.** There is design material
  for a Termux-hosted agent on the same device; no Android application is in this
  repository.
- **Wake word, pause/resume and a general execution-consent boundary are planned.**
  The current consent surface covers specific actions such as camera capture; a
  finished universal permission system for high-risk actions does not exist.
- **Memory is a first slice.** It deliberately has no embeddings, summarization
  calls or cloud sync.
- **Tool ownership is split by design.** In local mode, LM Studio owns its
  configured tools and MCP servers; Vokel does not execute them on the model's
  behalf unless the native LM Studio chat path is used. In external-agent mode,
  the agent owns its tools. There is no bundled tool marketplace or universal
  tool runtime.
- **No packaged release is evidenced.** There is no CI build, published wheel or
  installer artifact in the repository.

## Install, run or inspect

Requirements:

- Python 3.11 or newer.
- Node.js and npm for the React dashboard (only needed to build or develop it).
- For local voice: PortAudio, a working microphone and speakers/headset, and the
  Sherpa-ONNX and Kokoro model files. Model weights are not in the repository.
- For local reasoning: an OpenAI-compatible server such as LM Studio.
- For external-agent mode: a running agent gateway.
- `ffmpeg` only if you want to load your own musical backing track.

Install with the provided script:

```bash
cd <path-to-repo>
chmod +x scripts/install.sh
./scripts/install.sh
```

The script creates `.venv`, installs the package with its dev and audio extras,
tries to install PortAudio through `apt-get`, downloads the minimal models and
creates `.env` from `.env.example`. On a non-Debian system, install PortAudio
with the distribution's package manager instead and run the remaining steps
manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
uv pip install -e ".[dev,audio]"
```

Run a single text turn against a local OpenAI-compatible server:

```bash
vokel "Give me a short assessment of the live voice loop architecture."
# or, without the console script:
python3 -m vokel.cli "Give me a short assessment of the live voice loop architecture."
```

Run the dashboard:

```bash
# after the installer has created .venv
make start      # installs frontend modules if needed, builds, serves on port 8000
```

Then open `http://localhost:8000`. `make start` binds to `0.0.0.0` by default;
`python3 -m vokel.cli --web --host 127.0.0.1` keeps the server local.

Run the terminal UI:

```bash
python3 -m vokel.tui
```

Run the tests and benchmarks:

```bash
python3 -m pytest -q
PYTHONPATH=src python3 -m benchmarks.stst_latency
```

Inspect the design in `docs/`, especially `docs/latency-budget.md`,
`docs/memory.md`, `docs/agent-tools.md`, `docs/agent-extensions.md` and
`docs/android-termux-hermes.md`.

## Tests and build evidence

**Observed in this review**

- 29 `test_*.py` modules are present under `tests/`.
- `pyproject.toml` configures pytest, Ruff and mypy; the frontend has a Vite
  build and an ESLint configuration. The Makefile has `install`, `build`,
  `start`, `test` and `clean` targets.
- There is no `.github/` directory and therefore no CI workflow.
- No test, build, install or benchmark command was executed. No model files,
  virtual environment or running model server were present.

**Claimed by the project**

- `SESSION_HANDOVER.md` (28 May 2026) records `96 passed` in the project virtual
  environment.
- The README's June 2026 status says the external-agent WebSocket bridge is
  "fully working", with token-by-token streaming validated against a stub on a
  separate phone.
- `ROADMAP.md` marks phases 0–4 and two dashboard/agent slices complete.
- The README quotes a local microphone benchmark of
  `turn_to_first_token_ms=368.7` and `turn_to_playback_start_ms=674.6`.

Those are project records. They were not reproduced here, and the quoted
benchmark numbers depend on the machine, microphone, model server and audio
route used at the time.

## Limitations and dependencies

- **External services are not bundled.** A local model server, model weights and,
  in external mode, an agent gateway are prerequisites. `install.sh` downloads
  some models but cannot provide a model server.
- **The audio stack is platform-dependent.** PortAudio, microphone routing and
  echo behaviour vary by machine. The installer's PortAudio step assumes an
  `apt`-based distribution; other systems need manual setup.
- **Laptop speakers make interruption unreliable.** The project's own guidance is
  to use a headset with a separate microphone, or the dashboard's barge-in
  control, because the microphone otherwise hears the synthesized speech.
- **The dashboard imports the optional audio stack.** Its web module imports
  NumPy and the audio package at startup, so a Python install without the audio
  extra is not sufficient to run the dashboard.
- **The web server binds to all interfaces by default.** Use `--host 127.0.0.1`
  when the dashboard should only be reachable from the same machine.
- **Memory is local and basic.** Recall is off by default and limited to recent
  turns and saved notes; it is not a general knowledge store and does not sync.
- **Musical mode is local-playback only and experimental**, with manual drift
  correction.
- **Vision depends on hardware and explicit arming.** It needs a camera and
  GStreamer for the local path; the external-agent forwarding path is not
  finished.
- **The project is dormant and has no CI.** A fresh checkout has no automated
  evidence that the suite or the frontend build passes on your machine.
- **Dependency licences were not audited here.** The installed stack is larger
  than the repository and includes audio, web and frontend dependencies.

## Licence

**No licence is granted.** There is no `LICENSE` file in the repository, and the
reuse and redistribution terms are undecided. Contact the copyright holder
before reusing the code.
