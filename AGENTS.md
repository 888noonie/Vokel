# AGENTS

This file keeps human and AI contributors aligned while Voyce is still small enough to steer deliberately.

## Product Test

Voyce must preserve the live conversation loop:

You speak. It answers. You interrupt. It stops. It listens again. No button.

Every architectural choice should protect that loop.

## Current Shape

- Desktop reference core first.
- Android Foreground Service later, after behavior is measured locally.
- Local-first by default.
- Cloud services may be used for research or audit, but core voice-loop behavior should not depend on them.

## Engineering Rules

- Prefer small, measurable slices over large feature drops.
- Keep ASR, LLM, and TTS behind explicit interfaces.
- Add latency marks before optimizing a path.
- Keep every stream cancellable.
- Do not let microphone, playback, or model code leak through the whole system.
- Prefer named audio profiles over scattering one-off VAD settings through commands.
- Verify Pulse/PipeWire route availability before trusting headset profile benchmarks.
- Use `laptop-mic-headphones` as the default desktop dev profile until a true headset mic route is available.
- Preserve clear listen/capture cues so manual audio tests start from a fair point.
- Avoid committing model weights, generated audio, caches, or local build notes.
- Run `python3 -m pytest -q` before publishing code changes.

## Latency Priorities

Optimize in this order:

1. interruption correctness
2. first-token latency
3. first-playback latency
4. ASR turn completion latency
5. total response duration

Fast output is not useful if barge-in fails.

## Research Notes

When adding a new model, library, or repo pattern, record the reason in `ROADMAP.md` or the local ignored `build-log.md`.

Prefer primary sources for fast-moving dependencies such as Sherpa-ONNX, llama.cpp, Android NNAPI/GPU backends, and on-device TTS projects.

## Cursor Cloud specific instructions

### Activate the virtualenv

All commands assume the project venv is active:

```bash
source .venv/bin/activate
```

### Key commands

| Task | Command |
|---|---|
| Lint | `ruff check src/ tests/` |
| Type-check | `mypy src/voyce/` |
| Tests | `python3 -m pytest -q` |
| Synthetic benchmark | `PYTHONPATH=src python3 -m benchmarks.stst_latency` |
| CLI (needs LM Studio) | `python3 -m voyce.cli "prompt"` |
| TUI (needs LM Studio) | `python3 -m voyce.tui` |

### Notes

- **LM Studio is unavailable** in the cloud VM. The CLI and TUI require it at `localhost:1234`. Use the synthetic benchmark mode (no external deps) to verify the engine pipeline.
- **mypy has a pre-existing error** in `src/voyce/audio.py:41` (assigning `None` to a module-typed variable in an optional-import guard). This is expected and not a regression.
- **numpy is needed for all tests to pass.** The update script installs `.[dev]` only. If 4 tests in `test_audio_probe` / `test_vad_probe` fail with `ModuleNotFoundError: numpy`, run `pip install numpy` inside the venv.
- The `[audio]` optional extra (`pip install -e ".[audio,dev]"`) pulls heavier native deps (sounddevice, sherpa-onnx, kokoro-onnx) and requires `portaudio19-dev`. Only install when working on audio-path code.
