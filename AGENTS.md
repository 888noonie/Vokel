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
