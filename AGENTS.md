# AGENTS

This file keeps human and AI contributors aligned while Vokel is still small enough to steer deliberately.

## Product Test

Vokel must preserve the live conversation loop:

You speak. It answers. You interrupt. It stops. It listens again. No button.

Every architectural choice should protect that loop.

## Current Shape

- Desktop reference core first.
- Android companion and Foreground Service next, after desktop behavior has been measured locally.
- Local-first by default.
- External agents and cloud services may be used when explicitly selected, but core voice-loop behavior should not depend on them.
- Vokel owns voice capture, playback, interruption, routing, consent, and audit; external agents own their own tools and memory.

## Engineering Rules

- Prefer small, measurable slices over large feature drops.
- Keep ASR, LLM, TTS, and tools behind explicit interfaces.
- Add latency marks before optimizing a path.
- Keep every stream cancellable.
- Do not let microphone, playback, model, or tool code leak through the whole system.
- Register new external capabilities as `ToolDefinition` entries in the `ToolRegistry`.
- In external-agent mode, do not duplicate agent-owned tools inside Vokel; expose state and consent instead.
- Sanitize LLM output for TTS through the `sanitize_for_speech` path; never read raw Markdown or URLs aloud.
- Use deterministic forced tool execution for capabilities that small models cannot reliably self-invoke.
- Prefer named audio profiles over scattering one-off VAD settings through commands.
- Verify Pulse/PipeWire route availability before trusting headset profile benchmarks.
- Use `laptop-mic-headphones` as the default desktop dev profile until a true headset mic route is available.
- Preserve clear listen/capture cues so manual audio tests start from a fair point.
- Avoid committing model weights, generated audio, caches, API keys, or local build notes.
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
