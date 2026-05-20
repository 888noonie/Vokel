# Latency Budget

Voyce optimizes for perceived conversational immediacy, not just raw throughput.

## Primary Checkpoints

| Metric | Target | Notes |
| --- | ---: | --- |
| `barge_in_to_playback_stop_ms` | `<150` | Most important interactive safety metric. |
| `turn_to_first_token_ms` | `<500` | Keeps the model from feeling absent. |
| `turn_to_first_phrase_ms` | `<900` | First speakable text unit. |
| `turn_to_playback_start_ms` | `<1000` | The key "it is answering me" moment. |
| `asr_duration_ms` | `<350` | Applies after VAD turn completion for batch ASR. |
| `generation_duration_ms` | variable | Less important than first useful audio. |

## Current Measurement Modes

Synthetic benchmark:

```bash
PYTHONPATH=src python3 -m benchmarks.stst_latency
```

Live LM Studio text-path benchmark:

```bash
PYTHONPATH=src python3 -m benchmarks.stst_latency --mode lm-studio
```

Full JSON trace:

```bash
PYTHONPATH=src python3 -m benchmarks.stst_latency --json
```

## Interpretation

- If first token is slow, tune or replace the LLM/runtime.
- If first phrase is slow, tune phrase chunking and prompts.
- If playback start is slow, tune TTS synthesis and audio output.
- If ASR duration is slow, use streaming ASR or a smaller model.
- If barge-in is slow, stop optimizing everything else until interruption is fixed.

## Design Bias

The fastest local STST path is expected to be cascaded and streaming:

`VAD / streaming ASR -> local LLM stream -> phrase chunker -> local TTS stream`

Native end-to-end speech models remain worth watching, but the current production bet is still a measurable cascaded pipeline.
