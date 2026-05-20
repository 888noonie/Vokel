# Latency Budget

Voyce optimizes for perceived conversational immediacy, not just raw throughput.

## Primary Checkpoints

| Metric | Target | Notes |
| --- | ---: | --- |
| `barge_in_to_playback_stop_ms` | `<150` | Most important interactive safety metric. |
| `capture_duration_ms` | variable | Time spent waiting for VAD to close a user turn. |
| `capture_to_playback_start_ms` | context-dependent | Full benchmark wall-clock path from listen start. |
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

Mic + LM Studio benchmark:

```bash
PYTHONPATH=src python3 -m benchmarks.stst_latency \
  --mode mic-lm-studio \
  --audio-device 8 \
  --input-gain 4 \
  --vad-model models/silero_vad.onnx \
  --asr-tokens models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/tokens.txt \
  --sense-voice-model models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx
```

Playback stop benchmark:

```bash
PYTHONPATH=src python3 -m benchmarks.playback_latency --backend spd-say
```

LM Studio with audible `spd-say` playback:

```bash
PYTHONPATH=src python3 -m benchmarks.stst_latency \
  --mode lm-studio \
  --playback spd-say
```

Full JSON trace:

```bash
PYTHONPATH=src python3 -m benchmarks.stst_latency --json
```

JSON mode disables progress cues automatically. For normal human tests, the cue
to start speaking or play a clip is:

```text
[listen] Start speaking or play test clip now...
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
