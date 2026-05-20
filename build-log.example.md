# Build Log

`build-log.md` is ignored by Git. Use it for local notes, timings, install attempts, model experiments, and rough observations that are useful now but should not become project history.

Copy this file to `build-log.md` when needed.

## Template

### YYYY-MM-DD

Context:

- 

Commands:

```bash

```

Results:

- 

Latency:

- `asr_duration_ms=`
- `asr_to_first_token_ms=`
- `turn_to_first_token_ms=`
- `turn_to_first_phrase_ms=`
- `turn_to_playback_start_ms=`

Decision:

- 

## Example Benchmark Entry

### 2026-05-20

Context:

- LM Studio text-path benchmark against the current local model.

Commands:

```bash
PYTHONPATH=src python3 -m benchmarks.stst_latency --mode lm-studio
```

Results:

- Record exact values from your local run.

Latency:

- `asr_duration_ms=0.0`
- `asr_to_first_token_ms=`
- `turn_to_first_token_ms=`
- `turn_to_first_phrase_ms=`
- `turn_to_playback_start_ms=`

Decision:

- Keep, tune, or reject the backend based on the latency budget.

## Example Mic Benchmark Entry

### 2026-05-20

Context:

- First real desktop mic path.
- DC offset removal enabled.
- Input gain `4`.
- Device `8`.

Commands:

```bash
PYTHONPATH=src python3 -m benchmarks.stst_latency \
  --mode mic-lm-studio \
  --audio-device 8 \
  --input-gain 4 \
  --vad-model models/silero_vad.onnx \
  --asr-tokens models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/tokens.txt \
  --sense-voice-model models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx
```

Results:

- VAD produced a segment.
- SenseVoice ASR returned a non-empty transcript.
- LM Studio returned a response.

Latency:

- `asr_duration_ms=112.8`
- `asr_to_first_token_ms=368.7`
- `turn_to_first_token_ms=368.7`
- `turn_to_first_phrase_ms=674.1`
- `turn_to_playback_start_ms=674.6`

Decision:

- Plausible. Proceed to real TTS playback and barge-in measurement.
