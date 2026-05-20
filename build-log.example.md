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

## Example Playback Benchmark Entry

### 2026-05-20

Context:

- First cancellable audible backend using `spd-say`.

Commands:

```bash
PYTHONPATH=src python3 -m benchmarks.playback_latency --backend spd-say --interrupt-after-ms 500
```

Results:

- Playback stop latency was low enough for barge-in experiments.

Latency:

- `stop_latency_ms=3.1`
- `speak_completion_after_stop_ms=0.0`

Decision:

- Use `spd-say` as a bridge backend while benchmarking better local TTS options.

## Example Headset Profile Entry

### 2026-05-20

Context:

- `headset-wired` profile against speech/noise loop.

Commands:

```bash
PYTHONPATH=src python3 -m benchmarks.stst_latency \
  --mode mic-lm-studio \
  --audio-profile headset-wired \
  --vad-model models/silero_vad.onnx \
  --asr-tokens models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/tokens.txt \
  --sense-voice-model models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx
```

Latency:

- `capture_duration_ms=13452.3`
- `asr_duration_ms=132.8`
- `turn_to_first_token_ms=405.6`
- `turn_to_playback_start_ms=558.3`

Decision:

- Keep headset-first product constraint. Continue route-specific tuning.

## Example Recommended Dev Profile Entry

### 2026-05-20

Context:

- `laptop-mic-headphones` profile.
- Laptop digital mic input.
- Headphone-isolated `spd-say` playback.

Commands:

```bash
PYTHONPATH=src python3 -m benchmarks.stst_latency \
  --mode mic-lm-studio \
  --audio-profile laptop-mic-headphones \
  --playback spd-say \
  --vad-model models/silero_vad.onnx \
  --asr-tokens models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/tokens.txt \
  --sense-voice-model models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx
```

Latency:

- `capture_duration_ms=15362.6`
- `asr_duration_ms=256.1`
- `turn_to_first_token_ms=379.0`
- `turn_to_playback_start_ms=1122.5`

Decision:

- Use as the default desktop development profile.
- Tune phrase chunking/prompting to bring first playback below `1000ms`.
