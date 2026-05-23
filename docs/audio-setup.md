# Audio Setup

This guide prepares the first measurable desktop speech path:

`microphone -> Silero VAD -> SenseVoice ASR (offline) -> LM Studio -> phrase playback sink`

For **streaming** ASR (Phase 3), the ASR stage is Sherpa streaming Zipformer instead
of SenseVoice; see [Streaming Zipformer mic benchmark](#streaming-zipformer-mic-benchmark)
below.

## System Dependencies

On Pop!_OS/Ubuntu:

```bash
sudo apt update
sudo apt install -y python3.12-venv portaudio19-dev
```

Then create a local environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[audio,dev]"
```

If `python3.12-venv` is not installed yet, the venv step will fail until the OS package is present.

## Model Downloads

Model weights are stored under `models/`, which is ignored by Git.

```bash
python3 scripts/download_models.py
```

This downloads (among others):

- `models/silero_vad.onnx`
- `models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx`
- `models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/tokens.txt`
- `models/sherpa-onnx-streaming-zipformer-en-2023-06-26/` (streaming Zipformer, Phase 3)

Sources:

- Sherpa-ONNX Silero VAD documentation: https://k2-fsa.github.io/sherpa/onnx/vad/silero-vad.html
- Sherpa-ONNX SenseVoice documentation: https://k2-fsa.github.io/sherpa/onnx/sense-voice/pretrained.html

## First Mic Benchmark

Start LM Studio, load the local chat model, and enable the server on port `1234`.

Then run:

```bash
PYTHONPATH=src python3 -m benchmarks.stst_latency \
  --mode mic-lm-studio \
  --audio-profile laptop-mic-headphones \
  --vad-model models/silero_vad.onnx \
  --asr-tokens models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/tokens.txt \
  --sense-voice-model models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx
```

Speak one short sentence and pause. The benchmark should report ASR and LLM timings.
Start the test clip only after this cue appears:

```text
[listen] Start speaking or play test clip now...
```

Use `--no-progress` for machine-readable or unattended runs.

## Streaming Zipformer mic benchmark

After the default download (includes the streaming Zipformer directory), you can
measure the online recognizer path:

```bash
PYTHONPATH=src python3 -m benchmarks.stst_latency \
  --mode mic-streaming-lm-studio \
  --audio-profile laptop-mic-headphones \
  --vad-model models/silero_vad.onnx \
  --streaming-asr-dir models/sherpa-onnx-streaming-zipformer-en-2023-06-26 \
  --playback spd-say
```

The CLI accepts the same `--streaming-asr-dir` with `--mic` for interactive runs;
see `README.md`.

## Audio Profiles

List profiles:

```bash
PYTHONPATH=src python3 scripts/audio_profiles.py
```

Inspect actual Pulse/PipeWire routes for a profile:

```bash
PYTHONPATH=src python3 scripts/audio_routes.py --profile headset-wired
```

The route script marks the current default source and whether the profile's
preferred source is available. Do not trust a profile benchmark until the route
is usable.

Example route warning:

```text
Family 17h/19h HD Audio Controller Headphones Stereo Microphone [profile]
  available=not available usable=False
Family 17h/19h HD Audio Controller Digital Microphone [default]
  available=availability unknown usable=True
```

That means the headset profile exists, but the OS is not currently exposing the
headset mic as an available input. For wired headsets, check that the plug/cable
supports a microphone path and reseat the connector before comparing benchmarks.

Current profiles:

- `laptop-mic-headphones` - recommended dev baseline
- `laptop-open`
- `headset-wired`
- `headset-bluetooth`
- `noisy-handset`

Use a profile, then override individual values only when needed:

```bash
PYTHONPATH=src python3 -m benchmarks.stst_latency \
  --mode mic-lm-studio \
  --audio-profile headset-wired \
  --input-gain 2 \
  --vad-threshold 0.35 \
  --vad-model models/silero_vad.onnx \
  --asr-tokens models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/tokens.txt \
  --sense-voice-model models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx
```

To prevent false headset benchmarks, require the profile route:

```bash
PYTHONPATH=src python3 -m benchmarks.stst_latency \
  --mode mic-lm-studio \
  --audio-profile headset-wired \
  --require-profile-route \
  --vad-model models/silero_vad.onnx \
  --asr-tokens models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/tokens.txt \
  --sense-voice-model models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx
```

## Audio Probes

List input devices:

```bash
python3 scripts/audio_probe.py --list
```

Check raw input level:

```bash
python3 scripts/audio_probe.py --device 8 --seconds 2
```

Probe Sherpa VAD directly:

```bash
python3 scripts/vad_probe.py --device 8 --threshold 0.25 --input-gain 4
```

On the first Pop!_OS test machine, PipeWire/Pulse input had a large DC offset.
Vokel now removes DC offset before VAD by default. If RMS looks loud but VAD never
detects speech, check `raw_mean` in `scripts/vad_probe.py` and use `--input-gain`
to lift the centered signal.

## First Plausibility Result

On the initial local test run:

```text
benchmark=mic-lm-studio
asr_duration_ms=112.8
asr_to_first_token_ms=368.7
turn_to_first_token_ms=368.7
turn_to_first_phrase_ms=674.1
turn_to_playback_start_ms=674.6
generation_duration_ms=675.0
turn_duration_ms=675.0
```

This used:

- device `8`
- `--input-gain 4`
- Silero VAD threshold `0.25`
- SenseVoice int8 ASR
- LM Studio local chat server

## Headset Profile Result

Using `--audio-profile headset-wired` against the speech/noise loop:

```text
benchmark=mic-lm-studio
capture_duration_ms=13452.3
capture_to_first_token_ms=13990.7
capture_to_playback_start_ms=14143.4
asr_duration_ms=132.8
asr_to_first_token_ms=405.6
turn_to_first_token_ms=405.6
turn_to_first_phrase_ms=557.7
turn_to_playback_start_ms=558.3
generation_duration_ms=669.4
turn_duration_ms=669.4
```

The desktop default source may still point to the internal digital mic even when
the headset stereo mic is present. Use `pactl list short sources` and the probe
scripts to confirm routing before comparing profile results.

## Recommended Dev Baseline

For current desktop work, use laptop digital mic input with headphone-isolated
playback:

```bash
PYTHONPATH=src python3 -m benchmarks.stst_latency \
  --mode mic-lm-studio \
  --audio-profile laptop-mic-headphones \
  --playback spd-say \
  --vad-model models/silero_vad.onnx \
  --asr-tokens models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/tokens.txt \
  --sense-voice-model models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx
```

First `laptop-mic-headphones` audible benchmark:

```text
benchmark=mic-lm-studio
capture_duration_ms=15362.6
capture_to_first_token_ms=16000.6
capture_to_playback_start_ms=16744.2
asr_duration_ms=256.1
asr_to_first_token_ms=382.0
turn_to_first_token_ms=379.0
turn_to_first_phrase_ms=1122.4
turn_to_playback_start_ms=1122.5
generation_duration_ms=1275.7
turn_duration_ms=1275.8
```

This profile is honest and stable for development, but the first audible run
also shows why phrase-boundary tuning matters: first playback crossed the
`1000ms` target on that response.

## Current Limits

- Offline mic path (`mic-lm-studio`) is VAD-bounded batch ASR on each completed turn.
- Streaming Zipformer (`mic-streaming-lm-studio` / `--streaming-asr-dir`) is the
  integrated Phase 3 path; partial-to-LLM overlap is still future work.
- Playback is still a benchmark sink, not real TTS audio.
- Barge-in measurement starts after real playback exists.
