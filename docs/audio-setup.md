# Audio Setup

This guide prepares the first measurable desktop speech path:

`microphone -> Silero VAD -> SenseVoice ASR -> LM Studio -> phrase playback sink`

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

This downloads:

- `models/silero_vad.onnx`
- `models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx`
- `models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/tokens.txt`

Sources:

- Sherpa-ONNX Silero VAD documentation: https://k2-fsa.github.io/sherpa/onnx/vad/silero-vad.html
- Sherpa-ONNX SenseVoice documentation: https://k2-fsa.github.io/sherpa/onnx/sense-voice/pretrained.html

## First Mic Benchmark

Start LM Studio, load the local chat model, and enable the server on port `1234`.

Then run:

```bash
PYTHONPATH=src python3 -m benchmarks.stst_latency \
  --mode mic-lm-studio \
  --audio-device 8 \
  --input-gain 4 \
  --vad-model models/silero_vad.onnx \
  --asr-tokens models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/tokens.txt \
  --sense-voice-model models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx
```

Speak one short sentence and pause. The benchmark should report ASR and LLM timings.

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
Voyce now removes DC offset before VAD by default. If RMS looks loud but VAD never
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

## Current Limits

- This first mic path is VAD-bounded, not true streaming ASR.
- Playback is still a benchmark sink, not real TTS audio.
- Barge-in measurement starts after real playback exists.
