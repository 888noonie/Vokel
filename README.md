# Voyce

Voyce is a local-first conversational voice loop. The product test is simple:

You speak. It answers. You interrupt. It stops. It listens again. No button.

This repository starts with a desktop reference core before the Android port. The desktop core gives us a production-quality behavioral baseline for:

- async streaming from an OpenAI-compatible local server such as LM Studio
- cancellable generation for barge-in
- phrase chunking for low-latency TTS
- clean queue boundaries between ASR, LLM, and playback

## First Milestone

The first milestone is not a full voice app. It is a reliable turn-in/phrase-out stream that behaves like the future voice loop:

1. send a user turn to LM Studio
2. stream tokens asynchronously
3. split text into speakable phrases
4. hand phrases to a playback worker
5. cancel generation and flush playback when interrupted

Once that loop is stable, we add microphone/VAD/ASR as producers and real TTS as a consumer.

## Latency Scoreboard

Voyce measures the points that decide whether the loop feels alive:

- `asr_duration_ms`
- `asr_to_first_token_ms`
- `turn_to_first_token_ms`
- `turn_to_first_phrase_ms`
- `turn_to_playback_start_ms`
- `generation_duration_ms`
- `turn_duration_ms`

The current CLI uses a passthrough ASR producer so the async LLM and phrase playback path can be measured before microphone capture is introduced.

## Research Direction

Current latency work points toward three production choices:

- use Sherpa-ONNX as the Android speech spine for VAD, streaming ASR, and potentially TTS
- prefer streaming ASR over batch Whisper-style transcription
- overlap ASR, LLM, and TTS wherever cancellation remains reliable

The immediate code path is still desktop-first. That keeps the behavior easy to measure before the Android Foreground Service port.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional audio dependencies:

```bash
sudo apt install -y portaudio19-dev
pip install -e ".[audio,dev]"
```

## Run Against LM Studio

Start LM Studio's local server on `http://localhost:1234`, then:

```bash
voyce "Give me a compact assessment of the live voice loop architecture."
```

Or without installing the console script:

```bash
python3 -m voyce.cli "Give me a compact assessment of the live voice loop architecture."
```

To capture one microphone turn, install the optional audio dependencies and provide
Sherpa-ONNX model paths:

```bash
python3 -m voyce.cli "" \
  --mic \
  --vad-model models/silero_vad.onnx \
  --asr-tokens models/sense-voice/tokens.txt \
  --sense-voice-model models/sense-voice/model.onnx
```

The microphone path currently uses VAD-bounded turns. Streaming ASR is the next
step after this first measurable audio producer.

## Test

```bash
python3 -m unittest discover -s tests
```
