# Model Matrix

This file tracks candidates before we commit deeper integration work.

## ASR / VAD

| Candidate | Role | Why It Matters | Current Status |
| --- | --- | --- | --- |
| Sherpa-ONNX Silero VAD | VAD | Existing Python and Android path. | Scaffolded in `src/voyce/audio.py`. |
| Sherpa-ONNX SenseVoice | ASR | Good offline multilingual candidate. | Adapter scaffolded. Needs model files. |
| Sherpa-ONNX Moonshine | ASR | Interesting low-latency edge direction. | Adapter scaffolded. Needs measurement. |
| Streaming Zipformer | ASR | True streaming ASR path through Sherpa. | Future adapter. |
| Whisper | ASR | Strong baseline, likely slower for live turns. | Adapter scaffolded for comparison. |

## TTS

| Candidate | Role | Why It Matters | Current Status |
| --- | --- | --- | --- |
| Kokoro ONNX | TTS | Lightweight, Android-capable, pleasant quality. | Preferred first real TTS benchmark. |
| Sherpa-ONNX TTS | TTS | One speech runtime across ASR/VAD/TTS. | Candidate. |
| Piper | TTS | Simple, proven local baseline. | Candidate for quick desktop sink. |
| Speech Dispatcher `spd-say` | TTS bridge | Already available on the desktop test machine and cancellable. | Working bridge backend; not final fastest voice. |

## LLM Runtime

| Candidate | Role | Why It Matters | Current Status |
| --- | --- | --- | --- |
| LM Studio | Desktop LLM server | Already fast on RTX 4050. | Active baseline. |
| llama.cpp | Desktop/Android GGUF | Important local/mobile runtime. | Future benchmark. |
| LiteRT / AI Edge | Android runtime | May matter on Pixel 8 Pro. | Future benchmark. |
| MLC | Android runtime | Alternative mobile LLM path. | Future benchmark. |

## Selection Rule

No backend graduates from this matrix until it produces benchmark numbers for:

- first token
- first phrase
- first playback
- interruption stop time
- memory and thermal behavior where relevant

## Audio Profiles

| Profile | Purpose | Current Status |
| --- | --- | --- |
| `laptop-mic-headphones` | Recommended dev baseline: stable digital mic input with headphone playback. | Added. |
| `laptop-open` | Existing laptop mic baseline with DC offset correction. | Working. |
| `headset-wired` | Headset-first baseline for low feedback and cleaner barge-in. | Added; needs repeated runs. |
| `headset-bluetooth` | Bluetooth headset path for Android-like use. | Added; needs route-specific tuning. |
| `noisy-handset` | Future open-air/noisy profile. | Placeholder. |
