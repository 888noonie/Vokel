# Android And Termux Hermes

This document tracks the intended Android companion path for Vokel.

The first Android goal is not to run every model locally on the phone. The first
goal is to preserve Vokel's voice loop on Android while routing conversation to
a Hermes agent running in Termux on the same device.

## Target Shape

```text
Android Vokel
-> microphone, playback, wake/listen cues, interruption, consent
-> Hermes API Server in Termux
-> Hermes-selected model/provider and Hermes-owned tools
```

Vokel remains the voice, interruption, routing, and audit layer. Hermes remains
the reasoning agent with its own tools, memory, provider configuration, and
sessions.

## First Slice

- Android-friendly web/PWA dashboard layout
- configurable Hermes gateway URL
- documented Termux Hermes API Server setup
- `/health` check from Vokel to Hermes
- voice handoff to Hermes mode
- visible active backend and privacy state
- touchless interrupt and resume where the Android audio path permits it

## Hermes Setup In Termux

Hermes should expose its API Server:

```bash
# ~/.hermes/.env
API_SERVER_ENABLED=true
API_SERVER_PORT=8642
```

If an API key is configured, the same value must be entered in Vokel:

```bash
API_SERVER_KEY=change-me-local-dev
```

The gateway should report:

```text
API server listening on http://127.0.0.1:8642
```

Verify:

```bash
curl http://127.0.0.1:8642/health
```

Expected:

```json
{"status":"ok","platform":"hermes-agent"}
```

## Open Questions

- Android loopback behavior between the app/PWA and Termux.
- Whether the first Android client is a PWA, native shell, or Foreground Service
  plus web dashboard.
- Audio focus and echo-cancellation strategy for phone speaker/mic use.
- Local ASR/TTS fallback when Hermes is unavailable.
- Later local model runtime: llama.cpp, MLC, Google AI Edge, or another measured
  backend.

## Non-Goals For The First Slice

- continuous ambient screen or camera capture
- silent cloud routing
- destructive tool execution
- hidden long-term memory

Android should extend the same product test, not create a second product:

You speak. It answers. You interrupt. It stops. It listens again.
