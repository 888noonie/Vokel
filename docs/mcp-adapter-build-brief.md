# Platform Adapter Build Brief

## Purpose

Complete the next isolated adapter slice after the local-vision checkpoint.
Vokel remains the voice, interruption, routing, consent, audit, and display
layer. LM Studio and Hermes remain the owners of their platform capabilities.

## Measured Starting Point

- Branch: `mcp_adapter`
- Base checkpoint: `2db0575`
- Local LM Studio vision works with an explicitly armed webcam frame.
- The PlayStation Eye is available as `/dev/video4` on the current Pop!_OS host.
- Hermes HTTP and WebSocket text streaming work.
- Hermes webcam vision is not wired: Hermes receives text only and correctly
  reports that no image was supplied.
- `~/.lmstudio/mcp.json` currently contains `"mcpServers": {}`.
- Bundled SerpApi, Unsplash, and Giphy providers were removed from Vokel.

## Product Rule

Preserve the live loop:

> You speak. It answers. You interrupt. It stops. It listens again. No button.

Keep all new streams cancellable. Do not place camera, MCP, or provider-specific
logic throughout the engine. Add adapters and narrow contracts.

## Build Tasks

1. Add an LM Studio native chat adapter for `/api/v1/chat` with configured MCP
   integration identifiers. LM Studio must execute its own MCP integrations.
2. Keep the existing OpenAI-compatible LM Studio path available for ordinary
   text and visual turns where it remains the appropriate route.
3. Define and implement one Hermes gateway media payload contract for an
   explicitly armed camera frame. Cover HTTP and WebSocket transports where the
   gateway supports them.
4. Add visible route, consent, and audit events before a private frame is sent
   to Hermes or another external endpoint.
5. Render a media card only from a real returned URL or artifact. Never invent a
   card from prose such as "I found an image."
6. Update setup guidance and optional installer hints for LM Studio MCP and
   Hermes gateway configuration. Do not make either dependency mandatory.

## Android Constraint

Treat webcam capture as a media-input adapter. The future Android implementation
may use CameraX and a Foreground Service instead of V4L2, but it should feed the
same routing and consent contract. Keep platform adapters replaceable so an
on-device model, LAN LM Studio endpoint, or Termux Hermes gateway can occupy the
reasoning seat without rewriting the voice loop.

## Acceptance Checks

- LM Studio text turn still streams and remains interruptible.
- LM Studio local webcam question still describes a fresh frame.
- LM Studio MCP route executes one configured platform integration without a
  Vokel-owned provider implementation.
- Hermes text turn still streams over HTTP and `ws://`.
- Hermes webcam question receives one explicitly approved frame and answers
  from that frame.
- External camera routing is visible in the transcript or audit stream.
- Barge-in cancels active generation and pending media work.
- Prose-only media claims do not produce image cards.
- `.venv/bin/python -m pytest -q` passes.
- Frontend production build passes.

## Scope Guard

Do not add ambient capture, silent background camera access, provider API keys,
or a broad plugin framework in this slice. Prefer the smallest adapter changes
that prove both platform routes while preserving the Android path.
