# Platform Capabilities

Vokel is the voice, interruption, routing, consent, audit, and display layer.
It does not bundle provider-specific web, image, or GIF APIs.

## Ownership

| Connection | Capability owner | Vokel responsibility |
| --- | --- | --- |
| **LM Studio** | LM Studio and its configured MCP integrations | Stream local model turns, capture explicitly armed camera context, show activity, and render returned artifacts |
| **Hermes** | Hermes and the services configured behind its gateway | Stream Hermes turns, preserve cancellation and consent cues, and render returned artifacts |

The `ToolRegistry` / `ToolDefinition` types remain as an explicit extension
point for narrow runtime capabilities. No provider registry is created by
default, and Vokel does not duplicate capabilities owned by LM Studio or Hermes.

## LM Studio Integration Note

Vokel's `LocalInferenceClient` uses the OpenAI-compatible `/v1/chat/completions`
for ordinary text and explicitly-armed webcam visual turns (image_url blocks).

The `LmStudioNativeMcpClient` (added in mcp_adapter) targets `/api/v1/chat`
(derived from the configured URL) when the native toggle is on. It sends
`integrations` (labels from `~/.lmstudio/mcp.json` or ephemeral objects).
LM Studio executes its own MCP servers and tool calls; Vokel only consumes the
text stream and ToolActivity events and renders real returned artifacts. The
compat path remains available and is the default for vision.

See the build brief for the exact acceptance surface.

## Media Contract

Vokel renders a transcript media card only when the backend returns a usable
URL or artifact. A prose-only response such as "I found an image" remains text;
there is no image payload to display.

Camera questions follow a separate local route. When the user explicitly arms
Camera Questions in Voice Loop and asks a visual question, Vokel captures one
fresh frame, attaches it to the current local-model turn, and discards the
frame. This route does not depend on a web provider.

The camera route supports both LM Studio (compat + native /api/v1/chat image inputs)
and Hermes (via the explicit VisualContext + camera_frame contract on HTTP/ws,
with visible routing/consent/audit/cancellation before the private frame is sent).
Hermes gateway must still be updated to accept/forward the frame (separate change).

## Speech Sanitization

The TTS path receives a sanitized version of each phrase before Kokoro or
spd-say synthesizes it. Raw URLs, Markdown image syntax, formatting markers,
and tool-call syntax are stripped or replaced with short spoken captions.
The browser transcript keeps useful links and media visible.

## Audio Cues

The web UI plays a short cue when a backend reports tool activity and clears
the active state when the tool or generation finishes. Tool activity remains
visible without leaking serialized calls into speech.

## Android Port Notes

The mobile rule remains the same: platform capabilities stay behind the active
connection, while latency, cancellation, consent, and user-visible cues belong
to Vokel. Camera capture must remain an input adapter with an explicit routing
decision, not a desktop-specific assumption inside the conversation engine.
