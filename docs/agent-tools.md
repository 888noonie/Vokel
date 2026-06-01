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

Vokel's current `LocalInferenceClient` uses LM Studio's OpenAI-compatible chat
completions endpoint. That endpoint can carry model tool-call events, but the
client remains responsible for execution.

LM Studio also provides a native `/api/v1/chat` path for platform-managed MCP
integrations. Using that path requires configured MCP servers and an adapter
that selects the intended integration identifiers. Until that adapter is added,
Vokel does not imply that an LM Studio UI integration is automatically active
inside a Vokel session.

## Media Contract

Vokel renders a transcript media card only when the backend returns a usable
URL or artifact. A prose-only response such as "I found an image" remains text;
there is no image payload to display.

Camera questions follow a separate local route. When the user explicitly arms
Camera Questions in Voice Loop and asks a visual question, Vokel captures one
fresh frame, attaches it to the current local-model turn, and discards the
frame. This route does not depend on a web provider.

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
to Vokel.
