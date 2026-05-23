# Agent Tools

Voyce supports a small agent tool layer for capabilities that should stay outside
the model itself. The first production tool is `search_web`, backed by SerpApi's
DuckDuckGo engine with a BBC RSS shortcut for BBC headline requests.

## Runtime Contract

The conversation engine keeps inference and tool execution separate:

- `LocalInferenceClient` streams OpenAI-compatible text and tool-call events.
- `ToolRegistry` owns available tool schemas and Python implementations.
- `ConversationEngine` executes tools and records timing marks.
- The web UI listens for tool telemetry and exposes user-visible cues.

Explicit web, news, current, latest, or search requests run `search_web`
deterministically before the model can answer. This is deliberate: smaller local
models often imitate browsing instead of emitting valid tool-call JSON. Direct
tool execution protects the product loop from hallucinated search claims.

## User Experience

Search results are returned as numbered rows with title, snippet, and URL. The
browser transcript renders URLs as clickable links while the TTS path receives a
sanitized speech version, so it does not read raw URLs, Markdown markers, or
formatting symbols aloud.

The web UI also plays a subtle filtered noise cue while an external tool call is
running. The cue starts on `tool_call_forced` or `tool_call_started` telemetry
and fades when the tool or generation finishes.

## Android Port Notes

The tool layer is intentionally independent from LM Studio. Android can keep the
same contract while replacing:

- `LocalInferenceClient` with a llama.cpp, MLC LLM, or native inference adapter.
- `ToolRegistry` implementations with Android services or local databases.
- Web-only audio cues with Android audio focus and notification-safe earcons.

The rule for mobile remains the same: model output can request a tool, but
latency, cancellation, and user-visible cues belong to the runtime.
