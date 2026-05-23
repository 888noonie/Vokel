# Agent Tools

Vokel supports a small agent tool layer for capabilities that should stay outside
the model itself. Three production tools are registered by default:

- `search_web` — SerpApi DuckDuckGo with BBC RSS shortcut and page scraping
- `search_image` — Unsplash photo search with inline transcript rendering
- `search_gif` — Giphy GIF/reaction/meme search with context-aware follow-ups

## Runtime Contract

The conversation engine keeps inference and tool execution separate:

- `LocalInferenceClient` streams OpenAI-compatible text and tool-call events.
- `ToolRegistry` owns available tool schemas and Python implementations.
- `ConversationEngine` executes tools, records timing marks, and manages the
  hybrid synthesis loop.
- The web UI listens for tool telemetry and exposes user-visible cues.

Explicit web, news, current, latest, or search requests run `search_web`
deterministically before the model can answer. This is deliberate: smaller local
models (tested with Gemma 4 E4B Q4 K M) often imitate browsing instead of
emitting valid tool-call JSON. Direct tool execution protects the product loop
from hallucinated search claims.

## Hybrid Search-Then-Synthesize

The conversation engine uses a three-stage approach:

1. **Search first.** The tool runs the DuckDuckGo API call. If the returned
   snippets are too thin (generic page descriptions like "Reuters.com is your
   online source..."), Vokel scrapes the actual page content from the top result
   URLs to get real article text, weather data, or headlines.

2. **Synthesize with evidence.** The search evidence is injected into a strict
   system prompt that tells the model to answer from only that evidence. The
   model cannot claim it cannot browse; the search already happened.

3. **Fallback to raw evidence.** If the model still hedges ("I couldn't get",
   "not available right now", etc.), Vokel discards the model's response and
   speaks the raw numbered evidence with clickable links instead.

This gives the local model the best chance to produce a natural spoken answer
while guaranteeing the user always gets real data.

## Image Search (Unsplash)

When the user says "show me an image of", "picture of", or "photo of", the
engine forces `search_image` deterministically. A landscape photo is fetched
and rendered inline in the transcript as a rounded card with alt text and
Unsplash attribution. The LLM receives a strict prompt to give a brief, warm
spoken intro (1-2 sentences) without reading URLs or photographer metadata.

## GIF Search (Giphy)

When the user says "gif", "meme", "sticker", "reaction", "something funny",
or "make me laugh", the engine forces `search_gif`. The detection is
context-aware: if the last few turns mention GIFs, short follow-ups like
"anything" or "cats" automatically trigger a new search without requiring
explicit trigger words again.

GIFs are rendered in a compact card (max 280px) with a purple border and a
small "GIF" badge. The LLM receives a playful prompt to react like a friend
sharing a GIF — one expressive sentence, no metadata.

Query extraction strips filler words and caps at 60 characters to avoid
414 URI Too Long errors from the Giphy API. ASR-tolerant triggers accept
"show me a g" when speech recognition truncates "gif".

## Speech Sanitization

The TTS path receives a sanitized version of each phrase before Kokoro or
spd-say synthesizes it. The sanitizer strips:

- Markdown bold/italic markers, headers, and bullet points
- Raw URLs (replaced with "Link available in transcript")
- Image markdown `![alt](url)` (replaced with alt text + "Image shown in transcript")
- GIF markdown `![gif:title](url)` (replaced with "GIF shown in transcript")
- Code blocks and backtick formatting
- Angle brackets and pipe characters
- Repeated punctuation

The browser transcript keeps the original text with clickable links intact.

## Audio Cues

The web UI plays a quiet two-tone sine cue (not white noise) while an external
tool call is running. The cue starts on `tool_call_forced` or `tool_call_started`
telemetry and fades smoothly when the tool or generation finishes.

## Voice Preview

Users can audition any of the 28 bundled Kokoro voices before starting a
session. The preview sends a `preview_voice` websocket command that synthesizes
a short sample without creating transcript entries or touching memory state.

## Android Port Notes

The tool layer is intentionally independent from LM Studio. Android can keep the
same contract while replacing:

- `LocalInferenceClient` with a llama.cpp, MLC LLM, or native inference adapter.
- `ToolRegistry` implementations with Android services or local databases.
- Web-only audio cues with Android audio focus and notification-safe earcons.

The rule for mobile remains the same: model output can request a tool, but
latency, cancellation, and user-visible cues belong to the runtime.
