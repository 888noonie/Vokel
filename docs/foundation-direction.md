# Vokel Foundation Direction

This note captures the current product direction before the next build session. It is intentionally practical: keep the foundation clean, transparent, safe, and easy to explain.

## Core Product Truth

Vokel is not an assistant identity.

Vokel is the clean voice and display layer between a person and the intelligence systems they already run.

The user may connect to LM Studio, Hermes, or another backend later. The selected backend owns its own model, personality, memory, tools, and provider configuration. Vokel owns the human-facing loop:

- capture
- speech-to-text
- text-to-speech
- interruption
- pause and resume
- routing
- visible connection state
- consent boundary
- transcript and card display
- local audit notes

The user should never have to wonder what they are speaking to, where their words are going, or who owns the tools.

## Language Decision: Connections, Not Profiles

Use **Connections** as the user-facing concept.

Avoid **Profiles** for v1 because LM Studio and Hermes already have their own model/profile/agent configuration layers. Vokel should not duplicate or reinterpret those systems.

A saved connection is simply a remembered route:

- connection name
- connection type: `lm_studio` or `hermes`
- endpoint or gateway URL
- optional API key
- local voice and speed preference, if useful
- local/external visibility state

User-facing wording:

```text
Connected to: Hermes
Route: External agent
Voice: Local Kokoro
Tools: Hermes-owned
Interrupt: Available
```

or:

```text
Connected to: LM Studio
Route: Local model
Voice: Local Kokoro
Tools: Vokel-owned
Interrupt: Available
```

## First-Class v1 Connections

Only two first-class connection types should be treated as product commitments for now:

1. **LM Studio**
   - local OpenAI-compatible endpoint
   - model selection remains inside LM Studio
   - Vokel-owned local tools may be available
   - best for private local voice use

2. **Hermes**
   - Hermes gateway / API server
   - Hermes owns reasoning, tools, memory, provider config, and session state
   - Vokel owns voice, display, interruption, routing, and consent
   - best for speaking to a richer external agent stack

Other backend types can remain future adapters.

## Trust Display Principle

The product moat is not only speech. The product moat is legibility.

Vokel should make invisible AI plumbing visible without making it noisy.

Default view should be calm:

```text
Connected to Hermes
Listening
Local voice active
Interrupt available
```

Expanded details should show the full truth:

```text
Gateway: 127.0.0.1:8642
ASR: local
TTS: local Kokoro
Tools: Hermes-owned
Memory: Hermes-owned
Route: external agent
Context shared: current user turn only
Consent: not armed
```

Everything powerful should be visible. Everything visible should be collapsible. Everything external should be explicit.

## Media Cards

Media Cards are Vokel's repeatable visual grammar for useful things entering or leaving a conversation.

A card can represent:

- web result
- image
- GIF
- screenshot
- OCR result
- file
- voice note
- device status
- Hermes result
- local tool result
- saved context
- consent request

The user learns one interaction pattern:

```text
card appears -> glance -> expand -> save/tag/use/hide/delete
```

Cards first serve trust. Sharing comes later.

### Card Responsibilities

A Media Card should answer:

- What is this?
- Where did it come from?
- Was it local or external?
- Which connection received or produced it?
- Can it be used as context?
- Can it be saved, tagged, hidden, or deleted?
- What is safe for TTS to say about it?

### Minimal Internal Shape

```ts
type MediaCard = {
  id: string;
  kind: "web" | "image" | "gif" | "file" | "screenshot" | "ocr" | "voice" | "device" | "tool" | "context" | "consent";
  title: string;
  summary?: string;
  source?: string;
  route: "local" | "lm_studio" | "hermes" | "external";
  privacy: "local_only" | "external_allowed" | "external_sent";
  tags: string[];
  createdAt: string;
  saved: boolean;
  collapsed: boolean;
  speechText?: string;
  displayData: unknown;
  actions: string[];
};
```

The most important fields are `kind`, `route`, `privacy`, `tags`, `speechText`, and `actions`.

## Live Context Control

The user should be able to pause the stream and inject context before continuing.

Useful voice commands:

- "pause"
- "hold on"
- "wait"
- "let me add something"
- "use this as context"
- "continue"
- "forget that last bit"
- "switch to Hermes"
- "switch back to local"

This turns Vokel from a simple voice wrapper into a human-controlled context layer.

Do not build the whole context system at once. Start with typed notes or selected transcript text, then later add screenshots, files, OCR, clipboard, and device state.

## Phone Direction

The near-term phone goal is simple:

```text
Android/PWA Vokel -> Hermes running in Termux -> selected Hermes model/provider/tools
```

Do not begin by trying to run every model locally on Android. First prove the voice/control surface on the phone:

- connect to Hermes on-device or over LAN
- show active connection and privacy state
- speak one turn
- interrupt reliably
- pause/resume
- keep Hermes tools behind Hermes

The long-term direction is that any capable device can become a Vokel surface, and Vokel can speak to available local or agent backends across the user's devices.

## Deferred Ideas

These are valuable but should not distract from the current foundation:

- shareable Media Cards
- card import/export format
- cross-device card sending
- OCR and embedding model integration
- local vector memory v2
- file write-back
- camera/screen context
- plugin or skill marketplace
- multi-agent routing beyond Hermes

The rule: ground the idea, do not kill it; defer it until the foundation can safely carry it.

## Next Build Slice

The next practical build slice should be:

1. Rename/shape user-facing backend selection around **Connections**.
2. Support only LM Studio and Hermes as first-class connection types.
3. Show active connection, route, local/external state, voice state, and tool ownership clearly.
4. Stabilize Media Cards as a single UI/data primitive for web/image/GIF/tool outputs.
5. Add Expand, Save, Hide, Tag, and Use as Context actions where safe.
6. Add pause/resume/hold voice-command handling before larger context input features.
7. Keep Android/Termux Hermes as the next major product proof after the desktop/PWA connection layer is clean.

## Product Sentence

Vokel is a clean, local-first voice interface for the AI systems you already run. Connect LM Studio or Hermes, speak naturally, interrupt instantly, and always see what you are connected to.
