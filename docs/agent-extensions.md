# Agent Extensions

Vokel can act as a realtime voice front-end for an external agent backend.
Hermes gateway support is the first implementation alongside the built-in local
model path.

In external-agent mode, Vokel is not the reasoning agent. It owns voice capture,
playback, interruption, consent, and audit visibility. The external agent owns
its own memory, tools, and provider configuration.

## Modes

| Mode | Who reasons | Who owns history | Vokel tools |
| --- | --- | --- | --- |
| **Built-in** | Local OpenAI-compatible endpoint via `LocalInferenceClient` | Vokel `ConversationEngine` | Disabled in Vokel; LM Studio owns platform capabilities |
| **Hermes** | Hermes API Server (`hermes gateway`) | Hermes `conversation` id | Disabled in Vokel; Hermes owns its tools |

## Hermes setup

1. Enable the API server in `~/.hermes/.env`:

```bash
API_SERVER_ENABLED=true
API_SERVER_PORT=8642
# Optional for local development. If set, enter the same value in Vokel.
API_SERVER_KEY=change-me-local-dev
```

2. Start the gateway (leave this running in its own terminal):

```bash
hermes gateway run
```

You should see: `[API Server] API server listening on http://127.0.0.1:8642`

3. In the Vokel dashboard, choose **Agent Extension -> HERMES**, set the gateway URL
   (default `http://127.0.0.1:8642`), and enter the same API key if you set one.

Verify from another terminal:

```bash
curl http://127.0.0.1:8642/health
```

Expected: `{"status":"ok"}`

Hermes uses the provider and model configured in `~/.hermes/config.yaml`. That
may be XAI/Grok, OpenRouter, LM Studio, or another provider supported by Hermes.
Vokel does not call the model provider directly in Hermes mode.

## Hermes Vision (mcp_adapter slice)

Explicit camera frame support is wired for Hermes (HTTP + ws://) using the
`VisualContext` + `camera_frame` contract (see agent_backend.py and extract
in inference.py + injection in the two clients).

- Armed only via the Camera Questions toggle (visible consent + audit events
  including "external_media_route_initiated").
- Barge-in during capture or send prevents the frame from reaching the gateway
  (proven by engine test).
- Metadata (source, captured_at, consent, contract) is carried; no hard-coded
  device.
- Gateway-side changes still required on Hermes (accept the payload on both
  transports, forward frame to model, return artifacts as real markdown/URLs
  only for Vokel cards). Vokel side complete.

Android path remains open (CameraX will emit equivalent VisualContext).

## LM Studio Native MCP Adapter

See docs/mcp-adapter-build-brief.md and the new `LmStudioNativeMcpClient` in
inference.py. When "Use native /api/v1/chat" is selected in the dashboard (and
MCP servers configured in `~/.lmstudio/mcp.json` + allowed in LM Studio Server
Settings), Vokel calls `/api/v1/chat` (deriving from the LM url), passes
`integrations` (e.g. `["mcp/playwright"]`), and streams via LM Studio's named
SSE events. LM Studio + its MCPs execute everything; Vokel only renders returned
text and real artifacts.

## Boundaries

- LM Studio-owned tools stay behind the LM Studio platform boundary.
- Hermes-owned tools stay behind the Hermes gateway.
- Vokel displays connection state and returned media artifacts, but does not
  duplicate LM Studio or Hermes browser, email, repository, or messaging tools.
- Future execution flows should use Vokel's consent boundary before high-risk
  actions are allowed.

## Runtime contract

- `AgentBackend` — `stream_chat`, `cancel_active`, async context manager.
- `HermesAgentClient` — streams `/v1/responses` with `conversation` session chaining;
  falls back to `/v1/chat/completions` if Responses is unavailable.
- Barge-in closes the active HTTP stream so Hermes can interrupt the running turn.
- Agent Console events show gateway health, selected backend, session id, and
  execution-consent state.

## Reset

**Reset** in the UI starts a new Hermes conversation id. Transcript lines in Vokel are
local display only; Hermes retains authoritative session state on the gateway.
