# Agent Extensions

Vokel can act as a realtime voice front-end for an external agent backend. Phase A adds
**Hermes gateway** support alongside the built-in LM Studio path.

## Modes

| Mode | Who reasons | Who owns history | Vokel tools |
| --- | --- | --- | --- |
| **Built-in** | LM Studio via `LocalInferenceClient` | Vokel `ConversationEngine` | Optional web/image/GIF |
| **Hermes** | Hermes API server (`hermes gateway`) | Hermes `conversation` id | Disabled |

## Hermes setup

1. Enable the API server in `~/.hermes/.env`:

```bash
API_SERVER_ENABLED=true
API_SERVER_KEY=change-me-local-dev
```

2. Start the gateway (leave this running in its own terminal):

```bash
hermes gateway run
```

You should see: `[API Server] API server listening on http://127.0.0.1:8642`

3. In the Vokel dashboard, choose **Agent Extension → HERMES**, set the gateway URL
   (default `http://127.0.0.1:8642`), and enter the same API key if you set one.

Verify from another terminal:

```bash
curl http://127.0.0.1:8642/health
```

Expected: `{"status":"ok"}`

Hermes uses the LM Studio model configured in `~/.hermes/config.yaml`. Vokel does
not call LM Studio directly in Hermes mode.

## Runtime contract

- `AgentBackend` — `stream_chat`, `cancel_active`, async context manager.
- `HermesAgentClient` — streams `/v1/responses` with `conversation` session chaining;
  falls back to `/v1/chat/completions` if Responses is unavailable.
- Barge-in closes the active HTTP stream so Hermes can interrupt the running turn.

## Reset

**Reset** in the UI starts a new Hermes conversation id. Transcript lines in Vokel are
local display only; Hermes retains authoritative session state on the gateway.
