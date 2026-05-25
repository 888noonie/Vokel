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
| **Built-in** | Local OpenAI-compatible endpoint via `LocalInferenceClient` | Vokel `ConversationEngine` | Optional web/image/GIF |
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

## Boundaries

- Vokel-owned tools are for Built-in mode.
- Hermes-owned tools stay behind the Hermes gateway.
- Vokel displays Hermes session and gateway state, but does not duplicate Hermes
  browser, email, repository, or messaging tools.
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
