from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from types import TracebackType
from typing import Any, TYPE_CHECKING

from .config import LmStudioConfig
from .events import Event, TextDeltaEvent, ToolCallEvent, ToolActivityEvent
from .agent_backend import AgentBackendCapabilities

if TYPE_CHECKING:
    import httpx

    from .vision import VisualContext

ChatMessage = dict[str, Any]


def extract_camera_frame(messages: Sequence[ChatMessage]) -> VisualContext | None:
    """Shared helper: detect explicitly-armed frame (VisualContext) in multimodal user content.

    Returns VisualContext (narrow, CameraX-compatible) or None. Only data: URLs accepted.
    Upper layers ensure visible consent/audit + cancellation before the block reaches any
    stream_chat (Hermes or native).
    Metadata (source/device, captured_at) is threaded via the image block's visual_context
    sub-dict when available (populated at capture time in web layer); falls back to
    minimal source derived from context if missing (never hard-coded device).
    """
    from .vision import VisualContext  # runtime for constructing return value

    for message in reversed(messages or []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        blocks = content if isinstance(content, list) else ([content] if isinstance(content, dict) else [])
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image_url":
                img = block.get("image_url") or {}
                url = str(img.get("url") or "").strip()
                if url.startswith("data:image"):
                    meta = img.get("visual_context") or img.get("capture") or {}
                    source = str(meta.get("source") or meta.get("device") or "webcam")
                    return VisualContext(
                        data_url=url,
                        source=source,
                        captured_at=meta.get("captured_at"),
                        consent=str(meta.get("consent") or "explicit_visual_context_for_turn"),
                        contract=str(meta.get("contract") or "hermes_camera_frame_v1"),
                    )
            if block.get("type") == "visual_context":
                # direct block (future proof)
                url = str(block.get("data_url") or "").strip()
                if url.startswith("data:image"):
                    return VisualContext(
                        data_url=url,
                        source=str(block.get("source") or "webcam"),
                        captured_at=block.get("captured_at"),
                        consent=str(block.get("consent") or "explicit_visual_context_for_turn"),
                        contract=str(block.get("contract") or "hermes_camera_frame_v1"),
                    )
    return None


class InferenceError(RuntimeError):
    pass


def parse_sse_delta(line: str) -> dict[str, Any] | None:
    """Extract a content token or tool call from one OpenAI-compatible SSE line."""

    if not line.startswith("data:"):
        return None

    payload = line.removeprefix("data:").strip()
    if not payload or payload == "[DONE]":
        return None

    data = json.loads(payload)
    choices = data.get("choices") or []
    if not choices:
        return None

    return choices[0].get("delta")


_JAN_401_HINT = (
    "Jan rejected the API key (HTTP 401). Restart Jan with ~/.local/bin/jan-voice "
    "or set VOKEL_LLM_API_KEY to the key from `pgrep -af 'llama-server.*6767'`."
)


def derive_models_url(completions_url: str) -> str:
    """Derive the OpenAI-compatible /v1/models endpoint from a chat URL."""
    base = completions_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")] + "/models"
    if "/v1/" in base:
        return base.split("/v1/")[0] + "/v1/models"
    return base + "/v1/models"


def format_local_error(config: "LmStudioConfig", exc: Exception) -> str:
    """Translate a transport/HTTP failure into an actionable local-LLM message."""
    import httpx

    base = config.url
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return (
            f"Cannot reach the local LLM at {base}. Start Jan with ~/.local/bin/jan-voice "
            "(or your LM Studio server), confirm it is listening, then retry."
        )
    if isinstance(exc, (httpx.ReadTimeout, httpx.PoolTimeout, httpx.WriteTimeout)):
        return (
            f"The local LLM at {base} accepted the connection but did not respond in time. "
            f"Check that the model '{config.model}' is loaded and not stuck."
        )
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 401:
            return _JAN_401_HINT
        return f"The local LLM at {base} returned HTTP {status}."
    return f"Local LLM request failed: {exc}"


async def check_local_health(config: "LmStudioConfig", client: "httpx.AsyncClient") -> None:
    """Fail fast when the local LLM socket is unreachable or rejects the key."""
    headers: dict[str, str] = {}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    try:
        response = await client.get(derive_models_url(config.url), headers=headers)
        response.raise_for_status()
    except Exception as exc:
        raise InferenceError(format_local_error(config, exc)) from exc


def _local_timeout(config: "LmStudioConfig") -> Any:
    import httpx

    return httpx.Timeout(config.timeout_seconds, connect=config.connect_timeout_seconds)


class LocalInferenceClient:
    capabilities = AgentBackendCapabilities(
        owns_tools=False,
        emits_tool_activity=True,
    )

    def __init__(self, config: LmStudioConfig, client: "httpx.AsyncClient | None" = None):
        self.config = config
        self._client = client
        self._owns_client = client is None
        self._active_response: Any = None

    def _request_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def __aenter__(self) -> "LocalInferenceClient":
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                timeout=_local_timeout(self.config),
                headers=self._request_headers(),
            )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.cancel_active()
        if self._owns_client and self._client is not None:
            await self._client.aclose()
        self._client = None

    async def cancel_active(self) -> None:
        response = self._active_response
        self._active_response = None
        if response is not None:
            await response.aclose()

    async def stream_chat(
        self, messages: Sequence[ChatMessage], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[Event]:
        if self._client is None:
            raise InferenceError("LocalInferenceClient must be used as an async context manager")

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        active_tool_calls: dict[int, dict[str, Any]] = {}

        try:
            async with self._client.stream("POST", self.config.url, json=payload) as response:
                self._active_response = response
                if response.status_code == 401:
                    raise InferenceError(_JAN_401_HINT)
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip() == "data: [DONE]":
                        break
                    delta = parse_sse_delta(line)
                    if not delta:
                        continue

                    if "content" in delta and delta["content"]:
                        yield TextDeltaEvent(content=delta["content"])

                    if "tool_calls" in delta:
                        for tc in delta["tool_calls"]:
                            index = tc.get("index")
                            if index is None:
                                continue

                            if index not in active_tool_calls:
                                active_tool_calls[index] = {
                                    "id": tc.get("id", ""),
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }

                            if "id" in tc and tc["id"]:
                                active_tool_calls[index]["id"] = tc["id"]

                            fn = tc.get("function", {})
                            if "name" in fn and fn["name"]:
                                active_tool_calls[index]["function"]["name"] += fn["name"]
                            if "arguments" in fn and fn["arguments"]:
                                active_tool_calls[index]["function"]["arguments"] += fn["arguments"]
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(format_local_error(self.config, exc)) from exc
        finally:
            self._active_response = None

        # After streaming is done, yield completed tool calls
        for tc in active_tool_calls.values():
            name = tc["function"]["name"]
            arguments_str = tc["function"]["arguments"]
            try:
                arguments = json.loads(arguments_str) if arguments_str else {}
            except json.JSONDecodeError:
                arguments = {}
            yield ToolCallEvent(call_id=tc["id"], name=name, arguments=arguments)


# --- LM Studio native /api/v1/chat MCP adapter (smallest slice) ---


def _derive_native_chat_url(completions_url: str) -> str:
    """Derive the native chat endpoint from the common OpenAI-compat URL."""
    base = completions_url.rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base[: -len("/v1/chat/completions")] + "/api/v1/chat"
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")] + "/api/v1/chat"
    # Fallback: assume host:port and append
    if "/v1/" in base:
        return base.split("/v1/")[0] + "/api/v1/chat"
    return base.rstrip("/") + "/api/v1/chat"


async def _parse_native_chat_stream(
    response: Any,
    *,
    final_result_holder: list[dict[str, Any]] | None = None,
) -> AsyncIterator[Event]:
    """Parse LM Studio /api/v1/chat SSE (named events) into Vokel Events.

    Emits TextDeltaEvent for message content and ToolActivityEvent for MCP/tool starts.
    Tool execution and media artifacts (if any) remain inside LM Studio / its MCP servers.
    If final_result_holder (list of len 1) is provided, the chat.end result (incl. response_id)
    is placed in it for continuity (previous_response_id).
    """
    import json as _json

    current_event: str | None = None
    async for raw_line in response.aiter_lines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("event:"):
            current_event = line[6:].strip()
            continue
        if line.startswith("data:"):
            payload_str = line[5:].strip()
            if not payload_str:
                continue
            try:
                data = _json.loads(payload_str)
            except _json.JSONDecodeError:
                continue

            ev = current_event or data.get("type", "")
            if ev == "message.delta":
                content = data.get("content")
                if isinstance(content, str) and content:
                    yield TextDeltaEvent(content=content)
            elif ev == "tool_call.start":
                tool = data.get("tool") or data.get("name") or "mcp_tool"
                if isinstance(tool, str) and tool:
                    yield ToolActivityEvent(name=tool, status="started")
            elif ev == "tool_call.success":
                tool = data.get("tool") or data.get("name") or "mcp_tool"
                if isinstance(tool, str) and tool:
                    yield ToolActivityEvent(name=tool, status="finished")
            elif ev == "tool_call.failure":
                tool = data.get("tool") or (data.get("metadata") or {}).get("tool_name") or "mcp_tool"
                reason = (data.get("error") or data).get("message", "tool failed")
                if isinstance(tool, str) and tool:
                    yield ToolActivityEvent(name=tool, status="failed")
                # Surface hard failure for the engine/trace
                raise InferenceError(f"Native tool failure: {reason}")
            elif ev == "chat.end":
                if final_result_holder is not None:
                    result = data.get("result") or data
                    if isinstance(result, dict):
                        final_result_holder.append(result)
                    else:
                        final_result_holder.append({"raw": result})
                break
            elif ev == "error":
                detail = data.get("error") or data
                raise InferenceError(f"LM Studio native /api/v1/chat error: {detail}")
            # Ignore reasoning.*, prompt_*, model_load_* ; full tool lifecycle (start/success/failure) emitted with status


class LmStudioNativeMcpClient:
    """Adapter for LM Studio's native /api/v1/chat with configured MCP integrations.

    LM Studio (and the MCP servers listed in ~/.lmstudio/mcp.json or per-request) own
    all tool execution, memory, and provider calls. Vokel only:
      - streams the text deltas for TTS/transcript
      - surfaces tool activity for UI cues + audit
      - renders real artifacts returned in final text (markdown links etc.)
      - provides explicit camera frames when armed (native input also supports images)

    This keeps the existing OpenAI-compat path (LocalInferenceClient) for ordinary
    text/visual turns where MCP is not required.
    """

    capabilities = AgentBackendCapabilities(
        owns_tools=True,  # LM Studio + its MCPs execute
        emits_tool_activity=True,
        emits_media_artifacts=True,
    )

    def __init__(
        self,
        config: LmStudioConfig,
        client: "httpx.AsyncClient | None" = None,
        integrations: list[str] | None = None,
    ):
        self.config = config
        self._client = client
        self._owns_client = client is None
        self._active_response: Any = None
        self.integrations = list(integrations or config.mcp_integrations or [])
        self._last_response_id: str | None = None

    def reset_conversation(self) -> None:
        """Clear previous_response_id chain for native LM continuity (engine calls on Vokel Reset)."""
        self._last_response_id = None

    async def __aenter__(self) -> "LmStudioNativeMcpClient":
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=_local_timeout(self.config))
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.cancel_active()
        if self._owns_client and self._client is not None:
            await self._client.aclose()
        self._client = None

    async def cancel_active(self) -> None:
        response = self._active_response
        self._active_response = None
        if response is not None:
            await response.aclose()

    async def stream_chat(
        self, messages: Sequence[ChatMessage], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[Event]:
        del tools  # Native path: LM Studio + configured MCPs own execution
        if self._client is None:
            raise InferenceError("LmStudioNativeMcpClient must be used as an async context manager")

        # Support armed visual (image) via native input array form (data_url + VisualContext).
        # Also use previous_response_id for conversation continuity when available.
        user_input: str | list[dict[str, Any]] = ""
        vc = extract_camera_frame(messages)
        text_part = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    text_part = c
                elif isinstance(c, list):
                    for b in c:
                        if isinstance(b, dict) and b.get("type") in ("text", "input_text"):
                            text_part = str(b.get("text", ""))
                            break
                if text_part or vc:
                    break

        if vc:
            user_input = [{"type": "image", "data_url": vc.data_url}]
            if text_part:
                # Use "text" + "content" for compatibility with installed LM Studio server
                # (the /api/v1/chat array form for mixed image+text on the production endpoint).
                # Preserves full image + text native support.
                user_input.append({"type": "text", "content": text_part})
        else:
            user_input = text_part or " "

        system_prompt = ""
        for m in messages:
            if m.get("role") == "system":
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    system_prompt = c.strip()
                    break

        api_url = _derive_native_chat_url(self.config.url)

        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": user_input,
            "stream": True,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt
        if self.integrations:
            payload["integrations"] = self.integrations
        if self._last_response_id:
            payload["previous_response_id"] = self._last_response_id

        final_holder: list[dict[str, Any]] = []
        try:
            async with self._client.stream("POST", api_url, json=payload) as response:
                self._active_response = response
                response.raise_for_status()
                async for event in _parse_native_chat_stream(response, final_result_holder=final_holder):
                    yield event
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(format_local_error(self.config, exc)) from exc
        finally:
            self._active_response = None

        # Preserve continuity id from this turn for next turn (or visual->text handoff)
        if final_holder:
            rid = final_holder[0].get("response_id") or final_holder[0].get("id")
            if isinstance(rid, str) and rid.startswith("resp_"):
                self._last_response_id = rid
