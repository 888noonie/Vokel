from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable

from .agent_backend import (
    AgentBackendCapabilities,
    TOOL_ACTIVITY_REPORTING_CONTRACT,
)
from .events import Event, TextDeltaEvent, ToolActivityEvent
from .inference import ChatMessage, InferenceError, extract_camera_frame

if TYPE_CHECKING:
    import httpx


@dataclass
class HermesConfig:
    """Hermes API server (gateway) connection settings."""

    base_url: str = "http://127.0.0.1:8642"
    model: str = "hermes-agent"
    api_key: str = ""
    session_id: str = ""
    timeout_seconds: float = 120.0

    def responses_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/responses"

    def chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/chat/completions"


def _extract_user_input(messages: Sequence[ChatMessage]) -> str:
    parts: list[str] = []
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in ("text", "input_text"):
                    text = block.get("text", "")
                    if text:
                        parts.append(str(text))
        if parts:
            break
    return "\n".join(reversed(parts)).strip()


def _extract_system_instructions(messages: Sequence[ChatMessage]) -> str:
    parts: list[str] = []
    for message in messages:
        if message.get("role") != "system":
            continue
        content = message.get("content", "")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
    return "\n\n".join(parts)




def _parse_responses_sse_payload(payload: dict[str, Any]) -> str:
    event_type = payload.get("type", "")
    if event_type == "response.output_text.delta":
        delta = payload.get("delta", "")
        if isinstance(delta, str):
            return delta
        if isinstance(delta, dict):
            return str(delta.get("text", "") or delta.get("content", ""))
    if event_type == "response.text.delta":
        return str(payload.get("delta", ""))
    return ""


def _parse_chat_sse_payload(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    delta = first.get("delta") or {}
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def _extract_tool_activity_name(payload: dict[str, Any]) -> str | None:
    event_type = str(payload.get("type") or payload.get("event") or "").lower()
    item = payload.get("item") or payload.get("delta")
    item_type = str(item.get("type") or "").lower() if isinstance(item, dict) else ""
    if (
        "tool" not in event_type
        and "function_call" not in event_type
        and "tool" not in item_type
        and "function_call" not in item_type
    ):
        return None

    direct = payload.get("tool_name") or payload.get("name")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    function = payload.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()

    if isinstance(item, dict):
        name = item.get("name") or item.get("tool_name")
        if isinstance(name, str) and name.strip() and (
            "tool" in item_type or "function_call" in item_type or "tool" in event_type
        ):
            return name.strip()
        function = item.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()

    return None


def _parse_responses_sse_payload_probe(payload: dict[str, Any]) -> str:
    token = _parse_responses_sse_payload(payload)
    if token:
        return token
    if payload.get("type") == "response.output_text.done":
        text = payload.get("text", "")
        return text if isinstance(text, str) else ""
    return ""


def _parse_chat_sse_payload_probe(payload: dict[str, Any]) -> str:
    token = _parse_chat_sse_payload(payload)
    if token:
        return token
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = (choices[0] or {}).get("message") or {}
    final_content = message.get("content")
    return final_content if isinstance(final_content, str) else ""


def format_gateway_error(config: HermesConfig, exc: Exception) -> str:
    import httpx

    base = config.base_url.rstrip("/")
    if isinstance(exc, httpx.ConnectError):
        return (
            f"Cannot reach Hermes gateway at {base}. "
            "Add API_SERVER_ENABLED=true to ~/.hermes/.env, then run "
            "`hermes gateway run` in a separate terminal and retry."
        )
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 401:
            return (
                f"Hermes gateway at {base} rejected the API key (HTTP 401). "
                "Set the same value in Vokel and API_SERVER_KEY in ~/.hermes/.env."
            )
        return f"Hermes gateway at {base} returned HTTP {status}."
    return f"Hermes gateway request failed: {exc}"


async def check_gateway_health(config: HermesConfig, client: "httpx.AsyncClient") -> None:
    """Fail fast when the Hermes API server is unreachable or misconfigured."""

    headers: dict[str, str] = {}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    health_url = f"{config.base_url.rstrip('/')}/health"
    try:
        response = await client.get(health_url, headers=headers)
        response.raise_for_status()
    except Exception as exc:
        raise InferenceError(format_gateway_error(config, exc)) from exc


async def _stream_has_text(
    client: "httpx.AsyncClient",
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    parser: Callable[[dict[str, Any]], str],
) -> bool:
    async with client.stream("POST", url, json=payload, headers=headers) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line.removeprefix("data:").strip()
            if not raw or raw == "[DONE]":
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if parser(data):
                return True
    return False


async def check_gateway_inference(
    config: HermesConfig,
    client: "httpx.AsyncClient",
    *,
    timeout_seconds: float = 6.0,
) -> None:
    """Validate that the configured model can stream text tokens."""
    import httpx

    probe_session = f"vokel-smoke-{uuid.uuid4().hex}"
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "X-Hermes-Session-Id": probe_session,
    }
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    probe_prompt = "Reply with exactly: READY"
    responses_payload: dict[str, Any] = {
        "model": config.model,
        "input": probe_prompt,
        "conversation": probe_session,
        "stream": True,
    }

    base = config.base_url.rstrip("/")

    try:
        has_text = await asyncio.wait_for(
            _stream_has_text(
                client,
                url=config.responses_url(),
                payload=responses_payload,
                headers=headers,
                parser=_parse_responses_sse_payload_probe,
            ),
            timeout=timeout_seconds,
        )
        if has_text:
            return
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code not in (404, 405, 501):
            raise InferenceError(format_gateway_error(config, exc)) from exc
    except asyncio.TimeoutError as exc:
        raise InferenceError(
            f"Hermes gateway at {base} is reachable, but model '{config.model}' did not "
            f"stream text within {timeout_seconds:.0f}s. Check Hermes provider/model wiring."
        ) from exc
    except Exception as exc:
        raise InferenceError(format_gateway_error(config, exc)) from exc

    chat_payload: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": probe_prompt}],
        "stream": True,
    }
    try:
        has_text = await asyncio.wait_for(
            _stream_has_text(
                client,
                url=config.chat_completions_url(),
                payload=chat_payload,
                headers=headers,
                parser=_parse_chat_sse_payload_probe,
            ),
            timeout=timeout_seconds,
        )
        if has_text:
            return
    except asyncio.TimeoutError as exc:
        raise InferenceError(
            f"Hermes gateway at {base} is reachable, but model '{config.model}' did not "
            f"stream text within {timeout_seconds:.0f}s. Check Hermes provider/model wiring."
        ) from exc
    except Exception as exc:
        raise InferenceError(format_gateway_error(config, exc)) from exc

    raise InferenceError(
        f"Hermes gateway at {base} is reachable, but model '{config.model}' returned no "
        "streamed text. Check Hermes provider/model wiring."
    )


class HermesAgentClient:
    """Voice I/O adapter: Hermes owns reasoning, memory, and tools."""

    capabilities = AgentBackendCapabilities(
        owns_tools=True,
        supports_session_reset=True,
        emits_tool_activity=True,
    )

    def __init__(self, config: HermesConfig, client: httpx.AsyncClient | None = None):
        self.config = config
        self._client = client
        self._owns_client = client is None
        self._session_id = config.session_id or f"vokel-{uuid.uuid4().hex}"
        self._active_response: httpx.Response | None = None

    @property
    def session_id(self) -> str:
        return self._session_id

    def reset_session(self) -> str:
        self._session_id = f"vokel-{uuid.uuid4().hex}"
        return self._session_id

    async def __aenter__(self) -> HermesAgentClient:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self.config.timeout_seconds)
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

    def _auth_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "X-Hermes-Session-Id": self._session_id,
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[Event]:
        del tools  # Hermes owns tool execution in agent mode.
        if self._client is None:
            raise InferenceError("HermesAgentClient must be used as an async context manager")

        user_input = _extract_user_input(messages)
        if not user_input:
            raise InferenceError("Hermes agent mode requires a non-empty user message")

        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": user_input,
            "instructions": _extract_system_instructions(messages) or TOOL_ACTIVITY_REPORTING_CONTRACT,
            "conversation": self._session_id,
            "stream": True,
        }

        camera_frame = extract_camera_frame(messages)
        if camera_frame:
            # Convert narrow VisualContext to the wire payload shape for the contract.
            # (Gateway side still needs to accept + forward this.)
            payload["camera_frame"] = {
                "data_url": camera_frame.data_url,
                "source": camera_frame.source,
                "captured_at": camera_frame.captured_at,
                "consent": camera_frame.consent,
                "contract": camera_frame.contract,
            }

        try:
            async for event in self._stream_responses(payload):
                yield event
        except InferenceError:
            raise
        except Exception as exc:
            import httpx

            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (
                404,
                405,
                501,
            ):
                async for event in self._stream_chat_completions(messages, user_input):
                    yield event
                return
            raise InferenceError(format_gateway_error(self.config, exc)) from exc

    async def _stream_responses(self, payload: dict[str, Any]) -> AsyncIterator[Event]:
        assert self._client is not None
        try:
            async with self._client.stream(
                "POST",
                self.config.responses_url(),
                json=payload,
                headers=self._auth_headers(),
            ) as response:
                self._active_response = response
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line.removeprefix("data:").strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    tool_name = _extract_tool_activity_name(data)
                    if tool_name:
                        yield ToolActivityEvent(name=tool_name)
                    token = _parse_responses_sse_payload(data)
                    if token:
                        yield TextDeltaEvent(content=token)
        finally:
            self._active_response = None

    async def _stream_chat_completions(
        self,
        messages: Sequence[ChatMessage],
        user_input: str,
    ) -> AsyncIterator[Event]:
        assert self._client is not None
        chat_messages = [
            {"role": "system", "content": _extract_system_instructions(messages) or TOOL_ACTIVITY_REPORTING_CONTRACT},
            {"role": "user", "content": user_input},
        ]
        payload = {
            "model": self.config.model,
            "messages": chat_messages,
            "stream": True,
        }

        camera_frame = extract_camera_frame(messages)
        if camera_frame:
            # Fallback path also carries the frame for gateway compatibility
            payload["camera_frame"] = {
                "data_url": camera_frame.data_url,
                "source": camera_frame.source,
                "captured_at": camera_frame.captured_at,
                "consent": camera_frame.consent,
                "contract": camera_frame.contract,
            }
        try:
            async with self._client.stream(
                "POST",
                self.config.chat_completions_url(),
                json=payload,
                headers=self._auth_headers(),
            ) as response:
                self._active_response = response
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line.removeprefix("data:").strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    for choice in data.get("choices") or []:
                        delta = choice.get("delta") or {}
                        for tc in delta.get("tool_calls") or []:
                            function = tc.get("function") or {}
                            name = function.get("name")
                            if isinstance(name, str) and name.strip():
                                yield ToolActivityEvent(name=name.strip())
                    token = _parse_chat_sse_payload(data)
                    if token:
                        yield TextDeltaEvent(content=token)
        finally:
            self._active_response = None
