from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any

from .events import Event, TextDeltaEvent
from .inference import ChatMessage, InferenceError

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
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    return content if isinstance(content, str) else ""


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
    import httpx

    headers: dict[str, str] = {}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    health_url = f"{config.base_url.rstrip('/')}/health"
    try:
        response = await client.get(health_url, headers=headers)
        response.raise_for_status()
    except Exception as exc:
        raise InferenceError(format_gateway_error(config, exc)) from exc


class HermesAgentClient:
    """Voice I/O adapter: Hermes owns reasoning, memory, and tools."""

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
            "conversation": self._session_id,
            "stream": True,
        }

        try:
            async for token in self._stream_responses(payload):
                if token:
                    yield TextDeltaEvent(content=token)
        except InferenceError:
            raise
        except Exception as exc:
            import httpx

            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (
                404,
                405,
                501,
            ):
                async for token in self._stream_chat_completions(user_input):
                    if token:
                        yield TextDeltaEvent(content=token)
                return
            raise InferenceError(format_gateway_error(self.config, exc)) from exc

    async def _stream_responses(self, payload: dict[str, Any]) -> AsyncIterator[str]:
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
                    token = _parse_responses_sse_payload(data)
                    if token:
                        yield token
        finally:
            self._active_response = None

    async def _stream_chat_completions(self, user_input: str) -> AsyncIterator[str]:
        assert self._client is not None
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": user_input}],
            "stream": True,
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
                    token = _parse_chat_sse_payload(data)
                    if token:
                        yield token
        finally:
            self._active_response = None
