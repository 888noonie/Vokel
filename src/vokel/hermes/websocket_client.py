from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import websockets
from websockets.asyncio.client import ClientConnection

from ..agent_backend import AgentBackend
from ..events import Event, TextDeltaEvent
from ..inference import ChatMessage, InferenceError


class HermesWebSocketClient(AgentBackend):
    """Lightweight direct WebSocket client for Hermes agents (Termux / local Android path).

    Auto-selected when the configured URL starts with ws:// or wss://.
    """

    def __init__(self, url: str, timeout: float = 30.0, max_context_messages: int = 12):
        self.url = url
        self.timeout = timeout
        self.max_context_messages = max_context_messages

        self.ws: ClientConnection | None = None
        self.remote_schema: dict[str, Any] | None = None
        self._current_turn_id: str | None = None
        self._cancel_event = asyncio.Event()

    async def connect(self) -> None:
        if self.ws:
            return

        self.ws = await websockets.connect(
            self.url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        )

        # Proven handshake (exact pattern validated on Pixel 8 Pro)
        await self._send({"type": "ping"})
        pong = await self._recv()
        if pong.get("type") != "pong":
            raise InferenceError(f"Expected pong during handshake, got: {pong}")

        await self._send({"type": "get_schema"})
        schema = await self._recv()
        if schema.get("type") != "tool_schema":
            raise InferenceError(f"Expected tool_schema during handshake, got: {schema}")

        self.remote_schema = schema

    # === AgentBackend protocol ===

    async def __aenter__(self) -> HermesWebSocketClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[Event]:
        """Stream a turn using the direct WebSocket protocol.

        Ignores the `tools` parameter (remote agent owns tools, same as HTTP Hermes client).
        """
        await self.connect()
        self._cancel_event.clear()

        user_input = self._extract_last_user_message(messages)
        if not user_input:
            raise InferenceError("HermesWebSocketClient requires at least one user message")

        turn_id = f"turn-{asyncio.get_running_loop().time():.6f}"
        self._current_turn_id = turn_id

        context = self._build_context(messages)

        await self._send({
            "type": "start_turn",
            "turn_id": turn_id,
            "prompt": user_input,
            "context": context,
            "tools": self.remote_schema.get("tools", []) if self.remote_schema else [],
        })

        try:
            async for raw in self.ws:
                if self._cancel_event.is_set():
                    break

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if msg.get("turn_id") != turn_id:
                    continue

                if msg["type"] == "delta":
                    yield TextDeltaEvent(content=msg.get("content", ""))
                elif msg["type"] == "tool_call":
                    # v0.1: surface as text delta until proper ToolCallEvent support
                    name = msg.get("name", "unknown")
                    yield TextDeltaEvent(content=f"[tool_call:{name}]")
                elif msg["type"] == "turn_complete":
                    break
                elif msg["type"] == "error":
                    raise InferenceError(msg.get("message", "Unknown error from Hermes agent"))
        finally:
            self._current_turn_id = None

    async def cancel_active(self) -> None:
        if self._current_turn_id:
            await self._send({"type": "cancel", "turn_id": self._current_turn_id})
            self._cancel_event.set()

    async def close(self) -> None:
        if self.ws:
            await self.ws.close()
            self.ws = None
        self.remote_schema = None
        self._current_turn_id = None

    # --- helpers ---

    async def _send(self, obj: dict[str, Any]) -> None:
        assert self.ws is not None
        await self.ws.send(json.dumps(obj))

    async def _recv(self, timeout: float | None = None) -> dict[str, Any]:
        assert self.ws is not None
        raw = await asyncio.wait_for(self.ws.recv(), timeout or self.timeout)
        return json.loads(raw)

    def _extract_last_user_message(self, messages: Sequence[ChatMessage]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content") or ""
                if isinstance(content, str):
                    return content.strip()
                # Handle list content blocks if present
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") in ("text", "input_text"):
                            return str(block.get("text", "")).strip()
        return ""

    def _build_context(self, messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
        """Send a recent window. Desktop owns memory."""
        recent = list(messages)[-self.max_context_messages:]
        return [
            {
                "role": m.get("role"),
                "content": m.get("content") if isinstance(m.get("content"), str) else "",
            }
            for m in recent
        ]
