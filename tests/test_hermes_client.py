from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, MagicMock

import httpx

from vokel.hermes_client import (
    HermesAgentClient,
    HermesConfig,
    _extract_user_input,
    check_gateway_inference,
    format_gateway_error,
)
from vokel.agent_backend import TOOL_ACTIVITY_REPORTING_CONTRACT
from vokel.inference import InferenceError
from vokel.events import TextDeltaEvent, ToolActivityEvent


class HermesClientTests(unittest.IsolatedAsyncioTestCase):
    def test_extract_user_input_from_last_user_message(self) -> None:
        messages = [
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "Hello Hermes"},
        ]
        self.assertEqual(_extract_user_input(messages), "Hello Hermes")

    def test_format_gateway_error_for_connect_failure(self) -> None:
        config = HermesConfig(base_url="http://127.0.0.1:8642")
        message = format_gateway_error(config, httpx.ConnectError("All connection attempts failed"))
        self.assertIn("Cannot reach Hermes gateway", message)
        self.assertIn("hermes gateway run", message)

    async def test_stream_chat_yields_text_deltas(self) -> None:
        sse_lines = [
            'data: {"type":"response.output_text.delta","delta":"Hi "}',
            'data: {"type":"response.output_text.delta","delta":"there"}',
            "data: [DONE]",
        ]

        async def aiter_lines() -> object:
            for line in sse_lines:
                yield line

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = aiter_lines
        mock_response.aclose = AsyncMock()

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_http = MagicMock()
        mock_http.stream = MagicMock(return_value=mock_stream_ctx)
        mock_http.aclose = AsyncMock()

        config = HermesConfig(base_url="http://127.0.0.1:8642", session_id="vokel-test")
        client = HermesAgentClient(config, client=mock_http)

        tokens: list[str] = []
        async for event in client.stream_chat([{"role": "user", "content": "ping"}]):
            if isinstance(event, TextDeltaEvent):
                tokens.append(event.content)

        self.assertEqual("".join(tokens), "Hi there")
        mock_http.stream.assert_called_once()
        call_kwargs = mock_http.stream.call_args.kwargs
        payload = call_kwargs["json"]
        self.assertEqual(payload["conversation"], "vokel-test")
        self.assertEqual(payload["input"], "ping")
        self.assertIn(TOOL_ACTIVITY_REPORTING_CONTRACT, payload["instructions"])

    async def test_stream_chat_ignores_output_text_done_to_avoid_duplicates(self) -> None:
        sse_lines = [
            'data: {"type":"response.output_text.delta","delta":"Hi "}',
            'data: {"type":"response.output_text.delta","delta":"there"}',
            'data: {"type":"response.output_text.done","text":"Hi there"}',
            "data: [DONE]",
        ]

        async def aiter_lines() -> object:
            for line in sse_lines:
                yield line

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = aiter_lines
        mock_response.aclose = AsyncMock()

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_http = MagicMock()
        mock_http.stream = MagicMock(return_value=mock_stream_ctx)
        mock_http.aclose = AsyncMock()

        client = HermesAgentClient(HermesConfig(base_url="http://127.0.0.1:8642"), client=mock_http)

        tokens: list[str] = []
        async for event in client.stream_chat([{"role": "user", "content": "ping"}]):
            if isinstance(event, TextDeltaEvent):
                tokens.append(event.content)

        self.assertEqual("".join(tokens), "Hi there")

    async def test_stream_chat_yields_tool_activity_from_responses_events(self) -> None:
        sse_lines = [
            'data: {"type":"response.output_item.added","item":{"type":"function_call","name":"search_image"}}',
            'data: {"type":"response.output_text.delta","delta":"Done."}',
            "data: [DONE]",
        ]

        async def aiter_lines() -> object:
            for line in sse_lines:
                yield line

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = aiter_lines
        mock_response.aclose = AsyncMock()

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_http = MagicMock()
        mock_http.stream = MagicMock(return_value=mock_stream_ctx)
        mock_http.aclose = AsyncMock()

        client = HermesAgentClient(HermesConfig(base_url="http://127.0.0.1:8642"), client=mock_http)

        events = [event async for event in client.stream_chat([{"role": "user", "content": "image"}])]

        self.assertIsInstance(events[0], ToolActivityEvent)
        self.assertEqual(events[0].name, "search_image")
        self.assertIsInstance(events[1], TextDeltaEvent)
        self.assertEqual(events[1].content, "Done.")

    async def test_reset_session_rotates_conversation_id(self) -> None:
        client = HermesAgentClient(HermesConfig(session_id="vokel-old"))
        old = client.session_id
        new = client.reset_session()
        self.assertNotEqual(old, new)
        self.assertEqual(client.session_id, new)

    async def test_check_gateway_inference_accepts_output_text_done(self) -> None:
        sse_lines = [
            'data: {"type":"response.output_text.done","text":"READY"}',
            "data: [DONE]",
        ]

        async def aiter_lines() -> object:
            for line in sse_lines:
                yield line

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = aiter_lines
        mock_response.aclose = AsyncMock()

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_http = MagicMock()
        mock_http.stream = MagicMock(return_value=mock_stream_ctx)

        await check_gateway_inference(HermesConfig(), mock_http, timeout_seconds=1.0)
        self.assertEqual(mock_http.stream.call_count, 1)

    async def test_check_gateway_inference_raises_if_stream_has_no_text(self) -> None:
        sse_lines = [
            "data: [DONE]",
        ]

        async def aiter_lines() -> object:
            for line in sse_lines:
                yield line

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = aiter_lines
        mock_response.aclose = AsyncMock()

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_http = MagicMock()
        mock_http.stream = MagicMock(return_value=mock_stream_ctx)

        with self.assertRaises(InferenceError) as ctx:
            await check_gateway_inference(HermesConfig(), mock_http, timeout_seconds=1.0)
        self.assertIn("returned no streamed text", str(ctx.exception))
        self.assertEqual(mock_http.stream.call_count, 2)


if __name__ == "__main__":
    unittest.main()
