import json
import unittest
from unittest.mock import MagicMock

from vokel.events import TextDeltaEvent, ToolActivityEvent
from vokel.inference import (
    InferenceError,
    _parse_native_chat_stream,
    extract_camera_frame,
    parse_sse_delta,
)
from vokel.vision import VisualContext


class InferenceParsingTests(unittest.TestCase):
    def test_parses_content_token(self):
        payload = {"choices": [{"delta": {"content": "hello"}}]}

        self.assertEqual(parse_sse_delta(f"data: {json.dumps(payload)}"), {"content": "hello"})

    def test_ignores_done_line(self):
        self.assertIsNone(parse_sse_delta("data: [DONE]"))

    def test_ignores_non_data_lines(self):
        self.assertIsNone(parse_sse_delta(": keepalive"))


if __name__ == "__main__":
    unittest.main()


# --- Native LM Studio /api/v1/chat SSE parser + VisualContext extract tests (stop-ship) ---


class NativeSSEParserTests(unittest.IsolatedAsyncioTestCase):
    async def _collect(self, mock_response) -> list:
        events = []
        async for ev in _parse_native_chat_stream(mock_response):
            events.append(ev)
        return events

    def _make_lines(self, lines: list[str]) -> MagicMock:
        mock = MagicMock()
        mock.aiter_lines = MagicMock(return_value=self._async_iter(lines))
        return mock

    async def _async_iter(self, items):
        for item in items:
            yield item

    async def test_parses_message_delta(self):
        lines = [
            "event: message.delta",
            'data: {"content": "Hello from native"}',
            "event: chat.end",
            'data: {"result": {}}',
        ]
        resp = self._make_lines(lines)
        events = await self._collect(resp)
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], TextDeltaEvent)
        self.assertEqual(events[0].content, "Hello from native")

    async def test_parses_tool_activity_start_and_success(self):
        lines = [
            "event: tool_call.start",
            'data: {"tool": "mcp_playwright_navigate", "provider_info": {"type": "plugin"}}',
            "event: tool_call.success",
            'data: {"tool": "mcp_playwright_navigate"}',
            "event: chat.end",
            'data: {"result": {}}',
        ]
        resp = self._make_lines(lines)
        events = await self._collect(resp)
        # start + success now both emit (lifecycle); at least the start is present
        self.assertGreaterEqual(len(events), 1)
        self.assertIsInstance(events[0], ToolActivityEvent)
        self.assertEqual(events[0].name, "mcp_playwright_navigate")

    async def test_raises_inference_error_on_error_event(self):
        lines = [
            "event: error",
            'data: {"error": {"type": "mcp_connection_error", "message": "MCP server down"}}',
        ]
        resp = self._make_lines(lines)
        with self.assertRaises(InferenceError) as ctx:
            await self._collect(resp)
        self.assertIn("MCP server down", str(ctx.exception))

    async def test_extract_returns_visual_context_and_prefers_metadata(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,Zm9v",
                            "visual_context": {
                                "source": "/dev/video4",
                                "captured_at": "2026-06-02T12:00:00+00:00",
                                "consent": "explicit_visual_context_for_turn",
                                "contract": "hermes_camera_frame_v1",
                            },
                        },
                    },
                    {"type": "text", "text": "what do you see?"},
                ],
            }
        ]
        vc = extract_camera_frame(messages)
        self.assertIsInstance(vc, VisualContext)
        self.assertEqual(vc.data_url, "data:image/jpeg;base64,Zm9v")
        self.assertEqual(vc.source, "/dev/video4")
        self.assertEqual(vc.captured_at, "2026-06-02T12:00:00+00:00")

    async def test_extract_falls_back_without_hardcoded_device(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,Zm9v"}}],
            }
        ]
        vc = extract_camera_frame(messages)
        self.assertIsInstance(vc, VisualContext)
        self.assertEqual(vc.data_url, "data:image/jpeg;base64,Zm9v")
        self.assertEqual(vc.source, "webcam")  # no /dev/video4 hardcode


    async def test_native_tool_lifecycle_events_emit_correct_status(self):
        """Covers start->started, success->finished, failure->failed + InferenceError for native SSE."""
        # start
        lines = ["event: tool_call.start", 'data: {"tool": "search"}', "event: chat.end", 'data: {}']
        resp = self._make_lines(lines)
        events = await self._collect(resp)
        self.assertEqual(events[0].status, "started")

        # success
        lines2 = ["event: tool_call.success", 'data: {"tool": "search"}', "event: chat.end", 'data: {}']
        resp2 = self._make_lines(lines2)
        evs2 = await self._collect(resp2)
        self.assertEqual(evs2[0].status, "finished")

        # failure raises
        lines3 = ["event: tool_call.failure", 'data: {"tool": "search", "error": {"message": "boom"}}']
        resp3 = self._make_lines(lines3)
        with self.assertRaises(InferenceError):
            await self._collect(resp3)


def test_native_visual_payload_via_real_stream_chat_with_fake_transport():
    """Call the production LmStudioNativeMcpClient.stream_chat() with fake transport
    and assert the exact input sent to the installed server:
    [ {"type": "image", "data_url": ...}, {"type": "text", "content": ...} ]
    """
    import asyncio
    import httpx
    from unittest.mock import AsyncMock, MagicMock

    from vokel.inference import LmStudioNativeMcpClient
    from vokel.config import LmStudioConfig
    from vokel.vision import VisualContext
    from vokel.events import TextDeltaEvent

    async def _run():
        vc = VisualContext(
            data_url="data:image/jpeg;base64,Zm9v", source="/dev/video4"
        )
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": vc.data_url, "visual_context": {"source": vc.source}},
                    },
                    {"type": "text", "text": "describe this"},
                ],
            }
        ]

        recorded_payloads = []

        def fake_stream(method, url, **kwargs):
            recorded_payloads.append(kwargs.get("json", {}))
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            async def aiter_lines():
                yield 'event: message.delta'
                yield 'data: {"content": "ok"}'
                yield 'event: chat.end'
                yield 'data: {"result": {"response_id": "resp_123"}}'
                yield ""
            resp.aiter_lines = aiter_lines
            resp.aclose = AsyncMock()
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=resp)
            ctx.__aexit__ = AsyncMock(return_value=None)
            return ctx

        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.stream.side_effect = fake_stream
        mock_client.aclose = AsyncMock()

        cfg = LmStudioConfig(url="http://127.0.0.1:1234/v1/chat/completions")
        native = LmStudioNativeMcpClient(cfg, client=mock_client)

        events = []
        async with native:
            async for ev in native.stream_chat(msgs):
                events.append(ev)

        assert any(isinstance(e, TextDeltaEvent) for e in events)
        assert len(recorded_payloads) >= 1
        sent_input = recorded_payloads[0]["input"]
        assert isinstance(sent_input, list)
        assert sent_input[0] == {"type": "image", "data_url": "data:image/jpeg;base64,Zm9v"}
        assert sent_input[1] == {"type": "text", "content": "describe this"}

    asyncio.run(_run())
