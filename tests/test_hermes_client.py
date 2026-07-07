from __future__ import annotations

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
from vokel.vision import VisualContext
from vokel.hermes.websocket_client import HermesWebSocketClient


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
        self.assertIn("API_SERVER_KEY", message)
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


class HermesCameraFrameTransportTests(unittest.IsolatedAsyncioTestCase):
    """Actual payload tests for the camera_frame contract over both Hermes transports."""

    def _make_visual_messages(self) -> list[dict]:
        vc = VisualContext(
            data_url="data:image/jpeg;base64,TESTFRAME",
            source="/dev/video4",
            captured_at="2026-06-02T12:34:56Z",
            consent="explicit_visual_context_for_turn",
            contract="hermes_camera_frame_v1",
        )
        return [
            {"role": "system", "content": "ignore"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": vc.data_url, "visual_context": {
                        "source": vc.source,
                        "captured_at": vc.captured_at,
                        "consent": vc.consent,
                        "contract": vc.contract,
                    }}},
                    {"type": "text", "text": "what is this?"},
                ],
            },
        ]

    async def test_hermes_agent_client_responses_includes_full_camera_frame(self) -> None:
        sse_lines = ['data: {"type":"response.output_text.delta","delta":"ok"}', "data: [DONE]"]

        async def aiter_lines():
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
        msgs = self._make_visual_messages()

        _ = [e async for e in client.stream_chat(msgs)]

        call = mock_http.stream.call_args
        payload = call.kwargs["json"]
        self.assertIsInstance(payload["input"], list)
        content = payload["input"][0]["content"]
        self.assertEqual(content[0]["image_url"]["url"], "data:image/jpeg;base64,TESTFRAME")
        self.assertEqual(content[1]["text"], "what is this?")
        self.assertIn("camera_frame", payload)
        cf = payload["camera_frame"]
        self.assertEqual(cf["data_url"], "data:image/jpeg;base64,TESTFRAME")
        self.assertEqual(cf["source"], "/dev/video4")
        self.assertEqual(cf["captured_at"], "2026-06-02T12:34:56Z")
        self.assertEqual(cf["consent"], "explicit_visual_context_for_turn")
        self.assertEqual(cf["contract"], "hermes_camera_frame_v1")

    async def test_hermes_http_fallback_chat_includes_camera_frame(self) -> None:
        sse_lines = ['data: {"choices":[{"delta":{"content":"ok"}}]}', "data: [DONE]"]

        async def aiter_lines():
            for line in sse_lines:
                yield line

        # Real 404 response for /v1/responses
        responses_resp = MagicMock()
        responses_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=MagicMock(status_code=404)
        )
        responses_resp.aiter_lines = aiter_lines
        responses_resp.aclose = AsyncMock()

        responses_ctx = MagicMock()
        responses_ctx.__aenter__ = AsyncMock(return_value=responses_resp)
        responses_ctx.__aexit__ = AsyncMock(return_value=None)

        # Success response for fallback
        chat_resp = MagicMock()
        chat_resp.raise_for_status = MagicMock()
        chat_resp.aiter_lines = aiter_lines
        chat_resp.aclose = AsyncMock()

        chat_ctx = MagicMock()
        chat_ctx.__aenter__ = AsyncMock(return_value=chat_resp)
        chat_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_http = MagicMock()
        call_count = 0

        def stream_side(method, url, **kw):
            nonlocal call_count
            call_count += 1
            if "responses" in url:
                return responses_ctx
            return chat_ctx

        mock_http.stream.side_effect = stream_side
        mock_http.aclose = AsyncMock()

        client = HermesAgentClient(HermesConfig(base_url="http://127.0.0.1:8642"), client=mock_http)
        msgs = self._make_visual_messages()

        # No broad swallowing: let any real error surface if test is wrong
        _ = [e async for e in client.stream_chat(msgs)]

        self.assertEqual(call_count, 2)
        # Second call must be the fallback
        second_call = mock_http.stream.call_args_list[1]
        second_url = second_call.args[1]
        self.assertIn("/v1/chat/completions", second_url)
        second_payload = second_call.kwargs["json"]
        user_content = second_payload["messages"][-1]["content"]
        self.assertEqual(user_content[0]["image_url"]["url"], "data:image/jpeg;base64,TESTFRAME")
        self.assertEqual(user_content[1]["text"], "what is this?")
        self.assertIn("camera_frame", second_payload)
        cf = second_payload["camera_frame"]
        self.assertEqual(cf["data_url"], "data:image/jpeg;base64,TESTFRAME")
        self.assertEqual(cf["source"], "/dev/video4")
        self.assertEqual(cf["captured_at"], "2026-06-02T12:34:56Z")
        self.assertEqual(cf["consent"], "explicit_visual_context_for_turn")
        self.assertEqual(cf["contract"], "hermes_camera_frame_v1")

    async def test_hermes_ws_client_start_turn_includes_camera_frame(self) -> None:
        # Minimal ws test: monkey the internal send and verify start_turn dict
        client = HermesWebSocketClient("ws://127.0.0.1:9999")

        # fake ws
        sent = []

        async def fake_send(obj):
            # real code json.dumps before send; accept str or dict
            if isinstance(obj, str):
                import json
                sent.append(json.loads(obj))
            else:
                sent.append(obj)

        client.ws = MagicMock()
        client.ws.send = fake_send
        client.remote_schema = {"tools": []}
        # connect already "done" by setting ws

        msgs = self._make_visual_messages()
        # We only care about the start_turn payload; run until first send
        try:
            agen = client.stream_chat(msgs)
            # trigger the send inside
            await agen.__anext__()  # will try to recv etc, but we only need the start send
        except Exception:
            pass  # expected, we only inspect the sent start_turn

        # Find the start_turn
        start = next((s for s in sent if s.get("type") == "start_turn"), None)
        self.assertIsNotNone(start)
        self.assertIn("camera_frame", start)
        cf = start["camera_frame"]
        self.assertEqual(cf["data_url"], "data:image/jpeg;base64,TESTFRAME")
        self.assertEqual(cf["source"], "/dev/video4")
        self.assertEqual(cf["captured_at"], "2026-06-02T12:34:56Z")
        self.assertEqual(cf["consent"], "explicit_visual_context_for_turn")
        self.assertEqual(cf["contract"], "hermes_camera_frame_v1")

    async def test_hermes_ws_connect_refused_raises_inference_error(self) -> None:
        from vokel.inference import InferenceError

        # Nothing is listening on this port; connect must surface an actionable error
        # instead of a raw OSError, and must fail fast (short timeout).
        client = HermesWebSocketClient("ws://127.0.0.1:9", timeout=2.0)
        with self.assertRaises(InferenceError) as ctx:
            await client.connect()
        self.assertIn("ws://127.0.0.1:9", str(ctx.exception))
        self.assertIsNone(client.ws)


if __name__ == "__main__":
    unittest.main()
