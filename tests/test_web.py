from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from voyce.web import app


def test_root_endpoint_returns_api_status() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    if "html" in response.headers.get("content-type", ""):
        assert "<html" in response.text
    else:
        assert "message" in response.json()
        assert "Voyce Backend API" in response.json()["message"]


@patch("voyce.web.LocalInferenceClient")
@patch("voyce.web.create_streaming_asr")
@patch("voyce.web.KokoroPlaybackSink")
def test_websocket_browser_mode_flow(
    mock_kokoro_class: MagicMock,
    mock_create_asr: MagicMock,
    mock_lm_client_class: MagicMock,
) -> None:
    # Setup mocks
    mock_llm = MagicMock()
    mock_lm_client_class.return_value = mock_llm
    mock_llm.__aenter__.return_value = mock_llm
    mock_llm.__aexit__.return_value = None
    mock_llm.stream_chat = MagicMock()

    # Async generator mock for stream_chat
    async def dummy_stream_chat(*args: any, **kwargs: any):
        yield "Hello"
        yield " world!"

    mock_llm.stream_chat.return_value = dummy_stream_chat()

    mock_asr = MagicMock()
    mock_create_asr.return_value = mock_asr
    mock_stream = MagicMock()
    mock_asr.create_stream.return_value = mock_stream

    # Setup stream behaviour
    mock_stream.get_result.return_value = "hello"
    mock_stream.is_endpoint.return_value = True

    # Mock Kokoro and its create_stream
    mock_kokoro = MagicMock()
    mock_kokoro_class.return_value = mock_kokoro
    
    async def dummy_kokoro_stream(*args: any, **kwargs: any):
        yield np.zeros(1000, dtype=np.float32), 24000

    mock_kokoro.kokoro.create_stream.return_value = dummy_kokoro_stream()
    mock_kokoro.voice = "af_heart"

    # Run WebSocket Test
    client = TestClient(app)
    with client.websocket_connect("/api/ws") as websocket:
        # Start session
        websocket.send_json({
            "type": "start_session",
            "mode": "browser",
            "url": "http://mocked:1234",
            "model": "mock-model",
        })

        # Helper to receive non-telemetry text messages and safely ignore binary messages
        def receive_expected(target_types: tuple[str, ...]) -> dict[str, any]:
            while True:
                msg = websocket.receive()
                if "text" in msg:
                    data = json.loads(msg["text"])
                    if data.get("type") in target_types:
                        return data

        # Receive session started and listening status
        res1 = receive_expected(("session_started",))
        assert res1["mode"] == "browser"

        res2 = receive_expected(("status",))
        assert res2["status"] == "listening"

        # Send binary float32 data
        audio_frame = np.zeros(1600, dtype=np.float32)
        websocket.send_bytes(audio_frame.tobytes())

        # Receive transcripts
        res3 = receive_expected(("partial_transcript", "final_transcript"))
        assert res3["type"] in ("partial_transcript", "final_transcript")

        # End session
        websocket.send_json({
            "type": "stop_session",
        })
        res_stopped = receive_expected(("session_stopped",))
        assert res_stopped["type"] == "session_stopped"


@patch("voyce.web.LocalInferenceClient")
@patch("voyce.web.create_streaming_asr")
@patch("voyce.web.KokoroPlaybackSink")
def test_websocket_barge_in_bypasses(
    mock_kokoro_class: MagicMock,
    mock_create_asr: MagicMock,
    mock_lm_client_class: MagicMock,
) -> None:
    # Setup mocks for LLM and ASR
    mock_llm = MagicMock()
    mock_lm_client_class.return_value = mock_llm
    mock_llm.__aenter__.return_value = mock_llm
    mock_llm.__aexit__.return_value = None

    mock_asr = MagicMock()
    mock_create_asr.return_value = mock_asr
    mock_stream = MagicMock()
    mock_asr.create_stream.return_value = mock_stream
    mock_stream.get_result.return_value = "interrupt word"
    mock_stream.is_endpoint.return_value = False

    client = TestClient(app)
    with client.websocket_connect("/api/ws") as websocket:
        # Start browser session
        websocket.send_json({
            "type": "start_session",
            "mode": "browser",
        })
        
        # Helper to receive non-telemetry text messages
        def receive_expected(target_types: tuple[str, ...]) -> dict[str, any]:
            while True:
                msg = websocket.receive()
                if "text" in msg:
                    data = json.loads(msg["text"])
                    if data.get("type") in target_types:
                        return data

        receive_expected(("session_started",))
        receive_expected(("status",))

        # Send binary packet
        websocket.send_bytes(np.zeros(800, dtype=np.float32).tobytes())
        res = receive_expected(("partial_transcript", "stable_transcript"))
        assert res["type"] in ("partial_transcript", "stable_transcript")
