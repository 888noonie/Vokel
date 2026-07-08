from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from vokel.audio.beatclock import BeatClock
from vokel.audio.quantized_sink import QuantizedPlaybackSink
from vokel.web import (
    MAX_MUSICAL_TRACK_UPLOAD_BYTES,
    _MusicalTrackSlot,
    _musical_track_slots,
    app,
    detect_voice_session_command,
)
from vokel.vision import CameraDevice, VisionFrameAnalysis


def receive_expected(websocket: any, target_types: tuple[str, ...]) -> dict[str, any]:
    while True:
        msg = websocket.receive()
        if "text" in msg:
            data = json.loads(msg["text"])
            if data.get("type") in target_types:
                return data


def collect_messages(
    websocket: any,
    *,
    target_type: str,
    count: int,
    timeout_seconds: float = 2.0,
) -> list[dict[str, any]]:
    messages: list[dict[str, any]] = []
    deadline = time.monotonic() + timeout_seconds
    while len(messages) < count and time.monotonic() < deadline:
        msg = websocket.receive()
        if "text" not in msg:
            continue
        data = json.loads(msg["text"])
        if data.get("type") == target_type:
            messages.append(data)
    return messages


def _local_session_patches() -> tuple[Any, ...]:
    mock_engine = MagicMock()
    mock_engine.start = AsyncMock()
    mock_engine.close = AsyncMock()

    async def idle_run_turns(*_args: any, **_kwargs: any) -> None:
        try:
            while True:
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            return None

    mock_engine.run_turns = idle_run_turns
    mock_engine.trace = MagicMock()
    mock_engine.trace.summary_ms.return_value = {}

    return (
        patch("vokel.web.check_local_health", new_callable=AsyncMock),
        patch("vokel.web.LocalInferenceClient"),
        patch("vokel.web.KokoroPlaybackSink"),
        patch("vokel.web.SileroVadTurnProducer"),
        patch("vokel.web.SherpaOfflineAsr"),
        patch("vokel.web.BeatTrackPlayer"),
        patch("vokel.web.ConversationEngine", return_value=mock_engine),
    )


def test_root_endpoint_returns_api_status() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    if "html" in response.headers.get("content-type", ""):
        assert "<html" in response.text
    else:
        assert "message" in response.json()
        assert "Vokel Backend API" in response.json()["message"]


@patch("vokel.web.list_camera_devices")
def test_vision_camera_discovery_prefers_ps3_eye(mock_list_cameras: MagicMock) -> None:
    mock_list_cameras.return_value = [
        CameraDevice(path="/dev/video0", name="Built-in webcam (color)"),
    ]

    response = TestClient(app).get("/api/vision/cameras")

    assert response.status_code == 200
    assert response.json() == {
        "cameras": [
            {"path": "/dev/video0", "name": "Built-in webcam (color)"},
        ],
        "default_device": "/dev/video0",
        "local_only": True,
    }


@patch("vokel.web.capture_camera_frame")
@patch("vokel.web.list_camera_devices")
def test_vision_snapshot_returns_preview_frame(
    mock_list_cameras: MagicMock,
    mock_capture: MagicMock,
) -> None:
    from vokel.vision import CapturedVisionFrame

    mock_list_cameras.return_value = [CameraDevice(path="/dev/video0", name="Integrated Camera")]
    mock_capture.return_value = CapturedVisionFrame(
        device="/dev/video0",
        image_data_url="data:image/jpeg;base64,anBlZw==",
        capture_seconds=0.2,
    )

    response = TestClient(app).get("/api/vision/snapshot", params={"device": "/dev/video0"})

    assert response.status_code == 200
    assert response.json()["image_data_url"].startswith("data:image/jpeg;base64,")


@patch("vokel.web.analyze_camera_frame")
@patch("vokel.web.list_camera_devices")
def test_vision_analysis_returns_preview_and_latency(
    mock_list_cameras: MagicMock,
    mock_analyze: MagicMock,
) -> None:
    mock_list_cameras.return_value = [CameraDevice(path="/dev/video4", name="gspca main driver")]
    mock_analyze.return_value = VisionFrameAnalysis(
        device="/dev/video4",
        description="A person is looking at the camera.",
        image_data_url="data:image/jpeg;base64,anBlZw==",
        capture_seconds=1.2,
        inference_seconds=1.8,
    )

    response = TestClient(app).post(
        "/api/vision/analyze",
        json={
            "device": "/dev/video4",
            "url": "http://127.0.0.1:1234/v1/chat/completions",
            "model": "local-vlm",
            "prompt": "Describe this.",
        },
    )

    assert response.status_code == 200
    assert response.json()["description"] == "A person is looking at the camera."
    assert response.json()["image_data_url"] == "data:image/jpeg;base64,anBlZw=="
    mock_analyze.assert_called_once()


def test_vision_analysis_rejects_external_endpoint() -> None:
    response = TestClient(app).post(
        "/api/vision/analyze",
        json={"url": "https://example.com/v1/chat/completions"},
    )

    assert response.status_code == 400
    assert "loopback" in response.json()["detail"]


@patch("vokel.web.check_local_health", new_callable=AsyncMock)
@patch("vokel.web.LocalInferenceClient")
@patch("vokel.web.create_streaming_asr")
@patch("vokel.web.KokoroPlaybackSink")
def test_websocket_browser_mode_flow(
    mock_kokoro_class: MagicMock,
    mock_create_asr: MagicMock,
    mock_lm_client_class: MagicMock,
    mock_check_local_health: AsyncMock,
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

        # Receive session started and listening status
        res1 = receive_expected(websocket, ("session_started",))
        assert res1["mode"] == "browser"

        res2 = receive_expected(websocket, ("status",))
        assert res2["status"] == "listening"

        # Send binary float32 data
        audio_frame = np.zeros(1600, dtype=np.float32)
        websocket.send_bytes(audio_frame.tobytes())

        # Receive transcripts
        res3 = receive_expected(websocket, ("partial_transcript", "final_transcript"))
        assert res3["type"] in ("partial_transcript", "final_transcript")

        # End session
        websocket.send_json({
            "type": "stop_session",
        })
        res_stopped = receive_expected(websocket, ("session_stopped",))
        assert res_stopped["type"] == "session_stopped"


@patch("vokel.web.check_local_health", new_callable=AsyncMock)
@patch("vokel.web.LocalInferenceClient")
@patch("vokel.web.create_streaming_asr")
@patch("vokel.web.KokoroPlaybackSink")
def test_websocket_barge_in_bypasses(
    mock_kokoro_class: MagicMock,
    mock_create_asr: MagicMock,
    mock_lm_client_class: MagicMock,
    mock_check_local_health: AsyncMock,
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
        
        receive_expected(websocket, ("session_started",))
        receive_expected(websocket, ("status",))

        # Send binary packet
        websocket.send_bytes(np.zeros(800, dtype=np.float32).tobytes())
        res = receive_expected(websocket, ("partial_transcript", "stable_transcript"))
        assert res["type"] in ("partial_transcript", "stable_transcript")


@patch("vokel.web.KokoroPlaybackSink")
def test_websocket_voice_preview_does_not_start_session(mock_kokoro_class: MagicMock) -> None:
    mock_kokoro = MagicMock()
    mock_kokoro_class.return_value = mock_kokoro
    mock_kokoro.voice = "af_heart"
    mock_kokoro.speed = 1.0

    async def dummy_kokoro_stream(*args: any, **kwargs: any):
        yield np.zeros(1000, dtype=np.float32), 24000

    mock_kokoro.kokoro.create_stream.return_value = dummy_kokoro_stream()

    client = TestClient(app)
    with client.websocket_connect("/api/ws") as websocket:
        websocket.send_json({
            "type": "preview_voice",
            "voice": "af_heart",
            "tts_speed": 1.0,
        })

        started = receive_expected(websocket, ("voice_preview_started",))
        assert started == {"type": "voice_preview_started", "voice": "af_heart"}

        binary = websocket.receive_bytes()
        assert len(binary) == 4000

        finished = receive_expected(websocket, ("voice_preview_finished",))
        assert finished == {"type": "voice_preview_finished", "voice": "af_heart"}


def test_websocket_execute_state_reports_ffmpeg_available() -> None:
    client = TestClient(app)
    with (
        patch("vokel.web.shutil.which", return_value="/usr/bin/ffmpeg"),
        client.websocket_connect("/api/ws") as websocket,
    ):
        initial = receive_expected(websocket, ("execute_state",))
        assert initial["ffmpeg_available"] is True

    with (
        patch("vokel.web.shutil.which", return_value=None),
        client.websocket_connect("/api/ws") as websocket,
    ):
        initial = receive_expected(websocket, ("execute_state",))
        assert initial["ffmpeg_available"] is False


def test_websocket_execute_consent_scaffold() -> None:
    client = TestClient(app)
    with client.websocket_connect("/api/ws") as websocket:
        initial = receive_expected(websocket, ("execute_state",))
        assert initial["armed"] is False
        assert initial["risk"] == "none"

        websocket.send_json({"type": "arm_execute", "risk": "medium"})
        armed = receive_expected(websocket, ("execute_state",))
        assert armed["armed"] is True
        assert armed["risk"] == "medium"

        websocket.send_json({"type": "confirm_execute"})
        event = receive_expected(websocket, ("agent_event",))
        while event["event"] != "execute_confirm_ignored":
            event = receive_expected(websocket, ("agent_event",))
        assert event["level"] == "warning"

        websocket.send_json({"type": "cancel_execute"})
        cancelled = receive_expected(websocket, ("execute_state",))
        while cancelled["armed"] is not False:
            cancelled = receive_expected(websocket, ("execute_state",))
        assert cancelled["armed"] is False


def test_websocket_musical_mode_wraps_local_sink_and_emits_beats() -> None:
    mock_llm = MagicMock()
    mock_llm.__aenter__ = AsyncMock(return_value=mock_llm)
    mock_llm.__aexit__ = AsyncMock(return_value=None)

    patches = _local_session_patches()
    with (
        patches[0] as _health,
        patches[1] as mock_lm_client_class,
        patches[2] as mock_kokoro_class,
        patches[3] as _vad,
        patches[4] as _asr,
        patches[5] as mock_beat_track_class,
        patches[6] as _engine_class,
        patch("vokel.web.QuantizedPlaybackSink", wraps=QuantizedPlaybackSink) as qps_mock,
    ):
        mock_lm_client_class.return_value = mock_llm
        mock_kokoro_class.return_value = MagicMock()
        mock_beat_track_class.return_value.start = AsyncMock()
        mock_beat_track_class.return_value.stop = AsyncMock()

        client = TestClient(app)
        with client.websocket_connect("/api/ws") as websocket:
            receive_expected(websocket, ("execute_state",))
            websocket.send_json({
                "type": "start_session",
                "mode": "local",
                "playback": "kokoro",
                "url": "http://mocked:1234",
                "model": "mock-model",
                "musical_mode": True,
                "musical_bpm": 120,
            })
            started = receive_expected(websocket, ("session_started",))
            assert started["mode"] == "local"
            beats = collect_messages(websocket, target_type="beat", count=2, timeout_seconds=2.0)
            assert len(beats) == 2
            assert beats[0]["bpm"] == 120
            qps_mock.assert_called_once()
            websocket.send_json({"type": "stop_session"})
            receive_expected(websocket, ("session_stopped",))


def test_websocket_musical_mode_off_does_not_wrap_sink() -> None:
    mock_llm = MagicMock()
    mock_llm.__aenter__ = AsyncMock(return_value=mock_llm)
    mock_llm.__aexit__ = AsyncMock(return_value=None)

    patches = _local_session_patches()
    with (
        patches[0],
        patches[1] as mock_lm_client_class,
        patches[2] as mock_kokoro_class,
        patches[3],
        patches[4],
        patches[5],
        patches[6] as mock_engine_class,
        patch("vokel.web.QuantizedPlaybackSink") as qps_mock,
    ):
        mock_lm_client_class.return_value = mock_llm
        mock_kokoro_class.return_value = MagicMock()

        client = TestClient(app)
        with client.websocket_connect("/api/ws") as websocket:
            receive_expected(websocket, ("execute_state",))
            websocket.send_json({
                "type": "start_session",
                "mode": "local",
                "playback": "kokoro",
                "url": "http://mocked:1234",
                "model": "mock-model",
            })
            receive_expected(websocket, ("session_started",))
            qps_mock.assert_not_called()
            mock_engine_class.assert_called_once()
            assert "config" not in mock_engine_class.call_args.kwargs
            websocket.send_json({"type": "stop_session"})
            receive_expected(websocket, ("session_stopped",))


def test_websocket_musical_mode_passes_rap_system_prompt() -> None:
    mock_llm = MagicMock()
    mock_llm.__aenter__ = AsyncMock(return_value=mock_llm)
    mock_llm.__aexit__ = AsyncMock(return_value=None)

    patches = _local_session_patches()
    with (
        patches[0],
        patches[1] as mock_lm_client_class,
        patches[2] as mock_kokoro_class,
        patches[3],
        patches[4],
        patches[5] as mock_beat_track_class,
        patches[6] as mock_engine_class,
        patch("vokel.web.QuantizedPlaybackSink", wraps=QuantizedPlaybackSink),
    ):
        mock_lm_client_class.return_value = mock_llm
        mock_kokoro_class.return_value = MagicMock()
        mock_beat_track_class.return_value.start = AsyncMock()
        mock_beat_track_class.return_value.stop = AsyncMock()

        client = TestClient(app)
        with client.websocket_connect("/api/ws") as websocket:
            receive_expected(websocket, ("execute_state",))
            websocket.send_json({
                "type": "start_session",
                "mode": "local",
                "playback": "kokoro",
                "url": "http://mocked:1234",
                "model": "mock-model",
                "musical_mode": True,
                "musical_bpm": 120,
            })
            receive_expected(websocket, ("session_started",))
            mock_engine_class.assert_called_once()
            config = mock_engine_class.call_args.kwargs["config"]
            assert "120 BPM" in config.system_prompt
            assert "rapper" in config.system_prompt.lower()
            websocket.send_json({"type": "stop_session"})
            receive_expected(websocket, ("session_stopped",))


def test_websocket_session_context_empty_keeps_voice_loop_config_none() -> None:
    mock_llm = MagicMock()
    mock_llm.__aenter__ = AsyncMock(return_value=mock_llm)
    mock_llm.__aexit__ = AsyncMock(return_value=None)

    patches = _local_session_patches()
    with (
        patches[0],
        patches[1] as mock_lm_client_class,
        patches[2] as mock_kokoro_class,
        patches[3],
        patches[4],
        patches[5],
        patches[6] as mock_engine_class,
    ):
        mock_lm_client_class.return_value = mock_llm
        mock_kokoro_class.return_value = MagicMock()

        client = TestClient(app)
        with client.websocket_connect("/api/ws") as websocket:
            receive_expected(websocket, ("execute_state",))
            websocket.send_json({
                "type": "start_session",
                "mode": "local",
                "playback": "kokoro",
                "url": "http://mocked:1234",
                "model": "mock-model",
                "session_context": "   \t  ",
            })
            receive_expected(websocket, ("session_started",))
            mock_engine_class.assert_called_once()
            assert "config" not in mock_engine_class.call_args.kwargs
            websocket.send_json({"type": "stop_session"})
            receive_expected(websocket, ("session_stopped",))


def test_websocket_session_context_non_musical_injects_topic_and_mishearing() -> None:
    mock_llm = MagicMock()
    mock_llm.__aenter__ = AsyncMock(return_value=mock_llm)
    mock_llm.__aexit__ = AsyncMock(return_value=None)

    patches = _local_session_patches()
    with (
        patches[0],
        patches[1] as mock_lm_client_class,
        patches[2] as mock_kokoro_class,
        patches[3],
        patches[4],
        patches[5],
        patches[6] as mock_engine_class,
    ):
        mock_lm_client_class.return_value = mock_llm
        mock_kokoro_class.return_value = MagicMock()

        client = TestClient(app)
        with client.websocket_connect("/api/ws") as websocket:
            receive_expected(websocket, ("execute_state",))
            websocket.send_json({
                "type": "start_session",
                "mode": "local",
                "playback": "kokoro",
                "url": "http://mocked:1234",
                "model": "mock-model",
                "session_context": "  rapping   about   breakfast  " + ("x" * 250),
            })
            receive_expected(websocket, ("session_started",))
            mock_engine_class.assert_called_once()
            config = mock_engine_class.call_args.kwargs["config"]
            prompt = config.system_prompt
            assert "Session topic:" in prompt
            assert "rapping about breakfast" in prompt
            assert "imperfect speech-to-text" in prompt
            assert "'wrap' when the topic is rap" in prompt
            # whitespace collapsed + 200-char cap
            topic_part = prompt.split("Session topic: ", 1)[1]
            topic_text = topic_part.split(". The user's words", 1)[0]
            assert "  " not in topic_text
            assert len(topic_text) <= 200
            websocket.send_json({"type": "stop_session"})
            receive_expected(websocket, ("session_stopped",))


def test_websocket_musical_mode_context_with_track_orders_addendum() -> None:
    mock_llm = MagicMock()
    mock_llm.__aenter__ = AsyncMock(return_value=mock_llm)
    mock_llm.__aexit__ = AsyncMock(return_value=None)

    slot_id = "ctxtrackslot12345"
    samples = np.linspace(-0.1, 0.1, 800, dtype=np.float32)

    patches = _local_session_patches()
    try:
        with (
            patches[0],
            patches[1] as mock_lm_client_class,
            patches[2] as mock_kokoro_class,
            patches[3],
            patches[4],
            patches[5] as mock_beat_track_class,
            patches[6] as mock_engine_class,
            patch("vokel.web.QuantizedPlaybackSink", wraps=QuantizedPlaybackSink),
            patch("vokel.web.secrets.token_hex", return_value=slot_id),
        ):
            mock_lm_client_class.return_value = mock_llm
            mock_kokoro_class.return_value = MagicMock()
            mock_beat_track_class.return_value.start = AsyncMock()
            mock_beat_track_class.return_value.stop = AsyncMock()
            mock_beat_track_class.return_value.load_buffer = MagicMock()

            client = TestClient(app)
            with client.websocket_connect("/api/ws") as websocket:
                receive_expected(websocket, ("execute_state",))
                # Slot is allocated empty on connect; seed filename as upload would.
                slot = _musical_track_slots[slot_id]
                slot.buffer = samples
                slot.filename = "breakfast-beats.mp3"
                websocket.send_json({
                    "type": "start_session",
                    "mode": "local",
                    "playback": "kokoro",
                    "url": "http://mocked:1234",
                    "model": "mock-model",
                    "musical_mode": True,
                    "musical_bpm": 120,
                    "session_context": "rapping about breakfast",
                })
                receive_expected(websocket, ("session_started",))
                config = mock_engine_class.call_args.kwargs["config"]
                prompt = config.system_prompt
                bpm_i = prompt.index("120 BPM")
                topic_i = prompt.index("The performance topic is: rapping about breakfast.")
                track_i = prompt.index(
                    "The backing track file is named 'breakfast-beats.mp3'"
                )
                mishear_i = prompt.index("imperfect speech-to-text")
                assert bpm_i < topic_i < track_i < mishear_i
                # Slice 10 no-context musical prompt is a strict prefix of this one.
                base_addendum_end = prompt.index(" The performance topic is:")
                no_context_prompt = prompt[:base_addendum_end]
                assert "120 BPM" in no_context_prompt
                assert "rapper" in no_context_prompt.lower()
                assert "performance topic" not in no_context_prompt
                websocket.send_json({"type": "stop_session"})
                receive_expected(websocket, ("session_stopped",))
    finally:
        _musical_track_slots.pop(slot_id, None)


def test_websocket_musical_mode_context_without_track_omits_filename() -> None:
    mock_llm = MagicMock()
    mock_llm.__aenter__ = AsyncMock(return_value=mock_llm)
    mock_llm.__aexit__ = AsyncMock(return_value=None)

    patches = _local_session_patches()
    with (
        patches[0],
        patches[1] as mock_lm_client_class,
        patches[2] as mock_kokoro_class,
        patches[3],
        patches[4],
        patches[5] as mock_beat_track_class,
        patches[6] as mock_engine_class,
        patch("vokel.web.QuantizedPlaybackSink", wraps=QuantizedPlaybackSink),
    ):
        mock_lm_client_class.return_value = mock_llm
        mock_kokoro_class.return_value = MagicMock()
        mock_beat_track_class.return_value.start = AsyncMock()
        mock_beat_track_class.return_value.stop = AsyncMock()

        client = TestClient(app)
        with client.websocket_connect("/api/ws") as websocket:
            receive_expected(websocket, ("execute_state",))
            websocket.send_json({
                "type": "start_session",
                "mode": "local",
                "playback": "kokoro",
                "url": "http://mocked:1234",
                "model": "mock-model",
                "musical_mode": True,
                "musical_bpm": 90,
                "session_context": "space opera freestyle",
            })
            receive_expected(websocket, ("session_started",))
            prompt = mock_engine_class.call_args.kwargs["config"].system_prompt
            assert "The performance topic is: space opera freestyle." in prompt
            assert "backing track file is named" not in prompt
            assert "imperfect speech-to-text" in prompt
            websocket.send_json({"type": "stop_session"})
            receive_expected(websocket, ("session_stopped",))


def test_websocket_musical_bpm_is_clamped() -> None:
    mock_llm = MagicMock()
    mock_llm.__aenter__ = AsyncMock(return_value=mock_llm)
    mock_llm.__aexit__ = AsyncMock(return_value=None)

    patches = _local_session_patches()
    with (
        patches[0],
        patches[1] as mock_lm_client_class,
        patches[2] as mock_kokoro_class,
        patches[3],
        patches[4],
        patches[5] as mock_beat_track_class,
        patches[6],
        patch("vokel.web.BeatClock", wraps=BeatClock) as clock_mock,
    ):
        mock_lm_client_class.return_value = mock_llm
        mock_kokoro_class.return_value = MagicMock()
        mock_beat_track_class.return_value.start = AsyncMock()
        mock_beat_track_class.return_value.stop = AsyncMock()

        client = TestClient(app)
        with client.websocket_connect("/api/ws") as websocket:
            receive_expected(websocket, ("execute_state",))
            websocket.send_json({
                "type": "start_session",
                "mode": "local",
                "playback": "kokoro",
                "url": "http://mocked:1234",
                "model": "mock-model",
                "musical_mode": True,
                "musical_bpm": 240,
            })
            receive_expected(websocket, ("session_started",))
            clock_mock.assert_called_with(bpm=160.0)
            beats = collect_messages(websocket, target_type="beat", count=1, timeout_seconds=1.0)
            assert beats[0]["bpm"] == 160.0
            websocket.send_json({"type": "stop_session"})
            receive_expected(websocket, ("session_stopped",))


def test_websocket_musical_mode_teardown_stops_clock() -> None:
    mock_llm = MagicMock()
    mock_llm.__aenter__ = AsyncMock(return_value=mock_llm)
    mock_llm.__aexit__ = AsyncMock(return_value=None)
    clocks: list[BeatClock] = []

    def capture_clock(*args: Any, **kwargs: Any) -> BeatClock:
        clock = BeatClock(*args, **kwargs)
        clocks.append(clock)
        return clock

    patches = _local_session_patches()
    with (
        patches[0],
        patches[1] as mock_lm_client_class,
        patches[2] as mock_kokoro_class,
        patches[3],
        patches[4],
        patches[5] as mock_beat_track_class,
        patches[6],
        patch("vokel.web.BeatClock", side_effect=capture_clock),
    ):
        mock_lm_client_class.return_value = mock_llm
        mock_kokoro_class.return_value = MagicMock()
        mock_beat_track_class.return_value.start = AsyncMock()
        mock_beat_track_class.return_value.stop = AsyncMock()

        client = TestClient(app)
        with client.websocket_connect("/api/ws") as websocket:
            receive_expected(websocket, ("execute_state",))
            websocket.send_json({
                "type": "start_session",
                "mode": "local",
                "playback": "kokoro",
                "url": "http://mocked:1234",
                "model": "mock-model",
                "musical_mode": True,
            })
            receive_expected(websocket, ("session_started",))
            assert len(clocks) == 1
            assert clocks[0].running is True
            websocket.send_json({"type": "stop_session"})
            receive_expected(websocket, ("session_stopped",))
            assert clocks[0].running is False


def test_websocket_set_musical_level_routes_to_beat_track() -> None:
    mock_llm = MagicMock()
    mock_llm.__aenter__ = AsyncMock(return_value=mock_llm)
    mock_llm.__aexit__ = AsyncMock(return_value=None)
    mock_beat_track = MagicMock()
    mock_beat_track.start = AsyncMock()
    mock_beat_track.stop = AsyncMock()

    patches = _local_session_patches()
    with (
        patches[0],
        patches[1] as mock_lm_client_class,
        patches[2] as mock_kokoro_class,
        patches[3],
        patches[4],
        patches[5] as mock_beat_track_class,
        patches[6],
        patch("vokel.web.QuantizedPlaybackSink", wraps=QuantizedPlaybackSink),
    ):
        mock_lm_client_class.return_value = mock_llm
        mock_kokoro_class.return_value = MagicMock()
        mock_beat_track_class.return_value = mock_beat_track

        client = TestClient(app)
        with client.websocket_connect("/api/ws") as websocket:
            receive_expected(websocket, ("execute_state",))
            websocket.send_json({
                "type": "start_session",
                "mode": "local",
                "playback": "kokoro",
                "url": "http://mocked:1234",
                "model": "mock-model",
                "musical_mode": True,
                "musical_level": 0.5,
            })
            receive_expected(websocket, ("session_started",))
            collect_messages(websocket, target_type="beat", count=1, timeout_seconds=1.0)
            mock_beat_track_class.assert_called_once()
            assert mock_beat_track_class.call_args.kwargs["level"] == 0.5
            websocket.send_json({"type": "set_musical_level", "level": 0.35})
            collect_messages(websocket, target_type="beat", count=1, timeout_seconds=1.0)
            mock_beat_track.set_level.assert_called_once_with(0.35)
            websocket.send_json({"type": "stop_session"})
            receive_expected(websocket, ("session_stopped",))


def test_websocket_set_musical_level_ignores_malformed_level() -> None:
    mock_llm = MagicMock()
    mock_llm.__aenter__ = AsyncMock(return_value=mock_llm)
    mock_llm.__aexit__ = AsyncMock(return_value=None)
    mock_beat_track = MagicMock()
    mock_beat_track.start = AsyncMock()
    mock_beat_track.stop = AsyncMock()

    patches = _local_session_patches()
    with (
        patches[0],
        patches[1] as mock_lm_client_class,
        patches[2] as mock_kokoro_class,
        patches[3],
        patches[4],
        patches[5] as mock_beat_track_class,
        patches[6],
        patch("vokel.web.QuantizedPlaybackSink", wraps=QuantizedPlaybackSink),
    ):
        mock_lm_client_class.return_value = mock_llm
        mock_kokoro_class.return_value = MagicMock()
        mock_beat_track_class.return_value = mock_beat_track

        client = TestClient(app)
        with client.websocket_connect("/api/ws") as websocket:
            receive_expected(websocket, ("execute_state",))
            websocket.send_json({
                "type": "start_session",
                "mode": "local",
                "playback": "kokoro",
                "url": "http://mocked:1234",
                "model": "mock-model",
                "musical_mode": True,
            })
            receive_expected(websocket, ("session_started",))
            collect_messages(websocket, target_type="beat", count=1, timeout_seconds=1.0)
            websocket.send_json({"type": "set_musical_level", "level": "loud"})
            collect_messages(websocket, target_type="beat", count=1, timeout_seconds=1.0)
            mock_beat_track.set_level.assert_not_called()
            websocket.send_json({"type": "set_musical_level", "level": 0.2})
            collect_messages(websocket, target_type="beat", count=1, timeout_seconds=1.0)
            mock_beat_track.set_level.assert_called_once_with(0.2)
            beats = collect_messages(websocket, target_type="beat", count=1, timeout_seconds=1.0)
            assert len(beats) == 1
            websocket.send_json({"type": "stop_session"})
            receive_expected(websocket, ("session_stopped",))


def test_websocket_musical_style_whitelist_falls_back_to_beat() -> None:
    mock_llm = MagicMock()
    mock_llm.__aenter__ = AsyncMock(return_value=mock_llm)
    mock_llm.__aexit__ = AsyncMock(return_value=None)

    patches = _local_session_patches()
    with (
        patches[0],
        patches[1] as mock_lm_client_class,
        patches[2] as mock_kokoro_class,
        patches[3],
        patches[4],
        patches[5] as mock_beat_track_class,
        patches[6],
        patch("vokel.web.QuantizedPlaybackSink", wraps=QuantizedPlaybackSink),
    ):
        mock_lm_client_class.return_value = mock_llm
        mock_kokoro_class.return_value = MagicMock()
        mock_beat_track_class.return_value.start = AsyncMock()
        mock_beat_track_class.return_value.stop = AsyncMock()

        client = TestClient(app)
        with client.websocket_connect("/api/ws") as websocket:
            execute_state = receive_expected(websocket, ("execute_state",))
            websocket.send_json({
                "type": "start_session",
                "mode": "local",
                "playback": "kokoro",
                "url": "http://mocked:1234",
                "model": "mock-model",
                "musical_mode": True,
                "musical_style": "wobble",
            })
            receive_expected(websocket, ("session_started",))
            mock_beat_track_class.assert_called_once()
            assert mock_beat_track_class.call_args.kwargs["style"] == "beat"
            websocket.send_json({"type": "stop_session"})
            receive_expected(websocket, ("session_stopped",))
            assert execute_state["musical_track_slot"]


def test_websocket_musical_style_metronome_is_passed() -> None:
    mock_llm = MagicMock()
    mock_llm.__aenter__ = AsyncMock(return_value=mock_llm)
    mock_llm.__aexit__ = AsyncMock(return_value=None)

    patches = _local_session_patches()
    with (
        patches[0],
        patches[1] as mock_lm_client_class,
        patches[2] as mock_kokoro_class,
        patches[3],
        patches[4],
        patches[5] as mock_beat_track_class,
        patches[6],
        patch("vokel.web.QuantizedPlaybackSink", wraps=QuantizedPlaybackSink),
    ):
        mock_lm_client_class.return_value = mock_llm
        mock_kokoro_class.return_value = MagicMock()
        mock_beat_track_class.return_value.start = AsyncMock()
        mock_beat_track_class.return_value.stop = AsyncMock()

        client = TestClient(app)
        with client.websocket_connect("/api/ws") as websocket:
            receive_expected(websocket, ("execute_state",))
            websocket.send_json({
                "type": "start_session",
                "mode": "local",
                "playback": "kokoro",
                "url": "http://mocked:1234",
                "model": "mock-model",
                "musical_mode": True,
                "musical_style": "metronome",
            })
            receive_expected(websocket, ("session_started",))
            assert mock_beat_track_class.call_args.kwargs["style"] == "metronome"
            websocket.send_json({"type": "stop_session"})
            receive_expected(websocket, ("session_stopped",))


def test_upload_musical_track_decodes_and_stores_buffer() -> None:
    samples = np.linspace(-0.25, 0.25, 2400, dtype=np.float32)
    slot_id = "testslot12345678"
    _musical_track_slots[slot_id] = _MusicalTrackSlot()

    try:
        with patch("vokel.web.decode_audio_file", return_value=samples):
            client = TestClient(app)
            response = client.post(
                "/api/musical/track",
                params={"slot": slot_id},
                files={"file": ("loop.wav", b"RIFFfake", "audio/wav")},
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["samples"] == len(samples)
        assert abs(payload["duration_seconds"] - (len(samples) / 24000.0)) < 0.001
        stored = _musical_track_slots[slot_id].buffer
        assert stored is not None
        assert stored.shape == samples.shape
        assert np.allclose(stored, samples)
        assert _musical_track_slots[slot_id].filename == "loop.wav"
    finally:
        _musical_track_slots.pop(slot_id, None)


def test_upload_musical_track_sanitizes_filename() -> None:
    samples = np.linspace(-0.25, 0.25, 1200, dtype=np.float32)
    slot_id = "nameslot12345678"
    _musical_track_slots[slot_id] = _MusicalTrackSlot()
    long_name = "a" * 100 + ".mp3"
    try:
        with patch("vokel.web.decode_audio_file", return_value=samples):
            client = TestClient(app)
            response = client.post(
                "/api/musical/track",
                params={"slot": slot_id},
                files={"file": ("../../evil.mp3", b"RIFFfake", "audio/wav")},
            )
            assert response.status_code == 200
            assert _musical_track_slots[slot_id].filename == "evil.mp3"

            response = client.post(
                "/api/musical/track",
                params={"slot": slot_id},
                files={"file": (long_name, b"RIFFfake", "audio/wav")},
            )
            assert response.status_code == 200
            stored_name = _musical_track_slots[slot_id].filename
            assert stored_name is not None
            assert len(stored_name) == 80
            assert stored_name == long_name[:80]
    finally:
        _musical_track_slots.pop(slot_id, None)


def test_upload_musical_track_rejects_oversized_upload() -> None:
    slot_id = "oversize12345678"
    _musical_track_slots[slot_id] = _MusicalTrackSlot()
    try:
        client = TestClient(app)
        response = client.post(
            "/api/musical/track",
            params={"slot": slot_id},
            files={
                "file": (
                    "huge.wav",
                    b"x" * (MAX_MUSICAL_TRACK_UPLOAD_BYTES + 1),
                    "audio/wav",
                )
            },
        )
        assert response.status_code == 413
    finally:
        _musical_track_slots.pop(slot_id, None)


def test_upload_musical_track_returns_503_when_ffmpeg_missing() -> None:
    from vokel.audio.beattrack import MusicalTrackDecodeError

    slot_id = "ffmpegmissing1234"
    _musical_track_slots[slot_id] = _MusicalTrackSlot()
    try:
        with patch(
            "vokel.web.decode_audio_file",
            side_effect=MusicalTrackDecodeError(
                "ffmpeg is required to decode backing tracks but was not found on PATH"
            ),
        ):
            client = TestClient(app)
            response = client.post(
                "/api/musical/track",
                params={"slot": slot_id},
                files={"file": ("loop.mp3", b"fake", "audio/mpeg")},
            )
        assert response.status_code == 503
        assert "ffmpeg" in response.json()["detail"]
    finally:
        _musical_track_slots.pop(slot_id, None)


def test_delete_musical_track_returns_404_for_unknown_slot() -> None:
    client = TestClient(app)
    response = client.delete(
        "/api/musical/track",
        params={"slot": "unknownslot1234"},
    )
    assert response.status_code == 404
    assert "Unknown musical track slot" in response.json()["detail"]


def test_delete_musical_track_clears_slot_and_live_player() -> None:
    from vokel.audio.beattrack import BeatTrackPlayer

    slot_id = "clearslot12345678"
    samples = np.linspace(-0.1, 0.1, 1200, dtype=np.float32)
    player = BeatTrackPlayer(bpm=120.0)
    player.load_buffer(samples)
    player._stream = object()
    _musical_track_slots[slot_id] = _MusicalTrackSlot(buffer=samples, player=player)

    try:
        client = TestClient(app)
        response = client.delete(
            "/api/musical/track",
            params={"slot": slot_id},
        )
        assert response.status_code == 204
        assert response.content == b""
        slot = _musical_track_slots[slot_id]
        assert slot.buffer is None
        assert slot.filename is None
        assert slot.player is not None
        assert not player._external_buffer
        assert player._buffer is not None
        assert float(np.max(np.abs(player._buffer))) > 0.01
        assert player._cursor == 0
    finally:
        _musical_track_slots.pop(slot_id, None)


def test_upload_musical_track_rejects_bad_decode() -> None:
    from vokel.audio.beattrack import MusicalTrackDecodeError

    slot_id = "badfile123456789"
    _musical_track_slots[slot_id] = _MusicalTrackSlot()
    try:
        with patch(
            "vokel.web.decode_audio_file",
            side_effect=MusicalTrackDecodeError("not audio"),
        ):
            client = TestClient(app)
            response = client.post(
                "/api/musical/track",
                params={"slot": slot_id},
                files={"file": ("nope.txt", b"hello", "text/plain")},
            )
        assert response.status_code == 400
        assert "not audio" in response.json()["detail"]
    finally:
        _musical_track_slots.pop(slot_id, None)


def test_websocket_set_musical_nudge_routes_to_beat_track() -> None:
    mock_llm = MagicMock()
    mock_llm.__aenter__ = AsyncMock(return_value=mock_llm)
    mock_llm.__aexit__ = AsyncMock(return_value=None)
    mock_beat_track = MagicMock()
    mock_beat_track.start = AsyncMock()
    mock_beat_track.stop = AsyncMock()

    patches = _local_session_patches()
    with (
        patches[0],
        patches[1] as mock_lm_client_class,
        patches[2] as mock_kokoro_class,
        patches[3],
        patches[4],
        patches[5] as mock_beat_track_class,
        patches[6],
        patch("vokel.web.QuantizedPlaybackSink", wraps=QuantizedPlaybackSink),
    ):
        mock_lm_client_class.return_value = mock_llm
        mock_kokoro_class.return_value = MagicMock()
        mock_beat_track_class.return_value = mock_beat_track

        client = TestClient(app)
        with client.websocket_connect("/api/ws") as websocket:
            receive_expected(websocket, ("execute_state",))
            websocket.send_json({
                "type": "start_session",
                "mode": "local",
                "playback": "kokoro",
                "url": "http://mocked:1234",
                "model": "mock-model",
                "musical_mode": True,
            })
            receive_expected(websocket, ("session_started",))
            collect_messages(websocket, target_type="beat", count=1, timeout_seconds=1.0)
            websocket.send_json({"type": "set_musical_nudge", "ms": -25})
            collect_messages(websocket, target_type="beat", count=1, timeout_seconds=1.0)
            mock_beat_track.nudge_cursor.assert_called_once_with(-25.0)
            websocket.send_json({"type": "stop_session"})
            receive_expected(websocket, ("session_stopped",))


def test_websocket_set_musical_nudge_ignores_malformed_ms() -> None:
    mock_llm = MagicMock()
    mock_llm.__aenter__ = AsyncMock(return_value=mock_llm)
    mock_llm.__aexit__ = AsyncMock(return_value=None)
    mock_beat_track = MagicMock()
    mock_beat_track.start = AsyncMock()
    mock_beat_track.stop = AsyncMock()

    patches = _local_session_patches()
    with (
        patches[0],
        patches[1] as mock_lm_client_class,
        patches[2] as mock_kokoro_class,
        patches[3],
        patches[4],
        patches[5] as mock_beat_track_class,
        patches[6],
        patch("vokel.web.QuantizedPlaybackSink", wraps=QuantizedPlaybackSink),
    ):
        mock_lm_client_class.return_value = mock_llm
        mock_kokoro_class.return_value = MagicMock()
        mock_beat_track_class.return_value = mock_beat_track

        client = TestClient(app)
        with client.websocket_connect("/api/ws") as websocket:
            receive_expected(websocket, ("execute_state",))
            websocket.send_json({
                "type": "start_session",
                "mode": "local",
                "playback": "kokoro",
                "url": "http://mocked:1234",
                "model": "mock-model",
                "musical_mode": True,
            })
            receive_expected(websocket, ("session_started",))
            collect_messages(websocket, target_type="beat", count=1, timeout_seconds=1.0)
            websocket.send_json({"type": "set_musical_nudge", "ms": "early"})
            collect_messages(websocket, target_type="beat", count=1, timeout_seconds=1.0)
            mock_beat_track.nudge_cursor.assert_not_called()
            websocket.send_json({"type": "set_musical_nudge", "ms": 25})
            collect_messages(websocket, target_type="beat", count=1, timeout_seconds=1.0)
            mock_beat_track.nudge_cursor.assert_called_once_with(25.0)
            websocket.send_json({"type": "stop_session"})
            receive_expected(websocket, ("session_stopped",))


def test_detect_voice_session_command_pause() -> None:
    assert detect_voice_session_command("pause") == "pause"
    assert detect_voice_session_command("Okay, just hang on for now.") == "pause"
    assert detect_voice_session_command("wait") == "pause"


def test_detect_voice_session_command_resume() -> None:
    assert detect_voice_session_command("continue") == "resume"
    assert detect_voice_session_command("please resume now") == "resume"


def test_detect_voice_session_command_ignores_normal_utterances() -> None:
    assert detect_voice_session_command("what time is it") is None
    assert detect_voice_session_command("search latest UK AI news") is None


def test_voice_vision_capture_cancel_stops_process_before_releasing_lock():
    import asyncio
    from contextlib import suppress

    from vokel.vision import capture_camera_frame_async
    from vokel.web import _VISION_CAPTURE_LOCK

    async def _run():
        process_started = asyncio.Event()
        terminate_requested = asyncio.Event()
        process_may_exit = asyncio.Event()
        second_acquired = asyncio.Event()

        class FakeProcess:
            returncode = None
            terminate_called = False
            kill_called = False

            async def wait(self):
                await process_may_exit.wait()
                self.returncode = -15
                return self.returncode

            def terminate(self):
                self.terminate_called = True
                terminate_requested.set()

            def kill(self):
                self.kill_called = True

        process = FakeProcess()

        async def create_subprocess_exec(*_args):
            process_started.set()
            return process

        async def protected_capture():
            async with _VISION_CAPTURE_LOCK:
                return await capture_camera_frame_async(
                    device="/dev/video4",
                    width=640,
                    height=480,
                    framerate=30,
                    warmup_frames=8,
                )

        async def try_second_capture():
            async with _VISION_CAPTURE_LOCK:
                second_acquired.set()

        with patch("vokel.vision.asyncio.create_subprocess_exec", create_subprocess_exec):
            outer = asyncio.create_task(protected_capture())
            await asyncio.wait_for(process_started.wait(), timeout=1.0)
            outer.cancel()
            await asyncio.wait_for(terminate_requested.wait(), timeout=1.0)
            assert process.terminate_called

            second = asyncio.create_task(try_second_capture())
            await asyncio.sleep(0)
            assert not second_acquired.is_set()
            assert not outer.done()

            process_may_exit.set()
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(outer, timeout=1.0)
            await asyncio.wait_for(second, timeout=1.0)
            assert second_acquired.is_set()
            assert outer.cancelled()
            assert not process.kill_called

    asyncio.run(_run())
