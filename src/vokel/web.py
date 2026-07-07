from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import tempfile
import traceback
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Literal

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .audio import (
    MicVadConfig,
    SherpaOfflineAsr,
    SherpaOfflineAsrConfig,
    SileroVadTurnProducer,
    create_streaming_asr,
)
from .audio.beatclock import BeatClock, ClockStopped
from .audio.beattrack import (
    MAX_TRACK_DECODE_SECONDS,
    BeatTrackPlayer,
    MusicalTrackDecodeError,
    SAMPLE_RATE as MUSICAL_TRACK_SAMPLE_RATE,
    decode_audio_file,
)
from .audio.quantized_sink import QuantizedPlaybackSink
from .auto_followup import (
    AutoFollowupScheduler,
    clamp_auto_followup_seconds,
    DEFAULT_AUTO_FOLLOWUP_SECONDS,
)
from .agent_backend import AgentBackend
from .config import LmStudioConfig, VoiceLoopConfig
from .engine import AgentMode, ConversationEngine
from .hermes_client import (
    HermesAgentClient,
    HermesConfig,
    check_gateway_health,
    check_gateway_inference,
)
from .inference import (
    InferenceError,
    LocalInferenceClient,
    LmStudioNativeMcpClient,
    check_local_health,
)
from .memory import MemoryConfig, SQLiteMemoryStore
from .playback import (
    KOKORO_VOICES,
    KokoroPlaybackSink,
    SpdSayPlaybackSink,
    ConsolePlaybackSink,
    PlaybackSink,
    sanitize_for_speech,
)
from .telemetry import LatencyTrace, TraceEvent
from .vision import (
    CAMERA_GUIDANCE_OFFER,
    CAMERA_VISION_PROMPT,
    DEFAULT_VISION_PROMPT,
    SpokenReply,
    VisualContext,
    analyze_camera_frame,
    capture_camera_frame,
    capture_camera_frame_async,
    capture_prompt_for,
    detect_camera_command,
    is_loopback_url,
    list_camera_devices,
    pick_default_camera,
    should_capture_visual_context,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vokel.web")

_FRONTEND_DIST = Path("frontend/dist")
_FAVICON_PATH = _FRONTEND_DIST / "favicon.svg"
_VISION_CAPTURE_LOCK = asyncio.Lock()
MAX_MUSICAL_TRACK_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_MUSICAL_TRACK_SAMPLES = 15 * 60 * MUSICAL_TRACK_SAMPLE_RATE
_MUSICAL_STYLES = frozenset({"beat", "metronome"})


@dataclass
class _MusicalTrackSlot:
    buffer: np.ndarray | None = None
    player: BeatTrackPlayer | None = None


_musical_track_slots: dict[str, _MusicalTrackSlot] = {}


def _parse_musical_style(raw: Any) -> Literal["beat", "metronome"]:
    style = str(raw or "beat").strip().lower()
    if style in _MUSICAL_STYLES:
        return style  # type: ignore[return-value]
    return "beat"


def _apply_musical_track_buffer(slot_id: str, samples: np.ndarray) -> None:
    slot = _musical_track_slots.get(slot_id)
    if slot is None:
        return
    slot.buffer = samples
    if slot.player is not None:
        slot.player.load_buffer(samples)


def _clear_musical_track_buffer(slot_id: str) -> None:
    slot = _musical_track_slots.get(slot_id)
    if slot is None:
        return
    slot.buffer = None
    if slot.player is not None:
        slot.player.clear_buffer()


VoiceSessionCommand = Literal["pause", "resume"]
_VOICE_FILLER_PREFIX_RE = re.compile(r"^(?:ok(?:ay)?|please|just|now)\s+")
_VOICE_SPACE_RE = re.compile(r"\s+")


def detect_voice_session_command(text: str) -> VoiceSessionCommand | None:
    """Recognize lightweight voice control commands for session state only."""
    normalized = _VOICE_SPACE_RE.sub(" ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()
    if not normalized:
        return None
    while True:
        stripped = _VOICE_FILLER_PREFIX_RE.sub("", normalized)
        if stripped == normalized:
            break
        normalized = stripped.strip()
    words = normalized.split()
    if not words or len(words) > 8:
        return None

    if normalized in {
        "pause",
        "hold on",
        "wait",
        "hang on",
        "hang on for now",
        "hold on for now",
        "pause for now",
    }:
        return "pause"
    if normalized in {"continue", "resume", "continue now", "resume now"}:
        return "resume"
    return None


class _QuietStaticAccessLogFilter(logging.Filter):
    """Keep uvicorn access logs focused on API traffic, not favicon/assets."""

    _SKIP_FRAGMENTS = ('"GET /favicon', '"GET /assets/')

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(fragment in message for fragment in self._SKIP_FRAGMENTS)


logging.getLogger("uvicorn.access").addFilter(_QuietStaticAccessLogFilter())

app = FastAPI(title="Vokel Web Server")

# Allow CORS for development React servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WebSocketTraceObserver:
    """Observer that sends LatencyTrace events to a WebSocket."""

    def __init__(self, send_json_coro: Callable[[dict[str, Any]], Coroutine[Any, Any, None]]):
        self.send_json_coro = send_json_coro
        self._loop = asyncio.get_running_loop()

    def on_event(self, event: TraceEvent) -> None:
        async def send() -> None:
            try:
                await self.send_json_coro({
                    "type": "telemetry",
                    "event": event.name,
                    "timestamp_ns": event.timestamp_ns,
                    "fields": event.fields,
                })
                # Explicit transcript messages for the web UI (local hardware mode).
                if event.name == "asr_finished" and event.fields.get("text"):
                    await self.send_json_coro({
                        "type": "user_transcript",
                        "text": str(event.fields["text"]),
                    })
                if event.name == "generation_finished" and event.fields.get("text"):
                    await self.send_json_coro({
                        "type": "assistant_reply",
                        "text": str(event.fields["text"]),
                    })
            except Exception as e:
                logger.debug(f"Failed to send telemetry: {e}")

        self._loop.create_task(send())


class WebSocketPlaybackSink:
    """Playback sink that stream phrases and synthesized audio over a WebSocket."""

    def __init__(
        self,
        send_json_coro: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
        send_bytes_coro: Callable[[bytes], Coroutine[Any, Any, None]],
        kokoro_sink: KokoroPlaybackSink | None = None,
    ):
        self.send_json_coro = send_json_coro
        self.send_bytes_coro = send_bytes_coro
        self.kokoro_sink = kokoro_sink

    async def speak(self, phrase: str) -> None:
        await self.send_json_coro({
            "type": "assistant_phrase",
            "phrase": phrase,
        })
        speech_phrase = sanitize_for_speech(phrase)

        if self.kokoro_sink:
            try:
                async for samples, sample_rate in self.kokoro_sink.kokoro.create_stream(
                    speech_phrase, self.kokoro_sink.voice, self.kokoro_sink.speed, "en-us"
                ):
                    samples_f32 = np.asarray(samples, dtype=np.float32)
                    await self.send_bytes_coro(samples_f32.tobytes())
            except Exception as e:
                logger.error(f"Kokoro Web synthesis failed: {e}")
        else:
            # Fallback when Kokoro is not loaded: wait to simulate reading speed
            await asyncio.sleep(len(speech_phrase) * 0.05)

    async def stop(self) -> None:
        await self.send_json_coro({
            "type": "playback_stop",
        })


class VisionAnalyzeRequest(BaseModel):
    device: str = "/dev/video0"
    url: str = LmStudioConfig.url
    model_name: str = Field(default=LmStudioConfig.model, alias="model")
    prompt: str = DEFAULT_VISION_PROMPT
    api_key: str = ""
    backend: str = "local"
    max_tokens: int = Field(default=120, ge=1, le=400)
    width: int = Field(default=640, ge=160, le=1920)
    height: int = Field(default=480, ge=120, le=1080)
    framerate: int = Field(default=30, ge=1, le=60)
    warmup_frames: int = Field(default=8, ge=1, le=30)


@app.get("/api/vision/cameras")
def get_vision_cameras() -> dict[str, Any]:
    cameras = list_camera_devices()
    default_device = pick_default_camera(cameras)
    return {
        "cameras": [{"path": camera.path, "name": camera.name} for camera in cameras],
        "default_device": default_device,
        "local_only": True,
    }


@app.get("/api/vision/snapshot")
async def vision_snapshot(device: str) -> dict[str, Any]:
    cameras = list_camera_devices()
    if device not in {camera.path for camera in cameras}:
        raise HTTPException(status_code=400, detail=f"Camera device is not available: {device}")
    if _VISION_CAPTURE_LOCK.locked():
        raise HTTPException(status_code=409, detail="Camera busy — AI is capturing a frame.")

    try:
        async with _VISION_CAPTURE_LOCK:
            captured = await asyncio.to_thread(
                capture_camera_frame,
                device=device,
                width=640,
                height=480,
                framerate=15,
                warmup_frames=2,
            )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "device": captured.device,
        "image_data_url": captured.image_data_url,
        "capture_seconds": captured.capture_seconds,
    }


@app.post("/api/vision/analyze")
async def analyze_vision(request: VisionAnalyzeRequest) -> dict[str, Any]:
    from .vision import is_hermes_gateway_url

    hermes_backend = request.backend == "hermes" or is_hermes_gateway_url(request.url)
    if not hermes_backend and not is_loopback_url(request.url):
        raise HTTPException(
            status_code=400,
            detail="Live vision only sends frames to a loopback local or Hermes gateway endpoint.",
        )

    cameras = list_camera_devices()
    if request.device not in {camera.path for camera in cameras}:
        raise HTTPException(status_code=400, detail=f"Camera device is not available: {request.device}")
    if _VISION_CAPTURE_LOCK.locked():
        raise HTTPException(status_code=409, detail="A live vision capture is already running.")

    try:
        from .jan_key import resolve_llm_api_key

        api_key = (
            request.api_key
            if hermes_backend
            else resolve_llm_api_key(
                request.url,
                explicit_key=request.api_key,
                env_key=LmStudioConfig.api_key,
            )
        )
        async with _VISION_CAPTURE_LOCK:
            analysis = await asyncio.to_thread(
                analyze_camera_frame,
                device=request.device,
                url=request.url,
                model=request.model_name,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                width=request.width,
                height=request.height,
                framerate=request.framerate,
                warmup_frames=request.warmup_frames,
                api_key=api_key,
                backend="hermes" if hermes_backend else "local",
            )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "device": analysis.device,
        "description": analysis.description,
        "image_data_url": analysis.image_data_url,
        "capture_seconds": analysis.capture_seconds,
        "inference_seconds": analysis.inference_seconds,
        "local_only": True,
    }


@app.post("/api/musical/track")
async def upload_musical_track(
    slot: str = Query(..., min_length=8, max_length=64),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    if slot not in _musical_track_slots:
        raise HTTPException(status_code=404, detail="Unknown musical track slot")

    upload_path: Path | None = None
    try:
        chunks: list[bytes] = []
        total_bytes = 0
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_MUSICAL_TRACK_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Backing track upload exceeds the 50 MB limit",
                )
            chunks.append(chunk)
        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Backing track upload is empty")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".upload") as handle:
            for chunk in chunks:
                handle.write(chunk)
            upload_path = Path(handle.name)

        try:
            samples = await asyncio.to_thread(
                decode_audio_file,
                upload_path,
                timeout_seconds=MAX_TRACK_DECODE_SECONDS,
            )
        except MusicalTrackDecodeError as exc:
            message = str(exc)
            if "ffmpeg is required" in message:
                raise HTTPException(status_code=503, detail=message) from exc
            raise HTTPException(status_code=400, detail=message) from exc

        if samples.size > MAX_MUSICAL_TRACK_SAMPLES:
            raise HTTPException(
                status_code=400,
                detail="Decoded backing track exceeds the 15 minute limit",
            )

        _apply_musical_track_buffer(slot, samples)
        return {
            "duration_seconds": float(samples.size) / MUSICAL_TRACK_SAMPLE_RATE,
            "samples": int(samples.size),
        }
    finally:
        if upload_path is not None:
            upload_path.unlink(missing_ok=True)


@app.delete("/api/musical/track", status_code=204)
async def delete_musical_track(
    slot: str = Query(..., min_length=8, max_length=64),
) -> Response:
    if slot not in _musical_track_slots:
        raise HTTPException(status_code=404, detail="Unknown musical track slot")
    _clear_musical_track_buffer(slot)
    return Response(status_code=204)


@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("WebSocket connection established")

    # Serialize turns and auto follow-up so they cannot double-fire the LLM.
    turn_lock = asyncio.Lock()

    # Keep track of active tasks and engines to clean them up on disconnect
    session_mode: str | None = None
    engine: ConversationEngine | None = None
    agent_client: AgentBackend | None = None
    local_loop_task: asyncio.Task[None] | None = None
    session_paused = False
    active_memory_store: SQLiteMemoryStore | None = None
    auto_followup_scheduler: AutoFollowupScheduler | None = None
    auto_followup_enabled = True
    auto_followup_seconds = DEFAULT_AUTO_FOLLOWUP_SECONDS
    execute_armed = False
    execute_risk = "none"
    current_agent_backend = "vokel"
    local_run_loop: Callable[[], Coroutine[Any, Any, None]] | None = None
    browser_asr: Any | None = None
    browser_asr_stream: Any | None = None
    browser_all_samples: list[float] = []
    browser_last_text = ""
    browser_last_changed_time = asyncio.get_running_loop().time()
    browser_stable_fired = False
    vision_voice_enabled = False
    vision_device = "/dev/video0"
    vision_lm_url = LmStudioConfig.url
    # Resolved by the browser when it already holds the camera (Live Feed preview);
    # lets "shoot" grab that frame instead of losing a race for a busy V4L2 device.
    browser_frame_future: asyncio.Future[str | None] | None = None
    musical_clock: BeatClock | None = None
    beat_track: BeatTrackPlayer | None = None
    beat_forward_task: asyncio.Task[None] | None = None
    musical_track_slot = secrets.token_hex(16)
    _musical_track_slots[musical_track_slot] = _MusicalTrackSlot()

    async def teardown_musical_mode() -> None:
        nonlocal musical_clock, beat_track, beat_forward_task
        if beat_forward_task is not None and not beat_forward_task.done():
            beat_forward_task.cancel()
            try:
                await beat_forward_task
            except asyncio.CancelledError:
                pass
        beat_forward_task = None
        if musical_clock is not None:
            await musical_clock.stop()
            musical_clock = None
        track_slot = _musical_track_slots.get(musical_track_slot)
        if track_slot is not None:
            track_slot.player = None
        if beat_track is not None:
            await beat_track.stop()
            beat_track = None

    async def run_auto_followup() -> None:
        if engine is None or session_paused:
            return
        if engine._current_generation and not engine._current_generation.done():
            return
        if not engine._playback_queue.empty():
            return
        async with turn_lock:
            try:
                await send_json({"type": "auto_followup", "status": "triggered"})
                await send_json({"type": "status", "status": "generating"})
                if auto_followup_scheduler:
                    auto_followup_scheduler.on_user_activity()
                await engine.submit_auto_followup()
                await engine.wait_for_playback()
                await send_json({
                    "type": "summary",
                    "metrics": engine.trace.summary_ms(),
                })
            except Exception as exc:
                logger.error(f"Auto follow-up failed: {exc}")
                await send_json({"type": "error", "message": str(exc)})
            finally:
                if session_paused or engine is None:
                    return
                await send_json({"type": "status", "status": "listening"})
                if auto_followup_scheduler and not session_paused:
                    auto_followup_scheduler.arm_listening()

    def attach_auto_followup_scheduler() -> None:
        nonlocal auto_followup_scheduler
        auto_followup_scheduler = AutoFollowupScheduler(
            enabled=auto_followup_enabled,
            delay_s=auto_followup_seconds,
            on_trigger=run_auto_followup,
        )

    def cancel_auto_followup() -> None:
        if auto_followup_scheduler:
            auto_followup_scheduler.cancel()

    def memory_entry_payload(entry: Any) -> dict[str, Any]:
        return {
            "id": entry.id,
            "text": entry.user_text,
            "created_at_ns": entry.created_at_ns,
            "kind": entry.kind,
        }

    async def send_memory_facts(store: SQLiteMemoryStore | None) -> None:
        if store is None:
            await send_json({"type": "memory_facts", "facts": []})
            return
        facts = await store.list_facts(limit=50)
        await send_json({
            "type": "memory_facts",
            "facts": [memory_entry_payload(fact) for fact in facts],
        })

    async def send_json(data: dict[str, Any]) -> None:
        await websocket.send_json(data)

    async def send_bytes(data: bytes) -> None:
        await websocket.send_bytes(data)

    async def send_agent_event(
        event: str,
        *,
        backend: str | None = None,
        detail: str = "",
        level: str = "info",
        **fields: Any,
    ) -> None:
        payload: dict[str, Any] = {
            "type": "agent_event",
            "event": event,
            "level": level,
            "backend": backend or current_agent_backend,
            "detail": detail,
        }
        payload.update(fields)
        await send_json(payload)

    async def send_execute_state(detail: str = "") -> None:
        await send_json({
            "type": "execute_state",
            "armed": execute_armed,
            "risk": execute_risk,
            "detail": detail,
            "musical_track_slot": musical_track_slot,
        })

    async def request_browser_frame(timeout: float = 3.0) -> str | None:
        """Ask the dashboard for one frame from its live preview and await the reply.

        The browser already owns the camera while Live Feed is on, so a server-side
        GStreamer grab would fail with "device busy". Capturing from the browser avoids
        the conflict entirely and works for any camera the browser can see. Returns
        ``None`` quickly when no preview is active (the browser replies with null).
        """
        nonlocal browser_frame_future
        if browser_frame_future is not None and not browser_frame_future.done():
            browser_frame_future.cancel()
        loop = asyncio.get_running_loop()
        browser_frame_future = loop.create_future()
        await send_json({"type": "request_camera_frame", "device": vision_device})
        try:
            data_url = await asyncio.wait_for(browser_frame_future, timeout=timeout)
        except (TimeoutError, asyncio.CancelledError):
            return None
        finally:
            browser_frame_future = None
        if isinstance(data_url, str) and data_url.startswith("data:image"):
            return data_url
        return None

    async def capture_voice_vision_context() -> str | VisualContext | None:
        # Builtin LM route: private frame only to localhost LM endpoint.
        # Hermes route: local capture; frame sent only via explicitly consented
        # camera_frame payload contract (visible in audit/transcript). No LM loopback required.
        if not hermes_mode and not is_loopback_url(vision_lm_url):
            await send_json({
                "type": "error",
                "message": "Voice camera context requires a loopback LM Studio endpoint for the local model route.",
            })
            return None

        # Prefer a frame from the browser preview when it is live: it already owns the
        # camera (server capture would hit "device busy") and covers cameras GStreamer
        # cannot negotiate. Falls through to local capture when no preview is active.
        browser_url = await request_browser_frame()
        if browser_url:
            await send_json({
                "type": "vision_context_captured",
                "device": vision_device,
                "image_data_url": browser_url,
                "capture_seconds": 0.0,
            })
            await send_agent_event(
                "visual_context_attached",
                backend="vokel",
                detail="Fresh browser-preview frame attached to spoken question",
                device=vision_device,
                source="browser_preview",
            )
            if hermes_mode:
                await send_agent_event(
                    "external_media_route_initiated",
                    backend="hermes",
                    detail="Explicitly approved camera frame routed to Hermes (camera_frame contract)",
                    device=vision_device,
                    route="camera_frame",
                    consent="voice_context_arm",
                )
            return VisualContext(
                data_url=browser_url,
                source=vision_device,
                captured_at=datetime.now(timezone.utc).isoformat(),
                consent="explicit_visual_context_for_turn",
                contract="hermes_camera_frame_v1",
            )

        device = await resolve_voice_camera_device()
        if not device:
            await send_json({
                "type": "error",
                "message": "No camera is available to capture a frame.",
            })
            return None

        await send_agent_event(
            "visual_context_capture_started",
            backend="vokel",
            detail=f"Capturing one local frame from {device}",
            device=device,
        )
        try:
            async with _VISION_CAPTURE_LOCK:
                captured = await capture_camera_frame_async(
                    device=device,
                    width=640,
                    height=480,
                    framerate=30,
                    warmup_frames=8,
                )
        except asyncio.CancelledError:
            # The capture helper stopped GStreamer before the lock was released.
            raise
        except RuntimeError as exc:
            await send_agent_event(
                "visual_context_capture_failed",
                backend="vokel",
                level="error",
                detail=str(exc),
                device=vision_device,
            )
            await send_json({"type": "error", "message": str(exc)})
            return None

        await send_json({
            "type": "vision_context_captured",
            "device": captured.device,
            "image_data_url": captured.image_data_url,
            "capture_seconds": captured.capture_seconds,
        })
        await send_agent_event(
            "visual_context_attached",
            backend="vokel",
            detail="Fresh local camera frame attached to spoken question",
            device=captured.device,
            capture_seconds=captured.capture_seconds,
        )
        if hermes_mode:
            await send_agent_event(
                "external_media_route_initiated",
                backend="hermes",
                detail="Explicitly approved camera frame routed to Hermes (camera_frame contract)",
                device=captured.device,
                route="camera_frame",
                consent="voice_context_arm",
            )

        captured_at = datetime.now(timezone.utc).isoformat()
        vc = VisualContext(
            data_url=captured.image_data_url,
            source=captured.device or vision_device,
            captured_at=captured_at,
            consent="explicit_visual_context_for_turn",
            contract="hermes_camera_frame_v1",
        )
        return vc

    async def resolve_voice_camera_device() -> str:
        """Return a camera path that can actually pull frames.

        The dashboard often ships a stale default (``/dev/video0``) that is not a
        capturable node, which silently broke every "shoot". When the configured
        device cannot capture, fall back to autodetect, adopt it, and tell the UI so
        the toggle and panel reflect the camera Vokel is really using.
        """
        nonlocal vision_device
        cameras = list_camera_devices()
        if vision_device in {camera.path for camera in cameras}:
            return vision_device
        fallback = pick_default_camera(cameras)
        if not fallback:
            return ""
        if fallback != vision_device:
            vision_device = fallback
            await send_json({
                "type": "vision_context_state",
                "enabled": vision_voice_enabled,
                "device": vision_device,
            })
        return fallback

    async def ensure_voice_camera_armed(reason: str) -> bool:
        """Arm Camera Questions from an explicit spoken command and sync the UI toggle.

        Saying "watch me" or "shoot" is itself explicit consent to be seen, so we flip
        the arm state, audit it, and tell the dashboard so the switch reflects reality.
        """
        nonlocal vision_voice_enabled
        if vision_voice_enabled:
            return True
        device = await resolve_voice_camera_device()
        if not device:
            await send_json({
                "type": "error",
                "message": "No camera is available to start the voice loop.",
            })
            return False
        vision_voice_enabled = True
        await send_agent_event(
            "visual_context_auto_armed",
            backend="vokel",
            detail=f"Camera Questions armed by voice command ({reason})",
            device=device,
            reason=reason,
        )
        await send_json({
            "type": "vision_context_state",
            "enabled": True,
            "device": device,
        })
        return True

    async def send_vision_control(action: str) -> None:
        """Tell the dashboard to start/stop its live AI watch loop."""
        await send_json({
            "type": "vision_control",
            "action": action,
            "device": vision_device,
        })
        await send_agent_event(
            "vision_control",
            backend="vokel",
            detail=f"Live video loop {action.replace('_', ' ')}",
            action=action,
            device=vision_device,
        )

    async def provide_voice_visual_context(
        user_text: str,
    ) -> str | VisualContext | SpokenReply | None:
        command = detect_camera_command(user_text)

        # Control commands are deterministic: Vokel acts and speaks the result itself,
        # so even a small local model never has to decide or narrate camera actions.
        if command == "watch":
            await ensure_voice_camera_armed("watch_me")
            return SpokenReply(CAMERA_GUIDANCE_OFFER)
        if command == "action":
            await send_vision_control("start_live")
            return SpokenReply(
                "Action. I'm watching the live feed now — say cut when you want me to stop."
            )
        if command == "cut":
            await send_vision_control("stop_live")
            return SpokenReply("Cut. I've stopped watching the live feed.")

        wants_capture = command == "shoot" or should_capture_visual_context(user_text)
        if not wants_capture:
            return None

        if not vision_voice_enabled:
            if command == "shoot":
                # Explicit photo intent counts as consent — arm if a camera exists.
                if not await ensure_voice_camera_armed("shoot"):
                    return VisualContext(
                        data_url="", source=vision_device, consent="capture_failed"
                    )
            else:
                await send_agent_event(
                    "visual_context_not_armed",
                    backend="vokel",
                    level="warning",
                    detail="Visual question blocked until Camera Questions in Voice Loop is enabled",
                )
                return VisualContext(
                    data_url="",
                    source=vision_device,
                    consent="voice_context_not_armed",
                )

        await send_json({"type": "status", "status": "capturing_vision"})
        captured = await capture_voice_vision_context()
        await send_json({"type": "status", "status": "generating"})
        if captured is None:
            return ""
        # A bare "shoot" makes a poor vision prompt, so answer a clean instruction
        # about the frame; but if the user attached a real question ("take a picture
        # and tell me what's on my head"), answer that instead. The transcript always
        # keeps what the user actually said.
        if command == "shoot" and captured.data_url:
            prompt = capture_prompt_for(user_text, bare_prompt=CAMERA_VISION_PROMPT)
            return replace(captured, prompt=prompt)
        return captured

    async def pause_active_session(source: str) -> None:
        nonlocal session_paused, local_loop_task
        if not engine:
            return
        await send_agent_event(
            "session_paused",
            detail="Session paused",
            backend="vokel",
            source=source,
        )
        session_paused = True
        cancel_auto_followup()
        await engine.interrupt()
        if session_mode == "local" and local_loop_task and not local_loop_task.done():
            local_loop_task.cancel()
        await send_json({"type": "status", "status": "paused"})

    async def resume_active_session(source: str) -> None:
        nonlocal session_paused, local_loop_task, browser_asr_stream, browser_last_text, browser_stable_fired
        if not engine:
            return
        await send_agent_event(
            "session_resumed",
            detail="Session resumed",
            backend="vokel",
            source=source,
        )
        session_paused = False
        if session_mode == "browser":
            browser_asr_stream = browser_asr.create_stream()
            browser_last_text = ""
            browser_all_samples.clear()
            browser_stable_fired = False
            engine.trace.mark("capture_started")
        elif session_mode == "local":
            if local_loop_task is None or local_loop_task.done():
                local_loop_task = asyncio.create_task(local_run_loop())
        await send_json({"type": "status", "status": "listening"})
        if auto_followup_scheduler:
            auto_followup_scheduler.arm_listening()

    try:
        await send_agent_event("socket_connected", detail="WebSocket accepted", backend="vokel")
        await send_execute_state("idle")
        while True:
            # We can receive either JSON messages or raw binary frames
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            # Handle JSON text configuration
            if "text" in message:
                data = json.loads(message["text"])
                msg_type = data.get("type")

                if msg_type == "start_session":
                    execute_armed = False
                    execute_risk = "none"
                    await send_execute_state("idle")
                    # Clean up any existing session
                    if local_loop_task and not local_loop_task.done():
                        local_loop_task.cancel()
                    await teardown_musical_mode()
                    if engine:
                        await engine.close()
                    if agent_client:
                        await agent_client.__aexit__(None, None, None)

                    session_paused = False
                    agent_backend_name = str(data.get("agent_backend", "builtin")).strip().lower()
                    if agent_backend_name not in ("builtin", "hermes"):
                        agent_backend_name = "builtin"
                    current_agent_backend = agent_backend_name
                    await send_agent_event(
                        "backend_selected",
                        backend=agent_backend_name,
                        detail=f"{agent_backend_name} connection requested",
                    )
                    hermes_mode = agent_backend_name == "hermes"
                    agent_mode: AgentMode = "hermes" if hermes_mode else "builtin"
                    auto_followup_enabled = bool(data.get("auto_followup", True)) and not hermes_mode
                    auto_followup_seconds = clamp_auto_followup_seconds(
                        float(data.get("auto_followup_seconds", DEFAULT_AUTO_FOLLOWUP_SECONDS))
                    )
                    session_mode = data.get("mode", "local")
                    url = data.get("url", LmStudioConfig.url)
                    model = data.get("model", LmStudioConfig.model)
                    # Allow voice camera (explicit armed) for both LM and Hermes.
                    # For Hermes the frame travels under the camera_frame contract with visible consent.
                    vision_voice_enabled = bool(data.get("vision_voice_enabled", False))
                    vision_device = str(data.get("vision_device", "/dev/video0"))
                    vision_lm_url = str(url)
                    voice = str(data.get("voice", "af_heart"))
                    if voice not in KOKORO_VOICES:
                        voice = "af_heart"
                    tts_speed = float(data.get("tts_speed", 1.0))
                    tts_speed = min(1.25, max(0.75, tts_speed))
                    memory_config = MemoryConfig(
                        enabled=bool(data.get("memory", False)) and not hermes_mode,
                        path=Path(str(data.get("memory_db", MemoryConfig.path))),
                        max_results=int(data.get("memory_results", MemoryConfig.max_results)),
                    )
                    active_memory_store = SQLiteMemoryStore(
                        memory_config.path,
                        scan_limit=memory_config.scan_limit,
                    )
                    memory_store = active_memory_store if memory_config.enabled else None

                    # Captured only for the plain OpenAI-compat (Jan) path so we can
                    # preflight its socket before the first turn.
                    local_preflight_config: LmStudioConfig | None = None
                    if hermes_mode:
                        hermes_base = str(
                            data.get("hermes_url", HermesConfig.base_url)
                        ).rstrip("/")
                        hermes_session = str(data.get("hermes_session_id", "")).strip()
                        hermes_config = HermesConfig(
                            base_url=hermes_base,
                            model=str(data.get("hermes_model", HermesConfig.model)),
                            api_key=str(data.get("hermes_api_key", "")),
                            session_id=hermes_session,
                        )
                        agent_client = HermesAgentClient(hermes_config)
                    else:
                        use_native = bool(data.get("use_lm_native_chat") or data.get("lm_native_mcp"))
                        mcp_ids = data.get("lm_mcp_integrations") or data.get("mcp_integrations") or []
                        if isinstance(mcp_ids, str):
                            mcp_ids = [s.strip() for s in mcp_ids.split(",") if s.strip()]
                        if use_native:
                            lm_config = LmStudioConfig(
                                url=url, model=model, use_native_chat=True, mcp_integrations=list(mcp_ids)
                            )
                            agent_client = LmStudioNativeMcpClient(lm_config, integrations=list(mcp_ids))
                        else:
                            from .jan_key import resolve_llm_api_key

                            api_key = resolve_llm_api_key(
                                url,
                                explicit_key=str(data.get("api_key", "")),
                                env_key=LmStudioConfig.api_key,
                            )
                            lm_config = LmStudioConfig(url=url, model=model, api_key=api_key)
                            agent_client = LocalInferenceClient(lm_config)
                            local_preflight_config = lm_config
                    await agent_client.__aenter__()
                    await send_agent_event(
                        "agent_client_ready",
                        backend=agent_backend_name,
                        detail="Agent client context opened",
                    )

                    if hermes_mode and isinstance(agent_client, HermesAgentClient):
                        assert agent_client._client is not None
                        try:
                            await check_gateway_health(hermes_config, agent_client._client)
                            await send_agent_event(
                                "gateway_health_ok",
                                backend="hermes",
                                detail=f"Hermes gateway reachable at {hermes_config.base_url}",
                                session_id=agent_client.session_id,
                            )
                        except InferenceError as exc:
                            await agent_client.__aexit__(None, None, None)
                            agent_client = None
                            await send_agent_event(
                                "gateway_health_failed",
                                backend="hermes",
                                level="error",
                                detail=str(exc),
                            )
                            await send_json({"type": "error", "message": str(exc)})
                            continue

                        try:
                            await check_gateway_inference(hermes_config, agent_client._client)
                            await send_agent_event(
                                "gateway_inference_ok",
                                backend="hermes",
                                detail=f"Hermes model '{hermes_config.model}' is streaming text",
                                session_id=agent_client.session_id,
                            )
                        except InferenceError as exc:
                            await agent_client.__aexit__(None, None, None)
                            agent_client = None
                            await send_agent_event(
                                "gateway_inference_failed",
                                backend="hermes",
                                level="error",
                                detail=str(exc),
                            )
                            await send_json({"type": "error", "message": str(exc)})
                            continue

                    if local_preflight_config is not None and agent_client is not None:
                        try:
                            await check_local_health(local_preflight_config, agent_client._client)
                            await send_agent_event(
                                "local_llm_health_ok",
                                backend=agent_backend_name,
                                detail=f"Local LLM reachable at {local_preflight_config.url}",
                            )
                        except InferenceError as exc:
                            await agent_client.__aexit__(None, None, None)
                            agent_client = None
                            await send_agent_event(
                                "local_llm_health_failed",
                                backend=agent_backend_name,
                                level="error",
                                detail=str(exc),
                            )
                            await send_json({"type": "error", "message": str(exc)})
                            continue

                    trace = LatencyTrace()
                    trace.add_observer(WebSocketTraceObserver(send_json))

                    # Initialize Kokoro sink locally as a helper
                    kokoro_sink: KokoroPlaybackSink | None = None
                    try:
                        kokoro_sink = KokoroPlaybackSink(voice=voice, speed=tts_speed)
                    except Exception as e:
                        logger.warning(f"Kokoro not available for backend: {e}")

                    if session_mode == "local":
                        # Local hardware mode: VAD, microphone, and speakers run on server/host
                        playback_backend = data.get("playback", "kokoro")
                        if playback_backend == "kokoro" and kokoro_sink:
                            playback: PlaybackSink = kokoro_sink
                        elif playback_backend == "spd-say":
                            playback = SpdSayPlaybackSink()
                        else:
                            playback = ConsolePlaybackSink()

                        musical_mode = bool(data.get("musical_mode", False))
                        try:
                            musical_bpm = min(
                                160.0, max(60.0, float(data.get("musical_bpm", 90.0)))
                            )
                        except (TypeError, ValueError):
                            musical_bpm = 90.0
                        try:
                            musical_level = min(
                                1.0, max(0.0, float(data.get("musical_level", 1.0)))
                            )
                        except (TypeError, ValueError):
                            musical_level = 1.0
                        musical_style = _parse_musical_style(data.get("musical_style"))
                        voice_loop_config: VoiceLoopConfig | None = None
                        if (
                            musical_mode
                            and playback_backend in ("kokoro", "spd-say")
                        ):
                            musical_clock = BeatClock(bpm=musical_bpm)
                            beat_track = BeatTrackPlayer(
                                bpm=musical_bpm,
                                level=musical_level,
                                style=musical_style,
                            )
                            track_slot = _musical_track_slots.get(musical_track_slot)
                            if track_slot is not None:
                                track_slot.player = beat_track
                                if track_slot.buffer is not None:
                                    beat_track.load_buffer(track_slot.buffer)
                            await musical_clock.start()
                            await beat_track.start()
                            playback = QuantizedPlaybackSink(
                                playback,
                                musical_clock,
                                quantum="beat",
                            )
                            base = VoiceLoopConfig()
                            voice_loop_config = VoiceLoopConfig(
                                system_prompt=base.system_prompt + (
                                    f" You are currently performing over a live beat at "
                                    f"{musical_bpm:.0f} BPM. "
                                    "Deliver every reply as short rhythmic rap lines with rhyme "
                                    "and flow, a few words per line, so each phrase lands on the "
                                    "beat. You are the rapper — never say you are playing, "
                                    "finding, queueing, or displaying a track or lyrics; speak "
                                    "the bars yourself."
                                )
                            )
                            beat_clock = musical_clock

                            async def forward_beats() -> None:
                                try:
                                    while True:
                                        info = await beat_clock.wait_for_beat()
                                        await send_json({
                                            "type": "beat",
                                            "bar": info.bar,
                                            "beat": info.beat,
                                            "bpm": musical_bpm,
                                        })
                                except ClockStopped:
                                    pass
                                except Exception as exc:
                                    logger.debug("beat forwarder exited: %s", exc)

                            beat_forward_task = asyncio.create_task(forward_beats())

                        # Configure offline ASR and VAD
                        vad_model_path = data.get("vad_model", "models/silero_vad.onnx")
                        asr_tokens = data.get(
                            "asr_tokens",
                            "models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/tokens.txt",
                        )
                        sense_voice_model = data.get(
                            "sense_voice_model",
                            "models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx",
                        )

                        producer = SileroVadTurnProducer(MicVadConfig(vad_model_path=vad_model_path))
                        asr = SherpaOfflineAsr(
                            SherpaOfflineAsrConfig(
                                tokens=asr_tokens,
                                sense_voice_model=sense_voice_model,
                            )
                        )

                        local_engine_kwargs: dict[str, Any] = {
                            "agent": agent_client,
                            "playback": playback,
                            "trace": trace,
                            "echo_tokens": False,
                            "memory_store": memory_store,
                            "memory_config": memory_config,
                            "agent_mode": agent_mode,
                            "visual_context_provider": provide_voice_visual_context,
                        }
                        if voice_loop_config is not None:
                            local_engine_kwargs["config"] = voice_loop_config
                        engine = ConversationEngine(**local_engine_kwargs)
                        await engine.start()
                        attach_auto_followup_scheduler()

                        local_engine = engine

                        # Run local loop in the background
                        async def local_run_loop() -> None:
                            try:
                                while True:
                                    if session_paused:
                                        await asyncio.sleep(0.1)
                                        continue
                                    await send_json({"type": "status", "status": "listening"})
                                    if auto_followup_scheduler:
                                        auto_followup_scheduler.arm_listening()
                                    await local_engine.run_turns(producer, asr, max_turns=1)
                                    if auto_followup_scheduler:
                                        auto_followup_scheduler.on_user_activity()
                                    # Push latest summary metrics
                                    await send_json({
                                        "type": "summary",
                                        "metrics": local_engine.trace.summary_ms(),
                                    })
                                    await asyncio.sleep(0.1)
                            except asyncio.CancelledError:
                                pass
                            except Exception as e:
                                logger.error(f"Local run loop failed: {e}")
                                await send_json({"type": "error", "message": str(e)})

                        local_loop_task = asyncio.create_task(local_run_loop())
                        session_payload: dict[str, Any] = {
                            "type": "session_started",
                            "mode": "local",
                            "voice": voice,
                            "agent_backend": agent_backend_name,
                        }
                        if isinstance(agent_client, HermesAgentClient):
                            session_payload["hermes_session_id"] = agent_client.session_id
                        await send_json(session_payload)
                        await send_agent_event(
                            "session_started",
                            backend=agent_backend_name,
                            detail="Local hardware session started",
                            mode="local",
                            session_id=(
                                agent_client.session_id
                                if isinstance(agent_client, HermesAgentClient)
                                else None
                            ),
                        )
                        await send_memory_facts(active_memory_store)

                    elif session_mode == "browser":
                        # Browser mode: WebSocket serves as microphone & speaker
                        web_playback = WebSocketPlaybackSink(send_json, send_bytes, kokoro_sink)
                        engine = ConversationEngine(
                            agent=agent_client,
                            playback=web_playback,
                            trace=trace,
                            echo_tokens=False,
                            memory_store=memory_store,
                            memory_config=memory_config,
                            agent_mode=agent_mode,
                            visual_context_provider=provide_voice_visual_context,
                        )
                        await engine.start()
                        attach_auto_followup_scheduler()

                        # Create the online ASR engine for browser stream
                        streaming_asr_dir = data.get(
                            "streaming_asr_dir",
                            "models/sherpa-onnx-streaming-zipformer-en-2023-06-26",
                        )
                        browser_asr = create_streaming_asr(streaming_asr_dir)
                        browser_asr_stream = browser_asr.create_stream()

                        # We'll maintain state in the websocket handler for this stream
                        browser_all_samples: list[float] = []
                        browser_last_text = ""
                        browser_last_changed_time = asyncio.get_running_loop().time()
                        browser_stable_fired = False

                        engine.trace.mark("capture_started")

                        browser_session_payload: dict[str, Any] = {
                            "type": "session_started",
                            "mode": "browser",
                            "voice": voice,
                            "agent_backend": agent_backend_name,
                        }
                        if isinstance(agent_client, HermesAgentClient):
                            browser_session_payload["hermes_session_id"] = agent_client.session_id
                        await send_json(browser_session_payload)
                        await send_agent_event(
                            "session_started",
                            backend=agent_backend_name,
                            detail="Browser audio session started",
                            mode="browser",
                            session_id=(
                                agent_client.session_id
                                if isinstance(agent_client, HermesAgentClient)
                                else None
                            ),
                        )
                        await send_json({"type": "status", "status": "listening"})
                        if auto_followup_scheduler:
                            auto_followup_scheduler.arm_listening()
                        await send_memory_facts(active_memory_store)

                    else:
                        await send_json({
                            "type": "error",
                            "message": f"Unsupported audio route: {session_mode}",
                        })

                elif msg_type == "preview_voice":
                    voice = str(data.get("voice", "af_heart"))
                    if voice not in KOKORO_VOICES:
                        await send_json({
                            "type": "error",
                            "message": f"Unknown Kokoro voice: {voice}",
                        })
                        continue
                    tts_speed = float(data.get("tts_speed", 1.0))
                    tts_speed = min(1.25, max(0.75, tts_speed))
                    sample_text = str(
                        data.get(
                            "text",
                            "Hello, I am Vokel. Interrupt me anytime and I will stop talking.",
                        )
                    )
                    await send_json({"type": "voice_preview_started", "voice": voice})
                    try:
                        preview_sink = KokoroPlaybackSink(voice=voice, speed=tts_speed)
                        async for samples, sample_rate in preview_sink.kokoro.create_stream(
                            sanitize_for_speech(sample_text),
                            preview_sink.voice,
                            preview_sink.speed,
                            "en-us",
                        ):
                            samples_f32 = np.asarray(samples, dtype=np.float32)
                            await send_bytes(samples_f32.tobytes())
                        await send_json({"type": "voice_preview_finished", "voice": voice})
                    except Exception as e:
                        logger.warning(f"Voice preview failed: {e}")
                        await send_json({
                            "type": "voice_preview_finished",
                            "voice": voice,
                            "error": str(e),
                        })

                elif msg_type == "set_vision_context":
                    requested_device = str(data.get("device", vision_device))
                    requested_enabled = bool(data.get("enabled", False))
                    available_devices = {camera.path for camera in list_camera_devices()}
                    # Camera context now supported for Hermes too (external frame route with consent).
                    if requested_enabled and current_agent_backend not in ("builtin", "hermes"):
                        await send_json({
                            "type": "error",
                            "message": "Voice camera context requires a supported agent backend (LM Studio or Hermes).",
                        })
                        continue
                    if requested_enabled and requested_device not in available_devices:
                        await send_json({
                            "type": "error",
                            "message": f"Voice camera is not available: {requested_device}",
                        })
                        continue
                    vision_device = requested_device
                    vision_voice_enabled = requested_enabled
                    await send_agent_event(
                        "visual_context_setting_changed",
                        backend="vokel",
                        detail=(
                            f"Voice camera context {'enabled' if vision_voice_enabled else 'disabled'}"
                        ),
                        device=vision_device,
                        enabled=vision_voice_enabled,
                    )

                elif msg_type == "camera_frame":
                    # Reply to a request_camera_frame: the browser sends a preview frame
                    # (or null when no preview is live). Hand it to the waiting capture.
                    if browser_frame_future is not None and not browser_frame_future.done():
                        browser_frame_future.set_result(data.get("image_data_url"))

                elif msg_type == "stop_session":
                    await send_agent_event("session_stop_requested", detail="Stop requested", backend="vokel")
                    cancel_auto_followup()
                    if local_loop_task and not local_loop_task.done():
                        local_loop_task.cancel()
                    await teardown_musical_mode()
                    if engine:
                        await engine.close()
                    if agent_client:
                        await agent_client.__aexit__(None, None, None)
                    engine = None
                    agent_client = None
                    session_mode = None
                    session_paused = False
                    execute_armed = False
                    execute_risk = "none"
                    await send_execute_state("idle")
                    await send_json({"type": "session_stopped"})
                    await send_agent_event("session_stopped", detail="Session stopped", backend="vokel")

                elif msg_type == "set_musical_level":
                    if beat_track is not None:
                        try:
                            level = min(1.0, max(0.0, float(data.get("level", 1.0))))
                            beat_track.set_level(level)
                        except (TypeError, ValueError):
                            pass

                elif msg_type == "set_musical_nudge":
                    if beat_track is not None:
                        try:
                            beat_track.nudge_cursor(float(data.get("ms", 0)))
                        except (TypeError, ValueError):
                            pass

                elif msg_type == "interrupt":
                    if engine:
                        await send_agent_event("interrupt_requested", detail="Barge-in requested", backend="vokel")
                        cancel_auto_followup()
                        await engine.interrupt()
                        await send_json({"type": "status", "status": "listening"})
                        if auto_followup_scheduler:
                            auto_followup_scheduler.arm_listening()

                elif msg_type == "pause_session":
                    if engine:
                        await pause_active_session("ui_control")

                elif msg_type == "resume_session":
                    if engine:
                        await resume_active_session("ui_control")

                elif msg_type == "reset_session":
                    if engine:
                        await send_agent_event("session_reset_requested", detail="Reset requested", backend="vokel")
                        cancel_auto_followup()
                        await engine.reset_conversation()
                        execute_armed = False
                        execute_risk = "none"
                        await send_execute_state("idle")
                        if session_mode == "browser":
                            browser_asr_stream = browser_asr.create_stream()
                            browser_last_text = ""
                            browser_all_samples.clear()
                            browser_stable_fired = False
                            if not session_paused:
                                engine.trace.mark("capture_started")
                        await send_json({"type": "session_reset"})
                        await send_json({
                            "type": "status",
                            "status": "paused" if session_paused else "listening",
                        })

                elif msg_type == "memory_list":
                    await send_memory_facts(active_memory_store)

                elif msg_type == "memory_save":
                    if active_memory_store:
                        await active_memory_store.record_fact(str(data.get("text", "")))
                    await send_memory_facts(active_memory_store)

                elif msg_type == "memory_update":
                    if active_memory_store:
                        await active_memory_store.update_fact(
                            int(data.get("id", 0)),
                            str(data.get("text", "")),
                        )
                    await send_memory_facts(active_memory_store)

                elif msg_type == "memory_delete":
                    if active_memory_store:
                        await active_memory_store.delete_fact(int(data.get("id", 0)))
                    await send_memory_facts(active_memory_store)

                elif msg_type == "arm_execute":
                    execute_armed = True
                    execute_risk = str(data.get("risk", "medium"))
                    await send_execute_state("execute armed")
                    await send_agent_event(
                        "execute_armed",
                        backend="vokel",
                        detail="Execution consent armed",
                        risk=execute_risk,
                    )

                elif msg_type == "cancel_execute":
                    execute_armed = False
                    execute_risk = "none"
                    await send_execute_state("idle")
                    await send_agent_event(
                        "execute_cancelled",
                        backend="vokel",
                        detail="Execution consent cancelled",
                    )

                elif msg_type == "confirm_execute":
                    await send_agent_event(
                        "execute_confirm_ignored",
                        backend="vokel",
                        level="warning",
                        detail="No executable action is registered yet",
                        armed=execute_armed,
                        risk=execute_risk,
                    )
                    await send_execute_state("no executable action pending")

                elif msg_type == "probe_hermes":
                    probe_url = str(data.get("hermes_url", HermesConfig.base_url)).rstrip("/")
                    probe_key = str(data.get("hermes_api_key", ""))
                    probe_config = HermesConfig(base_url=probe_url, api_key=probe_key)
                    import httpx
                    try:
                        async with httpx.AsyncClient(timeout=5.0) as probe_client:
                            await check_gateway_health(probe_config, probe_client)
                        await send_agent_event(
                            "gateway_health_ok",
                            backend="hermes",
                            detail=f"Hermes gateway reachable at {probe_url}",
                        )
                    except InferenceError as exc:
                        await send_agent_event(
                            "gateway_health_failed",
                            backend="hermes",
                            level="error",
                            detail=str(exc),
                        )

            # Handle binary audio packets in browser streaming mode
            elif "bytes" in message and session_mode == "browser" and engine:
                audio_data = message["bytes"]
                # Convert raw bytes (Float32 PCM) to a numpy float32 array
                samples = np.frombuffer(audio_data, dtype=np.float32)

                # Check for interruption (barge-in)
                # If assistant is currently speaking or generating, and we receive audio speech,
                # we should interrupt instantly.
                is_active = (
                    engine._current_generation and not engine._current_generation.done()
                ) or not engine._playback_queue.empty()

                frame_rms = (
                    float(np.sqrt(np.mean(samples * samples)))
                    if samples.size
                    else 0.0
                )
                speech_threshold = 0.018

                # Feed to ASR stream
                browser_asr_stream.accept_waveform(16000, samples)
                browser_asr_stream.decode()
                text = browser_asr_stream.get_result()

                current_time = asyncio.get_running_loop().time()

                if frame_rms >= speech_threshold and auto_followup_scheduler and not session_paused:
                    auto_followup_scheduler.on_user_activity()

                if text != browser_last_text:
                    # Barge-in only on real microphone energy (avoids hallucinated partials stopping playback).
                    if is_active and frame_rms >= speech_threshold:
                        logger.info(
                            "Barge-in: mic activity while assistant active "
                            f"(rms={frame_rms:.4f}, partial={text!r})"
                        )
                        await engine.interrupt()
                        browser_asr_stream = browser_asr.create_stream()
                        browser_last_text = ""
                        browser_all_samples.clear()
                        browser_stable_fired = False
                        await send_json({"type": "status", "status": "listening"})
                        continue

                    browser_last_text = text
                    browser_last_changed_time = current_time
                    browser_stable_fired = False
                    if not session_paused:
                        engine.trace.mark("partial_transcript", text=text)
                        await send_json({"type": "partial_transcript", "text": text})

                elif text and not browser_stable_fired:
                    if current_time - browser_last_changed_time >= 0.6:
                        if not session_paused:
                            engine.trace.mark("stable_transcript", text=text)
                        browser_stable_fired = True
                        if not session_paused:
                            await send_json({"type": "stable_transcript", "text": text})

                browser_all_samples.extend(samples.tolist())

                if browser_asr_stream.is_endpoint():
                    if browser_last_text and not browser_stable_fired and not session_paused:
                        engine.trace.mark("stable_transcript", text=browser_last_text)

                    utterance = browser_last_text.strip()
                    # Ignore empty or noise-only endpoints so we do not spam the LLM.
                    if len(utterance) >= 2:
                        async with turn_lock:
                            await send_json({"type": "final_transcript", "text": utterance})
                            voice_command = detect_voice_session_command(utterance)
                            if voice_command == "pause":
                                if not session_paused:
                                    await pause_active_session("voice_command")
                                else:
                                    await send_json({"type": "status", "status": "paused"})
                            elif voice_command == "resume":
                                if session_paused:
                                    await resume_active_session("voice_command")
                                else:
                                    await send_json({"type": "status", "status": "listening"})
                            elif not session_paused:
                                if auto_followup_scheduler:
                                    auto_followup_scheduler.on_user_activity()
                                engine.trace.mark(
                                    "capture_finished",
                                    has_audio=True,
                                    audio_seconds=float(len(browser_all_samples)) / 16000.0,
                                    samples=len(browser_all_samples),
                                )
                                engine.trace.mark(
                                    "asr_finished", chars=len(utterance), text=utterance
                                )
                                await send_json({"type": "status", "status": "generating"})
                                await send_agent_event(
                                    "turn_submitted",
                                    backend=agent_backend_name if "agent_backend_name" in locals() else None,
                                    detail="Transcript sent to active agent",
                                    chars=len(utterance),
                                )

                                await engine.submit_turn(utterance, reset_trace=False)
                                await engine.wait_for_playback()

                                await send_json({
                                    "type": "summary",
                                    "metrics": engine.trace.summary_ms(),
                                })
                            else:
                                await send_agent_event(
                                    "paused_input_ignored",
                                    backend="vokel",
                                    detail="Input ignored while paused. Say 'continue' or 'resume' to continue.",
                                    level="info",
                                )

                    # Reset recognizer after every endpoint (including ignored silence).
                    browser_asr_stream = browser_asr.create_stream()
                    browser_last_text = ""
                    browser_all_samples.clear()
                    browser_stable_fired = False
                    if session_paused:
                        await send_json({"type": "status", "status": "paused"})
                    else:
                        await send_json({"type": "status", "status": "listening"})
                        engine.trace.mark("capture_started")
                        if auto_followup_scheduler:
                            auto_followup_scheduler.arm_listening()

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}\n{traceback.format_exc()}")
        try:
            await send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        # Guarantee session resources are cleaned up
        cancel_auto_followup()
        if local_loop_task and not local_loop_task.done():
            local_loop_task.cancel()
        await teardown_musical_mode()
        _musical_track_slots.pop(musical_track_slot, None)
        if engine:
            await engine.close()
        if agent_client:
            await agent_client.__aexit__(None, None, None)


@app.get("/favicon.ico", include_in_schema=False)
def favicon_ico() -> RedirectResponse:
    return RedirectResponse(url="/favicon.svg", status_code=307)


@app.get("/favicon.svg", include_in_schema=False)
def favicon_svg() -> FileResponse:
    if not _FAVICON_PATH.is_file():
        raise HTTPException(status_code=404, detail="favicon not found; run: make build")
    return FileResponse(
        _FAVICON_PATH,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=604800, immutable"},
    )


# Mount static files to serve the built frontend
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="static")
else:
    @app.get("/")
    def read_root() -> dict[str, str]:
        return {
            "message": "Vokel Backend API running. Please build the frontend React app in /frontend to view the UI."
        }
