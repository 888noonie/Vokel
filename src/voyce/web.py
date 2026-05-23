from __future__ import annotations

import asyncio
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Callable, Coroutine

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .audio import (
    MicVadConfig,
    SherpaOfflineAsr,
    SherpaOfflineAsrConfig,
    SileroVadTurnProducer,
    create_streaming_asr,
)
from .config import LmStudioConfig
from .engine import ConversationEngine
from .web_search import create_default_registry
from .inference import LocalInferenceClient
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voyce.web")

app = FastAPI(title="Voyce Web Server")

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


@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("WebSocket connection established")

    # Serialize browser-mode submit/playback so endpoint bursts cannot double-fire the LLM.
    browser_turn_lock = asyncio.Lock()

    # Keep track of active tasks and engines to clean them up on disconnect
    session_mode: str | None = None
    engine: ConversationEngine | None = None
    llm_client: LocalInferenceClient | None = None
    local_loop_task: asyncio.Task[None] | None = None
    session_paused = False
    active_memory_store: SQLiteMemoryStore | None = None

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

    try:
        while True:
            # We can receive either JSON messages or raw binary frames
            message = await websocket.receive()

            # Handle JSON text configuration
            if "text" in message:
                data = json.loads(message["text"])
                msg_type = data.get("type")

                if msg_type == "start_session":
                    # Clean up any existing session
                    if local_loop_task and not local_loop_task.done():
                        local_loop_task.cancel()
                    if engine:
                        await engine.close()
                    if llm_client:
                        await llm_client.__aexit__(None, None, None)

                    session_paused = False
                    session_mode = data.get("mode", "local")
                    url = data.get("url", LmStudioConfig.url)
                    model = data.get("model", LmStudioConfig.model)
                    voice = str(data.get("voice", "af_heart"))
                    if voice not in KOKORO_VOICES:
                        voice = "af_heart"
                    tts_speed = float(data.get("tts_speed", 1.0))
                    tts_speed = min(1.25, max(0.75, tts_speed))
                    memory_config = MemoryConfig(
                        enabled=bool(data.get("memory", False)),
                        path=Path(str(data.get("memory_db", MemoryConfig.path))),
                        max_results=int(data.get("memory_results", MemoryConfig.max_results)),
                    )
                    active_memory_store = SQLiteMemoryStore(
                        memory_config.path,
                        scan_limit=memory_config.scan_limit,
                    )
                    memory_store = active_memory_store if memory_config.enabled else None

                    lm_config = LmStudioConfig(url=url, model=model)
                    llm_client = LocalInferenceClient(lm_config)
                    await llm_client.__aenter__()

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

                        engine = ConversationEngine(
                            llm=llm_client,
                            playback=playback,
                            trace=trace,
                            echo_tokens=False,
                            memory_store=memory_store,
                            memory_config=memory_config,
                            tool_registry=create_default_registry(),
                        )
                        await engine.start()

                        local_engine = engine

                        # Run local loop in the background
                        async def local_run_loop() -> None:
                            try:
                                while True:
                                    if session_paused:
                                        await asyncio.sleep(0.1)
                                        continue
                                    await send_json({"type": "status", "status": "listening"})
                                    await local_engine.run_turns(producer, asr, max_turns=1)
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
                        await send_json({
                            "type": "session_started",
                            "mode": "local",
                            "voice": voice,
                        })
                        await send_memory_facts(active_memory_store)

                    elif session_mode == "browser":
                        # Browser mode: WebSocket serves as microphone & speaker
                        web_playback = WebSocketPlaybackSink(send_json, send_bytes, kokoro_sink)
                        engine = ConversationEngine(
                            llm=llm_client,
                            playback=web_playback,
                            trace=trace,
                            echo_tokens=False,
                            memory_store=memory_store,
                            memory_config=memory_config,
                            tool_registry=create_default_registry(),
                        )
                        await engine.start()

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

                        await send_json({
                            "type": "session_started",
                            "mode": "browser",
                            "voice": voice,
                        })
                        await send_json({"type": "status", "status": "listening"})
                        await send_memory_facts(active_memory_store)

                    else:
                        await send_json({
                            "type": "error",
                            "message": f"Unsupported mode: {session_mode}",
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
                            "Hello, I am Voyce. Interrupt me anytime and I will stop listening.",
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

                elif msg_type == "stop_session":
                    if local_loop_task and not local_loop_task.done():
                        local_loop_task.cancel()
                    if engine:
                        await engine.close()
                    if llm_client:
                        await llm_client.__aexit__(None, None, None)
                    engine = None
                    llm_client = None
                    session_mode = None
                    session_paused = False
                    await send_json({"type": "session_stopped"})

                elif msg_type == "interrupt":
                    if engine:
                        await engine.interrupt()
                        await send_json({"type": "status", "status": "listening"})

                elif msg_type == "pause_session":
                    if engine:
                        session_paused = True
                        await engine.interrupt()
                        if session_mode == "local" and local_loop_task and not local_loop_task.done():
                            local_loop_task.cancel()
                        await send_json({"type": "status", "status": "paused"})

                elif msg_type == "resume_session":
                    if engine:
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

                elif msg_type == "reset_session":
                    if engine:
                        await engine.reset_conversation()
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

            # Handle binary audio packets in browser streaming mode
            elif "bytes" in message and session_mode == "browser" and engine:
                if session_paused:
                    continue
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
                    engine.trace.mark("partial_transcript", text=text)
                    await send_json({"type": "partial_transcript", "text": text})

                elif text and not browser_stable_fired:
                    if current_time - browser_last_changed_time >= 0.6:
                        engine.trace.mark("stable_transcript", text=text)
                        browser_stable_fired = True
                        await send_json({"type": "stable_transcript", "text": text})

                browser_all_samples.extend(samples.tolist())

                if browser_asr_stream.is_endpoint():
                    if browser_last_text and not browser_stable_fired:
                        engine.trace.mark("stable_transcript", text=browser_last_text)

                    utterance = browser_last_text.strip()
                    # Ignore empty or noise-only endpoints so we do not spam the LLM.
                    if len(utterance) >= 2:
                        async with browser_turn_lock:
                            engine.trace.mark(
                                "capture_finished",
                                has_audio=True,
                                audio_seconds=float(len(browser_all_samples)) / 16000.0,
                                samples=len(browser_all_samples),
                            )
                            engine.trace.mark(
                                "asr_finished", chars=len(utterance), text=utterance
                            )

                            await send_json({"type": "final_transcript", "text": utterance})
                            await send_json({"type": "status", "status": "generating"})

                            await engine.submit_turn(utterance, reset_trace=False)
                            await engine.wait_for_playback()

                            await send_json({
                                "type": "summary",
                                "metrics": engine.trace.summary_ms(),
                            })

                    # Reset recognizer after every endpoint (including ignored silence).
                    browser_asr_stream = browser_asr.create_stream()
                    browser_last_text = ""
                    browser_all_samples.clear()
                    browser_stable_fired = False
                    await send_json({"type": "status", "status": "listening"})
                    engine.trace.mark("capture_started")

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
        if local_loop_task and not local_loop_task.done():
            local_loop_task.cancel()
        if engine:
            await engine.close()
        if llm_client:
            await llm_client.__aexit__(None, None, None)


# Mount static files to serve the built frontend
frontend_dist = Path("frontend/dist")
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")
else:
    @app.get("/")
    def read_root() -> dict[str, str]:
        return {
            "message": "Voyce Backend API running. Please build the frontend React app in /frontend to view the UI."
        }
