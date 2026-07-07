from __future__ import annotations

import asyncio
import base64
import json
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


DEFAULT_VISION_PROMPT = "Briefly describe what is visible in this live webcam frame."

# Instruction sent with the frame when a bare "shoot" command (no spoken question)
# captures a still — keeps a 4B vision model focused instead of guessing at "shoot".
CAMERA_VISION_PROMPT = "Describe exactly what you see in this camera frame in one short, natural sentence."

# What Vokel says when the user signals they want to be seen ("watch me") so the
# control vocabulary is taught out loud rather than left for the model to invent.
CAMERA_GUIDANCE_OFFER = (
    "I've got my eye on the camera. Just say shoot and I'll take the picture, "
    "say action to start a live video loop, and cut to stop."
)


@dataclass(frozen=True)
class CameraDevice:
    path: str
    name: str


@dataclass(frozen=True)
class VisionFrameAnalysis:
    device: str
    description: str
    image_data_url: str
    capture_seconds: float
    inference_seconds: float


@dataclass(frozen=True)
class CapturedVisionFrame:
    device: str
    image_data_url: str
    capture_seconds: float


@dataclass(frozen=True)
class VisualContext:
    """Narrow payload for an explicitly user-armed camera (or future CameraX) frame.

    Carries everything needed for external agent routing (Hermes camera_frame contract)
    or native LM vision. Source is the device id or platform identifier (e.g. "/dev/video4",
    "android:camera:0"). Never used for ambient capture.
    """

    data_url: str
    source: str
    captured_at: str | None = None  # ISO-8601 UTC
    consent: str = "explicit_visual_context_for_turn"
    contract: str = "hermes_camera_frame_v1"
    # Optional instruction the vision model should answer about this frame. When the
    # user gave a bare command ("shoot"), the spoken words make a poor prompt, so the
    # host substitutes CAMERA_VISION_PROMPT here while the transcript keeps what was said.
    prompt: str | None = None


@dataclass(frozen=True)
class SpokenReply:
    """A deterministic line for Vokel to say without invoking the model.

    Used for camera control acknowledgements ("watch me", "action", "cut") so the
    voice loop stays in human control and the model cannot hallucinate camera modes.
    """

    text: str


_VISUAL_CONTEXT_SPACE_RE = re.compile(r"\s+")
_VISUAL_CONTEXT_PUNCT_RE = re.compile(r"[^a-z0-9\s']")
_CAMERA_FILLER_PREFIX_RE = re.compile(r"^(?:ok(?:ay)?|alright|hey|so|now|please|just|and|um|uh)\s+")

CameraCommand = Literal["shoot", "watch", "action", "cut"]


def _normalize_camera_text(user_text: str) -> str:
    normalized = _VISUAL_CONTEXT_SPACE_RE.sub(
        " ",
        _VISUAL_CONTEXT_PUNCT_RE.sub(" ", user_text.lower()),
    ).strip()
    while True:
        stripped = _CAMERA_FILLER_PREFIX_RE.sub("", normalized).strip()
        if stripped == normalized:
            return normalized
        normalized = stripped


def should_capture_visual_context(user_text: str) -> bool:
    normalized = _normalize_camera_text(user_text)
    visual_phrases = (
        "what am i holding",
        "what i'm holding",
        "what is in my hand",
        "what's in my hand",
        "what am i showing you",
        "what can you see",
        "what do you see",
        "can you see this",
        "can you see what",
        "describe what you see",
        "look at this",
        "take a look",
    )
    return any(phrase in normalized for phrase in visual_phrases)


# Single-word triggers must equal the whole utterance so "I cut my finger" or
# "lights, camera, action!" said mid-sentence don't fire the camera by accident.
_CAMERA_EXACT_COMMANDS: dict[str, CameraCommand] = {
    "shoot": "shoot",
    "snap": "shoot",
    "click": "shoot",
    "action": "action",
    "roll": "action",
    "cut": "cut",
    "stop watching": "cut",
    "stop the watch": "cut",
    "thats a wrap": "cut",
    "that's a wrap": "cut",
    "watch me": "watch",
    "look at me": "watch",
    "watch this": "watch",
}

# Unambiguous capture phrases: no benign sentence says "take a picture" without
# meaning it, so these fire regardless of how much politeness padding surrounds
# them ("I can see how that confused you, please take a picture").
_CAMERA_STRONG_PHRASES: tuple[tuple[str, CameraCommand], ...] = (
    ("take a picture", "shoot"),
    ("take the picture", "shoot"),
    ("take another picture", "shoot"),
    ("take a photo", "shoot"),
    ("take the photo", "shoot"),
    ("take another photo", "shoot"),
    ("take a pic", "shoot"),
    ("take a shot", "shoot"),
    ("take the shot", "shoot"),
    ("snap a photo", "shoot"),
    ("snap a pic", "shoot"),
    ("snap a picture", "shoot"),
    ("get a picture", "shoot"),
)

# Distinctive but more context-dependent phrases may appear only inside a short
# utterance, so a passing mention ("we should go live on the radio") won't fire.
_CAMERA_PHRASE_COMMANDS: tuple[tuple[str, CameraCommand], ...] = (
    ("grab a frame", "shoot"),
    ("capture this", "shoot"),
    ("capture that", "shoot"),
    ("start the live loop", "action"),
    ("start a live loop", "action"),
    ("start live loop", "action"),
    ("start watching", "action"),
    ("go live", "action"),
    ("roll camera", "action"),
    ("stop the live loop", "cut"),
    ("stop live loop", "cut"),
    ("stop the camera", "cut"),
    ("watch me", "watch"),
    ("watch what i'm doing", "watch"),
    ("let me show you", "watch"),
    ("i want to show you", "watch"),
    ("i'll show you something", "watch"),
)


def detect_camera_command(user_text: str, *, max_words: int = 7) -> CameraCommand | None:
    """Deterministically map a short spoken utterance to a camera control command.

    Returns ``shoot`` (take one still), ``action`` (start a live video loop),
    ``cut`` (stop the live loop), or ``watch`` (user wants to be seen — Vokel offers
    the control words). Returns ``None`` for anything that isn't a clear command so the
    normal conversational path is never hijacked.
    """

    normalized = _normalize_camera_text(user_text)
    if not normalized:
        return None

    exact = _CAMERA_EXACT_COMMANDS.get(normalized)
    if exact is not None:
        return exact

    for phrase, command in _CAMERA_STRONG_PHRASES:
        if phrase in normalized:
            return command

    if len(normalized.split()) > max_words:
        return None

    for phrase, command in _CAMERA_PHRASE_COMMANDS:
        if phrase in normalized:
            return command
    return None


# Connective filler that can wrap a capture command without adding a real question.
_CAPTURE_FILLER_RE = re.compile(
    r"\b(?:and|then|please|now|for me|ok|okay|so|just|go on|go ahead|"
    r"can you|could you|would you|will you|i want you to|i'd like you to)\b"
)
_SHOOT_WORDS = tuple(word for word, command in _CAMERA_EXACT_COMMANDS.items() if command == "shoot")


def capture_prompt_for(user_text: str, *, bare_prompt: str) -> str:
    """Choose the instruction to send the vision model for a capture turn.

    A bare trigger ("shoot", "take a picture") carries no question, so the generic
    ``bare_prompt`` keeps a small model focused. But when the user attaches a real
    question ("take a picture and tell me what's on my head"), that question must be
    answered, so the full utterance is sent instead.
    """
    residual = _normalize_camera_text(user_text)
    for phrase, _ in (*_CAMERA_STRONG_PHRASES, *_CAMERA_PHRASE_COMMANDS):
        residual = residual.replace(phrase, " ")
    for word in _SHOOT_WORDS:
        residual = re.sub(rf"\b{re.escape(word)}\b", " ", residual)
    residual = _CAPTURE_FILLER_RE.sub(" ", residual)
    residual = re.sub(r"\s+", " ", residual).strip(" .,!?")
    if len(residual.split()) >= 2:
        return user_text.strip()
    return bare_prompt


_CAPTURE_PROBE_CACHE: dict[str, bool] = {}


def device_supports_capture(device: str) -> bool:
    """Return True when GStreamer can pull at least one frame from the V4L2 node."""

    cached = _CAPTURE_PROBE_CACHE.get(device)
    if cached is not None:
        return cached

    try:
        result = subprocess.run(
            [
                "gst-launch-1.0",
                "-q",
                "v4l2src",
                f"device={device}",
                "num-buffers=1",
                "!",
                "fakesink",
            ],
            capture_output=True,
            timeout=4,
            check=False,
        )
        ok = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        ok = False

    _CAPTURE_PROBE_CACHE[device] = ok
    return ok


def _friendly_camera_label(raw_name: str, path: str) -> str:
    lower = raw_name.lower()
    if path == "/dev/video0" and "integrated" in lower:
        return "Built-in webcam (color)"
    if "integrated" in lower and raw_name.endswith("I"):
        return "Built-in webcam (IR)"
    return raw_name


def list_camera_devices() -> list[CameraDevice]:
    devices: list[CameraDevice] = []
    for node in sorted(Path("/sys/class/video4linux").glob("video*")):
        path = f"/dev/{node.name}"
        if not device_supports_capture(path):
            continue
        raw_name = (node / "name").read_text().strip()
        devices.append(CameraDevice(path=path, name=_friendly_camera_label(raw_name, path)))
    return devices


def pick_default_camera(cameras: list[CameraDevice]) -> str:
    """Pick a sensible default capture camera, avoiding IR nodes when a color one exists.

    IR webcams ("(IR)") only emit a grayscale infrared image, which is useless for
    "what's on my head"-style questions, so they are chosen only as a last resort.
    """

    if not cameras:
        return ""
    color_cameras = [camera for camera in cameras if "(ir)" not in camera.name.lower()]
    preferred = color_cameras or cameras
    for camera in preferred:
        if camera.path == "/dev/video0":
            return camera.path
    for camera in preferred:
        if "integrated" in camera.name.lower():
            return camera.path
    return preferred[0].path


def is_loopback_url(url: str) -> bool:
    hostname = urllib.parse.urlparse(url).hostname
    return hostname in {"127.0.0.1", "::1", "localhost"}


def build_capture_command(
    *,
    device: str,
    output_pattern: Path,
    width: int,
    height: int,
    framerate: int,
    warmup_frames: int,
) -> list[str]:
    # Negotiate whatever the camera natively offers — YUY2, MJPG, or the GREY/Y8 an
    # IR webcam emits — instead of forcing YUY2 (which made IR and MJPG-only cameras
    # fail with "not-negotiated"). decodebin auto-plugs a decoder when needed and
    # passes raw video straight through; videoscale then fits the vision model size.
    del framerate  # native framerate is fine for a single still
    return [
        "gst-launch-1.0",
        "-q",
        "v4l2src",
        f"device={device}",
        f"num-buffers={warmup_frames}",
        "!",
        "decodebin",
        "!",
        "videoconvert",
        "!",
        "videoscale",
        "!",
        f"video/x-raw,width={width},height={height}",
        "!",
        "jpegenc",
        "!",
        "multifilesink",
        f"location={output_pattern}",
    ]


def capture_frame(
    *,
    device: str,
    output_dir: Path,
    width: int,
    height: int,
    framerate: int,
    warmup_frames: int,
) -> Path:
    output_pattern = output_dir / "frame-%02d.jpg"
    command = build_capture_command(
        device=device,
        output_pattern=output_pattern,
        width=width,
        height=height,
        framerate=framerate,
        warmup_frames=warmup_frames,
    )
    try:
        subprocess.run(command, check=True, timeout=15)
    except FileNotFoundError as exc:
        raise RuntimeError("Install GStreamer so `gst-launch-1.0` is available.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Camera capture failed for {device}.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Camera capture timed out for {device}.") from exc

    frames = sorted(output_dir.glob("frame-*.jpg"))
    if not frames:
        raise RuntimeError(f"Camera capture produced no JPEG frame for {device}.")
    return frames[-1]


async def _stop_capture_process(
    process: asyncio.subprocess.Process,
    *,
    terminate_grace_seconds: float = 0.25,
) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=terminate_grace_seconds)
    except TimeoutError:
        if process.returncode is None:
            process.kill()
        await process.wait()


async def capture_frame_async(
    *,
    device: str,
    output_dir: Path,
    width: int,
    height: int,
    framerate: int,
    warmup_frames: int,
) -> Path:
    """Capture one frame while keeping the GStreamer process cancellable."""
    output_pattern = output_dir / "frame-%02d.jpg"
    command = build_capture_command(
        device=device,
        output_pattern=output_pattern,
        width=width,
        height=height,
        framerate=framerate,
        warmup_frames=warmup_frames,
    )
    try:
        process = await asyncio.create_subprocess_exec(*command)
    except FileNotFoundError as exc:
        raise RuntimeError("Install GStreamer so `gst-launch-1.0` is available.") from exc

    try:
        await asyncio.wait_for(process.wait(), timeout=15)
    except asyncio.CancelledError:
        await _stop_capture_process(process)
        raise
    except TimeoutError as exc:
        await _stop_capture_process(process)
        raise RuntimeError(f"Camera capture timed out for {device}.") from exc

    if process.returncode:
        raise RuntimeError(f"Camera capture failed for {device}.")

    frames = sorted(output_dir.glob("frame-*.jpg"))
    if not frames:
        raise RuntimeError(f"Camera capture produced no JPEG frame for {device}.")
    return frames[-1]


def image_data_url(image_bytes: bytes) -> str:
    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded_image}"


def build_chat_payload(*, model: str, image_bytes: bytes, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url(image_bytes)},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def is_hermes_gateway_url(url: str) -> bool:
    normalized = url.rstrip("/").lower()
    if "/v1/chat/completions" in normalized or "/v1/responses" in normalized:
        normalized = normalized.rsplit("/v1/", 1)[0]
    return normalized.endswith(":8642")


def hermes_gateway_base_url(url: str) -> str:
    normalized = url.rstrip("/")
    if "/v1/" in normalized:
        return normalized.rsplit("/v1/", 1)[0]
    return normalized


def _parse_hermes_responses_body(result: dict[str, Any]) -> str:
    for item in result.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") not in ("output_text", "text"):
                continue
            text = str(part.get("text") or "").strip()
            if text:
                return text
    raise RuntimeError(f"Unexpected Hermes gateway response: {result}")


def analyze_frame_hermes(
    *,
    base_url: str,
    model: str,
    frame: Path,
    prompt: str,
    api_key: str = "",
) -> str:
    payload = {
        "model": model,
        "stream": False,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url(frame.read_bytes())},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{hermes_gateway_base_url(base_url)}/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Hermes gateway returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Hermes gateway at {hermes_gateway_base_url(base_url)}: {exc.reason}"
        ) from exc
    return _parse_hermes_responses_body(result)


def analyze_frame(
    *, url: str, model: str, frame: Path, prompt: str, max_tokens: int, api_key: str = ""
) -> str:
    payload = build_chat_payload(
        model=model,
        image_bytes=frame.read_bytes(),
        prompt=prompt,
        max_tokens=max_tokens,
    )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Vision endpoint returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach LM Studio at {url}: {exc.reason}") from exc

    try:
        message = result["choices"][0]["message"]
        text = str(message.get("content") or "").strip()
        if not text:
            text = str(message.get("reasoning_content") or "").strip()
        if not text:
            raise RuntimeError(f"Unexpected LM Studio response: {result}")
        return text
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected LM Studio response: {result}") from exc


def analyze_camera_frame(
    *,
    device: str,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    width: int,
    height: int,
    framerate: int,
    warmup_frames: int,
    api_key: str = "",
    backend: str = "local",
) -> VisionFrameAnalysis:
    captured = capture_camera_frame(
        device=device,
        width=width,
        height=height,
        framerate=framerate,
        warmup_frames=warmup_frames,
    )
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="vokel-live-vision-analysis-") as tmp:
        frame = Path(tmp) / "frame.jpg"
        frame.write_bytes(base64.b64decode(captured.image_data_url.partition(",")[2]))
        use_hermes = backend == "hermes" or is_hermes_gateway_url(url)
        if use_hermes:
            description = analyze_frame_hermes(
                base_url=url,
                model=model,
                frame=frame,
                prompt=prompt,
                api_key=api_key,
            )
        else:
            description = analyze_frame(
                url=url,
                model=model,
                frame=frame,
                prompt=prompt,
                max_tokens=max_tokens,
                api_key=api_key,
            )
    finished = time.monotonic()

    return VisionFrameAnalysis(
        device=device,
        description=description,
        image_data_url=captured.image_data_url,
        capture_seconds=captured.capture_seconds,
        inference_seconds=finished - started,
    )


def capture_camera_frame(
    *,
    device: str,
    width: int,
    height: int,
    framerate: int,
    warmup_frames: int,
) -> CapturedVisionFrame:
    with tempfile.TemporaryDirectory(prefix="vokel-live-vision-") as tmp:
        started = time.monotonic()
        frame = capture_frame(
            device=device,
            output_dir=Path(tmp),
            width=width,
            height=height,
            framerate=framerate,
            warmup_frames=warmup_frames,
        )
        captured = time.monotonic()
        frame_data_url = image_data_url(frame.read_bytes())

    return CapturedVisionFrame(
        device=device,
        image_data_url=frame_data_url,
        capture_seconds=captured - started,
    )


async def capture_camera_frame_async(
    *,
    device: str,
    width: int,
    height: int,
    framerate: int,
    warmup_frames: int,
) -> CapturedVisionFrame:
    with tempfile.TemporaryDirectory(prefix="vokel-live-vision-") as tmp:
        started = time.monotonic()
        frame = await capture_frame_async(
            device=device,
            output_dir=Path(tmp),
            width=width,
            height=height,
            framerate=framerate,
            warmup_frames=warmup_frames,
        )
        captured = time.monotonic()
        frame_data_url = image_data_url(frame.read_bytes())

    return CapturedVisionFrame(
        device=device,
        image_data_url=frame_data_url,
        capture_seconds=captured - started,
    )
