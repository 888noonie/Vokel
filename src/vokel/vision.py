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
from typing import Any


DEFAULT_VISION_PROMPT = "Briefly describe what is visible in this live webcam frame."


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


_VISUAL_CONTEXT_SPACE_RE = re.compile(r"\s+")
_VISUAL_CONTEXT_PUNCT_RE = re.compile(r"[^a-z0-9\s']")


def should_capture_visual_context(user_text: str) -> bool:
    normalized = _VISUAL_CONTEXT_SPACE_RE.sub(
        " ",
        _VISUAL_CONTEXT_PUNCT_RE.sub(" ", user_text.lower()),
    ).strip()
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


def list_camera_devices() -> list[CameraDevice]:
    devices: list[CameraDevice] = []
    for node in sorted(Path("/sys/class/video4linux").glob("video*")):
        devices.append(CameraDevice(path=f"/dev/{node.name}", name=(node / "name").read_text().strip()))
    return devices


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
    return [
        "gst-launch-1.0",
        "-q",
        "v4l2src",
        f"device={device}",
        f"num-buffers={warmup_frames}",
        "!",
        f"video/x-raw,format=YUY2,width={width},height={height},framerate={framerate}/1",
        "!",
        "videoconvert",
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


def analyze_frame(*, url: str, model: str, frame: Path, prompt: str, max_tokens: int) -> str:
    payload = build_chat_payload(
        model=model,
        image_bytes=frame.read_bytes(),
        prompt=prompt,
        max_tokens=max_tokens,
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LM Studio returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach LM Studio at {url}: {exc.reason}") from exc

    try:
        return str(result["choices"][0]["message"]["content"]).strip()
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
        description = analyze_frame(
            url=url,
            model=model,
            frame=frame,
            prompt=prompt,
            max_tokens=max_tokens,
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
