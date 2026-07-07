#!/usr/bin/env python3
"""Backend-only musical mode probe: start local session, print beat WS events."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from urllib.parse import urlparse

import websockets


def _llm_base(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.removesuffix("/v1/chat/completions").removesuffix("/v1")
    return f"{parsed.scheme}://{parsed.netloc}{path or ''}"


async def main() -> int:
    ws_url = os.environ.get("VOKEL_WS_URL", "ws://127.0.0.1:8000/api/ws")
    llm_url = os.environ.get(
        "VOKEL_LLM_URL",
        os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:6767/v1/chat/completions"),
    )
    model = os.environ.get(
        "VOKEL_LLM_MODEL",
        os.environ.get("LM_STUDIO_MODEL", "local-model"),
    )
    bpm = float(os.environ.get("VOKEL_MUSICAL_BPM", "90"))
    beat_target = int(os.environ.get("VOKEL_PROBE_BEATS", "4"))

    print(f"Connecting to {ws_url} …")
    async with websockets.connect(ws_url) as ws:
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            data = json.loads(raw)
            if data.get("type") == "execute_state":
                break

        payload = {
            "type": "start_session",
            "mode": "local",
            "playback": "kokoro",
            "url": _llm_base(llm_url),
            "model": model,
            "musical_mode": True,
            "musical_bpm": bpm,
        }
        print(f"Starting local musical session (bpm={bpm}) …")
        await ws.send(json.dumps(payload))

        beats = 0
        deadline = asyncio.get_running_loop().time() + 8.0
        while beats < beat_target and asyncio.get_running_loop().time() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=8.0)
            data = json.loads(raw)
            msg_type = data.get("type")
            if msg_type == "beat":
                beats += 1
                print(f"  beat #{beats}: bar={data.get('bar')} beat={data.get('beat')} bpm={data.get('bpm')}")
            elif msg_type == "error":
                print(f"error: {data.get('message')}", file=sys.stderr)
                return 1
            elif msg_type == "session_started":
                print(f"  session_started mode={data.get('mode')}")

        await ws.send(json.dumps({"type": "stop_session"}))
        print(f"OK — received {beats} beat event(s)")
        return 0 if beats >= 2 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))