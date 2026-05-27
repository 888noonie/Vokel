#!/data/data/com.termux/files/usr/bin/env python
"""
Hermes WebSocket Stub v2 — Full protocol support (ready for Vokel)
Run this on your Pixel in Termux.

This version supports:
- ping / pong
- get_schema
- start_turn → streams deltas + turn_complete
- cancel (basic)

Replace the fake streaming section with your actual on-device model
(Phi-3, Gemma, Llama.cpp, etc.) when you're ready.
"""

import asyncio
import json
import time
import sys

import websockets

# --- CONFIG ---
PORT = 8765
FAKE_LATENCY_BETWEEN_TOKENS = 0.07   # seconds (simulate streaming)


async def handler(websocket):
    print("Hermes connected", flush=True)

    # Send ready immediately on connect (like before)
    await websocket.send(json.dumps({"type": "ready", "ts": time.time()}))

    async for raw in websocket:
        try:
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "ping":
                await websocket.send(json.dumps({"type": "pong", "ts": time.time()}))

            elif msg_type == "get_schema":
                await websocket.send(json.dumps({
                    "type": "tool_schema",
                    "tools": ["web_search", "calculate", "remind_me"]
                }))

            elif msg_type == "start_turn":
                turn_id = msg["turn_id"]
                prompt = msg.get("prompt", "")
                print(f"[stub] start_turn received: {prompt[:80]}...", flush=True)

                # === TODO: Replace this block with your real model ===
                # For now: simple echo that streams token-by-token
                fake_response = f"Echo from Pixel Pro 8: {prompt[:80]} ... (this is simulated streaming)"

                for token in fake_response.split():
                    await websocket.send(json.dumps({
                        "type": "delta",
                        "turn_id": turn_id,
                        "content": token + " "
                    }))
                    await asyncio.sleep(FAKE_LATENCY_BETWEEN_TOKENS)

                await websocket.send(json.dumps({
                    "type": "turn_complete",
                    "turn_id": turn_id
                }))
                print(f"[stub] turn_complete sent for {turn_id}", flush=True)

            elif msg_type == "cancel":
                turn_id = msg.get("turn_id")
                print(f"[stub] cancel requested for {turn_id}", flush=True)
                # In a real implementation you would set a flag that stops generation
                # For the fake version we just acknowledge
                await websocket.send(json.dumps({
                    "type": "error",
                    "turn_id": turn_id,
                    "message": "cancel not fully implemented in fake mode"
                }))

            else:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"unknown message type: {msg_type}"
                }))

        except Exception as e:
            print(f"[stub] error handling message: {e}", flush=True)
            try:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
            except Exception:
                pass


async def main():
    print(f"Hermes WebSocket stub v2 listening on :{PORT}", flush=True)
    async with websockets.serve(handler, "0.0.0.0", PORT):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStub stopped.", flush=True)
        sys.exit(0)
