#!/usr/bin/env python
"""Desktop Hermes test — ping + schema exercise"""
import asyncio
import json
import time
import websockets

async def test_messages(host: str, runs: int = 8):
    uri = f"ws://{host}:8765"
    print(f"Testing full message exchange → {uri}\n")

    for i in range(1, runs + 1):
        start = time.perf_counter()
        async with websockets.connect(uri) as ws:
            # Test ping
            await ws.send(json.dumps({"type": "ping"}))
            pong = await ws.recv()

            # Test get_schema
            await ws.send(json.dumps({"type": "get_schema"}))
            schema = await ws.recv()

            total = (time.perf_counter() - start) * 1000
            print(f"Run {i}: ping→pong + get_schema = {total:.2f} ms")
            print(f"     pong: {pong}")
            print(f"     schema: {schema}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--runs", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(test_messages(args.host, args.runs))
