#!/usr/bin/env python3
"""
Desktop Hermes WebSocket Handshake Test
Run on the 8845HS desktop against a Termux Hermes stub on Pixel 8 Pro.

Usage:
    PYTHONPATH=src python benchmarks/hermes_handshake.py --host 192.168.1.42
    PYTHONPATH=src python benchmarks/hermes_handshake.py --host 192.168.1.42 --port 8765 --runs 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import websockets


async def run_handshake(uri: str) -> tuple[float, float]:
    """Returns (connect_ms, rtt_ms) for a single ping roundtrip."""
    start = time.perf_counter()
    async with websockets.connect(uri) as ws:
        connect_time = (time.perf_counter() - start) * 1000

        # Send ping immediately after connect
        await ws.send(json.dumps({"type": "ping"}))
        reply = await ws.recv()

        rtt = (time.perf_counter() - start) * 1000
        return connect_time, rtt, reply


async def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes WebSocket handshake latency test")
    parser.add_argument("--host", required=True, help="IP address of the Pixel 8 Pro running the stub")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port (default 8765)")
    parser.add_argument("--runs", type=int, default=5, help="Number of handshake attempts")
    args = parser.parse_args()

    uri = f"ws://{args.host}:{args.port}"

    print(f"Hermes handshake test → {uri}")
    print(f"Runs: {args.runs}")
    print()

    connect_times: list[float] = []
    rtt_times: list[float] = []

    for i in range(1, args.runs + 1):
        try:
            connect_ms, rtt_ms, reply = await run_handshake(uri)
            connect_times.append(connect_ms)
            rtt_times.append(rtt_ms)
            print(f"Run {i:2d}: connect={connect_ms:6.2f} ms | RTT={rtt_ms:6.2f} ms | reply={reply}")
        except Exception as e:
            print(f"Run {i:2d}: FAILED — {e}")

    if connect_times:
        print()
        print("=== Summary (ms) ===")
        print(f"Connect  P50: {sorted(connect_times)[len(connect_times)//2]:6.2f}  "
              f"P95: {sorted(connect_times)[int(len(connect_times)*0.95)-1]:6.2f}  "
              f"Max: {max(connect_times):6.2f}")
        print(f"RTT      P50: {sorted(rtt_times)[len(rtt_times)//2]:6.2f}  "
              f"P95: {sorted(rtt_times)[int(len(rtt_times)*0.95)-1]:6.2f}  "
              f"Max: {max(rtt_times):6.2f}")


if __name__ == "__main__":
    asyncio.run(main())
