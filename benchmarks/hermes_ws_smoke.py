#!/usr/bin/env python3
"""
Hermes WebSocket Smoke Test
Runs against the real Termux stub on Pixel 8 Pro.

Usage:
    PYTHONPATH=src python benchmarks/hermes_ws_smoke.py --host 192.168.6.151
"""

import argparse
import asyncio
import sys

from vokel.hermes.websocket_client import HermesWebSocketClient
from vokel.inference import ChatMessage


async def run_smoke(host: str) -> None:
    url = f"ws://{host}:8765"
    print(f"HermesWebSocketClient smoke test → {url}\n")

    client = HermesWebSocketClient(url)

    messages: list[ChatMessage] = [
        {"role": "user", "content": "What is 17 times 23? Please calculate it step by step."}
    ]

    print("Sending turn...\n")

    try:
        async with client:
            token_count = 0
            async for event in client.stream_chat(messages):
                if hasattr(event, "content"):
                    content = getattr(event, "content", "")
                    print(content, end="", flush=True)
                    token_count += len(content.split()) if content else 0
            print(f"\n\n[smoke] Turn complete. Approx tokens: {token_count}")
    except Exception as e:
        print(f"\n[smoke] ERROR: {e}", file=sys.stderr)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.6.151", help="IP of the Pixel running the stub")
    args = parser.parse_args()

    asyncio.run(run_smoke(args.host))


if __name__ == "__main__":
    main()
