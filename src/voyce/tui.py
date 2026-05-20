from __future__ import annotations

import argparse
import asyncio
import sys

from .config import LmStudioConfig
from .engine import ConversationEngine
from .lm_studio import LmStudioClient
from .playback import build_playback_sink, available_playback_backends
from .turns import AudioTurn


class InteractiveTextProducer:
    """Produces turns by reading from standard input asynchronously."""

    async def next_turn(self) -> AudioTurn:
        loop = asyncio.get_running_loop()
        print("\n> ", end="", flush=True)
        # We read from sys.stdin in an executor to avoid blocking the asyncio loop.
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            raise StopAsyncIteration

        text = line.strip()
        if text.lower() in ("/exit", "/quit"):
            raise StopAsyncIteration

        return AudioTurn(audio_samples=(), sample_rate=0, text=text)


async def run_tui(args: argparse.Namespace) -> None:
    print("Welcome to Voyce Interactive Loop.")
    print(f"Using Playback Sink: {args.playback}")
    print("Type your message and press Enter. Type /exit to quit.")
    print("If you interrupt, the assistant will stop speaking immediately.")

    from .turns import PassthroughAsr

    asr = PassthroughAsr()
    producer = InteractiveTextProducer()

    lm_config = LmStudioConfig(url=args.url, model=args.model)
    sink = build_playback_sink(args.playback)

    async with LmStudioClient(lm_config) as llm:
        engine = ConversationEngine(llm=llm, playback=sink, echo_tokens=True)
        await engine.start()
        try:
            await engine.run_turns(producer=producer, asr=asr, max_turns=None)
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\nShutting down...")
        finally:
            await engine.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an interactive Voyce CLI loop.")
    parser.add_argument("--url", default=LmStudioConfig.url)
    parser.add_argument("--model", default=LmStudioConfig.model)
    parser.add_argument(
        "--playback",
        choices=available_playback_backends(),
        default="kokoro" if "kokoro" in available_playback_backends() else "console",
        help="The playback backend to use.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        asyncio.run(run_tui(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
