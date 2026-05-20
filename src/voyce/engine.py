from __future__ import annotations

import asyncio
from contextlib import suppress

from .config import VoiceLoopConfig
from .lm_studio import ChatMessage, LmStudioClient
from .playback import PlaybackSink
from .telemetry import LatencyTrace
from .text_chunker import PhraseChunker
from .turns import AsrEngine, TurnProducer


class ConversationEngine:
    def __init__(
        self,
        llm: LmStudioClient,
        playback: PlaybackSink,
        config: VoiceLoopConfig | None = None,
        trace: LatencyTrace | None = None,
        echo_tokens: bool = True,
    ):
        self.llm = llm
        self.playback = playback
        self.config = config or VoiceLoopConfig()
        self.trace = trace or LatencyTrace()
        self.echo_tokens = echo_tokens
        self.history: list[ChatMessage] = [
            {"role": "system", "content": self.config.system_prompt}
        ]
        self._playback_queue: asyncio.Queue[str] = asyncio.Queue()
        self._current_generation: asyncio.Task[None] | None = None
        self._playback_worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._playback_worker is None or self._playback_worker.done():
            self._playback_worker = asyncio.create_task(self._run_playback())

    async def close(self) -> None:
        await self.interrupt()
        if self._playback_worker:
            self._playback_worker.cancel()
            with suppress(asyncio.CancelledError):
                await self._playback_worker

    async def wait_for_playback(self) -> None:
        await self._playback_queue.join()

    async def submit_turn(self, user_text: str, reset_trace: bool = True) -> None:
        await self.interrupt()
        if reset_trace:
            self.trace.reset()
        self.trace.mark("turn_submitted", chars=len(user_text))
        self.history.append({"role": "user", "content": user_text})
        self._trim_history()
        self._current_generation = asyncio.create_task(self._generate_reply())
        await self._current_generation

    async def run_turns(
        self,
        producer: TurnProducer,
        asr: AsrEngine,
        max_turns: int | None = None,
    ) -> None:
        turns_processed = 0
        while max_turns is None or turns_processed < max_turns:
            self.trace.reset()
            self.trace.mark("capture_started")
            try:
                turn = await producer.next_turn()
            except StopAsyncIteration:
                return

            audio_seconds = (
                len(turn.audio_samples) / turn.sample_rate
                if turn.audio_samples and turn.sample_rate
                else 0
            )
            self.trace.mark(
                "capture_finished",
                has_audio=turn.has_audio,
                audio_seconds=audio_seconds,
                samples=len(turn.audio_samples),
            )
            self.trace.mark("asr_started", has_audio=turn.has_audio)
            transcript = await asr.transcribe(turn)
            self.trace.mark("asr_finished", chars=len(transcript))
            if transcript.strip():
                await self.submit_turn(transcript, reset_trace=False)
                await self.wait_for_playback()
                turns_processed += 1

    async def interrupt(self) -> None:
        if self._current_generation and not self._current_generation.done():
            self.trace.mark("interruption_requested")
            self._current_generation.cancel()
            with suppress(asyncio.CancelledError):
                await self._current_generation

        self._drain_playback_queue()
        await self.playback.stop()
        self.trace.mark("playback_stop_requested")

    async def _generate_reply(self) -> None:
        chunker = PhraseChunker()
        assistant_text: list[str] = []
        saw_token = False
        saw_phrase = False

        self.trace.mark("generation_started")

        try:
            async for token in self.llm.stream_chat(self.history):
                if not saw_token:
                    self.trace.mark("first_token")
                    saw_token = True
                if self.echo_tokens:
                    print(token, end="", flush=True)
                assistant_text.append(token)
                for phrase in chunker.push(token):
                    if not saw_phrase:
                        self.trace.mark("first_phrase_queued", chars=len(phrase))
                        saw_phrase = True
                    await self._playback_queue.put(phrase)

            final_phrase = chunker.flush()
            if final_phrase:
                if not saw_phrase:
                    self.trace.mark("first_phrase_queued", chars=len(final_phrase))
                await self._playback_queue.put(final_phrase)

            reply = "".join(assistant_text).strip()
            if reply:
                self.history.append({"role": "assistant", "content": reply})
                self._trim_history()
            self.trace.mark("generation_finished", chars=len(reply))
        except asyncio.CancelledError:
            chunker.reset()
            self.trace.mark("generation_cancelled")
            raise

    async def _run_playback(self) -> None:
        while True:
            phrase = await self._playback_queue.get()
            try:
                self.trace.mark("playback_started", chars=len(phrase))
                await self.playback.speak(phrase)
                self.trace.mark("playback_finished", chars=len(phrase))
            finally:
                self._playback_queue.task_done()

    def _drain_playback_queue(self) -> None:
        while True:
            try:
                self._playback_queue.get_nowait()
                self._playback_queue.task_done()
            except asyncio.QueueEmpty:
                break

    def _trim_history(self) -> None:
        max_messages = self.config.max_history_messages
        if len(self.history) <= max_messages:
            return
        system = self.history[:1]
        recent = self.history[-(max_messages - 1) :]
        self.history = system + recent
