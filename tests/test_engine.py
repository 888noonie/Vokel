import asyncio
import unittest
from collections.abc import AsyncIterator, Sequence
from typing import Any

from voyce.engine import ConversationEngine
from voyce.events import Event, TextDeltaEvent, ToolCallEvent
from voyce.inference import ChatMessage
from voyce.memory import MemoryConfig, MemoryEntry
from voyce.tools import ToolDefinition, ToolRegistry
from voyce.turns import PassthroughAsr, TextTurnProducer


class FakeLlm:
    def __init__(self, events: list[Event], delay: float = 0) -> None:
        self.events = events
        self.delay = delay
        self.messages: list[list[ChatMessage]] = []

    async def stream_chat(
        self, messages: Sequence[ChatMessage], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[Event]:
        self.messages.append(list(messages))
        for event in self.events:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield event


class RecordingPlayback:
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.stop_count = 0

    async def speak(self, phrase: str) -> None:
        self.spoken.append(phrase)

    async def stop(self) -> None:
        self.stop_count += 1


class FakeMemoryStore:
    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    async def retrieve(self, query: str, limit: int) -> list[MemoryEntry]:
        return [
            MemoryEntry(
                user_text="We discussed local model memory.",
                assistant_text="Keep it local, explicit, and fast.",
                created_at_ns=1,
                score=1,
            )
        ][:limit]

    async def record_turn(self, user_text: str, assistant_text: str) -> None:
        self.records.append((user_text, assistant_text))

    async def close(self) -> None:
        pass


class ConversationEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_turn_records_reply_and_speaks_phrases(self) -> None:
        llm: Any = FakeLlm([TextDeltaEvent("Hello, "), TextDeltaEvent("Richard.")])
        playback = RecordingPlayback()
        engine = ConversationEngine(llm=llm, playback=playback)

        await engine.start()
        try:
            await engine.submit_turn("Hello")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        self.assertEqual(playback.spoken, ["Hello, Richard."])
        self.assertEqual(engine.history[-1], {"role": "assistant", "content": "Hello, Richard."})

    async def test_interrupt_cancels_generation_and_stops_playback(self) -> None:
        llm: Any = FakeLlm([TextDeltaEvent("This response will take a while. ")], delay=1)
        playback = RecordingPlayback()
        engine = ConversationEngine(llm=llm, playback=playback)

        await engine.start()
        turn = asyncio.create_task(engine.submit_turn("Start"))
        await asyncio.sleep(0.01)
        await engine.interrupt()
        await engine.close()

        self.assertTrue(turn.cancelled() or turn.done())
        self.assertGreaterEqual(playback.stop_count, 1)

    async def test_run_turns_transcribes_and_submits_turn(self) -> None:
        llm: Any = FakeLlm([TextDeltaEvent("Ready.")])
        playback = RecordingPlayback()
        engine = ConversationEngine(llm=llm, playback=playback)

        await engine.start()
        try:
            await engine.run_turns(
                producer=TextTurnProducer(["Start"]),
                asr=PassthroughAsr(),
                max_turns=1,
            )
        finally:
            await engine.close()

        self.assertIn({"role": "user", "content": "Start"}, engine.history)
        self.assertEqual(engine.history[-1], {"role": "assistant", "content": "Ready."})

    async def test_enabled_memory_is_injected_and_recorded(self) -> None:
        llm: Any = FakeLlm([TextDeltaEvent("Remembered.")])
        playback = RecordingPlayback()
        memory = FakeMemoryStore()
        engine = ConversationEngine(
            llm=llm,
            playback=playback,
            memory_store=memory,
            memory_config=MemoryConfig(enabled=True),
        )

        await engine.start()
        try:
            await engine.submit_turn("What did we decide about memory?")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        messages = llm.messages[0]
        self.assertEqual(messages[1]["role"], "system")
        self.assertIn("Relevant saved context", messages[1]["content"])
        self.assertEqual(memory.records, [("What did we decide about memory?", "Remembered.")])
        self.assertIn("memory_retrieval", engine.trace.summary_ms())

    async def test_web_search_prompt_forces_search_before_generation(self) -> None:
        queries: list[str] = []

        async def fake_search_web(query: str) -> str:
            queries.append(query)
            return "Story one. Story two. Story three."

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="search_web",
                description="Search the web.",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                func=fake_search_web,
            )
        )
        llm: Any = FakeLlm([TextDeltaEvent("The model should not answer this.")])
        playback = RecordingPlayback()
        engine = ConversationEngine(llm=llm, playback=playback, tool_registry=registry)

        await engine.start()
        try:
            await engine.submit_turn("Search for the top UK political news results today.")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        self.assertEqual(queries, ["Search for the top UK political news results today."])
        self.assertEqual(llm.messages, [])
        self.assertIn(
            "I searched the web. Here are the top results I found: Story one. Story two. Story three.",
            " ".join(playback.spoken),
        )
        messages = engine.history
        self.assertIn(
            {
                "role": "tool",
                "tool_call_id": "forced_search_web",
                "name": "search_web",
                "content": "Story one. Story two. Story three.",
            },
            messages,
        )


if __name__ == "__main__":
    unittest.main()
