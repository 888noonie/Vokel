import asyncio
import unittest
from collections.abc import AsyncIterator, Sequence
from typing import Any

from vokel.engine import ConversationEngine
from vokel.events import Event, TextDeltaEvent, ToolCallEvent
from vokel.agent_backend import TOOL_ACTIVITY_REPORTING_CONTRACT
from vokel.inference import ChatMessage
from vokel.memory import MemoryConfig, MemoryEntry
from vokel.tools import ToolDefinition, ToolRegistry
from vokel.turns import PassthroughAsr, TextTurnProducer


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

    async def cancel_active(self) -> None:
        return None

    async def __aenter__(self) -> "FakeLlm":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


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

    async def test_web_search_prompt_uses_search_evidence_for_synthesis(self) -> None:
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
        llm: Any = FakeLlm([TextDeltaEvent("Based on the search results, story one is the lead.")])
        playback = RecordingPlayback()
        engine = ConversationEngine(llm=llm, playback=playback, tool_registry=registry, enabled_tools={"search_web"})

        await engine.start()
        try:
            await engine.submit_turn("Search for the top UK political news results today.")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        self.assertEqual(queries, ["Search for the top UK political news results today."])
        self.assertEqual(len(llm.messages), 1)
        self.assertIn("Search evidence:\nStory one. Story two. Story three.", llm.messages[0][-1]["content"])
        self.assertEqual(playback.spoken[0], "I'm searching now.")
        self.assertIn(
            "Based on the search results, story one is the lead.",
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

    async def test_web_search_synthesis_falls_back_to_raw_evidence(self) -> None:
        async def fake_search_web(query: str) -> str:
            return "1. BBC headline\n   https://www.bbc.com/news/example"

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="search_web",
                description="Search the web.",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                func=fake_search_web,
            )
        )
        llm: Any = FakeLlm([TextDeltaEvent("I couldn't get the result right now.")])
        playback = RecordingPlayback()
        engine = ConversationEngine(llm=llm, playback=playback, tool_registry=registry, enabled_tools={"search_web"})

        await engine.start()
        try:
            await engine.submit_turn("Find the top BBC headline today.")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        spoken = " ".join(playback.spoken)
        self.assertEqual(playback.spoken[0], "I'm searching now.")
        self.assertIn("I searched the web.", spoken)
        self.assertIn("BBC headline", spoken)
        self.assertNotIn("couldn't get", spoken)


    async def test_hermes_mode_sends_only_user_turn_and_skips_forced_search(self) -> None:
        llm: Any = FakeLlm([TextDeltaEvent("Hermes says hi.")])
        playback = RecordingPlayback()

        async def fake_search_web(query: str) -> str:
            return "should not run"

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="search_web",
                description="Search the web.",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                func=fake_search_web,
            )
        )
        engine = ConversationEngine(
            agent=llm,
            playback=playback,
            tool_registry=registry,
            enabled_tools={"search_web"},
            agent_mode="hermes",
        )

        await engine.start()
        try:
            await engine.submit_turn("Search for the latest UK news today.")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        self.assertEqual(llm.messages[0][0], {"role": "system", "content": TOOL_ACTIVITY_REPORTING_CONTRACT})
        self.assertEqual(llm.messages[0][1], {"role": "user", "content": "Search for the latest UK news today."})
        self.assertEqual(engine.history[-1], {"role": "assistant", "content": "Hermes says hi."})

    async def test_hermes_mode_queues_speech_after_full_response_for_link_sanitizing(self) -> None:
        llm: Any = FakeLlm([
            TextDeltaEvent("Here is [generated_image_20260528.png]"),
            TextDeltaEvent("(sandbox:/mnt/data/generated_image_20260528.png)."),
        ])
        playback = RecordingPlayback()
        engine = ConversationEngine(
            agent=llm,
            playback=playback,
            agent_mode="hermes",
        )

        await engine.start()
        try:
            await engine.submit_turn("Show me the image.")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        self.assertEqual(len(playback.spoken), 1)
        self.assertIn("[generated_image_20260528.png](sandbox:/mnt/data/generated_image_20260528.png)", engine.history[-1]["content"])

    async def test_hermes_tool_markers_queue_reassurance_without_visible_marker(self) -> None:
        llm: Any = FakeLlm([
            TextDeltaEvent("[tool_call:search_image]"),
            TextDeltaEvent("Here is the image."),
        ])
        playback = RecordingPlayback()
        engine = ConversationEngine(
            agent=llm,
            playback=playback,
            agent_mode="hermes",
        )

        await engine.start()
        try:
            await engine.submit_turn("Show me an image.")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        self.assertEqual(playback.spoken[0], "I'm fetching that image now.")
        self.assertEqual(engine.history[-1], {"role": "assistant", "content": "Here is the image."})

    async def test_local_serialized_tool_call_is_not_visible_text(self) -> None:
        llm: Any = FakeLlm([
            TextDeltaEvent("Nice image.\n\n"),
            TextDeltaEvent('<|tool_call>call:search_image{query:<|"|>simple background texture<|"|>}<tool_call|>'),
        ])
        playback = RecordingPlayback()
        engine = ConversationEngine(llm=llm, playback=playback)

        await engine.start()
        try:
            await engine.submit_turn("That image failed to load.")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        self.assertEqual(playback.spoken, ["Nice image."])
        self.assertEqual(engine.history[-1], {"role": "assistant", "content": "Nice image."})
        self.assertNotIn("<|tool_call>", " ".join(playback.spoken))

    async def test_image_followup_context_does_not_hijack_unrelated_short_turn(self) -> None:
        queries: list[str] = []

        async def fake_search_image(query: str) -> str:
            queries.append(query)
            return "![forest](https://example.com/forest.jpg)"

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="search_image",
                description="Search images.",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                func=fake_search_image,
            )
        )
        llm: Any = FakeLlm([TextDeltaEvent("Regular text response.")])
        playback = RecordingPlayback()
        engine = ConversationEngine(
            llm=llm,
            playback=playback,
            tool_registry=registry,
            enabled_tools={"search_image"},
        )
        # Prior conversation included image terms; this used to trigger over-eager context mode.
        engine.history.append({"role": "assistant", "content": "Here is an image of a forest."})

        await engine.start()
        try:
            await engine.submit_turn("what time is it")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        self.assertEqual(queries, [])
        self.assertEqual(engine.history[-1], {"role": "assistant", "content": "Regular text response."})

    async def test_image_followup_context_allows_another_one_like_that(self) -> None:
        queries: list[str] = []

        async def fake_search_image(query: str) -> str:
            queries.append(query)
            return "![forest](https://example.com/forest.jpg)"

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="search_image",
                description="Search images.",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                func=fake_search_image,
            )
        )
        llm: Any = FakeLlm([TextDeltaEvent("Nice choice.")])
        playback = RecordingPlayback()
        engine = ConversationEngine(
            llm=llm,
            playback=playback,
            tool_registry=registry,
            enabled_tools={"search_image"},
        )
        engine.history.append({"role": "assistant", "content": "Here is an image of a mountain lake."})

        await engine.start()
        try:
            await engine.submit_turn("another one like that")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        self.assertEqual(queries, ["another one like that"])
        self.assertEqual(playback.spoken[0], "I'm fetching that image now.")

    async def test_media_fallback_is_transparent_when_tool_returns_no_renderable_markdown(self) -> None:
        async def fake_search_image(query: str) -> str:
            return "Unsplash lookup succeeded but no markdown media block was returned."

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="search_image",
                description="Search images.",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                func=fake_search_image,
            )
        )
        llm: Any = FakeLlm([TextDeltaEvent("Here is your image!")])
        playback = RecordingPlayback()
        engine = ConversationEngine(
            llm=llm,
            playback=playback,
            tool_registry=registry,
            enabled_tools={"search_image"},
        )

        await engine.start()
        try:
            await engine.submit_turn("show me an image of a sunset lake")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        assistant_reply = engine.history[-1]["content"]
        self.assertIn("no displayable image or GIF URL was returned", assistant_reply)
        self.assertNotIn("![", assistant_reply)

    async def test_web_search_intent_guard_ignores_planning_feedback(self) -> None:
        queries: list[str] = []

        async def fake_search_web(query: str) -> str:
            queries.append(query)
            return "1. Example result"

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="search_web",
                description="Search the web.",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                func=fake_search_web,
            )
        )
        llm: Any = FakeLlm([TextDeltaEvent("Understood.")])
        playback = RecordingPlayback()
        engine = ConversationEngine(llm=llm, playback=playback, tool_registry=registry, enabled_tools={"search_web"})

        await engine.start()
        try:
            await engine.submit_turn("I'm going to enhance the web search images soon.")
            await engine.wait_for_playback()
            await engine.submit_turn("That web search worked well.")
            await engine.wait_for_playback()
            await engine.submit_turn("We can improve image search later.")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        self.assertEqual(queries, [])

    async def test_web_search_intent_guard_allows_current_info_requests(self) -> None:
        queries: list[str] = []

        async def fake_search_web(query: str) -> str:
            queries.append(query)
            return "1. Artemis update"

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="search_web",
                description="Search the web.",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                func=fake_search_web,
            )
        )
        llm: Any = FakeLlm([TextDeltaEvent("Here is the update.")])
        playback = RecordingPlayback()
        engine = ConversationEngine(llm=llm, playback=playback, tool_registry=registry, enabled_tools={"search_web"})

        await engine.start()
        try:
            await engine.submit_turn("What is the latest on Artemis?")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        self.assertEqual(queries, ["What is the latest on Artemis?"])

    async def test_ambiguous_dot_utterance_is_ignored(self) -> None:
        queries: list[str] = []

        async def fake_search_web(query: str) -> str:
            queries.append(query)
            return "1. Example"

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="search_web",
                description="Search the web.",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                func=fake_search_web,
            )
        )
        llm: Any = FakeLlm([TextDeltaEvent("Should not run")])
        playback = RecordingPlayback()
        engine = ConversationEngine(llm=llm, playback=playback, tool_registry=registry, enabled_tools={"search_web"})

        await engine.start()
        try:
            await engine.submit_turn(".")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        self.assertEqual(llm.messages, [])
        self.assertEqual(queries, [])
        self.assertNotIn({"role": "user", "content": "."}, engine.history)

    async def test_ambiguous_cjk_filler_utterance_is_ignored(self) -> None:
        queries: list[str] = []

        async def fake_search_web(query: str) -> str:
            queries.append(query)
            return "1. Example"

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="search_web",
                description="Search the web.",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                func=fake_search_web,
            )
        )
        llm: Any = FakeLlm([TextDeltaEvent("Should not run")])
        playback = RecordingPlayback()
        engine = ConversationEngine(llm=llm, playback=playback, tool_registry=registry, enabled_tools={"search_web"})

        await engine.start()
        try:
            await engine.submit_turn("嗯")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        self.assertEqual(llm.messages, [])
        self.assertEqual(queries, [])
        self.assertNotIn({"role": "user", "content": "嗯"}, engine.history)

    async def test_media_followup_allows_another_one_with_prior_context(self) -> None:
        queries: list[str] = []

        async def fake_search_image(query: str) -> str:
            queries.append(query)
            return "![forest](https://example.com/forest.jpg)"

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="search_image",
                description="Search images.",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                func=fake_search_image,
            )
        )
        llm: Any = FakeLlm([TextDeltaEvent("Nice one.")])
        playback = RecordingPlayback()
        engine = ConversationEngine(
            llm=llm,
            playback=playback,
            tool_registry=registry,
            enabled_tools={"search_image"},
        )
        engine.history.append({"role": "assistant", "content": "Here is an image of a mountain lake."})

        await engine.start()
        try:
            await engine.submit_turn("another one")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        self.assertEqual(queries, ["another one"])

    async def test_no_is_not_swallowed_as_filler(self) -> None:
        llm: Any = FakeLlm([TextDeltaEvent("Okay, noted.")])
        playback = RecordingPlayback()
        engine = ConversationEngine(llm=llm, playback=playback)

        await engine.start()
        try:
            await engine.submit_turn("no")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        self.assertEqual(len(llm.messages), 1)
        self.assertIn({"role": "user", "content": "no"}, engine.history)


if __name__ == "__main__":
    unittest.main()
