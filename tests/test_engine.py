import asyncio
import contextlib
import unittest
from collections.abc import AsyncIterator, Sequence
from typing import Any

from vokel.config import VoiceLoopConfig
from vokel.engine import ConversationEngine
from vokel.events import Event, TextDeltaEvent
from vokel.agent_backend import TOOL_ACTIVITY_REPORTING_CONTRACT
from vokel.inference import ChatMessage
from vokel.memory import MemoryConfig, MemoryEntry
from vokel.tools import ToolDefinition, ToolRegistry
from vokel.turns import PassthroughAsr, TextTurnProducer
from vokel.vision import VisualContext, should_capture_visual_context


class FakeLlm:
    def __init__(self, events: list[Event], delay: float = 0) -> None:
        self.events = events
        self.delay = delay
        self.messages: list[list[ChatMessage]] = []
        self.tools: list[list[dict[str, Any]] | None] = []

    async def stream_chat(
        self, messages: Sequence[ChatMessage], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[Event]:
        self.messages.append(list(messages))
        self.tools.append(tools)
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
    def test_system_prompt_is_grounded_with_current_date(self) -> None:
        import datetime

        engine = ConversationEngine(llm=FakeLlm([]), playback=RecordingPlayback())
        system = engine.history[0]["content"]
        today = datetime.datetime.now().astimezone()
        # The live build guessed "October 24th, 2024"; the real date must be injected
        # so a clock-less local model stops confabulating it.
        self.assertIn("today's date is", system.lower())
        self.assertIn(str(today.year), system)
        self.assertIn(today.strftime("%B"), system)

    def test_system_prompt_forbids_fabricating_tools_and_results(self) -> None:
        # "BS-Buddy" guardrails: the live build claimed a weather/image tool it lacks,
        # narrated fake tool calls, and invented headlines/weather/an image description.
        engine = ConversationEngine(llm=FakeLlm([]), playback=RecordingPlayback())
        system = engine.history[0]["content"].lower()
        self.assertIn("no weather tool", system)
        self.assertIn("never narrate tool usage", system)
        self.assertIn("could not retrieve", system)
        self.assertIn("never invent", system)

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

        self.assertEqual(
            llm.messages[0][0],
            {"role": "system", "content": engine._session_system_prompt()},
        )
        self.assertEqual(llm.messages[0][1], {"role": "system", "content": TOOL_ACTIVITY_REPORTING_CONTRACT})
        self.assertEqual(llm.messages[0][2], {"role": "user", "content": "Search for the latest UK news today."})
        self.assertEqual(engine.history[-1], {"role": "assistant", "content": "Hermes says hi."})

    async def test_hermes_mode_forwards_session_system_prompt_with_topic(self) -> None:
        llm: Any = FakeLlm([TextDeltaEvent("Bars about breakfast.")])
        playback = RecordingPlayback()
        base = VoiceLoopConfig()
        engine = ConversationEngine(
            agent=llm,
            playback=playback,
            agent_mode="hermes",
            config=VoiceLoopConfig(
                system_prompt=base.system_prompt
                + " Session topic: rapping about breakfast."
            ),
        )

        await engine.start()
        try:
            await engine.submit_turn("It's time to rap.")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        system_texts = [
            m["content"] for m in llm.messages[0] if m["role"] == "system"
        ]
        self.assertTrue(
            any("Session topic: rapping about breakfast." in text for text in system_texts)
        )
        self.assertTrue(
            any(TOOL_ACTIVITY_REPORTING_CONTRACT in text for text in system_texts)
        )
        self.assertEqual(llm.messages[0][-1], {"role": "user", "content": "It's time to rap."})

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

    async def test_analyze_image_serialized_tool_call_is_stripped(self) -> None:
        llm: Any = FakeLlm([
            TextDeltaEvent(
                '<|tool_call>call: analyze_image{image_description:<|"|>steaming mug<|"|>}<tool_call|> '
            ),
            TextDeltaEvent("You are holding a mug."),
        ])
        playback = RecordingPlayback()
        engine = ConversationEngine(llm=llm, playback=playback)

        await engine.start()
        try:
            await engine.submit_turn("Describe this object.")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        self.assertNotIn("<|tool_call>", engine.history[-1]["content"])
        self.assertNotIn("analyze_image", engine.history[-1]["content"])

    async def test_visual_question_without_camera_armed_fails_closed(self) -> None:
        async def provider(user_text: str) -> VisualContext | None:
            if should_capture_visual_context(user_text):
                return VisualContext(
                    data_url="",
                    source="/dev/video0",
                    consent="voice_context_not_armed",
                )
            return None

        llm: Any = FakeLlm([TextDeltaEvent("You are holding a mug.")])
        playback = RecordingPlayback()
        engine = ConversationEngine(
            llm=llm,
            playback=playback,
            visual_context_provider=provider,
        )

        await engine.start()
        try:
            await engine.submit_turn("What am I holding?")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        self.assertEqual(len(llm.messages), 0)
        self.assertIn("Camera Questions", playback.spoken[0])
        self.assertNotIn("mug", " ".join(playback.spoken).lower())

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

    async def test_visual_turn_attaches_frame_and_suppresses_search_tools(self) -> None:
        queries: list[str] = []

        async def fake_search_image(query: str) -> str:
            queries.append(query)
            return "![camera](https://example.com/camera.jpg)"

        async def visual_context_provider(user_text: str) -> str | None:
            if user_text == "What am I holding?":
                return "data:image/jpeg;base64,anBlZw=="
            return None

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="search_image",
                description="Search images.",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                func=fake_search_image,
            )
        )
        llm: Any = FakeLlm([TextDeltaEvent("You are holding a mug.")])
        playback = RecordingPlayback()
        engine = ConversationEngine(
            llm=llm,
            playback=playback,
            tool_registry=registry,
            enabled_tools={"search_image"},
            visual_context_provider=visual_context_provider,
        )

        await engine.start()
        try:
            await engine.submit_turn("What am I holding?")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        self.assertEqual(queries, [])
        self.assertIsNone(llm.tools[0])
        visual_message = llm.messages[0][-1]
        self.assertEqual(visual_message["role"], "user")
        self.assertEqual(
            visual_message["content"][0],
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,anBlZw=="}},
        )
        # The spoken question rides as-is; the anti-tool-call directive is appended so a
        # small model answers in words instead of echoing a describe_image tool call.
        text_part = visual_message["content"][1]
        self.assertEqual(text_part["type"], "text")
        self.assertTrue(text_part["text"].startswith("What am I holding?"))
        self.assertIn("Do not output any tool calls", text_part["text"])
        self.assertIn({"role": "user", "content": "What am I holding?"}, engine.history)
        self.assertNotIn("data:image", str(engine.history))

    async def test_visual_turn_falls_back_when_model_only_emits_tool_call(self) -> None:
        from vokel.engine import VISUAL_REPLY_FALLBACK

        async def visual_context_provider(user_text: str) -> str | None:
            if user_text == "What am I holding?":
                return "data:image/jpeg;base64,anBlZw=="
            return None

        # The frame is attached, but a 4B model parrots a describe_image tool call
        # instead of answering — exactly the live failure. Nothing else is emitted.
        leak = 'call:describe_image{image_base64:<|"|>[Image data provided]<|"|>}<tool_call|>'
        llm: Any = FakeLlm([TextDeltaEvent(leak)])
        playback = RecordingPlayback()
        engine = ConversationEngine(
            llm=llm,
            playback=playback,
            visual_context_provider=visual_context_provider,
        )

        await engine.start()
        try:
            await engine.submit_turn("What am I holding?")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        # The raw tool-call tokens never reach the speaker or the transcript history.
        spoken = " ".join(playback.spoken)
        self.assertNotIn("describe_image", spoken)
        self.assertNotIn("tool_call", spoken)
        self.assertNotIn("describe_image", str(engine.history))
        # Instead of silence, the user hears a graceful fallback.
        self.assertEqual(playback.spoken, [VISUAL_REPLY_FALLBACK])
        self.assertEqual(engine.history[-1], {"role": "assistant", "content": VISUAL_REPLY_FALLBACK})

    async def test_visual_capture_failure_is_spoken_without_model_guessing(self) -> None:
        async def visual_context_provider(user_text: str) -> str | None:
            return "" if user_text == "What am I holding?" else None

        llm: Any = FakeLlm([TextDeltaEvent("Should not run.")])
        playback = RecordingPlayback()
        engine = ConversationEngine(
            llm=llm,
            playback=playback,
            visual_context_provider=visual_context_provider,
        )

        await engine.start()
        try:
            await engine.submit_turn("What am I holding?")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        self.assertEqual(llm.messages, [])
        self.assertEqual(
            playback.spoken,
            ["I couldn't capture a camera frame, so I can't answer that visual question yet."],
        )

    async def test_spoken_directive_is_spoken_without_calling_model(self) -> None:
        from vokel.vision import SpokenReply

        async def provider(user_text: str) -> SpokenReply | None:
            if user_text == "watch me":
                return SpokenReply("Just say shoot and I'll take the picture.")
            return None

        llm: Any = FakeLlm([TextDeltaEvent("model should not run")])
        playback = RecordingPlayback()
        engine = ConversationEngine(
            llm=llm,
            playback=playback,
            visual_context_provider=provider,
        )

        await engine.start()
        try:
            await engine.submit_turn("watch me")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        self.assertEqual(llm.messages, [])
        self.assertEqual(playback.spoken, ["Just say shoot and I'll take the picture."])
        self.assertEqual(engine.history[-2], {"role": "user", "content": "watch me"})
        self.assertEqual(
            engine.history[-1],
            {"role": "assistant", "content": "Just say shoot and I'll take the picture."},
        )

    async def test_shoot_command_sends_clean_prompt_but_keeps_spoken_words(self) -> None:
        async def provider(user_text: str) -> VisualContext | None:
            if user_text == "shoot":
                return VisualContext(
                    data_url="data:image/jpeg;base64,anBlZw==",
                    source="/dev/video0",
                    prompt="Describe exactly what you see.",
                )
            return None

        llm: Any = FakeLlm([TextDeltaEvent("A red mug on a desk.")])
        playback = RecordingPlayback()
        engine = ConversationEngine(
            llm=llm,
            playback=playback,
            visual_context_provider=provider,
        )

        await engine.start()
        try:
            await engine.submit_turn("shoot")
            await engine.wait_for_playback()
        finally:
            await engine.close()

        # The vision model receives the clean instruction (plus the anti-tool-call
        # directive), not the bare "shoot".
        text_part = llm.messages[0][-1]["content"][1]
        self.assertEqual(text_part["type"], "text")
        self.assertTrue(text_part["text"].startswith("Describe exactly what you see."))
        self.assertIn("Do not output any tool calls", text_part["text"])
        # The transcript/history still records what the user actually said.
        self.assertIn({"role": "user", "content": "shoot"}, engine.history)

    async def test_barge_in_during_visual_capture_prevents_frame_from_being_sent(self) -> None:
        """Strict proof (per tightening): engine tracks capture task, interrupt cancels it,
        late completion of provider cannot cause submit_visual_turn or backend calls.
        Assertions: capture task done+cancelled, FakeLlm (backend) received zero calls,
        even after unblocking the "late" capture, no frame is ever sent.
        """
        import asyncio

        capture_started = asyncio.Event()
        capture_blocker = asyncio.Event()
        provider_call_count = 0

        async def slow_visual_provider(user_text: str) -> str | None:
            nonlocal provider_call_count
            if "holding" not in user_text.lower():
                return None
            provider_call_count += 1
            capture_started.set()
            await capture_blocker.wait()
            return "data:image/jpeg;base64,LATEFRAME"

        llm: Any = FakeLlm([TextDeltaEvent("must not receive visual frame")])
        playback = RecordingPlayback()
        engine = ConversationEngine(
            llm=llm,
            playback=playback,
            visual_context_provider=slow_visual_provider,
        )
        await engine.start()

        turn_task = asyncio.create_task(engine.submit_turn("What am I holding?"))

        await asyncio.wait_for(capture_started.wait(), timeout=1.0)

        # Snapshot the tracked task before interrupt (submit_turn finally clears the attr)
        capture_task = engine._pending_visual_capture_task
        self.assertIsNotNone(capture_task)

        # Barge in while capture pending
        await engine.interrupt()

        # Strict: the tracked task is done (cancelled by interrupt)
        self.assertTrue(capture_task.done())

        # Unblock the provider "late" completion (simulates slow gst finishing after cancel)
        capture_blocker.set()

        with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
            await asyncio.wait_for(turn_task, timeout=0.5)

        await engine.close()

        # Key strict assertions:
        # - No frame data ever entered history
        self.assertNotIn("LATEFRAME", str(engine.history))
        # - Backend (FakeLlm) received zero calls whatsoever from this visual attempt
        self.assertEqual(len(llm.messages), 0)
        self.assertNotIn("must not receive visual frame", "".join(playback.spoken))
        # - Provider was called (we started it), but result was discarded
        self.assertGreaterEqual(provider_call_count, 1)

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
