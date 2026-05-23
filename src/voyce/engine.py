from __future__ import annotations

import asyncio
import json
from contextlib import suppress

from .config import VoiceLoopConfig
from .events import TextDeltaEvent, ToolCallEvent
from .inference import ChatMessage, LocalInferenceClient
from .memory import MemoryConfig, MemoryStore, NullMemoryStore, build_memory_context
from .playback import PlaybackSink
from .telemetry import LatencyTrace
from .text_chunker import PhraseChunker
from .tools import ToolRegistry
from .turns import AsrEngine, TurnProducer


class ConversationEngine:
    def __init__(
        self,
        llm: LocalInferenceClient,
        playback: PlaybackSink,
        config: VoiceLoopConfig | None = None,
        trace: LatencyTrace | None = None,
        echo_tokens: bool = True,
        memory_store: MemoryStore | None = None,
        memory_config: MemoryConfig | None = None,
        tool_registry: ToolRegistry | None = None,
    ):
        self.llm = llm
        self.playback = playback
        self.config = config or VoiceLoopConfig()
        self.trace = trace or LatencyTrace()
        self.echo_tokens = echo_tokens
        self.memory_config = memory_config or MemoryConfig()
        self.memory_store = memory_store or NullMemoryStore()
        self.tool_registry = tool_registry
        self.history: list[ChatMessage] = [{"role": "system", "content": self.config.system_prompt}]
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
        await self.memory_store.close()

    async def wait_for_playback(self) -> None:
        await self._playback_queue.join()

    async def reset_conversation(self) -> None:
        await self.interrupt()
        self.history = [{"role": "system", "content": self.config.system_prompt}]
        self.trace.reset()
        self.trace.mark("conversation_reset")

    async def submit_turn(self, user_text: str, reset_trace: bool = True) -> None:
        await self.interrupt()
        if reset_trace:
            self.trace.reset()
        self.trace.mark("turn_submitted", chars=len(user_text))
        memory_context = await self._retrieve_memory_context(user_text)
        self.history.append({"role": "user", "content": user_text})
        self._trim_history()
        self._current_generation = asyncio.create_task(self._generate_reply(user_text, memory_context))
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
            self.trace.mark("asr_finished", chars=len(transcript), text=transcript)
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

    async def _generate_reply(self, user_text: str, memory_context: str = "") -> None:
        chunker = PhraseChunker()
        assistant_text: list[str] = []
        saw_token = False
        saw_phrase = False

        self.trace.mark("generation_started")

        try:
            direct_web_reply = await self._maybe_run_required_web_search(user_text)
            if direct_web_reply:
                assistant_text.append(direct_web_reply)
                for phrase in chunker.push(direct_web_reply):
                    if not saw_phrase:
                        self.trace.mark("first_phrase_queued", chars=len(phrase))
                        saw_phrase = True
                    await self._playback_queue.put(phrase)

                final_phrase = chunker.flush()
                if final_phrase:
                    if not saw_phrase:
                        self.trace.mark("first_phrase_queued", chars=len(final_phrase))
                    await self._playback_queue.put(final_phrase)

                self.history.append({"role": "assistant", "content": direct_web_reply})
                self._trim_history()
                self.trace.mark(
                    "generation_finished",
                    chars=len(direct_web_reply),
                    text=direct_web_reply,
                )
                await self._record_memory_turn(user_text, direct_web_reply)
                return

            while True:
                tool_calls_made = []
                messages = self._messages_for_generation(memory_context)
                tools = self.tool_registry.get_all_schemas() if self.tool_registry else None
                
                async for event in self.llm.stream_chat(messages, tools):
                    if isinstance(event, TextDeltaEvent):
                        token = event.content
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
                    elif isinstance(event, ToolCallEvent):
                        tool_calls_made.append(event)
                
                if tool_calls_made:
                    # Append assistant's tool call request
                    tool_calls_payload = [
                        {
                            "id": tc.call_id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}
                        } for tc in tool_calls_made
                    ]
                    
                    self.history.append({
                        "role": "assistant",
                        "content": "".join(assistant_text),
                        "tool_calls": tool_calls_payload
                    })
                    
                    if self.tool_registry:
                        for tc in tool_calls_made:
                            self.trace.mark("tool_call_started", tool_name=tc.name)
                            result = await self.tool_registry.execute(tc)
                            self.history.append({
                                "role": "tool",
                                "tool_call_id": tc.call_id,
                                "name": tc.name,
                                "content": result
                            })
                            self.trace.mark(
                                "tool_call_finished",
                                tool_name=tc.name,
                                chars=len(result),
                            )
                    
                    # Reset text for the next iteration to get final answer
                    assistant_text = []
                else:
                    break

            final_phrase = chunker.flush()
            if final_phrase:
                if not saw_phrase:
                    self.trace.mark("first_phrase_queued", chars=len(final_phrase))
                await self._playback_queue.put(final_phrase)

            reply = "".join(assistant_text).strip()
            if reply:
                self.history.append({"role": "assistant", "content": reply})
                self._trim_history()
            
            self.trace.mark("generation_finished", chars=len(reply), text=reply)
            
            if reply:
                await self._record_memory_turn(user_text, reply)
        except asyncio.CancelledError:
            chunker.reset()
            self.trace.mark("generation_cancelled")
            raise

    async def _maybe_run_required_web_search(self, user_text: str) -> str:
        if not self.tool_registry or not self.tool_registry.get_tool("search_web"):
            return ""

        normalized = user_text.lower()
        requires_search = any(
            keyword in normalized
            for keyword in (
                "search",
                "web",
                "news",
                "today",
                "latest",
                "current",
                "breaking",
                "real-time",
            )
        )
        if not requires_search:
            return ""

        self.trace.mark("tool_call_forced", tool_name="search_web")
        tool_call = ToolCallEvent(
            call_id="forced_search_web",
            name="search_web",
            arguments={"query": user_text},
        )
        self.history.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tool_call.call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }
                ],
            }
        )
        result = await self.tool_registry.execute(tool_call)
        self.history.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.call_id,
                "name": tool_call.name,
                "content": result,
            }
        )
        self.trace.mark("tool_call_finished", tool_name="search_web", chars=len(result))
        return f"I searched the web. Here are the top results I found:\n\n{result}"

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

    async def _retrieve_memory_context(self, user_text: str) -> str:
        if not self.memory_config.enabled:
            return ""
        self.trace.mark("memory_retrieval_started")
        try:
            entries = await self.memory_store.retrieve(user_text, self.memory_config.max_results)
            context = build_memory_context(entries, self.memory_config.max_context_chars)
            self.trace.mark(
                "memory_retrieval_finished",
                entries=len(entries),
                chars=len(context),
            )
            return context
        except Exception as exc:
            self.trace.mark("memory_retrieval_failed", error=str(exc))
            return ""

    async def _record_memory_turn(self, user_text: str, assistant_text: str) -> None:
        if not self.memory_config.enabled:
            return
        self.trace.mark("memory_write_started", chars=len(user_text) + len(assistant_text))
        try:
            await self.memory_store.record_turn(user_text, assistant_text)
            self.trace.mark("memory_write_finished")
        except Exception as exc:
            self.trace.mark("memory_write_failed", error=str(exc))

    def _messages_for_generation(self, memory_context: str) -> list[ChatMessage]:
        if not memory_context:
            messages = list(self.history)
        else:
            system = self.history[:1]
            rest = self.history[1:]
            memory_message: ChatMessage = {"role": "system", "content": memory_context}
            messages = system + [memory_message] + rest
            
        if self.tool_registry:
            tool_names = ", ".join(s["function"]["name"] for s in self.tool_registry.get_all_schemas())
            if tool_names:
                instruction = (
                    f"You have access to the following tools: {tool_names}. "
                    "You MUST use these tools when the user asks for real-time information, web searches, or news. "
                    "DO NOT pretend or hallucinate search results. You MUST call the tool."
                )
                messages.insert(1, {"role": "system", "content": instruction})
                
        return messages
