from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from typing import Literal

from .agent_backend import AgentBackend
from .config import VoiceLoopConfig
from .events import TextDeltaEvent, ToolCallEvent
from .hermes_client import HermesAgentClient
from .inference import ChatMessage
from .memory import MemoryConfig, MemoryStore, NullMemoryStore, build_memory_context
from .playback import PlaybackSink
from .telemetry import LatencyTrace
from .text_chunker import PhraseChunker
from .tools import ToolRegistry
from .auto_followup import AUTO_FOLLOWUP_NUDGE
from .turns import AsrEngine, TurnProducer


AgentMode = Literal["builtin", "hermes"]


class ConversationEngine:
    def __init__(
        self,
        agent: AgentBackend | None = None,
        playback: PlaybackSink | None = None,
        *,
        llm: AgentBackend | None = None,
        config: VoiceLoopConfig | None = None,
        trace: LatencyTrace | None = None,
        echo_tokens: bool = True,
        memory_store: MemoryStore | None = None,
        memory_config: MemoryConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        enabled_tools: set[str] | None = None,
        agent_mode: AgentMode = "builtin",
    ):
        resolved_agent = agent if agent is not None else llm
        if resolved_agent is None or playback is None:
            raise TypeError("ConversationEngine requires agent (or llm) and playback")
        self.agent = resolved_agent
        self.llm = resolved_agent  # Backward-compatible alias for callers using .llm
        self.playback = playback
        self.agent_mode = agent_mode
        self.config = config or VoiceLoopConfig()
        self.trace = trace or LatencyTrace()
        self.echo_tokens = echo_tokens
        self.memory_config = memory_config or MemoryConfig()
        self.memory_store = memory_store or NullMemoryStore()
        self.tool_registry = tool_registry if agent_mode == "builtin" else None
        self.enabled_tools: set[str] = (
            enabled_tools if enabled_tools is not None and agent_mode == "builtin" else set()
        )
        self.history: list[ChatMessage] = [{"role": "system", "content": self.config.system_prompt}]
        self._playback_queue: asyncio.Queue[str] = asyncio.Queue()
        self._current_generation: asyncio.Task[None] | None = None
        self._playback_worker: asyncio.Task[None] | None = None
        self._suppress_memory_write = False

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
        if isinstance(self.agent, HermesAgentClient):
            new_session = self.agent.reset_session()
            self.trace.mark("hermes_session_reset", session_id=new_session)
        self.trace.reset()
        self.trace.mark("conversation_reset")

    async def submit_turn(self, user_text: str, reset_trace: bool = True) -> None:
        await self.interrupt()
        if reset_trace:
            self.trace.reset()
        self.trace.mark("turn_submitted", chars=len(user_text))
        memory_context = (
            "" if self.agent_mode == "hermes" else await self._retrieve_memory_context(user_text)
        )
        self.history.append({"role": "user", "content": user_text})
        self._trim_history()
        self._current_generation = asyncio.create_task(self._generate_reply(user_text, memory_context))
        await self._current_generation

    async def submit_auto_followup(self) -> None:
        """Prompt the model to re-engage after listening idle (no visible user transcript)."""
        if self.agent_mode == "hermes":
            return
        if not self._should_auto_followup():
            return
        await self.interrupt()
        self.trace.mark("auto_followup_triggered")
        nudge = AUTO_FOLLOWUP_NUDGE
        memory_context = await self._retrieve_memory_context(nudge)
        self.history.append({"role": "user", "content": nudge})
        self._trim_history()
        self._suppress_memory_write = True
        try:
            self._current_generation = asyncio.create_task(self._generate_reply(nudge, memory_context))
            await self._current_generation
        finally:
            self._suppress_memory_write = False

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
        cancel = getattr(self.agent, "cancel_active", None)
        if cancel is not None:
            await cancel()
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
            media_result = ""
            search_evidence = ""
            if self.agent_mode == "builtin":
                media_result = (
                    await self._maybe_run_required_gif_search(user_text)
                    or await self._maybe_run_required_image_search(user_text)
                )
            if media_result:
                spoken_parts: list[str] = []
                async for event in self.agent.stream_chat(
                    self._messages_for_media_synthesis(memory_context, user_text, media_result)
                ):
                    if isinstance(event, TextDeltaEvent):
                        token = event.content
                        if not saw_token:
                            self.trace.mark("first_token")
                            saw_token = True
                        if self.echo_tokens:
                            print(token, end="", flush=True)
                        spoken_parts.append(token)

                spoken = "".join(spoken_parts).strip()
                if not spoken or len(spoken) < 5:
                    spoken = "Here you go!"

                # Transcript gets media markdown + caption; TTS only gets the caption
                full_reply = f"{media_result}\n\n{spoken}"
                for phrase in chunker.push(spoken):
                    if not saw_phrase:
                        self.trace.mark("first_phrase_queued", chars=len(phrase))
                        saw_phrase = True
                    await self._playback_queue.put(phrase)
                final_phrase = chunker.flush()
                if final_phrase:
                    if not saw_phrase:
                        self.trace.mark("first_phrase_queued", chars=len(final_phrase))
                    await self._playback_queue.put(final_phrase)
                self.history.append({"role": "assistant", "content": full_reply})
                self._trim_history()
                self.trace.mark("generation_finished", chars=len(full_reply), text=full_reply)
                await self._record_memory_turn(user_text, full_reply)
                return

            if self.agent_mode == "builtin":
                search_evidence = await self._maybe_run_required_web_search(user_text)
            if search_evidence:
                async for event in self.agent.stream_chat(
                    self._messages_for_web_synthesis(memory_context, user_text, search_evidence)
                ):
                    if isinstance(event, TextDeltaEvent):
                        token = event.content
                        if not saw_token:
                            self.trace.mark("first_token")
                            saw_token = True
                        if self.echo_tokens:
                            print(token, end="", flush=True)
                        assistant_text.append(token)

                reply = "".join(assistant_text).strip()
                if self._web_synthesis_failed(reply):
                    assistant_text = [self._format_raw_web_evidence(search_evidence)]

                for phrase in chunker.push("".join(assistant_text)):
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
                self.trace.mark("generation_finished", chars=len(reply), text=reply)
                if reply:
                    await self._record_memory_turn(user_text, reply)
                return

            while True:
                tool_calls_made = []
                messages = self._messages_for_generation(memory_context, user_text)
                tools = (
                    self.tool_registry.get_all_schemas()
                    if self.agent_mode == "builtin" and self.tool_registry
                    else None
                )

                async for event in self.agent.stream_chat(messages, tools):
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

    def _recent_history_mentions_gif(self) -> bool:
        """Check if the last few conversation turns were about GIFs."""
        gif_words = ("gif", "giphy", "reaction gif", "meme", "sticker")
        for msg in self.history[-4:]:
            content = str(msg.get("content") or "").lower()
            if any(w in content for w in gif_words):
                return True
        return False

    async def _maybe_run_required_gif_search(self, user_text: str) -> str:
        if "search_gif" not in self.enabled_tools:
            return ""
        if not self.tool_registry or not self.tool_registry.get_tool("search_gif"):
            return ""

        normalized = user_text.lower()
        triggers = (
            "gif",
            "show me a g",
            "send me a g",
            "find me a g",
            "reaction gif",
            "meme",
            "sticker",
            "send a reaction",
            "show a reaction",
            "something funny",
            "make me laugh",
        )
        explicit_match = any(t in normalized for t in triggers)

        # Context-aware: if recent conversation was about GIFs, treat short
        # follow-ups (like "anything", "cats", "a happy one") as GIF requests
        context_match = (
            not explicit_match
            and self._recent_history_mentions_gif()
            and len(user_text.split()) <= 8
        )

        if not explicit_match and not context_match:
            return ""

        query = ""
        for prefix in ("show me a gif of", "send me a gif of", "find me a gif of",
                        "show me a gif about", "send me a gif about",
                        "gif of", "gif for", "gif about",
                        "show me a gif", "send me a gif", "find me a gif",
                        "show me a g of", "send me a g of",
                        "show me a g", "send me a g", "find me a g",
                        "reaction gif for", "reaction gif about", "reaction gif",
                        "meme about", "meme of", "sticker of", "sticker for"):
            idx = normalized.find(prefix)
            if idx != -1:
                query = user_text[idx + len(prefix):].strip().rstrip(".!?,")
                break

        # For context follow-ups, use the whole user text as the query
        if not query and context_match:
            query = user_text.strip().rstrip(".!?,")

        # Fallback: use a few content words if no prefix matched or query is empty
        if not query or len(query) > 80:
            filler = {"show", "me", "a", "the", "an", "of", "for", "about", "that",
                      "is", "going", "to", "make", "want", "see", "find", "send",
                      "please", "can", "you", "i", "yeah", "gif", "funny", "meme",
                      "something", "reaction", "sticker", "it", "with", "some", "get"}
            words = [w for w in normalized.split() if w.strip(".,!?'\"") not in filler]
            query = " ".join(words[:5]) if words else "funny"

        # Hard cap to avoid 414 URI Too Long
        if len(query) > 60:
            query = query[:60].rsplit(" ", 1)[0]

        self.trace.mark("tool_call_forced", tool_name="search_gif")
        tool_call = ToolCallEvent(
            call_id="forced_search_gif",
            name="search_gif",
            arguments={"query": query},
        )
        self.history.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": tool_call.call_id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments),
                },
            }],
        })
        result = await self.tool_registry.execute(tool_call)
        self.history.append({
            "role": "tool",
            "tool_call_id": tool_call.call_id,
            "name": tool_call.name,
            "content": result,
        })
        self.trace.mark("tool_call_finished", tool_name="search_gif", chars=len(result))
        return result

    def _recent_history_mentions_image(self) -> bool:
        """Check if the last few conversation turns were about images/pictures."""
        image_words = ("image", "picture", "photo", "unsplash", "photograph")
        for msg in self.history[-4:]:
            content = str(msg.get("content") or "").lower()
            if any(w in content for w in image_words):
                return True
        return False

    async def _maybe_run_required_image_search(self, user_text: str) -> str:
        if "search_image" not in self.enabled_tools:
            return ""
        if not self.tool_registry or not self.tool_registry.get_tool("search_image"):
            return ""

        normalized = user_text.lower()
        triggers = (
            "show me an image",
            "show me a picture",
            "show me a photo",
            "find an image",
            "find a picture",
            "find a photo",
            "find me an image",
            "find me a picture",
            "find me a photo",
            "get an image",
            "get a picture",
            "get a photo",
            "image of",
            "picture of",
            "photo of",
        )
        explicit_match = any(t in normalized for t in triggers)

        # Context-aware: if recent conversation was about images, treat short
        # follow-ups as image requests
        context_match = (
            not explicit_match
            and self._recent_history_mentions_image()
            and len(user_text.split()) <= 8
        )

        if not explicit_match and not context_match:
            return ""

        # Extract a clean query from the user request
        query = ""
        for prefix in ("show me an image of", "show me a picture of", "show me a photo of",
                        "find me an image of", "find me a picture of", "find me a photo of",
                        "find an image of", "find a picture of", "find a photo of",
                        "get an image of", "get a picture of", "get a photo of",
                        "show me an image", "show me a picture", "show me a photo",
                        "find me an image", "find me a picture", "find an image",
                        "image of", "picture of", "photo of"):
            idx = normalized.find(prefix)
            if idx != -1:
                query = user_text[idx + len(prefix):].strip().rstrip(".!?,")
                break

        # For context follow-ups, use the whole user text as the query
        if not query and context_match:
            query = user_text.strip().rstrip(".!?,")

        # Fallback: extract content words if query is still empty or too long
        if not query or len(query) > 80:
            filler = {"show", "me", "a", "the", "an", "of", "for", "about", "that",
                      "is", "to", "want", "see", "find", "get", "any", "try", "just",
                      "please", "can", "you", "i", "use", "internet", "image", "picture",
                      "photo", "it", "some", "test", "do", "look", "like", "what"}
            words = [w for w in normalized.split() if w.strip(".,!?'\"") not in filler]
            query = " ".join(words[:5]) if words else user_text.strip()[:40]

        if not query:
            query = "cute animal"

        # Hard cap
        if len(query) > 60:
            query = query[:60].rsplit(" ", 1)[0]

        self.trace.mark("tool_call_forced", tool_name="search_image")
        tool_call = ToolCallEvent(
            call_id="forced_search_image",
            name="search_image",
            arguments={"query": query},
        )
        self.history.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": tool_call.call_id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments),
                },
            }],
        })
        result = await self.tool_registry.execute(tool_call)
        self.history.append({
            "role": "tool",
            "tool_call_id": tool_call.call_id,
            "name": tool_call.name,
            "content": result,
        })
        self.trace.mark("tool_call_finished", tool_name="search_image", chars=len(result))
        return result

    async def _maybe_run_required_web_search(self, user_text: str) -> str:
        if "search_web" not in self.enabled_tools:
            return ""
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
        return result

    def _messages_for_web_synthesis(
        self,
        memory_context: str,
        user_text: str,
        search_evidence: str,
    ) -> list[ChatMessage]:
        messages = self._messages_for_generation(memory_context, user_text)
        messages.append(
            {
                "role": "system",
                "content": (
                    "A web search was just completed for the user. The search evidence is "
                    "provided below. Your job is to read the evidence and give a clear, "
                    "direct spoken answer. Rules:\n"
                    "- State the facts from the evidence naturally, as if reading news aloud.\n"
                    "- Name sources when available (e.g. 'According to AP News...').\n"
                    "- NEVER say you cannot browse, search, or access the web. The search "
                    "already happened.\n"
                    "- NEVER say 'based on the limited search results'. Just answer.\n"
                    "- If the evidence genuinely has no answer, say 'The search didn't return "
                    "specific details on that' and share what it did find.\n"
                    "- Keep it concise for voice playback. No bullet points or formatting."
                ),
            }
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"My question: {user_text}\n\n"
                    f"Search evidence:\n{search_evidence}\n\n"
                    "Answer my question using this evidence."
                ),
            }
        )
        return messages

    def _messages_for_media_synthesis(
        self,
        memory_context: str,
        user_text: str,
        media_result: str,
    ) -> list[ChatMessage]:
        is_gif = "gif:" in media_result or "GIPHY" in media_result
        messages = self._messages_for_generation(memory_context, user_text)
        if is_gif:
            messages.append({
                "role": "system",
                "content": (
                    "A GIF was just fetched for the user. Your job is to give a SHORT, "
                    "fun, playful spoken response (1 sentence max). Rules:\n"
                    "- React like a friend sharing a funny GIF: 'Ha! Check this out!' or "
                    "'This one's perfect!' or 'Oh this is so good.'\n"
                    "- Match the energy of the GIF topic.\n"
                    "- DO NOT read out URLs, titles, or attribution text.\n"
                    "- DO NOT describe what happens in the GIF frame by frame.\n"
                    "- Keep it to one short, expressive sentence."
                ),
            })
        else:
            messages.append({
                "role": "system",
                "content": (
                    "An image was just fetched for the user. Your job is to give a SHORT, "
                    "warm, spoken response (1-2 sentences max). Rules:\n"
                    "- Say something brief like 'Here's a beautiful shot of X for you' or "
                    "'I found this lovely image of X'.\n"
                    "- DO NOT read out URLs, photographer names, or attribution text.\n"
                    "- DO NOT describe the image in exhaustive detail.\n"
                    "- DO NOT use bullet points or formatting.\n"
                    "- Keep it natural and conversational, as if showing a friend a photo."
                ),
            })
        messages.append({
            "role": "user",
            "content": (
                f"My request: {user_text}\n\n"
                f"Media data returned:\n{media_result}\n\n"
                "Give me a brief spoken reaction."
            ),
        })
        return messages

    def _web_synthesis_failed(self, reply: str) -> bool:
        normalized = reply.strip().lower()
        if not normalized:
            return True
        failure_phrases = (
            "couldn't get",
            "could not get",
            "can't browse",
            "cannot browse",
            "i don't have access",
            "i do not have access",
            "try again",
            "not available right now",
            "was not available",
            "i'm unable to",
            "i am unable to",
            "i cannot provide",
            "i can't provide",
            "direct result",
            "i'm not able to",
        )
        return any(phrase in normalized for phrase in failure_phrases)

    def _format_raw_web_evidence(self, search_evidence: str) -> str:
        return f"I searched the web. Here are the top results I found:\n\n{search_evidence}"

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

    def _should_auto_followup(self) -> bool:
        non_system = [message for message in self.history if message["role"] != "system"]
        if not non_system:
            return True
        return non_system[-1]["role"] == "assistant"

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
        if (
            self.agent_mode == "hermes"
            or self._suppress_memory_write
            or not self.memory_config.enabled
        ):
            return
        self.trace.mark("memory_write_started", chars=len(user_text) + len(assistant_text))
        try:
            await self.memory_store.record_turn(user_text, assistant_text)
            self.trace.mark("memory_write_finished")
        except Exception as exc:
            self.trace.mark("memory_write_failed", error=str(exc))

    def _messages_for_generation(self, memory_context: str, user_text: str) -> list[ChatMessage]:
        if self.agent_mode == "hermes":
            return [{"role": "user", "content": user_text}]

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
