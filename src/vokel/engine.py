from __future__ import annotations

import asyncio
import datetime
import inspect
import json
import re
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Literal

from .agent_backend import AgentBackend, TOOL_ACTIVITY_REPORTING_CONTRACT
from .config import VoiceLoopConfig
from .events import TextDeltaEvent, ToolActivityEvent, ToolCallEvent
from .hermes_client import HermesAgentClient
from .inference import ChatMessage
from .memory import MemoryConfig, MemoryStore, NullMemoryStore, build_memory_context
from .playback import PlaybackSink
from .telemetry import LatencyTrace
from .text_chunker import PhraseChunker
from .tools import ToolRegistry
from .auto_followup import AUTO_FOLLOWUP_NUDGE
from .turns import AsrEngine, TurnProducer
from .vision import SpokenReply, VisualContext
from typing import Any


AgentMode = Literal["builtin", "hermes"]
MEDIA_MARKDOWN_RE = re.compile(r"!\[[^\]]*\]\((https?://[^)]+)\)")
TOOL_CALL_MARKER_RE = re.compile(r"\[tool_call:([^\]]+)\]")
# A serialized tool call the model leaked as plain text. Small local models emit
# two shapes, and the closer-required pattern we used before let the second leak:
#   1. `<|tool_call>call: camera.capture()`                  — opener, optional body, no closer
#   2. `call:describe_image{image_base64:<|"|>...}<tool_call|>` — no opener, {} body, closer
# Branch 1 keys on the opener, so a bare opener (and any optional body) is caught.
# Branch 2 has no opener, so it requires at least one (...)/{...} arg group as the
# anchor — that keeps natural prose like "give me a call: tomorrow" untouched.
SERIALIZED_TOOL_CALL_RE = re.compile(
    r"<\|tool_call\|?>\s*"
    r"(?:call\s*:\s*[\w.:-]+)?"
    r"(?:\s*(?:\([^)]*\)|\{[^}]*\}))*"
    r"(?:\s*<\|?/?tool_call\|?>)?"
    r"|"
    r"call\s*:\s*[\w.:-]+"
    r"(?:\s*(?:\([^)]*\)|\{[^}]*\}))+"
    r"(?:\s*<\|?/?tool_call\|?>)?",
    re.DOTALL,
)
# The function name inside a matched tool call, used only for spoken reassurance.
# Searched within a SERIALIZED match (never the raw token) so prose isn't mined.
_TOOL_CALL_NAME_RE = re.compile(r"call\s*:\s*([\w.:-]+)")
# Mop up orphaned delimiters and the stray `<|"|>` quote token the model wraps
# around base64 image args, so a split or partial emission never reaches output.
RESIDUAL_TOOL_TOKEN_RE = re.compile(r"<\|?/?tool_call\|?>|<\|[\"']\|>", re.I)
_PUNCT_ONLY_RE = re.compile(r"^[\s\.\,\!\?\:\;\-\_~…·。！？、]+$")

# Spoken when a visual turn yields only stripped tool-call scaffolding, so an
# image question never ends in silence (a known small-model failure mode).
VISUAL_REPLY_FALLBACK = (
    "Sorry, I had the camera frame but didn't catch a clear answer that time. "
    "Could you ask me again?"
)
# Appended to the prompt that rides with a camera frame, steering small models to
# answer in words instead of echoing a describe_image / camera tool call.
VISUAL_ANSWER_DIRECTIVE = (
    "Look at the attached image and answer in one or two plain spoken sentences. "
    "Do not output any tool calls, function calls, JSON, or code."
)


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
        visual_context_provider: Callable[
            [str], Awaitable[str | VisualContext | SpokenReply | None]
        ]
        | None = None,
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
        # Visual context (explicit armed camera) is now supported for Hermes too via the
        # camera_frame payload contract. The provider itself (capture + consent UI) lives
        # in the host layer (web.py) and is passed for both modes.
        self.visual_context_provider = visual_context_provider
        self.history: list[ChatMessage] = [
            {"role": "system", "content": self._session_system_prompt()}
        ]
        self._playback_queue: asyncio.Queue[str] = asyncio.Queue()
        self._current_generation: asyncio.Task[None] | None = None
        self._playback_worker: asyncio.Task[None] | None = None
        self._pending_visual_capture_task: asyncio.Task | None = None
        self._suppress_memory_write = False

    async def _handle_tool_markers(
        self,
        token: str,
        reassured_tools: set[str],
        *,
        queue_reassurance: bool,
    ) -> str:
        marker_names = list(TOOL_CALL_MARKER_RE.findall(token))
        for match in SERIALIZED_TOOL_CALL_RE.finditer(token):
            name_match = _TOOL_CALL_NAME_RE.search(match.group(0))
            if name_match:
                marker_names.append(name_match.group(1))
        for tool_name in marker_names:
            if not tool_name:
                # A bare opener with no `call: name` carries no tool to reassure about.
                continue
            self.trace.mark("tool_call_text_suppressed", tool_name=tool_name)
            if queue_reassurance and tool_name not in reassured_tools:
                reassured_tools.add(tool_name)
                self.trace.mark("tool_call_started", tool_name=tool_name)
                await self._queue_tool_reassurance(tool_name)
        token = TOOL_CALL_MARKER_RE.sub("", token)
        token = SERIALIZED_TOOL_CALL_RE.sub("", token)
        token = RESIDUAL_TOOL_TOKEN_RE.sub("", token)
        return token

    async def start(self) -> None:
        if self._playback_worker is None or self._playback_worker.done():
            self._playback_worker = asyncio.create_task(self._run_playback())

    async def close(self) -> None:
        await self.interrupt()
        # Ensure any stray pending capture task is cleaned (interrupt should have done it)
        if self._pending_visual_capture_task and not self._pending_visual_capture_task.done():
            self._pending_visual_capture_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._pending_visual_capture_task
        self._pending_visual_capture_task = None
        if self._playback_worker:
            self._playback_worker.cancel()
            with suppress(asyncio.CancelledError):
                await self._playback_worker
        await self.memory_store.close()

    async def wait_for_playback(self) -> None:
        await self._playback_queue.join()

    async def reset_conversation(self) -> None:
        await self.interrupt()
        self.history = [{"role": "system", "content": self._session_system_prompt()}]
        if isinstance(self.agent, HermesAgentClient):
            new_session = self.agent.reset_session()
            self.trace.mark("hermes_session_reset", session_id=new_session)
        elif hasattr(self.agent, "reset_conversation"):
            # Native LM client (and future) continuity reset
            self.agent.reset_conversation()
            self.trace.mark("native_lm_conversation_reset")
        self.trace.reset()
        self.trace.mark("conversation_reset")

    async def submit_turn(self, user_text: str, reset_trace: bool = True) -> None:
        if self._is_ambiguous_filler_utterance(user_text):
            if reset_trace:
                self.trace.reset()
            self.trace.mark("utterance_ignored", reason="ambiguous_filler", text=user_text)
            return
        if self.visual_context_provider:
            # Track the capture task separately so interrupt() can cancel it *before*
            # any result (even late) reaches submit_visual_turn or a backend.
            self._pending_visual_capture_task = asyncio.create_task(
                self.visual_context_provider(user_text)
            )
            try:
                provided = await self._pending_visual_capture_task
            finally:
                self._pending_visual_capture_task = None

            if provided is not None:
                if isinstance(provided, SpokenReply):
                    # Deterministic camera-control acknowledgement ("watch me", "action",
                    # "cut"): Vokel speaks the line itself so the model never narrates or
                    # invents a camera capability.
                    await self._submit_canned_reply(
                        user_text, provided.text, reset_trace=reset_trace
                    )
                    return
                if isinstance(provided, VisualContext):
                    vc = provided
                    if vc.data_url:
                        await self.submit_visual_turn(
                            user_text, vc.data_url, reset_trace=reset_trace, visual_context=vc
                        )
                    else:
                        reason = (
                            "not_armed"
                            if vc.consent == "voice_context_not_armed"
                            else "capture_failed"
                        )
                        await self._submit_visual_capture_failure(
                            user_text,
                            reset_trace=reset_trace,
                            reason=reason,
                        )
                elif provided:
                    # backward compat for tests/providers returning plain data_url str
                    await self.submit_visual_turn(user_text, provided, reset_trace=reset_trace)
                else:
                    await self._submit_visual_capture_failure(user_text, reset_trace=reset_trace)
                return
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

    async def _submit_canned_reply(
        self,
        user_text: str,
        reply: str,
        *,
        reset_trace: bool,
    ) -> None:
        """Record the turn and speak a fixed line without calling the model."""
        await self.interrupt()
        if reset_trace:
            self.trace.reset()
        self.trace.mark("turn_submitted", chars=len(user_text), camera_directive=True)
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})
        self._trim_history()
        await self._playback_queue.put(reply)
        self.trace.mark("generation_finished", chars=len(reply), text=reply)

    async def _submit_visual_capture_failure(
        self,
        user_text: str,
        *,
        reset_trace: bool,
        reason: str = "capture_failed",
    ) -> None:
        await self.interrupt()
        if reset_trace:
            self.trace.reset()
        if reason == "not_armed":
            reply = (
                "I need Camera Questions in Voice Loop turned on before I can look. "
                "Enable it in the Live Feed panel, then ask again."
            )
        else:
            reply = "I couldn't capture a camera frame, so I can't answer that visual question yet."
        self.trace.mark("turn_submitted", chars=len(user_text), visual_context=True)
        self.trace.mark("visual_context_unavailable")
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})
        self._trim_history()
        await self._playback_queue.put(reply)
        self.trace.mark("generation_finished", chars=len(reply), text=reply)

    async def submit_visual_turn(
        self,
        user_text: str,
        image_data_url: str,
        reset_trace: bool = True,
        *,
        visual_context: VisualContext | None = None,
    ) -> None:
        """Submit one local camera frame with the current spoken question.

        Supported for both builtin (LM Studio compat or native) and Hermes (via explicit
        camera_frame payload contract using VisualContext). The caller (web layer) is
        responsible for visible routing, consent, audit, and cancellation.
        """
        await self.interrupt()
        if reset_trace:
            self.trace.reset()
        self.trace.mark("turn_submitted", chars=len(user_text), visual_context=True)
        self.trace.mark("visual_context_attached", chars=len(image_data_url))
        memory_context = await self._retrieve_memory_context(user_text)
        self.history.append({"role": "user", "content": user_text})
        self._trim_history()

        img: dict[str, Any] = {"url": image_data_url}
        if visual_context is not None:
            img["visual_context"] = {
                "source": visual_context.source,
                "captured_at": visual_context.captured_at,
                "consent": visual_context.consent,
                "contract": visual_context.contract,
            }

        # Bare commands ("shoot") carry a clean instruction in visual_context.prompt;
        # spoken questions ("what am I holding?") use the words as the prompt directly.
        frame_prompt = (
            visual_context.prompt
            if visual_context is not None and visual_context.prompt
            else user_text
        )
        # The directive (invisible to the transcript) keeps small models from echoing
        # a describe_image / camera tool call instead of just answering the question.
        visual_message: ChatMessage = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": img},
                {"type": "text", "text": f"{frame_prompt}\n\n{VISUAL_ANSWER_DIRECTIVE}"},
            ],
        }
        self._current_generation = asyncio.create_task(
            self._generate_reply(
                user_text,
                memory_context,
                current_user_message=visual_message,
                allow_tools=False,
            )
        )
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
            result = cancel()
            if inspect.isawaitable(result):
                await result

        # Cancel any in-flight visual capture *first* so late GStreamer completion
        # cannot produce a frame that reaches submit_visual_turn or backend.
        if self._pending_visual_capture_task and not self._pending_visual_capture_task.done():
            self.trace.mark("visual_capture_interrupted")
            self._pending_visual_capture_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._pending_visual_capture_task
            self._pending_visual_capture_task = None

        if self._current_generation and not self._current_generation.done():
            self.trace.mark("interruption_requested")
            self._current_generation.cancel()
            with suppress(asyncio.CancelledError):
                await self._current_generation

        self._drain_playback_queue()
        await self.playback.stop()
        self.trace.mark("playback_stop_requested")

    async def _generate_reply(
        self,
        user_text: str,
        memory_context: str = "",
        *,
        current_user_message: ChatMessage | None = None,
        allow_tools: bool = True,
    ) -> None:
        chunker = PhraseChunker()
        assistant_text: list[str] = []
        saw_token = False
        saw_phrase = False
        reassured_tools: set[str] = set()

        self.trace.mark("generation_started")

        try:
            media_result = ""
            search_evidence = ""
            if self.agent_mode == "builtin" and allow_tools:
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
                        token = await self._handle_tool_markers(
                            token,
                            reassured_tools,
                            queue_reassurance=self.agent_mode == "hermes",
                        )
                        if not token:
                            continue
                        if not saw_token:
                            self.trace.mark("first_token")
                            saw_token = True
                        if self.echo_tokens:
                            print(token, end="", flush=True)
                        spoken_parts.append(token)

                spoken = "".join(spoken_parts).strip()
                if not spoken or len(spoken) < 5:
                    spoken = "Here you go!"

                media_block = self._first_media_markdown_block(media_result)
                if media_block:
                    # Transcript gets media markdown + caption; TTS only gets the caption.
                    full_reply = f"{media_result}\n\n{spoken}"
                else:
                    # Transparent fallback: never claim media was shown when no renderable block exists.
                    full_reply = (
                        "I tried to fetch that media, but no displayable image or GIF URL was returned. "
                        "Want me to try again with a different query?"
                    )
                    spoken = full_reply
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

            if self.agent_mode == "builtin" and allow_tools:
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
                messages = self._messages_for_generation(
                    memory_context,
                    user_text,
                    current_user_message=current_user_message,
                    include_tools=allow_tools,
                )
                tools = (
                    self.tool_registry.get_all_schemas()
                    if self.agent_mode == "builtin" and self.tool_registry and allow_tools
                    else None
                )

                async for event in self.agent.stream_chat(messages, tools):
                    if isinstance(event, TextDeltaEvent):
                        token = event.content
                        token = await self._handle_tool_markers(
                            token,
                            reassured_tools,
                            queue_reassurance=self.agent_mode == "hermes",
                        )
                        if not token:
                            continue
                        if not saw_token:
                            self.trace.mark("first_token")
                            saw_token = True
                        if self.echo_tokens:
                            print(token, end="", flush=True)
                        assistant_text.append(token)
                        if self.agent_mode == "hermes":
                            continue
                        for phrase in chunker.push(token):
                            if not saw_phrase:
                                self.trace.mark("first_phrase_queued", chars=len(phrase))
                                saw_phrase = True
                            await self._playback_queue.put(phrase)
                    elif isinstance(event, ToolActivityEvent):
                        status = getattr(event, "status", "started")
                        if status == "started":
                            if event.name not in reassured_tools:
                                reassured_tools.add(event.name)
                                self.trace.mark("tool_call_started", tool_name=event.name)
                                await self._queue_tool_reassurance(event.name)
                        elif status in ("finished", "failed"):
                            self.trace.mark(
                                "tool_call_finished" if status == "finished" else "tool_call_failed",
                                tool_name=event.name,
                                status=status,
                            )
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
                            await self._queue_tool_reassurance(tc.name)
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

            if self.agent_mode == "hermes" and assistant_text:
                phrase = "".join(assistant_text).strip()
                if phrase:
                    if not saw_phrase:
                        self.trace.mark("first_phrase_queued", chars=len(phrase))
                        saw_phrase = True
                    await self._playback_queue.put(phrase)
                chunker.reset()

            final_phrase = chunker.flush()
            if final_phrase:
                if not saw_phrase:
                    self.trace.mark("first_phrase_queued", chars=len(final_phrase))
                await self._playback_queue.put(final_phrase)

            reply = self._strip_serialized_tool_calls("".join(assistant_text).strip())
            used_visual_fallback = False
            if not reply and self._is_visual_turn(current_user_message):
                # The model emitted only tool-call scaffolding (now stripped) instead of
                # an answer — a known small-model failure when an image is in context.
                # Speak a graceful fallback rather than going silent on a visual question.
                reply = VISUAL_REPLY_FALLBACK
                used_visual_fallback = True
                self.trace.mark("visual_reply_fallback_used")
                await self._playback_queue.put(reply)
            if reply:
                self.history.append({"role": "assistant", "content": reply})
                self._trim_history()

            self.trace.mark("generation_finished", chars=len(reply), text=reply)

            if reply and not used_visual_fallback:
                await self._record_memory_turn(user_text, reply)
        except asyncio.CancelledError:
            chunker.reset()
            self.trace.mark("generation_cancelled")
            raise

    @staticmethod
    def _strip_serialized_tool_calls(text: str) -> str:
        cleaned = TOOL_CALL_MARKER_RE.sub("", text)
        cleaned = SERIALIZED_TOOL_CALL_RE.sub("", cleaned)
        cleaned = RESIDUAL_TOOL_TOKEN_RE.sub("", cleaned)
        return re.sub(r"\s{2,}", " ", cleaned).strip()

    def _session_system_prompt(self) -> str:
        """Base system prompt grounded with the real current date.

        A local model has no clock, so without this it confidently guesses the date
        (the live build answered "today is October 24th, 2024"). Computed when the
        history is built or reset so each session reflects the actual day.
        """
        today = datetime.datetime.now().astimezone()
        date_line = (
            f" For reference, today's date is {today.strftime('%A, %B')} {today.day}, "
            f"{today.year}. Treat this as the authoritative current date and never "
            "state or guess a different one."
        )
        return self.config.system_prompt + date_line

    @staticmethod
    def _is_visual_turn(current_user_message: ChatMessage | None) -> bool:
        """True when this turn carried a camera frame (image_url) to the model."""
        if not current_user_message:
            return False
        content = current_user_message.get("content")
        if not isinstance(content, list):
            return False
        return any(
            isinstance(part, dict) and part.get("type") == "image_url" for part in content
        )

    @staticmethod
    def _first_media_markdown_block(text: str) -> str | None:
        match = MEDIA_MARKDOWN_RE.search(text or "")
        if not match:
            return None
        return match.group(0)

    @staticmethod
    def _is_ambiguous_filler_utterance(user_text: str) -> bool:
        text = (user_text or "").strip()
        if not text:
            return True
        if _PUNCT_ONLY_RE.fullmatch(text):
            return True

        normalized = " ".join(text.lower().split())
        filler_tokens = {
            "um",
            "uh",
            "hmm",
            "mmm",
            "mm",
            "umm",
            "uhh",
            "erm",
            "huh",
            "eh",
            "嗯",
            "嗯嗯",
            "呃",
            "啊",
        }
        tokens = normalized.split()
        if tokens and all(token in filler_tokens for token in tokens):
            return True
        return False

    def _recent_history_mentions_gif(self) -> bool:
        """Check if the last few conversation turns were about GIFs."""
        gif_words = ("gif", "giphy", "reaction gif", "meme", "sticker")
        for msg in self.history[-4:]:
            content = str(msg.get("content") or "").lower()
            if any(w in content for w in gif_words):
                return True
        return False

    @staticmethod
    def _looks_like_media_followup(user_text: str) -> bool:
        """Only treat short follow-ups as media requests when intent is still obvious."""
        text = user_text.strip().lower()
        if not text:
            return False
        # Direct media words still count as clear intent.
        media_words = ("gif", "image", "picture", "photo", "meme", "sticker")
        if any(word in text for word in media_words):
            return True
        # Lightweight "another one like that" style follow-ups.
        followup_phrases = (
            "another",
            "one like",
            "like that",
            "like this",
            "more like",
            "same vibe",
            "same style",
            "same energy",
        )
        return any(phrase in text for phrase in followup_phrases)

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
            and self._looks_like_media_followup(user_text)
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
        await self._queue_tool_reassurance("search_gif")
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
            and self._looks_like_media_followup(user_text)
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
        await self._queue_tool_reassurance("search_image")
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

        if not self._should_force_web_search(user_text):
            return ""

        self.trace.mark("tool_call_forced", tool_name="search_web")
        await self._queue_tool_reassurance("search_web")
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

    async def _queue_tool_reassurance(self, tool_name: str) -> None:
        phrase_by_tool = {
            "search_web": "I'm searching now.",
            "search_image": "I'm fetching that image now.",
            "search_gif": "I'm fetching that GIF now.",
        }
        phrase = phrase_by_tool.get(tool_name, "I'm using that tool now.")
        self.trace.mark("tool_reassurance_queued", tool_name=tool_name, chars=len(phrase))
        await self._playback_queue.put(phrase)

    @staticmethod
    def _should_force_web_search(user_text: str) -> bool:
        normalized = " ".join(user_text.lower().split())
        if not normalized:
            return False

        explicit_lookup_phrases = (
            "search the web for",
            "search web for",
            "search for",
            "look up",
            "find me information about",
            "find information about",
            "find me info about",
            "look up current",
            "search latest",
            "latest on",
        )
        if any(phrase in normalized for phrase in explicit_lookup_phrases):
            return True

        meta_or_planning_phrases = (
            "i'm going to",
            "i am going to",
            "we can",
            "we should",
            "worked well",
            "work well",
            "improve",
            "later",
            "soon",
            "set up the api",
            "setup the api",
            "enhance",
            "future plan",
            "next step",
        )
        if any(phrase in normalized for phrase in meta_or_planning_phrases):
            return False

        is_question = (
            "?" in user_text
            or normalized.startswith(("what", "who", "when", "where", "why", "how"))
        )
        current_info_cues = (
            "latest",
            "current",
            "today",
            "news",
            "weather",
            "breaking",
            "real-time",
            "recent",
            "headline",
            "headlines",
        )
        has_current_info_cue = any(cue in normalized for cue in current_info_cues)
        if is_question and has_current_info_cue:
            return True

        words = normalized.split()
        if len(words) <= 10 and has_current_info_cue and not normalized.startswith(
            ("i ", "we ", "that ", "this ", "it ")
        ):
            return True
        return False

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
                    "fun, playful spoken response (1-2 short sentences). Rules:\n"
                    "- React like a friend sharing a funny GIF: 'Ha! Check this out!' or "
                    "'This one's perfect!' or 'Oh this is so good.'\n"
                    "- Match the energy of the GIF topic.\n"
                    "- DO NOT read out URLs, titles, or attribution text.\n"
                    "- DO NOT describe what happens in the GIF frame by frame.\n"
                    "- Keep it brief and expressive, but do not sound abruptly cut off."
                ),
            })
        else:
            messages.append({
                "role": "system",
                "content": (
                    "An image was just fetched for the user. Your job is to give a SHORT, "
                    "warm, spoken response (1-2 short sentences). Rules:\n"
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

    def _messages_for_generation(
        self,
        memory_context: str,
        user_text: str,
        *,
        current_user_message: ChatMessage | None = None,
        include_tools: bool = True,
    ) -> list[ChatMessage]:
        if self.agent_mode == "hermes":
            # Preserve multimodal current_user_message (e.g. visual with image_url block)
            # so that extract_camera_frame() in Hermes clients receives the armed frame.
            # Hermes owns full history/session, but the session system prompt (persona,
            # musical addendum, session topic) must still ride along — the Hermes
            # clients join all system messages into per-request instructions.
            if current_user_message is not None:
                return [
                    {"role": "system", "content": self._session_system_prompt()},
                    {"role": "system", "content": TOOL_ACTIVITY_REPORTING_CONTRACT},
                    current_user_message,
                ]
            return [
                {"role": "system", "content": self._session_system_prompt()},
                {"role": "system", "content": TOOL_ACTIVITY_REPORTING_CONTRACT},
                {"role": "user", "content": user_text},
            ]

        if not memory_context:
            messages = list(self.history)
        else:
            system = self.history[:1]
            rest = self.history[1:]
            memory_message: ChatMessage = {"role": "system", "content": memory_context}
            messages = system + [memory_message] + rest

        if current_user_message is not None and messages[-1]["role"] == "user":
            messages[-1] = current_user_message

        if self.tool_registry and include_tools:
            tool_names = ", ".join(s["function"]["name"] for s in self.tool_registry.get_all_schemas())
            if tool_names:
                instruction = (
                    f"You have access to the following tools: {tool_names}. "
                    "You MUST use these tools when the user asks for real-time information, web searches, or news. "
                    "DO NOT pretend or hallucinate search results. You MUST call the tool. "
                    f"{TOOL_ACTIVITY_REPORTING_CONTRACT}"
                )
                messages.insert(1, {"role": "system", "content": instruction})
                
        return messages
