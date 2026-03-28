"""Dialogue manager — routes speech to the brain, executes actions, speaks responses.

Simplified for the state machine architecture:
  STT text → Brain (with contextual tools) → tool execution → TTS
  State machine handles control acquire/release.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import structlog

from src.brain import BrainResponse

if TYPE_CHECKING:
    from src.brain import Brain
    from src.butler_client import ButlerClient
    from src.memory import MemoryStore
    from src.tts import TTSClient
    from src.vector_control import VectorController
    from src.vision import VisionPipeline

log = structlog.get_logger()


class ConversationManager:
    """Manages dialogue flow between the user, brain, and TTS."""

    def __init__(
        self,
        config: dict,
        *,
        brain: Brain,
        tts: TTSClient | None = None,
        vector: VectorController | None = None,
        vision: VisionPipeline | None = None,
        butler: ButlerClient | None = None,
        memory: MemoryStore | None = None,
    ) -> None:
        self._brain = brain
        self._tts = tts
        self._vector = vector
        self._vision = vision
        self._butler = butler
        self._memory = memory

        self._active = False
        self._last_spoke_at: float = 0.0

        # Lock to prevent concurrent processing of transcriptions.
        self._processing_lock = asyncio.Lock()

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def last_spoke_at(self) -> float:
        return self._last_spoke_at

    async def handle_transcription(
        self, text: str, *, timestamp: float = 0.0
    ) -> BrainResponse | None:
        """Process a new transcription from STT."""
        async with self._processing_lock:
            # Drop stale transcriptions.
            if timestamp and (time.perf_counter() - timestamp) > 10.0:
                age = time.perf_counter() - timestamp
                log.info("conversation.dropped_stale", age_s=f"{age:.1f}", text=text[:40])
                return None

            if not self._active:
                self._active = True
                log.info("conversation.started", trigger=text[:60])

            # Load memory context.
            await self._load_memory_context(text)

            # Get brain response.
            response = await self._brain.think(text)

            # Speak the initial response.
            if response.speech:
                await self._speak(response.speech)

            # Execute tool calls.
            response, _spoke = await self._execute_tools(response)

            # Persist history in background.
            speech = response.speech or (response.speech if _spoke else "")
            if self._memory and speech:
                asyncio.create_task(self._persist_history(text, speech))

            return response

    async def handle_learning(self, event_summary: str) -> BrainResponse | None:
        """React to something interesting — brief, may comment."""
        async with self._processing_lock:
            response = await self._brain.handle_vision_event(event_summary)

            if response is None:
                return None

            if response.speech:
                await self._speak(response.speech)

            return response

    async def _execute_tools(self, response: BrainResponse) -> tuple[BrainResponse, bool]:
        """Execute tool calls, max 2 rounds."""
        max_rounds = 2
        current = response
        spoke = False
        look_used = False

        for _ in range(max_rounds):
            if not current.tool_calls:
                break

            for tool_call in current.tool_calls:
                tool_name = tool_call["name"]
                params = tool_call.get("parameters", {})

                if tool_name == "look":
                    if look_used:
                        continue
                    look_used = True

                result = await self._dispatch_tool(tool_name, params)

                log.info(
                    "conversation.tool_executed",
                    tool=tool_name,
                    result=result[:80],
                )

                current = await self._brain.handle_tool_result(tool_name, result)

                if current.tool_calls:
                    current.tool_calls = [
                        tc for tc in current.tool_calls if tc["name"] != "look"
                    ]

                if current.speech:
                    await self._speak(current.speech)
                    spoke = True

        return current, spoke

    async def _dispatch_tool(self, tool_name: str, params: dict) -> str:
        """Route tool calls to the appropriate handler."""
        if tool_name == "look":
            if self._vision:
                return await self._vision.describe_scene()
            return "[no vision available]"

        if tool_name == "remember":
            fact = params.get("fact", "")
            category = params.get("category", "other")
            if self._memory and self._memory.available:
                await self._memory.remember(fact, category=category)
                return f"Remembered: {fact}"
            return "[memory not available]"

        if tool_name == "recall":
            query = params.get("query", "")
            if self._memory and self._memory.available:
                facts = await self._memory.recall(query, limit=5)
                if facts:
                    return "\n".join(f"- [{f['category']}] {f['fact']}" for f in facts)
                return "No memories found."
            return "[memory not available]"

        if tool_name == "ask_butler":
            question = params.get("question", "")
            if self._butler:
                try:
                    return await self._butler.ask(question)
                except Exception as e:
                    return f"[Butler unavailable: {e}]"
            return "[Butler not configured]"

        # Physical tools go through VectorController.
        if self._vector:
            return await self._vector.execute_tool(tool_name, params)

        return f"[{tool_name} unavailable]"

    async def _speak(self, text: str) -> None:
        """Speak text via TTS, show emoji on screen briefly."""
        from src.screen import extract_emoji, render_emoji

        self._last_spoke_at = time.monotonic()

        # Extract emoji before TTS.
        _, emojis = extract_emoji(text)

        # Show emoji on screen while speaking (fire-and-forget, don't block TTS).
        if emojis and self._vector:
            try:
                img = render_emoji(emojis)
                asyncio.create_task(self._vector.display_on_screen(img, duration_sec=1.0))
            except Exception:
                pass

        if self._tts:
            try:
                await self._tts.speak(text)
            except Exception:
                log.exception("tts.speak failed, falling back to vector")
                if self._vector:
                    try:
                        await self._vector.say(text)
                    except Exception:
                        log.exception("vector.say failed")

    async def _load_memory_context(self, text: str) -> None:
        """Load relevant memories into the brain's system prompt."""
        if not self._memory or not self._memory.available:
            return
        try:
            ctx = await self._memory.load_context(current_message=text)
            self._brain.set_memory_context(ctx)
        except Exception:
            log.exception("memory.load_context_failed")

    async def _persist_history(self, user_text: str, assistant_text: str) -> None:
        """Store conversation messages in Postgres."""
        if not self._memory:
            return
        try:
            await self._memory.store_message("user", user_text)
            await self._memory.store_message("assistant", assistant_text)
        except Exception:
            log.exception("memory.persist_failed")

    def end_conversation(self) -> None:
        """End the current conversation and reset brain context."""
        if not self._active:
            return
        self._active = False
        self._brain.reset_context()
        log.info("conversation.ended")
