"""Dialogue manager — routes speech to the brain, executes actions, speaks responses.

Orchestrates the full loop:
  STT text → direction check → Brain → tool execution → TTS
  Vision events → Brain → optional response → TTS
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

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

# Default follow-up window (kept for timeout logic).
_DEFAULT_FOLLOWUP_WINDOW_S = 10.0


class ConversationManager:
    """Manages dialogue flow between the user, brain, and TTS.

    Decides whether speech is directed at Vector, maintains conversation
    state, and orchestrates the brain → tool → speech loop.
    """

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
        thresholds = config.get("thresholds", {})
        self.timeout_s: float = thresholds.get("conversation_timeout_seconds", 30.0)

        self._brain = brain
        self._tts = tts
        self._vector = vector
        self._vision = vision
        self._butler = butler
        self._memory = memory

        self._active = False
        self._last_spoke_at: float = 0.0
        self._last_interaction_at: float = 0.0

        # Lock to prevent concurrent processing of transcriptions.
        self._processing_lock = asyncio.Lock()

    @property
    def is_active(self) -> bool:
        """Whether a conversation is currently active."""
        return self._active

    @property
    def last_spoke_at(self) -> float:
        """Timestamp of Vector's last speech."""
        return self._last_spoke_at

    async def handle_transcription(self, text: str) -> BrainResponse | None:
        """Process a new transcription from STT.

        Args:
            text: Transcribed speech text.

        Returns:
            BrainResponse if Vector responds, None if speech was ignored.
        """
        async with self._processing_lock:
            return await self._handle_transcription_inner(text)

    async def _handle_transcription_inner(self, text: str) -> BrainResponse | None:
        now = time.monotonic()

        # Check for conversation timeout — reset context after long silence.
        if self._active and self._last_interaction_at > 0:
            if (now - self._last_interaction_at) > self.timeout_s:
                self._end_conversation()

        # Always-on: every transcription is treated as directed at us.
        if not self._active:
            self._active = True
            log.info("conversation.started", trigger=text[:60])

        self._ignored_count = 0
        self._last_interaction_at = now

        # Load memory context before brain call.
        await self._load_memory_context(text)

        # Get brain response.
        response = await self._brain.think(text)

        # Execute tool calls and follow up.
        response, spoke = await self._execute_tools(response)

        # Speak the response (skip if tool chain already spoke it).
        if response.speech and not spoke:
            await self._speak(response.speech)

        # Store conversation and auto-learn in background.
        if self._memory and response.speech:
            asyncio.create_task(self._persist_exchange(text, response.speech))

        return response

    async def handle_vision_event(self, event_summary: str) -> BrainResponse | None:
        """Process a vision event and optionally respond.

        Vision events get the scene context already injected into the system
        prompt (including the last VLM description), so the brain should NOT
        need to call 'look' again. We strip look calls to prevent the
        look → speak → look infinite chain.

        Args:
            event_summary: Text summary of the vision event.

        Returns:
            BrainResponse if Vector reacts, None if ignored.
        """
        async with self._processing_lock:
            response = await self._brain.handle_vision_event(event_summary)

            if response is None:
                return None

            # Strip 'look' tool calls from vision reactions — the brain
            # already has scene context and doesn't need to look again.
            if response.tool_calls:
                response.tool_calls = [
                    tc for tc in response.tool_calls if tc["name"] != "look"
                ]

            # Execute remaining tool calls (movement, animations, etc).
            response, spoke = await self._execute_tools(response)

            if response.speech and not spoke:
                await self._speak(response.speech)

            return response

    async def _execute_tools(self, response: BrainResponse) -> tuple[BrainResponse, bool]:
        """Execute tool calls from a brain response and feed results back.

        Processes tools sequentially — each result may trigger follow-up
        tool calls from the brain.

        Args:
            response: Brain response potentially containing tool calls.

        Returns:
            Tuple of (final BrainResponse, whether speech was already spoken).
        """
        max_rounds = 5  # Prevent infinite tool loops.
        current = response
        spoke = False

        for _ in range(max_rounds):
            if not current.tool_calls:
                break

            for tool_call in current.tool_calls:
                tool_name = tool_call["name"]
                params = tool_call.get("parameters", {})

                result = await self._dispatch_tool(tool_name, params)

                log.info(
                    "conversation.tool_executed",
                    tool=tool_name,
                    result=result[:80],
                )

                # Feed result back to brain for follow-up.
                current = await self._brain.handle_tool_result(tool_name, result)

                # If follow-up has speech, speak it before next tool.
                if current.speech:
                    await self._speak(current.speech)
                    spoke = True

        return current, spoke

    async def _dispatch_tool(self, tool_name: str, params: dict) -> str:
        """Dispatch a single tool call to the appropriate handler.

        Args:
            tool_name: Name of the tool to call.
            params: Tool parameters.

        Returns:
            Result string for feeding back to the brain.
        """
        # "look" tool uses the vision pipeline.
        if tool_name == "look":
            if self._vision:
                description = await self._vision.describe_scene()
                return description
            return "[no vision available]"

        # "ask_butler" escalates to Claude via Butler API.
        if tool_name == "ask_butler":
            question = params.get("question", "")
            if self._butler:
                try:
                    return await self._butler.ask(question)
                except Exception as e:
                    log.warning("butler.escalation_failed", error=str(e))
                    return f"[Butler unavailable: {e}]"
            return "[Butler not configured]"

        # All other tools go through VectorController.
        if self._vector:
            return await self._vector.execute_tool(tool_name, params)

        return f"[{tool_name} unavailable — no vector controller]"

    async def _speak(self, text: str) -> None:
        """Speak text via TTS or fallback to Vector's built-in TTS.

        Args:
            text: Text to speak.
        """
        self._last_spoke_at = time.monotonic()

        if self._tts:
            try:
                await self._tts.speak(text)
                return
            except NotImplementedError:
                pass
            except Exception:
                log.exception("tts.speak failed, falling back to vector")

        # Fallback to Vector's built-in TTS.
        if self._vector:
            try:
                await self._vector.say(text)
                return
            except Exception:
                log.exception("vector.say failed")

        log.warning("conversation.no_tts_available", text=text[:60])

    async def _load_memory_context(self, text: str) -> None:
        """Load relevant memories and inject into the brain's system prompt."""
        if not self._memory or not self._memory.available:
            return
        try:
            ctx = await self._memory.load_context(current_message=text)
            self._brain.set_memory_context(ctx)
        except Exception:
            log.exception("memory.load_context_failed")

    async def _persist_exchange(self, user_text: str, assistant_text: str) -> None:
        """Store conversation messages and auto-learn facts (background task)."""
        if not self._memory:
            return
        try:
            await self._memory.store_message("user", user_text)
            await self._memory.store_message("assistant", assistant_text)
            await self._memory.auto_learn(user_text, assistant_text)
        except Exception:
            log.exception("memory.persist_failed")

    def _end_conversation(self) -> None:
        """End the current conversation and reset state."""
        if not self._active:
            return
        self._active = False
        self._brain.reset_context()
        log.info("conversation.ended")

    def end_conversation(self) -> None:
        """Public method to end the current conversation."""
        self._end_conversation()

    async def run_vision_event_loop(
        self, shutdown: asyncio.Event
    ) -> None:
        """Continuously process vision events from the pipeline.

        Applies a cooldown between reactions to avoid being too chatty.

        Args:
            shutdown: Event that signals when to stop.
        """
        if not self._vision:
            return

        log.info("conversation.vision_event_loop started")

        # Minimum seconds between vision-triggered reactions.
        cooldown_s = 30.0
        last_reaction_at = 0.0

        while not shutdown.is_set():
            try:
                # Wait for events with a timeout so we can check shutdown.
                try:
                    event = await asyncio.wait_for(
                        self._vision.events.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Enforce cooldown between vision reactions.
                now = time.monotonic()
                if (now - last_reaction_at) < cooldown_s:
                    continue

                summary = (
                    f"{event.event_type}: "
                    f"{', '.join(f'{k}={v}' for k, v in event.data.items())}"
                )
                result = await self.handle_vision_event(summary)

                # Only update cooldown if brain actually reacted.
                if result is not None:
                    last_reaction_at = time.monotonic()

            except Exception:
                log.exception("conversation.vision_event_loop error")

        log.info("conversation.vision_event_loop stopped")
