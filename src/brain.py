"""LLM reasoning engine — uses Ollama native tool calling for actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from src.ollama_client import ChatMessage, OllamaClient

if TYPE_CHECKING:
    from src.memory import MemoryContext
    from src.vision import VisionPipeline

log = structlog.get_logger()

# Maximum conversation history messages to keep (sliding window).
_MAX_CONTEXT_MESSAGES = 16

# Tools in Ollama native format (JSON Schema).
# Conversation tools — lightweight, for chatting.
CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "look",
            "description": "Look closely at the scene using your camera.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Remember a fact (name, preference, anything personal).",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "category": {"type": "string", "enum": ["preference", "relationship", "observation", "other"]},
                },
                "required": ["fact", "category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Search your memories.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

# Action tools — for physical commands (move, turn, dock, etc.).
ACTION_TOOLS = CHAT_TOOLS + [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Drive forward or backward.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["forward", "backward"]},
                    "distance_mm": {"type": "number", "description": "10-500"},
                },
                "required": ["direction", "distance_mm"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "turn",
            "description": "Turn left or right.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["left", "right"]},
                    "angle_degrees": {"type": "number"},
                },
                "required": ["direction", "angle_degrees"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dock",
            "description": "Drive back onto the charging dock.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_animation",
            "description": "Express an emotion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": ["happy", "sad", "curious", "surprised", "excited", "greeting", "goodbye", "bored"]},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_eye_color",
            "description": "Change eye color.",
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {"type": "string", "enum": ["red", "orange", "yellow", "green", "blue", "purple", "pink", "white"]},
                },
                "required": ["color"],
            },
        },
    },
]

# Keywords that suggest the user wants physical action.
_ACTION_KEYWORDS = {"move", "forward", "backward", "turn", "left", "right", "spin", "dock", "charger", "undock", "eyes", "color", "animation", "happy", "sad", "dance"}


@dataclass
class BrainResponse:
    """Structured output from the brain."""

    speech: str = ""
    tool_calls: list[dict] = field(default_factory=list)


class Brain:
    """LLM with native tool calling for actions and on-demand vision.

    Uses Ollama's structured tool calling — the model returns tool calls
    as structured JSON rather than embedded text, eliminating parsing issues.
    """

    def __init__(self, config: dict, *, client: OllamaClient | None = None) -> None:
        self.model = config["models"]["llm"]
        self.system_prompt = config["personality"]["system_prompt"]
        self.max_tokens = config["thresholds"]["max_response_tokens"]
        self._client = client or OllamaClient(config["endpoints"]["ollama"])
        self._context: list[ChatMessage] = []
        self._recent_speeches: list[str] = []  # Track recent outputs for repetition detection.
        self._active_tools: list[dict] = CHAT_TOOLS  # Currently selected tool set.
        self._vision: VisionPipeline | None = None
        self._memory_context: MemoryContext | None = None

    def set_vision(self, vision: VisionPipeline) -> None:
        """Connect the vision pipeline so the brain can read scene state."""
        self._vision = vision

    def set_memory_context(self, ctx: MemoryContext | None) -> None:
        """Update the memory context for the next brain call."""
        self._memory_context = ctx

    def _build_system_prompt(self) -> str:
        """Build the full system prompt including memory, history, and vision."""
        parts = [self.system_prompt.strip()]

        # Inject remembered facts from persistent memory.
        # NOTE: history injection disabled — it creates feedback loops where
        # bad responses get stored, loaded, and copied by the model.
        if self._memory_context:
            facts_text = self._memory_context.facts_prompt()
            if facts_text:
                parts.append(facts_text)

        # Inject current vision state (YOLO objects with positions + motion).
        if self._vision:
            summary = self._vision.scene.summary()
            if summary:
                parts.append(f"\n[Current vision] {summary}")

        return "\n".join(parts)

    def _trim_context(self) -> None:
        """Keep conversation history within the sliding window."""
        while len(self._context) > _MAX_CONTEXT_MESSAGES:
            self._context.pop(0)

    def _make_response(self, chat_response) -> BrainResponse:
        """Convert an Ollama ChatResponse into a BrainResponse."""
        msg = chat_response.message
        return BrainResponse(
            speech=msg.content.strip(),
            tool_calls=msg.tool_calls or [],
        )

    async def _call_llm(
        self, *, tools: list[dict] | None = None
    ) -> tuple[BrainResponse, float]:
        """Send the current context to the LLM and return a structured response."""
        messages = [
            ChatMessage(role="system", content=self._build_system_prompt()),
            *self._context,
        ]

        response = await self._client.chat(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            tools=tools,
        )

        # Store the assistant's response (including tool calls) in context.
        self._context.append(response.message)
        self._trim_context()

        result = self._make_response(response)

        return result, response.total_duration_ms

    async def think(self, user_message: str) -> BrainResponse:
        """Generate a response to the user's message.

        Args:
            user_message: Transcribed speech or internal prompt.

        Returns:
            BrainResponse with speech and optional tool calls.
        """
        self._context.append(ChatMessage(role="user", content=user_message))

        # Pick tools based on what the user is asking for.
        words = set(user_message.lower().split())
        tools = ACTION_TOOLS if words & _ACTION_KEYWORDS else CHAT_TOOLS
        self._active_tools = tools

        result, duration_ms = await self._call_llm(tools=tools)

        # Detect repetition — fuzzy match (first 30 chars).
        if result.speech:
            prefix = result.speech[:30].lower()
            recent_prefixes = [s[:30].lower() for s in self._recent_speeches]
            if prefix in recent_prefixes:
                log.warning("brain.repetition_detected", speech=result.speech[:40])
                self._context.clear()
                self._context.append(ChatMessage(
                    role="user",
                    content=f"(Say something COMPLETELY different) {user_message}",
                ))
                result, duration_ms = await self._call_llm(tools=tools)

        # Track recent speeches (keep last 8).
        if result.speech:
            self._recent_speeches.append(result.speech)
            if len(self._recent_speeches) > 8:
                self._recent_speeches.pop(0)

        log.info(
            "brain.think",
            speech=result.speech[:80] if result.speech else "",
            tool_calls=len(result.tool_calls),
            duration_ms=duration_ms,
        )

        return result

    async def handle_tool_result(self, tool_name: str, result: str) -> BrainResponse:
        """Feed a tool call result back to the brain for follow-up.

        Args:
            tool_name: The tool that was called.
            result: The result text (e.g., VLM description).

        Returns:
            BrainResponse with follow-up speech/actions.
        """
        self._context.append(ChatMessage(
            role="tool",
            content=result,
            name=tool_name,
        ))

        parsed, duration_ms = await self._call_llm(tools=self._active_tools)

        log.info(
            "brain.handle_tool_result",
            tool=tool_name,
            speech=parsed.speech[:80] if parsed.speech else "",
            duration_ms=duration_ms,
        )

        return parsed

    async def handle_vision_event(self, event_summary: str) -> BrainResponse | None:
        """React to a vision event — can look and remember."""
        prompt = (
            f"[vision event] {event_summary}\n"
            "React briefly if interesting. Say NOTHING if not."
        )
        self._context.append(ChatMessage(role="user", content=prompt))

        result, duration_ms = await self._call_llm(tools=CHAT_TOOLS)

        # Filter out non-responses and repetitions.
        if not result.speech or "nothing" in result.speech.lower().strip("., !"):
            return None

        prefix = result.speech[:30].lower()
        recent_prefixes = [s[:30].lower() for s in self._recent_speeches]
        if prefix in recent_prefixes:
            log.info("brain.vision_repetition_suppressed", speech=result.speech[:40])
            return None

        self._recent_speeches.append(result.speech)
        if len(self._recent_speeches) > 8:
            self._recent_speeches.pop(0)

        log.info(
            "brain.handle_vision_event",
            vision_event=event_summary[:60],
            speech=result.speech[:80] if result.speech else "",
        )

        return result

    def reset_context(self) -> None:
        """Clear conversation history."""
        self._context.clear()
        log.info("brain context reset")
