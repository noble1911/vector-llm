"""LLM reasoning engine — text-only model with tool use for actions and vision."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from src.ollama_client import ChatMessage, OllamaClient

if TYPE_CHECKING:
    from src.memory import MemoryContext
    from src.vision import VisionPipeline

log = structlog.get_logger()

# Maximum conversation history messages to keep (sliding window).
_MAX_CONTEXT_MESSAGES = 20

# Tools the brain can invoke, formatted for the system prompt.
TOOLS = [
    {
        "name": "look",
        "description": "Look closely at what's in front of you using your camera. "
        "Use this when you want a detailed description of the scene.",
        "parameters": {},
    },
    {
        "name": "move",
        "description": "Drive forward or backward.",
        "parameters": {
            "direction": "forward or backward",
            "distance_mm": "distance in millimeters (10-500)",
        },
    },
    {
        "name": "turn",
        "description": "Turn left or right.",
        "parameters": {
            "angle_degrees": "degrees to turn (positive=left, negative=right)",
        },
    },
    {
        "name": "set_head_angle",
        "description": "Tilt your head up or down to look at different things.",
        "parameters": {
            "angle_degrees": "head angle (-22=down to 45=up)",
        },
    },
    {
        "name": "play_animation",
        "description": "Express an emotion through body language.",
        "parameters": {
            "name": "animation name (e.g. happy, sad, curious, surprised)",
        },
    },
    {
        "name": "ask_butler",
        "description": "Escalate a question to Butler (Claude) for complex queries "
        "you can't handle — real-time data, home automation, deep reasoning, "
        "web searches. Rephrase the question clearly for Butler.",
        "parameters": {
            "question": "the question to ask Butler",
        },
    },
]

# Regex to extract tool calls from LLM output: [tool_name({"key": "value"})]
_TOOL_CALL_PATTERN = re.compile(
    r"\[(\w+)\((\{.*?\})\)\]",
    re.DOTALL,
)


@dataclass
class BrainResponse:
    """Structured output from the brain."""

    speech: str = ""
    tool_calls: list[dict] = field(default_factory=list)


def parse_response(text: str) -> BrainResponse:
    """Parse raw LLM output into speech and tool calls.

    The LLM is instructed to output tool calls as [tool_name({"param": "value"})].
    Everything else is treated as speech.

    Args:
        text: Raw LLM response text.

    Returns:
        BrainResponse with separated speech and tool calls.
    """
    tool_calls = []
    speech = text

    for match in _TOOL_CALL_PATTERN.finditer(text):
        tool_name = match.group(1)
        try:
            params = json.loads(match.group(2))
        except json.JSONDecodeError:
            log.warning("failed to parse tool params", tool=tool_name, raw=match.group(2))
            continue
        tool_calls.append({"name": tool_name, "parameters": params})
        speech = speech.replace(match.group(0), "")

    speech = speech.strip()

    return BrainResponse(speech=speech, tool_calls=tool_calls)


def build_tool_prompt() -> str:
    """Format tool definitions for inclusion in the system prompt."""
    lines = [
        "You can perform actions by including tool calls in your response.",
        "Format: [tool_name({\"param\": \"value\"})]",
        "",
        "Available tools:",
    ]
    for tool in TOOLS:
        params_str = ", ".join(
            f'"{k}": {v}' for k, v in tool["parameters"].items()
        )
        if params_str:
            lines.append(f'- {tool["name"]}({{{params_str}}}): {tool["description"]}')
        else:
            lines.append(f'- {tool["name"]}(): {tool["description"]}')
    lines.append("")
    lines.append("You may include zero or more tool calls alongside your speech.")
    lines.append("Example: That's interesting! Let me take a closer look. [look({})]")
    return "\n".join(lines)


class Brain:
    """Text-only LLM with tool use for actions and on-demand vision.

    The brain receives:
    - Transcribed speech from the user
    - Structured vision events (objects, motion, faces) from the CV pipeline
    - Tool call results (VLM descriptions, movement confirmations)

    The brain outputs:
    - Speech text to say
    - Tool calls to execute (look, move, turn, animate)
    """

    def __init__(self, config: dict, *, client: OllamaClient | None = None) -> None:
        self.model = config["models"]["llm"]
        self.system_prompt = config["personality"]["system_prompt"]
        self.max_tokens = config["thresholds"]["max_response_tokens"]
        self._client = client or OllamaClient(config["endpoints"]["ollama"])
        self._context: list[ChatMessage] = []
        self._vision: VisionPipeline | None = None
        self._memory_context: MemoryContext | None = None

    def set_vision(self, vision: VisionPipeline) -> None:
        """Connect the vision pipeline so the brain can read scene state."""
        self._vision = vision

    def set_memory_context(self, ctx: MemoryContext | None) -> None:
        """Update the memory context for the next brain call."""
        self._memory_context = ctx

    def _build_system_prompt(self) -> str:
        """Build the full system prompt including memory, vision context, and tools."""
        parts = [self.system_prompt.strip()]

        # Inject remembered facts from persistent memory.
        if self._memory_context:
            facts_text = self._memory_context.facts_prompt()
            if facts_text:
                parts.append(facts_text)

        # Inject current vision state
        if self._vision:
            scene_summary = self._vision.scene.summary()
            parts.append(f"\n[Current vision] {scene_summary}")

        # Append tool definitions
        parts.append(f"\n{build_tool_prompt()}")

        return "\n".join(parts)

    def _trim_context(self) -> None:
        """Keep conversation history within the sliding window.

        Called after appending new messages, so we trim to max size.
        """
        while len(self._context) > _MAX_CONTEXT_MESSAGES:
            self._context.pop(0)

    async def think(self, user_message: str) -> BrainResponse:
        """Generate a response to the user's message.

        Args:
            user_message: Transcribed speech or internal prompt.

        Returns:
            BrainResponse with speech and optional tool calls.
        """
        self._context.append(ChatMessage(role="user", content=user_message))

        messages = [
            ChatMessage(role="system", content=self._build_system_prompt()),
            *self._context,
        ]

        response = await self._client.chat(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )

        raw_text = response.message.content
        self._context.append(ChatMessage(role="assistant", content=raw_text))
        self._trim_context()

        result = parse_response(raw_text)

        log.info(
            "brain.think",
            speech=result.speech[:80] if result.speech else "",
            tool_calls=len(result.tool_calls),
            duration_ms=response.total_duration_ms,
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
        tool_msg = f"[{tool_name} result] {result}"
        self._context.append(ChatMessage(role="user", content=tool_msg))

        messages = [
            ChatMessage(role="system", content=self._build_system_prompt()),
            *self._context,
        ]

        response = await self._client.chat(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )

        raw_text = response.message.content
        self._context.append(ChatMessage(role="assistant", content=raw_text))
        self._trim_context()

        parsed = parse_response(raw_text)

        log.info(
            "brain.handle_tool_result",
            tool=tool_name,
            speech=parsed.speech[:80] if parsed.speech else "",
        )

        return parsed

    async def handle_vision_event(self, event_summary: str) -> BrainResponse | None:
        """React to a vision event (motion, new face, scene change).

        Not every event needs a response — the brain decides.

        Args:
            event_summary: Text summary of what changed.

        Returns:
            BrainResponse if the brain wants to react, None if it ignores it.
        """
        prompt = (
            f"[vision event] {event_summary}\n"
            "React briefly if this is interesting, or say nothing if it's not noteworthy."
        )
        self._context.append(ChatMessage(role="user", content=prompt))

        messages = [
            ChatMessage(role="system", content=self._build_system_prompt()),
            *self._context,
        ]

        response = await self._client.chat(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )

        raw_text = response.message.content
        self._context.append(ChatMessage(role="assistant", content=raw_text))
        self._trim_context()

        parsed = parse_response(raw_text)

        # If the brain has nothing to say, treat it as ignoring the event.
        if not parsed.speech and not parsed.tool_calls:
            return None

        log.info(
            "brain.handle_vision_event",
            vision_event=event_summary[:60],
            speech=parsed.speech[:80] if parsed.speech else "",
        )

        return parsed

    def reset_context(self) -> None:
        """Clear conversation history."""
        self._context.clear()
        log.info("brain context reset")
