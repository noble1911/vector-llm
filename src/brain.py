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
_MAX_CONTEXT_MESSAGES = 40

# Tools in Ollama native format (JSON Schema).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "look",
            "description": "Look closely at the scene using your camera for a detailed description.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Drive forward or backward.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["forward", "backward"],
                        "description": "Direction to move",
                    },
                    "distance_mm": {
                        "type": "number",
                        "description": "Distance in millimeters (10-500)",
                    },
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
                    "direction": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Direction to turn",
                    },
                    "angle_degrees": {
                        "type": "number",
                        "description": "Degrees to turn (e.g. 45, 90, 180)",
                    },
                },
                "required": ["direction", "angle_degrees"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_head_angle",
            "description": "Tilt your head up or down to look at different things.",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle_degrees": {
                        "type": "number",
                        "description": "Head angle from -22 (down) to 45 (up)",
                    },
                },
                "required": ["angle_degrees"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_animation",
            "description": "Express an emotion through body language.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": [
                            "happy", "sad", "curious", "surprised", "excited",
                            "greeting", "goodbye", "love", "bored", "frustrated",
                            "goodnight", "goodmorning",
                        ],
                        "description": "Emotion to express",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dock",
            "description": "Drive back onto the charging dock. Vector will navigate to the charger automatically.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "undock",
            "description": "Drive off the charging dock so you can move around.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_eye_color",
            "description": "Change your eye color to express mood or emotion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "enum": [
                            "red", "orange", "yellow", "green", "cyan",
                            "blue", "purple", "pink", "white",
                        ],
                        "description": "Eye color to set",
                    },
                },
                "required": ["color"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Remember a fact about someone or something for later. Use this when you learn names, preferences, or anything worth keeping.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The fact to remember (e.g. 'Ron likes coffee')",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["preference", "relationship", "observation", "schedule", "other"],
                        "description": "Category of the fact",
                    },
                },
                "required": ["fact", "category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Search your memories for something you learned before.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for (e.g. 'what is their name')",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_butler",
            "description": "Escalate a question to Butler (Claude) for complex queries you can't handle.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask Butler",
                    },
                },
                "required": ["question"],
            },
        },
    },
]


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
        if self._memory_context:
            facts_text = self._memory_context.facts_prompt()
            if facts_text:
                parts.append(facts_text)
            history_text = self._memory_context.history_prompt()
            if history_text:
                parts.append(history_text)

        # Inject current vision state.
        if self._vision:
            scene = self._vision.scene
            # Quick CV summary (objects, motion).
            cv_parts = []
            if scene.objects:
                cv_parts.append(f"objects: {', '.join(scene.objects)}")
            if scene.motion_detected:
                cv_parts.append(f"motion: {scene.motion_region}")
            if cv_parts:
                parts.append(f"\n[Current vision] {' | '.join(cv_parts)}")
            # Last VLM description (richer scene understanding).
            if scene.last_description:
                parts.append(f"[Last scene] {scene.last_description}")

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
    ) -> BrainResponse:
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

        result, duration_ms = await self._call_llm(tools=TOOLS)

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

        parsed, duration_ms = await self._call_llm(tools=TOOLS)

        log.info(
            "brain.handle_tool_result",
            tool=tool_name,
            speech=parsed.speech[:80] if parsed.speech else "",
            duration_ms=duration_ms,
        )

        return parsed

    async def handle_vision_event(self, event_summary: str) -> BrainResponse | None:
        """React to a vision event — kept lightweight, no tools."""
        prompt = f"[vision event] {event_summary}"
        self._context.append(ChatMessage(role="user", content=prompt))

        # No tools for vision events — just speak or stay quiet.
        result, duration_ms = await self._call_llm(tools=None)

        # Filter out non-responses — model should say NOTHING if not noteworthy.
        if not result.speech or "nothing" in result.speech.lower().strip("., !"):
            return None

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
