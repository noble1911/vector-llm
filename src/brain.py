"""LLM reasoning engine — text-only model with tool use for actions and vision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.vision import VisionPipeline

log = structlog.get_logger()


@dataclass
class BrainResponse:
    """Structured output from the brain."""

    speech: str = ""  # Text to speak
    tool_calls: list[dict] = None  # Tool invocations (look, move, turn, etc.)

    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []


# Tools the brain can invoke
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
]


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

    def __init__(self, config: dict) -> None:
        self.model = config["models"]["llm"]
        self.endpoint = config["endpoints"]["ollama"]
        self.system_prompt = config["personality"]["system_prompt"]
        self.max_tokens = config["thresholds"]["max_response_tokens"]
        self._context: list[dict[str, str]] = []
        self._vision: VisionPipeline | None = None

    def set_vision(self, vision: VisionPipeline) -> None:
        """Connect the vision pipeline so the brain can read scene state."""
        self._vision = vision

    def _build_system_prompt(self) -> str:
        """Build the full system prompt including vision context and tools."""
        parts = [self.system_prompt.strip()]

        # Inject current vision state
        if self._vision:
            scene_summary = self._vision.scene.summary()
            parts.append(f"\n[Current vision] {scene_summary}")

        return "\n".join(parts)

    async def think(self, user_message: str) -> BrainResponse:
        """Generate a response to the user's message.

        Args:
            user_message: Transcribed speech or internal prompt.

        Returns:
            BrainResponse with speech and optional tool calls.
        """
        raise NotImplementedError("Brain.think not yet implemented")

    async def handle_tool_result(self, tool_name: str, result: str) -> BrainResponse:
        """Feed a tool call result back to the brain for follow-up.

        Args:
            tool_name: The tool that was called.
            result: The result text (e.g., VLM description).

        Returns:
            BrainResponse with follow-up speech/actions.
        """
        raise NotImplementedError("Brain.handle_tool_result not yet implemented")

    async def handle_vision_event(self, event_summary: str) -> BrainResponse | None:
        """React to a vision event (motion, new face, scene change).

        Args:
            event_summary: Text summary of what changed.

        Returns:
            BrainResponse if the brain wants to react, None if it ignores it.
        """
        raise NotImplementedError("Brain.handle_vision_event not yet implemented")

    def reset_context(self) -> None:
        """Clear conversation history."""
        self._context.clear()
        log.info("brain context reset")
