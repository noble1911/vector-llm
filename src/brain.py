"""LLM reasoning engine — generates responses and decides actions."""

from __future__ import annotations

import structlog

log = structlog.get_logger()


class Brain:
    """Core LLM interface using Ollama.

    Maintains conversation context and generates responses
    based on speech input and vision descriptions.
    """

    def __init__(self, config: dict) -> None:
        self.model = config["models"]["llm"]
        self.endpoint = config["endpoints"]["ollama"]
        self.system_prompt = config["personality"]["system_prompt"]
        self.max_tokens = config["thresholds"]["max_response_tokens"]
        self._context: list[dict[str, str]] = []

    async def think(self, user_message: str) -> str:
        """Generate a response to the user's message.

        Args:
            user_message: Transcribed speech or internal prompt.

        Returns:
            The LLM's text response.
        """
        raise NotImplementedError("Brain.think not yet implemented")

    async def incorporate_vision(self, scene_description: str) -> None:
        """Update the brain's context with a new scene description.

        Args:
            scene_description: Text description of what Vector sees.
        """
        raise NotImplementedError("Brain.incorporate_vision not yet implemented")

    def reset_context(self) -> None:
        """Clear conversation history."""
        self._context.clear()
        log.info("brain context reset")
