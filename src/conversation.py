"""Dialogue manager — tracks conversation state and routes messages."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.brain import Brain
    from src.butler_client import ButlerClient
    from src.tts import TTSClient

log = structlog.get_logger()


class ConversationManager:
    """Manages dialogue flow between the user, brain, and TTS.

    Decides whether speech is directed at Vector, maintains conversation
    state, and handles escalation to Butler for complex queries.
    """

    def __init__(
        self,
        config: dict,
        *,
        brain: Brain,
        tts: TTSClient,
        butler: ButlerClient,
    ) -> None:
        self.timeout = config["thresholds"]["conversation_timeout_seconds"]
        self.escalation_threshold = config["thresholds"]["escalation_confidence"]
        self._brain = brain
        self._tts = tts
        self._butler = butler
        self._active = False

    async def handle_transcription(self, text: str) -> None:
        """Process a new transcription from STT.

        Args:
            text: Transcribed speech text.
        """
        log.info("transcription received", text=text[:80])
        raise NotImplementedError(
            "ConversationManager.handle_transcription not yet implemented"
        )

    async def _is_directed_at_vector(self, text: str) -> bool:
        """Determine if the speech is directed at Vector.

        Args:
            text: Transcribed text to evaluate.

        Returns:
            True if the speech appears to be for Vector.
        """
        raise NotImplementedError(
            "ConversationManager._is_directed_at_vector not yet implemented"
        )

    async def _should_escalate(self, text: str) -> bool:
        """Determine if the query should be escalated to Butler/Claude.

        Args:
            text: The user's query.

        Returns:
            True if the query is too complex for the local LLM.
        """
        raise NotImplementedError(
            "ConversationManager._should_escalate not yet implemented"
        )

    def end_conversation(self) -> None:
        """End the current conversation and reset state."""
        self._active = False
        self._brain.reset_context()
        log.info("conversation ended")
