"""Always-on speech-to-text — listens via external mic with VAD filtering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.conversation import ConversationManager

log = structlog.get_logger()


class STTListener:
    """Captures audio from the external mic, runs VAD, and transcribes speech."""

    def __init__(self, config: dict) -> None:
        self.model_size = config["models"]["stt"]
        self.vad_sensitivity = config["thresholds"]["vad_sensitivity"]

    async def listen(self, conversation: ConversationManager) -> None:
        """Main listening loop — capture audio, detect speech, transcribe, forward.

        Args:
            conversation: The ConversationManager to send transcriptions to.
        """
        log.info("stt listener starting", model=self.model_size)
        raise NotImplementedError("STTListener.listen not yet implemented")

    async def _detect_speech(self, audio_chunk: bytes) -> bool:
        """Run VAD on an audio chunk.

        Args:
            audio_chunk: Raw PCM audio bytes.

        Returns:
            True if speech is detected.
        """
        raise NotImplementedError("STTListener._detect_speech not yet implemented")

    async def _transcribe(self, audio: bytes) -> str:
        """Transcribe a speech segment using Whisper.

        Args:
            audio: Raw PCM audio of a speech segment.

        Returns:
            Transcribed text.
        """
        raise NotImplementedError("STTListener._transcribe not yet implemented")
