"""Text-to-speech via Kokoro — converts brain responses to audio."""

from __future__ import annotations

import structlog

log = structlog.get_logger()


class TTSClient:
    """Sends text to Kokoro TTS and plays audio on Vector's speaker."""

    def __init__(self, config: dict) -> None:
        self.endpoint = config["endpoints"]["kokoro_tts"]
        self.voice = config["personality"]["voice"]

    async def speak(self, text: str) -> None:
        """Convert text to speech and play it.

        Args:
            text: The text to speak.
        """
        log.info("tts speak", text=text[:50])
        raise NotImplementedError("TTSClient.speak not yet implemented")

    async def _synthesize(self, text: str) -> bytes:
        """Call Kokoro TTS API to generate audio.

        Args:
            text: Text to synthesize.

        Returns:
            Raw audio bytes.
        """
        raise NotImplementedError("TTSClient._synthesize not yet implemented")

    async def _play_on_vector(self, audio: bytes) -> None:
        """Stream audio to Vector's speaker.

        Args:
            audio: Raw audio bytes to play.
        """
        raise NotImplementedError("TTSClient._play_on_vector not yet implemented")
