"""Camera capture and VLM processing — gives Vector spatial awareness."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.brain import Brain

log = structlog.get_logger()


class VisionLoop:
    """Periodically captures frames from Vector's camera and describes them via VLM."""

    def __init__(self, config: dict) -> None:
        self.model = config["models"]["vlm"]
        self.endpoint = config["endpoints"]["ollama"]
        self.interval = config["thresholds"]["vision_interval_seconds"]

    async def capture(self) -> bytes:
        """Capture a frame from Vector's camera.

        Returns:
            Raw image bytes (JPEG).
        """
        raise NotImplementedError("VisionLoop.capture not yet implemented")

    async def describe(self, frame: bytes) -> str:
        """Send a frame to the VLM and get a scene description.

        Args:
            frame: Raw image bytes.

        Returns:
            Text description of the scene.
        """
        raise NotImplementedError("VisionLoop.describe not yet implemented")

    async def run(self, brain: Brain) -> None:
        """Main vision loop — capture, describe, feed to brain.

        Args:
            brain: The Brain instance to send descriptions to.
        """
        log.info("vision loop starting", interval=self.interval)
        while True:
            try:
                frame = await self.capture()
                description = await self.describe(frame)
                await brain.incorporate_vision(description)
            except NotImplementedError:
                log.debug("vision not yet implemented, sleeping")
            except Exception:
                log.exception("vision loop error")
            await asyncio.sleep(self.interval)
