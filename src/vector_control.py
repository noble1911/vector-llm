"""Vector SDK wrapper — movement, animations, and expressions."""

from __future__ import annotations

import structlog

log = structlog.get_logger()


class VectorController:
    """Wraps the anki_vector SDK for robot control."""

    def __init__(self, config: dict) -> None:
        self._robot = None

    async def connect(self) -> None:
        """Establish connection to Vector via wire-pod tokens."""
        raise NotImplementedError("VectorController.connect not yet implemented")

    async def disconnect(self) -> None:
        """Cleanly disconnect from Vector."""
        raise NotImplementedError("VectorController.disconnect not yet implemented")

    async def say(self, text: str) -> None:
        """Make Vector speak using his built-in TTS (fallback).

        Args:
            text: Text for Vector to say.
        """
        raise NotImplementedError("VectorController.say not yet implemented")

    async def move(self, direction: str, distance_mm: float) -> None:
        """Drive Vector in a direction.

        Args:
            direction: "forward" or "backward".
            distance_mm: Distance in millimeters.
        """
        raise NotImplementedError("VectorController.move not yet implemented")

    async def turn(self, angle_degrees: float) -> None:
        """Turn Vector by an angle.

        Args:
            angle_degrees: Degrees to turn (positive = left).
        """
        raise NotImplementedError("VectorController.turn not yet implemented")

    async def set_head_angle(self, angle_degrees: float) -> None:
        """Set Vector's head angle.

        Args:
            angle_degrees: Head angle (-22 to 45 degrees).
        """
        raise NotImplementedError("VectorController.set_head_angle not yet implemented")

    async def play_animation(self, name: str) -> None:
        """Play a named animation on Vector.

        Args:
            name: Animation trigger name.
        """
        raise NotImplementedError("VectorController.play_animation not yet implemented")

    async def capture_image(self) -> bytes:
        """Capture a single frame from Vector's camera.

        Returns:
            JPEG image bytes.
        """
        raise NotImplementedError("VectorController.capture_image not yet implemented")
