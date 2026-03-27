"""Vision pipeline — real-time CV + on-demand VLM for spatial awareness.

Tier 1 (every frame, ~5fps): Fast CV via YOLO + OpenCV
  - Object detection, motion detection, face detection
  - Produces structured events for the brain

Tier 2 (on-demand): VLM scene description
  - Triggered by brain tool call or significant scene change
  - Full natural-language description of what Vector sees
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.vector_control import VectorController

log = structlog.get_logger()


@dataclass
class VisionEvent:
    """A structured event from the CV pipeline."""

    event_type: str  # "objects", "motion", "face", "scene_change"
    data: dict = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class SceneState:
    """Current understanding of what Vector sees."""

    objects: list[str] = field(default_factory=list)
    faces: list[str] = field(default_factory=list)
    motion_detected: bool = False
    motion_region: str = ""  # "left", "right", "center", etc.
    last_description: str = ""  # From VLM, if available

    def summary(self) -> str:
        """One-line summary for the brain's context."""
        parts = []
        if self.objects:
            parts.append(f"objects: {', '.join(self.objects)}")
        if self.faces:
            parts.append(f"faces: {', '.join(self.faces)}")
        if self.motion_detected:
            parts.append(f"motion: {self.motion_region or 'detected'}")
        else:
            parts.append("motion: none")
        return " | ".join(parts) if parts else "no visual data"


class VisionPipeline:
    """Real-time vision: fast CV on every frame, VLM on demand.

    Runs continuously, updating scene state that the brain can read.
    Emits events when significant changes are detected.
    """

    def __init__(self, config: dict, *, vector: VectorController) -> None:
        self._vector = vector
        self._scene = SceneState()
        self._event_queue: asyncio.Queue[VisionEvent] = asyncio.Queue()
        self._previous_frame = None

        # Config
        vision_cfg = config.get("vision", {})
        self.yolo_model = vision_cfg.get("yolo_model", "yolov8n.pt")
        self.confidence_threshold = vision_cfg.get("confidence_threshold", 0.5)
        self.motion_threshold = vision_cfg.get("motion_threshold", 0.05)
        self.scene_change_threshold = vision_cfg.get("scene_change_threshold", 0.3)

        # VLM config (on-demand)
        self.vlm_model = config.get("models", {}).get("vlm", "")
        self.vlm_endpoint = config.get("endpoints", {}).get("ollama", "")

    @property
    def scene(self) -> SceneState:
        """Current scene state — read by the brain for context."""
        return self._scene

    @property
    def events(self) -> asyncio.Queue[VisionEvent]:
        """Queue of vision events for the brain to consume."""
        return self._event_queue

    async def run(self, shutdown: asyncio.Event) -> None:
        """Main vision loop — process every frame from the camera feed.

        Args:
            shutdown: Event that signals when to stop.
        """
        log.info("vision pipeline starting")
        raise NotImplementedError("VisionPipeline.run not yet implemented")

    async def _process_frame(self, frame_bytes: bytes) -> list[VisionEvent]:
        """Run all Tier 1 CV on a single frame.

        Args:
            frame_bytes: Raw image bytes from the camera.

        Returns:
            List of events detected in this frame.
        """
        raise NotImplementedError("VisionPipeline._process_frame not yet implemented")

    async def _detect_objects(self, frame) -> list[str]:
        """Run YOLOv8-nano object detection.

        Args:
            frame: numpy array (BGR image).

        Returns:
            List of detected object labels.
        """
        raise NotImplementedError(
            "VisionPipeline._detect_objects not yet implemented"
        )

    async def _detect_motion(self, frame) -> tuple[bool, str]:
        """Detect motion by comparing with previous frame.

        Args:
            frame: numpy array (BGR image).

        Returns:
            Tuple of (motion_detected, region).
        """
        raise NotImplementedError(
            "VisionPipeline._detect_motion not yet implemented"
        )

    async def _check_scene_change(self, current_objects: list[str]) -> bool:
        """Determine if the scene has changed significantly.

        Args:
            current_objects: Objects detected in current frame.

        Returns:
            True if scene changed enough to warrant a VLM call.
        """
        raise NotImplementedError(
            "VisionPipeline._check_scene_change not yet implemented"
        )

    async def describe_scene(self) -> str:
        """On-demand VLM: capture a frame and get a full description.

        Called by the brain via tool use when it wants to "look closely".

        Returns:
            Natural language description of the current scene.
        """
        log.info("vlm scene description requested")
        raise NotImplementedError(
            "VisionPipeline.describe_scene not yet implemented"
        )
