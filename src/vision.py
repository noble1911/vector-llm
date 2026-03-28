"""Vision pipeline — real-time CV + on-demand VLM for spatial awareness.

Tier 1 (every frame, ~5fps): Fast CV via YOLO + OpenCV
  - Object detection, motion detection
  - Produces structured events for the brain

Tier 2 (on-demand): VLM scene description
  - Triggered by brain tool call or significant scene change
  - Full natural-language description of what Vector sees
"""

from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from src.ollama_client import ChatMessage, OllamaClient

if TYPE_CHECKING:
    from src.vector_control import VectorController

log = structlog.get_logger()


@dataclass
class VisionEvent:
    """A structured event from the CV pipeline."""

    event_type: str  # "objects_changed", "motion", "scene_change"
    data: dict = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class SceneState:
    """Current understanding of what Vector sees."""

    objects: list[str] = field(default_factory=list)
    faces: list[str] = field(default_factory=list)
    motion_detected: bool = False
    motion_region: str = ""  # "left", "right", "center"
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


def detect_motion(
    current_gray: np.ndarray,
    previous_gray: np.ndarray | None,
    threshold: float,
) -> tuple[bool, str]:
    """Detect motion via frame differencing and locate the region.

    Args:
        current_gray: Current frame as grayscale numpy array.
        previous_gray: Previous frame as grayscale, or None if first frame.
        threshold: Fraction of pixels that must differ to count as motion (0-1).

    Returns:
        Tuple of (motion_detected, region) where region is "left"/"center"/"right".
    """
    if previous_gray is None:
        return False, ""

    import cv2

    diff = cv2.absdiff(current_gray, previous_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    total_pixels = thresh.size
    motion_pixels = np.count_nonzero(thresh)
    motion_ratio = motion_pixels / total_pixels

    if motion_ratio < threshold:
        return False, ""

    # Determine region: split frame into thirds.
    h, w = thresh.shape
    third = w // 3
    left_motion = np.count_nonzero(thresh[:, :third])
    center_motion = np.count_nonzero(thresh[:, third : 2 * third])
    right_motion = np.count_nonzero(thresh[:, 2 * third :])

    max_region = max(left_motion, center_motion, right_motion)
    if max_region == left_motion:
        region = "left"
    elif max_region == center_motion:
        region = "center"
    else:
        region = "right"

    return True, region


def detect_objects(
    frame: np.ndarray,
    model: Any,
    confidence_threshold: float,
) -> list[str]:
    """Run YOLO object detection on a frame.

    Args:
        frame: BGR numpy array.
        model: Loaded YOLO model instance.
        confidence_threshold: Minimum confidence to include a detection.

    Returns:
        Sorted list of unique object labels detected.
    """
    results = model(frame, verbose=False)
    labels = set()
    for result in results:
        for box in result.boxes:
            if box.conf.item() >= confidence_threshold:
                cls_id = int(box.cls.item())
                label = model.names.get(cls_id, f"class_{cls_id}")
                labels.add(label)
    return sorted(labels)


def check_scene_change(
    current_objects: list[str],
    previous_objects: list[str],
    threshold: float,
) -> bool:
    """Check if the set of detected objects has changed significantly.

    Uses Jaccard distance: 1 - |intersection| / |union|.

    Args:
        current_objects: Currently detected object labels.
        previous_objects: Previously detected object labels.
        threshold: Minimum change ratio to trigger (0-1).

    Returns:
        True if the scene changed enough.
    """
    if not previous_objects and not current_objects:
        return False
    if not previous_objects or not current_objects:
        return True

    current_set = set(current_objects)
    previous_set = set(previous_objects)
    union = current_set | previous_set
    intersection = current_set & previous_set

    if not union:
        return False

    change_ratio = 1.0 - len(intersection) / len(union)
    return change_ratio >= threshold


class VisionPipeline:
    """Real-time vision: fast CV on every frame, VLM on demand.

    Runs continuously, updating scene state that the brain can read.
    Emits events when significant changes are detected.
    """

    def __init__(
        self,
        config: dict,
        *,
        vector: VectorController,
        ollama_client: OllamaClient | None = None,
    ) -> None:
        self._vector = vector
        self._scene = SceneState()
        self._event_queue: asyncio.Queue[VisionEvent] = asyncio.Queue(maxsize=50)
        self._previous_gray: np.ndarray | None = None
        self._yolo_model: Any = None

        # Config
        vision_cfg = config.get("vision", {})
        self.yolo_model_name = vision_cfg.get("yolo_model", "yolov8n.pt")
        self.confidence_threshold = vision_cfg.get("confidence_threshold", 0.5)
        self.motion_threshold = vision_cfg.get("motion_threshold", 0.05)
        self.scene_change_threshold = vision_cfg.get("scene_change_threshold", 0.3)

        # VLM config (on-demand)
        self.vlm_model = config.get("models", {}).get("vlm", "")
        endpoint = config.get("endpoints", {}).get("ollama", "http://localhost:11434")
        self._ollama = ollama_client or OllamaClient(endpoint)

    @property
    def scene(self) -> SceneState:
        """Current scene state — read by the brain for context."""
        return self._scene

    @property
    def events(self) -> asyncio.Queue[VisionEvent]:
        """Queue of vision events for the brain to consume."""
        return self._event_queue

    async def _load_yolo(self) -> None:
        """Load the YOLO model in a thread (heavy import + download)."""
        def _load():
            from ultralytics import YOLO
            return YOLO(self.yolo_model_name)

        self._yolo_model = await asyncio.to_thread(_load)
        log.info("yolo model loaded", model=self.yolo_model_name)

    async def run(self, shutdown: asyncio.Event) -> None:
        """Main vision loop — process every frame from the camera feed.

        Args:
            shutdown: Event that signals when to stop.
        """
        log.info("vision pipeline starting")

        await self._load_yolo()
        await self._vector.start_camera_feed()

        # Wait for first frame.
        for _ in range(50):
            if shutdown.is_set():
                return
            image_bytes = await self._vector.get_latest_image()
            if image_bytes is not None:
                break
            await asyncio.sleep(0.1)
        else:
            log.error("vision pipeline: no frames received after 5s")
            return

        log.info("vision pipeline running")

        while not shutdown.is_set():
            try:
                image_bytes = await self._vector.get_latest_image()
                if image_bytes is None:
                    await asyncio.sleep(0.05)
                    continue

                events = await self._process_frame(image_bytes)

                for event in events:
                    try:
                        self._event_queue.put_nowait(event)
                    except asyncio.QueueFull:
                        # Drop oldest event if queue is full.
                        try:
                            self._event_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        self._event_queue.put_nowait(event)

            except Exception:
                log.exception("vision pipeline error")

            # Small sleep to yield control — camera runs at ~5fps anyway.
            await asyncio.sleep(0.05)

        log.info("vision pipeline stopped")

    async def _process_frame(self, image_bytes: bytes) -> list[VisionEvent]:
        """Run all Tier 1 CV on a single frame.

        Args:
            image_bytes: JPEG image bytes from the camera.

        Returns:
            List of events detected in this frame.
        """
        import cv2

        # Decode JPEG to numpy array.
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        now = time.time()
        events: list[VisionEvent] = []

        # Motion detection (pure numpy/cv2, very fast).
        motion_detected, motion_region = detect_motion(
            gray, self._previous_gray, self.motion_threshold
        )
        self._previous_gray = gray

        if motion_detected != self._scene.motion_detected or (
            motion_detected and motion_region != self._scene.motion_region
        ):
            self._scene.motion_detected = motion_detected
            self._scene.motion_region = motion_region
            if motion_detected:
                events.append(VisionEvent(
                    event_type="motion",
                    data={"region": motion_region},
                    timestamp=now,
                ))

        # Object detection (YOLO — run in thread to avoid blocking).
        if self._yolo_model is not None:
            objects = await asyncio.to_thread(
                detect_objects, frame, self._yolo_model, self.confidence_threshold
            )

            if objects != self._scene.objects:
                scene_changed = check_scene_change(
                    objects, self._scene.objects, self.scene_change_threshold
                )
                self._scene.objects = objects

                if scene_changed:
                    events.append(VisionEvent(
                        event_type="objects_changed",
                        data={"objects": objects},
                        timestamp=now,
                    ))

        return events

    async def describe_scene(self) -> str:
        """On-demand VLM: capture a frame and get a full description.

        Called by the brain via tool use when it wants to "look closely".

        Returns:
            Natural language description of the current scene.
        """
        log.info("vlm scene description requested")

        image_bytes = await self._vector.capture_image()

        # Encode to base64 for the Ollama vision API.
        image_b64 = base64.b64encode(image_bytes).decode("ascii")

        # Ollama vision models accept images in the message content.
        # We use the /api/chat endpoint with an image payload.
        import httpx

        payload = {
            "model": self.vlm_model,
            "messages": [
                {
                    "role": "user",
                    "content": "Describe what you see in this image in 1-2 sentences. "
                    "Focus on objects, people, and spatial layout.",
                    "images": [image_b64],
                }
            ],
            "stream": False,
            "options": {"num_predict": 100},
        }

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                response = await client.post(
                    f"{self._ollama._base_url}/api/chat",
                    json=payload,
                )

            if response.status_code != 200:
                log.warning("vlm request failed", status=response.status_code)
                return f"[VLM error: {response.status_code}]"

            data = response.json()
            description = data.get("message", {}).get("content", "")
            self._scene.last_description = description

            log.info("vlm description", description=description[:80])
            return description

        except Exception as e:
            log.exception("vlm request error")
            return f"[VLM error: {e}]"
