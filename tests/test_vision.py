"""Tests for the vision pipeline — CV functions and VisionPipeline integration."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.vision import (
    SceneState,
    VisionEvent,
    VisionPipeline,
    check_scene_change,
    detect_motion,
    detect_objects,
)


# ---------------------------------------------------------------------------
# SceneState
# ---------------------------------------------------------------------------


class TestSceneState:
    def test_empty_state(self):
        state = SceneState()
        assert state.summary() == "motion: none"

    def test_with_objects(self):
        state = SceneState(objects=["mug", "keyboard"])
        summary = state.summary()
        assert "mug" in summary
        assert "keyboard" in summary

    def test_with_motion(self):
        state = SceneState(motion_detected=True, motion_region="left")
        assert "motion: left" in state.summary()

    def test_with_faces(self):
        state = SceneState(faces=["person_1"])
        assert "person_1" in state.summary()

    def test_full_state(self):
        state = SceneState(
            objects=["laptop"],
            faces=["ron"],
            motion_detected=True,
            motion_region="center",
        )
        summary = state.summary()
        assert "laptop" in summary
        assert "ron" in summary
        assert "motion: center" in summary

    def test_no_motion_explicit(self):
        state = SceneState(objects=["cup"], motion_detected=False)
        assert "motion: none" in state.summary()


# ---------------------------------------------------------------------------
# detect_motion
# ---------------------------------------------------------------------------


class TestDetectMotion:
    def test_no_previous_frame(self):
        frame = np.zeros((360, 640), dtype=np.uint8)
        detected, region = detect_motion(frame, None, 0.05)
        assert detected is False
        assert region == ""

    def test_identical_frames_no_motion(self):
        frame = np.ones((360, 640), dtype=np.uint8) * 128
        detected, region = detect_motion(frame, frame.copy(), 0.05)
        assert detected is False

    def test_different_frames_motion_detected(self):
        prev = np.zeros((360, 640), dtype=np.uint8)
        curr = np.ones((360, 640), dtype=np.uint8) * 255
        detected, region = detect_motion(curr, prev, 0.05)
        assert detected is True
        assert region in ("left", "center", "right")

    def test_motion_region_left(self):
        prev = np.zeros((360, 640), dtype=np.uint8)
        curr = prev.copy()
        # Only the left third changes.
        curr[:, :213] = 255
        detected, region = detect_motion(curr, prev, 0.01)
        assert detected is True
        assert region == "left"

    def test_motion_region_right(self):
        prev = np.zeros((360, 640), dtype=np.uint8)
        curr = prev.copy()
        # Only the right third changes.
        curr[:, 427:] = 255
        detected, region = detect_motion(curr, prev, 0.01)
        assert detected is True
        assert region == "right"

    def test_below_threshold_no_motion(self):
        prev = np.zeros((360, 640), dtype=np.uint8)
        curr = prev.copy()
        # Change only a tiny area.
        curr[0:5, 0:5] = 255
        detected, region = detect_motion(curr, prev, 0.05)
        assert detected is False


# ---------------------------------------------------------------------------
# check_scene_change
# ---------------------------------------------------------------------------


class TestCheckSceneChange:
    def test_no_change(self):
        assert check_scene_change(["mug", "keyboard"], ["mug", "keyboard"], 0.3) is False

    def test_complete_change(self):
        assert check_scene_change(["cat"], ["mug", "keyboard"], 0.3) is True

    def test_partial_change_below_threshold(self):
        # 2 shared out of 3 total = Jaccard distance = 1/3 ≈ 0.33
        assert check_scene_change(["a", "b", "c"], ["a", "b"], 0.5) is False

    def test_partial_change_above_threshold(self):
        assert check_scene_change(["a", "b", "c"], ["a", "b"], 0.3) is True

    def test_empty_to_objects(self):
        assert check_scene_change(["mug"], [], 0.3) is True

    def test_objects_to_empty(self):
        assert check_scene_change([], ["mug"], 0.3) is True

    def test_both_empty(self):
        assert check_scene_change([], [], 0.3) is False


# ---------------------------------------------------------------------------
# detect_objects (with mocked YOLO)
# ---------------------------------------------------------------------------


class TestDetectObjects:
    def test_basic_detection(self):
        # Mock a YOLO model and its output.
        mock_model = MagicMock()
        mock_model.names = {0: "person", 67: "cell phone"}

        box1 = MagicMock()
        box1.conf.item.return_value = 0.85
        box1.cls.item.return_value = 0

        box2 = MagicMock()
        box2.conf.item.return_value = 0.72
        box2.cls.item.return_value = 67

        result = MagicMock()
        result.boxes = [box1, box2]
        mock_model.return_value = [result]

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        labels = detect_objects(frame, mock_model, confidence_threshold=0.5)

        assert labels == ["cell phone", "person"]

    def test_below_confidence_filtered(self):
        mock_model = MagicMock()
        mock_model.names = {0: "person"}

        box = MagicMock()
        box.conf.item.return_value = 0.3  # Below threshold
        box.cls.item.return_value = 0

        result = MagicMock()
        result.boxes = [box]
        mock_model.return_value = [result]

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        labels = detect_objects(frame, mock_model, confidence_threshold=0.5)

        assert labels == []

    def test_duplicate_labels_deduplicated(self):
        mock_model = MagicMock()
        mock_model.names = {0: "person"}

        box1 = MagicMock()
        box1.conf.item.return_value = 0.9
        box1.cls.item.return_value = 0

        box2 = MagicMock()
        box2.conf.item.return_value = 0.8
        box2.cls.item.return_value = 0

        result = MagicMock()
        result.boxes = [box1, box2]
        mock_model.return_value = [result]

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        labels = detect_objects(frame, mock_model, confidence_threshold=0.5)

        assert labels == ["person"]  # Only one entry

    def test_no_detections(self):
        mock_model = MagicMock()
        result = MagicMock()
        result.boxes = []
        mock_model.return_value = [result]

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        labels = detect_objects(frame, mock_model, confidence_threshold=0.5)

        assert labels == []


# ---------------------------------------------------------------------------
# VisionPipeline integration
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> dict:
    config = {
        "models": {"llm": "qwen2.5:3b", "vlm": "qwen2.5-vl:3b"},
        "personality": {"system_prompt": "test"},
        "thresholds": {"max_response_tokens": 150},
        "endpoints": {"ollama": "http://localhost:11434"},
        "vision": {
            "yolo_model": "yolov8n.pt",
            "confidence_threshold": 0.5,
            "motion_threshold": 0.05,
            "scene_change_threshold": 0.3,
        },
    }
    config.update(overrides)
    return config


class TestVisionPipeline:
    def test_initial_scene_state(self):
        vector = MagicMock()
        pipeline = VisionPipeline(_make_config(), vector=vector)
        assert pipeline.scene.objects == []
        assert pipeline.scene.motion_detected is False

    def test_scene_property(self):
        vector = MagicMock()
        pipeline = VisionPipeline(_make_config(), vector=vector)
        pipeline._scene.objects = ["mug"]
        assert pipeline.scene.objects == ["mug"]

    def test_events_queue_exists(self):
        vector = MagicMock()
        pipeline = VisionPipeline(_make_config(), vector=vector)
        assert isinstance(pipeline.events, asyncio.Queue)

    @pytest.mark.asyncio
    async def test_process_frame_detects_motion(self):
        """Process two frames — second should detect motion."""
        vector = MagicMock()
        pipeline = VisionPipeline(_make_config(), vector=vector)
        pipeline._yolo_model = MagicMock()
        result = MagicMock()
        result.boxes = []
        pipeline._yolo_model.return_value = [result]

        # Create two different JPEG-encoded frames.
        import cv2

        frame1 = np.zeros((360, 640, 3), dtype=np.uint8)
        frame2 = np.ones((360, 640, 3), dtype=np.uint8) * 200

        _, buf1 = cv2.imencode(".jpg", frame1)
        _, buf2 = cv2.imencode(".jpg", frame2)

        # First frame — no motion (no previous).
        events1 = await pipeline._process_frame(buf1.tobytes())
        assert not any(e.event_type == "motion" for e in events1)

        # Second frame — motion detected.
        events2 = await pipeline._process_frame(buf2.tobytes())
        assert any(e.event_type == "motion" for e in events2)
        assert pipeline.scene.motion_detected is True

    @pytest.mark.asyncio
    async def test_process_frame_detects_objects(self):
        """YOLO detection updates scene state."""
        vector = MagicMock()
        pipeline = VisionPipeline(_make_config(), vector=vector)

        # Mock YOLO to return a "person" detection.
        mock_model = MagicMock()
        mock_model.names = {0: "person"}
        box = MagicMock()
        box.conf.item.return_value = 0.9
        box.cls.item.return_value = 0
        result = MagicMock()
        result.boxes = [box]
        mock_model.return_value = [result]
        pipeline._yolo_model = mock_model

        import cv2

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", frame)

        events = await pipeline._process_frame(buf.tobytes())

        assert pipeline.scene.objects == ["person"]
        assert any(e.event_type == "objects_changed" for e in events)

    @pytest.mark.asyncio
    async def test_process_frame_no_event_if_objects_unchanged(self):
        """No objects_changed event when objects stay the same."""
        vector = MagicMock()
        pipeline = VisionPipeline(_make_config(), vector=vector)
        pipeline._scene.objects = ["person"]

        mock_model = MagicMock()
        mock_model.names = {0: "person"}
        box = MagicMock()
        box.conf.item.return_value = 0.9
        box.cls.item.return_value = 0
        result = MagicMock()
        result.boxes = [box]
        mock_model.return_value = [result]
        pipeline._yolo_model = mock_model

        import cv2

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", frame)

        # Process twice with same frame — second time should have no object event.
        await pipeline._process_frame(buf.tobytes())
        events = await pipeline._process_frame(buf.tobytes())

        assert not any(e.event_type == "objects_changed" for e in events)
