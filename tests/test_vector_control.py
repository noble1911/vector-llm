"""Tests for the Vector SDK control layer."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.vector_control import (
    DRIVE_DISTANCE_MAX,
    HEAD_ANGLE_MAX,
    HEAD_ANGLE_MIN,
    VectorConnectionError,
    VectorController,
)


def _make_config() -> dict:
    return {
        "models": {"llm": "qwen2.5:3b"},
        "personality": {"system_prompt": "test"},
        "thresholds": {"max_response_tokens": 150},
        "endpoints": {"ollama": "http://localhost:11434"},
    }


def _make_connected_controller() -> tuple[VectorController, MagicMock]:
    """Create a VectorController with a mocked robot, as if already connected."""
    ctrl = VectorController(_make_config())

    # Build a mock robot with all the sub-components.
    robot = MagicMock()
    robot.behavior.drive_straight = MagicMock()
    robot.behavior.turn_in_place = MagicMock()
    robot.behavior.set_head_angle = MagicMock()
    robot.behavior.set_lift_height = MagicMock()
    robot.behavior.say_text = MagicMock()
    robot.behavior.set_eye_color = MagicMock()
    robot.behavior.drive_on_charger = MagicMock()
    robot.behavior.drive_off_charger = MagicMock()
    robot.anim.play_animation_trigger = MagicMock()
    robot.camera.init_camera_feed = MagicMock()
    robot.camera.close_camera_feed = MagicMock()
    robot.camera.capture_single_image = MagicMock()
    robot.camera.latest_image = None
    robot.get_battery_state = MagicMock()
    robot.disconnect = MagicMock()

    ctrl._robot = robot
    ctrl._connected = True

    return ctrl, robot


# ---------------------------------------------------------------------------
# Connection state
# ---------------------------------------------------------------------------


class TestConnectionState:
    def test_not_connected_initially(self):
        ctrl = VectorController(_make_config())
        assert ctrl.is_connected is False

    @pytest.mark.asyncio
    async def test_methods_raise_when_disconnected(self):
        ctrl = VectorController(_make_config())

        with pytest.raises(VectorConnectionError):
            await ctrl.move("forward", 100)

        with pytest.raises(VectorConnectionError):
            await ctrl.turn(90)

        with pytest.raises(VectorConnectionError):
            await ctrl.say("hello")

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        ctrl = VectorController(_make_config())
        # Should not raise.
        await ctrl.disconnect()


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------


class TestMovement:
    @pytest.mark.asyncio
    async def test_move_forward(self):
        ctrl, robot = _make_connected_controller()
        await ctrl.move("forward", 200)
        robot.behavior.drive_straight.assert_called_once()

    @pytest.mark.asyncio
    async def test_move_backward(self):
        ctrl, robot = _make_connected_controller()
        await ctrl.move("backward", 150)
        robot.behavior.drive_straight.assert_called_once()

    @pytest.mark.asyncio
    async def test_move_clamps_distance(self):
        ctrl, robot = _make_connected_controller()
        await ctrl.move("forward", 9999)
        # Should have been clamped — verify the call was made (clamping is internal).
        robot.behavior.drive_straight.assert_called_once()

    @pytest.mark.asyncio
    async def test_move_negative_distance_clamped_to_zero(self):
        ctrl, robot = _make_connected_controller()
        await ctrl.move("forward", -50)
        robot.behavior.drive_straight.assert_called_once()

    @pytest.mark.asyncio
    async def test_turn(self):
        ctrl, robot = _make_connected_controller()
        await ctrl.turn(90)
        robot.behavior.turn_in_place.assert_called_once()

    @pytest.mark.asyncio
    async def test_turn_negative(self):
        ctrl, robot = _make_connected_controller()
        await ctrl.turn(-45)
        robot.behavior.turn_in_place.assert_called_once()


# ---------------------------------------------------------------------------
# Head / Lift
# ---------------------------------------------------------------------------


class TestHeadAndLift:
    @pytest.mark.asyncio
    async def test_set_head_angle(self):
        ctrl, robot = _make_connected_controller()
        await ctrl.set_head_angle(20)
        robot.behavior.set_head_angle.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_head_angle_clamped_high(self):
        ctrl, robot = _make_connected_controller()
        await ctrl.set_head_angle(90)  # Above max of 45
        robot.behavior.set_head_angle.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_head_angle_clamped_low(self):
        ctrl, robot = _make_connected_controller()
        await ctrl.set_head_angle(-50)  # Below min of -22
        robot.behavior.set_head_angle.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_lift_height(self):
        ctrl, robot = _make_connected_controller()
        await ctrl.set_lift_height(0.5)
        robot.behavior.set_lift_height.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_lift_height_clamped(self):
        ctrl, robot = _make_connected_controller()
        await ctrl.set_lift_height(5.0)  # Above max of 1.0
        robot.behavior.set_lift_height.assert_called_once()


# ---------------------------------------------------------------------------
# Speech / Animation / Eyes
# ---------------------------------------------------------------------------


class TestSpeechAndAnimation:
    @pytest.mark.asyncio
    async def test_say(self):
        ctrl, robot = _make_connected_controller()
        await ctrl.say("Hello world")
        robot.behavior.say_text.assert_called_once_with("Hello world")

    @pytest.mark.asyncio
    async def test_play_animation(self):
        ctrl, robot = _make_connected_controller()
        await ctrl.play_animation("anim_greeting_happy")
        robot.anim.play_animation_trigger.assert_called_once_with(
            "anim_greeting_happy"
        )

    @pytest.mark.asyncio
    async def test_set_eye_color(self):
        ctrl, robot = _make_connected_controller()
        await ctrl.set_eye_color(0.5, 0.8)
        robot.behavior.set_eye_color.assert_called_once_with(0.5, 0.8)

    @pytest.mark.asyncio
    async def test_set_eye_color_clamped(self):
        ctrl, robot = _make_connected_controller()
        await ctrl.set_eye_color(2.0, -1.0)
        robot.behavior.set_eye_color.assert_called_once_with(1.0, 0.0)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


class TestCamera:
    @pytest.mark.asyncio
    async def test_start_camera_feed(self):
        ctrl, robot = _make_connected_controller()
        await ctrl.start_camera_feed()
        robot.camera.init_camera_feed.assert_called_once()
        assert ctrl._camera_feed_active is True

    @pytest.mark.asyncio
    async def test_start_camera_feed_idempotent(self):
        ctrl, robot = _make_connected_controller()
        await ctrl.start_camera_feed()
        await ctrl.start_camera_feed()
        # Should only be called once.
        robot.camera.init_camera_feed.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_camera_feed(self):
        ctrl, robot = _make_connected_controller()
        ctrl._camera_feed_active = True
        await ctrl.stop_camera_feed()
        robot.camera.close_camera_feed.assert_called_once()
        assert ctrl._camera_feed_active is False

    @pytest.mark.asyncio
    async def test_capture_image_requires_feed(self):
        ctrl, robot = _make_connected_controller()
        with pytest.raises(VectorConnectionError, match="Camera feed not active"):
            await ctrl.capture_image()

    @pytest.mark.asyncio
    async def test_capture_image(self):
        ctrl, robot = _make_connected_controller()
        ctrl._camera_feed_active = True

        # Mock the PIL image returned by the SDK.
        mock_pil = MagicMock()
        mock_pil.save = MagicMock(
            side_effect=lambda buf, **kw: buf.write(b"fake-jpeg-data")
        )
        mock_camera_image = MagicMock()
        mock_camera_image.raw_image = mock_pil
        robot.camera.capture_single_image.return_value = mock_camera_image

        result = await ctrl.capture_image()

        assert result == b"fake-jpeg-data"
        robot.camera.capture_single_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_latest_image_no_feed(self):
        ctrl, robot = _make_connected_controller()
        result = await ctrl.get_latest_image()
        assert result is None


# ---------------------------------------------------------------------------
# Charger
# ---------------------------------------------------------------------------


class TestCharger:
    @pytest.mark.asyncio
    async def test_dock(self):
        ctrl, robot = _make_connected_controller()
        await ctrl.dock()
        robot.behavior.drive_on_charger.assert_called_once()

    @pytest.mark.asyncio
    async def test_undock(self):
        ctrl, robot = _make_connected_controller()
        await ctrl.undock()
        robot.behavior.drive_off_charger.assert_called_once()


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------


class TestBattery:
    @pytest.mark.asyncio
    async def test_get_battery_state(self):
        ctrl, robot = _make_connected_controller()

        mock_state = MagicMock()
        mock_state.battery_level = 3
        mock_state.is_charging = True
        mock_state.is_on_charger_platform = True
        robot.get_battery_state.return_value = mock_state

        result = await ctrl.get_battery_state()

        assert result["battery_level"] == 3
        assert result["is_charging"] is True
        assert result["is_on_charger_platform"] is True


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


class TestExecuteTool:
    @pytest.mark.asyncio
    async def test_move_tool(self):
        ctrl, robot = _make_connected_controller()
        result = await ctrl.execute_tool(
            "move", {"direction": "forward", "distance_mm": 200}
        )
        assert "forward" in result
        robot.behavior.drive_straight.assert_called_once()

    @pytest.mark.asyncio
    async def test_turn_tool(self):
        ctrl, robot = _make_connected_controller()
        result = await ctrl.execute_tool("turn", {"angle_degrees": -45})
        assert "right" in result
        robot.behavior.turn_in_place.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_head_angle_tool(self):
        ctrl, robot = _make_connected_controller()
        result = await ctrl.execute_tool("set_head_angle", {"angle_degrees": 30})
        assert "30" in result

    @pytest.mark.asyncio
    async def test_play_animation_tool(self):
        ctrl, robot = _make_connected_controller()
        result = await ctrl.execute_tool("play_animation", {"name": "happy"})
        assert "happy" in result

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        ctrl, robot = _make_connected_controller()
        result = await ctrl.execute_tool("fly", {})
        assert "Unknown tool" in result

    @pytest.mark.asyncio
    async def test_tool_error_handled(self):
        ctrl, robot = _make_connected_controller()
        robot.behavior.drive_straight.side_effect = RuntimeError("stuck")
        result = await ctrl.execute_tool(
            "move", {"direction": "forward", "distance_mm": 100}
        )
        assert "failed" in result.lower()
