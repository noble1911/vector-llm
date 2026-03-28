"""Vector SDK wrapper — movement, animations, camera, and expressions.

The anki_vector SDK is synchronous (returns concurrent.futures.Future).
This module wraps it for asyncio by running SDK calls in a thread executor.
"""

from __future__ import annotations

import asyncio
import math
from io import BytesIO
from typing import Any

import structlog

log = structlog.get_logger()

# Head angle limits (degrees).
HEAD_ANGLE_MIN = -22.0
HEAD_ANGLE_MAX = 45.0

# Lift height limits (0.0 = low, 1.0 = high).
LIFT_HEIGHT_MIN = 0.0
LIFT_HEIGHT_MAX = 1.0

# Drive distance limits (mm).
DRIVE_DISTANCE_MAX = 500.0

# Default drive speed (mm/s).
DEFAULT_SPEED_MMPS = 100.0

# Map color names to (hue, saturation) for Vector's eyes.
COLOR_NAME_TO_HS = {
    "red": (0.0, 1.0),
    "orange": (0.05, 1.0),
    "yellow": (0.12, 1.0),
    "green": (0.35, 1.0),
    "cyan": (0.5, 1.0),
    "blue": (0.65, 1.0),
    "purple": (0.78, 1.0),
    "pink": (0.9, 0.7),
    "white": (0.0, 0.0),
}

# Map simple emotion names to actual SDK animation triggers.
EMOTION_TO_TRIGGER = {
    "happy": "DriveEndHappy",
    "sad": "FrustratedByFailureMajor",
    "curious": "ObservingIdleWithHeadLookingUp",
    "surprised": "ReactToUnexpectedMovement",
    "excited": "DriveStartHappy",
    "greeting": "ReactToGreeting",
    "hello": "ReactToGreeting",
    "goodbye": "ReactToGoodBye",
    "love": "Feedback_ILoveYou",
    "bored": "NothingToDoBoredIdle",
    "frustrated": "FrustratedByFailureMajor",
    "goodnight": "ReactToGoodNight",
    "goodmorning": "ReactToGoodMorning",
    "petting": "PettingLevel2",
    "idle": "ObservingIdleEyesOnly",
}


class VectorConnectionError(Exception):
    """Raised when Vector is unreachable or connection fails."""


class VectorController:
    """Async wrapper around the anki_vector SDK.

    All SDK calls are blocking, so they're dispatched to a thread executor
    via asyncio.to_thread to avoid blocking the event loop.
    """

    def __init__(self, config: dict) -> None:
        self._robot: Any = None  # anki_vector.Robot, typed as Any to avoid import at module level
        self._connected = False
        self._camera_feed_active = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def robot(self) -> Any:
        """Direct access to the underlying SDK robot object."""
        return self._robot

    async def connect(self) -> None:
        """Connect to Vector via wire-pod tokens.

        Raises:
            VectorConnectionError: If connection fails.
        """
        try:
            import anki_vector

            def _connect():
                robot = anki_vector.Robot(
                    cache_animation_lists=False,
                    behavior_activation_timeout=30,
                )
                robot.connect()
                return robot

            self._robot = await asyncio.to_thread(_connect)
            self._connected = True
            log.info("vector connected")
        except Exception as e:
            raise VectorConnectionError(f"Failed to connect to Vector: {e}") from e

    async def disconnect(self) -> None:
        """Cleanly disconnect from Vector."""
        if not self._connected or self._robot is None:
            return
        try:
            if self._camera_feed_active:
                await asyncio.to_thread(self._robot.camera.close_camera_feed)
                self._camera_feed_active = False
            await asyncio.to_thread(self._robot.disconnect)
        except Exception:
            log.exception("error during disconnect")
        finally:
            self._connected = False
            self._robot = None
            log.info("vector disconnected")

    def _ensure_connected(self) -> None:
        if not self._connected or self._robot is None:
            raise VectorConnectionError("Not connected to Vector")

    async def _ensure_off_charger(self) -> None:
        """Drive off charger if currently docked, so wheels can move."""
        def _check_and_undock():
            state = self._robot.get_battery_state()
            if state.is_on_charger_platform:
                log.info("vector.undocking_for_movement")
                self._robot.behavior.drive_off_charger()

        await asyncio.to_thread(_check_and_undock)

    # ---- Movement ----

    async def move(self, direction: str, distance_mm: float) -> None:
        """Drive Vector forward or backward.

        Args:
            direction: "forward" or "backward".
            distance_mm: Distance in millimeters (clamped to 0-500).
        """
        self._ensure_connected()
        await self._ensure_off_charger()

        distance_mm = max(0.0, min(float(distance_mm), DRIVE_DISTANCE_MAX))
        if direction == "backward":
            distance_mm = -distance_mm

        from anki_vector.util import Distance, Speed

        def _drive():
            self._robot.behavior.drive_straight(
                Distance(distance_mm=distance_mm),
                Speed(speed_mmps=DEFAULT_SPEED_MMPS),
            )

        await asyncio.to_thread(_drive)
        log.info("vector.move", direction=direction, distance_mm=abs(distance_mm))

    async def turn(self, angle_degrees: float) -> None:
        """Turn Vector by an angle.

        Args:
            angle_degrees: Degrees to turn (positive = left, negative = right).
        """
        self._ensure_connected()
        await self._ensure_off_charger()

        from anki_vector.util import Angle

        def _turn():
            self._robot.behavior.turn_in_place(
                Angle(degrees=float(angle_degrees)),
            )

        await asyncio.to_thread(_turn)
        log.info("vector.turn", angle_degrees=angle_degrees)

    async def set_head_angle(self, angle_degrees: float) -> None:
        """Set Vector's head angle.

        Args:
            angle_degrees: Head angle (-22 to 45 degrees, clamped).
        """
        self._ensure_connected()

        angle_degrees = max(HEAD_ANGLE_MIN, min(float(angle_degrees), HEAD_ANGLE_MAX))

        from anki_vector.util import Angle

        def _set_head():
            self._robot.behavior.set_head_angle(
                Angle(degrees=angle_degrees),
            )

        await asyncio.to_thread(_set_head)
        log.info("vector.set_head_angle", angle_degrees=angle_degrees)

    async def set_lift_height(self, height: float) -> None:
        """Set Vector's lift height.

        Args:
            height: Lift height 0.0 (low) to 1.0 (high), clamped.
        """
        self._ensure_connected()

        height = max(LIFT_HEIGHT_MIN, min(float(height), LIFT_HEIGHT_MAX))

        def _set_lift():
            self._robot.behavior.set_lift_height(height)

        await asyncio.to_thread(_set_lift)
        log.info("vector.set_lift_height", height=height)

    # ---- Speech ----

    async def display_on_screen(self, text: str, duration_sec: float = 2.0) -> None:
        """Display text (including emoji) on Vector's face screen, then restore eyes.

        Args:
            text: Text or emoji to render on the 184x96 screen.
            duration_sec: How long to show it before restoring eyes.
        """
        self._ensure_connected()

        def _display():
            import time as _time
            from PIL import Image, ImageDraw, ImageFont
            import anki_vector.screen

            img = Image.new("RGB", (184, 96), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)

            # Use Apple Color Emoji for full-colour rendering.
            font = None
            for font_path in [
                "/System/Library/Fonts/Apple Color Emoji.ttc",
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            ]:
                try:
                    font = ImageFont.truetype(font_path, 64)
                    break
                except (OSError, IOError):
                    continue

            if font is None:
                font = ImageFont.load_default()

            # Center the text.
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (184 - tw) // 2
            y = (96 - th) // 2
            draw.text((x, y), text, font=font, embedded_color=True)

            screen_data = anki_vector.screen.convert_image_to_screen_data(img)
            self._robot.screen.set_screen_with_image_data(
                screen_data, duration_sec, interrupt_running=True
            )
            # Wait then trigger an animation to restore the default face.
            _time.sleep(duration_sec)
            self._robot.anim.play_animation_trigger("ObservingIdleEyesOnly")

        await asyncio.to_thread(_display)
        log.info("vector.display_on_screen", text=text[:20])

    async def say(self, text: str) -> None:
        """Make Vector speak using his built-in TTS (fallback for Kokoro).

        Args:
            text: Text for Vector to say.
        """
        self._ensure_connected()

        def _say():
            self._robot.behavior.say_text(text)

        await asyncio.to_thread(_say)
        log.info("vector.say", text=text[:50])

    # ---- Animations ----

    async def play_animation(self, name: str) -> None:
        """Play an animation on Vector, mapping emotion names to SDK triggers.

        Args:
            name: Emotion name (e.g. "happy") or raw SDK trigger name.
        """
        self._ensure_connected()

        # Map emotion words to actual trigger names.
        trigger = EMOTION_TO_TRIGGER.get(name.lower(), name)

        def _play():
            self._robot.anim.play_animation_trigger(trigger)

        await asyncio.to_thread(_play)
        log.info("vector.play_animation", emotion=name, trigger=trigger)

    # ---- Eyes ----

    async def set_eye_color(self, hue: float, saturation: float) -> None:
        """Set Vector's eye color.

        Args:
            hue: Hue value 0.0-1.0.
            saturation: Saturation value 0.0-1.0.
        """
        self._ensure_connected()

        hue = max(0.0, min(float(hue), 1.0))
        saturation = max(0.0, min(float(saturation), 1.0))

        def _set_eyes():
            self._robot.behavior.set_eye_color(hue, saturation)

        await asyncio.to_thread(_set_eyes)
        log.info("vector.set_eye_color", hue=hue, saturation=saturation)

    # ---- Camera ----

    async def start_camera_feed(self) -> None:
        """Start the camera streaming feed. Must be called before capture_image."""
        self._ensure_connected()

        if self._camera_feed_active:
            return

        await asyncio.to_thread(self._robot.camera.init_camera_feed)
        self._camera_feed_active = True
        log.info("vector camera feed started")

    async def stop_camera_feed(self) -> None:
        """Stop the camera streaming feed."""
        self._ensure_connected()

        if not self._camera_feed_active:
            return

        await asyncio.to_thread(self._robot.camera.close_camera_feed)
        self._camera_feed_active = False
        log.info("vector camera feed stopped")

    async def capture_image(self, jpeg_quality: int = 75) -> bytes:
        """Capture a single frame from Vector's camera as JPEG.

        The camera feed must be started first via start_camera_feed().

        Args:
            jpeg_quality: JPEG compression quality (1-100).

        Returns:
            JPEG image bytes.

        Raises:
            VectorConnectionError: If camera feed is not active.
        """
        self._ensure_connected()

        if not self._camera_feed_active:
            raise VectorConnectionError(
                "Camera feed not active — call start_camera_feed() first"
            )

        def _capture():
            image = self._robot.camera.capture_single_image()
            buf = BytesIO()
            image.raw_image.save(buf, format="JPEG", quality=jpeg_quality)
            return buf.getvalue()

        return await asyncio.to_thread(_capture)

    async def get_latest_image(self) -> bytes | None:
        """Get the latest frame from the camera feed as JPEG.

        Non-blocking — returns whatever frame is currently buffered.
        Returns None if no frame is available yet.
        """
        self._ensure_connected()

        if not self._camera_feed_active:
            return None

        def _get_latest():
            try:
                img = self._robot.camera.latest_image
                if img is None:
                    return None
                buf = BytesIO()
                img.raw_image.save(buf, format="JPEG", quality=75)
                return buf.getvalue()
            except Exception:
                return None

        return await asyncio.to_thread(_get_latest)

    # ---- Charger ----

    async def dock(self) -> None:
        """Drive Vector onto the charger."""
        self._ensure_connected()
        await asyncio.to_thread(self._robot.behavior.drive_on_charger)
        log.info("vector.dock")

    async def undock(self) -> None:
        """Drive Vector off the charger."""
        self._ensure_connected()
        await asyncio.to_thread(self._robot.behavior.drive_off_charger)
        log.info("vector.undock")

    # ---- Status ----

    async def get_battery_state(self) -> dict:
        """Get Vector's battery state.

        Returns:
            Dict with battery_level, is_charging, is_on_charger_platform, etc.
        """
        self._ensure_connected()

        def _battery():
            state = self._robot.get_battery_state()
            return {
                "battery_level": state.battery_level,
                "is_charging": state.is_charging,
                "is_on_charger_platform": state.is_on_charger_platform,
            }

        return await asyncio.to_thread(_battery)

    # ---- Tool dispatch ----

    async def execute_tool(self, tool_name: str, parameters: dict) -> str:
        """Execute a brain tool call on Vector.

        This is the bridge between Brain tool calls and SDK actions.

        Args:
            tool_name: Tool name from brain response.
            parameters: Tool parameters from brain response.

        Returns:
            Result description string for feeding back to the brain.
        """
        try:
            if tool_name == "move":
                direction = parameters.get("direction", "forward")
                distance = parameters.get("distance_mm", 100)
                await self.move(direction, distance)
                return f"Moved {direction} {distance}mm"

            elif tool_name == "turn":
                angle = abs(float(parameters.get("angle_degrees", 90)))
                direction = parameters.get("direction", "left")
                # Convert direction to signed angle: left=positive, right=negative.
                if direction == "right":
                    angle = -angle
                await self.turn(angle)
                return f"Turned {direction} {abs(angle)} degrees"

            elif tool_name == "set_head_angle":
                angle = parameters.get("angle_degrees", 0)
                await self.set_head_angle(angle)
                return f"Head angle set to {angle} degrees"

            elif tool_name == "play_animation":
                name = parameters.get("name", "")
                await self.play_animation(name)
                return f"Played animation: {name}"

            elif tool_name == "dock":
                await self.dock()
                return "Successfully docked on charger"

            elif tool_name == "undock":
                await self.undock()
                return "Drove off charger, ready to move"

            elif tool_name == "set_eye_color":
                color_name = parameters.get("color", "white")
                hue, sat = COLOR_NAME_TO_HS.get(color_name, (0.0, 0.0))
                await self.set_eye_color(hue, sat)
                return f"Eye color set to {color_name}"

            else:
                return f"Unknown tool: {tool_name}"

        except VectorConnectionError as e:
            return f"Action failed: {e}"
        except Exception as e:
            log.exception("tool execution error", tool=tool_name)
            return f"Action failed: {e}"
