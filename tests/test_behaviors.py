"""Tests for autonomous behavior engine."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.behaviors import (
    BehaviorEngine,
    _LOW_BATTERY_LEVEL,
    is_quiet_hours,
    pick_idle_behavior,
)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


class TestIsQuietHours:
    def test_late_night(self):
        assert is_quiet_hours(23) is True

    def test_midnight(self):
        assert is_quiet_hours(0) is True

    def test_early_morning(self):
        assert is_quiet_hours(5) is True

    def test_boundary_start(self):
        assert is_quiet_hours(22) is True

    def test_boundary_end(self):
        assert is_quiet_hours(7) is False

    def test_daytime(self):
        assert is_quiet_hours(12) is False

    def test_evening(self):
        assert is_quiet_hours(20) is False


class TestPickIdleBehavior:
    def test_quiet_mode_only_subtle(self):
        for _ in range(50):
            behavior = pick_idle_behavior(quiet=True)
            assert behavior in ("look_around", "head_tilt")

    def test_normal_mode_includes_all(self):
        behaviors_seen = set()
        for _ in range(200):
            behaviors_seen.add(pick_idle_behavior(quiet=False))
        # Should see at least look_around and one active behavior.
        assert "look_around" in behaviors_seen
        assert len(behaviors_seen) >= 3


# ---------------------------------------------------------------------------
# BehaviorEngine
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> dict:
    config: dict = {}
    config.update(overrides)
    return config


def _make_vector() -> MagicMock:
    vector = MagicMock()
    vector.set_head_angle = AsyncMock()
    vector.turn = AsyncMock()
    vector.move = AsyncMock()
    vector.play_animation = AsyncMock()
    vector.dock = AsyncMock()
    vector.get_battery_state = AsyncMock(
        return_value={"battery_level": 3, "is_on_charger_platform": False}
    )
    return vector


class TestBehaviorEngineInit:
    def test_default_config(self):
        vector = _make_vector()
        engine = BehaviorEngine(_make_config(), vector=vector)
        assert engine._idle_interval == 15.0

    def test_custom_interval(self):
        vector = _make_vector()
        config = _make_config(behaviors={"idle_interval_seconds": 30.0})
        engine = BehaviorEngine(config, vector=vector)
        assert engine._idle_interval == 30.0


class TestIdleTick:
    @pytest.mark.asyncio
    async def test_look_around(self):
        vector = _make_vector()
        engine = BehaviorEngine(_make_config(), vector=vector)

        with patch("src.behaviors.pick_idle_behavior", return_value="look_around"):
            await engine._idle_tick()

        vector.set_head_angle.assert_called_once()

    @pytest.mark.asyncio
    async def test_head_tilt(self):
        vector = _make_vector()
        engine = BehaviorEngine(_make_config(), vector=vector)

        with patch("src.behaviors.pick_idle_behavior", return_value="head_tilt"):
            await engine._idle_tick()

        vector.set_head_angle.assert_called_once()

    @pytest.mark.asyncio
    async def test_wander_small(self):
        vector = _make_vector()
        engine = BehaviorEngine(_make_config(), vector=vector)

        with patch("src.behaviors.pick_idle_behavior", return_value="wander_small"):
            await engine._idle_tick()

        vector.turn.assert_called_once()
        vector.move.assert_called_once()

    @pytest.mark.asyncio
    async def test_curious_animation(self):
        vector = _make_vector()
        engine = BehaviorEngine(_make_config(), vector=vector)

        with patch(
            "src.behaviors.pick_idle_behavior", return_value="curious_animation"
        ):
            await engine._idle_tick()

        vector.play_animation.assert_called_once()


class TestBatteryCheck:
    @pytest.mark.asyncio
    async def test_low_battery_docks(self):
        vector = _make_vector()
        vector.get_battery_state = AsyncMock(
            return_value={
                "battery_level": _LOW_BATTERY_LEVEL,
                "is_on_charger_platform": False,
            }
        )

        engine = BehaviorEngine(_make_config(), vector=vector)
        docked = await engine._check_battery()

        assert docked is True
        vector.dock.assert_called_once()

    @pytest.mark.asyncio
    async def test_low_battery_already_on_charger(self):
        vector = _make_vector()
        vector.get_battery_state = AsyncMock(
            return_value={
                "battery_level": _LOW_BATTERY_LEVEL,
                "is_on_charger_platform": True,
            }
        )

        engine = BehaviorEngine(_make_config(), vector=vector)
        docked = await engine._check_battery()

        assert docked is False
        vector.dock.assert_not_called()

    @pytest.mark.asyncio
    async def test_normal_battery_no_dock(self):
        vector = _make_vector()
        engine = BehaviorEngine(_make_config(), vector=vector)
        docked = await engine._check_battery()

        assert docked is False
        vector.dock.assert_not_called()

    @pytest.mark.asyncio
    async def test_battery_check_error_ignored(self):
        vector = _make_vector()
        vector.get_battery_state = AsyncMock(side_effect=RuntimeError("disconnected"))

        engine = BehaviorEngine(_make_config(), vector=vector)
        docked = await engine._check_battery()

        assert docked is False


class TestConversationYielding:
    @pytest.mark.asyncio
    async def test_skips_tick_when_conversation_active(self):
        vector = _make_vector()
        conversation = MagicMock()
        conversation.is_active = True

        engine = BehaviorEngine(
            _make_config(), vector=vector, conversation=conversation
        )

        # Run the loop briefly.
        shutdown = asyncio.Event()

        async def stop_soon():
            await asyncio.sleep(0.15)
            shutdown.set()

        await asyncio.gather(engine.run(shutdown), stop_soon())

        # No behaviors should have been executed.
        vector.set_head_angle.assert_not_called()
        vector.turn.assert_not_called()
        vector.move.assert_not_called()
        vector.play_animation.assert_not_called()

    @pytest.mark.asyncio
    async def test_runs_when_no_conversation(self):
        vector = _make_vector()
        conversation = MagicMock()
        conversation.is_active = False

        engine = BehaviorEngine(
            _make_config(behaviors={"idle_interval_seconds": 0.05}),
            vector=vector,
            conversation=conversation,
        )

        shutdown = asyncio.Event()

        async def stop_soon():
            await asyncio.sleep(0.15)
            shutdown.set()

        await asyncio.gather(engine.run(shutdown), stop_soon())

        # At least one behavior should have executed.
        total_calls = (
            vector.set_head_angle.call_count
            + vector.turn.call_count
            + vector.move.call_count
            + vector.play_animation.call_count
        )
        assert total_calls >= 1


class TestRunLoop:
    @pytest.mark.asyncio
    async def test_shutdown_stops_loop(self):
        vector = _make_vector()
        engine = BehaviorEngine(
            _make_config(behaviors={"idle_interval_seconds": 0.01}),
            vector=vector,
        )

        shutdown = asyncio.Event()

        async def stop_soon():
            await asyncio.sleep(0.05)
            shutdown.set()

        await asyncio.gather(engine.run(shutdown), stop_soon())
        # Should complete without hanging.

    @pytest.mark.asyncio
    async def test_error_in_tick_doesnt_crash_loop(self):
        vector = _make_vector()
        vector.set_head_angle = AsyncMock(side_effect=RuntimeError("oops"))
        vector.turn = AsyncMock(side_effect=RuntimeError("oops"))
        vector.move = AsyncMock(side_effect=RuntimeError("oops"))
        vector.play_animation = AsyncMock(side_effect=RuntimeError("oops"))

        engine = BehaviorEngine(
            _make_config(behaviors={"idle_interval_seconds": 0.01}),
            vector=vector,
        )

        shutdown = asyncio.Event()

        async def stop_soon():
            await asyncio.sleep(0.1)
            shutdown.set()

        # Should not raise despite all actions failing.
        await asyncio.gather(engine.run(shutdown), stop_soon())
