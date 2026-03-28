"""Autonomous idle behaviors — makes Vector feel alive when not in conversation.

Priority system (highest first):
  1. Conversation — brain is handling user speech (behavior engine yields)
  2. Low battery — return to charger
  3. Quiet hours — minimal activity at night
  4. Stimulus reaction — person/motion detected, comment if interesting
  5. Idle exploration — wander, look around, head movements
"""

from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.conversation import ConversationManager
    from src.vector_control import VectorController

log = structlog.get_logger()

# Battery threshold to trigger return-to-charger.
_LOW_BATTERY_LEVEL = 1  # battery_level enum: 1 = low

# Quiet hours — minimal behavior during these times.
_QUIET_HOUR_START = 22  # 10 PM
_QUIET_HOUR_END = 7  # 7 AM

# Timing between behaviors.
_IDLE_INTERVAL_S = 15.0  # Base interval between idle actions
_QUIET_INTERVAL_S = 60.0  # Longer interval during quiet hours


def pick_idle_behavior(*, quiet: bool) -> str:
    """Pick a random idle behavior weighted by context.

    Args:
        quiet: Whether we're in quiet hours.

    Returns:
        Behavior name to execute.
    """
    if quiet:
        # During quiet hours, only subtle head movements.
        return random.choice(["look_around", "look_around", "head_tilt"])

    behaviors = [
        "look_around",
        "look_around",
        "look_around",
        "head_tilt",
        "head_tilt",
        "wander_small",
        "curious_animation",
    ]
    return random.choice(behaviors)


def is_quiet_hours(hour: int) -> bool:
    """Check if the given hour falls in quiet hours.

    Args:
        hour: Hour of the day (0-23).

    Returns:
        True if in quiet hours.
    """
    if _QUIET_HOUR_START > _QUIET_HOUR_END:
        # Wraps midnight: e.g. 22-7
        return hour >= _QUIET_HOUR_START or hour < _QUIET_HOUR_END
    return _QUIET_HOUR_START <= hour < _QUIET_HOUR_END


class BehaviorEngine:
    """Runs idle behaviors when Vector isn't actively conversing.

    Behaviors are interruptible — if a conversation starts, the engine
    yields immediately and waits until the conversation ends.
    """

    def __init__(
        self,
        config: dict,
        *,
        vector: VectorController,
        conversation: ConversationManager | None = None,
    ) -> None:
        self._vector = vector
        self._conversation = conversation
        self._idle_interval = config.get("behaviors", {}).get(
            "idle_interval_seconds", _IDLE_INTERVAL_S
        )

    async def run(self, shutdown: asyncio.Event) -> None:
        """Main behavior loop — runs until shutdown is signaled.

        Args:
            shutdown: Event that signals when to stop.
        """
        log.info("behavior_engine.started")

        while not shutdown.is_set():
            try:
                # Skip if conversation is active.
                if self._conversation and self._conversation.is_active:
                    await asyncio.sleep(1.0)
                    continue

                quiet = is_quiet_hours(datetime.now().hour)
                interval = _QUIET_INTERVAL_S if quiet else self._idle_interval

                await self._idle_tick(quiet=quiet)
                await asyncio.sleep(interval)

            except Exception:
                log.exception("behavior_engine.error")
                await asyncio.sleep(5.0)

        log.info("behavior_engine.stopped")

    async def _idle_tick(self, *, quiet: bool = False) -> None:
        """Execute one idle behavior cycle.

        Args:
            quiet: Whether we're in quiet hours.
        """
        # Check battery first.
        if await self._check_battery():
            return

        behavior = pick_idle_behavior(quiet=quiet)

        log.debug("behavior_engine.tick", behavior=behavior, quiet=quiet)

        try:
            if behavior == "look_around":
                await self._look_around()
            elif behavior == "head_tilt":
                await self._head_tilt()
            elif behavior == "wander_small":
                await self._wander_small()
            elif behavior == "curious_animation":
                await self._curious_animation()
        except Exception:
            log.debug("behavior_engine.action_failed", behavior=behavior)

    async def _check_battery(self) -> bool:
        """Check battery and dock if low. Returns True if docking."""
        try:
            state = await self._vector.get_battery_state()
            if state.get("battery_level") == _LOW_BATTERY_LEVEL:
                if not state.get("is_on_charger_platform"):
                    log.info("behavior_engine.low_battery_docking")
                    await self._vector.dock()
                    return True
        except Exception:
            pass
        return False

    async def _look_around(self) -> None:
        """Scan the environment with head movements."""
        angle = random.uniform(-15.0, 35.0)
        await self._vector.set_head_angle(angle)

    async def _head_tilt(self) -> None:
        """Small curious head tilt."""
        angle = random.uniform(-5.0, 15.0)
        await self._vector.set_head_angle(angle)

    async def _wander_small(self) -> None:
        """Small exploratory movement."""
        # Turn a random direction, then move a small distance.
        turn_angle = random.uniform(-45.0, 45.0)
        await self._vector.turn(turn_angle)
        await asyncio.sleep(0.5)
        distance = random.uniform(30.0, 100.0)
        await self._vector.move("forward", distance)

    async def _curious_animation(self) -> None:
        """Play a curious or idle animation."""
        animations = [
            "anim_explorer_idle_01",
            "anim_observing_look_up_01",
            "anim_observing_self_01",
        ]
        await self._vector.play_animation(random.choice(animations))
