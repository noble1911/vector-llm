"""State machine — manages Vector's behavioral states.

States:
  EXPLORING  — firmware controls Vector, we passively observe via STT + YOLO
  CONVERSING — we hold SDK control, brain responds to speech
  LEARNING   — brief SDK control to look at something interesting
"""

from __future__ import annotations

import asyncio
import enum
import time
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.conversation import ConversationManager
    from src.vector_control import VectorController
    from src.vision import VisionPipeline

log = structlog.get_logger()

# Timeout before returning to EXPLORING from CONVERSING.
_CONVERSATION_TIMEOUT_S = 30.0

# Cooldown between LEARNING triggers.
_LEARNING_COOLDOWN_S = 60.0


class State(enum.Enum):
    EXPLORING = "exploring"
    CONVERSING = "conversing"
    LEARNING = "learning"


class StateMachine:
    """Manages transitions between Vector's behavioral states."""

    def __init__(
        self,
        *,
        vector: VectorController,
        conversation: ConversationManager,
        vision: VisionPipeline | None = None,
    ) -> None:
        self._vector = vector
        self._conversation = conversation
        self._vision = vision
        self._state = State.EXPLORING
        self._last_interaction_at: float = 0.0
        self._last_learning_at: float = 0.0
        self._has_control = False  # Control acquired on demand via request_control()

    @property
    def state(self) -> State:
        return self._state

    async def transition_to(self, new_state: State) -> None:
        """Transition to a new state, handling control acquire/release."""
        if new_state == self._state:
            return

        old_state = self._state
        self._state = new_state
        log.info("state.transition", old=old_state.value, new=new_state.value)

        if new_state == State.EXPLORING:
            await self._enter_exploring()
        elif new_state == State.CONVERSING:
            await self._enter_conversing()
        elif new_state == State.LEARNING:
            await self._enter_learning()

    async def _enter_exploring(self) -> None:
        """Release control — let Vector's firmware take over."""
        self._conversation.end_conversation()
        if self._has_control:
            try:
                await asyncio.to_thread(
                    self._vector.robot.conn.release_control
                )
                self._has_control = False
                log.info("state.control_released")
            except Exception:
                log.exception("state.release_control_failed")

    async def _enter_conversing(self) -> None:
        """Acquire control for conversation."""
        self._last_interaction_at = time.monotonic()
        if not self._has_control:
            try:
                await asyncio.to_thread(
                    self._vector.robot.conn.request_control, timeout=10.0
                )
                self._has_control = True
                log.info("state.control_acquired")
            except Exception:
                log.exception("state.request_control_failed")

    async def _enter_learning(self) -> None:
        """Acquire control briefly to look at something."""
        self._last_learning_at = time.monotonic()
        if not self._has_control:
            try:
                await asyncio.to_thread(
                    self._vector.robot.conn.request_control, timeout=10.0
                )
                self._has_control = True
                log.info("state.control_acquired_for_learning")
            except Exception:
                log.exception("state.request_control_failed")

    def touch_interaction(self) -> None:
        """Mark that an interaction just happened (resets timeout)."""
        self._last_interaction_at = time.monotonic()

    def can_learn(self) -> bool:
        """Whether enough time has passed since the last LEARNING trigger."""
        return (time.monotonic() - self._last_learning_at) >= _LEARNING_COOLDOWN_S

    async def run(self, shutdown: asyncio.Event) -> None:
        """Main state machine loop — handles timeouts and transitions."""
        log.info("state_machine.started", state=self._state.value)

        # Start in EXPLORING — release control.
        await self.transition_to(State.EXPLORING)

        while not shutdown.is_set():
            try:
                now = time.monotonic()

                # CONVERSING → EXPLORING on timeout.
                if self._state == State.CONVERSING:
                    if (now - self._last_interaction_at) > _CONVERSATION_TIMEOUT_S:
                        log.info("state.conversation_timeout")
                        await self.transition_to(State.EXPLORING)

                # LEARNING → EXPLORING after look completes (handled by caller).
                # Just a safety timeout here.
                if self._state == State.LEARNING:
                    if (now - self._last_learning_at) > 15.0:
                        await self.transition_to(State.EXPLORING)

                await asyncio.sleep(1.0)

            except Exception:
                log.exception("state_machine.error")
                await asyncio.sleep(5.0)

        log.info("state_machine.stopped")
