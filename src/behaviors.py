"""Autonomous idle behaviors — makes Vector feel alive when not in conversation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.vector_control import VectorController

log = structlog.get_logger()


class BehaviorEngine:
    """Runs idle behaviors when Vector isn't actively conversing.

    Examples: looking around, small movements, reacting to vision changes.
    """

    def __init__(self, config: dict, *, vector: VectorController) -> None:
        self._vector = vector

    async def run(self, shutdown: asyncio.Event) -> None:
        """Main behavior loop — runs until shutdown is signaled.

        Args:
            shutdown: Event that signals when to stop.
        """
        log.info("behavior engine starting")
        while not shutdown.is_set():
            try:
                await self._idle_tick()
            except NotImplementedError:
                log.debug("behaviors not yet implemented, sleeping")
            except Exception:
                log.exception("behavior engine error")
            await asyncio.sleep(5.0)
        log.info("behavior engine stopped")

    async def _idle_tick(self) -> None:
        """Execute one idle behavior cycle."""
        raise NotImplementedError("BehaviorEngine._idle_tick not yet implemented")
