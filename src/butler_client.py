"""Butler API client — escalates complex queries to Claude."""

from __future__ import annotations

import structlog

log = structlog.get_logger()


class ButlerClient:
    """HTTP client for the Butler API running on the home server."""

    def __init__(self, config: dict) -> None:
        self.endpoint = config["endpoints"]["butler_api"]

    async def ask(self, question: str) -> str:
        """Send a question to Butler API for a more capable response.

        Args:
            question: The user's complex query.

        Returns:
            Butler/Claude's response text.
        """
        log.info("escalating to butler", question=question[:80])
        raise NotImplementedError("ButlerClient.ask not yet implemented")
