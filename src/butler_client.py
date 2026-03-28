"""Butler API client — escalates complex queries to Claude.

The local LLM handles casual conversation; complex queries (real-time data,
home automation, deep reasoning) are routed to Butler which has Claude + tools.
"""

from __future__ import annotations

import httpx
import structlog

log = structlog.get_logger()

# Default timeout for Butler API calls (longer than local LLM since
# Claude may use tools like web search).
_DEFAULT_TIMEOUT_S = 30.0


class ButlerError(Exception):
    """Raised when Butler API call fails."""


class ButlerClient:
    """HTTP client for the Butler API running on the home server.

    Butler provides Claude-powered chat with access to tools like
    home automation, web search, calendar, etc.
    """

    def __init__(
        self,
        config: dict,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.endpoint = config["endpoints"]["butler_api"]
        self._token: str | None = config.get("butler", {}).get("token")
        self._client = http_client or httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT_S)

    def _auth_headers(self) -> dict[str, str]:
        """Build authorization headers."""
        if not self._token:
            return {}
        return {"Authorization": f"Bearer {self._token}"}

    async def ask(self, question: str) -> str:
        """Send a question to Butler API for a more capable response.

        Args:
            question: The user's complex query.

        Returns:
            Butler/Claude's response text.

        Raises:
            ButlerError: If the API call fails.
        """
        log.info("butler.ask", question=question[:80])

        url = f"{self.endpoint}/api/chat"
        payload = {"message": question, "type": "text"}

        try:
            response = await self._client.post(
                url,
                json=payload,
                headers=self._auth_headers(),
            )
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise ButlerError(
                f"Cannot connect to Butler API at {self.endpoint}: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise ButlerError(
                f"Butler API error {e.response.status_code}: "
                f"{e.response.text[:200]}"
            ) from e
        except httpx.TimeoutException as e:
            raise ButlerError(f"Butler API timeout: {e}") from e

        data = response.json()
        answer = data.get("response", "")

        log.info("butler.response", length=len(answer))
        return answer

    async def is_available(self) -> bool:
        """Check if Butler API is reachable.

        Returns:
            True if the health endpoint responds.
        """
        try:
            response = await self._client.get(f"{self.endpoint}/health")
            return response.status_code == 200
        except Exception:
            return False
