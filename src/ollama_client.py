"""Async HTTP client for Ollama's chat API."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx
import structlog

log = structlog.get_logger()

# Ollama /api/chat request timeout — generous to allow for cold model loading.
_DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)


@dataclass
class ChatMessage:
    """A single message in a conversation."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatResponse:
    """Response from Ollama /api/chat."""

    message: ChatMessage
    done: bool = True
    total_duration_ns: int = 0
    eval_count: int = 0

    @property
    def total_duration_ms(self) -> float:
        return self.total_duration_ns / 1_000_000


class OllamaClient:
    """Async HTTP client for Ollama's /api/chat endpoint."""

    def __init__(self, base_url: str, timeout: httpx.Timeout | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout or _DEFAULT_TIMEOUT

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 150,
    ) -> ChatResponse:
        """Send a chat completion request to Ollama.

        Args:
            model: Model name (e.g. "qwen2.5:3b").
            messages: Conversation messages.
            max_tokens: Maximum tokens in the response.

        Returns:
            ChatResponse with the assistant's reply.

        Raises:
            OllamaConnectionError: If Ollama is unreachable.
            OllamaAPIError: If Ollama returns an error response.
        """
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "num_predict": max_tokens,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                )
        except httpx.ConnectError as e:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self._base_url}"
            ) from e
        except httpx.TimeoutException as e:
            raise OllamaConnectionError(
                f"Ollama request timed out at {self._base_url}"
            ) from e

        if response.status_code != 200:
            raise OllamaAPIError(
                f"Ollama returned {response.status_code}: {response.text}"
            )

        data = response.json()
        msg = data.get("message", {})

        return ChatResponse(
            message=ChatMessage(
                role=msg.get("role", "assistant"),
                content=msg.get("content", ""),
            ),
            done=data.get("done", True),
            total_duration_ns=data.get("total_duration", 0),
            eval_count=data.get("eval_count", 0),
        )

    async def is_available(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.get(f"{self._base_url}/api/tags")
                return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False


class OllamaConnectionError(Exception):
    """Raised when Ollama is unreachable."""


class OllamaAPIError(Exception):
    """Raised when Ollama returns an error response."""
