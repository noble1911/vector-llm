"""Tests for the Ollama HTTP client."""

from __future__ import annotations

import json

import httpx
import pytest

from src.ollama_client import (
    ChatMessage,
    ChatResponse,
    OllamaAPIError,
    OllamaClient,
    OllamaConnectionError,
)


def _ollama_response(content: str = "Hello!", **kwargs) -> dict:
    """Build a realistic Ollama /api/chat response body."""
    return {
        "model": "qwen2.5:3b",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "total_duration": 150_000_000,
        "eval_count": 12,
        **kwargs,
    }


class TestChatMessage:
    def test_creation(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"


class TestChatResponse:
    def test_duration_ms(self):
        resp = ChatResponse(
            message=ChatMessage(role="assistant", content="Hi"),
            total_duration_ns=150_000_000,
        )
        assert resp.total_duration_ms == 150.0

    def test_zero_duration(self):
        resp = ChatResponse(
            message=ChatMessage(role="assistant", content="Hi"),
        )
        assert resp.total_duration_ms == 0.0


class TestOllamaClient:
    @pytest.mark.asyncio
    async def test_chat_success(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:11434/api/chat",
            json=_ollama_response("I'm Vector!"),
        )

        client = OllamaClient("http://localhost:11434")
        result = await client.chat(
            model="qwen2.5:3b",
            messages=[ChatMessage(role="user", content="Hi")],
        )

        assert result.message.content == "I'm Vector!"
        assert result.message.role == "assistant"
        assert result.done is True
        assert result.total_duration_ms == 150.0
        assert result.eval_count == 12

    @pytest.mark.asyncio
    async def test_chat_sends_correct_payload(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:11434/api/chat",
            json=_ollama_response(),
        )

        client = OllamaClient("http://localhost:11434")
        await client.chat(
            model="test-model",
            messages=[
                ChatMessage(role="system", content="Be helpful"),
                ChatMessage(role="user", content="Hello"),
            ],
            max_tokens=200,
        )

        request = httpx_mock.get_request()
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["stream"] is False
        assert body["options"]["num_predict"] == 200
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_chat_connection_error(self, httpx_mock):
        httpx_mock.add_exception(httpx.ConnectError("refused"))

        client = OllamaClient("http://localhost:11434")

        with pytest.raises(OllamaConnectionError, match="Cannot connect"):
            await client.chat(
                model="qwen2.5:3b",
                messages=[ChatMessage(role="user", content="Hi")],
            )

    @pytest.mark.asyncio
    async def test_chat_timeout_error(self, httpx_mock):
        httpx_mock.add_exception(httpx.ReadTimeout("timed out"))

        client = OllamaClient("http://localhost:11434")

        with pytest.raises(OllamaConnectionError, match="timed out"):
            await client.chat(
                model="qwen2.5:3b",
                messages=[ChatMessage(role="user", content="Hi")],
            )

    @pytest.mark.asyncio
    async def test_chat_api_error(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:11434/api/chat",
            status_code=404,
            text="model not found",
        )

        client = OllamaClient("http://localhost:11434")

        with pytest.raises(OllamaAPIError, match="404"):
            await client.chat(
                model="nonexistent",
                messages=[ChatMessage(role="user", content="Hi")],
            )

    @pytest.mark.asyncio
    async def test_trailing_slash_stripped(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:11434/api/chat",
            json=_ollama_response(),
        )

        client = OllamaClient("http://localhost:11434/")
        result = await client.chat(
            model="qwen2.5:3b",
            messages=[ChatMessage(role="user", content="Hi")],
        )
        assert result.message.content == "Hello!"

    @pytest.mark.asyncio
    async def test_is_available_true(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:11434/api/tags",
            json={"models": []},
        )

        client = OllamaClient("http://localhost:11434")
        assert await client.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_false(self, httpx_mock):
        httpx_mock.add_exception(
            httpx.ConnectError("refused"),
            url="http://localhost:11434/api/tags",
        )

        client = OllamaClient("http://localhost:11434")
        assert await client.is_available() is False
