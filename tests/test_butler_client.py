"""Tests for the Butler API client — escalation to Claude."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.butler_client import ButlerClient, ButlerError


def _make_config(**overrides) -> dict:
    config = {
        "endpoints": {"butler_api": "http://localhost:8000"},
    }
    config.update(overrides)
    return config


class TestButlerClientAsk:
    @pytest.mark.asyncio
    async def test_ask_success(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "response": "It's 3pm and sunny.",
            "message_id": "abc123",
        }
        mock_client.post = AsyncMock(return_value=mock_response)

        butler = ButlerClient(_make_config(), http_client=mock_client)
        result = await butler.ask("What's the weather?")

        assert result == "It's 3pm and sunny."
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/api/chat" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["message"] == "What's the weather?"
        assert payload["type"] == "text"

    @pytest.mark.asyncio
    async def test_ask_with_auth_token(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "response": "Done.",
            "message_id": "x",
        }
        mock_client.post = AsyncMock(return_value=mock_response)

        config = _make_config(butler={"token": "my-jwt-token"})
        butler = ButlerClient(config, http_client=mock_client)
        await butler.ask("Turn off the lights")

        headers = mock_client.post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer my-jwt-token"

    @pytest.mark.asyncio
    async def test_ask_without_token_no_auth_header(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "response": "Ok.",
            "message_id": "x",
        }
        mock_client.post = AsyncMock(return_value=mock_response)

        butler = ButlerClient(_make_config(), http_client=mock_client)
        await butler.ask("Hello")

        headers = mock_client.post.call_args[1]["headers"]
        assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_ask_connection_error(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        butler = ButlerClient(_make_config(), http_client=mock_client)

        with pytest.raises(ButlerError, match="Cannot connect"):
            await butler.ask("Hello?")

    @pytest.mark.asyncio
    async def test_ask_http_error(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "401", request=MagicMock(), response=mock_response
            )
        )
        mock_client.post = AsyncMock(return_value=mock_response)

        butler = ButlerClient(_make_config(), http_client=mock_client)

        with pytest.raises(ButlerError, match="error 401"):
            await butler.ask("Hello?")

    @pytest.mark.asyncio
    async def test_ask_timeout(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        butler = ButlerClient(_make_config(), http_client=mock_client)

        with pytest.raises(ButlerError, match="timeout"):
            await butler.ask("Complex question")

    @pytest.mark.asyncio
    async def test_ask_empty_response(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"response": "", "message_id": "x"}
        mock_client.post = AsyncMock(return_value=mock_response)

        butler = ButlerClient(_make_config(), http_client=mock_client)
        result = await butler.ask("Hmm")
        assert result == ""


class TestButlerClientAvailability:
    @pytest.mark.asyncio
    async def test_is_available_true(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_response)

        butler = ButlerClient(_make_config(), http_client=mock_client)
        assert await butler.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_false_on_error(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        butler = ButlerClient(_make_config(), http_client=mock_client)
        assert await butler.is_available() is False
