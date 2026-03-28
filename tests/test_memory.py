"""Tests for persistent memory module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory import (
    EMBEDDING_DIM,
    VALID_CATEGORIES,
    VECTOR_USER_ID,
    EmbeddingClient,
    MemoryContext,
    MemoryStore,
)


# ---------------------------------------------------------------------------
# MemoryContext
# ---------------------------------------------------------------------------


class TestMemoryContext:
    def test_empty_facts_prompt(self):
        ctx = MemoryContext()
        assert ctx.facts_prompt() == ""

    def test_facts_prompt_formatting(self):
        ctx = MemoryContext(facts=[
            {"fact": "Ron likes coffee", "category": "preference"},
            {"fact": "Cat named Luna", "category": "relationship"},
        ])
        result = ctx.facts_prompt()
        assert "WHAT YOU REMEMBER" in result
        assert "[preference] Ron likes coffee" in result
        assert "[relationship] Cat named Luna" in result

    def test_facts_prompt_missing_category(self):
        ctx = MemoryContext(facts=[{"fact": "something"}])
        result = ctx.facts_prompt()
        assert "[other]" in result


# ---------------------------------------------------------------------------
# EmbeddingClient
# ---------------------------------------------------------------------------


class TestEmbeddingClient:
    @pytest.mark.asyncio
    async def test_embed_success(self):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [[0.1] * EMBEDDING_DIM]
        }
        mock_client.post = AsyncMock(return_value=mock_response)

        client = EmbeddingClient("http://localhost:11434", http_client=mock_client)
        result = await client.embed("test text")

        assert result is not None
        assert len(result) == EMBEDDING_DIM
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_http_error(self):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_client.post = AsyncMock(return_value=mock_response)

        client = EmbeddingClient("http://localhost:11434", http_client=mock_client)
        result = await client.embed("test")

        assert result is None

    @pytest.mark.asyncio
    async def test_embed_empty_response(self):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": []}
        mock_client.post = AsyncMock(return_value=mock_response)

        client = EmbeddingClient("http://localhost:11434", http_client=mock_client)
        result = await client.embed("test")

        assert result is None

    @pytest.mark.asyncio
    async def test_embed_wrong_dimension(self):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [[0.1] * 100]}
        mock_client.post = AsyncMock(return_value=mock_response)

        client = EmbeddingClient("http://localhost:11434", http_client=mock_client)
        result = await client.embed("test")

        assert result is None

    @pytest.mark.asyncio
    async def test_embed_connection_error(self):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("connection refused"))

        client = EmbeddingClient("http://localhost:11434", http_client=mock_client)
        result = await client.embed("test")

        assert result is None


# ---------------------------------------------------------------------------
# MemoryStore — unit tests with mocked pool
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> dict:
    config = {
        "database": {
            "url": "postgresql://test:test@localhost/test",
            "user_id": "test-vector",
        },
        "endpoints": {"ollama": "http://localhost:11434"},
    }
    config.update(overrides)
    return config


def _make_store(pool=None, embedding=None) -> MemoryStore:
    """Create a MemoryStore with mocked dependencies."""
    mock_pool = pool or AsyncMock()
    if embedding is None:
        mock_embedding = AsyncMock()
        mock_embedding.embed = AsyncMock(return_value=None)  # No embeddings by default.
    else:
        mock_embedding = embedding

    store = MemoryStore(_make_config(), pool=mock_pool, embedding_client=mock_embedding)
    return store


class TestMemoryStoreAvailability:
    def test_available_with_pool(self):
        store = _make_store()
        assert store.available is True

    def test_unavailable_without_pool(self):
        store = MemoryStore(_make_config())
        assert store.available is False


class TestRemember:
    @pytest.mark.asyncio
    async def test_remember_without_embedding(self):
        mock_pool = AsyncMock()
        store = _make_store(pool=mock_pool)

        await store.remember("Ron likes coffee", category="preference")

        mock_pool.execute.assert_called_once()
        call_args = mock_pool.execute.call_args
        assert "INSERT INTO butler.user_facts" in call_args[0][0]
        assert call_args[0][2] == "Ron likes coffee"
        assert call_args[0][3] == "preference"

    @pytest.mark.asyncio
    async def test_remember_with_embedding(self):
        mock_pool = AsyncMock()
        mock_embedding = AsyncMock()
        mock_embedding.embed = AsyncMock(return_value=[0.1] * EMBEDDING_DIM)

        store = _make_store(pool=mock_pool, embedding=mock_embedding)

        await store.remember("Ron likes coffee", category="preference")

        mock_pool.execute.assert_called_once()
        call_args = mock_pool.execute.call_args
        assert "::vector" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_remember_invalid_category_defaults(self):
        mock_pool = AsyncMock()
        store = _make_store(pool=mock_pool)

        await store.remember("something", category="invalid_category")

        call_args = mock_pool.execute.call_args
        assert call_args[0][3] == "other"

    @pytest.mark.asyncio
    async def test_remember_no_pool_noop(self):
        store = MemoryStore(_make_config())
        # Should not raise.
        await store.remember("test fact")


class TestRecall:
    @pytest.mark.asyncio
    async def test_recall_by_confidence(self):
        mock_pool = AsyncMock()
        mock_pool.fetch = AsyncMock(return_value=[
            {"fact": "likes coffee", "category": "preference", "confidence": 0.9},
        ])
        store = _make_store(pool=mock_pool)

        facts = await store.recall()

        assert len(facts) == 1
        assert facts[0]["fact"] == "likes coffee"

    @pytest.mark.asyncio
    async def test_recall_with_category(self):
        mock_pool = AsyncMock()
        mock_pool.fetch = AsyncMock(return_value=[])
        store = _make_store(pool=mock_pool)

        await store.recall(category="preference")

        call_args = mock_pool.fetch.call_args
        assert "category = $2" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_recall_no_pool_returns_empty(self):
        store = MemoryStore(_make_config())
        facts = await store.recall()
        assert facts == []

    @pytest.mark.asyncio
    async def test_recall_semantic_search(self):
        mock_pool = AsyncMock()
        mock_pool.fetch = AsyncMock(return_value=[
            {"id": 1, "fact": "likes coffee", "category": "preference", "confidence": 0.9},
        ])
        mock_embedding = AsyncMock()
        mock_embedding.embed = AsyncMock(return_value=[0.1] * EMBEDDING_DIM)

        store = _make_store(pool=mock_pool, embedding=mock_embedding)

        facts = await store.recall(query="what drinks does he like")

        assert len(facts) >= 1
        mock_embedding.embed.assert_called_once()


class TestStoreMessage:
    @pytest.mark.asyncio
    async def test_store_message(self):
        mock_pool = AsyncMock()
        store = _make_store(pool=mock_pool)

        await store.store_message("user", "hello vector")

        mock_pool.execute.assert_called_once()
        call_args = mock_pool.execute.call_args
        assert "conversation_history" in call_args[0][0]
        assert call_args[0][3] == "user"
        assert call_args[0][4] == "hello vector"

    @pytest.mark.asyncio
    async def test_store_message_no_pool_noop(self):
        store = MemoryStore(_make_config())
        await store.store_message("user", "hello")


class TestLoadHistory:
    @pytest.mark.asyncio
    async def test_load_history_chronological(self):
        mock_pool = AsyncMock()
        # DB returns DESC order, load_history should reverse.
        mock_pool.fetch = AsyncMock(return_value=[
            {"role": "assistant", "content": "hi there!", "created_at": MagicMock()},
            {"role": "user", "content": "hello", "created_at": MagicMock()},
        ])
        store = _make_store(pool=mock_pool)

        history = await store.load_history()

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_load_history_merges_same_role(self):
        mock_pool = AsyncMock()
        mock_pool.fetch = AsyncMock(return_value=[
            {"role": "user", "content": "part 2", "created_at": MagicMock()},
            {"role": "user", "content": "part 1", "created_at": MagicMock()},
        ])
        store = _make_store(pool=mock_pool)

        history = await store.load_history()

        assert len(history) == 1
        assert "part 1" in history[0]["content"]
        assert "part 2" in history[0]["content"]

    @pytest.mark.asyncio
    async def test_load_history_no_pool_returns_empty(self):
        store = MemoryStore(_make_config())
        history = await store.load_history()
        assert history == []


class TestLoadContext:
    @pytest.mark.asyncio
    async def test_load_context_combines_facts_and_history(self):
        mock_pool = AsyncMock()
        # recall() and load_history() both call fetch — return facts first, then empty history.
        mock_pool.fetch = AsyncMock(side_effect=[
            [{"fact": "likes coffee", "category": "preference", "confidence": 0.9}],
            [],  # history
        ])
        store = _make_store(pool=mock_pool)

        ctx = await store.load_context(current_message="what do I like")

        assert isinstance(ctx, MemoryContext)
        assert len(ctx.facts) == 1

    @pytest.mark.asyncio
    async def test_load_context_no_pool_returns_empty(self):
        store = MemoryStore(_make_config())
        ctx = await store.load_context()

        assert ctx.facts == []
        assert ctx.history == []


class TestAutoLearn:
    @pytest.mark.asyncio
    async def test_short_message_skipped(self):
        mock_pool = AsyncMock()
        store = _make_store(pool=mock_pool)

        await store.auto_learn("hi", "hello!")

        # Should not have tried to extract facts.
        mock_pool.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_pool_noop(self):
        store = MemoryStore(_make_config())
        await store.auto_learn("I love coffee in the morning", "Great taste!")

    @pytest.mark.asyncio
    async def test_extract_facts_parses_json(self):
        mock_pool = AsyncMock()
        store = _make_store(pool=mock_pool)

        fake_response = {
            "message": {
                "content": json.dumps([
                    {"fact": "likes coffee", "category": "preference", "confidence": 0.8}
                ])
            }
        }
        mock_http_response = MagicMock()
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = fake_response

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_http_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client_instance

            await store.auto_learn(
                "I really love coffee every morning",
                "That's a great habit!",
            )

        # Should have stored the extracted fact.
        assert mock_pool.execute.call_count >= 1
