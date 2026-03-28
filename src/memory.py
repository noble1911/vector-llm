"""Persistent memory via Postgres+pgvector — reuses Butler's schema.

Stores and retrieves:
- User facts with semantic embeddings (butler.user_facts)
- Conversation history (butler.conversation_history)

Vector uses its own user_id ('vector-robot') in the same tables Butler uses,
keeping memories separate while sharing the infrastructure.

Embeddings are generated via Ollama's nomic-embed-text model (768 dimensions).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import asyncpg
import httpx
import structlog

log = structlog.get_logger()

# Embedding config — must match Butler's setup.
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768

# Vector's identity in the database.
VECTOR_USER_ID = "vector-robot"
VECTOR_CHANNEL = "voice"

# Valid fact categories (matches Butler's schema).
VALID_CATEGORIES = frozenset({
    "preference", "schedule", "relationship", "work", "health",
    "environment", "observation", "other",
})


@dataclass
class MemoryContext:
    """Loaded context for injection into the brain's system prompt."""

    facts: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)

    def facts_prompt(self) -> str:
        """Format facts for inclusion in the system prompt."""
        if not self.facts:
            return ""

        lines = ["\n[Memories]"]
        for fact in self.facts:
            category = fact.get("category", "other")
            lines.append(f"- [{category}] {fact['fact']}")
        return "\n".join(lines)

    def history_prompt(self) -> str:
        """Format recent conversation history for context continuity."""
        if not self.history:
            return ""

        lines = ["\n[Recent history]"]
        for msg in self.history[-8:]:  # Last 8 messages
            role = msg.get("role", "?")
            content = msg.get("content", "")[:100]
            lines.append(f"- {role}: {content}")
        return "\n".join(lines)


class EmbeddingClient:
    """Generate text embeddings via Ollama's local API."""

    def __init__(self, ollama_url: str, *, http_client: httpx.AsyncClient | None = None) -> None:
        self._url = ollama_url.rstrip("/")
        self._client = http_client or httpx.AsyncClient(timeout=10.0)

    async def embed(self, text: str) -> list[float] | None:
        """Generate an embedding vector for the given text.

        Returns None on any failure so callers degrade gracefully.
        """
        try:
            resp = await self._client.post(
                f"{self._url}/api/embed",
                json={"model": EMBEDDING_MODEL, "input": text},
            )
            if resp.status_code != 200:
                log.warning("embedding.failed", status=resp.status_code)
                return None
            data = resp.json()
            embeddings = data.get("embeddings")
            if not embeddings or not embeddings[0]:
                return None
            vector = embeddings[0]
            if len(vector) != EMBEDDING_DIM:
                log.warning("embedding.dim_mismatch", got=len(vector), expected=EMBEDDING_DIM)
                return None
            return vector
        except Exception as e:
            log.warning("embedding.error", error=str(e))
            return None


class MemoryStore:
    """Persistent memory backed by Postgres+pgvector.

    Reads and writes to Butler's existing butler.* tables using
    Vector's own user_id for isolation.
    """

    def __init__(
        self,
        config: dict,
        *,
        pool: asyncpg.Pool | None = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        db_cfg = config.get("database", {})
        self._db_url = db_cfg.get(
            "url", "postgresql://postgres:postgres@localhost:5432/immich"
        )
        self._pool = pool
        self._embedding = embedding_client or EmbeddingClient(
            config.get("endpoints", {}).get("ollama", "http://localhost:11434")
        )
        self._user_id = db_cfg.get("user_id", VECTOR_USER_ID)
        self._llm_model = config.get("models", {}).get("llm", "qwen3:8b")

    async def connect(self) -> None:
        """Create the connection pool and ensure Vector's user exists."""
        if self._pool is not None:
            return

        try:
            self._pool = await asyncpg.create_pool(
                self._db_url,
                min_size=1,
                max_size=5,
                init=self._init_connection,
            )
            await self._ensure_user()
            log.info("memory.connected", user_id=self._user_id)
        except Exception as e:
            log.warning("memory.connect_failed", error=str(e))
            self._pool = None

    @staticmethod
    async def _init_connection(conn: asyncpg.Connection) -> None:
        """Set up JSON codecs for each connection."""
        await conn.set_type_codec(
            "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
        )
        await conn.set_type_codec(
            "json", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
        )

    async def _ensure_user(self) -> None:
        """Create Vector's user record if it doesn't exist."""
        if not self._pool:
            return
        await self._pool.execute(
            """
            INSERT INTO butler.users (id, name, soul)
            VALUES ($1, $2, $3::jsonb)
            ON CONFLICT (id) DO NOTHING
            """,
            self._user_id,
            "Vector",
            json.dumps({
                "personality": "curious and playful desktop robot",
                "verbosity": "concise",
                "humor": "light",
            }),
        )

    @property
    def available(self) -> bool:
        """Whether the memory store is connected and usable."""
        return self._pool is not None

    async def close(self) -> None:
        """Close the connection pool and HTTP clients."""
        if self._pool:
            await self._pool.close()
            self._pool = None
        if hasattr(self._embedding, "_client"):
            await self._embedding._client.aclose()

    # --- Fact storage ---

    async def remember(
        self,
        fact: str,
        *,
        category: str = "other",
        confidence: float = 1.0,
        source: str = "conversation",
    ) -> None:
        """Store a fact about the world or a person Vector has met."""
        if not self._pool:
            return

        if category not in VALID_CATEGORIES:
            category = "other"

        embedding = await self._embedding.embed(fact)

        if embedding is not None:
            vector_str = "[" + ",".join(str(v) for v in embedding) + "]"
            await self._pool.execute(
                """
                INSERT INTO butler.user_facts
                    (user_id, fact, category, confidence, source, embedding)
                VALUES ($1, $2, $3, $4, $5, $6::vector)
                """,
                self._user_id, fact, category, confidence, source, vector_str,
            )
        else:
            await self._pool.execute(
                """
                INSERT INTO butler.user_facts
                    (user_id, fact, category, confidence, source)
                VALUES ($1, $2, $3, $4, $5)
                """,
                self._user_id, fact, category, confidence, source,
            )

        log.debug("memory.remembered", fact=fact[:60], category=category)

    async def recall(
        self,
        query: str | None = None,
        *,
        category: str | None = None,
        limit: int = 15,
    ) -> list[dict]:
        """Recall stored facts, optionally using semantic search.

        Args:
            query: Semantic search query. If provided, finds facts by meaning.
            category: Filter by fact category.
            limit: Maximum number of facts to return.

        Returns:
            List of fact dicts with 'fact', 'category', 'confidence' keys.
        """
        if not self._pool:
            return []

        # Try semantic search if we have a query.
        if query:
            embedding = await self._embedding.embed(query)
            if embedding is not None:
                return await self._semantic_recall(embedding, category, limit)

        # Fallback: top facts by confidence.
        if category:
            rows = await self._pool.fetch(
                """
                SELECT fact, category, confidence
                FROM butler.user_facts
                WHERE user_id = $1 AND category = $2
                  AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY confidence DESC, created_at DESC
                LIMIT $3
                """,
                self._user_id, category, limit,
            )
        else:
            rows = await self._pool.fetch(
                """
                SELECT fact, category, confidence
                FROM butler.user_facts
                WHERE user_id = $1
                  AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY confidence DESC, created_at DESC
                LIMIT $2
                """,
                self._user_id, limit,
            )

        return [dict(r) for r in rows]

    async def _semantic_recall(
        self,
        query_embedding: list[float],
        category: str | None,
        limit: int,
    ) -> list[dict]:
        """Hybrid recall: semantic similarity + top confidence, deduplicated."""
        if not self._pool:
            return []

        vector_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        # Semantic: closest by embedding.
        if category:
            semantic_rows = await self._pool.fetch(
                """
                SELECT id, fact, category, confidence
                FROM butler.user_facts
                WHERE user_id = $1 AND category = $2
                  AND embedding IS NOT NULL
                  AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY embedding <=> $3::vector
                LIMIT $4
                """,
                self._user_id, category, vector_str, limit,
            )
        else:
            semantic_rows = await self._pool.fetch(
                """
                SELECT id, fact, category, confidence
                FROM butler.user_facts
                WHERE user_id = $1
                  AND embedding IS NOT NULL
                  AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY embedding <=> $2::vector
                LIMIT $3
                """,
                self._user_id, vector_str, limit,
            )

        # Confidence: top general-purpose facts.
        confidence_rows = await self._pool.fetch(
            """
            SELECT id, fact, category, confidence
            FROM butler.user_facts
            WHERE user_id = $1
              AND (expires_at IS NULL OR expires_at > NOW())
            ORDER BY confidence DESC, created_at DESC
            LIMIT $2
            """,
            self._user_id, limit,
        )

        # Deduplicate, semantic first.
        seen = set()
        combined = []
        for row in list(semantic_rows) + list(confidence_rows):
            fid = row["id"]
            if fid not in seen:
                seen.add(fid)
                combined.append(dict(row))

        return combined[:limit]

    # --- Conversation history ---

    async def store_message(self, role: str, content: str) -> None:
        """Store a conversation message."""
        if not self._pool:
            return

        await self._pool.execute(
            """
            INSERT INTO butler.conversation_history
                (user_id, channel, role, content)
            VALUES ($1, $2, $3, $4)
            """,
            self._user_id, VECTOR_CHANNEL, role, content,
        )

    async def load_history(self, limit: int = 10, days: int = 7) -> list[dict]:
        """Load recent conversation history in chronological order."""
        if not self._pool:
            return []

        rows = await self._pool.fetch(
            """
            SELECT role, content, created_at
            FROM butler.conversation_history
            WHERE user_id = $1 AND channel = $2
              AND created_at > NOW() - INTERVAL '1 day' * $3
            ORDER BY created_at DESC, id DESC
            LIMIT $4
            """,
            self._user_id, VECTOR_CHANNEL, days, limit,
        )

        # Reverse to chronological, merge consecutive same-role.
        messages: list[dict] = []
        for row in reversed(rows):
            role = row["role"]
            content = row["content"]
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] += "\n" + content
            else:
                messages.append({"role": role, "content": content})

        return messages

    # --- Context loading ---

    async def load_context(self, current_message: str | None = None) -> MemoryContext:
        """Load full context for a brain call — facts + recent history.

        Args:
            current_message: The user's current message, used for
                semantic fact search to find relevant memories.

        Returns:
            MemoryContext with facts and history ready for prompt injection.
        """
        if not self._pool:
            return MemoryContext()

        facts = await self.recall(query=current_message, limit=15)
        history = await self.load_history(limit=10)

        return MemoryContext(facts=facts, history=history)

    # --- Auto-learning ---

    async def auto_learn(
        self,
        user_message: str,
        assistant_response: str,
        *,
        ollama_url: str | None = None,
    ) -> None:
        """Extract and store facts from a conversation turn.

        Uses the local LLM (via Ollama) to identify personal facts,
        preferences, and observations. Runs as a background task.
        """
        if not self._pool:
            return

        if len(user_message) < 5:
            return

        try:
            facts = await self._extract_facts(
                user_message, assistant_response, ollama_url=ollama_url
            )
            for fact_data in facts:
                await self.remember(
                    fact_data["fact"],
                    category=fact_data.get("category", "other"),
                    confidence=fact_data.get("confidence", 0.7),
                    source="auto_extraction",
                )

            if facts:
                log.info("memory.auto_learned", count=len(facts))

        except Exception:
            log.exception("memory.auto_learn_failed")

    async def _extract_facts(
        self,
        user_message: str,
        assistant_response: str,
        *,
        ollama_url: str | None = None,
    ) -> list[dict]:
        """Use the local LLM to extract facts from a conversation."""
        url = ollama_url or "http://localhost:11434"

        prompt = (
            "Analyze this conversation and extract personal facts about the user "
            "or notable observations about the environment.\n\n"
            f"User: {user_message}\n"
            f"Assistant: {assistant_response[:500]}\n\n"
            "Return a JSON array of facts. Each: "
            '{"fact": "...", "category": "preference|schedule|relationship|'
            'environment|observation|other", "confidence": 0.5-0.9}\n'
            "Return [] if nothing to learn. JSON only, no other text."
        )

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{url}/api/chat",
                json={
                    "model": self._llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"num_predict": 256},
                },
            )
            if resp.status_code != 200:
                return []

            text = resp.json().get("message", {}).get("content", "").strip()

        try:
            facts = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from wrapped response.
            import re
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                try:
                    facts = json.loads(match.group())
                except json.JSONDecodeError:
                    return []
            else:
                return []

        if not isinstance(facts, list):
            return []

        validated = []
        for item in facts:
            if not isinstance(item, dict) or "fact" not in item:
                continue
            category = item.get("category", "other")
            if category not in VALID_CATEGORIES:
                category = "other"
            confidence = item.get("confidence", 0.7)
            if not isinstance(confidence, (int, float)):
                confidence = 0.7
            confidence = max(0.5, min(0.9, float(confidence)))
            validated.append({
                "fact": str(item["fact"])[:500],
                "category": category,
                "confidence": confidence,
            })

        return validated
