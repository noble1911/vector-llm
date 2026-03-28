"""Tests for the Brain reasoning engine."""

from __future__ import annotations

import pytest

from src.brain import (
    Brain,
    BrainResponse,
    _MAX_CONTEXT_MESSAGES,
    build_tool_prompt,
    parse_response,
)
from src.ollama_client import ChatMessage, ChatResponse, OllamaClient


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_plain_speech(self):
        result = parse_response("Hello! I'm Vector.")
        assert result.speech == "Hello! I'm Vector."
        assert result.tool_calls == []

    def test_single_tool_call(self):
        result = parse_response('Let me look! [look({})]')
        assert result.speech == "Let me look!"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "look"
        assert result.tool_calls[0]["parameters"] == {}

    def test_tool_call_with_params(self):
        result = parse_response(
            'Moving forward! [move({"direction": "forward", "distance_mm": 100})]'
        )
        assert result.speech == "Moving forward!"
        assert result.tool_calls[0]["name"] == "move"
        assert result.tool_calls[0]["parameters"]["direction"] == "forward"
        assert result.tool_calls[0]["parameters"]["distance_mm"] == 100

    def test_multiple_tool_calls(self):
        text = 'Interesting! [look({})] [turn({"angle_degrees": 45})]'
        result = parse_response(text)
        assert result.speech == "Interesting!"
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "look"
        assert result.tool_calls[1]["name"] == "turn"

    def test_tool_call_only_no_speech(self):
        result = parse_response('[look({})]')
        assert result.speech == ""
        assert len(result.tool_calls) == 1

    def test_malformed_json_skipped(self):
        result = parse_response("Hmm [look({bad json})]")
        assert result.speech == "Hmm [look({bad json})]"
        assert result.tool_calls == []

    def test_empty_string(self):
        result = parse_response("")
        assert result.speech == ""
        assert result.tool_calls == []


# ---------------------------------------------------------------------------
# build_tool_prompt
# ---------------------------------------------------------------------------


class TestBuildToolPrompt:
    def test_contains_all_tool_names(self):
        prompt = build_tool_prompt()
        assert "look" in prompt
        assert "move" in prompt
        assert "turn" in prompt
        assert "set_head_angle" in prompt
        assert "play_animation" in prompt

    def test_contains_format_instruction(self):
        prompt = build_tool_prompt()
        assert "[tool_name(" in prompt

    def test_contains_example(self):
        prompt = build_tool_prompt()
        assert "[look({})]" in prompt


# ---------------------------------------------------------------------------
# Brain
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> dict:
    """Create a minimal valid config dict for Brain."""
    config = {
        "models": {"llm": "qwen2.5:3b"},
        "personality": {
            "system_prompt": "You are Vector, a small robot.",
        },
        "thresholds": {"max_response_tokens": 150},
        "endpoints": {"ollama": "http://localhost:11434"},
    }
    config.update(overrides)
    return config


class FakeOllamaClient(OllamaClient):
    """OllamaClient that returns canned responses without HTTP calls."""

    def __init__(self, responses: list[str]) -> None:
        super().__init__("http://fake:11434")
        self._responses = list(responses)
        self.requests: list[dict] = []  # Recorded for assertions.

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 150,
    ) -> ChatResponse:
        self.requests.append({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        })
        text = self._responses.pop(0) if self._responses else ""
        return ChatResponse(
            message=ChatMessage(role="assistant", content=text),
            total_duration_ns=50_000_000,  # 50ms
            eval_count=10,
        )


class TestBrain:
    @pytest.mark.asyncio
    async def test_think_returns_speech(self):
        fake = FakeOllamaClient(["Hello there!"])
        brain = Brain(_make_config(), client=fake)

        result = await brain.think("Hi Vector")

        assert result.speech == "Hello there!"
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_think_returns_tool_calls(self):
        fake = FakeOllamaClient(["Let me see! [look({})]"])
        brain = Brain(_make_config(), client=fake)

        result = await brain.think("What do you see?")

        assert result.speech == "Let me see!"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "look"

    @pytest.mark.asyncio
    async def test_think_sends_system_prompt(self):
        fake = FakeOllamaClient(["Hi!"])
        brain = Brain(_make_config(), client=fake)

        await brain.think("Hello")

        messages = fake.requests[0]["messages"]
        assert messages[0].role == "system"
        assert "Vector" in messages[0].content

    @pytest.mark.asyncio
    async def test_think_includes_tool_definitions(self):
        fake = FakeOllamaClient(["Hi!"])
        brain = Brain(_make_config(), client=fake)

        await brain.think("Hello")

        system_msg = fake.requests[0]["messages"][0].content
        assert "look" in system_msg
        assert "move" in system_msg
        assert "[tool_name(" in system_msg

    @pytest.mark.asyncio
    async def test_conversation_history_maintained(self):
        fake = FakeOllamaClient(["First reply", "Second reply"])
        brain = Brain(_make_config(), client=fake)

        await brain.think("Message 1")
        await brain.think("Message 2")

        # Second request should include both exchanges.
        messages = fake.requests[1]["messages"]
        roles = [m.role for m in messages]
        # system, user, assistant, user
        assert roles == ["system", "user", "assistant", "user"]

    @pytest.mark.asyncio
    async def test_context_trimmed_at_max(self):
        responses = [f"Reply {i}" for i in range(_MAX_CONTEXT_MESSAGES + 5)]
        fake = FakeOllamaClient(responses)
        brain = Brain(_make_config(), client=fake)

        for i in range(_MAX_CONTEXT_MESSAGES + 5):
            await brain.think(f"Message {i}")

        assert len(brain._context) == _MAX_CONTEXT_MESSAGES

    @pytest.mark.asyncio
    async def test_reset_context_clears_history(self):
        fake = FakeOllamaClient(["Reply"])
        brain = Brain(_make_config(), client=fake)

        await brain.think("Hello")
        assert len(brain._context) == 2  # user + assistant

        brain.reset_context()
        assert len(brain._context) == 0

    @pytest.mark.asyncio
    async def test_handle_tool_result(self):
        fake = FakeOllamaClient(["I see a mug on the desk!"])
        brain = Brain(_make_config(), client=fake)

        result = await brain.handle_tool_result(
            "look", "A coffee mug and a keyboard on a wooden desk."
        )

        assert "mug" in result.speech
        # The tool result should be in context as a user message.
        messages = fake.requests[0]["messages"]
        tool_msg = [m for m in messages if "look result" in m.content]
        assert len(tool_msg) == 1

    @pytest.mark.asyncio
    async def test_handle_vision_event_with_reaction(self):
        fake = FakeOllamaClient(["Oh, someone's here!"])
        brain = Brain(_make_config(), client=fake)

        result = await brain.handle_vision_event("face detected: unknown person")

        assert result is not None
        assert result.speech == "Oh, someone's here!"

    @pytest.mark.asyncio
    async def test_handle_vision_event_ignored(self):
        fake = FakeOllamaClient([""])
        brain = Brain(_make_config(), client=fake)

        result = await brain.handle_vision_event("motion: slight movement left")

        assert result is None

    @pytest.mark.asyncio
    async def test_uses_configured_model(self):
        config = _make_config()
        config["models"]["llm"] = "custom-model:7b"
        fake = FakeOllamaClient(["Hi"])
        brain = Brain(config, client=fake)

        await brain.think("Hello")

        assert fake.requests[0]["model"] == "custom-model:7b"

    @pytest.mark.asyncio
    async def test_vision_context_injected(self):
        """When vision is connected, scene summary appears in system prompt."""
        from src.vision import SceneState, VisionPipeline

        fake = FakeOllamaClient(["I see things!"])
        brain = Brain(_make_config(), client=fake)

        # Create a mock-like vision object with a known scene.
        class FakeVision:
            scene = SceneState(
                objects=["mug", "keyboard"],
                motion_detected=True,
                motion_region="left",
            )

        brain.set_vision(FakeVision())  # type: ignore[arg-type]

        await brain.think("What do you see?")

        system_msg = fake.requests[0]["messages"][0].content
        assert "mug" in system_msg
        assert "keyboard" in system_msg
        assert "motion: left" in system_msg
