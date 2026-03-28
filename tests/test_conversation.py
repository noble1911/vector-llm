"""Tests for the conversation manager — always-on orchestration."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.brain import BrainResponse
from src.conversation import ConversationManager


# ---------------------------------------------------------------------------
# ConversationManager
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> dict:
    config = {
        "models": {"llm": "qwen2.5:3b"},
        "personality": {"system_prompt": "test"},
        "thresholds": {
            "max_response_tokens": 150,
            "conversation_timeout_seconds": 30.0,
        },
        "endpoints": {"ollama": "http://localhost:11434"},
    }
    config.update(overrides)
    return config


def _make_brain_response(speech: str = "", tool_calls: list | None = None) -> BrainResponse:
    return BrainResponse(speech=speech, tool_calls=tool_calls or [])


class TestConversationManagerInit:
    def test_default_config(self):
        brain = MagicMock()
        mgr = ConversationManager(_make_config(), brain=brain)
        assert mgr.timeout_s == 30.0
        assert mgr.is_active is False

    def test_custom_timeout(self):
        config = _make_config(
            thresholds={
                "max_response_tokens": 150,
                "conversation_timeout_seconds": 60.0,
            }
        )
        brain = MagicMock()
        mgr = ConversationManager(config, brain=brain)
        assert mgr.timeout_s == 60.0


class TestHandleTranscription:
    @pytest.mark.asyncio
    async def test_any_speech_activates_conversation(self):
        brain = MagicMock()
        brain.think = AsyncMock(return_value=_make_brain_response("Hello!"))
        mgr = ConversationManager(_make_config(), brain=brain)

        result = await mgr.handle_transcription("what's up?")

        assert mgr.is_active is True
        assert result is not None
        assert result.speech == "Hello!"
        brain.think.assert_called_once_with("what's up?")

    @pytest.mark.asyncio
    async def test_all_speech_gets_response(self):
        """No speech is ignored — always-on mode."""
        brain = MagicMock()
        brain.think = AsyncMock(return_value=_make_brain_response("Hi!"))
        mgr = ConversationManager(_make_config(), brain=brain)

        result = await mgr.handle_transcription("I was talking about the weather")

        assert result is not None
        brain.think.assert_called_once()

    @pytest.mark.asyncio
    async def test_speech_output_updates_last_spoke(self):
        brain = MagicMock()
        brain.think = AsyncMock(return_value=_make_brain_response("Hi there!"))

        vector = MagicMock()
        vector.say = AsyncMock()
        mgr = ConversationManager(_make_config(), brain=brain, vector=vector)

        before = mgr.last_spoke_at
        await mgr.handle_transcription("hello")
        assert mgr.last_spoke_at > before

    @pytest.mark.asyncio
    async def test_empty_speech_no_tts_call(self):
        brain = MagicMock()
        brain.think = AsyncMock(return_value=_make_brain_response(""))
        vector = MagicMock()
        vector.say = AsyncMock()
        mgr = ConversationManager(_make_config(), brain=brain, vector=vector)

        await mgr.handle_transcription("hello")
        vector.say.assert_not_called()

    @pytest.mark.asyncio
    async def test_conversation_timeout_resets(self):
        brain = MagicMock()
        brain.think = AsyncMock(return_value=_make_brain_response("Hi!"))
        brain.reset_context = MagicMock()
        mgr = ConversationManager(_make_config(), brain=brain)

        # Activate conversation, then simulate timeout.
        mgr._active = True
        mgr._last_interaction_at = time.monotonic() - 60.0  # 60s ago, > 30s timeout

        await mgr.handle_transcription("hello again")

        # Should have ended and restarted.
        brain.reset_context.assert_called_once()


class TestToolExecution:
    @pytest.mark.asyncio
    async def test_tool_calls_dispatched_to_vector(self):
        brain = MagicMock()
        brain.think = AsyncMock(
            return_value=_make_brain_response(
                "Let me move!",
                [{"name": "move", "parameters": {"direction": "forward", "distance_mm": 100}}],
            )
        )
        brain.handle_tool_result = AsyncMock(
            return_value=_make_brain_response("Done moving!")
        )

        vector = MagicMock()
        vector.execute_tool = AsyncMock(return_value="Moved forward 100mm")
        vector.say = AsyncMock()

        mgr = ConversationManager(_make_config(), brain=brain, vector=vector)

        await mgr.handle_transcription("move forward")

        vector.execute_tool.assert_called_once_with(
            "move", {"direction": "forward", "distance_mm": 100}
        )
        brain.handle_tool_result.assert_called_once_with("move", "Moved forward 100mm")

    @pytest.mark.asyncio
    async def test_look_tool_uses_vision_pipeline(self):
        brain = MagicMock()
        brain.think = AsyncMock(
            return_value=_make_brain_response(
                "Let me look.",
                [{"name": "look", "parameters": {}}],
            )
        )
        brain.handle_tool_result = AsyncMock(
            return_value=_make_brain_response("I see a mug!")
        )

        vision = MagicMock()
        vision.describe_scene = AsyncMock(return_value="A coffee mug on a wooden desk.")

        vector = MagicMock()
        vector.say = AsyncMock()

        mgr = ConversationManager(
            _make_config(), brain=brain, vector=vector, vision=vision
        )

        await mgr.handle_transcription("what do you see?")

        vision.describe_scene.assert_called_once()
        brain.handle_tool_result.assert_called_once_with(
            "look", "A coffee mug on a wooden desk."
        )

    @pytest.mark.asyncio
    async def test_look_without_vision_returns_unavailable(self):
        brain = MagicMock()
        brain.think = AsyncMock(
            return_value=_make_brain_response(
                "Let me look.",
                [{"name": "look", "parameters": {}}],
            )
        )
        brain.handle_tool_result = AsyncMock(
            return_value=_make_brain_response("I can't see right now.")
        )

        mgr = ConversationManager(_make_config(), brain=brain)

        await mgr.handle_transcription("look around")

        brain.handle_tool_result.assert_called_once_with("look", "[no vision available]")

    @pytest.mark.asyncio
    async def test_max_tool_rounds_prevents_infinite_loop(self):
        """Tool calls that keep producing more tool calls should stop."""
        brain = MagicMock()

        loop_response = _make_brain_response(
            "Still going...",
            [{"name": "move", "parameters": {"direction": "forward", "distance_mm": 10}}],
        )
        brain.think = AsyncMock(return_value=loop_response)
        brain.handle_tool_result = AsyncMock(return_value=loop_response)

        vector = MagicMock()
        vector.execute_tool = AsyncMock(return_value="Moved")
        vector.say = AsyncMock()

        mgr = ConversationManager(_make_config(), brain=brain, vector=vector)

        await mgr.handle_transcription("keep moving")

        assert vector.execute_tool.call_count <= 5

    @pytest.mark.asyncio
    async def test_no_tool_calls_no_dispatch(self):
        brain = MagicMock()
        brain.think = AsyncMock(return_value=_make_brain_response("Just talking."))

        vector = MagicMock()
        vector.execute_tool = AsyncMock()
        vector.say = AsyncMock()

        mgr = ConversationManager(_make_config(), brain=brain, vector=vector)

        await mgr.handle_transcription("how are you?")

        vector.execute_tool.assert_not_called()


class TestButlerEscalation:
    @pytest.mark.asyncio
    async def test_ask_butler_tool_dispatches_to_butler(self):
        brain = MagicMock()
        brain.think = AsyncMock(
            return_value=_make_brain_response(
                "Let me check with Butler.",
                [{"name": "ask_butler", "parameters": {"question": "What's the weather?"}}],
            )
        )
        brain.handle_tool_result = AsyncMock(
            return_value=_make_brain_response("Butler says it's sunny!")
        )

        butler = MagicMock()
        butler.ask = AsyncMock(return_value="It's 22°C and sunny in London.")

        vector = MagicMock()
        vector.say = AsyncMock()

        mgr = ConversationManager(
            _make_config(), brain=brain, vector=vector, butler=butler
        )

        await mgr.handle_transcription("what's the weather?")

        butler.ask.assert_called_once_with("What's the weather?")
        brain.handle_tool_result.assert_called_once_with(
            "ask_butler", "It's 22°C and sunny in London."
        )

    @pytest.mark.asyncio
    async def test_butler_unavailable_returns_error(self):
        brain = MagicMock()
        brain.think = AsyncMock(
            return_value=_make_brain_response(
                "Let me ask Butler.",
                [{"name": "ask_butler", "parameters": {"question": "Test"}}],
            )
        )
        brain.handle_tool_result = AsyncMock(
            return_value=_make_brain_response("Sorry, Butler is down.")
        )

        butler = MagicMock()
        butler.ask = AsyncMock(side_effect=RuntimeError("Connection refused"))

        mgr = ConversationManager(_make_config(), brain=brain, butler=butler)

        await mgr.handle_transcription("ask Butler something")

        result_arg = brain.handle_tool_result.call_args[0][1]
        assert "unavailable" in result_arg.lower() or "Butler" in result_arg

    @pytest.mark.asyncio
    async def test_no_butler_configured(self):
        brain = MagicMock()
        brain.think = AsyncMock(
            return_value=_make_brain_response(
                "Asking Butler.",
                [{"name": "ask_butler", "parameters": {"question": "Test"}}],
            )
        )
        brain.handle_tool_result = AsyncMock(
            return_value=_make_brain_response("I can't reach Butler.")
        )

        mgr = ConversationManager(_make_config(), brain=brain)

        await mgr.handle_transcription("ask Butler")

        result_arg = brain.handle_tool_result.call_args[0][1]
        assert "not configured" in result_arg.lower()


class TestSpeech:
    @pytest.mark.asyncio
    async def test_tts_preferred_over_vector_say(self):
        brain = MagicMock()
        brain.think = AsyncMock(return_value=_make_brain_response("Hello!"))

        tts = MagicMock()
        tts.speak = AsyncMock()
        vector = MagicMock()
        vector.say = AsyncMock()

        mgr = ConversationManager(
            _make_config(), brain=brain, tts=tts, vector=vector
        )

        await mgr.handle_transcription("hello")

        tts.speak.assert_called_once_with("Hello!")
        vector.say.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_to_vector_say_on_tts_not_implemented(self):
        brain = MagicMock()
        brain.think = AsyncMock(return_value=_make_brain_response("Hi!"))

        tts = MagicMock()
        tts.speak = AsyncMock(side_effect=NotImplementedError)
        vector = MagicMock()
        vector.say = AsyncMock()

        mgr = ConversationManager(
            _make_config(), brain=brain, tts=tts, vector=vector
        )

        await mgr.handle_transcription("hello")

        vector.say.assert_called_once_with("Hi!")

    @pytest.mark.asyncio
    async def test_fallback_to_vector_say_on_tts_error(self):
        brain = MagicMock()
        brain.think = AsyncMock(return_value=_make_brain_response("Oops!"))

        tts = MagicMock()
        tts.speak = AsyncMock(side_effect=RuntimeError("TTS down"))
        vector = MagicMock()
        vector.say = AsyncMock()

        mgr = ConversationManager(
            _make_config(), brain=brain, tts=tts, vector=vector
        )

        await mgr.handle_transcription("hello")

        vector.say.assert_called_once_with("Oops!")

    @pytest.mark.asyncio
    async def test_no_tts_no_vector_logs_warning(self):
        brain = MagicMock()
        brain.think = AsyncMock(return_value=_make_brain_response("Nobody hears me"))

        mgr = ConversationManager(_make_config(), brain=brain)

        result = await mgr.handle_transcription("hello")
        assert result.speech == "Nobody hears me"


class TestVisionEvents:
    @pytest.mark.asyncio
    async def test_vision_event_triggers_brain(self):
        brain = MagicMock()
        brain.handle_vision_event = AsyncMock(
            return_value=_make_brain_response("Oh, motion!")
        )
        vector = MagicMock()
        vector.say = AsyncMock()

        mgr = ConversationManager(_make_config(), brain=brain, vector=vector)

        result = await mgr.handle_vision_event("motion: region=left")

        assert result is not None
        assert result.speech == "Oh, motion!"
        brain.handle_vision_event.assert_called_once_with("motion: region=left")

    @pytest.mark.asyncio
    async def test_vision_event_ignored_by_brain(self):
        brain = MagicMock()
        brain.handle_vision_event = AsyncMock(return_value=None)

        mgr = ConversationManager(_make_config(), brain=brain)

        result = await mgr.handle_vision_event("motion: region=left")

        assert result is None

    @pytest.mark.asyncio
    async def test_vision_event_strips_look_calls(self):
        """Vision events should NOT trigger look tool — prevents infinite chain."""
        brain = MagicMock()
        brain.handle_vision_event = AsyncMock(
            return_value=_make_brain_response(
                "Interesting!",
                [{"name": "look", "parameters": {}}],
            )
        )

        vision = MagicMock()
        vision.describe_scene = AsyncMock(return_value="A cat on the desk.")

        vector = MagicMock()
        vector.say = AsyncMock()

        mgr = ConversationManager(
            _make_config(), brain=brain, vector=vector, vision=vision
        )

        await mgr.handle_vision_event("objects_changed: objects=['cat']")

        # look should have been stripped — scene context is already in the prompt.
        vision.describe_scene.assert_not_called()

    @pytest.mark.asyncio
    async def test_vision_event_allows_non_look_tools(self):
        """Vision events can still trigger movement/animation tools."""
        brain = MagicMock()
        brain.handle_vision_event = AsyncMock(
            return_value=_make_brain_response(
                "Oh!",
                [{"name": "play_animation", "parameters": {"name": "surprised"}}],
            )
        )
        brain.handle_tool_result = AsyncMock(
            return_value=_make_brain_response("Wow!")
        )

        vector = MagicMock()
        vector.execute_tool = AsyncMock(return_value="Animation played")
        vector.say = AsyncMock()

        mgr = ConversationManager(
            _make_config(), brain=brain, vector=vector
        )

        await mgr.handle_vision_event("objects_changed: objects=['cat']")

        vector.execute_tool.assert_called_once_with(
            "play_animation", {"name": "surprised"}
        )


class TestEndConversation:
    def test_end_resets_state(self):
        brain = MagicMock()
        brain.reset_context = MagicMock()
        mgr = ConversationManager(_make_config(), brain=brain)
        mgr._active = True

        mgr.end_conversation()

        assert mgr.is_active is False
        brain.reset_context.assert_called_once()

    def test_end_when_not_active_noop(self):
        brain = MagicMock()
        brain.reset_context = MagicMock()
        mgr = ConversationManager(_make_config(), brain=brain)

        mgr.end_conversation()

        brain.reset_context.assert_not_called()
