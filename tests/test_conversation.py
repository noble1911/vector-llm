"""Tests for the conversation manager — direction detection and orchestration."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.brain import BrainResponse
from src.conversation import (
    ConversationManager,
    _DEFAULT_FOLLOWUP_WINDOW_S,
    is_directed_at_vector,
)


# ---------------------------------------------------------------------------
# is_directed_at_vector (pure function)
# ---------------------------------------------------------------------------


class TestIsDirectedAtVector:
    def test_name_mention(self):
        assert is_directed_at_vector(
            "Hey Vector, what's up?",
            last_spoke_at=0,
            now=100.0,
        )

    def test_name_mention_case_insensitive(self):
        assert is_directed_at_vector(
            "VECTOR do something",
            last_spoke_at=0,
            now=100.0,
        )

    def test_hey_robot(self):
        assert is_directed_at_vector(
            "hey robot come here",
            last_spoke_at=0,
            now=100.0,
        )

    def test_little_guy(self):
        assert is_directed_at_vector(
            "little guy, turn around",
            last_spoke_at=0,
            now=100.0,
        )

    def test_hey_buddy(self):
        assert is_directed_at_vector(
            "hey buddy what do you see",
            last_spoke_at=0,
            now=100.0,
        )

    def test_no_trigger_background_speech(self):
        assert not is_directed_at_vector(
            "I was talking to Sarah about the weather",
            last_spoke_at=0,
            now=100.0,
        )

    def test_active_conversation_assumes_directed(self):
        assert is_directed_at_vector(
            "what about the kitchen?",
            last_spoke_at=0,
            now=100.0,
            is_active=True,
        )

    def test_followup_window_after_speech(self):
        assert is_directed_at_vector(
            "thanks for that",
            last_spoke_at=95.0,
            now=100.0,  # 5s after Vector spoke, within 10s window
        )

    def test_outside_followup_window(self):
        assert not is_directed_at_vector(
            "thanks for that",
            last_spoke_at=85.0,
            now=100.0,  # 15s after Vector spoke, outside 10s window
        )

    def test_custom_followup_window(self):
        assert is_directed_at_vector(
            "okay",
            last_spoke_at=95.0,
            now=100.0,
            followup_window_s=3.0,
        ) is False  # 5s > 3s custom window

    def test_never_spoke_no_followup(self):
        assert not is_directed_at_vector(
            "hello there",
            last_spoke_at=0,
            now=100.0,
        )

    def test_vector_in_middle_of_sentence(self):
        assert is_directed_at_vector(
            "can you see that vector",
            last_spoke_at=0,
            now=100.0,
        )


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
    async def test_directed_speech_activates_conversation(self):
        brain = MagicMock()
        brain.think = AsyncMock(return_value=_make_brain_response("Hello!"))
        mgr = ConversationManager(_make_config(), brain=brain)

        result = await mgr.handle_transcription("Hey Vector, what's up?")

        assert mgr.is_active is True
        assert result is not None
        assert result.speech == "Hello!"
        brain.think.assert_called_once_with("Hey Vector, what's up?")

    @pytest.mark.asyncio
    async def test_background_speech_ignored(self):
        brain = MagicMock()
        brain.think = AsyncMock()
        mgr = ConversationManager(_make_config(), brain=brain)

        result = await mgr.handle_transcription("I was talking about the weather")

        assert result is None
        assert mgr.is_active is False
        brain.think.assert_not_called()

    @pytest.mark.asyncio
    async def test_active_conversation_continues(self):
        brain = MagicMock()
        brain.think = AsyncMock(return_value=_make_brain_response("Sure!"))
        mgr = ConversationManager(_make_config(), brain=brain)
        mgr._active = True

        result = await mgr.handle_transcription("what about that?")

        assert result is not None
        assert result.speech == "Sure!"

    @pytest.mark.asyncio
    async def test_followup_after_speaking(self):
        brain = MagicMock()
        brain.think = AsyncMock(return_value=_make_brain_response("You're welcome!"))
        mgr = ConversationManager(_make_config(), brain=brain)
        mgr._last_spoke_at = time.monotonic() - 3.0  # Spoke 3s ago

        result = await mgr.handle_transcription("thanks")

        assert result is not None
        assert result.speech == "You're welcome!"

    @pytest.mark.asyncio
    async def test_speech_output_updates_last_spoke(self):
        brain = MagicMock()
        brain.think = AsyncMock(return_value=_make_brain_response("Hi there!"))

        # Use a mock vector for fallback TTS
        vector = MagicMock()
        vector.say = AsyncMock()
        mgr = ConversationManager(_make_config(), brain=brain, vector=vector)

        before = mgr.last_spoke_at
        await mgr.handle_transcription("Hey Vector")
        assert mgr.last_spoke_at > before

    @pytest.mark.asyncio
    async def test_empty_speech_no_tts_call(self):
        brain = MagicMock()
        brain.think = AsyncMock(return_value=_make_brain_response(""))
        vector = MagicMock()
        vector.say = AsyncMock()
        mgr = ConversationManager(_make_config(), brain=brain, vector=vector)

        await mgr.handle_transcription("Hey Vector")
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

        await mgr.handle_transcription("Hey Vector")

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

        await mgr.handle_transcription("Vector, move forward")

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

        await mgr.handle_transcription("Vector, what do you see?")

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

        await mgr.handle_transcription("Vector, look around")

        brain.handle_tool_result.assert_called_once_with("look", "[no vision available]")

    @pytest.mark.asyncio
    async def test_max_tool_rounds_prevents_infinite_loop(self):
        """Tool calls that keep producing more tool calls should stop."""
        brain = MagicMock()

        # Brain always returns a tool call.
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

        await mgr.handle_transcription("Vector, keep moving")

        # Should cap at 5 rounds (max_rounds in _execute_tools).
        assert vector.execute_tool.call_count <= 5

    @pytest.mark.asyncio
    async def test_no_tool_calls_no_dispatch(self):
        brain = MagicMock()
        brain.think = AsyncMock(return_value=_make_brain_response("Just talking."))

        vector = MagicMock()
        vector.execute_tool = AsyncMock()
        vector.say = AsyncMock()

        mgr = ConversationManager(_make_config(), brain=brain, vector=vector)

        await mgr.handle_transcription("Hey Vector, how are you?")

        vector.execute_tool.assert_not_called()


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

        await mgr.handle_transcription("Hey Vector")

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

        await mgr.handle_transcription("Hey Vector")

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

        await mgr.handle_transcription("Hey Vector")

        vector.say.assert_called_once_with("Oops!")

    @pytest.mark.asyncio
    async def test_no_tts_no_vector_logs_warning(self):
        brain = MagicMock()
        brain.think = AsyncMock(return_value=_make_brain_response("Nobody hears me"))

        mgr = ConversationManager(_make_config(), brain=brain)

        # Should not raise, just log.
        result = await mgr.handle_transcription("Hey Vector")
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
    async def test_vision_event_with_tool_calls(self):
        brain = MagicMock()
        brain.handle_vision_event = AsyncMock(
            return_value=_make_brain_response(
                "Interesting!",
                [{"name": "look", "parameters": {}}],
            )
        )
        brain.handle_tool_result = AsyncMock(
            return_value=_make_brain_response("I see a cat!")
        )

        vision = MagicMock()
        vision.describe_scene = AsyncMock(return_value="A cat on the desk.")

        vector = MagicMock()
        vector.say = AsyncMock()

        mgr = ConversationManager(
            _make_config(), brain=brain, vector=vector, vision=vision
        )

        await mgr.handle_vision_event("objects_changed: objects=['cat']")

        vision.describe_scene.assert_called_once()


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
