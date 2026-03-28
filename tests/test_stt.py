"""Tests for the STT pipeline — segment assembly and configuration."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.stt import (
    MAX_SPEECH_MS,
    MIN_SPEECH_MS,
    SAMPLE_RATE,
    SILENCE_TIMEOUT_MS,
    STTListener,
    VAD_CHUNK_SAMPLES,
    assemble_speech_segments,
)


def _make_chunk(duration_ms: int = 30, value: float = 0.0) -> np.ndarray:
    """Create a dummy audio chunk."""
    n_samples = int(SAMPLE_RATE * duration_ms / 1000)
    return np.full(n_samples, value, dtype=np.float32)


def _always_speech(chunk: np.ndarray) -> bool:
    return True


def _never_speech(chunk: np.ndarray) -> bool:
    return False


# ---------------------------------------------------------------------------
# assemble_speech_segments
# ---------------------------------------------------------------------------


class TestAssembleSpeechSegments:
    def test_silence_returns_none(self):
        state: dict = {}
        chunk = _make_chunk()
        result = assemble_speech_segments(chunk, _never_speech, state=state)
        assert result is None

    def test_speech_starts_segment(self):
        state: dict = {}
        chunk = _make_chunk()
        result = assemble_speech_segments(chunk, _always_speech, state=state)
        # First speech chunk starts a segment but doesn't complete it.
        assert result is None
        assert state["in_segment"] is True

    def test_speech_then_silence_completes_segment(self):
        state: dict = {}

        # Feed speech chunks to build up a segment above MIN_SPEECH_MS.
        n_speech_chunks = int(MIN_SPEECH_MS / 30) + 5
        for _ in range(n_speech_chunks):
            assemble_speech_segments(_make_chunk(), _always_speech, state=state)

        # Feed silence chunks until timeout triggers.
        n_silence_chunks = int(SILENCE_TIMEOUT_MS / 30) + 5
        result = None
        for _ in range(n_silence_chunks):
            result = assemble_speech_segments(_make_chunk(), _never_speech, state=state)
            if result is not None:
                break

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_too_short_segment_discarded(self):
        """Segments where total audio (speech + silence) is below MIN_SPEECH_MS
        are discarded."""
        state: dict = {}

        # Use very small chunks (1ms) so total audio stays short.
        tiny_chunk = np.zeros(int(SAMPLE_RATE * 0.001), dtype=np.float32)

        # One speech chunk (1ms).
        assemble_speech_segments(
            tiny_chunk, _always_speech, state=state, chunk_ms=1
        )

        # Feed silence to trigger end — total will be ~1ms + ~1ms * n < MIN_SPEECH_MS
        # At chunk_ms=1, need SILENCE_TIMEOUT_MS=800 silence chunks.
        # Total audio: 1ms + 800ms ≈ 801ms — this exceeds MIN_SPEECH_MS.
        # So instead, set chunk_ms high enough that few silence chunks trigger
        # timeout while keeping total audio under MIN_SPEECH_MS.
        # With chunk_ms=200: 1 speech(200ms) + 4 silence(800ms) = 5 * 200ms = 1000ms
        # That's over MIN_SPEECH_MS. The min-length check is correct —
        # it protects against very short *audio*, not short *speech*.
        # Let's just verify the function returns a segment for valid audio.
        # This is actually correct behavior — just test that min filter works
        # by using 0-sample chunks.
        state2: dict = {}
        zero_chunk = np.zeros(0, dtype=np.float32)
        assemble_speech_segments(zero_chunk, _always_speech, state=state2, chunk_ms=30)

        n_silence = int(SILENCE_TIMEOUT_MS / 30) + 5
        result = None
        for _ in range(n_silence):
            result = assemble_speech_segments(
                zero_chunk, _never_speech, state=state2, chunk_ms=30
            )
            if result is not None:
                break

        # 0-sample chunks → 0ms audio → below MIN_SPEECH_MS → discarded.
        assert result is None

    def test_max_duration_forces_end(self):
        state: dict = {}

        # Feed many speech chunks to exceed MAX_SPEECH_MS.
        n_chunks = int(MAX_SPEECH_MS / 30) + 10
        result = None
        for i in range(n_chunks):
            result = assemble_speech_segments(
                _make_chunk(30), _always_speech, state=state
            )
            if result is not None:
                break

        assert result is not None
        assert state["in_segment"] is False

    def test_state_resets_after_segment(self):
        state: dict = {}

        # Build and complete one segment.
        n_speech = int(MIN_SPEECH_MS / 30) + 5
        for _ in range(n_speech):
            assemble_speech_segments(_make_chunk(), _always_speech, state=state)

        n_silence = int(SILENCE_TIMEOUT_MS / 30) + 5
        for _ in range(n_silence):
            result = assemble_speech_segments(_make_chunk(), _never_speech, state=state)
            if result is not None:
                break

        # State should be reset.
        assert state["in_segment"] is False
        assert state["buffers"] == []

    def test_continuous_silence_no_accumulation(self):
        state: dict = {}

        # Feed lots of silence — nothing should accumulate.
        for _ in range(100):
            result = assemble_speech_segments(_make_chunk(), _never_speech, state=state)
            assert result is None

        assert "buffers" not in state or state["buffers"] == []

    def test_intermittent_speech_stays_in_segment(self):
        """Speech, brief silence, then speech again — should not end segment."""
        state: dict = {}

        # Speech.
        for _ in range(5):
            assemble_speech_segments(_make_chunk(), _always_speech, state=state)

        # Brief silence (less than timeout).
        for _ in range(3):  # 90ms < 800ms timeout
            assemble_speech_segments(_make_chunk(), _never_speech, state=state)

        # More speech.
        assemble_speech_segments(_make_chunk(), _always_speech, state=state)

        assert state["in_segment"] is True


# ---------------------------------------------------------------------------
# STTListener init
# ---------------------------------------------------------------------------


class TestSTTListener:
    def test_init_from_config(self):
        config = {
            "models": {"stt": "tiny"},
            "thresholds": {"vad_sensitivity": 0.5},
        }
        listener = STTListener(config)
        assert listener.model_size == "tiny"
        assert listener.vad_sensitivity == 0.5

    def test_init_custom_device(self):
        config = {
            "models": {"stt": "tiny"},
            "thresholds": {"vad_sensitivity": 0.5},
            "audio": {"device_index": 3},
        }
        listener = STTListener(config)
        assert listener._device_index == 3

    def test_init_default_device(self):
        config = {
            "models": {"stt": "base"},
            "thresholds": {"vad_sensitivity": 0.7},
        }
        listener = STTListener(config)
        assert listener._device_index is None
