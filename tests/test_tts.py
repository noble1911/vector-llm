"""Tests for TTS client — WAV resampling, synthesis, and playback."""

from __future__ import annotations

import struct
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.tts import (
    TTSClient,
    TTSError,
    _decode_pcm,
    _encode_wav,
    _parse_wav_header,
    _resample_linear,
    resample_wav,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav(
    samples: list[int],
    sample_rate: int = 24000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Build a minimal WAV file from samples."""
    if sample_width == 2:
        pcm = struct.pack(f"<{len(samples)}h", *samples)
    elif sample_width == 1:
        pcm = bytes(s + 128 for s in samples)  # unsigned 8-bit
    else:
        raise ValueError(f"Unsupported sample_width: {sample_width}")

    data_size = len(pcm)
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        sample_width * 8,
        b"data",
        data_size,
    )
    return header + pcm


def _make_config(**overrides) -> dict:
    config = {
        "endpoints": {"kokoro_tts": "http://localhost:8880"},
        "personality": {"voice": "af_heart"},
    }
    config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# WAV parsing
# ---------------------------------------------------------------------------


class TestParseWavHeader:
    def test_valid_wav(self):
        wav = _make_wav([100, -200, 300], sample_rate=16000)
        header = _parse_wav_header(wav)
        assert header["sample_rate"] == 16000
        assert header["channels"] == 1
        assert header["sample_width"] == 2
        assert len(header["pcm_data"]) == 6  # 3 samples * 2 bytes

    def test_stereo_wav(self):
        wav = _make_wav([100, -100, 200, -200], sample_rate=44100, channels=2)
        header = _parse_wav_header(wav)
        assert header["channels"] == 2
        assert header["sample_rate"] == 44100

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            _parse_wav_header(b"RIFF")

    def test_not_riff_raises(self):
        with pytest.raises(ValueError, match="Not a RIFF"):
            _parse_wav_header(b"\x00" * 44)

    def test_not_wave_raises(self):
        data = b"RIFF" + b"\x00" * 4 + b"XXXX" + b"\x00" * 32
        with pytest.raises(ValueError, match="Not a WAVE"):
            _parse_wav_header(data)


class TestDecodePcm:
    def test_16bit(self):
        pcm = struct.pack("<3h", 100, -200, 300)
        assert _decode_pcm(pcm, 2) == [100, -200, 300]

    def test_8bit_unsigned(self):
        # 8-bit PCM is unsigned: 128=silence, 0=-128, 255=127
        pcm = bytes([128, 0, 255])
        assert _decode_pcm(pcm, 1) == [0, -128, 127]

    def test_32bit_scaled(self):
        pcm = struct.pack("<2i", 65536 * 100, 65536 * -200)
        result = _decode_pcm(pcm, 4)
        assert result == [100, -200]


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


class TestResampleLinear:
    def test_same_rate_noop(self):
        samples = [100, 200, 300]
        assert _resample_linear(samples, 16000, 16000) == samples

    def test_downsample(self):
        # 48kHz to 16kHz = 3:1 ratio. Output should be ~1/3 the length.
        samples = list(range(300))
        result = _resample_linear(samples, 48000, 16000)
        assert len(result) == 100

    def test_upsample(self):
        # 8kHz to 16kHz = 1:2 ratio. Output should be ~2x the length.
        samples = [0, 1000, 2000, 3000]
        result = _resample_linear(samples, 8000, 16000)
        assert len(result) == 8

    def test_values_interpolated(self):
        samples = [0, 1000]
        result = _resample_linear(samples, 8000, 16000)
        # Should interpolate: 0, ~500, 1000, ~...
        assert result[0] == 0
        assert 400 <= result[1] <= 600  # roughly midpoint

    def test_clamps_to_16bit(self):
        samples = [32767, 32767]
        result = _resample_linear(samples, 8000, 16000)
        assert all(-32768 <= s <= 32767 for s in result)


class TestResampleWav:
    def test_already_correct_format(self):
        wav = _make_wav([100, -200], sample_rate=16000)
        result = resample_wav(wav)
        header = _parse_wav_header(result)
        assert header["sample_rate"] == 16000
        assert header["channels"] == 1
        assert header["sample_width"] == 2

    def test_resample_24k_to_16k(self):
        samples = list(range(0, 2400, 10))  # 240 samples at 24kHz
        wav = _make_wav(samples, sample_rate=24000)
        result = resample_wav(wav)
        header = _parse_wav_header(result)
        assert header["sample_rate"] == 16000
        # ~240 * (16/24) = 160 samples
        expected_samples = len(samples) * 16000 // 24000
        actual_samples = len(header["pcm_data"]) // 2
        assert abs(actual_samples - expected_samples) <= 1

    def test_stereo_to_mono(self):
        # 4 samples: L=100 R=200, L=300 R=400
        wav = _make_wav([100, 200, 300, 400], sample_rate=16000, channels=2)
        result = resample_wav(wav)
        header = _parse_wav_header(result)
        assert header["channels"] == 1
        decoded = _decode_pcm(header["pcm_data"], 2)
        assert decoded[0] == 150  # (100+200)/2
        assert decoded[1] == 350  # (300+400)/2

    def test_invalid_wav_raises(self):
        with pytest.raises(TTSError, match="Invalid WAV"):
            resample_wav(b"not a wav file at all")


class TestEncodeWav:
    def test_roundtrip(self):
        original = [100, -200, 300, -400]
        wav = _encode_wav(original, 16000)
        header = _parse_wav_header(wav)
        decoded = _decode_pcm(header["pcm_data"], 2)
        assert decoded == original

    def test_correct_header(self):
        wav = _encode_wav([0, 0], 16000)
        header = _parse_wav_header(wav)
        assert header["sample_rate"] == 16000
        assert header["channels"] == 1
        assert header["sample_width"] == 2


# ---------------------------------------------------------------------------
# TTSClient
# ---------------------------------------------------------------------------


class TestTTSClientSynthesize:
    @pytest.mark.asyncio
    async def test_synthesize_calls_kokoro(self):
        wav = _make_wav([100, -100] * 100, sample_rate=24000)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = wav
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        tts = TTSClient(_make_config(), http_client=mock_client)
        result = await tts._synthesize("Hello world")

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/v1/audio/speech" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["input"] == "Hello world"
        assert payload["voice"] == "af_heart"
        assert payload["response_format"] == "wav"

        # Result should be resampled WAV.
        header = _parse_wav_header(result)
        assert header["sample_rate"] == 16000

    @pytest.mark.asyncio
    async def test_synthesize_connection_error(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        tts = TTSClient(_make_config(), http_client=mock_client)

        with pytest.raises(TTSError, match="Cannot connect"):
            await tts._synthesize("Hello")

    @pytest.mark.asyncio
    async def test_synthesize_http_error(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal error"
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "500", request=MagicMock(), response=mock_response
            )
        )
        mock_client.post = AsyncMock(return_value=mock_response)

        tts = TTSClient(_make_config(), http_client=mock_client)

        with pytest.raises(TTSError, match="error 500"):
            await tts._synthesize("Hello")

    @pytest.mark.asyncio
    async def test_synthesize_timeout(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        tts = TTSClient(_make_config(), http_client=mock_client)

        with pytest.raises(TTSError, match="timeout"):
            await tts._synthesize("Hello")

    @pytest.mark.asyncio
    async def test_synthesize_too_little_data(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"tiny"
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        tts = TTSClient(_make_config(), http_client=mock_client)

        with pytest.raises(TTSError, match="too little data"):
            await tts._synthesize("Hello")


class TestTTSClientSpeak:
    @pytest.mark.asyncio
    async def test_speak_empty_text_noop(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        tts = TTSClient(_make_config(), http_client=mock_client)

        await tts.speak("")
        await tts.speak("   ")

        mock_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_speak_falls_back_to_vector_say(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        vector = MagicMock()
        vector.say = AsyncMock()

        tts = TTSClient(_make_config(), http_client=mock_client, vector=vector)

        await tts.speak("Hello!")

        vector.say.assert_called_once_with("Hello!")

    @pytest.mark.asyncio
    async def test_speak_no_fallback_raises(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        tts = TTSClient(_make_config(), http_client=mock_client)

        with pytest.raises(TTSError, match="no fallback"):
            await tts.speak("Hello!")


class TestTTSClientInterrupt:
    def test_interrupt_sets_cancel_event(self):
        tts = TTSClient(_make_config())
        assert not tts._cancel_event.is_set()
        tts.interrupt()
        assert tts._cancel_event.is_set()


class TestTTSClientAvailability:
    @pytest.mark.asyncio
    async def test_is_available_true(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_response)

        tts = TTSClient(_make_config(), http_client=mock_client)
        assert await tts.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_false_on_error(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        tts = TTSClient(_make_config(), http_client=mock_client)
        assert await tts.is_available() is False
