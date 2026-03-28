"""Text-to-speech via Kokoro TTS — synthesizes speech and plays on Vector.

Pipeline:
  Text → Kokoro API (WAV) → resample to 16kHz mono 16-bit → Vector speaker
"""

from __future__ import annotations

import asyncio
import io
import struct
import tempfile
from typing import TYPE_CHECKING, Any

import httpx
import structlog

if TYPE_CHECKING:
    from src.vector_control import VectorController

log = structlog.get_logger()

# Kokoro TTS API path (OpenAI-compatible).
_TTS_PATH = "/v1/audio/speech"

# Vector speaker requirements.
_TARGET_SAMPLE_RATE = 16000
_TARGET_CHANNELS = 1
_TARGET_SAMPLE_WIDTH = 2  # 16-bit

# Default TTS timeout.
_DEFAULT_TIMEOUT_S = 10.0


class TTSError(Exception):
    """Raised when TTS synthesis or playback fails."""


def resample_wav(
    wav_data: bytes,
    *,
    target_rate: int = _TARGET_SAMPLE_RATE,
) -> bytes:
    """Resample WAV data to target format for Vector's speaker.

    Converts to 16kHz mono 16-bit PCM WAV. Uses linear interpolation
    for resampling — good enough for speech audio.

    Args:
        wav_data: Input WAV file bytes.
        target_rate: Target sample rate in Hz.

    Returns:
        Resampled WAV file bytes.

    Raises:
        TTSError: If the WAV data is invalid.
    """
    try:
        # Parse WAV header manually to avoid external dependency.
        header = _parse_wav_header(wav_data)
    except Exception as e:
        raise TTSError(f"Invalid WAV data: {e}") from e

    src_rate = header["sample_rate"]
    src_channels = header["channels"]
    src_sample_width = header["sample_width"]
    pcm_data = header["pcm_data"]

    # Decode PCM samples.
    samples = _decode_pcm(pcm_data, src_sample_width)

    # Convert stereo to mono by averaging channels.
    if src_channels == 2:
        mono = []
        for i in range(0, len(samples), 2):
            if i + 1 < len(samples):
                mono.append((samples[i] + samples[i + 1]) // 2)
            else:
                mono.append(samples[i])
        samples = mono

    # Resample if needed.
    if src_rate != target_rate:
        samples = _resample_linear(samples, src_rate, target_rate)

    # Encode as 16-bit PCM WAV.
    return _encode_wav(samples, target_rate)


def _parse_wav_header(data: bytes) -> dict:
    """Parse a WAV file header and extract PCM data.

    Returns:
        Dict with sample_rate, channels, sample_width, pcm_data.
    """
    if len(data) < 44:
        raise ValueError("WAV data too short")

    # RIFF header.
    riff = data[:4]
    if riff != b"RIFF":
        raise ValueError(f"Not a RIFF file: {riff!r}")

    wave = data[8:12]
    if wave != b"WAVE":
        raise ValueError(f"Not a WAVE file: {wave!r}")

    # Find 'fmt ' and 'data' chunks.
    pos = 12
    fmt_info = None
    pcm_data = None

    while pos < len(data) - 8:
        chunk_id = data[pos : pos + 4]
        chunk_size = struct.unpack_from("<I", data, pos + 4)[0]
        chunk_start = pos + 8

        if chunk_id == b"fmt ":
            if chunk_size < 16:
                raise ValueError("fmt chunk too small")
            audio_format = struct.unpack_from("<H", data, chunk_start)[0]
            if audio_format != 1:  # PCM only
                raise ValueError(f"Unsupported audio format: {audio_format}")
            channels = struct.unpack_from("<H", data, chunk_start + 2)[0]
            sample_rate = struct.unpack_from("<I", data, chunk_start + 4)[0]
            bits_per_sample = struct.unpack_from("<H", data, chunk_start + 14)[0]
            fmt_info = {
                "channels": channels,
                "sample_rate": sample_rate,
                "sample_width": bits_per_sample // 8,
            }

        elif chunk_id == b"data":
            pcm_data = data[chunk_start : chunk_start + chunk_size]

        pos = chunk_start + chunk_size
        # Chunks are word-aligned.
        if pos % 2 == 1:
            pos += 1

    if fmt_info is None:
        raise ValueError("No fmt chunk found")
    if pcm_data is None:
        raise ValueError("No data chunk found")

    fmt_info["pcm_data"] = pcm_data
    return fmt_info


def _decode_pcm(data: bytes, sample_width: int) -> list[int]:
    """Decode raw PCM bytes into a list of integer samples."""
    if sample_width == 2:
        count = len(data) // 2
        return list(struct.unpack(f"<{count}h", data[: count * 2]))
    elif sample_width == 1:
        # 8-bit PCM is unsigned, convert to signed range.
        return [b - 128 for b in data]
    elif sample_width == 4:
        # 32-bit, scale down to 16-bit range.
        count = len(data) // 4
        samples_32 = struct.unpack(f"<{count}i", data[: count * 4])
        return [s >> 16 for s in samples_32]
    else:
        raise TTSError(f"Unsupported sample width: {sample_width}")


def _resample_linear(
    samples: list[int], src_rate: int, dst_rate: int
) -> list[int]:
    """Resample using linear interpolation."""
    if src_rate == dst_rate or len(samples) < 2:
        return samples

    ratio = src_rate / dst_rate
    out_len = int(len(samples) / ratio)
    result = []

    for i in range(out_len):
        src_pos = i * ratio
        idx = int(src_pos)
        frac = src_pos - idx

        if idx + 1 < len(samples):
            val = samples[idx] * (1 - frac) + samples[idx + 1] * frac
        else:
            val = samples[idx] if idx < len(samples) else 0

        # Clamp to 16-bit range.
        result.append(max(-32768, min(32767, int(val))))

    return result


def _encode_wav(samples: list[int], sample_rate: int) -> bytes:
    """Encode samples as a 16-bit mono PCM WAV file."""
    pcm = struct.pack(f"<{len(samples)}h", *samples)
    data_size = len(pcm)
    channels = 1
    sample_width = 2
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,  # fmt chunk size
        1,  # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        sample_width * 8,  # bits per sample
        b"data",
        data_size,
    )

    return header + pcm


class TTSClient:
    """Synthesizes speech via Kokoro TTS and plays on Vector's speaker.

    Falls back to Vector's built-in say_text() if Kokoro is unavailable.
    """

    def __init__(
        self,
        config: dict,
        *,
        vector: VectorController | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.endpoint = config["endpoints"]["kokoro_tts"]
        self.voice = config["personality"]["voice"]
        self._vector = vector
        self._client = http_client or httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT_S)
        self._speaking = False
        self._cancel_event = asyncio.Event()

    @property
    def is_speaking(self) -> bool:
        """Whether audio is currently playing."""
        return self._speaking

    async def speak(self, text: str) -> None:
        """Synthesize text and play on Vector's speaker.

        Args:
            text: Text to speak.

        Raises:
            TTSError: If synthesis fails and no fallback is available.
        """
        if not text.strip():
            return

        log.info("tts.speak", text=text[:60])
        self._cancel_event.clear()

        try:
            wav_data = await self._synthesize(text)

            if self._cancel_event.is_set():
                log.info("tts.cancelled_before_playback")
                return

            await self._play_on_vector(wav_data)

        except Exception as e:
            log.warning("tts.kokoro_failed", error=str(e))
            # Fallback to Vector's built-in TTS.
            if self._vector:
                await self._vector.say(text)
            else:
                raise TTSError(f"TTS failed and no fallback: {e}") from e
        finally:
            self._speaking = False

    def interrupt(self) -> None:
        """Signal to stop current speech playback."""
        self._cancel_event.set()
        log.info("tts.interrupt")

    async def _synthesize(self, text: str) -> bytes:
        """Call Kokoro TTS API to generate WAV audio.

        Args:
            text: Text to synthesize.

        Returns:
            Resampled WAV bytes ready for Vector's speaker.

        Raises:
            TTSError: If the API call fails.
        """
        url = f"{self.endpoint}{_TTS_PATH}"
        payload = {
            "model": "kokoro",
            "input": text,
            "voice": self.voice,
            "response_format": "wav",
            "speed": 1.0,
        }

        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise TTSError(f"Cannot connect to Kokoro TTS at {self.endpoint}: {e}") from e
        except httpx.HTTPStatusError as e:
            raise TTSError(
                f"Kokoro TTS error {e.response.status_code}: {e.response.text[:200]}"
            ) from e
        except httpx.TimeoutException as e:
            raise TTSError(f"Kokoro TTS timeout: {e}") from e

        wav_data = response.content

        if len(wav_data) < 44:
            raise TTSError(f"Kokoro returned too little data ({len(wav_data)} bytes)")

        # Resample to Vector's required format.
        return resample_wav(wav_data)

    async def _play_on_vector(self, wav_data: bytes) -> None:
        """Stream WAV audio to Vector's speaker.

        Args:
            wav_data: 16kHz mono 16-bit WAV bytes.

        Raises:
            TTSError: If playback fails.
        """
        if self._vector is None:
            raise TTSError("No Vector controller for audio playback")

        self._speaking = True

        # Vector SDK needs a file path, so write to a temp file.
        def _play():
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                f.write(wav_data)
                f.flush()
                self._vector.robot.audio.stream_wav_file(
                    f.name, volume=75
                )

        try:
            await asyncio.to_thread(_play)
            log.info("tts.played", size_bytes=len(wav_data))
        except Exception as e:
            raise TTSError(f"Failed to play audio on Vector: {e}") from e

    async def is_available(self) -> bool:
        """Check if Kokoro TTS is reachable.

        Returns:
            True if the TTS endpoint responds.
        """
        try:
            response = await self._client.get(f"{self.endpoint}/v1/audio/voices")
            return response.status_code == 200
        except Exception:
            return False
