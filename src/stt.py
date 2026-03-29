"""Always-on speech-to-text — listens via external mic with VAD filtering.

Architecture:
  Mic (sounddevice) → Audio chunks → VAD (silero) → Speech segments → Whisper → Text
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import structlog

if TYPE_CHECKING:
    from src.conversation import ConversationManager

log = structlog.get_logger()

# Audio settings — match Whisper's expected input.
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"

# VAD settings — silero-vad v5+ requires chunks >= 512 samples at 16kHz.
VAD_CHUNK_MS = 32
VAD_CHUNK_SAMPLES = 512

# Speech segment assembly.
MIN_SPEECH_MS = 1500  # Ignore segments shorter than this (Whisper hallucinates on short clips)
MAX_SPEECH_MS = 30000  # Force-stop segments longer than this
SILENCE_TIMEOUT_MS = 800  # End segment after this much silence
PRE_ROLL_CHUNKS = 20  # Keep ~640ms of audio before VAD triggers to capture word onsets


def assemble_speech_segments(
    audio_chunk: np.ndarray,
    vad_fn: Callable[[np.ndarray], bool],
    *,
    state: dict,
    chunk_ms: int = VAD_CHUNK_MS,
) -> np.ndarray | None:
    """Accumulate VAD-triggered audio into complete speech segments.

    Call this repeatedly with incoming audio chunks. When a complete speech
    segment is detected (speech followed by silence timeout), returns the
    assembled audio. Otherwise returns None.

    Timing is derived from chunk count rather than wall clock, making this
    function deterministic and testable.

    Args:
        audio_chunk: Float32 audio samples at 16kHz.
        vad_fn: Function that returns True if the chunk contains speech.
        state: Mutable dict tracking segment assembly state. Must persist
            between calls. Initialize as empty dict.
        chunk_ms: Duration of each chunk in milliseconds.

    Returns:
        Complete speech segment as float32 numpy array, or None.
    """
    is_speech = vad_fn(audio_chunk)

    in_segment = state.get("in_segment", False)
    buffers = state.setdefault("buffers", [])
    silence_chunks = state.get("silence_chunks", 0)
    pre_roll = state.setdefault("pre_roll", [])

    if is_speech:
        state["silence_chunks"] = 0

        if not in_segment:
            state["in_segment"] = True
            # Prepend pre-roll audio so we don't lose the start of speech.
            state["buffers"] = list(pre_roll) + [audio_chunk]
            state["pre_roll"] = []
            return None

        buffers.append(audio_chunk)

        # Force-end very long segments.
        duration_ms = len(buffers) * chunk_ms
        if duration_ms >= MAX_SPEECH_MS:
            segment = np.concatenate(state["buffers"])
            state["in_segment"] = False
            state["buffers"] = []
            return segment

        return None

    # Silence.
    if not in_segment:
        # Maintain a rolling pre-roll buffer of recent silence/non-speech audio.
        pre_roll.append(audio_chunk)
        if len(pre_roll) > PRE_ROLL_CHUNKS:
            pre_roll.pop(0)
        return None

    buffers.append(audio_chunk)
    state["silence_chunks"] = silence_chunks + 1

    silence_ms = state["silence_chunks"] * chunk_ms
    if silence_ms >= SILENCE_TIMEOUT_MS:
        segment = np.concatenate(state["buffers"])
        state["in_segment"] = False
        state["buffers"] = []
        state["silence_chunks"] = 0

        # Check minimum length.
        duration_ms = len(segment) / SAMPLE_RATE * 1000
        if duration_ms < MIN_SPEECH_MS:
            return None

        return segment

    return None


class STTListener:
    """Captures audio from the external mic, runs VAD, and transcribes speech."""

    def __init__(self, config: dict, *, tts: Any = None) -> None:
        self.model_size = config["models"]["stt"]
        self.vad_sensitivity = config["thresholds"]["vad_sensitivity"]
        self._whisper_model: Any = None
        self._vad_model: Any = None
        self._device_index: int | None = config.get("audio", {}).get("device_index")
        self._tts = tts  # Used to suppress STT while Vector is speaking.

    async def _load_models(self) -> None:
        """Load Whisper and VAD models in a thread."""
        def _load():
            from faster_whisper import WhisperModel

            whisper = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
            )

            import torch
            vad, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            return whisper, vad

        self._whisper_model, self._vad_model = await asyncio.to_thread(_load)
        log.info("stt models loaded", whisper=self.model_size)

    def _vad_check(self, audio_chunk: np.ndarray) -> bool:
        """Run VAD on a single chunk.

        Args:
            audio_chunk: Float32 audio at 16kHz.

        Returns:
            True if speech is detected.
        """
        import torch

        tensor = torch.from_numpy(audio_chunk)
        confidence = self._vad_model(tensor, SAMPLE_RATE).item()
        return confidence >= self.vad_sensitivity

    async def _transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a speech segment using Whisper.

        Args:
            audio: Float32 audio at 16kHz.

        Returns:
            Transcribed text, stripped.
        """
        def _run():
            segments, _ = self._whisper_model.transcribe(
                audio,
                language="en",
                beam_size=1,
                vad_filter=False,  # We already ran VAD.
                condition_on_previous_text=False,  # Each segment is independent.
            )
            # Fully consume the generator to release internal buffers.
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text

        return await asyncio.to_thread(_run)

    async def _handle_segment(
        self, segment: np.ndarray, callback: Any
    ) -> None:
        """Transcribe a segment and forward via callback (runs as background task)."""
        try:
            t0 = time.perf_counter()
            text = await self._transcribe(segment)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if text:
                log.info(
                    "stt transcription",
                    text=text[:80],
                    audio_ms=int(len(segment) / SAMPLE_RATE * 1000),
                    latency_ms=int(elapsed_ms),
                )
                await callback(text, t0)
        except Exception:
            log.exception("stt transcription_error")

    async def listen_with_callback(self, callback: Any) -> None:
        """Main listening loop — capture audio, detect speech, transcribe, forward.

        Args:
            callback: Async function(text, timestamp) called for each transcription.
        """
        log.info("stt listener starting", model=self.model_size)

        await self._load_models()

        import sounddevice as sd

        audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _audio_callback(indata, frames, time_info, status):
            if status:
                log.warning("audio callback status", status=str(status))
            loop.call_soon_threadsafe(
                audio_queue.put_nowait,
                indata[:, 0].copy(),  # Mono, float32.
            )

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=VAD_CHUNK_SAMPLES,
            device=self._device_index,
            callback=_audio_callback,
        )

        vad_state: dict = {}
        _audio_debug_counter = 0

        with stream:
            log.info("stt listening", device=self._device_index if self._device_index is not None else "default")
            while True:
                chunk = await audio_queue.get()

                # Log audio level periodically to confirm mic is working.
                _audio_debug_counter += 1
                if _audio_debug_counter % 200 == 1:  # Every ~6 seconds
                    peak = float(np.max(np.abs(chunk)))
                    log.info("stt audio_level", peak=f"{peak:.4f}", chunks=_audio_debug_counter)

                # Suppress STT while Vector is speaking to avoid self-hearing.
                if self._tts and self._tts.is_speaking:
                    # Discard any in-progress segment — it's contaminated.
                    if vad_state.get("in_segment"):
                        vad_state["in_segment"] = False
                        vad_state["buffers"] = []
                        vad_state["silence_chunks"] = 0
                    continue

                try:
                    segment = assemble_speech_segments(
                        chunk, self._vad_check, state=vad_state
                    )
                except Exception:
                    log.exception("stt vad_error")
                    continue

                if segment is not None:
                    asyncio.create_task(
                        self._handle_segment(segment, callback)
                    )
