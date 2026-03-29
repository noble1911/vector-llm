"""Entry point — state machine architecture with auto-recovery.

States:
  EXPLORING  — firmware controls Vector, we passively observe
  CONVERSING — we hold control, brain responds to speech
  LEARNING   — brief control to look at something interesting
"""

import asyncio
import logging
import signal

import structlog
import yaml

from src.brain import Brain
from src.butler_client import ButlerClient
from src.conversation import ConversationManager
from src.memory import MemoryStore
from src.state_machine import State, StateMachine
from src.stt import STTListener
from src.tts import TTSClient
from src.vector_control import VectorController
from src.vision import VisionPipeline

log = structlog.get_logger()

_MAX_CONNECT_RETRIES = 10
_CONNECT_RETRY_DELAY_S = 15


def load_config(path: str = "config/personality.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


async def connect_with_retry(vector: VectorController) -> None:
    """Connect to Vector with retries."""
    for attempt in range(1, _MAX_CONNECT_RETRIES + 1):
        try:
            await vector.connect()
            return
        except Exception as e:
            if attempt == _MAX_CONNECT_RETRIES:
                raise
            log.warning("vector.connect_retry", attempt=attempt, error=str(e)[:80])
            await asyncio.sleep(_CONNECT_RETRY_DELAY_S)


async def run(config: dict) -> None:
    """Initialize components and run the state machine."""
    log.info("initializing components")

    brain = Brain(config)
    butler = ButlerClient(config)
    vector = VectorController(config)
    tts = TTSClient(config, vector=vector)
    stt = STTListener(config, tts=tts)
    vision = VisionPipeline(config, vector=vector)

    memory = MemoryStore(config)
    await memory.connect()

    conversation = ConversationManager(
        config,
        brain=brain,
        tts=tts,
        vector=vector,
        vision=vision,
        butler=butler,
        memory=memory,
    )

    brain.set_vision(vision)

    await connect_with_retry(vector)

    sm = StateMachine(
        vector=vector,
        conversation=conversation,
        vision=vision,
    )

    shutdown = asyncio.Event()

    def handle_signal() -> None:
        log.info("shutdown signal received")
        shutdown.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    log.info("starting main loop")

    # Robot names that trigger conversation (case-insensitive).
    _wake_names = {"vector", "chili", "chilli"}

    # STT callback — only enters conversation if name is spoken (or already active).
    async def on_transcription(text: str, timestamp: float) -> None:
        text_lower = text.lower()

        if sm.state != State.CONVERSING:
            # Check if the robot's name was spoken.
            if not any(name in text_lower for name in _wake_names):
                return  # Ignore — not addressed to us.
            await sm.transition_to(State.CONVERSING)

        sm.touch_interaction()

        # Show thinking animation while LLM processes (non-blocking).
        asyncio.create_task(sm._play_thinking())

        await conversation.handle_transcription(text, timestamp=timestamp)

    # Vision event callback — triggers LEARNING on significant changes.
    _learning_in_progress = False

    async def on_scene_change(event_summary: str) -> None:
        nonlocal _learning_in_progress
        if sm.state == State.CONVERSING or not sm.can_learn() or _learning_in_progress:
            return
        _learning_in_progress = True
        try:
            await sm.transition_to(State.LEARNING)
            await conversation.handle_learning(event_summary)
            await sm.transition_to(State.EXPLORING)
        finally:
            _learning_in_progress = False

    # Vision event drain loop.
    async def vision_event_loop() -> None:
        if not vision:
            return
        while not shutdown.is_set():
            try:
                try:
                    event = await asyncio.wait_for(
                        vision.events.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                summary = (
                    f"{event.event_type}: "
                    f"{', '.join(f'{k}={v}' for k, v in event.data.items())}"
                )
                asyncio.create_task(on_scene_change(summary))

            except Exception:
                log.exception("vision_event_loop error")

    tasks = [
        asyncio.create_task(
            stt.listen_with_callback(on_transcription), name="stt"
        ),
        asyncio.create_task(vision.run(shutdown), name="vision"),
        asyncio.create_task(vision_event_loop(), name="vision_events"),
        asyncio.create_task(sm.run(shutdown), name="state_machine"),
    ]

    # Monitor tasks — restart on crash.
    shutdown_task = asyncio.create_task(shutdown.wait(), name="shutdown_waiter")

    while not shutdown.is_set():
        done, _ = await asyncio.wait(
            tasks + [shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if shutdown.is_set():
            break

        for task in done:
            if task is shutdown_task:
                continue
            name = task.get_name()
            exc = task.exception() if not task.cancelled() else None
            if exc:
                log.error("task_crashed", task=name, error=str(exc)[:100])

                if "Vector" in str(exc) or "grpc" in str(exc).lower():
                    log.info("attempting_reconnect", task=name)
                    try:
                        await vector.disconnect()
                    except Exception:
                        pass
                    try:
                        await connect_with_retry(vector)
                    except Exception as e:
                        log.error("reconnect_failed", error=str(e)[:80])
                        shutdown.set()
                        break

                log.info("task_restarting", task=name)
                if name == "stt":
                    new_task = asyncio.create_task(
                        stt.listen_with_callback(on_transcription), name="stt"
                    )
                elif name == "vision":
                    new_task = asyncio.create_task(vision.run(shutdown), name="vision")
                elif name == "vision_events":
                    new_task = asyncio.create_task(vision_event_loop(), name="vision_events")
                elif name == "state_machine":
                    new_task = asyncio.create_task(sm.run(shutdown), name="state_machine")
                else:
                    continue

                tasks = [t for t in tasks if t is not task] + [new_task]

    log.info("shutting down")
    shutdown_task.cancel()
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await memory.close()
    log.info("shutdown complete")


def main() -> None:
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )
    config = load_config()
    log.info("vector-llm starting", models=config.get("models"))
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
