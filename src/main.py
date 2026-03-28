"""Entry point — starts all async loops with auto-recovery on failure."""

import asyncio
import logging
import signal

import structlog
import yaml

from src.brain import Brain
from src.butler_client import ButlerClient
from src.conversation import ConversationManager
from src.behaviors import BehaviorEngine
from src.memory import MemoryStore
from src.stt import STTListener
from src.tts import TTSClient
from src.vector_control import VectorController
from src.vision import VisionPipeline

log = structlog.get_logger()

# Retry settings for Vector connection.
_MAX_CONNECT_RETRIES = 10
_CONNECT_RETRY_DELAY_S = 15


def load_config(path: str = "config/personality.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


async def connect_with_retry(vector: VectorController) -> None:
    """Connect to Vector with retries — handles reboots and docking."""
    for attempt in range(1, _MAX_CONNECT_RETRIES + 1):
        try:
            await vector.connect()
            return
        except Exception as e:
            if attempt == _MAX_CONNECT_RETRIES:
                raise
            log.warning(
                "vector.connect_retry",
                attempt=attempt,
                max=_MAX_CONNECT_RETRIES,
                error=str(e)[:80],
            )
            await asyncio.sleep(_CONNECT_RETRY_DELAY_S)


async def run(config: dict) -> None:
    """Initialize all components and run the main loop."""
    log.info("initializing components")

    brain = Brain(config)
    butler = ButlerClient(config)
    vector = VectorController(config)
    tts = TTSClient(config, vector=vector)
    stt = STTListener(config, tts=tts)
    vision = VisionPipeline(config, vector=vector)

    # Persistent memory (optional — gracefully degrades if Postgres unavailable).
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
    behaviors = BehaviorEngine(config, vector=vector, conversation=conversation)

    # Wire up cross-references
    brain.set_vision(vision)

    await connect_with_retry(vector)

    shutdown = asyncio.Event()

    def handle_signal() -> None:
        log.info("shutdown signal received")
        shutdown.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    log.info("starting main loop")

    tasks = [
        asyncio.create_task(stt.listen(conversation), name="stt"),
        asyncio.create_task(vision.run(shutdown), name="vision"),
        asyncio.create_task(
            conversation.run_vision_event_loop(shutdown), name="vision_events"
        ),
        asyncio.create_task(behaviors.run(shutdown), name="behaviors"),
    ]

    # Monitor tasks — restart any that die unexpectedly.
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

                # Attempt to reconnect if it was a Vector connection issue.
                if "Vector" in str(exc) or "grpc" in str(exc).lower():
                    log.info("attempting_reconnect", task=name)
                    try:
                        await vector.disconnect()
                    except Exception:
                        pass
                    try:
                        await connect_with_retry(vector)
                        if name == "vision":
                            await vector.start_camera_feed()
                    except Exception as e:
                        log.error("reconnect_failed", error=str(e)[:80])
                        shutdown.set()
                        break

                # Restart the crashed task.
                log.info("task_restarting", task=name)
                if name == "stt":
                    new_task = asyncio.create_task(stt.listen(conversation), name="stt")
                elif name == "vision":
                    new_task = asyncio.create_task(vision.run(shutdown), name="vision")
                elif name == "vision_events":
                    new_task = asyncio.create_task(
                        conversation.run_vision_event_loop(shutdown), name="vision_events"
                    )
                elif name == "behaviors":
                    new_task = asyncio.create_task(behaviors.run(shutdown), name="behaviors")
                else:
                    continue

                tasks = [t for t in tasks if t is not task] + [new_task]

    log.info("shutting down")

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    await memory.close()
    log.info("shutdown complete")


def main() -> None:
    """Configure logging and start the async runtime."""
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )
    config = load_config()
    log.info("vector-llm starting", models=config.get("models"))
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
