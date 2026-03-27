"""Entry point — starts all async loops and initializes components."""

import asyncio
import logging
import signal

import structlog
import yaml

from src.brain import Brain
from src.butler_client import ButlerClient
from src.conversation import ConversationManager
from src.behaviors import BehaviorEngine
from src.stt import STTListener
from src.tts import TTSClient
from src.vector_control import VectorController
from src.vision import VisionPipeline

log = structlog.get_logger()


def load_config(path: str = "config/personality.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


async def run(config: dict) -> None:
    """Initialize all components and run the main loop."""
    log.info("initializing components")

    tts = TTSClient(config)
    stt = STTListener(config)
    brain = Brain(config)
    butler = ButlerClient(config)
    vector = VectorController(config)
    vision = VisionPipeline(config, vector=vector)
    conversation = ConversationManager(config, brain=brain, tts=tts, butler=butler)
    behaviors = BehaviorEngine(config, vector=vector)

    # Wire up cross-references
    brain.set_vision(vision)

    await vector.connect()

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
        asyncio.create_task(behaviors.run(shutdown), name="behaviors"),
    ]

    await shutdown.wait()
    log.info("shutting down")

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

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
