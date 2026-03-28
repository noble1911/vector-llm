"""Wire-pod integration — SDK auth, custom intents, and robot discovery.

Wire-pod runs as a native macOS app on the Mac Mini and provides:
  - SDK token generation (GUID-based auth, no Anki cloud needed)
  - Custom intent routing for Vector's on-board mic
  - Robot discovery and configuration

Note: The "Hey Vector" wake word is firmware-level and cannot be disabled.
Our system bypasses it entirely by using an external USB mic + VAD.
Wire-pod custom intents handle the case where someone *does* use the
wake word — those commands get routed to our brain too.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import structlog

log = structlog.get_logger()

_DEFAULT_TIMEOUT_S = 10.0


class WirePodError(Exception):
    """Raised when wire-pod communication fails."""


@dataclass
class RobotInfo:
    """Information about a Vector robot registered with wire-pod."""

    esn: str
    ip_address: str
    guid: str
    activated: bool


@dataclass
class WirePodConfig:
    """Wire-pod connection configuration."""

    global_guid: str
    robots: list[RobotInfo]


def parse_sdk_info(data: dict) -> WirePodConfig:
    """Parse wire-pod SDK info response into structured config.

    Args:
        data: JSON response from /api-sdk/get_sdk_info.

    Returns:
        WirePodConfig with global GUID and robot list.
    """
    robots = []
    for r in data.get("robots", []):
        robots.append(
            RobotInfo(
                esn=r.get("esn", ""),
                ip_address=r.get("ip_address", ""),
                guid=r.get("guid", ""),
                activated=r.get("activated", False),
            )
        )

    return WirePodConfig(
        global_guid=data.get("global_guid", ""),
        robots=robots,
    )


@dataclass
class CustomIntent:
    """A wire-pod custom intent definition."""

    name: str
    description: str
    utterances: list[str]
    intent: str
    exec_command: str = ""
    exec_args: str = ""


def build_intent_json(intent: CustomIntent) -> dict:
    """Convert a CustomIntent to wire-pod JSON format.

    Args:
        intent: The intent to serialize.

    Returns:
        Dict matching wire-pod's customIntents.json format.
    """
    return {
        "name": intent.name,
        "description": intent.description,
        "utterances": intent.utterances,
        "intent": intent.intent,
        "exec": intent.exec_command,
        "execargs": intent.exec_args,
    }


# Default intents to route Vector's on-board voice commands to our system.
DEFAULT_INTENTS = [
    CustomIntent(
        name="vector_llm_catchall",
        description="Route all unmatched voice commands to the LLM brain",
        utterances=[],  # Empty = catch-all for unmatched intents
        intent="intent_custom_llm",
    ),
]


class WirePodClient:
    """Client for wire-pod's HTTP API.

    Used for:
    - Discovering registered robots and their SDK tokens
    - Configuring custom intents
    - Health checks
    """

    def __init__(
        self,
        config: dict,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        wirepod_cfg = config.get("wirepod", {})
        self.endpoint = wirepod_cfg.get("endpoint", "http://localhost:8090")
        self._client = http_client or httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT_S)

    async def get_sdk_info(self) -> WirePodConfig:
        """Fetch SDK info including robot GUIDs.

        Returns:
            WirePodConfig with global GUID and robot list.

        Raises:
            WirePodError: If the request fails.
        """
        url = f"{self.endpoint}/api-sdk/get_sdk_info"

        try:
            response = await self._client.get(url)
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise WirePodError(
                f"Cannot connect to wire-pod at {self.endpoint}: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise WirePodError(
                f"wire-pod error {e.response.status_code}: {e.response.text[:200]}"
            ) from e

        data = response.json()
        config = parse_sdk_info(data)

        log.info(
            "wirepod.sdk_info",
            robots=len(config.robots),
            global_guid=config.global_guid[:8] + "...",
        )
        return config

    async def get_robot_guid(self, esn: str | None = None) -> str:
        """Get the SDK GUID for a specific robot or the first available.

        Args:
            esn: Robot ESN (serial number). If None, uses first robot.

        Returns:
            The robot's SDK GUID token.

        Raises:
            WirePodError: If no robot is found.
        """
        config = await self.get_sdk_info()

        if not config.robots:
            raise WirePodError("No robots registered with wire-pod")

        if esn:
            for robot in config.robots:
                if robot.esn == esn:
                    return robot.guid
            raise WirePodError(f"Robot {esn} not found in wire-pod")

        return config.robots[0].guid

    async def is_available(self) -> bool:
        """Check if wire-pod is reachable.

        Returns:
            True if wire-pod responds.
        """
        try:
            response = await self._client.get(
                f"{self.endpoint}/api-sdk/get_sdk_info"
            )
            return response.status_code == 200
        except Exception:
            return False
