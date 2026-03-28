"""Tests for wire-pod integration — SDK auth and robot discovery."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.wirepod import (
    CustomIntent,
    WirePodClient,
    WirePodError,
    build_intent_json,
    parse_sdk_info,
)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

_SDK_INFO_RESPONSE = {
    "global_guid": "tni1TRsTRTaNSapjo0Y+Sw==",
    "robots": [
        {
            "esn": "00603151",
            "ip_address": "192.168.1.125",
            "guid": "E5L3Z/2FESsTU4s/V5ACHQ==",
            "activated": True,
        }
    ],
}


class TestParseSdkInfo:
    def test_parses_single_robot(self):
        config = parse_sdk_info(_SDK_INFO_RESPONSE)
        assert config.global_guid == "tni1TRsTRTaNSapjo0Y+Sw=="
        assert len(config.robots) == 1
        assert config.robots[0].esn == "00603151"
        assert config.robots[0].ip_address == "192.168.1.125"
        assert config.robots[0].guid == "E5L3Z/2FESsTU4s/V5ACHQ=="
        assert config.robots[0].activated is True

    def test_parses_empty_robots(self):
        config = parse_sdk_info({"global_guid": "abc", "robots": []})
        assert config.global_guid == "abc"
        assert config.robots == []

    def test_parses_multiple_robots(self):
        data = {
            "global_guid": "g",
            "robots": [
                {"esn": "001", "ip_address": "1.1.1.1", "guid": "a", "activated": True},
                {"esn": "002", "ip_address": "2.2.2.2", "guid": "b", "activated": False},
            ],
        }
        config = parse_sdk_info(data)
        assert len(config.robots) == 2
        assert config.robots[1].activated is False

    def test_missing_fields_default(self):
        config = parse_sdk_info({})
        assert config.global_guid == ""
        assert config.robots == []


class TestBuildIntentJson:
    def test_basic_intent(self):
        intent = CustomIntent(
            name="test_intent",
            description="A test",
            utterances=["do something", "make it work"],
            intent="intent_custom_test",
        )
        result = build_intent_json(intent)
        assert result["name"] == "test_intent"
        assert result["description"] == "A test"
        assert result["utterances"] == ["do something", "make it work"]
        assert result["intent"] == "intent_custom_test"
        assert result["exec"] == ""
        assert result["execargs"] == ""

    def test_intent_with_exec(self):
        intent = CustomIntent(
            name="run_script",
            description="Run a script",
            utterances=["run script"],
            intent="intent_custom_script",
            exec_command="python3",
            exec_args="/path/to/script.py !botSerial",
        )
        result = build_intent_json(intent)
        assert result["exec"] == "python3"
        assert "!botSerial" in result["execargs"]


# ---------------------------------------------------------------------------
# WirePodClient
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> dict:
    config = {"wirepod": {"endpoint": "http://localhost:8090"}}
    config.update(overrides)
    return config


def _mock_sdk_response() -> MagicMock:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = _SDK_INFO_RESPONSE
    return mock_response


class TestWirePodClientGetSdkInfo:
    @pytest.mark.asyncio
    async def test_success(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=_mock_sdk_response())

        client = WirePodClient(_make_config(), http_client=mock_client)
        config = await client.get_sdk_info()

        assert config.global_guid == "tni1TRsTRTaNSapjo0Y+Sw=="
        assert len(config.robots) == 1
        mock_client.get.assert_called_once()
        assert "get_sdk_info" in mock_client.get.call_args[0][0]

    @pytest.mark.asyncio
    async def test_connection_error(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        client = WirePodClient(_make_config(), http_client=mock_client)

        with pytest.raises(WirePodError, match="Cannot connect"):
            await client.get_sdk_info()

    @pytest.mark.asyncio
    async def test_http_error(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal error"
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "500", request=MagicMock(), response=mock_response
            )
        )
        mock_client.get = AsyncMock(return_value=mock_response)

        client = WirePodClient(_make_config(), http_client=mock_client)

        with pytest.raises(WirePodError, match="error 500"):
            await client.get_sdk_info()


class TestGetRobotGuid:
    @pytest.mark.asyncio
    async def test_first_robot(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=_mock_sdk_response())

        client = WirePodClient(_make_config(), http_client=mock_client)
        guid = await client.get_robot_guid()

        assert guid == "E5L3Z/2FESsTU4s/V5ACHQ=="

    @pytest.mark.asyncio
    async def test_specific_esn(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=_mock_sdk_response())

        client = WirePodClient(_make_config(), http_client=mock_client)
        guid = await client.get_robot_guid(esn="00603151")

        assert guid == "E5L3Z/2FESsTU4s/V5ACHQ=="

    @pytest.mark.asyncio
    async def test_unknown_esn_raises(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=_mock_sdk_response())

        client = WirePodClient(_make_config(), http_client=mock_client)

        with pytest.raises(WirePodError, match="not found"):
            await client.get_robot_guid(esn="99999999")

    @pytest.mark.asyncio
    async def test_no_robots_raises(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"global_guid": "x", "robots": []}
        mock_client.get = AsyncMock(return_value=mock_response)

        client = WirePodClient(_make_config(), http_client=mock_client)

        with pytest.raises(WirePodError, match="No robots"):
            await client.get_robot_guid()


class TestAvailability:
    @pytest.mark.asyncio
    async def test_available(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_response)

        client = WirePodClient(_make_config(), http_client=mock_client)
        assert await client.is_available() is True

    @pytest.mark.asyncio
    async def test_unavailable(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        client = WirePodClient(_make_config(), http_client=mock_client)
        assert await client.is_available() is False
