"""Tests for Player UPnP event subscription (async_start_upnp_events / async_stop_upnp_events)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def player(mock_client):
    """Player with a reachable mock UPnP client and a health tracker."""
    from pywiim.player import Player

    p = Player(mock_client)
    p._upnp_client = MagicMock()
    p._upnp_client.unwind_notify_server = AsyncMock()
    p._ensure_upnp_client = AsyncMock(return_value=True)
    p._upnp_health_tracker = MagicMock()
    return p


@pytest.fixture
def eventer_cls():
    with patch("pywiim.upnp.eventer.UpnpEventer") as cls:
        cls.return_value.start = AsyncMock()
        cls.return_value.async_unsubscribe = AsyncMock()
        yield cls


@pytest.mark.asyncio
async def test_start_subscribes_with_player_as_state_manager(player, eventer_cls):
    assert player.upnp_events_active is False

    assert await player.async_start_upnp_events(callback_host="10.0.0.10", callback_port=8123) is True

    args, kwargs = eventer_cls.call_args
    assert args[0] is player._upnp_client
    assert args[1] is player
    assert kwargs["state_updated_callback"] == player._on_upnp_event
    eventer_cls.return_value.start.assert_awaited_once_with(callback_host="10.0.0.10", callback_port=8123)
    assert player.upnp_events_active is True
    # Misses counted before subscribing are discarded.
    player._upnp_health_tracker.reset_statistics.assert_called_once()


@pytest.mark.asyncio
async def test_start_is_idempotent(player, eventer_cls):
    await player.async_start_upnp_events()
    await player.async_start_upnp_events()

    assert eventer_cls.call_count == 1


@pytest.mark.asyncio
async def test_start_failure_cleans_up_and_returns_false(player, eventer_cls):
    eventer_cls.return_value.start = AsyncMock(side_effect=RuntimeError("subscribe rejected"))

    assert await player.async_start_upnp_events() is False

    assert player.upnp_events_active is False
    player._upnp_client.unwind_notify_server.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_without_upnp_client_returns_false(player, eventer_cls):
    player._ensure_upnp_client = AsyncMock(return_value=False)

    assert await player.async_start_upnp_events() is False

    eventer_cls.assert_not_called()
    assert player.upnp_events_active is False


@pytest.mark.asyncio
async def test_stop_unsubscribes(player, eventer_cls):
    await player.async_start_upnp_events()

    await player.async_stop_upnp_events()

    eventer_cls.return_value.async_unsubscribe.assert_awaited_once()
    assert player.upnp_events_active is False

    # Stopping again is a no-op.
    await player.async_stop_upnp_events()
    eventer_cls.return_value.async_unsubscribe.assert_awaited_once()


def test_upnp_event_notifies_owner(player):
    callback = MagicMock()
    player._on_state_changed = callback

    player._on_upnp_event()

    callback.assert_called_once()
