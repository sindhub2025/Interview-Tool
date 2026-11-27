"""
BrowserTransport — abstract transport interface for the Browser Input feature.

Separating the transport interface from the service and UI means the cloud
provider (Supabase) can be replaced without touching either the bridge
service or the browser input panel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional


@dataclass
class SessionInfo:
    """
    Data returned when a session is created server-side.

    Attributes:
        session_id: Opaque identifier used to join the channel.
        pairing_code: Human-readable code displayed to the receiver user.
        channel_topic: The Supabase Realtime channel topic string.
        expires_at: ISO-8601 expiry timestamp (informational).
    """

    session_id: str
    pairing_code: str
    channel_topic: str
    expires_at: str


# Callback types
OnMessageCallback = Callable[[dict], None]
OnConnectedCallback = Callable[[], None]
OnDisconnectedCallback = Callable[[], None]


class BrowserTransport(ABC):
    """
    Abstract async transport interface.

    Implementations must be thread-safe and must not import PyQt widgets.
    """

    @abstractmethod
    async def create_session(self) -> SessionInfo:
        """
        Authenticate and create a new browser session.

        Returns:
            SessionInfo containing the pairing code and channel topic.

        Raises:
            BrowserTransportError: on any network or auth failure.
        """
        ...

    @abstractmethod
    async def join_channel(
        self,
        session_id: str,
        on_message: OnMessageCallback,
        on_connected: OnConnectedCallback,
        on_disconnected: OnDisconnectedCallback,
    ) -> None:
        """
        Subscribe to the private Realtime channel for the given session.

        This method must return promptly; channel listening happens in
        background tasks or callbacks managed by the implementation.

        Args:
            session_id: The session identifier returned by create_session.
            on_message: Called with the raw payload dict for each event.
            on_connected: Called when the sender pairs and the channel is live.
            on_disconnected: Called when the channel drops unexpectedly.
        """
        ...

    @abstractmethod
    async def close_session(self) -> None:
        """
        Disconnect from the channel and invalidate the session server-side.

        Must release all network resources; safe to call multiple times.
        """
        ...


class BrowserTransportError(Exception):
    """Raised by transport implementations on network or auth failure."""


class NullBrowserTransport(BrowserTransport):
    """
    No-op transport used when Supabase is not configured.

    Allows the application to start without crashing when
    ``browser_bridge.supabase_url`` or ``browser_bridge.publishable_key``
    are not set.
    """

    async def create_session(self) -> SessionInfo:
        raise BrowserTransportError(
            "Supabase is not configured. "
            "Set supabase_url and publishable_key in Settings → Browser Input."
        )

    async def join_channel(
        self,
        session_id: str,
        on_message: OnMessageCallback,
        on_connected: OnConnectedCallback,
        on_disconnected: OnDisconnectedCallback,
    ) -> None:
        pass

    async def close_session(self) -> None:
        pass
