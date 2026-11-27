"""
SupabaseRealtimeTransport — Supabase-specific BrowserTransport implementation.

Uses the official Supabase Python SDK (``supabase-py``) with:
- Anonymous authentication.
- Edge Function invocations for session lifecycle.
- Private Realtime Broadcast channels.

Supabase-specific details are fully contained here.
The bridge service and UI are unaware of Supabase.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
from typing import Optional

from ghostmic.domain.browser_message import generate_pairing_code
from ghostmic.services.browser_transport import (
    BrowserTransport,
    BrowserTransportError,
    OnConnectedCallback,
    OnDisconnectedCallback,
    OnMessageCallback,
    SessionInfo,
)
from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)

# Event name used by the sender to broadcast messages
BROWSER_MESSAGE_EVENT = "browser_message"
SENDER_JOINED_EVENT = "sender_joined"


class SupabaseRealtimeTransport(BrowserTransport):
    """
    Supabase Realtime Broadcast transport.

    Args:
        supabase_url: The Supabase project URL.
        publishable_key: The Supabase anon/publishable key.
        session_ttl_minutes: How long a pairing code remains valid.
        max_message_bytes: Maximum payload size to accept (defence in depth).
    """

    def __init__(
        self,
        supabase_url: str,
        publishable_key: str,
        session_ttl_minutes: int = 10,
        max_message_bytes: int = 102_400,
    ) -> None:
        if not supabase_url or not publishable_key:
            raise BrowserTransportError(
                "supabase_url and publishable_key must both be provided."
            )
        self._url = supabase_url.rstrip("/")
        self._key = publishable_key
        self._ttl = session_ttl_minutes
        self._max_bytes = max_message_bytes
        self._client = None
        self._channel = None
        self._session_id: Optional[str] = None
        self._channel_topic: Optional[str] = None

    # ------------------------------------------------------------------
    # BrowserTransport interface
    # ------------------------------------------------------------------

    async def create_session(self) -> SessionInfo:
        """
        1. Create the Supabase client and authenticate anonymously.
        2. Invoke the create-browser-session Edge Function.
        3. Return pairing code and session info.
        """
        await self._ensure_client()
        assert self._client is not None

        try:
            # Anonymous sign-in
            auth_response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.auth.sign_in_anonymously(),
            )
            logger.debug(
                "SupabaseRealtimeTransport: anonymous auth ok, uid=%s",
                getattr(getattr(auth_response, "user", None), "id", "?"),
            )
        except Exception as exc:
            raise BrowserTransportError(f"Anonymous authentication failed: {exc}") from exc

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.functions.invoke(
                    "create-browser-session",
                    invoke_options={"body": {"ttl_minutes": self._ttl}},
                ),
            )
        except Exception as exc:
            raise BrowserTransportError(
                f"create-browser-session Edge Function failed: {exc}"
            ) from exc

        try:
            if hasattr(response, "data"):
                data = response.data
                if isinstance(data, (bytes, bytearray)):
                    data = json.loads(data)
            else:
                data = response

            self._session_id = data["session_id"]
            self._channel_topic = data["channel_topic"]
            pairing_code = data["pairing_code"]
            expires_at = data.get("expires_at", "")
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise BrowserTransportError(
                f"Unexpected response from create-browser-session: {exc}"
            ) from exc

        return SessionInfo(
            session_id=self._session_id,
            pairing_code=pairing_code,
            channel_topic=self._channel_topic,
            expires_at=expires_at,
        )

    async def join_channel(
        self,
        session_id: str,
        on_message: OnMessageCallback,
        on_connected: OnConnectedCallback,
        on_disconnected: OnDisconnectedCallback,
    ) -> None:
        """Subscribe to the private Realtime broadcast channel."""
        assert self._client is not None, "Must call create_session first."
        topic = self._channel_topic or f"browser_session:{session_id}"

        loop = asyncio.get_event_loop()

        def _subscribe():
            channel = self._client.channel(topic, {"config": {"private": True}})

            def _on_broadcast(payload):
                """Called by Supabase SDK for each broadcast event."""
                try:
                    event_type = payload.get("event", "")
                    if event_type == SENDER_JOINED_EVENT:
                        loop.call_soon_threadsafe(on_connected)
                        return
                    if event_type == BROWSER_MESSAGE_EVENT:
                        body = payload.get("payload", payload)
                        if isinstance(body, str):
                            body = json.loads(body)
                        loop.call_soon_threadsafe(on_message, body)
                except Exception as exc:
                    logger.warning(
                        "SupabaseRealtimeTransport: broadcast handler error: %s", exc
                    )

            def _on_system(status, *args):
                logger.debug(
                    "SupabaseRealtimeTransport: channel system event: %s %s",
                    status,
                    args,
                )
                if status in ("CHANNEL_ERROR", "TIMED_OUT"):
                    loop.call_soon_threadsafe(on_disconnected)

            channel.on_broadcast(event=BROWSER_MESSAGE_EVENT, callback=_on_broadcast)
            channel.on_broadcast(event=SENDER_JOINED_EVENT, callback=_on_broadcast)
            channel.subscribe(_on_system)
            self._channel = channel

        await loop.run_in_executor(None, _subscribe)
        logger.info(
            "SupabaseRealtimeTransport: subscribed to channel %s", topic
        )

    async def close_session(self) -> None:
        """Unsubscribe from the channel and close the Edge Function session."""
        if self._channel is not None:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._channel.unsubscribe()
                )
            except Exception as exc:
                logger.warning(
                    "SupabaseRealtimeTransport: unsubscribe error: %s", exc
                )
            finally:
                self._channel = None

        if self._session_id and self._client is not None:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.functions.invoke(
                        "close-browser-session",
                        invoke_options={"body": {"session_id": self._session_id}},
                    ),
                )
            except Exception as exc:
                logger.warning(
                    "SupabaseRealtimeTransport: close-browser-session error: %s", exc
                )
            finally:
                self._session_id = None

        if self._client is not None:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._client.auth.sign_out()
                )
            except Exception as exc:
                logger.warning(
                    "SupabaseRealtimeTransport: sign_out error: %s", exc
                )
            finally:
                self._client = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _ensure_client(self) -> None:
        """Lazily create the Supabase client (import is deferred so the app
        starts even when supabase-py is not installed)."""
        if self._client is not None:
            return
        try:
            from supabase import create_client, Client  # type: ignore[import]
        except ImportError as exc:
            raise BrowserTransportError(
                "supabase Python package is not installed. "
                "Run: pip install supabase"
            ) from exc

        def _create():
            return create_client(self._url, self._key)

        self._client = await asyncio.get_event_loop().run_in_executor(None, _create)
        logger.debug(
            "SupabaseRealtimeTransport: Supabase client created for %s", self._url
        )
