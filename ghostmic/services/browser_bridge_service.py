"""
BrowserBridgeService — session lifecycle and Supabase Realtime communication.

Responsibilities:
- Authenticate the desktop client anonymously.
- Create / close browser sessions via Supabase Edge Functions.
- Connect to the private Realtime channel.
- Subscribe to incoming messages and emit Qt signals.
- Run the Supabase Realtime client on a dedicated background thread.
- Never block the PyQt main thread.
- Never access or modify Qt widgets directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

from ghostmic.domain.browser_message import (
    BrowserMessage,
    BrowserMessageValidationError,
    generate_pairing_code,
)
from ghostmic.services.browser_transport import BrowserTransport, SessionInfo
from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Connection state machine
# ---------------------------------------------------------------------------

STATE_DISCONNECTED = "DISCONNECTED"
STATE_STARTING_SESSION = "STARTING_SESSION"
STATE_WAITING_FOR_PAIRING = "WAITING_FOR_PAIRING"
STATE_CONNECTING = "CONNECTING"
STATE_CONNECTED = "CONNECTED"
STATE_RECONNECTING = "RECONNECTING"
STATE_ERROR = "ERROR"
STATE_CLOSING = "CLOSING"


class BrowserBridgeService(QObject):
    """
    Qt-aware service that bridges the GhostMic receiver to Supabase Realtime.

    All signals are queued so they cross thread boundaries safely from the
    background asyncio loop into the Qt main thread.

    Signals:
        connection_state_changed(state: str, detail: str):
            Emitted on every state transition.
        pairing_code_ready(code: str, sender_url: str):
            Emitted when a session has been created and the receiver is
            subscribed and ready to accept a pairing connection.
        message_received(msg: BrowserMessage):
            Emitted for each validated message.
        message_error(description: str):
            Emitted when a received payload is invalid.
        session_closed():
            Emitted when the session has been fully torn down.
    """

    connection_state_changed = pyqtSignal(str, str)
    pairing_code_ready = pyqtSignal(str, str)
    message_received = pyqtSignal(object)
    message_error = pyqtSignal(str)
    session_closed = pyqtSignal()

    def __init__(
        self,
        transport: BrowserTransport,
        sender_url: str = "",
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._transport = transport
        self._sender_url = sender_url
        self._state = STATE_DISCONNECTED
        self._session: Optional[SessionInfo] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._seen_message_ids: set[str] = set()
        self._max_seen_ids = 1000  # prevent unbounded growth

    # ------------------------------------------------------------------
    # Public API (called from Qt main thread)
    # ------------------------------------------------------------------

    def start_session(self) -> None:
        """Begin a new browser session in the background."""
        if self._state not in (STATE_DISCONNECTED, STATE_ERROR):
            logger.warning(
                "BrowserBridgeService: start_session called in state %s — ignored.",
                self._state,
            )
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_event_loop,
            name="BrowserBridgeWorker",
            daemon=True,
        )
        self._thread.start()

    def stop_session(self) -> None:
        """Shut down the active session and release all network resources."""
        if self._state in (STATE_DISCONNECTED, STATE_CLOSING):
            return
        self._set_state(STATE_CLOSING, "User requested stop.")
        self._stop_event.set()
        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._async_close(), self._loop
            )

    def shutdown(self) -> None:
        """Hard shutdown for application exit."""
        self._stop_event.set()
        if self._loop is not None and self._loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(
                    self._async_close(), self._loop
                ).result(timeout=3.0)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=4.0)

    @property
    def state(self) -> str:
        return self._state

    @property
    def session(self) -> Optional[SessionInfo]:
        return self._session

    # ------------------------------------------------------------------
    # Background thread entry point
    # ------------------------------------------------------------------

    def _run_event_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_run())
        except Exception as exc:
            logger.exception("BrowserBridgeService: event loop error: %s", exc)
            self._set_state(STATE_ERROR, str(exc))
        finally:
            try:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                self._loop.close()
            except Exception:
                pass
            self._loop = None

    # ------------------------------------------------------------------
    # Async session lifecycle
    # ------------------------------------------------------------------

    async def _async_run(self) -> None:
        self._set_state(STATE_STARTING_SESSION, "Creating session…")
        try:
            session = await self._transport.create_session()
        except Exception as exc:
            logger.error("BrowserBridgeService: create_session failed: %s", exc)
            self._set_state(STATE_ERROR, f"Session creation failed: {exc}")
            return

        self._session = session
        self._set_state(STATE_CONNECTING, "Connecting to channel…")

        try:
            await self._transport.join_channel(
                session_id=session.session_id,
                on_message=self._on_raw_message,
                on_connected=self._on_channel_connected,
                on_disconnected=self._on_channel_disconnected,
            )
        except Exception as exc:
            logger.error("BrowserBridgeService: join_channel failed: %s", exc)
            self._set_state(STATE_ERROR, f"Channel connection failed: {exc}")
            await self._async_close()
            return

        # Emit pairing code once channel is subscribed
        self._set_state(STATE_WAITING_FOR_PAIRING, "Waiting for sender…")
        self.pairing_code_ready.emit(session.pairing_code, self._sender_url)

        # Keep alive until stop is requested
        while not self._stop_event.is_set():
            await asyncio.sleep(0.5)

        await self._async_close()

    async def _async_close(self) -> None:
        try:
            await self._transport.close_session()
        except Exception as exc:
            logger.warning("BrowserBridgeService: close_session error: %s", exc)
        finally:
            self._session = None
            self._set_state(STATE_DISCONNECTED, "Session closed.")
            self.session_closed.emit()

    # ------------------------------------------------------------------
    # Transport callbacks (called from transport's asyncio context)
    # ------------------------------------------------------------------

    def _on_raw_message(self, payload: dict) -> None:
        """Validate and dispatch a raw payload received from Supabase."""
        try:
            msg = BrowserMessage.from_dict(payload)
        except BrowserMessageValidationError as exc:
            logger.warning("BrowserBridgeService: invalid message: %s", exc)
            self.message_error.emit(str(exc))
            return
        except Exception as exc:
            logger.warning("BrowserBridgeService: unexpected parse error: %s", exc)
            self.message_error.emit(f"Message parse error: {exc}")
            return

        # Deduplication
        if msg.message_id in self._seen_message_ids:
            logger.debug(
                "BrowserBridgeService: duplicate message_id %s — skipped.",
                msg.message_id,
            )
            return

        self._seen_message_ids.add(msg.message_id)
        if len(self._seen_message_ids) > self._max_seen_ids:
            # Evict oldest by rebuilding the set keeping only recent half
            ids = list(self._seen_message_ids)
            self._seen_message_ids = set(ids[len(ids) // 2 :])

        logger.debug(
            "BrowserBridgeService: message received type=%s id=%s",
            msg.type,
            msg.message_id,
        )
        self.message_received.emit(msg)

    def _on_channel_connected(self) -> None:
        self._set_state(STATE_CONNECTED, "Sender connected.")

    def _on_channel_disconnected(self) -> None:
        if self._state not in (STATE_CLOSING, STATE_DISCONNECTED):
            self._set_state(STATE_RECONNECTING, "Reconnecting…")

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _set_state(self, state: str, detail: str = "") -> None:
        self._state = state
        self.connection_state_changed.emit(state, detail)
        logger.debug("BrowserBridgeService: state → %s (%s)", state, detail)
