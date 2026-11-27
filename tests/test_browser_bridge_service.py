"""
Tests for BrowserBridgeService using mocked transport.

No live Supabase required. All network calls are simulated.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from PyQt6.QtWidgets import QApplication
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _QT_AVAILABLE, reason="PyQt6 not available in this environment"
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


def _make_session():
    from ghostmic.services.browser_transport import SessionInfo
    return SessionInfo(
        session_id=str(uuid.uuid4()),
        pairing_code="ABCDE12345",
        channel_topic="browser_session:test",
        expires_at="2099-01-01T00:00:00Z",
    )


class MockTransport:
    """In-memory mock transport that simulates Supabase without network."""

    def __init__(self, session=None, raise_on_create=False, raise_on_join=False):
        self._session = session or _make_session()
        self._raise_on_create = raise_on_create
        self._raise_on_join = raise_on_join
        self._closed = False
        self._on_message = None
        self._on_connected = None
        self._on_disconnected = None

    async def create_session(self):
        from ghostmic.services.browser_transport import BrowserTransportError
        if self._raise_on_create:
            raise BrowserTransportError("Simulated create failure")
        return self._session

    async def join_channel(self, session_id, on_message, on_connected, on_disconnected):
        from ghostmic.services.browser_transport import BrowserTransportError
        if self._raise_on_join:
            raise BrowserTransportError("Simulated join failure")
        self._on_message = on_message
        self._on_connected = on_connected
        self._on_disconnected = on_disconnected

    async def close_session(self):
        self._closed = True

    def simulate_message(self, payload: dict) -> None:
        if self._on_message:
            self._on_message(payload)

    def simulate_connected(self) -> None:
        if self._on_connected:
            self._on_connected()

    def simulate_disconnected(self) -> None:
        if self._on_disconnected:
            self._on_disconnected()


@pytest.fixture
def service(qapp):
    from ghostmic.services.browser_bridge_service import BrowserBridgeService
    transport = MockTransport()
    svc = BrowserBridgeService(transport, sender_url="https://test.example.com")
    return svc, transport


class TestBrowserBridgeServiceInitial:
    def test_initial_state_disconnected(self, service) -> None:
        from ghostmic.services.browser_bridge_service import STATE_DISCONNECTED
        svc, _ = service
        assert svc.state == STATE_DISCONNECTED

    def test_no_session_initially(self, service) -> None:
        svc, _ = service
        assert svc.session is None


class TestBrowserBridgeServiceMessages:
    def test_valid_message_emits_signal(self, service, qapp) -> None:
        svc, transport = service
        received = []
        svc.message_received.connect(received.append)

        payload = {
            "protocol_version": 1,
            "message_id": str(uuid.uuid4()),
            "type": "question",
            "content": "What is O(1)?",
            "language": None,
            "created_at": "2026-11-26T10:00:00Z",
        }
        svc._on_raw_message(payload)
        qapp.processEvents()
        assert len(received) == 1
        assert received[0].type == "question"

    def test_invalid_message_emits_error_signal(self, service, qapp) -> None:
        svc, transport = service
        errors = []
        svc.message_error.connect(errors.append)

        # Empty content is invalid
        payload = {
            "protocol_version": 1,
            "message_id": str(uuid.uuid4()),
            "type": "question",
            "content": "",
            "created_at": "2026-11-26T10:00:00Z",
        }
        svc._on_raw_message(payload)
        qapp.processEvents()
        assert len(errors) >= 1

    def test_duplicate_message_id_ignored(self, service, qapp) -> None:
        svc, transport = service
        received = []
        svc.message_received.connect(received.append)

        msg_id = str(uuid.uuid4())
        payload = {
            "protocol_version": 1,
            "message_id": msg_id,
            "type": "answer",
            "content": "It depends.",
            "created_at": "2026-11-26T10:00:00Z",
        }
        svc._on_raw_message(payload)
        qapp.processEvents()
        before = len(received)
        svc._on_raw_message(payload)  # duplicate
        qapp.processEvents()
        assert len(received) == before  # not incremented

    def test_malformed_non_dict_payload(self, service, qapp) -> None:
        svc, transport = service
        errors = []
        svc.message_error.connect(errors.append)
        svc._on_raw_message("not a dict")  # type: ignore[arg-type]
        qapp.processEvents()
        assert len(errors) >= 1


class TestNullTransport:
    def test_null_transport_raises_on_create(self) -> None:
        from ghostmic.services.browser_transport import NullBrowserTransport, BrowserTransportError
        t = NullBrowserTransport()
        coro = t.create_session()
        with pytest.raises(BrowserTransportError):
            coro.send(None)
        coro.close()
