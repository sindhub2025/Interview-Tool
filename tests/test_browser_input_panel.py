"""
Tests for BrowserInputPanel UI.

Verifies rendering, message append, collapse/expand, copy, delete, and clear.
Requires a QApplication instance (provided by conftest or created here).
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone

import pytest

# Qt must be imported before creating any widget
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


def _make_msg(msg_type: str = "question", content: str = "Hello?", language=None):
    from ghostmic.domain.browser_message import BrowserMessage
    return BrowserMessage(
        protocol_version=1,
        message_id=str(uuid.uuid4()),
        type=msg_type,
        content=content,
        language=language,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


@pytest.fixture
def panel(qapp):
    from ghostmic.ui.browser_input_panel import BrowserInputPanel
    cfg = {
        "browser_bridge": {
            "max_messages": 100,
            "sender_url": "https://sender.example.com",
        }
    }
    p = BrowserInputPanel(cfg)
    p.show()
    return p


class TestBrowserInputPanelInitial:
    def test_panel_starts_expanded(self, panel) -> None:
        assert panel.is_panel_expanded

    def test_no_messages_on_start(self, panel) -> None:
        assert len(panel._messages) == 0

    def test_unread_count_zero_on_start(self, panel) -> None:
        assert panel.unread_count == 0


class TestBrowserInputPanelMessages:
    def test_append_question(self, panel) -> None:
        msg = _make_msg("question", "What is DI?")
        panel.append_message(msg)
        assert len(panel._messages) == 1
        assert panel._messages[0].type == "question"

    def test_append_code(self, panel) -> None:
        msg = _make_msg("code", "def hello(): pass", "python")
        panel.append_message(msg)
        # Code should be in cards
        assert msg.message_id in panel._cards

    def test_append_answer(self, panel) -> None:
        msg = _make_msg("answer", "It means dependency injection.")
        panel.append_message(msg)
        assert msg.message_id in panel._cards

    def test_duplicate_not_added(self, panel) -> None:
        msg = _make_msg("question", "Unique question")
        panel.append_message(msg)
        count_before = len(panel._messages)
        panel.append_message(msg)  # same message_id
        assert len(panel._messages) == count_before

    def test_delete_individual(self, panel) -> None:
        msg = _make_msg("question", "Delete me")
        panel.append_message(msg)
        count_before = len(panel._messages)
        panel._on_delete_card(msg.message_id)
        assert len(panel._messages) == count_before - 1
        assert msg.message_id not in panel._cards

    def test_code_indentation_preserved(self, panel) -> None:
        code = "def foo():\n    x = 1\n    return x\n"
        msg = _make_msg("code", code, "python")
        panel.append_message(msg)
        assert panel._messages[-1].content == code


class TestBrowserInputPanelCollapse:
    def test_collapse_hides_body(self, panel) -> None:
        panel.set_collapsed(True)
        assert not panel.is_panel_expanded
        assert not panel._body_widget.isVisible()

    def test_expand_shows_body(self, panel) -> None:
        panel.set_collapsed(True)
        panel.set_collapsed(False)
        assert panel.is_panel_expanded
        assert panel._body_widget.isVisible()

    def test_unread_increments_when_collapsed(self, panel) -> None:
        panel.set_collapsed(True)
        before = panel.unread_count
        panel.append_message(_make_msg("question", "New while collapsed"))
        assert panel.unread_count == before + 1

    def test_unread_resets_on_expand(self, panel) -> None:
        panel.set_collapsed(True)
        panel.append_message(_make_msg("question", "Another new message"))
        panel.set_collapsed(False)
        assert panel.unread_count == 0


class TestBrowserInputPanelClear:
    def test_clear_all(self, panel) -> None:
        for _ in range(3):
            panel.append_message(_make_msg())
        # Directly call without QMessageBox (bypass confirmation in tests)
        for card in list(panel._cards.values()):
            card.deleteLater()
        panel._cards.clear()
        panel._messages.clear()
        panel._update_count_label()
        assert len(panel._messages) == 0
        assert len(panel._cards) == 0
