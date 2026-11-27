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


class TestRemoteModeWindowBounds:
    def _make_window(self, monkeypatch, screen_rect):
        from PyQt6.QtWidgets import QApplication
        from ghostmic.ui.main_window import MainWindow

        class _FakeScreen:
            def availableGeometry(self):
                return screen_rect

        fake_screen = _FakeScreen()
        monkeypatch.setattr(QApplication, "screenAt", staticmethod(lambda _pos: fake_screen))
        monkeypatch.setattr(QApplication, "primaryScreen", staticmethod(lambda: fake_screen))

        cfg = {
            "ui": {
                "window_width": 1200,
                "window_height": 700,
                "browser_panel_visible": False,
                "browser_panel_width": 380,
                "docked": False,
            }
        }
        window = MainWindow(cfg)
        window.show()
        QApplication.instance().processEvents()
        return window

    def test_remote_toggle_button_does_not_double_expand(self, monkeypatch):
        from PyQt6.QtCore import QRect
        from PyQt6.QtWidgets import QApplication

        screen_rect = QRect(0, 0, 1920, 1080)
        window = self._make_window(monkeypatch, screen_rect)
        try:
            initial_w = window.width()
            window._title_bar._browser_input_btn.click()
            QApplication.instance().processEvents()

            # Should expand by a single panel width (380), not twice (760)
            assert window.width() == initial_w + 380
            assert window.x() + window.width() <= screen_rect.x() + screen_rect.width()
        finally:
            window.close()

    def test_remote_mode_stays_within_laptop_screen_bounds(self, monkeypatch):
        from PyQt6.QtCore import QRect
        from PyQt6.QtWidgets import QApplication

        # Typical 1366x768 laptop screen; window starts near bottom-right
        screen_rect = QRect(0, 0, 1366, 768)
        window = self._make_window(monkeypatch, screen_rect)
        try:
            window._title_bar._browser_input_btn.click()
            QApplication.instance().processEvents()

            geo = window.geometry()
            assert geo.width() <= screen_rect.width()
            assert geo.height() <= screen_rect.height()
            assert geo.x() >= screen_rect.x()
            assert geo.y() >= screen_rect.y()
            assert geo.x() + geo.width() <= screen_rect.x() + screen_rect.width()
            assert geo.y() + geo.height() <= screen_rect.y() + screen_rect.height()

            # Calling again while already open must be idempotent
            window.toggle_browser_input_panel(True)
            QApplication.instance().processEvents()
            assert window.geometry() == geo
        finally:
            window.close()

    def test_remote_collapse_preserves_panel_width_and_restores_width(self, monkeypatch):
        from PyQt6.QtCore import QRect
        from PyQt6.QtWidgets import QApplication

        screen_rect = QRect(0, 0, 1366, 768)
        window = self._make_window(monkeypatch, screen_rect)
        try:
            initial_w = window.width()
            window._title_bar._browser_input_btn.click()
            QApplication.instance().processEvents()
            window._title_bar._browser_input_btn.click()
            QApplication.instance().processEvents()

            assert window._browser_panel_width == 380
            assert window.width() == initial_w
        finally:
            window.close()

    def test_startup_offscreen_saved_position_is_clamped_to_screen(self, monkeypatch):
        from PyQt6.QtCore import QRect
        from PyQt6.QtWidgets import QApplication
        from ghostmic.ui.main_window import MainWindow

        screen_rect = QRect(0, 0, 1366, 768)

        class _FakeScreen:
            def availableGeometry(self):
                return screen_rect

        fake_screen = _FakeScreen()
        monkeypatch.setattr(QApplication, "screenAt", staticmethod(lambda _pos: None))
        monkeypatch.setattr(QApplication, "primaryScreen", staticmethod(lambda: fake_screen))

        cfg = {
            "ui": {
                "window_width": 829,
                "window_height": 90,
                "window_x": -1703,
                "window_y": 66,
                "docked": False,
            }
        }
        window = MainWindow(cfg)
        try:
            geo = window.geometry()
            assert geo.x() >= screen_rect.x()
            assert geo.y() >= screen_rect.y()
            assert geo.x() + geo.width() <= screen_rect.x() + screen_rect.width()
            assert geo.y() + geo.height() <= screen_rect.y() + screen_rect.height()
        finally:
            window.close()

    def test_clear_popup_stays_on_top_of_main_window(self, monkeypatch):
        from PyQt6.QtCore import QRect, Qt
        from PyQt6.QtWidgets import QApplication, QMessageBox

        screen_rect = QRect(0, 0, 1366, 768)
        window = self._make_window(monkeypatch, screen_rect)
        try:
            window._title_bar._browser_input_btn.click()
            QApplication.instance().processEvents()
            window._browser_panel.append_message(_make_msg())

            seen_flags = []

            def fake_exec(self_box):
                seen_flags.append(self_box.windowFlags())
                return QMessageBox.StandardButton.Yes

            monkeypatch.setattr(QMessageBox, "exec", fake_exec)
            window._browser_panel._clear_btn.click()

            assert len(seen_flags) == 1
            assert bool(seen_flags[0] & Qt.WindowType.WindowStaysOnTopHint)
            assert len(window._browser_panel._messages) == 0
        finally:
            window.close()

    def test_clear_popup_applies_stealth_and_no_tooltips(self, monkeypatch):
        from PyQt6.QtCore import QRect
        from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget

        screen_rect = QRect(0, 0, 1366, 768)
        window = self._make_window(monkeypatch, screen_rect)
        try:
            # Verify no tooltips on BrowserInputPanel or its children
            assert window._browser_panel.toolTip() == ""
            for child in window._browser_panel.findChildren(QWidget):
                assert child.toolTip() == ""

            stealth_calls = []
            monkeypatch.setattr(
                window._browser_panel,
                "_apply_dialog_stealth",
                lambda box: stealth_calls.append(box.windowTitle()),
            )
            monkeypatch.setattr(
                QMessageBox, "exec", lambda _self: QMessageBox.StandardButton.Cancel
            )
            window._browser_panel._clear_btn.click()

            assert "Clear All Messages" in stealth_calls
        finally:
            window.close()

    def test_browser_panel_collapse_button_collapses_into_main_app(self, monkeypatch):
        from PyQt6.QtCore import QRect
        from PyQt6.QtWidgets import QApplication

        screen_rect = QRect(0, 0, 1366, 768)
        window = self._make_window(monkeypatch, screen_rect)
        try:
            initial_w = window.width()
            window._title_bar._browser_input_btn.click()
            QApplication.instance().processEvents()
            assert window._browser_panel.isVisible()
            assert window._title_bar._browser_input_btn.isChecked()

            # Click collapse button inside the browser panel itself
            window._browser_panel._collapse_btn.click()
            QApplication.instance().processEvents()

            assert not window._browser_panel.isVisible()
            assert not window._title_bar._browser_input_btn.isChecked()
            assert window.width() == initial_w
        finally:
            window.close()
