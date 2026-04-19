from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from ghostmic.ui.system_tray import SystemTrayIcon


def _qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_stealth_action_is_checkable_and_enabled_by_default() -> None:
    app = _qt_app()
    assert app is not None

    tray = SystemTrayIcon()
    try:
        assert tray._stealth_action.isCheckable() is True
        assert tray._stealth_action.isChecked() is True
        assert "Hide from Windows Capture" in tray._stealth_action.text()
    finally:
        tray.hide()
        tray.deleteLater()


def test_set_stealth_enabled_updates_action_without_emitting_signal() -> None:
    app = _qt_app()
    assert app is not None

    tray = SystemTrayIcon()
    try:
        emitted: list[bool] = []
        tray.stealth_toggled.connect(emitted.append)

        tray._stealth_action.trigger()
        assert emitted == [False]
        assert tray._stealth_action.isChecked() is False
        assert "Allow Screenshots" in tray._stealth_action.text()

        tray.set_stealth_enabled(True)
        assert emitted == [False]
        assert tray._stealth_action.isChecked() is True
        assert "Hide from Windows Capture" in tray._stealth_action.text()
    finally:
        tray.hide()
        tray.deleteLater()
