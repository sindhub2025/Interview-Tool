from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from ghostmic.ui.controls_bar import ControlsBar


def _qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_capture_button_is_before_settings_button() -> None:
    app = _qt_app()
    assert app is not None

    controls = ControlsBar()
    layout = controls.layout()

    capture_index = None
    settings_index = None
    for idx in range(layout.count()):
        item = layout.itemAt(idx)
        if item is None:
            continue

        widget = item.widget()
        if widget is None:
            continue

        if widget is controls._screenshot_btn:
            capture_index = idx
        if widget is controls._settings_btn:
            settings_index = idx

    assert capture_index is not None
    stealth_index = None
    assert settings_index is not None
    for idx in range(layout.count()):
        item = layout.itemAt(idx)
        if item is None:
            continue

        widget = item.widget()
        if widget is None:
            continue

        if widget is controls._stealth_btn:
            stealth_index = idx

    assert stealth_index is not None
    assert capture_index < settings_index
    assert capture_index < stealth_index < settings_index


def test_stealth_button_toggles_and_can_be_synced() -> None:
    app = _qt_app()
    assert app is not None

    controls = ControlsBar()

    emitted: list[bool] = []
    controls.stealth_toggled.connect(emitted.append)

    controls._stealth_btn.click()
    assert emitted == [False]
    assert controls.is_stealth_enabled is False
    assert controls._stealth_btn.text() == "Stealth Off"

    controls.set_stealth_enabled(True)
    assert emitted == [False]
    assert controls.is_stealth_enabled is True
    assert controls._stealth_btn.text() == "Stealth On"
