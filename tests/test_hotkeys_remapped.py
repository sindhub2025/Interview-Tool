from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget

from ghostmic.main import GhostMicApp, _load_config
from ghostmic.ui.settings_dialog import SettingsDialog
from ghostmic.utils.hotkeys import DEFAULT_HOTKEYS, HotkeyManager


def _qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_default_hotkeys_remapped() -> None:
    assert DEFAULT_HOTKEYS["quit_app"] == "<ctrl>+<shift>+q"
    assert DEFAULT_HOTKEYS["toggle_window"] == "<ctrl>+<shift>+a"
    assert DEFAULT_HOTKEYS["screenshot_email"] == "<ctrl>+<shift>+z"


def test_settings_dialog_loads_and_saves_remapped_hotkeys() -> None:
    app = _qt_app()
    assert app is not None

    base_config = {
        "transcription": {"language": "en"},
        "ui": {"stealth_enabled": False},
        "hotkeys": {
            "toggle_recording": "ctrl+shift+g",
            "toggle_window": "ctrl+shift+a",
            "generate_response": "ctrl+g",
            "copy_response": "ctrl+shift+c",
            "clear_transcript": "ctrl+shift+x",
            "win_h_dictation": "win+h",
            "screenshot_email": "ctrl+shift+z",
            "quit_app": "ctrl+shift+q",
        },
        "dictation": {"enabled": True, "commit_idle_ms": 1200},
        "screenshot_email": {"enabled": False},
    }

    dialog = SettingsDialog(base_config)
    try:
        assert dialog._hotkey_quit.text() == "ctrl+shift+q"
        assert dialog._hotkey_window.text() == "ctrl+shift+a"
        assert dialog._hotkey_screenshot_email.text() == "ctrl+shift+z"

        emitted: list[dict] = []
        dialog.settings_saved.connect(emitted.append)
        dialog._save()

        assert emitted
        hotkeys = emitted[0]["hotkeys"]
        assert hotkeys["quit_app"] == "ctrl+shift+q"
        assert hotkeys["toggle_window"] == "ctrl+shift+a"
        assert hotkeys["screenshot_email"] == "ctrl+shift+z"
    finally:
        dialog.deleteLater()


def test_load_config_migrates_old_hotkeys(tmp_path) -> None:
    legacy_cfg = {
        "hotkeys": {
            "toggle_recording": "ctrl+shift+g",
            "toggle_window": "ctrl+shift+h",
            "screenshot_email": "ctrl+shift+e",
        }
    }
    legacy_file = tmp_path / "legacy_config.json"
    legacy_file.write_text(json.dumps(legacy_cfg), encoding="utf-8")

    loaded_legacy = _load_config(str(legacy_file))
    assert loaded_legacy["hotkeys"]["toggle_window"] == "ctrl+shift+a"
    assert loaded_legacy["hotkeys"]["screenshot_email"] == "ctrl+shift+z"
    assert loaded_legacy["hotkeys"]["quit_app"] == "ctrl+shift+q"

    prev_cfg = {
        "hotkeys": {
            "toggle_recording": "ctrl+shift+g",
            "toggle_window": "ctrl+a",
            "screenshot_email": "ctrl+z",
            "quit_app": "ctrl+q",
        }
    }
    prev_file = tmp_path / "prev_config.json"
    prev_file.write_text(json.dumps(prev_cfg), encoding="utf-8")

    loaded_prev = _load_config(str(prev_file))
    assert loaded_prev["hotkeys"]["toggle_window"] == "ctrl+shift+a"
    assert loaded_prev["hotkeys"]["screenshot_email"] == "ctrl+shift+z"
    assert loaded_prev["hotkeys"]["quit_app"] == "ctrl+shift+q"

    custom_cfg = {
        "hotkeys": {
            "toggle_window": "ctrl+alt+a",
            "screenshot_email": "ctrl+alt+z",
            "quit_app": "ctrl+alt+q",
        }
    }
    custom_file = tmp_path / "custom_config.json"
    custom_file.write_text(json.dumps(custom_cfg), encoding="utf-8")

    loaded_custom = _load_config(str(custom_file))
    assert loaded_custom["hotkeys"]["toggle_window"] == "ctrl+alt+a"
    assert loaded_custom["hotkeys"]["screenshot_email"] == "ctrl+alt+z"
    assert loaded_custom["hotkeys"]["quit_app"] == "ctrl+alt+q"


def test_hotkey_manager_registers_remapped_hotkeys(monkeypatch) -> None:
    registered = {}

    class FakeGlobalHotKeys:
        def __init__(self, hotkeys):
            registered.update(hotkeys)

        def start(self):
            pass

        def stop(self):
            pass

    fake_keyboard = types.SimpleNamespace(GlobalHotKeys=FakeGlobalHotKeys)
    fake_pynput = types.ModuleType("pynput")
    fake_pynput.keyboard = fake_keyboard
    monkeypatch.setitem(sys.modules, "pynput", fake_pynput)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", fake_keyboard)

    manager = HotkeyManager(
        {
            "toggle_window": "ctrl+shift+a",
            "screenshot_email": "ctrl+shift+z",
            "quit_app": "ctrl+shift+q",
        },
        {
            "toggle_window": lambda: None,
            "screenshot_email": lambda: None,
            "quit_app": lambda: None,
        },
    )
    manager.start()
    assert "<ctrl>+<shift>+a" in registered
    assert "<ctrl>+<shift>+z" in registered
    assert "<ctrl>+<shift>+q" in registered
    manager.stop()


def test_toggle_window_hides_and_brings_back() -> None:
    app = _qt_app()
    assert app is not None

    fake_app = types.SimpleNamespace()
    window = QWidget()
    window.show()
    fake_app._window = window
    fake_app._tray = MagicMock()

    # Initial state: visible and normal -> calling _toggle_window should hide it
    GhostMicApp._toggle_window(fake_app)
    assert not window.isVisible()

    # Calling _toggle_window again should restore it
    GhostMicApp._toggle_window(fake_app)
    assert window.isVisible()
    assert not window.isMinimized()

    # If minimized -> calling _toggle_window should unminimize and restore it
    window.showMinimized()
    GhostMicApp._toggle_window(fake_app)
    assert window.isVisible()
    assert not window.isMinimized()

    window.deleteLater()
