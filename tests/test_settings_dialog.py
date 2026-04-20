from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from ghostmic.ui.settings_dialog import SettingsDialog


def _qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _base_config(*, stealth_enabled: bool = True) -> dict:
    return {
        "ai": {
            "backend": "groq",
            "main_backend": "groq",
            "groq_api_key": "",
            "temperature": 0.7,
            "trigger_mode": "auto",
            "resume_context_enabled": True,
            "sql_profile_enabled": False,
        },
        "audio": {
            "sample_rate": 16000,
            "capture_mic": False,
        },
        "transcription": {
            "language": "en",
            "model_size": "base.en",
            "compute_type": "int8",
            "beam_size": 5,
            "remote_fallback": True,
        },
        "ui": {
            "opacity": 0.95,
            "font_size": 11,
            "window_width": 420,
            "window_height": 650,
            "stealth_enabled": stealth_enabled,
        },
        "hotkeys": {
            "toggle_recording": "ctrl+shift+g",
            "toggle_window": "ctrl+shift+h",
            "generate_response": "ctrl+g",
            "copy_response": "ctrl+shift+c",
            "clear_transcript": "ctrl+shift+x",
            "win_h_dictation": "win+h",
        },
        "dictation": {
            "enabled": True,
            "commit_idle_ms": 1200,
        },
    }


def test_settings_dialog_loads_stealth_checkbox_from_config() -> None:
    app = _qt_app()
    assert app is not None

    dialog = SettingsDialog(_base_config(stealth_enabled=False))
    try:
        assert dialog._stealth_enabled_check.isChecked() is False
    finally:
        dialog.deleteLater()


def test_settings_dialog_saves_stealth_checkbox_state() -> None:
    app = _qt_app()
    assert app is not None

    dialog = SettingsDialog(_base_config(stealth_enabled=True))
    try:
        emitted: list[dict] = []
        dialog.settings_saved.connect(emitted.append)

        dialog._stealth_enabled_check.setChecked(False)
        dialog._save()

        assert emitted
        assert emitted[0]["ui"]["stealth_enabled"] is False
    finally:
        dialog.deleteLater()
