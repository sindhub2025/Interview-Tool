"""
GhostMic — Real-time AI meeting/interview assistant.

Entry point: parses CLI args, sets up logging, loads config,
initialises the PyQt6 application, starts background threads,
applies stealth, and enters the event loop.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import List, Optional

# ── Ensure the project root is on sys.path ────────────────────────────
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    _HERE = os.path.join(sys._MEIPASS, "ghostmic")
else:
    _HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if not getattr(sys, "frozen", False) and _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ghostmic.utils.logger import configure_logging, get_logger

# Logger is set up in _main() after parsing --debug, but we need it here
# for module-level imports that may log warnings.
_early_logger = get_logger("ghostmic.startup")

# ── Default config path ───────────────────────────────────────────────
DEFAULT_CONFIG_PATH = os.path.join(_HERE, "config.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ghostmic",
        description="GhostMic — Real-time AI meeting/interview assistant",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH, help="Path to config.json"
    )
    parser.add_argument(
        "--minimized", action="store_true", help="Start minimised to tray"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _default_config() -> dict:
    """Return a hardcoded minimal default configuration."""
    return {
        "ai": {
            "backend": "openai",
            "openai_api_key": "",
            "openai_model": "gpt-5-mini",
            "groq_api_key": "",
            "groq_model": "llama-3.1-70b-versatile",
            "system_prompt": (
                "You are a real-time interview/meeting assistant helping the user "
                "respond to questions. Keep responses concise and natural."
            ),
            "temperature": 0.7,
            "trigger_mode": "auto",
            "context_segments": 10,
            "session_context": "",
        },
        "audio": {"input_device": None, "loopback_device": None, "sample_rate": 16000},
        "transcription": {
            "model_size": "base.en",
            "compute_type": "int8",
            "language": "en",
            "beam_size": 3,
        },
        "ui": {
            "opacity": 0.95,
            "font_size": 11,
            "window_width": 420,
            "window_height": 650,
            "window_x": None,
            "window_y": None,
            "always_on_top": True,
            "compact_mode": False,
        },
        "hotkeys": {
            "toggle_recording": "ctrl+shift+g",
            "toggle_window": "ctrl+shift+h",
            "generate_response": "ctrl+g",
            "copy_response": "ctrl+shift+c",
            "clear_transcript": "ctrl+shift+x",
        },
    }


def _load_config(path: str) -> dict:
    """Load config.json; create a default if it does not exist."""
    def _merge_defaults(base: dict, defaults: dict) -> dict:
        merged = dict(base)
        for key, default_value in defaults.items():
            if key not in merged:
                merged[key] = default_value
                continue
            if isinstance(default_value, dict) and isinstance(merged[key], dict):
                merged[key] = _merge_defaults(merged[key], default_value)
        return merged

    if not os.path.exists(path):
        # Try to read the bundled default config.json
        default_path = DEFAULT_CONFIG_PATH
        cfg: dict
        if os.path.exists(default_path):
            with open(default_path, encoding="utf-8") as fh:
                cfg = json.load(fh)
        else:
            cfg = _default_config()

        # Write it to the requested path if it differs
        if path != default_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(cfg, fh, indent=2)
            except OSError as exc:
                _early_logger.warning("Could not create config file %s: %s", path, exc)
        return cfg

    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)

    cfg = _merge_defaults(cfg, _default_config())
    ai_cfg = cfg.get("ai", {})
    if isinstance(ai_cfg, dict):
        ai_cfg.pop("ollama_model", None)
        ai_cfg.pop("ollama_url", None)
        if ai_cfg.get("backend") == "ollama":
            ai_cfg["backend"] = "openai"

    return cfg


def _save_config(config: dict, path: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)
    except OSError as exc:
        _early_logger.error("Could not save config: %s", exc)


# ---------------------------------------------------------------------------
# Windows version check
# ---------------------------------------------------------------------------


def _check_windows_version() -> bool:
    """Return True if Windows 10 build ≥ 19041 (2004) is detected."""
    if sys.platform != "win32":
        return False
    try:
        ver = sys.getwindowsversion()  # type: ignore[attr-defined]
        # Build 19041 = Windows 10 version 2004
        if ver.major >= 10 and ver.build >= 19041:
            return True
        _early_logger.warning(
            "Windows build %d is older than 19041 (Win10 2004). "
            "WDA_EXCLUDEFROMCAPTURE may not be available; "
            "falling back to WDA_MONITOR.",
            ver.build,
        )
        return False
    except Exception:  # pylint: disable=broad-except
        return False


# ---------------------------------------------------------------------------
# Application orchestrator
# ---------------------------------------------------------------------------


class GhostMicApp:
    """Top-level application controller.

    Wires together the Qt application, UI components, background threads,
    hotkeys, and system tray.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._config_path = args.config
        self._config = _load_config(self._config_path)
        self._logger = get_logger("ghostmic.app")

        # Lazily-imported heavy dependencies
        self._sys_audio_thread = None
        self._mic_thread = None
        self._vad_thread = None
        self._transcription_thread = None
        self._ai_thread = None
        self._model_loader = None
        self._hotkey_manager = None
        self._tray = None
        self._window = None
        self._transcript_history: List = []

    def run(self) -> int:
        """Initialise everything and start the Qt event loop."""
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt

        # High-DPI
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        app = QApplication(sys.argv)
        app.setApplicationName("GhostMic")
        app.setQuitOnLastWindowClosed(False)

        from ghostmic.ui.styles import MAIN_STYLE
        app.setStyleSheet(MAIN_STYLE)

        self._setup_main_window()
        self._setup_system_tray(app)
        self._start_model_loader()
        self._start_audio_threads()
        self._setup_hotkeys()
        self._test_api_startup()  # Test API connectivity and display result

        if not self._args.minimized and self._window:
            self._window.show()

        app.aboutToQuit.connect(self._on_quit)

        return app.exec()

    # ------------------------------------------------------------------
    # Main window
    # ------------------------------------------------------------------

    def _setup_main_window(self) -> None:
        from ghostmic.ui.main_window import MainWindow

        self._window = MainWindow(self._config)
        # Wire controls
        self._window.controls.record_toggled.connect(self._on_record_toggled)
        self._window.controls.settings_requested.connect(self._on_settings_requested)

    def _test_api_startup(self) -> None:
        """Test API connectivity on startup and display result."""
        from ghostmic.core.ai_engine import AIThread
        
        if not self._window:
            return
        
        # Create temporary AI thread for testing
        ai_test = AIThread(self._config.get("ai", {}))
        success, backend, message = ai_test.test_api_connectivity()
        
        # Update status indicator
        if success:
            self._window.set_api_status(True, backend.title())
            # Display startup test response
            self._window.ai_panel.start_response()
            self._window.ai_panel.finish_response(message)
            self._logger.info("Startup API test successful: %s", backend)
        else:
            self._window.set_api_status(False)
            # Display error in AI panel
            self._window.ai_panel.show_error(f"API Connection Failed: {message}")
            self._logger.warning("Startup API test failed: %s", message)

    # ------------------------------------------------------------------
    # System tray
    # ------------------------------------------------------------------

    def _setup_system_tray(self, app) -> None:
        from ghostmic.ui.system_tray import SystemTrayIcon

        self._tray = SystemTrayIcon()
        self._tray.show_hide_requested.connect(self._toggle_window)
        self._tray.start_stop_requested.connect(self._toggle_recording)
        self._tray.mode_changed.connect(self._on_mode_changed)
        self._tray.quit_requested.connect(app.quit)

    # ------------------------------------------------------------------
    # Whisper model loading
    # ------------------------------------------------------------------

    def _start_model_loader(self) -> None:
        try:
            from ghostmic.core.transcription_engine import ModelLoader, TranscriptionThread

            tcfg = self._config.get("transcription", {})
            loader = ModelLoader(
                model_size=tcfg.get("model_size", "base.en"),
                compute_type=tcfg.get("compute_type", "int8"),
                device="auto",
            )

            # Set up transcription thread
            self._transcription_thread = TranscriptionThread(
                language=tcfg.get("language", "en"),
                beam_size=tcfg.get("beam_size", 3),
            )
            self._transcription_thread.transcription_ready.connect(
                self._on_transcription_ready
            )
            self._transcription_thread.transcribing.connect(
                self._on_transcribing
            )

            def _on_model_ready():
                self._transcription_thread.set_model(loader.model)
                self._transcription_thread.start()
                if self._window:
                    self._window.controls.set_status("Model ready", "#3fb950")

            def _on_model_error(msg: str):
                self._logger.error("Model load error: %s", msg)
                if self._window:
                    self._window.controls.set_status(
                        f"Model error: {msg[:40]}", "#f85149"
                    )

            loader.model_ready.connect(_on_model_ready)
            loader.model_error.connect(_on_model_error)
            loader.progress.connect(
                lambda msg: self._window.controls.set_status(msg) if self._window else None
            )

            self._model_loader = loader
            loader.start()

        except ImportError as exc:
            self._logger.warning("faster-whisper not available: %s", exc)

    # ------------------------------------------------------------------
    # Audio threads
    # ------------------------------------------------------------------

    def _start_audio_threads(self) -> None:
        try:
            from ghostmic.core.audio_buffer import AudioBuffer
            from ghostmic.core.audio_capture import (
                SystemAudioCaptureThread,
                MicCaptureThread,
            )
            from ghostmic.core.vad import VADThread

            audio_cfg = self._config.get("audio", {})
            self._buffer = AudioBuffer(
                sample_rate=audio_cfg.get("sample_rate", 16_000)
            )

            self._vad_thread = VADThread(self._buffer)
            self._vad_thread.speech_segment_ready.connect(self._on_speech_segment)
            # Don't start yet — wait for user to click Record

            self._sys_audio_thread = SystemAudioCaptureThread(
                self._buffer,
                device_index=audio_cfg.get("loopback_device"),
            )
            self._sys_audio_thread.audio_chunk_ready.connect(
                self._vad_thread.push_chunk
            )

            self._mic_thread = MicCaptureThread(
                self._buffer,
                device_index=audio_cfg.get("input_device"),
            )
            self._mic_thread.audio_chunk_ready.connect(
                self._vad_thread.push_chunk
            )

        except ImportError as exc:
            self._logger.warning("Audio libraries not available: %s", exc)

    def _start_audio_capture(self) -> None:
        if self._vad_thread and not self._vad_thread.isRunning():
            self._vad_thread.start()
        if self._sys_audio_thread and not self._sys_audio_thread.isRunning():
            self._sys_audio_thread.start()
        if self._mic_thread and not self._mic_thread.isRunning():
            self._mic_thread.start()

    def _stop_audio_capture(self) -> None:
        for thread in (self._sys_audio_thread, self._mic_thread):
            if thread and thread.isRunning():
                thread.stop()
                thread.wait(2000)
        if self._vad_thread and self._vad_thread.isRunning():
            self._vad_thread.stop()
            self._vad_thread.wait(2000)

    # ------------------------------------------------------------------
    # AI engine
    # ------------------------------------------------------------------

    def _ensure_ai_thread(self) -> None:
        if self._ai_thread is not None:
            return
        try:
            from ghostmic.core.ai_engine import AIThread

            self._ai_thread = AIThread(self._config.get("ai", {}))
            self._ai_thread.ai_thinking.connect(
                lambda: self._window.ai_panel.show_thinking() if self._window else None
            )
            self._ai_thread.ai_response_chunk.connect(
                lambda c: self._window.ai_panel.append_chunk(c) if self._window else None
            )
            self._ai_thread.ai_response_ready.connect(self._on_ai_response_ready)
            self._ai_thread.ai_error.connect(
                lambda msg: self._window.ai_panel.show_error(msg) if self._window else None
            )
            self._ai_thread.start()
        except ImportError as exc:
            self._logger.warning("AI engine not available: %s", exc)

    # ------------------------------------------------------------------
    # Hotkeys
    # ------------------------------------------------------------------

    def _setup_hotkeys(self) -> None:
        try:
            from ghostmic.utils.hotkeys import HotkeyManager

            callbacks = {
                "toggle_recording": self._toggle_recording,
                "toggle_window": self._toggle_window,
                "generate_response": self._generate_ai_response,
                "copy_response": self._copy_last_response,
                "clear_transcript": self._clear_transcript,
            }
            self._hotkey_manager = HotkeyManager(
                self._config.get("hotkeys", {}), callbacks
            )
            self._hotkey_manager.start()
        except ImportError as exc:
            self._logger.warning("pynput not available: %s", exc)

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_record_toggled(self, recording: bool) -> None:
        if recording:
            self._start_audio_capture()
            self._ensure_ai_thread()
            if self._tray:
                self._tray.set_recording(True)
        else:
            self._stop_audio_capture()
            if self._tray:
                self._tray.set_recording(False)

    def _on_speech_segment(self, audio, source: str) -> None:
        if self._transcription_thread:
            self._transcription_thread.push_segment(audio, source)

    def _on_transcribing(self, source: str) -> None:
        if self._window:
            self._window.transcript_panel.show_pending(source)
            self._window.controls.set_status("Transcribing…", "#58a6ff")

    def _on_transcription_ready(self, segment) -> None:
        from ghostmic.utils.text_processing import clean_text

        segment.text = clean_text(segment.text)
        self._transcript_history.append(segment)
        # Cap history
        if len(self._transcript_history) > 1000:
            self._transcript_history = self._transcript_history[-1000:]

        if self._window:
            self._window.transcript_panel.add_segment(segment)
            self._window.controls.set_status("Listening…", "#3fb950")

        # Auto-trigger AI for speaker segments
        ai_cfg = self._config.get("ai", {})
        if (
            ai_cfg.get("trigger_mode", "auto") == "auto"
            and segment.source == "speaker"
        ):
            self._generate_ai_response()

    def _on_ai_response_ready(self, full_text: str) -> None:
        if self._window:
            self._window.ai_panel.finish_response(full_text)

    def _on_settings_requested(self) -> None:
        from ghostmic.ui.settings_dialog import SettingsDialog

        if self._window:
            dlg = SettingsDialog(self._config, parent=self._window)
            dlg.settings_saved.connect(self._on_settings_saved)
            dlg.exec()

    def _on_settings_saved(self, new_config: dict) -> None:
        self._config = new_config
        _save_config(new_config, self._config_path)
        if self._window:
            self._window.update_config(new_config)
        if self._ai_thread:
            self._ai_thread.update_config(new_config.get("ai", {}))
        if self._hotkey_manager:
            self._hotkey_manager.reload(new_config.get("hotkeys", {}))

    def _on_mode_changed(self, mode: str) -> None:
        self._logger.info("Mode changed to %s", mode)

    # ------------------------------------------------------------------
    # Actions (called by hotkeys or tray)
    # ------------------------------------------------------------------

    def _toggle_window(self) -> None:
        if self._window:
            if self._window.isVisible():
                self._window.hide()
            else:
                self._window.show()
            if self._tray:
                self._tray.set_window_visible(self._window.isVisible())

    def _toggle_recording(self) -> None:
        if self._window:
            recording = not self._window.controls.is_recording
            self._window.controls.set_recording(recording)
            self._on_record_toggled(recording)

    def _generate_ai_response(self) -> None:
        if self._ai_thread and self._transcript_history:
            if self._window:
                self._window.ai_panel.start_response()
            self._ai_thread.request_response(self._transcript_history)

    def _copy_last_response(self) -> None:
        # Copy from the most recent AI response card (best effort)
        if self._window:
            cards = self._window.ai_panel._cards
            if cards:
                from PyQt6.QtWidgets import QApplication
                QApplication.clipboard().setText(cards[-1].get_text())

    def _clear_transcript(self) -> None:
        if self._window:
            self._window.transcript_panel.clear_transcript()
        self._transcript_history.clear()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _on_quit(self) -> None:
        self._logger.info("GhostMic: shutting down …")
        self._stop_audio_capture()

        for thread in (self._transcription_thread, self._ai_thread, self._model_loader):
            if thread and hasattr(thread, "isRunning") and thread.isRunning():
                if hasattr(thread, "stop"):
                    thread.stop()
                thread.wait(3000)

        if self._hotkey_manager:
            self._hotkey_manager.stop()

        # Save transcript
        if self._transcript_history:
            self._save_transcript()

        # Save window position
        if self._window:
            geo = self._window.geometry()
            self._config.setdefault("ui", {})
            self._config["ui"]["window_x"] = geo.x()
            self._config["ui"]["window_y"] = geo.y()
            self._config["ui"]["window_width"] = geo.width()
            self._config["ui"]["window_height"] = geo.height()

        _save_config(self._config, self._config_path)
        self._logger.info("GhostMic: shutdown complete.")

    def _save_transcript(self) -> None:
        log_dir = os.path.join(os.path.expanduser("~"), ".ghostmic")
        os.makedirs(log_dir, exist_ok=True)
        filename = os.path.join(
            log_dir,
            f"transcript_{time.strftime('%Y%m%d_%H%M%S')}.txt",
        )
        try:
            if self._window:
                text = self._window.transcript_panel.get_full_text()
            else:
                lines = []
                for seg in self._transcript_history:
                    label = "Speaker" if seg.source == "speaker" else "You"
                    ts = time.strftime("%H:%M:%S", time.localtime(seg.timestamp))
                    lines.append(f"[{ts}] [{label}]: {seg.text}")
                text = "\n".join(lines)
            with open(filename, "w", encoding="utf-8") as fh:
                fh.write(text)
            self._logger.info("Transcript saved to %s", filename)
        except OSError as exc:
            self._logger.error("Could not save transcript: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    configure_logging(debug=args.debug)
    logger = get_logger("ghostmic.main")

    if sys.platform == "win32":
        supports_capture_exclusion = _check_windows_version()
        logger.info(
            "Windows version check: WDA_EXCLUDEFROMCAPTURE supported = %s",
            supports_capture_exclusion,
        )
    else:
        logger.info("Running on %s — stealth features are Windows-only.", sys.platform)

    app = GhostMicApp(args)
    exit_code = app.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
