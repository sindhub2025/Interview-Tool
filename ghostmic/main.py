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
import threading
import time
from typing import List, Optional, Tuple

# ── Ensure the project root is on sys.path ────────────────────────────
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    _HERE = os.path.join(sys._MEIPASS, "ghostmic")
else:
    _HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if not getattr(sys, "frozen", False) and _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ghostmic.utils.logger import configure_logging, get_log_file_path, get_logger
from ghostmic.services.resume_service import ResumeService
from ghostmic.services.thread_coordinator import ThreadCoordinator
from ghostmic.services.runtime_state import RuntimeStateCache

# Logger is set up in _main() after parsing --debug, but we need it here
# for module-level imports that may log warnings.
_early_logger = get_logger("ghostmic.startup")

def _default_user_config_path() -> str:
    """Return the writable user config path for the current platform."""
    if sys.platform == "win32":
        base_dir = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA")
        if base_dir:
            return os.path.join(base_dir, "GhostMic", "config.json")
    return os.path.join(os.path.expanduser("~"), ".ghostmic", "config.json")


# ── Config paths ──────────────────────────────────────────────────────
BUNDLED_CONFIG_PATH = os.path.join(_HERE, "config.json")
DEFAULT_CONFIG_PATH = (
    _default_user_config_path()
    if getattr(sys, "frozen", False)
    else BUNDLED_CONFIG_PATH
)


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
            "backend": "groq",
            "main_backend": "groq",
            "fallback_backend": "groq",
            "enable_fallback": False,
            "expose_openai_provider": False,
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
            "resume_context_enabled": True,
            "sql_profile_enabled": False,
            "resume_correction_threshold_high": 0.87,
            "resume_correction_threshold_medium": 0.74,
        },
        "audio": {"input_device": None, "loopback_device": None, "sample_rate": 16000},
        "transcription": {
            "model_size": "small.en",
            "compute_type": "int8",
            "language": "en",
            "beam_size": 5,
            "use_context_prompt": False,
            "min_segment_seconds": 0.45,
            "min_segment_rms": 140.0,
            "trim_silence": True,
            "silence_trim_threshold": 220,
            "silence_trim_pad_seconds": 0.08,
            "target_rms": 2200.0,
            "max_gain": 8.0,
            "no_speech_threshold": 0.7,
            "log_prob_threshold": -1.0,
            "compression_ratio_threshold": 2.0,
            "temperature": 0.0,
            "patience": 1.2,
            "repetition_penalty": 1.05,
            "remote_fallback": False,
            "remote_backend": "auto",
            "remote_model_groq": "whisper-large-v3-turbo",
            "remote_model_openai": "gpt-4o-mini-transcribe",
            "auto_skip_local_on_windows_broken": False,
            "known_local_torch_broken": False,
            "known_local_torch_error": "",
            "known_local_torch_marked_at": 0,
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
            "docked": False,
            "dock_height": 56,
            "pre_dock_x": None,
            "pre_dock_y": None,
            "pre_dock_width": None,
            "pre_dock_height": None,
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
        default_path = BUNDLED_CONFIG_PATH
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
        expose_openai = bool(ai_cfg.get("expose_openai_provider", False))
        if ai_cfg.get("backend") == "ollama":
            ai_cfg["backend"] = "openai" if expose_openai else "groq"

        # Groq-only mode unless the feature flag re-exposes OpenAI.
        if not expose_openai:
            ai_cfg["backend"] = "groq"
            ai_cfg["main_backend"] = "groq"
            ai_cfg["fallback_backend"] = "groq"
            ai_cfg["enable_fallback"] = False

    return cfg


def _save_config(config: dict, path: str) -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
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
        self._model_loader_thread = None
        self._hotkey_manager = None
        self._tray = None
        self._window = None
        self._transcript_history: List = []
        self._ai_context_history: List = []
        self._last_ai_response_text: str = ""
        self._last_question_text: str = ""  # tracks most recent speaker question for topic-shift detection
        self._transcript_lock = threading.Lock()
        self._model_ready = False
        self._model_error_message: Optional[str] = None
        self._last_transcription_drop_log = 0.0
        self._recording_active = False
        self._recording_lock = threading.Lock()
        self._dictation_hotkey_guard_until = 0.0
        self._startup_api_worker = None
        self._resume_upload_worker = None
        self._ui_dispatcher = None
        self._audio_backends_prewarmed = False

        # Services
        self._thread_coordinator = ThreadCoordinator()
        runtime_state_path = os.path.join(
            os.path.expanduser("~"), ".ghostmic", ".runtime_state.json"
        )
        self._runtime_state = RuntimeStateCache(runtime_state_path)
        self._resume_service = ResumeService()
        self._resume_profile = self._resume_service.get_profile()

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
        self._test_api_startup_async()  # Test API connectivity in background

        if not self._args.minimized and self._window:
            self._window.show()

        if self._tray and self._window:
            self._tray.set_window_visible(self._window.isVisible())
            self._tray.set_docked(self._window.is_docked)

        app.aboutToQuit.connect(self._on_quit)

        return app.exec()

    # ------------------------------------------------------------------
    # Main window
    # ------------------------------------------------------------------

    def _setup_main_window(self) -> None:
        from ghostmic.ui.main_window import MainWindow
        from PyQt6.QtCore import QObject, pyqtSignal

        class UiActionDispatcher(QObject):
            toggle_recording_requested = pyqtSignal()
            toggle_window_requested = pyqtSignal()
            toggle_dock_requested = pyqtSignal()
            generate_response_requested = pyqtSignal()
            copy_response_requested = pyqtSignal()
            clear_transcript_requested = pyqtSignal()
            win_h_dictation_requested = pyqtSignal()

        self._window = MainWindow(self._config, config_path=self._config_path)
        self._ui_dispatcher = UiActionDispatcher()
        self._ui_dispatcher.toggle_recording_requested.connect(self._toggle_recording)
        self._ui_dispatcher.toggle_window_requested.connect(self._toggle_window)
        self._ui_dispatcher.toggle_dock_requested.connect(self._toggle_dock_window)
        self._ui_dispatcher.generate_response_requested.connect(self._generate_ai_response)
        self._ui_dispatcher.copy_response_requested.connect(self._copy_last_response)
        self._ui_dispatcher.clear_transcript_requested.connect(self._clear_transcript)
        self._ui_dispatcher.win_h_dictation_requested.connect(self._on_win_h_dictation)
        self._window.dock_state_changed.connect(self._on_window_dock_state_changed)
        # Wire controls
        self._window.controls.record_toggled.connect(self._on_record_toggled)
        self._window.controls.settings_requested.connect(self._on_settings_requested)
        self._window.dictation_committed.connect(self._on_dictation_committed)
        self._window.ai_panel.text_prompt_submitted.connect(self._on_ai_text_prompt_submitted)

    def _test_api_startup_async(self) -> None:
        """Test API connectivity in a background thread (non-blocking)."""
        if not self._window:
            return

        from ghostmic.ui.settings_dialog import ApiConnectivityWorker

        self._window.controls.set_status("Testing API…", "#58a6ff")
        worker = ApiConnectivityWorker(self._config.get("ai", {}), parent=None)
        worker.result_ready.connect(self._on_startup_api_test_result)
        worker.finished.connect(lambda: self._cleanup_startup_worker())
        self._startup_api_worker = worker
        worker.start()

    def _on_startup_api_test_result(self, success: bool, backend: str, message: str) -> None:
        if not self._window:
            return
        if success:
            self._window.set_api_status(True, backend.title())
            self._logger.info("Startup API test successful: %s", backend)
        else:
            self._window.set_api_status(False)
            self._window.controls.set_status("API connection failed", "#f85149")
            self._logger.warning("Startup API test failed: %s", message)

    def _cleanup_startup_worker(self) -> None:
        if self._startup_api_worker:
            self._startup_api_worker.deleteLater()
            self._startup_api_worker = None

    # ------------------------------------------------------------------
    # System tray
    # ------------------------------------------------------------------

    def _setup_system_tray(self, app) -> None:
        from ghostmic.ui.system_tray import SystemTrayIcon

        self._tray = SystemTrayIcon()
        self._tray.show_hide_requested.connect(self._toggle_window)
        self._tray.dock_toggle_requested.connect(self._toggle_dock_window)
        self._tray.start_stop_requested.connect(self._toggle_recording)
        self._tray.mode_changed.connect(self._on_mode_changed)
        self._tray.quit_requested.connect(app.quit)
        if self._window:
            self._tray.set_docked(self._window.is_docked)
            self._tray.set_window_visible(self._window.isVisible())

    # ------------------------------------------------------------------
    # Whisper model loading
    # ------------------------------------------------------------------

    @staticmethod
    def _is_torch_dll_failure(msg: str) -> bool:
        text = msg.lower()
        return "winerror 1114" in text and ("c10.dll" in text or "torch" in text)

    def _should_skip_local_model_load(self) -> bool:
        if sys.platform != "win32":
            return False
        tcfg = self._config.get("transcription", {})
        return bool(
            tcfg.get("auto_skip_local_on_windows_broken", True)
            and self._runtime_state.get("known_local_torch_broken", False)
        )

    def _is_model_loader_running(self) -> bool:
        return bool(self._model_loader_thread and self._model_loader_thread.is_alive())

    def _preload_torch_runtime(self) -> None:
        """Import torch on the main thread before background loaders start."""
        try:
            import torch  # type: ignore[import]

            _ = torch
        except Exception as exc:  # pylint: disable=broad-except
            self._logger.debug("Torch preload skipped: %s", exc)

    def _mark_local_model_known_broken(self, msg: str) -> None:
        if sys.platform != "win32" or not self._is_torch_dll_failure(msg):
            return
        self._runtime_state.set("known_local_torch_broken", True, ttl=604800.0)  # 7 days
        self._runtime_state.set("known_local_torch_error", msg[:800], ttl=604800.0)
        self._logger.info("Marked local torch runtime as known-broken for fast startup fallback.")

    def _clear_local_model_known_broken(self) -> None:
        if not self._runtime_state.get("known_local_torch_broken", False):
            return
        self._runtime_state.delete("known_local_torch_broken")
        self._runtime_state.delete("known_local_torch_error")
        self._logger.info("Cleared known-broken torch marker after successful local model load.")

    def _start_model_loader(self) -> None:
        try:
            # Start transcription thread but skip local Whisper model entirely.
            from ghostmic.core.transcription_engine import TranscriptionThread

            tcfg = self._config.get("transcription", {})
            self._model_ready = False
            self._model_error_message = None

            # Create transcription thread which will use remote STT only.
            self._transcription_thread = TranscriptionThread(
                language=tcfg.get("language", "en"),
                beam_size=tcfg.get("beam_size", 5),
                ai_config=self._config.get("ai", {}),
                remote_config=tcfg,
            )
            self._transcription_thread.transcription_ready.connect(
                self._on_transcription_ready
            )
            self._transcription_thread.transcribing.connect(self._on_transcribing)
            self._thread_coordinator.register("transcription", self._transcription_thread)

            # Attempt to enable cloud transcription immediately and mark ready.
            enabled, detail = self._transcription_thread.enable_remote_fallback()
            if enabled:
                self._model_loader = None
                self._model_loader_thread = None
                self._model_ready = True
                self._model_error_message = None
                if not self._transcription_thread.isRunning():
                    self._transcription_thread.start()
                self._logger.info(
                    "Cloud-only STT enabled at startup (%s).", detail
                )
                if self._window:
                    self._window.controls.set_status(
                        "Cloud transcription ready", "#f0883e"
                    )
                return

            # Remote STT not configured/available.
            self._model_ready = False
            self._model_error_message = detail
            self._model_loader = None
            self._model_loader_thread = None
            self._logger.warning("Cloud transcription unavailable at startup: %s", detail)
            if self._window:
                self._window.controls.set_status(
                    "Cloud transcription unavailable. See AI panel/logs.", "#f85149"
                )

        except Exception as exc:  # pylint: disable=broad-except
            self._model_ready = False
            self._model_error_message = str(exc)
            self._logger.warning("Cloud STT startup failed: %s", exc)

    # ------------------------------------------------------------------
    # Audio threads
    # ------------------------------------------------------------------

    def _start_audio_threads(self) -> None:
        """One-time setup: create the shared audio buffer.

        The actual capture and VAD threads are created fresh per
        recording session in ``_create_audio_threads()`` because
        QThread objects cannot be restarted once finished.
        """
        try:
            self._preload_torch_runtime()
            from ghostmic.core.audio_buffer import AudioBuffer
            from ghostmic.core.vad import VADThread

            audio_cfg = self._config.get("audio", {})
            self._buffer = AudioBuffer(
                sample_rate=audio_cfg.get("sample_rate", 16_000)
            )

            # Preload Silero in the background so first Record click is responsive.
            threading.Thread(
                target=VADThread.preload_model,
                name="vad-preload",
                daemon=True,
            ).start()

            # Warm audio imports + device probing off the UI thread so
            # first Record click avoids lazy backend initialization cost.
            threading.Thread(
                target=self._prewarm_audio_backends,
                name="audio-backend-prewarm",
                daemon=True,
            ).start()
        except ImportError as exc:
            self._logger.warning("Audio libraries not available: %s", exc)

    def _prewarm_audio_backends(self) -> None:
        """Warm common audio backend imports and basic device enumeration."""
        if self._audio_backends_prewarmed:
            return
        try:
            from ghostmic.core import audio_capture

            # Trigger import of heavy native bindings during startup idle.
            _ = audio_capture.list_input_devices()
            _ = audio_capture.list_loopback_devices()
            self._audio_backends_prewarmed = True
            self._logger.info("Audio backends prewarmed.")
        except Exception as exc:  # pylint: disable=broad-except
            self._logger.debug("Audio backend prewarm skipped: %s", exc)

    def _create_audio_threads(self) -> bool:
        """Create fresh audio capture and VAD threads for a new recording.

        QThread cannot be restarted after finishing, so we recreate
        the capture and VAD threads each time recording begins.
        """
        try:
            from ghostmic.core.audio_buffer import AudioBuffer
            from ghostmic.core.audio_capture import (
                SystemAudioCaptureThread,
                MicCaptureThread,
            )
            from ghostmic.core.vad import VADThread

            if not hasattr(self, '_buffer') or self._buffer is None:
                audio_cfg = self._config.get("audio", {})
                self._buffer = AudioBuffer(
                    sample_rate=audio_cfg.get("sample_rate", 16_000)
                )

            audio_cfg = self._config.get("audio", {})

            self._vad_thread = VADThread(self._buffer)
            self._vad_thread.speech_segment_ready.connect(self._on_speech_segment)
            self._thread_coordinator.register("vad", self._vad_thread)

            self._sys_audio_thread = SystemAudioCaptureThread(
                self._buffer,
                device_index=audio_cfg.get("loopback_device"),
            )
            self._sys_audio_thread.audio_chunk_ready.connect(
                self._vad_thread.push_chunk
            )
            self._thread_coordinator.register("sys_audio", self._sys_audio_thread)

            self._mic_thread = MicCaptureThread(
                self._buffer,
                device_index=audio_cfg.get("input_device"),
            )
            self._mic_thread.audio_chunk_ready.connect(
                self._vad_thread.push_chunk
            )
            self._thread_coordinator.register("mic", self._mic_thread)

            self._logger.info("Recreated audio/VAD threads for new recording session.")
            return True

        except ImportError as exc:
            self._logger.warning("Audio libraries not available: %s", exc)
            return False

    def _start_audio_capture(self) -> bool:
        if not self._is_transcription_ready():
            return False

        # Recreate threads each time (QThread cannot restart after finishing)
        if not self._create_audio_threads():
            return False

        if self._vad_thread:
            self._vad_thread.start()
        if self._sys_audio_thread:
            self._sys_audio_thread.start()
        if self._mic_thread:
            self._mic_thread.start()
        return True

    def _stop_audio_capture(self) -> None:
        for name in ("sys_audio", "mic"):
            self._thread_coordinator.stop_one(name, timeout_ms=2000)
        self._thread_coordinator.stop_one("vad", timeout_ms=2000)

        # Release references so fresh threads are created on next recording
        self._sys_audio_thread = None
        self._mic_thread = None
        self._vad_thread = None

    # ------------------------------------------------------------------
    # AI engine
    # ------------------------------------------------------------------

    def _ensure_ai_thread(self) -> None:
        if self._ai_thread is not None:
            return
        try:
            from ghostmic.core.ai_engine import AIThread

            self._ai_thread = AIThread(self._build_ai_runtime_config())
            # Use default QueuedConnection (non-blocking) for cross-thread signals
            # This allows the AI thread to emit and continue without waiting
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
            self._thread_coordinator.register("ai", self._ai_thread)
            self._ai_thread.start()
        except ImportError as exc:
            self._logger.warning("AI engine not available: %s", exc)

    def _build_ai_runtime_config(self) -> dict:
        """Return AI config augmented with transient runtime context."""
        ai_config = dict(self._config.get("ai", {}))
        ai_config["resume_profile"] = self._resume_profile
        return ai_config

    def _refresh_ai_runtime_context(self) -> None:
        if self._ai_thread:
            self._ai_thread.update_config(self._build_ai_runtime_config())

    # ------------------------------------------------------------------
    # Hotkeys
    # ------------------------------------------------------------------

    def _setup_hotkeys(self) -> None:
        try:
            from ghostmic.utils.hotkeys import HotkeyManager

            dispatcher = self._ui_dispatcher

            callbacks = {
                "toggle_recording": (
                    (lambda: dispatcher.toggle_recording_requested.emit())
                    if dispatcher
                    else self._toggle_recording
                ),
                "toggle_window": (
                    (lambda: dispatcher.toggle_window_requested.emit())
                    if dispatcher
                    else self._toggle_window
                ),
                "generate_response": (
                    (lambda: dispatcher.generate_response_requested.emit())
                    if dispatcher
                    else self._generate_ai_response
                ),
                "copy_response": (
                    (lambda: dispatcher.copy_response_requested.emit())
                    if dispatcher
                    else self._copy_last_response
                ),
                "clear_transcript": (
                    (lambda: dispatcher.clear_transcript_requested.emit())
                    if dispatcher
                    else self._clear_transcript
                ),
                "win_h_dictation": (
                    (lambda: dispatcher.win_h_dictation_requested.emit())
                    if dispatcher
                    else self._on_win_h_dictation
                ),
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

    def _is_transcription_ready(self) -> bool:
        return bool(
            self._model_ready
            and self._transcription_thread
            and self._transcription_thread.is_ready()
        )

    def _try_enable_cloud_fast_start(self) -> Tuple[bool, str]:
        """Enable remote transcription so recording can start before local model is ready."""
        if not self._transcription_thread:
            return False, "transcription thread unavailable"

        enabled, detail = self._transcription_thread.enable_remote_fallback()
        if not enabled:
            return False, detail

        # Mark transcription backend as ready when cloud STT is active.
        self._model_ready = True

        if not self._transcription_thread.isRunning():
            self._transcription_thread.start()
            self._logger.info(
                "Transcription thread started with startup cloud fallback (%s).",
                detail,
            )

        return True, detail

    def _is_dictation_enabled(self) -> bool:
        dcfg = self._config.get("dictation", {})
        return bool(dcfg.get("enabled", True))

    def _on_record_toggled(self, recording: bool) -> None:
        if recording:
            started_with_cloud_fallback = False
            if self._is_model_loader_running():
                enabled, detail = self._try_enable_cloud_fast_start()
                if enabled:
                    started_with_cloud_fallback = True
                    self._logger.info(
                        "Recording started while local model loads; using cloud transcription (%s).",
                        detail,
                    )
                    if self._window:
                        self._window.controls.set_status(
                            "Starting now with cloud transcription while local model loads…",
                            "#f0883e",
                        )
                else:
                    self._logger.info(
                        "Recording blocked: local model still loading and cloud fallback unavailable (%s).",
                        detail,
                    )
                    self._recording_active = False
                    if self._window:
                        self._window.controls.set_recording(False)
                        self._window.controls.set_status(
                            "Model is still loading and cloud fallback is unavailable.",
                            "#f0883e",
                        )
                        self._window.ai_panel.show_error(
                            "Cannot start recording yet because local model is still loading and "
                            f"cloud transcription fallback is unavailable. Details: {detail}"
                        )
                    if self._tray:
                        self._tray.set_recording(False)
                    return

            if not self._start_audio_capture():
                self._logger.warning(
                    "Recording blocked: transcription model unavailable. last_error=%s",
                    self._model_error_message,
                )
                self._recording_active = False
                if self._window:
                    self._window.controls.set_recording(False)
                    self._window.controls.set_status(
                        "Transcription unavailable. Check AI panel/logs.",
                        "#f85149",
                    )
                    if self._model_error_message:
                        self._window.ai_panel.show_error(
                            "Cannot start recording for transcription until model is ready.\n"
                            f"Reason: {self._model_error_message}"
                        )
                if self._tray:
                    self._tray.set_recording(False)
                return

            self._ensure_ai_thread()
            self._recording_active = True
            if started_with_cloud_fallback and self._window:
                self._window.controls.set_status(
                    "Recording… cloud transcription active",
                    "#f0883e",
                )
            if self._tray:
                self._tray.set_recording(True)
        else:
            self._recording_active = False
            self._stop_audio_capture()
            if self._transcription_thread:
                self._transcription_thread.clear_pending_segments()
            if self._ai_thread:
                self._ai_thread.clear_pending_requests()
            with self._transcript_lock:
                self._ai_context_history.clear()
            if self._tray:
                self._tray.set_recording(False)

    def _on_speech_segment(self, audio, source: str) -> None:
        if not self._transcription_thread:
            now = time.time()
            if now - self._last_transcription_drop_log >= 5.0:
                self._logger.warning(
                    "Dropping speech segment because transcription thread is unavailable."
                )
                self._last_transcription_drop_log = now
            return

        if not self._transcription_thread.is_ready():
            now = time.time()
            if now - self._last_transcription_drop_log >= 5.0:
                self._logger.warning(
                    "Dropping speech segment because transcription backend is not ready."
                )
                self._last_transcription_drop_log = now
            return

        self._transcription_thread.push_segment(audio, source)

    def _on_transcribing(self, source: str) -> None:
        if self._window:
            self._window.transcript_panel.show_pending(source)
            self._window.controls.set_status("Transcribing…", "#58a6ff")

    def _on_transcription_ready(self, segment) -> None:
        self._append_transcript_segment(segment, require_recording=True)

    def _append_transcript_segment(self, segment, require_recording: bool) -> bool:
        if require_recording and not self._recording_active:
            self._logger.debug(
                "Ignoring transcription while recording is off (source=%s).",
                getattr(segment, "source", "unknown"),
            )
            return False

        from ghostmic.utils.text_processing import clean_text

        segment.text = clean_text(getattr(segment, "text", ""))
        if not segment.text:
            self._logger.debug("Transcript segment text was empty after cleaning; ignoring.")
            return False

        with self._transcript_lock:
            self._transcript_history.append(segment)
            self._ai_context_history.append(segment)
            self._logger.info(
                "Transcript segment added: source=%s, text=%r, total_segments=%d",
                getattr(segment, "source", "unknown"),
                segment.text[:50],  # First 50 chars
                len(self._transcript_history),
            )
            if len(self._transcript_history) > 1000:
                self._transcript_history = self._transcript_history[-1000:]

        if self._window:
            self._window.transcript_panel.add_segment(segment)
            self._window.controls.set_status("Listening…", "#3fb950")

        ai_cfg = self._config.get("ai", {})
        if (
            ai_cfg.get("trigger_mode", "auto") == "auto"
            and getattr(segment, "source", "") == "speaker"
        ):
            self._generate_ai_response()

        return True

    def _on_dictation_committed(self, text: str) -> None:
        if not text.strip():
            return

        from ghostmic.core.transcription_engine import TranscriptSegment

        segment = TranscriptSegment(text=text, source="user", confidence=1.0)
        accepted = self._append_transcript_segment(segment, require_recording=False)
        if accepted and self._window:
            self._window.controls.set_status("Dictation captured", "#58a6ff")

    def _on_ai_text_prompt_submitted(self, prompt: str, refine: bool) -> None:
        prompt = prompt.strip()
        if not prompt:
            return

        from ghostmic.core.transcription_engine import TranscriptSegment

        with self._transcript_lock:
            # Treat manual prompt as current question/topic from the speaker side
            # so it can run through existing topic-shift and generation pipeline.
            self._ai_context_history.append(
                TranscriptSegment(
                    text=prompt,
                    source="speaker",
                    confidence=1.0,
                )
            )

        if self._window:
            self._window.controls.set_status("Generating from typed prompt…", "#58a6ff")
            self._window.ai_panel.show_thinking()

        self._generate_ai_response(force_follow_up=refine)

    def _on_ai_response_ready(self, full_text: str) -> None:
        self._logger.info("_on_ai_response_ready called with %d chars", len(full_text))
        with self._transcript_lock:
            self._last_ai_response_text = full_text
            self._ai_context_history.clear()
        if self._window:
            self._window.ai_panel.finish_response(full_text)
            self._window.controls.set_status("✓ Response ready", "#3fb950")
            self._logger.info("AI response displayed successfully (%d chars).", len(full_text))

    def _on_settings_requested(self) -> None:
        from ghostmic.ui.settings_dialog import SettingsDialog

        if self._window:
            dlg = SettingsDialog(
                self._config,
                resume_status=self._resume_service.get_status(),
                parent=self._window,
            )
            dlg.settings_saved.connect(self._on_settings_saved)
            dlg.resume_upload_requested.connect(
                lambda file_path, dialog=dlg: self._on_resume_upload_requested(file_path, dialog)
            )
            dlg.resume_remove_requested.connect(
                lambda dialog=dlg: self._on_resume_remove_requested(dialog)
            )
            dlg.exec()

    def _on_resume_upload_requested(self, file_path: str, dialog=None) -> None:
        # Run resume ingestion on a background thread to avoid freezing the UI.
        if dialog and hasattr(dialog, "set_resume_busy"):
            dialog.set_resume_busy(True, "Processing resume…")

        # Define a small QThread-based worker that emits (status, error_message)
        # when ingestion completes. We define it here to avoid importing
        # PyQt6 at module import time.
        from PyQt6.QtCore import QThread, pyqtSignal

        class ResumeIngestWorker(QThread):
            finished = pyqtSignal(object, str)

            def __init__(self, resume_service: ResumeService, file_path: str, parent=None) -> None:
                super().__init__(parent)
                self._resume_service = resume_service
                self._file_path = file_path

            def run(self) -> None:
                try:
                    status = self._resume_service.ingest_resume(self._file_path)
                    self.finished.emit(status, "")
                except Exception as exc:  # pylint: disable=broad-except
                    self.finished.emit({}, str(exc))

        worker_parent = dialog if dialog is not None else None
        worker = ResumeIngestWorker(self._resume_service, file_path, parent=worker_parent)
        self._resume_upload_worker = worker

        def _on_ingest_finished(status: object, error: str) -> None:
            try:
                if error:
                    # Failure path: report error to UI
                    self._logger.warning("Resume upload failed: %s", error)
                    if self._window:
                        self._window.controls.set_status("Resume upload failed", "#f85149")
                    if dialog and hasattr(dialog, "set_resume_error"):
                        dialog.set_resume_error(error)
                else:
                    # Success path: refresh profile/context and update UI
                    self._resume_profile = self._resume_service.get_profile()
                    self._refresh_ai_runtime_context()
                    if self._window:
                        self._window.controls.set_status("Resume uploaded", "#3fb950")
                    if dialog and hasattr(dialog, "set_resume_status"):
                        dialog.set_resume_status(status, "Resume uploaded successfully.")
            finally:
                if dialog and hasattr(dialog, "set_resume_busy"):
                    dialog.set_resume_busy(False)
                # Ensure worker is cleaned up
                try:
                    worker.deleteLater()
                except Exception:
                    pass
                self._resume_upload_worker = None

        worker.finished.connect(_on_ingest_finished)
        worker.start()

    def _on_resume_remove_requested(self, dialog=None) -> None:
        try:
            self._resume_service.remove_resume()
            self._resume_profile = None
            self._refresh_ai_runtime_context()
            if self._window:
                self._window.controls.set_status("Resume removed", "#f0883e")
            if dialog and hasattr(dialog, "set_resume_status"):
                dialog.set_resume_status(self._resume_service.get_status(), "Resume removed.")
        except Exception as exc:  # pylint: disable=broad-except
            self._logger.warning("Resume removal failed: %s", exc)
            if dialog and hasattr(dialog, "set_resume_error"):
                dialog.set_resume_error(str(exc))
        finally:
            # Clear the busy state set by SettingsDialog._on_resume_remove_clicked
            # (triggered via the `resume_remove_requested` signal) so the
            # UI buttons are re-enabled regardless of success or failure.
            if dialog and hasattr(dialog, "set_resume_busy"):
                dialog.set_resume_busy(False)

    def _on_settings_saved(self, new_config: dict) -> None:
        self._config = new_config
        _save_config(new_config, self._config_path)
        if self._window:
            self._window.update_config(new_config)
        self._refresh_ai_runtime_context()
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

    def _toggle_dock_window(self) -> None:
        if not self._window:
            return
        if not self._window.isVisible():
            self._window.show()
        self._window.toggle_dock_mode()
        if self._tray:
            self._tray.set_docked(self._window.is_docked)
            self._tray.set_window_visible(self._window.isVisible())

    def _on_window_dock_state_changed(self, docked: bool) -> None:
        if self._tray:
            self._tray.set_docked(docked)

    def _toggle_recording(self) -> None:
        if self._window:
            recording = not self._window.controls.is_recording
            self._window.controls.set_recording(recording)
            self._on_record_toggled(recording)

    def _on_win_h_dictation(self) -> None:
        if sys.platform != "win32":
            self._logger.debug("Win+H dictation is only supported on Windows.")
            return
        if not self._is_dictation_enabled() or not self._window:
            return

        now = time.time()
        if now < self._dictation_hotkey_guard_until:
            return
        self._dictation_hotkey_guard_until = now + 0.35

        focused = self._window.focus_dictation_target()
        if focused:
            self._window.controls.set_status("Win+H dictation ready", "#58a6ff")
        else:
            self._logger.warning("Unable to focus dictation target for Win+H capture.")

    def _generate_ai_response(self, force_follow_up: bool = False) -> None:
        from ghostmic.core.ai_engine import AIThread, MAX_CONTEXT_SEGMENTS
        from ghostmic.core.transcription_engine import TranscriptSegment

        with self._transcript_lock:
            transcript_source = self._ai_context_history or self._transcript_history
            if not transcript_source:
                self._logger.debug("AI request skipped: transcript history is empty.")
                if self._window:
                    self._window.controls.set_status("No transcript available for AI", "#f0883e")
                return
            transcript_snapshot = list(transcript_source)[-MAX_CONTEXT_SEGMENTS:]
            if not self._ai_context_history and self._transcript_history:
                self._logger.debug(
                    "AI request: using transcript history fallback because AI context history is empty."
                )
            latest_speaker_text = ""
            for seg in reversed(transcript_snapshot):
                if getattr(seg, "source", "") == "speaker":
                    latest_speaker_text = str(getattr(seg, "text", "")).strip()
                    break

            prev_question_text = self._last_question_text.strip()
            is_follow_up = force_follow_up or AIThread.is_context_dependent_follow_up(
                prev_question_text,
                latest_speaker_text,
            )

            # ── Dynamic topic-shift detection ─────────────────────────────
            # 1. An explicit follow-up phrase ("explain more", "elaborate", …)
            #    is never a topic shift — carry existing answer context.
            # 2. Otherwise use Jaccard word-overlap: <25 % shared content words
            #    → the interviewer moved to a new topic.
            if force_follow_up:
                is_new_topic = False
            else:
                is_new_topic = AIThread.classify_topic_shift(
                    prev_question_text,
                    latest_speaker_text,
                )

            def _snapshot_has_prefix(prefix: str) -> bool:
                return any(
                    str(getattr(seg, "text", "")).startswith(prefix)
                    for seg in transcript_snapshot
                )

            if is_new_topic:
                # Wipe the old answer so it is never silently re-injected.
                self._last_ai_response_text = ""
                self._logger.info(
                    "AI request: topic shift detected — clearing previous answer context. "
                    "prev=%r new=%r",
                    prev_question_text[:60],
                    latest_speaker_text[:60],
                )
            else:
                if (
                    is_follow_up
                    and prev_question_text
                    and latest_speaker_text
                    and prev_question_text.lower() != latest_speaker_text.lower()
                    and not _snapshot_has_prefix("Previous speaker question context:")
                ):
                    transcript_snapshot.insert(
                        0,
                        TranscriptSegment(
                            text=f"Previous speaker question context: {prev_question_text}",
                            source="user",
                            confidence=1.0,
                        ),
                    )
                    self._logger.info(
                        "AI request: injected previous speaker question context for follow-up."
                    )

            if (
                not is_new_topic
                and self._last_ai_response_text
                and (is_follow_up or force_follow_up)
                and not _snapshot_has_prefix("Previous AI answer context:")
            ):
                prior_answer = self._last_ai_response_text.strip()
                if len(prior_answer) > 700:
                    prior_answer = prior_answer[:700] + "..."
                transcript_snapshot.insert(
                    0,
                    TranscriptSegment(
                        text=f"Previous AI answer context: {prior_answer}",
                        source="user",
                        confidence=1.0,
                    ),
                )
                if force_follow_up:
                    self._logger.info(
                        "AI request: forced follow-up mode, previous answer context injected."
                    )
                else:
                    self._logger.info(
                        "AI request: follow-up detected, previous answer context injected."
                    )
            elif not is_new_topic:
                self._logger.info(
                    "AI request: same topic detected, no previous-answer context injection."
                )
            else:
                self._logger.info(
                    "AI request: new topic detected, no follow-up context injection."
                )

            # Update last question for the next comparison.
            if latest_speaker_text:
                self._last_question_text = latest_speaker_text

            self._logger.info(
                "AI request: preparing with %d context segment(s), is_new_topic=%s.",
                len(transcript_snapshot),
                is_new_topic,
            )

        self._ensure_ai_thread()
        if self._ai_thread is None:
            self._logger.warning("AI request skipped: AI thread is unavailable.")
            if self._window:
                self._window.controls.set_status("AI unavailable", "#f85149")
                self._window.ai_panel.show_error("AI engine is unavailable. Check dependencies/API settings.")
            return

        accepted = self._ai_thread.request_response(transcript_snapshot, is_new_topic=is_new_topic)
        if accepted:
            self._logger.info("AI request accepted and queued for generation.")
            if self._window:
                self._window.ai_panel.start_response()
                self._window.controls.set_status("Generating response…", "#58a6ff")
            return

        reason = self._ai_thread.last_reject_reason()
        self._logger.info("AI request was not queued: %s", reason)
        if self._window:
            self._window.controls.set_status(f"AI request skipped ({reason})", "#f0883e")

    def _copy_last_response(self) -> None:
        """Copy the most recent AI response to clipboard with user feedback."""
        if not self._window:
            return
        text = self._window.ai_panel.get_last_response_text()
        if text:
            from PyQt6.QtWidgets import QApplication
            QApplication.clipboard().setText(text)
            self._window.controls.set_status("✓ Copied to clipboard", "#3fb950")
        else:
            self._window.controls.set_status("Nothing to copy", "#f0883e")

    def _clear_transcript(self) -> None:
        """Clear transcript with confirmation dialog."""
        if not self._window:
            with self._transcript_lock:
                self._transcript_history.clear()
                self._ai_context_history.clear()
                self._last_ai_response_text = ""
            return

        from PyQt6.QtWidgets import QMessageBox

        segment_count = len(self._transcript_history)
        if segment_count == 0:
            self._window.controls.set_status("Transcript is already empty", "#f0883e")
            return

        reply = QMessageBox.question(
            self._window,
            "Clear Transcript",
            f"Clear all {segment_count} transcript segment(s)?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._window.transcript_panel.clear_transcript()
            with self._transcript_lock:
                self._transcript_history.clear()
                self._ai_context_history.clear()
                self._last_ai_response_text = ""
            self._window.controls.set_status("Transcript cleared", "#3fb950")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _on_quit(self) -> None:
        self._logger.info("GhostMic: shutting down …")
        self._stop_audio_capture()

        # Graceful shutdown of all registered threads in reverse order
        orphans = self._thread_coordinator.shutdown(timeout_ms=5000)
        if orphans:
            self._logger.warning("Threads that did not stop cleanly: %s", orphans)

        if self._hotkey_manager:
            self._hotkey_manager.stop()

        # Save transcript
        with self._transcript_lock:
            if self._transcript_history:
                self._save_transcript()

        # Save window position and UI state
        if self._window:
            self._config.setdefault("ui", {})
            ui_cfg = self._config["ui"]
            ui_cfg["docked"] = self._window.is_docked

            if self._window.is_docked and self._window.pre_dock_geometry is not None:
                geo = self._window.pre_dock_geometry
            else:
                geo = self._window.geometry()

            ui_cfg["window_x"] = geo.x()
            ui_cfg["window_y"] = geo.y()
            ui_cfg["window_width"] = geo.width()
            ui_cfg["window_height"] = geo.height()
            ui_cfg["compact_mode"] = getattr(
                self._window, "_compact_mode", False
            )

            if self._window.pre_dock_geometry is not None:
                pre_geo = self._window.pre_dock_geometry
                ui_cfg["pre_dock_x"] = pre_geo.x()
                ui_cfg["pre_dock_y"] = pre_geo.y()
                ui_cfg["pre_dock_width"] = pre_geo.width()
                ui_cfg["pre_dock_height"] = pre_geo.height()

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

    # ── Preload torch BEFORE QApplication is created ──────────────
    # On Windows, PyQt6's QApplication alters the DLL search order.
    # If torch is imported *after* QApplication, c10.dll fails to load
    # (WinError 1114).  Importing torch first avoids the conflict
    # because its DLLs are already mapped into the process.
    try:
        import torch  # type: ignore[import]
        _ = torch
        logger.info("Torch preloaded successfully: %s", torch.__version__)
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("Torch preload skipped: %s", exc)

    exit_code = app.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
