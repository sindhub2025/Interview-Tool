"""
GhostMic — Real-time AI meeting/interview assistant.

Entry point: parses CLI args, sets up logging, loads config,
initialises the PyQt6 application, starts background threads,
applies stealth, and enters the event loop.
"""

from __future__ import annotations

import argparse
import atexit
import faulthandler
import json
import os
import platform
import sys
import threading
import time
from typing import Callable, List, Optional, Tuple

# ── Ensure the project root is on sys.path ────────────────────────────
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    _HERE = os.path.join(sys._MEIPASS, "ghostmic")
else:
    _HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if not getattr(sys, "frozen", False) and _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ghostmic.utils.logger import (
    configure_logging,
    get_log_dir,
    get_log_file_path,
    get_logger,
)
from ghostmic.core.ai_engine import DEFAULT_SYSTEM_PROMPT
from ghostmic.services.config_service import redact_api_keys_for_disk
from ghostmic.services.session_context_compactor import SessionContextCompactor
from ghostmic.services.resume_service import ResumeService
from ghostmic.services.session_context_store import SessionContextStore
from ghostmic.services.thread_coordinator import ThreadCoordinator
from ghostmic.services.runtime_state import RuntimeStateCache
from ghostmic.services.transcript_store import TranscriptStore
from ghostmic.services.normalizer_service import NormalizerService
from ghostmic.services.segment_manager import SegmentManager
from ghostmic.services.ai_trigger_service import AITriggerService

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

DIAGNOSTIC_DIR = get_log_dir()
STARTUP_TRACE_FILE = os.path.join(DIAGNOSTIC_DIR, "startup_trace.log")
FAULT_TRACE_FILE = os.path.join(DIAGNOSTIC_DIR, "python_faulthandler.log")

_startup_trace_lock = threading.Lock()
_fault_trace_stream = None


def _write_startup_trace(event: str, **fields: object) -> None:
    """Append a lightweight startup breadcrumb for crash diagnostics."""
    try:
        os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"{timestamp} | {event}"
        if fields:
            kv_pairs = []
            for key, value in fields.items():
                text = str(value).replace("\n", "\\n")
                kv_pairs.append(f"{key}={text}")
            line = f"{line} | {' '.join(kv_pairs)}"
        with _startup_trace_lock:
            with open(STARTUP_TRACE_FILE, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:
        # Startup diagnostics must never block app startup.
        return


def _enable_faulthandler_trace(logger) -> None:
    """Enable Python faulthandler output into a persistent file."""
    global _fault_trace_stream  # noqa: PLW0603

    if _fault_trace_stream is not None:
        return

    try:
        os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
        _fault_trace_stream = open(FAULT_TRACE_FILE, "a", encoding="utf-8")
        _fault_trace_stream.write(
            f"\n=== startup pid={os.getpid()} at "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} ===\n"
        )
        _fault_trace_stream.flush()
        faulthandler.enable(file=_fault_trace_stream, all_threads=True)
        logger.info("Faulthandler enabled: %s", FAULT_TRACE_FILE)
        _write_startup_trace("faulthandler.enabled", path=FAULT_TRACE_FILE)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Could not enable faulthandler: %s", exc)
        _write_startup_trace("faulthandler.enable_failed", error=exc)


def _close_faulthandler_trace() -> None:
    """Best-effort cleanup for the faulthandler output stream."""
    global _fault_trace_stream  # noqa: PLW0603

    if _fault_trace_stream is None:
        return

    try:
        faulthandler.disable()
    except Exception:
        pass

    try:
        _fault_trace_stream.flush()
        _fault_trace_stream.close()
    except Exception:
        pass

    _fault_trace_stream = None


def _install_uncaught_excepthook(logger) -> None:
    """Log uncaught exceptions with a clear breadcrumb."""
    previous_hook = sys.excepthook

    def _hook(exc_type, exc_value, exc_traceback) -> None:  # type: ignore[no-untyped-def]
        _write_startup_trace(
            "uncaught_exception",
            exc_type=getattr(exc_type, "__name__", str(exc_type)),
            message=exc_value,
        )
        logger.critical(
            "Uncaught exception reached sys.excepthook",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        if previous_hook is not None:
            previous_hook(exc_type, exc_value, exc_traceback)

    sys.excepthook = _hook


atexit.register(_close_faulthandler_trace)


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
            "gemini_api_key": "",
            "gemini_model": "gemini-3-flash-preview",
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "temperature": 0.7,
            "trigger_mode": "auto",
            "auto_speaker_analysis_enabled": True,
            "auto_speaker_silence_seconds": 1.2,
            "auto_speaker_min_words": 4,
            "auto_speaker_min_chars": 18,
            "auto_speaker_retrigger_cooldown_seconds": 5.0,
            "context_segments": 10,
            "session_context": "",
            "resume_context_enabled": True,
            "sql_profile_enabled": False,
            "resume_correction_threshold_high": 0.87,
            "resume_correction_threshold_medium": 0.74,
            "context_compaction_enabled": True,
            "context_compaction_min_interval_seconds": 30,
            "context_compaction_max_interval_seconds": 45,
            "context_compaction_full_refresh_every": 5,
            "context_compaction_source_max_events": 80,
            "context_compaction_source_max_chars": 7000,
            "context_compaction_timeout": 20.0,
            "context_compaction_temperature": 0.1,
            "context_compaction_max_tokens": 320,
        },
        "audio": {
            "input_device": None,
            "loopback_device": None,
            "sample_rate": 16000,
            "capture_mic": False,
        },
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
            "streaming_normalization_enabled": True,
            "streaming_processing_interval_ms": 3500,
            "streaming_buffer_seconds": 30.0,
            "streaming_buffer_max_chunks": 260,
            "streaming_pause_boundary_seconds": 1.15,
            "streaming_soft_flush_seconds": 10.0,
            "streaming_soft_flush_chunks": 8,
        },
        "ui": {
            "opacity": 0.95,
            "font_size": 11,
            "window_width": 420,
            "window_height": 650,
            "window_x": None,
            "window_y": None,
            "always_on_top": True,
            "stealth_enabled": True,
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


INITIAL_RECORDING_QUESTION_NORMALIZATION_DELAY_MS = 9_000
STREAMING_SEGMENT_LOOP_INTERVAL_MS = 2_500


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
        cfg: dict
        frozen = bool(getattr(sys, "frozen", False))
        if frozen:
            # Never seed packaged builds from bundled developer config.
            cfg = _default_config()
        else:
            default_path = BUNDLED_CONFIG_PATH
            if os.path.exists(default_path):
                with open(default_path, encoding="utf-8") as fh:
                    cfg = json.load(fh)
            else:
                cfg = _default_config()

        should_persist = not (not frozen and path == BUNDLED_CONFIG_PATH)
        if should_persist:
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

        def _sanitize_backend(raw_value: object, allowed: set[str], default: str) -> str:
            value = str(raw_value or "").strip().lower()
            if value in allowed:
                return value
            return default

        allowed = {"groq", "gemini"}
        if expose_openai:
            allowed.add("openai")

        requested_backend = ai_cfg.get("backend", "groq")
        requested_main = ai_cfg.get("main_backend", requested_backend)
        active_backend = _sanitize_backend(requested_main, allowed, "groq")
        ai_cfg["backend"] = _sanitize_backend(requested_backend, allowed, active_backend)
        ai_cfg["main_backend"] = active_backend

        if expose_openai:
            requested_fallback = ai_cfg.get("fallback_backend", active_backend)
            ai_cfg["fallback_backend"] = _sanitize_backend(requested_fallback, allowed, active_backend)
        else:
            # Keep fallback disabled in standard mode; only OpenAI remains feature-gated.
            ai_cfg["fallback_backend"] = active_backend
            ai_cfg["enable_fallback"] = False

    return cfg


def _save_config(config: dict, path: str) -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(redact_api_keys_for_disk(config), fh, indent=2)
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
        self._active_screen_summary_text: str = ""
        self._active_screen_anchor_question: str = ""
        self._transcript_lock = threading.Lock()
        self._model_ready = False
        self._model_error_message: Optional[str] = None
        self._last_transcription_drop_log = 0.0
        self._recording_active = False
        self._mic_recording_active = False
        self._recording_lock = threading.Lock()
        self._dictation_hotkey_guard_until = 0.0
        self._startup_api_worker = None
        self._screen_analysis_worker = None
        self._question_normalization_worker = None
        self._follow_up_suggestion_worker = None
        self._resume_upload_worker = None
        self._ui_dispatcher = None
        self._audio_backends_prewarmed = False
        self._mic_prime_in_progress = False
        self._mic_device_fallback_attempted = False
        self._mic_recovery_pending = False
        self._mic_recovery_generation = 0
        self._initial_recording_question_pending = False
        self._initial_recording_question_consumed = False
        self._initial_recording_question_generation = 0
        self._auto_speaker_silence_generation = 0
        self._auto_speaker_last_signature = ""
        self._auto_speaker_last_trigger_ts = 0.0
        self._auto_primary_question_sent = False
        self._active_primary_question_text = ""
        self._queued_normalized_questions: List[dict] = []
        self._streaming_segment_timer = None
        self._transcript_store: TranscriptStore | None = None
        self._normalizer_service: NormalizerService | None = None
        self._segment_manager: SegmentManager | None = None
        self._ai_trigger_service: AITriggerService | None = None
        self._normalized_segment_items: list[dict] = []
        self._normalized_segment_lookup: dict[str, dict] = {}
        # Queue of (segment_id, raw_text) tuples awaiting background AI normalization.
        self._pending_normalization_queue: list[tuple[str, str]] = []


        # Services
        self._thread_coordinator = ThreadCoordinator()
        runtime_state_path = os.path.join(
            os.path.expanduser("~"), ".ghostmic", ".runtime_state.json"
        )
        self._runtime_state = RuntimeStateCache(runtime_state_path)
        self._session_context_store = SessionContextStore()
        self._logger.info("Session context file created: %s", self._session_context_store.path)
        self._session_context_compactor = SessionContextCompactor(
            self._session_context_store,
            self._config.get("ai", {}),
        )
        self._resume_service = ResumeService()
        self._resume_profile = self._resume_service.get_profile()
        self._ensure_streaming_pipeline_initialized()

    def run(self) -> int:
        """Initialise everything and start the Qt event loop."""
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt

        startup_step = "begin"
        self._logger.info(
            "Startup sequence begin: minimized=%s config_path=%s",
            self._args.minimized,
            self._config_path,
        )
        _write_startup_trace(
            "run.begin",
            minimized=self._args.minimized,
            config=self._config_path,
        )

        try:
            # High-DPI
            startup_step = "set_high_dpi_policy"
            QApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
            )
            _write_startup_trace("run.high_dpi_policy.ok")

            startup_step = "create_qapplication"
            app = QApplication(sys.argv)
            app.setApplicationName("GhostMic")
            app.setQuitOnLastWindowClosed(False)
            app.setOverrideCursor(Qt.CursorShape.ArrowCursor)
            _write_startup_trace("run.qapplication.ok")

            startup_step = "apply_stylesheet"
            from ghostmic.ui.styles import MAIN_STYLE

            app.setStyleSheet(MAIN_STYLE)
            _write_startup_trace("run.stylesheet.ok")

            startup_step = "setup_main_window"
            self._setup_main_window()
            _write_startup_trace("run.setup_main_window.ok")

            startup_step = "setup_system_tray"
            self._setup_system_tray(app)
            _write_startup_trace("run.setup_system_tray.ok")

            startup_step = "start_model_loader"
            self._start_model_loader()
            _write_startup_trace("run.start_model_loader.ok")

            startup_step = "start_audio_threads"
            self._start_audio_threads()
            _write_startup_trace("run.start_audio_threads.ok")

            startup_step = "sync_startup_mic_state"
            self._sync_startup_mic_state()
            _write_startup_trace(
                "run.sync_startup_mic_state.ok",
                capture_mic=self._is_mic_capture_enabled(),
            )

            startup_step = "setup_hotkeys"
            self._setup_hotkeys()
            _write_startup_trace("run.setup_hotkeys.ok")

            startup_step = "test_api_startup_async"
            self._test_api_startup_async()  # Test API connectivity in background
            _write_startup_trace("run.test_api_startup_async.ok")

            startup_step = "start_session_context_compactor"
            self._session_context_compactor.start()
            _write_startup_trace("run.session_context_compactor.ok")

            startup_step = "show_window"
            if not self._args.minimized and self._window:
                self._window.show()
            _write_startup_trace(
                "run.show_window.ok",
                shown=bool(not self._args.minimized and self._window),
            )

            startup_step = "sync_tray_state"
            if self._tray and self._window:
                self._tray.set_window_visible(self._window.isVisible())
                self._tray.set_docked(self._window.is_docked)
            _write_startup_trace("run.sync_tray_state.ok")

            app.aboutToQuit.connect(self._on_quit)

            startup_step = "event_loop_exec"
            _write_startup_trace("run.event_loop.begin")
            exit_code = app.exec()
            self._logger.info("Qt event loop exited with code %s", exit_code)
            _write_startup_trace("run.event_loop.end", exit_code=exit_code)
            return exit_code

        except Exception as exc:  # pylint: disable=broad-except
            self._logger.exception("Startup failed at step: %s", startup_step)
            _write_startup_trace("run.failed", step=startup_step, error=exc)
            raise

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
        self._window.controls.mic_toggled.connect(self._on_mic_toggled)
        self._window.controls.settings_requested.connect(self._on_settings_requested)
        self._window.controls.screenshot_requested.connect(self._on_screen_analysis_requested)
        self._window.controls.set_mic_enabled(self._is_mic_capture_enabled())
        self._window.dictation_committed.connect(self._on_dictation_committed)
        self._window.ai_panel.text_prompt_submitted.connect(self._on_ai_text_prompt_submitted)
        self._window.suggested_follow_up_selected.connect(
            self._on_suggested_follow_up_selected
        )
        self._window.queued_question_send_requested.connect(
            self._on_queued_question_send_requested
        )
        if hasattr(self._window, "normalized_segment_send_requested"):
            self._window.normalized_segment_send_requested.connect(
                self._on_normalized_segment_send_requested
            )
        self._window.transcript_panel.speaker_text_edited.connect(self._on_speaker_transcript_edited)
        self._window.transcript_panel.speaker_normalize_requested.connect(
            self._on_speaker_question_normalize_requested
        )
        self._window.transcript_panel.speaker_send_requested.connect(
            self._on_speaker_send_requested
        )

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

    def _cleanup_screen_analysis_worker(self) -> None:
        if self._screen_analysis_worker:
            self._screen_analysis_worker.deleteLater()
            self._screen_analysis_worker = None

    def _cleanup_question_normalization_worker(self) -> None:
        if self._question_normalization_worker:
            self._question_normalization_worker.deleteLater()
            self._question_normalization_worker = None
        # Process the next pending streaming segment normalization, if any.
        self._drain_streaming_segment_normalization_queue()


    def _cleanup_follow_up_suggestion_worker(self) -> None:
        if self._follow_up_suggestion_worker:
            self._follow_up_suggestion_worker.deleteLater()
            self._follow_up_suggestion_worker = None

    # ------------------------------------------------------------------
    # Fix 1 & 5: Streaming segment AI normalization queue
    # All question segments (not just the first) are AI-cleaned before
    # they appear in the Normalized Segments panel.
    # ------------------------------------------------------------------

    def _enqueue_streaming_segment_normalization(self, segment_id: str, raw_text: str) -> None:
        """Add a streaming segment to the background AI normalization queue."""
        queue = getattr(self, "_pending_normalization_queue", None)
        if queue is None:
            self._pending_normalization_queue = []
            queue = self._pending_normalization_queue

        # Avoid double-queuing the same segment
        for queued_id, _ in queue:
            if queued_id == segment_id:
                return

        queue.append((segment_id, raw_text))
        self._logger.debug(
            "Queued streaming segment for AI normalization: %s (%d in queue)",
            segment_id,
            len(queue),
        )
        self._drain_streaming_segment_normalization_queue()

    def _drain_streaming_segment_normalization_queue(self) -> None:
        """Start a background AI normalization for the next queued segment, if the worker is free."""
        if getattr(self, "_question_normalization_worker", None) is not None:
            return  # Worker busy — will be called again from _cleanup_question_normalization_worker


        queue = getattr(self, "_pending_normalization_queue", [])
        if not queue:
            return

        segment_id, raw_text = queue.pop(0)

        # Safety: skip if segment no longer exists in the lookup
        lookup = getattr(self, "_normalized_segment_lookup", {})
        if segment_id not in lookup:
            # Try the next one
            self._drain_streaming_segment_normalization_queue()
            return

        try:
            from ghostmic.services.question_normalization_service import (
                QuestionNormalizationWorker,
            )

            worker = QuestionNormalizationWorker(
                self._build_ai_runtime_config(),
                raw_text,
                parent=None,
            )
            if hasattr(worker, "normalized_with_followups_ready"):
                worker.normalized_with_followups_ready.connect(
                    lambda normalized, follow_ups, sid=segment_id: self._on_streaming_segment_ai_normalized(
                        sid, normalized, list(follow_ups or [])
                    )
                )
            else:
                worker.normalized_ready.connect(
                    lambda normalized, sid=segment_id: self._on_streaming_segment_ai_normalized(
                        sid, normalized, []
                    )
                )
            worker.normalization_error.connect(
                lambda message, sid=segment_id: self._on_streaming_segment_ai_normalization_error(
                    sid, message
                )
            )
            worker.finished.connect(self._cleanup_question_normalization_worker)
            self._question_normalization_worker = worker
            self._logger.debug("Starting AI normalization worker for segment %s", segment_id)
            worker.start()
        except Exception as exc:  # pylint: disable=broad-except
            self._logger.warning(
                "Could not start streaming segment AI normalization for %s: %s",
                segment_id,
                exc,
            )
            # Still drain the queue even if this one failed
            self._drain_streaming_segment_normalization_queue()

    def _on_streaming_segment_ai_normalized(
        self,
        segment_id: str,
        normalized_text: str,
        follow_up_questions: List[str],
    ) -> None:
        """Called when background AI normalization of a streaming segment succeeds."""
        from ghostmic.utils.text_processing import clean_text, ensure_question_format

        normalized = ensure_question_format(clean_text(str(normalized_text or "")))
        if not normalized:
            self._logger.debug(
                "AI normalization for segment %s returned empty; keeping original text.",
                segment_id,
            )
            return

        item = self._update_normalized_segment_item(segment_id, text=normalized)
        if item is None:
            return  # Segment was evicted from the lookup; nothing to update

        self._logger.info(
            "Streaming segment %s AI-normalized: %r",
            segment_id,
            normalized[:80],
        )

        # Update follow-up suggestions on the item for later use
        cleaned_follow_ups: List[str] = []
        seen: set[str] = set()
        for q in list(follow_up_questions or []):
            candidate = clean_text(str(q or ""))
            if not candidate:
                continue
            if not candidate.endswith("?"):
                candidate = f"{candidate.rstrip('.!')}?"
            canonical = candidate.rstrip(".?!").lower()
            if canonical in seen:
                continue
            seen.add(canonical)
            cleaned_follow_ups.append(candidate)
            if len(cleaned_follow_ups) >= 3:
                break
        if cleaned_follow_ups:
            item["follow_up_questions"] = cleaned_follow_ups

        self._sync_normalized_segments_to_window()

    def _on_streaming_segment_ai_normalization_error(
        self,
        segment_id: str,
        message: str,
    ) -> None:
        """Called when AI normalization of a streaming segment fails — keep original text."""
        self._logger.warning(
            "AI normalization failed for streaming segment %s: %s", segment_id, message
        )
        # Original text stays in place — no UI update needed, segment remains usable.

    def _auto_send_next_pending_segment(self) -> None:
        """Fix 5: After an AI response completes, auto-clean the next pending segment.

        This keeps the Normalized Segments panel continuously fresh without
        requiring manual 'Send to AI' clicks for every new question.
        """
        if not self._is_streaming_normalization_enabled():
            return

        lookup = getattr(self, "_normalized_segment_lookup", {})
        items = getattr(self, "_normalized_segment_items", [])
        for item in items:
            if str(item.get("status", "pending")).lower() != "pending":
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            from ghostmic.services.ai_trigger_service import is_question_like
            if not is_question_like(text):
                continue
            seg_id = str(item.get("segment_id", "")).strip()
            if not seg_id or seg_id not in lookup:
                continue
            # Only re-queue for normalization if it hasn't been cleaned yet by AI
            # (i.e. text still looks like raw transcript).  We check by seeing
            # whether a queued normalization entry already exists.
            queue = getattr(self, "_pending_normalization_queue", [])
            already_queued = any(qid == seg_id for qid, _ in queue)
            if not already_queued:
                self._enqueue_streaming_segment_normalization(seg_id, text)
            break  # Process one at a time — subsequent ones will be drained in order

    @staticmethod
    def _build_local_follow_up_suggestions(question: str) -> List[str]:
        stem = " ".join(str(question or "").split()).strip().rstrip(".?!")
        if not stem:
            return [
                "Can you walk me through a real project where you handled this end-to-end?",
                "What trade-offs did you evaluate before choosing your approach?",
                "How did you monitor outcomes and respond when issues appeared in production?",
            ]

        return [
            "Can you walk me through a real production example where this was critical?",
            f"What constraints or trade-offs did you consider when solving: {stem}?",
            "How would you measure success and troubleshoot this if it failed after release?",
        ]

    def _start_follow_up_suggestion_refresh(self, question: str) -> None:
        if not getattr(self, "_window", None):
            return
        if getattr(self, "_follow_up_suggestion_worker", None) is not None:
            return

        try:
            from ghostmic.services.question_normalization_service import (
                QuestionNormalizationWorker,
            )

            worker = QuestionNormalizationWorker(
                self._build_ai_runtime_config(),
                question,
                parent=None,
            )
            if hasattr(worker, "normalized_with_followups_ready"):
                worker.normalized_with_followups_ready.connect(
                    lambda normalized, follow_ups, base=question: self._on_follow_up_suggestion_refresh_ready(
                        base,
                        normalized,
                        list(follow_ups or []),
                    )
                )
            else:
                worker.normalized_ready.connect(
                    lambda normalized, base=question: self._on_follow_up_suggestion_refresh_ready(
                        base,
                        normalized,
                        [],
                    )
                )
            worker.normalization_error.connect(
                lambda message, base=question: self._on_follow_up_suggestion_refresh_error(
                    base,
                    message,
                )
            )
            worker.finished.connect(self._cleanup_follow_up_suggestion_worker)
            self._follow_up_suggestion_worker = worker
            worker.start()
        except Exception as exc:  # pylint: disable=broad-except
            self._logger.warning(
                "Could not start follow-up suggestion refresh: %s",
                exc,
            )

    def _on_follow_up_suggestion_refresh_ready(
        self,
        base_question: str,
        normalized_text: str,
        follow_up_questions: List[str],
    ) -> None:
        from ghostmic.utils.text_processing import clean_text, ensure_question_format

        if not self._window:
            return

        normalized = ensure_question_format(clean_text(str(normalized_text or "")))
        display_question = normalized or ensure_question_format(clean_text(str(base_question or "")))
        if display_question:
            self._window.set_current_question_text(display_question)

        cleaned_follow_ups: List[str] = []
        seen: set[str] = set()
        for question in list(follow_up_questions or []):
            candidate = clean_text(str(question or ""))
            if not candidate:
                continue
            if not candidate.endswith("?"):
                candidate = f"{candidate.rstrip('.!')}?"
            canonical = candidate.rstrip(".?!").lower()
            if canonical in seen:
                continue
            seen.add(canonical)
            cleaned_follow_ups.append(candidate)
            if len(cleaned_follow_ups) >= 3:
                break

        if cleaned_follow_ups:
            self._window.set_question_follow_up_suggestions(cleaned_follow_ups)
            self._window.show_follow_up_sent_confirmation(base_question)

    def _on_follow_up_suggestion_refresh_error(
        self,
        base_question: str,
        message: str,
    ) -> None:
        self._logger.warning(
            "Follow-up suggestion refresh failed for %r: %s",
            base_question,
            message,
        )

    # ------------------------------------------------------------------
    # System tray
    # ------------------------------------------------------------------

    def _setup_system_tray(self, app) -> None:
        from ghostmic.ui.system_tray import SystemTrayIcon

        self._tray = SystemTrayIcon()
        self._tray.show_hide_requested.connect(self._toggle_window)
        self._tray.dock_toggle_requested.connect(self._toggle_dock_window)
        self._tray.stealth_toggled.connect(self._on_stealth_toggled)
        self._tray.start_stop_requested.connect(self._toggle_recording)
        self._tray.mode_changed.connect(self._on_mode_changed)
        self._tray.quit_requested.connect(app.quit)
        self._tray.set_stealth_enabled(self._is_stealth_enabled())
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
            self._logger.info("Audio startup: initialising shared audio components.")
            _write_startup_trace("audio.start.begin")
            from ghostmic.core.audio_buffer import AudioBuffer
            from ghostmic.core.vad import VADThread

            audio_cfg = self._config.get("audio", {})
            self._buffer = AudioBuffer(
                sample_rate=audio_cfg.get("sample_rate", 16_000)
            )
            _write_startup_trace(
                "audio.buffer.initialized",
                sample_rate=audio_cfg.get("sample_rate", 16_000),
            )

            preload_vad = os.environ.get("GHOSTMIC_PRELOAD_VAD", "").strip() == "1"
            if preload_vad:
                # Optional warm-up for local VAD (off by default for startup stability).
                threading.Thread(
                    target=VADThread.preload_model,
                    name="vad-preload",
                    daemon=True,
                ).start()
                _write_startup_trace("audio.vad_preload.started")
            else:
                _write_startup_trace("audio.vad_preload.skipped")

            # Warm audio imports + device probing off the UI thread so
            # first Record click avoids lazy backend initialization cost.
            threading.Thread(
                target=self._prewarm_audio_backends,
                name="audio-backend-prewarm",
                daemon=True,
            ).start()
            self._logger.info(
                "Audio startup: launched preload workers (%s).",
                "vad-preload, audio-backend-prewarm"
                if preload_vad
                else "audio-backend-prewarm",
            )
            _write_startup_trace("audio.start.ok", preload_vad=preload_vad)
        except ImportError as exc:
            self._logger.warning("Audio libraries not available: %s", exc)
            _write_startup_trace("audio.start.import_error", error=exc)
        except Exception as exc:  # pylint: disable=broad-except
            self._logger.exception("Audio startup failed unexpectedly.")
            _write_startup_trace("audio.start.failed", error=exc)

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

    def _is_mic_capture_enabled(self) -> bool:
        audio_cfg = self._config.get("audio", {})
        return bool(audio_cfg.get("capture_mic", False))

    def _prime_mic_capture_async(self) -> None:
        """Warm microphone backend/device so toggle-on feels immediate."""
        if getattr(self, "_mic_prime_in_progress", False):
            return

        self._mic_prime_in_progress = True

        def _worker() -> None:
            detail = ""
            ok = False
            try:
                from ghostmic.core.audio_capture import prime_input_device

                audio_cfg = self._config.get("audio", {})
                ok, detail = prime_input_device(
                    device_index=audio_cfg.get("input_device"),
                    sample_rate=int(audio_cfg.get("sample_rate", 16_000)),
                )
            except Exception as exc:  # pylint: disable=broad-except
                detail = str(exc)
                ok = False
            finally:
                self._mic_prime_in_progress = False

            if ok:
                self._logger.info("Microphone primed for fast start.")
            else:
                self._logger.warning("Microphone prime failed: %s", detail)

        threading.Thread(
            target=_worker,
            name="mic-prime",
            daemon=True,
        ).start()

    def _sync_startup_mic_state(self) -> None:
        """Prime microphone capture during startup when mic mode is enabled."""
        if not self._is_mic_capture_enabled():
            return

        self._logger.info("Startup mic capture is enabled; priming microphone backend.")
        self._prime_mic_capture_async()

    def _enable_mic_capture_live(self) -> bool:
        """Enable microphone capture without restarting speaker/VAD threads."""
        if self._mic_thread and hasattr(self._mic_thread, "isRunning"):
            try:
                if self._mic_thread.isRunning():
                    return True
            except Exception:  # pylint: disable=broad-except
                pass

        if not self._vad_thread:
            return False

        try:
            from ghostmic.core.audio_capture import MicCaptureThread

            audio_cfg = self._config.get("audio", {})
            self._mic_thread = MicCaptureThread(
                self._buffer,
                device_index=audio_cfg.get("input_device"),
            )
            self._mic_thread.audio_chunk_ready.connect(self._vad_thread.push_chunk)
            if hasattr(self._mic_thread, "mic_capture_failed"):
                self._mic_thread.mic_capture_failed.connect(self._on_mic_capture_failed)
            self._thread_coordinator.register("mic", self._mic_thread)
            self._mic_thread.start()
            self._logger.info("Microphone capture enabled live during recording.")
            return True
        except ImportError as exc:
            self._logger.warning("Microphone capture unavailable: %s", exc)
            return False

    def _disable_mic_capture_live(self) -> None:
        """Disable microphone capture without touching speaker/VAD threads."""
        self._thread_coordinator.stop_one("mic", timeout_ms=2000)
        self._mic_thread = None
        self._logger.info("Microphone capture disabled live during recording.")

    def _cancel_pending_mic_recovery(self) -> None:
        """Invalidate pending delayed mic recovery callbacks."""
        self._mic_recovery_pending = False
        self._mic_recovery_generation = int(
            getattr(self, "_mic_recovery_generation", 0)
        ) + 1

    @staticmethod
    def _is_transient_mic_failure(detail: str) -> bool:
        text = str(detail or "").strip().lower()
        if not text:
            return False

        transient_markers = (
            "overflow",
            "input overflow",
            "device busy",
            "resource busy",
            "temporarily unavailable",
            "timed out",
            "timeout",
            "unanticipated host error",
            "stream closed",
        )
        return any(marker in text for marker in transient_markers)

    def _schedule_mic_live_restart(self, detail: str, *, delay_ms: int = 700) -> bool:
        if getattr(self, "_mic_recovery_pending", False):
            return True

        try:
            from PyQt6.QtCore import QTimer
        except Exception as exc:  # pylint: disable=broad-except
            self._logger.debug("Mic recovery timer unavailable: %s", exc)
            return False

        generation = int(getattr(self, "_mic_recovery_generation", 0)) + 1
        self._mic_recovery_generation = generation
        self._mic_recovery_pending = True

        if self._window:
            self._window.controls.set_status(
                "Microphone hiccup detected. Retrying...",
                "#f0883e",
            )

        self._logger.info(
            "Scheduling microphone restart in %dms after transient failure (%s).",
            delay_ms,
            detail,
        )

        QTimer.singleShot(
            max(120, int(delay_ms)),
            lambda token=generation, error_detail=str(detail): self._run_mic_live_restart(
                token,
                error_detail,
            ),
        )
        return True

    def _run_mic_live_restart(self, generation: int, detail: str) -> None:
        if generation != int(getattr(self, "_mic_recovery_generation", 0)):
            return

        self._mic_recovery_pending = False
        if not self._is_any_recording_active() or not self._is_mic_capture_enabled():
            return

        self._disable_mic_capture_live()
        if self._enable_mic_capture_live():
            if self._window:
                self._window.controls.set_status(
                    "Microphone recovered after transient error",
                    "#3fb950",
                )
            return

        # The delayed restart already retried the default device once.
        audio_cfg = self._config.setdefault("audio", {})
        if audio_cfg.get("input_device") is None:
            self._mic_device_fallback_attempted = True

        self._on_mic_capture_failed(detail, allow_transient_retry=False)

    def _on_mic_capture_failed(
        self,
        detail: str,
        *,
        allow_transient_retry: bool = True,
    ) -> None:
        """Recover from runtime mic failures and keep UI/config in sync."""
        if not self._is_mic_capture_enabled():
            return

        if not self._is_any_recording_active() and self._mic_thread is None:
            # Ignore stale failure notifications that can arrive after shutdown.
            return

        detail_text = str(detail or "unknown microphone error")
        self._logger.warning("Microphone capture failed: %s", detail_text)

        if (
            allow_transient_retry
            and self._is_any_recording_active()
            and self._is_transient_mic_failure(detail_text)
            and self._schedule_mic_live_restart(detail_text, delay_ms=700)
        ):
            return

        audio_cfg = self._config.setdefault("audio", {})
        configured_input = audio_cfg.get("input_device")

        def _retry_default_input_device() -> bool:
            if self._window:
                self._window.controls.set_status(
                    "Microphone failed. Retrying default input device...",
                    "#f0883e",
                )

            self._disable_mic_capture_live()
            if self._enable_mic_capture_live():
                if self._window:
                    self._window.controls.set_status(
                        "Microphone recovered on default input device",
                        "#3fb950",
                    )
                return True
            return False

        if configured_input is not None and not self._mic_device_fallback_attempted:
            self._mic_device_fallback_attempted = True
            audio_cfg["input_device"] = None
            _save_config(self._config, self._config_path)

            if _retry_default_input_device():
                return

        if configured_input is None and not self._mic_device_fallback_attempted:
            self._mic_device_fallback_attempted = True
            if _retry_default_input_device():
                return

        self._cancel_pending_mic_recovery()
        self._disable_mic_capture_live()
        audio_cfg["capture_mic"] = False
        _save_config(self._config, self._config_path)
        self._mic_device_fallback_attempted = False

        if self._window:
            self._window.controls.set_mic_enabled(False)
            self._window.controls.set_status(
                "Microphone unavailable. Switched to speaker-only mode.",
                "#f0883e",
            )

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
            self._cancel_pending_mic_recovery()
            self._mic_device_fallback_attempted = False

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

            self._mic_thread = None
            if self._is_mic_capture_enabled():
                self._mic_thread = MicCaptureThread(
                    self._buffer,
                    device_index=audio_cfg.get("input_device"),
                )
                self._mic_thread.audio_chunk_ready.connect(
                    self._vad_thread.push_chunk
                )
                if hasattr(self._mic_thread, "mic_capture_failed"):
                    self._mic_thread.mic_capture_failed.connect(self._on_mic_capture_failed)
                self._thread_coordinator.register("mic", self._mic_thread)
                self._logger.info("Microphone capture enabled for this recording session.")
            else:
                self._logger.info("Microphone capture disabled; speaker-only recording active.")

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

    def _is_any_recording_active(self) -> bool:
        """Return True if either the Record session or mic-only session is active."""
        return bool(
            getattr(self, "_recording_active", False)
            or getattr(self, "_mic_recording_active", False)
        )

    # ------------------------------------------------------------------
    # Mic-only recording session
    # ------------------------------------------------------------------

    def _create_mic_only_audio_threads(self) -> bool:
        """Create VAD + mic capture threads for a mic-only recording session.

        Unlike _create_audio_threads(), this skips system audio capture
        and only sets up the microphone → VAD → transcription pipeline.
        """
        try:
            from ghostmic.core.audio_buffer import AudioBuffer
            from ghostmic.core.audio_capture import MicCaptureThread
            from ghostmic.core.vad import VADThread

            if not hasattr(self, '_buffer') or self._buffer is None:
                audio_cfg = self._config.get("audio", {})
                self._buffer = AudioBuffer(
                    sample_rate=audio_cfg.get("sample_rate", 16_000)
                )

            audio_cfg = self._config.get("audio", {})
            self._cancel_pending_mic_recovery()
            self._mic_device_fallback_attempted = False

            self._vad_thread = VADThread(self._buffer)
            self._vad_thread.speech_segment_ready.connect(self._on_speech_segment)
            self._thread_coordinator.register("vad", self._vad_thread)

            # No system audio thread for mic-only mode
            self._sys_audio_thread = None

            self._mic_thread = MicCaptureThread(
                self._buffer,
                device_index=audio_cfg.get("input_device"),
            )
            self._mic_thread.audio_chunk_ready.connect(
                self._vad_thread.push_chunk
            )
            if hasattr(self._mic_thread, "mic_capture_failed"):
                self._mic_thread.mic_capture_failed.connect(self._on_mic_only_capture_failed)
            self._thread_coordinator.register("mic", self._mic_thread)

            self._logger.info("Created mic-only audio threads (no system audio).")
            return True

        except ImportError as exc:
            self._logger.warning("Audio libraries not available for mic-only: %s", exc)
            return False

    def _start_mic_only_capture(self) -> bool:
        """Start a mic-only recording session with full transcription pipeline."""
        if not self._is_transcription_ready():
            return False

        if not self._create_mic_only_audio_threads():
            return False

        if self._vad_thread:
            self._vad_thread.start()
        if self._mic_thread:
            self._mic_thread.start()
        return True

    def _stop_mic_only_capture(self) -> None:
        """Stop the mic-only recording session."""
        self._thread_coordinator.stop_one("mic", timeout_ms=2000)
        self._thread_coordinator.stop_one("vad", timeout_ms=2000)

        self._sys_audio_thread = None
        self._mic_thread = None
        self._vad_thread = None

    def _on_mic_only_capture_failed(self, detail: str) -> None:
        """Handle mic capture failure during mic-only recording."""
        if not self._mic_recording_active:
            return

        detail_text = str(detail or "unknown microphone error")
        self._logger.warning("Mic-only capture failed: %s", detail_text)

        # Stop the mic-only session and update UI
        if self._is_streaming_normalization_enabled():
            self._stop_streaming_processing_loop(flush=True)
        self._mic_recording_active = False
        self._cancel_pending_mic_recovery()
        self._stop_mic_only_capture()

        if self._transcription_thread:
            self._transcription_thread.clear_pending_segments()

        audio_cfg = self._config.setdefault("audio", {})
        audio_cfg["capture_mic"] = False
        _save_config(self._config, self._config_path)

        if self._window:
            self._window.controls.set_mic_enabled(False)
            self._window.controls.set_status(
                "Microphone capture failed. Check settings.",
                "#f85149",
            )

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
            # If mic-only session is active, stop it before starting full recording
            if self._mic_recording_active:
                self._logger.info("Stopping mic-only session before starting full recording.")
                if self._is_streaming_normalization_enabled():
                    self._stop_streaming_processing_loop(flush=True)
                self._mic_recording_active = False
                self._stop_mic_only_capture()
                if self._window:
                    self._window.controls.set_mic_enabled(False)
                audio_cfg = self._config.setdefault("audio", {})
                audio_cfg["capture_mic"] = False
                _save_config(self._config, self._config_path)

            self._reset_auto_question_session_state()
            if self._is_streaming_normalization_enabled():
                self._reset_streaming_normalization_state(clear_ui=True)
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
            if self._is_streaming_normalization_enabled():
                self._start_streaming_processing_loop()
            if started_with_cloud_fallback and self._window:
                self._window.controls.set_status(
                    "Recording… cloud transcription active",
                    "#f0883e",
                )
            if self._tray:
                self._tray.set_recording(True)
        else:
            if self._is_streaming_normalization_enabled():
                self._stop_streaming_processing_loop(flush=True)
            self._recording_active = False
            self._cancel_pending_mic_recovery()
            self._reset_auto_question_session_state(clear_ui=False)
            self._auto_speaker_silence_generation += 1
            self._auto_speaker_last_signature = ""
            self._auto_speaker_last_trigger_ts = 0.0
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

    def _should_merge_speaker_segments(self, previous, incoming) -> bool:
        """Return True when *incoming* should continue the previous question line."""
        previous_source = str(getattr(previous, "source", "")).strip().lower()
        incoming_source = str(getattr(incoming, "source", "")).strip().lower()
        if previous_source != incoming_source:
            return False
        if not self._is_question_segment_source(previous_source):
            return False

        prev_text = str(getattr(previous, "text", "")).strip()
        incoming_text = str(getattr(incoming, "text", "")).strip()
        if not prev_text or not incoming_text:
            return False

        cfg = self._config.get("transcription", {})
        max_gap = float(cfg.get("speaker_merge_gap_seconds", 2.4))
        max_chars = int(cfg.get("speaker_merge_max_chars", 720))

        prev_ts = float(getattr(previous, "timestamp", 0.0) or 0.0)
        incoming_ts = float(getattr(incoming, "timestamp", 0.0) or 0.0)
        gap = max(0.0, incoming_ts - prev_ts)
        if gap > max_gap:
            return False

        if len(prev_text) + 1 + len(incoming_text) > max_chars:
            return False

        first_word = incoming_text.split()[0].lower()
        continuation_starters = {
            "and",
            "also",
            "then",
            "so",
            "because",
            "which",
            "that",
            "where",
            "when",
            "while",
            "with",
            "without",
            "if",
            "but",
        }
        question_starters = {
            "what",
            "why",
            "how",
            "when",
            "where",
            "who",
            "which",
            "can",
            "could",
            "would",
            "should",
            "do",
            "does",
            "did",
            "is",
            "are",
            "tell",
            "explain",
        }

        previous_looks_complete = prev_text.endswith(("?", ".", "!"))
        if not previous_looks_complete:
            return True

        if first_word in continuation_starters:
            return True

        if first_word in question_starters and gap >= 0.8:
            return False

        # Very short tails after brief pauses are usually chunk splits.
        if len(incoming_text.split()) <= 5 and gap <= 1.2:
            return True

        return False

    def _append_transcript_segment(self, segment, require_recording: bool) -> bool:
        if require_recording and not self._is_any_recording_active():
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

        merged_target = None
        with self._transcript_lock:
            if self._transcript_history and self._should_merge_speaker_segments(
                self._transcript_history[-1], segment
            ):
                merged_target = self._transcript_history[-1]
                merged_text = clean_text(
                    f"{getattr(merged_target, 'text', '')} {segment.text}"
                )
                if not merged_text:
                    self._logger.debug("Merged transcript text was empty after cleaning; ignoring.")
                    return False

                merged_target.text = merged_text
                merged_target.confidence = max(
                    float(getattr(merged_target, "confidence", 1.0)),
                    float(getattr(segment, "confidence", 1.0)),
                )
                merged_target.timestamp = max(
                    float(getattr(merged_target, "timestamp", 0.0) or 0.0),
                    float(getattr(segment, "timestamp", 0.0) or 0.0),
                )

                if self._ai_context_history:
                    last_context = self._ai_context_history[-1]
                    if getattr(last_context, "source", "") == getattr(
                        merged_target,
                        "source",
                        "",
                    ):
                        last_context.text = merged_text
                        last_context.confidence = max(
                            float(getattr(last_context, "confidence", 1.0)),
                            float(getattr(segment, "confidence", 1.0)),
                        )
                        last_context.timestamp = max(
                            float(getattr(last_context, "timestamp", 0.0) or 0.0),
                            float(getattr(segment, "timestamp", 0.0) or 0.0),
                        )

                self._logger.info(
                    "Transcript segment merged: source=%s, merged_len=%d",
                    getattr(segment, "source", "unknown"),
                    len(merged_text),
                )
            else:
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

        question_segment = merged_target if merged_target is not None else segment

        if merged_target is not None:
            self._session_context_store.append_transcript(merged_target)
            if self._window:
                self._window.transcript_panel.set_segment_text(
                    merged_target,
                    merged_target.text,
                )
                self._window.controls.set_status("Listening…", "#3fb950")
            if require_recording:
                if self._is_streaming_normalization_enabled():
                    self._ingest_streaming_transcript_chunk(segment)
                elif self._segment_is_question_source(question_segment):
                    if self._maybe_schedule_initial_recording_question_normalization(question_segment):
                        return True
                    self._schedule_auto_speaker_analysis(question_segment)
            return True

        self._session_context_store.append_transcript(segment)

        if self._window:
            self._window.transcript_panel.add_segment(segment)
            self._window.controls.set_status("Listening…", "#3fb950")

        if require_recording:
            if self._is_streaming_normalization_enabled():
                self._ingest_streaming_transcript_chunk(segment)
            elif self._segment_is_question_source(question_segment):
                if self._maybe_schedule_initial_recording_question_normalization(question_segment):
                    return True
                self._schedule_auto_speaker_analysis(question_segment)

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

        self._session_context_store.append_typed_prompt(prompt, refine)

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

    def _on_suggested_follow_up_selected(self, prompt: str) -> None:
        selected = str(prompt or "").strip()
        if not selected:
            return

        window = getattr(self, "_window", None)
        if window:
            window.set_current_question_text(selected, force=True)
            window.set_question_follow_up_suggestions(
                self._build_local_follow_up_suggestions(selected)
            )
            window.show_follow_up_sent_confirmation(selected)
            self._start_follow_up_suggestion_refresh(selected)

        self._on_ai_text_prompt_submitted(selected, refine=True)

    def _on_ai_response_ready(self, full_text: str) -> None:
        self._logger.info("_on_ai_response_ready called with %d chars", len(full_text))
        self._session_context_store.append_ai_response(full_text)
        with self._transcript_lock:
            self._last_ai_response_text = full_text
            self._ai_context_history.clear()
        if self._window:
            self._window.ai_panel.finish_response(full_text)
            self._window.controls.set_status("✓ Response ready", "#3fb950")
            self._logger.info("AI response displayed successfully (%d chars).", len(full_text))
        # Fix 5: Kick off AI normalization for the next pending streaming segment
        # so subsequent questions get cleaned up while the user reads the response.
        self._auto_send_next_pending_segment()


    def _on_screen_analysis_requested(self) -> None:
        if not self._window:
            return
        if self._screen_analysis_worker is not None:
            self._window.controls.set_status("Screen analysis already running", "#f0883e")
            return

        ai_cfg = self._config.get("ai", {})
        from ghostmic.services.screen_analysis_service import (
            ScreenAnalysisWorker,
            resolve_screen_analysis_provider,
        )

        provider = resolve_screen_analysis_provider(ai_cfg)
        provider_label = "Gemini" if provider == "gemini" else "Groq"
        key_field = "gemini_api_key" if provider == "gemini" else "groq_api_key"
        api_key = str(ai_cfg.get(key_field, "")).strip()
        if not api_key:
            self._window.ai_panel.show_error(
                f"Add your {provider_label} API key in Settings before using screen analysis.",
                title="Screen Analysis",
            )
            self._window.controls.set_status(
                f"{provider_label} API key required",
                "#f85149",
            )
            return

        self._window.controls.set_status(
            f"Capturing screen ({provider_label})…",
            "#58a6ff",
        )
        self._window.controls.set_screen_analysis_busy(True)
        self._window.ai_panel.start_response(title="Screen Analysis")
        self._window.ai_panel.show_thinking(
            f"Analyzing screenshot with {provider_label}…"
        )

        worker = ScreenAnalysisWorker(ai_cfg, parent=None)
        worker.analysis_ready.connect(self._on_screen_analysis_ready)
        worker.analysis_error.connect(self._on_screen_analysis_error)
        worker.finished.connect(self._cleanup_screen_analysis_worker)
        self._screen_analysis_worker = worker
        worker.start()

    def _on_screen_analysis_ready(self, full_text: str) -> None:
        if not self._window:
            return
        summary_text = str(full_text or "").strip()
        if len(summary_text) > 8000:
            summary_text = summary_text[:8000] + "..."
        self._active_screen_summary_text = summary_text
        self._active_screen_anchor_question = ""
        self._session_context_store.append_screen_summary(full_text)
        self._window.ai_panel.finish_response(full_text)
        self._window.controls.set_screen_analysis_busy(False)
        self._window.controls.set_status("✓ Screen analysis ready", "#3fb950")

    def _resolve_active_screen_summary_context(
        self,
        latest_speaker_text: str,
        *,
        force_follow_up: bool = False,
        topic_shift_classifier: Optional[Callable[[str, str], bool]] = None,
    ) -> str:
        """Return sticky screen-summary context while questions stay on-topic."""
        summary_text = str(getattr(self, "_active_screen_summary_text", "")).strip()
        if not summary_text:
            return ""

        latest_question = str(latest_speaker_text or "").strip()
        anchor_question = str(getattr(self, "_active_screen_anchor_question", "")).strip()

        if not anchor_question:
            if latest_question:
                self._active_screen_anchor_question = latest_question
            return summary_text

        if not latest_question:
            return summary_text

        moved_away = False
        if topic_shift_classifier and not force_follow_up:
            try:
                moved_away = bool(topic_shift_classifier(anchor_question, latest_question))
            except Exception as exc:  # pylint: disable=broad-except
                self._logger.debug(
                    "AI request: screen-context topic classification failed: %s",
                    exc,
                )

        if moved_away:
            self._active_screen_summary_text = ""
            self._active_screen_anchor_question = ""
            self._logger.info(
                "AI request: cleared active screen context after topic shift. "
                "anchor=%r new=%r",
                anchor_question[:60],
                latest_question[:60],
            )
            return ""

        self._active_screen_anchor_question = latest_question
        return summary_text

    def _on_screen_analysis_error(self, message: str) -> None:
        if self._window:
            self._window.ai_panel.show_error(message, title="Screen Analysis")
            self._window.controls.set_screen_analysis_busy(False)
            self._window.controls.set_status("Screen analysis failed", "#f85149")

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
        prev_audio_cfg = dict(self._config.get("audio", {}))
        was_recording = bool(self._recording_active)
        was_mic_recording = bool(self._mic_recording_active)

        self._config = new_config
        _save_config(new_config, self._config_path)
        if self._window:
            self._window.update_config(new_config)
            self._window.controls.set_mic_enabled(self._is_mic_capture_enabled())
        if self._tray:
            self._tray.set_stealth_enabled(self._is_stealth_enabled())
        self._refresh_ai_runtime_context()
        self._session_context_compactor.update_ai_config(new_config.get("ai", {}))
        if self._hotkey_manager:
            self._hotkey_manager.reload(new_config.get("hotkeys", {}))

        if self._is_streaming_normalization_enabled():
            # Rebuild runtime thresholds from fresh config.
            self._normalizer_service = None
            self._segment_manager = None
            self._ensure_streaming_pipeline_initialized()
            self._ensure_streaming_segment_timer()
            if self._is_any_recording_active():
                self._start_streaming_processing_loop()
        else:
            self._stop_streaming_processing_loop(flush=False)
            self._reset_streaming_normalization_state(clear_ui=True)

        new_audio_cfg = dict(new_config.get("audio", {}))
        if new_audio_cfg != prev_audio_cfg:
            self._logger.info("Audio settings changed; reinitializing capture threads.")
            self._stop_audio_capture()
            self._start_audio_threads()
            if was_recording:
                if not self._start_audio_capture():
                    self._recording_active = False
                    if self._window:
                        self._window.controls.set_recording(False)
                        self._window.controls.set_status(
                            "Could not restart audio capture with new settings.",
                            "#f85149",
                        )
                    if self._tray:
                        self._tray.set_recording(False)
            elif was_mic_recording:
                if not self._start_mic_only_capture():
                    self._mic_recording_active = False
                    if self._window:
                        self._window.controls.set_mic_enabled(False)
                        self._window.controls.set_status(
                            "Could not restart mic capture with new settings.",
                            "#f85149",
                        )

    def _on_mic_toggled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        audio_cfg = self._config.setdefault("audio", {})

        self._cancel_pending_mic_recovery()
        self._mic_device_fallback_attempted = False
        audio_cfg["capture_mic"] = enabled
        _save_config(self._config, self._config_path)

        # ── Case 1: Record is already active → add/remove mic to existing session ──
        if self._recording_active:
            if enabled and self._enable_mic_capture_live():
                if self._window:
                    self._window.controls.set_status("Microphone capture enabled", "#3fb950")
                return

            if not enabled:
                self._disable_mic_capture_live()
                if self._window:
                    self._window.controls.set_status("Speaker-only mode enabled", "#58a6ff")
                return

            # Live enable failed; fall back to full capture restart
            self._logger.info(
                "Live mic enable failed; falling back to full capture restart."
            )
            self._stop_audio_capture()
            if not self._start_audio_capture():
                self._recording_active = False
                if self._window:
                    self._window.controls.set_recording(False)
                    self._window.controls.set_status(
                        "Could not restart capture after mic toggle.",
                        "#f85149",
                    )
                if self._tray:
                    self._tray.set_recording(False)
            return

        # Lightweight test/app stubs may bypass full initialization. In that
        # mode, keep backward-compatible prime-only behavior.
        if not hasattr(self, "_mic_recording_active"):
            if enabled and hasattr(self, "_prime_mic_capture_async"):
                self._prime_mic_capture_async()
            return

        # ── Case 2: No active recording → start/stop a mic-only session ──
        if enabled:
            # Start a mic-only recording session with full pipeline
            self._reset_auto_question_session_state()
            if self._is_streaming_normalization_enabled():
                self._reset_streaming_normalization_state(clear_ui=True)

            if not self._start_mic_only_capture():
                self._logger.warning(
                    "Mic-only session blocked: transcription unavailable. last_error=%s",
                    self._model_error_message,
                )
                self._mic_recording_active = False
                audio_cfg["capture_mic"] = False
                _save_config(self._config, self._config_path)
                if self._window:
                    self._window.controls.set_mic_enabled(False)
                    self._window.controls.set_status(
                        "Transcription unavailable. Check AI panel/logs.",
                        "#f85149",
                    )
                    if self._model_error_message:
                        self._window.ai_panel.show_error(
                            "Cannot start mic capture until transcription is ready.\n"
                            f"Reason: {self._model_error_message}"
                        )
                return

            self._ensure_ai_thread()
            self._mic_recording_active = True
            if self._is_streaming_normalization_enabled():
                self._start_streaming_processing_loop()
            if self._window:
                self._window.controls.set_status(
                    "Mic recording active — listening…",
                    "#3fb950",
                )
            self._logger.info("Mic-only recording session started.")
        else:
            # Stop the mic-only recording session
            if not self._mic_recording_active:
                return

            if self._is_streaming_normalization_enabled():
                self._stop_streaming_processing_loop(flush=True)
            self._mic_recording_active = False
            self._cancel_pending_mic_recovery()
            self._reset_auto_question_session_state(clear_ui=False)
            self._auto_speaker_silence_generation += 1
            self._auto_speaker_last_signature = ""
            self._auto_speaker_last_trigger_ts = 0.0
            self._stop_mic_only_capture()
            if self._transcription_thread:
                self._transcription_thread.clear_pending_segments()
            if self._ai_thread:
                self._ai_thread.clear_pending_requests()
            with self._transcript_lock:
                self._ai_context_history.clear()
            if self._window:
                self._window.controls.set_status("Mic stopped", "#58a6ff")
            self._logger.info("Mic-only recording session stopped.")

    def _is_stealth_enabled(self) -> bool:
        return bool(self._config.get("ui", {}).get("stealth_enabled", True))

    def _on_stealth_toggled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        ui_cfg = self._config.setdefault("ui", {})
        if bool(ui_cfg.get("stealth_enabled", True)) == enabled:
            return

        ui_cfg["stealth_enabled"] = enabled
        _save_config(self._config, self._config_path)

        if self._window:
            self._window.update_config(self._config)
            if enabled:
                self._window.controls.set_status("Stealth mode enabled", "#58a6ff")
            else:
                self._window.controls.set_status(
                    "Stealth mode disabled for screenshots",
                    "#f0883e",
                )

        if self._tray:
            self._tray.set_stealth_enabled(enabled)

        self._logger.info("Stealth mode toggled: enabled=%s", enabled)

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

    def _is_streaming_normalization_enabled(self) -> bool:
        config = getattr(self, "_config", {})
        transcription_cfg = config.get("transcription", {}) if isinstance(config, dict) else {}
        return bool(transcription_cfg.get("streaming_normalization_enabled", False))

    def _ensure_streaming_pipeline_initialized(self) -> None:
        config = getattr(self, "_config", {})
        transcription_cfg = config.get("transcription", {}) if isinstance(config, dict) else {}

        if getattr(self, "_transcript_store", None) is None:
            raw_buffer_seconds = transcription_cfg.get("streaming_buffer_seconds", 30.0)
            raw_buffer_chunks = transcription_cfg.get("streaming_buffer_max_chunks", 260)
            try:
                buffer_seconds = float(raw_buffer_seconds)
            except (TypeError, ValueError):
                buffer_seconds = 30.0
            try:
                buffer_chunks = int(raw_buffer_chunks)
            except (TypeError, ValueError):
                buffer_chunks = 260

            self._transcript_store = TranscriptStore(
                max_window_seconds=max(20.0, min(45.0, buffer_seconds)),
                max_chunks=max(60, min(600, buffer_chunks)),
            )

        if getattr(self, "_normalizer_service", None) is None:
            raw_pause_seconds = transcription_cfg.get("streaming_pause_boundary_seconds", 1.15)
            raw_soft_flush_seconds = transcription_cfg.get("streaming_soft_flush_seconds", 10.0)
            raw_soft_flush_chunks = transcription_cfg.get("streaming_soft_flush_chunks", 8)
            try:
                pause_seconds = float(raw_pause_seconds)
            except (TypeError, ValueError):
                pause_seconds = 1.15
            try:
                soft_flush_seconds = float(raw_soft_flush_seconds)
            except (TypeError, ValueError):
                soft_flush_seconds = 10.0
            try:
                soft_flush_chunks = int(raw_soft_flush_chunks)
            except (TypeError, ValueError):
                soft_flush_chunks = 8

            self._normalizer_service = NormalizerService(
                pause_boundary_seconds=pause_seconds,
                soft_flush_seconds=soft_flush_seconds,
                soft_flush_chunks=soft_flush_chunks,
            )

        if getattr(self, "_segment_manager", None) is None and self._normalizer_service is not None:
            self._segment_manager = SegmentManager(
                self._normalizer_service,
                max_overlap_chunks=2,
            )

        if getattr(self, "_ai_trigger_service", None) is None:
            self._ai_trigger_service = AITriggerService()

        if not hasattr(self, "_normalized_segment_items"):
            self._normalized_segment_items = []
        if not hasattr(self, "_normalized_segment_lookup"):
            self._normalized_segment_lookup = {}
        if not hasattr(self, "_pending_normalization_queue"):
            self._pending_normalization_queue = []


    def _streaming_processing_interval_ms(self) -> int:
        config = getattr(self, "_config", {})
        transcription_cfg = config.get("transcription", {}) if isinstance(config, dict) else {}
        raw_interval = transcription_cfg.get(
            "streaming_processing_interval_ms",
            STREAMING_SEGMENT_LOOP_INTERVAL_MS,
        )
        try:
            interval_ms = int(raw_interval)
        except (TypeError, ValueError):
            interval_ms = STREAMING_SEGMENT_LOOP_INTERVAL_MS
        return max(1_500, min(4_000, interval_ms))

    def _ensure_streaming_segment_timer(self) -> None:
        timer = getattr(self, "_streaming_segment_timer", None)
        if timer is not None:
            timer.setInterval(self._streaming_processing_interval_ms())
            return

        try:
            from PyQt6.QtCore import QTimer
        except Exception as exc:  # pylint: disable=broad-except
            self._logger.debug("Streaming segment timer unavailable: %s", exc)
            return

        timer_parent = self._window if self._window is not None else None
        timer = QTimer(timer_parent)
        timer.setInterval(self._streaming_processing_interval_ms())
        timer.timeout.connect(self._process_streaming_transcript_chunks)
        self._streaming_segment_timer = timer

    def _start_streaming_processing_loop(self) -> None:
        if not self._is_streaming_normalization_enabled():
            return

        self._ensure_streaming_pipeline_initialized()
        self._ensure_streaming_segment_timer()
        timer = getattr(self, "_streaming_segment_timer", None)
        if timer is not None and not timer.isActive():
            timer.start()

    def _stop_streaming_processing_loop(self, *, flush: bool = False) -> None:
        if flush:
            self._process_streaming_transcript_chunks(force_flush=True)

        timer = getattr(self, "_streaming_segment_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()

    def _reset_streaming_normalization_state(self, *, clear_ui: bool) -> None:
        self._ensure_streaming_pipeline_initialized()

        if self._transcript_store is not None:
            self._transcript_store.reset()
        if self._segment_manager is not None:
            self._segment_manager.reset()
        trigger_service = getattr(self, "_ai_trigger_service", None)
        if trigger_service is not None:
            trigger_service.reset()

        self._normalized_segment_items.clear()
        self._normalized_segment_lookup.clear()
        # Clear any pending segment normalization jobs so stale segments from a
        # previous recording session are not processed after a restart.
        queue = getattr(self, "_pending_normalization_queue", None)
        if queue is not None:
            queue.clear()

        if clear_ui and self._window and hasattr(self._window, "set_normalized_segments"):
            self._window.set_normalized_segments([])


    def _ingest_streaming_transcript_chunk(self, segment) -> None:
        if not self._is_streaming_normalization_enabled():
            return
        if not self._is_any_recording_active():
            return

        self._ensure_streaming_pipeline_initialized()
        if self._transcript_store is None:
            return

        chunk = self._transcript_store.append_from_transcript_segment(segment)
        if chunk is None:
            return

        # Always attempt to process after every ingested chunk.  The
        # NormalizerService's boundary and soft-flush logic decides whether
        # enough material has accumulated for a complete segment.  Limiting
        # this to only terminal-punctuation chunks caused transcripts to sit
        # unprocessed for long intervals.
        self._process_streaming_transcript_chunks(force_flush=False)

    def _process_streaming_transcript_chunks(self, force_flush: bool = False) -> None:
        if not self._is_streaming_normalization_enabled():
            return
        if not self._is_any_recording_active() and not force_flush:
            return

        self._ensure_streaming_pipeline_initialized()
        if self._segment_manager is None or self._transcript_store is None:
            return

        segments = self._segment_manager.consume_ready_segments(
            self._transcript_store,
            force_flush=force_flush,
        )
        if not segments:
            return

        for normalized_segment in segments:
            self._register_normalized_segment(normalized_segment)

        self._sync_normalized_segments_to_window()

    def _register_normalized_segment(self, normalized_segment) -> None:
        segment_id = str(getattr(normalized_segment, "segment_id", "")).strip()
        if not segment_id:
            return
        if segment_id in self._normalized_segment_lookup:
            return

        normalized_text = str(getattr(normalized_segment, "normalized_text", "")).strip()
        if not normalized_text:
            return

        source = str(getattr(normalized_segment, "source", "speaker") or "speaker").strip().lower() or "speaker"
        source_chunk_ids = list(getattr(normalized_segment, "source_chunk_ids", []) or [])

        # Only register question-like segments for UI display and AI dispatch.
        # Statements, greetings, filler, and other non-question text are kept
        # as context only — they don't clutter the normalized segments panel.
        from ghostmic.services.ai_trigger_service import is_question_like
        from ghostmic.utils.text_processing import ensure_question_format

        if not is_question_like(normalized_text):
            self._logger.debug(
                "Normalized segment skipped (not question-like): %r",
                normalized_text[:80],
            )
            return

        # Minimum word count guard — rejects fragments like "Has more records than?"
        word_count = len(normalized_text.split())
        if word_count < 5:
            self._logger.debug(
                "Normalized segment skipped (too short, %d words): %r",
                word_count,
                normalized_text[:80],
            )
            return

        normalized_text = ensure_question_format(normalized_text)

        # Always log to session context so the AI has full conversation awareness,
        # regardless of whether this segment is a question.
        self._session_context_store.append_event(
            "normalized_segment",
            normalized_text,
            source=source,
            metadata={
                "segment_id": segment_id,
                "source_chunk_ids": source_chunk_ids,
            },
        )

        item = {
            "segment_id": segment_id,
            "text": normalized_text,
            "status": "pending",
            "source": source,
            "source_chunk_ids": source_chunk_ids,
        }
        self._normalized_segment_items.append(item)
        self._normalized_segment_lookup[segment_id] = item

        max_items = 3
        if len(self._normalized_segment_items) > max_items:
            removed = self._normalized_segment_items[:-max_items]
            self._normalized_segment_items = self._normalized_segment_items[-max_items:]
            for old in removed:
                old_id = str(old.get("segment_id", "")).strip()
                if old_id:
                    self._normalized_segment_lookup.pop(old_id, None)

        trigger_service = getattr(self, "_ai_trigger_service", None)
        # Auto-promote ANY question-like segment during recording to the top
        # Question area — not just the first one.  Each new question replaces
        # the previous primary question and triggers a fresh AI response.
        should_auto_promote = bool(
            self._is_any_recording_active()
            and is_question_like(normalized_text)
        )
        # Keep the trigger service in sync so other subsystems know a question
        # has been dispatched.
        if should_auto_promote and trigger_service is not None:
            trigger_service.mark_first_question_sent()

        if should_auto_promote:
            # Promote to top Question area, normalize, and trigger AI response.
            try:
                setattr(normalized_segment, "text", normalized_text)
            except Exception:
                pass
            self._on_speaker_question_normalize_requested(
                normalized_segment,
                normalized_text,
                auto_send_after=True,
            )
        else:
            # Non-question text: queue for background AI normalization only.
            self._enqueue_streaming_segment_normalization(segment_id, normalized_text)


    def _sync_normalized_segments_to_window(self) -> None:
        if not self._window or not hasattr(self._window, "set_normalized_segments"):
            return

        rows = [
            {
                "segment_id": str(item.get("segment_id", "")),
                "normalized_text": str(item.get("text", "")),
                "status": str(item.get("status", "pending")),
            }
            for item in list(getattr(self, "_normalized_segment_items", []))
        ]
        self._window.set_normalized_segments(rows)

    def _update_normalized_segment_item(
        self,
        segment_id: str,
        *,
        text: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Optional[dict]:
        key = str(segment_id or "").strip()
        if not key:
            return None

        lookup = getattr(self, "_normalized_segment_lookup", {})
        item = lookup.get(key)
        if item is None:
            return None

        if text is not None:
            cleaned_text = " ".join(str(text or "").split()).strip()
            if cleaned_text:
                item["text"] = cleaned_text

        if status is not None:
            normalized_status = str(status).strip().lower()
            if normalized_status in {"pending", "sent"}:
                item["status"] = normalized_status

        return item

    def _find_pending_normalized_segment_by_text(self, question: str) -> Optional[str]:
        key = self._canonical_question_key(question)
        if not key:
            return None

        for item in list(getattr(self, "_normalized_segment_items", [])):
            if str(item.get("status", "pending")) == "sent":
                continue
            text = str(item.get("text", ""))
            if key == self._canonical_question_key(text):
                return str(item.get("segment_id", "")).strip() or None

        return None

    def _on_normalized_segment_send_requested(self, segment_id: str) -> None:
        self._send_normalized_segment_to_ai(str(segment_id or "").strip(), auto=False)

    def _send_normalized_segment_to_ai(self, segment_id: str, *, auto: bool) -> None:
        key = str(segment_id or "").strip()
        if not key:
            return

        lookup = getattr(self, "_normalized_segment_lookup", {})
        item = lookup.get(key)
        if item is None:
            return
        if str(item.get("status", "pending")).strip().lower() == "sent":
            return

        from ghostmic.utils.text_processing import ensure_question_format

        question_text = ensure_question_format(str(item.get("text", "")).strip())
        if not question_text:
            return

        source = str(item.get("source", "speaker") or "speaker").strip().lower() or "speaker"

        from ghostmic.core.transcription_engine import TranscriptSegment

        question_segment = TranscriptSegment(
            text=question_text,
            source=source,
            confidence=1.0,
            timestamp=time.time(),
        )

        item["status"] = "sent"
        trigger_service = getattr(self, "_ai_trigger_service", None)
        if trigger_service is not None:
            trigger_service.mark_first_question_sent()

        self._auto_primary_question_sent = True
        self._active_primary_question_text = question_text
        self._pop_queued_normalized_question(question_text)
        self._prime_ai_context_with_question(question_segment, question_text)

        event_kind = "auto_ai_request" if auto else "manual_ai_request"
        self._session_context_store.append_event(
            event_kind,
            question_text,
            source=source,
            metadata={
                "segment_id": key,
                "source_chunk_ids": list(item.get("source_chunk_ids", []) or []),
            },
        )

        if self._window:
            self._window.set_question_lock_enabled(True)
            self._window.set_current_question_text(question_text, force=True)
            self._window.set_question_follow_up_suggestions([])
            if auto:
                self._window.controls.set_status(
                    "Question detected. Generating response…",
                    "#58a6ff",
                )
            else:
                self._window.controls.set_status(
                    "Sending selected segment to AI…",
                    "#58a6ff",
                )
            self._window.ai_panel.show_thinking("Generating response…")

        self._sync_normalized_segments_to_window()
        self._generate_ai_response(force_follow_up=False)

    def _on_speaker_transcript_edited(self, segment, text: str) -> None:
        from ghostmic.utils.text_processing import clean_text

        cleaned = clean_text(str(text or ""))
        if not cleaned:
            return
        with self._transcript_lock:
            segment.text = cleaned
        if self._window and self._segment_is_question_source(segment):
            self._window.set_question_follow_up_suggestions([])

    @staticmethod
    def _is_question_segment_source(source: str) -> bool:
        return str(source or "").strip().lower() in {"speaker", "user"}

    def _segment_is_question_source(self, segment) -> bool:
        return self._is_question_segment_source(getattr(segment, "source", ""))

    @staticmethod
    def _canonical_question_key(text: str) -> str:
        return " ".join(str(text or "").split()).strip().rstrip(".?!").lower()

    def _sync_queued_normalized_questions_to_window(self) -> None:
        if self._is_streaming_normalization_enabled():
            return
        if not self._window:
            return
        self._window.set_queued_normalized_questions(
            [str(item.get("text", "")) for item in self._queued_normalized_questions]
        )

    def _enqueue_normalized_question(
        self,
        segment,
        question: str,
        *,
        source: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        from ghostmic.utils.text_processing import ensure_question_format

        question = ensure_question_format(question)
        key = self._canonical_question_key(question)
        if not key:
            return False

        if key == self._canonical_question_key(self._active_primary_question_text):
            return False

        for item in self._queued_normalized_questions:
            if key == self._canonical_question_key(str(item.get("text", ""))):
                return False

        self._queued_normalized_questions.append(
            {
                "text": question,
                "source": source,
                "segment": segment,
                "metadata": dict(metadata or {}),
            }
        )

        max_queue_items = 3
        if len(self._queued_normalized_questions) > max_queue_items:
            self._queued_normalized_questions = self._queued_normalized_questions[-max_queue_items:]

        self._sync_queued_normalized_questions_to_window()
        return True

    def _pop_queued_normalized_question(self, question: str) -> Optional[dict]:
        key = self._canonical_question_key(question)
        for index, item in enumerate(self._queued_normalized_questions):
            if key == self._canonical_question_key(str(item.get("text", ""))):
                popped = self._queued_normalized_questions.pop(index)
                self._sync_queued_normalized_questions_to_window()
                return popped
        return None

    def _reset_auto_question_session_state(self, *, clear_ui: bool = True) -> None:
        self._auto_primary_question_sent = False
        self._active_primary_question_text = ""
        if not hasattr(self, "_queued_normalized_questions"):
            self._queued_normalized_questions = []
        self._queued_normalized_questions.clear()
        self._initial_recording_question_pending = False
        self._initial_recording_question_consumed = False
        self._initial_recording_question_generation = int(
            getattr(self, "_initial_recording_question_generation", 0)
        ) + 1
        trigger_service = getattr(self, "_ai_trigger_service", None)
        if trigger_service is not None:
            trigger_service.reset()
        if clear_ui and self._window:
            if hasattr(self._window, "set_question_lock_enabled"):
                self._window.set_question_lock_enabled(False)
            if hasattr(self._window, "clear_queued_normalized_questions"):
                self._window.clear_queued_normalized_questions()

    def _on_queued_question_send_requested(self, question: str) -> None:
        from ghostmic.core.transcription_engine import TranscriptSegment
        from ghostmic.utils.text_processing import clean_text, ensure_question_format

        cleaned = ensure_question_format(clean_text(str(question or "")))
        if not cleaned:
            return

        normalized_segment_id = self._find_pending_normalized_segment_by_text(cleaned)
        if normalized_segment_id:
            self._send_normalized_segment_to_ai(normalized_segment_id, auto=False)
            return

        queued_item = self._pop_queued_normalized_question(cleaned)
        if queued_item is None:
            queued_item = {
                "segment": TranscriptSegment(text=cleaned, source="speaker", confidence=1.0),
                "source": "speaker",
            }

        segment = queued_item.get("segment")
        if segment is None:
            segment = TranscriptSegment(text=cleaned, source="speaker", confidence=1.0)

        self._auto_primary_question_sent = True
        self._active_primary_question_text = cleaned
        trigger_service = getattr(self, "_ai_trigger_service", None)
        if trigger_service is not None:
            trigger_service.mark_first_question_sent()

        if self._window:
            self._window.set_question_lock_enabled(True)
            self._window.set_current_question_text(cleaned, force=True)

        self._on_speaker_send_requested(segment, cleaned)

    def _prime_ai_context_with_question(self, segment, question: str) -> None:
        from ghostmic.core.transcription_engine import TranscriptSegment

        with self._transcript_lock:
            self._ai_context_history = [
                TranscriptSegment(
                    text=question,
                    source="speaker",
                    confidence=float(getattr(segment, "confidence", 1.0)),
                    timestamp=float(getattr(segment, "timestamp", time.time()) or time.time()),
                )
            ]

    def _is_auto_speaker_analysis_enabled(self) -> bool:
        ai_cfg = self._config.get("ai", {})
        if not bool(ai_cfg.get("auto_speaker_analysis_enabled", True)):
            return False
        trigger_mode = str(ai_cfg.get("trigger_mode", "auto")).strip().lower()
        return trigger_mode in {"auto", "continuous"}

    def _auto_speaker_silence_delay_ms(self) -> int:
        ai_cfg = self._config.get("ai", {})
        raw_seconds = ai_cfg.get("auto_speaker_silence_seconds", 1.2)
        try:
            seconds = float(raw_seconds)
        except (TypeError, ValueError):
            seconds = 1.2
        seconds = max(0.3, min(6.0, seconds))
        return int(seconds * 1000)

    def _is_auto_speaker_candidate_text(self, text: str) -> bool:
        from ghostmic.utils.text_processing import clean_text

        cleaned = clean_text(str(text or ""))
        if not cleaned:
            return False

        ai_cfg = self._config.get("ai", {})
        raw_min_words = ai_cfg.get("auto_speaker_min_words", 4)
        raw_min_chars = ai_cfg.get("auto_speaker_min_chars", 18)
        try:
            min_words = int(raw_min_words)
        except (TypeError, ValueError):
            min_words = 4
        try:
            min_chars = int(raw_min_chars)
        except (TypeError, ValueError):
            min_chars = 18

        min_words = max(1, min(24, min_words))
        min_chars = max(1, min(500, min_chars))

        if len(cleaned) < min_chars:
            return False
        if len(cleaned.split()) < min_words:
            return False
        return any(ch.isalpha() for ch in cleaned)

    @staticmethod
    def _build_auto_speaker_signature(segment, text: str) -> str:
        normalized = " ".join(str(text or "").lower().split())
        return f"{id(segment)}::{normalized}"

    def _mark_auto_speaker_triggered(self, segment, text: str) -> None:
        self._auto_speaker_last_signature = self._build_auto_speaker_signature(segment, text)
        self._auto_speaker_last_trigger_ts = time.time()

    def _is_current_auto_speaker_generation(self, generation: Optional[int]) -> bool:
        if generation is None:
            return True
        return int(generation) == int(self._auto_speaker_silence_generation)

    def _maybe_schedule_initial_recording_question_normalization(self, segment) -> bool:
        if not self._is_auto_speaker_analysis_enabled():
            return False
        if not self._segment_is_question_source(segment):
            return False
        if bool(getattr(self, "_initial_recording_question_consumed", False)):
            return False
        if bool(getattr(self, "_initial_recording_question_pending", False)):
            return True

        from ghostmic.utils.text_processing import clean_text

        cleaned = clean_text(str(getattr(segment, "text", "")))
        if not self._is_auto_speaker_candidate_text(cleaned):
            return False

        try:
            from PyQt6.QtCore import QTimer
        except Exception as exc:  # pylint: disable=broad-except
            self._logger.debug("Initial question normalization timer unavailable: %s", exc)
            return False

        generation = int(getattr(self, "_initial_recording_question_generation", 0))
        self._initial_recording_question_pending = True
        self._logger.info(
            "Scheduling initial question normalization in %dms.",
            INITIAL_RECORDING_QUESTION_NORMALIZATION_DELAY_MS,
        )
        QTimer.singleShot(
            INITIAL_RECORDING_QUESTION_NORMALIZATION_DELAY_MS,
            lambda seg=segment, token=generation: self._on_initial_recording_question_normalization_elapsed(
                seg,
                token,
            ),
        )
        return True

    def _on_initial_recording_question_normalization_elapsed(self, segment, generation: int) -> None:
        if generation != int(getattr(self, "_initial_recording_question_generation", 0)):
            return
        if not self._is_any_recording_active():
            return
        if not self._is_auto_speaker_analysis_enabled():
            return
        if not self._segment_is_question_source(segment):
            return

        from ghostmic.utils.text_processing import clean_text

        cleaned = clean_text(str(getattr(segment, "text", "")))
        if not cleaned:
            self._logger.debug("Initial question normalization skipped: transcript was empty.")
            self._initial_recording_question_pending = False
            self._initial_recording_question_consumed = True
            return

        if self._question_normalization_worker is not None:
            self._logger.debug(
                "Initial question normalization deferred: normalization worker busy."
            )
            try:
                from PyQt6.QtCore import QTimer
            except Exception as exc:  # pylint: disable=broad-except
                self._logger.debug(
                    "Initial question normalization retry timer unavailable: %s",
                    exc,
                )
                return

            QTimer.singleShot(
                700,
                lambda seg=segment, token=generation: self._on_initial_recording_question_normalization_elapsed(
                    seg,
                    token,
                ),
            )
            return

        self._mark_auto_speaker_triggered(segment, cleaned)
        self._on_speaker_question_normalize_requested(
            segment,
            cleaned,
            auto_send_after=False,
            initial_recording_question=True,
            recording_question_generation=generation,
        )

    def _schedule_auto_speaker_analysis(self, segment, *, delay_ms: Optional[int] = None) -> None:
        if not self._is_auto_speaker_analysis_enabled():
            return
        if not self._segment_is_question_source(segment):
            return

        text = str(getattr(segment, "text", ""))
        if not self._is_auto_speaker_candidate_text(text):
            return

        self._auto_speaker_silence_generation += 1
        generation = self._auto_speaker_silence_generation
        timeout_ms = self._auto_speaker_silence_delay_ms() if delay_ms is None else int(delay_ms)
        timeout_ms = max(50, timeout_ms)

        try:
            from PyQt6.QtCore import QTimer
        except Exception as exc:  # pylint: disable=broad-except
            self._logger.debug("Auto speaker analysis timer unavailable: %s", exc)
            return

        QTimer.singleShot(
            timeout_ms,
            lambda seg=segment, token=generation: self._on_auto_speaker_silence_elapsed(
                seg,
                token,
            ),
        )

    def _on_auto_speaker_silence_elapsed(self, segment, generation: int) -> None:
        if generation != self._auto_speaker_silence_generation:
            return
        if not self._is_any_recording_active():
            return
        if not self._is_auto_speaker_analysis_enabled():
            return
        if not self._segment_is_question_source(segment):
            return

        from ghostmic.utils.text_processing import clean_text

        cleaned = clean_text(str(getattr(segment, "text", "")))
        if not self._is_auto_speaker_candidate_text(cleaned):
            return

        ai_cfg = self._config.get("ai", {})
        raw_cooldown = ai_cfg.get("auto_speaker_retrigger_cooldown_seconds", 5.0)
        try:
            cooldown = float(raw_cooldown)
        except (TypeError, ValueError):
            cooldown = 5.0
        cooldown = max(0.5, min(30.0, cooldown))

        signature = self._build_auto_speaker_signature(segment, cleaned)
        now = time.time()
        if (
            signature == self._auto_speaker_last_signature
            and (now - self._auto_speaker_last_trigger_ts) < cooldown
        ):
            self._logger.debug(
                "Auto speaker analysis skipped: duplicate within %.1fs cooldown.",
                cooldown,
            )
            return

        if self._question_normalization_worker is not None:
            self._logger.debug(
                "Auto speaker analysis deferred: normalization worker busy."
            )
            self._schedule_auto_speaker_analysis(segment, delay_ms=700)
            return

        self._mark_auto_speaker_triggered(segment, cleaned)
        self._on_speaker_question_normalize_requested(
            segment,
            cleaned,
            auto_send_after=True,
            auto_send_generation=generation,
        )

    def _on_speaker_question_normalize_requested(
        self,
        segment,
        text: str,
        *,
        auto_send_after: bool = False,
        auto_send_generation: Optional[int] = None,
        initial_recording_question: bool = False,
        recording_question_generation: Optional[int] = None,
    ) -> None:
        from ghostmic.utils.text_processing import clean_text

        if not self._segment_is_question_source(segment):
            return

        cleaned = clean_text(str(text or ""))
        if not cleaned:
            if self._window:
                self._window.controls.set_status("Question is empty", "#f0883e")
            return

        self._on_speaker_transcript_edited(segment, cleaned)

        if self._window:
            self._window.set_question_follow_up_suggestions([])

        if auto_send_after and auto_send_generation is None:
            auto_send_generation = int(self._auto_speaker_silence_generation)

        if self._question_normalization_worker is not None:
            if auto_send_after:
                self._schedule_auto_speaker_analysis(segment, delay_ms=700)
                return
            if self._window:
                self._window.controls.set_status(
                    "Question normalization already running", "#f0883e"
                )
            return

        try:
            from ghostmic.services.question_normalization_service import (
                QuestionNormalizationWorker,
            )

            worker = QuestionNormalizationWorker(
                self._build_ai_runtime_config(),
                cleaned,
                parent=None,
            )
            if hasattr(worker, "normalized_with_followups_ready"):
                worker.normalized_with_followups_ready.connect(
                    lambda normalized, follow_ups, seg=segment, auto_send=auto_send_after, auto_generation=auto_send_generation, initial=initial_recording_question, recording_generation=recording_question_generation: self._on_speaker_question_normalized(
                        seg,
                        normalized,
                        follow_up_questions=list(follow_ups or []),
                        auto_send=auto_send,
                        auto_send_generation=auto_generation,
                        initial_recording_question=initial,
                        recording_question_generation=recording_generation,
                    )
                )
            else:
                worker.normalized_ready.connect(
                    lambda normalized, seg=segment, auto_send=auto_send_after, auto_generation=auto_send_generation, initial=initial_recording_question, recording_generation=recording_question_generation: self._on_speaker_question_normalized(
                        seg,
                        normalized,
                        auto_send=auto_send,
                        auto_send_generation=auto_generation,
                        initial_recording_question=initial,
                        recording_question_generation=recording_generation,
                    )
                )
            worker.normalization_error.connect(
                lambda message, seg=segment, auto_send=auto_send_after, auto_generation=auto_send_generation, initial=initial_recording_question, recording_generation=recording_question_generation: self._on_speaker_question_normalization_error(
                    seg,
                    message,
                    auto_send=auto_send,
                    auto_send_generation=auto_generation,
                    initial_recording_question=initial,
                    recording_question_generation=recording_generation,
                )
            )
            worker.finished.connect(self._cleanup_question_normalization_worker)
            self._question_normalization_worker = worker
            if self._window:
                self._window.transcript_panel.set_segment_normalization_busy(
                    segment,
                    True,
                    "Analyzing question…" if auto_send_after else "Normalizing question…",
                )
                self._window.controls.set_status(
                    "Analyzing question…" if auto_send_after else "Normalizing question…",
                    "#58a6ff",
                )
            worker.start()
        except Exception as exc:  # pylint: disable=broad-except
            self._logger.warning("Could not start question normalization: %s", exc)
            if self._window:
                self._window.transcript_panel.set_segment_normalization_busy(
                    segment,
                    False,
                    "",
                )
                self._window.controls.set_status(
                    "Question normalization unavailable", "#f85149"
                )
            if auto_send_after:
                self._on_speaker_question_normalization_error(
                    segment,
                    str(exc),
                    auto_send=True,
                    auto_send_generation=auto_send_generation,
                )

    def _on_speaker_question_normalized(
        self,
        segment,
        normalized_text: str,
        follow_up_questions: Optional[List[str]] = None,
        *,
        auto_send: bool = False,
        auto_send_generation: Optional[int] = None,
        initial_recording_question: bool = False,
        recording_question_generation: Optional[int] = None,
    ) -> None:
        from ghostmic.utils.text_processing import clean_text, ensure_question_format

        if initial_recording_question and recording_question_generation is not None:
            if int(recording_question_generation) != int(
                getattr(self, "_initial_recording_question_generation", 0)
            ):
                self._logger.debug(
                    "Skipping stale initial recording question result: generation=%s current=%s",
                    recording_question_generation,
                    getattr(self, "_initial_recording_question_generation", 0),
                )
                if self._window:
                    self._window.transcript_panel.set_segment_normalization_busy(
                        segment,
                        False,
                        "",
                    )
                return

        if auto_send and not self._is_current_auto_speaker_generation(auto_send_generation):
            self._logger.debug(
                "Skipping stale auto normalization result: generation=%s current=%s",
                auto_send_generation,
                self._auto_speaker_silence_generation,
            )
            if self._window:
                self._window.transcript_panel.set_segment_normalization_busy(
                    segment,
                    False,
                    "",
                )
            return

        if initial_recording_question:
            self._initial_recording_question_pending = False
            self._initial_recording_question_consumed = True

        segment_source = str(getattr(segment, "source", "")).strip().lower() or "speaker"
        normalized = ensure_question_format(clean_text(str(normalized_text or "")))
        if not normalized:
            normalized = ensure_question_format(
                clean_text(str(getattr(segment, "text", "")).strip())
            )
        if not normalized:
            self._on_speaker_question_normalization_error(
                segment,
                "Normalization produced empty text.",
                auto_send=auto_send,
                auto_send_generation=auto_send_generation,
            )
            return

        with self._transcript_lock:
            segment.text = normalized

        self._session_context_store.append_event(
            "normalized_question",
            normalized,
            source=segment_source,
        )

        segment_id = str(getattr(segment, "segment_id", "")).strip()
        normalized_item = None
        if segment_id:
            normalized_item = self._update_normalized_segment_item(
                segment_id,
                text=normalized,
            )
        if normalized_item is None:
            pending_segment_id = self._find_pending_normalized_segment_by_text(normalized)
            if pending_segment_id:
                segment_id = pending_segment_id
                normalized_item = self._update_normalized_segment_item(
                    segment_id,
                    text=normalized,
                )
        if normalized_item is not None:
            self._sync_normalized_segments_to_window()

        # Every auto-sent question is promoted to the top Question area and
        # triggers an AI response — no queuing.  This ensures each new
        # question from recording mode is treated as the active primary
        # question rather than being tucked away in Normalized Segments.

        cleaned_follow_ups: List[str] = []
        seen: set[str] = set()
        for question in list(follow_up_questions or []):
            candidate = clean_text(str(question or ""))
            if not candidate:
                continue
            if not candidate.endswith("?"):
                candidate = f"{candidate.rstrip('.!')}?"
            canonical = candidate.rstrip(".?!").lower()
            if canonical in seen:
                continue
            seen.add(canonical)
            cleaned_follow_ups.append(candidate)
            if len(cleaned_follow_ups) >= 3:
                break

        if self._window:
            self._window.transcript_panel.set_segment_text(segment, normalized)
            self._window.transcript_panel.set_segment_normalization_busy(
                segment,
                False,
                "Question analyzed" if auto_send else "Question normalized",
            )
            self._window.set_question_follow_up_suggestions(cleaned_follow_ups)
            if auto_send:
                self._window.set_question_lock_enabled(True)
                self._window.set_current_question_text(normalized, force=True)
                self._window.controls.set_status(
                    "Question detected. Generating response…", "#58a6ff"
                )
                self._window.ai_panel.show_thinking("Generating response…")
            else:
                self._window.controls.set_status(
                    "Question normalized. Review, then Send to AI.", "#3fb950"
                )

        if auto_send:
            if normalized_item is not None:
                normalized_item["status"] = "sent"
                self._sync_normalized_segments_to_window()
            self._auto_primary_question_sent = True
            self._active_primary_question_text = normalized
            trigger_service = getattr(self, "_ai_trigger_service", None)
            if trigger_service is not None:
                trigger_service.mark_first_question_sent()
            self._prime_ai_context_with_question(segment, normalized)
            self._session_context_store.append_event(
                "auto_ai_request",
                normalized,
                source=segment_source,
            )
            self._generate_ai_response(force_follow_up=False)

    def _on_speaker_question_normalization_error(
        self,
        segment,
        message: str,
        *,
        auto_send: bool = False,
        auto_send_generation: Optional[int] = None,
        initial_recording_question: bool = False,
        recording_question_generation: Optional[int] = None,
    ) -> None:
        from ghostmic.utils.text_processing import clean_text

        if initial_recording_question and recording_question_generation is not None:
            if int(recording_question_generation) != int(
                getattr(self, "_initial_recording_question_generation", 0)
            ):
                self._logger.debug(
                    "Skipping stale initial recording question error: generation=%s current=%s",
                    recording_question_generation,
                    getattr(self, "_initial_recording_question_generation", 0),
                )
                if self._window:
                    self._window.transcript_panel.set_segment_normalization_busy(
                        segment,
                        False,
                        "",
                    )
                return

        if auto_send and not self._is_current_auto_speaker_generation(auto_send_generation):
            self._logger.debug(
                "Skipping stale auto normalization error: generation=%s current=%s",
                auto_send_generation,
                self._auto_speaker_silence_generation,
            )
            if self._window:
                self._window.transcript_panel.set_segment_normalization_busy(
                    segment,
                    False,
                    "",
                )
            return

        segment_source = str(getattr(segment, "source", "")).strip().lower() or "speaker"
        segment_id = str(getattr(segment, "segment_id", "")).strip()
        self._logger.warning("Question normalization failed: %s", message)
        if self._window:
            self._window.set_question_follow_up_suggestions([])
            self._window.transcript_panel.set_segment_normalization_busy(
                segment,
                False,
                "Normalization failed",
            )

        if initial_recording_question:
            self._initial_recording_question_pending = False
            self._initial_recording_question_consumed = True

        if auto_send:
            fallback_question = clean_text(str(getattr(segment, "text", "")))
            if not fallback_question:
                if self._window:
                    self._window.controls.set_status(
                        "Question normalization failed", "#f85149"
                    )
                return

            fallback_item = None
            if segment_id:
                fallback_item = self._update_normalized_segment_item(
                    segment_id,
                    text=fallback_question,
                )
            if fallback_item is None:
                pending_segment_id = self._find_pending_normalized_segment_by_text(
                    fallback_question,
                )
                if pending_segment_id:
                    segment_id = pending_segment_id
                    fallback_item = self._update_normalized_segment_item(
                        segment_id,
                        text=fallback_question,
                    )
            if fallback_item is not None:
                self._sync_normalized_segments_to_window()

            # Always promote the fallback question to the top Question area
            # and trigger an AI response — no queuing for subsequent questions.
            self._session_context_store.append_event(
                "auto_ai_request",
                fallback_question,
                source=segment_source,
                metadata={"normalization_error": str(message or "")[:120]},
            )
            if fallback_item is not None:
                fallback_item["status"] = "sent"
                self._sync_normalized_segments_to_window()
            self._auto_primary_question_sent = True
            self._active_primary_question_text = fallback_question
            trigger_service = getattr(self, "_ai_trigger_service", None)
            if trigger_service is not None:
                trigger_service.mark_first_question_sent()
            self._prime_ai_context_with_question(segment, fallback_question)
            if self._window:
                self._window.set_question_lock_enabled(True)
                self._window.set_current_question_text(fallback_question, force=True)
                self._window.controls.set_status(
                    "Normalization failed. Generating from transcript…", "#f0883e"
                )
                self._window.ai_panel.show_thinking("Generating response…")
            self._generate_ai_response(force_follow_up=False)
            return

        if self._window:
            self._window.controls.set_status(
                "Question normalization failed", "#f85149"
            )

    def _on_speaker_send_requested(self, segment, text: str) -> None:
        from ghostmic.utils.text_processing import clean_text

        if not self._segment_is_question_source(segment):
            return

        segment_source = str(getattr(segment, "source", "")).strip().lower() or "speaker"
        question = clean_text(str(text or ""))
        if not question:
            if self._window:
                self._window.controls.set_status("Question is empty", "#f0883e")
            return

        self._auto_speaker_silence_generation += 1
        self._mark_auto_speaker_triggered(segment, question)

        with self._transcript_lock:
            segment.text = question
        self._prime_ai_context_with_question(segment, question)

        self._auto_primary_question_sent = True
        self._active_primary_question_text = question
        trigger_service = getattr(self, "_ai_trigger_service", None)
        if trigger_service is not None:
            trigger_service.mark_first_question_sent()
        self._pop_queued_normalized_question(question)

        normalized_segment_id = self._find_pending_normalized_segment_by_text(question)
        if normalized_segment_id:
            lookup = getattr(self, "_normalized_segment_lookup", {})
            item = lookup.get(normalized_segment_id)
            if item is not None:
                item["status"] = "sent"
            self._sync_normalized_segments_to_window()

        self._session_context_store.append_event(
            "manual_ai_request",
            question,
            source=segment_source,
        )

        if self._window:
            self._window.transcript_panel.set_segment_text(segment, question)
            self._window.transcript_panel.set_segment_normalization_busy(
                segment,
                False,
                "",
            )
            self._window.set_question_lock_enabled(True)
            self._window.set_current_question_text(question, force=True)
            self._window.controls.set_status(
                "Sending selected question to AI…", "#58a6ff"
            )
            self._window.ai_panel.show_thinking("Generating response…")

        self._generate_ai_response(force_follow_up=False)

    def _generate_ai_response(self, force_follow_up: bool = False) -> None:
        from ghostmic.core.ai_engine import AIThread, MAX_CONTEXT_SEGMENTS
        from ghostmic.core.transcription_engine import TranscriptSegment

        runtime_context_tail = self._session_context_store.get_latest_organized_context(
            max_chars=1200,
        )
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

            if len(transcript_snapshot) < MAX_CONTEXT_SEGMENTS:
                source_payload = self._session_context_store.build_compaction_source(
                    max_events=20,
                    max_chars=1200,
                )
                raw_event_tail = str(source_payload.get("text", "")).strip()
                if raw_event_tail:
                    sections = []
                    if runtime_context_tail:
                        sections.append(
                            "Organized Snapshot:\n"
                            f"{runtime_context_tail}"
                        )
                    sections.append(
                        "Recent Raw Events:\n"
                        f"{raw_event_tail}"
                    )
                    runtime_context_tail = "\n\n".join(sections).strip()

            if runtime_context_tail:
                self._logger.debug(
                    "AI request: runtime context attached (%d chars).",
                    len(runtime_context_tail),
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

            active_screen_context = self._resolve_active_screen_summary_context(
                latest_speaker_text,
                force_follow_up=force_follow_up,
                topic_shift_classifier=AIThread.classify_topic_shift,
            )
            if active_screen_context:
                section = f"Active Screen Context:\n{active_screen_context}"
                runtime_context_tail = (
                    f"{section}\n\n{runtime_context_tail}".strip()
                    if runtime_context_tail
                    else section
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

        accepted = self._ai_thread.request_response(
            transcript_snapshot,
            is_new_topic=is_new_topic,
            runtime_context_tail=runtime_context_tail,
        )
        if accepted:
            self._logger.info("AI request accepted and queued for generation.")
            if self._window:
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
                self._active_screen_summary_text = ""
                self._active_screen_anchor_question = ""
            self._reset_auto_question_session_state()
            if self._is_streaming_normalization_enabled():
                self._reset_streaming_normalization_state(clear_ui=True)
            self._auto_speaker_silence_generation += 1
            self._auto_speaker_last_signature = ""
            self._auto_speaker_last_trigger_ts = 0.0
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
            self._reset_auto_question_session_state()
            if self._is_streaming_normalization_enabled():
                self._reset_streaming_normalization_state(clear_ui=True)
            self._window.transcript_panel.clear_transcript()
            self._window.set_current_question_text("", force=True)
            self._window.set_question_follow_up_suggestions([])
            with self._transcript_lock:
                self._transcript_history.clear()
                self._ai_context_history.clear()
                self._last_ai_response_text = ""
                self._active_screen_summary_text = ""
                self._active_screen_anchor_question = ""
            self._auto_speaker_silence_generation += 1
            self._auto_speaker_last_signature = ""
            self._auto_speaker_last_trigger_ts = 0.0
            self._window.controls.set_status("Transcript cleared", "#3fb950")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _on_quit(self) -> None:
        self._logger.info("GhostMic: shutting down …")
        from PyQt6.QtWidgets import QApplication

        QApplication.restoreOverrideCursor()
        if self._question_normalization_worker and self._question_normalization_worker.isRunning():
            self._question_normalization_worker.requestInterruption()
            self._question_normalization_worker.wait(1500)
        if self._follow_up_suggestion_worker and self._follow_up_suggestion_worker.isRunning():
            self._follow_up_suggestion_worker.requestInterruption()
            self._follow_up_suggestion_worker.wait(1500)
        self._stop_streaming_processing_loop(flush=False)
        self._stop_audio_capture()
        self._session_context_compactor.stop(timeout=1.5)

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

        self._session_context_store.cleanup(delete_file=True)

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
    _install_uncaught_excepthook(logger)
    _enable_faulthandler_trace(logger)
    _write_startup_trace(
        "main.start",
        pid=os.getpid(),
        executable=sys.executable,
        frozen=getattr(sys, "frozen", False),
        argv=" ".join(sys.argv),
        config=args.config,
    )

    logger.info(
        "Diagnostics paths: app_log=%s startup_trace=%s faulthandler=%s",
        get_log_file_path(),
        STARTUP_TRACE_FILE,
        FAULT_TRACE_FILE,
    )

    try:
        from PyQt6 import QtCore

        logger.info(
            "Qt runtime versions: PyQt=%s Qt=%s",
            getattr(QtCore, "PYQT_VERSION_STR", "unknown"),
            getattr(QtCore, "QT_VERSION_STR", "unknown"),
        )
        _write_startup_trace(
            "main.qt_versions",
            pyqt=getattr(QtCore, "PYQT_VERSION_STR", "unknown"),
            qt=getattr(QtCore, "QT_VERSION_STR", "unknown"),
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("Could not read Qt runtime versions: %s", exc)

    if sys.platform == "win32":
        supports_capture_exclusion = _check_windows_version()
        logger.info(
            "Windows version check: WDA_EXCLUDEFROMCAPTURE supported = %s",
            supports_capture_exclusion,
        )
        _write_startup_trace(
            "main.windows_version_check",
            supports_capture_exclusion=supports_capture_exclusion,
        )
    else:
        logger.info("Running on %s — stealth features are Windows-only.", sys.platform)
        _write_startup_trace("main.non_windows", platform=sys.platform)

    if getattr(sys, "frozen", False):
        logger.info(
            "Frozen runtime context: executable=%s _MEIPASS=%s",
            sys.executable,
            getattr(sys, "_MEIPASS", ""),
        )
        _write_startup_trace(
            "main.frozen_context",
            executable=sys.executable,
            meipass=getattr(sys, "_MEIPASS", ""),
        )

    _write_startup_trace("main.create_app.begin")
    app = GhostMicApp(args)
    _write_startup_trace("main.create_app.ok")

    # Torch preload is intentionally deferred to optional/background paths.
    # Some packaged environments hard-fail during eager torch import.
    _write_startup_trace("main.torch_preload.skipped", reason="deferred")
    logger.info("Torch eager preload skipped at startup for stability.")

    _write_startup_trace("main.app_run.begin")
    exit_code = app.run()
    _write_startup_trace("main.app_run.end", exit_code=exit_code)
    logger.info("Application exiting with code %s", exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
