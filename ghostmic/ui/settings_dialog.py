"""
Settings dialog with tabs for General, Audio, AI, Appearance, About.
Stealth (WDA_EXCLUDEFROMCAPTURE) is applied to this window too.
"""

from __future__ import annotations

import copy
import sys
from typing import Any, Dict

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ghostmic.ui.styles import ACCENT_BLUE, MAIN_STYLE
from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)


class ApiConnectivityWorker(QThread):
    """Runs API connectivity checks in a background thread."""

    result_ready = pyqtSignal(bool, str, str)

    def __init__(self, ai_config: dict, parent=None) -> None:
        super().__init__(parent)
        self._ai_config = ai_config

    def run(self) -> None:
        try:
            from ghostmic.core.ai_engine import AIThread

            ai_test = AIThread(self._ai_config)
            success, backend, message = ai_test.test_api_connectivity()
            self.result_ready.emit(bool(success), str(backend), str(message))
        except Exception as exc:  # pylint: disable=broad-except
            self.result_ready.emit(False, "unknown", f"Error testing connection: {exc}")


class SettingsDialog(QDialog):
    """Multi-tab settings dialog.

    Signals:
        settings_saved(dict): Full updated config dict when OK is clicked.
    """

    settings_saved = pyqtSignal(dict)

    def __init__(self, config: dict, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._expose_openai_provider = bool(
            self._config.get("ai", {}).get("expose_openai_provider", False)
        )
        self._api_test_worker: ApiConnectivityWorker | None = None
        self.setWindowTitle("GhostMic — Settings")
        self.setMinimumSize(520, 460)
        self.setStyleSheet(MAIN_STYLE)
        self._build_ui()
        self._load_values()
        # Apply stealth after the dialog has a valid HWND
        QTimer.singleShot(100, self._apply_stealth)

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        tabs = QTabWidget()
        tabs.addTab(self._make_general_tab(), "General")
        tabs.addTab(self._make_audio_tab(), "Audio")
        tabs.addTab(self._make_ai_tab(), "AI")
        tabs.addTab(self._make_appearance_tab(), "Appearance")
        tabs.addTab(self._make_about_tab(), "About")
        layout.addWidget(tabs)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ── Tab: General ──────────────────────────────────────────────────

    def _make_general_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self._lang_combo = QComboBox()
        self._lang_combo.addItems(["en", "es", "fr", "de", "zh", "ja", "auto"])
        form.addRow("Language:", self._lang_combo)

        self._hotkey_toggle = QLineEdit()
        form.addRow("Toggle recording hotkey:", self._hotkey_toggle)

        self._hotkey_window = QLineEdit()
        form.addRow("Toggle window hotkey:", self._hotkey_window)

        self._hotkey_ai = QLineEdit()
        form.addRow("Generate AI response hotkey:", self._hotkey_ai)

        self._hotkey_copy = QLineEdit()
        form.addRow("Copy response hotkey:", self._hotkey_copy)

        self._hotkey_clear = QLineEdit()
        form.addRow("Clear transcript hotkey:", self._hotkey_clear)

        self._hotkey_dictation = QLineEdit()
        form.addRow("Win+H dictation hotkey:", self._hotkey_dictation)

        self._dictation_enabled = QCheckBox("Enable Win+H dictation capture")
        form.addRow("Dictation:", self._dictation_enabled)

        self._dictation_idle_spin = QSpinBox()
        self._dictation_idle_spin.setRange(300, 5000)
        self._dictation_idle_spin.setSingleStep(100)
        self._dictation_idle_spin.setSuffix(" ms")
        form.addRow("Dictation commit idle:", self._dictation_idle_spin)

        return w

    # ── Tab: Audio ────────────────────────────────────────────────────

    def _make_audio_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self._input_combo = QComboBox()
        self._refresh_input_devices()
        btn_refresh = QPushButton("↻ Refresh")
        btn_refresh.setFixedWidth(80)
        btn_refresh.clicked.connect(self._refresh_input_devices)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self._input_combo)
        rl.addWidget(btn_refresh)
        form.addRow("Microphone:", row)

        self._loopback_combo = QComboBox()
        self._refresh_loopback_devices()
        btn_lb = QPushButton("↻ Refresh")
        btn_lb.setFixedWidth(80)
        btn_lb.clicked.connect(self._refresh_loopback_devices)
        row2 = QWidget()
        rl2 = QHBoxLayout(row2)
        rl2.setContentsMargins(0, 0, 0, 0)
        rl2.addWidget(self._loopback_combo)
        rl2.addWidget(btn_lb)
        form.addRow("System audio (loopback):", row2)

        self._model_combo = QComboBox()
        self._model_combo.addItems(
            ["tiny.en", "base.en", "small.en", "medium.en", "large-v3"]
        )
        form.addRow("Whisper model:", self._model_combo)

        self._compute_combo = QComboBox()
        self._compute_combo.addItems(["int8", "float16", "float32"])
        form.addRow("Compute type:", self._compute_combo)

        return w

    # ── Tab: AI ───────────────────────────────────────────────────────

    def _make_ai_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self._backend_combo = QComboBox()
        if self._expose_openai_provider:
            self._backend_combo.addItems(["groq", "openai"])
            self._backend_combo.setEnabled(True)
            form.addRow("Backend:", self._backend_combo)
        else:
            self._backend_combo.addItems(["groq"])
            self._backend_combo.setEnabled(False)
            form.addRow("Backend (fixed):", self._backend_combo)

        self._openai_api_key_edit = QLineEdit()
        self._openai_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._openai_api_key_edit.setPlaceholderText(
            "Get your key at platform.openai.com"
        )
        self._openai_key_toggle = QPushButton("👁")
        self._openai_key_toggle.setFixedSize(28, 28)
        self._openai_key_toggle.setCheckable(True)
        self._openai_key_toggle.setToolTip("Show/hide API key")
        self._openai_key_toggle.setStyleSheet("QPushButton { background: transparent; border: none; }")
        self._openai_key_toggle.toggled.connect(
            lambda checked: self._openai_api_key_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )

        self._openai_model_combo = QComboBox()
        self._openai_model_combo.addItems(["gpt-5-mini", "gpt-5-nano", "gpt-5.4-nano"])

        if self._expose_openai_provider:
            openai_key_row = QWidget()
            openai_key_layout = QHBoxLayout(openai_key_row)
            openai_key_layout.setContentsMargins(0, 0, 0, 0)
            openai_key_layout.addWidget(self._openai_api_key_edit)
            openai_key_layout.addWidget(self._openai_key_toggle)
            form.addRow("OpenAI API key:", openai_key_row)
            form.addRow("OpenAI model:", self._openai_model_combo)

        self._groq_api_key_edit = QLineEdit()
        self._groq_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._groq_api_key_edit.setPlaceholderText("Get a free key at console.groq.com")
        self._groq_key_toggle = QPushButton("👁")
        self._groq_key_toggle.setFixedSize(28, 28)
        self._groq_key_toggle.setCheckable(True)
        self._groq_key_toggle.setToolTip("Show/hide API key")
        self._groq_key_toggle.setStyleSheet("QPushButton { background: transparent; border: none; }")
        self._groq_key_toggle.toggled.connect(
            lambda checked: self._groq_api_key_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        groq_key_row = QWidget()
        groq_key_layout = QHBoxLayout(groq_key_row)
        groq_key_layout.setContentsMargins(0, 0, 0, 0)
        groq_key_layout.addWidget(self._groq_api_key_edit)
        groq_key_layout.addWidget(self._groq_key_toggle)
        form.addRow("Groq API key:", groq_key_row)

        self._groq_model_combo = QComboBox()
        self._groq_model_combo.addItems(
            [
                "llama-3.3-70b-versatile",
                "llama-3.1-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
            ]
        )
        form.addRow("Groq model:", self._groq_model_combo)

        self._trigger_combo = QComboBox()
        self._trigger_combo.addItems(["auto", "manual", "continuous"])
        form.addRow("Trigger mode:", self._trigger_combo)

        self._temp_slider = QSlider(Qt.Orientation.Horizontal)
        self._temp_slider.setRange(0, 100)
        self._temp_slider.setValue(70)
        self._temp_label = QLabel("0.7")
        self._temp_slider.valueChanged.connect(
            lambda v: self._temp_label.setText(f"{v / 100:.1f}")
        )
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self._temp_slider)
        rl.addWidget(self._temp_label)
        form.addRow("Temperature:", row)

        self._system_prompt = QTextEdit()
        self._system_prompt.setFixedHeight(100)
        form.addRow("System prompt:", self._system_prompt)

        self._session_ctx = QLineEdit()
        self._session_ctx.setPlaceholderText(
            "e.g. Python developer interview at Google"
        )
        form.addRow("Session context:", self._session_ctx)

        # Test connection button
        self._test_btn = QPushButton("Test API Connection")
        self._test_btn.clicked.connect(self._test_api_connection)
        form.addRow("", self._test_btn)

        self._test_result_label = QLabel("")
        self._test_result_label.setWordWrap(True)
        form.addRow("", self._test_result_label)

        return w

    # ── Tab: Appearance ───────────────────────────────────────────────

    def _make_appearance_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(30, 100)
        self._opacity_slider.setValue(95)
        self._opacity_label = QLabel("95%")
        self._opacity_slider.valueChanged.connect(
            lambda v: self._opacity_label.setText(f"{v}%")
        )
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self._opacity_slider)
        rl.addWidget(self._opacity_label)
        form.addRow("Opacity:", row)

        self._font_spin = QSpinBox()
        self._font_spin.setRange(8, 20)
        self._font_spin.setValue(11)
        form.addRow("Font size:", self._font_spin)

        self._width_spin = QSpinBox()
        self._width_spin.setRange(300, 1200)
        self._width_spin.setValue(420)
        form.addRow("Window width:", self._width_spin)

        self._height_spin = QSpinBox()
        self._height_spin.setRange(300, 1200)
        self._height_spin.setValue(650)
        form.addRow("Window height:", self._height_spin)

        return w

    # ── Tab: About ────────────────────────────────────────────────────

    def _make_about_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("<b>GhostMic</b> — Real-time AI meeting assistant"))
        layout.addWidget(QLabel("Version 1.0.0"))
        link = QLabel(
            '<a href="https://github.com/sindhub2025/Interview-Tool" '
            'style="color: #58a6ff;">GitHub Repository</a>'
        )
        link.setOpenExternalLinks(True)
        layout.addWidget(link)
        layout.addWidget(
            QLabel(
                "Invisible to screen-capture (WDA_EXCLUDEFROMCAPTURE).\n"
                "Your conversations are processed locally or via your own API keys."
            )
        )
        layout.addStretch()
        return w

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def _load_values(self) -> None:
        ai = self._config.get("ai", {})
        audio = self._config.get("audio", {})
        transcription = self._config.get("transcription", {})
        ui = self._config.get("ui", {})
        hotkeys = self._config.get("hotkeys", {})

        # General
        lang = transcription.get("language", "en")
        idx = self._lang_combo.findText(lang)
        if idx >= 0:
            self._lang_combo.setCurrentIndex(idx)
        self._hotkey_toggle.setText(hotkeys.get("toggle_recording", "ctrl+shift+g"))
        self._hotkey_window.setText(hotkeys.get("toggle_window", "ctrl+shift+h"))
        self._hotkey_ai.setText(hotkeys.get("generate_response", "ctrl+g"))
        self._hotkey_copy.setText(hotkeys.get("copy_response", "ctrl+shift+c"))
        self._hotkey_clear.setText(hotkeys.get("clear_transcript", "ctrl+shift+x"))
        self._hotkey_dictation.setText(hotkeys.get("win_h_dictation", "win+h"))

        dictation = self._config.get("dictation", {})
        self._dictation_enabled.setChecked(bool(dictation.get("enabled", True)))
        self._dictation_idle_spin.setValue(int(dictation.get("commit_idle_ms", 1200)))

        # Audio
        model_size = transcription.get("model_size", "base.en")
        m_idx = self._model_combo.findText(model_size)
        if m_idx >= 0:
            self._model_combo.setCurrentIndex(m_idx)
        compute = transcription.get("compute_type", "int8")
        c_idx = self._compute_combo.findText(compute)
        if c_idx >= 0:
            self._compute_combo.setCurrentIndex(c_idx)

        # AI
        backend = ai.get("main_backend") or ai.get("backend", "groq")
        if not self._expose_openai_provider and backend != "groq":
            backend = "groq"
        b_idx = self._backend_combo.findText(backend)
        if b_idx >= 0:
            self._backend_combo.setCurrentIndex(b_idx)

        self._openai_api_key_edit.setText(ai.get("openai_api_key", ""))
        openai_model = ai.get("openai_model", "gpt-5-mini")
        openai_model_idx = self._openai_model_combo.findText(openai_model)
        if openai_model_idx >= 0:
            self._openai_model_combo.setCurrentIndex(openai_model_idx)

        self._groq_api_key_edit.setText(ai.get("groq_api_key", ""))
        gm = ai.get("groq_model", "llama-3.3-70b-versatile")
        gm_idx = self._groq_model_combo.findText(gm)
        if gm_idx >= 0:
            self._groq_model_combo.setCurrentIndex(gm_idx)
        trigger = ai.get("trigger_mode", "auto")
        t_idx = self._trigger_combo.findText(trigger)
        if t_idx >= 0:
            self._trigger_combo.setCurrentIndex(t_idx)
        self._temp_slider.setValue(int(ai.get("temperature", 0.7) * 100))
        from ghostmic.core.ai_engine import DEFAULT_SYSTEM_PROMPT
        self._system_prompt.setPlainText(
            ai.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        )
        self._session_ctx.setText(ai.get("session_context", ""))

        # Appearance
        self._opacity_slider.setValue(int(ui.get("opacity", 0.95) * 100))
        self._font_spin.setValue(ui.get("font_size", 11))
        self._width_spin.setValue(ui.get("window_width", 420))
        self._height_spin.setValue(ui.get("window_height", 650))

    def _save(self) -> None:
        cfg = dict(self._config)

        # General / hotkeys
        cfg.setdefault("hotkeys", {})
        cfg["hotkeys"]["toggle_recording"] = self._hotkey_toggle.text()
        cfg["hotkeys"]["toggle_window"] = self._hotkey_window.text()
        cfg["hotkeys"]["generate_response"] = self._hotkey_ai.text()
        cfg["hotkeys"]["copy_response"] = self._hotkey_copy.text()
        cfg["hotkeys"]["clear_transcript"] = self._hotkey_clear.text()
        cfg["hotkeys"]["win_h_dictation"] = self._hotkey_dictation.text()

        cfg.setdefault("dictation", {})
        cfg["dictation"]["enabled"] = self._dictation_enabled.isChecked()
        cfg["dictation"]["commit_idle_ms"] = self._dictation_idle_spin.value()

        # Transcription
        cfg.setdefault("transcription", {})
        cfg["transcription"]["language"] = self._lang_combo.currentText()
        cfg["transcription"]["model_size"] = self._model_combo.currentText()
        cfg["transcription"]["compute_type"] = self._compute_combo.currentText()

        # AI
        cfg.setdefault("ai", {})
        cfg["ai"]["expose_openai_provider"] = self._expose_openai_provider

        selected_backend = (
            self._backend_combo.currentText()
            if self._expose_openai_provider
            else "groq"
        )
        cfg["ai"]["backend"] = selected_backend
        cfg["ai"]["main_backend"] = selected_backend
        if not self._expose_openai_provider:
            cfg["ai"]["fallback_backend"] = "groq"
            cfg["ai"]["enable_fallback"] = False

        if self._expose_openai_provider:
            cfg["ai"]["openai_api_key"] = self._openai_api_key_edit.text()
            cfg["ai"]["openai_model"] = self._openai_model_combo.currentText()

        cfg["ai"]["groq_api_key"] = self._groq_api_key_edit.text()
        cfg["ai"]["groq_model"] = self._groq_model_combo.currentText()
        cfg["ai"].pop("ollama_model", None)
        cfg["ai"].pop("ollama_url", None)
        cfg["ai"]["trigger_mode"] = self._trigger_combo.currentText()
        cfg["ai"]["temperature"] = self._temp_slider.value() / 100.0
        cfg["ai"]["system_prompt"] = self._system_prompt.toPlainText()
        cfg["ai"]["session_context"] = self._session_ctx.text()

        # Appearance / UI
        cfg.setdefault("ui", {})
        cfg["ui"]["opacity"] = self._opacity_slider.value() / 100.0
        cfg["ui"]["font_size"] = self._font_spin.value()
        cfg["ui"]["window_width"] = self._width_spin.value()
        cfg["ui"]["window_height"] = self._height_spin.value()

        self.settings_saved.emit(cfg)
        self.accept()

    def _test_api_connection(self) -> None:
        """Test the API connection with the currently configured settings."""
        if self._api_test_worker is not None and self._api_test_worker.isRunning():
            return

        self._test_result_label.setText("Testing API connection...")
        self._test_result_label.setStyleSheet("color: #888;")
        self._test_btn.setEnabled(False)

        # Build a temporary config with current dialog values
        test_config = copy.deepcopy(self._config)
        test_config.setdefault("ai", {})
        test_config["ai"]["expose_openai_provider"] = self._expose_openai_provider

        if self._expose_openai_provider:
            test_config["ai"]["openai_api_key"] = self._openai_api_key_edit.text()
            test_config["ai"]["openai_model"] = self._openai_model_combo.currentText()

        test_config["ai"]["groq_api_key"] = self._groq_api_key_edit.text()
        test_config["ai"]["groq_model"] = self._groq_model_combo.currentText()
        selected_backend = (
            self._backend_combo.currentText()
            if self._expose_openai_provider
            else "groq"
        )
        test_config["ai"]["backend"] = selected_backend
        test_config["ai"]["main_backend"] = selected_backend
        if not self._expose_openai_provider:
            test_config["ai"]["fallback_backend"] = "groq"
            test_config["ai"]["enable_fallback"] = False

        ai_test_config = test_config.get("ai")
        if not isinstance(ai_test_config, dict):
            self._test_result_label.setText("✗ Missing AI settings in config")
            self._test_result_label.setStyleSheet("color: #ef4444;")
            self._test_btn.setEnabled(True)
            return

        self._api_test_worker = ApiConnectivityWorker(copy.deepcopy(ai_test_config), self)
        self._api_test_worker.result_ready.connect(self._on_api_test_result)
        self._api_test_worker.finished.connect(self._on_api_test_finished)
        self._api_test_worker.start()

    def _on_api_test_result(self, success: bool, backend: str, message: str) -> None:
        if success:
            self._test_result_label.setText(
                f"✓ Connected to {backend.title()}!\n{message[:200]}"
            )
            self._test_result_label.setStyleSheet("color: #4ade80;")
            return

        self._test_result_label.setText(
            f"✗ Connection Failed ({backend}):\n{message[:200]}"
        )
        self._test_result_label.setStyleSheet("color: #ef4444;")

    def _on_api_test_finished(self) -> None:
        self._test_btn.setEnabled(True)
        if self._api_test_worker is None:
            return
        self._api_test_worker.deleteLater()
        self._api_test_worker = None

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    def _refresh_input_devices(self) -> None:
        self._input_combo.clear()
        self._input_combo.addItem("Default", None)
        try:
            from ghostmic.core.audio_capture import list_input_devices
            for dev in list_input_devices():
                self._input_combo.addItem(dev["name"], dev["index"])
        except Exception:  # pylint: disable=broad-except
            pass

    def _refresh_loopback_devices(self) -> None:
        self._loopback_combo.clear()
        self._loopback_combo.addItem("Default (WASAPI loopback)", None)
        try:
            from ghostmic.core.audio_capture import list_loopback_devices
            for dev in list_loopback_devices():
                self._loopback_combo.addItem(dev["name"], dev["index"])
        except Exception:  # pylint: disable=broad-except
            pass

    # ------------------------------------------------------------------
    # Stealth
    # ------------------------------------------------------------------

    def _apply_stealth(self) -> None:
        if sys.platform != "win32":
            return
        try:
            from ghostmic.core.stealth import apply_stealth
            hwnd = int(self.winId())
            apply_stealth(hwnd)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("SettingsDialog: could not apply stealth: %s", exc)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if sys.platform == "win32":
            from PyQt6.QtCore import QEvent
            if event.type() == QEvent.Type.WindowStateChange:
                QTimer.singleShot(100, self._apply_stealth)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(100, self._apply_stealth)
