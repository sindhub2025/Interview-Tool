"""
Controls bar: compact record, mic, capture, and settings actions.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from ghostmic.ui.styles import ACCENT_GREEN, ACCENT_RED, TEXT_SECONDARY


class APIStatusIndicator(QWidget):
    """Small API connection status indicator with accessible text."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(140, 20)
        self._connected = False
        self._backend = "Groq"
        self.setAccessibleName("API Status")
        self._update_accessible_description()

    def set_status(self, connected: bool, backend: str = "Groq") -> None:
        """Update the connection status."""
        self._connected = connected
        self._backend = backend
        self._update_accessible_description()
        self.update()

    def _update_accessible_description(self) -> None:
        if self._connected:
            self.setAccessibleDescription(f"Connected to {self._backend}")
        else:
            self.setAccessibleDescription("Offline, not connected")

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Dot color
        dot_color = ACCENT_GREEN if self._connected else "#666666"
        painter.fillRect(4, 6, 8, 8, QColor(dot_color))

        # Text (visible alongside dot)
        text = f"✓ {self._backend}" if self._connected else "○ Offline"
        text_color = ACCENT_GREEN if self._connected else TEXT_SECONDARY
        
        painter.setPen(QColor(text_color))
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(16, 15, text)


class ControlsBar(QWidget):
    """Top bar with recording controls.

    Signals:
        record_toggled(bool): Emitted when recording is toggled.
        mode_changed(str): Emitted when the mode dropdown changes.
        settings_requested(): Emitted when the settings button is clicked.
        screenshot_requested(): Emitted when the screen analysis button is clicked.
    """

    record_toggled = pyqtSignal(bool)
    mic_toggled = pyqtSignal(bool)
    mode_changed = pyqtSignal(str)
    settings_requested = pyqtSignal()
    screenshot_requested = pyqtSignal()

    MODES = ["Interview", "Meeting Notes", "Custom"]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._recording = False
        self._mic_enabled = False
        self._api_connected = False
        self._api_backend = "Groq"
        self._build_ui()

    def _build_ui(self) -> None:
        self.setObjectName("controls_bar")
        self.setFixedHeight(58)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(
            "QWidget#controls_bar {"
            " background-color: transparent;"
            " border: none;"
            "}"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        # Record button
        self._record_btn = QPushButton("Record")
        self._record_btn.setObjectName("record_btn")
        self._record_btn.setFixedSize(86, 38)
        self._record_btn.setCheckable(True)
        self._record_btn.clicked.connect(self._on_record_clicked)
        layout.addWidget(self._record_btn)

        # Optional microphone capture toggle (speaker-only by default)
        self._mic_btn = QPushButton("Mic Off")
        self._mic_btn.setObjectName("mic_btn")
        self._mic_btn.setFixedSize(78, 38)
        self._mic_btn.setCheckable(True)
        self._mic_btn.clicked.connect(self._on_mic_clicked)
        layout.addWidget(self._mic_btn)
        self._apply_mic_button_state()

        layout.addStretch(1)

        # API status, mode, and status text are kept as hidden state sinks so
        # controller calls remain UI-only and do not need to change.
        self._api_status = APIStatusIndicator(self)
        self._api_status.setVisible(False)

        self._mode_combo = QComboBox(self)
        self._mode_combo.addItems(self.MODES)
        self._mode_combo.setFixedWidth(130)
        self._mode_combo.currentTextChanged.connect(self.mode_changed)
        self._mode_combo.setVisible(False)

        self._status_label = QLabel("Ready", self)
        self._status_label.setObjectName("status_label")
        self._status_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 9pt;"
        )
        self._status_label.setVisible(False)

        # Screen analysis button
        self._screenshot_btn = QPushButton("Capture")
        self._screenshot_btn.setObjectName("screenshot_btn")
        self._screenshot_btn.setFixedSize(88, 38)
        self._screenshot_btn.clicked.connect(self._on_screenshot_clicked)
        layout.addWidget(self._screenshot_btn)

        # Settings button
        self._settings_btn = QPushButton("Settings")
        self._settings_btn.setObjectName("settings_btn")
        self._settings_btn.setFixedSize(86, 38)
        self._settings_btn.clicked.connect(self._on_settings_clicked)
        layout.addWidget(self._settings_btn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        """Return whether recording is currently active."""
        return self._recording

    @property
    def is_mic_enabled(self) -> bool:
        """Return whether microphone capture is enabled."""
        return self._mic_enabled

    def set_recording(self, recording: bool) -> None:
        """Sync button state with actual recording status."""
        self._recording = recording
        self._record_btn.setChecked(recording)
        if recording:
            self._record_btn.setText("Stop")
            self._record_btn.setStyleSheet(
                f"background-color: #5f1118; border-radius: 6px; "
                f"color: white; font-size: 10pt; font-weight: 700; "
                f"border: 1px solid {ACCENT_RED};"
            )
            self.set_status("Recording…", ACCENT_RED)
        else:
            self._record_btn.setText("Record")
            self._record_btn.setStyleSheet("")
            self.set_status("Ready", TEXT_SECONDARY)

    def set_mic_enabled(self, enabled: bool) -> None:
        """Sync microphone toggle state without re-emitting the signal."""
        self._mic_enabled = bool(enabled)
        prev = self._mic_btn.blockSignals(True)
        self._mic_btn.setChecked(self._mic_enabled)
        self._mic_btn.blockSignals(prev)
        self._apply_mic_button_state()

    def set_status(self, text: str, color: str = TEXT_SECONDARY) -> None:
        """Update the status label."""
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            f"color: {color}; font-size: 9pt;"
        )

    def set_api_status(self, connected: bool, backend: str = "Groq") -> None:
        """Update the API connection status indicator."""
        self._api_connected = connected
        self._api_backend = (backend.strip() if isinstance(backend, str) else backend) or "Groq"
        self._api_status.set_status(connected, backend)

    def set_screen_analysis_busy(self, busy: bool) -> None:
        """Disable the screenshot action while a request is in flight."""
        self._screenshot_btn.setEnabled(not busy)
        self._screenshot_btn.setText("Analyzing" if busy else "Capture")

    def current_mode(self) -> str:
        return self._mode_combo.currentText()

    def _apply_mic_button_state(self) -> None:
        if self._mic_enabled:
            self._mic_btn.setText("Mic On")
            self._mic_btn.setToolTip("Microphone capture enabled")
            self._mic_btn.setStyleSheet(
                "QPushButton {"
                " background-color: rgba(63, 185, 80, 0.20);"
                " border: 1px solid rgba(63, 185, 80, 0.65);"
                " border-radius: 6px;"
                " color: #d2ffd9;"
                " font-size: 10pt;"
                " font-weight: 600;"
                "}"
                "QPushButton:hover { background-color: rgba(63, 185, 80, 0.30); }"
            )
            return

        self._mic_btn.setText("Mic Off")
        self._mic_btn.setToolTip("Speaker-only capture (microphone ignored)")
        self._mic_btn.setStyleSheet(
            "QPushButton {"
            " background-color: rgba(139, 148, 158, 0.16);"
            " border: 1px solid rgba(139, 148, 158, 0.45);"
            " border-radius: 6px;"
            " color: #c9d1d9;"
            " font-size: 10pt;"
            " font-weight: 600;"
            "}"
            "QPushButton:hover { background-color: rgba(139, 148, 158, 0.24); }"
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_record_clicked(self, checked: bool) -> None:
        self.set_recording(checked)
        self.record_toggled.emit(checked)

    def _on_mic_clicked(self, checked: bool) -> None:
        self._mic_enabled = bool(checked)
        self._apply_mic_button_state()
        self.mic_toggled.emit(self._mic_enabled)

    def _on_settings_clicked(self) -> None:
        self.settings_requested.emit()

    def _on_screenshot_clicked(self) -> None:
        self.screenshot_requested.emit()
