"""
Controls bar: record toggle, mode dropdown, settings gear, status label.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from ghostmic.ui.styles import ACCENT_GREEN, ACCENT_RED, TEXT_SECONDARY


class APIStatusIndicator(QWidget):
    """Small API connection status indicator."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(140, 20)
        self._connected = False
        self._backend = "OpenAI"

    def set_status(self, connected: bool, backend: str = "OpenAI") -> None:
        """Update the connection status."""
        self._connected = connected
        self._backend = backend
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Dot color
        dot_color = ACCENT_GREEN if self._connected else "#666666"
        painter.fillRect(4, 6, 8, 8, QColor(dot_color))

        # Text
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
    """

    record_toggled = pyqtSignal(bool)
    mode_changed = pyqtSignal(str)
    settings_requested = pyqtSignal()

    MODES = ["Interview", "Meeting Notes", "Custom"]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._recording = False
        self._build_ui()

    def _build_ui(self) -> None:
        self.setFixedHeight(50)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Record button
        self._record_btn = QPushButton("⏺")
        self._record_btn.setObjectName("record_btn")
        self._record_btn.setFixedSize(36, 36)
        self._record_btn.setCheckable(True)
        self._record_btn.setToolTip("Start/Stop recording (Ctrl+Shift+G)")
        self._record_btn.clicked.connect(self._on_record_clicked)
        layout.addWidget(self._record_btn)

        # API Status indicator
        self._api_status = APIStatusIndicator()
        layout.addWidget(self._api_status)

        # Mode dropdown
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(self.MODES)
        self._mode_combo.setFixedWidth(130)
        self._mode_combo.currentTextChanged.connect(self.mode_changed)
        layout.addWidget(self._mode_combo)

        layout.addStretch()

        # Status label
        self._status_label = QLabel("Ready")
        self._status_label.setObjectName("status_label")
        self._status_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 9pt;"
        )
        layout.addWidget(self._status_label)

        layout.addStretch()

        # Settings button
        settings_btn = QPushButton("⚙")
        settings_btn.setObjectName("settings_btn")
        settings_btn.setFixedSize(32, 32)
        settings_btn.setToolTip("Settings")
        settings_btn.clicked.connect(self._on_settings_clicked)
        layout.addWidget(settings_btn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        """Return whether recording is currently active."""
        return self._recording

    def set_recording(self, recording: bool) -> None:
        """Sync button state with actual recording status."""
        self._recording = recording
        self._record_btn.setChecked(recording)
        if recording:
            self._record_btn.setText("⏹")
            self._record_btn.setStyleSheet(
                f"background-color: #6e0000; border-radius: 18px; "
                f"color: white; font-size: 16pt; border: 2px solid {ACCENT_RED};"
            )
            self.set_status("Recording…", ACCENT_RED)
        else:
            self._record_btn.setText("⏺")
            self._record_btn.setStyleSheet("")
            self.set_status("Ready", TEXT_SECONDARY)

    def set_status(self, text: str, color: str = TEXT_SECONDARY) -> None:
        """Update the status label."""
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            f"color: {color}; font-size: 9pt;"
        )

    def set_api_status(self, connected: bool, backend: str = "OpenAI") -> None:
        """Update the API connection status indicator."""
        self._api_status.set_status(connected, backend)

    def current_mode(self) -> str:
        return self._mode_combo.currentText()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_record_clicked(self, checked: bool) -> None:
        self.set_recording(checked)
        self.record_toggled.emit(checked)

    def _on_settings_clicked(self) -> None:
        self.settings_requested.emit()
