"""
Controls bar: compact record, mic, capture, and settings actions.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QMouseEvent, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpacerItem,
    QSizePolicy,
    QWidget,
)

from ghostmic.ui.styles import ACCENT_GREEN, ACCENT_RED, TEXT_SECONDARY


class _CameraActionButton(QPushButton):
    """Existing control button with a consistent vector icon in Camera Mode."""

    def __init__(self, kind: str, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self._kind = kind
        self._camera_mode = False
        self.setMouseTracking(True)

    def set_camera_mode(self, active: bool) -> None:
        self._camera_mode = active
        self.update()

    def paintEvent(self, event) -> None:
        if not self._camera_mode:
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = self.rect().adjusted(2, 2, -2, -2)
        checked = self.isCheckable() and self.isChecked()
        if checked:
            fill = QColor(ACCENT_RED if self._kind == "record" else ACCENT_GREEN)
            fill.setAlpha(34)
            painter.setPen(QPen(Qt.PenStyle.NoPen))
            painter.setBrush(fill)
            painter.drawRoundedRect(bounds, 9, 9)
        elif self.underMouse():
            painter.setPen(QPen(Qt.PenStyle.NoPen))
            painter.setBrush(QColor(255, 255, 255, 18))
            painter.drawRoundedRect(bounds, 9, 9)
        if self.hasFocus():
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(125, 173, 220, 110), 1.2))
            painter.drawRoundedRect(bounds.adjusted(0, 0, -1, -1), 9, 9)

        color = QColor(
            "#657181" if not self.isEnabled() else
            ACCENT_RED if checked and self._kind == "record" else
            ACCENT_GREEN if checked and self._kind == "mic" else "#d7dee8"
        )
        painter.setPen(QPen(color, 1.8, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        cx, cy = self.width() / 2, self.height() / 2

        if self._kind == "mic":
            painter.drawRoundedRect(int(cx - 4), int(cy - 10), 8, 14, 4, 4)
            path = QPainterPath()
            path.moveTo(cx - 8, cy - 1)
            path.cubicTo(cx - 8, cy + 9, cx + 8, cy + 9, cx + 8, cy - 1)
            painter.drawPath(path)
            painter.drawLine(int(cx), int(cy + 8), int(cx), int(cy + 12))
            painter.drawLine(int(cx - 4), int(cy + 12), int(cx + 4), int(cy + 12))
            if not checked:
                painter.drawLine(int(cx - 10), int(cy - 10), int(cx + 10), int(cy + 10))
        elif self._kind == "record":
            painter.drawEllipse(int(cx - 8), int(cy - 8), 16, 16)
            if checked:
                painter.setPen(QPen(Qt.PenStyle.NoPen))
                painter.setBrush(color)
                painter.drawEllipse(int(cx - 4), int(cy - 4), 8, 8)
        elif self._kind == "capture":
            painter.drawRoundedRect(int(cx - 10), int(cy - 6), 20, 15, 3, 3)
            painter.drawRoundedRect(int(cx - 5), int(cy - 9), 9, 4, 1, 1)
            painter.drawEllipse(int(cx - 4), int(cy - 3), 8, 8)
            painter.drawEllipse(int(cx - 1), int(cy), 2, 2)
        else:
            painter.drawEllipse(int(cx - 3), int(cy - 3), 6, 6)
            painter.drawEllipse(int(cx - 7), int(cy - 7), 14, 14)
            for angle in range(0, 360, 45):
                painter.save()
                painter.translate(cx, cy)
                painter.rotate(angle)
                painter.drawLine(0, -8, 0, -10)
                painter.restore()

    def enterEvent(self, event) -> None:
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.update()
        super().leaveEvent(event)


class _CameraDragHandle(QWidget):
    """Small grab surface for manually aligning the compact window."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._fallback_offset = None
        self.setFixedSize(18, 34)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip("Drag to align below your camera")
        self.setAccessibleName("Move Camera Mode window")

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor("#64748b" if not self.underMouse() else "#a9c9e8")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        for x in (6, 11):
            for y in (10, 15, 20, 25):
                painter.drawEllipse(x, y, 2, 2)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            window = self.window()
            handle = window.windowHandle()
            self._fallback_offset = event.globalPosition().toPoint() - window.frameGeometry().topLeft()
            if handle is not None and handle.startSystemMove():
                self._fallback_offset = None
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._fallback_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.window().move(event.globalPosition().toPoint() - self._fallback_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._fallback_offset = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().mouseReleaseEvent(event)


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
        self._camera_mode = False
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
        self._record_btn = _CameraActionButton("record", "Record")
        self._record_btn.setObjectName("record_btn")
        self._record_btn.setFixedSize(86, 38)
        self._record_btn.setCheckable(True)
        self._record_btn.clicked.connect(self._on_record_clicked)
        layout.addWidget(self._record_btn)

        # Optional microphone capture toggle (speaker-only by default)
        self._mic_btn = _CameraActionButton("mic", "Mic Off")
        self._mic_btn.setObjectName("mic_btn")
        self._mic_btn.setFixedSize(78, 38)
        self._mic_btn.setCheckable(True)
        self._mic_btn.clicked.connect(self._on_mic_clicked)
        layout.addWidget(self._mic_btn)
        self._camera_drag_handle = _CameraDragHandle(self)
        self._camera_drag_handle.hide()
        layout.addWidget(self._camera_drag_handle)
        self._apply_mic_button_state()

        self._controls_spacer = QSpacerItem(
            0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        layout.addItem(self._controls_spacer)

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
        self._screenshot_btn = _CameraActionButton("capture", "Capture")
        self._screenshot_btn.setObjectName("screenshot_btn")
        self._screenshot_btn.setFixedSize(88, 38)
        self._screenshot_btn.clicked.connect(self._on_screenshot_clicked)
        layout.addWidget(self._screenshot_btn)

        # Settings button
        self._settings_btn = _CameraActionButton("settings", "Settings")
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
        self.set_status(
            "Recording…" if recording else "Ready",
            ACCENT_RED if recording else TEXT_SECONDARY,
        )
        self._record_btn.setAccessibleDescription(
            "Recording is active" if recording else "Recording is stopped"
        )
        if self._camera_mode:
            self._record_btn.setToolTip("")
        if self._camera_mode:
            self._record_btn.setStyleSheet("")
        elif recording:
            self._record_btn.setText("Stop")
            self._record_btn.setStyleSheet(
                f"background-color: #5f1118; border-radius: 6px; "
                f"color: white; font-size: 10pt; font-weight: 700; "
                f"border: 1px solid {ACCENT_RED};"
            )
        else:
            self._record_btn.setText("Record")
            self._record_btn.setStyleSheet("")
        self._record_btn.update()

    def set_mic_enabled(self, enabled: bool) -> None:
        """Sync microphone toggle state without re-emitting the signal."""
        self._mic_enabled = bool(enabled)
        prev = self._mic_btn.blockSignals(True)
        self._mic_btn.setChecked(self._mic_enabled)
        self._mic_btn.blockSignals(prev)
        if not self._camera_mode:
            self._apply_mic_button_state()
        self._mic_btn.setAccessibleDescription(
            "Microphone is on" if self._mic_enabled else "Microphone is off"
        )
        if self._camera_mode:
            self._mic_btn.setToolTip("")
        self._mic_btn.update()

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

    def set_camera_mode(self, active: bool) -> None:
        """Use a compact icon toolbar while Camera Mode is active."""
        self._camera_mode = active
        layout = self.layout()
        if active:
            self.setStyleSheet(
                "QWidget#controls_bar { background: transparent; "
                "border-top: 1px solid rgba(203,213,225,0.12); }"
            )
            layout.setContentsMargins(8, 4, 8, 4)
            self._controls_spacer.changeSize(
                0, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
            )
        else:
            self.setStyleSheet(
                "QWidget#controls_bar { background-color: transparent; border: none; }"
            )
            layout.setContentsMargins(10, 8, 10, 8)
            self._controls_spacer.changeSize(
                0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
            )
        layout.setSpacing(5 if active else 8)
        layout.invalidate()
        self.setFixedHeight(46 if active else 58)
        self._record_btn.setVisible(True)
        self._mic_btn.setVisible(True)
        self._screenshot_btn.setVisible(True)
        self._settings_btn.setVisible(True)
        self._camera_drag_handle.setVisible(active)
        self._camera_drag_handle.setToolTip(
            "" if active else "Drag to align below your camera"
        )
        layout.removeWidget(self._record_btn)
        layout.removeWidget(self._mic_btn)
        layout.removeWidget(self._camera_drag_handle)
        if active:
            layout.insertWidget(0, self._camera_drag_handle)
            layout.insertWidget(1, self._mic_btn)
            layout.insertWidget(2, self._record_btn)
        else:
            layout.insertWidget(0, self._record_btn)
            layout.insertWidget(1, self._mic_btn)
        buttons = (
            (self._mic_btn, "Microphone"),
            (self._record_btn, "Record"),
            (self._screenshot_btn, "Capture"),
            (self._settings_btn, "Settings"),
        )
        if active:
            for button, name in buttons:
                button.setText("")
                button.setFixedSize(38, 38)
                button.set_camera_mode(True)
                button.setToolTip("")
                button.setAccessibleName(name)
                button.setAccessibleDescription(
                    "Microphone is on" if button is self._mic_btn and self._mic_enabled else
                    "Microphone is off" if button is self._mic_btn else
                    "Recording is active" if button is self._record_btn and self._recording else
                    "Recording is stopped" if button is self._record_btn else name
                )
                button.setStyleSheet(
                    "QPushButton { background: transparent; border: none; "
                    "border-radius: 9px; padding: 0px; color: transparent; }"
                )
        else:
            self._record_btn.setText("Stop" if self._recording else "Record")
            self._mic_btn.setText("Mic On" if self._mic_enabled else "Mic Off")
            self._screenshot_btn.setText("Capture")
            self._settings_btn.setText("Settings")
            for button, _ in buttons:
                button.set_camera_mode(False)
            self._record_btn.setFixedSize(86, 38)
            self._mic_btn.setFixedSize(78, 38)
            self._screenshot_btn.setFixedSize(88, 38)
            self._settings_btn.setFixedSize(86, 38)
            self._record_btn.setToolTip("")
            self._mic_btn.setToolTip("")
            self._screenshot_btn.setToolTip("")
            self._settings_btn.setToolTip("")
            self._record_btn.setAccessibleName("")
            self._mic_btn.setAccessibleName("")
            self._screenshot_btn.setAccessibleName("")
            self._settings_btn.setAccessibleName("")
            self._apply_mic_button_state()
            self.set_recording(self._recording)

    def current_mode(self) -> str:
        return self._mode_combo.currentText()

    def _apply_mic_button_state(self) -> None:
        if self._mic_enabled:
            self._mic_btn.setText("Mic On")
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
            self._mic_btn.setAccessibleDescription("Microphone is on")
            if self._camera_mode:
                self._mic_btn.setToolTip("Turn microphone off")
            return

        self._mic_btn.setText("Mic Off")
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
        self._mic_btn.setAccessibleDescription("Microphone is off")
        if self._camera_mode:
            self._mic_btn.setToolTip("Turn microphone on")

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
