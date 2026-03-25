"""
Main overlay window for GhostMic.

Features:
- Frameless, always-on-top, semi-transparent dark overlay
- Custom drag handle (title bar)
- Resizable from all edges/corners
- WDA_EXCLUDEFROMCAPTURE applied via stealth module
- Three panels: ControlsBar / TranscriptPanel / AIResponsePanel
- Compact / expanded mode toggle on double-click
- Ctrl+MouseWheel to adjust opacity
"""

from __future__ import annotations

import json
import sys
from typing import Optional

from PyQt6.QtCore import (
    QEvent,
    QPoint,
    QRect,
    QSize,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ghostmic.ui.ai_response_panel import AIResponsePanel
from ghostmic.ui.controls_bar import ControlsBar
from ghostmic.ui.styles import MAIN_STYLE
from ghostmic.ui.transcript_panel import TranscriptPanel
from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)

RESIZE_MARGIN = 8  # pixels from edge that trigger resize
QWIDGETSIZE_MAX = 16_777_215  # Qt maximum widget dimension


class TitleBar(QWidget):
    """Custom title bar with drag handle, pin, minimise and close buttons."""

    close_requested = pyqtSignal()
    minimise_requested = pyqtSignal()
    pin_toggled = pyqtSignal(bool)
    double_clicked = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(28)
        self._drag_pos: Optional[QPoint] = None
        self._pinned = True
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 6, 0)
        layout.setSpacing(4)

        # Drag label (acts as handle)
        self._title = QLabel("👻 GhostMic")
        self._title.setStyleSheet(
            "color: #8b949e; font-size: 10pt; font-weight: bold;"
        )
        layout.addWidget(self._title)
        layout.addStretch()

        # Pin button
        self._pin_btn = QPushButton("📌")
        self._pin_btn.setFixedSize(22, 22)
        self._pin_btn.setCheckable(True)
        self._pin_btn.setChecked(True)
        self._pin_btn.setToolTip("Pin on top")
        self._pin_btn.clicked.connect(lambda c: self.pin_toggled.emit(c))
        self._pin_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; }"
            "QPushButton:checked { color: #58a6ff; }"
        )
        layout.addWidget(self._pin_btn)

        # Minimise
        min_btn = QPushButton("—")
        min_btn.setFixedSize(22, 22)
        min_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; color: #8b949e; }"
            "QPushButton:hover { color: white; }"
        )
        min_btn.clicked.connect(self.minimise_requested)
        layout.addWidget(min_btn)

        # Close
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(22, 22)
        close_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; color: #8b949e; }"
            "QPushButton:hover { color: #f85149; }"
        )
        close_btn.clicked.connect(self.close_requested)
        layout.addWidget(close_btn)

    # Mouse events for dragging
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_pos and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.globalPosition().toPoint() - self._drag_pos
            window = self.window()
            window.move(window.pos() + delta)
            self._drag_pos = event.globalPosition().toPoint()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        self.double_clicked.emit()
        super().mouseDoubleClickEvent(event)


class MainWindow(QMainWindow):
    """Main GhostMic overlay window.

    Args:
        config: Full application config dict.
        config_path: Optional path to config.json for persistence.
        parent: Optional parent widget.
    """

    def __init__(self, config: dict, config_path: Optional[str] = None, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._config_path = config_path
        self._compact_mode = False
        self._resize_edge: Optional[str] = None
        self._resize_origin: Optional[QPoint] = None
        self._resize_geometry: Optional[QRect] = None

        self._setup_window()
        self._build_ui()
        self._apply_config()

    # ------------------------------------------------------------------
    # Window setup
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        ui = self._config.get("ui", {})

        # Window flags: frameless, always-on-top, tool (hides from taskbar)
        flags = (
            Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("GhostMic")

        w = ui.get("window_width", 420)
        h = ui.get("window_height", 650)
        self.resize(w, h)

        # Position: bottom-right corner with 20px margin
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = ui.get("window_x") or (geo.right() - w - 20)
            y = ui.get("window_y") or (geo.bottom() - h - 20)
            self.move(x, y)

        opacity = ui.get("opacity", 0.95)
        self.setWindowOpacity(opacity)

        # Mouse tracking for resize cursor
        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Root container (clips the rounded-rect background)
        self._root = QWidget(self)
        self._root.setObjectName("root_widget")
        self._root.setStyleSheet(
            "#root_widget {"
            "  background-color: rgba(26, 26, 46, 242);"  # #1a1a2e @ 95%
            "  border: 1px solid #333366;"
            "  border-radius: 12px;"
            "}"
        )
        self.setCentralWidget(self._root)

        root_layout = QVBoxLayout(self._root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Apply dark global styles
        self.setStyleSheet(MAIN_STYLE)

        # Title bar
        self._title_bar = TitleBar()
        self._title_bar.close_requested.connect(self._on_close)
        self._title_bar.minimise_requested.connect(self.showMinimized)
        self._title_bar.pin_toggled.connect(self._on_pin_toggled)
        self._title_bar.double_clicked.connect(self.toggle_compact_mode)
        root_layout.addWidget(self._title_bar)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #333366;")
        root_layout.addWidget(sep)

        # Controls bar
        self._controls = ControlsBar()
        self._controls.record_toggled.connect(self._on_record_toggled)
        self._controls.mode_changed.connect(self._on_mode_changed)
        root_layout.addWidget(self._controls)

        # Content splitter (transcript above, AI below)
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.setHandleWidth(4)
        self._splitter.setStyleSheet("QSplitter::handle { background: #333366; }")
        root_layout.addWidget(self._splitter, stretch=1)

        self.transcript_panel = TranscriptPanel()
        self._splitter.addWidget(self.transcript_panel)

        self.ai_panel = AIResponsePanel()
        self._splitter.addWidget(self.ai_panel)

        # 60/40 split
        total_h = self._config.get("ui", {}).get("window_height", 650) - 80
        self._splitter.setSizes([int(total_h * 0.6), int(total_h * 0.4)])

    # ------------------------------------------------------------------
    # Show + stealth
    # ------------------------------------------------------------------

    def show(self) -> None:
        super().show()
        # Apply stealth after window has a valid HWND (100 ms delay)
        QTimer.singleShot(100, self._apply_stealth)

    def _apply_stealth(self) -> None:
        if sys.platform != "win32":
            return
        try:
            from ghostmic.core.stealth import apply_stealth
            hwnd = int(self.winId())
            apply_stealth(hwnd)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("MainWindow: could not apply stealth: %s", exc)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _apply_config(self) -> None:
        ui = self._config.get("ui", {})
        font_size = ui.get("font_size", 11)
        font = QFont("Segoe UI", font_size)
        QApplication.setFont(font)

    def update_config(self, config: dict) -> None:
        """Apply updated config (called after settings dialog saves)."""
        self._config = config
        # Persist config to disk
        if self._config_path:
            try:
                with open(self._config_path, "w", encoding="utf-8") as fh:
                    json.dump(config, fh, indent=2)
            except OSError as exc:
                logger.error("Could not save config: %s", exc)
        ui = config.get("ui", {})
        self.setWindowOpacity(ui.get("opacity", 0.95))
        self.resize(
            ui.get("window_width", 420),
            ui.get("window_height", 650),
        )

    @property
    def controls(self) -> "ControlsBar":
        """Public accessor for the controls bar."""
        return self._controls

    def set_status(self, text: str, color: str = "") -> None:
        """Set the status label text (delegates to ControlsBar)."""
        if color:
            self._controls.set_status(text, color)
        else:
            self._controls.set_status(text)

    def set_api_status(self, connected: bool, backend: str = "Groq") -> None:
        """Update API connection status indicator."""
        self._controls.set_api_status(connected, backend)

    # ------------------------------------------------------------------
    # Compact mode
    # ------------------------------------------------------------------

    def toggle_compact_mode(self) -> None:
        self._compact_mode = not self._compact_mode
        if self._compact_mode:
            self.ai_panel.setMaximumHeight(80)
            self.transcript_panel.setMaximumHeight(60)
            self.resize(self.width(), 200)
        else:
            self.transcript_panel.setMaximumHeight(QWIDGETSIZE_MAX)
            self.ai_panel.setMaximumHeight(QWIDGETSIZE_MAX)
            ui = self._config.get("ui", {})
            self.resize(
                ui.get("window_width", 420), ui.get("window_height", 650)
            )

    # ------------------------------------------------------------------
    # Resize logic (frameless window)
    # ------------------------------------------------------------------

    def _get_resize_edge(self, pos: QPoint) -> Optional[str]:
        """Return which edge/corner the cursor is on, or None."""
        r = self.rect()
        m = RESIZE_MARGIN
        x, y = pos.x(), pos.y()
        on_left = x < m
        on_right = x > r.width() - m
        on_top = y < m
        on_bottom = y > r.height() - m

        if on_top and on_left:
            return "top-left"
        if on_top and on_right:
            return "top-right"
        if on_bottom and on_left:
            return "bottom-left"
        if on_bottom and on_right:
            return "bottom-right"
        if on_left:
            return "left"
        if on_right:
            return "right"
        if on_top:
            return "top"
        if on_bottom:
            return "bottom"
        return None

    def _cursor_for_edge(self, edge: str) -> Qt.CursorShape:
        mapping = {
            "top": Qt.CursorShape.SizeVerCursor,
            "bottom": Qt.CursorShape.SizeVerCursor,
            "left": Qt.CursorShape.SizeHorCursor,
            "right": Qt.CursorShape.SizeHorCursor,
            "top-left": Qt.CursorShape.SizeFDiagCursor,
            "bottom-right": Qt.CursorShape.SizeFDiagCursor,
            "top-right": Qt.CursorShape.SizeBDiagCursor,
            "bottom-left": Qt.CursorShape.SizeBDiagCursor,
        }
        return mapping.get(edge, Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        edge = self._get_resize_edge(event.position().toPoint())
        if edge and event.button() == Qt.MouseButton.LeftButton:
            self._resize_edge = edge
            self._resize_origin = event.globalPosition().toPoint()
            self._resize_geometry = self.geometry()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._resize_edge and self._resize_origin and self._resize_geometry:
            delta = event.globalPosition().toPoint() - self._resize_origin
            self._do_resize(delta)
        else:
            edge = self._get_resize_edge(event.position().toPoint())
            if edge:
                self.setCursor(self._cursor_for_edge(edge))
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._resize_edge = None
        self._resize_origin = None
        self._resize_geometry = None
        super().mouseReleaseEvent(event)

    def _do_resize(self, delta: QPoint) -> None:
        edge = self._resize_edge
        g = self._resize_geometry
        if not edge or not g:
            return
        dx, dy = delta.x(), delta.y()
        x, y, w, h = g.x(), g.y(), g.width(), g.height()
        min_w, min_h = 280, 200

        if "right" in edge:
            w = max(min_w, w + dx)
        if "left" in edge:
            new_w = max(min_w, w - dx)
            x = x + (w - new_w)
            w = new_w
        if "bottom" in edge:
            h = max(min_h, h + dy)
        if "top" in edge:
            new_h = max(min_h, h - dy)
            y = y + (h - new_h)
            h = new_h

        self.setGeometry(x, y, w, h)

    # ------------------------------------------------------------------
    # Opacity control via Ctrl+Wheel
    # ------------------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y() / 120  # steps
            new_opacity = max(0.3, min(1.0, self.windowOpacity() + delta * 0.05))
            self.setWindowOpacity(new_opacity)
        else:
            super().wheelEvent(event)

    # ------------------------------------------------------------------
    # changeEvent / showEvent — re-apply stealth
    # ------------------------------------------------------------------

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if sys.platform == "win32":
            if event.type() == QEvent.Type.WindowStateChange:
                QTimer.singleShot(100, self._apply_stealth)

    def showEvent(self, event: QEvent) -> None:
        super().showEvent(event)
        QTimer.singleShot(100, self._apply_stealth)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        QApplication.instance().quit()

    def _on_pin_toggled(self, pinned: bool) -> None:
        flags = self.windowFlags()
        if pinned:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show()

    def _on_record_toggled(self, recording: bool) -> None:
        # This signal is wired up by main.py to start/stop audio threads
        pass

    def _on_mode_changed(self, mode: str) -> None:
        logger.info("Mode changed to: %s", mode)

    def _show_settings(self) -> None:
        from ghostmic.ui.settings_dialog import SettingsDialog
        dlg = SettingsDialog(self._config, parent=self)
        dlg.settings_saved.connect(self.update_config)
        dlg.exec()
