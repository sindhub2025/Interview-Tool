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
    QEasingCurve,
    QEvent,
    QObject,
    QPoint,
    QParallelAnimationGroup,
    QPropertyAnimation,
    QRect,
    QSize,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
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
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QPlainTextEdit,
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

RESIZE_MARGIN = 12  # pixels from edge that trigger resize
QWIDGETSIZE_MAX = 16_777_215  # Qt maximum widget dimension


class TitleBar(QWidget):
    """Custom title bar with drag handle, pin, minimise and close buttons."""

    close_requested = pyqtSignal()
    minimise_requested = pyqtSignal()
    dock_toggle_requested = pyqtSignal()
    pin_toggled = pyqtSignal(bool)
    double_clicked = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(28)
        self._drag_pos: Optional[QPoint] = None
        self._pinned = True
        self._docked = False
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
        self._pin_btn.clicked.connect(lambda c: self.pin_toggled.emit(c))
        self._pin_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; }"
            "QPushButton:checked { color: #58a6ff; }"
        )
        layout.addWidget(self._pin_btn)

        # Dock toggle button
        self._dock_btn = QPushButton("⤓ Dock")
        self._dock_btn.setFixedSize(88, 26)
        self._dock_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: rgba(88, 166, 255, 0.16);"
            "  border: 1px solid rgba(88, 166, 255, 0.6);"
            "  border-radius: 6px;"
            "  color: #dce7ff;"
            "  font-size: 10pt;"
            "  font-weight: 600;"
            "  padding: 0px 12px;"
            "}"
            "QPushButton:hover { background-color: rgba(88, 166, 255, 0.28); }"
        )
        self._dock_btn.clicked.connect(self.dock_toggle_requested)
        layout.addWidget(self._dock_btn)

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

    def set_docked(self, docked: bool) -> None:
        self._docked = docked
        self._dock_btn.setText("⤒ Undock" if docked else "⤓ Dock")


class DockIndicator(QPushButton):
    """Small dock pill that can be dragged to reposition the dock without
    changing the mouse pointer.
    """

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._drag_origin: Optional[QPoint] = None
        self._window_start_pos: Optional[QPoint] = None
        self._dragging = False

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            # Only prepare drag state if dragging is enabled
            if getattr(self.window(), "_dock_draggable", True):
                self._drag_origin = event.globalPosition().toPoint()
                self._window_start_pos = self.window().pos()
                self._dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        # Only move the window when the user actually drags a few pixels
        if self._drag_origin and getattr(self.window(), "_dock_draggable", True):
            pos = event.globalPosition().toPoint()
            delta = pos - self._drag_origin
            if not self._dragging and (abs(delta.x()) > 3 or abs(delta.y()) > 3):
                self._dragging = True

            if self._dragging and self._window_start_pos is not None:
                new_pos = self._window_start_pos + delta
                # Move the whole window to follow the pill drag
                self.window().move(new_pos)
                return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            # End drag and consume the release so it doesn't trigger a click
            self._dragging = False
            self._drag_origin = None
            self._window_start_pos = None
            event.accept()
            return
        self._drag_origin = None
        self._window_start_pos = None
        super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    """Main GhostMic overlay window.

    Args:
        config: Full application config dict.
        config_path: Optional path to config.json for persistence.
        parent: Optional parent widget.
    """

    dictation_committed = pyqtSignal(str)
    dock_state_changed = pyqtSignal(bool)

    def __init__(self, config: dict, config_path: Optional[str] = None, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._config_path = config_path
        self._compact_mode = False
        self._qa_revealed = False
        self._qa_animation: QParallelAnimationGroup | None = None
        self._expanded_window_size = QSize(420, 650)
        self._collapsed_window_height = 90
        self._latest_question_text = ""
        self._move_origin: Optional[QPoint] = None
        self._move_window_origin: Optional[QPoint] = None
        self._move_started = False
        self._docked = bool(self._config.get("ui", {}).get("docked", False))
        self._dock_needs_apply = self._docked
        self._dock_transition_in_progress = False
        self._pre_dock_geometry: Optional[QRect] = self._load_pre_dock_geometry()
        self._resize_edge: Optional[str] = None
        self._resize_origin: Optional[QPoint] = None
        self._resize_geometry: Optional[QRect] = None
        self._resize_filter_targets: list[QWidget] = []
        self._dictation_idle_ms = int(
            self._config.get("dictation", {}).get("commit_idle_ms", 1200)
        )
        self._dictation_timer = QTimer(self)
        self._dictation_timer.setSingleShot(True)
        self._dictation_timer.timeout.connect(self._commit_dictation_text)

        self._setup_window()
        self._build_ui()
        if self._docked:
            self._title_bar.set_docked(True)
        self._apply_config()

        # Restore compact mode if previously saved
        if self._config.get("ui", {}).get("compact_mode", False):
            self.toggle_compact_mode()

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

        w = max(420, int(ui.get("window_width", 420) or 420))
        h = max(460, int(ui.get("window_height", 650) or 650))
        self._expanded_window_size = QSize(w, h)
        self.resize(w, h)

        # Position: bottom-right corner with 20px margin
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            cfg_x = ui.get("window_x")
            cfg_y = ui.get("window_y")
            x = cfg_x if cfg_x is not None else (geo.right() - w - 20)
            y = cfg_y if cfg_y is not None else (geo.bottom() - h - 20)
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
        self._set_root_style(docked=False)
        self.setCentralWidget(self._root)

        root_layout = QVBoxLayout(self._root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Apply dark global styles
        self.setStyleSheet(MAIN_STYLE)

        # Docked state indicator pill (shown only when docked)
        self._dock_indicator = DockIndicator("👻")
        self._dock_indicator.setObjectName("dock_restore_btn")
        self._dock_indicator.setFixedSize(120, 20)
        self._dock_indicator.setStyleSheet(
            "QPushButton#dock_restore_btn {"
            "  background-color: rgba(88, 166, 255, 0.85);"
            "  color: #0d1117;"
            "  border: 1px solid rgba(13, 17, 23, 0.45);"
            "  border-radius: 10px;"
            "  font-size: 9pt;"
            "  font-weight: 600;"
            "  padding: 0px 8px;"
            "}"
            "QPushButton#dock_restore_btn:hover {"
            "  background-color: rgba(88, 166, 255, 0.96);"
            "}"
        )
        self._dock_indicator.clicked.connect(self.exit_dock_mode)
        self._dock_indicator.hide()
        root_layout.addWidget(
            self._dock_indicator,
            alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
        )

        # Title bar
        self._title_bar = TitleBar()
        self._title_bar.close_requested.connect(self._on_close)
        self._title_bar.minimise_requested.connect(self.showMinimized)
        self._title_bar.dock_toggle_requested.connect(self.toggle_dock_mode)
        self._title_bar.pin_toggled.connect(self._on_pin_toggled)
        self._title_bar.double_clicked.connect(self.toggle_compact_mode)
        root_layout.addWidget(self._title_bar)

        # Separator
        self._title_separator = QFrame()
        self._title_separator.setFrameShape(QFrame.Shape.HLine)
        self._title_separator.setStyleSheet("color: #333366;")
        root_layout.addWidget(self._title_separator)

        # Controls bar
        self._controls = ControlsBar()
        self._controls.record_toggled.connect(self._on_record_toggled)
        self._controls.mode_changed.connect(self._on_mode_changed)
        root_layout.addWidget(self._controls, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Hidden transcript store remains wired for app logic; the visible Q&A
        # view mirrors its latest speaker question.
        self.transcript_panel = TranscriptPanel(self._root)
        self.transcript_panel.latest_question_changed.connect(self._set_question_text)
        self.transcript_panel.hide()

        self._qa_container = QWidget()
        self._qa_container.setObjectName("qa_container")
        self._qa_container.setStyleSheet(
            "QWidget#qa_container { background-color: transparent; border: none; }"
        )
        qa_layout = QVBoxLayout(self._qa_container)
        qa_layout.setContentsMargins(10, 0, 10, 10)
        qa_layout.setSpacing(0)

        # Content splitter (question above, AI answer below)
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.setHandleWidth(4)
        self._splitter.setStyleSheet(
            "QSplitter::handle { background: rgba(139, 148, 158, 0.22); }"
        )
        self._splitter.setChildrenCollapsible(False)
        qa_layout.addWidget(self._splitter, stretch=1)

        self._question_card = QFrame()
        self._question_card.setObjectName("question_card")
        self._question_card.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._question_card.setStyleSheet(
            "QFrame#question_card {"
            " background-color: rgba(33, 38, 45, 0.92);"
            " border: 1px solid rgba(139, 148, 158, 0.32);"
            " border-radius: 8px;"
            "}"
        )
        question_layout = QVBoxLayout(self._question_card)
        question_layout.setContentsMargins(12, 10, 12, 10)
        question_layout.setSpacing(6)

        question_title = QLabel("Question")
        question_title.setStyleSheet(
            "color: #7ee787; font-size: 10pt; font-weight: 700;"
        )
        question_layout.addWidget(question_title)

        self._question_text = QPlainTextEdit()
        self._question_text.setObjectName("question_text")
        self._question_text.setFrameShape(QFrame.Shape.NoFrame)
        self._question_text.setReadOnly(True)
        self._question_text.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._question_text.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._question_text.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.WidgetWidth
        )
        self._question_text.setStyleSheet(
            "QPlainTextEdit#question_text {"
            " background: transparent;"
            " color: #e6edf3;"
            " font-size: 11pt;"
            " padding: 0px;"
            "}"
        )
        question_layout.addWidget(self._question_text, 1)
        self._question_card.hide()
        self._splitter.addWidget(self._question_card)

        self.ai_panel = AIResponsePanel()
        self.ai_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.ai_panel.activity_started.connect(self._reveal_question_answer_area)
        self._splitter.addWidget(self.ai_panel)
        root_layout.addWidget(self._qa_container, stretch=1)
        self._qa_container.setMaximumHeight(0)
        self._qa_container.hide()

        self._install_resize_event_filters()

        # Hidden dictation target used for Windows native Win+H text input.
        self._dictation_input = QLineEdit()
        self._dictation_input.setFixedSize(1, 1)
        self._dictation_input.setStyleSheet(
            "QLineEdit { border: none; background: transparent; color: transparent; }"
        )
        self._dictation_input.setPlaceholderText("")
        self._dictation_input.textChanged.connect(self._on_dictation_text_changed)
        self._dictation_input.returnPressed.connect(self._commit_dictation_text)
        root_layout.addWidget(self._dictation_input)

        # 20/80 split so the AI response area gets the larger share.
        total_h = self._expanded_window_size.height() - 80
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 4)
        self._splitter.setSizes([int(total_h * 0.20), int(total_h * 0.80)])
        self._apply_startup_collapsed_layout()

    def _set_root_style(self, docked: bool) -> None:
        if docked:
            self._root.setStyleSheet(
                "#root_widget {"
                "  background-color: transparent;"
                "  border: none;"
                "  border-radius: 10px;"
                "}"
            )
            return

        self._root.setStyleSheet(
            "#root_widget {"
            "  background-color: rgba(17, 19, 24, 242);"
            "  border: 1px solid #30363d;"
            "  border-radius: 8px;"
            "}"
        )

    def _apply_splitter_proportions(self) -> None:
        """Keep the question/AI split close to the intended 20/80 ratio."""
        if (
            self._docked
            or self._compact_mode
            or not self._qa_revealed
            or not self._splitter.isVisible()
        ):
            return

        available_height = self._splitter.height()
        if available_height <= 0:
            return

        if not self._question_card.isVisible():
            self._splitter.setSizes([0, available_height])
            return

        question_height = self._target_question_height(available_height)
        ai_height = max(160, available_height - question_height)
        self._splitter.setSizes([question_height, ai_height])

    def _target_question_height(self, available_height: int) -> int:
        """Return a content-aware question-pane height that preserves answer space."""
        doc_height = int(self._question_text.document().size().height() or 0)
        natural_height = doc_height + 62  # title row + frame padding

        # Keep the AI panel dominant when speaker question is visible.
        ai_min_height = min(
            max(180, int(available_height * 0.70)),
            max(100, available_height - 40),
        )
        question_max = max(40, available_height - ai_min_height)
        question_min = min(120, max(60, int(available_height * 0.12)))
        if question_min > question_max:
            question_min = max(40, question_max)

        return max(question_min, min(question_max, natural_height))

    def _apply_startup_collapsed_layout(self) -> None:
        """Start as a toolbar-only rectangle."""
        if self._docked:
            return

        self._qa_revealed = False
        self._qa_container.setMaximumHeight(0)
        self._qa_container.hide()
        self._question_card.setVisible(False)
        self._title_bar.show()
        self._title_separator.show()
        self._dictation_input.hide()
        collapsed_width = max(420, self._controls.sizeHint().width() + 4)
        self.resize(collapsed_width, self._collapsed_window_height)

    def _set_question_text(self, text: str) -> None:
        self._latest_question_text = str(text or "").strip()
        self._question_text.setPlainText(self._latest_question_text)
        self._question_card.setVisible(
            self._qa_revealed and bool(self._latest_question_text)
        )
        if self._qa_revealed:
            QTimer.singleShot(0, self._apply_splitter_proportions)

    def _target_expanded_geometry(self) -> QRect:
        current = self.geometry()
        target_w = max(current.width(), self._expanded_window_size.width())
        target_h = max(460, self._expanded_window_size.height())
        target = QRect(current.x(), current.y(), target_w, target_h)

        screen = QApplication.screenAt(current.center()) or QApplication.primaryScreen()
        if screen is None:
            return target

        available = screen.availableGeometry()
        if target.right() > available.right():
            target.moveLeft(max(available.left(), available.right() - target_w))
        if target.bottom() > available.bottom():
            target.moveTop(max(available.top(), available.bottom() - target_h))
        return target

    def _reveal_question_answer_area(self) -> None:
        if self._docked:
            return

        self._question_card.setVisible(bool(self._latest_question_text))
        self._qa_container.show()
        self._splitter.show()

        if self._qa_revealed:
            self._qa_container.setMaximumHeight(QWIDGETSIZE_MAX)
            QTimer.singleShot(0, self._apply_splitter_proportions)
            return

        self._qa_revealed = True
        start_geometry = self.geometry()
        target_geometry = self._target_expanded_geometry()
        target_content_height = max(
            260,
            target_geometry.height() - self._controls.height() - 10,
        )

        if self._qa_animation is not None:
            self._qa_animation.stop()

        self._qa_container.setMaximumHeight(0)

        group = QParallelAnimationGroup(self)
        geometry_animation = QPropertyAnimation(self, b"geometry", group)
        geometry_animation.setDuration(240)
        geometry_animation.setStartValue(start_geometry)
        geometry_animation.setEndValue(target_geometry)
        geometry_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        content_animation = QPropertyAnimation(self._qa_container, b"maximumHeight", group)
        content_animation.setDuration(240)
        content_animation.setStartValue(0)
        content_animation.setEndValue(target_content_height)
        content_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        group.addAnimation(geometry_animation)
        group.addAnimation(content_animation)
        group.finished.connect(
            lambda: self._finish_qa_reveal(target_geometry)
        )
        self._qa_animation = group
        group.start()

    def _finish_qa_reveal(self, target_geometry: QRect) -> None:
        self.setGeometry(target_geometry)
        self._qa_container.setMaximumHeight(QWIDGETSIZE_MAX)
        self._qa_animation = None
        QTimer.singleShot(0, self._apply_splitter_proportions)

    def _install_resize_event_filters(self) -> None:
        """Capture edge-drag resize gestures even when child widgets are under cursor."""
        self._resize_filter_targets = [
            self._root,
            self._title_bar,
            self._controls,
            self._qa_container,
            self._splitter,
            self._question_card,
            self.transcript_panel,
            self.ai_panel,
        ]
        for widget in self._resize_filter_targets:
            widget.setMouseTracking(True)
            widget.installEventFilter(self)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if obj in self._resize_filter_targets and isinstance(event, QMouseEvent):
            event_type = event.type()

            if (
                self._docked
                and event_type == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
            ):
                self.exit_dock_mode()
                return True

            pos_in_window = obj.mapTo(self, event.position().toPoint())
            edge = self._get_resize_edge(pos_in_window)

            if (
                event_type == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
                and edge
            ):
                self._resize_edge = edge
                self._resize_origin = event.globalPosition().toPoint()
                self._resize_geometry = self.geometry()
                return True

            if (
                event_type == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
                and self._can_drag_from(obj)
                and edge is None
            ):
                self._move_origin = event.globalPosition().toPoint()
                self._move_window_origin = self.pos()
                self._move_started = False
                return False

            if event_type == QEvent.Type.MouseMove:
                if (
                    self._resize_edge
                    and self._resize_origin
                    and self._resize_geometry
                    and (event.buttons() & Qt.MouseButton.LeftButton)
                ):
                    delta = event.globalPosition().toPoint() - self._resize_origin
                    self._do_resize(delta)
                    return True
                if (
                    self._move_origin
                    and self._move_window_origin
                    and self._can_drag_from(obj)
                    and (event.buttons() & Qt.MouseButton.LeftButton)
                ):
                    delta = event.globalPosition().toPoint() - self._move_origin
                    if not self._move_started and (
                        abs(delta.x()) > 3 or abs(delta.y()) > 3
                    ):
                        self._move_started = True
                    if self._move_started:
                        self.move(self._move_window_origin + delta)
                        return True

            if event_type == QEvent.Type.MouseButtonRelease and self._resize_edge:
                self._resize_edge = None
                self._resize_origin = None
                self._resize_geometry = None
                return True
            if event_type == QEvent.Type.MouseButtonRelease and self._move_origin:
                consumed = self._move_started
                self._move_origin = None
                self._move_window_origin = None
                self._move_started = False
                if consumed:
                    return True

        return super().eventFilter(obj, event)

    def _can_drag_from(self, obj: QObject) -> bool:
        """Allow dragging from the compact shell without stealing button clicks."""
        if self._docked:
            return False
        return obj in (self._root, self._controls, self._qa_container, self._question_card)

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
        # Allow optional toggling of dock dragging behavior
        self._dock_draggable = bool(ui.get("dock_draggable", True))
        self._dictation_idle_ms = int(
            self._config.get("dictation", {}).get("commit_idle_ms", 1200)
        )

    def update_config(self, config: dict) -> None:
        """Apply updated config (called after settings dialog saves).

        Note: config persistence to disk is handled by the caller
        (GhostMicApp._on_settings_saved) to avoid double writes.
        """
        self._config = config
        ui = config.get("ui", {})
        self.setWindowOpacity(ui.get("opacity", 0.95))
        self._expanded_window_size = QSize(
            max(420, int(ui.get("window_width", 420) or 420)),
            max(460, int(ui.get("window_height", 650) or 650)),
        )
        if self._docked:
            self._apply_docked_state()
        elif self._qa_revealed:
            self.resize(
                self._expanded_window_size.width(),
                self._expanded_window_size.height(),
            )
            QTimer.singleShot(0, self._apply_splitter_proportions)
        else:
            self._apply_startup_collapsed_layout()
        self._dictation_idle_ms = int(config.get("dictation", {}).get("commit_idle_ms", 1200))

    @property
    def is_docked(self) -> bool:
        return self._docked

    @property
    def pre_dock_geometry(self) -> Optional[QRect]:
        if self._pre_dock_geometry is None:
            return None
        return QRect(self._pre_dock_geometry)

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

    def focus_dictation_target(self) -> bool:
        """Focus hidden text input so Win+H dictation inserts text into this app."""
        self.show()
        self.raise_()
        self.activateWindow()
        self._dictation_input.show()
        self._dictation_input.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self._dictation_input.selectAll()
        return self._dictation_input.hasFocus()

    def _on_dictation_text_changed(self, text: str) -> None:
        if not text.strip():
            self._dictation_timer.stop()
            return
        self._dictation_timer.start(self._dictation_idle_ms)

    def _commit_dictation_text(self) -> None:
        text = self._dictation_input.text().strip()
        if not text:
            return
        self._dictation_timer.stop()
        self._dictation_input.clear()
        self.dictation_committed.emit(text)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(0, self._apply_splitter_proportions)

    # ------------------------------------------------------------------
    # Compact mode
    # ------------------------------------------------------------------

    def toggle_compact_mode(self) -> None:
        if self._docked or not self._qa_revealed:
            return
        self._compact_mode = not self._compact_mode
        if self._compact_mode:
            self.ai_panel.setMaximumHeight(80)
            self._question_card.setMaximumHeight(60)
            self.resize(self.width(), 200)
        else:
            self._question_card.setMaximumHeight(QWIDGETSIZE_MAX)
            self.ai_panel.setMaximumHeight(QWIDGETSIZE_MAX)
            QTimer.singleShot(0, self._apply_splitter_proportions)

    # ------------------------------------------------------------------
    # Resize logic (frameless window)
    # ------------------------------------------------------------------

    def _get_resize_edge(self, pos: QPoint) -> Optional[str]:
        """Return which edge/corner the cursor is on, or None."""
        if self._docked:
            return None
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

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._docked and event.button() == Qt.MouseButton.LeftButton:
            self.exit_dock_mode()
            return
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
        min_w = 360
        min_h = 170 if self._qa_revealed else self._collapsed_window_height

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
        if self._dock_needs_apply:
            self._dock_needs_apply = False
            QTimer.singleShot(0, self._apply_docked_state)
        elif not self._docked and not self._compact_mode:
            QTimer.singleShot(0, self._apply_splitter_proportions)
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

    def toggle_dock_mode(self) -> None:
        if self._dock_transition_in_progress:
            return
        self._dock_transition_in_progress = True
        QTimer.singleShot(220, self._clear_dock_transition_lock)
        if self._docked:
            self.exit_dock_mode()
        else:
            self.enter_dock_mode()

    def enter_dock_mode(self) -> None:
        if self._docked:
            return
        self._pre_dock_geometry = QRect(self.geometry())
        self._persist_pre_dock_geometry(self._pre_dock_geometry)
        self._docked = True
        self._config.setdefault("ui", {})["docked"] = True
        self._title_bar.set_docked(True)
        self._apply_docked_state()
        self.dock_state_changed.emit(True)

    def exit_dock_mode(self) -> None:
        if not self._docked:
            return
        self._docked = False
        self._config.setdefault("ui", {})["docked"] = False
        self._title_bar.set_docked(False)
        self._set_docked_ui(False)

        geo = self._pre_dock_geometry or self._load_pre_dock_geometry()
        if geo is None:
            ui = self._config.get("ui", {})
            x = ui.get("window_x")
            y = ui.get("window_y")
            w = ui.get("window_width")
            h = ui.get("window_height")
            geo = QRect(
                int(x) if isinstance(x, int) else self.x(),
                int(y) if isinstance(y, int) else self.y(),
                int(w) if isinstance(w, int) else self.width(),
                int(h) if isinstance(h, int) else self.height(),
            )
        self.setGeometry(geo)
        if not self._qa_revealed:
            self._apply_startup_collapsed_layout()
        self.show()
        self.raise_()
        self.activateWindow()
        QTimer.singleShot(60, self._apply_stealth)
        self.dock_state_changed.emit(False)

    def _apply_docked_state(self) -> None:
        if not self._docked:
            return
        self._set_docked_ui(True)
        ui = self._config.get("ui", {})
        screen = QApplication.screenAt(self.frameGeometry().center()) or QApplication.primaryScreen()
        if screen is None:
            return

        # Make dock a visible pill centered at the top of the active screen.
        available = screen.availableGeometry()
        dock_width = 120
        dock_height = 20
        x = available.x() + (available.width() - dock_width) // 2
        y = available.y()

        # Do not set a window tooltip while docked; keep the dock pill passive.
        self.setGeometry(x, y, dock_width, dock_height)
        self.show()
        self.raise_()
        self.activateWindow()
        QTimer.singleShot(60, self._apply_stealth)

    def _set_docked_ui(self, docked: bool) -> None:
        self._set_root_style(docked)
        self._dock_indicator.setVisible(docked)
        # Hide full controls when docked so only the restore pill remains.
        self._title_bar.setVisible(not docked)
        self._title_separator.setVisible(not docked)
        self._controls.setVisible(not docked)
        self._qa_container.setVisible(not docked and self._qa_revealed)
        self._splitter.setVisible(not docked and self._qa_revealed)
        self._dictation_input.setVisible(False)

    def _persist_pre_dock_geometry(self, geo: QRect) -> None:
        ui = self._config.setdefault("ui", {})
        ui["pre_dock_x"] = geo.x()
        ui["pre_dock_y"] = geo.y()
        ui["pre_dock_width"] = geo.width()
        ui["pre_dock_height"] = geo.height()

    def _load_pre_dock_geometry(self) -> Optional[QRect]:
        ui = self._config.get("ui", {})
        x = ui.get("pre_dock_x")
        y = ui.get("pre_dock_y")
        w = ui.get("pre_dock_width")
        h = ui.get("pre_dock_height")
        if all(isinstance(v, int) for v in (x, y, w, h)):
            return QRect(x, y, w, h)
        return None

    def _clear_dock_transition_lock(self) -> None:
        self._dock_transition_in_progress = False

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
