"""
AI response panel: displays streaming AI-generated suggestions.
"""

from __future__ import annotations

from html import escape
import re
from typing import List

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QTextCursor, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ghostmic.ui.styles import (
    ACCENT_BLUE,
    ACCENT_GREEN,
    BG_CARD,
    BG_DEEP,
    BG_MID,
    BORDER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

QWIDGETSIZE_MAX = 16_777_215


def _code_block_style(language: str) -> tuple[str, str, str]:
    normalized = language.strip().lower()
    if normalized in {"sql", "postgres", "postgresql", "mysql", "sqlite", "tsql"}:
        return "SQL", ACCENT_GREEN, "#10241b"
    if normalized in {"python", "py"}:
        return "Python", ACCENT_BLUE, "#101826"
    label = language.strip().upper() if language.strip() else "Code"
    return label, TEXT_SECONDARY, "#11161d"


def _render_code_block(language: str, code: str) -> str:
    label, accent, background = _code_block_style(language)
    return (
        "<div style='margin:12px 0 16px 0;'>"
        f"<div style='margin:0 0 6px 0; color:{accent}; font-size:10pt; "
        f"font-weight:700; letter-spacing:0.1px;'>{escape(label)}</div>"
        f"<pre style='margin:0; padding:12px 14px; line-height:1.5; border:1px solid {accent}; "
        f"border-radius:8px; background-color:{background}; color:{TEXT_PRIMARY}; "
        'font-family:"Consolas","Cascadia Mono","Courier New",monospace; '
        "font-size:11pt; white-space:pre-wrap;'>"
        f"{escape(code)}</pre>"
        "</div>"
    )


def render_response_html(text: str) -> str:
    """Render plain AI response text into styled HTML for the response panel."""
    if not text:
        return ""

    bullet_re = re.compile(r"^(?:[-*•]|\d+[.)])\s+(.*)$")
    html_blocks: list[str] = []
    paragraph_lines: list[str] = []
    bullet_items: list[str] = []
    code_lines: list[str] = []
    code_language = ""
    in_code_block = False

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        paragraph = " ".join(part.strip() for part in paragraph_lines).strip()
        paragraph_lines = []
        if paragraph:
            html_blocks.append(
                "<p style='margin:0 0 10px 0; line-height:1.45; "
                f"color:{TEXT_PRIMARY};'>{escape(paragraph)}</p>"
            )

    def flush_bullets() -> None:
        nonlocal bullet_items
        if not bullet_items:
            return
        items_html = "".join(
            "<li style='margin:0 0 6px 0; padding:0;'>"
            f"{escape(item)}</li>"
            for item in bullet_items
        )
        bullet_items = []
        html_blocks.append(
            "<ul style='margin:0 0 12px 20px; padding:0 0 0 16px;'>"
            f"{items_html}</ul>"
        )

    def flush_code_block() -> None:
        nonlocal code_lines, code_language
        code = "\n".join(code_lines)
        code_lines = []
        if code:
            html_blocks.append(_render_code_block(code_language, code))
        code_language = ""

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code_block:
                flush_code_block()
                in_code_block = False
            else:
                flush_paragraph()
                flush_bullets()
                code_language = stripped[3:].strip()
                in_code_block = True
            continue

        if in_code_block:
            code_lines.append(raw_line)
            continue

        if not stripped:
            flush_paragraph()
            flush_bullets()
            continue

        bullet_match = bullet_re.match(stripped)
        if bullet_match:
            flush_paragraph()
            bullet_items.append(bullet_match.group(1).strip())
            continue

        flush_bullets()
        paragraph_lines.append(stripped)

    if in_code_block:
        flush_code_block()

    flush_paragraph()
    flush_bullets()

    if not html_blocks:
        return ""

    return (
        "<html><body style='margin:0; padding:0; background:transparent; "
        f"color:{TEXT_PRIMARY}; font-family:\"Segoe UI\"; font-size:11pt;'>"
        "<div style='padding:2px 4px 4px 4px;'>"
        + "".join(html_blocks)
        + "</div></body></html>"
    )


class AIResponseCard(QFrame):
    """A single AI answer card with normal and teleprompter presentations."""

    camera_mode_requested = pyqtSignal(bool)

    def __init__(self, text: str = "", title: str = "Answer", parent=None) -> None:
        super().__init__(parent)
        self._text = text
        self._title = title
        self._teleprompter_active = False
        self._reading_font_size = 24
        self._scroll_speed = 2.0
        self._scroll_remainder = 0.0
        self._scrolling = False
        self._camera_mode_active = False
        self._camera_shortcuts = []
        self._tooltips_before_camera = {}
        self._scroll_timer = QTimer(self)
        self._scroll_timer.setInterval(16)
        self._scroll_timer.timeout.connect(self._advance_scroll)
        self._build_ui()

    def _build_ui(self) -> None:
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setObjectName("ai_response_card")
        self.setStyleSheet(
            f"QFrame#ai_response_card {{ background-color: {BG_CARD}; "
            f"border: 1px solid {BORDER}; border-radius: 10px; }}"
        )
        self._style_before_camera = self.styleSheet()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        # Top row: title + buttons
        self._card_header = QWidget(self)
        header = QHBoxLayout(self._card_header)
        header.setContentsMargins(2, 0, 2, 0)
        header.setSpacing(6)
        self._title_label = QLabel(self._title)
        self._title_label.setStyleSheet(
            f"color: {ACCENT_BLUE}; font-weight: 700; font-size: 9pt;"
        )
        header.addWidget(self._title_label)
        header.addStretch()

        self._copy_btn = QPushButton("Copy")
        self._copy_btn.setFixedSize(66, 26)
        self._copy_btn.setStyleSheet(
            "font-size: 10pt; font-weight: 600; padding: 0px 12px;"
        )
        self._copy_btn.clicked.connect(self._copy)
        self._copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        header.addWidget(self._copy_btn)

        self._teleprompter_btn = QPushButton("Teleprompter")
        self._teleprompter_btn.setFixedSize(104, 26)
        self._teleprompter_btn.setStyleSheet(
            "font-size: 10pt; font-weight: 600; padding: 0px 8px;"
        )
        self._teleprompter_btn.clicked.connect(self.toggle_teleprompter)
        self._teleprompter_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        header.addWidget(self._teleprompter_btn)
        layout.addWidget(self._card_header)

        # Response text
        self._text_edit = QTextBrowser()
        self._text_edit.setReadOnly(True)
        self._text_edit.setOpenExternalLinks(False)
        self._text_edit.setStyleSheet(
            f"background-color: {BG_MID}; border: none; "
            "border-radius: 6px; padding: 6px;"
        )
        self._text_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._text_edit.setMinimumHeight(0)
        self._text_edit.setHtml(render_response_html(self._text))
        layout.addWidget(self._text_edit)

        self._teleprompter_toolbar = QWidget(self)
        self._teleprompter_toolbar.setObjectName("teleprompter_toolbar")
        toolbar = QHBoxLayout(self._teleprompter_toolbar)
        toolbar.setContentsMargins(2, 0, 2, 0)
        toolbar.setSpacing(5)
        self._font_down_btn = QPushButton("A−")
        self._font_up_btn = QPushButton("A+")
        self._start_scroll_btn = QPushButton("Start")
        self._camera_btn = QPushButton("Camera Mode")
        self._speed_down_btn = QPushButton("−")
        self._speed_up_btn = QPushButton("+")
        self._font_size_label = QLabel("24 pt")
        self._speed_label = QLabel("2x")
        self._font_down_btn.setToolTip("Decrease reading text size")
        self._font_up_btn.setToolTip("Increase reading text size")
        self._start_scroll_btn.setToolTip("Start or pause auto-scroll")
        self._speed_down_btn.setToolTip("Decrease scroll speed")
        self._speed_up_btn.setToolTip("Increase scroll speed")
        self._font_down_btn.setAccessibleName("Decrease reading text size")
        self._font_up_btn.setAccessibleName("Increase reading text size")
        self._start_scroll_btn.setAccessibleName("Start or pause auto-scroll")
        self._speed_down_btn.setAccessibleName("Decrease scroll speed")
        self._speed_up_btn.setAccessibleName("Increase scroll speed")
        for button in (self._font_down_btn, self._font_up_btn,
                       self._start_scroll_btn, self._speed_down_btn,
                       self._speed_up_btn):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumWidth(30)
            toolbar.addWidget(button)
        toolbar.insertWidget(1, self._font_size_label)
        self._font_size_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        toolbar.addWidget(self._speed_label)
        toolbar.addStretch()
        toolbar.addWidget(self._camera_btn)
        self._font_down_btn.clicked.connect(lambda: self._change_font_size(-2))
        self._font_up_btn.clicked.connect(lambda: self._change_font_size(2))
        self._start_scroll_btn.clicked.connect(self._toggle_scrolling)
        self._speed_down_btn.clicked.connect(lambda: self._change_speed(-1))
        self._speed_up_btn.clicked.connect(lambda: self._change_speed(1))
        self._camera_btn.clicked.connect(self.toggle_camera_mode)
        self._camera_btn.setAccessibleName("Camera Mode")
        self._apply_teleprompter_toolbar_style()
        self._teleprompter_toolbar.setVisible(False)
        layout.addWidget(self._teleprompter_toolbar)

        self._reading_view = QTextBrowser()
        self._reading_view.setReadOnly(True)
        self._reading_view.setOpenExternalLinks(False)
        self._reading_view.setFrameShape(QFrame.Shape.NoFrame)
        self._reading_view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._reading_view.setStyleSheet(
            "QTextBrowser { background: transparent; color: #eff4fa; "
            "border: none; padding: 2px 10px; "
            "selection-background-color: #315b83; }"
        )
        self._reading_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._reading_view.setMinimumHeight(0)
        self._reading_view.setVisible(False)
        layout.addWidget(self._reading_view)

    def _render_reading_view(self, *, scroll_to_end: bool = False) -> None:
        old_scroll = self._reading_view.verticalScrollBar().value()
        old_max = self._reading_view.verticalScrollBar().maximum()
        html = render_response_html(self._text)
        if html:
            html = html.replace(
                "<html><body style='margin:0; padding:0; background:transparent; "
                f"color:{TEXT_PRIMARY}; font-family:\"Segoe UI\"; font-size:11pt;'>",
                "<html><body style='margin:0; padding:0; background:transparent; "
                f"color:{TEXT_PRIMARY}; font-family:\"Segoe UI\"; font-size:{self._reading_font_size}pt; "
                "line-height:1.45;'>",
            )
            html = html.replace("font-size:11pt", f"font-size:{self._reading_font_size}pt")
            html = html.replace("font-size:10pt", f"font-size:{max(12, self._reading_font_size - 6)}pt")
            html = html.replace(
                f"font-size:{self._reading_font_size}pt; white-space:pre-wrap",
                f"font-size:{max(12, self._reading_font_size - 6)}pt; white-space:pre-wrap",
            )
        self._reading_view.setHtml(html)
        scrollbar = self._reading_view.verticalScrollBar()
        if scroll_to_end:
            scrollbar.setValue(scrollbar.maximum())
        elif old_max:
            scrollbar.setValue(min(scrollbar.maximum(), int(old_scroll / old_max * scrollbar.maximum())))
        else:
            scrollbar.setValue(scrollbar.minimum())

    def _apply_teleprompter_toolbar_style(self) -> None:
        button_style = (
            "QPushButton { background: #1b232e; color: #dce5ef; "
            "border: 1px solid rgba(148,163,184,0.16); border-radius: 7px; "
            "padding: 0 8px; font-size: 9pt; font-weight: 600; }"
            "QPushButton:hover { background: #263242; border-color: rgba(125,173,220,0.42); }"
            "QPushButton:pressed { background: #111821; }"
            "QPushButton:disabled { color: #687687; background: #151b23; }"
        )
        for button in (self._font_down_btn, self._font_up_btn,
                       self._speed_down_btn, self._speed_up_btn):
            button.setFixedSize(30, 28)
            button.setStyleSheet(button_style)
        self._start_scroll_btn.setFixedSize(70, 28)
        self._start_scroll_btn.setObjectName("scroll_toggle_btn")
        self._start_scroll_btn.setStyleSheet(
            button_style + "QPushButton#scroll_toggle_btn { color: #d9f5e1; "
            "background: rgba(46,160,67,0.16); border-color: rgba(63,185,80,0.34); }"
            "QPushButton#scroll_toggle_btn:hover { background: rgba(46,160,67,0.26); }"
        )
        self._camera_btn.setFixedSize(78, 28)
        self._camera_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #a9c9e8; "
            "border: 1px solid rgba(125,173,220,0.23); border-radius: 7px; "
            "padding: 0 8px; font-size: 9pt; font-weight: 600; }"
            "QPushButton:hover { background: rgba(88,166,255,0.12); "
            "border-color: rgba(125,173,220,0.50); }"
        )
        self._font_size_label.setStyleSheet(
            "color: #8795a5; font-size: 8pt; font-weight: 600;"
        )
        self._font_size_label.setFixedWidth(36)
        self._speed_label.setStyleSheet(
            "color: #e7edf5; font-size: 9pt; font-weight: 700; padding: 0 2px;"
        )
        self._speed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._speed_label.setFixedWidth(34)

    def _change_font_size(self, delta: int) -> None:
        self._reading_font_size = max(16, min(40, self._reading_font_size + delta))
        self._font_size_label.setText(f"{self._reading_font_size} pt")
        self._font_down_btn.setEnabled(self._reading_font_size > 16)
        self._font_up_btn.setEnabled(self._reading_font_size < 40)
        if self._teleprompter_active:
            self._render_reading_view()

    def _change_speed(self, delta: int) -> None:
        self._scroll_speed = max(1.0, min(5.0, self._scroll_speed + (delta * 0.5)))
        speed = f"{self._scroll_speed:.1f}".rstrip("0").rstrip(".")
        self._speed_label.setText(f"{speed}x")

    def _toggle_scrolling(self) -> None:
        scrollbar = self._reading_view.verticalScrollBar()
        if not self._scrolling and scrollbar.value() >= scrollbar.maximum() and scrollbar.maximum() > 0:
            scrollbar.setValue(scrollbar.minimum())
        self._scrolling = not self._scrolling
        if self._scrolling:
            self._scroll_timer.start()
        else:
            self._scroll_timer.stop()
        self._update_scroll_button_label()

    def _update_scroll_button_label(self) -> None:
        scrollbar = self._reading_view.verticalScrollBar()
        if self._scrolling:
            label = "Ⅱ" if self._camera_mode_active else "Pause"
        elif scrollbar.value() >= scrollbar.maximum() and scrollbar.maximum() > 0:
            label = "↻" if self._camera_mode_active else "Replay"
        elif scrollbar.value() > scrollbar.minimum():
            label = "▶" if self._camera_mode_active else "Resume"
        else:
            label = "▶" if self._camera_mode_active else "Start"
        self._start_scroll_btn.setText(label)

    def _advance_scroll(self) -> None:
        scrollbar = self._reading_view.verticalScrollBar()
        # Scale fractional speed to a steady pixels-per-second rate. At 60 Hz,
        # 2x advances about 20 px/s and 5x about 50 px/s.
        self._scroll_remainder += self._scroll_speed * 0.16
        pixels = int(self._scroll_remainder)
        self._scroll_remainder -= pixels
        value = min(scrollbar.maximum(), scrollbar.value() + pixels)
        scrollbar.setValue(value)
        if value >= scrollbar.maximum():
            self._scrolling = False
            self._scroll_timer.stop()
            self._update_scroll_button_label()

    def toggle_camera_mode(self) -> None:
        requested = not self._camera_mode_active
        self.camera_mode_requested.emit(requested)

    def set_camera_mode(self, active: bool) -> None:
        """Synchronize the card's presentation with MainWindow camera mode."""
        self._camera_mode_active = active
        if active:
            self._style_before_camera = self.styleSheet()
            camera_buttons = (
                self._font_down_btn,
                self._font_up_btn,
                self._start_scroll_btn,
                self._speed_down_btn,
                self._speed_up_btn,
                self._camera_btn,
            )
            self._tooltips_before_camera = {
                button: button.toolTip() for button in camera_buttons
            }
            for button in camera_buttons:
                button.setToolTip("")
        else:
            for button, tooltip in self._tooltips_before_camera.items():
                button.setToolTip(tooltip)
            self._tooltips_before_camera.clear()
        self._camera_btn.setText("×" if active else "Camera Mode")
        self._camera_btn.setAccessibleName(
            "Exit Camera Mode" if active else "Camera Mode"
        )
        if not active:
            self._camera_btn.setToolTip("Compact the window below your camera")
        self._card_header.setVisible(not active)
        if active:
            self.setStyleSheet(
                "QFrame#ai_response_card { background: transparent; border: none; }"
            )
        else:
            self.setStyleSheet(self._style_before_camera)
        if active:
            self.layout().setContentsMargins(10, 2, 10, 4)
        else:
            self.layout().setContentsMargins(10, 8, 10, 10)
        self.layout().setSpacing(5 if active else 6)
        self._teleprompter_toolbar.setStyleSheet(
            "QWidget#teleprompter_toolbar { background: transparent; "
            "border-top: 1px solid rgba(203,213,225,0.12); }"
            if active else ""
        )
        toolbar_layout = self._teleprompter_toolbar.layout()
        toolbar_layout.setContentsMargins(2, 5 if active else 0, 2, 0)
        toolbar_layout.setSpacing(4 if active else 5)
        if active:
            self._camera_btn.setFixedSize(30, 28)
            self._camera_btn.setStyleSheet(
                "QPushButton { background: transparent; color: #9ba9b9; "
                "border: 1px solid rgba(148,163,184,0.18); border-radius: 14px; "
                "padding: 0; font-size: 16pt; font-weight: 400; }"
                "QPushButton:hover { color: #f1f5f9; "
                "background: rgba(255,255,255,0.08); "
                "border-color: rgba(203,213,225,0.36); }"
            )
            self._start_scroll_btn.setFixedSize(34, 28)
            self._start_scroll_btn.setStyleSheet(
                "QPushButton { background: rgba(46,160,67,0.13); color: #b7e4c0; "
                "border: 1px solid rgba(63,185,80,0.24); border-radius: 14px; "
                "padding: 0; font-size: 10pt; font-weight: 700; }"
                "QPushButton:hover { background: rgba(46,160,67,0.24); }"
            )
            self._update_scroll_button_label()
        else:
            self._camera_btn.setFixedSize(78, 28)
            self._apply_teleprompter_toolbar_style()
            self._update_scroll_button_label()
        if active:
            layout = self.layout()
            layout.removeWidget(self._teleprompter_toolbar)
            layout.removeWidget(self._reading_view)
            layout.addWidget(self._reading_view, 1)
            layout.addWidget(self._teleprompter_toolbar)
        else:
            layout = self.layout()
            layout.removeWidget(self._reading_view)
            layout.removeWidget(self._teleprompter_toolbar)
            layout.insertWidget(2, self._teleprompter_toolbar)
            layout.insertWidget(3, self._reading_view, 1)
        if active:
            self._install_camera_shortcuts()
        else:
            for shortcut in self._camera_shortcuts:
                shortcut.setEnabled(False)
                shortcut.deleteLater()
            self._camera_shortcuts.clear()

    def _install_camera_shortcuts(self) -> None:
        if self._camera_shortcuts:
            return
        shortcuts = {
            "Space": self._toggle_scrolling,
            "Up": lambda: self._change_speed(1),
            "Down": lambda: self._change_speed(-1),
            "+": lambda: self._change_font_size(2),
            "=": lambda: self._change_font_size(2),
            "-": lambda: self._change_font_size(-2),
            "Home": self._scroll_home,
            "End": self._scroll_end,
            "C": self.toggle_camera_mode,
            "Esc": self.toggle_camera_mode,
        }
        for key, callback in shortcuts.items():
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(callback)
            self._camera_shortcuts.append(shortcut)

    def _scroll_home(self) -> None:
        self._scrolling = False
        self._scroll_timer.stop()
        self._reading_view.verticalScrollBar().setValue(0)
        self._start_scroll_btn.setText("Start")

    def _scroll_end(self) -> None:
        self._scrolling = False
        self._scroll_timer.stop()
        self._reading_view.verticalScrollBar().setValue(
            self._reading_view.verticalScrollBar().maximum()
        )
        self._start_scroll_btn.setText("Replay")

    def toggle_teleprompter(self) -> None:
        active = not self._teleprompter_active
        if active:
            self._render_reading_view()
        self._teleprompter_active = active
        self._text_edit.setVisible(not active)
        self._reading_view.setVisible(active)
        self._teleprompter_toolbar.setVisible(active)
        self._teleprompter_btn.setText("Normal View" if active else "Teleprompter")
        if not active:
            self._scrolling = False
            self._scroll_timer.stop()

    def is_teleprompter_active(self) -> bool:
        return self._teleprompter_active

    def _render_text(self, *, scroll_to_end: bool) -> None:
        self._text_edit.setHtml(render_response_html(self._text))
        if self._teleprompter_active:
            self._render_reading_view(scroll_to_end=scroll_to_end)
        cursor = self._text_edit.textCursor()
        cursor.movePosition(
            QTextCursor.MoveOperation.End
            if scroll_to_end
            else QTextCursor.MoveOperation.Start
        )
        self._text_edit.setTextCursor(cursor)
        self._text_edit.ensureCursorVisible()

    def append_text(self, chunk: str) -> None:
        """Append a streaming chunk to the response."""
        self._text += chunk
        self._render_text(scroll_to_end=not self._teleprompter_active)

    def set_text(self, text: str) -> None:
        from ghostmic.utils.logger import get_logger

        logger = get_logger(__name__)
        self._scrolling = False
        self._scroll_timer.stop()
        self._text = text
        self._render_text(scroll_to_end=False)
        if self._teleprompter_active:
            self._reading_view.verticalScrollBar().setValue(
                self._reading_view.verticalScrollBar().minimum()
            )
            self._start_scroll_btn.setText("Start")
        self._text_edit.updateGeometry()
        logger.info(
            "AIResponseCard.set_text() called - set %d chars to QTextBrowser, new height=%d",
            len(text),
            self._text_edit.height(),
        )

    def get_text(self) -> str:
        return self._text

    def _copy(self) -> None:
        QApplication.clipboard().setText(self._text)


class AIResponsePanel(QWidget):
    """Panel that shows only the latest AI answer with streaming support."""

    activity_started = pyqtSignal()
    text_prompt_submitted = pyqtSignal(str, bool)
    camera_mode_requested = pyqtSignal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cards: List[AIResponseCard] = []
        self._active_card: AIResponseCard | None = None
        self._thinking_label: QLabel | None = None
        self._collapsed = False
        self._camera_mode = False

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setObjectName("ai_response_panel")
        self.setStyleSheet(
            f"QWidget#ai_response_panel {{"
            f" background-color: {BG_DEEP};"
            f" border: none;"
            f"}}"
            f"QWidget#ai_panel_outer {{"
            f" background-color: {BG_DEEP};"
            f" border: none;"
            f" border-radius: 10px;"
            f"}}"
            f"QFrame#ai_panel_header {{"
            f" background-color: {BG_MID};"
            f" border-bottom: 1px solid {BORDER};"
            f" border-top-left-radius: 10px;"
            f" border-top-right-radius: 10px;"
            f"}}"
            f"QWidget#ai_panel_content {{"
            f" background-color: {BG_DEEP};"
            f" border: none;"
            f" border-bottom-left-radius: 10px;"
            f" border-bottom-right-radius: 10px;"
            f"}}"
        )
        self._style_before_camera = self.styleSheet()

        outer_layout = QVBoxLayout(self)
        outer_layout.setSpacing(0)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        header = QFrame()
        header.setObjectName("ai_panel_header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 6, 10, 6)
        header_layout.setSpacing(6)

        title = QLabel("Answer")
        title.setStyleSheet(f"color: {ACCENT_BLUE}; font-weight: bold;")
        header_layout.addWidget(title)
        header_layout.addStretch()

        self._toggle_btn = QPushButton("Hide")
        self._toggle_btn.setFixedSize(72, 26)
        self._toggle_btn.setStyleSheet(
            "font-size: 10pt; font-weight: 600; padding: 0px 12px;"
        )
        self._toggle_btn.clicked.connect(self.toggle_visibility)
        header_layout.addWidget(self._toggle_btn)

        header.setVisible(False)
        outer_layout.addWidget(header)

        self._content = QWidget()
        self._content.setObjectName("ai_panel_content")
        self._content.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        content_layout = QVBoxLayout(self._content)
        content_layout.setSpacing(8)
        content_layout.setContentsMargins(0, 0, 0, 0)

        self._responses_container = QWidget()
        self._responses_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._responses_container.setMinimumHeight(0)
        self._responses_layout = QVBoxLayout(self._responses_container)
        self._responses_layout.setSpacing(8)
        self._responses_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self._responses_container, 1)

        self._prompt_row = QWidget(self._content)
        input_row = QHBoxLayout(self._prompt_row)
        input_row.setSpacing(6)

        self._prompt_input = QLineEdit()
        self._prompt_input.setPlaceholderText(
            "Ask a follow-up or refine the answer"
        )
        self._prompt_input.returnPressed.connect(self._on_ask_clicked)
        input_row.addWidget(self._prompt_input, 1)

        self._ask_btn = QPushButton("Ask")
        self._ask_btn.setFixedSize(62, 30)
        self._ask_btn.clicked.connect(self._on_ask_clicked)
        input_row.addWidget(self._ask_btn)

        self._refine_btn = QPushButton("Refine")
        self._refine_btn.setFixedSize(74, 30)
        self._refine_btn.clicked.connect(self._on_refine_clicked)
        input_row.addWidget(self._refine_btn)

        content_layout.addWidget(self._prompt_row)

        outer_layout.addWidget(self._content, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _wire_camera_mode(self, card: AIResponseCard) -> None:
        card.camera_mode_requested.connect(self.camera_mode_requested.emit)
        card.set_camera_mode(self._camera_mode)

    def set_camera_mode(self, active: bool) -> None:
        """Compact the answer panel while keeping its response card in place."""
        if active == self._camera_mode:
            return
        self._camera_mode = active
        if active:
            self._prompt_row_was_hidden = self._prompt_row.isHidden()
            self._style_before_camera = self.styleSheet()
            self._prompt_row.hide()
            self._content.layout().setContentsMargins(0, 0, 0, 0)
            self._content.layout().setSpacing(0)
            self.setStyleSheet(
                "QWidget#ai_response_panel, QWidget#ai_panel_content { "
                "background: transparent; border: none; }"
            )
        else:
            self._prompt_row.setVisible(not getattr(self, "_prompt_row_was_hidden", False))
            self._content.layout().setContentsMargins(0, 0, 0, 0)
            self._content.layout().setSpacing(8)
            self.setStyleSheet(self._style_before_camera)
        for card in self._cards:
            card.set_camera_mode(active)

    def start_response(self, title: str = "Answer") -> None:
        """Create a new blank response card ready for streaming."""
        from ghostmic.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("AIResponsePanel.start_response() called")
        self.activity_started.emit()
        self._remove_thinking()
        card = self._active_card
        if card is None and self._cards:
            card = self._cards[-1]

        if card is not None and card.parent() is self._responses_container:
            self._active_card = card
            self._cards = [card]
            card.set_text("")
            self._responses_layout.setStretchFactor(card, 1)
            self._scroll_to_bottom()
            logger.info(
                "AIResponsePanel.start_response() - reused existing card, layout count=%d",
                self._responses_layout.count(),
            )
            return

        card = AIResponseCard(title=title, parent=self._responses_container)
        self._wire_camera_mode(card)
        self._active_card = card
        self._cards = [card]
        self._responses_layout.addWidget(card, 1)

        logger.info(
            "AIResponsePanel.start_response() - card inserted, layout count=%d",
            self._responses_layout.count(),
        )

        # Verify card is visible
        logger.info("AIResponsePanel.start_response() - card visible=%s, card height=%d",
                   card.isVisible(), card.height())

        self._scroll_to_bottom()
        logger.info("AIResponsePanel.start_response() - card added to layout")

    def append_chunk(self, chunk: str) -> None:
        """Append a streaming chunk to the active response card."""
        self.activity_started.emit()
        if self._active_card is None:
            self.start_response()
        if self._active_card:
            self._active_card.append_text(chunk)

    def finish_response(self, full_text: str | None = None) -> None:
        """Mark the current response as complete and display it."""
        from ghostmic.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("AIResponsePanel.finish_response() called with text: %s...", 
                   (full_text[:50] if full_text else "None"))
        self.activity_started.emit()
        logger.info("AIResponsePanel.finish_response() - panel visible=%s, width=%d, height=%d", 
                   self.isVisible(), self.width(), self.height())
        logger.info("AIResponsePanel.finish_response() - active_card=%s, cards count=%d", 
                   (self._active_card is not None), len(self._cards))
        
        self._remove_thinking()
        
        # If no active card yet, reuse the most recent card instead of
        # destroying and recreating the panel contents.
        if self._active_card is None:
            if self._cards:
                self._active_card = self._cards[-1]
            else:
                logger.warning("AIResponsePanel.finish_response() - no active card, creating one")
                card = AIResponseCard(title="Answer", parent=self._responses_container)
                self._wire_camera_mode(card)
                self._active_card = card
                self._cards = [card]
                self._responses_layout.addWidget(card, 1)
        
        # Now populate the card with the response
        if self._active_card and full_text is not None:
            logger.info("AIResponsePanel.finish_response() - setting text on card, text length=%d", len(full_text))
            self._active_card.set_text(full_text)
            self._responses_layout.setStretchFactor(self._active_card, 1)
            logger.info("AIResponsePanel.finish_response() - text set, card visible=%s, card height=%d", 
                       self._active_card.isVisible(), self._active_card.height())
            
            self._responses_layout.invalidate()
            self._responses_layout.activate()
            self._content.updateGeometry()
            self._content.update()
        else:
            logger.warning("AIResponsePanel.finish_response() - no active card or text is None")
        
        self._active_card = None

    def show_thinking(self, message: str = "Generating response...") -> None:
        """Show a loading indicator while the AI is processing."""
        self.activity_started.emit()
        if self._active_card is not None or self._cards:
            return
        if self._thinking_label is None:
            lbl = QLabel(message)
            lbl.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
            lbl.setStyleSheet(
                f"color: {TEXT_SECONDARY}; font-style: italic; padding: 8px;"
            )
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._thinking_label = lbl
            self._responses_layout.addWidget(lbl)
            self._scroll_to_bottom()
        else:
            self._thinking_label.setText(message)

    def show_error(self, message: str, title: str = "Answer") -> None:
        """Display an error message in the panel."""
        self.activity_started.emit()
        self._remove_thinking()
        self.clear_responses()
        card = AIResponseCard(f"⚠ {message}", title=title, parent=self._responses_container)
        self._wire_camera_mode(card)
        card.setStyleSheet(
            "background-color: #3d0000; border: 1px solid #f85149; "
            "border-radius: 8px; padding: 4px;"
        )
        self._cards = [card]
        self._responses_layout.addWidget(card, 1)
        self._scroll_to_bottom()

    def get_last_response_text(self) -> str:
        """Return the text of the most recent response card, or empty string."""
        if self._cards:
            return self._cards[-1].get_text()
        return ""

    def append_continuation(self, continuation_text: str) -> None:
        """Append continuation text to the most recent response card.

        Used when a response is delivered in multiple chunks and the latest
        chunk should be merged into the existing response card.
        """
        self._remove_thinking()
        if not self._cards:
            return
        card = self._cards[-1]
        existing = card.get_text()
        combined = f"{existing}\n\n{continuation_text}" if existing else continuation_text
        card.set_text(combined)

    def clear_responses(self) -> None:
        """Remove all response cards."""
        self._remove_thinking()
        self._active_card = None
        self._cards.clear()
        while self._responses_layout.count() > 0:
            item = self._responses_layout.takeAt(0)
            if item and item.widget():
                widget = item.widget()
                self._responses_layout.removeWidget(widget)
                widget.setParent(None)
                widget.close()
                widget.deleteLater()

    def clear_prompt_input(self) -> None:
        self._prompt_input.clear()

    def toggle_visibility(self) -> None:
        """Collapse or expand the response list to reclaim space."""
        if self._collapsed:
            self._content.show()
            self.setMaximumHeight(QWIDGETSIZE_MAX)
            self._responses_layout.invalidate()
            self._responses_layout.activate()
            self._content.updateGeometry()
            self._toggle_btn.setText("Hide")
            self._collapsed = False
            self._scroll_to_bottom()
            return

        self._content.hide()
        self.setMaximumHeight(34)
        self._toggle_btn.setText("Show")
        self._collapsed = True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _remove_thinking(self) -> None:
        if self._thinking_label:
            self._responses_layout.removeWidget(self._thinking_label)
            self._thinking_label.setParent(None)
            self._thinking_label.close()
            self._thinking_label.deleteLater()
            self._thinking_label = None

    def _submit_prompt(self, refine: bool) -> None:
        text = self._prompt_input.text().strip()
        if not text:
            return
        self.text_prompt_submitted.emit(text, refine)
        self._prompt_input.clear()

    def _on_ask_clicked(self) -> None:
        self._submit_prompt(refine=False)

    def _on_refine_clicked(self) -> None:
        self._submit_prompt(refine=True)

    def _scroll_to_bottom(self) -> None:
        from PyQt6.QtCore import QTimer

        def scroll() -> None:
            self._responses_layout.invalidate()
            self._responses_layout.activate()
            self._content.updateGeometry()

        QTimer.singleShot(10, scroll)
