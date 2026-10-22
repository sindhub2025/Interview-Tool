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

    def __init__(self, text: str = "", title: str = "Answer", parent=None) -> None:
        super().__init__(parent)
        self._text = text
        self._title = title
        self._teleprompter_active = False
        self._reading_font_size = 24
        self._scroll_speed = 2
        self._scrolling = False
        self._camera_window = None
        self._scroll_timer = QTimer(self)
        self._scroll_timer.setInterval(40)
        self._scroll_timer.timeout.connect(self._advance_scroll)
        self._build_ui()

    def _build_ui(self) -> None:
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setStyleSheet(
            f"background-color: {BG_CARD}; border: 1px solid {BORDER}; "
            "border-radius: 8px;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # Top row: title + buttons
        header = QHBoxLayout()
        title_label = QLabel(self._title)
        title_label.setStyleSheet(
            f"color: {ACCENT_BLUE}; font-weight: bold; font-size: 10pt;"
        )
        header.addWidget(title_label)
        header.addStretch()

        self._copy_btn = QPushButton("Copy")
        self._copy_btn.setFixedSize(66, 26)
        self._copy_btn.setStyleSheet(
            "font-size: 10pt; font-weight: 600; padding: 0px 12px;"
        )
        self._copy_btn.clicked.connect(self._copy)
        header.addWidget(self._copy_btn)

        self._teleprompter_btn = QPushButton("Teleprompter")
        self._teleprompter_btn.setFixedSize(104, 26)
        self._teleprompter_btn.setStyleSheet(
            "font-size: 10pt; font-weight: 600; padding: 0px 8px;"
        )
        self._teleprompter_btn.clicked.connect(self.toggle_teleprompter)
        header.addWidget(self._teleprompter_btn)
        layout.addLayout(header)

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

        self._teleprompter_toolbar = QWidget()
        toolbar = QHBoxLayout(self._teleprompter_toolbar)
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(4)
        self._font_down_btn = QPushButton("A−")
        self._font_up_btn = QPushButton("A+")
        self._start_scroll_btn = QPushButton("Start")
        self._camera_btn = QPushButton("Camera Mode")
        self._speed_down_btn = QPushButton("−")
        self._speed_up_btn = QPushButton("+")
        self._speed_label = QLabel("Speed 2/5")
        for button in (self._font_down_btn, self._font_up_btn,
                       self._start_scroll_btn, self._speed_down_btn,
                       self._speed_up_btn):
            button.setMinimumWidth(38)
            toolbar.addWidget(button)
        toolbar.addWidget(self._speed_label)
        toolbar.addWidget(self._camera_btn)
        toolbar.addStretch()
        self._font_down_btn.clicked.connect(lambda: self._change_font_size(-2))
        self._font_up_btn.clicked.connect(lambda: self._change_font_size(2))
        self._start_scroll_btn.clicked.connect(self._toggle_scrolling)
        self._speed_down_btn.clicked.connect(lambda: self._change_speed(-1))
        self._speed_up_btn.clicked.connect(lambda: self._change_speed(1))
        self._camera_btn.clicked.connect(self.toggle_camera_mode)
        self._teleprompter_toolbar.setVisible(False)
        layout.addWidget(self._teleprompter_toolbar)

        self._reading_view = QTextBrowser()
        self._reading_view.setReadOnly(True)
        self._reading_view.setOpenExternalLinks(False)
        self._reading_view.setStyleSheet(
            f"background-color: {BG_DEEP}; color: {TEXT_PRIMARY}; "
            "border: none; border-radius: 6px; padding: 20px 24px;"
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

    def _change_font_size(self, delta: int) -> None:
        self._reading_font_size = max(16, min(40, self._reading_font_size + delta))
        self._font_down_btn.setEnabled(self._reading_font_size > 16)
        self._font_up_btn.setEnabled(self._reading_font_size < 40)
        if self._teleprompter_active:
            self._render_reading_view()

    def _change_speed(self, delta: int) -> None:
        self._scroll_speed = max(1, min(5, self._scroll_speed + delta))
        self._speed_label.setText(f"Speed {self._scroll_speed}/5")

    def _toggle_scrolling(self) -> None:
        scrollbar = self._reading_view.verticalScrollBar()
        if not self._scrolling and scrollbar.value() >= scrollbar.maximum() and scrollbar.maximum() > 0:
            scrollbar.setValue(scrollbar.minimum())
        self._scrolling = not self._scrolling
        if self._scrolling:
            self._scroll_timer.start()
            self._start_scroll_btn.setText("Pause")
        else:
            self._scroll_timer.stop()
            self._start_scroll_btn.setText("Resume" if scrollbar.value() > scrollbar.minimum() else "Start")

    def _advance_scroll(self) -> None:
        scrollbar = self._reading_view.verticalScrollBar()
        value = min(scrollbar.maximum(), scrollbar.value() + self._scroll_speed)
        scrollbar.setValue(value)
        if value >= scrollbar.maximum():
            self._scrolling = False
            self._scroll_timer.stop()
            self._start_scroll_btn.setText("Replay")

    def _camera_geometry(self):
        screen = QApplication.screenAt(self.mapToGlobal(self.rect().center()))
        if screen is None:
            screen = QApplication.primaryScreen()
        if screen is None:
            return None
        available = screen.availableGeometry()
        width = min(640, max(420, available.width() - 32))
        height = min(210, max(150, available.height() // 5))
        x = available.x() + (available.width() - width) // 2
        y = available.y() + min(80, max(24, available.height() // 20))
        return available, x, y, width, height

    def _center_camera_window(self) -> None:
        if self._camera_window is None:
            return
        geometry = self._camera_geometry()
        if geometry is not None:
            _, x, y, width, height = geometry
            self._camera_window.setGeometry(x, y, width, height)

    def toggle_camera_mode(self) -> None:
        if self._camera_window is not None:
            self._exit_camera_mode()
            return
        from PyQt6.QtCore import QRect
        self._camera_window = QFrame(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self._camera_window.setObjectName("teleprompter_camera_window")
        self._camera_window.setStyleSheet(
            f"QFrame#teleprompter_camera_window {{ background-color: {BG_DEEP}; "
            f"border: 1px solid {BORDER}; border-radius: 8px; }}"
        )
        camera_layout = QVBoxLayout(self._camera_window)
        camera_layout.setContentsMargins(8, 6, 8, 6)
        camera_layout.addWidget(self._teleprompter_toolbar)
        camera_layout.addWidget(self._reading_view, 1)
        self._camera_window.setMinimumSize(320, 120)
        self._camera_window.resize(620, 190)
        self._center_camera_window()
        self._camera_btn.setText("Exit Camera Mode")
        self._camera_window.destroyed.connect(self._camera_window_closed)
        self._install_camera_shortcuts()
        self._camera_window.show()
        self._reading_view.setFocus()

    def _install_camera_shortcuts(self) -> None:
        if self._camera_window is None:
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
            "Esc": self._exit_camera_mode,
        }
        self._camera_shortcuts = []
        for key, callback in shortcuts.items():
            shortcut = QShortcut(QKeySequence(key), self._camera_window)
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

    def _exit_camera_mode(self) -> None:
        window = self._camera_window
        if window is None:
            return
        self._camera_window = None
        window.close()
        self._teleprompter_toolbar.setParent(self)
        self._reading_view.setParent(self)
        card_layout = self.layout()
        if card_layout is not None:
            card_layout.addWidget(self._teleprompter_toolbar)
            card_layout.addWidget(self._reading_view)
        self._camera_btn.setText("Camera Mode")
        self._teleprompter_toolbar.setVisible(self._teleprompter_active)
        self._reading_view.setVisible(self._teleprompter_active)

    def _camera_window_closed(self) -> None:
        if self._camera_window is not None:
            self._camera_window = None
            self._teleprompter_toolbar.setParent(self)
            self._reading_view.setParent(self)
            if self.layout() is not None:
                self.layout().addWidget(self._teleprompter_toolbar)
                self.layout().addWidget(self._reading_view)
            self._camera_btn.setText("Camera Mode")

    def toggle_teleprompter(self) -> None:
        active = not self._teleprompter_active
        if not active and self._camera_window is not None:
            self._exit_camera_mode()
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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cards: List[AIResponseCard] = []
        self._active_card: AIResponseCard | None = None
        self._thinking_label: QLabel | None = None
        self._collapsed = False

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

        input_row = QHBoxLayout()
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

        content_layout.addLayout(input_row)

        outer_layout.addWidget(self._content, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
