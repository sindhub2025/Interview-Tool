"""
AI response panel: displays streaming AI-generated suggestions.
"""

from __future__ import annotations

from html import escape
import re
from typing import List

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QTextCursor
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
    """A single AI answer card with a Copy action."""

    def __init__(self, text: str = "", title: str = "Answer", parent=None) -> None:
        super().__init__(parent)
        self._text = text
        self._title = title
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

    def _render_text(self) -> None:
        self._text_edit.setHtml(render_response_html(self._text))
        cursor = self._text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._text_edit.setTextCursor(cursor)
        self._text_edit.ensureCursorVisible()

    def append_text(self, chunk: str) -> None:
        """Append a streaming chunk to the response."""
        self._text += chunk
        self._render_text()

    def set_text(self, text: str) -> None:
        from ghostmic.utils.logger import get_logger

        logger = get_logger(__name__)
        self._text = text
        self._render_text()
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
        if self._thinking_label is None:
            lbl = QLabel(message)
            lbl.setStyleSheet(
                f"color: {TEXT_SECONDARY}; font-style: italic; padding: 8px;"
            )
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._thinking_label = lbl
            self._responses_layout.addWidget(lbl, 1)
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
