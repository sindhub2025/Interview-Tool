"""
AI response panel: displays streaming AI-generated suggestions.
"""

from __future__ import annotations

from typing import List

from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ghostmic.ui.styles import ACCENT_BLUE, BG_CARD, BG_DEEP, BG_MID, BORDER, TEXT_SECONDARY

QWIDGETSIZE_MAX = 16_777_215


class AIResponseCard(QFrame):
    """A single AI response card with Copy and Regenerate actions."""

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(parent)
        self._text = text
        self._build_ui()
        # Ensure the card is visible
        self.show()

    def _build_ui(self) -> None:
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setStyleSheet(
            f"background-color: {BG_CARD}; border: 1px solid #333366; "
            "border-radius: 8px; padding: 4px;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # Top row: title + buttons
        header = QHBoxLayout()
        title = QLabel("AI Suggestion")
        title.setStyleSheet(f"color: {ACCENT_BLUE}; font-weight: bold; font-size: 10pt;")
        header.addWidget(title)
        header.addStretch()

        self._copy_btn = QPushButton("Copy")
        self._copy_btn.setFixedHeight(24)
        self._copy_btn.setFixedWidth(54)
        self._copy_btn.clicked.connect(self._copy)
        header.addWidget(self._copy_btn)
        layout.addLayout(header)

        # Response text
        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setPlainText(self._text)
        self._text_edit.setStyleSheet(
            f"background-color: {BG_MID}; border: none; "
            "border-radius: 4px; padding: 4px;"
        )
        self._text_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._text_edit.setMinimumHeight(0)
        layout.addWidget(self._text_edit)

    def append_text(self, chunk: str) -> None:
        """Append a streaming chunk to the response."""
        cursor = self._text_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(chunk)
        self._text_edit.setTextCursor(cursor)
        self._text_edit.ensureCursorVisible()
        self._text = self._text_edit.toPlainText()

    def set_text(self, text: str) -> None:
        from ghostmic.utils.logger import get_logger
        logger = get_logger(__name__)
        self._text = text
        self._text_edit.setPlainText(text)
        self._text_edit.updateGeometry()
        logger.info("AIResponseCard.set_text() called - set %d chars to QTextEdit, new height=%d", 
                   len(text), self._text_edit.height())

    def get_text(self) -> str:
        return self._text

    def _copy(self) -> None:
        QApplication.clipboard().setText(self._text)


class AIResponsePanel(QWidget):
    """Panel that shows only the latest AI response with streaming support."""

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
        header_layout.setContentsMargins(8, 4, 8, 4)
        header_layout.setSpacing(6)

        title = QLabel("AI Responses")
        title.setStyleSheet(f"color: {ACCENT_BLUE}; font-weight: bold;")
        header_layout.addWidget(title)
        header_layout.addStretch()

        self._toggle_btn = QPushButton("Hide")
        self._toggle_btn.setFixedHeight(22)
        self._toggle_btn.setFixedWidth(54)
        self._toggle_btn.clicked.connect(self.toggle_visibility)
        header_layout.addWidget(self._toggle_btn)

        outer_layout.addWidget(header)

        self._content = QWidget()
        self._content.setObjectName("ai_panel_content")
        self._content.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        content_layout = QVBoxLayout(self._content)
        content_layout.setSpacing(8)
        content_layout.setContentsMargins(6, 6, 6, 6)

        self._responses_container = QWidget()
        self._responses_layout = QVBoxLayout(self._responses_container)
        self._responses_layout.setSpacing(8)
        self._responses_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self._responses_container, 1)

        input_row = QHBoxLayout()
        input_row.setSpacing(6)

        self._prompt_input = QLineEdit()
        self._prompt_input.setPlaceholderText(
            "Ask AI a question or type a refinement for the current response"
        )
        self._prompt_input.returnPressed.connect(self._on_ask_clicked)
        input_row.addWidget(self._prompt_input, 1)

        self._ask_btn = QPushButton("Ask")
        self._ask_btn.setFixedHeight(26)
        self._ask_btn.clicked.connect(self._on_ask_clicked)
        input_row.addWidget(self._ask_btn)

        self._refine_btn = QPushButton("Refine")
        self._refine_btn.setFixedHeight(26)
        self._refine_btn.clicked.connect(self._on_refine_clicked)
        input_row.addWidget(self._refine_btn)

        content_layout.addLayout(input_row)

        outer_layout.addWidget(self._content, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_response(self) -> None:
        """Create a new blank response card ready for streaming."""
        from ghostmic.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("AIResponsePanel.start_response() called")
        self.clear_responses()
        self._remove_thinking()
        card = AIResponseCard()
        self._active_card = card
        self._cards = [card]
        self._responses_layout.addWidget(card, 1)
        
        # CRITICAL: Explicitly show the card
        card.show()
        
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
        logger.info("AIResponsePanel.finish_response() - panel visible=%s, width=%d, height=%d", 
                   self.isVisible(), self.width(), self.height())
        logger.info("AIResponsePanel.finish_response() - active_card=%s, cards count=%d", 
                   (self._active_card is not None), len(self._cards))
        
        self._remove_thinking()
        
        # If no active card yet, create one before setting text
        if self._active_card is None:
            logger.warning("AIResponsePanel.finish_response() - no active card, creating one")
            self.start_response()
        
        # Now populate the card with the response
        if self._active_card and full_text is not None:
            logger.info("AIResponsePanel.finish_response() - setting text on card, text length=%d", len(full_text))
            self._active_card.set_text(full_text)
            self._responses_layout.setStretchFactor(self._active_card, 1)
            logger.info("AIResponsePanel.finish_response() - text set, card visible=%s, card height=%d", 
                       self._active_card.isVisible(), self._active_card.height())
            
            # Force layout recalculation
            self._responses_layout.update()
            self._content.adjustSize()
        else:
            logger.warning("AIResponsePanel.finish_response() - no active card or text is None")
        
        self._active_card = None

    def show_thinking(self) -> None:
        """Show a loading indicator while the AI is processing."""
        if self._thinking_label is None:
            lbl = QLabel("🤔 Generating response…")
            lbl.setStyleSheet(
                f"color: {TEXT_SECONDARY}; font-style: italic; padding: 8px;"
            )
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._thinking_label = lbl
            self._responses_layout.addWidget(lbl, 1)
            self._scroll_to_bottom()

    def show_error(self, message: str) -> None:
        """Display an error message in the panel."""
        self._remove_thinking()
        self.clear_responses()
        card = AIResponseCard(f"⚠ {message}")
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
            self._content.adjustSize()
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
            self._content.adjustSize()

        QTimer.singleShot(10, scroll)
