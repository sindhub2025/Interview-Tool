"""
AI response panel: displays streaming AI-generated suggestions.
"""

from __future__ import annotations

from typing import List

from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
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
        # Set minimum height so text is visible
        self._text_edit.setMinimumHeight(60)
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
        # Force layout recalculation
        self._text_edit.adjustSize()
        logger.info("AIResponseCard.set_text() called - set %d chars to QTextEdit, new height=%d", 
                   len(text), self._text_edit.height())

    def get_text(self) -> str:
        return self._text

    def _copy(self) -> None:
        QApplication.clipboard().setText(self._text)


class AIResponsePanel(QScrollArea):
    """Panel that shows the last AI responses with streaming support.

    Keeps the last 3 responses visible (scrollable).

    Signals emitted through the parent widget's ``regenerate_requested``.
    """

    MAX_RESPONSES = 3

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cards: List[AIResponseCard] = []
        self._active_card: AIResponseCard | None = None
        self._thinking_label: QLabel | None = None
        self._collapsed = False

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            f"QScrollArea {{"
            f" background-color: {BG_DEEP};"
            f" border: 1px solid {BORDER};"
            f" border-radius: 10px;"
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

        outer = QWidget()
        outer.setObjectName("ai_panel_outer")
        outer_layout = QVBoxLayout(outer)
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
        self._layout = QVBoxLayout(self._content)
        self._layout.setSpacing(8)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.addStretch()
        outer_layout.addWidget(self._content)

        self.setWidget(outer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_response(self) -> None:
        """Create a new blank response card ready for streaming."""
        from ghostmic.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("AIResponsePanel.start_response() called")
        self._remove_thinking()
        card = AIResponseCard()
        self._active_card = card
        self._cards.append(card)
        # Evict oldest if over limit
        while len(self._cards) > self.MAX_RESPONSES:
            old = self._cards.pop(0)
            old.deleteLater()
        
        # Insert before the stretch
        insert_pos = self._layout.count() - 1
        self._layout.insertWidget(insert_pos, card)
        
        # CRITICAL: Explicitly show the card
        card.show()
        
        logger.info("AIResponsePanel.start_response() - card inserted at position %d, layout count=%d", 
                   insert_pos, self._layout.count())
        
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
            logger.info("AIResponsePanel.finish_response() - text set, card visible=%s, card height=%d", 
                       self._active_card.isVisible(), self._active_card.height())
            
            # Force layout recalculation
            self._layout.update()
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
            self._layout.insertWidget(self._layout.count() - 1, lbl)
            self._scroll_to_bottom()

    def show_error(self, message: str) -> None:
        """Display an error message in the panel."""
        self._remove_thinking()
        card = AIResponseCard(f"⚠ {message}")
        card.setStyleSheet(
            "background-color: #3d0000; border: 1px solid #f85149; "
            "border-radius: 8px; padding: 4px;"
        )
        self._layout.insertWidget(self._layout.count() - 1, card)
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
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

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
            self._thinking_label.deleteLater()
            self._thinking_label = None

    def _scroll_to_bottom(self) -> None:
        from PyQt6.QtCore import QTimer
        def scroll():
            # Force the scroll area to update layout
            self._content.adjustSize()
            # Get scrollbar and set to maximum
            scrollbar = self.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        QTimer.singleShot(10, scroll)
