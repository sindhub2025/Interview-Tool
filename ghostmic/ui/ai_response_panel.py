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

from ghostmic.ui.styles import ACCENT_BLUE, BG_CARD, BG_MID, TEXT_SECONDARY


class AIResponseCard(QFrame):
    """A single AI response card with Copy and Regenerate actions."""

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(parent)
        self._text = text
        self._build_ui()

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
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
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
        self._text = text
        self._text_edit.setPlainText(text)

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

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._content = QWidget()
        self._layout = QVBoxLayout(self._content)
        self._layout.setSpacing(8)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.addStretch()
        self.setWidget(self._content)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_response(self) -> None:
        """Create a new blank response card ready for streaming."""
        self._remove_thinking()
        card = AIResponseCard()
        self._active_card = card
        self._cards.append(card)
        # Evict oldest if over limit
        while len(self._cards) > self.MAX_RESPONSES:
            old = self._cards.pop(0)
            old.deleteLater()
        self._layout.insertWidget(self._layout.count() - 1, card)
        self._scroll_to_bottom()

    def append_chunk(self, chunk: str) -> None:
        """Append a streaming chunk to the active response card."""
        if self._active_card is None:
            self.start_response()
        if self._active_card:
            self._active_card.append_text(chunk)

    def finish_response(self, full_text: str | None = None) -> None:
        """Mark the current response as complete."""
        if self._active_card and full_text is not None:
            self._active_card.set_text(full_text)
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

    def clear_responses(self) -> None:
        """Remove all response cards."""
        self._remove_thinking()
        self._active_card = None
        self._cards.clear()
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _remove_thinking(self) -> None:
        if self._thinking_label:
            self._thinking_label.deleteLater()
            self._thinking_label = None

    def _scroll_to_bottom(self) -> None:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(50, lambda: self.verticalScrollBar().setValue(
            self.verticalScrollBar().maximum()
        ))
