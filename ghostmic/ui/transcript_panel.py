"""
Transcript panel: scrollable, auto-scrolling display of
speaker/user transcript segments.
"""

from __future__ import annotations

import time
from typing import List

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QMenu,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ghostmic.core.transcription_engine import TranscriptSegment
from ghostmic.ui.styles import ACCENT_BLUE, FONT, TEXT_SECONDARY


class SegmentBubble(QFrame):
    """A single chat-bubble style transcript entry."""

    SPEAKER_BG = "#1e3a5f"
    USER_BG = "#1e5f3a"

    def __init__(self, segment: TranscriptSegment, parent=None) -> None:
        super().__init__(parent)
        self._segment = segment
        self._build_ui()

    def _build_ui(self) -> None:
        is_speaker = self._segment.source == "speaker"
        bg = self.SPEAKER_BG if is_speaker else self.USER_BG
        align = Qt.AlignmentFlag.AlignLeft if is_speaker else Qt.AlignmentFlag.AlignRight

        self.setStyleSheet(
            f"background-color: {bg}; border-radius: 8px; padding: 4px 8px;"
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(8, 4, 8, 4)

        # Text
        text_label = QLabel(self._segment.text)
        text_label.setWordWrap(True)
        text_label.setAlignment(align)
        text_label.setFont(QFont(FONT, 11))
        layout.addWidget(text_label)

        # Timestamp
        ts = time.strftime("%H:%M:%S", time.localtime(self._segment.timestamp))
        ts_label = QLabel(ts)
        ts_label.setAlignment(align)
        ts_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 9pt;")
        layout.addWidget(ts_label)


class TranscriptPanel(QScrollArea):
    """Auto-scrolling panel that shows TranscriptSegments as chat bubbles.

    Right-click for context menu (Copy, Copy All, Clear).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._segments: List[TranscriptSegment] = []
        self._auto_scroll = True
        self._pending_placeholder: QLabel | None = None

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        self._content = QWidget()
        self._layout = QVBoxLayout(self._content)
        self._layout.setSpacing(6)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.addStretch()
        self.setWidget(self._content)

        # Track manual scroll
        self.verticalScrollBar().valueChanged.connect(self._on_scroll_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_segment(self, segment: TranscriptSegment) -> None:
        """Append a new TranscriptSegment bubble."""
        self._remove_placeholder()
        self._segments.append(segment)
        bubble = SegmentBubble(segment)
        # Insert before the trailing stretch (last item)
        self._layout.insertWidget(self._layout.count() - 1, bubble)
        if self._auto_scroll:
            QTimer.singleShot(50, self._scroll_to_bottom)

    def show_pending(self, source: str) -> None:
        """Show a '…' placeholder while transcription is processing."""
        self._remove_placeholder()
        label = QLabel("…")
        label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-style: italic; padding: 4px 8px;"
        )
        align = (
            Qt.AlignmentFlag.AlignLeft
            if source == "speaker"
            else Qt.AlignmentFlag.AlignRight
        )
        label.setAlignment(align)
        self._pending_placeholder = label
        self._layout.insertWidget(self._layout.count() - 1, label)
        if self._auto_scroll:
            QTimer.singleShot(50, self._scroll_to_bottom)

    def clear_transcript(self) -> None:
        """Remove all bubbles and clear the segment history."""
        self._remove_placeholder()
        self._segments.clear()
        while self._layout.count() > 1:  # keep the stretch
            item = self._layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

    def get_full_text(self) -> str:
        """Return the full transcript as plain text."""
        lines = []
        for seg in self._segments:
            label = "Speaker" if seg.source == "speaker" else "You"
            ts = time.strftime("%H:%M:%S", time.localtime(seg.timestamp))
            lines.append(f"[{ts}] [{label}]: {seg.text}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Slots / internals
    # ------------------------------------------------------------------

    def _remove_placeholder(self) -> None:
        if self._pending_placeholder:
            self._pending_placeholder.deleteLater()
            self._pending_placeholder = None

    def _scroll_to_bottom(self) -> None:
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_scroll_changed(self, value: int) -> None:
        sb = self.verticalScrollBar()
        self._auto_scroll = value >= sb.maximum() - 20

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)
        copy_last = QAction("Copy Last", self)
        copy_all = QAction("Copy All", self)
        clear_action = QAction("Clear", self)
        copy_last.triggered.connect(self._copy_last)
        copy_all.triggered.connect(self._copy_all)
        clear_action.triggered.connect(self.clear_transcript)
        menu.addAction(copy_last)
        menu.addAction(copy_all)
        menu.addSeparator()
        menu.addAction(clear_action)
        menu.exec(self.mapToGlobal(pos))

    def _copy_last(self) -> None:
        if self._segments:
            QApplication.clipboard().setText(self._segments[-1].text)

    def _copy_all(self) -> None:
        QApplication.clipboard().setText(self.get_full_text())
