"""
Transcript panel: scrollable, auto-scrolling display of
speaker/user transcript segments.
"""

from __future__ import annotations

import time
from typing import List

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QFrame,
    QLabel,
    QMenu,
    QPushButton,
    QTextEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ghostmic.domain import TranscriptSegment
from ghostmic.ui.styles import FONT, FONT_SIZE_SMALL, TEXT_SECONDARY


class SegmentBubble(QFrame):
    """A single chat-bubble style transcript entry."""

    normalize_requested = pyqtSignal(object, str)
    send_requested = pyqtSignal(object, str)
    text_edited = pyqtSignal(object, str)

    SPEAKER_BG = "#1e3a5f"
    USER_BG = "#1e5f3a"

    def __init__(self, segment: TranscriptSegment, parent=None) -> None:
        super().__init__(parent)
        self._segment = segment
        self._text_editor: QTextEdit | None = None
        self._status_label: QLabel | None = None
        self._normalize_btn: QPushButton | None = None
        self._send_btn: QPushButton | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        is_speaker = self._segment.source == "speaker"
        bg = self.SPEAKER_BG if is_speaker else self.USER_BG
        align = Qt.AlignmentFlag.AlignLeft if is_speaker else Qt.AlignmentFlag.AlignRight

        self.setStyleSheet(
            f"background-color: {bg}; border-radius: 8px; padding: 8px 12px;"
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(10, 6, 10, 6)

        # Text
        if is_speaker:
            editor = QTextEdit()
            editor.setPlainText(self._segment.text)
            editor.setAcceptRichText(False)
            editor.setFixedHeight(78)
            editor.setStyleSheet(
                "QTextEdit {"
                " background-color: rgba(13, 17, 23, 0.45);"
                " border: 1px solid rgba(88, 166, 255, 0.35);"
                " border-radius: 6px;"
                " padding: 4px 6px;"
                " color: #f0f6fc;"
                "}"
            )
            editor.textChanged.connect(self._on_text_changed)
            self._text_editor = editor
            layout.addWidget(editor)

            actions = QHBoxLayout()
            actions.setContentsMargins(0, 0, 0, 0)
            actions.addStretch()

            normalize_btn = QPushButton("Normalize")
            normalize_btn.setFixedHeight(24)
            normalize_btn.setStyleSheet(
                "QPushButton {"
                " background-color: rgba(88, 166, 255, 0.18);"
                " border: 1px solid rgba(88, 166, 255, 0.55);"
                " border-radius: 5px;"
                " color: #dce7ff;"
                " font-size: 9pt;"
                " padding: 0px 10px;"
                "}"
                "QPushButton:hover { background-color: rgba(88, 166, 255, 0.30); }"
            )
            normalize_btn.clicked.connect(self._on_normalize_clicked)
            actions.addWidget(normalize_btn)
            self._normalize_btn = normalize_btn

            send_btn = QPushButton("Send to AI")
            send_btn.setFixedHeight(24)
            send_btn.setStyleSheet(
                "QPushButton {"
                " background-color: rgba(63, 185, 80, 0.18);"
                " border: 1px solid rgba(63, 185, 80, 0.55);"
                " border-radius: 5px;"
                " color: #d2ffd9;"
                " font-size: 9pt;"
                " padding: 0px 10px;"
                " font-weight: 600;"
                "}"
                "QPushButton:hover { background-color: rgba(63, 185, 80, 0.30); }"
            )
            send_btn.clicked.connect(self._on_send_clicked)
            actions.addWidget(send_btn)
            self._send_btn = send_btn

            layout.addLayout(actions)

            status_label = QLabel("")
            status_label.setStyleSheet(
                f"color: {TEXT_SECONDARY}; font-size: {FONT_SIZE_SMALL}pt;"
            )
            status_label.setVisible(False)
            self._status_label = status_label
            layout.addWidget(status_label)
        else:
            text_label = QLabel(self._segment.text)
            text_label.setWordWrap(True)
            text_label.setAlignment(align)
            text_label.setFont(QFont(FONT, 11))
            layout.addWidget(text_label)

        # Timestamp
        ts = time.strftime("%H:%M:%S", time.localtime(self._segment.timestamp))
        ts_label = QLabel(ts)
        ts_label.setAlignment(align)
        ts_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: {FONT_SIZE_SMALL}pt;"
        )
        layout.addWidget(ts_label)

    def segment(self) -> TranscriptSegment:
        return self._segment

    def current_text(self) -> str:
        if self._text_editor is not None:
            return self._text_editor.toPlainText().strip()
        return str(self._segment.text).strip()

    def set_segment_text(self, text: str) -> None:
        cleaned = str(text).strip()
        if not cleaned:
            return
        self._segment.text = cleaned
        if self._text_editor is not None:
            previous = self._text_editor.blockSignals(True)
            self._text_editor.setPlainText(cleaned)
            self._text_editor.blockSignals(previous)

    def set_normalization_busy(self, busy: bool, status: str = "") -> None:
        if self._normalize_btn is not None:
            self._normalize_btn.setEnabled(not busy)
            self._normalize_btn.setText("Normalizing…" if busy else "Normalize")
        if self._send_btn is not None:
            self._send_btn.setEnabled(not busy)
        if self._text_editor is not None:
            self._text_editor.setReadOnly(busy)

        if self._status_label is not None:
            message = status.strip()
            if message:
                self._status_label.setText(message)
                self._status_label.setVisible(True)
            elif not busy:
                self._status_label.setVisible(False)

    def _on_normalize_clicked(self) -> None:
        text = self.current_text()
        if text:
            self.normalize_requested.emit(self._segment, text)

    def _on_send_clicked(self) -> None:
        text = self.current_text()
        if text:
            self.send_requested.emit(self._segment, text)

    def _on_text_changed(self) -> None:
        text = self.current_text()
        self._segment.text = text
        self.text_edited.emit(self._segment, text)


class TranscriptPanel(QScrollArea):
    """Auto-scrolling panel that shows TranscriptSegments as chat bubbles.

    Right-click for context menu (Copy, Copy All, Clear).
    """

    speaker_normalize_requested = pyqtSignal(object, str)
    speaker_send_requested = pyqtSignal(object, str)
    speaker_text_edited = pyqtSignal(object, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._segments: List[TranscriptSegment] = []
        self._bubbles_by_segment_id: dict[int, SegmentBubble] = {}
        self._auto_scroll = True
        self._pending_placeholder: QLabel | None = None

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        self._content = QWidget()
        self._layout = QVBoxLayout(self._content)
        self._layout.setSpacing(10)
        self._layout.setContentsMargins(8, 8, 8, 8)
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
        if segment.source == "speaker":
            bubble.normalize_requested.connect(self.speaker_normalize_requested.emit)
            bubble.send_requested.connect(self.speaker_send_requested.emit)
            bubble.text_edited.connect(self.speaker_text_edited.emit)
            self._bubbles_by_segment_id[id(segment)] = bubble
        # Insert before the trailing stretch (last item)
        self._layout.insertWidget(self._layout.count() - 1, bubble)
        if self._auto_scroll:
            QTimer.singleShot(50, self._scroll_to_bottom)

    def set_segment_text(self, segment: TranscriptSegment, text: str) -> None:
        bubble = self._bubbles_by_segment_id.get(id(segment))
        if bubble is None:
            return
        bubble.set_segment_text(text)

    def set_segment_normalization_busy(
        self,
        segment: TranscriptSegment,
        busy: bool,
        status: str = "",
    ) -> None:
        bubble = self._bubbles_by_segment_id.get(id(segment))
        if bubble is None:
            return
        bubble.set_normalization_busy(busy, status)

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
        self._bubbles_by_segment_id.clear()
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
