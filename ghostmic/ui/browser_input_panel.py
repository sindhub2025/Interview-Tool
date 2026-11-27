"""
BrowserInputPanel — PyQt6 widget for displaying remotely-received messages.

Design constraints:
- Must NOT import or reference Supabase, the AI engine, or audio modules.
- Receives BrowserMessage objects via Qt signals.
- Renders all content as untrusted plain text (no HTML rendering).
- Code is displayed with Pygments syntax highlighting when available.
"""

from __future__ import annotations

import html
import sys
import textwrap
from datetime import datetime, timezone
from typing import List, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QClipboard, QColor, QFont, QFontDatabase
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ghostmic.domain.browser_message import BrowserMessage
from ghostmic.services.browser_bridge_service import (
    STATE_CONNECTED,
    STATE_CONNECTING,
    STATE_DISCONNECTED,
    STATE_ERROR,
    STATE_RECONNECTING,
    STATE_STARTING_SESSION,
    STATE_WAITING_FOR_PAIRING,
    STATE_CLOSING,
)
from ghostmic.ui.styles import (
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    BG_CARD,
    BG_DEEP,
    BG_MID,
    BORDER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pygments (optional — graceful fallback to plain text)
# ---------------------------------------------------------------------------

try:
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import get_lexer_by_name, TextLexer
    from pygments.util import ClassNotFound

    _PYGMENTS_AVAILABLE = True
except ImportError:
    _PYGMENTS_AVAILABLE = False
    logger.debug("Pygments not available; code will be displayed as plain text.")

# ---------------------------------------------------------------------------
# Language → Pygments alias map
# ---------------------------------------------------------------------------

_LANG_LEXER: dict[str, str] = {
    "python": "python3",
    "javascript": "javascript",
    "typescript": "typescript",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "csharp": "csharp",
    "go": "go",
    "rust": "rust",
    "sql": "sql",
    "bash": "bash",
    "powershell": "powershell",
    "json": "json",
    "yaml": "yaml",
    "html": "html",
    "css": "css",
    "php": "php",
    "ruby": "ruby",
    "plaintext": "text",
}

# Dark Pygments style compatible with our background
_PYGMENTS_STYLE = "monokai"

# Maximum messages kept in memory
DEFAULT_MAX_MESSAGES = 100

# ---------------------------------------------------------------------------
# Message type display helpers
# ---------------------------------------------------------------------------

_TYPE_LABELS = {
    "question": ("❓ Question", ACCENT_BLUE),
    "code": ("💻 Code", "#c678dd"),
    "answer": ("💡 Answer", ACCENT_GREEN),
}

# ---------------------------------------------------------------------------
# Syntax highlighting
# ---------------------------------------------------------------------------


def _highlight_code(code: str, language: Optional[str]) -> str:
    """
    Return HTML-highlighted code or plain escaped text.

    The returned HTML is set directly on a QTextEdit; it must be safe.
    Content is always treated as untrusted plain text — Pygments only uses it
    as source input, not as markup.
    """
    if not _PYGMENTS_AVAILABLE or not language:
        return f"<pre style='margin:0;white-space:pre;'>{html.escape(code)}</pre>"

    lexer_name = _LANG_LEXER.get(language, "text")
    try:
        from pygments.lexers import get_lexer_by_name, TextLexer
        from pygments.util import ClassNotFound
        from pygments.formatters import HtmlFormatter
        from pygments import highlight as _highlight

        lexer = get_lexer_by_name(lexer_name)
        formatter = HtmlFormatter(
            style=_PYGMENTS_STYLE,
            noclasses=True,
            prestyles="margin:0;white-space:pre;overflow-x:auto;font-family:Consolas,monospace;font-size:10pt;",
            nowrap=False,
        )
        return _highlight(code, lexer, formatter)
    except Exception:
        return f"<pre style='margin:0;white-space:pre;'>{html.escape(code)}</pre>"


# ---------------------------------------------------------------------------
# MessageCard widget
# ---------------------------------------------------------------------------


class MessageCard(QFrame):
    """A single message card inside the scrollable list."""

    delete_requested = pyqtSignal(str)  # emits message_id

    def __init__(self, msg: BrowserMessage, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._message = msg
        self._build_ui()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.setObjectName("message_card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setStyleSheet(
            "QFrame#message_card {"
            "  background-color: rgba(33, 38, 45, 0.88);"
            "  border: 1px solid rgba(48, 54, 61, 0.9);"
            "  border-radius: 10px;"
            "  margin: 2px 0px;"
            "}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # Header row
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        label_text, label_color = _TYPE_LABELS.get(
            self._message.type, (self._message.type, TEXT_SECONDARY)
        )
        type_label = QLabel(label_text)
        type_label.setStyleSheet(
            f"color: {label_color}; font-size: 9pt; font-weight: 700;"
            " border: 1px solid rgba(139, 148, 158, 0.22);"
            " border-radius: 6px; padding: 2px 8px;"
            " background-color: rgba(13, 17, 23, 0.45);"
        )
        header.addWidget(type_label)

        if self._message.language:
            lang_label = QLabel(self._message.language)
            lang_label.setStyleSheet(
                f"color: {TEXT_SECONDARY}; font-size: 8pt; font-weight: 600;"
                " border: 1px solid rgba(48, 54, 61, 0.9); border-radius: 5px; padding: 2px 6px;"
                " background-color: #0d1117;"
            )
            header.addWidget(lang_label)

        header.addStretch()

        ts = self._message.created_at
        try:
            # Try to format nicely
            for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S"):
                try:
                    dt = datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
                    ts = dt.strftime("%H:%M:%S")
                    break
                except ValueError:
                    continue
        except Exception:
            pass

        time_label = QLabel(ts)
        time_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 8pt; border: none; background: transparent;"
        )
        header.addWidget(time_label)

        copy_btn = QPushButton("Copy")
        copy_btn.setFixedHeight(24)
        copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_btn.setStyleSheet(
            "QPushButton {"
            f"  color: {ACCENT_BLUE}; font-size: 8.5pt; font-weight: 600;"
            "  border: 1px solid rgba(88,166,255,0.45); border-radius: 6px;"
            "  padding: 0px 9px; background: rgba(88,166,255,0.08);"
            "}"
            "QPushButton:hover { background: rgba(88,166,255,0.20); }"
        )
        copy_btn.clicked.connect(self._on_copy)
        header.addWidget(copy_btn)

        del_btn = QPushButton("✕")
        del_btn.setFixedSize(24, 24)
        del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        del_btn.setStyleSheet(
            "QPushButton {"
            f"  color: {TEXT_SECONDARY}; font-size: 9pt;"
            "  border: none; border-radius: 6px; background: transparent;"
            "}"
            "QPushButton:hover { color: #f85149; background: rgba(248,81,73,0.12); }"
        )
        del_btn.clicked.connect(self._on_delete)
        header.addWidget(del_btn)

        layout.addLayout(header)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet("border: none; background-color: rgba(48, 54, 61, 0.7);")
        layout.addWidget(sep)

        # Content
        if self._message.type == "code":
            self._build_code_view(layout)
        else:
            self._build_text_view(layout)

    def _build_code_view(self, layout: QVBoxLayout) -> None:
        view = QTextEdit()
        view.setReadOnly(True)
        view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        view.setMinimumHeight(60)
        view.setMaximumHeight(300)
        view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        view.setStyleSheet(
            f"QTextEdit {{ background-color: #0d1117; border: 1px solid {BORDER};"
            " border-radius: 4px; font-family: Consolas, 'Courier New', monospace; font-size: 10pt; }}"
        )

        highlighted = _highlight_code(self._message.content, self._message.language)
        view.setHtml(highlighted)
        layout.addWidget(view)

    def _build_text_view(self, layout: QVBoxLayout) -> None:
        view = QLabel()
        view.setTextFormat(Qt.TextFormat.PlainText)
        view.setText(self._message.content)
        view.setWordWrap(True)
        view.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        view.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 10.5pt;"
            " background: transparent; border: none;"
        )
        view.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(view)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_copy(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._message.content)

    def _on_delete(self) -> None:
        self.delete_requested.emit(self._message.message_id)

    @property
    def message_id(self) -> str:
        return self._message.message_id


# ---------------------------------------------------------------------------
# BrowserInputPanel
# ---------------------------------------------------------------------------


class BrowserInputPanel(QWidget):
    """
    Collapsible side panel that displays messages received from the browser.

    Public API (called from MainWindow or BrowserBridgeService signals):
        append_message(msg)       — add a new message card.
        on_state_changed(s, d)    — update connection state display.
        on_pairing_code_ready(c, url) — show pairing code and URL.
        on_session_closed()       — reset to disconnected state.
        set_collapsed(bool)       — toggle collapsed/expanded appearance.
        is_panel_expanded → bool  — query expansion state.

    Signals:
        start_session_requested()
        stop_session_requested()
        copy_pairing_code_requested()
    """

    start_session_requested = pyqtSignal()
    stop_session_requested = pyqtSignal()
    copy_pairing_code_requested = pyqtSignal()
    collapse_requested = pyqtSignal()

    def __init__(
        self,
        config: dict,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._messages: List[BrowserMessage] = []
        self._cards: dict[str, MessageCard] = {}  # message_id → card
        self._state = STATE_DISCONNECTED
        self._pairing_code = ""
        self._sender_url = config.get("browser_bridge", {}).get("sender_url", "")
        self._max_messages = int(
            config.get("browser_bridge", {}).get("max_messages", DEFAULT_MAX_MESSAGES)
        )
        self._unread_count = 0
        self._panel_expanded = True  # True = visible content; False = header-only
        self._auto_scroll = True
        self._build_ui()

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.setObjectName("browser_input_panel")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(300)
        self.setStyleSheet(
            "QWidget#browser_input_panel {"
            "  background-color: rgba(17, 19, 24, 244);"
            "  border-left: 1px solid rgba(48, 54, 61, 0.85);"
            "  border-radius: 0px;"
            "}"
        )

        self._outer_layout = QVBoxLayout(self)
        self._outer_layout.setContentsMargins(0, 0, 0, 0)
        self._outer_layout.setSpacing(0)

        # ── Header ──────────────────────────────────────────────────
        self._header_widget = QWidget()
        self._header_widget.setObjectName("bi_header")
        self._header_widget.setFixedHeight(38)
        self._header_widget.setStyleSheet(
            "QWidget#bi_header {"
            "  background-color: rgba(22, 27, 34, 0.85);"
            "  border-bottom: 1px solid rgba(48, 54, 61, 0.8);"
            "}"
        )
        header_layout = QHBoxLayout(self._header_widget)
        header_layout.setContentsMargins(12, 0, 10, 0)
        header_layout.setSpacing(8)

        self._status_dot = QLabel("●")
        self._status_dot.setStyleSheet(
            "color: #666666; font-size: 9pt; border: none; background: transparent;"
        )
        header_layout.addWidget(self._status_dot)

        self._title_label = QLabel("📡 Browser Input")
        self._title_label.setStyleSheet(
            "color: #79c0ff; font-size: 10pt; font-weight: 700; border: none; background: transparent;"
        )
        header_layout.addWidget(self._title_label)

        self._msg_count_label = QLabel("")
        self._msg_count_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 8.5pt; font-weight: 600;"
            " border: none; background: transparent;"
        )
        header_layout.addWidget(self._msg_count_label)

        header_layout.addStretch()

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setFixedHeight(26)
        self._clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clear_btn.setStyleSheet(
            "QPushButton {"
            f"  color: {ACCENT_RED}; font-size: 8.5pt; font-weight: 600;"
            "  border: 1px solid rgba(248,81,73,0.45); border-radius: 6px;"
            "  padding: 0px 10px; background: rgba(248,81,73,0.08);"
            "}"
            "QPushButton:hover { background: rgba(248,81,73,0.18); }"
        )
        self._clear_btn.clicked.connect(self._on_clear_all)
        header_layout.addWidget(self._clear_btn)

        self._collapse_btn = QPushButton("«")
        self._collapse_btn.setObjectName("bi_collapse_btn")
        self._collapse_btn.setFixedSize(26, 26)
        self._collapse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._collapse_btn.setStyleSheet(
            "QPushButton#bi_collapse_btn {"
            f"  color: {TEXT_SECONDARY}; font-size: 11pt; font-weight: 700;"
            "  border: 1px solid rgba(48, 54, 61, 0.9); border-radius: 6px;"
            "  padding: 0px 4px; background: transparent;"
            "}"
            "QPushButton#bi_collapse_btn:hover {"
            f"  color: {TEXT_PRIMARY}; background: rgba(139,148,158,0.18);"
            "}"
        )
        self._collapse_btn.clicked.connect(self.collapse_requested.emit)
        header_layout.addWidget(self._collapse_btn)

        self._outer_layout.addWidget(self._header_widget)

        # ── Collapsible body ─────────────────────────────────────────
        self._body_widget = QWidget()
        body_layout = QVBoxLayout(self._body_widget)
        body_layout.setContentsMargins(10, 10, 10, 10)
        body_layout.setSpacing(10)

        # Session area
        self._session_widget = self._build_session_area()
        body_layout.addWidget(self._session_widget)

        # Message scroll area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setStyleSheet(
            "QScrollArea { border: none; background-color: transparent; }"
        )

        self._messages_widget = QWidget()
        self._messages_widget.setStyleSheet("background-color: transparent;")
        self._messages_layout = QVBoxLayout(self._messages_widget)
        self._messages_layout.setContentsMargins(0, 0, 0, 0)
        self._messages_layout.setSpacing(8)
        self._messages_layout.addStretch()

        self._empty_label = QLabel("📡\n\nNo messages yet\nStart a session and connect a sender to begin.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setWordWrap(True)
        self._empty_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 9.5pt; padding: 24px 12px;"
            " border: none; background: transparent;"
        )
        self._messages_layout.insertWidget(0, self._empty_label)

        self._scroll.setWidget(self._messages_widget)
        self._scroll.verticalScrollBar().valueChanged.connect(self._on_scroll_changed)
        body_layout.addWidget(self._scroll, stretch=1)

        self._outer_layout.addWidget(self._body_widget, stretch=1)

    def _build_session_area(self) -> QWidget:
        w = QWidget()
        w.setObjectName("bi_session")
        w.setStyleSheet(
            "QWidget#bi_session {"
            "  background-color: rgba(22, 27, 34, 0.78);"
            "  border: 1px solid rgba(48, 54, 61, 0.85);"
            "  border-radius: 10px;"
            "}"
        )
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # Status label
        self._conn_status_label = QLabel("Not connected")
        self._conn_status_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 9.5pt; font-weight: 600;"
            " border: none; background: transparent;"
        )
        layout.addWidget(self._conn_status_label)

        # Pairing code area (hidden until session starts)
        self._pairing_widget = QWidget()
        self._pairing_widget.setObjectName("bi_pairing_card")
        self._pairing_widget.setStyleSheet(
            "QWidget#bi_pairing_card {"
            "  background-color: rgba(242, 204, 96, 0.08);"
            "  border: 1px solid rgba(242, 204, 96, 0.35);"
            "  border-radius: 8px;"
            "}"
        )
        pairing_layout = QHBoxLayout(self._pairing_widget)
        pairing_layout.setContentsMargins(12, 8, 10, 8)
        pairing_layout.setSpacing(8)

        code_col = QVBoxLayout()
        code_col.setContentsMargins(0, 0, 0, 0)
        code_col.setSpacing(1)
        code_hint = QLabel("PAIRING CODE")
        code_hint.setStyleSheet(
            "color: rgba(242, 204, 96, 0.75); font-size: 7.5pt; font-weight: 700;"
            " letter-spacing: 1px; border: none; background: transparent;"
        )
        code_col.addWidget(code_hint)

        self._pairing_code_label = QLabel("")
        self._pairing_code_label.setStyleSheet(
            "color: #f2cc60; font-size: 18pt; font-weight: 800; letter-spacing: 3px;"
            " border: none; background: transparent;"
        )
        code_col.addWidget(self._pairing_code_label)
        pairing_layout.addLayout(code_col)

        pairing_layout.addStretch()

        self._copy_code_btn = QPushButton("Copy Code")
        self._copy_code_btn.setFixedHeight(28)
        self._copy_code_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_code_btn.setStyleSheet(
            "QPushButton {"
            "  color: #f2cc60; font-size: 9pt; font-weight: 600;"
            "  border: 1px solid rgba(242,204,96,0.5); border-radius: 6px;"
            "  padding: 0px 12px; background: rgba(242,204,96,0.10);"
            "}"
            "QPushButton:hover { background: rgba(242,204,96,0.20); }"
        )
        self._copy_code_btn.clicked.connect(self._on_copy_pairing_code)
        pairing_layout.addWidget(self._copy_code_btn)

        self._pairing_widget.hide()
        layout.addWidget(self._pairing_widget)

        # Sender URL hint
        self._sender_url_label = QLabel("")
        self._sender_url_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 8.5pt; border: none; background: transparent;"
        )
        self._sender_url_label.setTextFormat(Qt.TextFormat.PlainText)
        self._sender_url_label.setWordWrap(True)
        self._sender_url_label.hide()
        layout.addWidget(self._sender_url_label)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._start_btn = QPushButton("▶ Start Session")
        self._start_btn.setFixedHeight(32)
        self._start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_btn.setStyleSheet(
            "QPushButton {"
            f"  color: {ACCENT_GREEN}; font-size: 9.5pt; font-weight: 700;"
            "  border: 1px solid rgba(63,185,80,0.65); border-radius: 6px;"
            "  padding: 0px 14px; background: rgba(63,185,80,0.14);"
            "}"
            "QPushButton:hover { background: rgba(63,185,80,0.24); }"
            "QPushButton:disabled { color: #555; border-color: #30363d; background: transparent; }"
        )
        self._start_btn.clicked.connect(self.start_session_requested)
        btn_row.addWidget(self._start_btn, 1)

        self._stop_btn = QPushButton("■ Stop Session")
        self._stop_btn.setFixedHeight(32)
        self._stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_btn.setStyleSheet(
            "QPushButton {"
            f"  color: {ACCENT_RED}; font-size: 9.5pt; font-weight: 700;"
            "  border: 1px solid rgba(248,81,73,0.65); border-radius: 6px;"
            "  padding: 0px 14px; background: rgba(248,81,73,0.14);"
            "}"
            "QPushButton:hover { background: rgba(248,81,73,0.24); }"
            "QPushButton:disabled { color: #555; border-color: #30363d; background: transparent; }"
        )
        self._stop_btn.clicked.connect(self.stop_session_requested)
        self._stop_btn.setEnabled(False)
        btn_row.addWidget(self._stop_btn, 1)

        layout.addLayout(btn_row)
        return w

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append_message(self, msg: BrowserMessage) -> None:
        """Add a new message card. Called from main thread via queued signal."""
        # Duplicate guard
        if msg.message_id in self._cards:
            return

        # Enforce max message limit
        while len(self._messages) >= self._max_messages:
            oldest = self._messages.pop(0)
            card = self._cards.pop(oldest.message_id, None)
            if card:
                card.deleteLater()

        self._messages.append(msg)
        card = MessageCard(msg)
        card.delete_requested.connect(self._on_delete_card)
        self._cards[msg.message_id] = card

        # Insert before the trailing stretch (last item)
        stretch_index = self._messages_layout.count() - 1
        self._messages_layout.insertWidget(stretch_index, card)

        # Remove empty state label if present
        if self._empty_label.isVisible():
            self._empty_label.hide()

        self._update_count_label()

        # Auto-scroll only if already at bottom
        QTimer.singleShot(0, self._maybe_scroll_to_bottom)

        # Unread tracking
        if not self._panel_expanded:
            self._unread_count += 1
            self._update_title_badge()

    def on_state_changed(self, state: str, detail: str) -> None:
        """Called when BrowserBridgeService emits connection_state_changed."""
        self._state = state
        self._update_state_ui(state, detail)

    def on_pairing_code_ready(self, code: str, sender_url: str) -> None:
        """Display the pairing code after a successful session creation."""
        self._pairing_code = code
        self._pairing_code_label.setText(code)
        self._pairing_widget.show()
        if sender_url:
            self._sender_url_label.setText(f"Sender page: {sender_url}")
            self._sender_url_label.show()

    def on_session_closed(self) -> None:
        """Reset session-related UI."""
        self._pairing_code = ""
        self._pairing_code_label.setText("")
        self._pairing_widget.hide()
        self._sender_url_label.hide()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._conn_status_label.setText("Session closed.")

    def set_collapsed(self, collapsed: bool) -> None:
        """Hide/show body widget (called from MainWindow toggle)."""
        self._body_widget.setVisible(not collapsed)
        self._panel_expanded = not collapsed
        if not collapsed:
            self._unread_count = 0
            self._update_title_badge()

    @property
    def is_panel_expanded(self) -> bool:
        return self._panel_expanded

    @property
    def unread_count(self) -> int:
        return self._unread_count

    # ------------------------------------------------------------------
    # Private UI helpers
    # ------------------------------------------------------------------

    def _update_state_ui(self, state: str, detail: str) -> None:
        STATE_DOT_COLORS = {
            STATE_DISCONNECTED: "#666666",
            STATE_STARTING_SESSION: "#f2cc60",
            STATE_WAITING_FOR_PAIRING: "#f2cc60",
            STATE_CONNECTING: "#f2cc60",
            STATE_CONNECTED: ACCENT_GREEN,
            STATE_RECONNECTING: "#f2cc60",
            STATE_ERROR: ACCENT_RED,
            STATE_CLOSING: TEXT_SECONDARY,
        }
        STATE_TEXTS = {
            STATE_DISCONNECTED: "Not connected",
            STATE_STARTING_SESSION: "Starting session…",
            STATE_WAITING_FOR_PAIRING: "Waiting for sender…",
            STATE_CONNECTING: "Connecting…",
            STATE_CONNECTED: "Sender connected",
            STATE_RECONNECTING: "Reconnecting…",
            STATE_ERROR: f"Error: {detail}",
            STATE_CLOSING: "Closing…",
        }

        dot_color = STATE_DOT_COLORS.get(state, TEXT_SECONDARY)
        status_text = STATE_TEXTS.get(state, state)

        self._status_dot.setStyleSheet(
            f"color: {dot_color}; font-size: 8pt; border: none; background: transparent;"
        )
        self._conn_status_label.setText(status_text)

        active = state in (
            STATE_STARTING_SESSION,
            STATE_WAITING_FOR_PAIRING,
            STATE_CONNECTING,
            STATE_CONNECTED,
            STATE_RECONNECTING,
            STATE_CLOSING,
        )
        self._start_btn.setEnabled(not active)
        self._stop_btn.setEnabled(active)

    def _update_count_label(self) -> None:
        count = len(self._messages)
        self._msg_count_label.setText(f"({count})" if count else "")

    def _update_title_badge(self) -> None:
        if self._unread_count > 0:
            self._title_label.setText(f"📡 Browser Input  🔴{self._unread_count}")
        else:
            self._title_label.setText("📡 Browser Input")

    def _maybe_scroll_to_bottom(self) -> None:
        sb = self._scroll.verticalScrollBar()
        if self._auto_scroll:
            sb.setValue(sb.maximum())

    def _on_scroll_changed(self, value: int) -> None:
        sb = self._scroll.verticalScrollBar()
        # If user scrolled away from bottom → disable auto-scroll
        self._auto_scroll = value >= sb.maximum() - 20

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_copy_pairing_code(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is not None and self._pairing_code:
            clipboard.setText(self._pairing_code)
        self.copy_pairing_code_requested.emit()

    def _on_delete_card(self, message_id: str) -> None:
        card = self._cards.pop(message_id, None)
        if card:
            card.deleteLater()
        self._messages = [m for m in self._messages if m.message_id != message_id]
        self._update_count_label()
        if not self._messages:
            self._empty_label.show()

    @staticmethod
    def _apply_dialog_stealth(box: QMessageBox) -> None:
        if sys.platform != "win32":
            return
        try:
            from ghostmic.core.stealth import apply_stealth, hide_from_taskbar

            hwnd = int(box.winId())
            hide_from_taskbar(hwnd)
            apply_stealth(hwnd)
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("BrowserInputPanel: could not apply stealth to dialog: %s", exc)

    def _on_clear_all(self) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Clear All Messages")
        box.setText("Remove all received messages from this session?")
        box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        box.setWindowModality(Qt.WindowModality.ApplicationModal)
        box.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        if bool(self._config.get("ui", {}).get("stealth_enabled", True)):
            self._apply_dialog_stealth(box)
            QTimer.singleShot(0, lambda: self._apply_dialog_stealth(box))
            QTimer.singleShot(50, lambda: self._apply_dialog_stealth(box))
        box.raise_()
        box.activateWindow()
        reply = box.exec()
        if reply != QMessageBox.StandardButton.Yes:
            return
        for card in list(self._cards.values()):
            card.deleteLater()
        self._cards.clear()
        self._messages.clear()
        self._unread_count = 0
        self._update_count_label()
        self._update_title_badge()
        self._empty_label.show()
        self._auto_scroll = True
