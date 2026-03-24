"""Dark-theme QSS stylesheet for GhostMic."""

from __future__ import annotations

# Colour palette
BG_DEEP = "#0d1117"
BG_MID = "#161b22"
BG_CARD = "#21262d"
BORDER = "#333366"
ACCENT_BLUE = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_RED = "#f85149"
TEXT_PRIMARY = "#e6edf3"
TEXT_SECONDARY = "#8b949e"
FONT = "Segoe UI"
FONT_SIZE = 11

MAIN_STYLE = f"""
/* ─── Global ──────────────────────────────────────────────────────── */
QWidget {{
    background-color: {BG_DEEP};
    color: {TEXT_PRIMARY};
    font-family: "{FONT}";
    font-size: {FONT_SIZE}pt;
}}

QMainWindow, QDialog {{
    background-color: {BG_DEEP};
}}

/* ─── Scroll bars ─────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {BG_MID};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {ACCENT_BLUE};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {BG_MID};
    height: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER};
    border-radius: 4px;
    min-width: 20px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {ACCENT_BLUE};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ─── Buttons ─────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 14px;
}}
QPushButton:hover {{
    background-color: #2d333b;
    border-color: {ACCENT_BLUE};
}}
QPushButton:pressed {{
    background-color: #1c2128;
}}
QPushButton:disabled {{
    color: {TEXT_SECONDARY};
    border-color: #30363d;
}}
QPushButton#record_btn {{
    background-color: {ACCENT_RED};
    border-color: {ACCENT_RED};
    font-weight: bold;
    border-radius: 20px;
    min-width: 40px;
    min-height: 40px;
}}
QPushButton#record_btn:checked {{
    background-color: #6e0000;
    border-color: {ACCENT_RED};
}}
QPushButton#record_btn:hover {{
    background-color: #da3633;
}}

/* ─── LineEdit / TextEdit ─────────────────────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {BG_MID};
    color: {TEXT_PRIMARY};
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 4px 8px;
    selection-background-color: {ACCENT_BLUE};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {ACCENT_BLUE};
}}

/* ─── ComboBox ────────────────────────────────────────────────────── */
QComboBox {{
    background-color: {BG_MID};
    color: {TEXT_PRIMARY};
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 3px 8px;
    min-height: 26px;
}}
QComboBox:hover {{
    border-color: {ACCENT_BLUE};
}}
QComboBox QAbstractItemView {{
    background-color: {BG_MID};
    color: {TEXT_PRIMARY};
    selection-background-color: {ACCENT_BLUE};
    border: 1px solid {BORDER};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

/* ─── Tabs ────────────────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    background: {BG_MID};
}}
QTabBar::tab {{
    background: {BG_CARD};
    color: {TEXT_SECONDARY};
    border: 1px solid {BORDER};
    border-bottom: none;
    padding: 6px 16px;
    border-radius: 4px 4px 0 0;
}}
QTabBar::tab:selected {{
    background: {BG_MID};
    color: {TEXT_PRIMARY};
    border-bottom: 2px solid {ACCENT_BLUE};
}}
QTabBar::tab:hover {{
    color: {TEXT_PRIMARY};
}}

/* ─── Slider ──────────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 4px;
    background: #30363d;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT_BLUE};
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT_BLUE};
    border-radius: 2px;
}}

/* ─── Checkboxes ──────────────────────────────────────────────────── */
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid #30363d;
    border-radius: 3px;
    background: {BG_MID};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT_BLUE};
    border-color: {ACCENT_BLUE};
}}

/* ─── Group box ───────────────────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    color: {TEXT_SECONDARY};
}}

/* ─── Tool tip ────────────────────────────────────────────────────── */
QToolTip {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px;
}}

/* ─── Status bar ──────────────────────────────────────────────────── */
QLabel#status_label {{
    color: {TEXT_SECONDARY};
    font-size: 9pt;
}}
"""
