"""
System tray icon and context menu for GhostMic.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QAction, QColor, QIcon, QPixmap
from PyQt6.QtWidgets import QMenu, QSystemTrayIcon

from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)


def _tray_icon() -> QIcon:
    """Load tray icon from assets or fall back to a blank coloured icon."""
    asset_dirs = []
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        asset_dirs.extend(
            [
                os.path.join(sys._MEIPASS, "ghostmic", "assets"),
                os.path.join(sys._MEIPASS, "assets"),
            ]
        )
    asset_dirs.append(os.path.join(os.path.dirname(__file__), "..", "assets"))

    for assets_dir in asset_dirs:
        for name in ("tray_icon.png", "icon.ico"):
            path = os.path.join(assets_dir, name)
            if os.path.exists(path):
                return QIcon(path)
    # Generate a small coloured pixmap as fallback
    pix = QPixmap(16, 16)
    pix.fill(QColor("#58a6ff"))
    return QIcon(pix)


class SystemTrayIcon(QSystemTrayIcon):
    """System tray icon with context menu.

    Signals:
        show_hide_requested()
        dock_toggle_requested()
        start_stop_requested()
        mode_changed(str)
        quit_requested()
    """

    show_hide_requested = pyqtSignal()
    dock_toggle_requested = pyqtSignal()
    start_stop_requested = pyqtSignal()
    mode_changed = pyqtSignal(str)
    quit_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(_tray_icon(), parent)
        self._recording = False
        self._build_menu()
        self.activated.connect(self._on_activated)
        self.show()

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        menu = QMenu()

        self._toggle_action = QAction("⏺  Start Recording", menu)
        self._toggle_action.triggered.connect(self.start_stop_requested)
        menu.addAction(self._toggle_action)

        self._show_action = QAction("👁  Show Window", menu)
        self._show_action.triggered.connect(self.show_hide_requested)
        menu.addAction(self._show_action)

        self._dock_action = QAction("📌  Dock Window", menu)
        self._dock_action.triggered.connect(self.dock_toggle_requested)
        menu.addAction(self._dock_action)

        menu.addSeparator()

        # Quick mode switch
        mode_menu = QMenu("Mode", menu)
        for mode in ["Interview", "Meeting Notes", "Custom"]:
            act = QAction(mode, mode_menu)
            act.triggered.connect(lambda _checked, m=mode: self.mode_changed.emit(m))
            mode_menu.addAction(act)
        menu.addMenu(mode_menu)

        menu.addSeparator()

        quit_action = QAction("✕  Quit GhostMic", menu)
        quit_action.triggered.connect(self.quit_requested)
        menu.addAction(quit_action)

        self.setContextMenu(menu)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_recording(self, recording: bool) -> None:
        self._recording = recording
        if recording:
            self._toggle_action.setText("⏹  Stop Recording")
        else:
            self._toggle_action.setText("⏺  Start Recording")

    def set_window_visible(self, visible: bool) -> None:
        self._show_action.setText(
            "🙈  Hide Window" if visible else "👁  Show Window"
        )

    def set_docked(self, docked: bool) -> None:
        self._dock_action.setText("↩  Undock Window" if docked else "📌  Dock Window")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show_hide_requested.emit()
