"""TeleprompterController: owns reading state and elapsed-time-based scrolling."""
from __future__ import annotations

import time
from typing import Optional

try:
    from PyQt6.QtCore import QElapsedTimer, QObject, QTimer, pyqtSignal
    from PyQt6.QtWidgets import QAbstractScrollArea, QScrollBar
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False
    QObject = object
    pyqtSignal = None

from ghostmic.utils.logger import get_logger
from ghostmic.ui.teleprompter_preferences import TeleprompterPreferences

logger = get_logger(__name__)

# Timer interval in ms. 16 ms ≈ 60 Hz tick rate.
_TICK_MS: int = 16
# Pixels-per-second at speed 1.0x
_PIXELS_PER_SECOND_BASE: float = 30.0


class TeleprompterController(QObject if _QT_AVAILABLE else object):  # type: ignore[misc]
    """Controls reading behaviour for a teleprompter reading view.
    
    - Elapsed-time-based scrolling (not tick-counted) → stable under timer jitter.
    - Manages font size and scroll speed preferences.
    - Works with any QAbstractScrollArea-based widget (QTextBrowser).
    - Independent of AI generation logic.
    
    Usage:
        ctrl = TeleprompterController(preferences)
        ctrl.attach(reading_view)
        ctrl.start()
        ctrl.stop()
        ctrl.reset()
    """

    if _QT_AVAILABLE and pyqtSignal is not None:
        scrolling_state_changed = pyqtSignal(bool)   # True=started, False=stopped/paused
        font_size_changed = pyqtSignal(int)
        speed_changed = pyqtSignal(float)

    def __init__(
        self,
        preferences: Optional[TeleprompterPreferences] = None,
        parent=None,
    ) -> None:
        if _QT_AVAILABLE:
            super().__init__(parent)
        self._prefs = preferences or TeleprompterPreferences()
        self._scrolling = False
        self._view: Optional[QAbstractScrollArea] = None  # type: ignore[type-arg]
        self._elapsed = QElapsedTimer() if _QT_AVAILABLE else None
        self._timer = None
        self._fractional_px: float = 0.0
        self._last_tick_ms: int = 0
        if _QT_AVAILABLE:
            self._timer = QTimer()
            self._timer.setInterval(_TICK_MS)
            self._timer.timeout.connect(self._on_tick)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def attach(self, view: "QAbstractScrollArea") -> None:  # type: ignore[type-arg]
        """Attach to a scroll view. Detaches previous view automatically."""
        self._view = view
        self._fractional_px = 0.0

    def detach(self) -> None:
        """Stop scrolling and detach from the current view."""
        self.stop()
        self._view = None

    @property
    def preferences(self) -> TeleprompterPreferences:
        return self._prefs

    def apply_preferences(self, prefs: TeleprompterPreferences) -> None:
        """Apply new preferences (e.g. from config or another card)."""
        self._prefs = prefs.clamp()
        if _QT_AVAILABLE and pyqtSignal is not None:
            self.font_size_changed.emit(self._prefs.font_size)  # type: ignore[attr-defined]
            self.speed_changed.emit(self._prefs.scroll_speed)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Scrolling control
    # ------------------------------------------------------------------

    def toggle(self) -> None:
        """Toggle play/pause."""
        if self._scrolling:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        """Start or resume scrolling from the current position."""
        if self._view is None:
            return
        scrollbar = self._scrollbar()
        if scrollbar is None:
            return
        # If already at end, restart from beginning
        if scrollbar.value() >= scrollbar.maximum() and scrollbar.maximum() > 0:
            scrollbar.setValue(0)
        self._fractional_px = 0.0
        self._scrolling = True
        if self._elapsed is not None:
            self._elapsed.start()
        self._last_tick_ms = 0
        if self._timer:
            self._timer.start()
        logger.debug("TeleprompterController: scrolling started")
        if _QT_AVAILABLE and pyqtSignal is not None:
            self.scrolling_state_changed.emit(True)  # type: ignore[attr-defined]

    def stop(self) -> None:
        """Pause scrolling, preserving position."""
        self._scrolling = False
        if self._timer:
            self._timer.stop()
        logger.debug("TeleprompterController: scrolling stopped")
        if _QT_AVAILABLE and pyqtSignal is not None:
            self.scrolling_state_changed.emit(False)  # type: ignore[attr-defined]

    def reset(self) -> None:
        """Stop scrolling and return to the beginning."""
        self.stop()
        scrollbar = self._scrollbar()
        if scrollbar:
            scrollbar.setValue(0)

    def go_to_end(self) -> None:
        """Jump to end of document."""
        self.stop()
        scrollbar = self._scrollbar()
        if scrollbar:
            scrollbar.setValue(scrollbar.maximum())

    @property
    def is_scrolling(self) -> bool:
        return self._scrolling

    # ------------------------------------------------------------------
    # Font size
    # ------------------------------------------------------------------

    @property
    def font_size(self) -> int:
        return self._prefs.font_size

    def change_font_size(self, delta: int) -> int:
        """Change font size by delta steps (each step = 2pt). Returns new size."""
        new_size = max(16, min(40, self._prefs.font_size + (delta * 2)))
        self._prefs = TeleprompterPreferences(
            font_size=new_size,
            scroll_speed=self._prefs.scroll_speed,
            line_spacing=self._prefs.line_spacing,
            reading_width=self._prefs.reading_width,
        )
        logger.debug("TeleprompterController: font_size=%d", new_size)
        if _QT_AVAILABLE and pyqtSignal is not None:
            self.font_size_changed.emit(new_size)  # type: ignore[attr-defined]
        return new_size

    # ------------------------------------------------------------------
    # Scroll speed
    # ------------------------------------------------------------------

    @property
    def scroll_speed(self) -> float:
        return self._prefs.scroll_speed

    def change_speed(self, delta: int) -> float:
        """Change scroll speed by delta steps (0.5x per step). Returns new speed."""
        new_speed = max(0.5, min(5.0, self._prefs.scroll_speed + (delta * 0.5)))
        self._prefs = TeleprompterPreferences(
            font_size=self._prefs.font_size,
            scroll_speed=new_speed,
            line_spacing=self._prefs.line_spacing,
            reading_width=self._prefs.reading_width,
        )
        logger.debug("TeleprompterController: scroll_speed=%.1f", new_speed)
        if _QT_AVAILABLE and pyqtSignal is not None:
            self.speed_changed.emit(new_speed)  # type: ignore[attr-defined]
        return new_speed

    # ------------------------------------------------------------------
    # Keyboard shortcuts mapping
    # ------------------------------------------------------------------

    def handle_key(self, key_name: str) -> bool:
        """Handle a named keyboard shortcut. Returns True if consumed."""
        if key_name == "Space":
            self.toggle()
            return True
        if key_name == "Up":
            self.change_speed(1)
            return True
        if key_name == "Down":
            self.change_speed(-1)
            return True
        if key_name in ("Plus", "+", "="):
            self.change_font_size(1)
            return True
        if key_name in ("Minus", "-"):
            self.change_font_size(-1)
            return True
        if key_name == "Home":
            self.reset()
            return True
        if key_name == "End":
            self.go_to_end()
            return True
        return False

    # ------------------------------------------------------------------
    # Reading position preservation
    # ------------------------------------------------------------------

    def save_reading_position(self) -> float:
        """Return current reading position as a fraction 0.0–1.0."""
        scrollbar = self._scrollbar()
        if scrollbar and scrollbar.maximum() > 0:
            return scrollbar.value() / scrollbar.maximum()
        return 0.0

    def restore_reading_position(self, fraction: float) -> None:
        """Restore reading position from a 0.0–1.0 fraction."""
        scrollbar = self._scrollbar()
        if scrollbar and scrollbar.maximum() > 0:
            scrollbar.setValue(int(fraction * scrollbar.maximum()))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _scrollbar(self):
        if self._view is None:
            return None
        return self._view.verticalScrollBar()

    def _on_tick(self) -> None:
        """Timer tick: advance scroll by elapsed-time-based pixel amount."""
        if not self._scrolling or self._view is None:
            return
        
        scrollbar = self._scrollbar()
        if scrollbar is None:
            return

        # Compute elapsed ms since last tick (or since start)
        if self._elapsed is not None:
            elapsed_ms = self._elapsed.elapsed()
        else:
            elapsed_ms = _TICK_MS
        
        delta_ms = elapsed_ms - self._last_tick_ms
        self._last_tick_ms = elapsed_ms
        if delta_ms <= 0:
            delta_ms = _TICK_MS

        # pixels = speed × base_rate × (delta_ms / 1000)
        pixels_exact = self._prefs.scroll_speed * _PIXELS_PER_SECOND_BASE * (delta_ms / 1000.0)
        self._fractional_px += pixels_exact
        pixels = int(self._fractional_px)
        self._fractional_px -= pixels

        if pixels <= 0:
            return

        new_value = min(scrollbar.maximum(), scrollbar.value() + pixels)
        scrollbar.setValue(new_value)

        if new_value >= scrollbar.maximum() and scrollbar.maximum() > 0:
            self._scrolling = False
            self._timer.stop()
            logger.debug("TeleprompterController: reached end of document")
            if _QT_AVAILABLE and pyqtSignal is not None:
                self.scrolling_state_changed.emit(False)  # type: ignore[attr-defined]
