"""
Global hotkey registration for GhostMic.

Uses the *pynput* library so that hotkeys work even when the application
window is not focused.

All hotkey bindings are read from the config dict and can be reconfigured
at runtime by calling :meth:`HotkeyManager.reload`.
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, Optional

from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)

# Default hotkeys (pynput combination syntax)
DEFAULT_HOTKEYS: Dict[str, str] = {
    "toggle_recording": "<ctrl>+<shift>+g",
    "toggle_window": "<ctrl>+<shift>+h",
    "generate_response": "<ctrl>+g",
    "copy_response": "<ctrl>+<shift>+c",
    "clear_transcript": "<ctrl>+<shift>+x",
}


def _to_pynput_combo(combo: str) -> str:
    """Convert a config key combo ('ctrl+shift+g') to pynput format ('<ctrl>+<shift>+g')."""
    parts = combo.lower().split("+")
    result = []
    for part in parts:
        part = part.strip()
        if part in ("ctrl", "shift", "alt", "meta", "cmd", "win"):
            result.append(f"<{part}>")
        else:
            result.append(part)
    return "+".join(result)


class HotkeyManager:
    """Registers and manages global hotkeys.

    Args:
        config: The 'hotkeys' section of config.json.
        callbacks: Dict mapping hotkey names to callable handlers.
    """

    def __init__(
        self,
        config: Dict[str, str],
        callbacks: Dict[str, Callable[[], None]],
    ) -> None:
        self._config = config
        self._callbacks = callbacks
        self._listener: Optional[object] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start listening for global hotkeys."""
        with self._lock:
            self._start_listener()

    def stop(self) -> None:
        """Stop the hotkey listener."""
        with self._lock:
            if self._listener:
                try:
                    self._listener.stop()  # type: ignore[attr-defined]
                except Exception:  # pylint: disable=broad-except
                    pass
                self._listener = None

    def reload(self, new_config: Dict[str, str]) -> None:
        """Reload hotkeys from an updated config."""
        self._config = new_config
        self.stop()
        self.start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_listener(self) -> None:
        try:
            from pynput import keyboard  # type: ignore[import]
        except ImportError:
            logger.warning(
                "pynput not installed – global hotkeys unavailable. "
                "Run: pip install pynput"
            )
            return

        hotkeys: Dict[str, Callable] = {}

        for name, default_combo in DEFAULT_HOTKEYS.items():
            combo_raw = self._config.get(name, default_combo)
            combo = _to_pynput_combo(combo_raw)
            handler = self._callbacks.get(name)
            if handler:
                hotkeys[combo] = handler

        if not hotkeys:
            return

        try:
            self._listener = keyboard.GlobalHotKeys(hotkeys)
            self._listener.start()  # type: ignore[attr-defined]
            logger.info("HotkeyManager: registered %d hotkeys.", len(hotkeys))
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("HotkeyManager: failed to start listener: %s", exc)
