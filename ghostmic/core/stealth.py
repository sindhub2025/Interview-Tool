"""
Stealth module: SetWindowDisplayAffinity logic to exclude the window
from screen-capture methods (Chrome getDisplayMedia, Zoom, Teams, OBS, etc.).
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import sys
from typing import Optional

from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)

# Win32 constants
WDA_NONE: int = 0x00000000
WDA_MONITOR: int = 0x00000001          # Shows as black rectangle in capture
WDA_EXCLUDEFROMCAPTURE: int = 0x00000011  # Completely invisible (Win10 2004+)


def _user32() -> ctypes.WinDLL:
    """Return the user32 DLL handle (Windows only)."""
    return ctypes.windll.user32  # type: ignore[attr-defined]


def apply_stealth(hwnd: int) -> bool:
    """Apply WDA_EXCLUDEFROMCAPTURE to *hwnd*.

    Falls back to WDA_MONITOR on older Windows builds.

    Args:
        hwnd: Native window handle.

    Returns:
        True if any display-affinity was applied successfully, False otherwise.
    """
    if sys.platform != "win32":
        logger.warning("apply_stealth: not on Windows – skipping.")
        return False

    if not hwnd:
        logger.error("apply_stealth: invalid HWND (0).")
        return False

    user32 = _user32()

    # Try the best option first
    ok = user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
    if ok:
        logger.info(
            "apply_stealth: WDA_EXCLUDEFROMCAPTURE applied to HWND=%d", hwnd
        )
        return True

    err = ctypes.get_last_error()
    logger.warning(
        "apply_stealth: WDA_EXCLUDEFROMCAPTURE failed for HWND=%d "
        "(win32 error %d) – falling back to WDA_MONITOR.",
        hwnd,
        err,
    )

    # Fallback: WDA_MONITOR (black rectangle, not invisible but not captured)
    ok = user32.SetWindowDisplayAffinity(hwnd, WDA_MONITOR)
    if ok:
        logger.info(
            "apply_stealth: WDA_MONITOR (fallback) applied to HWND=%d", hwnd
        )
        return True

    err = ctypes.get_last_error()
    logger.error(
        "apply_stealth: both WDA methods failed for HWND=%d "
        "(win32 error %d).",
        hwnd,
        err,
    )
    return False


def remove_stealth(hwnd: int) -> bool:
    """Remove display affinity from *hwnd* (restore normal capture).

    Args:
        hwnd: Native window handle.

    Returns:
        True if affinity was cleared successfully.
    """
    if sys.platform != "win32":
        return False

    if not hwnd:
        return False

    user32 = _user32()
    ok = user32.SetWindowDisplayAffinity(hwnd, WDA_NONE)
    if ok:
        logger.info("remove_stealth: WDA_NONE applied to HWND=%d", hwnd)
    else:
        err = ctypes.get_last_error()
        logger.error(
            "remove_stealth: failed for HWND=%d (win32 error %d).", hwnd, err
        )
    return bool(ok)


def verify_stealth(hwnd: int) -> bool:
    """Verify that *hwnd* is excluded from capture using mss.

    Takes a screenshot of the window's bounding rectangle and checks
    whether all pixels are black (indicating exclusion).

    Args:
        hwnd: Native window handle.

    Returns:
        True if the window appears hidden in a screen-capture.
        Returns False if verification fails or mss is unavailable.
    """
    if sys.platform != "win32":
        return False

    try:
        import mss  # type: ignore[import]
        import numpy as np

        user32 = _user32()

        # Get window rect
        rect = ctypes.wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            logger.warning("verify_stealth: GetWindowRect failed for HWND=%d", hwnd)
            return False

        left = rect.left
        top = rect.top
        width = rect.right - rect.left
        height = rect.bottom - rect.top

        if width <= 0 or height <= 0:
            logger.warning("verify_stealth: window has zero size for HWND=%d", hwnd)
            return False

        monitor = {"top": top, "left": left, "width": width, "height": height}

        with mss.mss() as sct:
            screenshot = sct.grab(monitor)

        img = np.array(screenshot)  # BGRA
        # Check if all pixels are black (RGB all zeros)
        rgb = img[:, :, :3]
        is_black = bool(np.all(rgb == 0))
        if is_black:
            logger.info(
                "verify_stealth: window HWND=%d is fully black in capture "
                "(stealth confirmed).",
                hwnd,
            )
        else:
            logger.warning(
                "verify_stealth: window HWND=%d is VISIBLE in capture "
                "(stealth may not be active).",
                hwnd,
            )
        return is_black

    except ImportError:
        logger.debug("verify_stealth: mss not installed, skipping verification.")
        return False
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("verify_stealth: unexpected error: %s", exc)
        return False


def get_current_affinity(hwnd: int) -> Optional[int]:
    """Return the current display-affinity value for *hwnd*.

    Args:
        hwnd: Native window handle.

    Returns:
        The affinity value, or None if the call failed.
    """
    if sys.platform != "win32":
        return None

    if not hwnd:
        return None

    user32 = _user32()
    affinity = ctypes.c_uint(0)
    ok = user32.GetWindowDisplayAffinity(hwnd, ctypes.byref(affinity))
    if ok:
        return affinity.value
    err = ctypes.get_last_error()
    logger.error(
        "get_current_affinity: failed for HWND=%d (win32 error %d).", hwnd, err
    )
    return None
