from __future__ import annotations

import ctypes

from ghostmic.core import stealth


def test_hide_from_taskbar_switches_to_toolwindow(monkeypatch) -> None:
    calls: dict[str, int] = {}
    initial_style = stealth.WS_EX_APPWINDOW

    class FakeUser32:
        def GetWindowLongPtrW(self, hwnd, index):
            calls["get_hwnd"] = hwnd
            calls["get_index"] = index
            return initial_style

        def SetWindowLongPtrW(self, hwnd, index, style):
            calls["set_hwnd"] = hwnd
            calls["set_index"] = index
            calls["style"] = style
            return initial_style

        def SetWindowPos(self, hwnd, insert_after, x, y, width, height, flags):
            calls["pos_hwnd"] = hwnd
            calls["pos_flags"] = flags
            return 1

    monkeypatch.setattr(stealth.sys, "platform", "win32")
    monkeypatch.setattr(stealth, "_user32", lambda: FakeUser32())
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 0)
    monkeypatch.setattr(ctypes, "set_last_error", lambda _value: None)

    assert stealth.hide_from_taskbar(12345) is True
    assert calls["style"] & stealth.WS_EX_TOOLWINDOW
    assert not calls["style"] & stealth.WS_EX_APPWINDOW
    assert calls["pos_flags"] & stealth.SWP_FRAMECHANGED
