from __future__ import annotations

import os
import sys
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from ghostmic.services.screenshot_email_service import (
    AuthenticationRequiredError,
    GmailSender,
    ScreenshotEmailWorker,
    capture_desktop_png,
)
from ghostmic.utils.hotkeys import HotkeyManager


def _qt_app():
    app = QApplication.instance()
    return app if app is not None else QApplication([])


def test_capture_desktop_uses_virtual_full_desktop_monitor(tmp_path, monkeypatch):
    fake_screenshot = types.SimpleNamespace(
        rgb=b"\x00\x00\x00",
        size=(1920, 1080),
        width=1920,
        height=1080,
    )
    fake_mss_module = types.ModuleType("mss")
    fake_mss_module.mss = lambda: types.SimpleNamespace(
        monitors=[
            {"left": 0, "top": 0, "width": 1920, "height": 1080},
            {"left": -1920, "top": 0, "width": 1920, "height": 1080},
        ],
        grab=lambda monitor: fake_screenshot,
        __enter__=lambda self: self,
        __exit__=lambda self, *args: None,
    )
    # Replace the context-manager object itself with one having enter/exit methods.
    class FakeMSS:
        monitors = [
            {"left": 0, "top": 0, "width": 1920, "height": 1080},
            {"left": -1920, "top": 0, "width": 1920, "height": 1080},
        ]
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def grab(self, monitor):
            assert monitor is self.monitors[0]
            return fake_screenshot
    fake_mss_module.mss = FakeMSS

    fake_tools = types.ModuleType("mss.tools")
    fake_tools.to_png = lambda rgb, size: b"png-data"
    fake_mss_module.tools = fake_tools
    monkeypatch.setitem(sys.modules, "mss", fake_mss_module)
    monkeypatch.setitem(sys.modules, "mss.tools", fake_tools)

    path = capture_desktop_png(str(tmp_path))
    assert os.path.exists(path)
    assert path.endswith(".png")
    os.remove(path)


def test_capture_failure_removes_partial_temporary_file(monkeypatch, tmp_path):
    class FakeMSS:
        monitors = [{"left": 0, "top": 0, "width": 0, "height": 0}]
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def grab(self, monitor):
            raise RuntimeError("display unavailable")
    fake_mss_module = types.ModuleType("mss")
    fake_mss_module.mss = FakeMSS
    fake_tools = types.ModuleType("mss.tools")
    fake_tools.to_png = lambda rgb, size: b"png"
    fake_mss_module.tools = fake_tools
    monkeypatch.setitem(sys.modules, "mss", fake_mss_module)
    monkeypatch.setitem(sys.modules, "mss.tools", fake_tools)

    try:
        capture_desktop_png(str(tmp_path))
        assert False
    except RuntimeError:
        assert not list(tmp_path.glob("ghostmic_screenshot_*.png"))


def test_gmail_sender_builds_attachment_and_requires_confirmed_message_id(monkeypatch):
    sender = GmailSender("")
    sent_calls = []

    class Messages:
        def send(self, **kwargs):
            sent_calls.append(kwargs)
            return types.SimpleNamespace(execute=lambda: {"id": "gmail-message-id"})

    class Users:
        def messages(self):
            return Messages()

    fake_gmail = types.SimpleNamespace(users=lambda: Users())
    monkeypatch.setattr(sender, "authenticate", lambda: fake_gmail)

    import tempfile
    fd, path = tempfile.mkstemp(suffix=".png")
    os.write(fd, b"png")
    os.close(fd)
    try:
        message_id = sender.send("person@example.com", "Test", "Attached", path)
        assert message_id == "gmail-message-id"
        assert sent_calls
        assert "raw" in sent_calls[0]["body"]
    finally:
        os.remove(path)


def test_gmail_sender_rejects_missing_confirmed_submission(monkeypatch):
    sender = GmailSender("")

    class Messages:
        def send(self, **kwargs):
            return types.SimpleNamespace(execute=lambda: {})

    class Users:
        def messages(self):
            return Messages()

    monkeypatch.setattr(
        sender,
        "authenticate",
        lambda: types.SimpleNamespace(users=lambda: Users()),
    )
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        try:
            sender.send("person@example.com", "Test", "Attached", path)
            assert False
        except RuntimeError:
            pass
    finally:
        os.remove(path)


def test_gmail_sender_requires_oauth_setup_when_credentials_are_missing(monkeypatch):
    sender = GmailSender("")
    fake_keyring = types.SimpleNamespace(
        get_password=lambda *args: None,
        set_password=lambda *args: None,
    )
    monkeypatch.setattr(sender, "_keyring", lambda: fake_keyring)

    try:
        sender.authenticate()
        assert False
    except AuthenticationRequiredError:
        pass


def test_worker_cleans_temporary_screenshot_after_success(monkeypatch, tmp_path):
    png_path = str(tmp_path / "desktop.png")
    with open(png_path, "wb") as fh:
        fh.write(b"png")
    monkeypatch.setattr(
        "ghostmic.services.screenshot_email_service.capture_desktop_png",
        lambda: png_path,
    )
    sent = []
    worker = ScreenshotEmailWorker(
        {"recipient": "person@example.com", "subject": "Test", "message": "Test"},
        sender=lambda recipient, subject, message, path: sent.append(path) or "id-1",
    )
    worker.run()
    assert sent == [png_path]
    assert not os.path.exists(png_path)


def test_duplicate_hotkey_binding_is_not_registered(monkeypatch):
    registered = {}

    class FakeGlobalHotKeys:
        def __init__(self, hotkeys):
            registered.update(hotkeys)
        def start(self):
            pass
        def stop(self):
            pass

    fake_keyboard = types.SimpleNamespace(GlobalHotKeys=FakeGlobalHotKeys)
    fake_pynput = types.ModuleType("pynput")
    fake_pynput.keyboard = fake_keyboard
    monkeypatch.setitem(sys.modules, "pynput", fake_pynput)
    monkeypatch.setitem(
        sys.modules,
        "pynput.keyboard",
        fake_keyboard,
    )

    manager = HotkeyManager(
        {
            "toggle_recording": "ctrl+shift+e",
            "screenshot_email": "ctrl+shift+e",
        },
        {
            "toggle_recording": lambda: None,
            "screenshot_email": lambda: None,
        },
    )
    manager.start()
    assert len(registered) == 1
    manager.stop()


