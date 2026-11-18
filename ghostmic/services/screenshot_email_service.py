"""Desktop screenshot capture and Gmail delivery."""

from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)


class AuthenticationRequiredError(RuntimeError):
    pass


class GmailSender:
    """Send one MIME message through the Gmail API."""

    CREDENTIAL_KEY = "ghostmic.gmail.oauth"

    def __init__(self, credentials_file: str = "") -> None:
        self._credentials_file = credentials_file

    def _keyring(self):
        try:
            import keyring

            return keyring
        except ImportError as exc:
            raise RuntimeError("Windows secure credential storage is unavailable") from exc

    def authenticate(self):
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise RuntimeError("Gmail API dependencies are not installed") from exc

        keyring = self._keyring()
        stored = keyring.get_password("GhostMic", self.CREDENTIAL_KEY)
        credentials = None

        if stored:
            try:
                credentials = Credentials.from_authorized_user_info(json.loads(stored))
            except Exception as exc:
                logger.warning("Stored Gmail authorization could not be loaded")
                raise AuthenticationRequiredError("Gmail authorization is invalid") from exc

        if credentials is not None:
            if credentials.expired or not credentials.valid:
                try:
                    credentials.refresh(Request())
                except Exception as exc:
                    raise AuthenticationRequiredError("Gmail authentication expired") from exc
            refreshed = credentials.to_json()
            keyring.set_password("GhostMic", self.CREDENTIAL_KEY, refreshed)
        else:
            if not self._credentials_file:
                raise AuthenticationRequiredError(
                    "Gmail OAuth credentials file is not configured"
                )
            path = Path(os.path.expandvars(os.path.expanduser(self._credentials_file)))
            if not path.is_file():
                raise AuthenticationRequiredError(
                    "Gmail OAuth credentials file was not found"
                )
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(path),
                    scopes=["https://www.googleapis.com/auth/gmail.send"],
                )
                credentials = flow.run_local_server(
                    port=0,
                    open_browser=True,
                    authorization_prompt_message="",
                )
            except Exception as exc:
                raise AuthenticationRequiredError(
                    "Gmail OAuth authorization is required"
                ) from exc
            keyring.set_password("GhostMic", self.CREDENTIAL_KEY, credentials.to_json())

        return build("gmail", "v1", credentials=credentials, cache_discovery=False)

    def send(self, recipient: str, subject: str, message: str, png_path: str) -> str:
        if not recipient or not png_path:
            raise ValueError("Recipient and screenshot path are required")

        gmail = self.authenticate()
        mime = MIMEMultipart()
        mime["To"] = recipient
        mime["Subject"] = subject or "Desktop screenshot"
        mime.attach(MIMEText(message or "Desktop screenshot attached.", "plain", "utf-8"))

        with open(png_path, "rb") as screenshot:
            attachment = MIMEApplication(screenshot.read(), _subtype="png")
        attachment.add_header(
            "Content-Disposition",
            "attachment",
            filename=Path(png_path).name,
        )
        mime.attach(attachment)

        encoded = base64.urlsafe_b64encode(mime.as_bytes()).decode("ascii")
        result = gmail.users().messages().send(
            userId="me",
            body={"raw": encoded},
        ).execute()
        message_id = str(result.get("id", "")).strip()
        if not message_id:
            raise RuntimeError("Gmail API did not confirm message submission")
        return message_id


def capture_desktop_png(directory: Optional[str] = None) -> str:
    """Capture the virtual full desktop (all monitors) to a temporary PNG."""
    try:
        import mss
        from mss.tools import to_png
    except ImportError as exc:
        raise RuntimeError("mss is not installed") from exc

    temp_dir = directory or tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    fd, png_path = tempfile.mkstemp(prefix="ghostmic_screenshot_", suffix=".png", dir=temp_dir)
    os.close(fd)
    try:
        with mss.mss() as sct:
            monitors = sct.monitors
            if not monitors or len(monitors) < 1:
                raise RuntimeError("No display was detected")
            screenshot = sct.grab(monitors[0])
            if screenshot.width <= 0 or screenshot.height <= 0:
                raise RuntimeError("Desktop capture produced an invalid image")
            with open(png_path, "wb") as output:
                output.write(to_png(screenshot.rgb, screenshot.size))
        return png_path
    except Exception:
        try:
            os.remove(png_path)
        except OSError:
            pass
        raise


class ScreenshotEmailWorker(QThread):
    capture_started = pyqtSignal()
    captured = pyqtSignal()
    sent = pyqtSignal(str)
    failed = pyqtSignal(str)
    authentication_required = pyqtSignal(str)

    def __init__(
        self,
        email_config: dict,
        sender: Optional[Callable[[str, str, str, str], str]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._email_config = dict(email_config)
        self._sender = sender
        self._png_path: Optional[str] = None

    def run(self) -> None:
        self.capture_started.emit()
        try:
            self._png_path = capture_desktop_png()
            self.captured.emit()

            if self._sender is None:
                gmail = GmailSender(
                    str(self._email_config.get("oauth_credentials_file", "")).strip()
                )
                sender = gmail.send
            else:
                sender = self._sender

            message_id = sender(
                str(self._email_config.get("recipient", "")).strip(),
                str(self._email_config.get("subject", "Desktop screenshot")).strip(),
                str(self._email_config.get("message", "")).strip(),
                self._png_path,
            )
            self.sent.emit(message_id)
        except AuthenticationRequiredError as exc:
            self.authentication_required.emit(str(exc))
        except Exception as exc:
            logger.error("Screenshot email operation failed")
            self.failed.emit(str(exc))
        finally:
            if self._png_path:
                try:
                    os.remove(self._png_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.warning("Could not remove temporary screenshot")
