"""Silent full-screen screenshot capture and multimodal vision analysis."""

from __future__ import annotations

import base64
from typing import Any, Dict

import requests

from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)

GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GEMINI_VISION_MODEL = "gemini-3-flash-preview"
GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
SCREEN_ANALYSIS_TEMPERATURE = 0.1
SCREEN_ANALYSIS_MAX_COMPLETION_TOKENS = 1024
GEMINI_INLINE_IMAGE_MAX_BYTES = 20 * 1024 * 1024
DEFAULT_SCREEN_PROMPT = (
    "Look at the screenshot for the main question, task, or problem the user is dealing with. "
    "Answer that directly using the on-screen context, and connect related labels, code, error messages, or UI clues when they help. "
    "Keep the response as a very short summary: 1-2 sentences, or up to 3 bullets if that is clearer. "
    "If no clear question is visible, give the briefest useful summary of what matters on screen. "
    "Do not include step-by-step analysis or filler."
)


def capture_full_screen_png_bytes() -> tuple[bytes, Dict[str, int]]:
    """Capture the full virtual desktop as PNG bytes without UI prompts."""
    import mss
    from mss import tools as mss_tools

    with mss.mss() as sct:
        monitor = sct.monitors[0]
        screenshot = sct.grab(monitor)
        png_bytes = mss_tools.to_png(screenshot.rgb, screenshot.size)

    metadata = {
        "left": int(monitor["left"]),
        "top": int(monitor["top"]),
        "width": int(monitor["width"]),
        "height": int(monitor["height"]),
    }
    return png_bytes, metadata


def encode_image_data_url(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """Return a data URL for a base64-encoded image payload."""
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_groq_text(response_json: Dict[str, Any]) -> str:
    """Extract assistant text from a Groq chat-completions response."""
    choices = response_json.get("choices") or []
    if not choices:
        raise RuntimeError("Groq response did not include any choices.")

    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        text = content.strip()
        if text:
            return text
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"text", "output_text"}:
                part = str(item.get("text", "")).strip()
                if part:
                    parts.append(part)
        text = "\n".join(parts).strip()
        if text:
            return text

    raise RuntimeError("Groq response did not contain readable text.")


def extract_gemini_text(response: Any) -> str:
    """Extract assistant text from a Gemini SDK response."""
    text = str(getattr(response, "text", "") or "").strip()
    if text:
        return text

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        chunks: list[str] = []
        for part in parts:
            part_text = str(getattr(part, "text", "") or "").strip()
            if part_text:
                chunks.append(part_text)
        merged = "\n".join(chunks).strip()
        if merged:
            return merged

    raise RuntimeError("Gemini response did not contain readable text.")


def analyze_screenshot_with_groq(
    image_bytes: bytes,
    api_key: str,
    *,
    model: str = GROQ_VISION_MODEL,
    prompt: str = DEFAULT_SCREEN_PROMPT,
    timeout: float = 90.0,
) -> str:
    """Send a screenshot to Groq vision and return the assistant response text."""
    if not api_key:
        raise ValueError("Groq API key is required for screen analysis.")
    if len(image_bytes) > 3_900_000:
        raise RuntimeError(
            "Screenshot is too large for Groq's base64 image limit. "
            "Try reducing display resolution or capturing fewer monitors."
        )

    data_url = encode_image_data_url(image_bytes)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": SCREEN_ANALYSIS_TEMPERATURE,
        "max_completion_tokens": SCREEN_ANALYSIS_MAX_COMPLETION_TOKENS,
        "top_p": 1,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            GROQ_CHAT_COMPLETIONS_URL,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Groq request failed with HTTP {response.status_code}: {response.text.strip()}"
            )
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Groq request failed: {exc}") from exc

    response_json = response.json()
    return extract_groq_text(response_json)


def analyze_screenshot_with_gemini(
    image_bytes: bytes,
    api_key: str,
    *,
    model: str = GEMINI_VISION_MODEL,
    prompt: str = DEFAULT_SCREEN_PROMPT,
    timeout: float = 90.0,
) -> str:
    """Send a screenshot to Gemini vision and return assistant response text."""
    if not api_key:
        raise ValueError("Gemini API key is required for screen analysis.")
    if len(image_bytes) > GEMINI_INLINE_IMAGE_MAX_BYTES:
        raise RuntimeError(
            "Screenshot is too large for Gemini inline image requests (20 MB limit). "
            "Try reducing display resolution or capturing fewer monitors."
        )

    try:
        from google import genai  # type: ignore[import]
        from google.genai import types  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "google-genai package not installed. Run: pip install google-genai"
        ) from exc

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            # Docs recommend placing the prompt after image content.
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png",
                ),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=SCREEN_ANALYSIS_TEMPERATURE,
                max_output_tokens=SCREEN_ANALYSIS_MAX_COMPLETION_TOKENS,
            ),
        )
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"Gemini request failed: {exc}") from exc

    return extract_gemini_text(response)


def resolve_screen_analysis_provider(ai_config: dict) -> str:
    """Return provider used for screen analysis based on active backend."""
    backend = str(
        ai_config.get("main_backend") or ai_config.get("backend") or "groq"
    ).strip().lower()
    if backend == "gemini":
        return "gemini"
    return "groq"


try:
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:  # pragma: no cover - fallback for non-Qt environments
    QThread = object  # type: ignore[misc,assignment]
    pyqtSignal = None  # type: ignore[assignment]


class ScreenAnalysisWorker(QThread):  # type: ignore[misc]
    """Qt-only background worker for screen capture and provider analysis.

    This class requires PyQt6 at import time so ``pyqtSignal`` is not ``None``;
    only then do the ``analysis_ready`` and ``analysis_error`` signals exist.
    In non-Qt environments the module falls back to a plain ``object`` base and
    the signal attributes are not defined, so calling :meth:`run` is not safe and
    will fail once the signal emits are reached, typically with ``AttributeError``.

    Expected environment: a Qt-enabled runtime with PyQt6 installed.
    Guard usage by checking ``pyqtSignal`` or importing this class only when
    PyQt6 is available.
    """

    if pyqtSignal is not None:
        analysis_ready = pyqtSignal(str)
        analysis_error = pyqtSignal(str)

    def __init__(self, ai_config: dict, parent=None) -> None:
        super().__init__(parent)
        self._ai_config = dict(ai_config or {})

    def run(self) -> None:
        try:
            image_bytes, metadata = capture_full_screen_png_bytes()
            logger.info(
                "Captured full-screen screenshot for analysis: %dx%d, %d bytes",
                metadata["width"],
                metadata["height"],
                len(image_bytes),
            )
            timeout = float(self._ai_config.get("screen_analysis_timeout", 90.0))
            provider = resolve_screen_analysis_provider(self._ai_config)

            if provider == "gemini":
                result = analyze_screenshot_with_gemini(
                    image_bytes,
                    str(self._ai_config.get("gemini_api_key", "")).strip(),
                    model=str(
                        self._ai_config.get("gemini_vision_model", GEMINI_VISION_MODEL)
                    ).strip() or GEMINI_VISION_MODEL,
                    prompt=DEFAULT_SCREEN_PROMPT,
                    timeout=timeout,
                )
            else:
                result = analyze_screenshot_with_groq(
                    image_bytes,
                    str(self._ai_config.get("groq_api_key", "")).strip(),
                    model=str(
                        self._ai_config.get("groq_vision_model", GROQ_VISION_MODEL)
                    ).strip() or GROQ_VISION_MODEL,
                    prompt=DEFAULT_SCREEN_PROMPT,
                    timeout=timeout,
                )

            self.analysis_ready.emit(result)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Screen analysis failed: %s", exc)
            self.analysis_error.emit(str(exc))