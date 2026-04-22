"""Periodic AI cleanup of session context logs for clearer downstream retrieval."""

from __future__ import annotations

import random
import threading
from typing import Any, Dict

from ghostmic.services.session_context_store import SessionContextStore
from ghostmic.utils.errors import is_rate_limited as _is_rate_limited_shared
from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MIN_INTERVAL_SECONDS = 30.0
DEFAULT_MAX_INTERVAL_SECONDS = 45.0
DEFAULT_MAX_COMPLETION_TOKENS = 320
DEFAULT_TEMPERATURE = 0.1
DEFAULT_FULL_REFRESH_EVERY = 5
DEFAULT_SOURCE_MAX_EVENTS = 80
DEFAULT_SOURCE_MAX_CHARS = 7000

COMPACTION_SYSTEM_PROMPT = """You are a context organizer for a live interview assistant.

Transform noisy event logs into a concise, readable, and structured snapshot for retrieval by another AI model.

Output plain text with these headings exactly:
Current Focus:
Recent Questions:
Important Facts:
Screen Insights:
Recent AI Guidance:
Open Follow-ups:

Rules:
- Remove duplication and filler.
- Keep technical terms, error strings, code symbols, and constraints accurate.
- Prefer the most recent state when details conflict.
- Keep the output short and skimmable (roughly <= 1200 characters).
- No markdown code fences.
"""


class SessionContextCompactor:
    """Background worker that periodically compacts session context with AI."""

    def __init__(
        self,
        store: SessionContextStore,
        ai_config: Dict[str, Any],
    ) -> None:
        self._store = store
        self._ai_config: Dict[str, Any] = dict(ai_config or {})
        self._config_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._source_cursor = 0
        self._sends_since_full = 0
        self._last_skip_reason = ""

    def start(self) -> None:
        """Start periodic compaction loop if not already running."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="session-context-compactor",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Stop periodic compaction loop."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def update_ai_config(self, ai_config: Dict[str, Any]) -> None:
        """Update runtime AI config used by compaction requests."""
        with self._config_lock:
            self._ai_config = dict(ai_config or {})

    def compact_once(self) -> bool:
        """Run one compaction pass; return True when snapshot is updated."""
        config = self._get_config_snapshot()
        if not bool(config.get("context_compaction_enabled", True)):
            self._log_skip_once("context compaction disabled in config")
            return False

        full_refresh_every = max(
            1,
            self._safe_int(
                config.get("context_compaction_full_refresh_every", DEFAULT_FULL_REFRESH_EVERY),
                DEFAULT_FULL_REFRESH_EVERY,
            ),
        )
        source_max_events = max(
            0,
            self._safe_int(
                config.get("context_compaction_source_max_events", DEFAULT_SOURCE_MAX_EVENTS),
                DEFAULT_SOURCE_MAX_EVENTS,
            ),
        )
        source_max_chars = max(
            1200,
            self._safe_int(
                config.get("context_compaction_source_max_chars", DEFAULT_SOURCE_MAX_CHARS),
                DEFAULT_SOURCE_MAX_CHARS,
            ),
        )

        force_full_refresh = self._source_cursor <= 0 or self._sends_since_full >= full_refresh_every
        source = self._store.build_compaction_source(
            max_events=0 if force_full_refresh else source_max_events,
            max_chars=source_max_chars,
            start_index=0 if force_full_refresh else self._source_cursor,
            consume_from_start=not force_full_refresh,
        )

        source_text = str(source.get("text", "")).strip()
        source_signature = str(source.get("signature", "")).strip()
        source_event_count = int(source.get("event_count", 0))
        total_event_count = int(source.get("total_event_count", 0))
        source_end_index = int(source.get("end_index", self._source_cursor))

        if not source_text or not source_signature:
            if not force_full_refresh and total_event_count <= self._source_cursor:
                self._log_skip_once("no newly added source events")
            else:
                self._log_skip_once("compaction source is empty")
            return False

        try:
            organized_text = self._request_compaction(source_text, config)
        except Exception as exc:  # pylint: disable=broad-except
            if _is_rate_limited_shared(exc):
                logger.warning("SessionContextCompactor: rate limited: %s", exc)
            else:
                logger.debug("SessionContextCompactor: compaction skipped: %s", exc)
            return False

        organized_text = str(organized_text).strip()
        if not organized_text:
            self._log_skip_once("empty compaction response")
            return False

        self._store.append_organized_context(
            organized_text,
            source_signature=source_signature,
            source_event_count=source_event_count,
        )

        if force_full_refresh:
            self._source_cursor = total_event_count
            self._sends_since_full = 0
        else:
            self._source_cursor = max(self._source_cursor, source_end_index)
            self._sends_since_full += 1

        self._last_skip_reason = ""
        logger.debug(
            "SessionContextCompactor: organized snapshot updated (%d source events, mode=%s, cursor=%d/%d).",
            source_event_count,
            "full" if force_full_refresh else "incremental",
            self._source_cursor,
            total_event_count,
        )
        return True

    def _run_loop(self) -> None:
        logger.info("SessionContextCompactor: started.")
        while not self._stop_event.is_set():
            interval = self._next_interval_seconds()
            if self._stop_event.wait(interval):
                break
            self.compact_once()
        logger.info("SessionContextCompactor: stopped.")

    def _next_interval_seconds(self) -> float:
        config = self._get_config_snapshot()
        min_interval = self._safe_float(
            config.get("context_compaction_min_interval_seconds", DEFAULT_MIN_INTERVAL_SECONDS),
            DEFAULT_MIN_INTERVAL_SECONDS,
        )
        max_interval = self._safe_float(
            config.get("context_compaction_max_interval_seconds", DEFAULT_MAX_INTERVAL_SECONDS),
            DEFAULT_MAX_INTERVAL_SECONDS,
        )
        if min_interval < 5.0:
            min_interval = 5.0
        if max_interval < min_interval:
            max_interval = min_interval
        return random.uniform(min_interval, max_interval)

    def _get_config_snapshot(self) -> Dict[str, Any]:
        with self._config_lock:
            return dict(self._ai_config)

    @staticmethod
    def _resolve_active_backend(config: Dict[str, Any]) -> str:
        expose_openai = bool(config.get("expose_openai_provider", False))
        requested_backend = str(
            config.get("main_backend") or config.get("backend", "groq")
        ).strip().lower()

        if requested_backend in ("", "groq"):
            return "groq"
        if requested_backend == "gemini":
            return "gemini"
        if requested_backend == "openai":
            return "openai" if expose_openai else "groq"
        return "groq"

    def _request_compaction(self, source_text: str, config: Dict[str, Any]) -> str:
        backend = self._resolve_active_backend(config)

        if backend == "gemini":
            api_key = str(config.get("gemini_api_key", "")).strip()
            if not api_key:
                raise ValueError("Gemini API key not configured for context compaction.")
            model = str(config.get("gemini_model", "gemini-3-flash-preview")).strip() or "gemini-3-flash-preview"

            try:
                from google import genai  # type: ignore[import]
            except ImportError as exc:
                raise RuntimeError("google-genai package not installed") from exc

            client = genai.Client(api_key=api_key)
            return self._request_with_gemini(client, model, source_text, config)

        if backend == "openai":
            api_key = str(config.get("openai_api_key", "")).strip()
            if not api_key:
                raise ValueError("OpenAI API key not configured for context compaction.")
            model = str(config.get("openai_model", "gpt-4o-mini")).strip() or "gpt-4o-mini"
            from openai import OpenAI  # type: ignore[import]

            client = OpenAI(api_key=api_key)
            return self._request_with_client(client, model, source_text, config)

        api_key = str(config.get("groq_api_key", "")).strip()
        if not api_key:
            raise ValueError("Groq API key not configured for context compaction.")
        model = str(config.get("groq_model", "llama-3.3-70b-versatile")).strip() or "llama-3.3-70b-versatile"
        from openai import OpenAI  # type: ignore[import]

        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        return self._request_with_client(client, model, source_text, config)

    @staticmethod
    def _extract_gemini_text(response: Any) -> str:
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

        return ""

    def _request_with_gemini(self, client, model: str, source_text: str, config: Dict[str, Any]) -> str:
        try:
            from google.genai import types  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("google-genai package not installed") from exc

        response = client.models.generate_content(
            model=model,
            contents=(
                "Recent session event log to compact:\n"
                f"{source_text}\n\n"
                "Rewrite this into a clean context snapshot now."
            ),
            config=types.GenerateContentConfig(
                system_instruction=COMPACTION_SYSTEM_PROMPT,
                temperature=self._safe_float(config.get("context_compaction_temperature", DEFAULT_TEMPERATURE), DEFAULT_TEMPERATURE),
                max_output_tokens=int(config.get("context_compaction_max_tokens", DEFAULT_MAX_COMPLETION_TOKENS) or DEFAULT_MAX_COMPLETION_TOKENS),
            ),
        )

        return self._extract_gemini_text(response)

    def _request_with_client(self, client, model: str, source_text: str, config: Dict[str, Any]) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Recent session event log to compact:\n"
                        f"{source_text}\n\n"
                        "Rewrite this into a clean context snapshot now."
                    ),
                },
            ],
            temperature=self._safe_float(config.get("context_compaction_temperature", DEFAULT_TEMPERATURE), DEFAULT_TEMPERATURE),
            max_tokens=int(config.get("context_compaction_max_tokens", DEFAULT_MAX_COMPLETION_TOKENS) or DEFAULT_MAX_COMPLETION_TOKENS),
            timeout=self._safe_float(config.get("context_compaction_timeout", 20.0), 20.0),
            stream=False,
        )

        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""

        first = choices[0]
        message = getattr(first, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        return str(content or "")

    def _log_skip_once(self, reason: str) -> None:
        if reason != self._last_skip_reason:
            logger.debug("SessionContextCompactor: %s", reason)
            self._last_skip_reason = reason

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)
