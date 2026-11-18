"""ScreenContextService: manages captured screen understanding across questions."""
from __future__ import annotations

import time
from typing import Optional

from ghostmic.utils.logger import get_logger
from ghostmic.domain.screen_understanding import ScreenUnderstanding

logger = get_logger(__name__)

# After this many seconds without a new question, context expires.
_DEFAULT_CONTEXT_TTL_SECONDS: float = 600.0  # 10 minutes
# Jaccard similarity below this → topic shift → expire context.
_TOPIC_SHIFT_THRESHOLD: float = 0.25


class ScreenContextService:
    """Manages the lifecycle of captured screen understanding.

    Responsibilities:
    - Stores latest ScreenUnderstanding
    - Tracks capture timestamp and anchor question
    - Determines context relevance to follow-up questions
    - Handles topic shifts and context expiry
    - Provides bounded context string for AI injection
    - Does NOT store raw screenshot bytes
    """

    def __init__(self, ttl_seconds: float = _DEFAULT_CONTEXT_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._understanding: Optional[ScreenUnderstanding] = None
        self._anchor_question: str = ""
        self._captured_at: float = 0.0
        self._last_accessed: float = 0.0

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def update(self, understanding: ScreenUnderstanding, anchor_question: str = "") -> None:
        """Store a new ScreenUnderstanding, replacing any previous one."""
        self._understanding = understanding
        self._anchor_question = anchor_question.strip()
        self._captured_at = time.time()
        self._last_accessed = self._captured_at
        logger.info(
            "ScreenContextService: updated with capture_id=%s, summary_len=%d",
            understanding.capture_id, len(understanding.summary),
        )

    def clear(self) -> None:
        """Explicitly clear the active context."""
        self._understanding = None
        self._anchor_question = ""
        self._captured_at = 0.0
        self._last_accessed = 0.0
        logger.info("ScreenContextService: context cleared.")

    def set_anchor_question(self, question: str) -> None:
        """Set or update the anchor question for topic tracking."""
        self._anchor_question = question.strip()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @property
    def has_context(self) -> bool:
        return self._understanding is not None and not self._is_expired()

    @property
    def understanding(self) -> Optional[ScreenUnderstanding]:
        if self._is_expired():
            return None
        return self._understanding

    def get_context_for_question(
        self,
        question: str,
        *,
        force_include: bool = False,
    ) -> str:
        """Return context string to inject into an AI request, or empty string.

        Args:
            question: Latest speaker question text.
            force_include: If True, always return context if available.

        Returns:
            Formatted context string, or "" if not relevant/expired.
        """
        if self._understanding is None:
            return ""
        if self._is_expired():
            logger.info("ScreenContextService: context expired, not injecting.")
            return ""

        question = question.strip()

        # Update last-accessed
        self._last_accessed = time.time()

        if force_include:
            self._anchor_question = question
            return self._format_context()

        # First question after capture — set anchor and return
        if not self._anchor_question:
            if question:
                self._anchor_question = question
            return self._format_context()

        if not question:
            return self._format_context()

        # Topic shift check
        if self._is_topic_shift(self._anchor_question, question):
            logger.info(
                "ScreenContextService: topic shift detected; clearing context. "
                "anchor=%r new=%r",
                self._anchor_question[:60], question[:60],
            )
            self.clear()
            return ""

        # Still on-topic
        self._anchor_question = question
        return self._format_context()

    def get_summary_text(self) -> str:
        """Return the raw summary text of the current understanding, or ''."""
        if self._understanding is None or self._is_expired():
            return ""
        return self._understanding.raw_text or self._understanding.summary

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_expired(self) -> bool:
        if self._captured_at == 0.0:
            return True
        return (time.time() - self._captured_at) > self._ttl

    def _is_topic_shift(self, anchor: str, question: str) -> bool:
        """Return True if question is a new topic (Jaccard similarity < threshold)."""
        try:
            anchor_words = set(w.lower() for w in anchor.split() if len(w) > 3)
            question_words = set(w.lower() for w in question.split() if len(w) > 3)
            if not anchor_words or not question_words:
                return False
            intersection = len(anchor_words & question_words)
            union = len(anchor_words | question_words)
            jaccard = intersection / union if union else 0.0
            return jaccard < _TOPIC_SHIFT_THRESHOLD
        except Exception:
            return False

    def _format_context(self) -> str:
        """Return a bounded context string for AI injection."""
        u = self._understanding
        if not u:
            return ""

        parts = []

        # Use raw_text if compact enough, else summarize
        if u.raw_text and len(u.raw_text) <= 8000:
            parts.append(u.raw_text)
        elif u.summary:
            parts.append(u.summary)
            if u.tables:
                for tbl in u.tables[:3]:
                    parts.append(f"\nTable: {tbl.name}")
                    if tbl.columns:
                        parts.append(f"  Columns: {', '.join(tbl.columns)}")
            if u.code_blocks:
                for cb in u.code_blocks[:2]:
                    parts.append(f"\n```{cb.language}\n{cb.code[:500]}\n```")

        result = "\n".join(parts).strip()
        return result[:8000]  # hard cap
