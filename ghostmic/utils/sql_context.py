"""Backward-compatible SQL profile helpers for GhostMic."""

from __future__ import annotations

from typing import Any, Dict, List

from ghostmic.utils.interview_profile import (
    SQL_INTERVIEW_PROFILE,
    apply_profile_corrections,
    build_profile_summary,
    is_profile_related_text,
)


def build_sql_profile_summary(max_items_per_section: int = 16) -> List[str]:
    """Return a compact SQL glossary safe to inject into prompt context."""
    return build_profile_summary(
        SQL_INTERVIEW_PROFILE,
        max_items_per_section=max_items_per_section,
    )


def is_sql_related_text(text: str) -> bool:
    """Return True when text looks SQL-related or mentions SQL functions."""
    return is_profile_related_text(text, SQL_INTERVIEW_PROFILE)


def apply_sql_corrections(text: str) -> Dict[str, Any]:
    """Normalize likely SQL function misrecognitions in transcript text."""
    return apply_profile_corrections(text, SQL_INTERVIEW_PROFILE)
