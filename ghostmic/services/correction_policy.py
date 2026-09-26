"""Strict conservative policy for automatic transcript corrections."""

from __future__ import annotations

import re
from typing import Iterable


STRONG_EVIDENCE = frozenset(
    {
        "exact alias",
        "technical vocabulary dictionary",
        "high string similarity",
        "strong phonetic similarity",
        "active topic",
        "resume/profile term",
        "immediately preceding conversation term",
    }
)

# These patterns protect semantic payloads from stylistic or speculative edits.
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|twenty|thirty|forty|fifty|hundred|thousand)\b", re.I)
_DATE_RE = re.compile(r"\b(?:19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b")
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+|\b[a-z0-9.-]+\.(?:com|org|net|io|dev|ai|co)\b", re.I)
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|API|SDK|CLI|ID|URL|HTTP|HTTPS|SQL|S3|EC2|EKS|ECS|RDS|JVM|ETL|ELK|SRE)\b|\b[A-Z][A-Z0-9_]{1,}\b")
_VERSION_RE = re.compile(r"\bv?\d+(?:\.\d+){1,3}(?:[-+][A-Za-z0-9.-]+)?\b", re.I)


def _matches(pattern: re.Pattern[str], value: str) -> set[str]:
    return {match.group(0).lower() for match in pattern.finditer(str(value or ""))}


def protected_semantics_changed(raw_term: str, normalized_term: str) -> bool:
    """Return True when a correction changes protected semantic tokens."""
    for pattern in (_URL_RE, _DATE_RE, _VERSION_RE, _NUMBER_RE, _IDENTIFIER_RE):
        raw_values = _matches(pattern, raw_term)
        normalized_values = _matches(pattern, normalized_term)
        if raw_values != normalized_values:
            return True
    return False


def allow_automatic_correction(
    raw_term: str,
    normalized_term: str,
    *,
    evidence: Iterable[str] = (),
    confidence: str = "LOW",
    score: float = 0.0,
) -> bool:
    """Decide whether an automatic correction is safe to apply.

    Numeric/entity-sensitive changes require an exact, dictionary-backed,
    high-confidence mapping. Otherwise protected values remain unchanged.
    """
    raw = str(raw_term or "").strip()
    normalized = str(normalized_term or "").strip()
    if not raw or not normalized or raw == normalized:
        return False
    evidence_set = {str(item).strip().lower() for item in evidence if str(item).strip()}
    strong = evidence_set & STRONG_EVIDENCE
    if not strong:
        return False
    if raw.lower() == normalized.lower() and raw != normalized:
        return bool(
            evidence_set & {"exact alias", "technical vocabulary dictionary"}
            and str(confidence or "LOW").upper() == "HIGH"
        )
    normalized_confidence = str(confidence or "LOW").upper()
    if normalized_confidence not in {"HIGH", "MEDIUM"}:
        return False
    if float(score or 0.0) < 0.82 and "exact alias" not in evidence_set:
        if not (
            "resume/profile term" in evidence_set
            and normalized_confidence == "HIGH"
            and float(score or 0.0) >= 0.74
        ):
            return False
    protected_changed = protected_semantics_changed(raw, normalized)
    if protected_changed:
        return (
            normalized_confidence == "HIGH"
            and float(score or 0.0) >= 0.95
            and bool(evidence_set & {"exact alias", "technical vocabulary dictionary", "resume/profile term"})
        )
    return True


__all__ = ["allow_automatic_correction", "protected_semantics_changed", "STRONG_EVIDENCE"]
