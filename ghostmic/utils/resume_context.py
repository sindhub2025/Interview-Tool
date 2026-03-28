"""
Resume-aware context helpers for AI prompting.

Provides:
- concise profile summary construction
- conservative transcript term disambiguation using resume-grounded terms
- lightweight resume-related query detection
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

RESUME_QUERY_KEYWORDS = frozenset(
    {
        "resume",
        "background",
        "experience",
        "worked",
        "work history",
        "company",
        "companies",
        "employer",
        "employers",
        "project",
        "projects",
        "education",
        "degree",
        "certification",
        "certifications",
        "career",
        "skill",
        "skills",
        "role",
        "roles",
        "title",
        "titles",
        "achievement",
        "achievements",
    }
)

# Precompile word-boundary regex patterns for resume-related query keywords.
# This avoids substring false-positives (e.g. matching 'resume' inside 'resumption').
# Patterns are matched against a lowercased input string.
_RESUME_QUERY_KEYWORD_PATTERNS = [
    re.compile(r"\b" + re.escape(k) + r"\b") for k in RESUME_QUERY_KEYWORDS
]

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#&./-]*")


# Confidence tuning constants for `_classify_confidence`.
# Boost applied to `medium_threshold` when a company-related cue appears in previous tokens.
_COMPANY_CUE_BOOST = 0.03
# Minimum absolute threshold required for company-cue-driven promotion to 'high'.
_COMPANY_CUE_MIN_THRESHOLD = 0.76
# Boost applied to `medium_threshold` when a skill-related cue appears in previous tokens.
_SKILL_CUE_BOOST = 0.08
# Minimum absolute threshold required for skill-cue-driven promotion to 'high'.
_SKILL_CUE_MIN_THRESHOLD = 0.80


@dataclass(frozen=True)
class _TermEntry:
    canonical: str
    category: str
    match_value: str


def build_resume_context_summary(
    profile: Optional[Dict[str, Any]],
    max_items: int = 6,
) -> List[str]:
    """Return concise resume summary lines safe for prompt injection."""
    if not isinstance(profile, dict):
        return []

    lines: List[str] = []
    identity = profile.get("identity", {}) if isinstance(profile.get("identity"), dict) else {}
    full_name = str(identity.get("full_name", "")).strip()
    location = str(identity.get("location", "")).strip()
    summary = str(profile.get("summary", "")).strip()

    if full_name:
        lines.append(f"Name: {full_name}")
    if location:
        lines.append(f"Location: {location}")
    if summary:
        lines.append(f"Summary: {summary[:220]}")

    for label, key in (
        ("Companies", "companies"),
        ("Job Titles", "job_titles"),
        ("Skills", "skills"),
        ("Projects", "projects"),
        ("Certifications", "certifications"),
        ("Tools/Tech", "tools"),
        ("Technologies", "technologies"),
    ):
        values = _string_list(profile.get(key, []))
        if values:
            lines.append(f"{label}: {', '.join(values[:max_items])}")

    work = profile.get("work_history", [])
    if isinstance(work, list) and work:
        rendered: List[str] = []
        for entry in work[:3]:
            if not isinstance(entry, dict):
                continue
            company = str(entry.get("company", "")).strip()
            title = str(entry.get("title", "")).strip()
            dates = str(entry.get("dates", "")).strip()
            chunk = " - ".join(part for part in (company, title, dates) if part)
            if chunk:
                rendered.append(chunk)
        if rendered:
            lines.append(f"Recent Roles: {'; '.join(rendered)}")

    return lines[:10]


def is_resume_related_text(text: str, profile: Optional[Dict[str, Any]]) -> bool:
    """Heuristic detection for whether the user query is resume/background related."""
    if not text.strip():
        return False

    lowered = text.lower()
    # Use precompiled word-boundary patterns to avoid substring false-positives.
    if any(p.search(lowered) for p in _RESUME_QUERY_KEYWORD_PATTERNS):
        return True

    terms = _collect_terms(profile)
    if not terms:
        return False

    normalized_text = _normalize(lowered)
    for term in terms[:80]:
        candidate = _normalize(term.canonical)
        if len(candidate) < 4:
            continue
        if candidate in normalized_text:
            return True
    return False


def apply_resume_corrections(
    text: str,
    profile: Optional[Dict[str, Any]],
    high_threshold: float = 0.87,
    medium_threshold: float = 0.74,
) -> Dict[str, Any]:
    """Resolve likely misrecognized transcript terms using resume-grounded terms.

    High-confidence corrections are applied directly to the returned text.
    Medium-confidence matches are returned as suggestions without mutation.
    """
    base = {
        "text": text,
        "high_confidence": [],
        "medium_confidence": [],
    }
    if not text.strip() or not isinstance(profile, dict):
        return base

    terms = _collect_terms(profile)
    if not terms:
        return base

    token_matches = list(_TOKEN_RE.finditer(text))
    if not token_matches:
        return base

    tokens = [m.group(0) for m in token_matches]
    used_token_indexes = set()
    high_with_spans: List[Dict[str, Any]] = []
    medium: List[Dict[str, Any]] = []

    for ngram_size in (3, 2, 1):
        for index in range(0, len(tokens) - ngram_size + 1):
            span = range(index, index + ngram_size)
            if any(i in used_token_indexes for i in span):
                continue

            phrase = " ".join(tokens[i] for i in span)
            norm_phrase = _normalize(phrase)
            if len(norm_phrase) < 4:
                continue

            best_term: Optional[_TermEntry] = None
            best_score = 0.0
            for term in terms:
                if _normalize(term.canonical) == norm_phrase:
                    best_term = None
                    best_score = 0.0
                    break
                if _normalize(term.match_value) == norm_phrase:
                    best_term = None
                    best_score = 0.0
                    break

                score = _similarity_score(phrase, term.match_value)
                if score >= medium_threshold and score > best_score:
                    best_term = term
                    best_score = score

            if best_term is None:
                continue

            prev_tokens = [t.lower() for t in tokens[max(0, index - 3):index]]
            confidence = _classify_confidence(
                score=best_score,
                category=best_term.category,
                prev_tokens=prev_tokens,
                high_threshold=high_threshold,
                medium_threshold=medium_threshold,
            )
            correction = {
                "original": phrase,
                "corrected": best_term.canonical,
                "category": best_term.category,
                "score": round(best_score, 3),
                "confidence": confidence,
            }

            if confidence == "high":
                start_char = token_matches[index].start()
                end_char = token_matches[index + ngram_size - 1].end()
                correction["start"] = start_char
                correction["end"] = end_char
                high_with_spans.append(correction)
                for i in span:
                    used_token_indexes.add(i)
            else:
                if len(medium) < 4:
                    medium.append(correction)

    corrected_text = text
    for correction in sorted(high_with_spans, key=lambda item: int(item["start"]), reverse=True):
        start = int(correction["start"])
        end = int(correction["end"])
        corrected = str(correction["corrected"])
        corrected_text = corrected_text[:start] + corrected + corrected_text[end:]

    for correction in high_with_spans:
        correction.pop("start", None)
        correction.pop("end", None)

    return {
        "text": corrected_text,
        "high_confidence": high_with_spans,
        "medium_confidence": medium,
    }


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _collect_terms(profile: Optional[Dict[str, Any]]) -> List[_TermEntry]:
    if not isinstance(profile, dict):
        return []

    terms: List[_TermEntry] = []
    category_by_term: Dict[str, str] = {}

    mappings = (
        ("companies", "companies"),
        ("job_titles", "job_titles"),
        ("skills", "skills"),
        ("projects", "projects"),
        ("certifications", "certifications"),
        ("tools", "tools"),
        ("technologies", "technologies"),
    )

    for category, key in mappings:
        for item in _string_list(profile.get(key, [])):
            if len(item) < 3:
                continue
            terms.append(_TermEntry(canonical=item, category=category, match_value=item))
            category_by_term[item.lower()] = category

    identity = profile.get("identity", {}) if isinstance(profile.get("identity"), dict) else {}
    full_name = str(identity.get("full_name", "")).strip()
    if full_name:
        terms.append(_TermEntry(canonical=full_name, category="identity", match_value=full_name))
        category_by_term[full_name.lower()] = "identity"

    aliases = profile.get("aliases", {}) if isinstance(profile.get("aliases"), dict) else {}
    for canonical, alias_values in aliases.items():
        canonical_clean = str(canonical).strip()
        if not canonical_clean:
            continue
        category = category_by_term.get(canonical_clean.lower(), "keywords")
        for alias in _string_list(alias_values):
            if len(alias) < 4:
                continue
            terms.append(_TermEntry(canonical=canonical_clean, category=category, match_value=alias))

    return _dedupe_terms(terms)


def _dedupe_terms(terms: Sequence[_TermEntry]) -> List[_TermEntry]:
    seen = set()
    result: List[_TermEntry] = []
    for term in terms:
        key = (_normalize(term.canonical), _normalize(term.match_value), term.category)
        if key in seen:
            continue
        seen.add(key)
        result.append(term)
    return result


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _similarity_score(a: str, b: str) -> float:
    a_lower = a.lower().strip()
    b_lower = b.lower().strip()
    a_compact = _normalize(a_lower)
    b_compact = _normalize(b_lower)

    if not a_compact or not b_compact:
        return 0.0

    ratio_text = SequenceMatcher(None, a_lower, b_lower).ratio()
    ratio_compact = SequenceMatcher(None, a_compact, b_compact).ratio()

    score = max(ratio_text, ratio_compact)

    # Prefix similarity is useful for brand/company corrections such as
    # "Micro hard" -> "Microsoft" where tokenization differs.
    if len(a_compact) >= 4 and len(b_compact) >= 4 and a_compact[:4] == b_compact[:4]:
        score = min(1.0, score + 0.08)

    token_overlap = _token_overlap(a_lower, b_lower)
    if token_overlap >= 0.5:
        score = min(1.0, score + 0.05)

    return score


def _token_overlap(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"[a-z0-9]+", a))
    b_tokens = set(re.findall(r"[a-z0-9]+", b))
    if not a_tokens or not b_tokens:
        return 0.0
    union = len(a_tokens | b_tokens)
    return len(a_tokens & b_tokens) / union


def _classify_confidence(
    score: float,
    category: str,
    prev_tokens: Sequence[str],
    high_threshold: float,
    medium_threshold: float,
) -> str:
    if score >= high_threshold:
        return "high"

    prev = " ".join(prev_tokens)
    company_cue = bool(re.search(r"\b(at|for|with|joined|join|worked|company|employer)\b", prev))
    skill_cue = bool(re.search(r"\b(using|use|skill|stack|tool|technology|tech)\b", prev))

    # Company cues can nudge a medium-confidence match into high-confidence.
    if category == "companies" and company_cue and score >= max(medium_threshold + _COMPANY_CUE_BOOST, _COMPANY_CUE_MIN_THRESHOLD):
        return "high"

    # Skill/tool/project categories get a slightly larger boost when a skill cue is present.
    if category in {"skills", "tools", "technologies", "projects", "job_titles", "certifications"}:
        if skill_cue and score >= max(medium_threshold + _SKILL_CUE_BOOST, _SKILL_CUE_MIN_THRESHOLD):
            return "high"

    return "medium"
