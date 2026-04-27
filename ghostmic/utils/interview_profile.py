"""Interview profile helpers for domain-specific context and corrections."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_PROFILE_ID = "sql"


@dataclass(frozen=True)
class ProfileTerm:
    section: str
    canonical: str
    definition: str
    aliases: Tuple[str, ...]


_SQL_PROFILE_SECTIONS: Tuple[Tuple[str, Tuple[ProfileTerm, ...]], ...] = (
    (
        "Aggregate Functions",
        (
            ProfileTerm("Aggregate Functions", "COUNT()", "number of rows", ("count",)),
            ProfileTerm("Aggregate Functions", "SUM()", "total of values", ("sum",)),
            ProfileTerm("Aggregate Functions", "AVG()", "average value", ("avg", "average")),
            ProfileTerm("Aggregate Functions", "MIN()", "smallest value", ("min", "minimum")),
            ProfileTerm("Aggregate Functions", "MAX()", "largest value", ("max", "maximum")),
        ),
    ),
    (
        "String Functions",
        (
            ProfileTerm("String Functions", "UPPER()", "change case to uppercase", ("upper",)),
            ProfileTerm("String Functions", "LOWER()", "change case to lowercase", ("lower",)),
            ProfileTerm("String Functions", "LENGTH()", "string length", ("length",)),
            ProfileTerm("String Functions", "LEN()", "string length", ("len",)),
            ProfileTerm("String Functions", "SUBSTRING()", "extract part", ("substring", "sub string")),
            ProfileTerm("String Functions", "SUBSTR()", "extract part", ("substr",)),
            ProfileTerm("String Functions", "TRIM()", "remove spaces", ("trim",)),
            ProfileTerm("String Functions", "LTRIM()", "remove spaces", ("ltrim",)),
            ProfileTerm("String Functions", "RTRIM()", "remove spaces", ("rtrim",)),
            ProfileTerm("String Functions", "CONCAT()", "join strings", ("concat", "concatenate")),
            ProfileTerm("String Functions", "REPLACE()", "replace text", ("replace",)),
        ),
    ),
    (
        "Numeric Functions",
        (
            ProfileTerm("Numeric Functions", "ROUND()", "round number", ("round",)),
            ProfileTerm("Numeric Functions", "CEILING()", "round up", ("ceiling", "ceil")),
            ProfileTerm("Numeric Functions", "FLOOR()", "round down", ("floor",)),
            ProfileTerm("Numeric Functions", "ABS()", "absolute value", ("abs",)),
            ProfileTerm("Numeric Functions", "POWER()", "exponent", ("power",)),
            ProfileTerm("Numeric Functions", "MOD()", "remainder", ("mod",)),
        ),
    ),
    (
        "Date & Time Functions",
        (
            ProfileTerm("Date & Time Functions", "NOW()", "current date-time", ("now",)),
            ProfileTerm(
                "Date & Time Functions",
                "CURRENT_TIMESTAMP",
                "current date-time",
                ("current timestamp", "current_timestamp"),
            ),
            ProfileTerm("Date & Time Functions", "CURDATE()", "current date", ("curdate",)),
            ProfileTerm("Date & Time Functions", "DATE()", "extract date", ("date",)),
            ProfileTerm("Date & Time Functions", "YEAR()", "year part", ("year",)),
            ProfileTerm("Date & Time Functions", "MONTH()", "month part", ("month",)),
            ProfileTerm("Date & Time Functions", "DAY()", "day part", ("day",)),
            ProfileTerm(
                "Date & Time Functions",
                "DATEDIFF()",
                "difference between dates",
                ("datediff", "date diff", "date difference"),
            ),
            ProfileTerm("Date & Time Functions", "DATEADD()", "add time", ("dateadd", "date add")),
            ProfileTerm("Date & Time Functions", "INTERVAL", "add time", ("interval",)),
        ),
    ),
    (
        "Conditional Functions",
        (
            ProfileTerm("Conditional Functions", "CASE", "if-else logic", ("case",)),
            ProfileTerm(
                "Conditional Functions",
                "COALESCE()",
                "first non-null value",
                ("coalesce", "call as"),
            ),
            ProfileTerm("Conditional Functions", "NULLIF()", "returns NULL if equal", ("nullif",)),
            ProfileTerm("Conditional Functions", "IF()", "if-else logic", ("if",)),
        ),
    ),
    (
        "Window (Analytical) Functions",
        (
            ProfileTerm(
                "Window (Analytical) Functions",
                "ROW_NUMBER()",
                "assigns row sequence",
                ("row number", "row_number"),
            ),
            ProfileTerm("Window (Analytical) Functions", "RANK()", "rank with gaps", ("rank",)),
            ProfileTerm(
                "Window (Analytical) Functions",
                "DENSE_RANK()",
                "rank without gaps",
                ("dense rank", "dense_rank"),
            ),
            ProfileTerm("Window (Analytical) Functions", "LEAD()", "next row value", ("lead",)),
            ProfileTerm("Window (Analytical) Functions", "LAG()", "previous row value", ("lag",)),
            ProfileTerm("Window (Analytical) Functions", "OVER()", "defines window", ("over",)),
        ),
    ),
)

_SQL_DETECTION_TERMS = tuple(
    {
        "sql",
        "query",
        "table",
        "tables",
        "column",
        "columns",
        "database",
        "select",
        "join",
        "where",
        "group by",
        "order by",
        "partition by",
        "window",
        "aggregate",
        "function",
        "functions",
        "procedure",
        "schema",
        "row number",
        "dense rank",
        "current timestamp",
        "current date",
        "count",
        "sum",
        "avg",
        "min",
        "max",
        "upper",
        "lower",
        "length",
        "len",
        "substring",
        "substr",
        "trim",
        "ltrim",
        "rtrim",
        "concat",
        "replace",
        "round",
        "ceiling",
        "ceil",
        "floor",
        "abs",
        "power",
        "mod",
        "now",
        "curdate",
        "dateadd",
        "datediff",
        "coalesce",
        "nullif",
        "case",
        "if",
        "lead",
        "lag",
    }
)


def _sql_sections_as_config() -> List[dict]:
    sections: List[dict] = []
    for section_name, entries in _SQL_PROFILE_SECTIONS:
        sections.append(
            {
                "name": section_name,
                "items": [
                    {
                        "canonical": entry.canonical,
                        "definition": entry.definition,
                        "aliases": list(entry.aliases),
                    }
                    for entry in entries
                ],
            }
        )
    return sections


SQL_INTERVIEW_PROFILE: Dict[str, Any] = {
    "id": DEFAULT_PROFILE_ID,
    "name": "SQL",
    "description": "SQL, database, data engineering, and analytics interview context.",
    "context": (
        "Use this profile for SQL, database, data engineering, ETL, analytics, "
        "and warehouse interview questions."
    ),
    "keywords": list(_SQL_DETECTION_TERMS),
    "sections": _sql_sections_as_config(),
    "normalization_guidance": [
        "Normalize likely speech-to-text mistakes to canonical SQL terms when the profile is active.",
        "Preserve DBMS and RDBMS as database acronyms; RDBMS means relational, not real-time.",
    ],
    "response_guidance": [
        "Use concise, practical SQL explanations when the speaker asks SQL or database questions.",
        "When a SQL example helps, provide a compact query in a fenced code block tagged sql.",
    ],
    "screen_analysis_focus": (
        "When tables, schemas, SQL queries, ER diagrams, database errors, or tabular data are visible, "
        "extract them precisely so SQL-related follow-up questions can be answered from the screen context."
    ),
}


def _normalize_whitespace(text: object) -> str:
    return " ".join(str(text or "").split()).strip()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return slug or "profile"


def _unique_profile_id(name: str, existing_ids: Iterable[str]) -> str:
    existing = {str(item or "").strip().lower() for item in existing_ids}
    base = _slugify(name)
    candidate = base
    suffix = 2
    while candidate.lower() in existing:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def create_blank_interview_profile(name: str, existing_profiles: Iterable[dict] = ()) -> Dict[str, Any]:
    """Return a new editable interview profile with a unique id."""
    clean_name = _normalize_whitespace(name) or "New Profile"
    existing_ids = [
        str(profile.get("id", "")).strip()
        for profile in existing_profiles
        if isinstance(profile, dict)
    ]
    return {
        "id": _unique_profile_id(clean_name, existing_ids),
        "name": clean_name,
        "description": "",
        "context": "",
        "keywords": [],
        "sections": [],
        "normalization_guidance": [],
        "response_guidance": [],
        "screen_analysis_focus": "",
    }


def default_interview_profiles() -> List[dict]:
    """Return built-in interview profiles."""
    return [copy.deepcopy(SQL_INTERVIEW_PROFILE)]


def _normalize_string_list(raw_value: object) -> List[str]:
    if isinstance(raw_value, str):
        raw_items = re.split(r"[,;\n]+", raw_value)
    elif isinstance(raw_value, (list, tuple)):
        raw_items = raw_value
    else:
        raw_items = []

    cleaned: List[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        item = _normalize_whitespace(raw)
        key = item.lower()
        if item and key not in seen:
            cleaned.append(item)
            seen.add(key)
    return cleaned


def _normalize_sections(raw_sections: object) -> List[dict]:
    if not isinstance(raw_sections, list):
        return []

    sections: List[dict] = []
    for raw_section in raw_sections:
        if not isinstance(raw_section, dict):
            continue
        section_name = _normalize_whitespace(raw_section.get("name", "")) or "Terms"
        raw_items = raw_section.get("items", [])
        if not isinstance(raw_items, list):
            raw_items = []

        items: List[dict] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            canonical = _normalize_whitespace(raw_item.get("canonical", ""))
            if not canonical:
                continue
            items.append(
                {
                    "canonical": canonical,
                    "definition": _normalize_whitespace(raw_item.get("definition", "")),
                    "aliases": _normalize_string_list(raw_item.get("aliases", [])),
                }
            )
        sections.append({"name": section_name, "items": items})
    return sections


def normalize_interview_profile(raw_profile: object) -> Optional[dict]:
    """Return a sanitized profile dict, or None when the input is unusable."""
    if not isinstance(raw_profile, dict):
        return None

    name = _normalize_whitespace(raw_profile.get("name", ""))
    profile_id = _normalize_whitespace(raw_profile.get("id", ""))
    if not name and not profile_id:
        return None
    if not name:
        name = profile_id.replace("-", " ").title()
    if not profile_id:
        profile_id = _slugify(name)

    profile = {
        "id": _slugify(profile_id),
        "name": name,
        "description": _normalize_whitespace(raw_profile.get("description", "")),
        "context": str(raw_profile.get("context", "") or "").strip(),
        "keywords": _normalize_string_list(raw_profile.get("keywords", [])),
        "sections": _normalize_sections(raw_profile.get("sections", [])),
        "normalization_guidance": _normalize_string_list(
            raw_profile.get("normalization_guidance", [])
        ),
        "response_guidance": _normalize_string_list(raw_profile.get("response_guidance", [])),
        "screen_analysis_focus": str(raw_profile.get("screen_analysis_focus", "") or "").strip(),
    }
    return profile


def normalize_interview_profiles(raw_profiles: object) -> List[dict]:
    """Return sanitized profiles, always including the built-in SQL profile."""
    profiles: List[dict] = []
    seen: set[str] = set()

    if isinstance(raw_profiles, list):
        for raw_profile in raw_profiles:
            profile = normalize_interview_profile(raw_profile)
            if profile is None:
                continue
            profile_id = str(profile.get("id", "")).lower()
            if not profile_id or profile_id in seen:
                continue
            profiles.append(profile)
            seen.add(profile_id)

    if DEFAULT_PROFILE_ID not in seen:
        profiles.insert(0, copy.deepcopy(SQL_INTERVIEW_PROFILE))

    return profiles


def resolve_interview_profiles(ai_config: dict) -> List[dict]:
    """Return all configured profiles plus built-ins."""
    if not isinstance(ai_config, dict):
        return default_interview_profiles()
    return normalize_interview_profiles(ai_config.get("interview_profiles", []))


def is_interview_profile_enabled(ai_config: dict) -> bool:
    """Return True when any interview profile should be applied."""
    if not isinstance(ai_config, dict):
        return False
    return bool(
        ai_config.get("interview_profile_enabled", False)
        or ai_config.get("sql_profile_enabled", False)
    )


def get_profile_by_id(profiles: Iterable[dict], profile_id: object) -> Optional[dict]:
    wanted = str(profile_id or "").strip().lower()
    if not wanted:
        return None
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        if str(profile.get("id", "")).strip().lower() == wanted:
            return profile
    return None


def get_active_interview_profile(ai_config: dict) -> Optional[dict]:
    """Return the active profile, honoring the legacy SQL toggle."""
    if not is_interview_profile_enabled(ai_config):
        return None

    profiles = resolve_interview_profiles(ai_config)
    active_id = ""
    if isinstance(ai_config, dict):
        active_id = str(ai_config.get("active_interview_profile_id", "") or "").strip()
        if not active_id and bool(ai_config.get("sql_profile_enabled", False)):
            active_id = DEFAULT_PROFILE_ID

    profile = get_profile_by_id(profiles, active_id)
    if profile is not None:
        return profile
    return profiles[0] if profiles else None


def is_sql_profile(profile: Optional[dict]) -> bool:
    if not isinstance(profile, dict):
        return False
    profile_id = str(profile.get("id", "")).strip().lower()
    name = str(profile.get("name", "")).strip().lower()
    return profile_id == DEFAULT_PROFILE_ID or name == "sql"


def iter_profile_terms(profile: dict) -> List[ProfileTerm]:
    terms: List[ProfileTerm] = []
    if not isinstance(profile, dict):
        return terms
    for section in profile.get("sections", []):
        if not isinstance(section, dict):
            continue
        section_name = _normalize_whitespace(section.get("name", "")) or "Terms"
        for item in section.get("items", []):
            if not isinstance(item, dict):
                continue
            canonical = _normalize_whitespace(item.get("canonical", ""))
            if not canonical:
                continue
            terms.append(
                ProfileTerm(
                    section=section_name,
                    canonical=canonical,
                    definition=_normalize_whitespace(item.get("definition", "")),
                    aliases=tuple(_normalize_string_list(item.get("aliases", []))),
                )
            )
    return terms


def build_profile_summary(profile: dict, max_items_per_section: int = 16) -> List[str]:
    """Return compact profile context safe to inject into prompts."""
    if not isinstance(profile, dict):
        return []

    lines: List[str] = []
    description = _normalize_whitespace(profile.get("description", ""))
    context = str(profile.get("context", "") or "").strip()
    if description:
        lines.append(f"Description: {description}")
    if context:
        for line in context.splitlines():
            cleaned = _normalize_whitespace(line)
            if cleaned:
                lines.append(cleaned)

    for section in profile.get("sections", []):
        if not isinstance(section, dict):
            continue
        section_name = _normalize_whitespace(section.get("name", "")) or "Terms"
        items = section.get("items", [])
        if not isinstance(items, list) or not items:
            continue
        lines.append(f"{section_name}:")
        for item in items[:max(1, int(max_items_per_section))]:
            if not isinstance(item, dict):
                continue
            canonical = _normalize_whitespace(item.get("canonical", ""))
            if not canonical:
                continue
            definition = _normalize_whitespace(item.get("definition", ""))
            if definition:
                lines.append(f"{canonical} - {definition}")
            else:
                lines.append(canonical)
    return lines


def _profile_detection_terms(profile: dict) -> List[str]:
    terms = _normalize_string_list(profile.get("keywords", []))
    for term in iter_profile_terms(profile):
        terms.append(term.canonical)
        terms.extend(term.aliases)
    return _normalize_string_list(terms)


def is_profile_related_text(text: str, profile: dict) -> bool:
    """Return True when text appears related to the profile."""
    cleaned = str(text or "").strip()
    if not cleaned or not isinstance(profile, dict):
        return False
    for term in _profile_detection_terms(profile):
        if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", cleaned, re.IGNORECASE):
            return True
    return False


def apply_profile_corrections(text: str, profile: dict) -> Dict[str, Any]:
    """Normalize likely profile-specific term misrecognitions."""
    base: Dict[str, Any] = {
        "text": text,
        "high_confidence": [],
        "medium_confidence": [],
    }
    if not str(text or "").strip() or not isinstance(profile, dict):
        return base

    if not is_profile_related_text(text, profile):
        return base

    corrected_text = str(text)
    high_confidence: List[Dict[str, Any]] = []
    seen_spans: set[Tuple[int, int, str]] = set()

    for term in iter_profile_terms(profile):
        aliases = term.aliases or (term.canonical,)
        for alias in aliases:
            if not alias:
                continue
            pattern = re.compile(
                r"(?<!\w)" + re.escape(alias) + r"(?!\w)",
                re.IGNORECASE,
            )
            matches = list(pattern.finditer(corrected_text))
            if not matches:
                continue

            for match in matches:
                span = (*match.span(), term.canonical.lower())
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                high_confidence.append(
                    {
                        "original": match.group(0),
                        "corrected": term.canonical,
                        "category": term.section,
                        "definition": term.definition,
                    }
                )

            corrected_text = pattern.sub(term.canonical, corrected_text)
            break

    return {
        "text": corrected_text,
        "high_confidence": high_confidence,
        "medium_confidence": [],
    }


def build_profile_system_policy(profile: dict) -> str:
    """Return system-prompt policy text for the active profile."""
    if not isinstance(profile, dict):
        return ""
    name = _normalize_whitespace(profile.get("name", "")) or "Interview"
    guidance = _normalize_string_list(profile.get("response_guidance", []))
    if is_sql_profile(profile):
        lines = [
            "\n\nSQL profile usage policy:",
            "- A SQL interview profile is available in the prompt context for SQL-related questions.",
            "- When a transcript word sounds like a listed SQL function, normalize it to the canonical function name if that improves clarity.",
            "- When you correct or explain a SQL function, include its definition briefly and directly.",
            "- Use the same normalization approach for other context-backed terms when the surrounding prompt makes the intended term clear.",
            "- If the question is not about SQL, do not force the glossary into the answer.",
        ]
    else:
        lines = [
            f"\n\nInterview profile usage policy ({name}):",
            f"- An interview profile named {name} is available in the prompt context.",
            "- Use it to resolve domain terminology, likely transcription mistakes, and expected interview depth.",
            "- If the current question is outside the active profile, answer normally without forcing profile details.",
        ]
    for item in guidance:
        lines.append(f"- {item}")
    return "\n".join(lines)


def format_profile_glossary(profile: dict) -> str:
    """Serialize profile glossary sections into editable text."""
    if not isinstance(profile, dict):
        return ""
    lines: List[str] = []
    for section in profile.get("sections", []):
        if not isinstance(section, dict):
            continue
        section_name = _normalize_whitespace(section.get("name", "")) or "Terms"
        items = section.get("items", [])
        if not isinstance(items, list):
            items = []
        lines.append(f"{section_name}:")
        for item in items:
            if not isinstance(item, dict):
                continue
            canonical = _normalize_whitespace(item.get("canonical", ""))
            if not canonical:
                continue
            definition = _normalize_whitespace(item.get("definition", ""))
            aliases = _normalize_string_list(item.get("aliases", []))
            line = canonical
            if definition:
                line += f" - {definition}"
            if aliases:
                line += f" | aliases: {', '.join(aliases)}"
            lines.append(line)
        lines.append("")
    return "\n".join(lines).strip()


def parse_profile_glossary(text: str) -> List[dict]:
    """Parse editable glossary text into profile sections."""
    sections: List[dict] = []
    current_section = {"name": "Terms", "items": []}

    def commit_section() -> None:
        nonlocal current_section
        if current_section["items"] or current_section["name"] != "Terms":
            sections.append(current_section)
        current_section = {"name": "Terms", "items": []}

    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.endswith(":") and " - " not in line and "|" not in line:
            commit_section()
            current_section = {"name": line[:-1].strip() or "Terms", "items": []}
            continue

        canonical = line
        definition = ""
        aliases: List[str] = []

        if "|" in canonical:
            canonical, meta = canonical.split("|", 1)
            meta = meta.strip()
            meta = re.sub(r"^aliases?\s*:\s*", "", meta, flags=re.IGNORECASE)
            aliases = _normalize_string_list(meta)

        if " - " in canonical:
            canonical, definition = canonical.split(" - ", 1)

        clean_canonical = _normalize_whitespace(canonical)
        if not clean_canonical:
            continue
        current_section["items"].append(
            {
                "canonical": clean_canonical,
                "definition": _normalize_whitespace(definition),
                "aliases": aliases,
            }
        )

    commit_section()
    return _normalize_sections(sections)
