"""SQL function glossary and prompt helpers for GhostMic."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class _SqlFunctionEntry:
    section: str
    canonical: str
    definition: str
    aliases: Tuple[str, ...]


_SQL_PROFILE_SECTIONS: Tuple[Tuple[str, Tuple[_SqlFunctionEntry, ...]], ...] = (
    (
        "Aggregate Functions",
        (
            _SqlFunctionEntry("Aggregate Functions", "COUNT()", "number of rows", ("count",)),
            _SqlFunctionEntry("Aggregate Functions", "SUM()", "total of values", ("sum",)),
            _SqlFunctionEntry("Aggregate Functions", "AVG()", "average value", ("avg", "average")),
            _SqlFunctionEntry("Aggregate Functions", "MIN()", "smallest value", ("min", "minimum")),
            _SqlFunctionEntry("Aggregate Functions", "MAX()", "largest value", ("max", "maximum")),
        ),
    ),
    (
        "String Functions",
        (
            _SqlFunctionEntry("String Functions", "UPPER()", "change case to uppercase", ("upper",)),
            _SqlFunctionEntry("String Functions", "LOWER()", "change case to lowercase", ("lower",)),
            _SqlFunctionEntry("String Functions", "LENGTH()", "string length", ("length",)),
            _SqlFunctionEntry("String Functions", "LEN()", "string length", ("len",)),
            _SqlFunctionEntry("String Functions", "SUBSTRING()", "extract part", ("substring", "sub string")),
            _SqlFunctionEntry("String Functions", "SUBSTR()", "extract part", ("substr",)),
            _SqlFunctionEntry("String Functions", "TRIM()", "remove spaces", ("trim",)),
            _SqlFunctionEntry("String Functions", "LTRIM()", "remove spaces", ("ltrim",)),
            _SqlFunctionEntry("String Functions", "RTRIM()", "remove spaces", ("rtrim",)),
            _SqlFunctionEntry("String Functions", "CONCAT()", "join strings", ("concat", "concatenate")),
            _SqlFunctionEntry("String Functions", "REPLACE()", "replace text", ("replace",)),
        ),
    ),
    (
        "Numeric Functions",
        (
            _SqlFunctionEntry("Numeric Functions", "ROUND()", "round number", ("round",)),
            _SqlFunctionEntry("Numeric Functions", "CEILING()", "round up", ("ceiling", "ceil")),
            _SqlFunctionEntry("Numeric Functions", "FLOOR()", "round down", ("floor",)),
            _SqlFunctionEntry("Numeric Functions", "ABS()", "absolute value", ("abs",)),
            _SqlFunctionEntry("Numeric Functions", "POWER()", "exponent", ("power",)),
            _SqlFunctionEntry("Numeric Functions", "MOD()", "remainder", ("mod",)),
        ),
    ),
    (
        "Date & Time Functions",
        (
            _SqlFunctionEntry("Date & Time Functions", "NOW()", "current date-time", ("now",)),
            _SqlFunctionEntry(
                "Date & Time Functions",
                "CURRENT_TIMESTAMP",
                "current date-time",
                ("current timestamp", "current_timestamp"),
            ),
            _SqlFunctionEntry("Date & Time Functions", "CURDATE()", "current date", ("curdate",)),
            _SqlFunctionEntry("Date & Time Functions", "DATE()", "extract date", ("date",)),
            _SqlFunctionEntry("Date & Time Functions", "YEAR()", "year part", ("year",)),
            _SqlFunctionEntry("Date & Time Functions", "MONTH()", "month part", ("month",)),
            _SqlFunctionEntry("Date & Time Functions", "DAY()", "day part", ("day",)),
            _SqlFunctionEntry("Date & Time Functions", "DATEDIFF()", "difference between dates", ("datediff", "date diff", "date difference")),
            _SqlFunctionEntry("Date & Time Functions", "DATEADD()", "add time", ("dateadd", "date add")),
            _SqlFunctionEntry("Date & Time Functions", "INTERVAL", "add time", ("interval",)),
        ),
    ),
    (
        "Conditional Functions",
        (
            _SqlFunctionEntry("Conditional Functions", "CASE", "if-else logic", ("case",)),
            _SqlFunctionEntry(
                "Conditional Functions",
                "COALESCE()",
                "first non-null value",
                ("coalesce", "call as"),
            ),
            _SqlFunctionEntry("Conditional Functions", "NULLIF()", "returns NULL if equal", ("nullif",)),
            _SqlFunctionEntry("Conditional Functions", "IF()", "if-else logic", ("if",)),
        ),
    ),
    (
        "Window (Analytical) Functions",
        (
            _SqlFunctionEntry("Window (Analytical) Functions", "ROW_NUMBER()", "assigns row sequence", ("row number", "row_number")),
            _SqlFunctionEntry("Window (Analytical) Functions", "RANK()", "rank with gaps", ("rank",)),
            _SqlFunctionEntry("Window (Analytical) Functions", "DENSE_RANK()", "rank without gaps", ("dense rank", "dense_rank")),
            _SqlFunctionEntry("Window (Analytical) Functions", "LEAD()", "next row value", ("lead",)),
            _SqlFunctionEntry("Window (Analytical) Functions", "LAG()", "previous row value", ("lag",)),
            _SqlFunctionEntry("Window (Analytical) Functions", "OVER()", "defines window", ("over",)),
        ),
    ),
)

_SQL_ENTRIES: Tuple[_SqlFunctionEntry, ...] = tuple(
    entry for _, entries in _SQL_PROFILE_SECTIONS for entry in entries
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

_SQL_DETECTION_PATTERNS = tuple(
    re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
    for term in _SQL_DETECTION_TERMS
)


def build_sql_profile_summary(max_items_per_section: int = 16) -> List[str]:
    """Return a compact SQL glossary safe to inject into the prompt context."""
    lines: List[str] = []
    for section, entries in _SQL_PROFILE_SECTIONS:
        lines.append(f"{section}:")
        for entry in entries[:max_items_per_section]:
            lines.append(f"{entry.canonical} - {entry.definition}")
    return lines


def is_sql_related_text(text: str) -> bool:
    """Return True when text looks SQL-related or mentions SQL functions."""
    lowered = str(text or "").strip()
    if not lowered:
        return False
    return any(pattern.search(lowered) for pattern in _SQL_DETECTION_PATTERNS)


def apply_sql_corrections(text: str) -> Dict[str, Any]:
    """Normalize likely SQL function misrecognitions in transcript text."""
    base: Dict[str, Any] = {
        "text": text,
        "high_confidence": [],
        "medium_confidence": [],
    }
    if not str(text or "").strip():
        return base

    if not is_sql_related_text(text):
        return base

    corrected_text = text
    high_confidence: List[Dict[str, Any]] = []
    seen_spans: set[Tuple[int, int]] = set()

    for entry in _SQL_ENTRIES:
        for alias in entry.aliases:
            pattern = re.compile(
                r"(?<!\w)" + re.escape(alias) + r"(?!\w)",
                re.IGNORECASE,
            )
            matches = list(pattern.finditer(corrected_text))
            if not matches:
                continue

            for match in matches:
                span = match.span()
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                high_confidence.append(
                    {
                        "original": match.group(0),
                        "corrected": entry.canonical,
                        "category": entry.section,
                        "definition": entry.definition,
                    }
                )

            corrected_text = pattern.sub(entry.canonical, corrected_text)
            break

    return {
        "text": corrected_text,
        "high_confidence": high_confidence,
        "medium_confidence": [],
    }
