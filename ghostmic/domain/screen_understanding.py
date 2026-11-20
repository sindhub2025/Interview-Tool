from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class ScreenObservation:
    category: str
    content: str
    confidence: Optional[float] = None   # 0.0-1.0; None = not assessed
    evidence: Optional[str] = None
    is_inferred: bool = False   # True -> inference, not direct observation

@dataclass
class ScreenTableColumn:
    name: str
    type: str = ""
    type_is_inferred: bool = False
    key_hint: str = ""   # "PK", "FK", ""
    key_hint_is_inferred: bool = False
    evidence: str = ""

@dataclass
class ScreenTable:
    name: str
    columns: List[str] = field(default_factory=list)
    column_types: List[str] = field(default_factory=list)
    column_key_hints: List[str] = field(default_factory=list)   # "PK", "FK", ""
    columns_meta: List[ScreenTableColumn] = field(default_factory=list)
    visible_rows: List[List[str]] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    uncertainty_notes: List[str] = field(default_factory=list)

@dataclass
class ScreenCodeBlock:
    language: str
    code: str
    description: str = ""

@dataclass
class ScreenUnderstanding:
    capture_id: str
    summary: str
    detected_application: Optional[str] = None
    detected_task: Optional[str] = None
    observations: List[ScreenObservation] = field(default_factory=list)
    tables: List[ScreenTable] = field(default_factory=list)
    code_blocks: List[ScreenCodeBlock] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    key_values: List[tuple] = field(default_factory=list)   # (key, value)
    inferred_relationships: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    suggested_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""   # original provider text before parsing


@dataclass
class ScreenAnalysisResult:
    """Atomic result of a single screen analysis request.

    Associates raw analysis text with its structured understanding (if any),
    capture metadata, and the request ID that produced it.

    ``screen_understanding`` is ``None`` when:
    - The structured pipeline failed and the legacy fallback was used.
    - The pipeline explicitly could not parse structured output.
    It is never borrowed from a previous capture.

    ``used_legacy_fallback`` is ``True`` when the new structured pipeline
    failed and the old Groq direct-call path produced ``raw_text``.
    """
    request_id: str                               # unique per analysis request
    capture_id: str                               # ID from CaptureRequest / PreparedCapture
    raw_text: str
    screen_understanding: Optional[ScreenUnderstanding] = None
    capture_metadata: Optional[Any] = None
    analysis_metadata: Optional[Dict[str, Any]] = None
    used_legacy_fallback: bool = False

