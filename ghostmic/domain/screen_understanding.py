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
class ScreenTable:
    name: str
    columns: List[str] = field(default_factory=list)
    column_types: List[str] = field(default_factory=list)
    column_key_hints: List[str] = field(default_factory=list)   # "PK", "FK", ""
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
