from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

BLOCK_HEADING = "heading"
BLOCK_PARAGRAPH = "paragraph"
BLOCK_BULLET_LIST = "bullet_list"
BLOCK_NUMBERED_LIST = "numbered_list"
BLOCK_CODE = "code"
BLOCK_TABLE = "table"
BLOCK_KEY_VALUE = "key_value"
BLOCK_WARNING = "warning"
BLOCK_UNCERTAINTY = "uncertainty"
BLOCK_RELATIONSHIP = "relationship"

RESPONSE_TYPE_SCREEN_ANALYSIS = "screen_analysis"
RESPONSE_TYPE_INTERVIEW_ANSWER = "interview_answer"
RESPONSE_TYPE_FOLLOW_UP = "follow_up"
RESPONSE_TYPE_REFINED = "refined"
RESPONSE_TYPE_GENERAL = "general"

@dataclass
class ResponseBlock:
    block_type: str
    content: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResponseDocument:
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    response_type: str = RESPONSE_TYPE_GENERAL
    title: str = "Answer"
    blocks: List[ResponseBlock] = field(default_factory=list)
    plain_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_plain_text(text: str, title: str = "Answer", response_type: str = RESPONSE_TYPE_GENERAL) -> "ResponseDocument":
        return ResponseDocument(
            title=title,
            response_type=response_type,
            plain_text=text,
            blocks=[],  # renderer will parse from plain_text if blocks empty
        )
