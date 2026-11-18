"""
Domain models shared across layers.

TranscriptSegment is the core data type passed between transcription,
AI, and UI layers.  Placing it here breaks the circular import
between ``utils/text_processing`` → ``core/transcription_engine``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TranscriptSegment:
    """A single transcribed speech segment."""

    text: str
    source: str                 # "speaker" or "user"
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    session_id: int | None = None
    raw_stt_text: str = ""
    normalized_text: str = ""
    segment_id: str = ""
    chunk_ids: list[str] = field(default_factory=list)
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0
    status: str = "finalized"
    corrections: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.raw_stt_text:
            self.raw_stt_text = self.text
        if not self.normalized_text:
            self.normalized_text = self.text


from ghostmic.domain.captured_screen import (  # noqa: E402
    CaptureRequest, CaptureMetadata, CapturedFrame, PreparedCapture, MonitorInfo,
)
from ghostmic.domain.screen_understanding import (  # noqa: E402
    ScreenObservation, ScreenTable, ScreenCodeBlock, ScreenUnderstanding,
)
from ghostmic.domain.response_document import (  # noqa: E402
    ResponseBlock, ResponseDocument,
    BLOCK_HEADING, BLOCK_PARAGRAPH, BLOCK_BULLET_LIST, BLOCK_NUMBERED_LIST,
    BLOCK_CODE, BLOCK_TABLE, BLOCK_KEY_VALUE, BLOCK_WARNING, BLOCK_UNCERTAINTY,
    BLOCK_RELATIONSHIP, RESPONSE_TYPE_SCREEN_ANALYSIS, RESPONSE_TYPE_INTERVIEW_ANSWER,
    RESPONSE_TYPE_FOLLOW_UP, RESPONSE_TYPE_REFINED, RESPONSE_TYPE_GENERAL,
)
