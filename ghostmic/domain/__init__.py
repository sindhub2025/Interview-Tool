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

    def __post_init__(self) -> None:
        if not self.raw_stt_text:
            self.raw_stt_text = self.text
        if not self.normalized_text:
            self.normalized_text = self.text
