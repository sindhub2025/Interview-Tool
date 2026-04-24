"""Incremental transcript normalizer for streaming segmentation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from ghostmic.services.transcript_store import TranscriptChunk
from ghostmic.utils.text_processing import clean_text

_TERMINAL_PUNCTUATION_RE = re.compile(r"[.?!][\"')\]]*\s*$")

# Chunks ending with these tail words are often mid-thought and should
# not be split solely because of a pause gap.
_CONTINUATION_TAIL_WORDS = {
    "and",
    "or",
    "to",
    "of",
    "for",
    "with",
    "without",
    "between",
    "than",
    "versus",
    "vs",
    "that",
    "which",
    "who",
    "whose",
    "where",
    "when",
    "while",
    "if",
    "because",
    "in",
    "on",
    "at",
    "by",
    "as",
    "about",
}


@dataclass(frozen=True)
class SegmentCandidate:
    """Represents one normalized segment slice in a chunk window."""

    start_index: int
    end_index: int
    normalized_text: str
    source: str


class NormalizerService:
    """Build normalized text segments from incremental transcript chunks."""

    def __init__(
        self,
        *,
        pause_boundary_seconds: float = 0.8,
        soft_flush_seconds: float = 6.0,
        soft_flush_chunks: int = 5,
        min_segment_chars: int = 20,
    ) -> None:
        self._pause_boundary_seconds = max(0.30, float(pause_boundary_seconds))
        self._soft_flush_seconds = max(2.0, float(soft_flush_seconds))
        self._soft_flush_chunks = max(3, int(soft_flush_chunks))
        self._min_segment_chars = max(1, int(min_segment_chars))


    def build_candidates(
        self,
        chunks: Sequence[TranscriptChunk],
        *,
        start_index: int,
        force_flush: bool = False,
    ) -> list[SegmentCandidate]:
        """Build finalized normalized segments from a chunk window.

        Args:
            chunks: Window of time-ordered transcript chunks.
            start_index: First index in ``chunks`` that must be considered new.
            force_flush: When True, flush trailing partial content as a segment.
        """
        if not chunks:
            return []

        cursor = max(0, min(int(start_index), len(chunks)))
        if cursor >= len(chunks):
            return []

        candidates: list[SegmentCandidate] = []

        while cursor < len(chunks):
            boundary = self._find_boundary(chunks, cursor)
            if boundary is None:
                if force_flush:
                    boundary = len(chunks) - 1
                else:
                    boundary = self._find_soft_flush_boundary(chunks, cursor)
                    if boundary is None:
                        break

            normalized = self._normalize_text(chunks[cursor : boundary + 1])
            if normalized:
                candidates.append(
                    SegmentCandidate(
                        start_index=cursor,
                        end_index=boundary,
                        normalized_text=normalized,
                        source=self._dominant_source(chunks[cursor : boundary + 1]),
                    )
                )
            cursor = boundary + 1

        return candidates

    def _find_boundary(
        self,
        chunks: Sequence[TranscriptChunk],
        start_index: int,
    ) -> int | None:
        for index in range(start_index, len(chunks)):
            current = chunks[index]
            if _TERMINAL_PUNCTUATION_RE.search(current.raw_text.strip()):
                return index

            if index + 1 >= len(chunks):
                continue

            next_chunk = chunks[index + 1]
            pause = max(0.0, next_chunk.timestamp_start - current.timestamp_end)
            if pause < self._pause_boundary_seconds:
                continue

            # Do not split on pause if the current chunk clearly ends as an
            # unfinished phrase (for example: "... between DBMS and").
            if self._ends_with_incomplete_tail(current.raw_text):
                continue

            joined = self._normalize_text(chunks[start_index : index + 1])
            if joined:
                return index

        return None

    def _find_soft_flush_boundary(
        self,
        chunks: Sequence[TranscriptChunk],
        start_index: int,
    ) -> int | None:
        remaining = len(chunks) - start_index
        if remaining <= 1:
            return None

        window_start = chunks[start_index].timestamp_start
        window_end = chunks[-1].timestamp_end
        span_seconds = max(0.0, window_end - window_start)

        if remaining >= self._soft_flush_chunks or span_seconds >= self._soft_flush_seconds:
            # Include ALL remaining chunks — the last chunk is part of the
            # content the speaker has already finished saying.  Excluding it
            # caused normalized text to be truncated mid-sentence.
            boundary = len(chunks) - 1
            if boundary >= start_index:
                return boundary
        return None

    def _normalize_text(self, chunks: Sequence[TranscriptChunk]) -> str:
        text = " ".join(chunk.raw_text.strip() for chunk in chunks if chunk.raw_text.strip())
        normalized = clean_text(text)
        if len(normalized) < self._min_segment_chars:
            return ""
        # Minimum word count guard — rejects very short fragments while still
        # allowing concise mic-only queries like "What is SQL?"
        if len(normalized.split()) < 3:
            return ""
        # Apply SQL term corrections synchronously (cheap, no API call).
        # This fixes common Whisper misrecognitions like "soda" → "COUNT()".
        try:
            from ghostmic.utils.sql_context import apply_sql_corrections, is_sql_related_text
            if is_sql_related_text(normalized):
                corrected = apply_sql_corrections(normalized)
                corrected_text = str(corrected.get("text", "") or "").strip()
                if corrected_text:
                    normalized = corrected_text
        except Exception:  # pylint: disable=broad-except
            pass  # Never let SQL correction break the normalization pipeline
        return normalized

    @staticmethod
    def _ends_with_incomplete_tail(text: str) -> bool:
        cleaned = clean_text(str(text or "")).strip()
        if not cleaned:
            return False

        if cleaned.endswith((",", ";", ":", "-", "(", "/")):
            return True

        tail = re.sub(r"[\"')\]]+$", "", cleaned).strip().lower()
        if not tail:
            return False

        last_token = re.sub(r"[^a-z0-9']+", "", tail.split()[-1])
        if not last_token:
            return False

        return last_token in _CONTINUATION_TAIL_WORDS

    @staticmethod
    def _dominant_source(chunks: Sequence[TranscriptChunk]) -> str:
        counts: dict[str, int] = {}
        for chunk in chunks:
            source = str(chunk.source or "speaker").strip().lower() or "speaker"
            counts[source] = counts.get(source, 0) + 1
        if not counts:
            return "speaker"
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]