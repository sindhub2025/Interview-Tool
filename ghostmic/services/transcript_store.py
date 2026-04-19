"""Rolling transcript chunk store used by streaming normalization."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Optional

from ghostmic.utils.text_processing import clean_text


@dataclass(frozen=True)
class TranscriptChunk:
    """One incremental transcript chunk captured during recording."""

    chunk_id: str
    timestamp_start: float
    timestamp_end: float
    raw_text: str
    source: str
    arrival_index: int


class TranscriptStore:
    """In-memory rolling transcript buffer for streaming pipelines.

    The store keeps a bounded, time-ordered window of chunks and supports
    prefix compaction once chunks are consumed by downstream processors.
    """

    def __init__(
        self,
        *,
        max_window_seconds: float = 30.0,
        max_chunks: int = 260,
    ) -> None:
        self._max_window_seconds = max(20.0, float(max_window_seconds))
        self._max_chunks = max(40, int(max_chunks))
        self._chunks: list[TranscriptChunk] = []
        self._seen_chunk_ids: set[str] = set()
        self._signature_by_chunk_id: dict[str, str] = {}
        self._seen_signatures: set[str] = set()
        self._arrival_counter = 0
        self._last_end_by_source: dict[str, float] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Clear all buffered chunks and dedupe state."""
        with self._lock:
            self._chunks.clear()
            self._seen_chunk_ids.clear()
            self._signature_by_chunk_id.clear()
            self._seen_signatures.clear()
            self._arrival_counter = 0
            self._last_end_by_source.clear()

    def append_from_transcript_segment(
        self,
        segment: Any,
        *,
        chunk_id: Optional[str] = None,
    ) -> Optional[TranscriptChunk]:
        """Append one chunk derived from a transcript segment object."""
        raw_text = clean_text(str(getattr(segment, "text", "") or ""))
        if not raw_text:
            return None

        source = str(getattr(segment, "source", "speaker") or "speaker").strip().lower()
        if not source:
            source = "speaker"

        timestamp_end = float(getattr(segment, "timestamp", 0.0) or 0.0)
        if timestamp_end <= 0.0:
            timestamp_end = time.time()

        estimated_duration = self._estimate_chunk_duration(raw_text)
        with self._lock:
            last_end = self._last_end_by_source.get(source)

        if last_end is not None:
            # The gap between the previous chunk's end and this chunk's arrival
            # time is what the normalizer uses for pause-boundary detection.
            # We set timestamp_start so that `next.start - prev.end` reflects
            # the real wall-clock gap.  This makes the pause boundary actually
            # fire when the speaker pauses between sentences.
            gap = max(0.0, timestamp_end - last_end - estimated_duration)
            timestamp_start = last_end + gap
            if timestamp_start >= timestamp_end:
                timestamp_start = max(0.0, timestamp_end - estimated_duration)
        else:
            timestamp_start = max(0.0, timestamp_end - estimated_duration)

        return self.append_chunk(
            raw_text=raw_text,
            source=source,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            chunk_id=chunk_id,
        )


    def append_chunk(
        self,
        *,
        raw_text: str,
        source: str,
        timestamp_start: float,
        timestamp_end: float,
        chunk_id: Optional[str] = None,
    ) -> Optional[TranscriptChunk]:
        """Append a prepared transcript chunk into the rolling buffer."""
        cleaned_text = clean_text(str(raw_text or ""))
        if not cleaned_text:
            return None

        normalized_source = str(source or "speaker").strip().lower() or "speaker"

        start = float(timestamp_start or 0.0)
        end = float(timestamp_end or 0.0)
        if end <= 0.0:
            end = time.time()
        if start <= 0.0:
            start = max(0.0, end - self._estimate_chunk_duration(cleaned_text))
        if start > end:
            start, end = end, start
        if start == end:
            end = start + 0.05

        with self._lock:
            if chunk_id is None:
                chunk_id = self._next_chunk_id_unlocked()
            if chunk_id in self._seen_chunk_ids:
                return None

            signature = self._chunk_signature(
                source=normalized_source,
                text=cleaned_text,
                timestamp_start=start,
                timestamp_end=end,
            )
            if signature in self._seen_signatures:
                return None

            self._arrival_counter += 1
            chunk = TranscriptChunk(
                chunk_id=chunk_id,
                timestamp_start=start,
                timestamp_end=end,
                raw_text=cleaned_text,
                source=normalized_source,
                arrival_index=self._arrival_counter,
            )

            self._chunks.append(chunk)
            self._chunks.sort(
                key=lambda item: (
                    item.timestamp_start,
                    item.timestamp_end,
                    item.arrival_index,
                    item.chunk_id,
                )
            )

            self._seen_chunk_ids.add(chunk.chunk_id)
            self._seen_signatures.add(signature)
            self._signature_by_chunk_id[chunk.chunk_id] = signature
            self._last_end_by_source[normalized_source] = max(
                self._last_end_by_source.get(normalized_source, 0.0),
                end,
            )

            self._trim_unlocked()
            return chunk

    def get_chunks(self) -> list[TranscriptChunk]:
        """Return a stable snapshot of current buffered chunks."""
        with self._lock:
            return list(self._chunks)

    def drop_processed_prefix(
        self,
        processed_chunk_ids: set[str],
        *,
        keep_last_chunks: int = 2,
    ) -> int:
        """Drop processed chunks from the head of the buffer.

        This keeps memory bounded while retaining a tiny overlap window for
        context reconstruction around segmentation boundaries.
        """
        keep = max(0, int(keep_last_chunks))
        dropped = 0

        with self._lock:
            while len(self._chunks) > keep:
                head = self._chunks[0]
                if head.chunk_id not in processed_chunk_ids:
                    break
                self._chunks.pop(0)
                self._seen_chunk_ids.discard(head.chunk_id)
                signature = self._signature_by_chunk_id.pop(head.chunk_id, "")
                if signature:
                    self._seen_signatures.discard(signature)
                dropped += 1

        return dropped

    @staticmethod
    def _estimate_chunk_duration(text: str) -> float:
        words = max(1, len(str(text or "").split()))
        return max(0.30, min(2.80, words * 0.26))

    @staticmethod
    def _chunk_signature(
        *,
        source: str,
        text: str,
        timestamp_start: float,
        timestamp_end: float,
    ) -> str:
        canonical = " ".join(str(text or "").lower().split())
        return (
            f"{source}|{round(float(timestamp_start), 2)}|"
            f"{round(float(timestamp_end), 2)}|{canonical}"
        )

    def _next_chunk_id_unlocked(self) -> str:
        return f"chunk-{self._arrival_counter + 1:08d}"

    def _trim_unlocked(self) -> None:
        while len(self._chunks) > self._max_chunks:
            removed = self._chunks.pop(0)
            self._seen_chunk_ids.discard(removed.chunk_id)
            signature = self._signature_by_chunk_id.pop(removed.chunk_id, "")
            if signature:
                self._seen_signatures.discard(signature)

        if not self._chunks:
            return

        latest_end = self._chunks[-1].timestamp_end
        cutoff = latest_end - self._max_window_seconds
        # Keep at least two chunks so segmentation context does not vanish.
        while len(self._chunks) > 2 and self._chunks[0].timestamp_end < cutoff:
            removed = self._chunks.pop(0)
            self._seen_chunk_ids.discard(removed.chunk_id)
            signature = self._signature_by_chunk_id.pop(removed.chunk_id, "")
            if signature:
                self._seen_signatures.discard(signature)