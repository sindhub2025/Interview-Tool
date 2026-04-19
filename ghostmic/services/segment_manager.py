"""Incremental normalized-segment manager for streaming transcript chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

from ghostmic.services.normalizer_service import NormalizerService
from ghostmic.services.transcript_store import TranscriptStore


@dataclass
class NormalizedTranscriptSegment:
    """Structured normalized segment emitted by the streaming pipeline."""

    segment_id: str
    source_chunk_ids: list[str]
    normalized_text: str
    source: str
    sent_to_ai: bool = False
    created_at: float = field(default_factory=time.time)


class SegmentManager:
    """Converts rolling transcript chunks into de-duplicated segments."""

    def __init__(
        self,
        normalizer: NormalizerService,
        *,
        max_overlap_chunks: int = 2,
        max_dedupe_keys: int = 1500,
    ) -> None:
        self._normalizer = normalizer
        self._max_overlap_chunks = max(1, int(max_overlap_chunks))
        self._max_dedupe_keys = max(200, int(max_dedupe_keys))
        self._processed_chunk_ids: set[str] = set()
        self._emitted_segment_keys: set[str] = set()
        self._emitted_segment_key_order: list[str] = []
        self._segment_counter = 0

    def reset(self) -> None:
        """Clear processing cursor and dedupe state."""
        self._processed_chunk_ids.clear()
        self._emitted_segment_keys.clear()
        self._emitted_segment_key_order.clear()
        self._segment_counter = 0

    def consume_ready_segments(
        self,
        store: TranscriptStore,
        *,
        force_flush: bool = False,
    ) -> list[NormalizedTranscriptSegment]:
        """Emit new normalized segments from the store's current chunk window."""
        chunks = store.get_chunks()
        if not chunks:
            self._processed_chunk_ids.clear()
            return []

        live_ids = {chunk.chunk_id for chunk in chunks}
        self._processed_chunk_ids.intersection_update(live_ids)

        first_new_index = self._first_unprocessed_index(chunks)
        if first_new_index is None:
            store.drop_processed_prefix(
                self._processed_chunk_ids,
                keep_last_chunks=self._max_overlap_chunks,
            )
            return []

        window_start = max(0, first_new_index - self._max_overlap_chunks)
        window_chunks = chunks[window_start:]
        required_start = first_new_index - window_start

        candidates = self._normalizer.build_candidates(
            window_chunks,
            start_index=required_start,
            force_flush=force_flush,
        )

        emitted: list[NormalizedTranscriptSegment] = []
        for candidate in candidates:
            span_chunks = window_chunks[candidate.start_index : candidate.end_index + 1]
            if not span_chunks:
                continue

            span_ids = [chunk.chunk_id for chunk in span_chunks]
            new_ids = [chunk_id for chunk_id in span_ids if chunk_id not in self._processed_chunk_ids]

            # Mark span as processed even when it de-duplicates against an existing output.
            self._processed_chunk_ids.update(span_ids)
            if not new_ids:
                continue

            segment_key = self._segment_key(candidate.normalized_text, new_ids)
            if segment_key in self._emitted_segment_keys:
                continue

            segment = NormalizedTranscriptSegment(
                segment_id=self._next_segment_id(),
                source_chunk_ids=span_ids,
                normalized_text=candidate.normalized_text,
                source=candidate.source,
            )
            emitted.append(segment)
            self._remember_segment_key(segment_key)

        store.drop_processed_prefix(
            self._processed_chunk_ids,
            keep_last_chunks=self._max_overlap_chunks,
        )

        # Keep processed IDs bounded to the current buffer.
        live_after_drop = {chunk.chunk_id for chunk in store.get_chunks()}
        self._processed_chunk_ids.intersection_update(live_after_drop)
        return emitted

    def _first_unprocessed_index(self, chunks) -> int | None:
        for index, chunk in enumerate(chunks):
            if chunk.chunk_id not in self._processed_chunk_ids:
                return index
        return None

    @staticmethod
    def _segment_key(text: str, new_chunk_ids: list[str]) -> str:
        normalized = " ".join(str(text or "").lower().split()).rstrip(".?!")
        return f"{normalized}|{'|'.join(new_chunk_ids)}"

    def _remember_segment_key(self, key: str) -> None:
        self._emitted_segment_keys.add(key)
        self._emitted_segment_key_order.append(key)
        while len(self._emitted_segment_key_order) > self._max_dedupe_keys:
            old_key = self._emitted_segment_key_order.pop(0)
            self._emitted_segment_keys.discard(old_key)

    def _next_segment_id(self) -> str:
        self._segment_counter += 1
        return f"seg-{self._segment_counter:08d}"