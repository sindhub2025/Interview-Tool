"""Unit tests for streaming segment manager behavior."""

from ghostmic.services.normalizer_service import NormalizerService
from ghostmic.services.segment_manager import SegmentManager
from ghostmic.services.transcript_store import TranscriptStore


def test_incremental_segmentation_emits_complete_question_chunks() -> None:
    store = TranscriptStore(max_window_seconds=30.0, max_chunks=120)
    manager = SegmentManager(
        NormalizerService(
            pause_boundary_seconds=1.15,
            soft_flush_seconds=10.0,
            soft_flush_chunks=8,
        )
    )

    store.append_chunk(
        raw_text="Can you explain your ETL validation strategy",
        source="speaker",
        timestamp_start=0.0,
        timestamp_end=1.0,
        chunk_id="c1",
    )
    store.append_chunk(
        raw_text="for late arriving data?",
        source="speaker",
        timestamp_start=1.1,
        timestamp_end=2.0,
        chunk_id="c2",
    )
    store.append_chunk(
        raw_text="What metrics do you track",
        source="speaker",
        timestamp_start=3.0,
        timestamp_end=4.0,
        chunk_id="c3",
    )

    first = manager.consume_ready_segments(store)
    assert len(first) == 1
    assert first[0].source_chunk_ids == ["c1", "c2"]
    assert first[0].normalized_text.endswith("?")

    store.append_chunk(
        raw_text="in production?",
        source="speaker",
        timestamp_start=4.1,
        timestamp_end=5.0,
        chunk_id="c4",
    )

    second = manager.consume_ready_segments(store)
    assert len(second) == 1
    assert second[0].source_chunk_ids == ["c3", "c4"]
    assert second[0].normalized_text.endswith("?")


def test_duplicate_chunks_do_not_reemit_duplicate_segments() -> None:
    store = TranscriptStore(max_window_seconds=30.0, max_chunks=120)
    manager = SegmentManager(NormalizerService())

    first_chunk = store.append_chunk(
        raw_text="How do you test retry backoff policies?",
        source="speaker",
        timestamp_start=0.0,
        timestamp_end=1.2,
        chunk_id="dup-1",
    )
    assert first_chunk is not None

    first_emit = manager.consume_ready_segments(store)
    assert len(first_emit) == 1

    duplicate_chunk = store.append_chunk(
        raw_text="How do you test retry backoff policies?",
        source="speaker",
        timestamp_start=0.0,
        timestamp_end=1.2,
        chunk_id="dup-2",
    )
    assert duplicate_chunk is None

    second_emit = manager.consume_ready_segments(store)
    assert second_emit == []