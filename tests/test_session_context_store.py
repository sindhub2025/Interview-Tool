"""Unit tests for per-session temporary context storage."""

import os

from ghostmic.domain import TranscriptSegment
from ghostmic.services.session_context_store import SessionContextStore


def test_session_context_store_uses_unique_file_per_instance(tmp_path):
    store1 = SessionContextStore(base_dir=str(tmp_path))
    store2 = SessionContextStore(base_dir=str(tmp_path))

    assert store1.path != store2.path
    assert os.path.exists(store1.path)
    assert os.path.exists(store2.path)


def test_session_context_store_builds_recent_context_tail(tmp_path):
    store = SessionContextStore(base_dir=str(tmp_path))

    store.append_transcript(
        TranscriptSegment(text="Explain window functions.", source="speaker")
    )
    store.append_typed_prompt("Can you refine that answer?", refine=True)
    store.append_screen_summary("IDE shows a failing assertion in test_ai_engine.")
    store.append_ai_response("Use DENSE_RANK when ties should not skip values.")

    tail = store.build_context_tail(max_events=10, max_chars=4000)

    assert "Speaker: Explain window functions." in tail
    assert "Typed Prompt: Can you refine that answer?" in tail
    assert "Screen Summary: IDE shows a failing assertion in test_ai_engine." in tail
    assert "AI Response: Use DENSE_RANK when ties should not skip values." in tail


def test_session_context_store_tail_respects_event_limit(tmp_path):
    store = SessionContextStore(base_dir=str(tmp_path))
    for i in range(1, 6):
        store.append_event("event", f"item {i}")

    tail = store.build_context_tail(max_events=2, max_chars=1000)

    assert "item 5" in tail
    assert "item 4" in tail
    assert "item 3" not in tail


def test_build_compaction_source_excludes_organized_context(tmp_path):
    store = SessionContextStore(base_dir=str(tmp_path))
    store.append_transcript(
        TranscriptSegment(text="What is a window function?", source="speaker")
    )
    store.append_organized_context(
        "Current Focus: SQL analytics",
        source_signature="sig-1",
        source_event_count=1,
    )

    payload = store.build_compaction_source(max_events=20, max_chars=4000)

    assert payload["event_count"] == 1
    assert "Speaker: What is a window function?" in payload["text"]
    assert "Current Focus: SQL analytics" not in payload["text"]


def test_build_compaction_source_supports_incremental_start_index(tmp_path):
    store = SessionContextStore(base_dir=str(tmp_path))
    store.append_event("transcript", "event one", source="speaker")
    store.append_event("transcript", "event two", source="speaker")
    store.append_event("transcript", "event three", source="speaker")

    payload = store.build_compaction_source(
        start_index=1,
        consume_from_start=True,
        max_events=1,
        max_chars=4000,
    )

    assert "event one" not in payload["text"]
    assert "event two" in payload["text"]
    assert "event three" not in payload["text"]
    assert payload["start_index"] == 1
    assert payload["end_index"] == 2
    assert payload["total_event_count"] == 3


def test_context_tail_includes_multiline_organized_snapshot(tmp_path):
    store = SessionContextStore(base_dir=str(tmp_path))
    store.append_organized_context(
        "Current Focus: SQL optimization\nOpen Follow-ups: indexing tradeoffs",
        source_signature="sig-2",
        source_event_count=3,
    )

    tail = store.build_context_tail(max_events=10, max_chars=4000)

    assert "Organized Context" in tail
    assert "Current Focus: SQL optimization" in tail
    assert "Open Follow-ups: indexing tradeoffs" in tail


def test_get_latest_organized_context_returns_latest_snapshot(tmp_path):
    store = SessionContextStore(base_dir=str(tmp_path))
    store.append_organized_context(
        "Current Focus: old snapshot",
        source_signature="sig-old",
        source_event_count=1,
    )
    store.append_organized_context(
        "Current Focus: latest snapshot\nOpen Follow-ups: discuss indexes",
        source_signature="sig-new",
        source_event_count=3,
    )

    latest = store.get_latest_organized_context(max_chars=1000)

    assert "Current Focus: latest snapshot" in latest
    assert "Open Follow-ups: discuss indexes" in latest
    assert "old snapshot" not in latest


def test_session_context_store_cleanup_deletes_file(tmp_path):
    store = SessionContextStore(base_dir=str(tmp_path))
    path = store.path

    assert os.path.exists(path)
    store.cleanup(delete_file=True)

    assert not os.path.exists(path)
