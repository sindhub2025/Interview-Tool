"""Unit tests for ghostmic.utils.text_processing — clean_text, merge_segments."""

import time

import pytest

from ghostmic.domain import TranscriptSegment
from ghostmic.utils.text_processing import clean_text, ensure_question_format, merge_segments


class TestCleanText:
    def test_removes_noise_tags(self):
        assert clean_text("[Music] Hello world [Applause]") == "Hello world"

    def test_removes_parenthetical_noise(self):
        assert clean_text("(inaudible) Test text (background noise)") == "Test text"

    def test_keeps_noise_tags_when_disabled(self):
        result = clean_text("[Music] Hello", remove_noise_tags=False)
        assert "[Music]" in result

    def test_removes_fillers(self):
        result = clean_text("Um, like, you know, it's important", remove_fillers=True)
        assert "um" not in result.lower()
        assert "like" not in result.lower()
        assert "important" in result.lower()

    def test_fix_capitalisation(self):
        result = clean_text("hello. world. test", fix_capitalisation=True)
        assert result.startswith("H")
        assert "World" in result

    def test_disables_capitalisation(self):
        result = clean_text("hello world", fix_capitalisation=False)
        assert result == "hello world"

    def test_collapses_whitespace(self):
        assert clean_text("too   many    spaces") == "Too many spaces"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_only_noise_tags(self):
        assert clean_text("[Music] [Applause]") == ""

    def test_single_char(self):
        assert clean_text("a") == "A"

    def test_ensure_question_format_appends_question_mark(self):
        assert (
            ensure_question_format("how do you validate row counts between source and target tables.")
            == "how do you validate row counts between source and target tables?"
        )


class TestMergeSegments:
    def _seg(self, text, source="speaker", ts=None, confidence=1.0):
        return TranscriptSegment(
            text=text,
            source=source,
            timestamp=ts if ts is not None else time.time(),
            confidence=confidence,
        )

    def test_empty_list(self):
        assert merge_segments([]) == []

    def test_single_segment(self):
        seg = self._seg("Hello")
        result = merge_segments([seg])
        assert len(result) == 1
        assert result[0].text == "Hello"

    def test_merges_same_source_close_timestamps(self):
        t = time.time()
        s1 = self._seg("Hello", ts=t)
        s2 = self._seg("World", ts=t + 0.5)
        result = merge_segments([s1, s2], gap_seconds=1.0)
        assert len(result) == 1
        assert result[0].text == "Hello World"

    def test_does_not_merge_different_sources(self):
        t = time.time()
        s1 = self._seg("Speaker says", source="speaker", ts=t)
        s2 = self._seg("User says", source="user", ts=t + 0.5)
        result = merge_segments([s1, s2], gap_seconds=1.0)
        assert len(result) == 2

    def test_does_not_merge_beyond_gap(self):
        t = time.time()
        s1 = self._seg("First", ts=t)
        s2 = self._seg("Second", ts=t + 5.0)
        result = merge_segments([s1, s2], gap_seconds=1.0)
        assert len(result) == 2

    def test_confidence_averaged(self):
        t = time.time()
        s1 = self._seg("A", ts=t, confidence=0.8)
        s2 = self._seg("B", ts=t + 0.5, confidence=0.6)
        result = merge_segments([s1, s2], gap_seconds=1.0)
        assert abs(result[0].confidence - 0.7) < 0.01
