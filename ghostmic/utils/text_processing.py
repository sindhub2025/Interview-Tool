"""
Text post-processing utilities for GhostMic.

Cleans up faster-whisper output and merges short adjacent segments.
"""

from __future__ import annotations

import re
from typing import List

from ghostmic.domain import TranscriptSegment
from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)

# Whisper noise tags: [Music], [Applause], (inaudible), etc.
_NOISE_TAG_RE = re.compile(r"\[.*?\]|\(.*?\)", re.IGNORECASE)

# Common filler words (word-boundary match)
_FILLER_WORDS = frozenset(
    [
        "um",
        "uh",
        "erm",
        "like",
        "you know",
        "sort of",
        "kind of",
        "i mean",
        "basically",
        "literally",
        "actually",
        "so",
    ]
)
_FILLER_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _FILLER_WORDS) + r")\b,?\s*",
    re.IGNORECASE,
)


def clean_text(
    text: str,
    remove_noise_tags: bool = True,
    remove_fillers: bool = False,
    fix_capitalisation: bool = True,
) -> str:
    """Clean up a whisper transcription string.

    Args:
        text: Raw transcription text.
        remove_noise_tags: Strip ``[Music]``, ``[Applause]`` etc.
        remove_fillers: Remove common filler words.
        fix_capitalisation: Capitalise the first letter of each sentence.

    Returns:
        Cleaned text string.
    """
    if remove_noise_tags:
        text = _NOISE_TAG_RE.sub("", text)

    if remove_fillers:
        text = _FILLER_RE.sub(" ", text)

    # Collapse multiple spaces / newlines
    text = re.sub(r"\s{2,}", " ", text).strip()

    if fix_capitalisation and text:
        # Capitalise after sentence-ending punctuation
        text = re.sub(
            r"([.!?]\s+)([a-z])",
            lambda m: m.group(1) + m.group(2).upper(),
            text,
        )
        # Capitalise the very first character
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

    return text


def merge_segments(
    segments: List[TranscriptSegment],
    gap_seconds: float = 1.0,
) -> List[TranscriptSegment]:
    """Merge consecutive segments from the same source if they are close in time.

    Args:
        segments: List of TranscriptSegment in chronological order.
        gap_seconds: Maximum gap (seconds) between segments to merge.

    Returns:
        Potentially shorter list of merged TranscriptSegments.
    """
    if not segments:
        return []

    merged: List[TranscriptSegment] = [segments[0]]

    for seg in segments[1:]:
        prev = merged[-1]
        same_source = prev.source == seg.source
        close_enough = (seg.timestamp - prev.timestamp) <= gap_seconds

        if same_source and close_enough:
            merged[-1] = TranscriptSegment(
                text=prev.text + " " + seg.text,
                source=prev.source,
                timestamp=prev.timestamp,
                confidence=(prev.confidence + seg.confidence) / 2,
            )
        else:
            merged.append(seg)

    return merged
