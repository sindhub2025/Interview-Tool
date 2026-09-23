"""Incremental transcript normalizer for streaming segmentation."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Sequence

from ghostmic.services.transcript_store import TranscriptChunk
from ghostmic.utils.text_processing import clean_text, ensure_question_format

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

CONFIDENCE_HIGH = "HIGH"
CONFIDENCE_MEDIUM = "MEDIUM"
CONFIDENCE_LOW = "LOW"


@dataclass(frozen=True)
class NormalizationCorrection:
    raw_term: str
    normalized_term: str
    reason: str
    confidence: str
    context_source: str = ""


@dataclass(frozen=True)
class NormalizationContext:
    """Bounded conversational evidence used to normalize one transcript item."""

    current_raw_transcript: str
    speaker_source: str = "speaker"
    timestamp: float = 0.0
    previous_normalized_interviewer_question: str = ""
    previous_candidate_answer: str = ""
    recent_conversation_turns: tuple[str, ...] = field(default_factory=tuple)
    current_detected_topic: str = ""
    known_technical_terms: tuple[str, ...] = field(default_factory=tuple)
    resume_profile_aliases: Mapping[str, Sequence[str]] = field(default_factory=dict)
    previous_normalized_terms: tuple[str, ...] = field(default_factory=tuple)
    screen_derived_context: str = ""
    confidence_metadata: Mapping[str, Any] = field(default_factory=dict)

    def prompt_block(self, *, max_turns: int = 6, max_chars: int = 3600) -> str:
        """Render only the compact, explicitly labelled context window."""
        aliases = [
            f"{canonical}: {', '.join(str(alias) for alias in aliases[:3])}"
            for canonical, aliases in self.resume_profile_aliases.items()
            if str(canonical).strip() and aliases
        ]
        sections = {
            "CURRENT AUDIO": self.current_raw_transcript,
            "PREVIOUS QUESTION": self.previous_normalized_interviewer_question,
            "PREVIOUS ANSWER": self.previous_candidate_answer,
            "RECENT TURNS": "\n".join(self.recent_conversation_turns[-max_turns:]),
            "KNOWN TERMS": ", ".join(self.known_technical_terms),
            "ACTIVE PROFILE TERMS": "\n".join(aliases),
            "SCREEN CONTEXT": self.screen_derived_context,
            "NORMALIZATION RULES": (
                "Preserve meaning. Fix only obvious STT, punctuation, grammar, "
                "spoken forms, or strongly supported terminology. Never guess. "
                "Prefer raw wording when confidence is low."
            ),
        }
        lines: list[str] = []
        for heading, value in sections.items():
            lines.append(f"{heading}:\n{str(value or '').strip() or '(none)'}")
        return "\n\n".join(lines)[:max_chars]


@dataclass(frozen=True)
class NormalizationResult:
    normalized_text: str
    is_question: bool
    question_type: str
    topic: str
    corrections: tuple[NormalizationCorrection, ...] = field(default_factory=tuple)
    continuation_of_previous: bool = False
    referenced_entities: tuple[str, ...] = field(default_factory=tuple)


_DETERMINISTIC_TERMS = {
    "elastic search": "Elasticsearch",
    "elastic search service": "Elasticsearch Service",
    "cube net ease": "Kubernetes",
    "kube net ease": "Kubernetes",
    "e k s": "EKS",
    "a w s": "AWS",
    "amazon web services": "AWS",
    "sequel": "SQL",
    "sequel server": "SQL Server",
    "post grass": "PostgreSQL",
    "post gres": "PostgreSQL",
    "my sequel": "MySQL",
    "mongo d b": "MongoDB",
    "data base": "database",
    "data bases": "databases",
    "py thon": "Python",
    "java script": "JavaScript",
    "type script": "TypeScript",
    "c sharp": "C#",
    "dot net": ".NET",
}


def normalize_deterministically(context: NormalizationContext) -> NormalizationResult:
    """Conservatively normalize without an external model or unbounded wait."""
    raw = " ".join(str(context.current_raw_transcript or "").split()).strip()
    normalized = clean_text(raw)
    corrections: list[NormalizationCorrection] = []
    terms = dict(_DETERMINISTIC_TERMS)
    for canonical, aliases in context.resume_profile_aliases.items():
        for alias in aliases:
            if str(alias).strip():
                terms[str(alias).strip().lower()] = str(canonical).strip()
    for term in context.known_technical_terms:
        canonical = str(term).strip()
        if canonical:
            terms.setdefault(canonical.lower(), canonical)

    for raw_term, canonical in sorted(terms.items(), key=lambda item: -len(item[0])):
        pattern = re.compile(rf"(?<!\w){re.escape(raw_term)}(?!\w)", re.IGNORECASE)
        if not pattern.search(normalized) or raw_term.lower() == canonical.lower():
            continue
        normalized = pattern.sub(canonical, normalized)
        corrections.append(
            NormalizationCorrection(
                raw_term=raw_term,
                normalized_term=canonical,
                reason="known technical/profile term supported by the normalization dictionary",
                confidence=CONFIDENCE_HIGH,
                context_source="known terms" if raw_term in _DETERMINISTIC_TERMS else "active profile terms",
            )
        )

    is_question = bool(re.search(r"\?|\b(what|why|how|when|where|who|which|can|could|would|should|do|does|did|is|are)\b", normalized, re.IGNORECASE))
    if is_question and not normalized.endswith("?"):
        normalized = ensure_question_format(normalized)
    continuation = bool(context.previous_normalized_interviewer_question and re.match(
        r"^(and|also|what about|how about|why|which one|can you elaborate|does that|is that)\b",
        normalized,
        re.IGNORECASE,
    ))
    topic = context.current_detected_topic.strip()
    if not topic and corrections:
        topic = corrections[-1].normalized_term
    question_type = "follow_up" if continuation else ("direct" if is_question else "statement")
    return NormalizationResult(
        normalized_text=normalized,
        is_question=is_question,
        question_type=question_type,
        topic=topic,
        corrections=tuple(corrections),
        continuation_of_previous=continuation,
        referenced_entities=tuple(dict.fromkeys(c.normalized_term for c in corrections)),
    )


@dataclass(frozen=True)
class SegmentCandidate:
    """Represents one normalized segment slice in a chunk window."""

    start_index: int
    end_index: int
    normalized_text: str
    source: str
    raw_stt_text: str = ""


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
                        raw_stt_text=" ".join(
                            chunk.raw_text.strip()
                            for chunk in chunks[cursor : boundary + 1]
                            if chunk.raw_text.strip()
                        ),
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
