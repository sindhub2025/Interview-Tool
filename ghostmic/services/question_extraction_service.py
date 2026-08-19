"""Question extraction and validation for conversational transcripts."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
from typing import Sequence

from ghostmic.utils.text_processing import clean_text, ensure_question_format


NO_QUESTION = "NO_QUESTION"
QUESTION_CANDIDATE = "QUESTION_CANDIDATE"
MULTIPLE_QUESTIONS = "MULTIPLE_QUESTIONS"
UNCERTAIN = "UNCERTAIN"

_QUESTION_START_RE = re.compile(
    r"\b("
    r"what|why|how|when|where|who|whom|which|"
    r"can|could|would|should|do|does|did|is|are|will|"
    r"have|has|tell\s+me|tell\s+us|explain|describe|"
    r"walk\s+me\s+through|walk\s+us\s+through|share|compare|contrast|define"
    r")\b",
    re.IGNORECASE,
)
_CORE_QUESTION_RE = re.compile(
    r"\b("
    r"what\s+(?:is|are|was|were|do|does|did|would|could|should|will)|"
    r"how\s+(?:do|does|did|would|could|can|is|are)|"
    r"why\s+(?:do|does|did|would|is|are)|"
    r"when\s+(?:do|does|did|would|is|are|would|will)|"
    r"where\s+(?:do|does|did|is|are|would)|"
    r"who\s+(?:is|are|was|does|would)|"
    r"which\s+(?:is|are|one|would)|"
    r"can\s+you|could\s+you|would\s+you|should\s+you|"
    r"do\s+you|did\s+you|have\s+you|is\s+there|are\s+there|"
    r"tell\s+(?:me|us)|explain|describe|walk\s+(?:me|us)\s+through|"
    r"share|compare|contrast|define"
    r")\b",
    re.IGNORECASE,
)
_CONTEXTUAL_START_RE = re.compile(
    r"\b(when|while|if|as|in\s+case|during)\b",
    re.IGNORECASE,
)
_LEADING_FILLER_RE = re.compile(
    r"^(?:okay|ok|so|right|yeah|yes|no|well|great|sure|alright|and|but)"
    r"(?:[, ]+|$)",
    re.IGNORECASE,
)
_FILLER_ONLY_RE = re.compile(
    r"^(?:okay|ok|yeah|yes|right|sure|great|interesting|perfect|thanks|thank you|"
    r"that's interesting|that is interesting|i see|got it|mm hmm|uh huh)"
    r"(?:[.!, ]|$)+$",
    re.IGNORECASE,
)
_FALSE_POSITIVE_RE = re.compile(
    r"\b(that'?s|that is|this is|it is)\s+(?:a\s+)?good\s+question\b|"
    r"\bno\s+question\b|\bany\s+questions?\b",
    re.IGNORECASE,
)
_RHETORICAL_RE = re.compile(
    r"^\s*(?:who\s+hasn'?t|who\s+doesn'?t|isn'?t\s+it|don'?t\s+we\s+all)\b",
    re.IGNORECASE,
)
_INCOMPLETE_TAIL_RE = re.compile(
    r"\b(and|or|to|of|for|with|between|than|versus|vs|that|which|where|when|"
    r"while|if|because|in|on|at|by|as|about|handle|handled|explain|describe)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExtractedQuestion:
    """A validated question span extracted from noisy interviewer speech."""

    has_question: bool
    confidence: float
    raw_question: str
    classification: str = QUESTION_CANDIDATE
    start_time: float | None = None
    end_time: float | None = None
    speaker: str = "speaker"
    question_type: str = "interview_question"
    source_chunk_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class QuestionExtractionService:
    """Extract actual interview questions from conversational transcript text."""

    def extract_from_text(
        self,
        text: str,
        *,
        source: str = "speaker",
        start_time: float | None = None,
        end_time: float | None = None,
        source_chunk_ids: Sequence[str] | None = None,
        force_finalize: bool = False,
    ) -> list[ExtractedQuestion]:
        cleaned = clean_text(str(text or "")).strip()
        if not cleaned:
            return []

        raw_candidates = self._extract_question_strings(cleaned)
        if not raw_candidates:
            return self._uncertain_or_empty(
                cleaned,
                source=source,
                start_time=start_time,
                end_time=end_time,
                source_chunk_ids=source_chunk_ids,
                force_finalize=force_finalize,
            )

        classification = MULTIPLE_QUESTIONS if len(raw_candidates) > 1 else QUESTION_CANDIDATE
        extracted: list[ExtractedQuestion] = []
        for raw in raw_candidates:
            question = ensure_question_format(raw)
            if not self._is_valid_question(question, force_finalize=force_finalize):
                continue
            extracted.append(
                ExtractedQuestion(
                    has_question=True,
                    confidence=self._confidence(question, cleaned),
                    raw_question=question,
                    classification=classification,
                    start_time=start_time,
                    end_time=end_time,
                    speaker=source,
                    source_chunk_ids=list(source_chunk_ids or []),
                )
            )
        if extracted:
            return extracted
        return self._uncertain_or_empty(
            cleaned,
            source=source,
            start_time=start_time,
            end_time=end_time,
            source_chunk_ids=source_chunk_ids,
            force_finalize=force_finalize,
        )

    def classify(self, text: str, *, force_finalize: bool = False) -> str:
        cleaned = clean_text(str(text or "")).strip()
        if not cleaned:
            return NO_QUESTION
        lowered = cleaned.lower().strip()
        if _FILLER_ONLY_RE.match(lowered) or _FALSE_POSITIVE_RE.search(lowered):
            return NO_QUESTION
        if _RHETORICAL_RE.match(lowered):
            return NO_QUESTION
        if self._looks_incomplete(cleaned) and not force_finalize:
            return UNCERTAIN
        questions = self._extract_question_strings(cleaned)
        if len(questions) > 1:
            return MULTIPLE_QUESTIONS
        if questions:
            return QUESTION_CANDIDATE
        return NO_QUESTION

    def _extract_question_strings(self, text: str) -> list[str]:
        parts = self._split_question_mark_questions(text)
        if not parts:
            start = self._find_question_start(text)
            if start is not None:
                parts = [text[start:]]

        extracted: list[str] = []
        seen: set[str] = set()
        for part in parts:
            cleaned = self._trim_conversation_edges(part)
            if not cleaned:
                continue
            for question in self._split_compound_questions(cleaned):
                key = self._canonical(question)
                if key and key not in seen:
                    seen.add(key)
                    extracted.append(question)
        return extracted

    def _split_question_mark_questions(self, text: str) -> list[str]:
        matches = list(re.finditer(r"\?", text))
        if not matches:
            return []

        questions: list[str] = []
        cursor = 0
        for match in matches:
            span = text[cursor : match.end()].strip()
            if "," in span and len(_CORE_QUESTION_RE.findall(span)) >= 2:
                first_start = self._find_question_start(span)
                if first_start is not None:
                    questions.append(span[first_start:])
                    cursor = match.end()
                    continue
            start = self._find_question_start(span)
            if start is not None:
                questions.append(span[start:])
            cursor = match.end()

        tail = text[cursor:].strip(" .,!;:")
        if tail:
            start = self._find_question_start(tail)
            if start is not None:
                questions.append(tail[start:])
        return questions

    def _find_question_start(self, text: str) -> int | None:
        if _FALSE_POSITIVE_RE.search(text) or _RHETORICAL_RE.match(text):
            return None

        core_matches = list(_CORE_QUESTION_RE.finditer(text))
        if not core_matches:
            if _QUESTION_START_RE.match(text.strip()):
                return 0
            return None

        core = core_matches[0]
        prefix = text[: core.start()]
        contextual_matches = list(_CONTEXTUAL_START_RE.finditer(prefix))
        if contextual_matches:
            context = contextual_matches[-1]
            between = prefix[context.end() :]
            if len(between.split()) <= 16:
                return context.start()
        return core.start()

    def _trim_conversation_edges(self, text: str) -> str:
        cleaned = clean_text(str(text or "")).strip(" ,;:")
        while True:
            updated = _LEADING_FILLER_RE.sub("", cleaned, count=1).strip(" ,;:")
            if updated == cleaned:
                break
            cleaned = updated
        cleaned = re.sub(r"\b(?:right|okay|ok|yeah)\s*[.?!]*$", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip(" ,;:")

    def _split_compound_questions(self, text: str) -> list[str]:
        cleaned = self._trim_conversation_edges(text)
        if not cleaned:
            return []

        # Preserve linked sub-components such as
        # "What is SCD Type 2, how is it different..., and when would you use it?"
        # as one primary question with clauses.
        if "," in cleaned and len(_CORE_QUESTION_RE.findall(cleaned)) >= 2:
            return [cleaned]

        pieces = re.split(
            r"\?\s*(?:and\s+before\s+we\s+move\s+on,?\s*)?|\.\s+",
            cleaned,
            flags=re.IGNORECASE,
        )
        questions: list[str] = []
        for piece in pieces:
            piece = self._trim_conversation_edges(piece)
            if not piece:
                continue
            start = self._find_question_start(piece)
            if start is not None:
                questions.append(piece[start:])
        return questions or [cleaned]

    def _is_valid_question(self, question: str, *, force_finalize: bool) -> bool:
        cleaned = clean_text(str(question or "")).strip()
        lowered = cleaned.lower()
        if len(cleaned.split()) < 3:
            return False
        if _FILLER_ONLY_RE.match(lowered) or _FALSE_POSITIVE_RE.search(lowered):
            return False
        if _RHETORICAL_RE.match(lowered):
            return False
        if self._looks_incomplete(cleaned) and not force_finalize:
            return False
        return bool(_CORE_QUESTION_RE.search(cleaned) or _QUESTION_START_RE.match(cleaned))

    @staticmethod
    def _looks_incomplete(text: str) -> bool:
        cleaned = clean_text(str(text or "")).strip().rstrip(".?!")
        if not cleaned:
            return False
        if cleaned.endswith((",", ";", ":", "-", "—")):
            return True
        return bool(_INCOMPLETE_TAIL_RE.search(cleaned))

    @staticmethod
    def _confidence(question: str, source_text: str) -> float:
        score = 0.72
        if str(question).strip().endswith("?"):
            score += 0.08
        if _CORE_QUESTION_RE.search(question):
            score += 0.12
        if len(question.split()) >= 6:
            score += 0.04
        if len(question) < max(1, len(source_text)) * 0.85:
            score += 0.03
        return min(0.97, score)

    @staticmethod
    def _canonical(text: str) -> str:
        return " ".join(str(text or "").lower().split()).strip().rstrip(".?!")

    def _uncertain_or_empty(
        self,
        cleaned: str,
        *,
        source: str,
        start_time: float | None,
        end_time: float | None,
        source_chunk_ids: Sequence[str] | None,
        force_finalize: bool,
    ) -> list[ExtractedQuestion]:
        classification = self.classify(cleaned, force_finalize=force_finalize)
        if classification != UNCERTAIN:
            return []
        return [
            ExtractedQuestion(
                has_question=False,
                confidence=0.35,
                raw_question=cleaned,
                classification=UNCERTAIN,
                start_time=start_time,
                end_time=end_time,
                speaker=source,
                source_chunk_ids=list(source_chunk_ids or []),
            )
        ]
