"""Deterministic semantic state layered above the durable session event store."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import re
import time
from typing import Any, Iterable, Mapping, Optional


TURN_QUESTION = "QUESTION"
TURN_ANSWER = "ANSWER"
TURN_FOLLOW_UP = "FOLLOW_UP"
TURN_CLARIFICATION = "CLARIFICATION"
TURN_STATEMENT = "STATEMENT"
TURN_COMMAND = "COMMAND"
TURN_UNKNOWN = "UNKNOWN"

CONFIDENCE_HIGH = "HIGH"
CONFIDENCE_MEDIUM = "MEDIUM"
CONFIDENCE_LOW = "LOW"

_REFERENCE_RE = re.compile(r"\b(this|that|it|these|those|the previous one|the latter|the former)\b", re.IGNORECASE)
_QUESTION_START_RE = re.compile(
    r"^(what|why|how|when|where|who|which|can|could|would|should|do|does|did|is|are|tell|explain)\b",
    re.IGNORECASE,
)
_FOLLOW_UP_START_RE = re.compile(
    r"^(and\s+)?(what about|how about|why|how|can you explain|could you explain|give me an example|which one|does that|is that|would that)\b",
    re.IGNORECASE,
)
_CLARIFICATION_RE = re.compile(r"\b(i mean|to clarify|in other words|let me rephrase|sorry)\b", re.IGNORECASE)
_COMMAND_RE = re.compile(r"^(please\s+)?(show|give|list|compare|define|summarize|repeat|stop|start)\b", re.IGNORECASE)
_SELF_CORRECTION_RE = re.compile(r"\b(i mean|rather|actually|sorry,? I meant|let me correct)\b", re.IGNORECASE)
_TOPIC_STOP_WORDS = {
    "what", "why", "how", "when", "where", "who", "which", "can", "could", "would", "should",
    "do", "does", "did", "is", "are", "the", "a", "an", "in", "on", "for", "to", "of", "and",
    "or", "about", "that", "this", "it", "you", "your", "me", "we", "they", "i", "with",
}


@dataclass(frozen=True)
class ReferenceResolution:
    reference: str
    resolved_to: str
    confidence: str


@dataclass
class ConversationTurn:
    turn_id: str
    timestamp: float
    speaker: str
    raw_text: str
    normalized_text: str
    turn_type: str
    topic: str = ""
    confidence: float = 1.0
    source_chunk_ids: list[str] = field(default_factory=list)
    references: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConversationState:
    """Bounded semantic interview state; it does not replace event storage."""

    def __init__(
        self,
        *,
        max_recent_turns: int = 12,
        max_recent_entities: int = 48,
        max_recent_topics: int = 24,
        max_state_chars: int = 12000,
    ) -> None:
        self.max_recent_turns = max(1, int(max_recent_turns))
        self.max_recent_entities = max(1, int(max_recent_entities))
        self.max_recent_topics = max(1, int(max_recent_topics))
        self.max_state_chars = max(1000, int(max_state_chars))
        self.reset()

    def reset(self) -> None:
        self.current_topic = ""
        self.current_interviewer_question = ""
        self.latest_candidate_answer = ""
        self.previous_interviewer_question = ""
        self.recent_turns: deque[ConversationTurn] = deque(maxlen=self.max_recent_turns)
        self.known_entities: deque[str] = deque(maxlen=self.max_recent_entities)
        self.technical_terms: deque[str] = deque(maxlen=self.max_recent_entities)
        self.unresolved_references: list[dict[str, str]] = []
        self.current_interview_phase = "opening"
        self.important_constraints_facts: deque[str] = deque(maxlen=self.max_recent_entities)
        self.current_screen_context = ""
        self.resume_profile_context: dict[str, Any] = {}
        self.recent_topics: deque[str] = deque(maxlen=self.max_recent_topics)
        self._turn_counter = 0

    def update_turn(
        self,
        *,
        speaker: str,
        raw_text: str,
        normalized_text: Optional[str] = None,
        timestamp: Optional[float] = None,
        confidence: float = 1.0,
        source_chunk_ids: Optional[Iterable[str]] = None,
        turn_type: Optional[str] = None,
        topic: Optional[str] = None,
        entities: Optional[Iterable[str]] = None,
        technical_terms: Optional[Iterable[str]] = None,
    ) -> ConversationTurn:
        raw = " ".join(str(raw_text or "").split()).strip()
        if not raw:
            raise ValueError("Conversation turn text is empty.")
        normalized = " ".join(str(normalized_text or raw).split()).strip() or raw
        source = str(speaker or "unknown").strip().lower() or "unknown"
        references = self._resolve_references(raw)
        inferred_type = self._classify_turn(raw, source, references)
        resolved_topic = str(topic or "").strip() or self._infer_topic(normalized, references)
        if inferred_type == TURN_FOLLOW_UP and (
            not resolved_topic or resolved_topic.lower() in {"production", "example", "it", "that"}
        ):
            resolved_topic = self.current_topic
        if self._is_self_correction(raw):
            inferred_type = TURN_CLARIFICATION
            resolved_topic = resolved_topic or self.current_topic

        self._turn_counter += 1
        turn = ConversationTurn(
            turn_id=f"turn-{self._turn_counter:08d}",
            timestamp=float(timestamp if timestamp is not None else time.time()),
            speaker=source,
            raw_text=raw,
            normalized_text=normalized,
            turn_type=str(turn_type or inferred_type).upper(),
            topic=resolved_topic,
            confidence=max(0.0, min(1.0, float(confidence))),
            source_chunk_ids=[str(item) for item in (source_chunk_ids or []) if str(item).strip()],
            references=references,
        )
        self.recent_turns.append(turn)
        self._update_semantic_fields(turn, entities=entities, technical_terms=technical_terms)
        self._enforce_state_budget()
        return turn

    def update_screen_context(self, context: str) -> None:
        self.current_screen_context = self._clip(context, 1800)
        self._enforce_state_budget()

    def update_resume_context(self, context: Mapping[str, Any] | None) -> None:
        self.resume_profile_context = {
            str(key): self._clip(value, 600)
            for key, value in dict(context or {}).items()
        }
        self._enforce_state_budget()

    def snapshot(self) -> dict[str, Any]:
        """Return semantic state without exposing or mutating durable event data."""
        payload = {
            "current_topic": self.current_topic,
            "current_interviewer_question": self.current_interviewer_question,
            "latest_candidate_answer": self.latest_candidate_answer,
            "previous_interviewer_question": self.previous_interviewer_question,
            "recent_conversation_turns": [turn.as_dict() for turn in self.recent_turns],
            "known_entities": list(self.known_entities),
            "technical_terms": list(self.technical_terms),
            "unresolved_references": list(self.unresolved_references),
            "current_interview_phase": self.current_interview_phase,
            "important_constraints_facts": list(self.important_constraints_facts),
            "current_screen_context": self.current_screen_context,
            "resume_profile_context": dict(self.resume_profile_context),
            "recent_topics": list(self.recent_topics),
        }
        while len(str(payload)) > self.max_state_chars:
            if len(payload["recent_conversation_turns"]) > 1:
                payload["recent_conversation_turns"].pop(0)
                continue
            if payload["current_screen_context"]:
                payload["current_screen_context"] = self._clip(
                    payload["current_screen_context"],
                    max(80, len(payload["current_screen_context"]) // 2),
                )
                continue
            if payload["resume_profile_context"]:
                payload["resume_profile_context"] = {}
                continue
            break
        return payload

    def _classify_turn(self, text: str, speaker: str, references: list[dict[str, str]]) -> str:
        lowered = text.lower().strip()
        if _CLARIFICATION_RE.search(lowered):
            return TURN_CLARIFICATION
        if self.current_interviewer_question and (
            _FOLLOW_UP_START_RE.search(lowered) or self._looks_like_short_follow_up(lowered)
        ):
            return TURN_FOLLOW_UP
        is_question = bool("?" in text or _QUESTION_START_RE.search(lowered))
        interviewer = speaker in {"speaker", "interviewer", "host", "system"}
        if is_question:
            if references or _FOLLOW_UP_START_RE.search(lowered) or self._looks_like_short_follow_up(lowered):
                return TURN_FOLLOW_UP if self.current_interviewer_question or self.current_topic else TURN_QUESTION
            return TURN_QUESTION
        if _COMMAND_RE.search(lowered) and not self.current_interviewer_question:
            return TURN_COMMAND
        if speaker in {"user", "candidate", "mic", "microphone", "self"}:
            return TURN_ANSWER
        if interviewer and self.current_interviewer_question:
            return TURN_STATEMENT
        return TURN_UNKNOWN

    def _resolve_references(self, text: str) -> list[dict[str, str]]:
        resolutions: list[dict[str, str]] = []
        target = self.current_topic or self._latest_topic_from_turns()
        if not target and self.previous_interviewer_question:
            target = self._infer_topic(self.previous_interviewer_question, [])
        for match in _REFERENCE_RE.finditer(text):
            resolved = target or ""
            confidence = CONFIDENCE_HIGH if resolved and match.group(0).lower() in {"that", "it"} else CONFIDENCE_MEDIUM if resolved else CONFIDENCE_LOW
            item = ReferenceResolution(match.group(0), resolved, confidence).__dict__
            resolutions.append(item)
        return resolutions

    def _infer_topic(self, text: str, references: list[dict[str, str]]) -> str:
        reference_topic = next(
            (reference.get("resolved_to", "") for reference in references if reference.get("resolved_to")),
            "",
        )
        cleaned = re.sub(r"[^A-Za-z0-9+#./-]+", " ", text)
        words = [word for word in cleaned.split() if word.lower() not in _TOPIC_STOP_WORDS]
        explicit_terms = [
            word for word in words
            if word[:1].isupper() or re.search(r"\d|[#.+/-]", word)
        ]
        if reference_topic:
            additions = [word for word in explicit_terms if word.lower() not in reference_topic.lower().split()]
            return " / ".join([reference_topic] + additions[:2])
        if not words:
            return ""
        # Preserve technical compounds and the first meaningful phrase without inventing entities.
        if len(words) >= 3 and words[0].lower() in {"explain", "describe", "implement", "compare"}:
            words = words[1:]
        return " ".join(words[:5]).strip(" .?!")

    def _update_semantic_fields(
        self,
        turn: ConversationTurn,
        *,
        entities: Optional[Iterable[str]],
        technical_terms: Optional[Iterable[str]],
    ) -> None:
        topic_owner = turn.speaker in {"speaker", "interviewer", "host", "system"}
        if turn.topic and topic_owner:
            self.current_topic = turn.topic
            self.recent_topics.append(turn.topic)
        if turn.turn_type in {TURN_QUESTION, TURN_FOLLOW_UP, TURN_CLARIFICATION} and turn.speaker in {"speaker", "interviewer", "host", "system"}:
            if self.current_interviewer_question:
                self.previous_interviewer_question = self.current_interviewer_question
            self.current_interviewer_question = turn.normalized_text
        self.unresolved_references = turn.references[-8:]
        for value in list(entities or ()) + self._extract_entities(turn.normalized_text):
            self._append_unique(self.known_entities, value, self.max_recent_entities)
        for value in list(technical_terms or ()) + self._extract_technical_terms(turn.normalized_text):
            self._append_unique(self.technical_terms, value, self.max_recent_entities)
        if turn.turn_type in {TURN_QUESTION, TURN_FOLLOW_UP}:
            self.current_interview_phase = "questioning"
        elif turn.turn_type == TURN_ANSWER:
            self.current_interview_phase = "answering"
            if re.search(r"\b(must|required|constraint|only|at least|no more than|cannot|can't)\b", turn.normalized_text, re.IGNORECASE):
                self._append_unique(
                    self.important_constraints_facts,
                    turn.normalized_text,
                    self.max_recent_entities,
                )
        elif turn.turn_type == TURN_CLARIFICATION:
            self.current_interview_phase = "clarification"

    def _latest_topic_from_turns(self) -> str:
        for turn in reversed(self.recent_turns):
            if turn.topic:
                return turn.topic
        return ""

    @staticmethod
    def _looks_like_short_follow_up(text: str) -> bool:
        return len(text.split()) <= 7 and bool(re.search(r"\b(why|how|what about|can you explain|give me an example)\b", text, re.IGNORECASE))

    @staticmethod
    def _is_self_correction(text: str) -> bool:
        return bool(_SELF_CORRECTION_RE.search(text))

    @staticmethod
    def _extract_entities(text: str) -> list[str]:
        return [match.group(0) for match in re.finditer(r"\b[A-Z][A-Za-z0-9+#.-]{2,}(?:\s+[A-Z][A-Za-z0-9+#.-]{2,})*", text)]

    @staticmethod
    def _extract_technical_terms(text: str) -> list[str]:
        return [match.group(0) for match in re.finditer(r"\b[A-Za-z][A-Za-z0-9+#./-]*(?:SQL|DB|API|AWS|S3|EKS|Kubernetes|Snowflake|Python|Java|Elastic|Kafka)[A-Za-z0-9+#./-]*\b", text, re.IGNORECASE)]

    @staticmethod
    def _append_unique(target: deque[str], value: str, limit: int) -> None:
        cleaned = " ".join(str(value or "").split()).strip()
        if not cleaned:
            return
        existing = [item for item in target if item.lower() != cleaned.lower()]
        existing.append(cleaned)
        target.clear()
        target.extend(existing[-limit:])

    def _enforce_state_budget(self) -> None:
        self.current_screen_context = self._clip(self.current_screen_context, 1800)
        while len(str(self.snapshot())) > self.max_state_chars and len(self.recent_turns) > 1:
            self.recent_turns.popleft()

    @staticmethod
    def _clip(value: Any, max_chars: int) -> str:
        text = " ".join(str(value or "").split()).strip()
        return text[:max_chars]
