"""
AI response engine.

Groq and Gemini are active backends exposed in the app.
OpenAI code paths are intentionally preserved for future re-enable.

Runs in a dedicated QThread and emits streaming response tokens.
"""

from __future__ import annotations

import queue
import random
import re
import threading
import time
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

from ghostmic.utils.errors import is_rate_limited as _is_rate_limited_shared

try:
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:
    QThread = object  # type: ignore[misc,assignment]
    pyqtSignal = None  # type: ignore[assignment]

from ghostmic.core.transcription_engine import TranscriptSegment
from ghostmic.utils.resume_context import (
    apply_resume_corrections,
    build_resume_context_summary,
    is_resume_related_text,
)
from ghostmic.utils.sql_context import (
    apply_sql_corrections,
    build_sql_profile_summary,
    is_sql_related_text,
)
from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You are a real-time interview/meeting assistant helping the user respond to \
questions and participate in a conversation.

Write like a helpful person answering in real time:
- Answer the current question directly first.
- Use plain conversational language and contractions when they sound natural.
- Keep it concise, but do not sound stiff, templated, or overly formal.
- If the answer benefits from it, add one short example or practical detail.

Assume the interview is for an experienced candidate:
- Answer at a senior level, not like a fresher.
- Do not start from absolute basics unless the question explicitly asks for basics.
- Focus on practical tradeoffs, implementation details, decision criteria, and risks.
- Avoid textbook-style definitions unless they are the shortest way to answer.
- Keep the answer straight to the point.

When the user needs a fuller answer, use this structure:
- Start with a direct answer in 2-3 sentences.
- Follow with exactly 5 bullet points covering the most important takeaways.
- Finish with examples when they help, especially for SQL or Python questions.
- Put SQL examples in fenced code blocks tagged `sql`.
- Put Python examples in fenced code blocks tagged `python`.
- Keep SQL and Python examples in separate fenced code blocks so they are easy to spot.

For technical questions:
- Start from the practical answer, not from first principles.
- If it helps, include a short example, but only after the direct answer.
- For SQL or Python questions, prefer a concise sample SQL query or Python \
script that the user can adapt.

If the user asks for a sample script, sample code, Python script, code \
example, snippet, or SQL query:
- Provide a short generic example in a fenced code block when the request does \
not include enough specifics.
- Keep the example realistic and easy to adapt.

Keep the response focused, avoid filler, avoid robotic phrasing, and do not \
include phrases like "Based on the transcript" or "Here's a suggestion". \
Write in first person as if the user is speaking.
"""

DEBOUNCE_SECONDS: float = 3.0
# Keep only the most recent few transcript segments so AI generation
# focuses on the latest 2-3 conversation turns instead of the full history.
MAX_CONTEXT_SEGMENTS: int = 6
AI_QUEUE_MAXSIZE: int = 8
RUNTIME_RATE_LIMIT_RETRIES: int = 3
RUNTIME_RATE_LIMIT_BASE_DELAY: float = 1.0
RUNTIME_RATE_LIMIT_MAX_DELAY: float = 8.0
RESUME_CORRECTION_HIGH_THRESHOLD: float = 0.87
RESUME_CORRECTION_MEDIUM_THRESHOLD: float = 0.74

# ---------------------------------------------------------------------------
# Two-stage Groq → Gemini continuation prompt
# ---------------------------------------------------------------------------
TWO_STAGE_CONTINUATION_SYSTEM_PROMPT = """\
You are continuing an AI response that was started by a fast initial model.
The initial response has already been delivered to the user.
Your job is to CONTINUE and COMPLETE the answer.

Rules:
- Do NOT repeat, rephrase, or summarize anything from the initial response.
- Start exactly where the initial response left off.
- Add depth, examples, nuance, or additional points that were missing.
- If the initial response was already complete and adequate, respond with \
just: [COMPLETE]
- Write naturally as a seamless continuation of the initial response.
- Match the same tone, style, and detail level as the initial response.
- Do not use phrases like "Building on the above" or "As mentioned".
"""

# Jaccard similarity threshold below which two questions are considered
# different topics.  0.25 means less than 25 % word overlap → new topic.
TOPIC_SHIFT_THRESHOLD: float = 0.25

# ---------------------------------------------------------------------------
# Follow-up confidence scoring thresholds
# ---------------------------------------------------------------------------
# Cumulative score at or above this value → treat as follow-up.
FOLLOW_UP_CONFIDENCE_THRESHOLD: float = 0.50

# Individual signal weights used by _compute_follow_up_confidence().
_WEIGHT_EXPLICIT_PATTERN: float = 1.0
_WEIGHT_REFERENCE_TERM: float = 0.35
_WEIGHT_BRIDGE_START: float = 0.30
_WEIGHT_SELECTION_PHRASE: float = 0.40
_WEIGHT_SHARED_DOMAIN_KEYWORD: float = 0.40
_WEIGHT_CONFIRMATION_STARTER: float = 0.35
_WEIGHT_CHALLENGE_PATTERN: float = 0.45
_WEIGHT_SCENARIO_EXTENSION: float = 0.50
_WEIGHT_COMPARISON_PHRASE: float = 0.45
_WEIGHT_REFERENCE_BACK: float = 0.55
_WEIGHT_SHORT_BONUS: float = 0.15  # bonus when ≤ 5 tokens and another signal
_WEIGHT_STANDALONE_PENALTY: float = -0.30  # penalty for ≥ 2 novel content words

# Maximum token count for a question to qualify as a "short" follow-up.
SHORT_FOLLOW_UP_MAX_TOKENS: int = 15

# Minimum content-word length for domain-keyword extraction.
_DOMAIN_KEYWORD_MIN_LEN: int = 4

# Common English stop-words excluded from topic-shift comparison.
_STOP_WORDS: FrozenSet[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "on", "at", "by", "for", "with", "from",
    "up", "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "out", "and", "but", "or", "nor",
    "so", "yet", "both", "either", "neither", "not", "only", "own",
    "same", "than", "too", "very", "just", "that", "this", "these",
    "those", "which", "who", "whom", "what", "how", "when", "where",
    "why", "if", "as", "its", "it", "your", "you", "me", "my",
    "we", "our", "they", "their", "them", "he", "she", "his", "her",
    "i", "im", "ive", "dont", "doesnt", "didnt", "wont", "wouldnt",
    "between", "difference", "explain", "describe", "give",
})

ETL_CONTEXT_KEYWORDS = (
    "etl",
    "data warehouse",
    "data warehousing",
    "source table",
    "target table",
    "staging",
    "pipeline",
    "fact table",
    "dimension table",
)

# ---------------------------------------------------------------------------
# Follow-up pattern libraries
# ---------------------------------------------------------------------------

# Explicit continuation / elaboration requests.
FOLLOW_UP_PATTERNS: Tuple[str, ...] = (
    # --- original patterns ---
    r"\bexplain more\b",
    r"\bexplain further\b",
    r"\bexplain further on (?:this|that) topic\b",
    r"\belaborate\b",
    r"\belaborate more\b",
    r"\bcan you elaborate\b",
    r"\bcan you explain (?:that|this|it) more\b",
    r"\btell me more\b",
    r"\bmore detail(?:s)?\b",
    r"\bgo deeper\b",
    r"\bexpand on (?:that|this|it|the topic)\b",
    r"\bcontinue\b",
    r"\bwhat else\b",
    r"\bcan you give (?:an?|another) example\b",
    r"\bgive (?:an?|another) example\b",
    r"\b(?:what|how) about (?:this|that|it)\b",
    r"\bwhy (?:is|was) (?:this|that|it)\b",
    r"\bhow so\b",
    r"\bcan you clarify\b",
    r"\bclarify (?:this|that|it)\b",
    r"\bwhat do you mean\b",
    r"\bwhat does (?:this|that|it) mean\b",
    r"\bcould you expand on (?:this|that|it)\b",
    r"\bwalk me through (?:this|that|it)\b",
    # --- new explicit continuation patterns ---
    r"\btell me about (?:this|that|it)\b",
    r"\bgo on\b",
    r"\bkeep going\b",
    r"\bsay more\b",
    r"\bsay more about (?:this|that|it)\b",
    r"\band then(?: what)?\b",
    r"\bwhat(?:'s| is) next\b",
    r"\bwhat comes after (?:that|this|it)\b",
    r"\bhow does (?:this|that|it) work\b",
    r"\bwhy does (?:this|that|it) matter\b",
    r"\bwhy is (?:this|that|it) important\b",
    r"\bcan you break (?:this|that|it) down\b",
    r"\bbreak (?:this|that|it) down\b",
    r"\bcan you simplify (?:this|that|it)\b",
    r"\bsimplify (?:this|that|it)\b",
    r"\bin simpler terms\b",
    r"\bin other words\b",
    r"\bput (?:it|that|this) differently\b",
    r"\bcan you rephrase (?:this|that|it)\b",
    r"\brephrase (?:this|that|it)\b",
    r"\bsummarize (?:this|that|it)\b",
    r"\bcan you summarize\b",
    r"\bgive me (?:a|the) summary\b",
    r"\bwhat are the (?:pros|cons|tradeoffs|trade-offs|advantages|disadvantages)\b",
    r"\bwhat(?:'s| is) the (?:benefit|downside|drawback|limitation)\b",
    r"\bwhen would (?:you|we|i) use (?:this|that|it)\b",
    r"\bwhere would (?:you|we|i) use (?:this|that|it)\b",
    r"\bhow would (?:you|we|i) (?:use|apply|implement) (?:this|that|it)\b",
    r"\bis there (?:a|an) (?:example|use case|scenario)\b",
    r"\bgive me (?:a|an) (?:example|use case|scenario)\b",
    r"\blike what\b",
    r"\bsuch as\b",
    r"\bfor (?:example|instance)\b",
    r"\bwhat(?:'s| is) (?:a|an) (?:real[- ]world|practical|concrete) example\b",
    r"\bin practice\b",
    r"\bin real life\b",
    r"\bhow does (?:this|that) (?:differ|compare)\b",
    r"\bhow (?:is|are) (?:this|that|they|those) different\b",
    r"\bwhat(?:'s| is) the catch\b",
    r"\bis there anything else\b",
    r"\banything else (?:to add|worth mentioning|i should know)\b",
    r"\bcan you (?:also|additionally) (?:mention|explain|cover)\b",
)

# Confirmation / agreement starters that usually precede a follow-up.
CONFIRMATION_CONTINUATION_PATTERNS: Tuple[str, ...] = (
    r"^(?:yes|yeah|yep|yup|right|correct|exactly|sure|okay|ok|alright|got it|i see|makes sense|fair enough)\s*[,.]?\s*(?:but|and|so|now|then|how|what|why|can|could|would|does|is|are|do)",
    r"^(?:yes|yeah|yep|yup|right|correct|exactly|sure|okay|ok|alright)\s*[,.]?\s*$",
    r"^(?:i (?:see|understand|get it|agree))\s*[,.]?\s*(?:but|and|so|now|then|how|what|why|can|could|would)",
    r"^(?:that makes sense|fair point|good point)\s*[,.]?\s*(?:but|and|so|now|then|how|what|why|can|could|would)",
    r"^(?:understood|noted)\s*[,.]?\s*(?:but|and|so|now|then|how|what|why|can|could|would)",
)

# Challenge / pushback follow-ups.
CHALLENGE_FOLLOW_UP_PATTERNS: Tuple[str, ...] = (
    r"\bbut (?:isn't|aren't|doesn't|don't|wouldn't|won't|couldn't|can't|shouldn't) (?:that|this|it)\b",
    r"\bbut what (?:if|about)\b",
    r"\bwhat if (?:that|this|it) (?:doesn't|fails|breaks|isn't)\b",
    r"\bisn't (?:that|this|it) (?:the same|similar|different|wrong|incorrect|bad|slow|expensive)\b",
    r"\baren't (?:those|these|they) (?:the same|similar|different)\b",
    r"\bwhy not (?:just|simply|instead)\b",
    r"\bwhy (?:can't|couldn't|shouldn't|wouldn't) (?:we|you|i)\b",
    r"\bdoesn't (?:that|this|it) (?:cause|create|lead to|mean|imply|break|violate)\b",
    r"\bwouldn't (?:that|this|it) (?:be|cause|create|lead to|mean)\b",
    r"\bwhat(?:'s| is) (?:wrong|bad|the problem|the issue|the risk) with (?:that|this|it)\b",
    r"\bwhat (?:are|were) the (?:risks|issues|problems|downsides|limitations|challenges)\b",
    r"\bhow do you (?:handle|deal with|address|avoid|prevent|mitigate)\b",
    r"\bwhat happens (?:if|when) (?:that|this|it) (?:fails|breaks|doesn't work|goes wrong)\b",
)

# Scenario extension: "What if instead we…", "In production, would that…"
SCENARIO_EXTENSION_PATTERNS: Tuple[str, ...] = (
    r"\bwhat if (?:we|you|i|the|there)\b",
    r"\bwhat would happen if\b",
    r"\bwhat about (?:in|for|with|when|during)\b",
    r"\bin (?:production|staging|testing|practice|a real[- ]world|a large[- ]scale)\b",
    r"\bat (?:scale|high volume|large scale)\b",
    r"\bwhat if (?:instead|rather)\b",
    r"\bsuppose (?:we|you|i|the)\b",
    r"\bimagine (?:we|you|i|the|that|a)\b",
    r"\blet(?:'s| us) say\b",
    r"\bhypothetically\b",
    r"\bin that case\b",
    r"\bin (?:this|that) scenario\b",
    r"\bwould (?:that|this|it) (?:still|also|even) (?:work|apply|hold|be true)\b",
    r"\bdoes (?:that|this|it) (?:still|also|even) (?:apply|hold|work)\b",
    r"\bif (?:we|you|i) (?:change|switch|replace|use|add|remove|modify|scale)\b",
    r"\bwhat if the (?:data|load|volume|traffic|input|schema|table|source|requirement) (?:is|was|were|changes|grows|increases)\b",
)

# Comparison follow-ups: "How does X compare to Y", "Is X better than Y"
COMPARISON_FOLLOW_UP_PATTERNS: Tuple[str, ...] = (
    r"\bhow (?:does|do|would|did) (?:that|this|it|they) compare\b",
    r"\bcompare (?:that|this|it|them) (?:to|with|against|versus|vs)\b",
    r"\b(?:what's|what is) the difference\b",
    r"\bhow (?:is|are) (?:that|this|it|they|those) different (?:from|than|compared)\b",
    r"\bversus\b",
    r"\b(?:vs|vs\.)\b",
    r"\bwhich (?:is|would be|performs) (?:faster|slower|better|worse|cheaper|more efficient|more scalable|easier|harder|simpler|more complex)\b",
    r"\bis (?:that|this|it) (?:faster|slower|better|worse|cheaper|more efficient|more scalable|easier|harder) than\b",
    r"\bhow does (?:that|this|it) (?:stack up|measure up|hold up)\b",
    r"\b(?:as opposed to|in contrast to|unlike|rather than)\b",
    r"\bwhat(?:'s| is) the (?:difference|distinction|relationship) between\b",
    r"\bare (?:they|those|these) (?:the same|similar|different|interchangeable|equivalent|related)\b",
)

# Reference-back patterns: "you mentioned…", "going back to…", "earlier you said…"
REFERENCE_BACK_PATTERNS: Tuple[str, ...] = (
    r"\byou (?:mentioned|said|talked about|brought up|pointed out|described|explained|noted|suggested|recommended)\b",
    r"\bgoing back to\b",
    r"\bback to (?:what|the|your|that|this)\b",
    r"\bearlier (?:you|we)\b",
    r"\bpreviously (?:you|we)\b",
    r"\bbefore (?:you|we) (?:said|mentioned|talked|discussed)\b",
    r"\blike (?:you|we) (?:said|mentioned|discussed|talked about)\b",
    r"\bas (?:you|we) (?:said|mentioned|discussed|talked about|noted)\b",
    r"\bregarding (?:what|the|that|this)\b",
    r"\babout (?:that|this) (?:thing|point|topic|concept|idea|approach|method|pattern)\b",
    r"\bon that (?:note|point|topic|subject)\b",
    r"\bspeaking of (?:which|that|this)\b",
    r"\brelated to (?:that|this|what)\b",
    r"\bcoming back to\b",
    r"\brevisiting\b",
    r"\bfollowing up on\b",
    r"\bto follow up\b",
    r"\bto clarify (?:what|the|that|this|your)\b",
)

SELECTION_FOLLOW_UP_PATTERNS: Tuple[str, ...] = (
    r"\bwhich one\b",
    r"\bwhich (?:is|one) (?:better|best)\b",
    r"\bwhich (?:should|would) (?:i|we) use\b",
    r"\bshould (?:i|we) (?:use|choose|pick)\b",
    r"\bwhat should (?:i|we) use\b",
    r"\bwhich (?:would you|do you) (?:use|choose|pick|recommend)\b",
    r"\bwhat (?:would you|do you) recommend\b",
    r"\bwhat(?:'s| is) (?:your|the) (?:preference|recommendation|pick|choice)\b",
    r"\bif you had to (?:choose|pick)\b",
    r"\bwhat(?:'s| is) (?:the|your) (?:go[- ]to|default|preferred|favorite)\b",
    r"\bwhy (?:do|would|did) (?:you|we|i) prefer\b",
    r"\bprefer\b.*\bover\b",
)

FOLLOW_UP_CONTEXT_PREFIX = "Previous AI answer context:"
PRIOR_QUESTION_CONTEXT_PREFIX = "Previous speaker question context:"

FOLLOW_UP_REFERENCE_TERMS: FrozenSet[str] = frozenset({
    # Pronouns & demonstratives
    "this", "that", "it", "those", "these", "them", "same",
    # Temporal/positional references
    "earlier", "previous", "above", "before", "prior", "last", "first",
    "second", "third", "latter", "former",
    # Elaboration cues
    "example", "examples", "again", "more", "detail", "details",
    "further", "deeper",
    # Structural references
    "part", "step", "steps", "approach", "method", "technique",
    "way", "pattern", "solution", "option", "alternative",
    # Conceptual references
    "concept", "idea", "topic", "point", "thing", "stuff",
    "process", "mechanism", "strategy", "logic", "flow",
    # Outcome references
    "result", "outcome", "output", "effect", "impact",
    "consequence", "behavior", "behaviour",
})

GENERIC_SELECTION_TERMS: FrozenSet[str] = frozenset({
    "better", "best", "choose", "choice", "option", "prefer",
    "recommend", "recommended", "versus", "vs", "compare",
    "comparison", "tradeoff", "tradeoffs",
})

# Bridge words/phrases that start a sentence and signal continuation.
_BRIDGE_STARTERS: Tuple[str, ...] = (
    "and ", "also ", "so ", "then ", "right ", "okay ", "ok ",
    "but ", "plus ", "additionally ", "furthermore ", "moreover ",
    "however ", "although ", "though ", "still ", "yet ",
    "similarly ", "likewise ", "meanwhile ",
    "in addition ", "on top of that ", "besides ",
    "in that case ", "if so ", "otherwise ",
)


class AIThread(QThread):  # type: ignore[misc]
    """Generates AI responses in a background thread.

    Emits:
        ``ai_response_chunk(str)`` – individual streaming tokens.
        ``ai_response_ready(str)`` – full assembled response.
        ``ai_error(str)`` – error message.
        ``ai_thinking()`` – emitted when generation starts.

    Args:
        config: The AI section of config.json.
        on_chunk: Optional streaming callback.
        on_ready: Optional completion callback.
    """

    if pyqtSignal is not None:
        ai_response_chunk = pyqtSignal(str)
        ai_response_ready = pyqtSignal(str)
        ai_error = pyqtSignal(str)
        ai_thinking = pyqtSignal()
        # Two-stage Groq → Gemini signals
        ai_continuation_thinking = pyqtSignal()
        ai_continuation_ready = pyqtSignal(str)

    def __init__(
        self,
        config: dict,
        on_chunk: Optional[Callable[[str], None]] = None,
        on_ready: Optional[Callable[[str], None]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._on_chunk = on_chunk
        self._on_ready = on_ready
        self._stop_event = threading.Event()
        self._queue: "queue.Queue[tuple]" = queue.Queue(
            maxsize=AI_QUEUE_MAXSIZE
        )
        self._last_request_time: float = 0.0
        self._last_reject_reason: str = ""

    def stop(self) -> None:
        self._stop_event.set()

    def request_response(
        self,
        transcript: List[TranscriptSegment],
        is_new_topic: bool = False,
        runtime_context_tail: str = "",
    ) -> bool:
        """Queue a response-generation request.

        Applies debounce: requests closer than DEBOUNCE_SECONDS apart are
        discarded.

        Args:
            transcript: Recent transcript segments for context.
            is_new_topic: When True the AI will not use any previous-answer
                context so the response focuses strictly on the new question.
            runtime_context_tail: Optional recent runtime context loaded from
                the per-session context file when live context is thin.

        Returns:
            True when accepted into the queue, otherwise False.
        """
        now = time.time()
        if now - self._last_request_time < DEBOUNCE_SECONDS:
            logger.debug("AIThread: debounce – request ignored.")
            self._last_reject_reason = "debounced"
            return False
        self._last_request_time = now
        payload = (transcript, is_new_topic, runtime_context_tail)
        try:
            self._queue.put_nowait(payload)
            self._last_reject_reason = ""
            return True
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                logger.debug("AIThread: queue was full but empty on readback.")
                self._last_reject_reason = "queue-full"
                return False

            try:
                self._queue.put_nowait(payload)
                logger.debug(
                    "AIThread: queue full – dropped oldest queued request."
                )
                self._last_reject_reason = ""
                return True
            except queue.Full:
                logger.debug(
                    "AIThread: queue remained full – dropping request."
                )
                self._last_reject_reason = "queue-full"
                return False

    def last_reject_reason(self) -> str:
        """Return the most recent queue-rejection reason."""
        return self._last_reject_reason or "unknown"

    def clear_pending_requests(self) -> int:
        """Drop queued requests that have not started execution yet."""
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        if dropped:
            logger.debug("AIThread: dropped %d pending request(s).", dropped)
        return dropped

    def update_config(self, config: dict) -> None:
        """Replace the AI configuration at runtime."""
        self._config = config

    def run(self) -> None:
        self._stop_event.clear()
        logger.info("AIThread: started.")
        while not self._stop_event.is_set():
            try:
                payload = self._queue.get(timeout=0.2)
                runtime_context_tail = ""
                if isinstance(payload, tuple):
                    if len(payload) >= 2:
                        transcript = payload[0]
                        is_new_topic = bool(payload[1])
                        if len(payload) >= 3:
                            runtime_context_tail = str(payload[2] or "").strip()
                    elif len(payload) == 1:
                        transcript = payload[0]
                        is_new_topic = False
                    else:
                        transcript = []
                        is_new_topic = False
                else:
                    transcript = payload
                    is_new_topic = False

                logger.info(
                    "AIThread: got request from queue with %d segments, "
                    "is_new_topic=%s",
                    len(transcript),
                    is_new_topic,
                )
            except queue.Empty:
                continue

            logger.info("AIThread: calling _generate() now...")
            try:
                self._generate(
                    transcript,
                    is_new_topic=is_new_topic,
                    runtime_context_tail=runtime_context_tail,
                )
                logger.info("AIThread: _generate() completed")
            except Exception as e:
                logger.error(
                    "AIThread: _generate() raised exception: %s",
                    e,
                    exc_info=True,
                )

        logger.info("AIThread: stopped.")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _resolve_active_backend(
        self, requested_backend: Optional[str]
    ) -> str:
        """Return the active backend for execution."""
        expose_openai = bool(
            self._config.get("expose_openai_provider", False)
        )

        normalized = str(requested_backend or "").strip().lower()

        if normalized in ("", "groq"):
            return "groq"

        if normalized == "gemini":
            return "gemini"

        if normalized == "openai":
            if expose_openai:
                return "openai"
            logger.info("AIThread: OpenAI backend is hidden; using 'groq'")
            return "groq"

        logger.info(
            "AIThread: backend %r is unsupported; using 'groq'",
            normalized,
        )
        return "groq"

    def _generate(
        self,
        transcript: List[TranscriptSegment],
        is_new_topic: bool = False,
        runtime_context_tail: str = "",
    ) -> None:
        logger.info(
            "AIThread: _generate() starting with %d segments, "
            "is_new_topic=%s",
            len(transcript),
            is_new_topic,
        )
        requested_main = self._config.get("main_backend") or self._config.get(
            "backend", "groq"
        )
        requested_fallback = self._config.get("fallback_backend", "groq")
        main_backend = self._resolve_active_backend(requested_main)
        fallback_backend = self._resolve_active_backend(requested_fallback)
        enable_fallback = bool(self._config.get("enable_fallback", False))

        logger.info(
            "AIThread: using main_backend=%s, fallback_backend=%s, "
            "enable_fallback=%s",
            main_backend,
            fallback_backend,
            enable_fallback,
        )

        session_context = str(
            self._config.get("session_context", "")
        ).strip()
        resume_context_enabled = bool(
            self._config.get("resume_context_enabled", True)
        )
        sql_profile_enabled = bool(self._config.get("sql_profile_enabled", False))
        resume_profile_raw = self._config.get("resume_profile")
        resume_profile = (
            resume_profile_raw
            if resume_context_enabled
            and isinstance(resume_profile_raw, dict)
            else None
        )
        correction_high_raw = self._config.get(
            "resume_correction_threshold_high",
            RESUME_CORRECTION_HIGH_THRESHOLD,
        )
        try:
            correction_high_threshold = float(correction_high_raw)
        except (TypeError, ValueError):
            logger.warning(
                "AIThread: invalid resume_correction_threshold_high=%r; "
                "using default %.2f",
                correction_high_raw,
                RESUME_CORRECTION_HIGH_THRESHOLD,
            )
            correction_high_threshold = RESUME_CORRECTION_HIGH_THRESHOLD

        correction_medium_raw = self._config.get(
            "resume_correction_threshold_medium",
            RESUME_CORRECTION_MEDIUM_THRESHOLD,
        )
        try:
            correction_medium_threshold = float(correction_medium_raw)
        except (TypeError, ValueError):
            logger.warning(
                "AIThread: invalid resume_correction_threshold_medium=%r; "
                "using default %.2f",
                correction_medium_raw,
                RESUME_CORRECTION_MEDIUM_THRESHOLD,
            )
            correction_medium_threshold = RESUME_CORRECTION_MEDIUM_THRESHOLD

        context = self._build_context(
            transcript,
            session_context=session_context,
            runtime_context_tail=runtime_context_tail,
            is_new_topic=is_new_topic,
            resume_profile=resume_profile,
            sql_profile_enabled=sql_profile_enabled,
            correction_high_threshold=correction_high_threshold,
            correction_medium_threshold=correction_medium_threshold,
        )
        logger.info(
            "AIThread: context built (length=%d, is_new_topic=%s)",
            len(context),
            is_new_topic,
        )
        system_prompt = self._build_system_prompt(
            self._config.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
            session_context,
            resume_profile=resume_profile,
            sql_profile_enabled=sql_profile_enabled,
        )
        temperature = float(self._config.get("temperature", 0.7))
        logger.info("AIThread: temperature=%.1f", temperature)

        # ── Two-stage Groq → Gemini mode ──────────────────────────────
        two_stage = bool(self._config.get("two_stage_enabled", False))
        has_groq_key = bool(
            str(self._config.get("groq_api_key", "")).strip()
        )
        has_gemini_key = bool(
            str(self._config.get("gemini_api_key", "")).strip()
        )

        if two_stage and has_groq_key and has_gemini_key:
            logger.info(
                "AIThread: two-stage mode active (Groq → Gemini)"
            )
            self._generate_two_stage(
                context, system_prompt, temperature
            )
            return

        if two_stage and not (has_groq_key and has_gemini_key):
            logger.warning(
                "AIThread: two_stage_enabled but missing API key(s); "
                "falling back to single-backend mode."
            )

        # ── Single-backend mode (original behaviour) ──────────────────
        logger.info("AIThread: trying main backend: %s", main_backend)
        success = self._try_generate(
            main_backend, context, system_prompt, temperature
        )
        logger.info("AIThread: main backend result: success=%s", success)

        if (
            not success
            and enable_fallback
            and fallback_backend != main_backend
        ):
            logger.info(
                "AIThread: main backend (%s) failed, trying fallback (%s)",
                main_backend,
                fallback_backend,
            )
            success = self._try_generate(
                fallback_backend, context, system_prompt, temperature
            )

        if not success:
            logger.error(
                "AIThread: both main and fallback backends failed"
            )

    # ------------------------------------------------------------------
    # Two-stage Groq → Gemini pipeline
    # ------------------------------------------------------------------

    def _generate_two_stage(
        self,
        context: str,
        system_prompt: str,
        temperature: float,
    ) -> None:
        """Fast Groq initial response followed by Gemini continuation.

        Stage 1 – Groq delivers a quick response and emits
        ``ai_response_ready`` so the user sees an answer immediately.

        Stage 2 – Gemini receives the original context *and* the Groq
        response, and is asked to continue/complete without repeating.
        The continuation is emitted via ``ai_continuation_ready``.
        """
        # ── Stage 1: Groq fast response ───────────────────────────────
        groq_text = ""
        try:
            groq_text = self._run_groq_initial(context, system_prompt, temperature)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "AIThread two-stage: Groq stage-1 failed (%s); "
                "falling back to Gemini-only.",
                exc,
            )

        if not groq_text:
            # Groq failed or returned empty → fall back to Gemini for
            # the whole answer (single-backend behaviour).
            logger.info(
                "AIThread two-stage: Groq returned empty; using Gemini only."
            )
            try:
                self._generate_gemini(context, system_prompt, temperature)
            except Exception as exc:  # pylint: disable=broad-except
                error_msg = f"Two-stage fallback Gemini failed: {exc}"
                logger.error("AIThread: %s", error_msg)
                if pyqtSignal is not None:
                    self.ai_error.emit(error_msg)  # type: ignore[attr-defined]
            return

        # Emit Groq response immediately so the user sees it.
        if self._on_ready:
            self._on_ready(groq_text)
        if pyqtSignal is not None:
            self.ai_response_ready.emit(groq_text)  # type: ignore[attr-defined]
        logger.info(
            "AIThread two-stage: Groq initial response delivered (%d chars).",
            len(groq_text),
        )

        # ── Stage 2: Gemini continuation ──────────────────────────────
        if self._stop_event.is_set():
            return

        if pyqtSignal is not None:
            self.ai_continuation_thinking.emit()  # type: ignore[attr-defined]

        try:
            continuation = self._run_gemini_continuation(
                context, system_prompt, groq_text, temperature
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "AIThread two-stage: Gemini continuation failed: %s", exc
            )
            # Groq response was already delivered; continuation failure
            # is non-critical — the user already has an answer.
            return

        # Skip if Gemini determined the answer was already complete.
        if not continuation or "[COMPLETE]" in continuation:
            logger.info(
                "AIThread two-stage: Gemini indicated response is complete."
            )
            return

        if pyqtSignal is not None:
            self.ai_continuation_ready.emit(continuation)  # type: ignore[attr-defined]
        logger.info(
            "AIThread two-stage: Gemini continuation delivered (%d chars).",
            len(continuation),
        )

    def _run_groq_initial(
        self,
        context: str,
        system_prompt: str,
        temperature: float,
    ) -> str:
        """Call Groq (non-streaming) and return the full response text."""
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            ) from exc

        api_key = str(self._config.get("groq_api_key", "")).strip()
        if not api_key:
            raise ValueError("Groq API key not set for two-stage mode.")

        model = str(
            self._config.get("groq_model", "llama-4-maverick-17b-128e-instruct")
        ).strip() or "llama-4-maverick-17b-128e-instruct"

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        logger.info(
            "AIThread two-stage: calling Groq (model=%s, context=%d chars)",
            model,
            len(context),
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
            temperature=temperature,
            max_tokens=512,
            stream=False,
            timeout=30.0,
        )

        if (
            response.choices
            and response.choices[0].message
            and response.choices[0].message.content
        ):
            return response.choices[0].message.content.strip()
        return ""

    def _run_gemini_continuation(
        self,
        original_context: str,
        original_system_prompt: str,
        groq_response: str,
        temperature: float,
    ) -> str:
        """Call Gemini with the Groq response to continue/complete the answer."""
        try:
            from google import genai  # type: ignore[import]
            from google.genai import types  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "google-genai package not installed. Run: pip install google-genai"
            ) from exc

        api_key = str(self._config.get("gemini_api_key", "")).strip()
        if not api_key:
            raise ValueError("Gemini API key not set for two-stage continuation.")

        model = str(
            self._config.get("gemini_model", "gemini-3-flash-preview")
        ).strip() or "gemini-3-flash-preview"

        continuation_user_prompt = (
            "Original question/context:\n"
            f"{original_context}\n\n"
            "Initial response already delivered to user:\n"
            f"{groq_response}\n\n"
            "Continue and complete this response. Add depth, examples, "
            "or additional points that are missing. "
            "If the initial response is already complete, respond with "
            "just: [COMPLETE]"
        )

        client = genai.Client(api_key=api_key)

        logger.info(
            "AIThread two-stage: calling Gemini continuation "
            "(model=%s, prompt=%d chars)",
            model,
            len(continuation_user_prompt),
        )
        response = client.models.generate_content(
            model=model,
            contents=continuation_user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=TWO_STAGE_CONTINUATION_SYSTEM_PROMPT,
                temperature=temperature,
            ),
        )

        return self._extract_gemini_text(response) or ""

    # ------------------------------------------------------------------
    # Single-backend generation
    # ------------------------------------------------------------------

    def _try_generate(
        self,
        backend: str,
        context: str,
        system_prompt: str,
        temperature: float,
    ) -> bool:
        """Try to generate response using specified backend."""
        logger.info(
            "AIThread: _try_generate() starting for backend=%s", backend
        )
        retries = int(
            self._config.get(
                "runtime_rate_limit_retries", RUNTIME_RATE_LIMIT_RETRIES
            )
        )
        base_delay = float(
            self._config.get(
                "runtime_rate_limit_retry_delay",
                RUNTIME_RATE_LIMIT_BASE_DELAY,
            )
        )
        retries = max(1, retries)
        logger.info(
            "AIThread: _try_generate() retries=%d, base_delay=%.1f",
            retries,
            base_delay,
        )

        for attempt in range(retries):
            logger.info(
                "AIThread: _try_generate() attempt %d/%d",
                attempt + 1,
                retries,
            )
            try:
                if backend == "openai":
                    logger.info("AIThread: calling _generate_openai()")
                    self._generate_openai(
                        context, system_prompt, temperature
                    )
                    logger.info("AIThread: _generate_openai() completed")
                    return True
                if backend == "groq":
                    logger.info("AIThread: calling _generate_groq()")
                    self._generate_groq(
                        context, system_prompt, temperature
                    )
                    logger.info("AIThread: _generate_groq() completed")
                    return True
                if backend == "gemini":
                    logger.info("AIThread: calling _generate_gemini()")
                    self._generate_gemini(
                        context, system_prompt, temperature
                    )
                    logger.info("AIThread: _generate_gemini() completed")
                    return True

                logger.error("AIThread: unknown backend %r", backend)
                if pyqtSignal is not None:
                    self.ai_error.emit(  # type: ignore[attr-defined]
                        f"Unknown backend: {backend}"
                    )
                return False
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "AIThread: _try_generate() caught exception: %s",
                    exc,
                    exc_info=True,
                )
                rate_limited = _is_rate_limited_shared(exc)
                can_retry = (
                    rate_limited
                    and attempt < retries - 1
                    and not self._stop_event.is_set()
                )
                if can_retry:
                    delay = min(
                        RUNTIME_RATE_LIMIT_MAX_DELAY,
                        base_delay * (2**attempt),
                    ) + random.uniform(0.0, 0.25)
                    logger.warning(
                        "AIThread: %s rate limited, retrying in %.2fs "
                        "(attempt %d/%d)",
                        backend,
                        delay,
                        attempt + 1,
                        retries,
                    )
                    self._stop_event.wait(delay)
                    continue

                if rate_limited:
                    error_msg = (
                        "Rate limit reached on AI generation (HTTP 429). "
                        "Please wait a moment and try again."
                    )
                else:
                    error_msg = (
                        f"{backend.title()} generation failed: {exc}"
                    )

                logger.debug(
                    "AIThread: backend %r failed: %s", backend, exc
                )
                if pyqtSignal is not None:
                    self.ai_error.emit(error_msg)  # type: ignore[attr-defined]
                return False
        return False

    @staticmethod
    def _is_rate_limited_exception(exc: Exception) -> bool:
        """Delegate to shared utility. Kept for backward compatibility."""
        return _is_rate_limited_shared(exc)

    def _generate_openai(
        self, context: str, system_prompt: str, temperature: float
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as e:
            error_msg = (
                "openai package not installed. Run: pip install openai"
            )
            if pyqtSignal is not None:
                self.ai_error.emit(error_msg)  # type: ignore[attr-defined]
            raise RuntimeError(error_msg) from e

        api_key = self._config.get("openai_api_key", "")
        if not api_key:
            error_msg = (
                "OpenAI API key not set. Add it in Settings -> AI."
            )
            if pyqtSignal is not None:
                self.ai_error.emit(error_msg)  # type: ignore[attr-defined]
            raise ValueError(error_msg)

        client = OpenAI(api_key=api_key)
        self._generate_stream(
            client, system_prompt, context, temperature, "openai"
        )

    def _generate_groq(
        self, context: str, system_prompt: str, temperature: float
    ) -> None:
        logger.info("AIThread: _generate_groq() called")
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as e:
            error_msg = (
                "openai package not installed. Run: pip install openai"
            )
            if pyqtSignal is not None:
                self.ai_error.emit(error_msg)  # type: ignore[attr-defined]
            raise RuntimeError(error_msg) from e

        api_key = self._config.get("groq_api_key", "")
        if not api_key:
            error_msg = "Groq API key not set. Add it in Settings → AI."
            if pyqtSignal is not None:
                self.ai_error.emit(error_msg)  # type: ignore[attr-defined]
            raise ValueError(error_msg)

        logger.info("AIThread: creating Groq client...")
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        logger.info(
            "AIThread: Groq client created, calling _generate_stream()..."
        )
        self._generate_stream(
            client, system_prompt, context, temperature, "groq"
        )

    def _generate_gemini(
        self, context: str, system_prompt: str, temperature: float
    ) -> None:
        logger.info("AIThread: _generate_gemini() called")
        try:
            from google import genai  # type: ignore[import]
        except ImportError as e:
            error_msg = (
                "google-genai package not installed. "
                "Run: pip install google-genai"
            )
            if pyqtSignal is not None:
                self.ai_error.emit(error_msg)  # type: ignore[attr-defined]
            raise RuntimeError(error_msg) from e

        api_key = str(self._config.get("gemini_api_key", "")).strip()
        if not api_key:
            error_msg = "Gemini API key not set. Add it in Settings -> AI."
            if pyqtSignal is not None:
                self.ai_error.emit(error_msg)  # type: ignore[attr-defined]
            raise ValueError(error_msg)

        logger.info("AIThread: creating Gemini client...")
        client = genai.Client(api_key=api_key)
        logger.info(
            "AIThread: Gemini client created, calling _generate_gemini_stream()..."
        )
        self._generate_gemini_stream(
            client, system_prompt, context, temperature
        )

    # ------------------------------------------------------------------
    # Stream Processing (shared by OpenAI and Groq)
    # ------------------------------------------------------------------

    def _generate_stream(
        self,
        client,
        system_prompt: str,
        context: str,
        temperature: float,
        backend_name: str,
    ) -> None:
        """Generate response using standard Chat Completions streaming API."""
        logger.info(
            "AIThread (%s): _generate_stream() starting", backend_name
        )
        full_response: List[str] = []
        stream = None

        try:
            default_model = (
                "llama-4-maverick-17b-128e-instruct"
                if backend_name == "groq"
                else "gpt-4o-mini"
            )
            model = self._config.get(
                f"{backend_name}_model", default_model
            )
            logger.info(
                "AIThread (%s): using model=%s, context_length=%d",
                backend_name,
                model,
                len(context),
            )

            logger.info(
                "AIThread (%s): calling "
                "client.chat.completions.create()",
                backend_name,
            )
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context},
                ],
                temperature=temperature,
                max_tokens=512,
                stream=True,
            )
            logger.info(
                "AIThread (%s): stream object created, starting iteration",
                backend_name,
            )

            stream_timeout = self._config.get("stream_timeout", 30.0)
            stream_start = time.time()
            chunk_count = 0
            logger.info(
                "AIThread (%s): stream_timeout=%s",
                backend_name,
                stream_timeout,
            )

            for chunk in stream:
                if self._stop_event.is_set():
                    logger.info(
                        "AIThread (%s): stream cancelled by stop event",
                        backend_name,
                    )
                    break

                elapsed = time.time() - stream_start
                if elapsed > stream_timeout:
                    error_msg = (
                        f"Stream timeout after {stream_timeout}s"
                    )
                    logger.error(
                        "AIThread (%s): %s", backend_name, error_msg
                    )
                    raise RuntimeError(error_msg)

                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]

                    if choice.finish_reason in (
                        "stop",
                        "length",
                        "tool_calls",
                    ):
                        logger.debug(
                            "AIThread (%s): stream finished "
                            "(reason: %s, chunks: %d)",
                            backend_name,
                            choice.finish_reason,
                            chunk_count,
                        )
                        break

                    if choice.delta and choice.delta.content:
                        delta = choice.delta.content
                        full_response.append(delta)
                        chunk_count += 1

        except Exception as exc:  # pylint: disable=broad-except
            error_msg = f"Stream error: {exc}"
            logger.error("AIThread (%s): %s", backend_name, error_msg)
            raise RuntimeError(error_msg) from exc

        finally:
            if stream is not None and hasattr(stream, "close"):
                try:
                    stream.close()
                except Exception as e:  # pylint: disable=broad-except
                    logger.debug(
                        "AIThread (%s): error closing stream: %s",
                        backend_name,
                        e,
                    )

        complete = "".join(full_response)
        if complete:
            if self._on_ready:
                self._on_ready(complete)
            if pyqtSignal is not None:
                self.ai_response_ready.emit(complete)  # type: ignore[attr-defined]
            logger.info(
                "AIThread (%s): response ready (%d chars)",
                backend_name,
                len(complete),
            )
        else:
            error_msg = "No response generated from AI"
            logger.warning("AIThread (%s): %s", backend_name, error_msg)
            raise RuntimeError(error_msg)

    def _generate_gemini_stream(
        self,
        client,
        system_prompt: str,
        context: str,
        temperature: float,
    ) -> None:
        """Generate response using Gemini streaming API."""
        logger.info("AIThread (gemini): _generate_gemini_stream() starting")
        full_response: List[str] = []
        stream = None

        try:
            from google.genai import types  # type: ignore[import]

            model = str(
                self._config.get("gemini_model", "gemini-3-flash-preview")
            ).strip() or "gemini-3-flash-preview"

            logger.info(
                "AIThread (gemini): using model=%s, context_length=%d",
                model,
                len(context),
            )

            stream = client.models.generate_content_stream(
                model=model,
                contents=context,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                ),
            )
            logger.info(
                "AIThread (gemini): stream object created, starting iteration"
            )

            stream_timeout = self._config.get("stream_timeout", 30.0)
            stream_start = time.time()
            chunk_count = 0

            for chunk in stream:
                if self._stop_event.is_set():
                    logger.info(
                        "AIThread (gemini): stream cancelled by stop event"
                    )
                    break

                elapsed = time.time() - stream_start
                if elapsed > stream_timeout:
                    error_msg = f"Stream timeout after {stream_timeout}s"
                    logger.error("AIThread (gemini): %s", error_msg)
                    raise RuntimeError(error_msg)

                delta = str(getattr(chunk, "text", "") or "")
                if delta:
                    full_response.append(delta)
                    chunk_count += 1

            logger.debug(
                "AIThread (gemini): stream finished (chunks: %d)",
                chunk_count,
            )

        except Exception as exc:  # pylint: disable=broad-except
            error_msg = f"Gemini stream error: {exc}"
            logger.error("AIThread (gemini): %s", error_msg)
            raise RuntimeError(error_msg) from exc

        finally:
            if stream is not None and hasattr(stream, "close"):
                try:
                    stream.close()
                except Exception as e:  # pylint: disable=broad-except
                    logger.debug(
                        "AIThread (gemini): error closing stream: %s",
                        e,
                    )

        complete = "".join(full_response)
        if complete.strip():
            if self._on_ready:
                self._on_ready(complete)
            if pyqtSignal is not None:
                self.ai_response_ready.emit(complete)  # type: ignore[attr-defined]
            logger.info(
                "AIThread (gemini): response ready (%d chars)",
                len(complete),
            )
        else:
            error_msg = "No response generated from Gemini"
            logger.warning("AIThread (gemini): %s", error_msg)
            raise RuntimeError(error_msg)

    @staticmethod
    def _extract_gemini_text(response: Any) -> str:
        """Extract plain text content from Gemini SDK responses."""
        text = str(getattr(response, "text", "") or "").strip()
        if text:
            return text

        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            chunks: List[str] = []
            for part in parts:
                part_text = str(getattr(part, "text", "") or "").strip()
                if part_text:
                    chunks.append(part_text)
            merged = "\n".join(chunks).strip()
            if merged:
                return merged

        return ""

    # ------------------------------------------------------------------
    # Connectivity Test
    # ------------------------------------------------------------------

    def test_api_connectivity(
        self, backend: Optional[str] = None
    ) -> tuple:
        """Test API connectivity by sending a simple message."""
        requested_backend = (
            backend
            or self._config.get("main_backend")
            or self._config.get("backend", "groq")
        )
        test_backend = self._resolve_active_backend(requested_backend)
        test_msg = self._config.get(
            "api_test_message", "Hi, testing API connectivity"
        )

        try:
            if test_backend == "openai":
                response = self._test_openai(test_msg)
                return (True, "openai", response)
            elif test_backend == "gemini":
                response = self._test_gemini(test_msg)
                return (True, "gemini", response)
            elif test_backend == "groq":
                response = self._test_groq(test_msg)
                return (True, "groq", response)
            else:
                return (
                    False,
                    test_backend,
                    f"Unknown backend: {test_backend}",
                )
        except Exception as exc:
            return (False, test_backend, f"Connection failed: {exc}")

    def _test_openai(self, message: str) -> str:
        """Send test message to OpenAI and return response."""
        api_key = self._config.get("openai_api_key", "")
        if not api_key:
            raise ValueError("OpenAI API key not set")

        model = self._config.get("openai_model", "gpt-4o-mini")
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                try:
                    from openai import OpenAI  # type: ignore[import]
                except ImportError as e:
                    raise RuntimeError(
                        "openai package not installed"
                    ) from e

                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": message}],
                    temperature=0.7,
                    max_tokens=100,
                    timeout=10.0,
                )

                if response.choices and len(response.choices) > 0:
                    choice = response.choices[0]
                    if choice.message and choice.message.content:
                        return choice.message.content

                logger.warning(
                    "AIThread: OpenAI response had no content"
                )
                return ""

            except TimeoutError as e:
                if attempt < max_retries - 1:
                    logger.debug(
                        "AIThread: OpenAI test timeout, retrying in "
                        "%.1fs... (attempt %d/%d)",
                        retry_delay,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise RuntimeError(
                        f"OpenAI test timeout after {max_retries} retries"
                    ) from e
            except Exception as e:
                if (
                    attempt < max_retries - 1
                    and "rate" in str(e).lower()
                ):
                    logger.debug(
                        "AIThread: OpenAI rate limit, retrying in "
                        "%.1fs... (attempt %d/%d)",
                        retry_delay,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise
        return ""

    def _test_gemini(self, message: str) -> str:
        """Send test message to Gemini and return response."""
        api_key = str(self._config.get("gemini_api_key", "")).strip()
        if not api_key:
            raise ValueError("Gemini API key not set")

        model = str(
            self._config.get("gemini_model", "gemini-3-flash-preview")
        ).strip() or "gemini-3-flash-preview"
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                try:
                    from google import genai  # type: ignore[import]
                    from google.genai import types  # type: ignore[import]
                except ImportError as e:
                    raise RuntimeError(
                        "google-genai package not installed"
                    ) from e

                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=model,
                    contents=message,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                    ),
                )

                text = self._extract_gemini_text(response)
                if text:
                    return text

                logger.warning(
                    "AIThread: Gemini response had no content"
                )
                return ""

            except TimeoutError as e:
                if attempt < max_retries - 1:
                    logger.debug(
                        "AIThread: Gemini test timeout, retrying in "
                        "%.1fs... (attempt %d/%d)",
                        retry_delay,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise RuntimeError(
                        f"Gemini test timeout after {max_retries} retries"
                    ) from e
            except Exception as e:
                if (
                    attempt < max_retries - 1
                    and "rate" in str(e).lower()
                ):
                    logger.debug(
                        "AIThread: Gemini rate limit, retrying in "
                        "%.1fs... (attempt %d/%d)",
                        retry_delay,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise
        return ""

    def _test_groq(self, message: str) -> str:
        """Send test message to Groq and return response."""
        api_key = self._config.get("groq_api_key", "")
        if not api_key:
            raise ValueError("Groq API key not set")

        model = self._config.get(
            "groq_model", "llama-4-maverick-17b-128e-instruct"
        )
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                try:
                    from openai import OpenAI  # type: ignore[import]
                except ImportError as e:
                    raise RuntimeError(
                        "openai package not installed"
                    ) from e

                client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1",
                )

                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": message}],
                    temperature=0.7,
                    max_tokens=100,
                    timeout=10.0,
                )

                if response.choices and len(response.choices) > 0:
                    choice = response.choices[0]
                    if choice.message and choice.message.content:
                        return choice.message.content

                logger.warning(
                    "AIThread: Groq response had no content"
                )
                return ""

            except TimeoutError as e:
                if attempt < max_retries - 1:
                    logger.debug(
                        "AIThread: Groq test timeout, retrying in "
                        "%.1fs... (attempt %d/%d)",
                        retry_delay,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise RuntimeError(
                        f"Groq test timeout after {max_retries} retries"
                    ) from e
            except Exception as e:
                if (
                    attempt < max_retries - 1
                    and "rate" in str(e).lower()
                ):
                    logger.debug(
                        "AIThread: Groq rate limit, retrying in "
                        "%.1fs... (attempt %d/%d)",
                        retry_delay,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise
        return ""

    # ------------------------------------------------------------------
    # Follow-up detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_domain_keywords(text: str) -> frozenset:
        """Extract content words suitable for domain-keyword comparison.

        Returns lowercase words with length ≥ ``_DOMAIN_KEYWORD_MIN_LEN``
        that are not in ``_STOP_WORDS`` or ``FOLLOW_UP_REFERENCE_TERMS``.
        """
        words = re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
        return frozenset(
            w
            for w in words
            if (
                len(w) >= _DOMAIN_KEYWORD_MIN_LEN
                and w not in _STOP_WORDS
                and w not in FOLLOW_UP_REFERENCE_TERMS
                and w not in GENERIC_SELECTION_TERMS
            )
        )

    @staticmethod
    def _compute_follow_up_confidence(
        prev_question: str,
        new_question: str,
    ) -> float:
        """Return a 0.0–1.0+ confidence score that *new_question* is a follow-up.

        Accumulates weighted signals from multiple detectors.  A score
        at or above ``FOLLOW_UP_CONFIDENCE_THRESHOLD`` means the engine
        should treat this as a follow-up.

        The score can exceed 1.0 when multiple strong signals fire
        simultaneously; callers should treat anything ≥ threshold as
        positive.
        """
        cleaned = new_question.strip().lower()
        if not cleaned:
            return 0.0

        score: float = 0.0
        tokens = re.findall(r"[a-z0-9']+", cleaned)
        token_count = len(tokens)

        # 1. Explicit follow-up pattern → immediate high confidence.
        if any(re.search(p, cleaned) for p in FOLLOW_UP_PATTERNS):
            score += _WEIGHT_EXPLICIT_PATTERN
            logger.debug(
                "follow-up confidence: +%.2f (explicit pattern)",
                _WEIGHT_EXPLICIT_PATTERN,
            )

        # 2. Reference terms (this, that, it, etc.).
        has_reference = any(
            t in FOLLOW_UP_REFERENCE_TERMS for t in tokens
        )
        if has_reference:
            score += _WEIGHT_REFERENCE_TERM
            logger.debug(
                "follow-up confidence: +%.2f (reference term)",
                _WEIGHT_REFERENCE_TERM,
            )

        # 3. Bridge-word starter.
        if any(cleaned.startswith(b) for b in _BRIDGE_STARTERS):
            score += _WEIGHT_BRIDGE_START
            logger.debug(
                "follow-up confidence: +%.2f (bridge start)",
                _WEIGHT_BRIDGE_START,
            )

        # 4. Selection / preference phrase.
        if any(
            re.search(p, cleaned)
            for p in SELECTION_FOLLOW_UP_PATTERNS
        ):
            score += _WEIGHT_SELECTION_PHRASE
            logger.debug(
                "follow-up confidence: +%.2f (selection phrase)",
                _WEIGHT_SELECTION_PHRASE,
            )

        # 5. Confirmation / agreement continuation.
        if any(
            re.search(p, cleaned)
            for p in CONFIRMATION_CONTINUATION_PATTERNS
        ):
            score += _WEIGHT_CONFIRMATION_STARTER
            logger.debug(
                "follow-up confidence: +%.2f (confirmation starter)",
                _WEIGHT_CONFIRMATION_STARTER,
            )

        # 6. Challenge / pushback pattern.
        if any(
            re.search(p, cleaned)
            for p in CHALLENGE_FOLLOW_UP_PATTERNS
        ):
            score += _WEIGHT_CHALLENGE_PATTERN
            logger.debug(
                "follow-up confidence: +%.2f (challenge pattern)",
                _WEIGHT_CHALLENGE_PATTERN,
            )

        # 7. Scenario extension.
        if any(
            re.search(p, cleaned)
            for p in SCENARIO_EXTENSION_PATTERNS
        ):
            score += _WEIGHT_SCENARIO_EXTENSION
            logger.debug(
                "follow-up confidence: +%.2f (scenario extension)",
                _WEIGHT_SCENARIO_EXTENSION,
            )

        # 8. Comparison follow-up.
        if any(
            re.search(p, cleaned)
            for p in COMPARISON_FOLLOW_UP_PATTERNS
        ):
            score += _WEIGHT_COMPARISON_PHRASE
            logger.debug(
                "follow-up confidence: +%.2f (comparison phrase)",
                _WEIGHT_COMPARISON_PHRASE,
            )

        # 9. Reference-back to prior answer / question.
        if any(
            re.search(p, cleaned) for p in REFERENCE_BACK_PATTERNS
        ):
            score += _WEIGHT_REFERENCE_BACK
            logger.debug(
                "follow-up confidence: +%.2f (reference back)",
                _WEIGHT_REFERENCE_BACK,
            )

        # 10. Shared domain keywords with previous question.
        if prev_question.strip():
            prev_kw = AIThread._extract_domain_keywords(prev_question)
            new_kw = AIThread._extract_domain_keywords(new_question)
            shared = prev_kw & new_kw
            if shared and token_count <= SHORT_FOLLOW_UP_MAX_TOKENS:
                # Short question that shares domain terms → likely
                # a follow-up drilling into the same topic.
                score += _WEIGHT_SHARED_DOMAIN_KEYWORD
                logger.debug(
                    "follow-up confidence: +%.2f (shared keywords: %s)",
                    _WEIGHT_SHARED_DOMAIN_KEYWORD,
                    shared,
                )

        # 11. Very-short bonus: ≤ 5 tokens with at least one other signal.
        if token_count <= 5 and score > 0:
            score += _WEIGHT_SHORT_BONUS
            logger.debug(
                "follow-up confidence: +%.2f (very short bonus)",
                _WEIGHT_SHORT_BONUS,
            )

        # 12. Standalone-content penalty: if the question introduces ≥ 2
        #     novel content words that are absent from the previous
        #     question, it is more likely a new standalone question.
        if score > 0 and prev_question.strip():
            new_content = AIThread._extract_domain_keywords(new_question)
            prev_content = AIThread._extract_domain_keywords(prev_question)
            novel = new_content - prev_content
            if len(novel) >= 2:
                score += _WEIGHT_STANDALONE_PENALTY
                logger.debug(
                    "follow-up confidence: %.2f (standalone penalty, "
                    "novel=%s)",
                    _WEIGHT_STANDALONE_PENALTY,
                    novel,
                )

        logger.debug(
            "follow-up confidence total: %.2f (threshold=%.2f) "
            "new=%r prev=%r",
            score,
            FOLLOW_UP_CONFIDENCE_THRESHOLD,
            new_question[:80],
            prev_question[:80],
        )
        return max(0.0, score)

    @staticmethod
    def is_explicit_follow_up_request(text: str) -> bool:
        """Return True when *text* explicitly asks to continue prior topic."""
        cleaned = text.strip().lower()
        if not cleaned:
            return False
        return any(
            re.search(pattern, cleaned) for pattern in FOLLOW_UP_PATTERNS
        )

    @staticmethod
    def is_context_dependent_follow_up(
        prev_question: str,
        new_question: str,
    ) -> bool:
        """Return True when *new_question* likely depends on prior context.

        Uses the confidence-scoring system which combines multiple signals:
        explicit patterns, reference terms, bridge words, challenge patterns,
        scenario extensions, comparisons, reference-back phrases, and shared
        domain keywords.
        """
        cleaned_new = new_question.strip().lower()
        if not cleaned_new:
            return False

        confidence = AIThread._compute_follow_up_confidence(
            prev_question, new_question
        )
        return confidence >= FOLLOW_UP_CONFIDENCE_THRESHOLD

    @staticmethod
    def classify_topic_shift(
        prev_question: str, new_question: str
    ) -> bool:
        """Return True when *new_question* is a semantically different topic.

        Uses Jaccard similarity on content words (non-stop-words ≥ 4 chars).
        A similarity below ``TOPIC_SHIFT_THRESHOLD`` is treated as a new
        topic.

        The follow-up confidence score overrides the Jaccard check: if the
        question scores above the follow-up threshold it is never considered
        a topic shift, even when lexical overlap is low (e.g. "Can you give
        an example?" shares almost no words with the prior question).
        """
        if not new_question.strip():
            return False
        if not prev_question.strip():
            return True  # First question is always a new topic.

        # High follow-up confidence overrides lexical similarity.
        follow_up_conf = AIThread._compute_follow_up_confidence(
            prev_question, new_question
        )
        if follow_up_conf >= FOLLOW_UP_CONFIDENCE_THRESHOLD:
            logger.debug(
                "classify_topic_shift: follow-up confidence %.2f "
                "≥ threshold → NOT a topic shift",
                follow_up_conf,
            )
            return False

        def _content_words(text: str) -> frozenset:
            words = re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
            return frozenset(
                w
                for w in words
                if len(w) >= 4 and w not in _STOP_WORDS
            )

        prev_words = _content_words(prev_question)
        new_words = _content_words(new_question)

        if not prev_words and not new_words:
            return (
                prev_question.strip().lower()
                != new_question.strip().lower()
            )

        intersection = len(prev_words & new_words)
        union = len(prev_words | new_words)
        similarity = intersection / union if union else 0.0

        logger.debug(
            "classify_topic_shift: similarity=%.2f (threshold=%.2f) "
            "prev=%r new=%r",
            similarity,
            TOPIC_SHIFT_THRESHOLD,
            prev_question[:60],
            new_question[:60],
        )
        return similarity < TOPIC_SHIFT_THRESHOLD

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(
        transcript: List[TranscriptSegment],
        session_context: str = "",
        runtime_context_tail: str = "",
        is_new_topic: bool = False,
        resume_profile: Optional[Dict[str, Any]] = None,
        sql_profile_enabled: bool = False,
        correction_high_threshold: float = RESUME_CORRECTION_HIGH_THRESHOLD,
        correction_medium_threshold: float = RESUME_CORRECTION_MEDIUM_THRESHOLD,
    ) -> str:
        """Format the most recent question-focused transcript block."""
        if not transcript:
            return ""

        last_speaker_index = -1
        speaker_block_start = -1
        for index in range(len(transcript) - 1, -1, -1):
            if transcript[index].source == "speaker":
                last_speaker_index = index
                break

        if last_speaker_index < 0:
            recent = transcript[-MAX_CONTEXT_SEGMENTS:]
        else:
            speaker_block_start = last_speaker_index
            while (
                speaker_block_start > 0
                and transcript[speaker_block_start - 1].source == "speaker"
            ):
                speaker_block_start -= 1
            recent = [
                seg
                for seg in transcript[speaker_block_start:]
                if seg.source == "speaker"
            ]

        previous_answer_context = ""
        previous_question_context = ""
        follow_up_type = ""
        if not is_new_topic:
            for seg in reversed(transcript):
                text = str(getattr(seg, "text", "")).strip()
                if (
                    not previous_answer_context
                    and text.startswith(FOLLOW_UP_CONTEXT_PREFIX)
                ):
                    previous_answer_context = text[
                        len(FOLLOW_UP_CONTEXT_PREFIX) :
                    ].strip()
                if (
                    not previous_question_context
                    and text.startswith(PRIOR_QUESTION_CONTEXT_PREFIX)
                ):
                    previous_question_context = text[
                        len(PRIOR_QUESTION_CONTEXT_PREFIX) :
                    ].strip()
                if previous_answer_context and previous_question_context:
                    break

        lower_session_context = session_context.lower()
        joined_recent_text = " ".join(
            seg.text.lower() for seg in recent
        )
        etl_context_active = any(
            keyword in lower_session_context
            or keyword in joined_recent_text
            for keyword in ETL_CONTEXT_KEYWORDS
        )

        segment_payloads: List[Dict[str, Any]] = []
        resume_related = False
        sql_related = False
        sql_summary_lines: List[str] = []
        latest_speaker_text = " ".join(
            seg.text.strip()
            for seg in recent
            if seg.source == "speaker" and seg.text.strip()
        ).strip()

        if sql_profile_enabled:
            sql_related = is_sql_related_text(latest_speaker_text) or is_sql_related_text(session_context)
            sql_summary_lines = build_sql_profile_summary()

        # ---- Enhanced follow-up detection with confidence scoring ----
        if (
            not is_new_topic
            and not previous_question_context
            and latest_speaker_text
            and speaker_block_start > 0
        ):
            prior_speaker_end = -1
            for index in range(speaker_block_start - 1, -1, -1):
                if transcript[index].source == "speaker":
                    prior_speaker_end = index
                    break

            if prior_speaker_end >= 0:
                prior_speaker_start = prior_speaker_end
                while (
                    prior_speaker_start > 0
                    and transcript[prior_speaker_start - 1].source
                    == "speaker"
                ):
                    prior_speaker_start -= 1

                prior_speaker_text = " ".join(
                    str(transcript[idx].text).strip()
                    for idx in range(
                        prior_speaker_start, prior_speaker_end + 1
                    )
                    if str(transcript[idx].text).strip()
                )

                # Use confidence scoring for richer follow-up classification.
                confidence = AIThread._compute_follow_up_confidence(
                    prior_speaker_text, latest_speaker_text
                )
                if confidence >= FOLLOW_UP_CONFIDENCE_THRESHOLD:
                    previous_question_context = prior_speaker_text
                    follow_up_type = (
                        AIThread._classify_follow_up_type(
                            latest_speaker_text
                        )
                    )
                    logger.debug(
                        "follow-up detected (conf=%.2f, type=%s): "
                        "prev=%r → new=%r",
                        confidence,
                        follow_up_type,
                        prior_speaker_text[:60],
                        latest_speaker_text[:60],
                    )

        if resume_profile and latest_speaker_text:
            resume_related = is_resume_related_text(
                latest_speaker_text, resume_profile
            )

        for seg in recent:
            label = "Speaker" if seg.source == "speaker" else "You"
            text = seg.text
            if etl_context_active:
                text = AIThread._normalize_etl_transcript_terms(text)

            high_matches: List[Dict[str, Any]] = []
            medium_matches: List[Dict[str, Any]] = []

            if resume_profile and seg.source == "speaker":
                try:
                    correction_result = apply_resume_corrections(
                        text,
                        resume_profile,
                        high_threshold=correction_high_threshold,
                        medium_threshold=correction_medium_threshold,
                    )
                except Exception as exc:
                    logger.exception(
                        "AIThread: apply_resume_corrections failed: %s",
                        exc,
                    )
                    correction_result = {
                        "high_confidence": [],
                        "medium_confidence": [],
                        "text": text,
                    }

                if not isinstance(correction_result, dict):
                    logger.debug(
                        "AIThread: apply_resume_corrections returned "
                        "non-dict; falling back to defaults"
                    )
                    correction_result = {
                        "high_confidence": [],
                        "medium_confidence": [],
                        "text": text,
                    }

                high_matches = list(
                    correction_result.get("high_confidence", [])
                )
                medium_matches = list(
                    correction_result.get("medium_confidence", [])
                )
                corrected_text = str(
                    correction_result.get("text", text)
                )
                if high_matches:
                    text = corrected_text
                if high_matches or medium_matches:
                    resume_related = True

            if sql_profile_enabled and sql_related and seg.source == "speaker":
                try:
                    sql_result = apply_sql_corrections(text)
                except Exception as exc:
                    logger.exception(
                        "AIThread: apply_sql_corrections failed: %s",
                        exc,
                    )
                    sql_result = {
                        "text": text,
                        "high_confidence": [],
                        "medium_confidence": [],
                    }

                if isinstance(sql_result, dict):
                    sql_text = str(sql_result.get("text", text))
                    sql_matches = list(sql_result.get("high_confidence", []))
                    if sql_matches and sql_text:
                        text = sql_text
                        sql_related = True
                        if sql_summary_lines:
                            for match in sql_matches[:2]:
                                original = str(match.get("original", "")).strip()
                                corrected = str(match.get("corrected", "")).strip()
                                definition = str(match.get("definition", "")).strip()
                                if original and corrected:
                                    lines = [
                                        f'[SQL Correction Applied]: "{original}" -> "{corrected}"',
                                    ]
                                    if definition:
                                        lines[0] += f" ({definition})"
                                    segment_payloads.append(
                                        {
                                            "label": "SQL",
                                            "text": lines[0],
                                            "high_matches": [],
                                            "medium_matches": [],
                                        }
                                    )

            segment_payloads.append(
                {
                    "label": label,
                    "text": text,
                    "high_matches": high_matches,
                    "medium_matches": medium_matches,
                }
            )

        resume_summary_lines: List[str] = []
        if resume_profile and resume_related:
            resume_summary_lines = build_resume_context_summary(
                resume_profile
            )

        lines: List[str] = []
        if session_context:
            lines.append(f"[Session Context]: {session_context}")
        runtime_context_tail = str(runtime_context_tail).strip()
        if runtime_context_tail:
            lines.append("[Runtime Session Context]:")
            for item in runtime_context_tail.splitlines():
                item = item.strip()
                if item:
                    lines.append(f"- {item}")
        if previous_question_context:
            lines.append(
                f"[Prior Question Context]: {previous_question_context}"
            )
        if previous_answer_context:
            lines.append(
                f"[Follow-up Context]: {previous_answer_context}"
            )
        # Inject follow-up type hint so the AI knows *how* to handle
        # the continuation (elaborate vs. compare vs. challenge, etc.).
        if follow_up_type:
            lines.append(
                f"[Follow-up Type]: {follow_up_type}"
            )

        if resume_summary_lines:
            lines.append("[Resume Context]:")
            for item in resume_summary_lines:
                lines.append(f"- {item}")

        if sql_profile_enabled and sql_summary_lines:
            lines.append("[SQL Profile]:")
            for item in sql_summary_lines:
                lines.append(f"- {item}")

            lines.append(
                "[Normalization Guidance]: When a transcript word or phrase "
                "sounds like a canonical term from the supplied context, "
                "normalize it before answering. Apply the same rule to SQL "
                "terms and any other domain terms present in session context "
                "or resume context."
            )

        for payload in segment_payloads:
            lines.append(f"[{payload['label']}]: {payload['text']}")

            high_matches = payload.get("high_matches", [])
            medium_matches = payload.get("medium_matches", [])
            for match in high_matches[:2]:
                original = str(match.get("original", "")).strip()
                corrected = str(match.get("corrected", "")).strip()
                category = str(
                    match.get("category", "term")
                ).strip()
                if original and corrected:
                    lines.append(
                        f'[Resume Correction Applied]: "{original}" '
                        f'-> "{corrected}" '
                        f"({category}, high confidence)"
                    )

            if not high_matches:
                for match in medium_matches[:2]:
                    original = str(
                        match.get("original", "")
                    ).strip()
                    corrected = str(
                        match.get("corrected", "")
                    ).strip()
                    category = str(
                        match.get("category", "term")
                    ).strip()
                    if original and corrected:
                        lines.append(
                            f'[Resume Correction Candidate]: '
                            f'"{original}" -> "{corrected}" '
                            f"({category}, medium confidence)"
                        )

        return "\n".join(lines)

    @staticmethod
    def _classify_follow_up_type(question: str) -> str:
        """Return a human-readable label for the kind of follow-up.

        This label is injected as ``[Follow-up Type]`` in the context so
        the LLM can tailor its response style accordingly.
        """
        cleaned = question.strip().lower()
        if not cleaned:
            return "continuation"

        # Order matters: check most specific patterns first.
        if any(
            re.search(p, cleaned)
            for p in CHALLENGE_FOLLOW_UP_PATTERNS
        ):
            return "challenge / pushback"

        if any(
            re.search(p, cleaned)
            for p in COMPARISON_FOLLOW_UP_PATTERNS
        ):
            return "comparison"

        if any(
            re.search(p, cleaned)
            for p in SCENARIO_EXTENSION_PATTERNS
        ):
            return "scenario extension"

        if any(
            re.search(p, cleaned)
            for p in SELECTION_FOLLOW_UP_PATTERNS
        ):
            return "selection / preference"

        if any(re.search(p, cleaned) for p in REFERENCE_BACK_PATTERNS):
            return "reference back"

        # Elaboration sub-types.
        if re.search(
            r"\b(?:example|instance|use case|scenario)\b", cleaned
        ):
            return "example request"

        if re.search(
            r"\b(?:summarize|summary|recap|overview)\b", cleaned
        ):
            return "summarization"

        if re.search(
            r"\b(?:simplify|simpler|plain|layman)\b", cleaned
        ):
            return "simplification"

        if re.search(
            r"\b(?:pro|cons|tradeoff|trade-off|advantage|disadvantage|"
            r"benefit|downside|drawback|limitation)\b",
            cleaned,
        ):
            return "pros and cons"

        if re.search(
            r"\b(?:clarify|clarification|mean|meaning)\b", cleaned
        ):
            return "clarification"

        # Catch-all for remaining explicit elaboration.
        if any(re.search(p, cleaned) for p in FOLLOW_UP_PATTERNS):
            return "elaboration"

        return "continuation"

    @staticmethod
    def _normalize_etl_transcript_terms(text: str) -> str:
        """Repair common ETL-specific speech-to-text homophone mistakes."""
        normalized = text
        replacements = [
            (
                r"\bsort and target table\b",
                "source and target table",
            ),
            (r"\bsort table\b", "source table"),
            (r"\bsort to target\b", "source to target"),
            (
                r"\bsort and target tables\b",
                "source and target tables",
            ),
            (r"\bsort tables\b", "source tables"),
            (r"\bsort-to-target\b", "source-to-target"),
        ]
        for pattern, replacement in replacements:
            normalized = re.sub(
                pattern, replacement, normalized, flags=re.IGNORECASE
            )
        return normalized

    @staticmethod
    def _build_system_prompt(
        system_prompt: str,
        session_context: str,
        resume_profile: Optional[Dict[str, Any]] = None,
        sql_profile_enabled: bool = False,
    ) -> str:
        prompt = system_prompt
        if session_context:
            addendum = (
                "\n\nSession context (role/background): "
                f"{session_context}\n"
                "Tailor your response to this context if relevant, but "
                "do not force it into answers for general concepts "
                "(e.g., general SQL or programming questions)."
            )
            prompt = f"{prompt}{addendum}"

        if resume_profile:
            resume_policy = (
                "\n\nResume usage policy:\n"
                "- The structured resume context is the source of truth "
                "for the user's professional background.\n"
                "- Use resume facts for resume/background questions "
                "(companies, roles, dates, projects, skills, education, "
                "certifications).\n"
                "- Apply transcript corrections only when confidence is "
                "high and the corrected term is grounded in the resume "
                "context.\n"
                "- When confidence is medium, mention the likely "
                "correction cautiously or ask for confirmation if "
                "needed.\n"
                "- If a detail is not present in the resume context, "
                "explicitly say it is not available rather than "
                "inventing details.\n"
                "- For non-resume questions, do not let resume context "
                "dominate the answer."
            )
            prompt = f"{prompt}{resume_policy}"

        if sql_profile_enabled:
            sql_policy = (
                "\n\nSQL profile usage policy:\n"
                "- A SQL function glossary is available in the prompt context for SQL-related questions.\n"
                "- When a transcript word sounds like a listed SQL function, normalize it to the canonical function name if that improves clarity.\n"
                "- When you correct or explain a SQL function, include its definition briefly and directly.\n"
                "- Use the same normalization approach for other context-backed terms when the surrounding prompt makes the intended term clear.\n"
                "- If the question is not about SQL, do not force the glossary into the answer."
            )
            prompt = f"{prompt}{sql_policy}"

        human_style_policy = (
            "\n\nResponse style guidance:\n"
            "- Sound like a helpful person answering in real time, not a template.\n"
            "- Answer the question directly first in plain conversational language.\n"
            "- Assume the interviewer wants an experienced candidate's answer, not a fresher-level explanation.\n"
            "- Do not start from absolute basics unless the question explicitly asks for them.\n"
            "- Focus on practical tradeoffs, decision criteria, implementation details, and risks.\n"
            "- Avoid long textbook definitions; keep it straight to the point.\n"
            "- When the user needs a full answer, use 2-3 sentences up front, then exactly 5 bullet points with the main takeaways.\n"
            "- Put SQL examples in fenced code blocks tagged sql and Python examples in fenced code blocks tagged python.\n"
            "- Keep SQL and Python examples in separate fenced code blocks so they stand out clearly.\n"
            "- For SQL or Python questions, prefer a practical query or script the user can adapt quickly."
        )
        prompt = f"{prompt}{human_style_policy}"

        follow_up_policy = (
            "\n\nFollow-up handling policy:\n"
            "- If [Prior Question Context] and/or [Follow-up Context] is "
            "provided, treat it as authoritative for resolving references "
            "like 'this', 'that', 'it', 'why', and 'example'.\n"
            "- If [Follow-up Type] is provided, adapt your response style "
            "accordingly:\n"
            "  • 'elaboration' → expand on the previous answer with more "
            "depth.\n"
            "  • 'example request' → provide a concrete, practical "
            "example. If the user asks for a sample script, sample code, "
            "Python script, SQL query, code example, or snippet, return a "
            "short generic example in a fenced code block when specifics are "
            "missing.\n"
            "  • 'clarification' → rephrase or simplify the previous "
            "answer.\n"
            "  • 'simplification' → re-explain in simpler, more "
            "accessible terms.\n"
            "  • 'challenge / pushback' → address the concern or "
            "counter-argument directly and honestly.\n"
            "  • 'scenario extension' → apply the concept to the "
            "described scenario.\n"
            "  • 'comparison' → clearly contrast the items, highlighting "
            "key differences.\n"
            "  • 'selection / preference' → compare both options with "
            "concise pros and cons for each, then give a clear "
            "recommendation with a brief rationale tied to likely "
            "interview expectations.\n"
            "  • 'pros and cons' → list advantages and disadvantages "
            "concisely.\n"
            "  • 'summarization' → provide a brief recap of the key "
            "points.\n"
            "  • 'reference back' → reconnect to the earlier point the "
            "speaker mentioned.\n"
            "  • 'continuation' → continue naturally from where the "
            "conversation left off.\n"
            "- Keep your answer focused on the current speaker question, "
            "while using prior context only to disambiguate what the "
            "follow-up refers to.\n"
            "- Do not ignore follow-up context unless it clearly conflicts "
            "with the current question."
        )
        prompt = f"{prompt}{follow_up_policy}"

        return prompt