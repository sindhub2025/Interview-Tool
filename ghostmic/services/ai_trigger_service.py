"""Policy service for controlling normalized-segment AI triggering."""

from __future__ import annotations

import re

_INTERROGATIVE_STARTERS = {
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "whom",
    "which",
    "can",
    "could",
    "would",
    "should",
    "do",
    "does",
    "did",
    "is",
    "are",
    "am",
    "will",
    "won't",
    "shall",
    "may",
    "might",
    "have",
    "has",
    "had",
}

_FALSE_POSITIVE_PREFIX_RE = re.compile(
    r"^(i think|i believe|we think|we believe|i know|we know|"
    r"there is|there are|i was|i am|i have|we have|we are|"
    r"that is|that was|it is|it was|this is|this was|"
    r"thank you|thanks|okay|ok |sure|right|yes|no |"
    r"i see|i understand|great|good|perfect|absolutely|exactly|"
    r"let me|i will|i would|we will|we would)\b",
    re.IGNORECASE,
)

# Imperative interview prompts — these are questions even without "?"
_IMPERATIVE_PROMPT_RE = re.compile(
    r"^(tell me|tell us|explain|describe|walk me through|walk us through|"
    r"give me|give us|give an|share|discuss|elaborate|outline|define|"
    r"compare|contrast|demonstrate|illustrate|list|mention|"
    r"talk about|talk me through|talk us through|show me|show us|"
    r"help me understand|let's talk about|let's discuss|"
    r"say you have|say we have|suppose|imagine|consider|"
    r"take me through|take us through|run me through|go through|"
    r"break down|walk through|step through)\b",
    re.IGNORECASE,
)

# Patterns that strongly indicate a question even mid-sentence
_EMBEDDED_QUESTION_RE = re.compile(
    r"\b(what is|what are|what was|what were|what do|what does|what did|"
    r"what would|what could|what should|what will|"
    r"how do|how does|how did|how would|how could|how can|how is|how are|"
    r"why do|why does|why did|why would|why is|why are|"
    r"where do|where does|where is|where are|"
    r"when do|when does|when is|when was|"
    r"who is|who are|who was|who does|"
    r"which one|which is|which are|"
    r"can you|could you|would you|should you|"
    r"do you|does it|did you|have you|has it|"
    r"is there|are there|is it|are they)\b",
    re.IGNORECASE,
)

# Declarative comparative prompts often appear in interview speech
# without an explicit interrogative starter (e.g. "The difference
# between DBMS and RDBMS").
_COMPARATIVE_PROMPT_RE = re.compile(
    r"\b(?:the\s+)?difference(?:s)?\s+between\s+.+?\s+and\s+\S",
    re.IGNORECASE,
)


def is_question_like(text: str) -> bool:
    """Return True when text resembles an interview question or prompt.

    Detects:
    - Sentences ending with ``?``
    - Interrogative sentence forms (what, how, why, …)
    - Imperative interview prompts (tell me about, explain, describe, …)
    - Embedded question patterns (what is the difference, how do you, …)
    - Declarative comparative prompts (the difference between X and Y)
    """
    cleaned = " ".join(str(text or "").split()).strip()
    if len(cleaned) < 8:
        return False

    # Explicit question mark — always a question
    if cleaned.endswith("?"):
        return True

    lowered = cleaned.lower()
    tokens = lowered.split()
    if len(tokens) < 3:
        return False

    # Imperative interview prompts (tell me about X, explain Y, …)
    # Checked early — these are always questions regardless of prefix
    if _IMPERATIVE_PROMPT_RE.match(lowered):
        return True

    # Embedded question patterns anywhere in the text
    # (e.g. "Okay how do you handle…", "So what is the difference…")
    # These override filler-word prefixes because the core is still a question.
    if len(tokens) >= 4 and _EMBEDDED_QUESTION_RE.search(lowered):
        return True

    # False positives: statements that start with answer/acknowledgement
    # words and contain no question markers at all
    if _FALSE_POSITIVE_PREFIX_RE.match(lowered):
        return False

    # Comparative prompts without explicit question words are still
    # interview questions when both sides of the comparison are present.
    if len(tokens) >= 5 and _COMPARATIVE_PROMPT_RE.search(lowered):
        return True

    # Standard interrogative starters
    if tokens[0] in _INTERROGATIVE_STARTERS:
        return True

    return False


class AITriggerService:
    """Tracks one-time auto-send policy for normalized segments."""

    def __init__(self) -> None:
        self._first_valid_question_sent = False

    def reset(self) -> None:
        """Reset per-recording auto-send state."""
        self._first_valid_question_sent = False

    def should_auto_send(self, text: str) -> bool:
        """Auto-send exactly one first valid question segment."""
        if self._first_valid_question_sent:
            return False
        if not is_question_like(text):
            return False
        self._first_valid_question_sent = True
        return True

    def mark_first_question_sent(self) -> None:
        """Mark that the first question has already been dispatched."""
        self._first_valid_question_sent = True

    @property
    def first_question_sent(self) -> bool:
        return self._first_valid_question_sent