"""Deterministic semantic conversation-state tests."""

from ghostmic.services.conversation_state import (
    ConversationState,
    TURN_ANSWER,
    TURN_CLARIFICATION,
    TURN_FOLLOW_UP,
    TURN_QUESTION,
)


def add(state, speaker, text, **kwargs):
    return state.update_turn(speaker=speaker, raw_text=text, **kwargs)


def test_follow_up_reference_resolves_to_previous_question_topic():
    state = ConversationState()
    add(state, "speaker", "What is SCD Type 2?")
    add(state, "user", "SCD Type 2 keeps historical versions of rows.")
    turn = add(state, "speaker", "How would you implement that in Snowflake?")

    assert turn.turn_type == TURN_FOLLOW_UP
    assert turn.references == [{
        "reference": "that",
        "resolved_to": "SCD Type 2",
        "confidence": "HIGH",
    }]
    assert "SCD Type 2" in turn.topic
    assert "Snowflake" in turn.topic
    assert state.current_interviewer_question == turn.normalized_text


def test_what_about_production_is_follow_up():
    state = ConversationState()
    add(state, "speaker", "Explain Kubernetes probes.")
    add(state, "user", "They check application health.")
    turn = add(state, "speaker", "What about production?")

    assert turn.turn_type == TURN_FOLLOW_UP
    assert turn.topic == "Kubernetes probes"
    assert state.current_topic == "Kubernetes probes"


def test_topic_switch_replaces_active_topic_but_keeps_bounded_history():
    state = ConversationState(max_recent_turns=3, max_recent_topics=3)
    add(state, "speaker", "What is SQL?")
    add(state, "user", "SQL queries relational databases.")
    add(state, "speaker", "How do indexes work?")
    add(state, "user", "They speed up lookups.")
    add(state, "speaker", "Explain Kubernetes deployments.")

    assert state.current_topic == "Kubernetes deployments"
    assert len(state.recent_turns) == 3
    assert len(state.recent_topics) <= 3
    assert "SQL" in state.recent_topics


def test_repeated_short_questions_and_why_are_follow_ups():
    state = ConversationState()
    add(state, "speaker", "How do retries work?")
    why = add(state, "speaker", "Why?")
    example = add(state, "speaker", "Give me an example")

    assert why.turn_type == TURN_FOLLOW_UP
    assert example.turn_type == TURN_FOLLOW_UP
    assert why.references == []
    assert why.topic == "retries work"


def test_self_correction_is_stored_as_clarification_without_rewriting_raw_text():
    state = ConversationState()
    turn = add(state, "speaker", "Use Redis, I mean Kafka.")

    assert turn.turn_type == TURN_CLARIFICATION
    assert turn.raw_text == "Use Redis, I mean Kafka."
    assert turn.normalized_text == turn.raw_text


def test_candidate_and_interviewer_interruptions_remain_separate_turns():
    state = ConversationState()
    question = add(state, "speaker", "Can you explain indexes?")
    answer = add(state, "user", "Yes, they speed up lookups.")
    interruption = add(state, "speaker", "Sorry, what about composite indexes?")

    assert question.turn_type == TURN_QUESTION
    assert answer.turn_type == TURN_ANSWER
    assert interruption.turn_type in {TURN_FOLLOW_UP, TURN_CLARIFICATION}
    assert [turn.speaker for turn in state.recent_turns] == ["speaker", "user", "speaker"]


def test_explicit_metadata_and_context_fields_are_bounded():
    state = ConversationState(max_state_chars=1800, max_recent_entities=2)
    state.update_screen_context("screen " * 1000)
    state.update_resume_context({"role": "Data Engineer", "skills": ["SQL", "Python"]})
    for index in range(10):
        add(state, "speaker", f"What is Service{index}?")

    snapshot = state.snapshot()
    assert len(snapshot["current_screen_context"]) <= 1800
    assert len(snapshot["recent_conversation_turns"]) <= 12
    assert len(snapshot["known_entities"]) <= 2
    assert len(str(snapshot)) <= state.max_state_chars
