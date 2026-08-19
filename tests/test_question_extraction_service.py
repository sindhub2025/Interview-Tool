"""Regression tests for conversational interview-question extraction."""

from ghostmic.services.question_extraction_service import (
    MULTIPLE_QUESTIONS,
    NO_QUESTION,
    QUESTION_CANDIDATE,
    UNCERTAIN,
    QuestionExtractionService,
)


def _questions(text: str) -> list[str]:
    service = QuestionExtractionService()
    return [item.raw_question for item in service.extract_from_text(text)]


def test_clean_question_is_detected() -> None:
    assert _questions("Can you explain SCD Type 2?") == [
        "Can you explain SCD Type 2?"
    ]


def test_long_preamble_extracts_actual_question() -> None:
    questions = _questions(
        "Okay, let's talk about your previous project. You worked with Snowflake "
        "and Hive, right? There were some migration challenges. Can you explain "
        "how you validated the migrated data?"
    )

    assert questions == ["Can you explain how you validated the migrated data?"]


def test_contextual_clause_before_core_question_is_preserved() -> None:
    questions = _questions(
        "Okay Jaime, before we get into the technical part, you mentioned that "
        "you worked on a migration from Hive to Snowflake. There were probably "
        "a few problems during that process, right? And when you were validating "
        "the migrated data, can you explain how you actually handled reconciliation?"
    )

    assert questions == [
        "When you were validating the migrated data, can you explain how you actually handled reconciliation?"
    ]


def test_long_post_question_conversation_yields_both_questions() -> None:
    questions = _questions(
        "Can you explain how you validated the migrated data? And before we move "
        "on, tell me what tools you used."
    )

    assert questions == [
        "Can you explain how you validated the migrated data?",
        "Tell me what tools you used?",
    ]


def test_filler_is_not_a_question() -> None:
    service = QuestionExtractionService()
    assert service.classify("Okay, yeah, that's interesting... right... okay...") == NO_QUESTION
    assert service.extract_from_text("Okay, yeah, that's interesting... right... okay...") == []


def test_question_fragment_remains_uncertain() -> None:
    service = QuestionExtractionService()
    assert service.classify("How did you handle...") == UNCERTAIN
    result = service.extract_from_text("How did you handle...")
    assert result[0].classification == UNCERTAIN
    assert result[0].has_question is False


def test_split_question_can_be_extracted_after_chunks_are_joined() -> None:
    questions = _questions("Can you explain how you handled incremental loads in Snowflake?")

    assert questions == [
        "Can you explain how you handled incremental loads in Snowflake?"
    ]


def test_multiple_clause_question_preserves_subcomponents() -> None:
    questions = _questions(
        "What is SCD Type 2, how is it different from Type 1, and when would you use it?"
    )

    assert questions == [
        "What is SCD Type 2, how is it different from Type 1, and when would you use it?"
    ]


def test_stt_type_words_are_preserved_for_downstream_normalization() -> None:
    questions = _questions(
        "What is the difference between SCD type one and type two and when would you use type two?"
    )

    assert questions == [
        "What is the difference between SCD type one and type two and when would you use type two?"
    ]


def test_rdbms_question_is_not_misclassified() -> None:
    questions = _questions("How does an RDBMS differ from NoSQL?")

    assert questions == ["How does an RDBMS differ from NoSQL?"]


def test_context_contamination_is_not_introduced_by_extraction() -> None:
    questions = _questions("How do you validate the data?")

    assert questions == ["How do you validate the data?"]


def test_false_positive_good_question_is_rejected() -> None:
    service = QuestionExtractionService()
    assert service.classify("That's a good question.") == NO_QUESTION
    assert service.extract_from_text("That's a good question.") == []


def test_rhetorical_question_is_rejected() -> None:
    service = QuestionExtractionService()
    assert service.classify("Who hasn't seen a problem like that?") == NO_QUESTION
    assert service.extract_from_text("Who hasn't seen a problem like that?") == []


def test_multiple_question_classification() -> None:
    service = QuestionExtractionService()
    result = service.extract_from_text(
        "What is partition pruning? How would you verify it is working?"
    )

    assert len(result) == 2
    assert result[0].classification == MULTIPLE_QUESTIONS
    assert service.classify(
        "What is partition pruning? How would you verify it is working?"
    ) == MULTIPLE_QUESTIONS


def test_very_long_conversation_selects_relevant_question_only() -> None:
    transcript = (
        "Thanks for joining today. I looked through your resume and we can start "
        "with data engineering. You have a few projects listed and some of them "
        "sound pretty broad. The migration work is interesting, and we will come "
        "back to performance later. For now, when you reconciled source and target "
        "tables, how did you investigate mismatched row counts?"
    )

    assert _questions(transcript) == [
        "When you reconciled source and target tables, how did you investigate mismatched row counts?"
    ]


def test_single_question_classification() -> None:
    service = QuestionExtractionService()
    assert service.classify("Can you explain SCD Type 2?") == QUESTION_CANDIDATE
