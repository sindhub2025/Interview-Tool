"""Focused tests for conservative context-aware transcript normalization."""

from ghostmic.services.normalizer_service import (
    NormalizationContext,
    normalize_deterministically,
)
from ghostmic.services.question_normalization_service import _parse_normalization_result


def normalize(text: str, **kwargs):
    return normalize_deterministically(
        NormalizationContext(current_raw_transcript=text, **kwargs)
    )


def test_technical_homophones_and_aws_names_are_context_supported():
    result = normalize("how does elastic search run on a w s", known_technical_terms=("Elasticsearch",))

    assert result.normalized_text == "How does Elasticsearch run on AWS?"
    assert {item.normalized_term for item in result.corrections} == {"Elasticsearch", "AWS"}
    assert all(item.confidence == "HIGH" for item in result.corrections)


def test_database_kubernetes_and_programming_language_spoken_forms():
    result = normalize(
        "compare post gres with my sequel on cube net ease using py thon and type script"
    )

    assert "PostgreSQL" in result.normalized_text
    assert "MySQL" in result.normalized_text
    assert "Kubernetes" in result.normalized_text
    assert "Python" in result.normalized_text
    assert "TypeScript" in result.normalized_text


def test_acronyms_and_resume_aliases_are_applied_without_history_leakage():
    result = normalize(
        "did jane use count in the project",
        resume_profile_aliases={"COUNT()": ("count",), "Acme Analytics": ("acme",)},
        recent_conversation_turns=("old unrelated secret transcript",),
    )

    assert "COUNT()" in result.normalized_text
    assert "old unrelated secret" not in result.normalized_text
    assert any(item.context_source == "active profile terms" for item in result.corrections)


def test_follow_up_question_uses_previous_question_only_for_classification():
    result = normalize(
        "and what about retries",
        previous_normalized_interviewer_question="How do you handle Elasticsearch failures?",
        current_detected_topic="Elasticsearch",
    )

    assert result.is_question is True
    assert result.question_type == "follow_up"
    assert result.continuation_of_previous is True
    assert result.topic == "Elasticsearch"


def test_ambiguous_hallucination_bait_preserves_raw_meaning():
    result = normalize("we used a custom internal search tool")

    assert result.normalized_text == "We used a custom internal search tool"
    assert result.corrections == ()
    assert "OpenAI" not in result.normalized_text
    assert "Elasticsearch" not in result.normalized_text


def test_prompt_block_has_explicit_bounded_sections():
    block = NormalizationContext(
        current_raw_transcript="and what about it",
        previous_normalized_interviewer_question="How does AWS work?",
        previous_candidate_answer="We used EKS.",
        recent_conversation_turns=("speaker: How does AWS work?", "user: We used EKS."),
        known_technical_terms=("AWS", "EKS"),
        resume_profile_aliases={"EKS": ("e k s",)},
        screen_derived_context="A Kubernetes deployment is visible.",
    ).prompt_block()

    for heading in (
        "CURRENT AUDIO", "PREVIOUS QUESTION", "PREVIOUS ANSWER", "RECENT TURNS",
        "KNOWN TERMS", "ACTIVE PROFILE TERMS", "SCREEN CONTEXT", "NORMALIZATION RULES",
    ):
        assert heading in block
    assert "old unrelated" not in block


def test_strict_schema_parser_records_corrections_and_follow_up_metadata():
    result = _parse_normalization_result(
        '{"normalized_text":"What is Elasticsearch?","is_question":true,'
        '"question_type":"follow_up","topic":"Elasticsearch",'
        '"corrections":[{"raw_term":"elastic search","normalized_term":"Elasticsearch",'
        '"confidence":"HIGH","reason":"known technical term"}],'
        '"continuation_of_previous":true,"referenced_entities":["Elasticsearch"],'
        '"follow_up_questions":[]}',
        fallback_question="elastic search",
    )

    assert result.normalized_text == "What is Elasticsearch?"
    assert result.question_type == "follow_up"
    assert result.corrections[0]["confidence"] == "HIGH"
    assert result.continuation_of_previous is True
