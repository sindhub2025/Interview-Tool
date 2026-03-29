"""Unit tests for AI context construction."""

from ghostmic.core.ai_engine import AIThread
from ghostmic.core.transcription_engine import TranscriptSegment


def _seg(text, source):
    return TranscriptSegment(text=text, source=source)


def test_build_context_uses_latest_speaker_turn_only():
    transcript = [
        _seg("What is the difference between rank and dense rank?", "speaker"),
        _seg("Rank skips numbers after ties.", "user"),
        _seg("How do you test late arriving data in ETL Testing?", "speaker"),
    ]

    context = AIThread._build_context(transcript)

    assert "rank and dense rank" not in context.lower()
    assert "late arriving data" in context.lower()
    assert context.strip().startswith("[Speaker]: How do you test late arriving data in ETL Testing?")


def test_build_context_falls_back_without_speaker_turns():
    transcript = [
        _seg("User note one", "user"),
        _seg("User note two", "user"),
    ]

    context = AIThread._build_context(transcript)

    assert "User note one" in context
    assert "User note two" in context


def test_build_context_includes_session_context_line():
    transcript = [_seg("How do you validate source and target table counts?", "speaker")]

    context = AIThread._build_context(
        transcript,
        session_context="ETL Tester with 8 years of Experience",
    )

    assert "[Session Context]: ETL Tester with 8 years of Experience" in context


def test_build_context_normalizes_sort_table_in_etl_context():
    transcript = [_seg("How do you compare sort and target table record counts?", "speaker")]

    context = AIThread._build_context(
        transcript,
        session_context="ETL Tester with 8 years of Experience",
    )

    assert "sort and target table" not in context.lower()
    assert "source and target table" in context.lower()


def test_follow_up_detector_matches_explicit_continuation_requests():
    assert AIThread.is_explicit_follow_up_request("Can you explain more on the topic?")
    assert AIThread.is_explicit_follow_up_request("Can you explain further on this topic?")
    assert AIThread.is_explicit_follow_up_request("Please elaborate.")
    assert AIThread.is_explicit_follow_up_request("Go deeper into this")
    assert AIThread.is_explicit_follow_up_request("Can you give an example?")
    assert AIThread.is_explicit_follow_up_request("What does that mean?")


def test_follow_up_detector_catches_short_context_dependent_requests():
    prev = "How do you validate ETL source and target table counts?"

    assert AIThread.is_context_dependent_follow_up(prev, "Can you give an example?")
    assert AIThread.is_context_dependent_follow_up(prev, "Why is that?")
    assert AIThread.is_context_dependent_follow_up(prev, "And what about this case?")
    assert AIThread.is_context_dependent_follow_up(prev, "Which one should I use?")

    assert not AIThread.is_context_dependent_follow_up(prev, "How do indexes improve query performance?")
    assert not AIThread.is_context_dependent_follow_up(
        prev,
        "Which algorithm should I use for graph traversal?",
    )


def test_follow_up_detector_rejects_new_topic_questions():
    assert not AIThread.is_explicit_follow_up_request(
        "How do you test late arriving data in ETL testing?"
    )
    assert not AIThread.is_explicit_follow_up_request(
        "What is difference between rank and dense rank?"
    )


def test_build_context_preserves_injected_follow_up_context():
    transcript = [
        _seg("Previous AI answer context: Focus on source and target table validation.", "user"),
        _seg("Can you explain further on this topic?", "speaker"),
    ]

    context = AIThread._build_context(
        transcript,
        session_context="ETL Tester with 8 years of Experience",
    )

    assert "[Follow-up Context]: Focus on source and target table validation." in context
    assert "source and target table validation" in context.lower()
    assert "can you explain further on this topic" in context.lower()


def test_build_context_prefers_latest_injected_follow_up_context():
    transcript = [
        _seg("Previous AI answer context: Old context should not be used.", "user"),
        _seg("Previous AI answer context: Use this latest context.", "user"),
        _seg("Can you explain more?", "speaker"),
    ]

    context = AIThread._build_context(transcript)

    assert "[Follow-up Context]: Use this latest context." in context
    assert "Old context should not be used" not in context


def test_build_context_includes_prior_question_context_for_follow_up():
    transcript = [
        _seg("What is the difference between rank and dense rank?", "speaker"),
        _seg("Rank does not skip tied positions while dense_rank does.", "user"),
        _seg("Can you give an example?", "speaker"),
    ]

    context = AIThread._build_context(transcript, is_new_topic=False)

    assert "[Prior Question Context]: What is the difference between rank and dense rank?" in context
    assert "[Speaker]: Can you give an example?" in context


# ---------------------------------------------------------------------------
# classify_topic_shift tests
# ---------------------------------------------------------------------------


def test_topic_shift_detects_completely_different_questions():
    prev = "What is the difference between rank and dense rank?"
    new = "How do you test late arriving data in ETL Testing?"
    assert AIThread.classify_topic_shift(prev, new) is True


def test_topic_shift_same_question_is_not_a_shift():
    q = "What is the difference between rank and dense rank?"
    # Identical input should never be treated as a topic shift.
    assert AIThread.classify_topic_shift(q, q) is False


def test_topic_shift_empty_prev_is_first_question():
    # No prior question → always treated as a new topic.
    assert AIThread.classify_topic_shift("", "What is a primary key?") is True


def test_topic_shift_empty_new_returns_false():
    # Empty new question → not enough info to judge.
    assert AIThread.classify_topic_shift("What is a primary key?", "") is False


def test_topic_shift_explicit_follow_up_is_not_a_shift():
    prev = "What is the difference between rank and dense rank?"
    # Explicit follow-up phrase must never be classified as a new topic.
    assert AIThread.classify_topic_shift(prev, "Can you explain more?") is False
    assert AIThread.classify_topic_shift(prev, "Please elaborate.") is False
    assert AIThread.classify_topic_shift(prev, "Tell me more.") is False


def test_topic_shift_context_dependent_follow_up_is_not_a_shift():
    prev = "What is the difference between rank and dense rank?"

    assert AIThread.classify_topic_shift(prev, "Can you give an example?") is False
    assert AIThread.classify_topic_shift(prev, "Why is that?") is False
    assert AIThread.classify_topic_shift(prev, "Which one should I use?") is False


def test_topic_shift_overlapping_etl_questions_are_same_topic():
    # Same question asked slightly differently — lots of word overlap.
    prev = "How do you validate ETL source and target table counts?"
    new = "How would you validate ETL source and target table record counts?"
    # Both share: validate, source, target, table, counts, etl → high overlap → not a shift.
    assert AIThread.classify_topic_shift(prev, new) is False


def test_build_context_suppresses_follow_up_block_on_new_topic():
    """When is_new_topic=True the [Follow-up Context] block must be omitted."""
    transcript = [
        _seg("Previous AI answer context: Rank skips numbers after ties.", "user"),
        _seg("How do you test late arriving data in ETL Testing?", "speaker"),
    ]

    context = AIThread._build_context(transcript, is_new_topic=True)

    assert "[Follow-up Context]" not in context
    assert "late arriving data" in context.lower()


def test_build_context_includes_follow_up_block_when_not_new_topic():
    """When is_new_topic=False (default) the [Follow-up Context] block is kept."""
    transcript = [
        _seg("Previous AI answer context: Rank skips numbers after ties.", "user"),
        _seg("Can you explain more?", "speaker"),
    ]

    context = AIThread._build_context(transcript, is_new_topic=False)

    assert "[Follow-up Context]" in context
    assert "rank skips numbers" in context.lower()


def test_build_context_includes_resume_context_for_resume_related_question():
    transcript = [_seg("Can you summarize your experience at Microsoft?", "speaker")]
    resume_profile = {
        "identity": {"full_name": "Jane Doe"},
        "companies": ["Microsoft"],
        "job_titles": ["Senior Data Engineer"],
        "skills": ["Python", "SQL"],
        "projects": ["Customer 360"],
        "certifications": ["AWS Certified Solutions Architect"],
        "tools": ["Azure Data Factory"],
        "technologies": ["Spark"],
        "aliases": {},
    }

    context = AIThread._build_context(
        transcript,
        resume_profile=resume_profile,
    )

    assert "[Resume Context]:" in context
    assert "Companies: Microsoft" in context
    assert "Job Titles: Senior Data Engineer" in context


def test_build_context_includes_sql_profile_for_sql_question():
    transcript = [_seg("How do COUNT and current timestamp work in SQL?", "speaker")]

    context = AIThread._build_context(
        transcript,
        sql_profile_enabled=True,
    )

    assert "[SQL Profile]:" in context
    assert "COUNT() - number of rows" in context
    assert "CURRENT_TIMESTAMP - current date-time" in context
    assert "COUNT()" in context


def test_build_system_prompt_includes_sql_policy_when_enabled():
    prompt = AIThread._build_system_prompt(
        "Base prompt",
        "",
        sql_profile_enabled=True,
    )

    assert "SQL profile usage policy" in prompt
    assert "canonical function name" in prompt
    assert "same normalization approach for other context-backed terms" in prompt


def test_build_context_applies_resume_grounded_high_confidence_correction():
    transcript = [_seg("I worked at Micro hard as a senior data engineer.", "speaker")]
    resume_profile = {
        "identity": {"full_name": "Jane Doe"},
        "companies": ["Microsoft"],
        "job_titles": ["Senior Data Engineer"],
        "skills": ["Python"],
        "projects": [],
        "certifications": [],
        "tools": [],
        "technologies": [],
        "aliases": {"Microsoft": ["micro soft"]},
    }

    context = AIThread._build_context(
        transcript,
        resume_profile=resume_profile,
    )

    assert "Microsoft" in context
    assert "[Resume Correction Applied]" in context


def test_build_context_does_not_force_resume_context_for_general_question():
    transcript = [_seg("What is CAP theorem in distributed systems?", "speaker")]
    resume_profile = {
        "identity": {"full_name": "Jane Doe"},
        "companies": ["Microsoft"],
        "job_titles": ["Senior Data Engineer"],
        "skills": ["Python"],
        "projects": [],
        "certifications": [],
        "tools": [],
        "technologies": [],
        "aliases": {},
    }

    context = AIThread._build_context(
        transcript,
        resume_profile=resume_profile,
    )

    assert "[Resume Context]:" not in context


def test_build_system_prompt_includes_resume_policy_when_profile_available():
    prompt = AIThread._build_system_prompt(
        "Base prompt",
        session_context="",
        resume_profile={"companies": ["Microsoft"]},
    )

    assert "Resume usage policy:" in prompt
    assert "source of truth" in prompt
