"""Unit tests for AI trigger policy over normalized transcript segments."""

from ghostmic.services.ai_trigger_service import AITriggerService, is_question_like


def test_ai_trigger_auto_sends_only_first_valid_question() -> None:
    service = AITriggerService()

    assert service.should_auto_send("Can you explain your ETL retry strategy?") is True
    assert service.should_auto_send("How do you monitor this in production?") is False


def test_question_detection_prefers_interrogative_forms_and_avoids_common_false_positives() -> None:
    # Interrogative starters
    assert is_question_like("What trade-offs did you evaluate in production") is True
    assert is_question_like("Can you walk through your rollback plan?") is True
    assert is_question_like("How do you handle state management in React") is True
    assert is_question_like("Why did you choose Python over Java") is True
    assert is_question_like("Where do you deploy your services") is True

    # Explicit question mark
    assert is_question_like("This statement ends with a question mark?") is True

    # False positives — should NOT be detected as questions
    assert is_question_like("I think this is the right architecture") is False
    assert is_question_like("Thanks") is False
    assert is_question_like("Thank you for that explanation") is False
    assert is_question_like("I understand, that makes sense") is False
    assert is_question_like("Great, let's move on") is False
    assert is_question_like("I have experience with Python and Java") is False
    assert is_question_like("Yes, I have worked with databases") is False


def test_imperative_interview_prompts_detected_as_questions() -> None:
    """Imperative forms used in interviews should be treated as questions."""
    assert is_question_like("Tell me about your experience with Python") is True
    assert is_question_like("Explain the difference between REST and GraphQL") is True
    assert is_question_like("Describe your approach to debugging production issues") is True
    assert is_question_like("Walk me through your last project architecture") is True
    assert is_question_like("Give me an example of a time you led a team") is True
    assert is_question_like("Share your experience with cloud deployments") is True
    assert is_question_like("Discuss the advantages of microservices") is True
    assert is_question_like("Compare SQL and NoSQL databases") is True
    assert is_question_like("Talk about your testing strategy") is True


def test_embedded_question_patterns_detected() -> None:
    """Questions embedded after a preamble should still be detected."""
    assert is_question_like("So what is the difference between a list and a tuple") is True
    assert is_question_like("Okay how do you handle error logging in that system") is True
    assert is_question_like("And can you explain what happens during a failover") is True
    assert is_question_like("Now is there a way to optimize that further") is True


def test_comparative_prompt_detected_without_interrogative_prefix() -> None:
    """Comparative prompts like 'the difference between X and Y' should be treated as questions."""
    assert is_question_like("The difference between DBMS and RDBMS") is True
    assert is_question_like("The difference between DBMS and") is False
    assert is_question_like("I know the difference between DBMS and RDBMS") is False


def test_short_text_rejected() -> None:
    """Very short text should never match as a question."""
    assert is_question_like("What?") is False
    assert is_question_like("How") is False
    assert is_question_like("Tell") is False
    assert is_question_like("") is False