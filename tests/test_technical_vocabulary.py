"""Representative domain-aware technical vocabulary coverage."""

import pytest

from ghostmic.services.technical_vocabulary import TechnicalVocabularyEngine, _BUILTIN_ROWS


@pytest.mark.parametrize("canonical, aliases, _category", _BUILTIN_ROWS)
def test_representative_aliases_normalize(canonical, aliases, _category):
    engine = TechnicalVocabularyEngine()
    alias = aliases[0] if aliases else canonical
    normalized, candidates = engine.normalize_text(f"we used {alias} in production")

    assert canonical in normalized
    assert candidates
    assert candidates[0].canonical_term == canonical


def test_catalog_has_at_least_100_representative_terms():
    assert len(_BUILTIN_ROWS) >= 100


def test_noisy_terms_have_supporting_evidence_and_source():
    engine = TechnicalVocabularyEngine()
    normalized, candidates = engine.normalize_text(
        "we used terra form with cube control and snow flow"
    )

    assert "Terraform" in normalized
    assert "kubectl" in normalized
    assert "Snowflake" in normalized
    assert all(candidate.source == "builtin" for candidate in candidates)
    assert all(candidate.evidence for candidate in candidates)


def test_resume_session_and_organization_sources_are_distinguished():
    engine = TechnicalVocabularyEngine(
        organization_terms={"AcmeStream": ("acme stream",)},
        resume_terms={"LegacyLake": ("legacy lake",)},
        session_terms=("Project Orion",),
    )

    assert engine.candidates("acme stream")[0].source == "organization"
    assert engine.candidates("legacy lake")[0].source == "resume"
    assert engine.candidates("project orion")[0].source == "session"


def test_low_confidence_candidates_do_not_replace_text():
    engine = TechnicalVocabularyEngine(
        replacement_threshold=0.98,
        suggestion_threshold=0.40,
    )
    normalized, candidates = engine.normalize_text("we used an internal platform")

    assert normalized == "we used an internal platform"
    assert all(candidate.score < 0.98 for candidate in candidates)


def test_topic_and_recent_term_evidence_can_raise_a_candidate():
    engine = TechnicalVocabularyEngine(replacement_threshold=0.85)
    normalized, candidates = engine.normalize_text(
        "how does snow flow work",
        current_topic="Snowflake warehouse",
        recent_terms=("Snowflake",),
    )

    assert normalized == "how does Snowflake work"
    candidate = next(item for item in candidates if item.canonical_term == "Snowflake")
    assert "current topic" in candidate.evidence
    assert "recent session term" in candidate.evidence


def test_engine_does_not_log_sensitive_input(caplog):
    engine = TechnicalVocabularyEngine(
        resume_terms={"PrivateCompany": ("private company",)},
    )
    engine.normalize_text("private company confidential project")

    assert "confidential project" not in " ".join(record.getMessage() for record in caplog.records)
    assert "private company" not in " ".join(record.getMessage() for record in caplog.records)
