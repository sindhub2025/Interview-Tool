"""Regression cases that must remain semantically unchanged."""

import pytest

from ghostmic.services.correction_policy import (
    allow_automatic_correction,
    protected_semantics_changed,
)
from ghostmic.services.normalizer_service import NormalizationContext, normalize_deterministically


DO_NOT_CHANGE = [
    ("eight point one five", "8.5"),
    ("eight point one five", "8.15"),
    ("version 2.7", "version 2.07"),
    ("v1.2.3", "v1.3.2"),
    ("Python 3.11", "Python 3.1"),
    ("Java 17", "Java 7"),
    ("Node 20", "Node 2.0"),
    ("2024", "2025"),
    ("03/04/2024", "04/03/2024"),
    ("June 12, 2024", "June 21, 2024"),
    ("error 404", "error 400"),
    ("HTTP 500", "HTTP 503"),
    ("ECONNRESET", "ECONN REFUSED"),
    ("ValueError", "TypeError"),
    ("ERR_MODULE_NOT_FOUND", "ERR_NOT_FOUND"),
    ("/api/v1/users", "/api/v2/users"),
    ("https://example.com/api", "https://example.org/api"),
    ("api.example.com", "api.example.net"),
    ("AWS Lambda", "AWS Lamba"),
    ("Amazon S3", "Amazon EC2"),
    ("Project Orion", "Project Apollo"),
    ("Acme Corp", "Acme Corporation"),
    ("Product X", "Product Y"),
    ("customer-portal", "customer portal"),
    ("internal_api_v2", "internal_api_v3"),
    ("getUserById", "getUserByID"),
    ("UserID", "User ID"),
    ("snake_case", "camelCase"),
    ("SQL", "sql"),
    ("AWS", "aws"),
    ("S3", "s3"),
    ("EC2", "ec2"),
    ("EKS", "eks"),
    ("API", "api"),
    ("JVM", "jvm"),
    ("ETL", "etl"),
    ("ELK", "elk"),
    ("SRE", "sre"),
    ("database migration", "DB migration"),
    ("we used a custom search tool", "we used Elasticsearch"),
    ("the team called it Atlas", "the team called it Aurora"),
    ("latency was about eight milliseconds", "latency was about eighty milliseconds"),
    ("five replicas", "fifteen replicas"),
    ("port 5432", "port 5433"),
    ("timeout 30 seconds", "timeout 300 seconds"),
    ("memory limit 512Mi", "memory limit 512Gi"),
    ("CPU 250m", "CPU 2500m"),
    ("SCD Type 2", "SCD Type 3"),
    ("PostgreSQL 14", "PostgreSQL 15"),
    ("Terraform 1.6", "Terraform 1.16"),
    ("release candidate rc1", "release candidate rc2"),
    ("build #1842", "build #1482"),
    ("sha256:abc123", "sha256:def456"),
    ("tenant-001", "tenant-002"),
]


@pytest.mark.parametrize("raw, candidate", DO_NOT_CHANGE)
def test_protected_semantic_change_is_not_automatically_allowed(raw, candidate):
    assert protected_semantics_changed(raw, candidate) or raw.upper() != candidate.upper()
    assert allow_automatic_correction(
        raw,
        candidate,
        evidence=("normalized string similarity",),
        confidence="HIGH",
        score=0.91,
    ) is False


def test_numeric_phrase_is_preserved_by_deterministic_normalization():
    result = normalize_deterministically(
        NormalizationContext(current_raw_transcript="The value was eight point one five")
    )

    assert result.normalized_text == "The value was eight point one five"
    assert result.corrections == ()


def test_style_only_change_requires_explicit_evidence():
    assert allow_automatic_correction(
        "AWS", "aws", evidence=("normalized string similarity",), confidence="HIGH", score=1.0
    ) is False
    assert allow_automatic_correction(
        "a w s", "AWS", evidence=("exact alias", "technical vocabulary dictionary"), confidence="HIGH", score=1.0
    ) is True
