"""Unit tests for ghostmic.services.resume_service."""

import os

import pytest

from ghostmic.services.resume_service import ResumeService


@pytest.fixture
def resume_service(tmp_path):
    return ResumeService(storage_root=str(tmp_path / "resume_store"))


def test_ingest_txt_resume_extracts_structured_profile(resume_service, tmp_path):
    resume_text = """
Jane Doe
jane.doe@example.com
+1 (555) 123-4567
Seattle, WA

Professional Summary
Data engineer with 7 years of ETL and analytics experience.

Experience
Microsoft - Senior Data Engineer | 2021 - Present
- Built Azure ETL pipelines in Python and SQL.
Contoso - Data Engineer | 2018 - 2021
- Developed data quality checks and Airflow DAGs.

Skills
Python, SQL, Azure, Airflow, Spark

Projects
Customer 360 Platform

Education
University of Washington - B.S. Computer Science | 2014 - 2018

Certifications
AWS Certified Solutions Architect
""".strip()

    resume_path = tmp_path / "resume.txt"
    resume_path.write_text(resume_text, encoding="utf-8")

    status = resume_service.ingest_resume(str(resume_path))
    profile = resume_service.get_profile()

    assert status["has_resume"] is True
    assert profile is not None
    assert profile["identity"]["full_name"] == "Jane Doe"
    assert profile["identity"]["email"] == "jane.doe@example.com"
    assert "Microsoft" in profile["companies"]
    assert "Senior Data Engineer" in profile["job_titles"]
    assert "Python" in profile["skills"]
    assert "AWS Certified Solutions Architect" in profile["certifications"]


def test_ingest_rejects_unsupported_file_type(resume_service, tmp_path):
    bad_file = tmp_path / "resume.exe"
    bad_file.write_text("binary-ish", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported resume format"):
        resume_service.ingest_resume(str(bad_file))


def test_ingest_rejects_oversized_file(tmp_path):
    service = ResumeService(
        storage_root=str(tmp_path / "resume_store"),
        max_file_size_bytes=32,
    )
    big_file = tmp_path / "resume.txt"
    big_file.write_text("A" * 2048, encoding="utf-8")

    with pytest.raises(ValueError, match="exceeds size limit"):
        service.ingest_resume(str(big_file))


def test_remove_resume_clears_profile_and_files(resume_service, tmp_path):
    resume_path = tmp_path / "resume.txt"
    resume_path.write_text("Jane Doe\nSkills\nPython, SQL", encoding="utf-8")
    resume_service.ingest_resume(str(resume_path))

    assert resume_service.has_resume() is True
    assert os.path.exists(resume_service.profile_path)

    resume_service.remove_resume()

    assert resume_service.has_resume() is False
    assert not os.path.exists(resume_service.profile_path)


def test_profile_persists_across_service_reloads(tmp_path):
    storage_root = tmp_path / "resume_store"
    service1 = ResumeService(storage_root=str(storage_root))

    resume_path = tmp_path / "resume.txt"
    resume_path.write_text(
        "Jane Doe\nExperience\nMicrosoft - Data Engineer | 2022 - Present\nSkills\nPython",
        encoding="utf-8",
    )
    service1.ingest_resume(str(resume_path))

    service2 = ResumeService(storage_root=str(storage_root))
    profile = service2.get_profile()

    assert profile is not None
    assert "Microsoft" in profile.get("companies", [])
