"""
Resume ingestion and structured profile persistence for GhostMic.

This service handles:
- Upload validation and safe local storage
- Text extraction from supported resume formats
- Heuristic structured extraction into a durable JSON profile
- Resume lifecycle operations (ingest/load/remove)
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
import dataclasses
from typing import Any, Dict, List, Optional, Sequence

from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)

RESUME_PROFILE_SCHEMA_VERSION = 1
DEFAULT_MAX_FILE_SIZE_BYTES = 8 * 1024 * 1024
SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".txt"})

_SECTION_ALIASES = {
    "summary": "summary",
    "professional summary": "summary",
    "profile": "summary",
    "objective": "summary",
    "about": "summary",
    "experience": "work_history",
    "work history": "work_history",
    "work experience": "work_history",
    "professional experience": "work_history",
    "employment history": "work_history",
    "career history": "work_history",
    "skills": "skills",
    "technical skills": "skills",
    "core skills": "skills",
    "core competencies": "skills",
    "key skills": "skills",
    "skills & tools": "skills",
    "skills and tools": "skills",
    "competencies": "skills",
    "expertise": "skills",
    "projects": "projects",
    "key projects": "projects",
    "education": "education",
    "academic background": "education",
    "certifications": "certifications",
    "certification": "certifications",
    "certifications & licenses": "certifications",
    "certifications and licenses": "certifications",
    "licenses": "certifications",
    "licenses and certifications": "certifications",
    "professional certifications": "certifications",
    "achievements": "achievements",
    "awards": "achievements",
    "tools": "tools",
    "technologies": "technologies",
    "tech stack": "technologies",
}

_TITLE_HINTS = (
    "engineer",
    "developer",
    "architect",
    "analyst",
    "manager",
    "consultant",
    "specialist",
    "lead",
    "tester",
    "scientist",
    "intern",
)

_STOP_WORDS = frozenset(
    {
        "the",
        "and",
        "with",
        "for",
        "that",
        "this",
        "from",
        "into",
        "using",
        "used",
        "have",
        "has",
        "had",
        "are",
        "was",
        "were",
        "your",
        "you",
        "our",
        "their",
        "about",
        "will",
        "can",
        "within",
        "across",
        "through",
        "over",
        "under",
        "years",
        "year",
        "months",
        "month",
        "team",
        "teams",
        "build",
        "built",
        "developed",
        "responsible",
        "experience",
    }
)

_DATE_RE = re.compile(
    r"(\b(?:19|20)\d{2}\b\s*(?:-|\u2013|to)\s*(?:present|current|now|\b(?:19|20)\d{2}\b))",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")

_SKILL_CANONICAL_BY_KEY = {
    "airflow": "Airflow",
    "ansible": "Ansible",
    "aspnet": "ASP.NET",
    "aws": "AWS",
    "azure": "Azure",
    "bash": "Bash",
    "bigquery": "BigQuery",
    "cicd": "CI/CD",
    "cpp": "C++",
    "csharp": "C#",
    "cplusplus": "C++",
    "databricks": "Databricks",
    "deeplearning": "Deep Learning",
    "django": "Django",
    "docker": "Docker",
    "dotnet": ".NET",
    "elt": "ELT",
    "etl": "ETL",
    "express": "Express",
    "fastapi": "FastAPI",
    "flask": "Flask",
    "gcp": "GCP",
    "githubactions": "GitHub Actions",
    "gitlabci": "GitLab CI",
    "go": "Go",
    "golang": "Go",
    "graphql": "GraphQL",
    "java": "Java",
    "javascript": "JavaScript",
    "jenkins": "Jenkins",
    "k8s": "Kubernetes",
    "kafka": "Kafka",
    "kubernetes": "Kubernetes",
    "linux": "Linux",
    "machinelearning": "Machine Learning",
    "microservices": "Microservices",
    "mongodb": "MongoDB",
    "mysql": "MySQL",
    "node": "Node.js",
    "nodejs": "Node.js",
    "nlp": "NLP",
    "nosql": "NoSQL",
    "numpy": "NumPy",
    "pandas": "Pandas",
    "php": "PHP",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "powershell": "PowerShell",
    "python": "Python",
    "python3": "Python",
    "pyspark": "PySpark",
    "pytorch": "PyTorch",
    "react": "React",
    "reactjs": "React",
    "redis": "Redis",
    "rest": "REST",
    "restapi": "REST APIs",
    "restapis": "REST APIs",
    "ruby": "Ruby",
    "rust": "Rust",
    "scikitlearn": "scikit-learn",
    "sklearn": "scikit-learn",
    "snowflake": "Snowflake",
    "spark": "Spark",
    "sql": "SQL",
    "sqlserver": "SQL Server",
    "mssql": "SQL Server",
    "swift": "Swift",
    "tensorflow": "TensorFlow",
    "terraform": "Terraform",
    "typescript": "TypeScript",
    "vue": "Vue",
}

_SKILL_REGEX_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Python", re.compile(r"\bpython(?:3)?\b", re.IGNORECASE)),
    ("Java", re.compile(r"\bjava\b", re.IGNORECASE)),
    ("JavaScript", re.compile(r"\bjavascript\b|\bjs\b", re.IGNORECASE)),
    ("TypeScript", re.compile(r"\btypescript\b", re.IGNORECASE)),
    ("C++", re.compile(r"(?<!\w)c\+\+(?!\w)|\bcpp\b", re.IGNORECASE)),
    ("C#", re.compile(r"(?<!\w)c#(?!\w)|\bcsharp\b", re.IGNORECASE)),
    (".NET", re.compile(r"\b\.net\b|\bdotnet\b", re.IGNORECASE)),
    ("ASP.NET", re.compile(r"\basp\.?net\b", re.IGNORECASE)),
    ("Go", re.compile(r"\bgolang\b", re.IGNORECASE)),
    ("Rust", re.compile(r"\brust\b", re.IGNORECASE)),
    ("Ruby", re.compile(r"\bruby\b", re.IGNORECASE)),
    ("PHP", re.compile(r"\bphp\b", re.IGNORECASE)),
    ("Swift", re.compile(r"\bswift\b", re.IGNORECASE)),
    ("SQL", re.compile(r"\bsql\b", re.IGNORECASE)),
    ("NoSQL", re.compile(r"\bnosql\b", re.IGNORECASE)),
    ("PostgreSQL", re.compile(r"\bpostgres(?:ql)?\b", re.IGNORECASE)),
    ("MySQL", re.compile(r"\bmysql\b", re.IGNORECASE)),
    ("SQL Server", re.compile(r"\bsql server\b|\bmssql\b", re.IGNORECASE)),
    ("MongoDB", re.compile(r"\bmongodb\b", re.IGNORECASE)),
    ("Redis", re.compile(r"\bredis\b", re.IGNORECASE)),
    ("Snowflake", re.compile(r"\bsnowflake\b", re.IGNORECASE)),
    ("BigQuery", re.compile(r"\bbigquery\b", re.IGNORECASE)),
    ("Databricks", re.compile(r"\bdatabricks\b", re.IGNORECASE)),
    ("Spark", re.compile(r"\bspark\b", re.IGNORECASE)),
    ("PySpark", re.compile(r"\bpyspark\b", re.IGNORECASE)),
    ("Airflow", re.compile(r"\bairflow\b", re.IGNORECASE)),
    ("Kafka", re.compile(r"\bkafka\b", re.IGNORECASE)),
    ("React", re.compile(r"\breact(?:\.?js)?\b", re.IGNORECASE)),
    ("Angular", re.compile(r"\bangular\b", re.IGNORECASE)),
    ("Vue", re.compile(r"\bvue(?:\.?js)?\b", re.IGNORECASE)),
    ("Node.js", re.compile(r"\bnode(?:\.?js)?\b", re.IGNORECASE)),
    ("Express", re.compile(r"\bexpress(?:\.?js)?\b", re.IGNORECASE)),
    ("Django", re.compile(r"\bdjango\b", re.IGNORECASE)),
    ("Flask", re.compile(r"\bflask\b", re.IGNORECASE)),
    ("FastAPI", re.compile(r"\bfastapi\b", re.IGNORECASE)),
    ("AWS", re.compile(r"\baws\b|\bamazon web services\b", re.IGNORECASE)),
    ("Azure", re.compile(r"\bazure\b", re.IGNORECASE)),
    ("GCP", re.compile(r"\bgcp\b|\bgoogle cloud\b", re.IGNORECASE)),
    ("Docker", re.compile(r"\bdocker\b", re.IGNORECASE)),
    ("Kubernetes", re.compile(r"\bkubernetes\b|\bk8s\b", re.IGNORECASE)),
    ("Terraform", re.compile(r"\bterraform\b", re.IGNORECASE)),
    ("Ansible", re.compile(r"\bansible\b", re.IGNORECASE)),
    ("Jenkins", re.compile(r"\bjenkins\b", re.IGNORECASE)),
    ("GitHub Actions", re.compile(r"\bgithub actions\b", re.IGNORECASE)),
    ("GitLab CI", re.compile(r"\bgitlab ci\b", re.IGNORECASE)),
    ("CI/CD", re.compile(r"\bci\s*/\s*cd\b|\bcicd\b", re.IGNORECASE)),
    ("Git", re.compile(r"\bgit\b", re.IGNORECASE)),
    ("Linux", re.compile(r"\blinux\b", re.IGNORECASE)),
    ("Bash", re.compile(r"\bbash\b|\bshell scripting\b", re.IGNORECASE)),
    ("PowerShell", re.compile(r"\bpowershell\b", re.IGNORECASE)),
    ("REST APIs", re.compile(r"\brest(?:ful)?\s+api(?:s)?\b", re.IGNORECASE)),
    ("GraphQL", re.compile(r"\bgraphql\b", re.IGNORECASE)),
    ("Microservices", re.compile(r"\bmicroservices?\b", re.IGNORECASE)),
    ("ETL", re.compile(r"\betl\b", re.IGNORECASE)),
    ("ELT", re.compile(r"\belt\b", re.IGNORECASE)),
    ("Machine Learning", re.compile(r"\bmachine learning\b", re.IGNORECASE)),
    ("Deep Learning", re.compile(r"\bdeep learning\b", re.IGNORECASE)),
    ("NLP", re.compile(r"\bnlp\b|\bnatural language processing\b", re.IGNORECASE)),
    ("Pandas", re.compile(r"\bpandas\b", re.IGNORECASE)),
    ("NumPy", re.compile(r"\bnumpy\b", re.IGNORECASE)),
    ("TensorFlow", re.compile(r"\btensorflow\b", re.IGNORECASE)),
    ("PyTorch", re.compile(r"\bpytorch\b", re.IGNORECASE)),
    ("scikit-learn", re.compile(r"\bscikit[- ]learn\b|\bsklearn\b", re.IGNORECASE)),
)

_SOFT_SKILL_KEYS = frozenset(
    {
        "adaptability",
        "collaboration",
        "communication",
        "communications",
        "detailoriented",
        "leadership",
        "problemsolving",
        "selfmotivated",
        "teamwork",
        "timemanagement",
    }
)

_SKILL_CONTEXT_CUES_RE = re.compile(
    r"\b(?:using|with|in|on|across|including|include|proficient\s+in|experienced\s+with|expertise\s+in)\b\s+([^.;:]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ResumeStatus:
    """Presentation-friendly status snapshot for UI and runtime logic."""

    has_resume: bool
    source_file_name: str
    uploaded_at: float
    updated_at: float
    skills_count: int
    companies_count: int
    projects_count: int
    certifications_count: int

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class ResumeService:
    """Ingests resumes and persists a structured profile in local storage."""

    def __init__(
        self,
        storage_root: Optional[str] = None,
        allowed_extensions: Optional[Sequence[str]] = None,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    ) -> None:
        root = storage_root or os.path.join(os.path.expanduser("~"), ".ghostmic", "resume")
        self._storage_root = root
        self._uploads_dir = os.path.join(root, "uploads")
        self._profile_path = os.path.join(root, "profile.json")
        self._raw_text_path = os.path.join(root, "raw_text.txt")

        allowed = allowed_extensions or sorted(SUPPORTED_EXTENSIONS)
        self._allowed_extensions = frozenset(ext.lower() for ext in allowed)
        self._max_file_size_bytes = max(1, int(max_file_size_bytes))
        self._profile_cache: Optional[Dict[str, Any]] = None

        self._load_cached_profile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def profile_path(self) -> str:
        return self._profile_path

    def has_resume(self) -> bool:
        return self._profile_cache is not None

    def get_profile(self) -> Optional[Dict[str, Any]]:
        if self._profile_cache is None:
            return None
        return copy.deepcopy(self._profile_cache)

    def get_status(self) -> Dict[str, Any]:
        profile = self._profile_cache
        if profile is None:
            return ResumeStatus(
                has_resume=False,
                source_file_name="",
                uploaded_at=0.0,
                updated_at=0.0,
                skills_count=0,
                companies_count=0,
                projects_count=0,
                certifications_count=0,
            ).to_dict()

        meta = profile.get("meta", {}) if isinstance(profile, dict) else {}
        status = ResumeStatus(
            has_resume=True,
            source_file_name=str(meta.get("source_file_name", "")),
            uploaded_at=float(meta.get("uploaded_at", 0.0) or 0.0),
            updated_at=float(meta.get("updated_at", 0.0) or 0.0),
            skills_count=len(profile.get("skills", []) or []),
            companies_count=len(profile.get("companies", []) or []),
            projects_count=len(profile.get("projects", []) or []),
            certifications_count=len(profile.get("certifications", []) or []),
        )
        return status.to_dict()

    def ingest_resume(self, file_path: str) -> Dict[str, Any]:
        """Validate, extract, structure, and persist an uploaded resume."""
        source_path = os.path.abspath(file_path)
        if not os.path.isfile(source_path):
            raise ValueError("Resume file does not exist.")

        extension = os.path.splitext(source_path)[1].lower()
        if extension not in self._allowed_extensions:
            allowed = ", ".join(sorted(self._allowed_extensions))
            raise ValueError(f"Unsupported resume format: {extension}. Allowed: {allowed}")

        size = os.path.getsize(source_path)
        if size <= 0:
            raise ValueError("Resume file is empty.")
        if size > self._max_file_size_bytes:
            max_mb = self._max_file_size_bytes / (1024 * 1024)
            raise ValueError(f"Resume file exceeds size limit ({max_mb:.1f} MB).")

        os.makedirs(self._uploads_dir, exist_ok=True)
        safe_name = self._sanitize_filename(os.path.basename(source_path))
        stored_name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{safe_name}"
        stored_path = os.path.join(self._uploads_dir, stored_name)
        shutil.copy2(source_path, stored_path)

        try:
            extracted_text = self._extract_text(stored_path, extension)
            normalized_text = self._normalize_text(extracted_text)
            if len(normalized_text) < 20:
                raise ValueError("Resume text extraction produced insufficient content.")

            profile = self._build_profile(
                normalized_text,
                source_file_name=os.path.basename(source_path),
                stored_file_name=stored_name,
            )

            self._write_profile(profile)
            self._write_raw_text(normalized_text)
            self._profile_cache = profile
        except Exception:
            try:
                if os.path.exists(stored_path):
                    os.remove(stored_path)
            except OSError:
                pass
            raise

        logger.info(
            "ResumeService: resume ingested. source=%s, skills=%d, companies=%d",
            profile.get("meta", {}).get("source_file_name", ""),
            len(profile.get("skills", []) or []),
            len(profile.get("companies", []) or []),
        )
        return self.get_status()

    def remove_resume(self) -> None:
        """Remove persisted resume profile and extracted text from local storage."""
        for path in (self._profile_path, self._raw_text_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as exc:
                logger.warning("ResumeService: could not delete %s: %s", path, exc)

        if os.path.isdir(self._uploads_dir):
            for name in os.listdir(self._uploads_dir):
                path = os.path.join(self._uploads_dir, name)
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                except OSError as exc:
                    logger.warning("ResumeService: could not delete %s: %s", path, exc)

        self._profile_cache = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_cached_profile(self) -> None:
        if not os.path.exists(self._profile_path):
            self._profile_cache = None
            return

        try:
            with open(self._profile_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self._profile_cache = data
            else:
                self._profile_cache = None
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("ResumeService: could not load profile: %s", exc)
            self._profile_cache = None

    def _write_profile(self, profile: Dict[str, Any]) -> None:
        os.makedirs(self._storage_root, exist_ok=True)
        with open(self._profile_path, "w", encoding="utf-8") as fh:
            json.dump(profile, fh, indent=2, ensure_ascii=False)

    def _write_raw_text(self, raw_text: str) -> None:
        os.makedirs(self._storage_root, exist_ok=True)
        with open(self._raw_text_path, "w", encoding="utf-8") as fh:
            fh.write(raw_text)

    # ------------------------------------------------------------------
    # Extraction pipeline
    # ------------------------------------------------------------------

    def _extract_text(self, path: str, extension: str) -> str:
        if extension == ".txt":
            return self._extract_txt(path)
        if extension == ".pdf":
            return self._extract_pdf(path)
        if extension == ".docx":
            return self._extract_docx(path)
        raise ValueError(f"Unsupported extension: {extension}")

    @staticmethod
    def _extract_txt(path: str) -> str:
        for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
            try:
                with open(path, "r", encoding=encoding) as fh:
                    return fh.read()
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode text resume with supported encodings.")

    @staticmethod
    def _extract_pdf(path: str) -> str:
        try:
            from pypdf import PdfReader  # type: ignore[import]
        except ImportError as exc:
            raise ValueError("PDF parsing dependency missing: install pypdf.") from exc

        reader = PdfReader(path)
        parts: List[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text:
                parts.append(page_text)
        return "\n".join(parts)

    @staticmethod
    def _extract_docx(path: str) -> str:
        try:
            from docx import Document  # type: ignore[import]
        except ImportError as exc:
            raise ValueError("DOCX parsing dependency missing: install python-docx.") from exc

        document = Document(path)
        parts = [p.text for p in document.paragraphs if p.text and p.text.strip()]
        return "\n".join(parts)

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        return "\n".join(lines)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
        if not sanitized:
            return "resume.txt"
        return sanitized

    # ------------------------------------------------------------------
    # Structured profile extraction
    # ------------------------------------------------------------------

    def _build_profile(
        self,
        normalized_text: str,
        source_file_name: str,
        stored_file_name: str,
    ) -> Dict[str, Any]:
        lines = [line.strip() for line in normalized_text.split("\n") if line.strip()]
        sections = self._split_sections(lines)
        identity = self._extract_identity(lines)

        summary = " ".join(sections.get("summary", [])[:3]).strip()
        tools = self._extract_list_items(sections.get("tools", []))
        technologies = self._extract_list_items(sections.get("technologies", []))
        projects = self._extract_list_items(sections.get("projects", []))
        certifications = self._extract_list_items(sections.get("certifications", []))
        achievements = self._extract_list_items(sections.get("achievements", []))

        work_history = self._extract_work_history(sections.get("work_history", []))
        education = self._extract_education(sections.get("education", []))

        skills = self._extract_relevant_skills(
            normalized_text=normalized_text,
            sections=sections,
            work_history=work_history,
            tools=tools,
            technologies=technologies,
        )

        companies = self._dedupe(
            [entry.get("company", "") for entry in work_history if entry.get("company")]
        )
        job_titles = self._dedupe(
            [entry.get("title", "") for entry in work_history if entry.get("title")]
        )

        notable_facts = self._extract_notable_facts(work_history, achievements)

        keywords = self._extract_keywords(
            normalized_text,
            seed_terms=skills + tools + technologies + projects + companies + job_titles,
        )

        aliases = self._build_aliases(
            companies + job_titles + skills + tools + technologies + projects + certifications
        )

        now = time.time()
        profile: Dict[str, Any] = {
            "schema_version": RESUME_PROFILE_SCHEMA_VERSION,
            "meta": {
                "source_file_name": source_file_name,
                "stored_file_name": stored_file_name,
                "uploaded_at": now,
                "updated_at": now,
                "raw_text_chars": len(normalized_text),
            },
            "identity": identity,
            "summary": summary,
            "work_history": work_history,
            "education": education,
            "skills": skills,
            "projects": projects,
            "certifications": certifications,
            "achievements": achievements,
            "companies": companies,
            "job_titles": job_titles,
            "tools": tools,
            "technologies": technologies,
            "keywords": keywords,
            "notable_facts": notable_facts,
            "aliases": aliases,
        }
        return profile

    def _split_sections(self, lines: Sequence[str]) -> Dict[str, List[str]]:
        sections: Dict[str, List[str]] = {"general": []}
        current = "general"

        for line in lines:
            heading, inline_content = self._match_section_heading(line)
            if heading:
                current = heading
                sections.setdefault(current, [])
                if inline_content:
                    sections[current].append(inline_content)
                continue
            sections.setdefault(current, []).append(line)

        return sections

    def _match_section_heading(self, line: str) -> tuple[Optional[str], str]:
        stripped = line.strip()
        normalized = self._normalize_heading(stripped)

        for alias, section_name in _SECTION_ALIASES.items():
            if normalized == alias:
                return section_name, ""

            match = re.match(
                rf"^\s*{re.escape(alias)}\s*[:\-|\u2013\u2014|]\s*(.+?)\s*$",
                stripped,
                flags=re.IGNORECASE,
            )
            if match:
                return section_name, match.group(1).strip()

        return None, ""

    @staticmethod
    def _normalize_heading(line: str) -> str:
        cleaned = line.strip().strip(":").lower()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    def _extract_identity(self, lines: Sequence[str]) -> Dict[str, str]:
        email = ""
        phone = ""
        location = ""
        full_name = ""

        for line in lines[:20]:
            if not email:
                match = _EMAIL_RE.search(line)
                if match:
                    email = match.group(0)
            if not phone:
                match = _PHONE_RE.search(line)
                if match:
                    phone = match.group(0).strip()

        for line in lines[:8]:
            if self._looks_like_name(line):
                full_name = line
                break

        # Narrow down location detection to avoid false positives from headings
        # or company lines. Require a city/state/country-like pattern.
        section_keywords = {
            "skills",
            "experience",
            "summary",
            "education",
            "projects",
            "certifications",
            "contact",
            "profile",
            "objective",
            "about",
            "achievements",
            "awards",
            "tools",
            "technologies",
            "work",
            "company",
        }

        for line in lines[:15]:
            if location:
                break

            # Skip obvious non-location lines
            if _EMAIL_RE.search(line) or _PHONE_RE.search(line):
                continue
            candidate = line.strip()
            if not candidate:
                continue
            # Skip headings and labeled lines
            if ":" in candidate:
                continue
            low = candidate.lower()
            if any(kw in low for kw in section_keywords):
                continue
            # Skip bullets or numbered lists
            if candidate.startswith(("-", "*", "\u2022")) or re.match(r"^\d+[\.)\s]", candidate):
                continue
            if len(candidate) > 80:
                continue

            # Require a comma-separated form like 'City, State' or 'City, Country'
            if "," not in candidate:
                continue
            parts = [p.strip() for p in candidate.split(",") if p.strip()]
            if len(parts) < 2:
                continue
            # First token should look like a city/name
            if not re.search(r"[A-Za-z]", parts[0]):
                continue

            last = parts[-1]
            # Accept two-letter state codes or longer state/country names
            if re.fullmatch(r"[A-Za-z]{2}", last) or re.fullmatch(r"[A-Za-z][A-Za-z .-]{2,}$", last):
                # avoid company-like suffixes
                if re.search(r"\b(?:llc|inc|ltd|co|corp|company|technologies|systems)\b", last.lower()):
                    continue
                location = candidate
                break

        return {
            "full_name": full_name,
            "email": email,
            "phone": phone,
            "location": location,
        }

    @staticmethod
    def _looks_like_name(line: str) -> bool:
        if _EMAIL_RE.search(line) or _PHONE_RE.search(line):
            return False
        line = line.strip()
        if len(line.split()) < 2 or len(line.split()) > 5:
            return False
        return bool(re.fullmatch(r"[A-Za-z][A-Za-z'`.-]+(?:\s+[A-Za-z][A-Za-z'`.-]+){1,4}", line))

    def _extract_work_history(self, lines: Sequence[str]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None

        for raw in lines:
            line = raw.strip()
            if not line:
                continue

            bullet = self._strip_bullet(line)
            if bullet is not None:
                if current is not None:
                    current.setdefault("highlights", []).append(bullet)
                continue

            parsed = self._parse_work_entry_header(line)
            if parsed is not None:
                if current is not None:
                    entries.append(current)
                current = parsed
                continue

            if current is not None:
                current.setdefault("highlights", []).append(line)

        if current is not None:
            entries.append(current)

        return entries

    def _extract_education(self, lines: Sequence[str]) -> List[Dict[str, str]]:
        entries: List[Dict[str, str]] = []
        for line in lines:
            item = line.strip()
            if not item:
                continue
            stripped = self._strip_bullet(item)
            if stripped is not None:
                item = stripped or ""

            dates = self._extract_date_span(item)
            without_dates = item
            if dates:
                without_dates = without_dates.replace(dates, "").strip(" ,-|")

            degree = ""
            institution = without_dates
            if " - " in without_dates:
                left, right = without_dates.split(" - ", 1)
                if self._looks_like_degree(left):
                    degree = left.strip()
                    institution = right.strip()
                elif self._looks_like_degree(right):
                    degree = right.strip()
                    institution = left.strip()

            entries.append(
                {
                    "institution": institution,
                    "degree": degree,
                    "dates": dates,
                }
            )
        return entries

    def _parse_work_entry_header(self, line: str) -> Optional[Dict[str, Any]]:
        dates = self._extract_date_span(line)
        core = line
        if dates:
            core = core.replace(dates, "").strip(" ,-|")

        separators = [" - ", " | ", " at "]
        left = core
        right = ""
        for sep in separators:
            if sep in core:
                left, right = core.split(sep, 1)
                break

        left = left.strip()
        right = right.strip()

        if not left and not right:
            return None

        company = ""
        title = ""

        if " at " in core.lower():
            pieces = re.split(r"\bat\b", core, maxsplit=1, flags=re.IGNORECASE)
            if len(pieces) == 2:
                title = pieces[0].strip(" ,-|")
                company = pieces[1].strip(" ,-|")
        elif right:
            if self._looks_like_title(left) and not self._looks_like_title(right):
                title = left
                company = right
            elif self._looks_like_title(right) and not self._looks_like_title(left):
                title = right
                company = left
            else:
                company = left
                title = right
        else:
            if self._looks_like_title(left):
                title = left
            else:
                company = left

        if not company and not title:
            return None

        return {
            "company": company,
            "title": title,
            "dates": dates,
            "highlights": [],
        }

    @staticmethod
    def _extract_date_span(text: str) -> str:
        match = _DATE_RE.search(text)
        if match:
            return match.group(1)
        return ""

    @staticmethod
    def _looks_like_title(text: str) -> bool:
        lowered = text.lower()
        return any(hint in lowered for hint in _TITLE_HINTS)

    @staticmethod
    def _looks_like_degree(text: str) -> bool:
        lowered = text.lower()
        return any(
            token in lowered
            for token in ("b.s", "bsc", "m.s", "msc", "phd", "mba", "bachelor", "master")
        )

    @staticmethod
    def _strip_bullet(line: str) -> Optional[str]:
        match = re.match(r"^(?:[-*\u2022]+)\s+(.*)$", line)
        if not match:
            return None
        return match.group(1).strip()

    def _extract_list_items(self, lines: Sequence[str]) -> List[str]:
        items: List[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            bullet = self._strip_bullet(stripped)
            if bullet is not None:
                stripped = bullet

            pieces = re.split(r"\s*[|,;]\s*|\s+/\s+", stripped)
            for piece in pieces:
                value = piece.strip()
                if not value:
                    continue
                if len(value) > 120:
                    continue
                items.append(value)

        return self._dedupe(items)

    def _extract_relevant_skills(
        self,
        normalized_text: str,
        sections: Dict[str, List[str]],
        work_history: Sequence[Dict[str, Any]],
        tools: Sequence[str],
        technologies: Sequence[str],
    ) -> List[str]:
        explicit_candidates: List[str] = []
        for section_name in ("skills", "tools", "technologies"):
            for line in sections.get(section_name, []):
                explicit_candidates.extend(self._extract_skill_candidates_from_line(line))

        explicit_candidates.extend(tools)
        explicit_candidates.extend(technologies)

        explicit_skills: List[str] = []
        for candidate in explicit_candidates:
            canonical = self._canonicalize_skill(candidate)
            if canonical and self._is_relevant_skill_candidate(canonical):
                explicit_skills.append(canonical)
        explicit_skills = self._dedupe(explicit_skills)

        # When explicit skills are rich enough, prefer precision over aggressive inference.
        if len(explicit_skills) >= 5:
            return explicit_skills[:50]

        scored_inferred: Dict[str, float] = {}

        for skill in self._extract_known_skills(normalized_text):
            scored_inferred[skill] = scored_inferred.get(skill, 0.0) + 2.0

        context_lines: List[str] = []
        context_lines.extend(sections.get("summary", [])[:4])
        context_lines.extend(sections.get("projects", [])[:6])
        context_lines.extend(sections.get("general", [])[:6])
        for entry in work_history:
            title = str(entry.get("title", "")).strip()
            if title:
                context_lines.append(title)
            highlights = entry.get("highlights", [])
            if isinstance(highlights, list):
                for item in highlights[:4]:
                    text = str(item).strip()
                    if text:
                        context_lines.append(text)

        for line in context_lines:
            for candidate in self._extract_skill_candidates_from_line(line):
                canonical = self._canonicalize_skill(candidate)
                if not canonical or not self._is_relevant_skill_candidate(canonical):
                    continue
                scored_inferred[canonical] = scored_inferred.get(canonical, 0.0) + 1.0

        explicit_lookup = {item.lower() for item in explicit_skills}
        inferred_ranked = [
            skill
            for skill, _score in sorted(
                scored_inferred.items(),
                key=lambda item: (-item[1], item[0].lower()),
            )
            if skill.lower() not in explicit_lookup
        ]

        return (explicit_skills + inferred_ranked)[:50]

    def _extract_skill_candidates_from_line(self, line: str) -> List[str]:
        text = line.strip()
        if not text:
            return []

        bullet = self._strip_bullet(text)
        if bullet is not None:
            text = bullet

        heading_match = re.match(
            r"^\s*(?:skills?|technical skills?|core skills?|tools?|technologies|tech stack|competencies|expertise)\s*[:\-]\s*(.+?)\s*$",
            text,
            flags=re.IGNORECASE,
        )
        if heading_match:
            text = heading_match.group(1).strip()

        candidates: List[str] = []
        pieces = re.split(r"\s*[|,;]\s*", text)
        if len(pieces) == 1:
            cue_matches = list(_SKILL_CONTEXT_CUES_RE.finditer(text))
            if cue_matches:
                pieces = [match.group(1).strip() for match in cue_matches]
            else:
                pieces = [text]

        for piece in pieces:
            cleaned_piece = self._clean_skill_fragment(piece)
            if not cleaned_piece:
                continue

            subpieces = re.split(r"\s+(?:and|or)\s+|\s*&\s*", cleaned_piece, flags=re.IGNORECASE)
            if len(subpieces) == 1:
                candidates.append(cleaned_piece)
                continue

            for subpiece in subpieces:
                cleaned_subpiece = self._clean_skill_fragment(subpiece)
                if cleaned_subpiece:
                    candidates.append(cleaned_subpiece)

        candidates.extend(self._extract_known_skills(text))
        return self._dedupe(candidates)

    @staticmethod
    def _extract_known_skills(text: str) -> List[str]:
        found: List[tuple[int, str]] = []
        for canonical, pattern in _SKILL_REGEX_PATTERNS:
            for match in pattern.finditer(text):
                found.append((match.start(), canonical))

        found.sort(key=lambda item: item[0])
        return ResumeService._dedupe([skill for _pos, skill in found])

    def _canonicalize_skill(self, value: str) -> str:
        cleaned = self._clean_skill_fragment(value)
        if not cleaned:
            return ""

        key = self._normalize_skill_key(cleaned)
        if not key:
            return ""

        canonical = _SKILL_CANONICAL_BY_KEY.get(key, cleaned)
        return re.sub(r"\s+", " ", canonical).strip()

    @staticmethod
    def _normalize_skill_key(value: str) -> str:
        lowered = value.lower().strip()
        lowered = lowered.replace("c#", "csharp")
        lowered = lowered.replace("f#", "fsharp")
        lowered = lowered.replace("c++", "cplusplus")
        lowered = lowered.replace(".net", "dotnet")
        return re.sub(r"[^a-z0-9]", "", lowered)

    @staticmethod
    def _clean_skill_fragment(value: str) -> str:
        cleaned = value.strip().strip(".,:;|-()[]{}")
        if not cleaned:
            return ""

        cleaned = re.sub(r"^(?:and|or)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            (
                r"^(?:proficient|experienced|hands-on|strong|expert|expertise|"
                r"skilled|familiar|knowledgeable)(?:\s+(?:in|with|on|across))?\s+"
            ),
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\b(?:skills?|tools?|technologies|tech stack|frameworks?)\b$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,;:-")
        return cleaned

    @staticmethod
    def _is_relevant_skill_candidate(value: str) -> bool:
        candidate = value.strip()
        if not candidate:
            return False
        if len(candidate) < 2 or len(candidate) > 64:
            return False
        if _EMAIL_RE.search(candidate) or _PHONE_RE.search(candidate) or _DATE_RE.search(candidate):
            return False
        if re.search(r"[!?]", candidate):
            return False

        key = ResumeService._normalize_skill_key(candidate)
        if not key:
            return False
        if key in _SOFT_SKILL_KEYS:
            return False
        if key in _SKILL_CANONICAL_BY_KEY:
            return True

        words = re.findall(r"[A-Za-z0-9+#./-]+", candidate)
        if not words or len(words) > 4:
            return False

        lowered = candidate.lower()
        if any(
            verb in lowered
            for verb in ("built", "developed", "responsible", "managed", "designed", "led")
        ):
            return False

        if len(words) >= 3 and lowered == candidate:
            return False

        stop_count = sum(1 for token in re.findall(r"[a-z]+", lowered) if token in _STOP_WORDS)
        if stop_count >= 2:
            return False

        return True

    @staticmethod
    def _extract_notable_facts(
        work_history: Sequence[Dict[str, Any]],
        achievements: Sequence[str],
    ) -> List[str]:
        facts: List[str] = []
        for entry in work_history:
            for bullet in entry.get("highlights", [])[:2]:
                if bullet:
                    facts.append(bullet)
        facts.extend([a for a in achievements if a])
        return ResumeService._dedupe(facts)[:20]

    def _extract_keywords(self, text: str, seed_terms: Sequence[str]) -> List[str]:
        seed = [term for term in seed_terms if term and len(term) >= 3]

        tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#.-]{2,}", text)
        freq: Dict[str, int] = {}
        for token in tokens:
            lowered = token.lower()
            if lowered in _STOP_WORDS:
                continue
            freq[lowered] = freq.get(lowered, 0) + 1

        ranked = sorted(freq.items(), key=lambda item: (-item[1], item[0]))
        auto_keywords = [word for word, _ in ranked[:30]]

        merged = seed + auto_keywords
        return self._dedupe(merged)[:50]

    def _build_aliases(self, terms: Sequence[str]) -> Dict[str, List[str]]:
        aliases: Dict[str, List[str]] = {}

        for term in self._dedupe(terms):
            canonical = term.strip()
            if not canonical:
                continue

            generated = set()
            lowered = canonical.lower()
            compact = re.sub(r"[^a-z0-9]", "", lowered)
            if compact and compact != lowered:
                generated.add(compact)

            cleaned = lowered.replace("&", " and ")
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned and cleaned != lowered:
                generated.add(cleaned)

            suffix_map = ("soft", "tech", "systems", "labs", "works", "ware")
            if compact and len(compact) > 7:
                for suffix in suffix_map:
                    if compact.endswith(suffix) and len(compact) > len(suffix) + 3:
                        generated.add(f"{compact[:-len(suffix)]} {suffix}".strip())

            # Remove weak/noisy aliases.
            filtered = [
                alias
                for alias in sorted(generated)
                if len(alias) >= 4 and alias != lowered
            ]
            if filtered:
                aliases[canonical] = filtered

        return aliases

    @staticmethod
    def _dedupe(values: Sequence[str]) -> List[str]:
        seen = set()
        result: List[str] = []
        for value in values:
            cleaned = value.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(cleaned)
        return result
