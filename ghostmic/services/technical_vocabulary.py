"""Domain-aware technical vocabulary matching for conservative STT normalization.

This module is intentionally pure and does not log input text. Resume and session
terms are treated as matching evidence only; callers decide what may be persisted.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Iterable, Mapping, Sequence

from ghostmic.services.correction_policy import allow_automatic_correction


DEFAULT_REPLACEMENT_THRESHOLD = 0.82
DEFAULT_SUGGESTION_THRESHOLD = 0.62
MAX_CANDIDATES = 8
MAX_INPUT_CHARS = 2400
MAX_CUSTOM_TERMS = 500


@dataclass(frozen=True)
class VocabularyTerm:
    canonical: str
    aliases: tuple[str, ...] = ()
    source: str = "builtin"
    category: str = "technical"


@dataclass(frozen=True)
class VocabularyCandidate:
    raw_term: str
    canonical_term: str
    score: float
    evidence: tuple[str, ...]
    source: str
    category: str = "technical"

    @property
    def confidence(self) -> str:
        if self.score >= 0.90:
            return "HIGH"
        if self.score >= DEFAULT_SUGGESTION_THRESHOLD:
            return "MEDIUM"
        return "LOW"


# More than 100 representative canonical terms and spoken/noisy aliases. The
# aliases are deliberately bounded and contain no user-specific resume data.
_BUILTIN_ROWS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("Elasticsearch", ("elastic search", "elastic-search", "elastic"), "search"),
    ("Kubernetes", ("kubernetes", "cube net ease", "kube net ease"), "cloud"),
    ("Terraform", ("terra form", "terraform"), "iac"),
    ("Snowflake", ("snow flow", "snow flake"), "warehouse"),
    ("PostgreSQL", ("post gres", "post grass", "postgres"), "database"),
    ("Amazon S3", ("amazon s three", "amazon s3", "s three"), "aws"),
    ("AWS Lambda", ("aws lambda", "a w s lambda", "lambda function"), "aws"),
    ("AWS", ("a w s", "amazon web services"), "aws"),
    ("K8s", ("k eight s", "k8s"), "abbreviation"),
    ("EKS", ("e k s", "eks"), "aws"),
    ("ECS", ("e c s", "ecs"), "aws"),
    ("EC2", ("e c two", "ec two", "ec2"), "aws"),
    ("RDS", ("r d s", "rds"), "aws"),
    ("S3", ("s three", "s3"), "aws"),
    ("APM", ("a p m", "apm"), "observability"),
    ("SRE", ("s r e", "sre"), "role"),
    ("ETL", ("e t l", "etl"), "data"),
    ("ELK", ("e l k", "elk stack"), "observability"),
    ("kubectl", ("cube control", "kube control", "kube ctl"), "kubernetes"),
    ("Docker", ("docker", "dock er"), "containers"),
    ("Podman", ("pod man",), "containers"),
    ("Helm", ("helm chart", "helm"), "kubernetes"),
    ("Istio", ("is tee oh", "istio"), "kubernetes"),
    ("Argo CD", ("argo c d", "argo cd"), "kubernetes"),
    ("Prometheus", ("prometheus", "prom eth eus"), "observability"),
    ("Grafana", ("grafana", "graph anna"), "observability"),
    ("OpenTelemetry", ("open telemetry", "open telly metry"), "observability"),
    ("Kafka", ("kafka", "kaf ka"), "messaging"),
    ("Amazon Kinesis", ("amazon kinesis", "kinesis"), "aws"),
    ("RabbitMQ", ("rabbit m q", "rabbit mq"), "messaging"),
    ("Apache Spark", ("apache spark", "spark"), "data"),
    ("Apache Flink", ("apache flink", "flink"), "data"),
    ("Databricks", ("data bricks", "databricks"), "data"),
    ("dbt", ("d b t", "dbt"), "data"),
    ("Airflow", ("air flow", "apache airflow"), "data"),
    ("Hadoop", ("hadoop", "had oop"), "data"),
    ("Hive", ("hive", "hive sql"), "data"),
    ("BigQuery", ("big query", "bigquery"), "warehouse"),
    ("Redshift", ("red shift", "redshift"), "warehouse"),
    ("Datadog", ("data dog", "datadog"), "observability"),
    ("New Relic", ("new relic", "newrelic"), "observability"),
    ("PostGIS", ("post gis", "postgis"), "database"),
    ("MySQL", ("my sequel", "mysql"), "database"),
    ("SQL Server", ("sequel server", "sql server"), "database"),
    ("SQLite", ("sequel lite", "sqlite"), "database"),
    ("Oracle Database", ("oracle database", "oracle db"), "database"),
    ("MongoDB", ("mongo d b", "mongo db", "mongodb"), "database"),
    ("DynamoDB", ("dynamo d b", "dynamo db", "dynamodb"), "aws"),
    ("Cassandra", ("cassandra", "cass and dra"), "database"),
    ("Redis", ("redis", "read dis"), "database"),
    ("Memcached", ("mem cache d", "memcached"), "database"),
    ("Neo4j", ("neo four j", "neo4j"), "database"),
    ("GraphQL", ("graph q l", "graphql"), "api"),
    ("REST API", ("rest a p i", "rest api"), "api"),
    ("gRPC", ("g r p c", "grpc"), "api"),
    ("WebSocket", ("web socket", "websocket"), "api"),
    ("OAuth", ("o auth", "oauth"), "security"),
    ("OpenID Connect", ("open id connect", "oidc"), "security"),
    ("JWT", ("j w t", "jwt token"), "security"),
    ("TLS", ("t l s", "tls"), "security"),
    ("mTLS", ("m t l s", "mtls"), "security"),
    ("IAM", ("i a m", "iam"), "security"),
    ("RBAC", ("r back", "r b a c", "rbac"), "security"),
    ("Cognito", ("cognito", "co g knee toe"), "aws"),
    ("Vault", ("hashicorp vault", "vault"), "security"),
    ("Python", ("py thon", "python"), "language"),
    ("JavaScript", ("java script", "javascript"), "language"),
    ("TypeScript", ("type script", "typescript"), "language"),
    ("C#", ("c sharp", "c hash"), "language"),
    ("C++", ("c plus plus", "c plus-plus"), "language"),
    (".NET", ("dot net", "dotnet"), "platform"),
    ("Java", ("java",), "language"),
    ("Go", ("golang", "go lang"), "language"),
    ("Rust", ("rust",), "language"),
    ("Kotlin", ("kotlin", "cotlin"), "language"),
    ("Swift", ("swift",), "language"),
    ("Scala", ("scala",), "language"),
    ("R programming", ("r programming", "r language"), "language"),
    ("React", ("react", "react js", "reactjs"), "frontend"),
    ("Angular", ("angular", "angular js", "angularjs"), "frontend"),
    ("Vue.js", ("vue", "view js", "vue js"), "frontend"),
    ("Next.js", ("next js", "nextjs"), "frontend"),
    ("Node.js", ("node js", "nodejs"), "backend"),
    ("Django", ("django",), "backend"),
    ("FastAPI", ("fast api", "fastapi"), "backend"),
    ("Spring Boot", ("spring boot", "springboot"), "backend"),
    ("Jenkins", ("jenkins",), "ci"),
    ("GitHub Actions", ("github actions", "git hub actions"), "ci"),
    ("GitLab CI", ("gitlab c i", "git lab ci"), "ci"),
    ("CircleCI", ("circle c i", "circle ci"), "ci"),
    ("CI/CD", ("c i c d", "continuous integration continuous delivery"), "ci"),
    ("Git", ("git",), "version-control"),
    ("GitHub", ("git hub", "github"), "version-control"),
    ("Jira", ("jira",), "workflow"),
    ("Confluence", ("confluence",), "workflow"),
    ("Linux", ("linux", "linکس"), "platform"),
    ("Windows", ("windows",), "platform"),
    ("macOS", ("mac os", "macos"), "platform"),
    ("HTTP", ("h t t p", "http"), "protocol"),
    ("HTTPS", ("h t t p s", "https"), "protocol"),
    ("DNS", ("d n s", "dns"), "networking"),
    ("TCP/IP", ("t c p i p", "tcp ip"), "networking"),
    ("Load Balancer", ("load balancer", "load-balancer"), "networking"),
    ("CDN", ("c d n", "cdn"), "networking"),
    ("Nginx", ("engine x", "nginx"), "networking"),
    ("Apache", ("apache",), "networking"),
    ("Machine Learning", ("machine learning", "machine-learning"), "ml"),
    ("TensorFlow", ("tensor flow", "tensorflow"), "ml"),
    ("PyTorch", ("pie torch", "pytorch"), "ml"),
    ("LLM", ("l l m", "large language model"), "ml"),
    ("RAG", ("r a g", "rag pipeline"), "ml"),
    ("OpenAI", ("open ai", "openai"), "ml"),
    ("AWS Step Functions", ("step functions", "aws step functions"), "aws"),
    ("AWS CloudFormation", ("cloud formation", "cloudformation"), "aws"),
    ("Azure DevOps", ("azure dev ops", "azure devops"), "cloud"),
    ("Google Cloud", ("google cloud", "g c p"), "cloud"),
    ("Cloud Run", ("cloud run",), "cloud"),
    ("Terraform Cloud", ("terraform cloud",), "iac"),
    ("Ansible", ("ansible", "and sibble"), "iac"),
    ("Pulumi", ("pulumi", "pull um i"), "iac"),
    ("SCD Type 2", ("scd type 2", "slowly changing dimension type 2"), "data"),
    ("ACID", ("a c i d", "acid transactions"), "database"),
    ("OLTP", ("o l t p", "oltp"), "database"),
    ("OLAP", ("o l a p", "olap"), "database"),
    ("CAP theorem", ("cap theorem", "cap theorem"), "distributed"),
)


def _normalize(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9+#./-]+", " ", str(value).lower()).split())


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(token for token in _normalize(value).split() if token)


def _phonetic(value: str) -> str:
    """Small English consonant skeleton; used only as supporting evidence."""
    text = _normalize(value).replace("ph", "f").replace("ck", "k")
    text = re.sub(r"[aeiouy]+", "", text)
    text = re.sub(r"(.)\1+", r"\1", text)
    return text.replace("q", "k").replace("c", "k")[:32]


def _similarity(left: str, right: str) -> tuple[float, tuple[str, ...]]:
    left_norm, right_norm = _normalize(left), _normalize(right)
    if left_norm == right_norm:
        return 1.0, ("exact alias",)
    string_score = SequenceMatcher(None, left_norm, right_norm).ratio()
    left_tokens, right_tokens = set(_tokens(left)), set(_tokens(right))
    token_score = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    left_phonetic, right_phonetic = _phonetic(left), _phonetic(right)
    phonetic_score = (
        SequenceMatcher(None, left_phonetic, right_phonetic).ratio()
        if len(left_norm) >= 3 and len(right_norm) >= 3 and left_phonetic and right_phonetic
        else 0.0
    )
    score = max(string_score * 0.72 + token_score * 0.28, phonetic_score * 0.62)
    evidence = []
    if string_score >= 0.72:
        evidence.append("normalized string similarity")
    if token_score >= 0.5:
        evidence.append("token similarity")
    if phonetic_score >= 0.78:
        evidence.append("phonetic similarity")
    return score, tuple(evidence)


class TechnicalVocabularyEngine:
    """Score technical candidates and apply only high-confidence replacements."""

    def __init__(
        self,
        *,
        replacement_threshold: float = DEFAULT_REPLACEMENT_THRESHOLD,
        suggestion_threshold: float = DEFAULT_SUGGESTION_THRESHOLD,
        organization_terms: Mapping[str, Sequence[str] | str] | None = None,
        resume_terms: Mapping[str, Sequence[str] | str] | None = None,
        session_terms: Iterable[str] = (),
    ) -> None:
        self.replacement_threshold = max(0.0, min(1.0, float(replacement_threshold)))
        self.suggestion_threshold = max(0.0, min(self.replacement_threshold, float(suggestion_threshold)))
        self._terms: list[VocabularyTerm] = [
            VocabularyTerm(canonical, tuple(aliases), "builtin", category)
            for canonical, aliases, category in _BUILTIN_ROWS
        ]
        self._add_mapping(organization_terms, "organization")
        self._add_mapping(resume_terms, "resume")
        self._add_session_terms(session_terms)

    def _add_mapping(self, values, source: str) -> None:
        if not isinstance(values, Mapping):
            return
        for canonical, aliases in list(values.items())[:MAX_CUSTOM_TERMS]:
            alias_values = (aliases,) if isinstance(aliases, str) else tuple(aliases or ())
            self._terms.append(VocabularyTerm(str(canonical).strip(), alias_values, source))

    def _add_session_terms(self, values: Iterable[str]) -> None:
        for value in list(values or ())[:MAX_CUSTOM_TERMS]:
            term = str(value).strip()
            if term:
                self._terms.append(VocabularyTerm(term, (), "session"))

    def candidates(
        self,
        text: str,
        *,
        current_topic: str = "",
        recent_terms: Iterable[str] = (),
        max_candidates: int = MAX_CANDIDATES,
    ) -> list[VocabularyCandidate]:
        source_text = " ".join(str(text or "").split())[:MAX_INPUT_CHARS]
        if not source_text:
            return []
        spans = self._exact_alias_spans(source_text)
        # Fuzzy/phonetic matching is only needed when no known alias is
        # present. This keeps the live normalizer cheap while retaining
        # suggestions for genuinely noisy input.
        if not spans:
            spans = self._candidate_spans(source_text)
        topic_norm = _normalize(current_topic)
        recent_norm = {_normalize(item) for item in recent_terms if str(item).strip()}
        scored: dict[tuple[str, str], VocabularyCandidate] = {}
        for raw_term in spans:
            for term in self._terms:
                aliases = (term.canonical,) + term.aliases
                best_score = 0.0
                best_evidence: tuple[str, ...] = ()
                for alias in aliases:
                    score, evidence = _similarity(raw_term, alias)
                    if score > best_score:
                        best_score, best_evidence = score, evidence
                canonical_norm = _normalize(term.canonical)
                evidence = list(best_evidence)
                if topic_norm and (canonical_norm in topic_norm or topic_norm in canonical_norm):
                    best_score = min(1.0, best_score + 0.08)
                    evidence.append("current topic")
                if canonical_norm in recent_norm:
                    best_score = min(1.0, best_score + 0.05)
                    evidence.append("recent session term")
                if best_score < self.suggestion_threshold:
                    continue
                candidate = VocabularyCandidate(
                    raw_term=raw_term,
                    canonical_term=term.canonical,
                    score=round(best_score, 4),
                    evidence=tuple(dict.fromkeys(evidence)),
                    source=term.source,
                    category=term.category,
                )
                key = (_normalize(raw_term), _normalize(term.canonical))
                if key not in scored or candidate.score > scored[key].score:
                    scored[key] = candidate
        return sorted(
            scored.values(),
            key=lambda item: (-item.score, -len(item.raw_term), len(item.canonical_term)),
        )[:max(1, int(max_candidates))]

    def normalize_text(
        self,
        text: str,
        *,
        current_topic: str = "",
        recent_terms: Iterable[str] = (),
    ) -> tuple[str, tuple[VocabularyCandidate, ...]]:
        normalized = " ".join(str(text or "").split())
        candidates = self.candidates(
            normalized,
            current_topic=current_topic,
            recent_terms=recent_terms,
            max_candidates=MAX_CANDIDATES,
        )
        # Automatic mutation requires an explicit alias/canonical match. Fuzzy
        # and phonetic candidates remain suggestions until a caller explicitly
        # chooses them, preventing words such as "does" from being swallowed
        # by a nearby topic match.
        accepted = [
            candidate
            for candidate in candidates
            if candidate.score >= self.replacement_threshold
            and "exact alias" in candidate.evidence
            and allow_automatic_correction(
                candidate.raw_term,
                candidate.canonical_term,
                evidence=self._policy_evidence(candidate),
                confidence=candidate.confidence,
                score=candidate.score,
            )
        ]
        for candidate in sorted(accepted, key=lambda item: len(item.raw_term), reverse=True):
            pattern = re.compile(rf"(?<!\w){re.escape(candidate.raw_term)}(?!\w)", re.IGNORECASE)
            normalized = pattern.sub(candidate.canonical_term, normalized)
        return normalized, tuple(accepted)

    @staticmethod
    def _policy_evidence(candidate: VocabularyCandidate) -> tuple[str, ...]:
        evidence = list(candidate.evidence)
        if candidate.source == "builtin":
            evidence.append("technical vocabulary dictionary")
        elif candidate.source == "resume":
            evidence.append("resume/profile term")
        elif candidate.source == "session":
            evidence.append("immediately preceding conversation term")
        return tuple(evidence)

    @staticmethod
    def _candidate_spans(text: str) -> list[str]:
        words = re.findall(r"[A-Za-z0-9+#./-]+", text)
        spans: list[str] = []
        for size in range(6, 0, -1):
            for index in range(0, len(words) - size + 1):
                spans.append(" ".join(words[index : index + size]))
        return list(dict.fromkeys(spans))

    def _exact_alias_spans(self, text: str) -> list[str]:
        found: list[str] = []
        for term in self._terms:
            for alias in (term.canonical,) + term.aliases:
                alias_text = str(alias or "").strip()
                if len(_normalize(alias_text)) < 2:
                    continue
                pattern = re.compile(rf"(?<!\w){re.escape(alias_text)}(?!\w)", re.IGNORECASE)
                found.extend(match.group(0) for match in pattern.finditer(text))
        return list(dict.fromkeys(found))


__all__ = ["TechnicalVocabularyEngine", "VocabularyCandidate", "VocabularyTerm"]
