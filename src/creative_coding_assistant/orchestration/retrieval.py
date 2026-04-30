"""Orchestration-facing retrieval boundary contracts and adapters."""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.contracts import AssistantRequest, CreativeCodingDomain
from creative_coding_assistant.orchestration.routing import (
    RouteCapability,
    RouteDecision,
    RouteName,
)
from creative_coding_assistant.rag.retrieval import (
    KnowledgeBaseRetrievalFilter,
    KnowledgeBaseRetrievalRequest,
    KnowledgeBaseRetriever,
)
from creative_coding_assistant.rag.sources import OfficialSourceType

DEFAULT_RETRIEVAL_LIMIT = 5
_WHITESPACE_PATTERN = re.compile(r"\s+")
_THREE_JS_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bthree(?:\.js|js|\s+js)\b"), 3),
)
_REACT_THREE_FIBER_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\breact\s+three\s+fiber\b"), 3),
    (re.compile(r"@react-three/fiber"), 3),
    (re.compile(r"\br3f\b"), 3),
    (re.compile(r"\buseframe\b"), 2),
)
_P5_JS_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bp5(?:\.js|js)?\b"), 3),
)
_GLSL_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bglsl\b"), 3),
    (re.compile(r"\bfragment\s+shader\b"), 2),
    (re.compile(r"\bvertex\s+shader\b"), 2),
    (re.compile(r"\bshader\b"), 1),
)
_EXPLICIT_DOMAIN_PATTERNS: tuple[
    tuple[CreativeCodingDomain, tuple[tuple[re.Pattern[str], int], ...]],
    ...,
] = (
    (CreativeCodingDomain.THREE_JS, _THREE_JS_PATTERNS),
    (CreativeCodingDomain.REACT_THREE_FIBER, _REACT_THREE_FIBER_PATTERNS),
    (CreativeCodingDomain.P5_JS, _P5_JS_PATTERNS),
    (CreativeCodingDomain.GLSL, _GLSL_PATTERNS),
)


class RetrievalContextSource(StrEnum):
    OFFICIAL_KB = "official_kb"


class RetrievalContextFilter(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    domain: CreativeCodingDomain | None = None
    domains: tuple[CreativeCodingDomain, ...] = Field(default_factory=tuple)
    source_id: str | None = Field(default=None, min_length=1)
    source_type: OfficialSourceType | None = None
    publisher: str | None = Field(default=None, min_length=1)

    @field_validator("domains", mode="before")
    @classmethod
    def normalize_domains(
        cls,
        value: Sequence[CreativeCodingDomain | str] | CreativeCodingDomain | str | None,
    ) -> tuple[CreativeCodingDomain, ...]:
        if value is None:
            return ()
        if isinstance(value, CreativeCodingDomain):
            return (value,)
        if isinstance(value, str):
            return (CreativeCodingDomain(value.strip()),)

        normalized: list[CreativeCodingDomain] = []
        for item in value:
            domain = (
                item
                if isinstance(item, CreativeCodingDomain)
                else CreativeCodingDomain(str(item).strip())
            )
            if domain not in normalized:
                normalized.append(domain)
        return tuple(normalized)

    @model_validator(mode="before")
    @classmethod
    def populate_legacy_domain_fields(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        domain = normalized.get("domain")
        domains = normalized.get("domains")

        if domain is not None and not domains:
            normalized["domains"] = (domain,)

        return normalized

    @model_validator(mode="after")
    def validate_domain_alignment(self) -> RetrievalContextFilter:
        if self.domain is None and len(self.domains) == 1:
            object.__setattr__(self, "domain", self.domains[0])

        if self.domain is not None and self.domain not in self.domains:
            raise ValueError(
                "Retrieval context domain must be included in domains "
                "when both are provided."
            )

        return self


class RetrievalContextRequest(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    query: str = Field(min_length=1)
    route: RouteName
    limit: int = Field(default=DEFAULT_RETRIEVAL_LIMIT, ge=1, le=20)
    filters: RetrievalContextFilter = Field(default_factory=RetrievalContextFilter)


class RetrievedKnowledgeChunk(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source_id: str = Field(min_length=1)
    domain: CreativeCodingDomain
    source_type: OfficialSourceType
    publisher: str = Field(min_length=1)
    registry_title: str = Field(min_length=1)
    document_title: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    resolved_url: str | None = None
    chunk_index: int = Field(ge=0)
    excerpt: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)


class RetrievalContextResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: RetrievalContextRequest
    source: RetrievalContextSource = RetrievalContextSource.OFFICIAL_KB
    chunks: tuple[RetrievedKnowledgeChunk, ...] = Field(default_factory=tuple)


class RetrievalGateway(Protocol):
    def retrieve_context(
        self,
        request: RetrievalContextRequest,
    ) -> RetrievalContextResponse:
        """Return orchestration-facing retrieval context for a single request."""


class KnowledgeBaseRetrievalAdapter:
    """Adapt KB retrieval outputs into orchestration-facing context results."""

    def __init__(self, *, retriever: KnowledgeBaseRetriever) -> None:
        self._retriever = retriever

    def retrieve_context(
        self,
        request: RetrievalContextRequest,
    ) -> RetrievalContextResponse:
        retrieval_request = KnowledgeBaseRetrievalRequest(
            query=request.query,
            limit=request.limit,
            filters=KnowledgeBaseRetrievalFilter(
                domain=request.filters.domain,
                domains=request.filters.domains,
                source_id=request.filters.source_id,
                source_type=request.filters.source_type,
                publisher=request.filters.publisher,
            ),
        )
        retrieval_response = self._retriever.search(retrieval_request)
        chunks = tuple(
            RetrievedKnowledgeChunk(
                source_id=result.source_id,
                domain=result.domain,
                source_type=result.source_type,
                publisher=result.publisher,
                registry_title=result.registry_title,
                document_title=result.document_title,
                source_url=result.source_url,
                resolved_url=result.resolved_url,
                chunk_index=result.chunk_index,
                excerpt=result.text,
                score=result.score,
            )
            for result in retrieval_response.results
        )
        logger.info(
            "Built orchestration retrieval context with {} chunk(s)",
            len(chunks),
        )
        return RetrievalContextResponse(request=request, chunks=chunks)


def build_retrieval_context_request(
    assistant_request: AssistantRequest,
    route_decision: RouteDecision,
    *,
    limit: int = DEFAULT_RETRIEVAL_LIMIT,
) -> RetrievalContextRequest | None:
    if RouteCapability.OFFICIAL_DOCS not in route_decision.capabilities:
        return None

    domain, domains = _resolve_retrieval_domains(assistant_request)
    return RetrievalContextRequest(
        query=assistant_request.query,
        route=route_decision.route,
        limit=limit,
        filters=RetrievalContextFilter(
            domain=domain,
            domains=domains,
        ),
    )


def _resolve_retrieval_domains(
    assistant_request: AssistantRequest,
) -> tuple[CreativeCodingDomain | None, tuple[CreativeCodingDomain, ...]]:
    query_domains = _detect_explicit_query_domains(assistant_request.query)
    if not query_domains:
        return assistant_request.domain, assistant_request.domains

    logger.info(
        "Using explicit query domains {} for retrieval instead of request domains {}",
        [domain.value for domain in query_domains],
        [domain.value for domain in assistant_request.domains],
    )
    if len(query_domains) == 1:
        return query_domains[0], query_domains
    return None, query_domains


def _detect_explicit_query_domains(query: str) -> tuple[CreativeCodingDomain, ...]:
    normalized_query = _normalize_query(query)
    if not normalized_query:
        return ()

    scores = [
        (domain, _score_domain(normalized_query, patterns))
        for domain, patterns in _EXPLICIT_DOMAIN_PATTERNS
    ]
    detected = tuple(domain for domain, score in scores if score > 0)
    return detected


def _normalize_query(value: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", value.strip().lower())


def _score_domain(
    query: str,
    patterns: tuple[tuple[re.Pattern[str], int], ...],
) -> int:
    return sum(weight for pattern, weight in patterns if pattern.search(query))
