"""Orchestration-facing retrieval boundary contracts and adapters."""

from __future__ import annotations

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
    KnowledgeBaseSearchResult,
)
from creative_coding_assistant.rag.retrieval.domain_intent import (
    detect_explicit_query_domains,
    resolve_effective_query_domains,
)
from creative_coding_assistant.rag.sources import OfficialSourceType

DEFAULT_RETRIEVAL_LIMIT = 5
_P5_EXAMPLE_SCORE_BOOST = 0.08
_P5_EXAMPLE_MARKERS = (
    "function setup(",
    "function draw(",
    "createCanvas(",
    "ellipse(",
    "rect(",
    "background(",
    "let ",
    "const ",
    "var ",
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
        ranked_results = _rank_retrieval_results(
            retrieval_response.results,
            request=request,
        )
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
            for result in ranked_results
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
    query_domains = detect_explicit_query_domains(assistant_request.query)
    if not query_domains:
        return assistant_request.domain, assistant_request.domains

    logger.info(
        "Using explicit query domains {} for retrieval instead of request domains {}",
        [domain.value for domain in query_domains],
        [domain.value for domain in assistant_request.domains],
    )
    effective_domains = resolve_effective_query_domains(
        query=assistant_request.query,
        selected_domains=assistant_request.domains,
    )
    if len(effective_domains) == 1:
        return effective_domains[0], effective_domains
    return None, effective_domains


def _rank_retrieval_results(
    results: tuple[KnowledgeBaseSearchResult, ...],
    *,
    request: RetrievalContextRequest,
) -> tuple[KnowledgeBaseSearchResult, ...]:
    if request.route is not RouteName.GENERATE:
        return results

    ranked: list[tuple[int, KnowledgeBaseSearchResult]] = []
    changed = False
    for index, result in enumerate(results):
        boosted = _boost_p5_example_result(result)
        changed = changed or boosted.score != result.score
        ranked.append((index, boosted))

    if not changed:
        return results

    ranked.sort(key=lambda item: (-item[1].score, item[0]))
    return tuple(result for _, result in ranked)


def _boost_p5_example_result(
    result: KnowledgeBaseSearchResult,
) -> KnowledgeBaseSearchResult:
    if result.domain is not CreativeCodingDomain.P5_JS:
        return result
    if not _looks_like_p5_example_chunk(result.text):
        return result

    boosted_score = min(1.0, result.score + _P5_EXAMPLE_SCORE_BOOST)
    return result.model_copy(update={"score": boosted_score})


def _looks_like_p5_example_chunk(text: str) -> bool:
    return any(marker in text for marker in _P5_EXAMPLE_MARKERS)
