"""Orchestration-facing retrieval boundary contracts and adapters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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
)
from creative_coding_assistant.rag.sources import OfficialSourceType

DEFAULT_RETRIEVAL_LIMIT = 5
_P5_EXAMPLE_SCORE_BOOST = 0.08
_P5_WEAK_REFERENCE_SCORE_PENALTY = 0.06
_P5_VERY_SHORT_REFERENCE_CHARS = 80
_P5_HEADING_LIKE_REFERENCE_CHARS = 140
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
_P5_HEADING_DETAIL_MARKERS = (".", ";", "{", "}", "=", "\n")


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
    rank: int | None = Field(default=None, ge=1)
    original_score: float | None = Field(default=None, ge=0, le=1)
    score_adjustment: float | None = Field(default=None, ge=-1, le=1)
    domain_match: bool | None = None
    selection_reason: str | None = Field(default=None, min_length=1)


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


@dataclass(frozen=True, slots=True)
class _RankedRetrievalResult:
    result: KnowledgeBaseSearchResult
    original_score: float


class KnowledgeBaseRetrievalAdapter:
    """Adapt KB retrieval outputs into orchestration-facing context results."""

    def __init__(self, *, retriever: KnowledgeBaseRetriever) -> None:
        self._retriever = retriever

    def retrieve_context(
        self,
        request: RetrievalContextRequest,
    ) -> RetrievalContextResponse:
        retrieval_request = _build_kb_retrieval_request(request)
        retrieval_response = self._retriever.search(retrieval_request)
        ranked_results = _rank_retrieval_results(
            retrieval_response.results,
            request=request,
        )
        chunks = tuple(
            _build_retrieved_chunk(
                ranked_result,
                request=request,
                rank=index,
            )
            for index, ranked_result in enumerate(ranked_results, start=1)
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

    explicit_domains = detect_explicit_query_domains(assistant_request.query)
    domains = explicit_domains or route_decision.domains or assistant_request.domains
    if len(domains) == 1:
        domain = domains[0]
    elif domains:
        domain = None
    else:
        domain = route_decision.domain or assistant_request.domain
    return RetrievalContextRequest(
        query=assistant_request.query,
        route=route_decision.route,
        limit=limit,
        filters=RetrievalContextFilter(
            domain=domain,
            domains=domains,
        ),
    )


def _build_kb_retrieval_request(
    request: RetrievalContextRequest,
) -> KnowledgeBaseRetrievalRequest:
    return KnowledgeBaseRetrievalRequest(
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


def _build_retrieved_chunk(
    ranked_result: _RankedRetrievalResult,
    *,
    request: RetrievalContextRequest,
    rank: int,
) -> RetrievedKnowledgeChunk:
    result = ranked_result.result
    score_adjustment = result.score - ranked_result.original_score
    domain_match = _domain_matches_request(result, request=request)

    return RetrievedKnowledgeChunk(
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
        rank=rank,
        original_score=ranked_result.original_score,
        score_adjustment=score_adjustment if score_adjustment != 0 else None,
        domain_match=domain_match,
        selection_reason=_build_selection_reason(
            domain_match=domain_match,
            score_adjustment=score_adjustment,
        ),
    )


def _rank_retrieval_results(
    results: tuple[KnowledgeBaseSearchResult, ...],
    *,
    request: RetrievalContextRequest,
) -> tuple[_RankedRetrievalResult, ...]:
    if request.route is not RouteName.GENERATE:
        return tuple(
            _RankedRetrievalResult(result=result, original_score=result.score)
            for result in results
        )

    ranked: list[tuple[int, _RankedRetrievalResult]] = []
    changed = False
    for index, result in enumerate(results):
        adjusted = _adjust_p5_generate_result_score(result)
        changed = changed or adjusted.score != result.score
        ranked.append(
            (
                index,
                _RankedRetrievalResult(
                    result=adjusted,
                    original_score=result.score,
                ),
            )
        )

    if not changed:
        return tuple(result for _, result in ranked)

    ranked.sort(key=lambda item: (-item[1].result.score, item[0]))
    return tuple(result for _, result in ranked)


def _domain_matches_request(
    result: KnowledgeBaseSearchResult,
    *,
    request: RetrievalContextRequest,
) -> bool | None:
    requested_domains = request.filters.domains
    if not requested_domains:
        return None

    return result.domain in requested_domains


def _build_selection_reason(
    *,
    domain_match: bool | None,
    score_adjustment: float,
) -> str:
    if score_adjustment != 0:
        return (
            "Selected after semantic ranking and route-specific generation "
            "relevance adjustment."
        )

    if domain_match:
        return "Selected for semantic relevance within the requested domain scope."

    if domain_match is False:
        return (
            "Selected for semantic relevance despite falling outside the requested "
            "domain scope."
        )

    return "Selected by semantic relevance from the official knowledge base."


def _adjust_p5_generate_result_score(
    result: KnowledgeBaseSearchResult,
) -> KnowledgeBaseSearchResult:
    if result.domain is not CreativeCodingDomain.P5_JS:
        return result

    if _looks_like_p5_example_chunk(result.text):
        adjusted_score = min(1.0, result.score + _P5_EXAMPLE_SCORE_BOOST)
    elif _looks_like_weak_p5_reference_chunk(result.text):
        adjusted_score = max(0.0, result.score - _P5_WEAK_REFERENCE_SCORE_PENALTY)
    else:
        return result

    return result.model_copy(update={"score": adjusted_score})


def _looks_like_p5_example_chunk(text: str) -> bool:
    return any(marker in text for marker in _P5_EXAMPLE_MARKERS)


def _looks_like_weak_p5_reference_chunk(text: str) -> bool:
    if _looks_like_p5_example_chunk(text):
        return False

    compact_text = " ".join(text.split())
    if len(compact_text) <= _P5_VERY_SHORT_REFERENCE_CHARS:
        return True

    return (
        len(compact_text) <= _P5_HEADING_LIKE_REFERENCE_CHARS
        and not any(marker in compact_text for marker in _P5_HEADING_DETAIL_MARKERS)
    )
