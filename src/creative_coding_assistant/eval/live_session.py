"""Typed models for real live-session evaluation samples."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantMode, CreativeCodingDomain
from creative_coding_assistant.orchestration.retrieval import RetrievedKnowledgeChunk
from creative_coding_assistant.orchestration.routing import (
    DomainSelectionShape,
    RouteCapability,
    RouteName,
)
from creative_coding_assistant.rag.sources import OfficialSourceType


class LiveSessionRetrievedContext(BaseModel):
    """One retrieved context excerpt captured from a real live session."""

    model_config = ConfigDict(frozen=True)

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

    @classmethod
    def from_chunk(
        cls,
        chunk: RetrievedKnowledgeChunk,
    ) -> LiveSessionRetrievedContext:
        return cls(
            source_id=chunk.source_id,
            domain=chunk.domain,
            source_type=chunk.source_type,
            publisher=chunk.publisher,
            registry_title=chunk.registry_title,
            document_title=chunk.document_title,
            source_url=chunk.source_url,
            resolved_url=chunk.resolved_url,
            chunk_index=chunk.chunk_index,
            excerpt=chunk.excerpt,
            score=chunk.score,
        )


class LiveSessionRouteMetadata(BaseModel):
    """Route metadata captured from the existing assistant trace."""

    model_config = ConfigDict(frozen=True)

    route: RouteName | None = None
    mode: AssistantMode
    domain: CreativeCodingDomain | None = None
    domains: tuple[CreativeCodingDomain, ...] = Field(default_factory=tuple)
    domain_selection: DomainSelectionShape | None = None
    capabilities: tuple[RouteCapability, ...] = Field(default_factory=tuple)


class LiveSessionEvalSample(BaseModel):
    """One real assistant turn recorded for later evaluation work."""

    model_config = ConfigDict(frozen=True)

    sample_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    conversation_id: str | None = None
    project_id: str | None = None
    route: LiveSessionRouteMetadata
    retrieved_contexts: tuple[LiveSessionRetrievedContext, ...] = Field(
        default_factory=tuple
    )
    started_at: datetime
    completed_at: datetime
    recorded_at: datetime
