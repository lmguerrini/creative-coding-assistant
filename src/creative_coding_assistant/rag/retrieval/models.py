"""Typed contracts for official knowledge-base retrieval."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.rag.sources import OfficialSourceType

_HASH_PATTERN = r"^[a-f0-9]{64}$"


class KnowledgeBaseRetrievalFilter(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    domain: CreativeCodingDomain | None = None
    source_id: str | None = Field(default=None, min_length=1)
    source_type: OfficialSourceType | None = None
    publisher: str | None = Field(default=None, min_length=1)


class KnowledgeBaseRetrievalRequest(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)
    filters: KnowledgeBaseRetrievalFilter = Field(
        default_factory=KnowledgeBaseRetrievalFilter
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Retrieval query must not be empty.")
        return value


class KnowledgeBaseSearchResult(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    record_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    domain: CreativeCodingDomain
    source_type: OfficialSourceType
    publisher: str = Field(min_length=1)
    registry_title: str = Field(min_length=1)
    document_title: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    resolved_url: str | None = None
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    char_count: int = Field(ge=1)
    content_hash: str = Field(pattern=_HASH_PATTERN)
    chunk_hash: str = Field(pattern=_HASH_PATTERN)
    distance: float = Field(ge=0)
    score: float = Field(ge=0, le=1)


class KnowledgeBaseRetrievalResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: KnowledgeBaseRetrievalRequest
    results: tuple[KnowledgeBaseSearchResult, ...] = Field(default_factory=tuple)
