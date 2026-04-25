"""Typed contracts for official knowledge-base retrieval."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.rag.sources import OfficialSourceType

_HASH_PATTERN = r"^[a-f0-9]{64}$"


class KnowledgeBaseRetrievalFilter(BaseModel):
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
    def validate_domain_alignment(self) -> KnowledgeBaseRetrievalFilter:
        if self.domain is None and len(self.domains) == 1:
            object.__setattr__(self, "domain", self.domains[0])

        if self.domain is not None and self.domain not in self.domains:
            raise ValueError(
                "Retrieval filter domain must be included in domains "
                "when both are provided."
            )

        return self


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
