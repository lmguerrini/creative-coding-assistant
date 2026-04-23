"""Typed sync models for approved official knowledge-base sources."""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.rag.sources import OfficialSourceType

_HASH_PATTERN = r"^[a-f0-9]{64}$"


class SourceContentFormat(StrEnum):
    HTML = "html"
    TEXT = "text"


def _require_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Sync timestamps must be timezone-aware.")
    return value


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class OfficialSourceSyncRequest(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source_id: str = Field(min_length=1)
    requested_at: datetime

    @field_validator("requested_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value)


class FetchedSourceDocument(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source_id: str = Field(min_length=1)
    domain: CreativeCodingDomain
    source_type: OfficialSourceType
    registry_title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    resolved_url: str = Field(min_length=1)
    fetched_at: datetime
    content_format: SourceContentFormat
    raw_content: str = Field(min_length=1)
    raw_content_hash: str = Field(pattern=_HASH_PATTERN)

    @field_validator("fetched_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value)

    @classmethod
    def from_content(
        cls,
        *,
        source_id: str,
        domain: CreativeCodingDomain,
        source_type: OfficialSourceType,
        registry_title: str,
        publisher: str,
        source_url: str,
        resolved_url: str,
        fetched_at: datetime,
        content_format: SourceContentFormat,
        raw_content: str,
    ) -> Self:
        return cls(
            source_id=source_id,
            domain=domain,
            source_type=source_type,
            registry_title=registry_title,
            publisher=publisher,
            source_url=source_url,
            resolved_url=resolved_url,
            fetched_at=fetched_at,
            content_format=content_format,
            raw_content=raw_content,
            raw_content_hash=_hash_text(raw_content),
        )


class NormalizedSourceDocument(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source_id: str = Field(min_length=1)
    domain: CreativeCodingDomain
    source_type: OfficialSourceType
    registry_title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    resolved_url: str = Field(min_length=1)
    fetched_at: datetime
    document_title: str = Field(min_length=1)
    normalized_text: str = Field(min_length=1)
    content_hash: str = Field(pattern=_HASH_PATTERN)

    @field_validator("fetched_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value)

    @classmethod
    def from_text(
        cls,
        *,
        fetched_document: FetchedSourceDocument,
        document_title: str,
        normalized_text: str,
    ) -> Self:
        return cls(
            source_id=fetched_document.source_id,
            domain=fetched_document.domain,
            source_type=fetched_document.source_type,
            registry_title=fetched_document.registry_title,
            publisher=fetched_document.publisher,
            source_url=fetched_document.source_url,
            resolved_url=fetched_document.resolved_url,
            fetched_at=fetched_document.fetched_at,
            document_title=document_title,
            normalized_text=normalized_text,
            content_hash=_hash_text(normalized_text),
        )


class OfficialSourceChunk(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source_id: str = Field(min_length=1)
    domain: CreativeCodingDomain
    source_type: OfficialSourceType
    registry_title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    resolved_url: str = Field(min_length=1)
    fetched_at: datetime
    document_title: str = Field(min_length=1)
    content_hash: str = Field(pattern=_HASH_PATTERN)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    chunk_hash: str = Field(pattern=_HASH_PATTERN)
    char_count: int = Field(ge=1)

    @field_validator("fetched_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value)

    @field_validator("char_count")
    @classmethod
    def require_positive_char_count(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Chunk character count must be positive.")
        return value

    @classmethod
    def from_text(
        cls,
        *,
        normalized_document: NormalizedSourceDocument,
        chunk_index: int,
        text: str,
    ) -> Self:
        stripped_text = text.strip()
        return cls(
            source_id=normalized_document.source_id,
            domain=normalized_document.domain,
            source_type=normalized_document.source_type,
            registry_title=normalized_document.registry_title,
            publisher=normalized_document.publisher,
            source_url=normalized_document.source_url,
            resolved_url=normalized_document.resolved_url,
            fetched_at=normalized_document.fetched_at,
            document_title=normalized_document.document_title,
            content_hash=normalized_document.content_hash,
            chunk_index=chunk_index,
            text=stripped_text,
            chunk_hash=_hash_text(stripped_text),
            char_count=len(stripped_text),
        )
