"""Flat metadata conventions for Chroma records."""

from __future__ import annotations

from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.contracts.requests import (
    AssistantMode,
    CreativeCodingDomain,
)
from creative_coding_assistant.vectorstore.collections import ChromaCollection
from creative_coding_assistant.vectorstore.ids import VectorRecordKind

MetadataValue: TypeAlias = str | int | float | bool
RESERVED_METADATA_KEYS = {
    "collection",
    "record_kind",
    "schema_version",
    "source_id",
    "domain",
    "mode",
    "conversation_id",
    "project_id",
    "trace_id",
}


class ChromaRecordMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    collection: ChromaCollection
    record_kind: VectorRecordKind
    source_id: str = Field(min_length=1)
    schema_version: int = Field(default=1, ge=1)
    domain: CreativeCodingDomain | None = None
    mode: AssistantMode | None = None
    conversation_id: str | None = None
    project_id: str | None = None
    trace_id: str | None = None
    extras: dict[str, MetadataValue] = Field(default_factory=dict)

    @field_validator("source_id", "conversation_id", "project_id", "trace_id")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Metadata text fields must not be blank.")
        return cleaned

    @model_validator(mode="after")
    def validate_extra_keys(self) -> ChromaRecordMetadata:
        reserved = RESERVED_METADATA_KEYS & set(self.extras)
        if reserved:
            raise ValueError(
                f"Extra metadata cannot override reserved keys: {reserved}"
            )
        return self

    def to_chroma(self) -> dict[str, MetadataValue]:
        """Return Chroma-compatible flat metadata."""

        payload: dict[str, MetadataValue] = {
            "collection": self.collection.value,
            "record_kind": self.record_kind.value,
            "schema_version": self.schema_version,
            "source_id": self.source_id,
        }
        optional_values: dict[str, object] = {
            "domain": self.domain.value if self.domain is not None else None,
            "mode": self.mode.value if self.mode is not None else None,
            "conversation_id": self.conversation_id,
            "project_id": self.project_id,
            "trace_id": self.trace_id,
        }
        payload.update(
            {
                key: value
                for key, value in optional_values.items()
                if isinstance(value, str)
            }
        )
        payload.update(self.extras)
        return payload
