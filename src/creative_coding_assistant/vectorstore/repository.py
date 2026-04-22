"""Small generic repository helpers for Chroma collections."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from creative_coding_assistant.vectorstore.client import ensure_collection
from creative_coding_assistant.vectorstore.collections import CollectionDefinition
from creative_coding_assistant.vectorstore.metadata import (
    ChromaRecordMetadata,
    MetadataValue,
)


class VectorRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    document: str = Field(min_length=1)
    metadata: ChromaRecordMetadata
    embedding: list[float] = Field(min_length=1)

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("Embedding must not be empty.")
        return value


class StoredVectorRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    document: str | None
    metadata: dict[str, MetadataValue]


class ChromaRepository:
    """Thin wrapper that keeps Chroma calls behind a stable local API."""

    def __init__(self, *, client: Any, definition: CollectionDefinition) -> None:
        self.definition = definition
        self.collection = ensure_collection(client, definition)

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return

        payload: dict[str, Any] = {
            "ids": [record.id for record in records],
            "documents": [record.document for record in records],
            "metadatas": [record.metadata.to_chroma() for record in records],
            "embeddings": [record.embedding for record in records],
        }
        self.collection.upsert(**payload)

    def get(self, record_id: str) -> StoredVectorRecord | None:
        result = self.collection.get(
            ids=[record_id],
            include=["documents", "metadatas"],
        )
        ids = result.get("ids") or []
        if not ids:
            return None

        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return StoredVectorRecord(
            id=ids[0],
            document=documents[0] if documents else None,
            metadata=metadatas[0] if metadatas else {},
        )

    def count(self) -> int:
        return int(self.collection.count())
