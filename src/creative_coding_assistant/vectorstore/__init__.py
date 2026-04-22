"""Chroma collection definitions and repository helpers."""

from creative_coding_assistant.vectorstore.client import (
    create_chroma_client,
    ensure_collection,
    ensure_project_collections,
)
from creative_coding_assistant.vectorstore.collections import (
    ChromaCollection,
    CollectionDefinition,
    collection_definitions,
    collection_names,
    get_collection_definition,
)
from creative_coding_assistant.vectorstore.ids import VectorRecordKind, make_record_id
from creative_coding_assistant.vectorstore.metadata import ChromaRecordMetadata
from creative_coding_assistant.vectorstore.repository import (
    ChromaRepository,
    StoredVectorRecord,
    VectorRecord,
)

__all__ = [
    "ChromaCollection",
    "ChromaRecordMetadata",
    "ChromaRepository",
    "CollectionDefinition",
    "StoredVectorRecord",
    "VectorRecord",
    "VectorRecordKind",
    "collection_definitions",
    "collection_names",
    "create_chroma_client",
    "ensure_collection",
    "ensure_project_collections",
    "get_collection_definition",
    "make_record_id",
]
