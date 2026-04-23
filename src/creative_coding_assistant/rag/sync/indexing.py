"""Chroma indexing boundaries for official knowledge-base chunks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from loguru import logger

from creative_coding_assistant.rag.sync.models import OfficialSourceChunk
from creative_coding_assistant.vectorstore import (
    ChromaCollection,
    ChromaRecordMetadata,
    ChromaRepository,
    VectorRecord,
    VectorRecordKind,
    get_collection_definition,
    make_record_id,
)


class OfficialKnowledgeBaseIndexer:
    """Index normalized official KB chunks into the dedicated Chroma collection."""

    def __init__(self, *, client: Any) -> None:
        definition = get_collection_definition(ChromaCollection.KB_OFFICIAL_DOCS)
        self._repository = ChromaRepository(client=client, definition=definition)

    def upsert_chunks(
        self,
        chunks: Sequence[OfficialSourceChunk],
        embeddings: Sequence[list[float]],
    ) -> tuple[str, ...]:
        if len(chunks) != len(embeddings):
            raise ValueError("Official KB chunks and embeddings must align one-to-one.")
        if not chunks:
            return ()

        records = [
            self._build_vector_record(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self._repository.upsert(records)
        logger.info(
            "Indexed {} official KB chunks for source '{}'",
            len(records),
            chunks[0].source_id,
        )
        return tuple(record.id for record in records)

    def list_source_chunks(self, source_id: str):
        return self._repository.list(where={"source_id": source_id})

    def _build_vector_record(
        self,
        *,
        chunk: OfficialSourceChunk,
        embedding: list[float],
    ) -> VectorRecord:
        record_id = make_record_id(
            collection=ChromaCollection.KB_OFFICIAL_DOCS,
            record_kind=VectorRecordKind.OFFICIAL_DOC_CHUNK,
            parts=(
                chunk.source_id,
                chunk.source_url,
                chunk.content_hash,
                str(chunk.chunk_index),
            ),
        )
        metadata = ChromaRecordMetadata(
            collection=ChromaCollection.KB_OFFICIAL_DOCS,
            record_kind=VectorRecordKind.OFFICIAL_DOC_CHUNK,
            source_id=chunk.source_id,
            domain=chunk.domain,
            extras={
                "source_url": chunk.source_url,
                "resolved_url": chunk.resolved_url,
                "source_type": chunk.source_type.value,
                "publisher": chunk.publisher,
                "registry_title": chunk.registry_title,
                "document_title": chunk.document_title,
                "chunk_index": chunk.chunk_index,
                "char_count": chunk.char_count,
                "content_hash": chunk.content_hash,
                "chunk_hash": chunk.chunk_hash,
                "fetched_at": chunk.fetched_at.isoformat(),
            },
        )
        return VectorRecord(
            id=record_id,
            document=chunk.text,
            metadata=metadata,
            embedding=embedding,
        )
