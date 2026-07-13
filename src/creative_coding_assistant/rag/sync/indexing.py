"""Chroma indexing boundaries for official knowledge-base chunks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from loguru import logger

from creative_coding_assistant.rag.sync.embedding import ChunkEmbedder
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
        records = self.build_vector_records(chunks, embeddings)
        return self.upsert_records(records)

    def upsert_records(
        self,
        records: Sequence[VectorRecord],
    ) -> tuple[str, ...]:
        if not records:
            return ()

        self._repository.upsert(records)
        logger.info(
            "Indexed {} official KB chunks for source '{}'",
            len(records),
            records[0].metadata.source_id,
        )
        return tuple(record.id for record in records)

    def replace_source_records(
        self,
        records: Sequence[VectorRecord],
        *,
        source_id: str,
    ) -> tuple[str, ...]:
        """Store one source snapshot and remove records superseded by it."""

        source_ids = {record.metadata.source_id for record in records}
        if source_ids and source_ids != {source_id}:
            raise ValueError("Replacement records must match the selected source.")

        previous_ids = {
            record.id for record in self.list_source_chunks(source_id=source_id)
        }
        current_ids = self.upsert_records(records)
        superseded_ids = tuple(sorted(previous_ids - set(current_ids)))
        self._repository.delete(superseded_ids)
        if superseded_ids:
            logger.info(
                "Removed {} superseded official KB chunks for source '{}'",
                len(superseded_ids),
                source_id,
            )
        return current_ids

    def embed_and_upsert_chunks(
        self,
        chunks: Sequence[OfficialSourceChunk],
        *,
        embedder: ChunkEmbedder,
    ) -> tuple[str, ...]:
        embeddings = embedder.embed_chunks(chunks)
        return self.upsert_chunks(chunks, embeddings)

    def build_vector_records(
        self,
        chunks: Sequence[OfficialSourceChunk],
        embeddings: Sequence[list[float]],
    ) -> tuple[VectorRecord, ...]:
        if len(chunks) != len(embeddings):
            raise ValueError("Official KB chunks and embeddings must align one-to-one.")
        if not chunks:
            return ()

        return tuple(
            self._build_vector_record(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        )

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
