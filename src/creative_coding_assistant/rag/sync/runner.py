"""End-to-end official KB sync runner."""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.rag.source_health import (
    OfficialSourceHealthSnapshot,
    OfficialSourceSyncMetadata,
    evaluate_official_source_health,
)
from creative_coding_assistant.rag.sync.chunking import OfficialSourceChunker
from creative_coding_assistant.rag.sync.embedding import ChunkEmbedder
from creative_coding_assistant.rag.sync.fetcher import OfficialSourceFetcher
from creative_coding_assistant.rag.sync.indexing import OfficialKnowledgeBaseIndexer
from creative_coding_assistant.rag.sync.models import (
    FetchedSourceDocument,
    NormalizedSourceDocument,
    OfficialSourceChunk,
    OfficialSourceSyncRequest,
)
from creative_coding_assistant.rag.sync.normalize import OfficialSourceNormalizer
from creative_coding_assistant.vectorstore import VectorRecord


class OfficialKnowledgeBaseSyncResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: OfficialSourceSyncRequest
    fetched_document: FetchedSourceDocument
    normalized_document: NormalizedSourceDocument
    chunks: tuple[OfficialSourceChunk, ...] = Field(default_factory=tuple)
    embeddings: tuple[tuple[float, ...], ...] = Field(default_factory=tuple)
    vector_records: tuple[VectorRecord, ...] = Field(default_factory=tuple)
    record_ids: tuple[str, ...] = Field(default_factory=tuple)
    sync_metadata: OfficialSourceSyncMetadata | None = None

    @model_validator(mode="after")
    def populate_sync_metadata(self) -> OfficialKnowledgeBaseSyncResult:
        if self.sync_metadata is not None:
            return self

        object.__setattr__(
            self,
            "sync_metadata",
            OfficialSourceSyncMetadata.from_success(
                source_id=self.fetched_document.source_id,
                requested_at=self.request.requested_at,
                last_synced_at=self.fetched_document.fetched_at,
                source_url=self.fetched_document.source_url,
                resolved_url=self.fetched_document.resolved_url,
                domain=self.fetched_document.domain,
                source_type=self.fetched_document.source_type,
                content_hash=self.normalized_document.content_hash,
                chunk_count=len(self.chunks),
                record_count=len(self.record_ids),
            ),
        )
        return self

    def health_snapshot(
        self,
        *,
        checked_at,
    ) -> OfficialSourceHealthSnapshot:
        return evaluate_official_source_health(
            self.fetched_document.source_id,
            sync_metadata=self.sync_metadata,
            checked_at=checked_at,
        )


class OfficialKnowledgeBaseSyncRunner:
    """Compose fetch, normalize, chunk, embed, and index for one official source."""

    def __init__(
        self,
        *,
        fetcher: OfficialSourceFetcher,
        embedder: ChunkEmbedder,
        indexer: OfficialKnowledgeBaseIndexer,
        normalizer: OfficialSourceNormalizer | None = None,
        chunker: OfficialSourceChunker | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._normalizer = normalizer or OfficialSourceNormalizer()
        self._chunker = chunker or OfficialSourceChunker()
        self._embedder = embedder
        self._indexer = indexer

    def run(
        self,
        request: OfficialSourceSyncRequest,
    ) -> OfficialKnowledgeBaseSyncResult:
        fetched_document = self._fetcher.fetch(request)
        normalized_document = self._normalizer.normalize(fetched_document)
        chunks = self._chunker.chunk(normalized_document)
        embeddings = self._embedder.embed_chunks(chunks)
        vector_records = self._indexer.build_vector_records(chunks, embeddings)
        record_ids = self._indexer.replace_source_records(
            vector_records,
            source_id=request.source_id,
        )

        logger.info(
            "Completed official KB sync for '{}' with {} chunk(s)",
            request.source_id,
            len(chunks),
        )
        return OfficialKnowledgeBaseSyncResult(
            request=request,
            fetched_document=fetched_document,
            normalized_document=normalized_document,
            chunks=chunks,
            embeddings=tuple(tuple(embedding) for embedding in embeddings),
            vector_records=vector_records,
            record_ids=record_ids,
        )
