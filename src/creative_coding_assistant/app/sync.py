"""Official KB sync composition helpers."""

from __future__ import annotations

from collections.abc import Sequence

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.core import Settings, load_settings
from creative_coding_assistant.rag.sources import (
    approved_official_sources,
    get_official_source,
)
from creative_coding_assistant.rag.sync import (
    ChunkEmbedder,
    OfficialKnowledgeBaseIndexer,
    OfficialKnowledgeBaseSyncResult,
    OfficialKnowledgeBaseSyncRunner,
    OfficialSourceChunker,
    OfficialSourceFetcher,
    OfficialSourceNormalizer,
    SourceTransport,
    UrllibSourceTransport,
    build_chunk_embedder,
    default_sync_request,
)
from creative_coding_assistant.vectorstore import (
    create_chroma_client,
    ensure_project_collections,
)


class OfficialKnowledgeBaseBatchSyncResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    results: tuple[OfficialKnowledgeBaseSyncResult, ...] = Field(default_factory=tuple)
    total_chunks: int = Field(ge=0)
    total_records: int = Field(ge=0)


def resolve_sync_source_ids(source_ids: Sequence[str] | None) -> tuple[str, ...]:
    """Resolve optional user selections into approved official source IDs."""

    if not source_ids:
        return tuple(source.source_id for source in approved_official_sources())

    resolved: list[str] = []
    for raw_source_id in source_ids:
        source_id = str(raw_source_id).strip()
        source = get_official_source(source_id)
        if source.source_id not in resolved:
            resolved.append(source.source_id)
    return tuple(resolved)


def build_official_kb_sync_runner(
    *,
    settings: Settings | None = None,
    transport: SourceTransport | None = None,
    chunk_embedder: ChunkEmbedder | None = None,
    normalizer: OfficialSourceNormalizer | None = None,
    chunker: OfficialSourceChunker | None = None,
) -> OfficialKnowledgeBaseSyncRunner:
    """Compose the official KB sync runner from runtime settings."""

    resolved_settings = settings or load_settings()
    resolved_chunk_embedder = (
        chunk_embedder
        if chunk_embedder is not None
        else build_chunk_embedder(resolved_settings)
    )
    if resolved_chunk_embedder is None:
        raise RuntimeError(
            "Official KB sync requires OpenAI embedding configuration. "
            "Set OPENAI_API_KEY or CCA_OPENAI_API_KEY."
        )

    client = create_chroma_client(settings=resolved_settings)
    ensure_project_collections(client)
    return OfficialKnowledgeBaseSyncRunner(
        fetcher=OfficialSourceFetcher(transport=transport or UrllibSourceTransport()),
        normalizer=normalizer,
        chunker=chunker,
        embedder=resolved_chunk_embedder,
        indexer=OfficialKnowledgeBaseIndexer(client=client),
    )


def sync_official_sources(
    *,
    source_ids: Sequence[str] | None = None,
    settings: Settings | None = None,
    transport: SourceTransport | None = None,
    chunk_embedder: ChunkEmbedder | None = None,
    runner: OfficialKnowledgeBaseSyncRunner | None = None,
) -> OfficialKnowledgeBaseBatchSyncResult:
    """Run the official KB sync pipeline for approved sources."""

    resolved_source_ids = resolve_sync_source_ids(source_ids)
    resolved_runner = (
        runner
        if runner is not None
        else build_official_kb_sync_runner(
            settings=settings,
            transport=transport,
            chunk_embedder=chunk_embedder,
        )
    )

    results: list[OfficialKnowledgeBaseSyncResult] = []
    for index, source_id in enumerate(resolved_source_ids, start=1):
        logger.info(
            "Syncing official KB source '{}' ({}/{})",
            source_id,
            index,
            len(resolved_source_ids),
        )
        results.append(resolved_runner.run(default_sync_request(source_id)))

    batch_result = OfficialKnowledgeBaseBatchSyncResult(
        source_ids=resolved_source_ids,
        results=tuple(results),
        total_chunks=sum(len(result.chunks) for result in results),
        total_records=sum(len(result.record_ids) for result in results),
    )
    logger.info(
        "Completed official KB sync for {} source(s), {} chunk(s), {} record(s)",
        len(batch_result.source_ids),
        batch_result.total_chunks,
        batch_result.total_records,
    )
    return batch_result
