"""Official KB sync composition helpers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from enum import StrEnum

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
    failed_source_ids: tuple[str, ...] = Field(default_factory=tuple)
    total_chunks: int = Field(ge=0)
    total_records: int = Field(ge=0)

    @property
    def failed_count(self) -> int:
        return len(self.failed_source_ids)

    @property
    def succeeded_count(self) -> int:
        return len(self.results)

    def summary_payload(self) -> dict[str, object]:
        return {
            "source_ids": list(self.source_ids),
            "succeeded_source_ids": [
                result.request.source_id for result in self.results
            ],
            "failed_source_ids": list(self.failed_source_ids),
            "total_chunks": self.total_chunks,
            "total_records": self.total_records,
        }

    def summary_json(self) -> str:
        return json.dumps(self.summary_payload(), sort_keys=True)


class SyncFailureMode(StrEnum):
    FAIL_FAST = "fail_fast"
    CONTINUE = "continue"


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
    failure_mode: SyncFailureMode = SyncFailureMode.FAIL_FAST,
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
    failed_source_ids: list[str] = []
    for index, source_id in enumerate(resolved_source_ids, start=1):
        logger.info(
            "Syncing official KB source '{}' ({}/{})",
            source_id,
            index,
            len(resolved_source_ids),
        )
        try:
            results.append(resolved_runner.run(default_sync_request(source_id)))
        except Exception:
            if failure_mode is SyncFailureMode.FAIL_FAST:
                raise
            logger.exception("Official KB sync failed for source '{}'", source_id)
            failed_source_ids.append(source_id)

    batch_result = OfficialKnowledgeBaseBatchSyncResult(
        source_ids=resolved_source_ids,
        results=tuple(results),
        failed_source_ids=tuple(failed_source_ids),
        total_chunks=sum(len(result.chunks) for result in results),
        total_records=sum(len(result.record_ids) for result in results),
    )
    logger.info(
        "Completed official KB sync for {} source(s), {} success, {} failed, "
        "{} chunk(s), {} record(s)",
        len(batch_result.source_ids),
        batch_result.succeeded_count,
        batch_result.failed_count,
        batch_result.total_chunks,
        batch_result.total_records,
    )
    return batch_result
