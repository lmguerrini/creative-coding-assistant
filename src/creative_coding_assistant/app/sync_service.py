"""Official KB sync service layer."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from loguru import logger

from creative_coding_assistant.app.sync import (
    OfficialKnowledgeBaseBatchSyncResult,
    SyncFailureMode,
    build_official_kb_sync_runner,
    resolve_sync_source_ids,
)
from creative_coding_assistant.core import Settings
from creative_coding_assistant.rag.sync import (
    ChunkEmbedder,
    OfficialKnowledgeBaseSyncResult,
    OfficialKnowledgeBaseSyncRunner,
    OfficialSourceChunker,
    OfficialSourceNormalizer,
    SourceTransport,
    default_sync_request,
)


class OfficialKBSyncService:
    """Service wrapper for syncing approved official KB sources."""

    def __init__(
        self,
        *,
        failure_mode: SyncFailureMode = SyncFailureMode.FAIL_FAST,
        runner: OfficialKnowledgeBaseSyncRunner | None = None,
        runner_builder: Callable[[], OfficialKnowledgeBaseSyncRunner] | None = None,
    ) -> None:
        if runner is None and runner_builder is None:
            raise ValueError(
                "OfficialKBSyncService requires a runner or runner builder."
            )

        self._failure_mode = failure_mode
        self._runner = runner
        self._runner_builder = runner_builder

    def sync_all_sources(self) -> OfficialKnowledgeBaseBatchSyncResult:
        return self._sync_resolved_sources(resolve_sync_source_ids(None))

    def sync_selected_sources(
        self,
        source_ids: Sequence[str],
    ) -> OfficialKnowledgeBaseBatchSyncResult:
        return self._sync_resolved_sources(resolve_sync_source_ids(source_ids))

    def _get_runner(self) -> OfficialKnowledgeBaseSyncRunner:
        if self._runner is None:
            assert self._runner_builder is not None
            self._runner = self._runner_builder()
        return self._runner

    def _sync_resolved_sources(
        self,
        source_ids: Sequence[str],
    ) -> OfficialKnowledgeBaseBatchSyncResult:
        runner = self._get_runner()
        results: list[OfficialKnowledgeBaseSyncResult] = []
        failed_source_ids: list[str] = []

        for index, source_id in enumerate(source_ids, start=1):
            logger.info(
                "Syncing official KB source '{}' ({}/{})",
                source_id,
                index,
                len(source_ids),
            )
            try:
                results.append(runner.run(default_sync_request(source_id)))
            except Exception:
                if self._failure_mode is SyncFailureMode.FAIL_FAST:
                    raise
                logger.exception("Official KB sync failed for source '{}'", source_id)
                failed_source_ids.append(source_id)

        batch_result = OfficialKnowledgeBaseBatchSyncResult(
            source_ids=tuple(source_ids),
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


def build_official_kb_sync_service(
    *,
    settings: Settings | None = None,
    transport: SourceTransport | None = None,
    chunk_embedder: ChunkEmbedder | None = None,
    normalizer: OfficialSourceNormalizer | None = None,
    chunker: OfficialSourceChunker | None = None,
    failure_mode: SyncFailureMode = SyncFailureMode.FAIL_FAST,
    runner: OfficialKnowledgeBaseSyncRunner | None = None,
) -> OfficialKBSyncService:
    """Compose the official KB sync service from runtime settings."""

    if runner is not None:
        return OfficialKBSyncService(
            failure_mode=failure_mode,
            runner=runner,
        )

    return OfficialKBSyncService(
        failure_mode=failure_mode,
        runner_builder=lambda: build_official_kb_sync_runner(
            settings=settings,
            transport=transport,
            chunk_embedder=chunk_embedder,
            normalizer=normalizer,
            chunker=chunker,
        ),
    )
