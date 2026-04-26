"""Composition helpers for the official KB sync service."""

from __future__ import annotations

from creative_coding_assistant.app.sync import (
    SyncFailureMode,
    build_official_kb_sync_runner,
)
from creative_coding_assistant.app.sync_service import OfficialKBSyncService
from creative_coding_assistant.core import Settings
from creative_coding_assistant.rag.sync import (
    ChunkEmbedder,
    OfficialKnowledgeBaseSyncRunner,
    OfficialSourceChunker,
    OfficialSourceNormalizer,
    SourceTransport,
)


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
