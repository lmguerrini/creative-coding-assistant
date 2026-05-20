"""Application composition helpers."""

from creative_coding_assistant.app.bootstrap import build_assistant_service
from creative_coding_assistant.app.rebuild import (
    OfficialKnowledgeBaseRebuildPlan,
    OfficialSourceRebuildCandidate,
    RebuildReason,
    build_official_kb_rebuild_plan,
    resolve_rebuild_source_ids,
    select_stale_rebuild_source_ids,
)
from creative_coding_assistant.app.sync import (
    OfficialKnowledgeBaseBatchSyncResult,
    build_official_kb_sync_runner,
    resolve_sync_source_ids,
    sync_official_sources,
)
from creative_coding_assistant.app.sync_bootstrap import build_official_kb_sync_service
from creative_coding_assistant.app.sync_service import OfficialKBSyncService

__all__ = [
    "build_assistant_service",
    "OfficialKnowledgeBaseBatchSyncResult",
    "OfficialKnowledgeBaseRebuildPlan",
    "OfficialKBSyncService",
    "OfficialSourceRebuildCandidate",
    "RebuildReason",
    "build_official_kb_rebuild_plan",
    "build_official_kb_sync_runner",
    "build_official_kb_sync_service",
    "resolve_rebuild_source_ids",
    "resolve_sync_source_ids",
    "select_stale_rebuild_source_ids",
    "sync_official_sources",
]
