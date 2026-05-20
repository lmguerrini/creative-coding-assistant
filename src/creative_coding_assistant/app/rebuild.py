"""Selective KB rebuild planning helpers built on source-health metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.rag import (
    OfficialSourceHealthSnapshot,
    OfficialSourceSyncMetadata,
    SourceFreshnessStatus,
    SourceHealthStatus,
    SourceSyncStatus,
    evaluate_official_source_health,
)
from creative_coding_assistant.rag.sources import (
    approved_official_sources,
    get_official_source,
)


class RebuildReason(StrEnum):
    EXPLICIT_SOURCE = "explicit_source"
    EXPLICIT_DOMAIN = "explicit_domain"
    STALE = "stale"
    REFRESH_RECOMMENDED = "refresh_recommended"
    SYNC_FAILED = "sync_failed"


class OfficialSourceRebuildCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str = Field(min_length=1)
    domain: CreativeCodingDomain
    reasons: tuple[RebuildReason, ...] = Field(min_length=1)
    health_status: SourceHealthStatus
    freshness_status: SourceFreshnessStatus
    refresh_recommended: bool = False
    last_synced_at: datetime | None = None

    @field_validator("reasons", mode="before")
    @classmethod
    def normalize_reasons(
        cls,
        value: Sequence[RebuildReason | str] | RebuildReason | str,
    ) -> tuple[RebuildReason, ...]:
        if isinstance(value, RebuildReason):
            return (value,)
        if isinstance(value, str):
            return (RebuildReason(value.strip()),)

        normalized: list[RebuildReason] = []
        for item in value:
            reason = (
                item if isinstance(item, RebuildReason) else RebuildReason(str(item))
            )
            if reason not in normalized:
                normalized.append(reason)
        return tuple(normalized)


class OfficialKnowledgeBaseRebuildPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    checked_at: datetime
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    affected_domains: tuple[CreativeCodingDomain, ...] = Field(default_factory=tuple)
    candidates: tuple[OfficialSourceRebuildCandidate, ...] = Field(
        default_factory=tuple
    )
    explicit_source_ids: tuple[str, ...] = Field(default_factory=tuple)
    explicit_domains: tuple[CreativeCodingDomain, ...] = Field(default_factory=tuple)
    stale_only: bool = False
    include_refresh_recommended: bool = True
    include_sync_failed: bool = True

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    def source_ids_for_reason(self, reason: RebuildReason) -> tuple[str, ...]:
        return tuple(
            candidate.source_id
            for candidate in self.candidates
            if reason in candidate.reasons
        )

    def summary_payload(self) -> dict[str, object]:
        return {
            "source_ids": list(self.source_ids),
            "affected_domains": [domain.value for domain in self.affected_domains],
            "explicit_source_ids": list(self.explicit_source_ids),
            "explicit_domains": [domain.value for domain in self.explicit_domains],
            "stale_only": self.stale_only,
            "candidate_count": self.candidate_count,
            "stale_source_ids": list(self.source_ids_for_reason(RebuildReason.STALE)),
            "sync_failed_source_ids": list(
                self.source_ids_for_reason(RebuildReason.SYNC_FAILED)
            ),
            "refresh_recommended_source_ids": list(
                self.source_ids_for_reason(RebuildReason.REFRESH_RECOMMENDED)
            ),
        }


def build_official_kb_rebuild_plan(
    *,
    source_ids: Sequence[str] | None = None,
    domains: Sequence[CreativeCodingDomain | str] | None = None,
    sync_metadata_by_source: Mapping[str, OfficialSourceSyncMetadata] | None = None,
    checked_at: datetime | None = None,
    stale_only: bool = False,
    include_refresh_recommended: bool = True,
    include_sync_failed: bool = True,
) -> OfficialKnowledgeBaseRebuildPlan:
    resolved_checked_at = checked_at or datetime.now(tz=UTC)
    explicit_source_ids = _normalize_source_ids(source_ids)
    explicit_domains = _normalize_domains(domains)
    metadata_by_source = dict(sync_metadata_by_source or {})

    candidates: list[OfficialSourceRebuildCandidate] = []
    affected_domains: list[CreativeCodingDomain] = []

    for source in approved_official_sources():
        snapshot = evaluate_official_source_health(
            source,
            sync_metadata=metadata_by_source.get(source.source_id),
            checked_at=resolved_checked_at,
        )
        reasons = _candidate_reasons(
            source_id=source.source_id,
            domain=source.domain,
            snapshot=snapshot,
            explicit_source_ids=explicit_source_ids,
            explicit_domains=explicit_domains,
            stale_only=stale_only,
            include_refresh_recommended=include_refresh_recommended,
            include_sync_failed=include_sync_failed,
        )
        if not reasons:
            continue

        candidates.append(
            OfficialSourceRebuildCandidate(
                source_id=source.source_id,
                domain=source.domain,
                reasons=reasons,
                health_status=snapshot.health_status,
                freshness_status=snapshot.freshness_status,
                refresh_recommended=snapshot.refresh_recommended,
                last_synced_at=(
                    snapshot.sync.last_synced_at if snapshot.sync is not None else None
                ),
            )
        )
        if source.domain not in affected_domains:
            affected_domains.append(source.domain)

    return OfficialKnowledgeBaseRebuildPlan(
        checked_at=resolved_checked_at,
        source_ids=tuple(candidate.source_id for candidate in candidates),
        affected_domains=tuple(affected_domains),
        candidates=tuple(candidates),
        explicit_source_ids=explicit_source_ids,
        explicit_domains=explicit_domains,
        stale_only=stale_only,
        include_refresh_recommended=include_refresh_recommended,
        include_sync_failed=include_sync_failed,
    )


def resolve_rebuild_source_ids(
    *,
    source_ids: Sequence[str] | None = None,
    domains: Sequence[CreativeCodingDomain | str] | None = None,
    sync_metadata_by_source: Mapping[str, OfficialSourceSyncMetadata] | None = None,
    checked_at: datetime | None = None,
    stale_only: bool = False,
    include_refresh_recommended: bool = True,
    include_sync_failed: bool = True,
) -> tuple[str, ...]:
    plan = build_official_kb_rebuild_plan(
        source_ids=source_ids,
        domains=domains,
        sync_metadata_by_source=sync_metadata_by_source,
        checked_at=checked_at,
        stale_only=stale_only,
        include_refresh_recommended=include_refresh_recommended,
        include_sync_failed=include_sync_failed,
    )
    return plan.source_ids


def select_stale_rebuild_source_ids(
    *,
    domains: Sequence[CreativeCodingDomain | str] | None = None,
    sync_metadata_by_source: Mapping[str, OfficialSourceSyncMetadata],
    checked_at: datetime | None = None,
    include_sync_failed: bool = True,
) -> tuple[str, ...]:
    return resolve_rebuild_source_ids(
        domains=domains,
        sync_metadata_by_source=sync_metadata_by_source,
        checked_at=checked_at,
        stale_only=True,
        include_refresh_recommended=False,
        include_sync_failed=include_sync_failed,
    )


def _normalize_source_ids(source_ids: Sequence[str] | None) -> tuple[str, ...]:
    if not source_ids:
        return ()

    normalized: list[str] = []
    for raw_source_id in source_ids:
        source = get_official_source(str(raw_source_id).strip())
        if source.source_id not in normalized:
            normalized.append(source.source_id)
    return tuple(normalized)


def _normalize_domains(
    domains: Sequence[CreativeCodingDomain | str] | None,
) -> tuple[CreativeCodingDomain, ...]:
    if not domains:
        return ()

    normalized: list[CreativeCodingDomain] = []
    for item in domains:
        domain = (
            item
            if isinstance(item, CreativeCodingDomain)
            else CreativeCodingDomain(str(item).strip())
        )
        if domain not in normalized:
            normalized.append(domain)
    return tuple(normalized)


def _candidate_reasons(
    *,
    source_id: str,
    domain: CreativeCodingDomain,
    snapshot: OfficialSourceHealthSnapshot,
    explicit_source_ids: tuple[str, ...],
    explicit_domains: tuple[CreativeCodingDomain, ...],
    stale_only: bool,
    include_refresh_recommended: bool,
    include_sync_failed: bool,
) -> tuple[RebuildReason, ...]:
    reasons: list[RebuildReason] = []

    if source_id in explicit_source_ids and not stale_only:
        reasons.append(RebuildReason.EXPLICIT_SOURCE)
    if domain in explicit_domains and not stale_only:
        reasons.append(RebuildReason.EXPLICIT_DOMAIN)

    if snapshot.health_status is SourceHealthStatus.STALE:
        reasons.append(RebuildReason.STALE)
    elif (
        include_sync_failed
        and snapshot.sync is not None
        and snapshot.sync.sync_status is SourceSyncStatus.FAILED
    ):
        reasons.append(RebuildReason.SYNC_FAILED)
    elif (
        include_refresh_recommended
        and snapshot.refresh_recommended
        and not stale_only
    ):
        reasons.append(RebuildReason.REFRESH_RECOMMENDED)

    return tuple(reasons)
