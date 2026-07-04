"""Typed source freshness and health helpers for official KB observability."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.rag.sources import (
    OfficialSource,
    OfficialSourceType,
    get_official_source,
)

_HASH_PATTERN = r"^[a-f0-9]{64}$"


def _require_timezone(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Source health timestamps must be timezone-aware.")
    return value


class SourceFreshnessStatus(StrEnum):
    FRESH = "fresh"
    STALE = "stale"
    UNKNOWN = "unknown"


class SourceHealthStatus(StrEnum):
    HEALTHY = "healthy"
    STALE = "stale"
    SYNC_FAILED = "sync_failed"
    UNKNOWN = "unknown"


class SourceSyncStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class SourceFreshnessPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    warn_after_hours: int = Field(ge=1)
    stale_after_hours: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> Self:
        if self.warn_after_hours >= self.stale_after_hours:
            raise ValueError(
                "Freshness warning threshold must be lower than stale threshold."
            )
        return self

    @property
    def warn_after(self) -> timedelta:
        return timedelta(hours=self.warn_after_hours)

    @property
    def stale_after(self) -> timedelta:
        return timedelta(hours=self.stale_after_hours)

    @classmethod
    def for_source_type(cls, source_type: OfficialSourceType) -> SourceFreshnessPolicy:
        if source_type is OfficialSourceType.API_REFERENCE:
            return cls(warn_after_hours=24 * 45, stale_after_hours=24 * 120)
        if source_type is OfficialSourceType.GUIDE:
            return cls(warn_after_hours=24 * 21, stale_after_hours=24 * 60)
        if source_type is OfficialSourceType.EXAMPLES:
            return cls(warn_after_hours=24 * 14, stale_after_hours=24 * 45)
        return cls(warn_after_hours=24 * 120, stale_after_hours=24 * 365)


class OfficialSourceHealthMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str = Field(min_length=1)
    domain: CreativeCodingDomain
    source_type: OfficialSourceType
    source_url: str = Field(min_length=1)
    additional_url_count: int = Field(ge=0)
    approved_url_count: int = Field(ge=1)
    tag_count: int = Field(ge=0)
    freshness_policy: SourceFreshnessPolicy


class OfficialSourceSyncMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source_id: str = Field(min_length=1)
    domain: CreativeCodingDomain
    source_type: OfficialSourceType
    source_url: str = Field(min_length=1)
    resolved_url: str | None = None
    sync_status: SourceSyncStatus
    requested_at: datetime
    last_synced_at: datetime | None = None
    completed_at: datetime | None = None
    content_hash: str | None = Field(default=None, pattern=_HASH_PATTERN)
    chunk_count: int = Field(default=0, ge=0)
    record_count: int = Field(default=0, ge=0)
    warnings: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("requested_at", "last_synced_at", "completed_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        return _require_timezone(value)

    @model_validator(mode="after")
    def validate_status_payload(self) -> Self:
        if self.sync_status is SourceSyncStatus.SUCCEEDED:
            if self.last_synced_at is None:
                raise ValueError(
                    "Successful source sync metadata requires a last_synced_at value."
                )
            if self.completed_at is None:
                raise ValueError(
                    "Successful source sync metadata requires a completed_at value."
                )
            if self.content_hash is None:
                raise ValueError(
                    "Successful source sync metadata requires a content hash."
                )
        return self

    @classmethod
    def from_success(
        cls,
        *,
        source_id: str,
        requested_at: datetime,
        last_synced_at: datetime,
        source_url: str,
        resolved_url: str,
        domain: CreativeCodingDomain,
        source_type: OfficialSourceType,
        content_hash: str,
        chunk_count: int,
        record_count: int,
        completed_at: datetime | None = None,
        warnings: tuple[str, ...] = (),
    ) -> OfficialSourceSyncMetadata:
        return cls(
            source_id=source_id,
            domain=domain,
            source_type=source_type,
            source_url=source_url,
            resolved_url=resolved_url,
            sync_status=SourceSyncStatus.SUCCEEDED,
            requested_at=requested_at,
            last_synced_at=last_synced_at,
            completed_at=completed_at or last_synced_at,
            content_hash=content_hash,
            chunk_count=chunk_count,
            record_count=record_count,
            warnings=warnings,
        )


class OfficialSourceHealthSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: OfficialSourceHealthMetadata
    sync: OfficialSourceSyncMetadata | None = None
    checked_at: datetime
    freshness_status: SourceFreshnessStatus
    health_status: SourceHealthStatus
    refresh_recommended: bool = False
    stale_after: datetime | None = None
    age_hours: float | None = None

    @field_validator("checked_at", "stale_after")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        return _require_timezone(value)

    @property
    def is_stale(self) -> bool:
        return self.freshness_status is SourceFreshnessStatus.STALE


def build_official_source_health_metadata(
    source: OfficialSource | str,
) -> OfficialSourceHealthMetadata:
    resolved_source = get_official_source(source) if isinstance(source, str) else source
    approved_url_count = 1 + len(resolved_source.additional_urls)
    return OfficialSourceHealthMetadata(
        source_id=resolved_source.source_id,
        domain=resolved_source.domain,
        source_type=resolved_source.source_type,
        source_url=resolved_source.url,
        additional_url_count=len(resolved_source.additional_urls),
        approved_url_count=approved_url_count,
        tag_count=len(resolved_source.tags),
        freshness_policy=SourceFreshnessPolicy.for_source_type(
            resolved_source.source_type
        ),
    )


def evaluate_official_source_health(
    source: OfficialSource | str,
    *,
    sync_metadata: OfficialSourceSyncMetadata | None,
    checked_at: datetime,
) -> OfficialSourceHealthSnapshot:
    checked_at = _validated_checked_at(checked_at)
    metadata = build_official_source_health_metadata(source)

    if sync_metadata is None:
        return OfficialSourceHealthSnapshot(
            source=metadata,
            sync=None,
            checked_at=checked_at,
            freshness_status=SourceFreshnessStatus.UNKNOWN,
            health_status=SourceHealthStatus.UNKNOWN,
        )

    if sync_metadata.sync_status is SourceSyncStatus.FAILED:
        return OfficialSourceHealthSnapshot(
            source=metadata,
            sync=sync_metadata,
            checked_at=checked_at,
            freshness_status=SourceFreshnessStatus.UNKNOWN,
            health_status=SourceHealthStatus.SYNC_FAILED,
        )

    assert sync_metadata.last_synced_at is not None
    warn_after = sync_metadata.last_synced_at + metadata.freshness_policy.warn_after
    stale_after = sync_metadata.last_synced_at + metadata.freshness_policy.stale_after
    age_hours = max(
        0.0,
        (checked_at - sync_metadata.last_synced_at).total_seconds() / 3600,
    )
    is_stale = checked_at >= stale_after
    return OfficialSourceHealthSnapshot(
        source=metadata,
        sync=sync_metadata,
        checked_at=checked_at,
        freshness_status=(
            SourceFreshnessStatus.STALE if is_stale else SourceFreshnessStatus.FRESH
        ),
        health_status=(
            SourceHealthStatus.STALE if is_stale else SourceHealthStatus.HEALTHY
        ),
        refresh_recommended=checked_at >= warn_after,
        stale_after=stale_after,
        age_hours=round(age_hours, 2),
    )


def _validated_checked_at(value: datetime) -> datetime:
    checked_at = _require_timezone(value)
    assert checked_at is not None
    return checked_at
