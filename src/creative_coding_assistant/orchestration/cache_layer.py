"""V5.1 bounded in-memory execution cache layer."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

ExecutionCacheNamespace = Literal[
    "context",
    "prompt",
    "retrieval",
    "memory",
    "workflow",
    "generic",
]
ExecutionCacheStatus = Literal["hit", "miss", "stale"]
ExecutionCachePayload = dict[str, str | int | float | bool | None]

EXECUTION_CACHE_ENTRY_SERIALIZATION_VERSION = "execution_cache_entry.v1"
EXECUTION_CACHE_LOOKUP_SERIALIZATION_VERSION = "execution_cache_lookup.v1"
EXECUTION_CACHE_AUTHORITY_BOUNDARY = (
    "The execution cache layer stores deterministic in-memory cache entries for "
    "the current process only; it does not write persistent storage, use network "
    "caches, route providers or models, control workflows, write memory, or "
    "modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "persistent_storage_write",
    "network_cache_access",
    "provider_or_model_routing",
    "workflow_control",
    "memory_write",
    "generated_output_modification",
)


class ExecutionCacheEntry(BaseModel):
    """One deterministic in-memory cache entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    namespace: ExecutionCacheNamespace
    cache_key: str = Field(min_length=1, max_length=120)
    payload: ExecutionCachePayload
    payload_fingerprint: str = Field(min_length=16, max_length=64)
    tags: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    created_at: datetime
    expires_at: datetime | None = None
    ttl_seconds: int | None = Field(default=None, ge=1, le=86_400)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    cache_layer_implemented: Literal[True] = True
    persistent_storage_write_implemented: Literal[False] = False
    network_cache_access_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["execution_cache_entry.v1"] = (
        EXECUTION_CACHE_ENTRY_SERIALIZATION_VERSION
    )
    in_memory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_payload(self) -> Self:
        if self.payload_fingerprint != _payload_fingerprint(self.payload):
            raise ValueError("payload_fingerprint must match payload")
        if self.ttl_seconds is None and self.expires_at is not None:
            raise ValueError("expires_at requires ttl_seconds")
        if self.ttl_seconds is not None and self.expires_at is None:
            raise ValueError("ttl_seconds requires expires_at")
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("expires_at must be after created_at")
        return self


class ExecutionCacheLookup(BaseModel):
    """Cache lookup result with explicit hit/miss/stale state."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["execution_cache_layer"] = "execution_cache_layer"
    serialization_version: Literal["execution_cache_lookup.v1"] = (
        EXECUTION_CACHE_LOOKUP_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=EXECUTION_CACHE_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    namespace: ExecutionCacheNamespace
    cache_key: str = Field(min_length=1, max_length=120)
    status: ExecutionCacheStatus
    entry: ExecutionCacheEntry | None = None
    stale_entry: ExecutionCacheEntry | None = None
    looked_up_at: datetime
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    cache_layer_implemented: Literal[True] = True
    persistent_storage_write_implemented: Literal[False] = False
    network_cache_access_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    in_memory_only: Literal[True] = True

    @model_validator(mode="after")
    def _lookup_matches_status(self) -> Self:
        if self.status == "hit" and self.entry is None:
            raise ValueError("hit lookup requires entry")
        if self.status == "hit" and self.stale_entry is not None:
            raise ValueError("hit lookup must not include stale_entry")
        if self.status == "miss" and (self.entry is not None or self.stale_entry is not None):
            raise ValueError("miss lookup must not include entries")
        if self.status == "stale" and self.stale_entry is None:
            raise ValueError("stale lookup requires stale_entry")
        if self.status == "stale" and self.entry is not None:
            raise ValueError("stale lookup must not include live entry")
        return self


class InMemoryExecutionCache:
    """Small process-local execution cache with deterministic keys."""

    def __init__(self) -> None:
        self._entries: dict[str, ExecutionCacheEntry] = {}

    def put(
        self,
        *,
        namespace: ExecutionCacheNamespace,
        components: ExecutionCachePayload,
        payload: ExecutionCachePayload,
        ttl_seconds: int | None = None,
        tags: tuple[str, ...] = (),
        now: datetime | None = None,
    ) -> ExecutionCacheEntry:
        created_at = now or _now()
        expires_at = (
            created_at + timedelta(seconds=ttl_seconds)
            if ttl_seconds is not None
            else None
        )
        entry = ExecutionCacheEntry(
            namespace=namespace,
            cache_key=build_execution_cache_key(
                namespace=namespace,
                components=components,
            ),
            payload=payload,
            payload_fingerprint=_payload_fingerprint(payload),
            tags=tags,
            created_at=created_at,
            expires_at=expires_at,
            ttl_seconds=ttl_seconds,
        )
        self._entries[entry.cache_key] = entry
        return entry

    def get(
        self,
        *,
        namespace: ExecutionCacheNamespace,
        components: ExecutionCachePayload,
        now: datetime | None = None,
    ) -> ExecutionCacheLookup:
        looked_up_at = now or _now()
        cache_key = build_execution_cache_key(namespace=namespace, components=components)
        entry = self._entries.get(cache_key)
        if entry is None:
            return _lookup(
                namespace=namespace,
                cache_key=cache_key,
                status="miss",
                looked_up_at=looked_up_at,
            )
        if entry.expires_at is not None and entry.expires_at <= looked_up_at:
            return _lookup(
                namespace=namespace,
                cache_key=cache_key,
                status="stale",
                stale_entry=entry,
                looked_up_at=looked_up_at,
            )
        return _lookup(
            namespace=namespace,
            cache_key=cache_key,
            status="hit",
            entry=entry,
            looked_up_at=looked_up_at,
        )

    def invalidate(
        self,
        *,
        namespace: ExecutionCacheNamespace,
        components: ExecutionCachePayload,
    ) -> bool:
        cache_key = build_execution_cache_key(namespace=namespace, components=components)
        return self._entries.pop(cache_key, None) is not None

    def snapshot(self) -> tuple[ExecutionCacheEntry, ...]:
        return tuple(self._entries.values())


def build_execution_cache_key(
    *,
    namespace: ExecutionCacheNamespace,
    components: ExecutionCachePayload,
) -> str:
    """Build a stable cache key without reading or writing storage."""

    digest = hashlib.sha256(_canonical_json(components).encode("utf-8")).hexdigest()
    return f"cache::{namespace}::{digest[:32]}"


def execution_cache_entry_is_fresh(
    entry: ExecutionCacheEntry,
    *,
    now: datetime | None = None,
) -> bool:
    """Return cache freshness without mutating cache state."""

    if entry.expires_at is None:
        return True
    return entry.expires_at > (now or _now())


def _lookup(
    *,
    namespace: ExecutionCacheNamespace,
    cache_key: str,
    status: ExecutionCacheStatus,
    looked_up_at: datetime,
    entry: ExecutionCacheEntry | None = None,
    stale_entry: ExecutionCacheEntry | None = None,
) -> ExecutionCacheLookup:
    return ExecutionCacheLookup(
        namespace=namespace,
        cache_key=cache_key,
        status=status,
        entry=entry,
        stale_entry=stale_entry,
        looked_up_at=looked_up_at,
        advisory_actions=_lookup_actions(status),
    )


def _lookup_actions(status: ExecutionCacheStatus) -> tuple[str, ...]:
    if status == "hit":
        return ("Use in-memory cached payload only within current process.",)
    if status == "stale":
        return ("Treat stale cache entry as metadata and recompute upstream value.",)
    return ("Compute upstream value before writing a process-local cache entry.",)


def _payload_fingerprint(payload: ExecutionCachePayload) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:32]


def _canonical_json(payload: ExecutionCachePayload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _now() -> datetime:
    return datetime.now(UTC)
