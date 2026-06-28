"""Passive V4.6 blackboard audit metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_identities import (
    agent_identity_registry,
)
from creative_coding_assistant.orchestration.agent_memory_contracts import (
    agent_memory_contract_registry,
)
from creative_coding_assistant.orchestration.blackboard_memory import (
    BLACKBOARD_MEMORY_REGISTRY,
    BlackboardAgentPermissionContract,
    BlackboardMemoryChannelContract,
)

BlackboardAuditStage = Literal["v4_6_blackboard_hardening"]
BlackboardAuditStatus = Literal["pass"]

BLACKBOARD_AUDIT_RECORD_SERIALIZATION_VERSION = "blackboard_audit_record.v1"
BLACKBOARD_AUDIT_REGISTRY_SERIALIZATION_VERSION = "blackboard_audit_registry.v1"
BLACKBOARD_AUDIT_REGISTRY_AUTHORITY_BOUNDARY = (
    "V4.6 blackboard audit metadata checks passive blackboard channel and "
    "permission coverage, source memory contract alignment, metadata key "
    "alignment, persistence flags, storage boundaries, runtime read/write "
    "blocks, and mutation blocks only; it does not create storage, persist "
    "records, read runtime blackboard state, write runtime blackboard state, "
    "mutate memory, materialize shared context, invoke agents, route providers "
    "or models, or modify generated output."
)

_VALIDATED_BLACKBOARD_SURFACES = (
    "channel_permission_alignment",
    "source_memory_contract_alignment",
    "metadata_key_alignment",
    "persistence_flags",
    "storage_boundary",
    "runtime_read_write_flags",
    "mutation_boundary",
)
_PASSIVE_BOUNDARY_FLAGS = (
    "persistence_blocked",
    "storage_backend_blocked",
    "database_schema_blocked",
    "runtime_read_blocked",
    "runtime_write_blocked",
    "memory_mutation_blocked",
    "shared_context_materialization_blocked",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "blackboard_persistence",
    "runtime_blackboard_materialization",
    "runtime_blackboard_mutation",
    "storage_backend_creation",
    "database_schema_creation",
    "memory_record_writes",
    "shared_context_view_materialization",
    "agent_invocation",
    "provider_or_model_routing",
    "generated_output_modification",
)
_AUDIT_FINDINGS = (
    "channel_permission_alignment_confirmed",
    "source_memory_contract_alignment_confirmed",
    "metadata_key_alignment_confirmed",
    "persistence_storage_blocks_confirmed",
    "runtime_read_write_blocks_confirmed",
    "memory_mutation_blocks_confirmed",
)


class BlackboardAuditRecord(BaseModel):
    """One passive V4.6 audit record for a blackboard channel/permission pair."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=80)
    channel_id: str = Field(min_length=1, max_length=120)
    permission_id: str = Field(min_length=1, max_length=140)
    owner_role_family: str = Field(min_length=1, max_length=80)
    audit_stage: BlackboardAuditStage = "v4_6_blackboard_hardening"
    channel_serialization_version: str = Field(min_length=1, max_length=80)
    permission_serialization_version: str = Field(min_length=1, max_length=80)
    source_memory_contract_id: str = Field(min_length=1, max_length=120)
    source_registries: tuple[str, ...] = Field(min_length=2, max_length=2)
    metadata_keys: tuple[str, ...] = Field(min_length=1, max_length=12)
    readable_channel_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    writable_channel_ids: tuple[str, ...] = Field(min_length=1, max_length=1)
    referenceable_channel_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    validated_blackboard_surfaces: tuple[str, ...] = Field(
        min_length=7,
        max_length=7,
    )
    passive_boundary_flags: tuple[str, ...] = Field(min_length=7, max_length=7)
    audit_findings: tuple[str, ...] = Field(min_length=6, max_length=6)
    missing_coverage_items: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    channel_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=16,
    )
    permission_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=16,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    audit_status: BlackboardAuditStatus = "pass"
    metadata_only_declared: Literal[True] = True
    persistence_implemented: Literal[False] = False
    storage_backend_implemented: Literal[False] = False
    database_schema_implemented: Literal[False] = False
    runtime_read_implemented: Literal[False] = False
    runtime_write_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    shared_context_materialization_implemented: Literal[False] = False
    serialization_version: Literal["blackboard_audit_record.v1"] = (
        BLACKBOARD_AUDIT_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class BlackboardAuditRegistry(BaseModel):
    """Stable passive V4.6 audit registry for blackboard metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["blackboard_audit_registry"] = "blackboard_audit_registry"
    serialization_version: Literal["blackboard_audit_registry.v1"] = (
        BLACKBOARD_AUDIT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=BLACKBOARD_AUDIT_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    audit_stage: BlackboardAuditStage = "v4_6_blackboard_hardening"
    audit_records: tuple[BlackboardAuditRecord, ...] = Field(
        min_length=12,
        max_length=12,
    )
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    channel_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    audit_count: int = Field(ge=12, le=12)
    source_blackboard_registry: Literal["blackboard_memory_registry"] = (
        "blackboard_memory_registry"
    )
    source_registries: tuple[str, ...] = Field(min_length=2, max_length=2)
    validated_blackboard_surfaces: tuple[str, ...] = Field(
        min_length=7,
        max_length=7,
    )
    passive_boundary_flags: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    all_channels_covered: Literal[True] = True
    all_permissions_covered: Literal[True] = True
    no_missing_coverage: Literal[True] = True
    active_runtime_audit_implemented: Literal[False] = False
    persistence_implemented: Literal[False] = False
    storage_backend_implemented: Literal[False] = False
    runtime_read_implemented: Literal[False] = False
    runtime_write_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_audit_records(self) -> Self:
        derived_agent_ids = tuple(record.agent_id for record in self.audit_records)
        derived_channel_ids = tuple(record.channel_id for record in self.audit_records)
        if len(set(derived_agent_ids)) != len(derived_agent_ids):
            raise ValueError("agent_ids must be unique")
        if len(set(derived_channel_ids)) != len(derived_channel_ids):
            raise ValueError("channel_ids must be unique")
        if self.agent_ids != derived_agent_ids:
            raise ValueError("agent_ids must match audit records")
        if self.channel_ids != derived_channel_ids:
            raise ValueError("channel_ids must match audit records")
        if self.audit_count != len(self.audit_records):
            raise ValueError("audit_count must match audit records")
        for record in self.audit_records:
            if record.source_registries != self.source_registries:
                raise ValueError("source_registries must match registry")
            if record.validated_blackboard_surfaces != self.validated_blackboard_surfaces:
                raise ValueError("validated_blackboard_surfaces must match registry")
            if record.passive_boundary_flags != self.passive_boundary_flags:
                raise ValueError("passive_boundary_flags must match registry")
            if record.missing_coverage_items:
                raise ValueError("audit records must not contain missing coverage")
        return self


def blackboard_audit_registry() -> BlackboardAuditRegistry:
    """Return passive V4.6 blackboard audit metadata."""

    return BLACKBOARD_AUDIT_REGISTRY


def blackboard_audit_by_agent_id(
    agent_id: str,
    registry: BlackboardAuditRegistry | None = None,
) -> BlackboardAuditRecord | None:
    """Return one passive blackboard audit record by agent id."""

    source_registry = registry or BLACKBOARD_AUDIT_REGISTRY
    for record in source_registry.audit_records:
        if record.agent_id == agent_id:
            return record
    return None


def blackboard_audit_by_channel_id(
    channel_id: str,
    registry: BlackboardAuditRegistry | None = None,
) -> BlackboardAuditRecord | None:
    """Return one passive blackboard audit record by channel id."""

    source_registry = registry or BLACKBOARD_AUDIT_REGISTRY
    for record in source_registry.audit_records:
        if record.channel_id == channel_id:
            return record
    return None


def blackboard_audits_for_source_registry(
    source_registry_ref: str,
    registry: BlackboardAuditRegistry | None = None,
) -> tuple[BlackboardAuditRecord, ...]:
    """Return passive blackboard audit records referencing one source registry."""

    source_registry = registry or BLACKBOARD_AUDIT_REGISTRY
    normalized_ref = str(source_registry_ref).strip()
    return tuple(
        record
        for record in source_registry.audit_records
        if normalized_ref in record.source_registries
    )


def _missing_coverage_items(
    channel: BlackboardMemoryChannelContract,
    permission: BlackboardAgentPermissionContract,
) -> tuple[str, ...]:
    missing: list[str] = []
    memory_registry = agent_memory_contract_registry()
    identity_registry = agent_identity_registry()
    if channel.owner_agent_id not in identity_registry.agent_ids:
        missing.append("owner_agent_identity_missing")
    if permission.agent_id not in memory_registry.agent_ids:
        missing.append("permission_memory_contract_missing")
    if permission.agent_id != channel.owner_agent_id:
        missing.append("permission_channel_owner_mismatch")
    if permission.source_memory_contract_id != channel.source_memory_contract_id:
        missing.append("source_memory_contract_id_mismatch")
    if permission.writable_channel_ids != (channel.channel_id,):
        missing.append("writable_channel_owner_mismatch")
    if channel.permitted_writer_agent_ids != (permission.agent_id,):
        missing.append("permitted_writer_agent_mismatch")
    if permission.readable_channel_ids != BLACKBOARD_MEMORY_REGISTRY.channel_ids:
        missing.append("readable_channel_coverage_mismatch")
    if permission.referenceable_channel_ids != BLACKBOARD_MEMORY_REGISTRY.channel_ids:
        missing.append("referenceable_channel_coverage_mismatch")
    if channel.metadata_keys != permission.writable_metadata_keys:
        missing.append("metadata_key_alignment_missing")
    if channel.persistence_implemented or permission.persists_blackboard_records:
        missing.append("persistence_enabled")
    if channel.storage_backend_implemented or permission.creates_storage_backend:
        missing.append("storage_backend_enabled")
    if channel.runtime_read_implemented or permission.reads_runtime_blackboard:
        missing.append("runtime_read_enabled")
    if channel.runtime_write_implemented or permission.writes_runtime_blackboard:
        missing.append("runtime_write_enabled")
    if channel.memory_mutation_implemented or permission.mutates_shared_context:
        missing.append("memory_mutation_enabled")
    for blocked_behavior in (
        "storage_backend_creation",
        "runtime_blackboard_mutation",
        "generated_output_modification",
    ):
        if blocked_behavior not in channel.blocked_runtime_behaviors:
            missing.append(f"channel_{blocked_behavior}_block_missing")
        if blocked_behavior not in permission.blocked_runtime_behaviors:
            missing.append(f"permission_{blocked_behavior}_block_missing")
    return tuple(missing)


def _audit_record(
    channel: BlackboardMemoryChannelContract,
    permission: BlackboardAgentPermissionContract,
) -> BlackboardAuditRecord:
    return BlackboardAuditRecord(
        agent_id=permission.agent_id,
        channel_id=channel.channel_id,
        permission_id=permission.permission_id,
        owner_role_family=channel.owner_role_family,
        channel_serialization_version=channel.serialization_version,
        permission_serialization_version=permission.serialization_version,
        source_memory_contract_id=channel.source_memory_contract_id,
        source_registries=BLACKBOARD_MEMORY_REGISTRY.source_registries,
        metadata_keys=channel.metadata_keys,
        readable_channel_ids=permission.readable_channel_ids,
        writable_channel_ids=permission.writable_channel_ids,
        referenceable_channel_ids=permission.referenceable_channel_ids,
        validated_blackboard_surfaces=_VALIDATED_BLACKBOARD_SURFACES,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
        audit_findings=_AUDIT_FINDINGS,
        missing_coverage_items=_missing_coverage_items(channel, permission),
        channel_blocked_runtime_behaviors=channel.blocked_runtime_behaviors,
        permission_blocked_runtime_behaviors=permission.blocked_runtime_behaviors,
        metadata_only_declared=channel.metadata_only and permission.metadata_only,
    )


_PERMISSIONS_BY_AGENT_ID = {
    permission.agent_id: permission
    for permission in BLACKBOARD_MEMORY_REGISTRY.permissions
}
BLACKBOARD_AUDIT_RECORDS = tuple(
    _audit_record(channel, _PERMISSIONS_BY_AGENT_ID[channel.owner_agent_id])
    for channel in BLACKBOARD_MEMORY_REGISTRY.channels
)
BLACKBOARD_AUDIT_REGISTRY = BlackboardAuditRegistry(
    audit_records=BLACKBOARD_AUDIT_RECORDS,
    agent_ids=tuple(record.agent_id for record in BLACKBOARD_AUDIT_RECORDS),
    channel_ids=tuple(record.channel_id for record in BLACKBOARD_AUDIT_RECORDS),
    audit_count=len(BLACKBOARD_AUDIT_RECORDS),
    source_registries=BLACKBOARD_MEMORY_REGISTRY.source_registries,
    validated_blackboard_surfaces=_VALIDATED_BLACKBOARD_SURFACES,
    passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
)
