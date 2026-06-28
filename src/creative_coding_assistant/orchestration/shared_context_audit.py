"""Passive V4.6 shared context audit metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_memory_contracts import (
    agent_memory_contract_registry,
)
from creative_coding_assistant.orchestration.blackboard_memory import (
    blackboard_memory_registry,
)
from creative_coding_assistant.orchestration.shared_context_views import (
    SHARED_CONTEXT_VIEW_REGISTRY,
    SharedContextViewContract,
)

SharedContextAuditStage = Literal["v4_6_shared_context_hardening"]
SharedContextAuditStatus = Literal["pass"]

SHARED_CONTEXT_AUDIT_RECORD_SERIALIZATION_VERSION = "shared_context_audit_record.v1"
SHARED_CONTEXT_AUDIT_REGISTRY_SERIALIZATION_VERSION = (
    "shared_context_audit_registry.v1"
)
SHARED_CONTEXT_AUDIT_REGISTRY_AUTHORITY_BOUNDARY = (
    "V4.6 shared context audit metadata checks passive per-agent shared "
    "context view coverage, source memory contract alignment, blackboard "
    "channel visibility scope, hidden-channel complements, visible metadata "
    "keys, and runtime materialization and mutation blocks only; it does not "
    "expose unrestricted global state, materialize runtime memory, mutate "
    "context, create storage, invoke agents, route providers or models, or "
    "modify generated output."
)

_VALIDATED_CONTEXT_SURFACES = (
    "view_scope",
    "source_memory_contract_alignment",
    "source_blackboard_registry_alignment",
    "visible_blackboard_channels",
    "hidden_blackboard_channels",
    "visible_metadata_keys",
    "runtime_materialization_flags",
)
_PASSIVE_BOUNDARY_FLAGS = (
    "unrestricted_global_state_blocked",
    "runtime_memory_materialization_blocked",
    "context_materialization_blocked",
    "context_mutation_blocked",
    "storage_backend_blocked",
    "blackboard_state_access_blocked",
    "generated_output_mutation_blocked",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "unrestricted_global_state_access",
    "runtime_memory_materialization",
    "shared_context_mutation",
    "blackboard_state_reads",
    "blackboard_state_writes",
    "storage_backend_creation",
    "agent_invocation",
    "provider_or_model_routing",
    "generated_output_modification",
)
_AUDIT_FINDINGS = (
    "shared_context_view_scope_confirmed",
    "source_memory_contract_alignment_confirmed",
    "blackboard_channel_scope_confirmed",
    "hidden_channel_complement_confirmed",
    "visible_metadata_keys_confirmed",
    "runtime_materialization_blocks_confirmed",
)


class SharedContextAuditRecord(BaseModel):
    """One passive V4.6 audit record for a shared context view."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=80)
    view_id: str = Field(min_length=1, max_length=140)
    audit_stage: SharedContextAuditStage = "v4_6_shared_context_hardening"
    view_serialization_version: str = Field(min_length=1, max_length=80)
    access_mode: str = Field(min_length=1, max_length=80)
    visible_memory_surfaces: tuple[str, ...] = Field(max_length=5)
    visible_blackboard_channel_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=6,
    )
    hidden_blackboard_channel_ids: tuple[str, ...] = Field(max_length=12)
    visible_metadata_keys: tuple[str, ...] = Field(min_length=1, max_length=36)
    source_memory_contract_id: str = Field(min_length=1, max_length=120)
    source_blackboard_registry: Literal["blackboard_memory_registry"] = (
        "blackboard_memory_registry"
    )
    source_registries: tuple[str, ...] = Field(min_length=2, max_length=2)
    validated_context_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=7, max_length=7)
    audit_findings: tuple[str, ...] = Field(min_length=6, max_length=6)
    missing_coverage_items: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    view_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=16,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    audit_status: SharedContextAuditStatus = "pass"
    metadata_only_declared: Literal[True] = True
    unrestricted_global_state_exposed: Literal[False] = False
    runtime_memory_implemented: Literal[False] = False
    context_materialization_implemented: Literal[False] = False
    context_mutation_implemented: Literal[False] = False
    storage_backend_implemented: Literal[False] = False
    blackboard_state_access_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["shared_context_audit_record.v1"] = (
        SHARED_CONTEXT_AUDIT_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class SharedContextAuditRegistry(BaseModel):
    """Stable passive V4.6 audit registry for shared context views."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["shared_context_audit_registry"] = "shared_context_audit_registry"
    serialization_version: Literal["shared_context_audit_registry.v1"] = (
        SHARED_CONTEXT_AUDIT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=SHARED_CONTEXT_AUDIT_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    audit_stage: SharedContextAuditStage = "v4_6_shared_context_hardening"
    audit_records: tuple[SharedContextAuditRecord, ...] = Field(
        min_length=12,
        max_length=12,
    )
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    view_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    blackboard_channel_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    audit_count: int = Field(ge=12, le=12)
    source_shared_context_registry: Literal["shared_context_view_registry"] = (
        "shared_context_view_registry"
    )
    source_registries: tuple[str, ...] = Field(min_length=2, max_length=2)
    validated_context_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    all_views_covered: Literal[True] = True
    scoped_visibility_confirmed: Literal[True] = True
    no_missing_coverage: Literal[True] = True
    active_runtime_audit_implemented: Literal[False] = False
    unrestricted_global_state_exposed: Literal[False] = False
    runtime_memory_implemented: Literal[False] = False
    context_materialization_implemented: Literal[False] = False
    context_mutation_implemented: Literal[False] = False
    storage_backend_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_audit_records(self) -> Self:
        derived_agent_ids = tuple(record.agent_id for record in self.audit_records)
        derived_view_ids = tuple(record.view_id for record in self.audit_records)
        if len(set(derived_agent_ids)) != len(derived_agent_ids):
            raise ValueError("agent_ids must be unique")
        if len(set(derived_view_ids)) != len(derived_view_ids):
            raise ValueError("view_ids must be unique")
        if self.agent_ids != derived_agent_ids:
            raise ValueError("agent_ids must match audit records")
        if self.view_ids != derived_view_ids:
            raise ValueError("view_ids must match audit records")
        if self.audit_count != len(self.audit_records):
            raise ValueError("audit_count must match audit records")
        for record in self.audit_records:
            if record.source_registries != self.source_registries:
                raise ValueError("source_registries must match registry")
            if record.validated_context_surfaces != self.validated_context_surfaces:
                raise ValueError("validated_context_surfaces must match registry")
            if record.passive_boundary_flags != self.passive_boundary_flags:
                raise ValueError("passive_boundary_flags must match registry")
            if record.missing_coverage_items:
                raise ValueError("audit records must not contain missing coverage")
        return self


def shared_context_audit_registry() -> SharedContextAuditRegistry:
    """Return passive V4.6 shared context audit metadata."""

    return SHARED_CONTEXT_AUDIT_REGISTRY


def shared_context_audit_by_agent_id(
    agent_id: str,
    registry: SharedContextAuditRegistry | None = None,
) -> SharedContextAuditRecord | None:
    """Return one passive shared context audit record by agent id."""

    source_registry = registry or SHARED_CONTEXT_AUDIT_REGISTRY
    for record in source_registry.audit_records:
        if record.agent_id == agent_id:
            return record
    return None


def shared_context_audit_by_view_id(
    view_id: str,
    registry: SharedContextAuditRegistry | None = None,
) -> SharedContextAuditRecord | None:
    """Return one passive shared context audit record by view id."""

    source_registry = registry or SHARED_CONTEXT_AUDIT_REGISTRY
    for record in source_registry.audit_records:
        if record.view_id == view_id:
            return record
    return None


def shared_context_audits_for_source_registry(
    source_registry_ref: str,
    registry: SharedContextAuditRegistry | None = None,
) -> tuple[SharedContextAuditRecord, ...]:
    """Return passive shared context audit records for one source registry."""

    source_registry = registry or SHARED_CONTEXT_AUDIT_REGISTRY
    normalized_ref = str(source_registry_ref).strip()
    return tuple(
        record
        for record in source_registry.audit_records
        if normalized_ref in record.source_registries
    )


def _missing_coverage_items(view: SharedContextViewContract) -> tuple[str, ...]:
    missing: list[str] = []
    memory_registry = agent_memory_contract_registry()
    blackboard_registry = blackboard_memory_registry()
    visible = set(view.visible_blackboard_channel_ids)
    hidden = set(view.hidden_blackboard_channel_ids)
    known_channels = set(blackboard_registry.channel_ids)
    if view.agent_id not in memory_registry.agent_ids:
        missing.append("agent_memory_contract_missing")
    if view.source_memory_contract_id not in {
        contract.memory_contract_id for contract in memory_registry.contracts
    }:
        missing.append("source_memory_contract_id_missing")
    if view.source_blackboard_registry != "blackboard_memory_registry":
        missing.append("source_blackboard_registry_mismatch")
    if not visible.issubset(known_channels):
        missing.append("visible_blackboard_channel_unknown")
    if visible == known_channels:
        missing.append("unrestricted_blackboard_visibility_enabled")
    if hidden != known_channels - visible:
        missing.append("hidden_channel_complement_mismatch")
    if not view.visible_metadata_keys:
        missing.append("visible_metadata_keys_missing")
    if view.unrestricted_global_state_exposed:
        missing.append("unrestricted_global_state_enabled")
    if view.runtime_memory_implemented:
        missing.append("runtime_memory_materialization_enabled")
    if view.context_materialization_implemented:
        missing.append("context_materialization_enabled")
    if view.context_mutation_implemented:
        missing.append("context_mutation_enabled")
    if view.storage_backend_implemented:
        missing.append("storage_backend_enabled")
    for blocked_behavior in (
        "unrestricted_global_state_access",
        "shared_context_mutation",
        "blackboard_state_reads",
        "blackboard_state_writes",
        "storage_backend_creation",
        "generated_output_modification",
    ):
        if blocked_behavior not in view.blocked_runtime_behaviors:
            missing.append(f"{blocked_behavior}_block_missing")
    return tuple(missing)


def _audit_record(view: SharedContextViewContract) -> SharedContextAuditRecord:
    return SharedContextAuditRecord(
        agent_id=view.agent_id,
        view_id=view.view_id,
        view_serialization_version=view.serialization_version,
        access_mode=view.access_mode,
        visible_memory_surfaces=view.visible_memory_surfaces,
        visible_blackboard_channel_ids=view.visible_blackboard_channel_ids,
        hidden_blackboard_channel_ids=view.hidden_blackboard_channel_ids,
        visible_metadata_keys=view.visible_metadata_keys,
        source_memory_contract_id=view.source_memory_contract_id,
        source_blackboard_registry=view.source_blackboard_registry,
        source_registries=SHARED_CONTEXT_VIEW_REGISTRY.source_registries,
        validated_context_surfaces=_VALIDATED_CONTEXT_SURFACES,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
        audit_findings=_AUDIT_FINDINGS,
        missing_coverage_items=_missing_coverage_items(view),
        view_blocked_runtime_behaviors=view.blocked_runtime_behaviors,
        metadata_only_declared=view.metadata_only,
    )


SHARED_CONTEXT_AUDIT_RECORDS = tuple(
    _audit_record(view) for view in SHARED_CONTEXT_VIEW_REGISTRY.views
)
SHARED_CONTEXT_AUDIT_REGISTRY = SharedContextAuditRegistry(
    audit_records=SHARED_CONTEXT_AUDIT_RECORDS,
    agent_ids=tuple(record.agent_id for record in SHARED_CONTEXT_AUDIT_RECORDS),
    view_ids=tuple(record.view_id for record in SHARED_CONTEXT_AUDIT_RECORDS),
    blackboard_channel_ids=SHARED_CONTEXT_VIEW_REGISTRY.blackboard_channel_ids,
    audit_count=len(SHARED_CONTEXT_AUDIT_RECORDS),
    source_registries=SHARED_CONTEXT_VIEW_REGISTRY.source_registries,
    validated_context_surfaces=_VALIDATED_CONTEXT_SURFACES,
    passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
)
