"""Passive V4.6 agent reliability audit metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_contracts import (
    AGENT_CONTRACT_REGISTRY,
)
from creative_coding_assistant.orchestration.agent_escalation_signals import (
    agent_escalation_signal_registry,
)
from creative_coding_assistant.orchestration.agent_lifecycle import (
    AgentLifecycleProfile,
    agent_lifecycle_profile_by_agent_id,
    agent_lifecycle_registry,
)
from creative_coding_assistant.orchestration.agent_state_synchronization import (
    AgentStateSyncProfile,
    agent_state_sync_profile_by_agent_id,
    agent_state_synchronization_registry,
)
from creative_coding_assistant.orchestration.engine_contract_consistency import (
    engine_contract_consistency_registry,
)

AgentReliabilityAuditStage = Literal["v4_6_agent_reliability_hardening"]
AgentReliabilityAuditStatus = Literal["pass"]

AGENT_RELIABILITY_AUDIT_RECORD_SERIALIZATION_VERSION = (
    "agent_reliability_audit_record.v1"
)
AGENT_RELIABILITY_AUDIT_REGISTRY_SERIALIZATION_VERSION = (
    "agent_reliability_audit_registry.v1"
)
AGENT_RELIABILITY_AUDIT_REGISTRY_AUTHORITY_BOUNDARY = (
    "V4.6 agent reliability audit metadata checks passive lifecycle profile "
    "coverage, lifecycle transition coverage, state synchronization profile "
    "coverage, checkpoint and stale-warning metadata, conflict-surface "
    "declarations, advisory escalation signal coverage, engine contract "
    "consistency references, and blocked runtime behavior declarations only; "
    "it does not run lifecycle transitions, synchronize runtime state, "
    "detect stale state, resolve conflicts, perform escalation, trigger "
    "retries, execute recovery, route providers or models, invoke agents, "
    "or modify generated output."
)

_SOURCE_RELIABILITY_REGISTRIES = (
    "agent_lifecycle_registry",
    "agent_state_synchronization_registry",
    "agent_escalation_signal_registry",
    "engine_contract_consistency_registry",
)
_VALIDATED_RELIABILITY_SURFACES = (
    "lifecycle_profile_coverage",
    "lifecycle_transition_coverage",
    "state_sync_profile_coverage",
    "sync_checkpoint_coverage",
    "stale_warning_coverage",
    "conflict_surface_coverage",
    "escalation_signal_coverage",
    "engine_contract_consistency_coverage",
)
_PASSIVE_BOUNDARY_FLAGS = (
    "runtime_lifecycle_engine_blocked",
    "runtime_state_synchronization_blocked",
    "stale_state_detection_blocked",
    "conflict_resolution_blocked",
    "escalation_execution_blocked",
    "retry_recovery_execution_blocked",
    "provider_model_routing_blocked",
    "generated_output_mutation_blocked",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "runtime_reliability_engine",
    "runtime_lifecycle_engine",
    "state_transition_execution",
    "runtime_state_synchronization",
    "stale_state_detection_execution",
    "conflict_resolution",
    "escalation_execution",
    "retry_or_refinement_triggering",
    "provider_or_model_routing",
    "agent_invocation",
    "generated_output_modification",
)
_AUDIT_FINDINGS = (
    "lifecycle_profile_coverage_confirmed",
    "state_sync_profile_coverage_confirmed",
    "stale_warning_surface_confirmed",
    "conflict_surface_declarations_confirmed",
    "escalation_signal_coverage_confirmed",
    "engine_contract_consistency_confirmed",
    "runtime_reliability_blocks_confirmed",
)


class AgentReliabilityAuditRecord(BaseModel):
    """One passive V4.6 reliability audit record for an agent."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=80)
    audit_stage: AgentReliabilityAuditStage = "v4_6_agent_reliability_hardening"
    lifecycle_profile_id: str = Field(min_length=1, max_length=140)
    lifecycle_transition_ids: tuple[str, ...] = Field(min_length=10, max_length=10)
    lifecycle_terminal_states: tuple[str, ...] = Field(min_length=5, max_length=5)
    state_sync_profile_id: str = Field(min_length=1, max_length=140)
    sync_checkpoint_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    consistency_constraint_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    stale_warning_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    conflict_surface_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    escalation_signal_ids: tuple[str, ...] = Field(min_length=7, max_length=7)
    escalation_signal_categories: tuple[str, ...] = Field(min_length=7, max_length=7)
    consistency_family_ids: tuple[str, ...] = Field(min_length=3, max_length=3)
    reliability_source_registries: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    validated_reliability_surfaces: tuple[str, ...] = Field(
        min_length=8,
        max_length=8,
    )
    passive_boundary_flags: tuple[str, ...] = Field(min_length=8, max_length=8)
    audit_findings: tuple[str, ...] = Field(min_length=7, max_length=7)
    missing_coverage_items: tuple[str, ...] = Field(
        default_factory=tuple, max_length=16
    )
    lifecycle_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    state_sync_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    escalation_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    consistency_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    audit_status: AgentReliabilityAuditStatus = "pass"
    metadata_only_declared: Literal[True] = True
    lifecycle_profile_present: Literal[True] = True
    state_sync_profile_present: Literal[True] = True
    escalation_signal_coverage_present: Literal[True] = True
    consistency_family_coverage_present: Literal[True] = True
    runtime_lifecycle_engine_implemented: Literal[False] = False
    runtime_state_synchronization_implemented: Literal[False] = False
    stale_state_detection_implemented: Literal[False] = False
    conflict_resolution_implemented: Literal[False] = False
    escalation_execution_implemented: Literal[False] = False
    retry_recovery_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["agent_reliability_audit_record.v1"] = (
        AGENT_RELIABILITY_AUDIT_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentReliabilityAuditRegistry(BaseModel):
    """Stable passive V4.6 audit registry for agent reliability metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_reliability_audit_registry"] = (
        "agent_reliability_audit_registry"
    )
    serialization_version: Literal["agent_reliability_audit_registry.v1"] = (
        AGENT_RELIABILITY_AUDIT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_RELIABILITY_AUDIT_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    audit_stage: AgentReliabilityAuditStage = "v4_6_agent_reliability_hardening"
    audit_records: tuple[AgentReliabilityAuditRecord, ...] = Field(
        min_length=12,
        max_length=12,
    )
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    audit_count: int = Field(ge=12, le=12)
    source_lifecycle_registry: Literal["agent_lifecycle_registry"] = (
        "agent_lifecycle_registry"
    )
    source_state_synchronization_registry: Literal[
        "agent_state_synchronization_registry"
    ] = "agent_state_synchronization_registry"
    source_escalation_signal_registry: Literal["agent_escalation_signal_registry"] = (
        "agent_escalation_signal_registry"
    )
    source_engine_contract_consistency_registry: Literal[
        "engine_contract_consistency_registry"
    ] = "engine_contract_consistency_registry"
    lifecycle_profile_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    state_sync_profile_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    lifecycle_transition_ids: tuple[str, ...] = Field(min_length=10, max_length=10)
    sync_checkpoint_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    consistency_constraint_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    stale_warning_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    conflict_surface_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    escalation_signal_ids: tuple[str, ...] = Field(min_length=7, max_length=7)
    escalation_signal_categories: tuple[str, ...] = Field(min_length=7, max_length=7)
    consistency_family_ids: tuple[str, ...] = Field(min_length=3, max_length=3)
    reliability_source_registries: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    validated_reliability_surfaces: tuple[str, ...] = Field(
        min_length=8,
        max_length=8,
    )
    passive_boundary_flags: tuple[str, ...] = Field(min_length=8, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    all_agents_covered: Literal[True] = True
    no_missing_coverage: Literal[True] = True
    runtime_reliability_engine_implemented: Literal[False] = False
    runtime_lifecycle_engine_implemented: Literal[False] = False
    runtime_state_synchronization_implemented: Literal[False] = False
    stale_state_detection_implemented: Literal[False] = False
    conflict_resolution_implemented: Literal[False] = False
    escalation_execution_implemented: Literal[False] = False
    retry_recovery_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_audit_records(self) -> Self:
        derived_agent_ids = tuple(record.agent_id for record in self.audit_records)
        if len(set(derived_agent_ids)) != len(derived_agent_ids):
            raise ValueError("agent_ids must be unique")
        if self.agent_ids != derived_agent_ids:
            raise ValueError("agent_ids must match audit records")
        if self.audit_count != len(self.audit_records):
            raise ValueError("audit_count must match audit records")
        if self.reliability_source_registries != _SOURCE_RELIABILITY_REGISTRIES:
            raise ValueError("reliability_source_registries must match sources")

        known_lifecycle_profiles = set(self.lifecycle_profile_ids)
        known_state_sync_profiles = set(self.state_sync_profile_ids)
        for record in self.audit_records:
            if record.audit_stage != self.audit_stage:
                raise ValueError("audit_stage must match registry")
            if record.lifecycle_profile_id not in known_lifecycle_profiles:
                raise ValueError("lifecycle_profile_id must be known")
            if record.state_sync_profile_id not in known_state_sync_profiles:
                raise ValueError("state_sync_profile_id must be known")
            if record.lifecycle_transition_ids != self.lifecycle_transition_ids:
                raise ValueError("lifecycle_transition_ids must match registry")
            if record.sync_checkpoint_ids != self.sync_checkpoint_ids:
                raise ValueError("sync_checkpoint_ids must match registry")
            if record.consistency_constraint_ids != self.consistency_constraint_ids:
                raise ValueError("consistency_constraint_ids must match registry")
            if record.stale_warning_ids != self.stale_warning_ids:
                raise ValueError("stale_warning_ids must match registry")
            if record.conflict_surface_ids != self.conflict_surface_ids:
                raise ValueError("conflict_surface_ids must match registry")
            if record.escalation_signal_ids != self.escalation_signal_ids:
                raise ValueError("escalation_signal_ids must match registry")
            if record.escalation_signal_categories != (
                self.escalation_signal_categories
            ):
                raise ValueError("escalation_signal_categories must match registry")
            if record.consistency_family_ids != self.consistency_family_ids:
                raise ValueError("consistency_family_ids must match registry")
            if record.reliability_source_registries != (
                self.reliability_source_registries
            ):
                raise ValueError("reliability_source_registries must match registry")
            if record.validated_reliability_surfaces != (
                self.validated_reliability_surfaces
            ):
                raise ValueError("validated_reliability_surfaces must match registry")
            if record.passive_boundary_flags != self.passive_boundary_flags:
                raise ValueError("passive_boundary_flags must match registry")
            if record.missing_coverage_items:
                raise ValueError("audit records must not contain missing coverage")
        return self


def agent_reliability_audit_registry() -> AgentReliabilityAuditRegistry:
    """Return passive V4.6 agent reliability audit metadata."""

    return AGENT_RELIABILITY_AUDIT_REGISTRY


def agent_reliability_audit_by_agent_id(
    agent_id: str,
    registry: AgentReliabilityAuditRegistry | None = None,
) -> AgentReliabilityAuditRecord | None:
    """Return one passive reliability audit record by agent id."""

    source_registry = registry or AGENT_RELIABILITY_AUDIT_REGISTRY
    for record in source_registry.audit_records:
        if record.agent_id == agent_id:
            return record
    return None


def agent_reliability_audits_for_escalation_category(
    escalation_category: str,
    registry: AgentReliabilityAuditRegistry | None = None,
) -> tuple[AgentReliabilityAuditRecord, ...]:
    """Return passive reliability audits referencing one escalation category."""

    source_registry = registry or AGENT_RELIABILITY_AUDIT_REGISTRY
    normalized_category = str(escalation_category).strip()
    return tuple(
        record
        for record in source_registry.audit_records
        if normalized_category in record.escalation_signal_categories
    )


def agent_reliability_audits_for_consistency_family(
    consistency_family_id: str,
    registry: AgentReliabilityAuditRegistry | None = None,
) -> tuple[AgentReliabilityAuditRecord, ...]:
    """Return passive reliability audits referencing one consistency family."""

    source_registry = registry or AGENT_RELIABILITY_AUDIT_REGISTRY
    normalized_family_id = str(consistency_family_id).strip()
    return tuple(
        record
        for record in source_registry.audit_records
        if normalized_family_id in record.consistency_family_ids
    )


def _missing_coverage_items(
    *,
    lifecycle_profile: AgentLifecycleProfile,
    state_sync_profile: AgentStateSyncProfile,
) -> tuple[str, ...]:
    lifecycle = agent_lifecycle_registry()
    state_sync = agent_state_synchronization_registry()
    escalation_signals = agent_escalation_signal_registry()
    consistency = engine_contract_consistency_registry()
    missing: list[str] = []
    if lifecycle_profile.agent_id != state_sync_profile.agent_id:
        missing.append("agent_profile_alignment_missing")
    if not lifecycle_profile.lifecycle_profile_id:
        missing.append("lifecycle_profile_id_missing")
    if not state_sync_profile.sync_profile_id:
        missing.append("state_sync_profile_id_missing")
    if lifecycle_profile.transition_ids != lifecycle.transition_ids:
        missing.append("lifecycle_transition_coverage_missing")
    if state_sync_profile.sync_checkpoint_ids != state_sync.checkpoint_ids:
        missing.append("sync_checkpoint_coverage_missing")
    if state_sync_profile.consistency_constraint_ids != state_sync.constraint_ids:
        missing.append("consistency_constraint_coverage_missing")
    if state_sync_profile.stale_warning_ids != state_sync.stale_warning_ids:
        missing.append("stale_warning_coverage_missing")
    if state_sync_profile.conflict_surface_ids != state_sync.conflict_surface_ids:
        missing.append("conflict_surface_coverage_missing")
    if not escalation_signals.signal_ids:
        missing.append("escalation_signal_coverage_missing")
    if not consistency.family_ids:
        missing.append("engine_contract_consistency_missing")
    if not lifecycle_profile.metadata_only or not state_sync_profile.metadata_only:
        missing.append("metadata_only_declaration_missing")
    if lifecycle.runtime_lifecycle_engine_implemented:
        missing.append("runtime_lifecycle_engine_enabled")
    if state_sync.runtime_synchronization_implemented:
        missing.append("runtime_state_synchronization_enabled")
    if state_sync.conflict_resolution_implemented:
        missing.append("conflict_resolution_enabled")
    if escalation_signals.escalation_performed:
        missing.append("escalation_execution_enabled")
    if not consistency.metadata_only:
        missing.append("engine_contract_consistency_runtime_enabled")
    if "runtime_lifecycle_engine" not in lifecycle_profile.blocked_runtime_behaviors:
        missing.append("runtime_lifecycle_engine_block_missing")
    if (
        "runtime_state_synchronization"
        not in state_sync_profile.blocked_runtime_behaviors
    ):
        missing.append("runtime_state_synchronization_block_missing")
    if "conflict_resolution" not in state_sync_profile.blocked_runtime_behaviors:
        missing.append("conflict_resolution_block_missing")
    if "escalation_execution" not in escalation_signals.blocked_runtime_behaviors:
        missing.append("escalation_execution_block_missing")
    if "retry_or_refinement_triggering" not in consistency.blocked_runtime_behaviors:
        missing.append("retry_triggering_block_missing")
    if "provider_or_model_routing" not in consistency.blocked_runtime_behaviors:
        missing.append("provider_model_routing_block_missing")
    if "generated_output_modification" not in consistency.blocked_runtime_behaviors:
        missing.append("generated_output_mutation_block_missing")
    return tuple(missing)


def _audit_record(agent_id: str) -> AgentReliabilityAuditRecord:
    lifecycle = agent_lifecycle_registry()
    state_sync = agent_state_synchronization_registry()
    escalation_signals = agent_escalation_signal_registry()
    consistency = engine_contract_consistency_registry()
    lifecycle_profile = agent_lifecycle_profile_by_agent_id(agent_id, lifecycle)
    state_sync_profile = agent_state_sync_profile_by_agent_id(agent_id, state_sync)
    if lifecycle_profile is None:
        raise ValueError(f"missing lifecycle profile for {agent_id}")
    if state_sync_profile is None:
        raise ValueError(f"missing state sync profile for {agent_id}")

    return AgentReliabilityAuditRecord(
        agent_id=agent_id,
        lifecycle_profile_id=lifecycle_profile.lifecycle_profile_id,
        lifecycle_transition_ids=lifecycle_profile.transition_ids,
        lifecycle_terminal_states=lifecycle_profile.terminal_states,
        state_sync_profile_id=state_sync_profile.sync_profile_id,
        sync_checkpoint_ids=state_sync_profile.sync_checkpoint_ids,
        consistency_constraint_ids=state_sync_profile.consistency_constraint_ids,
        stale_warning_ids=state_sync_profile.stale_warning_ids,
        conflict_surface_ids=state_sync_profile.conflict_surface_ids,
        escalation_signal_ids=escalation_signals.signal_ids,
        escalation_signal_categories=escalation_signals.categories,
        consistency_family_ids=consistency.family_ids,
        reliability_source_registries=_SOURCE_RELIABILITY_REGISTRIES,
        validated_reliability_surfaces=_VALIDATED_RELIABILITY_SURFACES,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
        audit_findings=_AUDIT_FINDINGS,
        missing_coverage_items=_missing_coverage_items(
            lifecycle_profile=lifecycle_profile,
            state_sync_profile=state_sync_profile,
        ),
        lifecycle_blocked_runtime_behaviors=lifecycle_profile.blocked_runtime_behaviors,
        state_sync_blocked_runtime_behaviors=(
            state_sync_profile.blocked_runtime_behaviors
        ),
        escalation_blocked_runtime_behaviors=(
            escalation_signals.blocked_runtime_behaviors
        ),
        consistency_blocked_runtime_behaviors=consistency.blocked_runtime_behaviors,
        metadata_only_declared=(
            lifecycle_profile.metadata_only and state_sync_profile.metadata_only
        ),
    )


AGENT_RELIABILITY_AUDIT_RECORDS = tuple(
    _audit_record(agent_id) for agent_id in AGENT_CONTRACT_REGISTRY.agent_ids
)
AGENT_RELIABILITY_AUDIT_REGISTRY = AgentReliabilityAuditRegistry(
    audit_records=AGENT_RELIABILITY_AUDIT_RECORDS,
    agent_ids=tuple(record.agent_id for record in AGENT_RELIABILITY_AUDIT_RECORDS),
    audit_count=len(AGENT_RELIABILITY_AUDIT_RECORDS),
    lifecycle_profile_ids=tuple(
        profile.lifecycle_profile_id for profile in agent_lifecycle_registry().profiles
    ),
    state_sync_profile_ids=tuple(
        profile.sync_profile_id
        for profile in agent_state_synchronization_registry().profiles
    ),
    lifecycle_transition_ids=agent_lifecycle_registry().transition_ids,
    sync_checkpoint_ids=agent_state_synchronization_registry().checkpoint_ids,
    consistency_constraint_ids=agent_state_synchronization_registry().constraint_ids,
    stale_warning_ids=agent_state_synchronization_registry().stale_warning_ids,
    conflict_surface_ids=agent_state_synchronization_registry().conflict_surface_ids,
    escalation_signal_ids=agent_escalation_signal_registry().signal_ids,
    escalation_signal_categories=agent_escalation_signal_registry().categories,
    consistency_family_ids=engine_contract_consistency_registry().family_ids,
    reliability_source_registries=_SOURCE_RELIABILITY_REGISTRIES,
    validated_reliability_surfaces=_VALIDATED_RELIABILITY_SURFACES,
    passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
)
