"""Passive V4.6 agent collaboration audit metadata."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_consensus import (
    consensus_builder_registry,
)
from creative_coding_assistant.orchestration.agent_contracts import (
    AGENT_CONTRACT_REGISTRY,
)
from creative_coding_assistant.orchestration.agent_coordination import (
    agent_coordination_registry,
)
from creative_coding_assistant.orchestration.agent_debate import (
    agent_debate_registry,
)
from creative_coding_assistant.orchestration.workflow_agent_handoff import (
    workflow_agent_handoff_registry,
)

AgentCollaborationAuditStage = Literal["v4_6_agent_collaboration_hardening"]
AgentCollaborationAuditStatus = Literal["pass"]
AgentCollaborationSurface = Literal[
    "coordination",
    "debate",
    "consensus",
    "workflow_handoff",
]

AGENT_COLLABORATION_AUDIT_RECORD_SERIALIZATION_VERSION = (
    "agent_collaboration_audit_record.v1"
)
AGENT_COLLABORATION_AUDIT_REGISTRY_SERIALIZATION_VERSION = (
    "agent_collaboration_audit_registry.v1"
)
AGENT_COLLABORATION_AUDIT_REGISTRY_AUTHORITY_BOUNDARY = (
    "V4.6 agent collaboration audit metadata checks passive coordination, "
    "debate, consensus, and workflow handoff registry coverage, participant "
    "alignment, collaboration contract ids, source registry references, "
    "metadata-only declarations, and runtime collaboration blocks only; it "
    "does not coordinate live agents, execute debate or voting, perform "
    "runtime handoffs, invoke agents, route providers or models, trigger "
    "retries, mutate state, schedule work, or modify generated output."
)

_VALIDATED_COLLABORATION_SURFACES = (
    "registry_role",
    "serialization_version",
    "source_registry_refs",
    "participant_agent_ids",
    "collaboration_contract_ids",
    "metadata_only",
    "blocked_runtime_behaviors",
)
_PASSIVE_BOUNDARY_FLAGS = (
    "live_coordination_blocked",
    "debate_execution_blocked",
    "voting_execution_blocked",
    "runtime_handoff_blocked",
    "agent_invocation_blocked",
    "provider_model_routing_blocked",
    "generated_output_mutation_blocked",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "live_coordination",
    "debate_loop_execution",
    "voting_execution",
    "runtime_handoff_execution",
    "agent_invocation",
    "provider_or_model_routing",
    "retry_or_refinement_triggering",
    "state_mutation",
    "generated_output_modification",
)
_AUDIT_FINDINGS = (
    "collaboration_registry_discoverability_confirmed",
    "source_registry_alignment_confirmed",
    "participant_agent_alignment_confirmed",
    "collaboration_contract_coverage_confirmed",
    "metadata_only_boundary_confirmed",
    "runtime_collaboration_blocks_confirmed",
)
_AGENT_INVOCATION_BLOCK_ALIASES = (
    "agent_invocation",
    "agent_execution",
    "agent_action_triggering",
)
_RETRY_BLOCK_ALIASES = (
    "retry_or_refinement_triggering",
    "retry_triggering",
)
_SURFACE_RUNTIME_BLOCKS: dict[AgentCollaborationSurface, tuple[str, ...]] = {
    "coordination": ("live_coordination",),
    "debate": ("debate_loop_execution",),
    "consensus": ("voting_execution",),
    "workflow_handoff": ("runtime_handoff_execution",),
}
_PASSIVE_IMPLEMENTATION_FLAGS = (
    "live_coordination_implemented",
    "agent_actions_triggered",
    "output_mutation_implemented",
    "debate_execution_implemented",
    "retry_triggering_implemented",
    "voting_execution_implemented",
    "final_answer_selection_implemented",
    "final_synthesis_mutation_implemented",
    "workflow_graph_change_implemented",
    "prompt_alteration_implemented",
    "agent_execution_implemented",
    "runtime_handoff_implemented",
)
_COLLABORATION_SPECS: tuple[
    tuple[str, AgentCollaborationSurface, Callable[[], Any]],
    ...,
] = (
    ("agent_coordination_registry", "coordination", agent_coordination_registry),
    ("agent_debate_registry", "debate", agent_debate_registry),
    ("consensus_builder_registry", "consensus", consensus_builder_registry),
    (
        "workflow_agent_handoff_registry",
        "workflow_handoff",
        workflow_agent_handoff_registry,
    ),
)


class AgentCollaborationAuditRecord(BaseModel):
    """One passive V4.6 audit record for an agent collaboration registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    registry_id: str = Field(min_length=1, max_length=160)
    registry_role: str = Field(min_length=1, max_length=160)
    collaboration_surface: AgentCollaborationSurface
    registry_serialization_version: str = Field(min_length=1, max_length=120)
    source_registry_refs: tuple[str, ...] = Field(min_length=1, max_length=8)
    participant_agent_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    collaboration_contract_ids: tuple[str, ...] = Field(min_length=1, max_length=24)
    collaboration_contract_count: int = Field(ge=1, le=24)
    validated_collaboration_surfaces: tuple[str, ...] = Field(
        min_length=7,
        max_length=7,
    )
    passive_boundary_flags: tuple[str, ...] = Field(min_length=7, max_length=7)
    audit_findings: tuple[str, ...] = Field(min_length=6, max_length=6)
    missing_coverage_items: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    registry_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=16,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    audit_stage: AgentCollaborationAuditStage = (
        "v4_6_agent_collaboration_hardening"
    )
    audit_status: AgentCollaborationAuditStatus = "pass"
    metadata_only_declared: Literal[True] = True
    live_coordination_implemented: Literal[False] = False
    debate_execution_implemented: Literal[False] = False
    voting_execution_implemented: Literal[False] = False
    runtime_handoff_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    state_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["agent_collaboration_audit_record.v1"] = (
        AGENT_COLLABORATION_AUDIT_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentCollaborationAuditRegistry(BaseModel):
    """Stable passive V4.6 audit registry for agent collaboration metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_collaboration_audit_registry"] = (
        "agent_collaboration_audit_registry"
    )
    serialization_version: Literal["agent_collaboration_audit_registry.v1"] = (
        AGENT_COLLABORATION_AUDIT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_COLLABORATION_AUDIT_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    audit_stage: AgentCollaborationAuditStage = (
        "v4_6_agent_collaboration_hardening"
    )
    audit_records: tuple[AgentCollaborationAuditRecord, ...] = Field(
        min_length=4,
        max_length=4,
    )
    registry_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    collaboration_surfaces: tuple[AgentCollaborationSurface, ...] = Field(
        min_length=4,
        max_length=4,
    )
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    audit_count: int = Field(ge=4, le=4)
    source_collaboration_registries: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    source_registry_refs: tuple[str, ...] = Field(min_length=7, max_length=7)
    validated_collaboration_surfaces: tuple[str, ...] = Field(
        min_length=7,
        max_length=7,
    )
    passive_boundary_flags: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    all_collaboration_registries_covered: Literal[True] = True
    all_agent_ids_aligned: Literal[True] = True
    no_missing_coverage: Literal[True] = True
    active_collaboration_execution_implemented: Literal[False] = False
    live_coordination_implemented: Literal[False] = False
    debate_execution_implemented: Literal[False] = False
    voting_execution_implemented: Literal[False] = False
    runtime_handoff_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_audit_records(self) -> Self:
        derived_registry_ids = tuple(record.registry_id for record in self.audit_records)
        derived_surfaces = tuple(
            record.collaboration_surface for record in self.audit_records
        )
        derived_source_refs = tuple(
            dict.fromkeys(
                source_ref
                for record in self.audit_records
                for source_ref in record.source_registry_refs
            )
        )
        if len(set(derived_registry_ids)) != len(derived_registry_ids):
            raise ValueError("registry_ids must be unique")
        if self.registry_ids != derived_registry_ids:
            raise ValueError("registry_ids must match audit records")
        if self.collaboration_surfaces != derived_surfaces:
            raise ValueError("collaboration_surfaces must match audit records")
        if self.audit_count != len(self.audit_records):
            raise ValueError("audit_count must match audit records")
        if self.source_collaboration_registries != derived_registry_ids:
            raise ValueError("source_collaboration_registries must match records")
        if self.source_registry_refs != derived_source_refs:
            raise ValueError("source_registry_refs must match audit records")

        known_agents = set(self.agent_ids)
        covered_agents = {
            agent_id
            for record in self.audit_records
            for agent_id in record.participant_agent_ids
        }
        if covered_agents != known_agents:
            raise ValueError("participant_agent_ids must cover known agents")
        for record in self.audit_records:
            if record.validated_collaboration_surfaces != (
                self.validated_collaboration_surfaces
            ):
                raise ValueError("validated_collaboration_surfaces must match registry")
            if record.passive_boundary_flags != self.passive_boundary_flags:
                raise ValueError("passive_boundary_flags must match registry")
            if not set(record.participant_agent_ids).issubset(known_agents):
                raise ValueError("participant_agent_ids must be known agents")
            if record.collaboration_contract_count != len(
                record.collaboration_contract_ids
            ):
                raise ValueError("collaboration_contract_count must match ids")
            if record.missing_coverage_items:
                raise ValueError("audit records must not contain missing coverage")
        return self


def agent_collaboration_audit_registry() -> AgentCollaborationAuditRegistry:
    """Return passive V4.6 agent collaboration audit metadata."""

    return AGENT_COLLABORATION_AUDIT_REGISTRY


def agent_collaboration_audit_by_registry_id(
    registry_id: str,
    registry: AgentCollaborationAuditRegistry | None = None,
) -> AgentCollaborationAuditRecord | None:
    """Return one passive collaboration audit record by source registry id."""

    source_registry = registry or AGENT_COLLABORATION_AUDIT_REGISTRY
    for record in source_registry.audit_records:
        if record.registry_id == registry_id:
            return record
    return None


def agent_collaboration_audits_for_surface(
    collaboration_surface: str,
    registry: AgentCollaborationAuditRegistry | None = None,
) -> tuple[AgentCollaborationAuditRecord, ...]:
    """Return passive collaboration audit records for one surface kind."""

    source_registry = registry or AGENT_COLLABORATION_AUDIT_REGISTRY
    normalized_surface = str(collaboration_surface).strip()
    return tuple(
        record
        for record in source_registry.audit_records
        if record.collaboration_surface == normalized_surface
    )


def agent_collaboration_audits_for_source_registry(
    source_registry_ref: str,
    registry: AgentCollaborationAuditRegistry | None = None,
) -> tuple[AgentCollaborationAuditRecord, ...]:
    """Return passive collaboration audits referencing one source registry."""

    source_registry = registry or AGENT_COLLABORATION_AUDIT_REGISTRY
    normalized_ref = str(source_registry_ref).strip()
    return tuple(
        record
        for record in source_registry.audit_records
        if normalized_ref in record.source_registry_refs
    )


def _source_registry_refs(registry: Any) -> tuple[str, ...]:
    return tuple(str(ref) for ref in getattr(registry, "source_registries", ()))


def _participant_agent_ids(registry: Any) -> tuple[str, ...]:
    if hasattr(registry, "participant_agent_ids"):
        return tuple(registry.participant_agent_ids)
    if hasattr(registry, "agent_ids"):
        return tuple(registry.agent_ids)
    if hasattr(registry, "voting_inputs"):
        return tuple(
            dict.fromkeys(
                agent_id
                for item in registry.voting_inputs
                for agent_id in item.participant_agent_ids
            )
        )
    return AGENT_CONTRACT_REGISTRY.agent_ids


def _coordination_agent_ids(registry: Any) -> tuple[str, ...]:
    agent_ids: list[str] = []
    for responsibility in registry.responsibilities:
        agent_ids.extend(responsibility.responsible_agent_ids)
    for channel in registry.handoff_channels:
        agent_ids.extend(
            (*channel.source_agent_ids, *channel.target_agent_ids),
        )
    return tuple(dict.fromkeys(agent_ids))


def _collaboration_contract_ids(registry: Any) -> tuple[str, ...]:
    role = getattr(registry, "role", "")
    if role == "agent_coordination_registry":
        return tuple(
            str(item)
            for item in (
                registry.coordinator_ids
                + registry.handoff_channel_ids
                + registry.event_types
            )
        )
    if role == "agent_debate_registry":
        return tuple(
            str(item)
            for item in registry.claim_ids + registry.round_ids + registry.topic_ids
        )
    if role == "consensus_builder_registry":
        return tuple(
            str(item)
            for item in (
                registry.voting_input_ids
                + registry.agreement_surface_ids
                + registry.topic_ids
            )
        )
    if role == "workflow_agent_handoff_registry":
        return tuple(
            str(item)
            for item in registry.handoff_ids + registry.profile_ids + registry.surfaces
        )
    return ()


def _registry_blocked_runtime_behaviors(registry: Any) -> tuple[str, ...]:
    blocked = getattr(registry, "blocked_runtime_behaviors", None)
    if blocked is None:
        return _BLOCKED_RUNTIME_BEHAVIORS
    return tuple(blocked)


def _missing_coverage_items(
    *,
    registry_id: str,
    registry: Any,
    collaboration_surface: AgentCollaborationSurface,
    source_registry_refs: tuple[str, ...],
    participant_agent_ids: tuple[str, ...],
    collaboration_contract_ids: tuple[str, ...],
    blocked_runtime_behaviors: tuple[str, ...],
) -> tuple[str, ...]:
    missing: list[str] = []
    if getattr(registry, "role", "") != registry_id:
        missing.append("registry_role_mismatch")
    if not getattr(registry, "serialization_version", ""):
        missing.append("serialization_version_missing")
    if not source_registry_refs:
        missing.append("source_registry_refs_missing")
    if not getattr(registry, "metadata_only", False):
        missing.append("metadata_only_declaration_missing")
    if not participant_agent_ids:
        missing.append("participant_agent_ids_missing")
    if not set(participant_agent_ids).issubset(AGENT_CONTRACT_REGISTRY.agent_ids):
        missing.append("unknown_participant_agent_id")
    if not collaboration_contract_ids:
        missing.append("collaboration_contract_ids_missing")
    for blocked_behavior in _SURFACE_RUNTIME_BLOCKS[collaboration_surface]:
        if blocked_behavior not in blocked_runtime_behaviors:
            missing.append(f"{blocked_behavior}_block_missing")
    if not any(
        blocked_behavior in blocked_runtime_behaviors
        for blocked_behavior in _AGENT_INVOCATION_BLOCK_ALIASES
    ):
        missing.append("agent_invocation_block_missing")
    if "provider_or_model_routing" not in blocked_runtime_behaviors:
        missing.append("provider_model_routing_block_missing")
    if not any(
        blocked_behavior in blocked_runtime_behaviors
        for blocked_behavior in _RETRY_BLOCK_ALIASES
    ):
        missing.append("retry_triggering_block_missing")
    if "generated_output_modification" not in blocked_runtime_behaviors:
        missing.append("generated_output_mutation_block_missing")
    for flag in _PASSIVE_IMPLEMENTATION_FLAGS:
        if getattr(registry, flag, False):
            missing.append(f"{flag}_enabled")
    return tuple(missing)


def _audit_record(
    registry_id: str,
    collaboration_surface: AgentCollaborationSurface,
    registry_builder: Callable[[], Any],
) -> AgentCollaborationAuditRecord:
    registry = registry_builder()
    participant_agent_ids = (
        _coordination_agent_ids(registry)
        if registry.role == "agent_coordination_registry"
        else _participant_agent_ids(registry)
    )
    source_registry_refs = _source_registry_refs(registry)
    collaboration_contract_ids = _collaboration_contract_ids(registry)
    blocked_runtime_behaviors = _registry_blocked_runtime_behaviors(registry)
    return AgentCollaborationAuditRecord(
        registry_id=registry_id,
        registry_role=registry.role,
        collaboration_surface=collaboration_surface,
        registry_serialization_version=registry.serialization_version,
        source_registry_refs=source_registry_refs,
        participant_agent_ids=participant_agent_ids,
        collaboration_contract_ids=collaboration_contract_ids,
        collaboration_contract_count=len(collaboration_contract_ids),
        validated_collaboration_surfaces=_VALIDATED_COLLABORATION_SURFACES,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
        audit_findings=_AUDIT_FINDINGS,
        missing_coverage_items=_missing_coverage_items(
            registry_id=registry_id,
            registry=registry,
            collaboration_surface=collaboration_surface,
            source_registry_refs=source_registry_refs,
            participant_agent_ids=participant_agent_ids,
            collaboration_contract_ids=collaboration_contract_ids,
            blocked_runtime_behaviors=blocked_runtime_behaviors,
        ),
        registry_blocked_runtime_behaviors=blocked_runtime_behaviors,
        metadata_only_declared=registry.metadata_only,
    )


AGENT_COLLABORATION_AUDIT_RECORDS = tuple(
    _audit_record(*spec) for spec in _COLLABORATION_SPECS
)
AGENT_COLLABORATION_AUDIT_REGISTRY = AgentCollaborationAuditRegistry(
    audit_records=AGENT_COLLABORATION_AUDIT_RECORDS,
    registry_ids=tuple(record.registry_id for record in AGENT_COLLABORATION_AUDIT_RECORDS),
    collaboration_surfaces=tuple(
        record.collaboration_surface for record in AGENT_COLLABORATION_AUDIT_RECORDS
    ),
    agent_ids=AGENT_CONTRACT_REGISTRY.agent_ids,
    audit_count=len(AGENT_COLLABORATION_AUDIT_RECORDS),
    source_collaboration_registries=tuple(
        record.registry_id for record in AGENT_COLLABORATION_AUDIT_RECORDS
    ),
    source_registry_refs=tuple(
        dict.fromkeys(
            source_ref
            for record in AGENT_COLLABORATION_AUDIT_RECORDS
            for source_ref in record.source_registry_refs
        )
    ),
    validated_collaboration_surfaces=_VALIDATED_COLLABORATION_SURFACES,
    passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
)
