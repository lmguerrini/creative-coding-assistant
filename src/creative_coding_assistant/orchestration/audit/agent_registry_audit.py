"""Passive V4.6 agent registry audit metadata."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_boundaries import (
    agent_boundary_registry,
)
from creative_coding_assistant.orchestration.agent_capabilities import (
    agent_capability_registry,
)
from creative_coding_assistant.orchestration.agent_capability_alignment import (
    agent_capability_alignment_registry,
)
from creative_coding_assistant.orchestration.agent_consensus import (
    consensus_builder_registry,
)
from creative_coding_assistant.orchestration.agent_contract_audit import (
    agent_contract_audit_registry,
)
from creative_coding_assistant.orchestration.agent_contracts import (
    AGENT_CONTRACT_REGISTRY,
    agent_contract_registry,
)
from creative_coding_assistant.orchestration.agent_coordination import (
    agent_coordination_registry,
)
from creative_coding_assistant.orchestration.agent_debate import agent_debate_registry
from creative_coding_assistant.orchestration.agent_dependency_graph import (
    agent_dependency_graph_registry,
)
from creative_coding_assistant.orchestration.agent_escalation_signals import (
    agent_escalation_signal_registry,
)
from creative_coding_assistant.orchestration.agent_identities import (
    agent_identity_registry,
)
from creative_coding_assistant.orchestration.agent_lifecycle import (
    agent_lifecycle_registry,
)
from creative_coding_assistant.orchestration.agent_memory_contracts import (
    agent_memory_contract_registry,
)
from creative_coding_assistant.orchestration.agent_metadata import (
    agent_metadata_registry,
)
from creative_coding_assistant.orchestration.agent_parallel_scheduling import (
    parallel_scheduling_registry,
)
from creative_coding_assistant.orchestration.agent_roles import agent_role_registry
from creative_coding_assistant.orchestration.agent_routing import (
    agent_routing_registry,
)
from creative_coding_assistant.orchestration.agent_state_synchronization import (
    agent_state_synchronization_registry,
)
from creative_coding_assistant.orchestration.orchestration_contract_integration import (
    orchestration_contract_integration_registry,
)
from creative_coding_assistant.orchestration.workflow_agent_handoff import (
    workflow_agent_handoff_registry,
)

AgentRegistryAuditKind = Literal[
    "v4_1_foundation",
    "v3_6_future_capability",
    "v4_2_agent_registry",
    "v4_2_integration",
    "v4_6_audit",
]
AgentRegistryAuditStage = Literal["v4_6_agent_registry_hardening"]
AgentRegistryAuditStatus = Literal["pass"]

AGENT_REGISTRY_AUDIT_ENTRY_SERIALIZATION_VERSION = "agent_registry_audit_entry.v1"
AGENT_REGISTRY_AUDIT_REGISTRY_SERIALIZATION_VERSION = "agent_registry_audit_registry.v1"
AGENT_REGISTRY_AUDIT_REGISTRY_AUTHORITY_BOUNDARY = (
    "V4.6 agent registry audit metadata checks passive agent registry "
    "discoverability, serialization, metadata-only declarations, agent-id "
    "alignment, source references, and blocked runtime behaviors only; it "
    "does not create agents, execute agents, route providers or models, "
    "select runtimes, trigger retries, control workflows, write memory, or "
    "modify generated output."
)

_COVERAGE_SURFACES = (
    "registry_role",
    "serialization_version",
    "metadata_only",
    "authority_boundary",
    "agent_alignment",
    "source_registry_refs",
    "blocked_runtime_behaviors",
)
_PASSIVE_BOUNDARY_FLAGS = (
    "agent_creation_blocked",
    "agent_execution_blocked",
    "provider_model_routing_blocked",
    "runtime_selection_blocked",
    "retry_triggering_blocked",
    "workflow_control_blocked",
    "generated_output_mutation_blocked",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "agent_creation",
    "agent_execution",
    "agent_invocation",
    "provider_or_model_routing",
    "runtime_selection",
    "workflow_control",
    "retry_or_refinement_triggering",
    "memory_write",
    "generated_output_modification",
)
_AUDIT_FINDINGS = (
    "registry_discoverability_confirmed",
    "serialization_metadata_confirmed",
    "metadata_only_boundary_confirmed",
    "agent_alignment_confirmed",
    "source_reference_surface_confirmed",
    "runtime_behavior_blocks_confirmed",
)
_ROUTING_BLOCK_ALIASES = (
    "provider_or_model_routing",
    "runtime_work_routing",
    "dynamic_agent_routing",
    "workflow_routing_change",
)
_REQUIRED_BLOCKS = ("generated_output_modification",)

_REGISTRY_SPECS: tuple[
    tuple[str, str, AgentRegistryAuditKind, Callable[[], Any]],
    ...,
] = (
    (
        "agent_identity_registry",
        "agent_identity_registry",
        "v4_1_foundation",
        agent_identity_registry,
    ),
    (
        "agent_contract_registry",
        "agent_contract_registry",
        "v4_1_foundation",
        agent_contract_registry,
    ),
    (
        "agent_memory_contract_registry",
        "agent_memory_contract_registry",
        "v4_1_foundation",
        agent_memory_contract_registry,
    ),
    (
        "agent_role_registry",
        "agent_role_registry",
        "v4_1_foundation",
        agent_role_registry,
    ),
    (
        "agent_boundary_registry",
        "agent_boundary_registry",
        "v4_1_foundation",
        agent_boundary_registry,
    ),
    (
        "agent_metadata_registry",
        "agent_metadata_registry",
        "v4_1_foundation",
        agent_metadata_registry,
    ),
    (
        "agent_capability_registry",
        "agent_capability_registry",
        "v3_6_future_capability",
        agent_capability_registry,
    ),
    (
        "agent_routing_registry",
        "agent_routing_registry",
        "v4_2_agent_registry",
        agent_routing_registry,
    ),
    (
        "agent_dependency_graph_registry",
        "agent_dependency_graph_registry",
        "v4_2_agent_registry",
        agent_dependency_graph_registry,
    ),
    (
        "parallel_scheduling_registry",
        "parallel_scheduling_registry",
        "v4_2_agent_registry",
        parallel_scheduling_registry,
    ),
    (
        "agent_coordination_registry",
        "agent_coordination_registry",
        "v4_2_agent_registry",
        agent_coordination_registry,
    ),
    (
        "agent_debate_registry",
        "agent_debate_registry",
        "v4_2_agent_registry",
        agent_debate_registry,
    ),
    (
        "consensus_builder_registry",
        "consensus_builder_registry",
        "v4_2_agent_registry",
        consensus_builder_registry,
    ),
    (
        "agent_capability_alignment_registry",
        "agent_capability_alignment_registry",
        "v4_2_agent_registry",
        agent_capability_alignment_registry,
    ),
    (
        "agent_escalation_signal_registry",
        "agent_escalation_signal_registry",
        "v4_2_agent_registry",
        agent_escalation_signal_registry,
    ),
    (
        "agent_lifecycle_registry",
        "agent_lifecycle_registry",
        "v4_2_agent_registry",
        agent_lifecycle_registry,
    ),
    (
        "agent_state_synchronization_registry",
        "agent_state_synchronization_registry",
        "v4_2_agent_registry",
        agent_state_synchronization_registry,
    ),
    (
        "workflow_agent_handoff_registry",
        "workflow_agent_handoff_registry",
        "v4_2_agent_registry",
        workflow_agent_handoff_registry,
    ),
    (
        "orchestration_contract_integration_registry",
        "orchestration_contract_integration_registry",
        "v4_2_integration",
        orchestration_contract_integration_registry,
    ),
    (
        "agent_contract_audit_registry",
        "agent_contract_audit_registry",
        "v4_6_audit",
        agent_contract_audit_registry,
    ),
)


class AgentRegistryAuditEntry(BaseModel):
    """One passive V4.6 audit entry for an agent registry surface."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    registry_id: str = Field(min_length=1, max_length=160)
    registry_role: str = Field(min_length=1, max_length=160)
    registry_kind: AgentRegistryAuditKind
    export_symbol: str = Field(min_length=1, max_length=160)
    registry_serialization_version: str = Field(min_length=1, max_length=120)
    linked_agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    source_registry_refs: tuple[str, ...] = Field(default_factory=tuple, max_length=24)
    coverage_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=7, max_length=7)
    audit_findings: tuple[str, ...] = Field(min_length=6, max_length=6)
    missing_coverage_items: tuple[str, ...] = Field(
        default_factory=tuple, max_length=16
    )
    registry_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=16,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    audit_stage: AgentRegistryAuditStage = "v4_6_agent_registry_hardening"
    audit_status: AgentRegistryAuditStatus = "pass"
    metadata_only_declared: Literal[True] = True
    active_runtime_path_implemented: Literal[False] = False
    agent_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["agent_registry_audit_entry.v1"] = (
        AGENT_REGISTRY_AUDIT_ENTRY_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentRegistryAuditRegistry(BaseModel):
    """Stable passive V4.6 registry-of-registries audit for agent metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_registry_audit_registry"] = "agent_registry_audit_registry"
    serialization_version: Literal["agent_registry_audit_registry.v1"] = (
        AGENT_REGISTRY_AUDIT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_REGISTRY_AUDIT_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    audit_stage: AgentRegistryAuditStage = "v4_6_agent_registry_hardening"
    audit_entries: tuple[AgentRegistryAuditEntry, ...] = Field(
        min_length=20,
        max_length=20,
    )
    registry_ids: tuple[str, ...] = Field(min_length=20, max_length=20)
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    audit_count: int = Field(ge=20, le=20)
    coverage_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    all_registries_covered: Literal[True] = True
    all_agent_ids_aligned: Literal[True] = True
    no_missing_coverage: Literal[True] = True
    active_runtime_audit_implemented: Literal[False] = False
    agent_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_audit_entries(self) -> Self:
        derived_registry_ids = tuple(entry.registry_id for entry in self.audit_entries)
        if len(set(derived_registry_ids)) != len(derived_registry_ids):
            raise ValueError("registry_ids must be unique")
        if self.registry_ids != derived_registry_ids:
            raise ValueError("registry_ids must match audit entries")
        if self.audit_count != len(self.audit_entries):
            raise ValueError("audit_count must match audit entries")
        known_agents = set(self.agent_ids)
        for entry in self.audit_entries:
            if entry.coverage_surfaces != self.coverage_surfaces:
                raise ValueError("coverage_surfaces must match registry")
            if entry.passive_boundary_flags != self.passive_boundary_flags:
                raise ValueError("passive_boundary_flags must match registry")
            if set(entry.linked_agent_ids) != known_agents:
                raise ValueError("linked_agent_ids must match known agents")
            if entry.missing_coverage_items:
                raise ValueError("audit entries must not contain missing coverage")
        return self


def agent_registry_audit_registry() -> AgentRegistryAuditRegistry:
    """Return passive V4.6 agent registry audit metadata."""

    return AGENT_REGISTRY_AUDIT_REGISTRY


def agent_registry_audit_by_registry_id(
    registry_id: str,
    registry: AgentRegistryAuditRegistry | None = None,
) -> AgentRegistryAuditEntry | None:
    """Return one passive agent registry audit entry by registry id."""

    source_registry = registry or AGENT_REGISTRY_AUDIT_REGISTRY
    for entry in source_registry.audit_entries:
        if entry.registry_id == registry_id:
            return entry
    return None


def agent_registry_audits_for_kind(
    registry_kind: str,
    registry: AgentRegistryAuditRegistry | None = None,
) -> tuple[AgentRegistryAuditEntry, ...]:
    """Return passive audit entries for one registry kind."""

    source_registry = registry or AGENT_REGISTRY_AUDIT_REGISTRY
    normalized_kind = str(registry_kind).strip()
    return tuple(
        entry
        for entry in source_registry.audit_entries
        if entry.registry_kind == normalized_kind
    )


def _linked_agent_ids(registry: Any) -> tuple[str, ...]:
    agent_ids = getattr(registry, "agent_ids", None)
    if agent_ids is None:
        return AGENT_CONTRACT_REGISTRY.agent_ids
    return tuple(agent_ids)


def _source_registry_refs(registry: Any) -> tuple[str, ...]:
    refs: list[str] = []
    for attr in (
        "source_contract_registry",
        "source_contract_registries",
        "source_registries",
        "source_v4_1_registries",
        "source_identity_registry",
        "audited_registry_refs",
    ):
        value = getattr(registry, attr, None)
        if isinstance(value, str):
            refs.append(value)
        elif value is not None:
            refs.extend(str(item) for item in value)
    return tuple(dict.fromkeys(refs))


def _registry_blocked_runtime_behaviors(registry: Any) -> tuple[str, ...]:
    blocked = getattr(registry, "blocked_runtime_behaviors", None)
    if blocked is None:
        return _BLOCKED_RUNTIME_BEHAVIORS
    return tuple(blocked)


def _missing_coverage_items(
    *,
    registry_id: str,
    registry_role: str,
    registry: Any,
    linked_agent_ids: tuple[str, ...],
    blocked_runtime_behaviors: tuple[str, ...],
) -> tuple[str, ...]:
    missing: list[str] = []
    if registry_role != registry_id:
        missing.append("registry_role_mismatch")
    if not getattr(registry, "serialization_version", ""):
        missing.append("serialization_version_missing")
    if not getattr(registry, "metadata_only", False):
        missing.append("metadata_only_declaration_missing")
    if not getattr(registry, "authority_boundary", ""):
        missing.append("authority_boundary_missing")
    if set(linked_agent_ids) != set(AGENT_CONTRACT_REGISTRY.agent_ids):
        missing.append("agent_id_alignment_missing")
    if len(linked_agent_ids) != len(AGENT_CONTRACT_REGISTRY.agent_ids):
        missing.append("agent_id_count_mismatch")
    for blocked_behavior in _REQUIRED_BLOCKS:
        if blocked_behavior not in blocked_runtime_behaviors:
            missing.append(f"{blocked_behavior}_block_missing")
    if not any(
        blocked_behavior in blocked_runtime_behaviors
        for blocked_behavior in _ROUTING_BLOCK_ALIASES
    ):
        missing.append("routing_block_missing")
    return tuple(missing)


def _audit_entry(
    registry_id: str,
    export_symbol: str,
    registry_kind: AgentRegistryAuditKind,
    registry_builder: Callable[[], Any],
) -> AgentRegistryAuditEntry:
    registry = registry_builder()
    linked_agent_ids = _linked_agent_ids(registry)
    blocked_runtime_behaviors = _registry_blocked_runtime_behaviors(registry)
    return AgentRegistryAuditEntry(
        registry_id=registry_id,
        registry_role=registry.role,
        registry_kind=registry_kind,
        export_symbol=export_symbol,
        registry_serialization_version=registry.serialization_version,
        linked_agent_ids=linked_agent_ids,
        source_registry_refs=_source_registry_refs(registry),
        coverage_surfaces=_COVERAGE_SURFACES,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
        audit_findings=_AUDIT_FINDINGS,
        missing_coverage_items=_missing_coverage_items(
            registry_id=registry_id,
            registry_role=registry.role,
            registry=registry,
            linked_agent_ids=linked_agent_ids,
            blocked_runtime_behaviors=blocked_runtime_behaviors,
        ),
        registry_blocked_runtime_behaviors=blocked_runtime_behaviors,
        metadata_only_declared=registry.metadata_only,
    )


AGENT_REGISTRY_AUDIT_ENTRIES = tuple(_audit_entry(*spec) for spec in _REGISTRY_SPECS)
AGENT_REGISTRY_AUDIT_REGISTRY = AgentRegistryAuditRegistry(
    audit_entries=AGENT_REGISTRY_AUDIT_ENTRIES,
    registry_ids=tuple(entry.registry_id for entry in AGENT_REGISTRY_AUDIT_ENTRIES),
    agent_ids=AGENT_CONTRACT_REGISTRY.agent_ids,
    audit_count=len(AGENT_REGISTRY_AUDIT_ENTRIES),
    coverage_surfaces=_COVERAGE_SURFACES,
    passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
)
