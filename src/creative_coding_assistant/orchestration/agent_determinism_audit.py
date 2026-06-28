"""Passive V4.6 agent determinism audit metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_contracts import (
    AGENT_CONTRACT_REGISTRY,
    AgentContract,
    agent_contract_by_id,
)
from creative_coding_assistant.orchestration.agent_dependency_graph import (
    AgentDependencyNode,
    agent_dependency_graph_registry,
    agent_dependency_node_by_id,
)
from creative_coding_assistant.orchestration.agent_parallel_scheduling import (
    ParallelSchedulingGroup,
    parallel_scheduling_group_for_agent,
    parallel_scheduling_registry,
)
from creative_coding_assistant.orchestration.agent_routing import (
    AgentRoutingProfile,
    agent_routing_profile_by_agent_id,
    agent_routing_registry,
)
from creative_coding_assistant.orchestration.engine_contract_consistency import (
    engine_contract_consistency_registry,
)

AgentDeterminismAuditStage = Literal["v4_6_agent_determinism_hardening"]
AgentDeterminismAuditStatus = Literal["pass"]

AGENT_DETERMINISM_AUDIT_RECORD_SERIALIZATION_VERSION = (
    "agent_determinism_audit_record.v1"
)
AGENT_DETERMINISM_AUDIT_REGISTRY_SERIALIZATION_VERSION = (
    "agent_determinism_audit_registry.v1"
)
AGENT_DETERMINISM_AUDIT_REGISTRY_AUTHORITY_BOUNDARY = (
    "V4.6 agent determinism audit metadata checks passive cacheability "
    "declarations, ordered contract inputs and outputs, dependency graph "
    "topological order, routing candidate order, parallel scheduling group "
    "order, engine contract consistency family order, serialization "
    "metadata, and blocked runtime behavior declarations only; it does not "
    "generate seeds, sample randomness, invoke agents, change workflow order, "
    "mutate route selection, run parallel scheduling, route providers or "
    "models, select runtimes, trigger retries, alter prompt rendering, or "
    "modify generated output."
)

_SOURCE_DETERMINISM_REGISTRIES = (
    "agent_contract_registry",
    "agent_dependency_graph_registry",
    "agent_routing_registry",
    "parallel_scheduling_registry",
    "engine_contract_consistency_registry",
)
_VALIDATED_DETERMINISM_SURFACES = (
    "contract_cacheability",
    "contract_input_output_order",
    "dependency_topological_order",
    "routing_candidate_order",
    "scheduling_group_order",
    "engine_contract_family_order",
    "serialization_versions",
    "blocked_runtime_behaviors",
)
_PASSIVE_BOUNDARY_FLAGS = (
    "random_seed_generation_blocked",
    "nondeterministic_sampling_blocked",
    "agent_invocation_blocked",
    "workflow_order_mutation_blocked",
    "route_selection_mutation_blocked",
    "parallel_execution_blocked",
    "provider_model_routing_blocked",
    "generated_output_mutation_blocked",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "random_seed_generation",
    "nondeterministic_sampling",
    "agent_invocation",
    "workflow_node_order_change",
    "route_selection_mutation",
    "parallel_task_execution",
    "provider_or_model_routing",
    "runtime_selection",
    "retry_or_refinement_triggering",
    "prompt_rendering_change",
    "generated_output_modification",
)
_AUDIT_FINDINGS = (
    "deterministic_cacheability_confirmed",
    "contract_order_surfaces_confirmed",
    "dependency_topological_order_confirmed",
    "routing_candidate_order_confirmed",
    "scheduling_group_order_confirmed",
    "engine_contract_family_order_confirmed",
    "runtime_determinism_blocks_confirmed",
)


class AgentDeterminismAuditRecord(BaseModel):
    """One passive V4.6 determinism audit record for an agent."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=80)
    role_id: str = Field(min_length=1, max_length=80)
    audit_stage: AgentDeterminismAuditStage = "v4_6_agent_determinism_hardening"
    contract_serialization_version: str = Field(min_length=1, max_length=80)
    contract_cacheability: str = Field(min_length=1, max_length=80)
    contract_required_input_order: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=16,
    )
    contract_optional_input_order: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=24,
    )
    contract_output_order: tuple[str, ...] = Field(min_length=1, max_length=16)
    dependency_node_id: str = Field(min_length=1, max_length=160)
    dependency_stage_id: str = Field(min_length=1, max_length=80)
    dependency_upstream_node_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=32,
    )
    dependency_downstream_node_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=32,
    )
    routing_priority_band: str = Field(min_length=1, max_length=80)
    route_candidates: tuple[str, ...] = Field(min_length=1, max_length=6)
    scheduling_group_id: str = Field(min_length=1, max_length=120)
    scheduling_group_agent_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    scheduling_blocking_group_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    scheduling_downstream_group_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    consistency_family_ids: tuple[str, ...] = Field(min_length=3, max_length=3)
    determinism_source_registries: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    validated_determinism_surfaces: tuple[str, ...] = Field(
        min_length=8,
        max_length=8,
    )
    passive_boundary_flags: tuple[str, ...] = Field(min_length=8, max_length=8)
    audit_findings: tuple[str, ...] = Field(min_length=7, max_length=7)
    missing_coverage_items: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    contract_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=16,
    )
    dependency_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    routing_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=16,
    )
    scheduling_blocked_runtime_behaviors: tuple[str, ...] = Field(
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
    audit_status: AgentDeterminismAuditStatus = "pass"
    metadata_only_declared: Literal[True] = True
    deterministic_cacheability_declared: Literal[True] = True
    dependency_order_reference_present: Literal[True] = True
    routing_order_reference_present: Literal[True] = True
    scheduling_order_reference_present: Literal[True] = True
    random_seed_generation_implemented: Literal[False] = False
    nondeterministic_sampling_implemented: Literal[False] = False
    active_agent_execution_implemented: Literal[False] = False
    workflow_order_mutation_implemented: Literal[False] = False
    route_selection_mutation_implemented: Literal[False] = False
    parallel_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_rendering_change_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["agent_determinism_audit_record.v1"] = (
        AGENT_DETERMINISM_AUDIT_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentDeterminismAuditRegistry(BaseModel):
    """Stable passive V4.6 audit registry for agent determinism metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_determinism_audit_registry"] = (
        "agent_determinism_audit_registry"
    )
    serialization_version: Literal["agent_determinism_audit_registry.v1"] = (
        AGENT_DETERMINISM_AUDIT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_DETERMINISM_AUDIT_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    audit_stage: AgentDeterminismAuditStage = "v4_6_agent_determinism_hardening"
    audit_records: tuple[AgentDeterminismAuditRecord, ...] = Field(
        min_length=12,
        max_length=12,
    )
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    audit_count: int = Field(ge=12, le=12)
    source_agent_contract_registry: Literal["agent_contract_registry"] = (
        "agent_contract_registry"
    )
    source_dependency_graph_registry: Literal["agent_dependency_graph_registry"] = (
        "agent_dependency_graph_registry"
    )
    source_agent_routing_registry: Literal["agent_routing_registry"] = (
        "agent_routing_registry"
    )
    source_parallel_scheduling_registry: Literal["parallel_scheduling_registry"] = (
        "parallel_scheduling_registry"
    )
    source_engine_contract_consistency_registry: Literal[
        "engine_contract_consistency_registry"
    ] = "engine_contract_consistency_registry"
    determinism_source_registries: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    cacheability_modes: tuple[str, ...] = Field(min_length=1, max_length=4)
    dependency_stage_order: tuple[str, ...] = Field(min_length=6, max_length=6)
    dependency_topological_node_order: tuple[str, ...] = Field(
        min_length=30,
        max_length=30,
    )
    route_names: tuple[str, ...] = Field(min_length=6, max_length=6)
    scheduling_group_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    scheduling_agent_order: tuple[str, ...] = Field(min_length=12, max_length=12)
    consistency_family_ids: tuple[str, ...] = Field(min_length=3, max_length=3)
    validated_determinism_surfaces: tuple[str, ...] = Field(
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
    stable_order_references_present: Literal[True] = True
    no_missing_coverage: Literal[True] = True
    active_determinism_engine_implemented: Literal[False] = False
    random_seed_generation_implemented: Literal[False] = False
    nondeterministic_sampling_implemented: Literal[False] = False
    active_agent_execution_implemented: Literal[False] = False
    workflow_order_mutation_implemented: Literal[False] = False
    route_selection_mutation_implemented: Literal[False] = False
    parallel_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_rendering_change_implemented: Literal[False] = False
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
        if self.determinism_source_registries != _SOURCE_DETERMINISM_REGISTRIES:
            raise ValueError("determinism_source_registries must match sources")

        known_topological_nodes = set(self.dependency_topological_node_order)
        known_group_ids = set(self.scheduling_group_ids)
        known_routes = set(self.route_names)
        for record in self.audit_records:
            if record.audit_stage != self.audit_stage:
                raise ValueError("audit_stage must match registry")
            if record.contract_cacheability not in self.cacheability_modes:
                raise ValueError("contract_cacheability must be known")
            if not record.contract_cacheability.startswith("deterministic_"):
                raise ValueError("contract_cacheability must be deterministic")
            if record.dependency_node_id not in known_topological_nodes:
                raise ValueError("dependency_node_id must be known")
            if not set(record.route_candidates).issubset(known_routes):
                raise ValueError("route_candidates must be known")
            if record.scheduling_group_id not in known_group_ids:
                raise ValueError("scheduling_group_id must be known")
            if record.consistency_family_ids != self.consistency_family_ids:
                raise ValueError("consistency_family_ids must match registry")
            if record.determinism_source_registries != (
                self.determinism_source_registries
            ):
                raise ValueError("determinism_source_registries must match registry")
            if record.validated_determinism_surfaces != (
                self.validated_determinism_surfaces
            ):
                raise ValueError("validated_determinism_surfaces must match registry")
            if record.passive_boundary_flags != self.passive_boundary_flags:
                raise ValueError("passive_boundary_flags must match registry")
            if record.missing_coverage_items:
                raise ValueError("audit records must not contain missing coverage")
        return self


def agent_determinism_audit_registry() -> AgentDeterminismAuditRegistry:
    """Return passive V4.6 agent determinism audit metadata."""

    return AGENT_DETERMINISM_AUDIT_REGISTRY


def agent_determinism_audit_by_agent_id(
    agent_id: str,
    registry: AgentDeterminismAuditRegistry | None = None,
) -> AgentDeterminismAuditRecord | None:
    """Return one passive determinism audit record by agent id."""

    source_registry = registry or AGENT_DETERMINISM_AUDIT_REGISTRY
    for record in source_registry.audit_records:
        if record.agent_id == agent_id:
            return record
    return None


def agent_determinism_audits_for_cacheability(
    cacheability: str,
    registry: AgentDeterminismAuditRegistry | None = None,
) -> tuple[AgentDeterminismAuditRecord, ...]:
    """Return passive determinism audits for one cacheability mode."""

    source_registry = registry or AGENT_DETERMINISM_AUDIT_REGISTRY
    normalized_cacheability = str(cacheability).strip()
    return tuple(
        record
        for record in source_registry.audit_records
        if record.contract_cacheability == normalized_cacheability
    )


def agent_determinism_audits_for_routing_priority_band(
    priority_band: str,
    registry: AgentDeterminismAuditRegistry | None = None,
) -> tuple[AgentDeterminismAuditRecord, ...]:
    """Return passive determinism audits for one routing priority band."""

    source_registry = registry or AGENT_DETERMINISM_AUDIT_REGISTRY
    normalized_band = str(priority_band).strip()
    return tuple(
        record
        for record in source_registry.audit_records
        if record.routing_priority_band == normalized_band
    )


def agent_determinism_audits_for_scheduling_group(
    scheduling_group_id: str,
    registry: AgentDeterminismAuditRegistry | None = None,
) -> tuple[AgentDeterminismAuditRecord, ...]:
    """Return passive determinism audits for one scheduling group."""

    source_registry = registry or AGENT_DETERMINISM_AUDIT_REGISTRY
    normalized_group_id = str(scheduling_group_id).strip()
    return tuple(
        record
        for record in source_registry.audit_records
        if record.scheduling_group_id == normalized_group_id
    )


def _missing_coverage_items(
    *,
    contract: AgentContract,
    dependency_node: AgentDependencyNode,
    routing_profile: AgentRoutingProfile,
    scheduling_group: ParallelSchedulingGroup,
) -> tuple[str, ...]:
    dependency_graph = agent_dependency_graph_registry()
    routing = agent_routing_registry()
    scheduling = parallel_scheduling_registry()
    consistency = engine_contract_consistency_registry()
    missing: list[str] = []
    if not contract.cacheability.startswith("deterministic_"):
        missing.append("deterministic_cacheability_missing")
    if not contract.produced_outputs:
        missing.append("contract_output_order_missing")
    if dependency_node.node_id not in dependency_graph.topological_node_order:
        missing.append("dependency_topological_order_missing")
    if not routing_profile.route_candidates:
        missing.append("routing_candidate_order_missing")
    if scheduling_group.group_id not in scheduling.group_ids:
        missing.append("scheduling_group_order_missing")
    if not consistency.family_ids:
        missing.append("engine_contract_family_order_missing")
    if not all(
        (
            contract.metadata_only,
            dependency_node.metadata_only,
            routing_profile.metadata_only,
            scheduling_group.metadata_only,
            consistency.metadata_only,
        )
    ):
        missing.append("metadata_only_declaration_missing")
    for blocked_behavior in (
        "agent_invocation",
        "provider_or_model_routing",
        "runtime_selection",
        "retry_or_refinement_triggering",
        "generated_output_modification",
    ):
        if blocked_behavior not in contract.blocked_runtime_behaviors:
            missing.append(f"{blocked_behavior}_block_missing")
    if "workflow_node_order_change" not in dependency_node.blocked_runtime_behaviors:
        missing.append("workflow_node_order_change_block_missing")
    if "active_dynamic_agent_routing" not in routing_profile.blocked_runtime_behaviors:
        missing.append("dynamic_agent_routing_block_missing")
    if "parallel_task_execution" not in scheduling_group.blocked_runtime_behaviors:
        missing.append("parallel_execution_block_missing")
    if "prompt_rendering_change" not in consistency.blocked_runtime_behaviors:
        missing.append("prompt_rendering_change_block_missing")
    if dependency_node.workflow_node_order_changed:
        missing.append("workflow_node_order_changed")
    if routing_profile.workflow_routing_implemented:
        missing.append("workflow_routing_enabled")
    if routing_profile.provider_model_routing_implemented:
        missing.append("provider_model_routing_enabled")
    if scheduling_group.parallel_execution_implemented:
        missing.append("parallel_execution_enabled")
    if scheduling_group.async_behavior_changed:
        missing.append("async_behavior_changed")
    return tuple(missing)


def _audit_record(agent_id: str) -> AgentDeterminismAuditRecord:
    contract = agent_contract_by_id(agent_id, AGENT_CONTRACT_REGISTRY)
    dependency_node = agent_dependency_node_by_id(
        f"agent::{agent_id}",
        agent_dependency_graph_registry(),
    )
    routing_profile = agent_routing_profile_by_agent_id(
        agent_id,
        agent_routing_registry(),
    )
    scheduling_group = parallel_scheduling_group_for_agent(
        agent_id,
        parallel_scheduling_registry(),
    )
    consistency = engine_contract_consistency_registry()
    if contract is None:
        raise ValueError(f"missing agent contract for {agent_id}")
    if dependency_node is None:
        raise ValueError(f"missing dependency graph node for {agent_id}")
    if routing_profile is None:
        raise ValueError(f"missing routing profile for {agent_id}")
    if scheduling_group is None:
        raise ValueError(f"missing scheduling group for {agent_id}")

    return AgentDeterminismAuditRecord(
        agent_id=agent_id,
        role_id=contract.role_id,
        contract_serialization_version=contract.serialization_version,
        contract_cacheability=contract.cacheability,
        contract_required_input_order=contract.required_inputs,
        contract_optional_input_order=contract.optional_inputs,
        contract_output_order=contract.produced_outputs,
        dependency_node_id=dependency_node.node_id,
        dependency_stage_id=dependency_node.stage_id,
        dependency_upstream_node_ids=dependency_node.upstream_node_ids,
        dependency_downstream_node_ids=dependency_node.downstream_node_ids,
        routing_priority_band=routing_profile.priority_band,
        route_candidates=tuple(route.value for route in routing_profile.route_candidates),
        scheduling_group_id=scheduling_group.group_id,
        scheduling_group_agent_ids=scheduling_group.agent_ids,
        scheduling_blocking_group_ids=scheduling_group.blocking_group_ids,
        scheduling_downstream_group_ids=scheduling_group.downstream_group_ids,
        consistency_family_ids=consistency.family_ids,
        determinism_source_registries=_SOURCE_DETERMINISM_REGISTRIES,
        validated_determinism_surfaces=_VALIDATED_DETERMINISM_SURFACES,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
        audit_findings=_AUDIT_FINDINGS,
        missing_coverage_items=_missing_coverage_items(
            contract=contract,
            dependency_node=dependency_node,
            routing_profile=routing_profile,
            scheduling_group=scheduling_group,
        ),
        contract_blocked_runtime_behaviors=contract.blocked_runtime_behaviors,
        dependency_blocked_runtime_behaviors=dependency_node.blocked_runtime_behaviors,
        routing_blocked_runtime_behaviors=routing_profile.blocked_runtime_behaviors,
        scheduling_blocked_runtime_behaviors=scheduling_group.blocked_runtime_behaviors,
        consistency_blocked_runtime_behaviors=consistency.blocked_runtime_behaviors,
        metadata_only_declared=(
            contract.metadata_only
            and dependency_node.metadata_only
            and routing_profile.metadata_only
            and scheduling_group.metadata_only
        ),
    )


AGENT_DETERMINISM_AUDIT_RECORDS = tuple(
    _audit_record(agent_id) for agent_id in AGENT_CONTRACT_REGISTRY.agent_ids
)
AGENT_DETERMINISM_AUDIT_REGISTRY = AgentDeterminismAuditRegistry(
    audit_records=AGENT_DETERMINISM_AUDIT_RECORDS,
    agent_ids=tuple(record.agent_id for record in AGENT_DETERMINISM_AUDIT_RECORDS),
    audit_count=len(AGENT_DETERMINISM_AUDIT_RECORDS),
    determinism_source_registries=_SOURCE_DETERMINISM_REGISTRIES,
    cacheability_modes=tuple(
        dict.fromkeys(
            contract.cacheability for contract in AGENT_CONTRACT_REGISTRY.contracts
        )
    ),
    dependency_stage_order=agent_dependency_graph_registry().stage_order,
    dependency_topological_node_order=(
        agent_dependency_graph_registry().topological_node_order
    ),
    route_names=tuple(route.value for route in agent_routing_registry().route_names),
    scheduling_group_ids=parallel_scheduling_registry().group_ids,
    scheduling_agent_order=parallel_scheduling_registry().agent_ids,
    consistency_family_ids=engine_contract_consistency_registry().family_ids,
    validated_determinism_surfaces=_VALIDATED_DETERMINISM_SURFACES,
    passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
)
