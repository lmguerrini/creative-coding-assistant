"""Passive V4.1 agent contract foundation metadata."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

AgentContractCategory = Literal["multi_agent_core"]
AgentContractStage = Literal["v4_1_contract_foundation"]
AgentContractCacheability = Literal[
    "deterministic_static_metadata",
    "deterministic_with_upstream_metadata",
]
AgentContractCostClass = Literal["none", "low", "medium"]
AgentContractLatencyClass = Literal["none", "low", "medium"]
AgentMemoryAccessMode = Literal[
    "no_runtime_memory",
    "metadata_reference_only",
]

AGENT_CONTRACT_SERIALIZATION_VERSION = "agent_contract.v1"
AGENT_CONTRACT_REGISTRY_SERIALIZATION_VERSION = "agent_contract_registry.v1"
AGENT_CONTRACT_REGISTRY_AUTHORITY_BOUNDARY = (
    "V4.1 agent contracts describe passive agent identities, roles, "
    "capabilities, boundaries, metadata inputs and outputs, memory access "
    "posture, cost hints, latency hints, and future orchestration hooks only; "
    "they do not create agents, invoke agents, route work, call providers, "
    "select models or runtimes, coordinate collaboration, trigger retries, "
    "write memory, execute artifacts, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "agent_instantiation",
    "agent_invocation",
    "dynamic_agent_routing",
    "multi_agent_orchestration",
    "provider_or_model_routing",
    "runtime_selection",
    "workflow_control",
    "retry_or_refinement_triggering",
    "memory_storage_or_mutation",
    "artifact_execution",
    "generated_output_modification",
)

_FORBIDDEN_MEMORY_OPERATIONS = (
    "memory_write",
    "memory_store_creation",
    "blackboard_memory_mutation",
    "shared_context_mutation",
)


class AgentMemoryAccessContract(BaseModel):
    """Metadata-only description of an agent contract's memory posture."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    access_mode: AgentMemoryAccessMode = "metadata_reference_only"
    allowed_memory_sources: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    forbidden_memory_operations: tuple[str, ...] = Field(
        default=_FORBIDDEN_MEMORY_OPERATIONS,
        min_length=1,
        max_length=12,
    )
    retention_policy: str = Field(
        default=(
            "Agent contracts may reference existing metadata but do not retain "
            "or write memory."
        ),
        min_length=1,
        max_length=260,
    )
    reads_runtime_memory: Literal[False] = False
    writes_runtime_memory: Literal[False] = False
    creates_memory_store: Literal[False] = False
    metadata_only: Literal[True] = True


class AgentContractCostMetadata(BaseModel):
    """Static estimated cost metadata for a passive agent contract."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    relative_cost: AgentContractCostClass = "none"
    external_provider_calls: Literal[False] = False
    cost_basis: str = Field(min_length=1, max_length=260)
    cache_sensitivity: str = Field(min_length=1, max_length=260)


class AgentContractLatencyMetadata(BaseModel):
    """Static estimated latency metadata for a passive agent contract."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    relative_latency: AgentContractLatencyClass = "none"
    network_required: Literal[False] = False
    latency_basis: str = Field(min_length=1, max_length=260)
    blocking_inputs: tuple[str, ...] = Field(default_factory=tuple, max_length=16)


class AgentContract(BaseModel):
    """Common metadata contract surface for a future V4.1 agent."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=80)
    agent_name: str = Field(min_length=1, max_length=140)
    agent_version: str = Field(min_length=1, max_length=24)
    agent_category: AgentContractCategory = "multi_agent_core"
    contract_stage: AgentContractStage = "v4_1_contract_foundation"
    role_id: str = Field(min_length=1, max_length=80)
    role_name: str = Field(min_length=1, max_length=140)
    role_purpose: str = Field(min_length=1, max_length=260)
    authority_boundary: str = Field(min_length=1, max_length=900)
    allowed_actions: tuple[str, ...] = Field(min_length=1, max_length=16)
    prohibited_actions: tuple[str, ...] = Field(min_length=1, max_length=16)
    capabilities: tuple[str, ...] = Field(min_length=1, max_length=16)
    required_inputs: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    optional_inputs: tuple[str, ...] = Field(default_factory=tuple, max_length=24)
    produced_outputs: tuple[str, ...] = Field(min_length=1, max_length=16)
    produced_metadata: tuple[str, ...] = Field(min_length=1, max_length=18)
    produced_signals: tuple[str, ...] = Field(min_length=1, max_length=18)
    memory_access: AgentMemoryAccessContract
    cacheability: AgentContractCacheability = "deterministic_static_metadata"
    estimated_cost_metadata: AgentContractCostMetadata
    estimated_latency_metadata: AgentContractLatencyMetadata
    future_orchestration_hooks: tuple[str, ...] = Field(min_length=1, max_length=12)
    source_contract_registries: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    serialization_version: Literal["agent_contract.v1"] = (
        AGENT_CONTRACT_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentContractRegistry(BaseModel):
    """Stable registry envelope for passive V4.1 agent contracts."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_contract_registry"] = "agent_contract_registry"
    agent_category: AgentContractCategory = "multi_agent_core"
    serialization_version: Literal["agent_contract_registry.v1"] = (
        AGENT_CONTRACT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_CONTRACT_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    contracts: tuple[AgentContract, ...] = Field(default_factory=tuple, max_length=32)
    agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=32)
    contract_count: int = Field(ge=0, le=32)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_contracts(self) -> Self:
        derived_agent_ids = tuple(contract.agent_id for contract in self.contracts)
        if len(set(derived_agent_ids)) != len(derived_agent_ids):
            raise ValueError("agent_ids must be unique")
        if self.agent_ids != derived_agent_ids:
            raise ValueError("agent_ids must match contracts")
        if self.contract_count != len(self.contracts):
            raise ValueError("contract_count must match contracts")
        return self


def build_agent_contract_registry(
    contracts: Iterable[AgentContract],
) -> AgentContractRegistry:
    """Build a passive registry envelope without creating executable agents."""

    contract_tuple = tuple(contracts)
    return AgentContractRegistry(
        contracts=contract_tuple,
        agent_ids=tuple(contract.agent_id for contract in contract_tuple),
        contract_count=len(contract_tuple),
    )


def agent_contract_registry() -> AgentContractRegistry:
    """Return the static V4.1 agent contract foundation registry."""

    return AGENT_CONTRACT_REGISTRY


def agent_contract_by_id(
    agent_id: str,
    registry: AgentContractRegistry | None = None,
) -> AgentContract | None:
    """Return one passive contract by id without invoking or routing agents."""

    source_registry = registry or AGENT_CONTRACT_REGISTRY
    for contract in source_registry.contracts:
        if contract.agent_id == agent_id:
            return contract
    return None


PLANNER_AGENT_CONTRACT = AgentContract(
    agent_id="planner_agent",
    agent_name="Planner Agent",
    agent_version="v4.1",
    role_id="planner",
    role_name="Planner Agent",
    role_purpose=(
        "Represent planning context, gaps, constraints, and handoff metadata "
        "for future V4 orchestration."
    ),
    authority_boundary=(
        "Planner Agent contract metadata maps existing V3 planning and "
        "intelligence outputs into a future planning handoff surface only; it "
        "does not execute a planner agent, change existing planner behavior, "
        "alter workflow node order, route providers or models, select "
        "runtimes, trigger retries, or modify generated output."
    ),
    allowed_actions=(
        "describe_planning_context_requirements",
        "map_upstream_planning_metadata",
        "declare_future_planning_handoff",
    ),
    prohibited_actions=(
        "planner_agent_execution",
        "workflow_node_order_change",
        "provider_or_model_routing",
        "runtime_selection",
        "retry_or_refinement_triggering",
        "generated_output_modification",
    ),
    capabilities=(
        "planning_context_mapping",
        "requirement_gap_detection",
        "handoff_metadata_preparation",
    ),
    required_inputs=(
        "assistant_request",
        "route_decision",
        "creative_execution_plan",
    ),
    optional_inputs=(
        "creative_translation",
        "creative_intent",
        "creative_hierarchy",
        "creative_constraints",
        "creative_strategy",
        "creative_technique",
        "artifact_engine_contract_registry",
        "evaluation_engine_contract_registry",
        "agent_memory_contract",
    ),
    produced_outputs=(
        "planner_context_packet_contract",
        "planning_gap_summary_contract",
        "planning_handoff_metadata_contract",
    ),
    produced_metadata=(
        "planning_scope_metadata",
        "planning_gap_metadata",
        "plan_step_metadata",
        "constraint_metadata",
        "planning_evidence_metadata",
    ),
    produced_signals=(
        "planning_confidence",
        "missing_information",
        "execution_complexity",
        "export_readiness",
        "runtime_availability",
    ),
    memory_access=AgentMemoryAccessContract(
        allowed_memory_sources=(
            "session_metadata",
            "artifact_metadata",
            "evaluation_metadata",
            "provenance_metadata",
            "future_blackboard_contract",
        ),
    ),
    cacheability="deterministic_with_upstream_metadata",
    estimated_cost_metadata=AgentContractCostMetadata(
        relative_cost="low",
        cost_basis=(
            "Static metadata mapping from existing planning and intelligence "
            "outputs; no provider calls or runtime execution."
        ),
        cache_sensitivity=(
            "Cache key must include request, route, and upstream planning "
            "metadata identifiers."
        ),
    ),
    estimated_latency_metadata=AgentContractLatencyMetadata(
        relative_latency="low",
        latency_basis=(
            "Bounded local metadata inspection with no network, storage, "
            "provider, workflow, or artifact execution."
        ),
        blocking_inputs=(
            "assistant_request",
            "route_decision",
            "creative_execution_plan",
        ),
    ),
    future_orchestration_hooks=(
        "v4_2_planner_context_handoff",
        "v4_2_planning_gap_review",
    ),
    source_contract_registries=(
        "agent_identity_registry",
        "agent_memory_contract_registry",
        "artifact_engine_contract_registry",
        "evaluation_engine_contract_registry",
    ),
)

RESEARCH_AGENT_CONTRACT = AgentContract(
    agent_id="research_agent",
    agent_name="Research Agent",
    agent_version="v4.1",
    role_id="research",
    role_name="Research Agent",
    role_purpose=(
        "Represent retrieval context, source evidence, and source gap metadata "
        "for future research-aware orchestration."
    ),
    authority_boundary=(
        "Research Agent contract metadata maps existing retrieval and source "
        "context boundaries for future source synthesis only; it does not add "
        "web research execution, change retrieval behavior, call external "
        "sources, route providers or models, select runtimes, trigger retries, "
        "or modify generated output."
    ),
    allowed_actions=(
        "describe_retrieval_context_requirements",
        "map_source_evidence_metadata",
        "declare_future_research_handoff",
    ),
    prohibited_actions=(
        "web_research_execution",
        "external_source_calling",
        "retrieval_behavior_change",
        "provider_or_model_routing",
        "runtime_selection",
        "generated_output_modification",
    ),
    capabilities=(
        "source_context_mapping",
        "evidence_gap_detection",
        "research_handoff_metadata_preparation",
    ),
    required_inputs=(
        "assistant_request",
        "retrieval_context",
        "source_metadata",
    ),
    optional_inputs=(
        "assembled_context",
        "retrieval_quality_metadata",
        "kb_source_health_metadata",
        "route_decision",
        "agent_memory_contract",
    ),
    produced_outputs=(
        "research_context_packet_contract",
        "source_gap_summary_contract",
        "evidence_handoff_metadata_contract",
    ),
    produced_metadata=(
        "source_context_metadata",
        "retrieval_gap_metadata",
        "evidence_summary_metadata",
        "source_reliability_metadata",
        "research_scope_metadata",
    ),
    produced_signals=(
        "source_coverage",
        "retrieval_confidence",
        "missing_source_context",
        "source_freshness",
        "evidence_density",
    ),
    memory_access=AgentMemoryAccessContract(
        allowed_memory_sources=(
            "session_metadata",
            "provenance_metadata",
            "retrieval_context_metadata",
            "future_blackboard_contract",
        ),
    ),
    cacheability="deterministic_with_upstream_metadata",
    estimated_cost_metadata=AgentContractCostMetadata(
        relative_cost="low",
        cost_basis=(
            "Static metadata mapping from existing retrieval and source "
            "context; no web, provider, or external source calls."
        ),
        cache_sensitivity=(
            "Cache key must include request, retrieval context, and source "
            "metadata identifiers."
        ),
    ),
    estimated_latency_metadata=AgentContractLatencyMetadata(
        relative_latency="low",
        latency_basis=(
            "Bounded local metadata inspection with no network, retrieval "
            "execution, storage, provider, or external source work."
        ),
        blocking_inputs=(
            "assistant_request",
            "retrieval_context",
            "source_metadata",
        ),
    ),
    future_orchestration_hooks=(
        "v4_2_research_context_handoff",
        "v4_2_source_gap_review",
    ),
    source_contract_registries=(
        "agent_identity_registry",
        "agent_memory_contract_registry",
        "retrieval_context_contract",
        "official_kb_source_registry",
        "source_health_metadata",
    ),
)

STYLE_AGENT_CONTRACT = AgentContract(
    agent_id="style_agent",
    agent_name="Style Agent",
    agent_version="v4.1",
    role_id="style",
    role_name="Style Agent",
    role_purpose=(
        "Represent style interpretation, visual coherence, and stylistic "
        "handoff metadata for future V4 orchestration."
    ),
    authority_boundary=(
        "Style Agent contract metadata maps existing creative style, motif, "
        "composition, and reference-fusion metadata boundaries only; it does "
        "not change style output generation, add style engine behavior, "
        "execute agents, route providers or models, select runtimes, trigger "
        "retries, or modify generated output."
    ),
    allowed_actions=(
        "describe_style_context_requirements",
        "map_visual_style_metadata",
        "declare_future_style_handoff",
    ),
    prohibited_actions=(
        "style_output_generation_change",
        "style_engine_behavior_change",
        "agent_invocation",
        "provider_or_model_routing",
        "runtime_selection",
        "generated_output_modification",
    ),
    capabilities=(
        "style_context_mapping",
        "stylistic_coherence_metadata",
        "visual_handoff_metadata_preparation",
    ),
    required_inputs=(
        "assistant_request",
        "creative_translation",
        "visual_style_guidance",
    ),
    optional_inputs=(
        "creative_intent",
        "creative_hierarchy",
        "creative_composition",
        "semantic_motif",
        "emotional_consistency",
        "reference_fusion",
        "shader_preset_guidance",
        "agent_memory_contract",
    ),
    produced_outputs=(
        "style_context_packet_contract",
        "stylistic_coherence_summary_contract",
        "style_handoff_metadata_contract",
    ),
    produced_metadata=(
        "visual_style_metadata",
        "style_constraint_metadata",
        "composition_style_metadata",
        "reference_style_metadata",
        "style_coherence_metadata",
    ),
    produced_signals=(
        "style_alignment",
        "style_confidence",
        "style_ambiguity",
        "reference_style_coverage",
        "visual_coherence",
    ),
    memory_access=AgentMemoryAccessContract(
        allowed_memory_sources=(
            "session_metadata",
            "artifact_metadata",
            "provenance_metadata",
            "future_blackboard_contract",
        ),
    ),
    cacheability="deterministic_with_upstream_metadata",
    estimated_cost_metadata=AgentContractCostMetadata(
        relative_cost="low",
        cost_basis=(
            "Static metadata mapping from existing creative style signals; no "
            "style generation, provider calls, or runtime execution."
        ),
        cache_sensitivity=(
            "Cache key must include request, creative translation, and visual "
            "style metadata identifiers."
        ),
    ),
    estimated_latency_metadata=AgentContractLatencyMetadata(
        relative_latency="low",
        latency_basis=(
            "Bounded local metadata inspection with no network, provider, "
            "style engine, workflow, or artifact execution."
        ),
        blocking_inputs=(
            "assistant_request",
            "creative_translation",
            "visual_style_guidance",
        ),
    ),
    future_orchestration_hooks=(
        "v4_2_style_context_handoff",
        "v4_2_stylistic_coherence_review",
    ),
    source_contract_registries=(
        "agent_identity_registry",
        "agent_memory_contract_registry",
        "visual_style_metadata",
        "creative_composition_metadata",
        "reference_fusion_metadata",
    ),
)

RUNTIME_AGENT_CONTRACT = AgentContract(
    agent_id="runtime_agent",
    agent_name="Runtime Agent",
    agent_version="v4.1",
    role_id="runtime",
    role_name="Runtime Agent",
    role_purpose=(
        "Represent runtime compatibility, environment constraints, and "
        "runtime-fit handoff metadata for future V4 orchestration."
    ),
    authority_boundary=(
        "Runtime Agent contract metadata maps existing runtime capability, "
        "compatibility, and artifact fit metadata boundaries only; it does not "
        "change runtime selection, route providers or models, execute runtime "
        "decisions, trigger retries, alter workflow control, or modify "
        "generated output."
    ),
    allowed_actions=(
        "describe_runtime_context_requirements",
        "map_runtime_compatibility_metadata",
        "declare_future_runtime_handoff",
    ),
    prohibited_actions=(
        "runtime_selection_change",
        "runtime_decision_execution",
        "provider_or_model_routing",
        "workflow_control",
        "retry_or_refinement_triggering",
        "generated_output_modification",
    ),
    capabilities=(
        "runtime_context_mapping",
        "compatibility_signal_metadata",
        "environment_handoff_metadata_preparation",
    ),
    required_inputs=(
        "assistant_request",
        "runtime_capabilities",
        "runtime_compatibility_profile",
    ),
    optional_inputs=(
        "artifact_capability_matrix",
        "creative_execution_plan",
        "route_decision",
        "workstation_engine_contract_registry",
        "agent_memory_contract",
    ),
    produced_outputs=(
        "runtime_context_packet_contract",
        "runtime_fit_summary_contract",
        "environment_handoff_metadata_contract",
    ),
    produced_metadata=(
        "runtime_capability_metadata",
        "runtime_compatibility_metadata",
        "environment_constraint_metadata",
        "runtime_fit_metadata",
        "unsupported_runtime_metadata",
    ),
    produced_signals=(
        "runtime_confidence",
        "runtime_fit_status",
        "supported_runtime",
        "unsupported_runtime",
        "environment_risk",
    ),
    memory_access=AgentMemoryAccessContract(
        allowed_memory_sources=(
            "session_metadata",
            "artifact_metadata",
            "provenance_metadata",
            "future_blackboard_contract",
        ),
    ),
    cacheability="deterministic_with_upstream_metadata",
    estimated_cost_metadata=AgentContractCostMetadata(
        relative_cost="low",
        cost_basis=(
            "Static metadata mapping from existing runtime compatibility "
            "signals; no runtime decision execution or provider calls."
        ),
        cache_sensitivity=(
            "Cache key must include request, runtime capability, and "
            "compatibility metadata identifiers."
        ),
    ),
    estimated_latency_metadata=AgentContractLatencyMetadata(
        relative_latency="low",
        latency_basis=(
            "Bounded local metadata inspection with no network, provider, "
            "runtime execution, workflow, or artifact execution."
        ),
        blocking_inputs=(
            "assistant_request",
            "runtime_capabilities",
            "runtime_compatibility_profile",
        ),
    ),
    future_orchestration_hooks=(
        "v4_2_runtime_context_handoff",
        "v4_2_runtime_fit_review",
    ),
    source_contract_registries=(
        "agent_identity_registry",
        "agent_memory_contract_registry",
        "artifact_engine_contract_registry",
        "workstation_engine_contract_registry",
        "runtime_capability_metadata",
    ),
)

AGENT_CONTRACTS: tuple[AgentContract, ...] = (
    PLANNER_AGENT_CONTRACT,
    RESEARCH_AGENT_CONTRACT,
    STYLE_AGENT_CONTRACT,
    RUNTIME_AGENT_CONTRACT,
)
AGENT_CONTRACT_REGISTRY = build_agent_contract_registry(AGENT_CONTRACTS)
