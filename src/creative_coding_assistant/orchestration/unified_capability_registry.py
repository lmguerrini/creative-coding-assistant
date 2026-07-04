"""V6.6 Unified Capability Registry metadata over the Cognitive OS stack."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_capabilities import (
    AgentCapabilityRegistry,
    agent_capability_registry,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
    CognitiveOSCapability,
    CognitiveOSLayer,
    CognitiveOSPosture,
)
from creative_coding_assistant.orchestration.model_capability_matrix import (
    ModelCapabilityMatrix,
    build_model_capability_matrix,
)
from creative_coding_assistant.orchestration.provider_capability_matrix import (
    ProviderCapabilityMatrix,
    build_provider_capability_matrix,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)
from creative_coding_assistant.orchestration.unified_agent_registry import (
    UnifiedAgentRegistryPlan,
    build_unified_agent_registry,
)

UNIFIED_CAPABILITY_REGISTRY_SERIALIZATION_VERSION = "unified_capability_registry.v1"
UNIFIED_CAPABILITY_REGISTRY_ROADMAP_ITEM = "Unified Capability Registry"
UNIFIED_CAPABILITY_REGISTRY_AUTHORITY_BOUNDARY = (
    "V6.6 Unified Capability Registry composes Cognitive OS capability "
    "metadata with the Unified Agent Registry, passive agent capability "
    "registry, model capability matrix, and provider capability matrix. It "
    "makes capability ownership, dependency traceability, explainability, "
    "governance, and HITL posture inspectable only; it does not score "
    "capabilities, activate capabilities, route work to agents, route "
    "providers or models, execute providers, control workflows, mutate "
    "registries, modify generated output, emit HITL requests, apply HITL "
    "decisions, or apply Runtime Evolution."
)

_SOURCE_REGISTRY_IDS = (
    "unified_agent_registry",
    "agent_capability_registry",
    "model_capability_matrix",
    "provider_capability_matrix",
)

_CAPABILITY_IDS = (
    "v6_1_adaptive_learning",
    "v6_2_creative_memory",
    "v6_3_knowledge_evolution",
    "v6_4_autonomous_research",
    "v6_5_self_evolution",
    "v6_6_cognitive_core",
)

_LAYER_KNOWLEDGE_NODE_IDS: dict[CognitiveOSLayer, str] = {
    "learning": "knowledge::v6_1_learning_node",
    "memory": "knowledge::v6_2_memory_node",
    "knowledge": "knowledge::v6_3_knowledge_node",
    "research": "knowledge::v6_4_research_node",
    "self_evolution": "knowledge::v6_5_self_evolution_node",
    "cognitive_core": "knowledge::v6_6_cognitive_core_node",
}


class UnifiedCapabilityRegistryEntry(BaseModel):
    """One Cognitive OS capability entry linked to source registries."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    capability_id: str = Field(min_length=1, max_length=80)
    capability_name: CognitiveOSCapability
    cognitive_layer: CognitiveOSLayer
    knowledge_node_id: str = Field(min_length=1, max_length=150)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    source_registry_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    agent_capability_profile_ids: tuple[str, ...] = Field(
        min_length=6,
        max_length=6,
    )
    model_capability_dimension_count: int = Field(ge=1, le=40)
    provider_candidate_count: int = Field(ge=1, le=12)
    dependency_contracts: tuple[str, ...] = Field(min_length=2, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=2, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=2, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_layer_and_sources(self) -> Self:
        expected_node = _LAYER_KNOWLEDGE_NODE_IDS[self.cognitive_layer]
        if self.knowledge_node_id != expected_node:
            raise ValueError("knowledge_node_id must match cognitive_layer")
        if self.source_registry_ids != _SOURCE_REGISTRY_IDS:
            raise ValueError("source_registry_ids must match capability sources")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class UnifiedCapabilityRegistryPlan(BaseModel):
    """Unified passive capability registry for the Cognitive OS."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["unified_capability_registry"] = "unified_capability_registry"
    serialization_version: Literal["unified_capability_registry.v1"] = (
        UNIFIED_CAPABILITY_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=UNIFIED_CAPABILITY_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1800,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    agent_registry_role: Literal["unified_agent_registry"]
    agent_registry_serialization_version: str = Field(min_length=1, max_length=120)
    knowledge_node_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    layer_order: tuple[CognitiveOSLayer, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    capability_entries: tuple[UnifiedCapabilityRegistryEntry, ...] = Field(
        min_length=6,
        max_length=6,
    )
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_count: int = Field(ge=6, le=6)
    source_registry_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    source_registry_serialization_versions: tuple[str, ...] = Field(
        min_length=4,
        max_length=4,
    )
    source_registry_count: int = Field(ge=4, le=4)
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    agent_count: int = Field(ge=12, le=12)
    agent_capability_profile_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    agent_capability_profile_count: int = Field(ge=6, le=6)
    model_capability_row_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    model_capability_row_count: int = Field(ge=4, le=4)
    model_capability_dimensions: tuple[str, ...] = Field(
        min_length=12,
        max_length=40,
    )
    model_capability_dimension_count: int = Field(ge=12, le=40)
    provider_capability_row_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    provider_capability_row_count: int = Field(ge=4, le=4)
    provider_candidate_ids: tuple[str, ...] = Field(min_length=5, max_length=12)
    provider_candidate_count: int = Field(ge=5, le=12)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    unified_capability_registry_implemented: Literal[True] = True
    unified_agent_registry_integrated: Literal[True] = True
    agent_capability_registry_integrated: Literal[True] = True
    model_capability_matrix_integrated: Literal[True] = True
    provider_capability_matrix_integrated: Literal[True] = True
    capability_lookup_implemented: Literal[True] = True
    capability_dependency_traceability_implemented: Literal[True] = True
    capability_governance_contract_implemented: Literal[True] = True
    capability_explainability_contract_implemented: Literal[True] = True
    capability_scoring_implemented: Literal[False] = False
    capability_activation_implemented: Literal[False] = False
    agent_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    activated_capability_ids: tuple[str, ...] = Field(default_factory=tuple)
    scored_capability_ids: tuple[str, ...] = Field(default_factory=tuple)
    routed_capability_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_capability_registry_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_sources(self) -> Self:
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_ids != tuple(
            entry.capability_id for entry in self.capability_entries
        ):
            raise ValueError("capability_ids must match capability entries")
        if self.capability_ids != _CAPABILITY_IDS:
            raise ValueError("capability_ids must match Cognitive OS capabilities")
        if self.capability_count != len(self.capability_entries):
            raise ValueError("capability_count must match capability entries")
        if self.source_registry_ids != _SOURCE_REGISTRY_IDS:
            raise ValueError("source_registry_ids must match capability sources")
        if self.source_registry_count != len(self.source_registry_ids):
            raise ValueError("source_registry_count must match source registries")
        if self.agent_count != len(self.agent_ids):
            raise ValueError("agent_count must match agent ids")
        if self.agent_capability_profile_count != len(
            self.agent_capability_profile_ids
        ):
            raise ValueError("agent_capability_profile_count must match profiles")
        if self.model_capability_row_count != len(self.model_capability_row_ids):
            raise ValueError("model_capability_row_count must match rows")
        if self.model_capability_dimension_count != len(
            self.model_capability_dimensions
        ):
            raise ValueError("model_capability_dimension_count must match dimensions")
        if self.provider_capability_row_count != len(self.provider_capability_row_ids):
            raise ValueError("provider_capability_row_count must match rows")
        if self.provider_candidate_count != len(self.provider_candidate_ids):
            raise ValueError("provider_candidate_count must match providers")
        declared_agents = set(self.agent_ids)
        declared_nodes = set(self.knowledge_node_ids)
        for entry in self.capability_entries:
            if entry.knowledge_node_id not in declared_nodes:
                raise ValueError("entry knowledge_node_id must be declared")
            if not set(entry.linked_agent_ids).issubset(declared_agents):
                raise ValueError("entry linked_agent_ids must be declared")
            if entry.agent_capability_profile_ids != self.agent_capability_profile_ids:
                raise ValueError("entry agent capability profiles must match")
        if self.covered_roadmap_items != (UNIFIED_CAPABILITY_REGISTRY_ROADMAP_ITEM,):
            raise ValueError("covered_roadmap_items must be Task 6 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.activated_capability_ids,
                self.scored_capability_ids,
                self.routed_capability_ids,
                self.mutated_capability_registry_ids,
                self.emitted_hitl_request_ids,
            )
        ):
            raise ValueError(
                "capability activation, scoring, routing, and mutation ids "
                "must be empty",
            )
        if not all(entry.advisory_only for entry in self.capability_entries):
            raise ValueError("all capability entries must be advisory only")
        return self


def build_unified_capability_registry(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    agent_registry: UnifiedAgentRegistryPlan | None = None,
    agent_capabilities: AgentCapabilityRegistry | None = None,
    model_capabilities: ModelCapabilityMatrix | None = None,
    provider_capabilities: ProviderCapabilityMatrix | None = None,
) -> UnifiedCapabilityRegistryPlan:
    """Build the passive V6.6 unified capability registry."""

    agents = agent_registry or build_unified_agent_registry(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    agent_caps = agent_capabilities or agent_capability_registry()
    model_matrix = model_capabilities or build_model_capability_matrix()
    provider_matrix = provider_capabilities or build_provider_capability_matrix()
    entries = _capability_entries(
        agent_registry=agents,
        agent_capabilities=agent_caps,
        model_capabilities=model_matrix,
        provider_capabilities=provider_matrix,
    )
    return UnifiedCapabilityRegistryPlan(
        route_name=agents.route_name,
        task_type=agents.task_type,
        execution_mode_ids=agents.execution_mode_ids,
        agent_registry_role=agents.role,
        agent_registry_serialization_version=agents.serialization_version,
        knowledge_node_ids=agents.knowledge_node_ids,
        layer_order=COGNITIVE_OS_LAYER_ORDER,
        capabilities=COGNITIVE_OS_CAPABILITIES,
        capability_entries=entries,
        capability_ids=tuple(entry.capability_id for entry in entries),
        capability_count=len(entries),
        source_registry_ids=_SOURCE_REGISTRY_IDS,
        source_registry_serialization_versions=(
            agents.serialization_version,
            agent_caps.serialization_version,
            model_matrix.serialization_version,
            provider_matrix.serialization_version,
        ),
        source_registry_count=len(_SOURCE_REGISTRY_IDS),
        agent_ids=agents.agent_ids,
        agent_count=agents.agent_count,
        agent_capability_profile_ids=agent_caps.capability_ids,
        agent_capability_profile_count=agent_caps.capability_count,
        model_capability_row_ids=model_matrix.row_ids,
        model_capability_row_count=model_matrix.row_count,
        model_capability_dimensions=model_matrix.capability_dimensions,
        model_capability_dimension_count=model_matrix.capability_dimension_count,
        provider_capability_row_ids=provider_matrix.row_ids,
        provider_capability_row_count=provider_matrix.row_count,
        provider_candidate_ids=provider_matrix.provider_candidate_ids,
        provider_candidate_count=provider_matrix.provider_candidate_count,
        covered_roadmap_items=(UNIFIED_CAPABILITY_REGISTRY_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture="guarded",
    )


def unified_capability_registry_entry_by_id(
    capability_id: str,
    registry: UnifiedCapabilityRegistryPlan | None = None,
) -> UnifiedCapabilityRegistryEntry | None:
    """Return one capability entry without scoring or activating it."""

    source_registry = registry or build_unified_capability_registry()
    for entry in source_registry.capability_entries:
        if entry.capability_id == capability_id:
            return entry
    return None


def unified_capability_registry_entry_for_layer(
    layer: CognitiveOSLayer,
    registry: UnifiedCapabilityRegistryPlan | None = None,
) -> UnifiedCapabilityRegistryEntry | None:
    """Return the capability entry for one cognitive layer."""

    source_registry = registry or build_unified_capability_registry()
    for entry in source_registry.capability_entries:
        if entry.cognitive_layer == layer:
            return entry
    return None


def unified_capability_registry_entries_for_agent(
    agent_id: str,
    registry: UnifiedCapabilityRegistryPlan | None = None,
) -> tuple[UnifiedCapabilityRegistryEntry, ...]:
    """Return capability entries linked to one passive agent id."""

    source_registry = registry or build_unified_capability_registry()
    return tuple(
        entry
        for entry in source_registry.capability_entries
        if agent_id in entry.linked_agent_ids
    )


def _capability_entries(
    *,
    agent_registry: UnifiedAgentRegistryPlan,
    agent_capabilities: AgentCapabilityRegistry,
    model_capabilities: ModelCapabilityMatrix,
    provider_capabilities: ProviderCapabilityMatrix,
) -> tuple[UnifiedCapabilityRegistryEntry, ...]:
    return tuple(
        UnifiedCapabilityRegistryEntry(
            capability_id=capability_id,
            capability_name=capability_name,
            cognitive_layer=layer,
            knowledge_node_id=_LAYER_KNOWLEDGE_NODE_IDS[layer],
            linked_agent_ids=tuple(
                entry.agent_id
                for entry in agent_registry.agent_entries
                if entry.cognitive_layer == layer
            ),
            source_registry_ids=_SOURCE_REGISTRY_IDS,
            agent_capability_profile_ids=agent_capabilities.capability_ids,
            model_capability_dimension_count=(
                model_capabilities.capability_dimension_count
            ),
            provider_candidate_count=provider_capabilities.provider_candidate_count,
            dependency_contracts=(
                "capability entry follows Cognitive OS layer order",
                f"knowledge node:{_LAYER_KNOWLEDGE_NODE_IDS[layer]}",
            ),
            governance_contracts=(
                "capability entry does not authorize capability activation",
                "HITL required before capability-driven runtime behavior",
            ),
            explanation_contracts=(
                "capability entry cites agent, model, and provider sources",
                "unified agent registry explains linked agent coverage",
            ),
            blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        for capability_id, capability_name, layer in zip(
            _CAPABILITY_IDS,
            COGNITIVE_OS_CAPABILITIES,
            COGNITIVE_OS_LAYER_ORDER,
            strict=True,
        )
    )
