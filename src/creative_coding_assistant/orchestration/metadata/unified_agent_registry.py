"""V6.6 Unified Agent Registry metadata over the cognitive OS graph stack."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_capabilities import (
    agent_capability_registry,
)
from creative_coding_assistant.orchestration.agent_contracts import (
    AgentContractRegistry,
    agent_contract_registry,
)
from creative_coding_assistant.orchestration.agent_identities import (
    AgentIdentityMetadata,
    AgentIdentityRegistry,
    AgentRoleFamily,
    agent_identity_registry,
)
from creative_coding_assistant.orchestration.agent_metadata import (
    AgentMetadataRegistry,
    AgentOperationalMetadata,
    agent_metadata_registry,
)
from creative_coding_assistant.orchestration.agent_registry_audit import (
    AgentRegistryAuditRegistry,
    agent_registry_audit_registry,
)
from creative_coding_assistant.orchestration.agent_roles import (
    AgentRoleMetadata,
    AgentRoleRegistry,
    agent_role_registry,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    CognitiveOSCapability,
    CognitiveOSLayer,
    CognitiveOSPosture,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)
from creative_coding_assistant.orchestration.unified_knowledge_graph import (
    UnifiedKnowledgeGraphPlan,
    build_unified_knowledge_graph,
)

UNIFIED_AGENT_REGISTRY_SERIALIZATION_VERSION = "unified_agent_registry.v1"
UNIFIED_AGENT_REGISTRY_ROADMAP_ITEM = "Unified Agent Registry"
UNIFIED_AGENT_REGISTRY_AUTHORITY_BOUNDARY = (
    "V6.6 Unified Agent Registry composes passive V4 agent identity, "
    "contract, role, metadata, capability, and registry-audit surfaces into "
    "one inspectable cognitive registry. It links every agent to the Unified "
    "Knowledge Graph and preserves ownership, dependency traceability, "
    "explainability, safety, and HITL governance; it does not create agents, "
    "invoke agents, route work to agents, perform provider/model routing, "
    "control workflows, mutate registries, modify generated output, emit HITL "
    "requests, apply HITL decisions, or apply Runtime Evolution."
)

_SOURCE_REGISTRY_IDS = (
    "agent_identity_registry",
    "agent_contract_registry",
    "agent_role_registry",
    "agent_metadata_registry",
    "agent_capability_registry",
    "agent_registry_audit_registry",
)

_ROLE_FAMILY_LAYER: dict[AgentRoleFamily, CognitiveOSLayer] = {
    "planning": "cognitive_core",
    "research": "research",
    "style": "memory",
    "runtime": "cognitive_core",
    "artifact": "cognitive_core",
    "art_direction": "cognitive_core",
    "critique": "knowledge",
    "narrative": "memory",
    "curation": "knowledge",
    "refinement": "self_evolution",
    "synthesis": "cognitive_core",
}

_LAYER_KNOWLEDGE_NODE_IDS: dict[CognitiveOSLayer, str] = {
    "learning": "knowledge::v6_1_learning_node",
    "memory": "knowledge::v6_2_memory_node",
    "knowledge": "knowledge::v6_3_knowledge_node",
    "research": "knowledge::v6_4_research_node",
    "self_evolution": "knowledge::v6_5_self_evolution_node",
    "cognitive_core": "knowledge::v6_6_cognitive_core_node",
}


class UnifiedAgentRegistryEntry(BaseModel):
    """One source-aligned passive agent entry in the unified registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=80)
    role_id: str = Field(min_length=1, max_length=80)
    agent_name: str = Field(min_length=1, max_length=140)
    role_family: AgentRoleFamily
    capability_family: str = Field(min_length=1, max_length=120)
    cognitive_layer: CognitiveOSLayer
    knowledge_node_id: str = Field(min_length=1, max_length=150)
    source_registry_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    audited_registry_ids: tuple[str, ...] = Field(min_length=20, max_length=20)
    contract_serialization_version: Literal["agent_contract.v1"]
    identity_serialization_version: Literal["agent_identity.v1"]
    role_serialization_version: Literal["agent_role.v1"]
    metadata_serialization_version: Literal["agent_metadata.v1"]
    produced_outputs: tuple[str, ...] = Field(min_length=1, max_length=16)
    future_orchestration_hooks: tuple[str, ...] = Field(min_length=1, max_length=12)
    agent_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=16,
    )
    dependency_contracts: tuple[str, ...] = Field(min_length=2, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=2, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=2, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_layer_and_boundaries(self) -> Self:
        expected_node_id = _LAYER_KNOWLEDGE_NODE_IDS[self.cognitive_layer]
        if self.knowledge_node_id != expected_node_id:
            raise ValueError("knowledge_node_id must match cognitive_layer")
        if self.source_registry_ids != _SOURCE_REGISTRY_IDS:
            raise ValueError("source_registry_ids must match agent source registries")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class UnifiedAgentRegistryPlan(BaseModel):
    """Unified passive agent registry connected to the V6.6 graph stack."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["unified_agent_registry"] = "unified_agent_registry"
    serialization_version: Literal["unified_agent_registry.v1"] = (
        UNIFIED_AGENT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=UNIFIED_AGENT_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1800,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    knowledge_graph_role: Literal["unified_knowledge_graph"]
    knowledge_graph_serialization_version: str = Field(min_length=1, max_length=120)
    knowledge_node_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    source_registry_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_registry_serialization_versions: tuple[str, ...] = Field(
        min_length=6,
        max_length=6,
    )
    source_registry_count: int = Field(ge=6, le=6)
    audited_registry_ids: tuple[str, ...] = Field(min_length=20, max_length=20)
    audited_registry_count: int = Field(ge=20, le=20)
    agent_entries: tuple[UnifiedAgentRegistryEntry, ...] = Field(
        min_length=12,
        max_length=12,
    )
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    role_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    agent_count: int = Field(ge=12, le=12)
    role_count: int = Field(ge=12, le=12)
    role_families: tuple[str, ...] = Field(min_length=11, max_length=11)
    capability_families: tuple[str, ...] = Field(min_length=12, max_length=12)
    capability_profile_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_profile_count: int = Field(ge=6, le=6)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    unified_agent_registry_implemented: Literal[True] = True
    unified_knowledge_graph_integrated: Literal[True] = True
    agent_registry_audit_integrated: Literal[True] = True
    agent_identity_alignment_implemented: Literal[True] = True
    agent_contract_alignment_implemented: Literal[True] = True
    agent_role_alignment_implemented: Literal[True] = True
    agent_metadata_alignment_implemented: Literal[True] = True
    agent_dependency_traceability_implemented: Literal[True] = True
    agent_governance_contract_implemented: Literal[True] = True
    agent_explainability_contract_implemented: Literal[True] = True
    agent_creation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    dynamic_agent_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    created_agent_ids: tuple[str, ...] = Field(default_factory=tuple)
    invoked_agent_ids: tuple[str, ...] = Field(default_factory=tuple)
    routed_agent_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_agent_registry_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_sources(self) -> Self:
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.source_registry_ids != _SOURCE_REGISTRY_IDS:
            raise ValueError("source_registry_ids must match unified agent sources")
        if self.source_registry_count != len(self.source_registry_ids):
            raise ValueError("source_registry_count must match source registries")
        if self.audited_registry_count != len(self.audited_registry_ids):
            raise ValueError("audited_registry_count must match audited registries")
        if len(set(self.audited_registry_ids)) != len(self.audited_registry_ids):
            raise ValueError("audited_registry_ids must be unique")
        if self.agent_ids != tuple(entry.agent_id for entry in self.agent_entries):
            raise ValueError("agent_ids must match agent entries")
        if len(set(self.agent_ids)) != len(self.agent_ids):
            raise ValueError("agent_ids must be unique")
        if self.role_ids != tuple(entry.role_id for entry in self.agent_entries):
            raise ValueError("role_ids must match agent entries")
        if len(set(self.role_ids)) != len(self.role_ids):
            raise ValueError("role_ids must be unique")
        if self.agent_count != len(self.agent_entries):
            raise ValueError("agent_count must match agent entries")
        if self.role_count != len(self.role_ids):
            raise ValueError("role_count must match role ids")
        if self.role_families != tuple(
            dict.fromkeys(entry.role_family for entry in self.agent_entries)
        ):
            raise ValueError("role_families must match agent entries")
        if self.capability_families != tuple(
            dict.fromkeys(entry.capability_family for entry in self.agent_entries)
        ):
            raise ValueError("capability_families must match agent entries")
        declared_knowledge_nodes = set(self.knowledge_node_ids)
        for entry in self.agent_entries:
            if entry.knowledge_node_id not in declared_knowledge_nodes:
                raise ValueError("entry knowledge_node_id must be declared")
            if entry.audited_registry_ids != self.audited_registry_ids:
                raise ValueError("entry audited_registry_ids must match registry")
        if self.covered_roadmap_items != (UNIFIED_AGENT_REGISTRY_ROADMAP_ITEM,):
            raise ValueError("covered_roadmap_items must be Task 5 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.created_agent_ids,
                self.invoked_agent_ids,
                self.routed_agent_ids,
                self.mutated_agent_registry_ids,
                self.emitted_hitl_request_ids,
            )
        ):
            raise ValueError(
                "agent creation, invocation, routing, and mutation ids must be empty",
            )
        if not all(entry.advisory_only for entry in self.agent_entries):
            raise ValueError("all agent registry entries must be advisory only")
        return self


def build_unified_agent_registry(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    knowledge_graph: UnifiedKnowledgeGraphPlan | None = None,
) -> UnifiedAgentRegistryPlan:
    """Build the passive V6.6 unified agent registry."""

    graph = knowledge_graph or build_unified_knowledge_graph(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    identities = agent_identity_registry()
    contracts = agent_contract_registry()
    roles = agent_role_registry()
    metadata = agent_metadata_registry()
    capabilities = agent_capability_registry()
    audit = agent_registry_audit_registry()
    agent_entries = _agent_entries(
        identities=identities,
        contracts=contracts,
        roles=roles,
        metadata=metadata,
        audit=audit,
    )
    return UnifiedAgentRegistryPlan(
        route_name=graph.route_name,
        task_type=graph.task_type,
        execution_mode_ids=graph.execution_mode_ids,
        knowledge_graph_role=graph.role,
        knowledge_graph_serialization_version=graph.serialization_version,
        knowledge_node_ids=graph.knowledge_node_ids,
        capabilities=graph.capabilities,
        source_registry_ids=_SOURCE_REGISTRY_IDS,
        source_registry_serialization_versions=(
            identities.serialization_version,
            contracts.serialization_version,
            roles.serialization_version,
            metadata.serialization_version,
            capabilities.serialization_version,
            audit.serialization_version,
        ),
        source_registry_count=len(_SOURCE_REGISTRY_IDS),
        audited_registry_ids=audit.registry_ids,
        audited_registry_count=audit.audit_count,
        agent_entries=agent_entries,
        agent_ids=tuple(entry.agent_id for entry in agent_entries),
        role_ids=tuple(entry.role_id for entry in agent_entries),
        agent_count=len(agent_entries),
        role_count=len(agent_entries),
        role_families=roles.role_families,
        capability_families=roles.capability_families,
        capability_profile_ids=capabilities.capability_ids,
        capability_profile_count=capabilities.capability_count,
        covered_roadmap_items=(UNIFIED_AGENT_REGISTRY_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture="guarded",
    )


def unified_agent_registry_entry_by_id(
    agent_id: str,
    registry: UnifiedAgentRegistryPlan | None = None,
) -> UnifiedAgentRegistryEntry | None:
    """Return one unified agent registry entry without invoking agents."""

    source_registry = registry or build_unified_agent_registry()
    for entry in source_registry.agent_entries:
        if entry.agent_id == agent_id:
            return entry
    return None


def unified_agent_registry_entries_for_layer(
    layer: CognitiveOSLayer,
    registry: UnifiedAgentRegistryPlan | None = None,
) -> tuple[UnifiedAgentRegistryEntry, ...]:
    """Return agent entries aligned to one cognitive layer."""

    source_registry = registry or build_unified_agent_registry()
    return tuple(
        entry
        for entry in source_registry.agent_entries
        if entry.cognitive_layer == layer
    )


def unified_agent_registry_entries_for_role_family(
    role_family: AgentRoleFamily,
    registry: UnifiedAgentRegistryPlan | None = None,
) -> tuple[UnifiedAgentRegistryEntry, ...]:
    """Return agent entries aligned to one role family."""

    source_registry = registry or build_unified_agent_registry()
    return tuple(
        entry
        for entry in source_registry.agent_entries
        if entry.role_family == role_family
    )


def _agent_entries(
    *,
    identities: AgentIdentityRegistry,
    contracts: AgentContractRegistry,
    roles: AgentRoleRegistry,
    metadata: AgentMetadataRegistry,
    audit: AgentRegistryAuditRegistry,
) -> tuple[UnifiedAgentRegistryEntry, ...]:
    identity_by_agent = {
        identity.agent_id: identity for identity in identities.identities
    }
    role_by_agent = {role.agent_id: role for role in roles.roles}
    metadata_by_agent = {item.agent_id: item for item in metadata.metadata}
    return tuple(
        _agent_entry(
            contract_agent_id=contract.agent_id,
            identity=identity_by_agent[contract.agent_id],
            role=role_by_agent[contract.agent_id],
            metadata=metadata_by_agent[contract.agent_id],
            contract_registry=contracts,
            audit=audit,
        )
        for contract in contracts.contracts
    )


def _agent_entry(
    *,
    contract_agent_id: str,
    identity: AgentIdentityMetadata,
    role: AgentRoleMetadata,
    metadata: AgentOperationalMetadata,
    contract_registry: AgentContractRegistry,
    audit: AgentRegistryAuditRegistry,
) -> UnifiedAgentRegistryEntry:
    contract = next(
        item
        for item in contract_registry.contracts
        if item.agent_id == contract_agent_id
    )
    cognitive_layer = _ROLE_FAMILY_LAYER[identity.role_family]
    return UnifiedAgentRegistryEntry(
        agent_id=contract.agent_id,
        role_id=contract.role_id,
        agent_name=contract.agent_name,
        role_family=identity.role_family,
        capability_family=identity.capability_class,
        cognitive_layer=cognitive_layer,
        knowledge_node_id=_LAYER_KNOWLEDGE_NODE_IDS[cognitive_layer],
        source_registry_ids=_SOURCE_REGISTRY_IDS,
        audited_registry_ids=audit.registry_ids,
        contract_serialization_version=contract.serialization_version,
        identity_serialization_version=identity.serialization_version,
        role_serialization_version=role.serialization_version,
        metadata_serialization_version=metadata.serialization_version,
        produced_outputs=role.produced_outputs,
        future_orchestration_hooks=role.future_orchestration_hooks,
        agent_blocked_runtime_behaviors=contract.blocked_runtime_behaviors,
        dependency_contracts=(
            "agent entry follows audited passive source registries",
            f"knowledge node:{_LAYER_KNOWLEDGE_NODE_IDS[cognitive_layer]}",
        ),
        governance_contracts=(
            "agent entry does not authorize agent creation or invocation",
            "HITL required before any agent orchestration behavior",
        ),
        explanation_contracts=(
            "agent entry cites identity, contract, role, and metadata sources",
            "knowledge graph explains cognitive-layer placement",
        ),
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    )
