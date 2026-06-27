"""Passive V4.2 orchestration contract integration manifest."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_capability_alignment import (
    agent_capability_alignment_registry,
)
from creative_coding_assistant.orchestration.agent_consensus import (
    consensus_builder_registry,
)
from creative_coding_assistant.orchestration.agent_contracts import (
    AGENT_CONTRACT_REGISTRY,
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
from creative_coding_assistant.orchestration.agent_lifecycle import (
    agent_lifecycle_registry,
)
from creative_coding_assistant.orchestration.agent_parallel_scheduling import (
    parallel_scheduling_registry,
)
from creative_coding_assistant.orchestration.agent_routing import (
    agent_routing_registry,
)
from creative_coding_assistant.orchestration.agent_state_synchronization import (
    agent_state_synchronization_registry,
)
from creative_coding_assistant.orchestration.blackboard_memory import (
    blackboard_memory_registry,
)
from creative_coding_assistant.orchestration.shared_context_views import (
    shared_context_view_registry,
)
from creative_coding_assistant.orchestration.workflow_agent_handoff import (
    workflow_agent_handoff_registry,
)

RegistryIntegrationKind = Literal[
    "per_agent",
    "relationship",
    "advisory",
    "workflow_bridge",
]

ORCHESTRATION_INTEGRATED_REGISTRY_SERIALIZATION_VERSION = (
    "orchestration_integrated_registry.v1"
)
ORCHESTRATION_CONTRACT_INTEGRATION_SERIALIZATION_VERSION = (
    "orchestration_contract_integration.v1"
)
ORCHESTRATION_CONTRACT_INTEGRATION_AUTHORITY_BOUNDARY = (
    "Orchestration contract integration metadata makes V4.2 passive "
    "orchestration registries discoverable against V4.1 agent contracts only; "
    "it does not execute orchestration, mutate runtime state, route providers "
    "or models, invoke agents, alter prompts, control workflows, write memory, "
    "or modify generated output."
)

_SOURCE_V4_1_REGISTRIES = (
    "agent_contract_registry",
    "agent_role_registry",
    "agent_boundary_registry",
    "agent_metadata_registry",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "orchestration_execution",
    "runtime_mutation",
    "provider_or_model_routing",
    "agent_invocation",
    "prompt_alteration",
    "workflow_control",
    "memory_write",
    "generated_output_modification",
)
_REGISTRY_SPECS: tuple[
    tuple[
        str,
        str,
        RegistryIntegrationKind,
        Callable[[], Any],
    ],
    ...,
] = (
    (
        "agent_routing_registry",
        "agent_routing_registry",
        "per_agent",
        agent_routing_registry,
    ),
    (
        "blackboard_memory_registry",
        "blackboard_memory_registry",
        "per_agent",
        blackboard_memory_registry,
    ),
    (
        "shared_context_view_registry",
        "shared_context_view_registry",
        "per_agent",
        shared_context_view_registry,
    ),
    (
        "agent_dependency_graph_registry",
        "agent_dependency_graph_registry",
        "relationship",
        agent_dependency_graph_registry,
    ),
    (
        "parallel_scheduling_registry",
        "parallel_scheduling_registry",
        "relationship",
        parallel_scheduling_registry,
    ),
    (
        "agent_coordination_registry",
        "agent_coordination_registry",
        "relationship",
        agent_coordination_registry,
    ),
    (
        "agent_debate_registry",
        "agent_debate_registry",
        "advisory",
        agent_debate_registry,
    ),
    (
        "consensus_builder_registry",
        "consensus_builder_registry",
        "advisory",
        consensus_builder_registry,
    ),
    (
        "agent_capability_alignment_registry",
        "agent_capability_alignment_registry",
        "per_agent",
        agent_capability_alignment_registry,
    ),
    (
        "agent_escalation_signal_registry",
        "agent_escalation_signal_registry",
        "advisory",
        agent_escalation_signal_registry,
    ),
    (
        "agent_lifecycle_registry",
        "agent_lifecycle_registry",
        "per_agent",
        agent_lifecycle_registry,
    ),
    (
        "agent_state_synchronization_registry",
        "agent_state_synchronization_registry",
        "per_agent",
        agent_state_synchronization_registry,
    ),
    (
        "workflow_agent_handoff_registry",
        "workflow_agent_handoff_registry",
        "workflow_bridge",
        workflow_agent_handoff_registry,
    ),
)


class IntegratedOrchestrationRegistryContract(BaseModel):
    """One discoverable V4.2 registry entry integrated with V4.1 contracts."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    registry_id: str = Field(min_length=1, max_length=140)
    registry_role: str = Field(min_length=1, max_length=140)
    registry_kind: RegistryIntegrationKind
    export_symbol: str = Field(min_length=1, max_length=140)
    registry_serialization_version: str = Field(min_length=1, max_length=120)
    linked_agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    source_v4_1_registries: tuple[str, ...] = Field(min_length=4, max_length=4)
    integration_boundary: str = Field(min_length=1, max_length=800)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    metadata_only_declared: Literal[True] = True
    active_orchestration_path_implemented: Literal[False] = False
    runtime_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    serialization_version: Literal["orchestration_integrated_registry.v1"] = (
        ORCHESTRATION_INTEGRATED_REGISTRY_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class OrchestrationContractIntegrationRegistry(BaseModel):
    """Stable passive manifest of integrated V4.2 orchestration contracts."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["orchestration_contract_integration_registry"] = (
        "orchestration_contract_integration_registry"
    )
    serialization_version: Literal["orchestration_contract_integration.v1"] = (
        ORCHESTRATION_CONTRACT_INTEGRATION_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ORCHESTRATION_CONTRACT_INTEGRATION_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    integrated_registries: tuple[IntegratedOrchestrationRegistryContract, ...] = Field(
        min_length=13,
        max_length=13,
    )
    registry_ids: tuple[str, ...] = Field(min_length=13, max_length=13)
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    source_v4_1_registries: tuple[str, ...] = Field(min_length=4, max_length=4)
    registry_count: int = Field(ge=13, le=13)
    contract_count: int = Field(ge=12, le=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    active_orchestration_path_implemented: Literal[False] = False
    runtime_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_integrated_contracts(self) -> Self:
        derived_registry_ids = tuple(
            entry.registry_id for entry in self.integrated_registries
        )
        if self.registry_ids != derived_registry_ids:
            raise ValueError("registry_ids must match integrated registries")
        if len(set(self.registry_ids)) != len(self.registry_ids):
            raise ValueError("registry_ids must be unique")
        if self.registry_count != len(self.integrated_registries):
            raise ValueError("registry_count must match integrated registries")
        if self.contract_count != len(self.agent_ids):
            raise ValueError("contract_count must match agent_ids")

        known_agents = set(self.agent_ids)
        expected_sources = set(self.source_v4_1_registries)
        for entry in self.integrated_registries:
            if not set(entry.linked_agent_ids).issubset(known_agents):
                raise ValueError("linked_agent_ids must be known V4.1 agents")
            if set(entry.source_v4_1_registries) != expected_sources:
                raise ValueError("source_v4_1_registries must match registry sources")
            if not entry.metadata_only_declared:
                raise ValueError("integrated registries must declare metadata only")
        return self


def orchestration_contract_integration_registry() -> (
    OrchestrationContractIntegrationRegistry
):
    """Return passive V4.2 orchestration contract integration metadata."""

    return ORCHESTRATION_CONTRACT_INTEGRATION_REGISTRY


def integrated_orchestration_registry_by_id(
    registry_id: str,
    registry: OrchestrationContractIntegrationRegistry | None = None,
) -> IntegratedOrchestrationRegistryContract | None:
    """Return one integrated registry entry without executing orchestration."""

    source_registry = registry or ORCHESTRATION_CONTRACT_INTEGRATION_REGISTRY
    for entry in source_registry.integrated_registries:
        if entry.registry_id == registry_id:
            return entry
    return None


def _linked_agent_ids(registry: Any) -> tuple[str, ...]:
    agent_ids = getattr(registry, "agent_ids", None)
    if agent_ids is not None:
        return tuple(agent_ids)
    return AGENT_CONTRACT_REGISTRY.agent_ids


def _integrated_registry_contract(
    registry_id: str,
    export_symbol: str,
    registry_kind: RegistryIntegrationKind,
    registry_builder: Callable[[], Any],
) -> IntegratedOrchestrationRegistryContract:
    registry = registry_builder()
    return IntegratedOrchestrationRegistryContract(
        registry_id=registry_id,
        registry_role=registry.role,
        registry_kind=registry_kind,
        export_symbol=export_symbol,
        registry_serialization_version=registry.serialization_version,
        linked_agent_ids=_linked_agent_ids(registry),
        source_v4_1_registries=_SOURCE_V4_1_REGISTRIES,
        metadata_only_declared=registry.metadata_only,
        integration_boundary=(
            "Integrated orchestration registry contracts are discoverability "
            "metadata only; they do not execute orchestration, mutate runtime "
            "state, invoke agents, or route providers and models."
        ),
    )


INTEGRATED_ORCHESTRATION_REGISTRIES = tuple(
    _integrated_registry_contract(*spec) for spec in _REGISTRY_SPECS
)
ORCHESTRATION_CONTRACT_INTEGRATION_REGISTRY = (
    OrchestrationContractIntegrationRegistry(
        integrated_registries=INTEGRATED_ORCHESTRATION_REGISTRIES,
        registry_ids=tuple(
            entry.registry_id for entry in INTEGRATED_ORCHESTRATION_REGISTRIES
        ),
        agent_ids=AGENT_CONTRACT_REGISTRY.agent_ids,
        source_v4_1_registries=_SOURCE_V4_1_REGISTRIES,
        registry_count=len(INTEGRATED_ORCHESTRATION_REGISTRIES),
        contract_count=AGENT_CONTRACT_REGISTRY.contract_count,
    )
)
