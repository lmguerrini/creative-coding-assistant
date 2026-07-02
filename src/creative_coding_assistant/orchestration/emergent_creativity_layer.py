"""V6.6 Emergent Creativity Layer metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
    CognitiveOSCapability,
    CognitiveOSLayer,
    CognitiveOSPosture,
)
from creative_coding_assistant.orchestration.creative_identity_layer import (
    CreativeIdentityLayerPlan,
    build_creative_identity_layer,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)

EMERGENT_CREATIVITY_LAYER_SERIALIZATION_VERSION = "emergent_creativity_layer.v1"
EMERGENT_CREATIVITY_LAYER_ROADMAP_ITEM = "Emergent Creativity Layer"
EMERGENT_CREATIVITY_LAYER_AUTHORITY_BOUNDARY = (
    "V6.6 Emergent Creativity Layer projects creative identity profiles into "
    "read-only emergence signals for novelty readiness, cross-layer synthesis "
    "posture, identity continuity, governed exploration boundaries, and HITL "
    "review posture. It exposes emergent creativity metadata only; it does "
    "not generate output, execute exploration, mutate identities, prompts, "
    "memory, retrieval, storage, provider selection, generated output, "
    "runtime state, or apply Runtime Evolution."
)
EMERGENT_CREATIVITY_DIMENSIONS = (
    "novel combination readiness",
    "cross-layer synthesis posture",
    "identity continuity",
    "governed exploration boundary",
    "HITL emergence review",
)


class EmergentCreativitySignal(BaseModel):
    """One read-only emergent creativity signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    emergence_id: str = Field(min_length=1, max_length=190)
    identity_id: str = Field(min_length=1, max_length=190)
    cognition_id: str = Field(min_length=1, max_length=190)
    governance_id: str = Field(min_length=1, max_length=190)
    planning_id: str = Field(min_length=1, max_length=170)
    reasoning_id: str = Field(min_length=1, max_length=170)
    profile_id: str = Field(min_length=1, max_length=170)
    state_id: str = Field(min_length=1, max_length=160)
    capability_id: str = Field(min_length=1, max_length=80)
    capability_name: CognitiveOSCapability
    cognitive_layer: CognitiveOSLayer
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    emergence_dimensions: tuple[str, ...] = Field(min_length=5, max_length=5)
    identity_posture: CognitiveOSPosture
    emergence_posture: CognitiveOSPosture
    emergence_execution_authorized: Literal[False] = False
    emergence_summary: str = Field(min_length=1, max_length=560)
    dependency_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_sources_and_boundary(self) -> Self:
        expected_emergence_id = f"emergent_creativity::{self.capability_id}"
        if self.emergence_id != expected_emergence_id:
            raise ValueError("emergence_id must match capability_id")
        expected_identity_id = f"creative_identity::{self.capability_id}"
        if self.identity_id != expected_identity_id:
            raise ValueError("identity_id must match capability_id")
        expected_cognition_id = f"creative_cognition::{self.capability_id}"
        if self.cognition_id != expected_cognition_id:
            raise ValueError("cognition_id must match capability_id")
        expected_governance_id = f"cognitive_governance::{self.capability_id}"
        if self.governance_id != expected_governance_id:
            raise ValueError("governance_id must match capability_id")
        expected_planning_id = f"meta_planning::{self.capability_id}"
        if self.planning_id != expected_planning_id:
            raise ValueError("planning_id must match capability_id")
        expected_reasoning_id = f"meta_reasoning::{self.capability_id}"
        if self.reasoning_id != expected_reasoning_id:
            raise ValueError("reasoning_id must match capability_id")
        expected_profile_id = f"cognitive_profile::{self.capability_id}"
        if self.profile_id != expected_profile_id:
            raise ValueError("profile_id must match capability_id")
        expected_state_id = f"cognitive_state::{self.capability_id}"
        if self.state_id != expected_state_id:
            raise ValueError("state_id must match capability_id")
        if self.emergence_dimensions != EMERGENT_CREATIVITY_DIMENSIONS:
            raise ValueError("emergence_dimensions must match V6.6 emergence")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class EmergentCreativityLayerPlan(BaseModel):
    """Read-only emergent creativity layer over creative identity."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["emergent_creativity_layer"] = "emergent_creativity_layer"
    serialization_version: Literal["emergent_creativity_layer.v1"] = (
        EMERGENT_CREATIVITY_LAYER_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=EMERGENT_CREATIVITY_LAYER_AUTHORITY_BOUNDARY,
        max_length=2000,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    creative_identity_layer_role: Literal["creative_identity_layer"]
    creative_identity_layer_serialization_version: Literal[
        "creative_identity_layer.v1"
    ]
    creative_cognition_layer_role: Literal["creative_cognition_layer"]
    cognitive_governance_layer_role: Literal["cognitive_governance_layer"]
    layer_order: tuple[CognitiveOSLayer, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_count: int = Field(ge=6, le=6)
    source_identity_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_identity_count: int = Field(ge=6, le=6)
    source_cognition_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_cognition_count: int = Field(ge=6, le=6)
    source_governance_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_governance_count: int = Field(ge=6, le=6)
    source_planning_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_planning_count: int = Field(ge=6, le=6)
    source_reasoning_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_reasoning_count: int = Field(ge=6, le=6)
    source_profile_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_profile_count: int = Field(ge=6, le=6)
    source_state_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_state_count: int = Field(ge=6, le=6)
    emergent_creativity_signals: tuple[EmergentCreativitySignal, ...] = Field(
        min_length=6,
        max_length=6,
    )
    emergence_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    emergence_count: int = Field(ge=6, le=6)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    emergent_creativity_layer_implemented: Literal[True] = True
    creative_identity_layer_integrated: Literal[True] = True
    emergence_signal_contract_implemented: Literal[True] = True
    emergence_dependency_traceability_implemented: Literal[True] = True
    emergence_governance_contract_implemented: Literal[True] = True
    emergence_explainability_contract_implemented: Literal[True] = True
    emergent_generation_implemented: Literal[False] = False
    autonomous_exploration_implemented: Literal[False] = False
    identity_mutation_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    generated_emergent_output_ids: tuple[str, ...] = Field(default_factory=tuple)
    executed_emergence_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_emergence_policy_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _emergent_creativity_layer_matches_sources(self) -> Self:
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_count != len(self.capability_ids):
            raise ValueError("capability_count must match capability ids")
        if self.source_identity_count != len(self.source_identity_ids):
            raise ValueError("source_identity_count must match identity ids")
        if self.source_cognition_count != len(self.source_cognition_ids):
            raise ValueError("source_cognition_count must match cognition ids")
        if self.source_governance_count != len(self.source_governance_ids):
            raise ValueError("source_governance_count must match governance ids")
        if self.source_planning_count != len(self.source_planning_ids):
            raise ValueError("source_planning_count must match planning ids")
        if self.source_reasoning_count != len(self.source_reasoning_ids):
            raise ValueError("source_reasoning_count must match reasoning ids")
        if self.source_profile_count != len(self.source_profile_ids):
            raise ValueError("source_profile_count must match profile ids")
        if self.source_state_count != len(self.source_state_ids):
            raise ValueError("source_state_count must match state ids")
        if self.emergence_ids != tuple(
            signal.emergence_id for signal in self.emergent_creativity_signals
        ):
            raise ValueError("emergence_ids must match signals")
        if self.emergence_count != len(self.emergent_creativity_signals):
            raise ValueError("emergence_count must match signals")
        if len(set(self.emergence_ids)) != len(self.emergence_ids):
            raise ValueError("emergence_ids must be unique")
        declared_capabilities = set(self.capability_ids)
        declared_identities = set(self.source_identity_ids)
        declared_cognition = set(self.source_cognition_ids)
        declared_governance = set(self.source_governance_ids)
        declared_planning = set(self.source_planning_ids)
        declared_reasoning = set(self.source_reasoning_ids)
        declared_profiles = set(self.source_profile_ids)
        declared_states = set(self.source_state_ids)
        declared_agents = set(self.linked_agent_ids)
        for signal in self.emergent_creativity_signals:
            if signal.capability_id not in declared_capabilities:
                raise ValueError("signal capability_id must be declared")
            if signal.identity_id not in declared_identities:
                raise ValueError("signal identity_id must be declared")
            if signal.cognition_id not in declared_cognition:
                raise ValueError("signal cognition_id must be declared")
            if signal.governance_id not in declared_governance:
                raise ValueError("signal governance_id must be declared")
            if signal.planning_id not in declared_planning:
                raise ValueError("signal planning_id must be declared")
            if signal.reasoning_id not in declared_reasoning:
                raise ValueError("signal reasoning_id must be declared")
            if signal.profile_id not in declared_profiles:
                raise ValueError("signal profile_id must be declared")
            if signal.state_id not in declared_states:
                raise ValueError("signal state_id must be declared")
            if not set(signal.linked_agent_ids).issubset(declared_agents):
                raise ValueError("signal linked_agent_ids must be declared")
        if self.covered_roadmap_items != (EMERGENT_CREATIVITY_LAYER_ROADMAP_ITEM,):
            raise ValueError("covered_roadmap_items must be Task 16 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.generated_emergent_output_ids,
                self.executed_emergence_ids,
                self.mutated_emergence_policy_ids,
                self.emitted_hitl_request_ids,
            )
        ):
            raise ValueError(
                "emergent generation, execution, mutation, and HITL ids "
                "must be empty",
            )
        if not all(
            signal.advisory_only for signal in self.emergent_creativity_signals
        ):
            raise ValueError("all emergent creativity signals must be advisory only")
        return self


def build_emergent_creativity_layer(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    creative_identity_layer: CreativeIdentityLayerPlan | None = None,
) -> EmergentCreativityLayerPlan:
    """Build read-only emergent creativity metadata."""

    identity = creative_identity_layer or build_creative_identity_layer(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    signals = _emergent_creativity_signals(identity)
    return EmergentCreativityLayerPlan(
        route_name=identity.route_name,
        task_type=identity.task_type,
        execution_mode_ids=identity.execution_mode_ids,
        creative_identity_layer_role=identity.role,
        creative_identity_layer_serialization_version=identity.serialization_version,
        creative_cognition_layer_role=identity.creative_cognition_layer_role,
        cognitive_governance_layer_role=identity.cognitive_governance_layer_role,
        layer_order=identity.layer_order,
        capabilities=identity.capabilities,
        capability_ids=identity.capability_ids,
        capability_count=identity.capability_count,
        source_identity_ids=identity.identity_ids,
        source_identity_count=identity.identity_count,
        source_cognition_ids=identity.source_cognition_ids,
        source_cognition_count=identity.source_cognition_count,
        source_governance_ids=identity.source_governance_ids,
        source_governance_count=identity.source_governance_count,
        source_planning_ids=identity.source_planning_ids,
        source_planning_count=identity.source_planning_count,
        source_reasoning_ids=identity.source_reasoning_ids,
        source_reasoning_count=identity.source_reasoning_count,
        source_profile_ids=identity.source_profile_ids,
        source_profile_count=identity.source_profile_count,
        source_state_ids=identity.source_state_ids,
        source_state_count=identity.source_state_count,
        emergent_creativity_signals=signals,
        emergence_ids=tuple(signal.emergence_id for signal in signals),
        emergence_count=len(signals),
        linked_agent_ids=identity.linked_agent_ids,
        covered_roadmap_items=(EMERGENT_CREATIVITY_LAYER_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture=identity.graph_posture,
    )


def emergent_creativity_signal_by_id(
    emergence_id: str,
    layer: EmergentCreativityLayerPlan | None = None,
) -> EmergentCreativitySignal | None:
    """Return one emergent creativity signal without applying it."""

    source_layer = layer or build_emergent_creativity_layer()
    for signal in source_layer.emergent_creativity_signals:
        if signal.emergence_id == emergence_id:
            return signal
    return None


def emergent_creativity_signals_for_layer(
    cognitive_layer: CognitiveOSLayer,
    layer: EmergentCreativityLayerPlan | None = None,
) -> tuple[EmergentCreativitySignal, ...]:
    """Return emergent creativity signals for one Cognitive OS layer."""

    source_layer = layer or build_emergent_creativity_layer()
    return tuple(
        signal
        for signal in source_layer.emergent_creativity_signals
        if signal.cognitive_layer == cognitive_layer
    )


def emergent_creativity_signals_for_agent(
    agent_id: str,
    layer: EmergentCreativityLayerPlan | None = None,
) -> tuple[EmergentCreativitySignal, ...]:
    """Return emergent creativity signals linked to one agent."""

    source_layer = layer or build_emergent_creativity_layer()
    return tuple(
        signal
        for signal in source_layer.emergent_creativity_signals
        if agent_id in signal.linked_agent_ids
    )


def _emergent_creativity_signals(
    identity_layer: CreativeIdentityLayerPlan,
) -> tuple[EmergentCreativitySignal, ...]:
    return tuple(
        EmergentCreativitySignal(
            emergence_id=f"emergent_creativity::{profile.capability_id}",
            identity_id=profile.identity_id,
            cognition_id=profile.cognition_id,
            governance_id=profile.governance_id,
            planning_id=profile.planning_id,
            reasoning_id=profile.reasoning_id,
            profile_id=profile.profile_id,
            state_id=profile.state_id,
            capability_id=profile.capability_id,
            capability_name=profile.capability_name,
            cognitive_layer=profile.cognitive_layer,
            linked_agent_ids=profile.linked_agent_ids,
            emergence_dimensions=EMERGENT_CREATIVITY_DIMENSIONS,
            identity_posture=profile.identity_posture,
            emergence_posture=profile.identity_posture,
            emergence_summary=(
                f"Read-only emergent creativity signal for "
                f"{profile.capability_name}; evaluates novelty readiness, "
                "cross-layer synthesis, identity continuity, governance "
                "boundary, and HITL review without generation authority."
            ),
            dependency_contracts=(
                "emergent creativity signal follows creative identity profile",
                f"creative identity profile:{profile.identity_id}",
                f"creative cognition signal:{profile.cognition_id}",
            ),
            governance_contracts=(
                "emergent creativity does not generate output",
                "emergent creativity does not execute exploration or mutate identity",
                "HITL required before any emergent creativity behavior",
            ),
            explanation_contracts=(
                "emergent creativity cites identity and cognition sources",
                "emergent creativity preserves capability and layer ownership",
                "emergent creativity explains why no emergence is applied",
            ),
            blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        for profile in identity_layer.identity_profiles
    )
