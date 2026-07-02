"""V6.6 Cognitive Profile Engine metadata."""

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
from creative_coding_assistant.orchestration.cognitive_state_engine import (
    CognitiveStateEnginePlan,
    build_cognitive_state_engine,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)

COGNITIVE_PROFILE_ENGINE_SERIALIZATION_VERSION = "cognitive_profile_engine.v1"
COGNITIVE_PROFILE_ENGINE_ROADMAP_ITEM = "Cognitive Profile Engine"
COGNITIVE_PROFILE_ENGINE_AUTHORITY_BOUNDARY = (
    "V6.6 Cognitive Profile Engine classifies read-only Cognitive OS state "
    "snapshots into capability, layer, dependency, governance, and "
    "explainability profiles. It exposes profile metadata for inspection and "
    "future capability-scoped analysis only; it does not persist profiles, "
    "mutate profiles, personalize runtime behavior, route agents, route "
    "providers or models, execute providers, control workflows, mutate "
    "generated output, emit HITL requests, apply HITL decisions, or apply "
    "Runtime Evolution."
)
COGNITIVE_PROFILE_DIMENSIONS = (
    "capability ownership",
    "state dependency trace",
    "agent linkage",
    "governance posture",
    "explanation posture",
)


class CognitiveProfile(BaseModel):
    """One read-only profile for a Cognitive OS state snapshot."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    profile_id: str = Field(min_length=1, max_length=170)
    state_id: str = Field(min_length=1, max_length=160)
    capability_id: str = Field(min_length=1, max_length=80)
    capability_name: CognitiveOSCapability
    cognitive_layer: CognitiveOSLayer
    source_optimization_signal_id: str = Field(min_length=1, max_length=180)
    source_learning_signal_id: str = Field(min_length=1, max_length=180)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    profile_dimensions: tuple[str, ...] = Field(min_length=5, max_length=5)
    optimization_posture: CognitiveOSPosture
    state_posture: CognitiveOSPosture
    profile_posture: CognitiveOSPosture
    profile_summary: str = Field(min_length=1, max_length=420)
    dependency_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _profile_matches_sources_and_boundary(self) -> Self:
        expected_profile_id = f"cognitive_profile::{self.capability_id}"
        if self.profile_id != expected_profile_id:
            raise ValueError("profile_id must match capability_id")
        expected_state_id = f"cognitive_state::{self.capability_id}"
        if self.state_id != expected_state_id:
            raise ValueError("state_id must match capability_id")
        expected_optimization_id = f"cross_system_optimization::{self.capability_id}"
        if self.source_optimization_signal_id != expected_optimization_id:
            raise ValueError("source_optimization_signal_id must match capability")
        expected_learning_id = f"cross_system_learning::{self.capability_id}"
        if self.source_learning_signal_id != expected_learning_id:
            raise ValueError("source_learning_signal_id must match capability")
        if self.profile_dimensions != COGNITIVE_PROFILE_DIMENSIONS:
            raise ValueError("profile_dimensions must match V6.6 profile contract")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class CognitiveProfileEnginePlan(BaseModel):
    """Read-only Cognitive OS profile layer."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cognitive_profile_engine"] = "cognitive_profile_engine"
    serialization_version: Literal["cognitive_profile_engine.v1"] = (
        COGNITIVE_PROFILE_ENGINE_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=COGNITIVE_PROFILE_ENGINE_AUTHORITY_BOUNDARY,
        max_length=1900,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    state_engine_role: Literal["cognitive_state_engine"]
    state_engine_serialization_version: Literal["cognitive_state_engine.v1"]
    optimization_layer_role: Literal["cross_system_optimization_layer"]
    learning_layer_role: Literal["cross_system_learning_layer"]
    layer_order: tuple[CognitiveOSLayer, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_count: int = Field(ge=6, le=6)
    source_state_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_state_count: int = Field(ge=6, le=6)
    source_optimization_signal_ids: tuple[str, ...] = Field(
        min_length=6,
        max_length=6,
    )
    source_optimization_signal_count: int = Field(ge=6, le=6)
    source_learning_signal_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_learning_signal_count: int = Field(ge=6, le=6)
    cognitive_profiles: tuple[CognitiveProfile, ...] = Field(
        min_length=6,
        max_length=6,
    )
    profile_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    profile_count: int = Field(ge=6, le=6)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    cognitive_profile_engine_implemented: Literal[True] = True
    cognitive_state_engine_integrated: Literal[True] = True
    profile_contract_implemented: Literal[True] = True
    profile_dependency_traceability_implemented: Literal[True] = True
    profile_governance_contract_implemented: Literal[True] = True
    profile_explainability_contract_implemented: Literal[True] = True
    profile_persistence_implemented: Literal[False] = False
    profile_mutation_implemented: Literal[False] = False
    personalized_runtime_behavior_implemented: Literal[False] = False
    profile_driven_agent_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    persisted_profile_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_profile_ids: tuple[str, ...] = Field(default_factory=tuple)
    routed_profile_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _profile_engine_matches_sources(self) -> Self:
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_count != len(self.capability_ids):
            raise ValueError("capability_count must match capability ids")
        if self.source_state_count != len(self.source_state_ids):
            raise ValueError("source_state_count must match state ids")
        if self.source_optimization_signal_count != len(
            self.source_optimization_signal_ids
        ):
            raise ValueError("source_optimization_signal_count must match signals")
        if self.source_learning_signal_count != len(self.source_learning_signal_ids):
            raise ValueError("source_learning_signal_count must match signals")
        if self.profile_ids != tuple(
            profile.profile_id for profile in self.cognitive_profiles
        ):
            raise ValueError("profile_ids must match cognitive profiles")
        if self.profile_count != len(self.cognitive_profiles):
            raise ValueError("profile_count must match cognitive profiles")
        if len(set(self.profile_ids)) != len(self.profile_ids):
            raise ValueError("profile_ids must be unique")
        declared_capabilities = set(self.capability_ids)
        declared_states = set(self.source_state_ids)
        declared_optimization_signals = set(self.source_optimization_signal_ids)
        declared_learning_signals = set(self.source_learning_signal_ids)
        declared_agents = set(self.linked_agent_ids)
        for profile in self.cognitive_profiles:
            if profile.capability_id not in declared_capabilities:
                raise ValueError("profile capability_id must be declared")
            if profile.state_id not in declared_states:
                raise ValueError("profile state_id must be declared")
            if (
                profile.source_optimization_signal_id
                not in declared_optimization_signals
            ):
                raise ValueError("profile optimization signal must be declared")
            if profile.source_learning_signal_id not in declared_learning_signals:
                raise ValueError("profile learning signal must be declared")
            if not set(profile.linked_agent_ids).issubset(declared_agents):
                raise ValueError("profile linked_agent_ids must be declared")
        if self.covered_roadmap_items != (COGNITIVE_PROFILE_ENGINE_ROADMAP_ITEM,):
            raise ValueError("covered_roadmap_items must be Task 10 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.persisted_profile_ids,
                self.mutated_profile_ids,
                self.routed_profile_ids,
                self.emitted_hitl_request_ids,
            )
        ):
            raise ValueError(
                "profile persistence, mutation, routing, and HITL ids must be empty",
            )
        if not all(profile.advisory_only for profile in self.cognitive_profiles):
            raise ValueError("all cognitive profiles must be advisory only")
        return self


def build_cognitive_profile_engine(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    state_engine: CognitiveStateEnginePlan | None = None,
) -> CognitiveProfileEnginePlan:
    """Build read-only Cognitive OS profile metadata."""

    state = state_engine or build_cognitive_state_engine(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    profiles = _cognitive_profiles(state)
    return CognitiveProfileEnginePlan(
        route_name=state.route_name,
        task_type=state.task_type,
        execution_mode_ids=state.execution_mode_ids,
        state_engine_role=state.role,
        state_engine_serialization_version=state.serialization_version,
        optimization_layer_role=state.optimization_layer_role,
        learning_layer_role=state.learning_layer_role,
        layer_order=state.layer_order,
        capabilities=state.capabilities,
        capability_ids=state.capability_ids,
        capability_count=state.capability_count,
        source_state_ids=state.state_ids,
        source_state_count=state.state_count,
        source_optimization_signal_ids=state.source_optimization_signal_ids,
        source_optimization_signal_count=state.source_optimization_signal_count,
        source_learning_signal_ids=state.source_learning_signal_ids,
        source_learning_signal_count=state.source_learning_signal_count,
        cognitive_profiles=profiles,
        profile_ids=tuple(profile.profile_id for profile in profiles),
        profile_count=len(profiles),
        linked_agent_ids=state.linked_agent_ids,
        covered_roadmap_items=(COGNITIVE_PROFILE_ENGINE_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture=state.graph_posture,
    )


def cognitive_profile_by_id(
    profile_id: str,
    engine: CognitiveProfileEnginePlan | None = None,
) -> CognitiveProfile | None:
    """Return one cognitive profile without persisting or applying it."""

    source_engine = engine or build_cognitive_profile_engine()
    for profile in source_engine.cognitive_profiles:
        if profile.profile_id == profile_id:
            return profile
    return None


def cognitive_profiles_for_layer(
    cognitive_layer: CognitiveOSLayer,
    engine: CognitiveProfileEnginePlan | None = None,
) -> tuple[CognitiveProfile, ...]:
    """Return read-only cognitive profiles for one layer."""

    source_engine = engine or build_cognitive_profile_engine()
    return tuple(
        profile
        for profile in source_engine.cognitive_profiles
        if profile.cognitive_layer == cognitive_layer
    )


def cognitive_profiles_for_agent(
    agent_id: str,
    engine: CognitiveProfileEnginePlan | None = None,
) -> tuple[CognitiveProfile, ...]:
    """Return read-only cognitive profiles linked to one agent."""

    source_engine = engine or build_cognitive_profile_engine()
    return tuple(
        profile
        for profile in source_engine.cognitive_profiles
        if agent_id in profile.linked_agent_ids
    )


def _cognitive_profiles(
    state_engine: CognitiveStateEnginePlan,
) -> tuple[CognitiveProfile, ...]:
    return tuple(
        CognitiveProfile(
            profile_id=f"cognitive_profile::{snapshot.capability_id}",
            state_id=snapshot.state_id,
            capability_id=snapshot.capability_id,
            capability_name=snapshot.capability_name,
            cognitive_layer=snapshot.cognitive_layer,
            source_optimization_signal_id=snapshot.source_optimization_signal_id,
            source_learning_signal_id=snapshot.source_learning_signal_id,
            linked_agent_ids=snapshot.linked_agent_ids,
            profile_dimensions=COGNITIVE_PROFILE_DIMENSIONS,
            optimization_posture=snapshot.optimization_posture,
            state_posture=snapshot.state_posture,
            profile_posture=snapshot.state_posture,
            profile_summary=(
                f"Read-only cognitive profile for {snapshot.capability_name}; "
                f"classifies {snapshot.state_id} without persistence, "
                "personalized routing, workflow control, or execution authority."
            ),
            dependency_contracts=(
                "profile follows cognitive state snapshot",
                f"state snapshot:{snapshot.state_id}",
                f"optimization signal:{snapshot.source_optimization_signal_id}",
            ),
            governance_contracts=(
                "profile does not authorize profile persistence",
                "profile does not authorize personalized runtime behavior",
                "HITL required before any profile-driven behavioral application",
            ),
            explanation_contracts=(
                "profile cites cognitive state source",
                "profile preserves capability and layer ownership",
                "profile explains why no runtime behavior is applied",
            ),
            blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        for snapshot in state_engine.state_snapshots
    )
