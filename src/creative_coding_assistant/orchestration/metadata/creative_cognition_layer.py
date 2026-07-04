"""V6.6 Creative Cognition Layer metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.cognitive_governance_layer import (
    CognitiveGovernanceLayerPlan,
    build_cognitive_governance_layer,
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
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)

CREATIVE_COGNITION_LAYER_SERIALIZATION_VERSION = "creative_cognition_layer.v1"
CREATIVE_COGNITION_LAYER_ROADMAP_ITEM = "Creative Cognition Layer"
CREATIVE_COGNITION_LAYER_AUTHORITY_BOUNDARY = (
    "V6.6 Creative Cognition Layer projects cognitive governance policies "
    "into read-only creative cognition signals for divergent framing, "
    "constraint awareness, composition awareness, evaluation awareness, and "
    "governed exploration posture. It exposes creative cognition metadata "
    "only; it does not generate creative output, execute exploration, mutate "
    "prompts, memory, retrieval, storage, provider selection, generated "
    "output, runtime state, or apply Runtime Evolution."
)
CREATIVE_COGNITION_MODES = (
    "divergent framing",
    "constraint sensitivity",
    "composition awareness",
    "evaluation awareness",
    "governed exploration",
)


class CreativeCognitionSignal(BaseModel):
    """One read-only creative cognition signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    cognition_id: str = Field(min_length=1, max_length=190)
    governance_id: str = Field(min_length=1, max_length=190)
    planning_id: str = Field(min_length=1, max_length=170)
    reasoning_id: str = Field(min_length=1, max_length=170)
    profile_id: str = Field(min_length=1, max_length=170)
    state_id: str = Field(min_length=1, max_length=160)
    capability_id: str = Field(min_length=1, max_length=80)
    capability_name: CognitiveOSCapability
    cognitive_layer: CognitiveOSLayer
    source_optimization_signal_id: str = Field(min_length=1, max_length=180)
    source_learning_signal_id: str = Field(min_length=1, max_length=180)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    creative_cognition_modes: tuple[str, ...] = Field(min_length=5, max_length=5)
    governance_posture: CognitiveOSPosture
    cognition_posture: CognitiveOSPosture
    exploration_authorized: Literal[False] = False
    creative_summary: str = Field(min_length=1, max_length=540)
    dependency_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_sources_and_boundary(self) -> Self:
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
        expected_optimization_id = f"cross_system_optimization::{self.capability_id}"
        if self.source_optimization_signal_id != expected_optimization_id:
            raise ValueError("source_optimization_signal_id must match capability")
        expected_learning_id = f"cross_system_learning::{self.capability_id}"
        if self.source_learning_signal_id != expected_learning_id:
            raise ValueError("source_learning_signal_id must match capability")
        if self.creative_cognition_modes != CREATIVE_COGNITION_MODES:
            raise ValueError("creative_cognition_modes must match V6.6 contract")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class CreativeCognitionLayerPlan(BaseModel):
    """Read-only creative cognition layer over cognitive governance."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_cognition_layer"] = "creative_cognition_layer"
    serialization_version: Literal["creative_cognition_layer.v1"] = (
        CREATIVE_COGNITION_LAYER_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CREATIVE_COGNITION_LAYER_AUTHORITY_BOUNDARY,
        max_length=2000,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    cognitive_governance_layer_role: Literal["cognitive_governance_layer"]
    cognitive_governance_layer_serialization_version: Literal[
        "cognitive_governance_layer.v1"
    ]
    meta_planning_layer_role: Literal["meta_planning_layer"]
    meta_reasoning_layer_role: Literal["meta_reasoning_layer"]
    layer_order: tuple[CognitiveOSLayer, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_count: int = Field(ge=6, le=6)
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
    creative_cognition_signals: tuple[CreativeCognitionSignal, ...] = Field(
        min_length=6,
        max_length=6,
    )
    cognition_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    cognition_count: int = Field(ge=6, le=6)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    creative_cognition_layer_implemented: Literal[True] = True
    cognitive_governance_layer_integrated: Literal[True] = True
    creative_cognition_contract_implemented: Literal[True] = True
    creative_dependency_traceability_implemented: Literal[True] = True
    creative_governance_contract_implemented: Literal[True] = True
    creative_explainability_contract_implemented: Literal[True] = True
    creative_generation_implemented: Literal[False] = False
    exploration_execution_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    generated_creative_output_ids: tuple[str, ...] = Field(default_factory=tuple)
    executed_exploration_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_creative_policy_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _creative_cognition_layer_matches_sources(self) -> Self:
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_count != len(self.capability_ids):
            raise ValueError("capability_count must match capability ids")
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
        if self.cognition_ids != tuple(
            signal.cognition_id for signal in self.creative_cognition_signals
        ):
            raise ValueError("cognition_ids must match signals")
        if self.cognition_count != len(self.creative_cognition_signals):
            raise ValueError("cognition_count must match signals")
        if len(set(self.cognition_ids)) != len(self.cognition_ids):
            raise ValueError("cognition_ids must be unique")
        declared_capabilities = set(self.capability_ids)
        declared_governance = set(self.source_governance_ids)
        declared_planning = set(self.source_planning_ids)
        declared_reasoning = set(self.source_reasoning_ids)
        declared_profiles = set(self.source_profile_ids)
        declared_states = set(self.source_state_ids)
        declared_agents = set(self.linked_agent_ids)
        for signal in self.creative_cognition_signals:
            if signal.capability_id not in declared_capabilities:
                raise ValueError("signal capability_id must be declared")
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
        if self.covered_roadmap_items != (CREATIVE_COGNITION_LAYER_ROADMAP_ITEM,):
            raise ValueError("covered_roadmap_items must be Task 14 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.generated_creative_output_ids,
                self.executed_exploration_ids,
                self.mutated_creative_policy_ids,
                self.emitted_hitl_request_ids,
            )
        ):
            raise ValueError(
                "creative generation, exploration, mutation, and HITL ids "
                "must be empty",
            )
        if not all(signal.advisory_only for signal in self.creative_cognition_signals):
            raise ValueError("all creative cognition signals must be advisory only")
        return self


def build_creative_cognition_layer(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    cognitive_governance_layer: CognitiveGovernanceLayerPlan | None = None,
) -> CreativeCognitionLayerPlan:
    """Build read-only creative cognition metadata."""

    governance = cognitive_governance_layer or build_cognitive_governance_layer(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    signals = _creative_cognition_signals(governance)
    return CreativeCognitionLayerPlan(
        route_name=governance.route_name,
        task_type=governance.task_type,
        execution_mode_ids=governance.execution_mode_ids,
        cognitive_governance_layer_role=governance.role,
        cognitive_governance_layer_serialization_version=(
            governance.serialization_version
        ),
        meta_planning_layer_role=governance.meta_planning_layer_role,
        meta_reasoning_layer_role=governance.meta_reasoning_layer_role,
        layer_order=governance.layer_order,
        capabilities=governance.capabilities,
        capability_ids=governance.capability_ids,
        capability_count=governance.capability_count,
        source_governance_ids=governance.governance_ids,
        source_governance_count=governance.governance_count,
        source_planning_ids=governance.source_planning_ids,
        source_planning_count=governance.source_planning_count,
        source_reasoning_ids=governance.source_reasoning_ids,
        source_reasoning_count=governance.source_reasoning_count,
        source_profile_ids=governance.source_profile_ids,
        source_profile_count=governance.source_profile_count,
        source_state_ids=governance.source_state_ids,
        source_state_count=governance.source_state_count,
        creative_cognition_signals=signals,
        cognition_ids=tuple(signal.cognition_id for signal in signals),
        cognition_count=len(signals),
        linked_agent_ids=governance.linked_agent_ids,
        covered_roadmap_items=(CREATIVE_COGNITION_LAYER_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture=governance.graph_posture,
    )


def creative_cognition_signal_by_id(
    cognition_id: str,
    layer: CreativeCognitionLayerPlan | None = None,
) -> CreativeCognitionSignal | None:
    """Return one creative cognition signal without applying it."""

    source_layer = layer or build_creative_cognition_layer()
    for signal in source_layer.creative_cognition_signals:
        if signal.cognition_id == cognition_id:
            return signal
    return None


def creative_cognition_signals_for_layer(
    cognitive_layer: CognitiveOSLayer,
    layer: CreativeCognitionLayerPlan | None = None,
) -> tuple[CreativeCognitionSignal, ...]:
    """Return creative cognition signals for one Cognitive OS layer."""

    source_layer = layer or build_creative_cognition_layer()
    return tuple(
        signal
        for signal in source_layer.creative_cognition_signals
        if signal.cognitive_layer == cognitive_layer
    )


def creative_cognition_signals_for_agent(
    agent_id: str,
    layer: CreativeCognitionLayerPlan | None = None,
) -> tuple[CreativeCognitionSignal, ...]:
    """Return creative cognition signals linked to one agent."""

    source_layer = layer or build_creative_cognition_layer()
    return tuple(
        signal
        for signal in source_layer.creative_cognition_signals
        if agent_id in signal.linked_agent_ids
    )


def _creative_cognition_signals(
    governance_layer: CognitiveGovernanceLayerPlan,
) -> tuple[CreativeCognitionSignal, ...]:
    return tuple(
        CreativeCognitionSignal(
            cognition_id=f"creative_cognition::{policy.capability_id}",
            governance_id=policy.governance_id,
            planning_id=policy.planning_id,
            reasoning_id=policy.reasoning_id,
            profile_id=policy.profile_id,
            state_id=policy.state_id,
            capability_id=policy.capability_id,
            capability_name=policy.capability_name,
            cognitive_layer=policy.cognitive_layer,
            source_optimization_signal_id=policy.source_optimization_signal_id,
            source_learning_signal_id=policy.source_learning_signal_id,
            linked_agent_ids=policy.linked_agent_ids,
            creative_cognition_modes=CREATIVE_COGNITION_MODES,
            governance_posture=policy.governance_posture,
            cognition_posture=policy.governance_posture,
            creative_summary=(
                f"Read-only creative cognition signal for "
                f"{policy.capability_name}; maps {policy.governance_id} into "
                "divergent, constrained, compositional, evaluative, and "
                "governed exploration posture without generation authority."
            ),
            dependency_contracts=(
                "creative cognition signal follows governance policy",
                f"cognitive governance policy:{policy.governance_id}",
                f"meta-planning projection:{policy.planning_id}",
            ),
            governance_contracts=(
                "creative cognition does not generate creative output",
                "creative cognition does not execute exploration or mutate prompts",
                "HITL required before any creative cognition behavior",
            ),
            explanation_contracts=(
                "creative cognition cites governance and planning sources",
                "creative cognition preserves capability and layer ownership",
                "creative cognition explains why no generation is applied",
            ),
            blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        for policy in governance_layer.governance_policies
    )
