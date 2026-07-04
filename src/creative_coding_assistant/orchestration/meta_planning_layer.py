"""V6.6 Meta-Planning Layer metadata."""

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
from creative_coding_assistant.orchestration.meta_reasoning_layer import (
    MetaReasoningLayerPlan,
    build_meta_reasoning_layer,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)

META_PLANNING_LAYER_SERIALIZATION_VERSION = "meta_planning_layer.v1"
META_PLANNING_LAYER_ROADMAP_ITEM = "Meta-Planning Layer"
META_PLANNING_LAYER_AUTHORITY_BOUNDARY = (
    "V6.6 Meta-Planning Layer projects meta-reasoning assessments into "
    "read-only planning metadata for capability ordering, dependency "
    "awareness, governance checkpoint placement, explainability, and HITL "
    "readiness. It does not autonomously plan or execute workflows, mutate "
    "plans, prompts, memory, retrieval, storage, provider selection, "
    "generated output, runtime state, or apply Runtime Evolution."
)
META_PLANNING_FOCUSES = (
    "reasoning dependency ordering",
    "capability handoff readiness",
    "governance checkpoint placement",
    "explanation continuity",
    "HITL readiness",
)


class MetaPlanningProjection(BaseModel):
    """One read-only planning projection for a meta-reasoning assessment."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

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
    planning_focuses: tuple[str, ...] = Field(min_length=5, max_length=5)
    reasoning_posture: CognitiveOSPosture
    planning_posture: CognitiveOSPosture
    planning_summary: str = Field(min_length=1, max_length=480)
    dependency_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _projection_matches_sources_and_boundary(self) -> Self:
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
        if self.planning_focuses != META_PLANNING_FOCUSES:
            raise ValueError("planning_focuses must match V6.6 planning contract")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class MetaPlanningLayerPlan(BaseModel):
    """Read-only meta-planning layer over meta-reasoning assessments."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["meta_planning_layer"] = "meta_planning_layer"
    serialization_version: Literal["meta_planning_layer.v1"] = (
        META_PLANNING_LAYER_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=META_PLANNING_LAYER_AUTHORITY_BOUNDARY,
        max_length=1900,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    meta_reasoning_layer_role: Literal["meta_reasoning_layer"]
    meta_reasoning_layer_serialization_version: Literal["meta_reasoning_layer.v1"]
    profile_engine_role: Literal["cognitive_profile_engine"]
    state_engine_role: Literal["cognitive_state_engine"]
    layer_order: tuple[CognitiveOSLayer, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_count: int = Field(ge=6, le=6)
    source_reasoning_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_reasoning_count: int = Field(ge=6, le=6)
    source_profile_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_profile_count: int = Field(ge=6, le=6)
    source_state_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_state_count: int = Field(ge=6, le=6)
    source_optimization_signal_ids: tuple[str, ...] = Field(
        min_length=6,
        max_length=6,
    )
    source_optimization_signal_count: int = Field(ge=6, le=6)
    source_learning_signal_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_learning_signal_count: int = Field(ge=6, le=6)
    planning_projections: tuple[MetaPlanningProjection, ...] = Field(
        min_length=6,
        max_length=6,
    )
    planning_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    planning_count: int = Field(ge=6, le=6)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    meta_planning_layer_implemented: Literal[True] = True
    meta_reasoning_layer_integrated: Literal[True] = True
    planning_projection_contract_implemented: Literal[True] = True
    planning_dependency_traceability_implemented: Literal[True] = True
    planning_governance_contract_implemented: Literal[True] = True
    planning_explainability_contract_implemented: Literal[True] = True
    autonomous_workflow_planning_implemented: Literal[False] = False
    planning_execution_implemented: Literal[False] = False
    plan_mutation_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    executed_planning_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_plan_ids: tuple[str, ...] = Field(default_factory=tuple)
    routed_planning_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _meta_planning_matches_sources(self) -> Self:
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_count != len(self.capability_ids):
            raise ValueError("capability_count must match capability ids")
        if self.source_reasoning_count != len(self.source_reasoning_ids):
            raise ValueError("source_reasoning_count must match reasoning ids")
        if self.source_profile_count != len(self.source_profile_ids):
            raise ValueError("source_profile_count must match profile ids")
        if self.source_state_count != len(self.source_state_ids):
            raise ValueError("source_state_count must match state ids")
        if self.source_optimization_signal_count != len(
            self.source_optimization_signal_ids
        ):
            raise ValueError("source_optimization_signal_count must match signals")
        if self.source_learning_signal_count != len(self.source_learning_signal_ids):
            raise ValueError("source_learning_signal_count must match signals")
        if self.planning_ids != tuple(
            projection.planning_id for projection in self.planning_projections
        ):
            raise ValueError("planning_ids must match projections")
        if self.planning_count != len(self.planning_projections):
            raise ValueError("planning_count must match projections")
        if len(set(self.planning_ids)) != len(self.planning_ids):
            raise ValueError("planning_ids must be unique")
        declared_capabilities = set(self.capability_ids)
        declared_reasoning = set(self.source_reasoning_ids)
        declared_profiles = set(self.source_profile_ids)
        declared_states = set(self.source_state_ids)
        declared_optimization_signals = set(self.source_optimization_signal_ids)
        declared_learning_signals = set(self.source_learning_signal_ids)
        declared_agents = set(self.linked_agent_ids)
        for projection in self.planning_projections:
            if projection.capability_id not in declared_capabilities:
                raise ValueError("projection capability_id must be declared")
            if projection.reasoning_id not in declared_reasoning:
                raise ValueError("projection reasoning_id must be declared")
            if projection.profile_id not in declared_profiles:
                raise ValueError("projection profile_id must be declared")
            if projection.state_id not in declared_states:
                raise ValueError("projection state_id must be declared")
            if (
                projection.source_optimization_signal_id
                not in declared_optimization_signals
            ):
                raise ValueError("projection optimization signal must be declared")
            if projection.source_learning_signal_id not in declared_learning_signals:
                raise ValueError("projection learning signal must be declared")
            if not set(projection.linked_agent_ids).issubset(declared_agents):
                raise ValueError("projection linked_agent_ids must be declared")
        if self.covered_roadmap_items != (META_PLANNING_LAYER_ROADMAP_ITEM,):
            raise ValueError("covered_roadmap_items must be Task 12 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.executed_planning_ids,
                self.mutated_plan_ids,
                self.routed_planning_ids,
                self.emitted_hitl_request_ids,
            )
        ):
            raise ValueError(
                "planning execution, mutation, routing, and HITL ids must be empty",
            )
        if not all(
            projection.advisory_only for projection in self.planning_projections
        ):
            raise ValueError("all meta-planning projections must be advisory only")
        return self


def build_meta_planning_layer(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    meta_reasoning_layer: MetaReasoningLayerPlan | None = None,
) -> MetaPlanningLayerPlan:
    """Build read-only meta-planning metadata."""

    reasoning = meta_reasoning_layer or build_meta_reasoning_layer(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    projections = _meta_planning_projections(reasoning)
    return MetaPlanningLayerPlan(
        route_name=reasoning.route_name,
        task_type=reasoning.task_type,
        execution_mode_ids=reasoning.execution_mode_ids,
        meta_reasoning_layer_role=reasoning.role,
        meta_reasoning_layer_serialization_version=reasoning.serialization_version,
        profile_engine_role=reasoning.profile_engine_role,
        state_engine_role=reasoning.state_engine_role,
        layer_order=reasoning.layer_order,
        capabilities=reasoning.capabilities,
        capability_ids=reasoning.capability_ids,
        capability_count=reasoning.capability_count,
        source_reasoning_ids=reasoning.reasoning_ids,
        source_reasoning_count=reasoning.reasoning_count,
        source_profile_ids=reasoning.source_profile_ids,
        source_profile_count=reasoning.source_profile_count,
        source_state_ids=reasoning.source_state_ids,
        source_state_count=reasoning.source_state_count,
        source_optimization_signal_ids=reasoning.source_optimization_signal_ids,
        source_optimization_signal_count=reasoning.source_optimization_signal_count,
        source_learning_signal_ids=reasoning.source_learning_signal_ids,
        source_learning_signal_count=reasoning.source_learning_signal_count,
        planning_projections=projections,
        planning_ids=tuple(projection.planning_id for projection in projections),
        planning_count=len(projections),
        linked_agent_ids=reasoning.linked_agent_ids,
        covered_roadmap_items=(META_PLANNING_LAYER_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture=reasoning.graph_posture,
    )


def meta_planning_projection_by_id(
    planning_id: str,
    layer: MetaPlanningLayerPlan | None = None,
) -> MetaPlanningProjection | None:
    """Return one meta-planning projection without executing it."""

    source_layer = layer or build_meta_planning_layer()
    for projection in source_layer.planning_projections:
        if projection.planning_id == planning_id:
            return projection
    return None


def meta_planning_projections_for_layer(
    cognitive_layer: CognitiveOSLayer,
    layer: MetaPlanningLayerPlan | None = None,
) -> tuple[MetaPlanningProjection, ...]:
    """Return meta-planning projections for one Cognitive OS layer."""

    source_layer = layer or build_meta_planning_layer()
    return tuple(
        projection
        for projection in source_layer.planning_projections
        if projection.cognitive_layer == cognitive_layer
    )


def meta_planning_projections_for_agent(
    agent_id: str,
    layer: MetaPlanningLayerPlan | None = None,
) -> tuple[MetaPlanningProjection, ...]:
    """Return meta-planning projections linked to one agent."""

    source_layer = layer or build_meta_planning_layer()
    return tuple(
        projection
        for projection in source_layer.planning_projections
        if agent_id in projection.linked_agent_ids
    )


def _meta_planning_projections(
    reasoning_layer: MetaReasoningLayerPlan,
) -> tuple[MetaPlanningProjection, ...]:
    return tuple(
        MetaPlanningProjection(
            planning_id=f"meta_planning::{assessment.capability_id}",
            reasoning_id=assessment.reasoning_id,
            profile_id=assessment.profile_id,
            state_id=assessment.state_id,
            capability_id=assessment.capability_id,
            capability_name=assessment.capability_name,
            cognitive_layer=assessment.cognitive_layer,
            source_optimization_signal_id=(assessment.source_optimization_signal_id),
            source_learning_signal_id=assessment.source_learning_signal_id,
            linked_agent_ids=assessment.linked_agent_ids,
            planning_focuses=META_PLANNING_FOCUSES,
            reasoning_posture=assessment.reasoning_posture,
            planning_posture=assessment.reasoning_posture,
            planning_summary=(
                f"Read-only meta-planning projection for "
                f"{assessment.capability_name}; orders "
                f"{assessment.reasoning_id} around dependencies, governance, "
                "explanation continuity, and HITL readiness without execution."
            ),
            dependency_contracts=(
                "meta-planning projection follows meta-reasoning assessment",
                f"meta-reasoning assessment:{assessment.reasoning_id}",
                f"cognitive profile:{assessment.profile_id}",
            ),
            governance_contracts=(
                "meta-planning does not execute autonomous workflows",
                "meta-planning does not mutate prompts, memory, retrieval, or storage",
                "HITL required before any planning-driven behavior",
            ),
            explanation_contracts=(
                "meta-planning cites reasoning, profile, and state sources",
                "meta-planning preserves capability and layer ownership",
                "meta-planning explains why no workflow behavior is applied",
            ),
            blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        for assessment in reasoning_layer.reasoning_assessments
    )
