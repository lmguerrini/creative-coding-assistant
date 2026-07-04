"""V6.6 Cognitive Safety Layer metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.cognitive_explanation_engine import (
    CognitiveExplanationEnginePlan,
    build_cognitive_explanation_engine,
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

COGNITIVE_SAFETY_LAYER_SERIALIZATION_VERSION = "cognitive_safety_layer.v1"
COGNITIVE_SAFETY_LAYER_ROADMAP_ITEM = "Cognitive Safety Layer"
COGNITIVE_SAFETY_LAYER_AUTHORITY_BOUNDARY = (
    "V6.6 Cognitive Safety Layer projects cognitive explanation traces into "
    "read-only safety boundaries for mutation controls, execution readiness, "
    "dependency traceability, explainability, and HITL governance. It exposes "
    "safety metadata only; it does not enforce policies, block workflows, "
    "classify live content, apply routing, mutate prompts, memory, retrieval, "
    "storage, provider selection, generated output, runtime state, or apply "
    "Runtime Evolution."
)
COGNITIVE_SAFETY_CONTROLS = (
    "mutation boundary preservation",
    "execution safety readiness",
    "governance dependency preservation",
    "explainability source-chain preservation",
    "HITL safety escalation boundary",
)


class CognitiveSafetyBoundary(BaseModel):
    """One read-only Cognitive OS safety boundary."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    safety_id: str = Field(min_length=1, max_length=190)
    explanation_id: str = Field(min_length=1, max_length=190)
    blackboard_entry_id: str = Field(min_length=1, max_length=190)
    route_decision_id: str = Field(min_length=1, max_length=190)
    plan_id: str = Field(min_length=1, max_length=190)
    schedule_id: str = Field(min_length=1, max_length=190)
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
    safety_rank: int = Field(ge=1, le=6)
    dependency_depth: int = Field(ge=0, le=5)
    safety_controls: tuple[str, ...] = Field(min_length=5, max_length=5)
    explanation_posture: CognitiveOSPosture
    safety_posture: CognitiveOSPosture
    safety_enforcement_authorized: Literal[False] = False
    workflow_blocking_authorized: Literal[False] = False
    source_trace_ids: tuple[str, ...] = Field(min_length=9, max_length=13)
    safety_summary: str = Field(min_length=1, max_length=720)
    dependency_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    safety_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _boundary_matches_sources_and_authority(self) -> Self:
        expected_safety_id = f"cognitive_safety::{self.capability_id}"
        if self.safety_id != expected_safety_id:
            raise ValueError("safety_id must match capability_id")
        expected_explanation_id = f"cognitive_explanation::{self.capability_id}"
        if self.explanation_id != expected_explanation_id:
            raise ValueError("explanation_id must match capability_id")
        expected_blackboard_id = f"cognitive_blackboard::{self.capability_id}"
        if self.blackboard_entry_id != expected_blackboard_id:
            raise ValueError("blackboard_entry_id must match capability_id")
        expected_route_id = f"cognitive_router::{self.capability_id}"
        if self.route_decision_id != expected_route_id:
            raise ValueError("route_decision_id must match capability_id")
        expected_plan_id = f"cognitive_planner::{self.capability_id}"
        if self.plan_id != expected_plan_id:
            raise ValueError("plan_id must match capability_id")
        expected_schedule_id = f"cognitive_scheduler::{self.capability_id}"
        if self.schedule_id != expected_schedule_id:
            raise ValueError("schedule_id must match capability_id")
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
        if self.safety_controls != COGNITIVE_SAFETY_CONTROLS:
            raise ValueError("safety_controls must match V6.6 safety controls")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class CognitiveSafetyLayerPlan(BaseModel):
    """Read-only safety layer over cognitive explanations."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cognitive_safety_layer"] = "cognitive_safety_layer"
    serialization_version: Literal["cognitive_safety_layer.v1"] = (
        COGNITIVE_SAFETY_LAYER_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=COGNITIVE_SAFETY_LAYER_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    cognitive_explanation_engine_role: Literal["cognitive_explanation_engine"]
    cognitive_explanation_engine_serialization_version: Literal[
        "cognitive_explanation_engine.v1"
    ]
    cognitive_blackboard_role: Literal["cognitive_blackboard"]
    cognitive_router_role: Literal["cognitive_router"]
    cognitive_planner_role: Literal["cognitive_planner"]
    cognitive_scheduler_role: Literal["cognitive_scheduler"]
    layer_order: tuple[CognitiveOSLayer, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_count: int = Field(ge=6, le=6)
    source_explanation_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_explanation_count: int = Field(ge=6, le=6)
    source_blackboard_entry_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_blackboard_entry_count: int = Field(ge=6, le=6)
    source_route_decision_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_route_decision_count: int = Field(ge=6, le=6)
    source_plan_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_plan_count: int = Field(ge=6, le=6)
    source_schedule_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_schedule_count: int = Field(ge=6, le=6)
    source_emergence_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_emergence_count: int = Field(ge=6, le=6)
    safety_boundaries: tuple[CognitiveSafetyBoundary, ...] = Field(
        min_length=6,
        max_length=6,
    )
    safety_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    candidate_safety_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    review_required_safety_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    guarded_safety_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    safety_count: int = Field(ge=6, le=6)
    candidate_safety_count: int = Field(ge=0, le=6)
    review_required_safety_count: int = Field(ge=0, le=6)
    guarded_safety_count: int = Field(ge=0, le=6)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    cognitive_safety_layer_implemented: Literal[True] = True
    cognitive_explanation_engine_integrated: Literal[True] = True
    safety_boundary_contract_implemented: Literal[True] = True
    safety_dependency_traceability_implemented: Literal[True] = True
    safety_explainability_contract_implemented: Literal[True] = True
    safety_governance_contract_implemented: Literal[True] = True
    safety_hitl_contract_implemented: Literal[True] = True
    safety_enforcement_implemented: Literal[False] = False
    workflow_blocking_implemented: Literal[False] = False
    live_content_classification_implemented: Literal[False] = False
    autonomous_workflow_planning_implemented: Literal[False] = False
    routing_application_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    enforced_safety_ids: tuple[str, ...] = Field(default_factory=tuple)
    blocked_workflow_ids: tuple[str, ...] = Field(default_factory=tuple)
    classified_content_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_safety_policy_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _safety_layer_matches_sources(self) -> Self:
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_count != len(self.capability_ids):
            raise ValueError("capability_count must match capability ids")
        if self.source_explanation_count != len(self.source_explanation_ids):
            raise ValueError("source_explanation_count must match explanation ids")
        if self.source_blackboard_entry_count != len(self.source_blackboard_entry_ids):
            raise ValueError("source_blackboard_entry_count must match entries")
        if self.source_route_decision_count != len(self.source_route_decision_ids):
            raise ValueError("source_route_decision_count must match route ids")
        if self.source_plan_count != len(self.source_plan_ids):
            raise ValueError("source_plan_count must match plan ids")
        if self.source_schedule_count != len(self.source_schedule_ids):
            raise ValueError("source_schedule_count must match schedule ids")
        if self.source_emergence_count != len(self.source_emergence_ids):
            raise ValueError("source_emergence_count must match emergence ids")
        if self.safety_ids != tuple(
            boundary.safety_id for boundary in self.safety_boundaries
        ):
            raise ValueError("safety_ids must match boundaries")
        if self.safety_count != len(self.safety_boundaries):
            raise ValueError("safety_count must match boundaries")
        if len(set(self.safety_ids)) != len(self.safety_ids):
            raise ValueError("safety_ids must be unique")
        if self.candidate_safety_ids != _safety_ids_for_posture(
            self.safety_boundaries,
            "candidate",
        ):
            raise ValueError("candidate_safety_ids must match boundaries")
        if self.review_required_safety_ids != _safety_ids_for_posture(
            self.safety_boundaries,
            "review_required",
        ):
            raise ValueError("review_required_safety_ids must match boundaries")
        if self.guarded_safety_ids != _safety_ids_for_posture(
            self.safety_boundaries,
            "guarded",
        ):
            raise ValueError("guarded_safety_ids must match boundaries")
        if self.candidate_safety_count != len(self.candidate_safety_ids):
            raise ValueError("candidate_safety_count must match ids")
        if self.review_required_safety_count != len(self.review_required_safety_ids):
            raise ValueError("review_required_safety_count must match ids")
        if self.guarded_safety_count != len(self.guarded_safety_ids):
            raise ValueError("guarded_safety_count must match ids")

        declared_capabilities = set(self.capability_ids)
        declared_explanations = set(self.source_explanation_ids)
        declared_blackboard = set(self.source_blackboard_entry_ids)
        declared_routes = set(self.source_route_decision_ids)
        declared_plans = set(self.source_plan_ids)
        declared_schedules = set(self.source_schedule_ids)
        declared_emergence = set(self.source_emergence_ids)
        declared_agents = set(self.linked_agent_ids)
        for boundary in self.safety_boundaries:
            if boundary.capability_id not in declared_capabilities:
                raise ValueError("boundary capability_id must be declared")
            if boundary.explanation_id not in declared_explanations:
                raise ValueError("boundary explanation_id must be declared")
            if boundary.blackboard_entry_id not in declared_blackboard:
                raise ValueError("boundary blackboard_entry_id must be declared")
            if boundary.route_decision_id not in declared_routes:
                raise ValueError("boundary route_decision_id must be declared")
            if boundary.plan_id not in declared_plans:
                raise ValueError("boundary plan_id must be declared")
            if boundary.schedule_id not in declared_schedules:
                raise ValueError("boundary schedule_id must be declared")
            if boundary.emergence_id not in declared_emergence:
                raise ValueError("boundary emergence_id must be declared")
            if not set(boundary.linked_agent_ids).issubset(declared_agents):
                raise ValueError("boundary linked_agent_ids must be declared")
        if self.covered_roadmap_items != (COGNITIVE_SAFETY_LAYER_ROADMAP_ITEM,):
            raise ValueError("covered_roadmap_items must be Task 22 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.enforced_safety_ids,
                self.blocked_workflow_ids,
                self.classified_content_ids,
                self.mutated_safety_policy_ids,
                self.emitted_hitl_request_ids,
            )
        ):
            raise ValueError(
                "safety enforcement, workflow blocking, classification, "
                "mutation, and HITL ids must be empty",
            )
        if not all(boundary.advisory_only for boundary in self.safety_boundaries):
            raise ValueError("all cognitive safety boundaries must be advisory only")
        return self


def build_cognitive_safety_layer(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    cognitive_explanation_engine: CognitiveExplanationEnginePlan | None = None,
) -> CognitiveSafetyLayerPlan:
    """Build read-only cognitive safety metadata."""

    explanation_engine = (
        cognitive_explanation_engine
        or build_cognitive_explanation_engine(
            route=route,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
        )
    )
    boundaries = _cognitive_safety_boundaries(explanation_engine)
    candidate_ids = _safety_ids_for_posture(boundaries, "candidate")
    review_required_ids = _safety_ids_for_posture(boundaries, "review_required")
    guarded_ids = _safety_ids_for_posture(boundaries, "guarded")
    return CognitiveSafetyLayerPlan(
        route_name=explanation_engine.route_name,
        task_type=explanation_engine.task_type,
        execution_mode_ids=explanation_engine.execution_mode_ids,
        cognitive_explanation_engine_role=explanation_engine.role,
        cognitive_explanation_engine_serialization_version=(
            explanation_engine.serialization_version
        ),
        cognitive_blackboard_role=explanation_engine.cognitive_blackboard_role,
        cognitive_router_role=explanation_engine.cognitive_router_role,
        cognitive_planner_role=explanation_engine.cognitive_planner_role,
        cognitive_scheduler_role=explanation_engine.cognitive_scheduler_role,
        layer_order=explanation_engine.layer_order,
        capabilities=explanation_engine.capabilities,
        capability_ids=explanation_engine.capability_ids,
        capability_count=explanation_engine.capability_count,
        source_explanation_ids=explanation_engine.explanation_ids,
        source_explanation_count=explanation_engine.explanation_count,
        source_blackboard_entry_ids=(explanation_engine.source_blackboard_entry_ids),
        source_blackboard_entry_count=(
            explanation_engine.source_blackboard_entry_count
        ),
        source_route_decision_ids=explanation_engine.source_route_decision_ids,
        source_route_decision_count=(explanation_engine.source_route_decision_count),
        source_plan_ids=explanation_engine.source_plan_ids,
        source_plan_count=explanation_engine.source_plan_count,
        source_schedule_ids=explanation_engine.source_schedule_ids,
        source_schedule_count=explanation_engine.source_schedule_count,
        source_emergence_ids=explanation_engine.source_emergence_ids,
        source_emergence_count=explanation_engine.source_emergence_count,
        safety_boundaries=boundaries,
        safety_ids=tuple(boundary.safety_id for boundary in boundaries),
        candidate_safety_ids=candidate_ids,
        review_required_safety_ids=review_required_ids,
        guarded_safety_ids=guarded_ids,
        safety_count=len(boundaries),
        candidate_safety_count=len(candidate_ids),
        review_required_safety_count=len(review_required_ids),
        guarded_safety_count=len(guarded_ids),
        linked_agent_ids=explanation_engine.linked_agent_ids,
        covered_roadmap_items=(COGNITIVE_SAFETY_LAYER_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture=explanation_engine.graph_posture,
    )


def cognitive_safety_boundary_by_id(
    safety_id: str,
    safety_layer: CognitiveSafetyLayerPlan | None = None,
) -> CognitiveSafetyBoundary | None:
    """Return one cognitive safety boundary without enforcing it."""

    source_layer = safety_layer or build_cognitive_safety_layer()
    for boundary in source_layer.safety_boundaries:
        if boundary.safety_id == safety_id:
            return boundary
    return None


def cognitive_safety_boundaries_for_layer(
    cognitive_layer: CognitiveOSLayer,
    safety_layer: CognitiveSafetyLayerPlan | None = None,
) -> tuple[CognitiveSafetyBoundary, ...]:
    """Return safety boundaries for one Cognitive OS layer."""

    source_layer = safety_layer or build_cognitive_safety_layer()
    return tuple(
        boundary
        for boundary in source_layer.safety_boundaries
        if boundary.cognitive_layer == cognitive_layer
    )


def cognitive_safety_boundaries_for_agent(
    agent_id: str,
    safety_layer: CognitiveSafetyLayerPlan | None = None,
) -> tuple[CognitiveSafetyBoundary, ...]:
    """Return safety boundaries linked to one agent."""

    source_layer = safety_layer or build_cognitive_safety_layer()
    return tuple(
        boundary
        for boundary in source_layer.safety_boundaries
        if agent_id in boundary.linked_agent_ids
    )


def cognitive_safety_boundaries_for_posture(
    posture: CognitiveOSPosture,
    safety_layer: CognitiveSafetyLayerPlan | None = None,
) -> tuple[CognitiveSafetyBoundary, ...]:
    """Return safety boundaries by posture without enforcing policy."""

    source_layer = safety_layer or build_cognitive_safety_layer()
    return tuple(
        boundary
        for boundary in source_layer.safety_boundaries
        if boundary.safety_posture == posture
    )


def _cognitive_safety_boundaries(
    explanation_engine: CognitiveExplanationEnginePlan,
) -> tuple[CognitiveSafetyBoundary, ...]:
    return tuple(
        CognitiveSafetyBoundary(
            safety_id=f"cognitive_safety::{trace.capability_id}",
            explanation_id=trace.explanation_id,
            blackboard_entry_id=trace.blackboard_entry_id,
            route_decision_id=trace.route_decision_id,
            plan_id=trace.plan_id,
            schedule_id=trace.schedule_id,
            emergence_id=trace.emergence_id,
            identity_id=trace.identity_id,
            cognition_id=trace.cognition_id,
            governance_id=trace.governance_id,
            planning_id=trace.planning_id,
            reasoning_id=trace.reasoning_id,
            profile_id=trace.profile_id,
            state_id=trace.state_id,
            capability_id=trace.capability_id,
            capability_name=trace.capability_name,
            cognitive_layer=trace.cognitive_layer,
            linked_agent_ids=trace.linked_agent_ids,
            safety_rank=trace.explanation_rank,
            dependency_depth=trace.dependency_depth,
            safety_controls=COGNITIVE_SAFETY_CONTROLS,
            explanation_posture=trace.explanation_posture,
            safety_posture=trace.explanation_posture,
            source_trace_ids=(trace.explanation_id, *trace.source_trace_ids),
            safety_summary=(
                f"Read-only cognitive safety boundary for "
                f"{trace.capability_name}; cites explanation, blackboard, "
                "routing, planning, scheduling, emergence, identity, "
                "cognition, governance, reasoning, profile, and state "
                "metadata without enforcing policy or blocking workflows."
            ),
            dependency_contracts=(
                "cognitive safety follows cognitive explanation trace",
                f"cognitive explanation:{trace.explanation_id}",
                f"cognitive blackboard entry:{trace.blackboard_entry_id}",
            ),
            governance_contracts=(
                "cognitive safety layer does not enforce policies",
                "cognitive safety layer does not block workflows",
                "HITL required before any safety-driven behavior",
            ),
            explanation_contracts=(
                "cognitive safety cites the full cognitive source chain",
                "cognitive safety preserves capability and agent ownership",
                "cognitive safety explains why enforcement is not authorized",
            ),
            safety_contracts=(
                "cognitive safety records mutation boundaries only",
                "cognitive safety records execution readiness only",
                "cognitive safety records HITL escalation boundaries only",
            ),
            blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        for trace in explanation_engine.explanation_traces
    )


def _safety_ids_for_posture(
    boundaries: tuple[CognitiveSafetyBoundary, ...],
    posture: CognitiveOSPosture,
) -> tuple[str, ...]:
    return tuple(
        boundary.safety_id
        for boundary in boundaries
        if boundary.safety_posture == posture
    )
