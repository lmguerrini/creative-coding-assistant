"""V6.6 Cognitive HITL Layer metadata."""

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
from creative_coding_assistant.orchestration.cognitive_safety_layer import (
    CognitiveSafetyLayerPlan,
    build_cognitive_safety_layer,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)

COGNITIVE_HITL_LAYER_SERIALIZATION_VERSION = "cognitive_hitl_layer.v1"
COGNITIVE_HITL_LAYER_ROADMAP_ITEM = "Cognitive HITL Layer"
COGNITIVE_HITL_LAYER_AUTHORITY_BOUNDARY = (
    "V6.6 Cognitive HITL Layer projects cognitive safety boundaries into "
    "read-only HITL checkpoints for review traceability, decision ownership, "
    "safety escalation, explainability, and dependency awareness. It exposes "
    "HITL readiness metadata only; it does not emit HITL requests, apply "
    "HITL decisions, enforce safety policies, block workflows, apply routing, "
    "mutate prompts, memory, retrieval, storage, provider selection, "
    "generated output, runtime state, or apply Runtime Evolution."
)
COGNITIVE_HITL_CONTROLS = (
    "HITL review boundary",
    "decision application boundary",
    "ownership escalation context",
    "safety escalation context",
    "explanation review context",
)


class CognitiveHITLCheckpoint(BaseModel):
    """One read-only Cognitive OS HITL checkpoint."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    hitl_id: str = Field(min_length=1, max_length=190)
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
    hitl_rank: int = Field(ge=1, le=6)
    dependency_depth: int = Field(ge=0, le=5)
    hitl_controls: tuple[str, ...] = Field(min_length=5, max_length=5)
    safety_posture: CognitiveOSPosture
    hitl_posture: CognitiveOSPosture
    hitl_required_before_application: Literal[True] = True
    hitl_request_emission_authorized: Literal[False] = False
    hitl_decision_application_authorized: Literal[False] = False
    source_trace_ids: tuple[str, ...] = Field(min_length=10, max_length=14)
    hitl_summary: str = Field(min_length=1, max_length=760)
    dependency_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    safety_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    hitl_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _checkpoint_matches_sources_and_authority(self) -> Self:
        expected_hitl_id = f"cognitive_hitl::{self.capability_id}"
        if self.hitl_id != expected_hitl_id:
            raise ValueError("hitl_id must match capability_id")
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
        if self.hitl_controls != COGNITIVE_HITL_CONTROLS:
            raise ValueError("hitl_controls must match V6.6 HITL controls")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class CognitiveHITLLayerPlan(BaseModel):
    """Read-only HITL layer over cognitive safety boundaries."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cognitive_hitl_layer"] = "cognitive_hitl_layer"
    serialization_version: Literal["cognitive_hitl_layer.v1"] = (
        COGNITIVE_HITL_LAYER_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=COGNITIVE_HITL_LAYER_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    cognitive_safety_layer_role: Literal["cognitive_safety_layer"]
    cognitive_safety_layer_serialization_version: Literal["cognitive_safety_layer.v1"]
    cognitive_explanation_engine_role: Literal["cognitive_explanation_engine"]
    cognitive_blackboard_role: Literal["cognitive_blackboard"]
    cognitive_router_role: Literal["cognitive_router"]
    cognitive_planner_role: Literal["cognitive_planner"]
    cognitive_scheduler_role: Literal["cognitive_scheduler"]
    layer_order: tuple[CognitiveOSLayer, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_count: int = Field(ge=6, le=6)
    source_safety_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_safety_count: int = Field(ge=6, le=6)
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
    hitl_checkpoints: tuple[CognitiveHITLCheckpoint, ...] = Field(
        min_length=6,
        max_length=6,
    )
    hitl_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    hitl_required_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    candidate_hitl_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    review_required_hitl_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    guarded_hitl_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    hitl_count: int = Field(ge=6, le=6)
    hitl_required_count: int = Field(ge=6, le=6)
    candidate_hitl_count: int = Field(ge=0, le=6)
    review_required_hitl_count: int = Field(ge=0, le=6)
    guarded_hitl_count: int = Field(ge=0, le=6)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    cognitive_hitl_layer_implemented: Literal[True] = True
    cognitive_safety_layer_integrated: Literal[True] = True
    hitl_checkpoint_contract_implemented: Literal[True] = True
    hitl_dependency_traceability_implemented: Literal[True] = True
    hitl_explainability_contract_implemented: Literal[True] = True
    hitl_safety_contract_implemented: Literal[True] = True
    hitl_governance_contract_implemented: Literal[True] = True
    hitl_request_emission_implemented: Literal[False] = False
    hitl_decision_application_implemented: Literal[False] = False
    safety_enforcement_implemented: Literal[False] = False
    workflow_blocking_implemented: Literal[False] = False
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
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    applied_hitl_decision_ids: tuple[str, ...] = Field(default_factory=tuple)
    enforced_safety_ids: tuple[str, ...] = Field(default_factory=tuple)
    blocked_workflow_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_hitl_policy_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _hitl_layer_matches_sources(self) -> Self:
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_count != len(self.capability_ids):
            raise ValueError("capability_count must match capability ids")
        if self.source_safety_count != len(self.source_safety_ids):
            raise ValueError("source_safety_count must match safety ids")
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
        if self.hitl_ids != tuple(
            checkpoint.hitl_id for checkpoint in self.hitl_checkpoints
        ):
            raise ValueError("hitl_ids must match checkpoints")
        if self.hitl_count != len(self.hitl_checkpoints):
            raise ValueError("hitl_count must match checkpoints")
        if len(set(self.hitl_ids)) != len(self.hitl_ids):
            raise ValueError("hitl_ids must be unique")
        if self.hitl_required_ids != tuple(
            checkpoint.hitl_id
            for checkpoint in self.hitl_checkpoints
            if checkpoint.hitl_required_before_application
        ):
            raise ValueError("hitl_required_ids must match checkpoints")
        if self.hitl_required_count != len(self.hitl_required_ids):
            raise ValueError("hitl_required_count must match ids")
        if self.candidate_hitl_ids != _hitl_ids_for_posture(
            self.hitl_checkpoints,
            "candidate",
        ):
            raise ValueError("candidate_hitl_ids must match checkpoints")
        if self.review_required_hitl_ids != _hitl_ids_for_posture(
            self.hitl_checkpoints,
            "review_required",
        ):
            raise ValueError("review_required_hitl_ids must match checkpoints")
        if self.guarded_hitl_ids != _hitl_ids_for_posture(
            self.hitl_checkpoints,
            "guarded",
        ):
            raise ValueError("guarded_hitl_ids must match checkpoints")
        if self.candidate_hitl_count != len(self.candidate_hitl_ids):
            raise ValueError("candidate_hitl_count must match ids")
        if self.review_required_hitl_count != len(self.review_required_hitl_ids):
            raise ValueError("review_required_hitl_count must match ids")
        if self.guarded_hitl_count != len(self.guarded_hitl_ids):
            raise ValueError("guarded_hitl_count must match ids")

        declared_capabilities = set(self.capability_ids)
        declared_safety = set(self.source_safety_ids)
        declared_explanations = set(self.source_explanation_ids)
        declared_blackboard = set(self.source_blackboard_entry_ids)
        declared_routes = set(self.source_route_decision_ids)
        declared_plans = set(self.source_plan_ids)
        declared_schedules = set(self.source_schedule_ids)
        declared_emergence = set(self.source_emergence_ids)
        declared_agents = set(self.linked_agent_ids)
        for checkpoint in self.hitl_checkpoints:
            if checkpoint.capability_id not in declared_capabilities:
                raise ValueError("checkpoint capability_id must be declared")
            if checkpoint.safety_id not in declared_safety:
                raise ValueError("checkpoint safety_id must be declared")
            if checkpoint.explanation_id not in declared_explanations:
                raise ValueError("checkpoint explanation_id must be declared")
            if checkpoint.blackboard_entry_id not in declared_blackboard:
                raise ValueError("checkpoint blackboard_entry_id must be declared")
            if checkpoint.route_decision_id not in declared_routes:
                raise ValueError("checkpoint route_decision_id must be declared")
            if checkpoint.plan_id not in declared_plans:
                raise ValueError("checkpoint plan_id must be declared")
            if checkpoint.schedule_id not in declared_schedules:
                raise ValueError("checkpoint schedule_id must be declared")
            if checkpoint.emergence_id not in declared_emergence:
                raise ValueError("checkpoint emergence_id must be declared")
            if not set(checkpoint.linked_agent_ids).issubset(declared_agents):
                raise ValueError("checkpoint linked_agent_ids must be declared")
        if self.covered_roadmap_items != (COGNITIVE_HITL_LAYER_ROADMAP_ITEM,):
            raise ValueError("covered_roadmap_items must be Task 23 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.emitted_hitl_request_ids,
                self.applied_hitl_decision_ids,
                self.enforced_safety_ids,
                self.blocked_workflow_ids,
                self.mutated_hitl_policy_ids,
            )
        ):
            raise ValueError(
                "HITL request emission, decision application, safety "
                "enforcement, workflow blocking, and mutation ids must be empty",
            )
        if not all(checkpoint.advisory_only for checkpoint in self.hitl_checkpoints):
            raise ValueError("all cognitive HITL checkpoints must be advisory only")
        return self


def build_cognitive_hitl_layer(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    cognitive_safety_layer: CognitiveSafetyLayerPlan | None = None,
) -> CognitiveHITLLayerPlan:
    """Build read-only cognitive HITL metadata."""

    safety_layer = cognitive_safety_layer or build_cognitive_safety_layer(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    checkpoints = _cognitive_hitl_checkpoints(safety_layer)
    candidate_ids = _hitl_ids_for_posture(checkpoints, "candidate")
    review_required_ids = _hitl_ids_for_posture(checkpoints, "review_required")
    guarded_ids = _hitl_ids_for_posture(checkpoints, "guarded")
    hitl_ids = tuple(checkpoint.hitl_id for checkpoint in checkpoints)
    return CognitiveHITLLayerPlan(
        route_name=safety_layer.route_name,
        task_type=safety_layer.task_type,
        execution_mode_ids=safety_layer.execution_mode_ids,
        cognitive_safety_layer_role=safety_layer.role,
        cognitive_safety_layer_serialization_version=(
            safety_layer.serialization_version
        ),
        cognitive_explanation_engine_role=(
            safety_layer.cognitive_explanation_engine_role
        ),
        cognitive_blackboard_role=safety_layer.cognitive_blackboard_role,
        cognitive_router_role=safety_layer.cognitive_router_role,
        cognitive_planner_role=safety_layer.cognitive_planner_role,
        cognitive_scheduler_role=safety_layer.cognitive_scheduler_role,
        layer_order=safety_layer.layer_order,
        capabilities=safety_layer.capabilities,
        capability_ids=safety_layer.capability_ids,
        capability_count=safety_layer.capability_count,
        source_safety_ids=safety_layer.safety_ids,
        source_safety_count=safety_layer.safety_count,
        source_explanation_ids=safety_layer.source_explanation_ids,
        source_explanation_count=safety_layer.source_explanation_count,
        source_blackboard_entry_ids=safety_layer.source_blackboard_entry_ids,
        source_blackboard_entry_count=safety_layer.source_blackboard_entry_count,
        source_route_decision_ids=safety_layer.source_route_decision_ids,
        source_route_decision_count=safety_layer.source_route_decision_count,
        source_plan_ids=safety_layer.source_plan_ids,
        source_plan_count=safety_layer.source_plan_count,
        source_schedule_ids=safety_layer.source_schedule_ids,
        source_schedule_count=safety_layer.source_schedule_count,
        source_emergence_ids=safety_layer.source_emergence_ids,
        source_emergence_count=safety_layer.source_emergence_count,
        hitl_checkpoints=checkpoints,
        hitl_ids=hitl_ids,
        hitl_required_ids=hitl_ids,
        candidate_hitl_ids=candidate_ids,
        review_required_hitl_ids=review_required_ids,
        guarded_hitl_ids=guarded_ids,
        hitl_count=len(checkpoints),
        hitl_required_count=len(hitl_ids),
        candidate_hitl_count=len(candidate_ids),
        review_required_hitl_count=len(review_required_ids),
        guarded_hitl_count=len(guarded_ids),
        linked_agent_ids=safety_layer.linked_agent_ids,
        covered_roadmap_items=(COGNITIVE_HITL_LAYER_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture=safety_layer.graph_posture,
    )


def cognitive_hitl_checkpoint_by_id(
    hitl_id: str,
    hitl_layer: CognitiveHITLLayerPlan | None = None,
) -> CognitiveHITLCheckpoint | None:
    """Return one cognitive HITL checkpoint without emitting a request."""

    source_layer = hitl_layer or build_cognitive_hitl_layer()
    for checkpoint in source_layer.hitl_checkpoints:
        if checkpoint.hitl_id == hitl_id:
            return checkpoint
    return None


def cognitive_hitl_checkpoints_for_layer(
    cognitive_layer: CognitiveOSLayer,
    hitl_layer: CognitiveHITLLayerPlan | None = None,
) -> tuple[CognitiveHITLCheckpoint, ...]:
    """Return HITL checkpoints for one Cognitive OS layer."""

    source_layer = hitl_layer or build_cognitive_hitl_layer()
    return tuple(
        checkpoint
        for checkpoint in source_layer.hitl_checkpoints
        if checkpoint.cognitive_layer == cognitive_layer
    )


def cognitive_hitl_checkpoints_for_agent(
    agent_id: str,
    hitl_layer: CognitiveHITLLayerPlan | None = None,
) -> tuple[CognitiveHITLCheckpoint, ...]:
    """Return HITL checkpoints linked to one agent."""

    source_layer = hitl_layer or build_cognitive_hitl_layer()
    return tuple(
        checkpoint
        for checkpoint in source_layer.hitl_checkpoints
        if agent_id in checkpoint.linked_agent_ids
    )


def cognitive_hitl_checkpoints_for_posture(
    posture: CognitiveOSPosture,
    hitl_layer: CognitiveHITLLayerPlan | None = None,
) -> tuple[CognitiveHITLCheckpoint, ...]:
    """Return HITL checkpoints by posture without applying decisions."""

    source_layer = hitl_layer or build_cognitive_hitl_layer()
    return tuple(
        checkpoint
        for checkpoint in source_layer.hitl_checkpoints
        if checkpoint.hitl_posture == posture
    )


def _cognitive_hitl_checkpoints(
    safety_layer: CognitiveSafetyLayerPlan,
) -> tuple[CognitiveHITLCheckpoint, ...]:
    return tuple(
        CognitiveHITLCheckpoint(
            hitl_id=f"cognitive_hitl::{boundary.capability_id}",
            safety_id=boundary.safety_id,
            explanation_id=boundary.explanation_id,
            blackboard_entry_id=boundary.blackboard_entry_id,
            route_decision_id=boundary.route_decision_id,
            plan_id=boundary.plan_id,
            schedule_id=boundary.schedule_id,
            emergence_id=boundary.emergence_id,
            identity_id=boundary.identity_id,
            cognition_id=boundary.cognition_id,
            governance_id=boundary.governance_id,
            planning_id=boundary.planning_id,
            reasoning_id=boundary.reasoning_id,
            profile_id=boundary.profile_id,
            state_id=boundary.state_id,
            capability_id=boundary.capability_id,
            capability_name=boundary.capability_name,
            cognitive_layer=boundary.cognitive_layer,
            linked_agent_ids=boundary.linked_agent_ids,
            hitl_rank=boundary.safety_rank,
            dependency_depth=boundary.dependency_depth,
            hitl_controls=COGNITIVE_HITL_CONTROLS,
            safety_posture=boundary.safety_posture,
            hitl_posture=boundary.safety_posture,
            source_trace_ids=(boundary.safety_id, *boundary.source_trace_ids),
            hitl_summary=(
                f"Read-only cognitive HITL checkpoint for "
                f"{boundary.capability_name}; cites safety, explanation, "
                "blackboard, routing, planning, scheduling, emergence, "
                "identity, cognition, governance, reasoning, profile, and "
                "state metadata without emitting HITL requests or applying "
                "decisions."
            ),
            dependency_contracts=(
                "cognitive HITL follows cognitive safety boundary",
                f"cognitive safety:{boundary.safety_id}",
                f"cognitive explanation:{boundary.explanation_id}",
            ),
            governance_contracts=(
                "cognitive HITL layer does not apply decisions",
                "cognitive HITL layer does not enforce safety policies",
                "HITL review must occur before any behavioral application",
            ),
            explanation_contracts=(
                "cognitive HITL cites the full cognitive source chain",
                "cognitive HITL preserves capability and agent ownership",
                "cognitive HITL explains why request emission is not authorized",
            ),
            safety_contracts=(
                "cognitive HITL preserves safety boundary metadata",
                "cognitive HITL preserves workflow blocking boundary",
                "cognitive HITL preserves mutation boundary metadata",
            ),
            hitl_contracts=(
                "cognitive HITL records review requirement only",
                "cognitive HITL records decision ownership only",
                "cognitive HITL records escalation readiness only",
            ),
            blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        for boundary in safety_layer.safety_boundaries
    )


def _hitl_ids_for_posture(
    checkpoints: tuple[CognitiveHITLCheckpoint, ...],
    posture: CognitiveOSPosture,
) -> tuple[str, ...]:
    return tuple(
        checkpoint.hitl_id
        for checkpoint in checkpoints
        if checkpoint.hitl_posture == posture
    )
