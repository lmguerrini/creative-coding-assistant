"""V6.6 Cognitive Planner metadata."""

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
from creative_coding_assistant.orchestration.cognitive_scheduler import (
    CognitiveSchedulerPlan,
    build_cognitive_scheduler,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)

COGNITIVE_PLANNER_SERIALIZATION_VERSION = "cognitive_planner.v1"
COGNITIVE_PLANNER_ROADMAP_ITEM = "Cognitive Planner"
COGNITIVE_PLANNER_AUTHORITY_BOUNDARY = (
    "V6.6 Cognitive Planner projects cognitive schedule slots into "
    "read-only cognitive plan steps for goal decomposition posture, "
    "schedule-to-plan continuity, dependency-aware sequencing, governance "
    "checkpoint mapping, explanation continuity, and HITL readiness. It "
    "exposes planner metadata only; it does not execute plans, autonomously "
    "create workflows, mutate workflows, route requests, invoke agents, "
    "change prompts, memory, retrieval, storage, provider selection, "
    "generated output, runtime state, or apply Runtime Evolution."
)
COGNITIVE_PLANNING_DIMENSIONS = (
    "goal decomposition posture",
    "schedule-to-plan continuity",
    "dependency-aware sequencing",
    "governance checkpoint mapping",
    "HITL readiness",
)


class CognitivePlanStep(BaseModel):
    """One read-only Cognitive OS plan step."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

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
    plan_rank: int = Field(ge=1, le=6)
    dependency_depth: int = Field(ge=0, le=5)
    upstream_plan_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=1)
    downstream_plan_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=1)
    planning_dimensions: tuple[str, ...] = Field(min_length=5, max_length=5)
    scheduling_posture: CognitiveOSPosture
    planning_posture: CognitiveOSPosture
    execution_plan_authorized: Literal[False] = False
    plan_summary: str = Field(min_length=1, max_length=580)
    dependency_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _step_matches_sources_and_boundary(self) -> Self:
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
        if self.planning_dimensions != COGNITIVE_PLANNING_DIMENSIONS:
            raise ValueError("planning_dimensions must match V6.6 planner")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class CognitivePlannerPlan(BaseModel):
    """Read-only cognitive planner over scheduler slots."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cognitive_planner"] = "cognitive_planner"
    serialization_version: Literal["cognitive_planner.v1"] = (
        COGNITIVE_PLANNER_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=COGNITIVE_PLANNER_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    cognitive_scheduler_role: Literal["cognitive_scheduler"]
    cognitive_scheduler_serialization_version: Literal["cognitive_scheduler.v1"]
    emergent_creativity_layer_role: Literal["emergent_creativity_layer"]
    creative_identity_layer_role: Literal["creative_identity_layer"]
    creative_cognition_layer_role: Literal["creative_cognition_layer"]
    cognitive_governance_layer_role: Literal["cognitive_governance_layer"]
    meta_planning_layer_role: Literal["meta_planning_layer"]
    layer_order: tuple[CognitiveOSLayer, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_count: int = Field(ge=6, le=6)
    source_schedule_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_schedule_count: int = Field(ge=6, le=6)
    source_emergence_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_emergence_count: int = Field(ge=6, le=6)
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
    plan_steps: tuple[CognitivePlanStep, ...] = Field(min_length=6, max_length=6)
    plan_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    candidate_plan_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    review_required_plan_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    guarded_plan_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    plan_count: int = Field(ge=6, le=6)
    candidate_plan_count: int = Field(ge=0, le=6)
    review_required_plan_count: int = Field(ge=0, le=6)
    guarded_plan_count: int = Field(ge=0, le=6)
    max_dependency_depth: int = Field(ge=0, le=5)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    cognitive_planner_implemented: Literal[True] = True
    cognitive_scheduler_integrated: Literal[True] = True
    plan_step_contract_implemented: Literal[True] = True
    plan_dependency_traceability_implemented: Literal[True] = True
    plan_governance_contract_implemented: Literal[True] = True
    plan_explainability_contract_implemented: Literal[True] = True
    autonomous_planning_implemented: Literal[False] = False
    plan_execution_implemented: Literal[False] = False
    plan_mutation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    routing_mutation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    executed_plan_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_plan_ids: tuple[str, ...] = Field(default_factory=tuple)
    routed_plan_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _planner_matches_sources(self) -> Self:
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_count != len(self.capability_ids):
            raise ValueError("capability_count must match capability ids")
        if self.source_schedule_count != len(self.source_schedule_ids):
            raise ValueError("source_schedule_count must match schedule ids")
        if self.source_emergence_count != len(self.source_emergence_ids):
            raise ValueError("source_emergence_count must match emergence ids")
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
        if self.plan_ids != tuple(step.plan_id for step in self.plan_steps):
            raise ValueError("plan_ids must match steps")
        if self.plan_count != len(self.plan_steps):
            raise ValueError("plan_count must match steps")
        if len(set(self.plan_ids)) != len(self.plan_ids):
            raise ValueError("plan_ids must be unique")
        expected_ranks = tuple(range(1, len(self.plan_steps) + 1))
        if tuple(step.plan_rank for step in self.plan_steps) != expected_ranks:
            raise ValueError("plan ranks must match step order")
        if self.candidate_plan_ids != _plan_ids_for_posture(
            self.plan_steps,
            "candidate",
        ):
            raise ValueError("candidate_plan_ids must match steps")
        if self.review_required_plan_ids != _plan_ids_for_posture(
            self.plan_steps,
            "review_required",
        ):
            raise ValueError("review_required_plan_ids must match steps")
        if self.guarded_plan_ids != _plan_ids_for_posture(
            self.plan_steps,
            "guarded",
        ):
            raise ValueError("guarded_plan_ids must match steps")
        if self.candidate_plan_count != len(self.candidate_plan_ids):
            raise ValueError("candidate_plan_count must match ids")
        if self.review_required_plan_count != len(self.review_required_plan_ids):
            raise ValueError("review_required_plan_count must match ids")
        if self.guarded_plan_count != len(self.guarded_plan_ids):
            raise ValueError("guarded_plan_count must match ids")
        expected_max_depth = max(step.dependency_depth for step in self.plan_steps)
        if self.max_dependency_depth != expected_max_depth:
            raise ValueError("max_dependency_depth must match steps")

        declared_steps = {
            step.plan_id: index for index, step in enumerate(self.plan_steps)
        }
        declared_capabilities = set(self.capability_ids)
        declared_schedules = set(self.source_schedule_ids)
        declared_emergence = set(self.source_emergence_ids)
        declared_identities = set(self.source_identity_ids)
        declared_cognition = set(self.source_cognition_ids)
        declared_governance = set(self.source_governance_ids)
        declared_planning = set(self.source_planning_ids)
        declared_reasoning = set(self.source_reasoning_ids)
        declared_profiles = set(self.source_profile_ids)
        declared_states = set(self.source_state_ids)
        declared_agents = set(self.linked_agent_ids)
        for step in self.plan_steps:
            if step.capability_id not in declared_capabilities:
                raise ValueError("step capability_id must be declared")
            if step.schedule_id not in declared_schedules:
                raise ValueError("step schedule_id must be declared")
            if step.emergence_id not in declared_emergence:
                raise ValueError("step emergence_id must be declared")
            if step.identity_id not in declared_identities:
                raise ValueError("step identity_id must be declared")
            if step.cognition_id not in declared_cognition:
                raise ValueError("step cognition_id must be declared")
            if step.governance_id not in declared_governance:
                raise ValueError("step governance_id must be declared")
            if step.planning_id not in declared_planning:
                raise ValueError("step planning_id must be declared")
            if step.reasoning_id not in declared_reasoning:
                raise ValueError("step reasoning_id must be declared")
            if step.profile_id not in declared_profiles:
                raise ValueError("step profile_id must be declared")
            if step.state_id not in declared_states:
                raise ValueError("step state_id must be declared")
            if not set(step.linked_agent_ids).issubset(declared_agents):
                raise ValueError("step linked_agent_ids must be declared")
            for upstream_id in step.upstream_plan_ids:
                if upstream_id not in declared_steps:
                    raise ValueError("upstream_plan_ids must be known steps")
                if declared_steps[upstream_id] >= declared_steps[step.plan_id]:
                    raise ValueError("upstream plans must precede step")
            for downstream_id in step.downstream_plan_ids:
                if downstream_id not in declared_steps:
                    raise ValueError("downstream_plan_ids must be known steps")
                if declared_steps[downstream_id] <= declared_steps[step.plan_id]:
                    raise ValueError("downstream plans must follow step")
        if self.covered_roadmap_items != (COGNITIVE_PLANNER_ROADMAP_ITEM,):
            raise ValueError("covered_roadmap_items must be Task 18 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.executed_plan_ids,
                self.mutated_plan_ids,
                self.routed_plan_ids,
                self.emitted_hitl_request_ids,
            )
        ):
            raise ValueError(
                "plan execution, mutation, routing, and HITL ids must be empty",
            )
        if not all(step.advisory_only for step in self.plan_steps):
            raise ValueError("all cognitive plan steps must be advisory only")
        return self


def build_cognitive_planner(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    cognitive_scheduler: CognitiveSchedulerPlan | None = None,
) -> CognitivePlannerPlan:
    """Build read-only cognitive planner metadata."""

    scheduler = cognitive_scheduler or build_cognitive_scheduler(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    steps = _cognitive_plan_steps(scheduler)
    return CognitivePlannerPlan(
        route_name=scheduler.route_name,
        task_type=scheduler.task_type,
        execution_mode_ids=scheduler.execution_mode_ids,
        cognitive_scheduler_role=scheduler.role,
        cognitive_scheduler_serialization_version=scheduler.serialization_version,
        emergent_creativity_layer_role=scheduler.emergent_creativity_layer_role,
        creative_identity_layer_role=scheduler.creative_identity_layer_role,
        creative_cognition_layer_role=scheduler.creative_cognition_layer_role,
        cognitive_governance_layer_role=scheduler.cognitive_governance_layer_role,
        meta_planning_layer_role=scheduler.meta_planning_layer_role,
        layer_order=scheduler.layer_order,
        capabilities=scheduler.capabilities,
        capability_ids=scheduler.capability_ids,
        capability_count=scheduler.capability_count,
        source_schedule_ids=scheduler.schedule_ids,
        source_schedule_count=scheduler.schedule_count,
        source_emergence_ids=scheduler.source_emergence_ids,
        source_emergence_count=scheduler.source_emergence_count,
        source_identity_ids=scheduler.source_identity_ids,
        source_identity_count=scheduler.source_identity_count,
        source_cognition_ids=scheduler.source_cognition_ids,
        source_cognition_count=scheduler.source_cognition_count,
        source_governance_ids=scheduler.source_governance_ids,
        source_governance_count=scheduler.source_governance_count,
        source_planning_ids=scheduler.source_planning_ids,
        source_planning_count=scheduler.source_planning_count,
        source_reasoning_ids=scheduler.source_reasoning_ids,
        source_reasoning_count=scheduler.source_reasoning_count,
        source_profile_ids=scheduler.source_profile_ids,
        source_profile_count=scheduler.source_profile_count,
        source_state_ids=scheduler.source_state_ids,
        source_state_count=scheduler.source_state_count,
        plan_steps=steps,
        plan_ids=tuple(step.plan_id for step in steps),
        candidate_plan_ids=_plan_ids_for_posture(steps, "candidate"),
        review_required_plan_ids=_plan_ids_for_posture(steps, "review_required"),
        guarded_plan_ids=_plan_ids_for_posture(steps, "guarded"),
        plan_count=len(steps),
        candidate_plan_count=len(_plan_ids_for_posture(steps, "candidate")),
        review_required_plan_count=len(
            _plan_ids_for_posture(steps, "review_required")
        ),
        guarded_plan_count=len(_plan_ids_for_posture(steps, "guarded")),
        max_dependency_depth=max(step.dependency_depth for step in steps),
        linked_agent_ids=scheduler.linked_agent_ids,
        covered_roadmap_items=(COGNITIVE_PLANNER_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture=scheduler.graph_posture,
    )


def cognitive_plan_step_by_id(
    plan_id: str,
    planner: CognitivePlannerPlan | None = None,
) -> CognitivePlanStep | None:
    """Return one cognitive plan step without executing it."""

    source_planner = planner or build_cognitive_planner()
    for step in source_planner.plan_steps:
        if step.plan_id == plan_id:
            return step
    return None


def cognitive_plan_steps_for_layer(
    cognitive_layer: CognitiveOSLayer,
    planner: CognitivePlannerPlan | None = None,
) -> tuple[CognitivePlanStep, ...]:
    """Return cognitive plan steps for one Cognitive OS layer."""

    source_planner = planner or build_cognitive_planner()
    return tuple(
        step
        for step in source_planner.plan_steps
        if step.cognitive_layer == cognitive_layer
    )


def cognitive_plan_steps_for_agent(
    agent_id: str,
    planner: CognitivePlannerPlan | None = None,
) -> tuple[CognitivePlanStep, ...]:
    """Return cognitive plan steps linked to one agent."""

    source_planner = planner or build_cognitive_planner()
    return tuple(
        step for step in source_planner.plan_steps if agent_id in step.linked_agent_ids
    )


def cognitive_plan_steps_for_posture(
    posture: CognitiveOSPosture,
    planner: CognitivePlannerPlan | None = None,
) -> tuple[CognitivePlanStep, ...]:
    """Return cognitive plan steps by posture without applying plans."""

    source_planner = planner or build_cognitive_planner()
    return tuple(
        step
        for step in source_planner.plan_steps
        if step.planning_posture == posture
    )


def _cognitive_plan_steps(
    scheduler: CognitiveSchedulerPlan,
) -> tuple[CognitivePlanStep, ...]:
    plan_ids = tuple(
        f"cognitive_planner::{slot.capability_id}" for slot in scheduler.schedule_slots
    )
    steps: list[CognitivePlanStep] = []
    for index, slot in enumerate(scheduler.schedule_slots):
        upstream_ids = (plan_ids[index - 1],) if index else ()
        downstream_ids = (
            (plan_ids[index + 1],)
            if index < len(scheduler.schedule_slots) - 1
            else ()
        )
        steps.append(
            CognitivePlanStep(
                plan_id=plan_ids[index],
                schedule_id=slot.schedule_id,
                emergence_id=slot.emergence_id,
                identity_id=slot.identity_id,
                cognition_id=slot.cognition_id,
                governance_id=slot.governance_id,
                planning_id=slot.planning_id,
                reasoning_id=slot.reasoning_id,
                profile_id=slot.profile_id,
                state_id=slot.state_id,
                capability_id=slot.capability_id,
                capability_name=slot.capability_name,
                cognitive_layer=slot.cognitive_layer,
                linked_agent_ids=slot.linked_agent_ids,
                plan_rank=slot.schedule_rank,
                dependency_depth=slot.dependency_depth,
                upstream_plan_ids=upstream_ids,
                downstream_plan_ids=downstream_ids,
                planning_dimensions=COGNITIVE_PLANNING_DIMENSIONS,
                scheduling_posture=slot.scheduling_posture,
                planning_posture=slot.scheduling_posture,
                plan_summary=(
                    f"Read-only cognitive plan step for {slot.capability_name}; "
                    f"projects {slot.schedule_id} into goal decomposition, "
                    "dependency sequencing, governance checkpoint, explanation, "
                    "and HITL metadata without plan execution."
                ),
                dependency_contracts=(
                    "cognitive plan step follows cognitive schedule slot",
                    f"cognitive schedule slot:{slot.schedule_id}",
                    f"meta-planning projection:{slot.planning_id}",
                ),
                governance_contracts=(
                    "cognitive planner does not execute plans",
                    "cognitive planner does not create or mutate workflows",
                    "HITL required before any planner-driven behavior",
                ),
                explanation_contracts=(
                    "cognitive planner cites scheduler and planning sources",
                    "cognitive planner preserves capability and layer ownership",
                    "cognitive planner explains why no plan is applied",
                ),
                blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
            )
        )
    return tuple(steps)


def _plan_ids_for_posture(
    steps: tuple[CognitivePlanStep, ...],
    posture: CognitiveOSPosture,
) -> tuple[str, ...]:
    return tuple(step.plan_id for step in steps if step.planning_posture == posture)
