"""V6.6 Cognitive Router metadata."""

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
from creative_coding_assistant.orchestration.cognitive_planner import (
    CognitivePlannerPlan,
    build_cognitive_planner,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)

COGNITIVE_ROUTER_SERIALIZATION_VERSION = "cognitive_router.v1"
COGNITIVE_ROUTER_ROADMAP_ITEM = "Cognitive Router"
COGNITIVE_ROUTER_AUTHORITY_BOUNDARY = (
    "V6.6 Cognitive Router projects cognitive plan steps into read-only "
    "route-decision metadata for plan-to-route continuity, capability "
    "ownership routing, dependency-aware handoff, governance checkpoint "
    "routing, explanation continuity, and HITL readiness. It exposes router "
    "metadata only; it does not route requests, select providers or models, "
    "invoke agents, execute workflows, mutate workflows, mutate routing "
    "state, change prompts, memory, retrieval, storage, generated output, "
    "runtime state, or apply Runtime Evolution."
)
COGNITIVE_ROUTING_DIMENSIONS = (
    "plan-to-route continuity",
    "capability ownership routing",
    "dependency-aware handoff",
    "governance checkpoint routing",
    "HITL routing boundary",
)


class CognitiveRouteDecision(BaseModel):
    """One read-only Cognitive OS route decision."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

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
    route_rank: int = Field(ge=1, le=6)
    dependency_depth: int = Field(ge=0, le=5)
    upstream_route_decision_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=1,
    )
    downstream_route_decision_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=1,
    )
    routing_dimensions: tuple[str, ...] = Field(min_length=5, max_length=5)
    planning_posture: CognitiveOSPosture
    routing_posture: CognitiveOSPosture
    routing_application_authorized: Literal[False] = False
    route_summary: str = Field(min_length=1, max_length=600)
    dependency_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _decision_matches_sources_and_boundary(self) -> Self:
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
        if self.routing_dimensions != COGNITIVE_ROUTING_DIMENSIONS:
            raise ValueError("routing_dimensions must match V6.6 router")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class CognitiveRouterPlan(BaseModel):
    """Read-only cognitive router over planner steps."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cognitive_router"] = "cognitive_router"
    serialization_version: Literal["cognitive_router.v1"] = (
        COGNITIVE_ROUTER_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=COGNITIVE_ROUTER_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    cognitive_planner_role: Literal["cognitive_planner"]
    cognitive_planner_serialization_version: Literal["cognitive_planner.v1"]
    cognitive_scheduler_role: Literal["cognitive_scheduler"]
    emergent_creativity_layer_role: Literal["emergent_creativity_layer"]
    creative_identity_layer_role: Literal["creative_identity_layer"]
    creative_cognition_layer_role: Literal["creative_cognition_layer"]
    cognitive_governance_layer_role: Literal["cognitive_governance_layer"]
    meta_planning_layer_role: Literal["meta_planning_layer"]
    layer_order: tuple[CognitiveOSLayer, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_count: int = Field(ge=6, le=6)
    source_plan_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_plan_count: int = Field(ge=6, le=6)
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
    route_decisions: tuple[CognitiveRouteDecision, ...] = Field(
        min_length=6,
        max_length=6,
    )
    route_decision_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    candidate_route_decision_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    review_required_route_decision_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    guarded_route_decision_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    route_decision_count: int = Field(ge=6, le=6)
    candidate_route_decision_count: int = Field(ge=0, le=6)
    review_required_route_decision_count: int = Field(ge=0, le=6)
    guarded_route_decision_count: int = Field(ge=0, le=6)
    max_dependency_depth: int = Field(ge=0, le=5)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    cognitive_router_implemented: Literal[True] = True
    cognitive_planner_integrated: Literal[True] = True
    route_decision_contract_implemented: Literal[True] = True
    route_dependency_traceability_implemented: Literal[True] = True
    route_governance_contract_implemented: Literal[True] = True
    route_explainability_contract_implemented: Literal[True] = True
    request_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_routing_implemented: Literal[False] = False
    routing_mutation_implemented: Literal[False] = False
    plan_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    applied_route_decision_ids: tuple[str, ...] = Field(default_factory=tuple)
    executed_route_decision_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_route_decision_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _router_matches_sources(self) -> Self:
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_count != len(self.capability_ids):
            raise ValueError("capability_count must match capability ids")
        if self.source_plan_count != len(self.source_plan_ids):
            raise ValueError("source_plan_count must match plan ids")
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
        if self.route_decision_ids != tuple(
            decision.route_decision_id for decision in self.route_decisions
        ):
            raise ValueError("route_decision_ids must match decisions")
        if self.route_decision_count != len(self.route_decisions):
            raise ValueError("route_decision_count must match decisions")
        if len(set(self.route_decision_ids)) != len(self.route_decision_ids):
            raise ValueError("route_decision_ids must be unique")
        expected_ranks = tuple(range(1, len(self.route_decisions) + 1))
        if tuple(decision.route_rank for decision in self.route_decisions) != (
            expected_ranks
        ):
            raise ValueError("route ranks must match decision order")
        if self.candidate_route_decision_ids != _route_ids_for_posture(
            self.route_decisions,
            "candidate",
        ):
            raise ValueError("candidate_route_decision_ids must match decisions")
        if self.review_required_route_decision_ids != _route_ids_for_posture(
            self.route_decisions,
            "review_required",
        ):
            raise ValueError("review_required_route_decision_ids must match decisions")
        if self.guarded_route_decision_ids != _route_ids_for_posture(
            self.route_decisions,
            "guarded",
        ):
            raise ValueError("guarded_route_decision_ids must match decisions")
        if self.candidate_route_decision_count != len(
            self.candidate_route_decision_ids
        ):
            raise ValueError("candidate_route_decision_count must match ids")
        if self.review_required_route_decision_count != len(
            self.review_required_route_decision_ids
        ):
            raise ValueError("review_required_route_decision_count must match ids")
        if self.guarded_route_decision_count != len(self.guarded_route_decision_ids):
            raise ValueError("guarded_route_decision_count must match ids")
        expected_max_depth = max(
            decision.dependency_depth for decision in self.route_decisions
        )
        if self.max_dependency_depth != expected_max_depth:
            raise ValueError("max_dependency_depth must match decisions")

        declared_decisions = {
            decision.route_decision_id: index
            for index, decision in enumerate(self.route_decisions)
        }
        declared_capabilities = set(self.capability_ids)
        declared_plans = set(self.source_plan_ids)
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
        for decision in self.route_decisions:
            if decision.capability_id not in declared_capabilities:
                raise ValueError("decision capability_id must be declared")
            if decision.plan_id not in declared_plans:
                raise ValueError("decision plan_id must be declared")
            if decision.schedule_id not in declared_schedules:
                raise ValueError("decision schedule_id must be declared")
            if decision.emergence_id not in declared_emergence:
                raise ValueError("decision emergence_id must be declared")
            if decision.identity_id not in declared_identities:
                raise ValueError("decision identity_id must be declared")
            if decision.cognition_id not in declared_cognition:
                raise ValueError("decision cognition_id must be declared")
            if decision.governance_id not in declared_governance:
                raise ValueError("decision governance_id must be declared")
            if decision.planning_id not in declared_planning:
                raise ValueError("decision planning_id must be declared")
            if decision.reasoning_id not in declared_reasoning:
                raise ValueError("decision reasoning_id must be declared")
            if decision.profile_id not in declared_profiles:
                raise ValueError("decision profile_id must be declared")
            if decision.state_id not in declared_states:
                raise ValueError("decision state_id must be declared")
            if not set(decision.linked_agent_ids).issubset(declared_agents):
                raise ValueError("decision linked_agent_ids must be declared")
            for upstream_id in decision.upstream_route_decision_ids:
                if upstream_id not in declared_decisions:
                    raise ValueError(
                        "upstream_route_decision_ids must be known decisions"
                    )
                if (
                    declared_decisions[upstream_id]
                    >= declared_decisions[decision.route_decision_id]
                ):
                    raise ValueError("upstream routes must precede decision")
            for downstream_id in decision.downstream_route_decision_ids:
                if downstream_id not in declared_decisions:
                    raise ValueError(
                        "downstream_route_decision_ids must be known decisions"
                    )
                if (
                    declared_decisions[downstream_id]
                    <= declared_decisions[decision.route_decision_id]
                ):
                    raise ValueError("downstream routes must follow decision")
        if self.covered_roadmap_items != (COGNITIVE_ROUTER_ROADMAP_ITEM,):
            raise ValueError("covered_roadmap_items must be Task 19 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.applied_route_decision_ids,
                self.executed_route_decision_ids,
                self.mutated_route_decision_ids,
                self.emitted_hitl_request_ids,
            )
        ):
            raise ValueError(
                "route application, execution, mutation, and HITL ids must be empty",
            )
        if not all(decision.advisory_only for decision in self.route_decisions):
            raise ValueError("all cognitive route decisions must be advisory only")
        return self


def build_cognitive_router(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    cognitive_planner: CognitivePlannerPlan | None = None,
) -> CognitiveRouterPlan:
    """Build read-only cognitive router metadata."""

    planner = cognitive_planner or build_cognitive_planner(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    decisions = _cognitive_route_decisions(planner)
    return CognitiveRouterPlan(
        route_name=planner.route_name,
        task_type=planner.task_type,
        execution_mode_ids=planner.execution_mode_ids,
        cognitive_planner_role=planner.role,
        cognitive_planner_serialization_version=planner.serialization_version,
        cognitive_scheduler_role=planner.cognitive_scheduler_role,
        emergent_creativity_layer_role=planner.emergent_creativity_layer_role,
        creative_identity_layer_role=planner.creative_identity_layer_role,
        creative_cognition_layer_role=planner.creative_cognition_layer_role,
        cognitive_governance_layer_role=planner.cognitive_governance_layer_role,
        meta_planning_layer_role=planner.meta_planning_layer_role,
        layer_order=planner.layer_order,
        capabilities=planner.capabilities,
        capability_ids=planner.capability_ids,
        capability_count=planner.capability_count,
        source_plan_ids=planner.plan_ids,
        source_plan_count=planner.plan_count,
        source_schedule_ids=planner.source_schedule_ids,
        source_schedule_count=planner.source_schedule_count,
        source_emergence_ids=planner.source_emergence_ids,
        source_emergence_count=planner.source_emergence_count,
        source_identity_ids=planner.source_identity_ids,
        source_identity_count=planner.source_identity_count,
        source_cognition_ids=planner.source_cognition_ids,
        source_cognition_count=planner.source_cognition_count,
        source_governance_ids=planner.source_governance_ids,
        source_governance_count=planner.source_governance_count,
        source_planning_ids=planner.source_planning_ids,
        source_planning_count=planner.source_planning_count,
        source_reasoning_ids=planner.source_reasoning_ids,
        source_reasoning_count=planner.source_reasoning_count,
        source_profile_ids=planner.source_profile_ids,
        source_profile_count=planner.source_profile_count,
        source_state_ids=planner.source_state_ids,
        source_state_count=planner.source_state_count,
        route_decisions=decisions,
        route_decision_ids=tuple(decision.route_decision_id for decision in decisions),
        candidate_route_decision_ids=_route_ids_for_posture(
            decisions,
            "candidate",
        ),
        review_required_route_decision_ids=_route_ids_for_posture(
            decisions,
            "review_required",
        ),
        guarded_route_decision_ids=_route_ids_for_posture(decisions, "guarded"),
        route_decision_count=len(decisions),
        candidate_route_decision_count=len(
            _route_ids_for_posture(decisions, "candidate")
        ),
        review_required_route_decision_count=len(
            _route_ids_for_posture(decisions, "review_required")
        ),
        guarded_route_decision_count=len(_route_ids_for_posture(decisions, "guarded")),
        max_dependency_depth=max(decision.dependency_depth for decision in decisions),
        linked_agent_ids=planner.linked_agent_ids,
        covered_roadmap_items=(COGNITIVE_ROUTER_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture=planner.graph_posture,
    )


def cognitive_route_decision_by_id(
    route_decision_id: str,
    router: CognitiveRouterPlan | None = None,
) -> CognitiveRouteDecision | None:
    """Return one cognitive route decision without applying it."""

    source_router = router or build_cognitive_router()
    for decision in source_router.route_decisions:
        if decision.route_decision_id == route_decision_id:
            return decision
    return None


def cognitive_route_decisions_for_layer(
    cognitive_layer: CognitiveOSLayer,
    router: CognitiveRouterPlan | None = None,
) -> tuple[CognitiveRouteDecision, ...]:
    """Return cognitive route decisions for one Cognitive OS layer."""

    source_router = router or build_cognitive_router()
    return tuple(
        decision
        for decision in source_router.route_decisions
        if decision.cognitive_layer == cognitive_layer
    )


def cognitive_route_decisions_for_agent(
    agent_id: str,
    router: CognitiveRouterPlan | None = None,
) -> tuple[CognitiveRouteDecision, ...]:
    """Return cognitive route decisions linked to one agent."""

    source_router = router or build_cognitive_router()
    return tuple(
        decision
        for decision in source_router.route_decisions
        if agent_id in decision.linked_agent_ids
    )


def cognitive_route_decisions_for_posture(
    posture: CognitiveOSPosture,
    router: CognitiveRouterPlan | None = None,
) -> tuple[CognitiveRouteDecision, ...]:
    """Return cognitive route decisions by posture without routing."""

    source_router = router or build_cognitive_router()
    return tuple(
        decision
        for decision in source_router.route_decisions
        if decision.routing_posture == posture
    )


def _cognitive_route_decisions(
    planner: CognitivePlannerPlan,
) -> tuple[CognitiveRouteDecision, ...]:
    route_ids = tuple(
        f"cognitive_router::{step.capability_id}" for step in planner.plan_steps
    )
    decisions: list[CognitiveRouteDecision] = []
    for index, step in enumerate(planner.plan_steps):
        upstream_ids = (route_ids[index - 1],) if index else ()
        downstream_ids = (
            (route_ids[index + 1],) if index < len(planner.plan_steps) - 1 else ()
        )
        decisions.append(
            CognitiveRouteDecision(
                route_decision_id=route_ids[index],
                plan_id=step.plan_id,
                schedule_id=step.schedule_id,
                emergence_id=step.emergence_id,
                identity_id=step.identity_id,
                cognition_id=step.cognition_id,
                governance_id=step.governance_id,
                planning_id=step.planning_id,
                reasoning_id=step.reasoning_id,
                profile_id=step.profile_id,
                state_id=step.state_id,
                capability_id=step.capability_id,
                capability_name=step.capability_name,
                cognitive_layer=step.cognitive_layer,
                linked_agent_ids=step.linked_agent_ids,
                route_rank=step.plan_rank,
                dependency_depth=step.dependency_depth,
                upstream_route_decision_ids=upstream_ids,
                downstream_route_decision_ids=downstream_ids,
                routing_dimensions=COGNITIVE_ROUTING_DIMENSIONS,
                planning_posture=step.planning_posture,
                routing_posture=step.planning_posture,
                route_summary=(
                    f"Read-only cognitive route decision for "
                    f"{step.capability_name}; projects {step.plan_id} into "
                    "capability ownership, dependency handoff, governance, "
                    "explanation, and HITL route metadata without routing."
                ),
                dependency_contracts=(
                    "cognitive route decision follows cognitive plan step",
                    f"cognitive plan step:{step.plan_id}",
                    f"cognitive schedule slot:{step.schedule_id}",
                ),
                governance_contracts=(
                    "cognitive router does not route requests",
                    "cognitive router does not select providers or models",
                    "HITL required before any router-driven behavior",
                ),
                explanation_contracts=(
                    "cognitive router cites planner and scheduler sources",
                    "cognitive router preserves capability and layer ownership",
                    "cognitive router explains why no route is applied",
                ),
                blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
            )
        )
    return tuple(decisions)


def _route_ids_for_posture(
    decisions: tuple[CognitiveRouteDecision, ...],
    posture: CognitiveOSPosture,
) -> tuple[str, ...]:
    return tuple(
        decision.route_decision_id
        for decision in decisions
        if decision.routing_posture == posture
    )
