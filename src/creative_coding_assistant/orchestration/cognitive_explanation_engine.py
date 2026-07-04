"""V6.6 Cognitive Explanation Engine metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.cognitive_blackboard import (
    CognitiveBlackboardPlan,
    build_cognitive_blackboard,
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

COGNITIVE_EXPLANATION_ENGINE_SERIALIZATION_VERSION = "cognitive_explanation_engine.v1"
COGNITIVE_EXPLANATION_ENGINE_ROADMAP_ITEM = "Cognitive Explanation Engine"
COGNITIVE_EXPLANATION_ENGINE_AUTHORITY_BOUNDARY = (
    "V6.6 Cognitive Explanation Engine projects cognitive blackboard entries "
    "into read-only explanation traces for graph source traceability, "
    "planning rationale, routing rationale, blackboard context rationale, "
    "governance checkpoint rationale, and HITL readiness. It exposes "
    "explanation metadata only; it does not generate explanation text, write "
    "audit records, mutate prompts, memory, retrieval, storage, provider "
    "selection, generated output, runtime state, or apply Runtime Evolution."
)
COGNITIVE_EXPLANATION_DIMENSIONS = (
    "graph source traceability",
    "planning rationale",
    "routing rationale",
    "blackboard context rationale",
    "HITL explanation boundary",
)


class CognitiveExplanationTrace(BaseModel):
    """One read-only Cognitive OS explanation trace."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

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
    explanation_rank: int = Field(ge=1, le=6)
    dependency_depth: int = Field(ge=0, le=5)
    explanation_dimensions: tuple[str, ...] = Field(min_length=5, max_length=5)
    blackboard_posture: CognitiveOSPosture
    explanation_posture: CognitiveOSPosture
    explanation_generation_authorized: Literal[False] = False
    source_trace_ids: tuple[str, ...] = Field(min_length=8, max_length=12)
    explanation_summary: str = Field(min_length=1, max_length=720)
    dependency_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _trace_matches_sources_and_boundary(self) -> Self:
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
        if self.explanation_dimensions != COGNITIVE_EXPLANATION_DIMENSIONS:
            raise ValueError("explanation_dimensions must match V6.6 explanation")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class CognitiveExplanationEnginePlan(BaseModel):
    """Read-only cognitive explanation engine over blackboard entries."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cognitive_explanation_engine"] = "cognitive_explanation_engine"
    serialization_version: Literal["cognitive_explanation_engine.v1"] = (
        COGNITIVE_EXPLANATION_ENGINE_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=COGNITIVE_EXPLANATION_ENGINE_AUTHORITY_BOUNDARY,
        max_length=2300,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    cognitive_blackboard_role: Literal["cognitive_blackboard"]
    cognitive_blackboard_serialization_version: Literal["cognitive_blackboard.v1"]
    cognitive_router_role: Literal["cognitive_router"]
    cognitive_planner_role: Literal["cognitive_planner"]
    cognitive_scheduler_role: Literal["cognitive_scheduler"]
    layer_order: tuple[CognitiveOSLayer, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_count: int = Field(ge=6, le=6)
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
    explanation_traces: tuple[CognitiveExplanationTrace, ...] = Field(
        min_length=6,
        max_length=6,
    )
    explanation_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    candidate_explanation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    review_required_explanation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    guarded_explanation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    explanation_count: int = Field(ge=6, le=6)
    candidate_explanation_count: int = Field(ge=0, le=6)
    review_required_explanation_count: int = Field(ge=0, le=6)
    guarded_explanation_count: int = Field(ge=0, le=6)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    cognitive_explanation_engine_implemented: Literal[True] = True
    cognitive_blackboard_integrated: Literal[True] = True
    explanation_trace_contract_implemented: Literal[True] = True
    explanation_dependency_traceability_implemented: Literal[True] = True
    explanation_governance_contract_implemented: Literal[True] = True
    explanation_hitl_contract_implemented: Literal[True] = True
    explanation_generation_implemented: Literal[False] = False
    explanation_text_generation_implemented: Literal[False] = False
    explanation_audit_record_write_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    generated_explanation_ids: tuple[str, ...] = Field(default_factory=tuple)
    written_explanation_record_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_explanation_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _engine_matches_sources(self) -> Self:
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_count != len(self.capability_ids):
            raise ValueError("capability_count must match capability ids")
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
        if self.explanation_ids != tuple(
            trace.explanation_id for trace in self.explanation_traces
        ):
            raise ValueError("explanation_ids must match traces")
        if self.explanation_count != len(self.explanation_traces):
            raise ValueError("explanation_count must match traces")
        if len(set(self.explanation_ids)) != len(self.explanation_ids):
            raise ValueError("explanation_ids must be unique")
        if self.candidate_explanation_ids != _explanation_ids_for_posture(
            self.explanation_traces,
            "candidate",
        ):
            raise ValueError("candidate_explanation_ids must match traces")
        if self.review_required_explanation_ids != _explanation_ids_for_posture(
            self.explanation_traces,
            "review_required",
        ):
            raise ValueError("review_required_explanation_ids must match traces")
        if self.guarded_explanation_ids != _explanation_ids_for_posture(
            self.explanation_traces,
            "guarded",
        ):
            raise ValueError("guarded_explanation_ids must match traces")
        if self.candidate_explanation_count != len(self.candidate_explanation_ids):
            raise ValueError("candidate_explanation_count must match ids")
        if self.review_required_explanation_count != len(
            self.review_required_explanation_ids
        ):
            raise ValueError("review_required_explanation_count must match ids")
        if self.guarded_explanation_count != len(self.guarded_explanation_ids):
            raise ValueError("guarded_explanation_count must match ids")

        declared_capabilities = set(self.capability_ids)
        declared_blackboard = set(self.source_blackboard_entry_ids)
        declared_routes = set(self.source_route_decision_ids)
        declared_plans = set(self.source_plan_ids)
        declared_schedules = set(self.source_schedule_ids)
        declared_emergence = set(self.source_emergence_ids)
        declared_agents = set(self.linked_agent_ids)
        for trace in self.explanation_traces:
            if trace.capability_id not in declared_capabilities:
                raise ValueError("trace capability_id must be declared")
            if trace.blackboard_entry_id not in declared_blackboard:
                raise ValueError("trace blackboard_entry_id must be declared")
            if trace.route_decision_id not in declared_routes:
                raise ValueError("trace route_decision_id must be declared")
            if trace.plan_id not in declared_plans:
                raise ValueError("trace plan_id must be declared")
            if trace.schedule_id not in declared_schedules:
                raise ValueError("trace schedule_id must be declared")
            if trace.emergence_id not in declared_emergence:
                raise ValueError("trace emergence_id must be declared")
            if not set(trace.linked_agent_ids).issubset(declared_agents):
                raise ValueError("trace linked_agent_ids must be declared")
        if self.covered_roadmap_items != (COGNITIVE_EXPLANATION_ENGINE_ROADMAP_ITEM,):
            raise ValueError("covered_roadmap_items must be Task 21 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.generated_explanation_ids,
                self.written_explanation_record_ids,
                self.mutated_explanation_ids,
                self.emitted_hitl_request_ids,
            )
        ):
            raise ValueError(
                "explanation generation, writes, mutation, and HITL ids must be empty",
            )
        if not all(trace.advisory_only for trace in self.explanation_traces):
            raise ValueError("all cognitive explanation traces must be advisory only")
        return self


def build_cognitive_explanation_engine(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    cognitive_blackboard: CognitiveBlackboardPlan | None = None,
) -> CognitiveExplanationEnginePlan:
    """Build read-only cognitive explanation metadata."""

    blackboard = cognitive_blackboard or build_cognitive_blackboard(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    traces = _cognitive_explanation_traces(blackboard)
    return CognitiveExplanationEnginePlan(
        route_name=blackboard.route_name,
        task_type=blackboard.task_type,
        execution_mode_ids=blackboard.execution_mode_ids,
        cognitive_blackboard_role=blackboard.role,
        cognitive_blackboard_serialization_version=blackboard.serialization_version,
        cognitive_router_role=blackboard.cognitive_router_role,
        cognitive_planner_role=blackboard.cognitive_planner_role,
        cognitive_scheduler_role=blackboard.cognitive_scheduler_role,
        layer_order=blackboard.layer_order,
        capabilities=blackboard.capabilities,
        capability_ids=blackboard.capability_ids,
        capability_count=blackboard.capability_count,
        source_blackboard_entry_ids=blackboard.blackboard_entry_ids,
        source_blackboard_entry_count=blackboard.blackboard_entry_count,
        source_route_decision_ids=blackboard.source_route_decision_ids,
        source_route_decision_count=blackboard.source_route_decision_count,
        source_plan_ids=blackboard.source_plan_ids,
        source_plan_count=blackboard.source_plan_count,
        source_schedule_ids=blackboard.source_schedule_ids,
        source_schedule_count=blackboard.source_schedule_count,
        source_emergence_ids=blackboard.source_emergence_ids,
        source_emergence_count=blackboard.source_emergence_count,
        explanation_traces=traces,
        explanation_ids=tuple(trace.explanation_id for trace in traces),
        candidate_explanation_ids=_explanation_ids_for_posture(
            traces,
            "candidate",
        ),
        review_required_explanation_ids=_explanation_ids_for_posture(
            traces,
            "review_required",
        ),
        guarded_explanation_ids=_explanation_ids_for_posture(traces, "guarded"),
        explanation_count=len(traces),
        candidate_explanation_count=len(
            _explanation_ids_for_posture(traces, "candidate")
        ),
        review_required_explanation_count=len(
            _explanation_ids_for_posture(traces, "review_required")
        ),
        guarded_explanation_count=len(_explanation_ids_for_posture(traces, "guarded")),
        linked_agent_ids=blackboard.linked_agent_ids,
        covered_roadmap_items=(COGNITIVE_EXPLANATION_ENGINE_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture=blackboard.graph_posture,
    )


def cognitive_explanation_trace_by_id(
    explanation_id: str,
    engine: CognitiveExplanationEnginePlan | None = None,
) -> CognitiveExplanationTrace | None:
    """Return one cognitive explanation trace without generation."""

    source_engine = engine or build_cognitive_explanation_engine()
    for trace in source_engine.explanation_traces:
        if trace.explanation_id == explanation_id:
            return trace
    return None


def cognitive_explanation_traces_for_layer(
    cognitive_layer: CognitiveOSLayer,
    engine: CognitiveExplanationEnginePlan | None = None,
) -> tuple[CognitiveExplanationTrace, ...]:
    """Return explanation traces for one Cognitive OS layer."""

    source_engine = engine or build_cognitive_explanation_engine()
    return tuple(
        trace
        for trace in source_engine.explanation_traces
        if trace.cognitive_layer == cognitive_layer
    )


def cognitive_explanation_traces_for_agent(
    agent_id: str,
    engine: CognitiveExplanationEnginePlan | None = None,
) -> tuple[CognitiveExplanationTrace, ...]:
    """Return explanation traces linked to one agent."""

    source_engine = engine or build_cognitive_explanation_engine()
    return tuple(
        trace
        for trace in source_engine.explanation_traces
        if agent_id in trace.linked_agent_ids
    )


def cognitive_explanation_traces_for_posture(
    posture: CognitiveOSPosture,
    engine: CognitiveExplanationEnginePlan | None = None,
) -> tuple[CognitiveExplanationTrace, ...]:
    """Return explanation traces by posture without generating output."""

    source_engine = engine or build_cognitive_explanation_engine()
    return tuple(
        trace
        for trace in source_engine.explanation_traces
        if trace.explanation_posture == posture
    )


def _cognitive_explanation_traces(
    blackboard: CognitiveBlackboardPlan,
) -> tuple[CognitiveExplanationTrace, ...]:
    return tuple(
        CognitiveExplanationTrace(
            explanation_id=f"cognitive_explanation::{entry.capability_id}",
            blackboard_entry_id=entry.blackboard_entry_id,
            route_decision_id=entry.route_decision_id,
            plan_id=entry.plan_id,
            schedule_id=entry.schedule_id,
            emergence_id=entry.emergence_id,
            identity_id=entry.identity_id,
            cognition_id=entry.cognition_id,
            governance_id=entry.governance_id,
            planning_id=entry.planning_id,
            reasoning_id=entry.reasoning_id,
            profile_id=entry.profile_id,
            state_id=entry.state_id,
            capability_id=entry.capability_id,
            capability_name=entry.capability_name,
            cognitive_layer=entry.cognitive_layer,
            linked_agent_ids=entry.linked_agent_ids,
            explanation_rank=entry.blackboard_rank,
            dependency_depth=entry.dependency_depth,
            explanation_dimensions=COGNITIVE_EXPLANATION_DIMENSIONS,
            blackboard_posture=entry.blackboard_posture,
            explanation_posture=entry.blackboard_posture,
            source_trace_ids=(
                entry.blackboard_entry_id,
                entry.route_decision_id,
                entry.plan_id,
                entry.schedule_id,
                entry.emergence_id,
                entry.identity_id,
                entry.cognition_id,
                entry.governance_id,
                entry.planning_id,
                entry.reasoning_id,
                entry.profile_id,
                entry.state_id,
            ),
            explanation_summary=(
                f"Read-only cognitive explanation trace for "
                f"{entry.capability_name}; cites blackboard, routing, "
                "planning, scheduling, emergence, identity, cognition, "
                "governance, reasoning, profile, and state metadata without "
                "generating explanation text."
            ),
            dependency_contracts=(
                "cognitive explanation follows cognitive blackboard entry",
                f"cognitive blackboard entry:{entry.blackboard_entry_id}",
                f"cognitive route decision:{entry.route_decision_id}",
            ),
            governance_contracts=(
                "cognitive explanation engine does not generate explanations",
                "cognitive explanation engine does not write audit records",
                "HITL required before any explanation-driven behavior",
            ),
            explanation_contracts=(
                "cognitive explanation cites the full cognitive source chain",
                "cognitive explanation preserves capability and agent ownership",
                "cognitive explanation explains why no text is generated",
            ),
            blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        for entry in blackboard.blackboard_entries
    )


def _explanation_ids_for_posture(
    traces: tuple[CognitiveExplanationTrace, ...],
    posture: CognitiveOSPosture,
) -> tuple[str, ...]:
    return tuple(
        trace.explanation_id for trace in traces if trace.explanation_posture == posture
    )
