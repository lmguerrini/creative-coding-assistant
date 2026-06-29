"""V5.1 advisory workflow pruning planner."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.execution_graph_analyzer import (
    ExecutionGraphAnalysis,
    analyze_assistant_execution_graph,
)
from creative_coding_assistant.orchestration.workflow_complexity_analyzer import (
    WorkflowComplexityAnalysis,
    WorkflowComplexityFactor,
    analyze_workflow_complexity,
)
from creative_coding_assistant.orchestration.workflow_cost_analyzer import (
    WorkflowCostAnalysis,
    WorkflowCostComponent,
    analyze_workflow_cost,
)

WorkflowPruningCandidateKind = Literal[
    "workflow_node",
    "retry_reserve",
    "failure_reserve",
    "complexity_factor",
]
WorkflowPruningStatus = Literal["prunable", "retain", "review"]
WorkflowPruningPriority = Literal["low", "medium", "high"]
WorkflowPruningPressure = Literal["low", "medium", "high"]

WORKFLOW_PRUNING_CANDIDATE_SERIALIZATION_VERSION = "workflow_pruning_candidate.v1"
WORKFLOW_PRUNING_PLAN_SERIALIZATION_VERSION = "workflow_pruning_plan.v1"
WORKFLOW_PRUNING_AUTHORITY_BOUNDARY = (
    "Workflow pruning planning derives advisory pruning candidates from static "
    "workflow graph, bounded cost, and structural complexity metadata only; it "
    "does not remove workflow nodes, mutate graph order, choose execution "
    "paths, select strategies, route providers or models, control workflow "
    "execution, trigger retries, mutate prompts, write storage, or modify "
    "generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "workflow_node_removal",
    "workflow_graph_mutation",
    "workflow_order_change",
    "execution_path_selection",
    "strategy_selection",
    "provider_or_model_routing",
    "workflow_control",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)
_COMPLEXITY_FACTOR_KINDS = {"branching", "retry", "cost_pressure"}


class WorkflowPruningCandidate(BaseModel):
    """One advisory workflow pruning candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=180)
    candidate_kind: WorkflowPruningCandidateKind
    source_id: str = Field(min_length=1, max_length=160)
    source_serialization_version: str = Field(min_length=1, max_length=80)
    status: WorkflowPruningStatus
    priority: WorkflowPruningPriority
    estimated_token_savings: int = Field(ge=0, le=240_000)
    retained_token_cost: int = Field(ge=0, le=240_000)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    workflow_pruning_implemented: Literal[True] = True
    workflow_node_removal_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_order_mutation_implemented: Literal[False] = False
    execution_path_selection_implemented: Literal[False] = False
    strategy_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["workflow_pruning_candidate.v1"] = (
        WORKFLOW_PRUNING_CANDIDATE_SERIALIZATION_VERSION
    )
    planning_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_status(self) -> Self:
        if self.status == "prunable":
            if self.estimated_token_savings <= 0:
                raise ValueError("prunable candidates require token savings")
            if self.retained_token_cost != 0:
                raise ValueError("prunable candidates cannot retain token cost")
        if self.status == "retain" and self.estimated_token_savings != 0:
            raise ValueError("retained candidates cannot report token savings")
        if self.status == "review":
            if self.estimated_token_savings != 0 or self.retained_token_cost != 0:
                raise ValueError("review candidates cannot report token totals")
        return self


class WorkflowPruningPlan(BaseModel):
    """Bounded V5.1 advisory workflow pruning plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["workflow_pruning_planner"] = "workflow_pruning_planner"
    serialization_version: Literal["workflow_pruning_plan.v1"] = (
        WORKFLOW_PRUNING_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=WORKFLOW_PRUNING_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    source_graph_serialization_version: str = Field(min_length=1, max_length=80)
    source_cost_serialization_version: str = Field(min_length=1, max_length=80)
    source_complexity_serialization_version: str = Field(min_length=1, max_length=80)
    candidates: tuple[WorkflowPruningCandidate, ...] = Field(
        min_length=1,
        max_length=80,
    )
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=80)
    prunable_candidate_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=80)
    retained_candidate_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=80)
    review_candidate_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=80)
    candidate_count: int = Field(ge=1, le=80)
    prunable_candidate_count: int = Field(ge=0, le=80)
    retained_candidate_count: int = Field(ge=0, le=80)
    review_candidate_count: int = Field(ge=0, le=80)
    estimated_token_savings: int = Field(ge=0, le=240_000)
    retained_token_cost: int = Field(ge=0, le=240_000)
    worst_case_token_estimate: int = Field(ge=1, le=240_000)
    savings_ratio: float = Field(ge=0, le=1)
    pruning_pressure: WorkflowPruningPressure
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    workflow_pruning_implemented: Literal[True] = True
    workflow_node_removal_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_order_mutation_implemented: Literal[False] = False
    execution_path_selection_implemented: Literal[False] = False
    strategy_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    planning_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_candidates(self) -> Self:
        derived_candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if len(set(derived_candidate_ids)) != len(derived_candidate_ids):
            raise ValueError("candidate_ids must be unique")
        if self.candidate_ids != derived_candidate_ids:
            raise ValueError("candidate_ids must match candidates")
        if self.candidate_count != len(self.candidates):
            raise ValueError("candidate_count must match candidates")

        prunable_ids = _candidate_ids_for_status(self.candidates, "prunable")
        retained_ids = _candidate_ids_for_status(self.candidates, "retain")
        review_ids = _candidate_ids_for_status(self.candidates, "review")
        if self.prunable_candidate_ids != prunable_ids:
            raise ValueError("prunable_candidate_ids must match candidates")
        if self.retained_candidate_ids != retained_ids:
            raise ValueError("retained_candidate_ids must match candidates")
        if self.review_candidate_ids != review_ids:
            raise ValueError("review_candidate_ids must match candidates")
        if self.prunable_candidate_count != len(prunable_ids):
            raise ValueError("prunable_candidate_count must match candidates")
        if self.retained_candidate_count != len(retained_ids):
            raise ValueError("retained_candidate_count must match candidates")
        if self.review_candidate_count != len(review_ids):
            raise ValueError("review_candidate_count must match candidates")

        savings = sum(candidate.estimated_token_savings for candidate in self.candidates)
        retained = sum(candidate.retained_token_cost for candidate in self.candidates)
        if self.estimated_token_savings != savings:
            raise ValueError("estimated_token_savings must match candidates")
        if self.retained_token_cost != retained:
            raise ValueError("retained_token_cost must match candidates")
        expected_ratio = savings / self.worst_case_token_estimate
        if abs(self.savings_ratio - expected_ratio) > 0.0001:
            raise ValueError("savings_ratio must match worst case estimate")
        if self.pruning_pressure != _pruning_pressure(
            savings_ratio=self.savings_ratio,
            prunable_count=self.prunable_candidate_count,
        ):
            raise ValueError("pruning_pressure must match candidate pressure")
        return self


def plan_workflow_pruning(
    *,
    execution_graph: ExecutionGraphAnalysis | None = None,
    cost_analysis: WorkflowCostAnalysis | None = None,
    complexity_analysis: WorkflowComplexityAnalysis | None = None,
) -> WorkflowPruningPlan:
    """Plan advisory workflow pruning without graph or execution mutation."""

    graph = execution_graph or analyze_assistant_execution_graph()
    costs = cost_analysis or analyze_workflow_cost(execution_graph=graph)
    complexity = complexity_analysis or analyze_workflow_complexity(
        execution_graph=graph,
        cost_analysis=costs,
    )
    candidates = _candidates(graph=graph, costs=costs, complexity=complexity)
    savings = sum(candidate.estimated_token_savings for candidate in candidates)
    retained = sum(candidate.retained_token_cost for candidate in candidates)
    ratio = savings / costs.worst_case_token_estimate
    prunable_ids = _candidate_ids_for_status(candidates, "prunable")
    retained_ids = _candidate_ids_for_status(candidates, "retain")
    review_ids = _candidate_ids_for_status(candidates, "review")
    pressure = _pruning_pressure(
        savings_ratio=ratio,
        prunable_count=len(prunable_ids),
    )

    return WorkflowPruningPlan(
        source_graph_serialization_version=graph.serialization_version,
        source_cost_serialization_version=costs.serialization_version,
        source_complexity_serialization_version=complexity.serialization_version,
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        prunable_candidate_ids=prunable_ids,
        retained_candidate_ids=retained_ids,
        review_candidate_ids=review_ids,
        candidate_count=len(candidates),
        prunable_candidate_count=len(prunable_ids),
        retained_candidate_count=len(retained_ids),
        review_candidate_count=len(review_ids),
        estimated_token_savings=savings,
        retained_token_cost=retained,
        worst_case_token_estimate=costs.worst_case_token_estimate,
        savings_ratio=ratio,
        pruning_pressure=pressure,
        advisory_actions=_plan_actions(savings, pressure),
    )


def workflow_pruning_candidate_by_id(
    candidate_id: str,
    plan: WorkflowPruningPlan | None = None,
) -> WorkflowPruningCandidate | None:
    """Return one pruning candidate without changing workflow topology."""

    source_plan = plan or plan_workflow_pruning()
    for candidate in source_plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def workflow_pruning_candidates_for_status(
    status: WorkflowPruningStatus,
    plan: WorkflowPruningPlan | None = None,
) -> tuple[WorkflowPruningCandidate, ...]:
    """Return pruning candidates by status without applying pruning."""

    source_plan = plan or plan_workflow_pruning()
    return tuple(candidate for candidate in source_plan.candidates if candidate.status == status)


def _candidates(
    *,
    graph: ExecutionGraphAnalysis,
    costs: WorkflowCostAnalysis,
    complexity: WorkflowComplexityAnalysis,
) -> tuple[WorkflowPruningCandidate, ...]:
    workflow_node_components = {
        component.source_id: component
        for component in costs.components
        if component.component_kind == "workflow_node"
    }
    candidates = [
        _workflow_node_candidate(
            node_id=node_id,
            graph=graph,
            component=workflow_node_components[node_id],
            costs=costs,
        )
        for node_id in graph.node_order
    ]
    candidates.extend(
        _reserve_candidate(component, costs)
        for component in costs.components
        if component.component_kind in {"retry_reserve", "failure_reserve"}
    )
    candidates.extend(
        _complexity_candidate(factor)
        for factor in complexity.factors
        if factor.factor_kind in _COMPLEXITY_FACTOR_KINDS
    )
    return tuple(candidates)


def _workflow_node_candidate(
    *,
    node_id: str,
    graph: ExecutionGraphAnalysis,
    component: WorkflowCostComponent,
    costs: WorkflowCostAnalysis,
) -> WorkflowPruningCandidate:
    if node_id in graph.critical_path_node_ids or node_id == "failure":
        status: WorkflowPruningStatus = "retain"
        savings = 0
        retained = component.estimated_token_cost
    else:
        status = "review"
        savings = 0
        retained = 0
    retry_path = node_id in costs.retry_path_node_ids

    return WorkflowPruningCandidate(
        candidate_id=f"workflow_pruning::node::{node_id}",
        candidate_kind="workflow_node",
        source_id=node_id,
        source_serialization_version=component.serialization_version,
        status=status,
        priority=_priority_for_component(component),
        estimated_token_savings=savings,
        retained_token_cost=retained,
        evidence=(
            f"node:{node_id}",
            f"critical_path:{node_id in graph.critical_path_node_ids}",
            f"retry_path:{retry_path}",
            f"estimated_tokens:{component.estimated_token_cost}",
        ),
        advisory_actions=_node_actions(status, retry_path),
    )


def _reserve_candidate(
    component: WorkflowCostComponent,
    costs: WorkflowCostAnalysis,
) -> WorkflowPruningCandidate:
    if component.component_kind == "retry_reserve":
        return WorkflowPruningCandidate(
            candidate_id="workflow_pruning::reserve::retry_path",
            candidate_kind="retry_reserve",
            source_id=component.source_id,
            source_serialization_version=component.serialization_version,
            status="prunable",
            priority="high" if costs.estimated_cost_pressure != "low" else "medium",
            estimated_token_savings=component.estimated_token_cost,
            retained_token_cost=0,
            evidence=(
                f"retry_iterations:{costs.retry_iteration_count}",
                f"retry_tokens:{costs.retry_token_reserve}",
                f"cost_pressure:{costs.estimated_cost_pressure}",
            ),
            advisory_actions=(
                "Expose retry reserve as a bounded pruning opportunity only.",
                "Defer actual retry behavior to explicit strategy selection.",
            ),
        )

    return WorkflowPruningCandidate(
        candidate_id="workflow_pruning::reserve::failure_path",
        candidate_kind="failure_reserve",
        source_id=component.source_id,
        source_serialization_version=component.serialization_version,
        status="retain",
        priority="high",
        estimated_token_savings=0,
        retained_token_cost=component.estimated_token_cost,
        evidence=(
            "failure_path_preserved",
            f"failure_tokens:{costs.failure_path_token_reserve}",
        ),
        advisory_actions=("Retain normalized failure path budget.",),
    )


def _complexity_candidate(
    factor: WorkflowComplexityFactor,
) -> WorkflowPruningCandidate:
    return WorkflowPruningCandidate(
        candidate_id=f"workflow_pruning::factor::{factor.factor_kind}",
        candidate_kind="complexity_factor",
        source_id=factor.factor_id,
        source_serialization_version=factor.serialization_version,
        status="review",
        priority=factor.level,
        estimated_token_savings=0,
        retained_token_cost=0,
        evidence=(
            f"factor:{factor.factor_kind}",
            f"score:{factor.score}",
            f"level:{factor.level}",
            *factor.evidence[:5],
        ),
        advisory_actions=(
            "Use complexity pressure as pruning review metadata only.",
        ),
    )


def _candidate_ids_for_status(
    candidates: tuple[WorkflowPruningCandidate, ...],
    status: WorkflowPruningStatus,
) -> tuple[str, ...]:
    return tuple(
        candidate.candidate_id for candidate in candidates if candidate.status == status
    )


def _priority_for_component(
    component: WorkflowCostComponent,
) -> WorkflowPruningPriority:
    if component.relative_cost == "high":
        return "high"
    if component.relative_cost == "medium":
        return "medium"
    return "low"


def _node_actions(
    status: WorkflowPruningStatus,
    retry_path: bool,
) -> tuple[str, ...]:
    if status == "retain":
        return ("Retain workflow node; preserve graph order.",)
    if retry_path:
        return ("Review retry-path node through reserve metadata only.",)
    return ("Review non-critical workflow node as advisory metadata only.",)


def _pruning_pressure(
    *,
    savings_ratio: float,
    prunable_count: int,
) -> WorkflowPruningPressure:
    if prunable_count > 0 and savings_ratio >= 0.35:
        return "high"
    if prunable_count > 0 or savings_ratio >= 0.15:
        return "medium"
    return "low"


def _plan_actions(
    estimated_token_savings: int,
    pressure: WorkflowPruningPressure,
) -> tuple[str, ...]:
    actions = [
        "Expose workflow pruning opportunities as advisory metadata only.",
        "Preserve graph, routing, retry, storage, and output boundaries.",
    ]
    if estimated_token_savings:
        actions.append("Limit token savings to explicit reserve candidates.")
    if pressure == "high":
        actions.append("Require downstream strategy selection before applying changes.")
    return tuple(actions)
