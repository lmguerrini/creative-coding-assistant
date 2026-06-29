"""V5.1 advisory execution path optimization planning."""

from __future__ import annotations

from typing import Literal, Self

from langgraph.graph import END, START
from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.execution_cost_forecasting import (
    ExecutionCostForecast,
    ExecutionCostForecastScenario,
    forecast_execution_cost,
)
from creative_coding_assistant.orchestration.execution_graph_analyzer import (
    ExecutionGraphAnalysis,
    analyze_assistant_execution_graph,
)
from creative_coding_assistant.orchestration.workflow_cost_analyzer import (
    WorkflowCostAnalysis,
    analyze_workflow_cost,
)
from creative_coding_assistant.orchestration.workflow_pruning import (
    WorkflowPruningPlan,
    plan_workflow_pruning,
)

ExecutionPathCandidateKind = Literal[
    "minimum_success_path",
    "single_retry_path",
    "worst_case_bound_path",
    "pruning_adjusted_path",
    "failure_normalization_path",
]
ExecutionPathCandidateStatus = Literal[
    "baseline",
    "optimization_candidate",
    "retain",
    "review",
]
ExecutionPathOptimizationPressure = Literal["low", "medium", "high"]

EXECUTION_PATH_CANDIDATE_SERIALIZATION_VERSION = "execution_path_candidate.v1"
EXECUTION_PATH_OPTIMIZATION_PLAN_SERIALIZATION_VERSION = (
    "execution_path_optimization_plan.v1"
)
EXECUTION_PATH_OPTIMIZATION_AUTHORITY_BOUNDARY = (
    "Execution path optimization derives advisory path candidates from static "
    "workflow topology, bounded cost forecasts, and pruning metadata only; it "
    "does not select execution paths, mutate graph order, compile or execute "
    "the workflow graph, invoke node handlers, route providers or models, "
    "trigger retries, mutate prompts, write storage, or modify generated "
    "output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "execution_path_selection",
    "workflow_graph_mutation",
    "workflow_order_change",
    "langgraph_compilation",
    "workflow_execution",
    "node_handler_invocation",
    "provider_or_model_routing",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class ExecutionPathCandidate(BaseModel):
    """One advisory execution path optimization candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=180)
    candidate_kind: ExecutionPathCandidateKind
    status: ExecutionPathCandidateStatus
    node_ids: tuple[str, ...] = Field(min_length=1, max_length=40)
    edge_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=80)
    source_scenario_id: str = Field(min_length=1, max_length=180)
    forecast_tokens: int = Field(ge=0, le=240_000)
    token_delta_from_worst_case: int = Field(ge=-240_000, le=240_000)
    advisory_rank: int = Field(ge=1, le=20)
    optimization_score: int = Field(ge=0, le=2400)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    execution_path_optimization_implemented: Literal[True] = True
    execution_path_selection_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_order_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["execution_path_candidate.v1"] = (
        EXECUTION_PATH_CANDIDATE_SERIALIZATION_VERSION
    )
    planning_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_status(self) -> Self:
        if self.status == "optimization_candidate" and self.optimization_score <= 0:
            raise ValueError("optimization candidates require a positive score")
        if self.candidate_kind == "failure_normalization_path" and self.status != "retain":
            raise ValueError("failure normalization path must be retained")
        return self


class ExecutionPathOptimizationPlan(BaseModel):
    """Bounded V5.1 advisory execution path optimization plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["execution_path_optimizer"] = "execution_path_optimizer"
    serialization_version: Literal["execution_path_optimization_plan.v1"] = (
        EXECUTION_PATH_OPTIMIZATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=EXECUTION_PATH_OPTIMIZATION_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    source_graph_serialization_version: str = Field(min_length=1, max_length=80)
    source_cost_serialization_version: str = Field(min_length=1, max_length=80)
    source_forecast_serialization_version: str = Field(min_length=1, max_length=80)
    source_pruning_serialization_version: str = Field(min_length=1, max_length=80)
    candidates: tuple[ExecutionPathCandidate, ...] = Field(
        min_length=1,
        max_length=20,
    )
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=20)
    baseline_candidate_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=20)
    optimization_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=20,
    )
    retained_candidate_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=20)
    review_candidate_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=20)
    candidate_count: int = Field(ge=1, le=20)
    highest_advisory_score: int = Field(ge=0, le=2400)
    largest_advisory_token_savings: int = Field(ge=0, le=240_000)
    optimization_pressure: ExecutionPathOptimizationPressure
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    execution_path_optimization_implemented: Literal[True] = True
    execution_path_selection_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_order_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
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

        if self.baseline_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "baseline",
        ):
            raise ValueError("baseline_candidate_ids must match candidates")
        if self.optimization_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "optimization_candidate",
        ):
            raise ValueError("optimization_candidate_ids must match candidates")
        if self.retained_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "retain",
        ):
            raise ValueError("retained_candidate_ids must match candidates")
        if self.review_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "review",
        ):
            raise ValueError("review_candidate_ids must match candidates")

        expected_score = max(candidate.optimization_score for candidate in self.candidates)
        if self.highest_advisory_score != expected_score:
            raise ValueError("highest_advisory_score must match candidates")
        expected_savings = max(
            max(0, -candidate.token_delta_from_worst_case)
            for candidate in self.candidates
        )
        if self.largest_advisory_token_savings != expected_savings:
            raise ValueError("largest_advisory_token_savings must match candidates")
        if self.optimization_pressure != _optimization_pressure(
            highest_score=self.highest_advisory_score,
            largest_savings=self.largest_advisory_token_savings,
        ):
            raise ValueError("optimization_pressure must match candidates")
        return self


def plan_execution_path_optimization(
    *,
    execution_graph: ExecutionGraphAnalysis | None = None,
    cost_analysis: WorkflowCostAnalysis | None = None,
    cost_forecast: ExecutionCostForecast | None = None,
    pruning_plan: WorkflowPruningPlan | None = None,
) -> ExecutionPathOptimizationPlan:
    """Plan advisory execution path optimization without selecting a path."""

    graph = execution_graph or analyze_assistant_execution_graph()
    costs = cost_analysis or analyze_workflow_cost(execution_graph=graph)
    pruning = pruning_plan or plan_workflow_pruning(
        execution_graph=graph,
        cost_analysis=costs,
    )
    forecast = cost_forecast or forecast_execution_cost(
        cost_analysis=costs,
        pruning_plan=pruning,
    )
    candidates = _candidates(graph=graph, costs=costs, forecast=forecast)
    savings = max(max(0, -candidate.token_delta_from_worst_case) for candidate in candidates)
    score = max(candidate.optimization_score for candidate in candidates)

    return ExecutionPathOptimizationPlan(
        source_graph_serialization_version=graph.serialization_version,
        source_cost_serialization_version=costs.serialization_version,
        source_forecast_serialization_version=forecast.serialization_version,
        source_pruning_serialization_version=pruning.serialization_version,
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        baseline_candidate_ids=_candidate_ids_for_status(candidates, "baseline"),
        optimization_candidate_ids=_candidate_ids_for_status(
            candidates,
            "optimization_candidate",
        ),
        retained_candidate_ids=_candidate_ids_for_status(candidates, "retain"),
        review_candidate_ids=_candidate_ids_for_status(candidates, "review"),
        candidate_count=len(candidates),
        highest_advisory_score=score,
        largest_advisory_token_savings=savings,
        optimization_pressure=_optimization_pressure(
            highest_score=score,
            largest_savings=savings,
        ),
        advisory_actions=_plan_actions(score, savings),
    )


def execution_path_candidate_by_id(
    candidate_id: str,
    plan: ExecutionPathOptimizationPlan | None = None,
) -> ExecutionPathCandidate | None:
    """Return one path candidate without selecting it."""

    source_plan = plan or plan_execution_path_optimization()
    for candidate in source_plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def execution_path_candidates_for_status(
    status: ExecutionPathCandidateStatus,
    plan: ExecutionPathOptimizationPlan | None = None,
) -> tuple[ExecutionPathCandidate, ...]:
    """Return path candidates by status without workflow control."""

    source_plan = plan or plan_execution_path_optimization()
    return tuple(candidate for candidate in source_plan.candidates if candidate.status == status)


def _candidates(
    *,
    graph: ExecutionGraphAnalysis,
    costs: WorkflowCostAnalysis,
    forecast: ExecutionCostForecast,
) -> tuple[ExecutionPathCandidate, ...]:
    minimum = _scenario_by_kind(forecast, "minimum_success_path")
    single_retry = _scenario_by_kind(forecast, "single_retry_path")
    worst_case = _scenario_by_kind(forecast, "worst_case_bound")
    pruning_adjusted = _scenario_by_kind(forecast, "pruning_adjusted_bound")

    return (
        _candidate(
            kind="pruning_adjusted_path",
            status="optimization_candidate",
            rank=1,
            node_ids=graph.critical_path_node_ids,
            graph=graph,
            scenario=pruning_adjusted,
            evidence=("pruning_adjusted_forecast",),
        ),
        _candidate(
            kind="minimum_success_path",
            status="baseline",
            rank=2,
            node_ids=graph.critical_path_node_ids,
            graph=graph,
            scenario=minimum,
            evidence=("critical_success_path",),
        ),
        _candidate(
            kind="single_retry_path",
            status="review",
            rank=3,
            node_ids=_single_retry_node_ids(graph, costs),
            graph=graph,
            scenario=single_retry,
            evidence=("single_retry_forecast",),
        ),
        _candidate(
            kind="worst_case_bound_path",
            status="retain",
            rank=4,
            node_ids=graph.node_order,
            graph=graph,
            scenario=worst_case,
            evidence=("worst_case_bound",),
        ),
        _candidate(
            kind="failure_normalization_path",
            status="retain",
            rank=5,
            node_ids=("failure",),
            graph=graph,
            scenario=worst_case,
            evidence=("failure_path_reachable",),
        ),
    )


def _single_retry_node_ids(
    graph: ExecutionGraphAnalysis,
    costs: WorkflowCostAnalysis,
) -> tuple[str, ...]:
    prefix: list[str] = []
    for node_id in graph.critical_path_node_ids:
        prefix.append(node_id)
        if node_id == "review":
            break
    return tuple((*prefix, *costs.retry_path_node_ids, "finalization"))


def _candidate(
    *,
    kind: ExecutionPathCandidateKind,
    status: ExecutionPathCandidateStatus,
    rank: int,
    node_ids: tuple[str, ...],
    graph: ExecutionGraphAnalysis,
    scenario: ExecutionCostForecastScenario,
    evidence: tuple[str, ...],
) -> ExecutionPathCandidate:
    score = _optimization_score(scenario)
    if status in {"baseline", "retain"}:
        score = 0
    return ExecutionPathCandidate(
        candidate_id=f"execution_path::{kind}",
        candidate_kind=kind,
        status=status,
        node_ids=node_ids,
        edge_ids=_edge_ids_for_node_path(graph, node_ids),
        source_scenario_id=scenario.scenario_id,
        forecast_tokens=scenario.forecast_tokens,
        token_delta_from_worst_case=scenario.token_delta_from_worst_case,
        advisory_rank=rank,
        optimization_score=score,
        evidence=(
            *evidence,
            f"forecast_tokens:{scenario.forecast_tokens}",
            f"token_delta:{scenario.token_delta_from_worst_case}",
        ),
        advisory_actions=_candidate_actions(kind, status),
    )


def _edge_ids_for_node_path(
    graph: ExecutionGraphAnalysis,
    node_ids: tuple[str, ...],
) -> tuple[str, ...]:
    edge_ids: list[str] = []
    first_node = node_ids[0]
    for edge in graph.edges:
        if edge.source_node_id == str(START) and edge.target_node_id == first_node:
            edge_ids.append(edge.edge_id)
            break
    for source, target in zip(node_ids, node_ids[1:], strict=False):
        for edge in graph.edges:
            if edge.source_node_id == source and edge.target_node_id == target:
                edge_ids.append(edge.edge_id)
                break
    last_node = node_ids[-1]
    for edge in graph.edges:
        if edge.source_node_id == last_node and edge.target_node_id == str(END):
            edge_ids.append(edge.edge_id)
            break
    return tuple(edge_ids)


def _scenario_by_kind(
    forecast: ExecutionCostForecast,
    scenario_kind: str,
) -> ExecutionCostForecastScenario:
    for scenario in forecast.scenarios:
        if scenario.scenario_kind == scenario_kind:
            return scenario
    raise ValueError("forecast scenario kind is required")


def _candidate_ids_for_status(
    candidates: tuple[ExecutionPathCandidate, ...],
    status: ExecutionPathCandidateStatus,
) -> tuple[str, ...]:
    return tuple(
        candidate.candidate_id for candidate in candidates if candidate.status == status
    )


def _optimization_score(scenario: ExecutionCostForecastScenario) -> int:
    token_savings = max(0, -scenario.token_delta_from_worst_case)
    return min(2400, token_savings // 100)


def _optimization_pressure(
    *,
    highest_score: int,
    largest_savings: int,
) -> ExecutionPathOptimizationPressure:
    if highest_score >= 80 or largest_savings >= 8_000:
        return "high"
    if highest_score >= 30 or largest_savings >= 3_000:
        return "medium"
    return "low"


def _candidate_actions(
    kind: ExecutionPathCandidateKind,
    status: ExecutionPathCandidateStatus,
) -> tuple[str, ...]:
    if kind == "failure_normalization_path":
        return ("Retain failure normalization path without path selection.",)
    if status == "optimization_candidate":
        return ("Expose path as an optimization candidate without selecting it.",)
    if status == "baseline":
        return ("Use baseline path metadata as comparison only.",)
    return ("Keep path visible for later strategy selection.",)


def _plan_actions(
    highest_score: int,
    largest_savings: int,
) -> tuple[str, ...]:
    actions = [
        "Expose execution path optimization candidates as advisory metadata only.",
        "Preserve path selection, graph mutation, routing, retry, and output boundaries.",
    ]
    if highest_score > 0:
        actions.append("Use advisory score only in downstream strategy selection.")
    if largest_savings:
        actions.append("Report token savings without changing workflow execution.")
    return tuple(actions)
