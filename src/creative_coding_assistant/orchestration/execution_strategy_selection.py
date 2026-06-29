"""V5.1 advisory execution strategy selection."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.execution_cost_forecasting import (
    ExecutionCostForecast,
    forecast_execution_cost,
)
from creative_coding_assistant.orchestration.execution_path_optimization import (
    ExecutionPathCandidate,
    ExecutionPathOptimizationPlan,
    plan_execution_path_optimization,
)
from creative_coding_assistant.orchestration.workflow_pruning import (
    WorkflowPruningPlan,
    plan_workflow_pruning,
)

ExecutionStrategyKind = Literal[
    "baseline_success",
    "cost_guarded_pruning",
    "retry_guarded_quality",
    "failure_safe",
]
ExecutionStrategyStatus = Literal["selected", "fallback", "deferred", "guardrail"]
ExecutionStrategyConfidence = Literal["low", "medium", "high"]

EXECUTION_STRATEGY_CANDIDATE_SERIALIZATION_VERSION = "execution_strategy_candidate.v1"
EXECUTION_STRATEGY_SELECTION_SERIALIZATION_VERSION = "execution_strategy_selection.v1"
EXECUTION_STRATEGY_SELECTION_AUTHORITY_BOUNDARY = (
    "Execution strategy selection chooses an advisory strategy profile from "
    "bounded path optimization, cost forecast, and pruning metadata only; it "
    "does not apply the strategy, select runtime execution paths, mutate graph "
    "order, compile or execute the workflow graph, invoke node handlers, route "
    "providers or models, enforce budgets, trigger retries, mutate prompts, "
    "write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "strategy_application",
    "execution_path_selection",
    "workflow_graph_mutation",
    "workflow_order_change",
    "langgraph_compilation",
    "workflow_execution",
    "node_handler_invocation",
    "provider_or_model_routing",
    "budget_enforcement",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class ExecutionStrategyCandidate(BaseModel):
    """One advisory execution strategy candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    strategy_id: str = Field(min_length=1, max_length=180)
    strategy_kind: ExecutionStrategyKind
    status: ExecutionStrategyStatus
    source_path_candidate_id: str = Field(min_length=1, max_length=180)
    forecast_tokens: int = Field(ge=0, le=240_000)
    estimated_token_savings: int = Field(ge=0, le=240_000)
    selection_score: int = Field(ge=0, le=2500)
    confidence: ExecutionStrategyConfidence
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    execution_strategy_selection_implemented: Literal[True] = True
    execution_strategy_application_implemented: Literal[False] = False
    execution_path_selection_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_order_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["execution_strategy_candidate.v1"] = (
        EXECUTION_STRATEGY_CANDIDATE_SERIALIZATION_VERSION
    )
    selection_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_status(self) -> Self:
        if self.status == "selected" and self.selection_score <= 0:
            raise ValueError("selected strategies require a positive score")
        if self.strategy_kind == "failure_safe" and self.status != "guardrail":
            raise ValueError("failure_safe strategy must remain a guardrail")
        return self


class ExecutionStrategySelection(BaseModel):
    """Bounded V5.1 advisory execution strategy selection."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["execution_strategy_selector"] = "execution_strategy_selector"
    serialization_version: Literal["execution_strategy_selection.v1"] = (
        EXECUTION_STRATEGY_SELECTION_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=EXECUTION_STRATEGY_SELECTION_AUTHORITY_BOUNDARY,
        max_length=1400,
    )
    source_path_optimization_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    source_cost_forecast_serialization_version: str = Field(min_length=1, max_length=80)
    source_pruning_serialization_version: str = Field(min_length=1, max_length=80)
    strategies: tuple[ExecutionStrategyCandidate, ...] = Field(
        min_length=1,
        max_length=12,
    )
    strategy_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    selected_strategy_id: str = Field(min_length=1, max_length=180)
    selected_path_candidate_id: str = Field(min_length=1, max_length=180)
    fallback_strategy_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    deferred_strategy_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    guardrail_strategy_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    strategy_count: int = Field(ge=1, le=12)
    selected_strategy_count: int = Field(ge=1, le=1)
    selected_strategy_score: int = Field(ge=0, le=2500)
    selected_estimated_token_savings: int = Field(ge=0, le=240_000)
    selection_confidence: ExecutionStrategyConfidence
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    execution_strategy_selection_implemented: Literal[True] = True
    execution_strategy_application_implemented: Literal[False] = False
    execution_path_selection_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_order_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    selection_only: Literal[True] = True

    @model_validator(mode="after")
    def _selection_matches_strategies(self) -> Self:
        derived_strategy_ids = tuple(strategy.strategy_id for strategy in self.strategies)
        if len(set(derived_strategy_ids)) != len(derived_strategy_ids):
            raise ValueError("strategy_ids must be unique")
        if self.strategy_ids != derived_strategy_ids:
            raise ValueError("strategy_ids must match strategies")
        if self.strategy_count != len(self.strategies):
            raise ValueError("strategy_count must match strategies")

        selected = tuple(
            strategy for strategy in self.strategies if strategy.status == "selected"
        )
        if len(selected) != 1:
            raise ValueError("exactly one selected strategy is required")
        selected_strategy = selected[0]
        if self.selected_strategy_id != selected_strategy.strategy_id:
            raise ValueError("selected_strategy_id must match selected strategy")
        if self.selected_path_candidate_id != selected_strategy.source_path_candidate_id:
            raise ValueError("selected_path_candidate_id must match selected strategy")
        if self.selected_strategy_count != 1:
            raise ValueError("selected_strategy_count must be one")
        if self.selected_strategy_score != selected_strategy.selection_score:
            raise ValueError("selected_strategy_score must match selected strategy")
        if (
            self.selected_estimated_token_savings
            != selected_strategy.estimated_token_savings
        ):
            raise ValueError("selected_estimated_token_savings must match selection")
        if self.selection_confidence != selected_strategy.confidence:
            raise ValueError("selection_confidence must match selected strategy")
        if self.fallback_strategy_ids != _strategy_ids_for_status(
            self.strategies,
            "fallback",
        ):
            raise ValueError("fallback_strategy_ids must match strategies")
        if self.deferred_strategy_ids != _strategy_ids_for_status(
            self.strategies,
            "deferred",
        ):
            raise ValueError("deferred_strategy_ids must match strategies")
        if self.guardrail_strategy_ids != _strategy_ids_for_status(
            self.strategies,
            "guardrail",
        ):
            raise ValueError("guardrail_strategy_ids must match strategies")
        return self


def select_execution_strategy(
    *,
    path_optimization: ExecutionPathOptimizationPlan | None = None,
    cost_forecast: ExecutionCostForecast | None = None,
    pruning_plan: WorkflowPruningPlan | None = None,
) -> ExecutionStrategySelection:
    """Select one advisory execution strategy without applying it."""

    pruning = pruning_plan or plan_workflow_pruning()
    forecast = cost_forecast or forecast_execution_cost(pruning_plan=pruning)
    path_plan = path_optimization or plan_execution_path_optimization(
        cost_forecast=forecast,
        pruning_plan=pruning,
    )
    strategies = _strategies(path_plan)
    selected = _selected_strategy(strategies)

    return ExecutionStrategySelection(
        source_path_optimization_serialization_version=path_plan.serialization_version,
        source_cost_forecast_serialization_version=forecast.serialization_version,
        source_pruning_serialization_version=pruning.serialization_version,
        strategies=strategies,
        strategy_ids=tuple(strategy.strategy_id for strategy in strategies),
        selected_strategy_id=selected.strategy_id,
        selected_path_candidate_id=selected.source_path_candidate_id,
        fallback_strategy_ids=_strategy_ids_for_status(strategies, "fallback"),
        deferred_strategy_ids=_strategy_ids_for_status(strategies, "deferred"),
        guardrail_strategy_ids=_strategy_ids_for_status(strategies, "guardrail"),
        strategy_count=len(strategies),
        selected_strategy_count=1,
        selected_strategy_score=selected.selection_score,
        selected_estimated_token_savings=selected.estimated_token_savings,
        selection_confidence=selected.confidence,
        advisory_actions=_selection_actions(selected),
    )


def execution_strategy_by_id(
    strategy_id: str,
    selection: ExecutionStrategySelection | None = None,
) -> ExecutionStrategyCandidate | None:
    """Return one strategy candidate without applying strategy behavior."""

    source_selection = selection or select_execution_strategy()
    for strategy in source_selection.strategies:
        if strategy.strategy_id == strategy_id:
            return strategy
    return None


def execution_strategies_for_status(
    status: ExecutionStrategyStatus,
    selection: ExecutionStrategySelection | None = None,
) -> tuple[ExecutionStrategyCandidate, ...]:
    """Return strategy candidates by status without workflow control."""

    source_selection = selection or select_execution_strategy()
    return tuple(strategy for strategy in source_selection.strategies if strategy.status == status)


def _strategies(
    path_plan: ExecutionPathOptimizationPlan,
) -> tuple[ExecutionStrategyCandidate, ...]:
    cost_guarded_path = _path_candidate(path_plan, "execution_path::pruning_adjusted_path")
    baseline_path = _path_candidate(path_plan, "execution_path::minimum_success_path")
    retry_path = _path_candidate(path_plan, "execution_path::single_retry_path")
    failure_path = _path_candidate(path_plan, "execution_path::failure_normalization_path")
    select_cost_guarded = bool(path_plan.optimization_candidate_ids)

    return (
        _strategy(
            strategy_kind="cost_guarded_pruning",
            status="selected" if select_cost_guarded else "fallback",
            path_candidate=cost_guarded_path,
            score_bonus=25,
            confidence=_confidence(path_plan.optimization_pressure),
            evidence=("optimization_candidate_available",),
        ),
        _strategy(
            strategy_kind="baseline_success",
            status="fallback" if select_cost_guarded else "selected",
            path_candidate=baseline_path,
            score_bonus=10,
            confidence="high",
            evidence=("baseline_success_path",),
        ),
        _strategy(
            strategy_kind="retry_guarded_quality",
            status="deferred",
            path_candidate=retry_path,
            score_bonus=5,
            confidence="medium",
            evidence=("retry_path_requires_strategy_review",),
        ),
        _strategy(
            strategy_kind="failure_safe",
            status="guardrail",
            path_candidate=failure_path,
            score_bonus=0,
            confidence="high",
            evidence=("failure_normalization_guardrail",),
        ),
    )


def _strategy(
    *,
    strategy_kind: ExecutionStrategyKind,
    status: ExecutionStrategyStatus,
    path_candidate: ExecutionPathCandidate,
    score_bonus: int,
    confidence: ExecutionStrategyConfidence,
    evidence: tuple[str, ...],
) -> ExecutionStrategyCandidate:
    savings = max(0, -path_candidate.token_delta_from_worst_case)
    selection_score = path_candidate.optimization_score + score_bonus
    if status in {"fallback", "guardrail"}:
        savings = 0
    return ExecutionStrategyCandidate(
        strategy_id=f"execution_strategy::{strategy_kind}",
        strategy_kind=strategy_kind,
        status=status,
        source_path_candidate_id=path_candidate.candidate_id,
        forecast_tokens=path_candidate.forecast_tokens,
        estimated_token_savings=savings,
        selection_score=selection_score,
        confidence=confidence,
        evidence=(
            *evidence,
            f"path_candidate:{path_candidate.candidate_id}",
            f"path_score:{path_candidate.optimization_score}",
        ),
        advisory_actions=_strategy_actions(strategy_kind, status),
    )


def _selected_strategy(
    strategies: tuple[ExecutionStrategyCandidate, ...],
) -> ExecutionStrategyCandidate:
    selected = tuple(strategy for strategy in strategies if strategy.status == "selected")
    if len(selected) != 1:
        raise ValueError("exactly one selected strategy is required")
    return selected[0]


def _path_candidate(
    path_plan: ExecutionPathOptimizationPlan,
    candidate_id: str,
) -> ExecutionPathCandidate:
    for candidate in path_plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise ValueError("required path candidate is missing")


def _strategy_ids_for_status(
    strategies: tuple[ExecutionStrategyCandidate, ...],
    status: ExecutionStrategyStatus,
) -> tuple[str, ...]:
    return tuple(strategy.strategy_id for strategy in strategies if strategy.status == status)


def _confidence(
    pressure: str,
) -> ExecutionStrategyConfidence:
    if pressure == "high":
        return "high"
    if pressure == "medium":
        return "medium"
    return "low"


def _strategy_actions(
    strategy_kind: ExecutionStrategyKind,
    status: ExecutionStrategyStatus,
) -> tuple[str, ...]:
    if strategy_kind == "failure_safe":
        return ("Retain failure-safe strategy as a non-optional guardrail.",)
    if status == "selected":
        return ("Select advisory strategy metadata without applying it.",)
    if status == "deferred":
        return ("Defer retry-sensitive strategy to later explicit execution control.",)
    return ("Keep strategy available as advisory fallback metadata.",)


def _selection_actions(
    selected: ExecutionStrategyCandidate,
) -> tuple[str, ...]:
    return (
        "Expose selected execution strategy as advisory metadata only.",
        "Preserve strategy application, path selection, routing, retry, and output boundaries.",
        f"Selected strategy candidate: {selected.strategy_id}.",
    )
