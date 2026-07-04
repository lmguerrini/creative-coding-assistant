"""V5.1 workflow complexity analyzer for structural advisory signals."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.creative_planning import (
    CreativeExecutionPlan,
)
from creative_coding_assistant.orchestration.execution_graph_analyzer import (
    ExecutionGraphAnalysis,
    analyze_assistant_execution_graph,
)
from creative_coding_assistant.orchestration.workflow_cost_analyzer import (
    WorkflowCostAnalysis,
    analyze_workflow_cost,
)

WorkflowComplexityFactorKind = Literal[
    "topology",
    "branching",
    "retry",
    "failure_path",
    "cost_pressure",
    "plan_shape",
]
WorkflowComplexityLevel = Literal["low", "medium", "high"]

WORKFLOW_COMPLEXITY_FACTOR_SERIALIZATION_VERSION = "workflow_complexity_factor.v1"
WORKFLOW_COMPLEXITY_ANALYSIS_SERIALIZATION_VERSION = "workflow_complexity_analysis.v1"
WORKFLOW_COMPLEXITY_ANALYZER_AUTHORITY_BOUNDARY = (
    "Workflow complexity analysis derives structural advisory signals from "
    "workflow graph topology, retry and failure surfaces, cost pressure, and "
    "optional plan shape only; it does not evaluate creative semantics, prune "
    "workflow nodes, choose execution paths, select strategies, route providers "
    "or models, control workflow execution, trigger retries, mutate prompts, "
    "write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "creative_semantic_scoring",
    "workflow_pruning",
    "execution_path_selection",
    "strategy_selection",
    "provider_or_model_routing",
    "workflow_control",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class WorkflowComplexityFactor(BaseModel):
    """One structural factor in a workflow complexity analysis."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    factor_id: str = Field(min_length=1, max_length=160)
    factor_kind: WorkflowComplexityFactorKind
    source_id: str = Field(min_length=1, max_length=120)
    level: WorkflowComplexityLevel
    score: int = Field(ge=0, le=200)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    creative_semantic_scoring_implemented: Literal[False] = False
    workflow_pruning_implemented: Literal[False] = False
    execution_path_selection_implemented: Literal[False] = False
    strategy_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["workflow_complexity_factor.v1"] = (
        WORKFLOW_COMPLEXITY_FACTOR_SERIALIZATION_VERSION
    )
    analysis_only: Literal[True] = True


class WorkflowComplexityAnalysis(BaseModel):
    """Bounded V5.1 structural complexity analysis for assistant workflow."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["workflow_complexity_analyzer"] = "workflow_complexity_analyzer"
    serialization_version: Literal["workflow_complexity_analysis.v1"] = (
        WORKFLOW_COMPLEXITY_ANALYSIS_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=WORKFLOW_COMPLEXITY_ANALYZER_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    source_graph_serialization_version: str = Field(min_length=1, max_length=80)
    source_cost_serialization_version: str = Field(min_length=1, max_length=80)
    factors: tuple[WorkflowComplexityFactor, ...] = Field(min_length=1, max_length=12)
    factor_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    node_count: int = Field(ge=1, le=40)
    edge_count: int = Field(ge=1, le=120)
    branch_count: int = Field(ge=0, le=40)
    failure_edge_count: int = Field(ge=0, le=40)
    critical_path_length: int = Field(ge=1, le=40)
    retry_cycle_present: bool
    cost_pressure: WorkflowComplexityLevel
    plan_shape_complexity: WorkflowComplexityLevel | None = None
    structural_complexity_score: int = Field(ge=0, le=400)
    complexity_level: WorkflowComplexityLevel
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    complexity_analysis_implemented: Literal[True] = True
    creative_semantic_scoring_implemented: Literal[False] = False
    workflow_pruning_implemented: Literal[False] = False
    execution_path_selection_implemented: Literal[False] = False
    strategy_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    analysis_only: Literal[True] = True

    @model_validator(mode="after")
    def _analysis_matches_factors(self) -> Self:
        derived_factor_ids = tuple(factor.factor_id for factor in self.factors)
        if len(set(derived_factor_ids)) != len(derived_factor_ids):
            raise ValueError("factor_ids must be unique")
        if self.factor_ids != derived_factor_ids:
            raise ValueError("factor_ids must match factors")
        derived_score = sum(factor.score for factor in self.factors)
        if self.structural_complexity_score != derived_score:
            raise ValueError("structural_complexity_score must match factors")
        if self.complexity_level != _complexity_level(derived_score):
            raise ValueError("complexity_level must match score")
        if self.branch_count > 0 and "factor::branching" not in self.factor_ids:
            raise ValueError("branching factor is required when branches exist")
        if self.retry_cycle_present and "factor::retry" not in self.factor_ids:
            raise ValueError("retry factor is required when retry cycle exists")
        return self


def analyze_workflow_complexity(
    *,
    creative_plan: CreativeExecutionPlan | None = None,
    execution_graph: ExecutionGraphAnalysis | None = None,
    cost_analysis: WorkflowCostAnalysis | None = None,
) -> WorkflowComplexityAnalysis:
    """Return structural workflow complexity analysis without runtime control."""

    graph = execution_graph or analyze_assistant_execution_graph()
    costs = cost_analysis or analyze_workflow_cost(
        creative_plan=creative_plan,
        execution_graph=graph,
    )
    factors = _factors(
        graph=graph,
        costs=costs,
        creative_plan=creative_plan,
    )
    score = sum(factor.score for factor in factors)
    plan_level = _plan_shape_level(creative_plan) if creative_plan is not None else None

    return WorkflowComplexityAnalysis(
        source_graph_serialization_version=graph.serialization_version,
        source_cost_serialization_version=costs.serialization_version,
        factors=factors,
        factor_ids=tuple(factor.factor_id for factor in factors),
        node_count=graph.node_count,
        edge_count=graph.edge_count,
        branch_count=graph.branch_count,
        failure_edge_count=graph.failure_edge_count,
        critical_path_length=len(graph.critical_path_node_ids),
        retry_cycle_present=graph.bounded_retry_cycle_detected,
        cost_pressure=costs.estimated_cost_pressure,
        plan_shape_complexity=plan_level,
        structural_complexity_score=score,
        complexity_level=_complexity_level(score),
        advisory_actions=_analysis_actions(score, plan_level),
    )


def workflow_complexity_factor_by_id(
    factor_id: str,
    analysis: WorkflowComplexityAnalysis | None = None,
) -> WorkflowComplexityFactor | None:
    """Return one complexity factor without controlling workflow execution."""

    source_analysis = analysis or analyze_workflow_complexity()
    for factor in source_analysis.factors:
        if factor.factor_id == factor_id:
            return factor
    return None


def workflow_complexity_factors_for_kind(
    factor_kind: WorkflowComplexityFactorKind,
    analysis: WorkflowComplexityAnalysis | None = None,
) -> tuple[WorkflowComplexityFactor, ...]:
    """Return complexity factors by kind without pruning or path selection."""

    source_analysis = analysis or analyze_workflow_complexity()
    return tuple(
        factor
        for factor in source_analysis.factors
        if factor.factor_kind == factor_kind
    )


def _factors(
    *,
    graph: ExecutionGraphAnalysis,
    costs: WorkflowCostAnalysis,
    creative_plan: CreativeExecutionPlan | None,
) -> tuple[WorkflowComplexityFactor, ...]:
    factors = [
        _topology_factor(graph),
        _branching_factor(graph),
        _retry_factor(graph, costs),
        _failure_factor(graph),
        _cost_pressure_factor(costs),
    ]
    if creative_plan is not None:
        factors.append(_plan_shape_factor(creative_plan))
    return tuple(factors)


def _topology_factor(graph: ExecutionGraphAnalysis) -> WorkflowComplexityFactor:
    score = graph.node_count + graph.edge_count // 2
    return WorkflowComplexityFactor(
        factor_id="factor::topology",
        factor_kind="topology",
        source_id="execution_graph_analysis",
        level=_complexity_level(score),
        score=score,
        evidence=(
            f"nodes:{graph.node_count}",
            f"edges:{graph.edge_count}",
            f"critical_path:{len(graph.critical_path_node_ids)}",
        ),
        advisory_actions=("Keep topology size visible for later pruning decisions.",),
    )


def _branching_factor(graph: ExecutionGraphAnalysis) -> WorkflowComplexityFactor:
    score = graph.branch_count * 2
    return WorkflowComplexityFactor(
        factor_id="factor::branching",
        factor_kind="branching",
        source_id="execution_graph_analysis",
        level=_complexity_level(score),
        score=score,
        evidence=(
            f"branch_nodes:{graph.branch_count}",
            f"failure_edges:{graph.failure_edge_count}",
        ),
        advisory_actions=("Track branch density without selecting execution paths.",),
    )


def _retry_factor(
    graph: ExecutionGraphAnalysis,
    costs: WorkflowCostAnalysis,
) -> WorkflowComplexityFactor:
    retry_entries = len(graph.retry_entry_node_ids)
    score = retry_entries * 5 + costs.retry_iteration_count * 3
    return WorkflowComplexityFactor(
        factor_id="factor::retry",
        factor_kind="retry",
        source_id="execution_graph_analysis",
        level=_complexity_level(score),
        score=score,
        evidence=(
            f"retry_entries:{retry_entries}",
            f"retry_iterations:{costs.retry_iteration_count}",
        ),
        advisory_actions=("Account for retry complexity without triggering retries.",),
    )


def _failure_factor(graph: ExecutionGraphAnalysis) -> WorkflowComplexityFactor:
    score = graph.failure_edge_count
    return WorkflowComplexityFactor(
        factor_id="factor::failure_path",
        factor_kind="failure_path",
        source_id="execution_graph_analysis",
        level=_complexity_level(score),
        score=score,
        evidence=(
            f"failure_edges:{graph.failure_edge_count}",
            f"terminal_nodes:{len(graph.terminal_node_ids)}",
        ),
        advisory_actions=("Preserve failure normalization while tracking complexity.",),
    )


def _cost_pressure_factor(costs: WorkflowCostAnalysis) -> WorkflowComplexityFactor:
    score = {"low": 3, "medium": 8, "high": 13}[costs.estimated_cost_pressure]
    return WorkflowComplexityFactor(
        factor_id="factor::cost_pressure",
        factor_kind="cost_pressure",
        source_id="workflow_cost_analysis",
        level=costs.estimated_cost_pressure,
        score=score,
        evidence=(
            f"worst_case_tokens:{costs.worst_case_token_estimate}",
            f"cost_pressure:{costs.estimated_cost_pressure}",
        ),
        advisory_actions=("Use cost pressure as complexity metadata only.",),
    )


def _plan_shape_factor(plan: CreativeExecutionPlan) -> WorkflowComplexityFactor:
    level = _plan_shape_level(plan)
    score = (
        plan.candidate_count * 3
        + plan.refinement_budget * 4
        + {"low": 1, "medium": 4, "high": 7}[plan.expected_complexity]
    )
    return WorkflowComplexityFactor(
        factor_id="factor::plan_shape",
        factor_kind="plan_shape",
        source_id="creative_execution_plan",
        level=level,
        score=score,
        evidence=(
            f"candidate_count:{plan.candidate_count}",
            f"refinement_budget:{plan.refinement_budget}",
            f"expected_complexity:{plan.expected_complexity}",
        ),
        advisory_actions=("Use plan shape only as workflow structure pressure.",),
    )


def _plan_shape_level(plan: CreativeExecutionPlan) -> WorkflowComplexityLevel:
    score = (
        plan.candidate_count
        + plan.refinement_budget
        + {"low": 0, "medium": 1, "high": 2}[plan.expected_complexity]
    )
    if score <= 3:
        return "low"
    if score <= 5:
        return "medium"
    return "high"


def _complexity_level(score: int) -> WorkflowComplexityLevel:
    if score < 24:
        return "low"
    if score < 72:
        return "medium"
    return "high"


def _analysis_actions(
    score: int,
    plan_level: WorkflowComplexityLevel | None,
) -> tuple[str, ...]:
    actions = [
        "Expose workflow complexity as advisory metadata only.",
        "Preserve provider and model routing boundaries.",
    ]
    if score >= 72:
        actions.append("Flag high structural complexity for later pruning review.")
    if plan_level == "high":
        actions.append("Keep plan-shape pressure visible for strategy selection.")
    return tuple(actions)
