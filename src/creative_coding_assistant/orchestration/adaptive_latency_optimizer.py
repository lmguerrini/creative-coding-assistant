"""V5.5 adaptive advisory latency optimization intelligence."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_hybrid_workflow_optimizer import (
    HybridWorkflowOptimizationPlan,
    optimize_hybrid_workflow,
)
from creative_coding_assistant.orchestration.agent_activation_optimizer import (
    AgentActivationOptimizationPlan,
    optimize_agent_activation,
)
from creative_coding_assistant.orchestration.latency_optimizer import (
    LatencyOptimizationBand,
    LatencyOptimizationPlan,
    optimize_latency,
)
from creative_coding_assistant.orchestration.performance_prediction import (
    PerformancePredictionPlan,
    predict_performance,
)
from creative_coding_assistant.orchestration.resource_utilization_optimizer import (
    ResourceUtilizationOptimizationPlan,
    ResourceUtilizationPressure,
    optimize_resource_utilization,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    EstimatedLatencyBand,
    ExecutionModeId,
    TaskRoutingType,
    routing_execution_mode_registry,
)

AdaptiveLatencyStatus = Literal["recommended", "fallback", "guardrail"]
AdaptiveLatencyPosture = Literal[
    "fast_path_candidate",
    "balanced_latency",
    "guarded_latency",
]

ADAPTIVE_LATENCY_CANDIDATE_SERIALIZATION_VERSION = "adaptive_latency_candidate.v1"
ADAPTIVE_LATENCY_PLAN_SERIALIZATION_VERSION = "adaptive_latency_plan.v1"
ADAPTIVE_LATENCY_AUTHORITY_BOUNDARY = (
    "V5.5 adaptive latency optimization combines advisory latency optimizer, "
    "performance prediction, resource utilization, hybrid workflow, and agent "
    "activation metadata into relative latency recommendations only; it does "
    "not measure latency, evaluate latency thresholds, route by latency, "
    "select runtimes, execute parallel or async tasks, change workflow timing, "
    "mutate workflow graphs, execute workflows, invoke agents, route providers "
    "or models, trigger retries, write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "latency_measurement",
    "latency_threshold_evaluation",
    "latency_based_routing",
    "runtime_selection",
    "parallel_task_execution",
    "async_task_creation",
    "workflow_timing_change",
    "workflow_graph_mutation",
    "langgraph_compilation",
    "workflow_execution",
    "agent_invocation",
    "node_handler_invocation",
    "provider_or_model_routing",
    "provider_execution",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class AdaptiveLatencyCandidate(BaseModel):
    """One advisory adaptive latency candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=180)
    source_latency_candidate_id: str = Field(min_length=1, max_length=180)
    source_hybrid_workflow_candidate_id: str = Field(min_length=1, max_length=180)
    stage_id: str = Field(min_length=1, max_length=80)
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_latency_band: LatencyOptimizationBand
    hybrid_estimated_latency: EstimatedLatencyBand
    resource_utilization_pressure: ResourceUtilizationPressure
    predicted_performance_midpoint: int = Field(ge=0, le=100)
    agent_activation_candidate_count: int = Field(ge=1, le=12)
    latency_savings_score: int = Field(ge=0, le=600)
    latency_pressure_score: int = Field(ge=0, le=1000)
    performance_weight: int = Field(ge=0, le=120)
    resource_weight: int = Field(ge=0, le=120)
    hybrid_latency_weight: int = Field(ge=0, le=120)
    adaptive_latency_score: int = Field(ge=0, le=240)
    adaptive_latency_posture: AdaptiveLatencyPosture
    status: AdaptiveLatencyStatus
    hitl_required: bool
    evidence: tuple[str, ...] = Field(min_length=1, max_length=10)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    adaptive_latency_optimizer_implemented: Literal[True] = True
    adaptive_latency_scoring_implemented: Literal[True] = True
    latency_measurement_implemented: Literal[False] = False
    latency_threshold_evaluation_implemented: Literal[False] = False
    latency_based_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    parallel_execution_implemented: Literal[False] = False
    async_execution_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["adaptive_latency_candidate.v1"] = (
        ADAPTIVE_LATENCY_CANDIDATE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_scores(self) -> Self:
        if self.candidate_id != f"adaptive_latency::{self.stage_id}":
            raise ValueError("candidate_id must match stage_id")
        expected = min(
            240,
            self.performance_weight
            + self.resource_weight
            + self.hybrid_latency_weight
            + self.latency_savings_score // 10
            - self.latency_pressure_score // 100,
        )
        if self.adaptive_latency_score != expected:
            raise ValueError("adaptive_latency_score must combine weights")
        if self.adaptive_latency_posture != _latency_posture(
            self.source_latency_band,
            self.hybrid_estimated_latency,
            self.adaptive_latency_score,
        ):
            raise ValueError("adaptive_latency_posture must match inputs")
        return self


class AdaptiveLatencyPlan(BaseModel):
    """Bounded V5.5 advisory adaptive latency optimization plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["adaptive_latency_optimizer"] = "adaptive_latency_optimizer"
    serialization_version: Literal["adaptive_latency_plan.v1"] = (
        ADAPTIVE_LATENCY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ADAPTIVE_LATENCY_AUTHORITY_BOUNDARY,
        max_length=1700,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_latency_optimization_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_performance_prediction_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_resource_utilization_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_hybrid_workflow_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_agent_activation_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    candidates: tuple[AdaptiveLatencyCandidate, ...] = Field(
        min_length=1,
        max_length=20,
    )
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=20)
    recommended_candidate_id: str = Field(min_length=1, max_length=180)
    fallback_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple, max_length=20
    )
    guardrail_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple, max_length=20
    )
    recommended_adaptive_latency_posture: AdaptiveLatencyPosture
    candidate_count: int = Field(ge=1, le=20)
    recommended_adaptive_latency_score: int = Field(ge=0, le=240)
    hitl_required_candidate_count: int = Field(ge=0, le=20)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    adaptive_latency_optimizer_implemented: Literal[True] = True
    adaptive_latency_scoring_implemented: Literal[True] = True
    latency_measurement_implemented: Literal[False] = False
    latency_threshold_evaluation_implemented: Literal[False] = False
    latency_based_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    parallel_execution_implemented: Literal[False] = False
    async_execution_implemented: Literal[False] = False
    workflow_timing_change_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    node_handler_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_candidates(self) -> Self:
        derived_candidate_ids = tuple(
            candidate.candidate_id for candidate in self.candidates
        )
        if len(set(derived_candidate_ids)) != len(derived_candidate_ids):
            raise ValueError("candidate_ids must be unique")
        if self.candidate_ids != derived_candidate_ids:
            raise ValueError("candidate_ids must match candidates")
        if self.candidate_count != len(self.candidates):
            raise ValueError("candidate_count must match candidates")
        recommended = tuple(
            candidate
            for candidate in self.candidates
            if candidate.status == "recommended"
        )
        if len(recommended) != 1:
            raise ValueError("exactly one recommended candidate is required")
        recommended_candidate = recommended[0]
        if self.recommended_candidate_id != recommended_candidate.candidate_id:
            raise ValueError("recommended_candidate_id must match candidate")
        if (
            self.recommended_adaptive_latency_posture
            != recommended_candidate.adaptive_latency_posture
        ):
            raise ValueError(
                "recommended_adaptive_latency_posture must match candidate"
            )
        if (
            self.recommended_adaptive_latency_score
            != recommended_candidate.adaptive_latency_score
        ):
            raise ValueError("recommended_adaptive_latency_score must match candidate")
        if self.fallback_candidate_ids != tuple(
            candidate.candidate_id
            for candidate in self.candidates
            if candidate.status == "fallback"
        ):
            raise ValueError("fallback_candidate_ids must match candidates")
        if self.guardrail_candidate_ids != tuple(
            candidate.candidate_id
            for candidate in self.candidates
            if candidate.status == "guardrail"
        ):
            raise ValueError("guardrail_candidate_ids must match candidates")
        if self.hitl_required_candidate_count != sum(
            1 for candidate in self.candidates if candidate.hitl_required
        ):
            raise ValueError("hitl_required_candidate_count must match candidates")
        return self


def optimize_adaptive_latency(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    latency: LatencyOptimizationPlan | None = None,
    performance: PerformancePredictionPlan | None = None,
    resources: ResourceUtilizationOptimizationPlan | None = None,
    hybrid_workflow: HybridWorkflowOptimizationPlan | None = None,
    agent_activation: AgentActivationOptimizationPlan | None = None,
) -> AdaptiveLatencyPlan:
    """Recommend adaptive latency posture without applying runtime behavior."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    latency_plan = latency or optimize_latency()
    performance_plan = performance or predict_performance()
    resource_plan = resources or optimize_resource_utilization()
    hybrid_plan = hybrid_workflow or optimize_hybrid_workflow(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=execution_mode_id,
    )
    agent_plan = agent_activation or optimize_agent_activation(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=execution_mode_id,
    )
    execution_modes = routing_execution_mode_registry()
    candidates = tuple(
        _candidate_from_latency(
            source=candidate,
            route_name=route_name,
            task_type=hybrid_plan.task_type,
            execution_mode_id=agent_plan.candidates[0].execution_mode_id,
            performance=performance_plan,
            resources=resource_plan,
            hybrid_workflow=hybrid_plan,
            agent_activation=agent_plan,
        )
        for candidate in latency_plan.candidates
    )
    recommended = max(
        candidates, key=lambda candidate: candidate.adaptive_latency_score
    )
    candidates = tuple(
        candidate.model_copy(
            update={
                "status": _status_for_candidate(candidate, recommended.candidate_id),
            }
        )
        for candidate in candidates
    )
    recommended = next(
        candidate for candidate in candidates if candidate.status == "recommended"
    )
    return AdaptiveLatencyPlan(
        route_name=route_name,
        task_type=hybrid_plan.task_type,
        source_latency_optimization_serialization_version=latency_plan.serialization_version,
        source_performance_prediction_serialization_version=performance_plan.serialization_version,
        source_resource_utilization_serialization_version=resource_plan.serialization_version,
        source_hybrid_workflow_serialization_version=hybrid_plan.serialization_version,
        source_agent_activation_serialization_version=agent_plan.serialization_version,
        execution_mode_ids=execution_modes.execution_mode_ids,
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        recommended_candidate_id=recommended.candidate_id,
        fallback_candidate_ids=tuple(
            candidate.candidate_id
            for candidate in candidates
            if candidate.status == "fallback"
        ),
        guardrail_candidate_ids=tuple(
            candidate.candidate_id
            for candidate in candidates
            if candidate.status == "guardrail"
        ),
        recommended_adaptive_latency_posture=recommended.adaptive_latency_posture,
        candidate_count=len(candidates),
        recommended_adaptive_latency_score=recommended.adaptive_latency_score,
        hitl_required_candidate_count=sum(
            1 for candidate in candidates if candidate.hitl_required
        ),
        advisory_actions=_plan_actions(recommended),
    )


def adaptive_latency_candidate_by_id(
    candidate_id: str,
    plan: AdaptiveLatencyPlan | None = None,
) -> AdaptiveLatencyCandidate | None:
    """Return one adaptive latency candidate without applying it."""

    source_plan = plan or optimize_adaptive_latency()
    for candidate in source_plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def adaptive_latency_candidates_for_posture(
    posture: AdaptiveLatencyPosture,
    plan: AdaptiveLatencyPlan | None = None,
) -> tuple[AdaptiveLatencyCandidate, ...]:
    """Return adaptive latency candidates for one posture."""

    source_plan = plan or optimize_adaptive_latency()
    return tuple(
        candidate
        for candidate in source_plan.candidates
        if candidate.adaptive_latency_posture == posture
    )


def _candidate_from_latency(
    *,
    source: object,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    performance: PerformancePredictionPlan,
    resources: ResourceUtilizationOptimizationPlan,
    hybrid_workflow: HybridWorkflowOptimizationPlan,
    agent_activation: AgentActivationOptimizationPlan,
) -> AdaptiveLatencyCandidate:
    performance_midpoint = performance.recommended_performance_midpoint
    resource_weight = _resource_weight(resources.resource_utilization_pressure)
    hybrid_latency = next(
        candidate
        for candidate in hybrid_workflow.candidates
        if candidate.candidate_id == hybrid_workflow.recommended_candidate_id
    ).estimated_latency
    hybrid_weight = _hybrid_latency_weight(hybrid_latency)
    performance_weight = min(
        120,
        performance_midpoint + source.advisory_latency_savings_score // 10,
    )
    score = min(
        240,
        performance_weight
        + resource_weight
        + hybrid_weight
        + source.advisory_latency_savings_score // 10
        - source.advisory_latency_pressure_score // 100,
    )
    return AdaptiveLatencyCandidate(
        candidate_id=f"adaptive_latency::{source.stage_id}",
        source_latency_candidate_id=source.candidate_id,
        source_hybrid_workflow_candidate_id=hybrid_workflow.recommended_candidate_id,
        stage_id=source.stage_id,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_latency_band=source.latency_band,
        hybrid_estimated_latency=hybrid_latency,
        resource_utilization_pressure=resources.resource_utilization_pressure,
        predicted_performance_midpoint=performance_midpoint,
        agent_activation_candidate_count=agent_activation.candidate_count,
        latency_savings_score=source.advisory_latency_savings_score,
        latency_pressure_score=source.advisory_latency_pressure_score,
        performance_weight=performance_weight,
        resource_weight=resource_weight,
        hybrid_latency_weight=hybrid_weight,
        adaptive_latency_score=score,
        adaptive_latency_posture=_latency_posture(
            source.latency_band,
            hybrid_latency,
            score,
        ),
        status="fallback",
        hitl_required=bool(agent_activation.hitl_required_candidate_ids),
        evidence=(
            f"source_latency:{source.candidate_id}",
            f"performance_midpoint:{performance_midpoint}",
            f"resource_pressure:{resources.resource_utilization_pressure}",
            f"hybrid_latency:{hybrid_latency}",
            f"agent_candidates:{agent_activation.candidate_count}",
        ),
        advisory_actions=(
            "Surface adaptive latency posture for review.",
            "Keep measurement, threshold evaluation, routing, parallelism, workflow timing, and execution disabled.",
        ),
    )


def _status_for_candidate(
    candidate: AdaptiveLatencyCandidate,
    recommended_candidate_id: str,
) -> AdaptiveLatencyStatus:
    if candidate.candidate_id == recommended_candidate_id:
        return "recommended"
    if candidate.adaptive_latency_posture == "guarded_latency":
        return "guardrail"
    return "fallback"


def _resource_weight(pressure: ResourceUtilizationPressure) -> int:
    return {
        "low": 36,
        "medium": 28,
        "high": 18,
        "guarded": 8,
    }[pressure]


def _hybrid_latency_weight(latency: EstimatedLatencyBand) -> int:
    return {
        "fast": 42,
        "moderate": 28,
        "slow": 12,
    }[latency]


def _latency_posture(
    latency_band: LatencyOptimizationBand,
    hybrid_latency: EstimatedLatencyBand,
    score: int,
) -> AdaptiveLatencyPosture:
    if latency_band == "guarded" or hybrid_latency == "slow":
        return "guarded_latency"
    if score >= 120 and hybrid_latency == "fast":
        return "fast_path_candidate"
    return "balanced_latency"


def _plan_actions(
    recommended: AdaptiveLatencyCandidate,
) -> tuple[str, ...]:
    return (
        f"Present {recommended.adaptive_latency_posture} as advisory adaptive latency posture.",
        "Use relative latency, performance, resource, hybrid, and agent metadata only.",
        "Preserve measurement, routing, parallel execution, workflow timing, workflow execution, storage, and output boundaries.",  # noqa: E501
    )


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
