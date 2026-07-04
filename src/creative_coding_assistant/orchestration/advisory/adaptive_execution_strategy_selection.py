"""V5.5 advisory dynamic execution strategy selection intelligence."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_cost_quality_optimizer import (
    AdaptiveCostQualityPlan,
    optimize_adaptive_cost_quality,
)
from creative_coding_assistant.orchestration.adaptive_escalation_optimizer import (
    EscalationOptimizationPlan,
    optimize_escalation_policy,
)
from creative_coding_assistant.orchestration.adaptive_hybrid_workflow_optimizer import (
    HybridWorkflowOptimizationCandidate,
    HybridWorkflowOptimizationPlan,
    optimize_hybrid_workflow,
)
from creative_coding_assistant.orchestration.adaptive_latency_optimizer import (
    AdaptiveLatencyPlan,
    optimize_adaptive_latency,
)
from creative_coding_assistant.orchestration.agent_activation_optimizer import (
    AgentActivationOptimizationPlan,
    optimize_agent_activation,
)
from creative_coding_assistant.orchestration.execution_strategy_selection import (
    ExecutionStrategySelection,
    select_execution_strategy,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    EstimatedCostBand,
    EstimatedLatencyBand,
    EstimatedQualityBand,
    ExecutionModeId,
    HybridRoutingPolicyDirection,
    ProviderId,
    TaskRoutingType,
    UnavailableReasonCode,
    routing_execution_mode_registry,
)

DynamicExecutionStrategyKind = Literal[
    "balanced_hybrid_strategy",
    "quality_priority_strategy",
    "latency_priority_strategy",
    "human_guarded_fallback_strategy",
]
DynamicExecutionStrategyStatus = Literal["selected", "fallback", "guardrail"]

ADAPTIVE_EXECUTION_STRATEGY_CANDIDATE_SERIALIZATION_VERSION = (
    "adaptive_execution_strategy_candidate.v1"
)
ADAPTIVE_EXECUTION_STRATEGY_SELECTION_SERIALIZATION_VERSION = (
    "adaptive_execution_strategy_selection_plan.v1"
)
ADAPTIVE_EXECUTION_STRATEGY_SELECTION_AUTHORITY_BOUNDARY = (
    "V5.5 dynamic execution strategy selection combines advisory execution "
    "strategy, hybrid workflow, escalation, agent activation, adaptive "
    "cost/quality, and adaptive latency metadata into inspectable strategy "
    "selection metadata only; it does not apply strategies, change provider "
    "or model routing, execute providers, probe local runtimes, scan or "
    "download local models, invoke agents, emit HITL requests, enforce "
    "budgets, control workflows, compile or execute workflow graphs, trigger "
    "retries, mutate prompts, write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "strategy_application",
    "silent_provider_or_model_routing_change",
    "automatic_provider_switching",
    "automatic_model_switching",
    "automatic_model_download",
    "provider_or_model_routing",
    "provider_execution",
    "local_runtime_probe",
    "local_model_inventory_scan",
    "agent_invocation",
    "multi_agent_orchestration",
    "human_review_request",
    "hitl_request_emission",
    "budget_enforcement",
    "workflow_control",
    "workflow_graph_mutation",
    "langgraph_compilation",
    "workflow_execution",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class AdaptiveExecutionStrategyCandidate(BaseModel):
    """One advisory V5.5 dynamic execution strategy candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    strategy_id: str = Field(min_length=1, max_length=180)
    strategy_kind: DynamicExecutionStrategyKind
    status: DynamicExecutionStrategyStatus
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_execution_strategy_id: str = Field(min_length=1, max_length=180)
    source_hybrid_workflow_candidate_id: str = Field(min_length=1, max_length=180)
    source_escalation_posture: str = Field(min_length=1, max_length=80)
    source_agent_activation_candidate_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=6,
    )
    source_cost_quality_candidate_id: str = Field(min_length=1, max_length=180)
    source_latency_candidate_id: str = Field(min_length=1, max_length=180)
    policy_direction: HybridRoutingPolicyDirection
    provider_sequence: tuple[ProviderId, ...] = Field(min_length=1, max_length=4)
    model_profile_sequence: tuple[str, ...] = Field(min_length=1, max_length=4)
    estimated_quality: EstimatedQualityBand
    estimated_cost: EstimatedCostBand
    estimated_latency: EstimatedLatencyBand
    unavailable_reason_codes: tuple[UnavailableReasonCode, ...] = Field(
        default_factory=tuple,
        max_length=9,
    )
    source_execution_strategy_score: int = Field(ge=0, le=2500)
    hybrid_adaptive_score: int = Field(ge=0, le=240)
    cost_quality_score: int = Field(ge=0, le=240)
    latency_score: int = Field(ge=0, le=240)
    agent_activation_score: int = Field(ge=0, le=240)
    escalation_pressure_score: int = Field(ge=0, le=240)
    strategy_bias: int = Field(ge=-300, le=120)
    dynamic_strategy_score: int = Field(ge=0, le=400)
    hitl_required: bool
    fallback_strategy_id: str | None = Field(default=None, max_length=180)
    fallback_reason_summary: str = Field(min_length=1, max_length=360)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    dynamic_execution_strategy_selection_implemented: Literal[True] = True
    adaptive_execution_intelligence_implemented: Literal[True] = True
    provider_intelligence_metadata_used: Literal[True] = True
    availability_awareness_metadata_used: Literal[True] = True
    execution_simulation_metadata_used: Literal[True] = True
    fallback_intelligence_implemented: Literal[True] = True
    strategy_application_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_runtime_probe_implemented: Literal[False] = False
    local_model_inventory_scan_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["adaptive_execution_strategy_candidate.v1"] = (
        ADAPTIVE_EXECUTION_STRATEGY_CANDIDATE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_contract(self) -> Self:
        if self.strategy_id != f"adaptive_execution_strategy::{self.strategy_kind}":
            raise ValueError("strategy_id must match strategy_kind")
        if len(self.model_profile_sequence) != len(self.provider_sequence):
            raise ValueError("model_profile_sequence must match provider_sequence")
        if self.dynamic_strategy_score != _dynamic_strategy_score(
            source_execution_strategy_score=self.source_execution_strategy_score,
            hybrid_adaptive_score=self.hybrid_adaptive_score,
            cost_quality_score=self.cost_quality_score,
            latency_score=self.latency_score,
            agent_activation_score=self.agent_activation_score,
            escalation_pressure_score=self.escalation_pressure_score,
            strategy_bias=self.strategy_bias,
        ):
            raise ValueError("dynamic_strategy_score must combine source scores")
        if self.unavailable_reason_codes and not self.hitl_required:
            raise ValueError("unavailable reasons require HITL")
        if self.status == "selected" and self.dynamic_strategy_score <= 0:
            raise ValueError("selected strategies require a positive score")
        if self.status == "guardrail" and self.strategy_kind != (
            "human_guarded_fallback_strategy"
        ):
            raise ValueError("only human guarded fallback can be a guardrail")
        return self


class AdaptiveExecutionStrategySelectionPlan(BaseModel):
    """Bounded V5.5 advisory dynamic execution strategy selection plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["dynamic_execution_strategy_selector"] = (
        "dynamic_execution_strategy_selector"
    )
    serialization_version: Literal["adaptive_execution_strategy_selection_plan.v1"] = (
        ADAPTIVE_EXECUTION_STRATEGY_SELECTION_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ADAPTIVE_EXECUTION_STRATEGY_SELECTION_AUTHORITY_BOUNDARY,
        max_length=1800,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_execution_strategy_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_hybrid_workflow_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_escalation_optimization_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_agent_activation_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_adaptive_cost_quality_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_adaptive_latency_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    provider_ids: tuple[ProviderId, ...] = Field(min_length=4, max_length=4)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    hybrid_policy_directions: tuple[HybridRoutingPolicyDirection, ...] = Field(
        min_length=4,
        max_length=4,
    )
    strategies: tuple[AdaptiveExecutionStrategyCandidate, ...] = Field(
        min_length=4,
        max_length=4,
    )
    strategy_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    selected_strategy_id: str = Field(min_length=1, max_length=180)
    applied_strategy_id: str | None = Field(default=None, max_length=180)
    selected_strategy_kind: DynamicExecutionStrategyKind
    selected_execution_mode_id: ExecutionModeId
    selected_policy_direction: HybridRoutingPolicyDirection
    selected_provider_sequence: tuple[ProviderId, ...] = Field(
        min_length=1,
        max_length=4,
    )
    selected_model_profile_sequence: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    fallback_strategy_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    guardrail_strategy_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    hitl_required_strategy_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    strategy_count: int = Field(ge=4, le=4)
    selected_strategy_count: int = Field(ge=1, le=1)
    selected_strategy_score: int = Field(ge=0, le=400)
    selected_strategy_hitl_required: bool
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    dynamic_execution_strategy_selection_implemented: Literal[True] = True
    adaptive_execution_intelligence_implemented: Literal[True] = True
    provider_intelligence_metadata_used: Literal[True] = True
    availability_awareness_metadata_used: Literal[True] = True
    execution_simulation_metadata_used: Literal[True] = True
    fallback_intelligence_implemented: Literal[True] = True
    strategy_application_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_runtime_probe_implemented: Literal[False] = False
    local_model_inventory_scan_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    graph_compilation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_strategies(self) -> Self:
        derived_strategy_ids = tuple(
            strategy.strategy_id for strategy in self.strategies
        )
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
        if self.selected_strategy_kind != selected_strategy.strategy_kind:
            raise ValueError("selected_strategy_kind must match selected strategy")
        if self.selected_execution_mode_id != selected_strategy.execution_mode_id:
            raise ValueError("selected_execution_mode_id must match selected strategy")
        if self.selected_policy_direction != selected_strategy.policy_direction:
            raise ValueError("selected_policy_direction must match selected strategy")
        if self.selected_provider_sequence != selected_strategy.provider_sequence:
            raise ValueError("selected_provider_sequence must match selected strategy")
        if self.selected_model_profile_sequence != (
            selected_strategy.model_profile_sequence
        ):
            raise ValueError(
                "selected_model_profile_sequence must match selected strategy"
            )
        if self.selected_strategy_count != 1:
            raise ValueError("selected_strategy_count must be one")
        if self.selected_strategy_score != selected_strategy.dynamic_strategy_score:
            raise ValueError("selected_strategy_score must match selected strategy")
        if self.selected_strategy_hitl_required != selected_strategy.hitl_required:
            raise ValueError(
                "selected_strategy_hitl_required must match selected strategy"
            )
        if self.applied_strategy_id is not None:
            raise ValueError("applied_strategy_id must remain unset")
        if self.fallback_strategy_ids != _strategy_ids_for_status(
            self.strategies,
            "fallback",
        ):
            raise ValueError("fallback_strategy_ids must match strategies")
        if self.guardrail_strategy_ids != _strategy_ids_for_status(
            self.strategies,
            "guardrail",
        ):
            raise ValueError("guardrail_strategy_ids must match strategies")
        if self.hitl_required_strategy_ids != tuple(
            strategy.strategy_id
            for strategy in self.strategies
            if strategy.hitl_required
        ):
            raise ValueError("hitl_required_strategy_ids must match strategies")
        return self


def select_dynamic_execution_strategy(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    execution_strategy: ExecutionStrategySelection | None = None,
    hybrid_workflow: HybridWorkflowOptimizationPlan | None = None,
    escalation: EscalationOptimizationPlan | None = None,
    agent_activation: AgentActivationOptimizationPlan | None = None,
    cost_quality: AdaptiveCostQualityPlan | None = None,
    latency: AdaptiveLatencyPlan | None = None,
) -> AdaptiveExecutionStrategySelectionPlan:
    """Select advisory dynamic execution strategy metadata without applying it."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    hybrid_plan = hybrid_workflow or optimize_hybrid_workflow(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=execution_mode_id,
    )
    normalized_mode = str(
        execution_mode_id or hybrid_plan.candidates[0].execution_mode_id
    )
    execution_modes = routing_execution_mode_registry()
    if normalized_mode not in execution_modes.execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")

    strategy_plan = execution_strategy or select_execution_strategy()
    escalation_plan = escalation or optimize_escalation_policy(
        task_type=hybrid_plan.task_type,
        route=route_name,
        execution_mode_id=normalized_mode,
        hybrid_workflow=hybrid_plan,
    )
    agent_plan = agent_activation or optimize_agent_activation(
        route=route_name,
        task_type=hybrid_plan.task_type,
        execution_mode_id=normalized_mode,
        escalation=escalation_plan,
    )
    cost_quality_plan = cost_quality or optimize_adaptive_cost_quality(
        route=route_name,
        task_type=hybrid_plan.task_type,
        execution_mode_id=normalized_mode,
        hybrid_workflow=hybrid_plan,
        agent_activation=agent_plan,
    )
    latency_plan = latency or optimize_adaptive_latency(
        route=route_name,
        task_type=hybrid_plan.task_type,
        execution_mode_id=normalized_mode,
        hybrid_workflow=hybrid_plan,
        agent_activation=agent_plan,
    )

    candidates = _strategy_candidates(
        route_name=route_name,
        task_type=hybrid_plan.task_type,
        execution_mode_id=normalized_mode,  # type: ignore[arg-type]
        execution_strategy=strategy_plan,
        hybrid_workflow=hybrid_plan,
        escalation=escalation_plan,
        agent_activation=agent_plan,
        cost_quality=cost_quality_plan,
        latency=latency_plan,
    )
    selected = max(candidates, key=lambda candidate: candidate.dynamic_strategy_score)
    strategies = tuple(
        candidate.model_copy(
            update={
                "status": (
                    "selected"
                    if candidate.strategy_id == selected.strategy_id
                    else candidate.status
                ),
            }
        )
        for candidate in candidates
    )
    selected = next(
        strategy for strategy in strategies if strategy.status == "selected"
    )

    return AdaptiveExecutionStrategySelectionPlan(
        route_name=route_name,
        task_type=hybrid_plan.task_type,
        source_execution_strategy_serialization_version=(
            strategy_plan.serialization_version
        ),
        source_hybrid_workflow_serialization_version=hybrid_plan.serialization_version,
        source_escalation_optimization_serialization_version=(
            escalation_plan.serialization_version
        ),
        source_agent_activation_serialization_version=agent_plan.serialization_version,
        source_adaptive_cost_quality_serialization_version=(
            cost_quality_plan.serialization_version
        ),
        source_adaptive_latency_serialization_version=latency_plan.serialization_version,
        provider_ids=hybrid_plan.provider_ids,
        execution_mode_ids=execution_modes.execution_mode_ids,
        hybrid_policy_directions=hybrid_plan.hybrid_policy_directions,
        strategies=strategies,
        strategy_ids=tuple(strategy.strategy_id for strategy in strategies),
        selected_strategy_id=selected.strategy_id,
        applied_strategy_id=None,
        selected_strategy_kind=selected.strategy_kind,
        selected_execution_mode_id=selected.execution_mode_id,
        selected_policy_direction=selected.policy_direction,
        selected_provider_sequence=selected.provider_sequence,
        selected_model_profile_sequence=selected.model_profile_sequence,
        fallback_strategy_ids=_strategy_ids_for_status(strategies, "fallback"),
        guardrail_strategy_ids=_strategy_ids_for_status(strategies, "guardrail"),
        hitl_required_strategy_ids=tuple(
            strategy.strategy_id for strategy in strategies if strategy.hitl_required
        ),
        strategy_count=len(strategies),
        selected_strategy_count=1,
        selected_strategy_score=selected.dynamic_strategy_score,
        selected_strategy_hitl_required=selected.hitl_required,
        advisory_actions=_plan_actions(selected),
    )


def adaptive_execution_strategy_by_id(
    strategy_id: str,
    plan: AdaptiveExecutionStrategySelectionPlan | None = None,
) -> AdaptiveExecutionStrategyCandidate | None:
    """Return one dynamic execution strategy candidate without applying it."""

    source_plan = plan or select_dynamic_execution_strategy()
    for strategy in source_plan.strategies:
        if strategy.strategy_id == strategy_id:
            return strategy
    return None


def adaptive_execution_strategies_for_status(
    status: DynamicExecutionStrategyStatus,
    plan: AdaptiveExecutionStrategySelectionPlan | None = None,
) -> tuple[AdaptiveExecutionStrategyCandidate, ...]:
    """Return dynamic execution strategy candidates by advisory status."""

    source_plan = plan or select_dynamic_execution_strategy()
    return tuple(
        strategy for strategy in source_plan.strategies if strategy.status == status
    )


def _strategy_candidates(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    execution_strategy: ExecutionStrategySelection,
    hybrid_workflow: HybridWorkflowOptimizationPlan,
    escalation: EscalationOptimizationPlan,
    agent_activation: AgentActivationOptimizationPlan,
    cost_quality: AdaptiveCostQualityPlan,
    latency: AdaptiveLatencyPlan,
) -> tuple[AdaptiveExecutionStrategyCandidate, ...]:
    selected_execution_strategy = next(
        strategy
        for strategy in execution_strategy.strategies
        if strategy.strategy_id == execution_strategy.selected_strategy_id
    )
    hybrid_recommended = _hybrid_candidate(
        hybrid_workflow,
        hybrid_workflow.recommended_candidate_id,
    )
    hybrid_fallback = _hybrid_candidate(
        hybrid_workflow,
        hybrid_workflow.fallback.fallback_candidate_id,
    )
    return (
        _strategy_candidate(
            kind="balanced_hybrid_strategy",
            status="fallback",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            source_execution_strategy_id=selected_execution_strategy.strategy_id,
            source_execution_strategy_score=execution_strategy.selected_strategy_score,
            hybrid_candidate=hybrid_recommended,
            escalation=escalation,
            agent_activation=agent_activation,
            cost_quality=cost_quality,
            latency=latency,
            strategy_bias=35,
            fallback_strategy_id=(
                "adaptive_execution_strategy::human_guarded_fallback_strategy"
            ),
            fallback_reason_summary=hybrid_workflow.fallback.reason_summary,
        ),
        _strategy_candidate(
            kind="quality_priority_strategy",
            status="fallback",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            source_execution_strategy_id=selected_execution_strategy.strategy_id,
            source_execution_strategy_score=execution_strategy.selected_strategy_score,
            hybrid_candidate=hybrid_recommended,
            escalation=escalation,
            agent_activation=agent_activation,
            cost_quality=cost_quality,
            latency=latency,
            strategy_bias=20,
            fallback_strategy_id=(
                "adaptive_execution_strategy::balanced_hybrid_strategy"
            ),
            fallback_reason_summary=(
                "Use balanced hybrid posture if quality priority is too expensive, "
                "slow, or unavailable."
            ),
        ),
        _strategy_candidate(
            kind="latency_priority_strategy",
            status="fallback",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            source_execution_strategy_id=selected_execution_strategy.strategy_id,
            source_execution_strategy_score=execution_strategy.selected_strategy_score,
            hybrid_candidate=hybrid_recommended,
            escalation=escalation,
            agent_activation=agent_activation,
            cost_quality=cost_quality,
            latency=latency,
            strategy_bias=-20,
            fallback_strategy_id=(
                "adaptive_execution_strategy::balanced_hybrid_strategy"
            ),
            fallback_reason_summary=(
                "Use balanced hybrid posture if latency metadata is guarded or "
                "insufficient for review."
            ),
        ),
        _strategy_candidate(
            kind="human_guarded_fallback_strategy",
            status="guardrail",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            source_execution_strategy_id=selected_execution_strategy.strategy_id,
            source_execution_strategy_score=execution_strategy.selected_strategy_score,
            hybrid_candidate=hybrid_fallback,
            escalation=escalation,
            agent_activation=agent_activation,
            cost_quality=cost_quality,
            latency=latency,
            strategy_bias=-220,
            fallback_strategy_id=None,
            fallback_reason_summary=(
                "Keep a human guarded fallback available when strategy selection "
                "cannot be safely reviewed."
            ),
        ),
    )


def _strategy_candidate(
    *,
    kind: DynamicExecutionStrategyKind,
    status: DynamicExecutionStrategyStatus,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    source_execution_strategy_id: str,
    source_execution_strategy_score: int,
    hybrid_candidate: HybridWorkflowOptimizationCandidate,
    escalation: EscalationOptimizationPlan,
    agent_activation: AgentActivationOptimizationPlan,
    cost_quality: AdaptiveCostQualityPlan,
    latency: AdaptiveLatencyPlan,
    strategy_bias: int,
    fallback_strategy_id: str | None,
    fallback_reason_summary: str,
) -> AdaptiveExecutionStrategyCandidate:
    source_agent_ids = (
        agent_activation.recommended_candidate_ids
        or agent_activation.hitl_required_candidate_ids
        or (agent_activation.candidates[0].candidate_id,)
    )
    hitl_required = (
        bool(hybrid_candidate.unavailable_reason_codes)
        or hybrid_candidate.hitl_required
        or escalation.optimized_escalation_posture == "requires_hitl"
        or bool(agent_activation.hitl_required_candidate_ids)
        or cost_quality.hitl_required_candidate_count > 0
        or latency.hitl_required_candidate_count > 0
    )
    dynamic_score = _dynamic_strategy_score(
        source_execution_strategy_score=source_execution_strategy_score,
        hybrid_adaptive_score=hybrid_candidate.adaptive_score,
        cost_quality_score=cost_quality.recommended_adaptive_score,
        latency_score=latency.recommended_adaptive_latency_score,
        agent_activation_score=agent_activation.highest_activation_score,
        escalation_pressure_score=escalation.highest_escalation_score,
        strategy_bias=strategy_bias,
    )
    return AdaptiveExecutionStrategyCandidate(
        strategy_id=f"adaptive_execution_strategy::{kind}",
        strategy_kind=kind,
        status=status,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_execution_strategy_id=source_execution_strategy_id,
        source_hybrid_workflow_candidate_id=hybrid_candidate.candidate_id,
        source_escalation_posture=escalation.optimized_escalation_posture,
        source_agent_activation_candidate_ids=source_agent_ids[:6],
        source_cost_quality_candidate_id=cost_quality.recommended_candidate_id,
        source_latency_candidate_id=latency.recommended_candidate_id,
        policy_direction=hybrid_candidate.policy_direction,
        provider_sequence=hybrid_candidate.provider_sequence,
        model_profile_sequence=hybrid_candidate.model_profile_sequence,
        estimated_quality=hybrid_candidate.estimated_quality,
        estimated_cost=hybrid_candidate.estimated_cost,
        estimated_latency=hybrid_candidate.estimated_latency,
        unavailable_reason_codes=hybrid_candidate.unavailable_reason_codes,
        source_execution_strategy_score=source_execution_strategy_score,
        hybrid_adaptive_score=hybrid_candidate.adaptive_score,
        cost_quality_score=cost_quality.recommended_adaptive_score,
        latency_score=latency.recommended_adaptive_latency_score,
        agent_activation_score=agent_activation.highest_activation_score,
        escalation_pressure_score=escalation.highest_escalation_score,
        strategy_bias=strategy_bias,
        dynamic_strategy_score=dynamic_score,
        hitl_required=hitl_required,
        fallback_strategy_id=fallback_strategy_id,
        fallback_reason_summary=fallback_reason_summary,
        advisory_actions=_candidate_actions(kind),
        evidence=(
            f"execution_strategy:{source_execution_strategy_id}",
            f"hybrid_candidate:{hybrid_candidate.candidate_id}",
            f"cost_quality:{cost_quality.recommended_candidate_id}",
            f"latency:{latency.recommended_candidate_id}",
            f"escalation_posture:{escalation.optimized_escalation_posture}",
            f"agent_candidates:{len(source_agent_ids)}",
        ),
    )


def _dynamic_strategy_score(
    *,
    source_execution_strategy_score: int,
    hybrid_adaptive_score: int,
    cost_quality_score: int,
    latency_score: int,
    agent_activation_score: int,
    escalation_pressure_score: int,
    strategy_bias: int,
) -> int:
    return min(
        400,
        max(
            0,
            source_execution_strategy_score
            + hybrid_adaptive_score
            + cost_quality_score // 2
            + latency_score // 2
            + agent_activation_score // 4
            - escalation_pressure_score // 5
            + strategy_bias,
        ),
    )


def _hybrid_candidate(
    plan: HybridWorkflowOptimizationPlan,
    candidate_id: str,
) -> HybridWorkflowOptimizationCandidate:
    for candidate in plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise ValueError("required hybrid workflow candidate is missing")


def _strategy_ids_for_status(
    strategies: tuple[AdaptiveExecutionStrategyCandidate, ...],
    status: DynamicExecutionStrategyStatus,
) -> tuple[str, ...]:
    return tuple(
        strategy.strategy_id for strategy in strategies if strategy.status == status
    )


def _candidate_actions(
    kind: DynamicExecutionStrategyKind,
) -> tuple[str, ...]:
    if kind == "human_guarded_fallback_strategy":
        return (
            "Keep human guarded fallback strategy available for review.",
            "Do not apply fallback, emit HITL, execute workflows, or mutate outputs.",
        )
    return (
        f"Surface {kind} as advisory dynamic execution strategy metadata.",
        "Preserve provider, model, local runtime, agent, workflow, budget, storage, and output boundaries.",
    )


def _plan_actions(
    selected: AdaptiveExecutionStrategyCandidate,
) -> tuple[str, ...]:
    return (
        f"Expose {selected.strategy_id} as selected advisory strategy metadata.",
        "Leave applied_strategy_id unset until an explicit runtime contract exists.",
        "Preserve provider/model routing, local runtime, agent, workflow, budget, storage, and output boundaries.",
    )


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
