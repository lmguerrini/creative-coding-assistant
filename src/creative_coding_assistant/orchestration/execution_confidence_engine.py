"""V5.5 advisory execution confidence intelligence."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_cost_quality_optimizer import (
    AdaptiveCostQualityPlan,
    optimize_adaptive_cost_quality,
)
from creative_coding_assistant.orchestration.adaptive_execution_strategy_selection import (
    AdaptiveExecutionStrategySelectionPlan,
    select_dynamic_execution_strategy,
)
from creative_coding_assistant.orchestration.adaptive_latency_optimizer import (
    AdaptiveLatencyPlan,
    optimize_adaptive_latency,
)
from creative_coding_assistant.orchestration.dynamic_agent_allocation import (
    DynamicAgentAllocationPlan,
    allocate_dynamic_agents,
)
from creative_coding_assistant.orchestration.dynamic_resource_allocation import (
    DynamicResourceAllocationCandidate,
    DynamicResourceAllocationPlan,
    allocate_dynamic_resources,
    dynamic_resource_allocation_by_id,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    HybridRoutingPolicyDirection,
    ProviderId,
    TaskRoutingType,
    UnavailableReasonCode,
    routing_execution_mode_registry,
)
from creative_coding_assistant.orchestration.workflow_self_tuning_policies import (
    WorkflowSelfTuningPolicy,
    WorkflowSelfTuningPolicyPlan,
    plan_workflow_self_tuning_policies,
    workflow_self_tuning_policy_by_id,
)

ExecutionConfidenceSignalKind = Literal[
    "strategy_execution_confidence",
    "agent_allocation_confidence",
    "resource_capacity_confidence",
    "workflow_self_tuning_confidence",
    "provider_availability_confidence",
    "fallback_safety_confidence",
]
ExecutionConfidenceStatus = Literal["ready", "review_required", "guardrail"]
ExecutionConfidenceBand = Literal["high", "moderate", "low", "guarded"]

EXECUTION_CONFIDENCE_SIGNAL_SERIALIZATION_VERSION = (
    "execution_confidence_signal.v1"
)
EXECUTION_CONFIDENCE_PLAN_SERIALIZATION_VERSION = (
    "execution_confidence_plan.v1"
)
EXECUTION_CONFIDENCE_AUTHORITY_BOUNDARY = (
    "V5.5 execution confidence combines advisory dynamic execution strategy, "
    "dynamic agent allocation, dynamic resource allocation, workflow "
    "self-tuning, adaptive cost/quality, and adaptive latency metadata into "
    "inspectable execution confidence posture only; it does not apply "
    "confidence decisions, evaluate generated output, change provider or "
    "model routing, execute providers, probe local runtimes, scan or download "
    "local models, invoke agents, allocate resources, apply self-tuning, emit "
    "HITL requests, enforce budgets, control workflows, mutate workflow "
    "graphs, trigger retries, mutate prompts, write storage, modify generated "
    "output, or apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "confidence_application",
    "generated_output_evaluation",
    "provider_or_model_routing",
    "automatic_provider_switching",
    "automatic_model_switching",
    "provider_execution",
    "local_runtime_probe",
    "local_model_inventory_scan",
    "automatic_model_download",
    "agent_invocation",
    "resource_allocation",
    "workflow_self_tuning_application",
    "human_review_request",
    "hitl_request_emission",
    "budget_enforcement",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class ExecutionConfidenceSignal(BaseModel):
    """One advisory execution confidence signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=180)
    signal_kind: ExecutionConfidenceSignalKind
    status: ExecutionConfidenceStatus
    confidence_band: ExecutionConfidenceBand
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    source_dynamic_strategy_id: str = Field(min_length=1, max_length=180)
    source_dynamic_agent_allocation_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    source_dynamic_resource_allocation_id: str = Field(min_length=1, max_length=180)
    source_self_tuning_policy_id: str = Field(min_length=1, max_length=180)
    source_cost_quality_candidate_id: str = Field(min_length=1, max_length=180)
    source_latency_candidate_id: str = Field(min_length=1, max_length=180)
    provider_sequence: tuple[ProviderId, ...] = Field(min_length=1, max_length=4)
    model_profile_sequence: tuple[str, ...] = Field(min_length=1, max_length=4)
    hybrid_policy_direction: HybridRoutingPolicyDirection
    unavailable_reason_codes: tuple[UnavailableReasonCode, ...] = Field(
        default_factory=tuple,
        max_length=9,
    )
    dynamic_strategy_score: int = Field(ge=0, le=400)
    agent_allocation_score: int = Field(ge=0, le=320)
    dynamic_resource_score: int = Field(ge=0, le=500)
    self_tuning_score: int = Field(ge=0, le=600)
    adaptive_cost_quality_score: int = Field(ge=0, le=240)
    adaptive_latency_score: int = Field(ge=0, le=240)
    availability_penalty: int = Field(ge=0, le=120)
    guardrail_penalty: int = Field(ge=0, le=120)
    execution_confidence_score: int = Field(ge=0, le=100)
    hitl_required: bool
    fallback_safety_summary: str = Field(min_length=1, max_length=360)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    execution_confidence_engine_implemented: Literal[True] = True
    execution_confidence_metadata_implemented: Literal[True] = True
    provider_intelligence_metadata_used: Literal[True] = True
    availability_awareness_metadata_used: Literal[True] = True
    manual_assisted_auto_mode_metadata_used: Literal[True] = True
    hybrid_transition_metadata_used: Literal[True] = True
    task_aware_category_metadata_used: Literal[True] = True
    execution_simulation_metadata_used: Literal[True] = True
    fallback_safety_metadata_used: Literal[True] = True
    confidence_application_implemented: Literal[False] = False
    generated_output_evaluation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_runtime_probe_implemented: Literal[False] = False
    local_model_inventory_scan_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    self_tuning_application_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["execution_confidence_signal.v1"] = (
        EXECUTION_CONFIDENCE_SIGNAL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_contract(self) -> Self:
        if self.signal_id != f"execution_confidence::{self.signal_kind}":
            raise ValueError("signal_id must match signal_kind")
        if len(self.model_profile_sequence) != len(self.provider_sequence):
            raise ValueError("model_profile_sequence must match provider_sequence")
        if self.execution_confidence_score != _execution_confidence_score(
            dynamic_strategy_score=self.dynamic_strategy_score,
            agent_allocation_score=self.agent_allocation_score,
            dynamic_resource_score=self.dynamic_resource_score,
            self_tuning_score=self.self_tuning_score,
            adaptive_cost_quality_score=self.adaptive_cost_quality_score,
            adaptive_latency_score=self.adaptive_latency_score,
            availability_penalty=self.availability_penalty,
            guardrail_penalty=self.guardrail_penalty,
        ):
            raise ValueError("execution_confidence_score must combine source scores")
        if self.confidence_band != _confidence_band(
            self.execution_confidence_score,
            self.status,
        ):
            raise ValueError("confidence_band must match score and status")
        if self.status == "guardrail" and not self.hitl_required:
            raise ValueError("guardrail confidence signals require HITL posture")
        return self


class ExecutionConfidencePlan(BaseModel):
    """Bounded V5.5 advisory execution confidence plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["execution_confidence_engine"] = "execution_confidence_engine"
    serialization_version: Literal["execution_confidence_plan.v1"] = (
        EXECUTION_CONFIDENCE_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=EXECUTION_CONFIDENCE_AUTHORITY_BOUNDARY,
        max_length=1900,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_dynamic_execution_strategy_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_dynamic_agent_allocation_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_dynamic_resource_allocation_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_workflow_self_tuning_serialization_version: str = Field(
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
    signals: tuple[ExecutionConfidenceSignal, ...] = Field(
        min_length=6,
        max_length=6,
    )
    signal_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    high_confidence_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    moderate_confidence_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    low_confidence_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    guarded_confidence_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    review_required_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    guardrail_signal_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    hitl_required_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    applied_confidence_signal_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    signal_count: int = Field(ge=6, le=6)
    review_required_signal_count: int = Field(ge=0, le=6)
    guardrail_signal_count: int = Field(ge=0, le=6)
    hitl_required_signal_count: int = Field(ge=0, le=6)
    overall_confidence_score: int = Field(ge=0, le=100)
    overall_confidence_band: ExecutionConfidenceBand
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    execution_confidence_engine_implemented: Literal[True] = True
    execution_confidence_metadata_implemented: Literal[True] = True
    provider_intelligence_metadata_used: Literal[True] = True
    availability_awareness_metadata_used: Literal[True] = True
    manual_assisted_auto_mode_metadata_used: Literal[True] = True
    hybrid_transition_metadata_used: Literal[True] = True
    task_aware_category_metadata_used: Literal[True] = True
    execution_simulation_metadata_used: Literal[True] = True
    fallback_safety_metadata_used: Literal[True] = True
    confidence_application_implemented: Literal[False] = False
    generated_output_evaluation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_runtime_probe_implemented: Literal[False] = False
    local_model_inventory_scan_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    self_tuning_application_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_signals(self) -> Self:
        derived_signal_ids = tuple(signal.signal_id for signal in self.signals)
        if len(set(derived_signal_ids)) != len(derived_signal_ids):
            raise ValueError("signal_ids must be unique")
        if self.signal_ids != derived_signal_ids:
            raise ValueError("signal_ids must match signals")
        if self.signal_count != len(self.signals):
            raise ValueError("signal_count must match signals")
        if self.high_confidence_signal_ids != _signal_ids_for_band(
            self.signals,
            "high",
        ):
            raise ValueError("high_confidence_signal_ids must match signals")
        if self.moderate_confidence_signal_ids != _signal_ids_for_band(
            self.signals,
            "moderate",
        ):
            raise ValueError("moderate_confidence_signal_ids must match signals")
        if self.low_confidence_signal_ids != _signal_ids_for_band(self.signals, "low"):
            raise ValueError("low_confidence_signal_ids must match signals")
        if self.guarded_confidence_signal_ids != _signal_ids_for_band(
            self.signals,
            "guarded",
        ):
            raise ValueError("guarded_confidence_signal_ids must match signals")
        if self.review_required_signal_ids != _signal_ids_for_status(
            self.signals,
            "review_required",
        ):
            raise ValueError("review_required_signal_ids must match signals")
        if self.guardrail_signal_ids != _signal_ids_for_status(
            self.signals,
            "guardrail",
        ):
            raise ValueError("guardrail_signal_ids must match signals")
        if self.hitl_required_signal_ids != tuple(
            signal.signal_id for signal in self.signals if signal.hitl_required
        ):
            raise ValueError("hitl_required_signal_ids must match signals")
        if self.applied_confidence_signal_ids:
            raise ValueError("applied_confidence_signal_ids must remain empty")
        if self.review_required_signal_count != len(self.review_required_signal_ids):
            raise ValueError("review_required_signal_count must match signals")
        if self.guardrail_signal_count != len(self.guardrail_signal_ids):
            raise ValueError("guardrail_signal_count must match signals")
        if self.hitl_required_signal_count != len(self.hitl_required_signal_ids):
            raise ValueError("hitl_required_signal_count must match signals")
        if self.overall_confidence_score != _overall_confidence_score(self.signals):
            raise ValueError("overall_confidence_score must match signals")
        if self.overall_confidence_band != _overall_confidence_band(self.signals):
            raise ValueError("overall_confidence_band must match signals")
        for signal in self.signals:
            if signal.route_name != self.route_name:
                raise ValueError("signal route_name must match plan")
        return self


def evaluate_execution_confidence(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    dynamic_strategy: AdaptiveExecutionStrategySelectionPlan | None = None,
    agent_allocation: DynamicAgentAllocationPlan | None = None,
    dynamic_resource_allocation: DynamicResourceAllocationPlan | None = None,
    self_tuning: WorkflowSelfTuningPolicyPlan | None = None,
    cost_quality: AdaptiveCostQualityPlan | None = None,
    latency: AdaptiveLatencyPlan | None = None,
) -> ExecutionConfidencePlan:
    """Evaluate advisory execution confidence without applying decisions."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    strategy_plan = dynamic_strategy or select_dynamic_execution_strategy(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=execution_mode_id,
    )
    normalized_mode = str(execution_mode_id or strategy_plan.selected_execution_mode_id)
    execution_modes = routing_execution_mode_registry()
    if normalized_mode not in execution_modes.execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")

    agent_plan = agent_allocation or allocate_dynamic_agents(
        route=route_name,
        task_type=strategy_plan.task_type,
        execution_mode_id=normalized_mode,
        dynamic_strategy=strategy_plan,
    )
    cost_quality_plan = cost_quality or optimize_adaptive_cost_quality(
        route=route_name,
        task_type=strategy_plan.task_type,
        execution_mode_id=normalized_mode,
    )
    latency_plan = latency or optimize_adaptive_latency(
        route=route_name,
        task_type=strategy_plan.task_type,
        execution_mode_id=normalized_mode,
    )
    resource_plan = dynamic_resource_allocation or allocate_dynamic_resources(
        route=route_name,
        task_type=strategy_plan.task_type,
        execution_mode_id=normalized_mode,
        agent_allocation=agent_plan,
        cost_quality=cost_quality_plan,
        latency=latency_plan,
    )
    tuning_plan = self_tuning or plan_workflow_self_tuning_policies(
        route=route_name,
        task_type=strategy_plan.task_type,
        execution_mode_id=normalized_mode,
        dynamic_resource_allocation=resource_plan,
        dynamic_strategy=strategy_plan,
    )
    signals = _signals(
        route_name=route_name,
        task_type=strategy_plan.task_type,
        execution_mode_id=normalized_mode,  # type: ignore[arg-type]
        dynamic_strategy=strategy_plan,
        agent_allocation=agent_plan,
        dynamic_resource_allocation=resource_plan,
        self_tuning=tuning_plan,
        cost_quality=cost_quality_plan,
        latency=latency_plan,
    )
    return ExecutionConfidencePlan(
        route_name=route_name,
        task_type=strategy_plan.task_type,
        source_dynamic_execution_strategy_serialization_version=(
            strategy_plan.serialization_version
        ),
        source_dynamic_agent_allocation_serialization_version=(
            agent_plan.serialization_version
        ),
        source_dynamic_resource_allocation_serialization_version=(
            resource_plan.serialization_version
        ),
        source_workflow_self_tuning_serialization_version=(
            tuning_plan.serialization_version
        ),
        source_adaptive_cost_quality_serialization_version=(
            cost_quality_plan.serialization_version
        ),
        source_adaptive_latency_serialization_version=latency_plan.serialization_version,
        provider_ids=strategy_plan.provider_ids,
        execution_mode_ids=execution_modes.execution_mode_ids,
        hybrid_policy_directions=strategy_plan.hybrid_policy_directions,
        signals=signals,
        signal_ids=tuple(signal.signal_id for signal in signals),
        high_confidence_signal_ids=_signal_ids_for_band(signals, "high"),
        moderate_confidence_signal_ids=_signal_ids_for_band(signals, "moderate"),
        low_confidence_signal_ids=_signal_ids_for_band(signals, "low"),
        guarded_confidence_signal_ids=_signal_ids_for_band(signals, "guarded"),
        review_required_signal_ids=_signal_ids_for_status(signals, "review_required"),
        guardrail_signal_ids=_signal_ids_for_status(signals, "guardrail"),
        hitl_required_signal_ids=tuple(
            signal.signal_id for signal in signals if signal.hitl_required
        ),
        applied_confidence_signal_ids=(),
        signal_count=len(signals),
        review_required_signal_count=len(
            _signal_ids_for_status(signals, "review_required")
        ),
        guardrail_signal_count=len(_signal_ids_for_status(signals, "guardrail")),
        hitl_required_signal_count=sum(1 for signal in signals if signal.hitl_required),
        overall_confidence_score=_overall_confidence_score(signals),
        overall_confidence_band=_overall_confidence_band(signals),
        advisory_actions=_plan_actions(signals),
    )


def execution_confidence_signal_by_id(
    signal_id: str,
    plan: ExecutionConfidencePlan | None = None,
) -> ExecutionConfidenceSignal | None:
    """Return one execution confidence signal without applying it."""

    source_plan = plan or evaluate_execution_confidence()
    for signal in source_plan.signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def execution_confidence_signals_for_band(
    band: ExecutionConfidenceBand,
    plan: ExecutionConfidencePlan | None = None,
) -> tuple[ExecutionConfidenceSignal, ...]:
    """Return execution confidence signals by advisory confidence band."""

    source_plan = plan or evaluate_execution_confidence()
    return tuple(signal for signal in source_plan.signals if signal.confidence_band == band)


def _signals(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    dynamic_strategy: AdaptiveExecutionStrategySelectionPlan,
    agent_allocation: DynamicAgentAllocationPlan,
    dynamic_resource_allocation: DynamicResourceAllocationPlan,
    self_tuning: WorkflowSelfTuningPolicyPlan,
    cost_quality: AdaptiveCostQualityPlan,
    latency: AdaptiveLatencyPlan,
) -> tuple[ExecutionConfidenceSignal, ...]:
    return (
        _signal(
            kind="strategy_execution_confidence",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            dynamic_strategy=dynamic_strategy,
            agent_allocation=agent_allocation,
            dynamic_resource_allocation=dynamic_resource_allocation,
            resource_allocation_id=(
                "dynamic_resource_allocation::throughput_capacity_utilization"
            ),
            self_tuning=self_tuning,
            self_tuning_policy_id="workflow_self_tuning::load_self_tuning_policy",
            cost_quality=cost_quality,
            latency=latency,
            availability_multiplier=5,
            guardrail_penalty=18,
        ),
        _signal(
            kind="agent_allocation_confidence",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            dynamic_strategy=dynamic_strategy,
            agent_allocation=agent_allocation,
            dynamic_resource_allocation=dynamic_resource_allocation,
            resource_allocation_id=(
                "dynamic_resource_allocation::reasoning_budget_utilization"
            ),
            self_tuning=self_tuning,
            self_tuning_policy_id="workflow_self_tuning::retry_self_tuning_policy",
            cost_quality=cost_quality,
            latency=latency,
            availability_multiplier=4,
            guardrail_penalty=14,
        ),
        _signal(
            kind="resource_capacity_confidence",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            dynamic_strategy=dynamic_strategy,
            agent_allocation=agent_allocation,
            dynamic_resource_allocation=dynamic_resource_allocation,
            resource_allocation_id=(
                "dynamic_resource_allocation::benchmark_workload_utilization"
            ),
            self_tuning=self_tuning,
            self_tuning_policy_id="workflow_self_tuning::resource_self_tuning_policy",
            cost_quality=cost_quality,
            latency=latency,
            availability_multiplier=4,
            guardrail_penalty=10,
        ),
        _signal(
            kind="workflow_self_tuning_confidence",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            dynamic_strategy=dynamic_strategy,
            agent_allocation=agent_allocation,
            dynamic_resource_allocation=dynamic_resource_allocation,
            resource_allocation_id=(
                "dynamic_resource_allocation::profiling_scope_utilization"
            ),
            self_tuning=self_tuning,
            self_tuning_policy_id="workflow_self_tuning::resource_self_tuning_policy",
            cost_quality=cost_quality,
            latency=latency,
            availability_multiplier=4,
            guardrail_penalty=12,
        ),
        _signal(
            kind="provider_availability_confidence",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            dynamic_strategy=dynamic_strategy,
            agent_allocation=agent_allocation,
            dynamic_resource_allocation=dynamic_resource_allocation,
            resource_allocation_id="dynamic_resource_allocation::runtime_resource_boundary",
            self_tuning=self_tuning,
            self_tuning_policy_id="workflow_self_tuning::strategy_guardrail_policy",
            cost_quality=cost_quality,
            latency=latency,
            availability_multiplier=8,
            guardrail_penalty=35,
        ),
        _signal(
            kind="fallback_safety_confidence",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            dynamic_strategy=dynamic_strategy,
            agent_allocation=agent_allocation,
            dynamic_resource_allocation=dynamic_resource_allocation,
            resource_allocation_id="dynamic_resource_allocation::runtime_resource_boundary",
            self_tuning=self_tuning,
            self_tuning_policy_id="workflow_self_tuning::strategy_guardrail_policy",
            cost_quality=cost_quality,
            latency=latency,
            availability_multiplier=6,
            guardrail_penalty=30,
        ),
    )


def _signal(
    *,
    kind: ExecutionConfidenceSignalKind,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    dynamic_strategy: AdaptiveExecutionStrategySelectionPlan,
    agent_allocation: DynamicAgentAllocationPlan,
    dynamic_resource_allocation: DynamicResourceAllocationPlan,
    resource_allocation_id: str,
    self_tuning: WorkflowSelfTuningPolicyPlan,
    self_tuning_policy_id: str,
    cost_quality: AdaptiveCostQualityPlan,
    latency: AdaptiveLatencyPlan,
    availability_multiplier: int,
    guardrail_penalty: int,
) -> ExecutionConfidenceSignal:
    selected_strategy = next(
        strategy
        for strategy in dynamic_strategy.strategies
        if strategy.strategy_id == dynamic_strategy.selected_strategy_id
    )
    resource_allocation = _required_resource_allocation(
        resource_allocation_id,
        dynamic_resource_allocation,
    )
    policy = _required_self_tuning_policy(self_tuning_policy_id, self_tuning)
    availability_penalty = (
        len(selected_strategy.unavailable_reason_codes) * availability_multiplier
    )
    status = _status(kind, resource_allocation, policy)
    score = _execution_confidence_score(
        dynamic_strategy_score=dynamic_strategy.selected_strategy_score,
        agent_allocation_score=agent_allocation.highest_allocation_score,
        dynamic_resource_score=resource_allocation.dynamic_resource_score,
        self_tuning_score=policy.self_tuning_score,
        adaptive_cost_quality_score=cost_quality.recommended_adaptive_score,
        adaptive_latency_score=latency.recommended_adaptive_latency_score,
        availability_penalty=availability_penalty,
        guardrail_penalty=guardrail_penalty,
    )
    hitl_required = (
        dynamic_strategy.selected_strategy_hitl_required
        or agent_allocation.hitl_required_allocation_count > 0
        or resource_allocation.hitl_required
        or policy.hitl_required
        or status == "guardrail"
    )
    return ExecutionConfidenceSignal(
        signal_id=f"execution_confidence::{kind}",
        signal_kind=kind,
        status=status,
        confidence_band=_confidence_band(score, status),
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        source_dynamic_strategy_id=dynamic_strategy.selected_strategy_id,
        source_dynamic_agent_allocation_ids=_source_agent_ids(agent_allocation),
        source_dynamic_resource_allocation_id=resource_allocation.allocation_id,
        source_self_tuning_policy_id=policy.policy_id,
        source_cost_quality_candidate_id=cost_quality.recommended_candidate_id,
        source_latency_candidate_id=latency.recommended_candidate_id,
        provider_sequence=dynamic_strategy.selected_provider_sequence,
        model_profile_sequence=dynamic_strategy.selected_model_profile_sequence,
        hybrid_policy_direction=dynamic_strategy.selected_policy_direction,
        unavailable_reason_codes=selected_strategy.unavailable_reason_codes,
        dynamic_strategy_score=dynamic_strategy.selected_strategy_score,
        agent_allocation_score=agent_allocation.highest_allocation_score,
        dynamic_resource_score=resource_allocation.dynamic_resource_score,
        self_tuning_score=policy.self_tuning_score,
        adaptive_cost_quality_score=cost_quality.recommended_adaptive_score,
        adaptive_latency_score=latency.recommended_adaptive_latency_score,
        availability_penalty=availability_penalty,
        guardrail_penalty=guardrail_penalty,
        execution_confidence_score=score,
        hitl_required=hitl_required,
        fallback_safety_summary=_fallback_safety_summary(kind, status),
        advisory_actions=_signal_actions(kind),
        evidence=(
            f"dynamic_strategy:{dynamic_strategy.selected_strategy_id}",
            f"agent_allocations:{len(_source_agent_ids(agent_allocation))}",
            f"resource_allocation:{resource_allocation.allocation_id}",
            f"self_tuning_policy:{policy.policy_id}",
            f"cost_quality:{cost_quality.recommended_candidate_id}",
            f"latency:{latency.recommended_candidate_id}",
            f"unavailable_reasons:{len(selected_strategy.unavailable_reason_codes)}",
            f"hitl_required:{hitl_required}",
        ),
    )


def _execution_confidence_score(
    *,
    dynamic_strategy_score: int,
    agent_allocation_score: int,
    dynamic_resource_score: int,
    self_tuning_score: int,
    adaptive_cost_quality_score: int,
    adaptive_latency_score: int,
    availability_penalty: int,
    guardrail_penalty: int,
) -> int:
    return min(
        100,
        max(
            0,
            dynamic_strategy_score // 12
            + agent_allocation_score // 12
            + dynamic_resource_score // 20
            + self_tuning_score // 20
            + adaptive_cost_quality_score // 16
            + adaptive_latency_score // 12
            - availability_penalty
            - guardrail_penalty,
        ),
    )


def _status(
    kind: ExecutionConfidenceSignalKind,
    resource_allocation: DynamicResourceAllocationCandidate,
    policy: WorkflowSelfTuningPolicy,
) -> ExecutionConfidenceStatus:
    if kind in {"provider_availability_confidence", "fallback_safety_confidence"}:
        return "guardrail"
    if resource_allocation.hitl_required or policy.hitl_required:
        return "review_required"
    return "ready"


def _confidence_band(
    score: int,
    status: ExecutionConfidenceStatus,
) -> ExecutionConfidenceBand:
    if status == "guardrail" or score < 42:
        return "guarded"
    if score >= 82:
        return "high"
    if score >= 62:
        return "moderate"
    return "low"


def _overall_confidence_score(
    signals: tuple[ExecutionConfidenceSignal, ...],
) -> int:
    return sum(signal.execution_confidence_score for signal in signals) // len(signals)


def _overall_confidence_band(
    signals: tuple[ExecutionConfidenceSignal, ...],
) -> ExecutionConfidenceBand:
    if any(signal.status == "guardrail" for signal in signals):
        return "guarded"
    return _confidence_band(_overall_confidence_score(signals), "ready")


def _signal_ids_for_band(
    signals: tuple[ExecutionConfidenceSignal, ...],
    band: ExecutionConfidenceBand,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.confidence_band == band)


def _signal_ids_for_status(
    signals: tuple[ExecutionConfidenceSignal, ...],
    status: ExecutionConfidenceStatus,
) -> tuple[str, ...]:
    return tuple(signal.signal_id for signal in signals if signal.status == status)


def _source_agent_ids(
    agent_allocation: DynamicAgentAllocationPlan,
) -> tuple[str, ...]:
    return agent_allocation.primary_allocation_ids or agent_allocation.allocation_ids[:3]


def _required_resource_allocation(
    allocation_id: str,
    plan: DynamicResourceAllocationPlan,
) -> DynamicResourceAllocationCandidate:
    allocation = dynamic_resource_allocation_by_id(allocation_id, plan)
    if allocation is None:
        raise ValueError("required execution confidence resource metadata is missing")
    return allocation


def _required_self_tuning_policy(
    policy_id: str,
    plan: WorkflowSelfTuningPolicyPlan,
) -> WorkflowSelfTuningPolicy:
    policy = workflow_self_tuning_policy_by_id(policy_id, plan)
    if policy is None:
        raise ValueError("required execution confidence self-tuning metadata is missing")
    return policy


def _fallback_safety_summary(
    kind: ExecutionConfidenceSignalKind,
    status: ExecutionConfidenceStatus,
) -> str:
    if status == "guardrail":
        return "Preserve execution confidence as guardrail metadata without routing or execution."
    if kind == "strategy_execution_confidence":
        return "Treat dynamic strategy confidence as advisory until runtime authority exists."
    if kind == "agent_allocation_confidence":
        return "Keep agent confidence detached from agent instantiation or invocation."
    if kind == "resource_capacity_confidence":
        return "Keep resource confidence detached from allocation and capacity enforcement."
    return "Keep self-tuning confidence detached from workflow control and retry behavior."


def _signal_actions(kind: ExecutionConfidenceSignalKind) -> tuple[str, ...]:
    return (
        f"Surface {kind} as advisory execution confidence metadata.",
        "Keep confidence application, provider routing, execution, agents, resources, workflow control, HITL emission, storage, and output behavior disabled.",
    )


def _plan_actions(
    signals: tuple[ExecutionConfidenceSignal, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose execution confidence posture as advisory metadata only.",
        "Keep applied confidence signal ids empty until explicit runtime authority exists.",
        "Preserve provider, model, runtime, agent, resource, workflow, HITL, storage, and output boundaries.",
    ]
    if any(signal.hitl_required for signal in signals):
        actions.append("Require review before any future execution confidence behavior.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
