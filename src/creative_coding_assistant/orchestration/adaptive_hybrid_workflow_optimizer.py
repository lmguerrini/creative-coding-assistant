"""V5.5 advisory hybrid workflow optimization intelligence."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.execution_path_optimization import (
    ExecutionPathOptimizationPlan,
    plan_execution_path_optimization,
)
from creative_coding_assistant.orchestration.hybrid_routing import (
    HybridRoutingPlan,
    route_hybrid_model_request,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    EstimatedCostBand,
    EstimatedLatencyBand,
    EstimatedQualityBand,
    ExecutionModeId,
    HybridRoutingPolicyDirection,
    ProviderId,
    RoutingCapabilityFamily,
    RoutingRiskBand,
    TaskRoutingType,
    UnavailableReasonCode,
    advisory_hybrid_routing_policy_registry,
    model_routing_intelligence_registry,
    provider_availability_registry,
    routing_execution_mode_registry,
    task_aware_routing_registry,
)

HybridWorkflowCandidateStatus = Literal[
    "recommended",
    "fallback",
    "requires_hitl",
]
HybridWorkflowSurface = Literal["local", "cloud"]

HYBRID_WORKFLOW_SIMULATION_SERIALIZATION_VERSION = (
    "adaptive_hybrid_workflow_simulation.v1"
)
HYBRID_WORKFLOW_CANDIDATE_SERIALIZATION_VERSION = (
    "adaptive_hybrid_workflow_candidate.v1"
)
HYBRID_WORKFLOW_FALLBACK_SERIALIZATION_VERSION = (
    "adaptive_hybrid_workflow_fallback.v1"
)
HYBRID_WORKFLOW_OPTIMIZATION_PLAN_SERIALIZATION_VERSION = (
    "adaptive_hybrid_workflow_optimization_plan.v1"
)
HYBRID_WORKFLOW_OPTIMIZER_AUTHORITY_BOUNDARY = (
    "V5.5 hybrid workflow optimization combines advisory V5.1 path "
    "optimization and V5.2 routing intelligence into inspectable hybrid "
    "workflow strategy recommendations only; it does not execute workflows, "
    "select or switch providers or models, call providers, probe local "
    "runtimes, list or download local models, assume API keys, emit HITL "
    "requests, mutate prompts, write storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "hybrid_workflow_execution",
    "execution_strategy_selection",
    "automatic_provider_switching",
    "automatic_model_switching",
    "automatic_model_download",
    "automatic_api_key_assumption",
    "provider_or_model_routing_application",
    "provider_execution",
    "local_runtime_probe",
    "local_model_inventory_scan",
    "provider_output_merging",
    "workflow_control",
    "retry_or_refinement_triggering",
    "human_input_request_emission",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class HybridWorkflowSimulationEstimate(BaseModel):
    """Pre-run hybrid workflow simulation estimate without execution."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    simulation_id: str = Field(min_length=1, max_length=180)
    policy_direction: HybridRoutingPolicyDirection
    provider_sequence: tuple[ProviderId, ...] = Field(min_length=1, max_length=4)
    model_profile_sequence: tuple[str, ...] = Field(min_length=1, max_length=4)
    workflow_path_candidate_id: str = Field(min_length=1, max_length=180)
    workflow_summary: str = Field(min_length=1, max_length=320)
    estimated_quality: EstimatedQualityBand
    estimated_cost: EstimatedCostBand
    estimated_latency: EstimatedLatencyBand
    confidence_score: float = Field(ge=0.0, le=1.0)
    unavailable_reason_codes: tuple[UnavailableReasonCode, ...] = Field(
        default_factory=tuple,
        max_length=9,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    simulation_estimate_implemented: Literal[True] = True
    execution_simulation_run: Literal[False] = False
    provider_call_performed: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    serialization_version: Literal["adaptive_hybrid_workflow_simulation.v1"] = (
        HYBRID_WORKFLOW_SIMULATION_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _simulation_identity_matches(self) -> Self:
        if self.simulation_id != f"hybrid_workflow_simulation::{self.policy_direction}":
            raise ValueError("simulation_id must match policy_direction")
        if len(self.model_profile_sequence) != len(self.provider_sequence):
            raise ValueError("model_profile_sequence must match provider_sequence")
        return self


class HybridWorkflowOptimizationCandidate(BaseModel):
    """One advisory V5.5 hybrid workflow optimization candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=180)
    task_type: TaskRoutingType
    capability_requirements: tuple[RoutingCapabilityFamily, ...] = Field(
        min_length=1,
        max_length=8,
    )
    policy_direction: HybridRoutingPolicyDirection
    source_policy_id: str = Field(min_length=1, max_length=140)
    source_task_decision_id: str = Field(min_length=1, max_length=180)
    source_hybrid_route_decision_id: str = Field(min_length=1, max_length=180)
    execution_mode_id: ExecutionModeId
    provider_sequence: tuple[ProviderId, ...] = Field(min_length=1, max_length=4)
    surface_sequence: tuple[HybridWorkflowSurface, ...] = Field(
        min_length=1,
        max_length=4,
    )
    model_profile_sequence: tuple[str, ...] = Field(min_length=1, max_length=4)
    workflow_path_candidate_id: str = Field(min_length=1, max_length=180)
    status: HybridWorkflowCandidateStatus
    estimated_quality: EstimatedQualityBand
    estimated_cost: EstimatedCostBand
    estimated_latency: EstimatedLatencyBand
    confidence_score: float = Field(ge=0.0, le=1.0)
    risk_band: RoutingRiskBand
    unavailable_reason_codes: tuple[UnavailableReasonCode, ...] = Field(
        default_factory=tuple,
        max_length=9,
    )
    hitl_required: bool
    adaptive_score: int = Field(ge=0, le=240)
    simulation: HybridWorkflowSimulationEstimate
    fallback_reason_summary: str = Field(min_length=1, max_length=320)
    suggested_action: str = Field(min_length=1, max_length=320)
    tradeoffs: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    hybrid_workflow_optimizer_implemented: Literal[True] = True
    adaptive_execution_intelligence_implemented: Literal[True] = True
    strategy_recommendation_implemented: Literal[True] = True
    execution_strategy_selection_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    automatic_api_key_assumption_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_runtime_probe_implemented: Literal[False] = False
    local_model_inventory_scan_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["adaptive_hybrid_workflow_candidate.v1"] = (
        HYBRID_WORKFLOW_CANDIDATE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_is_consistent(self) -> Self:
        if self.candidate_id != f"hybrid_workflow::{self.policy_direction}":
            raise ValueError("candidate_id must match policy_direction")
        if self.source_policy_id != f"hybrid_policy::{self.policy_direction}":
            raise ValueError("source_policy_id must match policy_direction")
        if len(self.surface_sequence) != len(self.provider_sequence):
            raise ValueError("surface_sequence must match provider_sequence")
        if len(self.model_profile_sequence) != len(self.provider_sequence):
            raise ValueError("model_profile_sequence must match provider_sequence")
        if self.simulation.policy_direction != self.policy_direction:
            raise ValueError("simulation policy_direction must match candidate")
        if self.simulation.provider_sequence != self.provider_sequence:
            raise ValueError("simulation provider_sequence must match candidate")
        if self.simulation.model_profile_sequence != self.model_profile_sequence:
            raise ValueError("simulation model_profile_sequence must match candidate")
        if self.simulation.workflow_path_candidate_id != self.workflow_path_candidate_id:
            raise ValueError("simulation workflow path must match candidate")
        if bool(self.unavailable_reason_codes) and not self.hitl_required:
            raise ValueError("unavailable reasons require HITL")
        if self.execution_mode_id == "auto_mode" and self.risk_band == "high":
            if not self.hitl_required:
                raise ValueError("high-risk auto-mode candidates require HITL")
        return self


class HybridWorkflowFallback(BaseModel):
    """Fallback intelligence for an advisory hybrid workflow recommendation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    fallback_id: str = Field(min_length=1, max_length=180)
    preferred_candidate_id: str = Field(min_length=1, max_length=180)
    fallback_candidate_id: str = Field(min_length=1, max_length=180)
    reason_codes: tuple[UnavailableReasonCode, ...] = Field(min_length=1, max_length=9)
    reason_summary: str = Field(min_length=1, max_length=320)
    suggested_action: str = Field(min_length=1, max_length=320)
    tradeoffs: tuple[str, ...] = Field(min_length=1, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    fallback_intelligence_implemented: Literal[True] = True
    fallback_application_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["adaptive_hybrid_workflow_fallback.v1"] = (
        HYBRID_WORKFLOW_FALLBACK_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _fallback_identity_matches(self) -> Self:
        expected = f"hybrid_workflow_fallback::{self.preferred_candidate_id}"
        if self.fallback_id != expected:
            raise ValueError("fallback_id must match preferred_candidate_id")
        if self.preferred_candidate_id == self.fallback_candidate_id:
            raise ValueError("fallback candidate must differ from preferred candidate")
        return self


class HybridWorkflowOptimizationPlan(BaseModel):
    """Bounded V5.5 advisory hybrid workflow optimization plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["hybrid_workflow_optimizer"] = "hybrid_workflow_optimizer"
    serialization_version: Literal[
        "adaptive_hybrid_workflow_optimization_plan.v1"
    ] = HYBRID_WORKFLOW_OPTIMIZATION_PLAN_SERIALIZATION_VERSION
    authority_boundary: str = Field(
        default=HYBRID_WORKFLOW_OPTIMIZER_AUTHORITY_BOUNDARY,
        max_length=1600,
    )
    task_type: TaskRoutingType
    source_path_optimization_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_hybrid_routing_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_routing_intelligence_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    provider_ids: tuple[ProviderId, ...] = Field(min_length=4, max_length=4)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    hybrid_policy_directions: tuple[HybridRoutingPolicyDirection, ...] = Field(
        min_length=4,
        max_length=4,
    )
    candidates: tuple[HybridWorkflowOptimizationCandidate, ...] = Field(
        min_length=4,
        max_length=4,
    )
    candidate_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    recommended_candidate_id: str = Field(min_length=1, max_length=180)
    fallback_candidate_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    selected_candidate_id: None = None
    fallback: HybridWorkflowFallback
    decision_count: int = Field(ge=4, le=4)
    hitl_required_candidate_count: int = Field(ge=0, le=4)
    highest_adaptive_score: int = Field(ge=0, le=240)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=18,
    )
    hybrid_workflow_optimizer_implemented: Literal[True] = True
    adaptive_execution_policy_metadata_implemented: Literal[True] = True
    execution_simulation_estimates_implemented: Literal[True] = True
    fallback_intelligence_implemented: Literal[True] = True
    execution_strategy_selection_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    automatic_api_key_assumption_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_runtime_probe_implemented: Literal[False] = False
    local_model_inventory_scan_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_candidates(self) -> Self:
        derived_candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if len(set(derived_candidate_ids)) != len(derived_candidate_ids):
            raise ValueError("candidate_ids must be unique")
        if self.candidate_ids != derived_candidate_ids:
            raise ValueError("candidate_ids must match candidates")
        if self.decision_count != len(self.candidates):
            raise ValueError("decision_count must match candidates")
        if self.hybrid_policy_directions != tuple(
            candidate.policy_direction for candidate in self.candidates
        ):
            raise ValueError("hybrid_policy_directions must match candidates")
        recommended = tuple(
            candidate for candidate in self.candidates if candidate.status == "recommended"
        )
        if len(recommended) != 1:
            raise ValueError("exactly one recommended candidate is required")
        if self.recommended_candidate_id != recommended[0].candidate_id:
            raise ValueError("recommended_candidate_id must match candidate")
        if self.fallback_candidate_ids != tuple(
            candidate.candidate_id
            for candidate in self.candidates
            if candidate.status != "recommended"
        ):
            raise ValueError("fallback_candidate_ids must match candidates")
        if self.fallback.preferred_candidate_id != self.recommended_candidate_id:
            raise ValueError("fallback must reference the recommended candidate")
        if self.fallback.fallback_candidate_id not in self.fallback_candidate_ids:
            raise ValueError("fallback candidate must be a fallback candidate")
        if self.hitl_required_candidate_count != sum(
            1 for candidate in self.candidates if candidate.hitl_required
        ):
            raise ValueError("hitl_required_candidate_count must match candidates")
        if self.highest_adaptive_score != max(
            candidate.adaptive_score for candidate in self.candidates
        ):
            raise ValueError("highest_adaptive_score must match candidates")
        return self


def optimize_hybrid_workflow(
    *,
    task_type: TaskRoutingType | str = "creative_coding",
    route: RouteName | str = RouteName.GENERATE,
    execution_mode_id: ExecutionModeId | str | None = None,
    path_optimization: ExecutionPathOptimizationPlan | None = None,
    hybrid_routing: HybridRoutingPlan | None = None,
) -> HybridWorkflowOptimizationPlan:
    """Recommend a hybrid workflow strategy without applying it."""

    normalized_task_type = str(task_type).strip()
    path_plan = path_optimization or plan_execution_path_optimization()
    hybrid_plan = hybrid_routing or route_hybrid_model_request(
        route=_resolve_route(route),
    )
    routing_intelligence = model_routing_intelligence_registry()
    task_registry = task_aware_routing_registry()
    execution_modes = routing_execution_mode_registry()
    policy_registry = advisory_hybrid_routing_policy_registry()
    availability = provider_availability_registry()

    task_decision = next(
        (
            decision
            for decision in task_registry.decisions
            if decision.task_type == normalized_task_type
        ),
        None,
    )
    if task_decision is None:
        raise ValueError("task_type must be present in task-aware routing registry")

    normalized_mode = str(execution_mode_id or task_decision.execution_mode_id).strip()
    if normalized_mode not in execution_modes.execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")

    hybrid_decision = hybrid_plan.decisions[0]
    preferred_path = path_plan.optimization_candidate_ids[0]
    candidates = tuple(
        _candidate_from_policy(
            policy_direction=policy.direction,
            task_type=task_decision.task_type,
            capability_requirements=task_decision.capability_requirements,
            source_task_decision_id=task_decision.decision_id,
            source_hybrid_route_decision_id=hybrid_decision.decision_id,
            execution_mode_id=normalized_mode,  # type: ignore[arg-type]
            provider_ids=routing_intelligence.provider_ids,
            availability=availability,
            model_profile_id=task_decision.recommended_model_profile_id,
            fallback_model_profile_id=task_decision.fallback_model_profile_id,
            workflow_path_candidate_id=preferred_path,
            task_quality=task_decision.estimated_quality,
            task_cost=task_decision.estimated_cost,
            task_latency=task_decision.estimated_latency,
            task_confidence=task_decision.confidence_score,
            task_risk=task_decision.risk_band,
            task_unavailable_reasons=task_decision.unavailable_reason_codes,
        )
        for policy in policy_registry.policies
    )
    recommended = _recommended_candidate(candidates)
    fallback_candidate = _fallback_candidate(candidates, recommended)
    fallback = HybridWorkflowFallback(
        fallback_id=f"hybrid_workflow_fallback::{recommended.candidate_id}",
        preferred_candidate_id=recommended.candidate_id,
        fallback_candidate_id=fallback_candidate.candidate_id,
        reason_codes=_fallback_reason_codes(recommended),
        reason_summary=(
            "Preferred hybrid workflow requires review because availability, "
            "credential, local runtime, cost, latency, or HITL metadata is not "
            "fully safe for automatic use."
        ),
        suggested_action=(
            "Request human confirmation before applying any provider, model, or "
            "hybrid workflow change."
        ),
        tradeoffs=fallback_candidate.tradeoffs,
    )

    return HybridWorkflowOptimizationPlan(
        task_type=task_decision.task_type,
        source_path_optimization_serialization_version=path_plan.serialization_version,
        source_hybrid_routing_serialization_version=hybrid_plan.serialization_version,
        source_routing_intelligence_serialization_version=(
            routing_intelligence.serialization_version
        ),
        provider_ids=routing_intelligence.provider_ids,
        execution_mode_ids=execution_modes.execution_mode_ids,
        hybrid_policy_directions=tuple(candidate.policy_direction for candidate in candidates),
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        recommended_candidate_id=recommended.candidate_id,
        fallback_candidate_ids=tuple(
            candidate.candidate_id
            for candidate in candidates
            if candidate.status != "recommended"
        ),
        fallback=fallback,
        decision_count=len(candidates),
        hitl_required_candidate_count=sum(
            1 for candidate in candidates if candidate.hitl_required
        ),
        highest_adaptive_score=max(candidate.adaptive_score for candidate in candidates),
    )


def hybrid_workflow_candidate_by_id(
    candidate_id: str,
    plan: HybridWorkflowOptimizationPlan | None = None,
) -> HybridWorkflowOptimizationCandidate | None:
    """Return one advisory hybrid workflow candidate without selecting it."""

    source_plan = plan or optimize_hybrid_workflow()
    for candidate in source_plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def hybrid_workflow_candidates_requiring_hitl(
    plan: HybridWorkflowOptimizationPlan | None = None,
) -> tuple[HybridWorkflowOptimizationCandidate, ...]:
    """Return candidates that would require HITL before future application."""

    source_plan = plan or optimize_hybrid_workflow()
    return tuple(candidate for candidate in source_plan.candidates if candidate.hitl_required)


def _candidate_from_policy(
    *,
    policy_direction: HybridRoutingPolicyDirection,
    task_type: TaskRoutingType,
    capability_requirements: tuple[RoutingCapabilityFamily, ...],
    source_task_decision_id: str,
    source_hybrid_route_decision_id: str,
    execution_mode_id: ExecutionModeId,
    provider_ids: tuple[ProviderId, ...],
    availability: object,
    model_profile_id: str,
    fallback_model_profile_id: str,
    workflow_path_candidate_id: str,
    task_quality: EstimatedQualityBand,
    task_cost: EstimatedCostBand,
    task_latency: EstimatedLatencyBand,
    task_confidence: float,
    task_risk: RoutingRiskBand,
    task_unavailable_reasons: tuple[UnavailableReasonCode, ...],
) -> HybridWorkflowOptimizationCandidate:
    provider_sequence = _provider_sequence(policy_direction, provider_ids)
    surface_sequence = tuple(
        "local" if provider == "local" else "cloud" for provider in provider_sequence
    )
    model_sequence = _model_profile_sequence(
        provider_sequence,
        model_profile_id,
        fallback_model_profile_id,
    )
    unavailable = _unavailable_reasons_for_sequence(
        provider_sequence,
        availability,
        task_unavailable_reasons,
    )
    quality, cost, latency = _estimates_for_policy(
        policy_direction,
        task_quality,
        task_cost,
        task_latency,
    )
    risk = _risk_for_policy(policy_direction, task_risk, unavailable)
    confidence = _confidence_for_policy(policy_direction, task_confidence, unavailable)
    score = _adaptive_score(quality, cost, latency, confidence, risk, unavailable)
    hitl_required = bool(unavailable) or risk == "high" or execution_mode_id == "manual_mode"
    status: HybridWorkflowCandidateStatus = "fallback"
    if policy_direction == "local_to_cloud":
        status = "recommended"
    elif hitl_required:
        status = "requires_hitl"

    simulation = HybridWorkflowSimulationEstimate(
        simulation_id=f"hybrid_workflow_simulation::{policy_direction}",
        policy_direction=policy_direction,
        provider_sequence=provider_sequence,
        model_profile_sequence=model_sequence,
        workflow_path_candidate_id=workflow_path_candidate_id,
        workflow_summary=_workflow_summary(policy_direction),
        estimated_quality=quality,
        estimated_cost=cost,
        estimated_latency=latency,
        confidence_score=confidence,
        unavailable_reason_codes=unavailable,
    )

    return HybridWorkflowOptimizationCandidate(
        candidate_id=f"hybrid_workflow::{policy_direction}",
        task_type=task_type,
        capability_requirements=capability_requirements,
        policy_direction=policy_direction,
        source_policy_id=f"hybrid_policy::{policy_direction}",
        source_task_decision_id=source_task_decision_id,
        source_hybrid_route_decision_id=source_hybrid_route_decision_id,
        execution_mode_id=execution_mode_id,
        provider_sequence=provider_sequence,
        surface_sequence=surface_sequence,
        model_profile_sequence=model_sequence,
        workflow_path_candidate_id=workflow_path_candidate_id,
        status=status,
        estimated_quality=quality,
        estimated_cost=cost,
        estimated_latency=latency,
        confidence_score=confidence,
        risk_band=risk,
        unavailable_reason_codes=unavailable,
        hitl_required=hitl_required,
        adaptive_score=score,
        simulation=simulation,
        fallback_reason_summary=_fallback_summary(unavailable),
        suggested_action=_suggested_action(unavailable),
        tradeoffs=_tradeoffs(policy_direction, quality, cost, latency),
        evidence=(
            f"Task {task_type} maps to {len(capability_requirements)} capability requirements.",
            f"Hybrid policy direction is {policy_direction}.",
            f"Execution mode is {execution_mode_id}.",
            "Provider availability is derived from routing intelligence metadata.",
            "Workflow path candidate is advisory path optimization metadata.",
        ),
    )


def _provider_sequence(
    direction: HybridRoutingPolicyDirection,
    provider_ids: tuple[ProviderId, ...],
) -> tuple[ProviderId, ...]:
    cloud = tuple(provider for provider in provider_ids if provider != "local")
    local = tuple(provider for provider in provider_ids if provider == "local")
    if direction == "local_to_cloud":
        return (*local, cloud[0])
    if direction == "cloud_to_local":
        return (cloud[0], *local)
    if direction == "cloud_to_cloud":
        return cloud
    return (*local, *local)


def _model_profile_sequence(
    provider_sequence: tuple[ProviderId, ...],
    model_profile_id: str,
    fallback_model_profile_id: str,
) -> tuple[str, ...]:
    return tuple(
        fallback_model_profile_id if provider == "local" else model_profile_id
        for provider in provider_sequence
    )


def _unavailable_reasons_for_sequence(
    provider_sequence: tuple[ProviderId, ...],
    availability: object,
    task_unavailable_reasons: tuple[UnavailableReasonCode, ...],
) -> tuple[UnavailableReasonCode, ...]:
    reason_codes: list[UnavailableReasonCode] = list(task_unavailable_reasons)
    provider_availability = getattr(availability, "provider_availability")
    for provider in provider_sequence:
        for item in provider_availability:
            if item.provider_id == provider:
                reason_codes.extend(item.unavailable_reason_codes)
                break
    return _dedupe_reason_codes(tuple(reason_codes))


def _dedupe_reason_codes(
    reason_codes: tuple[UnavailableReasonCode, ...],
) -> tuple[UnavailableReasonCode, ...]:
    deduped: list[UnavailableReasonCode] = []
    for reason_code in reason_codes:
        if reason_code not in deduped:
            deduped.append(reason_code)
    return tuple(deduped)


def _estimates_for_policy(
    direction: HybridRoutingPolicyDirection,
    task_quality: EstimatedQualityBand,
    task_cost: EstimatedCostBand,
    task_latency: EstimatedLatencyBand,
) -> tuple[EstimatedQualityBand, EstimatedCostBand, EstimatedLatencyBand]:
    if direction == "local_to_cloud":
        return _max_quality(task_quality, "high"), "medium", "moderate"
    if direction == "cloud_to_local":
        return _max_quality(task_quality, "high"), task_cost, task_latency
    if direction == "cloud_to_cloud":
        return "maximum", "high", "slow"
    return "medium", "low", "fast"


def _max_quality(
    first: EstimatedQualityBand,
    second: EstimatedQualityBand,
) -> EstimatedQualityBand:
    order = {"low": 0, "medium": 1, "high": 2, "maximum": 3}
    return first if order[first] >= order[second] else second


def _risk_for_policy(
    direction: HybridRoutingPolicyDirection,
    task_risk: RoutingRiskBand,
    unavailable: tuple[UnavailableReasonCode, ...],
) -> RoutingRiskBand:
    if "provider_unsupported" in unavailable or direction == "cloud_to_cloud":
        return "high"
    if unavailable or task_risk == "high":
        return "medium"
    return task_risk


def _confidence_for_policy(
    direction: HybridRoutingPolicyDirection,
    task_confidence: float,
    unavailable: tuple[UnavailableReasonCode, ...],
) -> float:
    policy_adjustment = {
        "local_to_cloud": 0.04,
        "cloud_to_local": 0.0,
        "cloud_to_cloud": -0.08,
        "local_to_local": -0.1,
    }[direction]
    availability_penalty = min(0.3, 0.04 * len(unavailable))
    return round(max(0.0, min(1.0, task_confidence + policy_adjustment - availability_penalty)), 2)


def _adaptive_score(
    quality: EstimatedQualityBand,
    cost: EstimatedCostBand,
    latency: EstimatedLatencyBand,
    confidence: float,
    risk: RoutingRiskBand,
    unavailable: tuple[UnavailableReasonCode, ...],
) -> int:
    quality_points = {"low": 10, "medium": 20, "high": 34, "maximum": 42}[quality]
    cost_points = {"low": 32, "medium": 22, "high": 10}[cost]
    latency_points = {"fast": 30, "moderate": 22, "slow": 10}[latency]
    risk_penalty = {"low": 0, "medium": 10, "high": 24}[risk]
    availability_penalty = min(32, 4 * len(unavailable))
    return max(
        0,
        int(
            quality_points
            + cost_points
            + latency_points
            + (confidence * 40)
            - risk_penalty
            - availability_penalty
        ),
    )


def _recommended_candidate(
    candidates: tuple[HybridWorkflowOptimizationCandidate, ...],
) -> HybridWorkflowOptimizationCandidate:
    for candidate in candidates:
        if candidate.status == "recommended":
            return candidate
    return max(candidates, key=lambda candidate: candidate.adaptive_score)


def _fallback_candidate(
    candidates: tuple[HybridWorkflowOptimizationCandidate, ...],
    recommended: HybridWorkflowOptimizationCandidate,
) -> HybridWorkflowOptimizationCandidate:
    fallbacks = tuple(
        candidate for candidate in candidates if candidate.candidate_id != recommended.candidate_id
    )
    return max(fallbacks, key=lambda candidate: candidate.adaptive_score)


def _fallback_reason_codes(
    recommended: HybridWorkflowOptimizationCandidate,
) -> tuple[UnavailableReasonCode, ...]:
    if recommended.unavailable_reason_codes:
        return recommended.unavailable_reason_codes
    return ("hitl_required",)


def _fallback_summary(
    unavailable: tuple[UnavailableReasonCode, ...],
) -> str:
    if not unavailable:
        return "No unavailable reasons recorded; fallback remains advisory."
    return "Unavailable provider, credential, local runtime, or policy metadata requires HITL."


def _suggested_action(
    unavailable: tuple[UnavailableReasonCode, ...],
) -> str:
    if not unavailable:
        return "Keep candidate available for manual or assisted review."
    return "Confirm provider credentials, local runtime readiness, model inventory, and policy risk before use."


def _tradeoffs(
    direction: HybridRoutingPolicyDirection,
    quality: EstimatedQualityBand,
    cost: EstimatedCostBand,
    latency: EstimatedLatencyBand,
) -> tuple[str, ...]:
    return (
        f"{direction} estimates {quality} quality, {cost} cost, and {latency} latency.",
        "Provider availability and local model readiness remain metadata-only.",
        "Applying this workflow would require an explicit future execution contract.",
    )


def _workflow_summary(direction: HybridRoutingPolicyDirection) -> str:
    summaries = {
        "local_to_cloud": "Local draft or exploration followed by cloud final synthesis.",
        "cloud_to_local": "Cloud reasoning followed by local variants or lower-cost iteration.",
        "cloud_to_cloud": "Cloud reasoning or synthesis compared across cloud provider profiles.",
        "local_to_local": "Local exploration and local synthesis across user-managed runtimes.",
    }
    return summaries[direction]


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
