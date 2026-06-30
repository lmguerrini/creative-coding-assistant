"""V5.5 advisory emergence optimization intelligence."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.creative_analytics import (
    CreativeAnalytics,
    CreativeAnalyticsPanel,
    build_creative_analytics,
    creative_analytics_panel_by_id,
)
from creative_coding_assistant.orchestration.creative_exploration_optimizer import (
    CreativeExplorationOptimizationCandidate,
    CreativeExplorationOptimizationPlan,
    CreativeExplorationOptimizationStatus,
    creative_exploration_candidate_by_id,
    optimize_creative_exploration,
)
from creative_coding_assistant.orchestration.exploration_budget_planner import (
    ExplorationBudgetTopic,
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

EmergenceOptimizationKind = Literal[
    "planning_emergence",
    "aesthetic_emergence",
    "curation_emergence",
    "synthesis_emergence",
]
EmergenceOptimizationMode = Literal[
    "pattern_amplification",
    "diversity_emergence",
    "risk_bounded_emergence",
    "synthesis_guardrail",
]
EmergenceOptimizationStatus = CreativeExplorationOptimizationStatus

EMERGENCE_OPTIMIZATION_CANDIDATE_SERIALIZATION_VERSION = (
    "emergence_optimization_candidate.v1"
)
EMERGENCE_OPTIMIZATION_PLAN_SERIALIZATION_VERSION = (
    "emergence_optimization_plan.v1"
)
EMERGENCE_OPTIMIZER_AUTHORITY_BOUNDARY = (
    "V5.5 emergence optimization combines advisory creative exploration "
    "optimization and creative analytics metadata into inspectable emergence "
    "potential recommendations only; it does not generate emergent variants, "
    "apply emergence behavior, select variants or artifacts, evaluate "
    "generated output, collect creative metrics, trigger refinement, enforce "
    "budgets, change provider or model routing, execute providers, probe "
    "local runtimes, scan or download local models, invoke agents, allocate "
    "resources, emit HITL requests, control workflows, mutate workflow graphs, "
    "execute workflows, trigger retries, mutate prompts, write storage, "
    "modify generated output, or apply Runtime Evolution."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "emergence_behavior_application",
    "emergent_variant_generation",
    "variant_generation",
    "variant_selection",
    "artifact_selection",
    "generated_output_evaluation",
    "creative_metric_collection",
    "refinement_triggering",
    "budget_enforcement",
    "provider_or_model_routing",
    "automatic_provider_switching",
    "automatic_model_switching",
    "provider_execution",
    "local_runtime_probe",
    "local_model_inventory_scan",
    "automatic_model_download",
    "agent_invocation",
    "resource_allocation",
    "human_review_request",
    "hitl_request_emission",
    "workflow_control",
    "workflow_graph_mutation",
    "workflow_execution",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)


class EmergenceOptimizationCandidate(BaseModel):
    """One advisory emergence optimization candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1, max_length=180)
    emergence_kind: EmergenceOptimizationKind
    emergence_mode: EmergenceOptimizationMode
    status: EmergenceOptimizationStatus
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_id: ExecutionModeId
    topic_id: ExplorationBudgetTopic
    source_creative_exploration_candidate_id: str = Field(
        min_length=1,
        max_length=180,
    )
    source_creative_analytics_panel_id: str = Field(min_length=1, max_length=180)
    source_workflow_risk_factor_id: str = Field(min_length=1, max_length=180)
    source_execution_confidence_signal_id: str = Field(min_length=1, max_length=180)
    provider_sequence: tuple[ProviderId, ...] = Field(min_length=1, max_length=4)
    model_profile_sequence: tuple[str, ...] = Field(min_length=1, max_length=4)
    hybrid_policy_direction: HybridRoutingPolicyDirection
    unavailable_reason_codes: tuple[UnavailableReasonCode, ...] = Field(
        default_factory=tuple,
        max_length=9,
    )
    exploration_status: CreativeExplorationOptimizationStatus
    diversity_band: str = Field(min_length=1, max_length=80)
    workflow_risk_severity: str = Field(min_length=1, max_length=80)
    analytics_panel_status: str = Field(min_length=1, max_length=80)
    creative_signal_count: int = Field(ge=0, le=12000)
    guardrail_signal_count: int = Field(ge=0, le=4000)
    creative_exploration_score: int = Field(ge=0, le=500)
    execution_confidence_score: int = Field(ge=0, le=100)
    analytics_signal_score: int = Field(ge=0, le=180)
    mode_weight: int = Field(ge=0, le=100)
    guardrail_penalty: int = Field(ge=0, le=320)
    emergence_potential_score: int = Field(ge=0, le=500)
    recommended_emergence_path_count: int = Field(ge=0, le=4)
    applied_emergence_path_count: int = Field(ge=0, le=0)
    hitl_required: bool
    emergence_summary: str = Field(min_length=1, max_length=360)
    fallback_summary: str = Field(min_length=1, max_length=360)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    emergence_optimizer_implemented: Literal[True] = True
    emergence_potential_metadata_implemented: Literal[True] = True
    creative_exploration_metadata_used: Literal[True] = True
    creative_analytics_metadata_used: Literal[True] = True
    workflow_risk_metadata_used: Literal[True] = True
    provider_intelligence_metadata_used: Literal[True] = True
    availability_awareness_metadata_used: Literal[True] = True
    manual_assisted_auto_mode_metadata_used: Literal[True] = True
    hybrid_transition_metadata_used: Literal[True] = True
    task_aware_category_metadata_used: Literal[True] = True
    execution_simulation_metadata_used: Literal[True] = True
    fallback_safety_metadata_used: Literal[True] = True
    emergence_behavior_application_implemented: Literal[False] = False
    emergent_variant_generation_implemented: Literal[False] = False
    variant_generation_implemented: Literal[False] = False
    variant_selection_implemented: Literal[False] = False
    artifact_selection_implemented: Literal[False] = False
    generated_output_evaluation_implemented: Literal[False] = False
    creative_metric_collection_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_runtime_probe_implemented: Literal[False] = False
    local_model_inventory_scan_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal["emergence_optimization_candidate.v1"] = (
        EMERGENCE_OPTIMIZATION_CANDIDATE_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _candidate_matches_contract(self) -> Self:
        if self.candidate_id != f"emergence_optimizer::{self.emergence_kind}":
            raise ValueError("candidate_id must match emergence_kind")
        if len(self.model_profile_sequence) != len(self.provider_sequence):
            raise ValueError("model_profile_sequence must match provider_sequence")
        if self.analytics_signal_score != _analytics_signal_score(
            creative_signal_count=self.creative_signal_count,
            guardrail_signal_count=self.guardrail_signal_count,
        ):
            raise ValueError("analytics_signal_score must match analytics counts")
        if self.guardrail_penalty != _guardrail_penalty(
            guardrail_signal_count=self.guardrail_signal_count,
            workflow_risk_severity=self.workflow_risk_severity,
            status=self.status,
            analytics_panel_status=self.analytics_panel_status,
        ):
            raise ValueError("guardrail_penalty must match source guardrails")
        if self.emergence_potential_score != _emergence_potential_score(
            creative_exploration_score=self.creative_exploration_score,
            analytics_signal_score=self.analytics_signal_score,
            execution_confidence_score=self.execution_confidence_score,
            mode_weight=self.mode_weight,
            guardrail_penalty=self.guardrail_penalty,
        ):
            raise ValueError("emergence_potential_score must combine source scores")
        if self.status == "guardrail" and self.recommended_emergence_path_count:
            raise ValueError("guardrail emergence must recommend no paths")
        if self.applied_emergence_path_count:
            raise ValueError("applied_emergence_path_count must remain zero")
        if self.status == "recommended" and (
            self.emergence_mode != "diversity_emergence"
        ):
            raise ValueError("recommended emergence must use diversity emergence")
        return self


class EmergenceOptimizationPlan(BaseModel):
    """Bounded V5.5 advisory emergence optimization plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["emergence_optimizer"] = "emergence_optimizer"
    serialization_version: Literal["emergence_optimization_plan.v1"] = (
        EMERGENCE_OPTIMIZATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=EMERGENCE_OPTIMIZER_AUTHORITY_BOUNDARY,
        max_length=2100,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_creative_exploration_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_creative_analytics_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    provider_ids: tuple[ProviderId, ...] = Field(min_length=4, max_length=4)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    hybrid_policy_directions: tuple[HybridRoutingPolicyDirection, ...] = Field(
        min_length=4,
        max_length=4,
    )
    candidates: tuple[EmergenceOptimizationCandidate, ...] = Field(
        min_length=4,
        max_length=4,
    )
    candidate_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    recommended_candidate_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    bounded_candidate_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    guardrail_candidate_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    hitl_required_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    applied_emergence_candidate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    candidate_count: int = Field(ge=4, le=4)
    recommended_candidate_count: int = Field(ge=0, le=4)
    guardrail_candidate_count: int = Field(ge=0, le=4)
    hitl_required_candidate_count: int = Field(ge=0, le=4)
    total_recommended_emergence_path_count: int = Field(ge=0, le=16)
    total_applied_emergence_path_count: int = Field(ge=0, le=0)
    highest_emergence_potential_score: int = Field(ge=0, le=500)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    emergence_optimizer_implemented: Literal[True] = True
    emergence_potential_metadata_implemented: Literal[True] = True
    creative_exploration_metadata_used: Literal[True] = True
    creative_analytics_metadata_used: Literal[True] = True
    workflow_risk_metadata_used: Literal[True] = True
    provider_intelligence_metadata_used: Literal[True] = True
    availability_awareness_metadata_used: Literal[True] = True
    manual_assisted_auto_mode_metadata_used: Literal[True] = True
    hybrid_transition_metadata_used: Literal[True] = True
    task_aware_category_metadata_used: Literal[True] = True
    execution_simulation_metadata_used: Literal[True] = True
    fallback_safety_metadata_used: Literal[True] = True
    emergence_behavior_application_implemented: Literal[False] = False
    emergent_variant_generation_implemented: Literal[False] = False
    variant_generation_implemented: Literal[False] = False
    variant_selection_implemented: Literal[False] = False
    artifact_selection_implemented: Literal[False] = False
    generated_output_evaluation_implemented: Literal[False] = False
    creative_metric_collection_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_runtime_probe_implemented: Literal[False] = False
    local_model_inventory_scan_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    resource_allocation_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
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
        if self.recommended_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "recommended",
        ):
            raise ValueError("recommended_candidate_ids must match candidates")
        if self.bounded_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "bounded",
        ):
            raise ValueError("bounded_candidate_ids must match candidates")
        if self.guardrail_candidate_ids != _candidate_ids_for_status(
            self.candidates,
            "guardrail",
        ):
            raise ValueError("guardrail_candidate_ids must match candidates")
        if self.hitl_required_candidate_ids != tuple(
            candidate.candidate_id
            for candidate in self.candidates
            if candidate.hitl_required
        ):
            raise ValueError("hitl_required_candidate_ids must match candidates")
        if self.applied_emergence_candidate_ids:
            raise ValueError("applied_emergence_candidate_ids must remain empty")
        if self.recommended_candidate_count != len(self.recommended_candidate_ids):
            raise ValueError("recommended_candidate_count must match candidates")
        if self.guardrail_candidate_count != len(self.guardrail_candidate_ids):
            raise ValueError("guardrail_candidate_count must match candidates")
        if self.hitl_required_candidate_count != len(self.hitl_required_candidate_ids):
            raise ValueError("hitl_required_candidate_count must match candidates")
        if self.total_recommended_emergence_path_count != sum(
            candidate.recommended_emergence_path_count
            for candidate in self.candidates
        ):
            raise ValueError("total_recommended_emergence_path_count must match candidates")
        if self.total_applied_emergence_path_count != 0:
            raise ValueError("total_applied_emergence_path_count must remain zero")
        if self.highest_emergence_potential_score != max(
            candidate.emergence_potential_score for candidate in self.candidates
        ):
            raise ValueError("highest_emergence_potential_score must match candidates")
        for candidate in self.candidates:
            if candidate.route_name != self.route_name:
                raise ValueError("candidate route_name must match plan")
        return self


def optimize_emergence(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "creative_coding",
    execution_mode_id: ExecutionModeId | str | None = None,
    creative_exploration: CreativeExplorationOptimizationPlan | None = None,
    creative_analytics: CreativeAnalytics | None = None,
) -> EmergenceOptimizationPlan:
    """Optimize emergence metadata without applying emergent behavior."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    exploration_plan = creative_exploration or optimize_creative_exploration(
        route=route_name,
        task_type=normalized_task_type,
        execution_mode_id=execution_mode_id,
    )
    normalized_mode = str(execution_mode_id or exploration_plan.candidates[0].execution_mode_id)
    execution_modes = routing_execution_mode_registry()
    if normalized_mode not in execution_modes.execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")
    analytics = creative_analytics or build_creative_analytics()
    candidates = _candidates(
        route_name=route_name,
        task_type=exploration_plan.task_type,
        execution_mode_id=normalized_mode,  # type: ignore[arg-type]
        creative_exploration=exploration_plan,
        creative_analytics=analytics,
    )
    return EmergenceOptimizationPlan(
        route_name=route_name,
        task_type=exploration_plan.task_type,
        source_creative_exploration_serialization_version=(
            exploration_plan.serialization_version
        ),
        source_creative_analytics_serialization_version=analytics.serialization_version,
        provider_ids=exploration_plan.provider_ids,
        execution_mode_ids=execution_modes.execution_mode_ids,
        hybrid_policy_directions=exploration_plan.hybrid_policy_directions,
        candidates=candidates,
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        recommended_candidate_ids=_candidate_ids_for_status(candidates, "recommended"),
        bounded_candidate_ids=_candidate_ids_for_status(candidates, "bounded"),
        guardrail_candidate_ids=_candidate_ids_for_status(candidates, "guardrail"),
        hitl_required_candidate_ids=tuple(
            candidate.candidate_id for candidate in candidates if candidate.hitl_required
        ),
        applied_emergence_candidate_ids=(),
        candidate_count=len(candidates),
        recommended_candidate_count=len(
            _candidate_ids_for_status(candidates, "recommended")
        ),
        guardrail_candidate_count=len(_candidate_ids_for_status(candidates, "guardrail")),
        hitl_required_candidate_count=sum(
            1 for candidate in candidates if candidate.hitl_required
        ),
        total_recommended_emergence_path_count=sum(
            candidate.recommended_emergence_path_count for candidate in candidates
        ),
        total_applied_emergence_path_count=0,
        highest_emergence_potential_score=max(
            candidate.emergence_potential_score for candidate in candidates
        ),
        advisory_actions=_plan_actions(candidates),
    )


def emergence_candidate_by_id(
    candidate_id: str,
    plan: EmergenceOptimizationPlan | None = None,
) -> EmergenceOptimizationCandidate | None:
    """Return one emergence candidate without applying behavior."""

    source_plan = plan or optimize_emergence()
    for candidate in source_plan.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def emergence_candidates_for_status(
    status: EmergenceOptimizationStatus,
    plan: EmergenceOptimizationPlan | None = None,
) -> tuple[EmergenceOptimizationCandidate, ...]:
    """Return emergence candidates by advisory status."""

    source_plan = plan or optimize_emergence()
    return tuple(candidate for candidate in source_plan.candidates if candidate.status == status)


def _candidates(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    creative_exploration: CreativeExplorationOptimizationPlan,
    creative_analytics: CreativeAnalytics,
) -> tuple[EmergenceOptimizationCandidate, ...]:
    return (
        _candidate(
            kind="planning_emergence",
            topic_id="planning_execution_fit",
            analytics_panel_id="creative_analytics::complexity_profile",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            creative_exploration=creative_exploration,
            creative_analytics=creative_analytics,
        ),
        _candidate(
            kind="aesthetic_emergence",
            topic_id="style_aesthetic_alignment",
            analytics_panel_id="creative_analytics::diversity_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            creative_exploration=creative_exploration,
            creative_analytics=creative_analytics,
        ),
        _candidate(
            kind="curation_emergence",
            topic_id="curation_refinement_need",
            analytics_panel_id="creative_analytics::quality_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            creative_exploration=creative_exploration,
            creative_analytics=creative_analytics,
        ),
        _candidate(
            kind="synthesis_emergence",
            topic_id="final_synthesis_readiness",
            analytics_panel_id="creative_analytics::consistency_readiness",
            route_name=route_name,
            task_type=task_type,
            execution_mode_id=execution_mode_id,
            creative_exploration=creative_exploration,
            creative_analytics=creative_analytics,
        ),
    )


def _candidate(
    *,
    kind: EmergenceOptimizationKind,
    topic_id: ExplorationBudgetTopic,
    analytics_panel_id: str,
    route_name: RouteName,
    task_type: TaskRoutingType,
    execution_mode_id: ExecutionModeId,
    creative_exploration: CreativeExplorationOptimizationPlan,
    creative_analytics: CreativeAnalytics,
) -> EmergenceOptimizationCandidate:
    exploration = _required_exploration_candidate(
        f"creative_exploration_optimizer::{topic_id}",
        creative_exploration,
    )
    analytics_panel = _required_analytics_panel(analytics_panel_id, creative_analytics)
    status = exploration.status
    mode = _mode(kind, status)
    signal_score = _analytics_signal_score(
        creative_signal_count=analytics_panel.creative_signal_count,
        guardrail_signal_count=analytics_panel.guardrail_signal_count,
    )
    mode_weight = _mode_weight(mode)
    penalty = _guardrail_penalty(
        guardrail_signal_count=analytics_panel.guardrail_signal_count,
        workflow_risk_severity=exploration.workflow_risk_severity,
        status=status,
        analytics_panel_status=analytics_panel.status,
    )
    score = _emergence_potential_score(
        creative_exploration_score=exploration.creative_exploration_score,
        analytics_signal_score=signal_score,
        execution_confidence_score=exploration.execution_confidence_score,
        mode_weight=mode_weight,
        guardrail_penalty=penalty,
    )
    recommended_paths = 0 if status == "guardrail" else 1
    return EmergenceOptimizationCandidate(
        candidate_id=f"emergence_optimizer::{kind}",
        emergence_kind=kind,
        emergence_mode=mode,
        status=status,
        route_name=route_name,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
        topic_id=topic_id,
        source_creative_exploration_candidate_id=exploration.candidate_id,
        source_creative_analytics_panel_id=analytics_panel.panel_id,
        source_workflow_risk_factor_id=exploration.source_workflow_risk_factor_id,
        source_execution_confidence_signal_id=(
            exploration.source_execution_confidence_signal_id
        ),
        provider_sequence=exploration.provider_sequence,
        model_profile_sequence=exploration.model_profile_sequence,
        hybrid_policy_direction=exploration.hybrid_policy_direction,
        unavailable_reason_codes=exploration.unavailable_reason_codes,
        exploration_status=exploration.status,
        diversity_band=exploration.diversity_band,
        workflow_risk_severity=exploration.workflow_risk_severity,
        analytics_panel_status=analytics_panel.status,
        creative_signal_count=analytics_panel.creative_signal_count,
        guardrail_signal_count=analytics_panel.guardrail_signal_count,
        creative_exploration_score=exploration.creative_exploration_score,
        execution_confidence_score=exploration.execution_confidence_score,
        analytics_signal_score=signal_score,
        mode_weight=mode_weight,
        guardrail_penalty=penalty,
        emergence_potential_score=score,
        recommended_emergence_path_count=recommended_paths,
        applied_emergence_path_count=0,
        hitl_required=exploration.hitl_required or status == "guardrail",
        emergence_summary=_emergence_summary(kind, status),
        fallback_summary=_fallback_summary(status),
        advisory_actions=_candidate_actions(status),
        evidence=(
            f"creative_exploration:{exploration.candidate_id}",
            f"creative_analytics_panel:{analytics_panel.panel_id}",
            f"workflow_risk:{exploration.source_workflow_risk_factor_id}",
            f"analytics_guardrails:{analytics_panel.guardrail_signal_count}",
            f"exploration_status:{exploration.status}",
            f"emergence_mode:{mode}",
        ),
    )


def _emergence_potential_score(
    *,
    creative_exploration_score: int,
    analytics_signal_score: int,
    execution_confidence_score: int,
    mode_weight: int,
    guardrail_penalty: int,
) -> int:
    return min(
        500,
        max(
            0,
            creative_exploration_score
            + analytics_signal_score
            + execution_confidence_score
            + mode_weight
            - guardrail_penalty,
        ),
    )


def _analytics_signal_score(
    *,
    creative_signal_count: int,
    guardrail_signal_count: int,
) -> int:
    return min(
        180,
        max(0, creative_signal_count // 8 - guardrail_signal_count * 2 + 60),
    )


def _guardrail_penalty(
    *,
    guardrail_signal_count: int,
    workflow_risk_severity: str,
    status: EmergenceOptimizationStatus,
    analytics_panel_status: str,
) -> int:
    penalty = guardrail_signal_count * 6
    if status == "guardrail":
        penalty += 120
    elif analytics_panel_status == "guarded":
        penalty += 40
    if workflow_risk_severity == "guarded":
        penalty += 60
    elif workflow_risk_severity == "high":
        penalty += 30
    return min(320, penalty)


def _mode(
    kind: EmergenceOptimizationKind,
    status: EmergenceOptimizationStatus,
) -> EmergenceOptimizationMode:
    if status == "guardrail":
        return "synthesis_guardrail"
    if kind == "aesthetic_emergence" and status == "recommended":
        return "diversity_emergence"
    if status == "bounded":
        return "pattern_amplification"
    return "risk_bounded_emergence"


def _mode_weight(mode: EmergenceOptimizationMode) -> int:
    return {
        "diversity_emergence": 80,
        "pattern_amplification": 50,
        "risk_bounded_emergence": 40,
        "synthesis_guardrail": 30,
    }[mode]


def _candidate_ids_for_status(
    candidates: tuple[EmergenceOptimizationCandidate, ...],
    status: EmergenceOptimizationStatus,
) -> tuple[str, ...]:
    return tuple(candidate.candidate_id for candidate in candidates if candidate.status == status)


def _required_exploration_candidate(
    candidate_id: str,
    plan: CreativeExplorationOptimizationPlan,
) -> CreativeExplorationOptimizationCandidate:
    candidate = creative_exploration_candidate_by_id(candidate_id, plan)
    if candidate is None:
        raise ValueError("required emergence exploration metadata is missing")
    return candidate


def _required_analytics_panel(
    panel_id: str,
    analytics: CreativeAnalytics,
) -> CreativeAnalyticsPanel:
    panel = creative_analytics_panel_by_id(panel_id, analytics)
    if panel is None:
        raise ValueError("required emergence analytics metadata is missing")
    return panel


def _emergence_summary(
    kind: EmergenceOptimizationKind,
    status: EmergenceOptimizationStatus,
) -> str:
    if status == "recommended":
        return f"Surface {kind} as the strongest advisory emergence path."
    if status == "bounded":
        return f"Keep {kind} bounded by analytics and workflow risk metadata."
    return f"Keep {kind} in guardrail posture without emergent variants."


def _fallback_summary(status: EmergenceOptimizationStatus) -> str:
    if status == "recommended":
        return "Fallback to bounded emergence if risk or analytics guardrails rise."
    if status == "bounded":
        return "Fallback to guardrail posture before any emergence behavior exists."
    return "Preserve guardrail posture without applying emergence behavior."


def _candidate_actions(
    status: EmergenceOptimizationStatus,
) -> tuple[str, ...]:
    return (
        f"Surface {status} emergence potential as advisory metadata.",
        "Keep emergence behavior, variants, artifact selection, evaluation, metrics, routing, workflow, storage, and output behavior disabled.",
    )


def _plan_actions(
    candidates: tuple[EmergenceOptimizationCandidate, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose emergence optimization as advisory metadata only.",
        "Keep applied emergence candidate ids empty and applied path count at zero.",
        "Preserve emergence behavior, variant generation, evaluation, routing, workflow, storage, and output boundaries.",
    ]
    if _candidate_ids_for_status(candidates, "guardrail"):
        actions.append("Require review before any future emergence behavior.")
    return tuple(actions)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route))
