"""Controlled V5.5 adaptive execution policy and simulation engine."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.core import Settings
from creative_coding_assistant.orchestration.adaptive_execution_strategy_selection import (
    AdaptiveExecutionStrategySelectionPlan,
    select_dynamic_execution_strategy,
)
from creative_coding_assistant.orchestration.adaptive_hybrid_workflow_optimizer import (
    HybridWorkflowOptimizationCandidate,
    HybridWorkflowOptimizationPlan,
    optimize_hybrid_workflow,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    EstimatedCostBand,
    EstimatedLatencyBand,
    EstimatedQualityBand,
    ExecutionModeId,
    HybridRoutingPolicyDirection,
    LocalRuntimeKind,
    ProviderId,
    RoutingCapabilityFamily,
    RoutingRiskBand,
    TaskAwareRoutingDecision,
    TaskRoutingType,
    UnavailableReasonCode,
    advisory_hybrid_routing_policy_registry,
    provider_availability_registry,
    routing_execution_mode_registry,
    routing_provider_profile_registry,
    task_aware_routing_registry,
)

ExecutionReadinessStatus = Literal[
    "ready_now",
    "requires_confirmation",
    "blocked",
]
AdaptiveExecutionOptionStatus = Literal[
    "recommended",
    "selected",
    "fallback",
    "blocked",
]
AdaptiveExecutionStrategyKind = Literal[
    "direct_provider_path",
    "hybrid_policy_path",
]
AdaptiveExecutionModeTransition = Literal[
    "manual_selection_required",
    "assisted_confirmation_required",
    "auto_selected",
    "auto_downgraded_to_assisted",
    "blocked",
]
ManualActionKind = Literal[
    "local_model_download",
    "provider_provisioning",
    "runtime_installation",
    "runtime_evolution_review",
]
SuggestedSourceCategory = Literal[
    "Ollama",
    "LM Studio",
    "Hugging Face",
    "Provider console",
    "Runtime vendor documentation",
    "Runtime Evolution Review",
]
TokenResourcePosture = Literal[
    "low_token_pressure",
    "moderate_token_pressure",
    "high_token_pressure",
]
RuntimeResourcePosture = Literal[
    "cloud_only",
    "local_runtime_required",
    "hybrid_runtime_required",
    "multi_provider_cloud_required",
]

ADAPTIVE_EXECUTION_PATH_STEP_SERIALIZATION_VERSION = (
    "adaptive_execution_path_step.v1"
)
ADAPTIVE_EXECUTION_SIMULATION_SERIALIZATION_VERSION = (
    "adaptive_execution_simulation.v1"
)
ADAPTIVE_EXECUTION_FALLBACK_SERIALIZATION_VERSION = (
    "adaptive_execution_fallback.v1"
)
ADAPTIVE_EXECUTION_MANUAL_ACTION_SERIALIZATION_VERSION = (
    "adaptive_execution_manual_action.v1"
)
ADAPTIVE_EXECUTION_HYBRID_POLICY_SERIALIZATION_VERSION = (
    "controlled_hybrid_execution_policy.v1"
)
ADAPTIVE_EXECUTION_ESCALATION_RULE_SERIALIZATION_VERSION = (
    "adaptive_execution_escalation_rule.v1"
)
ADAPTIVE_EXECUTION_DECISION_SERIALIZATION_VERSION = (
    "adaptive_execution_policy_decision.v1"
)
ADAPTIVE_EXECUTION_POLICY_PLAN_SERIALIZATION_VERSION = (
    "adaptive_execution_policy_plan.v1"
)
ADAPTIVE_EXECUTION_POLICY_AUTHORITY_BOUNDARY = (
    "V5.5 controlled adaptive execution policy evaluates task-aware options, "
    "availability, simulations, fallbacks, and escalation gates into actionable "
    "allow/confirm/block decisions only; it does not call providers, execute "
    "generation, probe local runtimes, list or download local models, provision "
    "providers, infer API keys, emit HITL requests, mutate workflow graphs, "
    "write storage, modify generated output, or apply Runtime Evolution."
)

_CONTROLLED_POLICY_BLOCKED_RUNTIME_BEHAVIORS = (
    "provider_execution",
    "provider_or_model_routing_mutation",
    "automatic_provider_switching",
    "automatic_model_switching",
    "automatic_model_download",
    "provider_provisioning",
    "runtime_auto_installation",
    "automatic_api_key_assumption",
    "local_runtime_probe",
    "local_model_inventory_scan",
    "hitl_request_emission",
    "workflow_graph_mutation",
    "workflow_execution",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)
_SAFETY_CONSTRAINTS = (
    "no automatic provider switching",
    "no automatic model download",
    "no automatic API key assumption",
    "no provider provisioning",
    "no runtime auto-installation",
    "HITL required before unavailable, expensive, provider-changing, model-changing, high-risk, or download-requiring decisions",
    "no generated output mutation",
    "no Runtime Evolution without HITL",
)
_REQUIRED_TASK_TYPES: tuple[TaskRoutingType, ...] = (
    "coding",
    "reasoning",
    "creative_coding",
    "creative_writing",
    "long_context_reasoning",
    "multimodal_understanding",
    "image_understanding",
    "tool_use",
    "structured_output",
    "fast_draft",
    "low_cost_execution",
    "maximum_quality_execution",
)
_REQUIRED_HYBRID_DIRECTIONS: tuple[HybridRoutingPolicyDirection, ...] = (
    "local_to_cloud",
    "cloud_to_local",
    "cloud_to_cloud",
    "local_to_local",
)


class AdaptiveExecutionAvailabilityContext(BaseModel):
    """Explicit availability inputs; no probing or network checks are performed."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    configured_provider_ids: tuple[ProviderId, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    supported_provider_ids: tuple[ProviderId, ...] = Field(
        default=("openai",),
        min_length=1,
        max_length=4,
    )
    confirmed_local_runtime_kinds: tuple[LocalRuntimeKind, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    installed_local_model_labels: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    safe_auto_cost_bands: tuple[EstimatedCostBand, ...] = Field(
        default=("low", "medium"),
        min_length=1,
        max_length=3,
    )
    safe_auto_risk_bands: tuple[RoutingRiskBand, ...] = Field(
        default=("low",),
        min_length=1,
        max_length=3,
    )
    safe_auto_latency_bands: tuple[EstimatedLatencyBand, ...] = Field(
        default=("fast", "moderate"),
        min_length=1,
        max_length=3,
    )
    availability_context_implemented: Literal[True] = True
    settings_metadata_used: Literal[True] = True
    network_call_performed: Literal[False] = False
    provider_probe_performed: Literal[False] = False
    local_runtime_probe_performed: Literal[False] = False
    local_model_inventory_scan_performed: Literal[False] = False
    local_model_download_attempted: Literal[False] = False
    provider_provisioning_attempted: Literal[False] = False
    runtime_installation_attempted: Literal[False] = False

    @model_validator(mode="after")
    def _deduped(self) -> Self:
        for values, label in (
            (self.configured_provider_ids, "configured_provider_ids"),
            (self.supported_provider_ids, "supported_provider_ids"),
            (self.confirmed_local_runtime_kinds, "confirmed_local_runtime_kinds"),
            (self.installed_local_model_labels, "installed_local_model_labels"),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{label} must be unique")
        return self


class AdaptiveExecutionPathStep(BaseModel):
    """One provider/model step in a simulated execution path."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    step_id: str = Field(min_length=1, max_length=180)
    provider_id: ProviderId
    provider_category: Literal["cloud", "local"]
    model_profile_id: str = Field(min_length=1, max_length=120)
    surface: Literal["cloud", "local"]
    credential_configured: bool
    provider_supported: bool
    local_runtime_confirmed: bool
    local_model_confirmed: bool
    capability_supported: bool
    unavailable_reason_codes: tuple[UnavailableReasonCode, ...] = Field(
        default_factory=tuple,
        max_length=9,
    )
    manual_action_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    immediate_execution_ready: bool
    serialization_version: Literal["adaptive_execution_path_step.v1"] = (
        ADAPTIVE_EXECUTION_PATH_STEP_SERIALIZATION_VERSION
    )
    policy_path_step_implemented: Literal[True] = True
    provider_model_path_selection_implemented: Literal[True] = True
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    local_runtime_probe_implemented: Literal[False] = False
    local_model_inventory_scan_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    automatic_api_key_assumption_implemented: Literal[False] = False

    @model_validator(mode="after")
    def _readiness_matches_reasons(self) -> Self:
        if self.immediate_execution_ready and self.unavailable_reason_codes:
            raise ValueError("ready path steps must not have unavailable reasons")
        if self.provider_category == "local" and self.surface != "local":
            raise ValueError("local providers must use local surface")
        if self.provider_category == "cloud" and self.surface != "cloud":
            raise ValueError("cloud providers must use cloud surface")
        return self


class AdaptiveExecutionSimulation(BaseModel):
    """Deterministic pre-run execution simulation without provider calls."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    simulation_id: str = Field(min_length=1, max_length=180)
    route_name: RouteName
    task_type: TaskRoutingType
    strategy_kind: AdaptiveExecutionStrategyKind
    strategy_id: str = Field(min_length=1, max_length=180)
    policy_direction: HybridRoutingPolicyDirection | None = None
    concrete_strategy_id: str = Field(min_length=1, max_length=180)
    workflow_path_candidate_id: str = Field(min_length=1, max_length=180)
    provider_model_path: tuple[AdaptiveExecutionPathStep, ...] = Field(
        min_length=1,
        max_length=4,
    )
    fallback_option_id: str | None = Field(default=None, max_length=180)
    estimated_quality: EstimatedQualityBand
    estimated_cost: EstimatedCostBand
    estimated_latency: EstimatedLatencyBand
    token_resource_posture: TokenResourcePosture
    runtime_resource_posture: RuntimeResourcePosture
    confidence_score: float = Field(ge=0.0, le=1.0)
    required_hitl_gates: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    unavailable_reason_codes: tuple[UnavailableReasonCode, ...] = Field(
        default_factory=tuple,
        max_length=9,
    )
    blocked_reasons: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    safety_constraints: tuple[str, ...] = Field(
        default=_SAFETY_CONSTRAINTS,
        min_length=1,
        max_length=12,
    )
    serialization_version: Literal["adaptive_execution_simulation.v1"] = (
        ADAPTIVE_EXECUTION_SIMULATION_SERIALIZATION_VERSION
    )
    execution_simulation_implemented: Literal[True] = True
    deterministic_simulation_implemented: Literal[True] = True
    provider_call_performed: Literal[False] = False
    network_call_performed: Literal[False] = False
    generation_execution_performed: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False

    @model_validator(mode="after")
    def _simulation_matches_path(self) -> Self:
        path_reasons = _dedupe_reason_codes(
            tuple(
                reason
                for step in self.provider_model_path
                for reason in step.unavailable_reason_codes
            )
        )
        for reason in path_reasons:
            if reason not in self.unavailable_reason_codes:
                raise ValueError("simulation reasons must include path reasons")
        needs_gate = "hitl_required" in self.unavailable_reason_codes or bool(
            self.blocked_reasons
        )
        manual_gate = "manual_provider_model_selection_required" in self.required_hitl_gates
        if needs_gate and not self.required_hitl_gates:
            raise ValueError("unavailable or blocked reasons require HITL gates")
        if self.required_hitl_gates and not needs_gate and not manual_gate:
            raise ValueError("HITL gates must match manual, unavailable, or blocked reasons")
        return self


class ControlledHybridExecutionPolicy(BaseModel):
    """Guarded hybrid execution policy definition for one cloud/local direction."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    policy_id: str = Field(min_length=1, max_length=180)
    direction: HybridRoutingPolicyDirection
    concrete_strategy_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    when_applies: str = Field(min_length=1, max_length=320)
    required_capabilities: tuple[RoutingCapabilityFamily, ...] = Field(
        min_length=1,
        max_length=8,
    )
    provider_requirements: tuple[str, ...] = Field(min_length=1, max_length=8)
    model_requirements: tuple[str, ...] = Field(min_length=1, max_length=8)
    availability_constraints: tuple[str, ...] = Field(min_length=1, max_length=8)
    fallback_behavior: str = Field(min_length=1, max_length=320)
    cost_quality_latency_tradeoff: str = Field(min_length=1, max_length=320)
    hitl_requirements: tuple[str, ...] = Field(min_length=1, max_length=8)
    execution_readiness_policy: str = Field(min_length=1, max_length=320)
    safety_constraints: tuple[str, ...] = Field(
        default=_SAFETY_CONSTRAINTS,
        min_length=1,
        max_length=12,
    )
    serialization_version: Literal["controlled_hybrid_execution_policy.v1"] = (
        ADAPTIVE_EXECUTION_HYBRID_POLICY_SERIALIZATION_VERSION
    )
    hybrid_execution_policy_implemented: Literal[True] = True
    policy_application_implemented: Literal[True] = True
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False

    @model_validator(mode="after")
    def _policy_id_matches_direction(self) -> Self:
        if self.policy_id != f"controlled_hybrid_policy::{self.direction}":
            raise ValueError("policy_id must match direction")
        return self


class AdaptiveExecutionFallbackDecision(BaseModel):
    """Fallback engine output for one blocked or guarded execution path."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    fallback_id: str = Field(min_length=1, max_length=220)
    primary_option_id: str = Field(min_length=1, max_length=180)
    fallback_option_id: str | None = Field(default=None, max_length=180)
    reason_code: str = Field(min_length=1, max_length=80)
    primary_path_summary: str = Field(min_length=1, max_length=320)
    fallback_path_summary: str = Field(min_length=1, max_length=320)
    reason_summary: str = Field(min_length=1, max_length=320)
    tradeoff_summary: str = Field(min_length=1, max_length=320)
    execution_mode_impact: str = Field(min_length=1, max_length=320)
    suggested_action: str = Field(min_length=1, max_length=320)
    hitl_required: bool
    fallback_available: bool
    serialization_version: Literal["adaptive_execution_fallback.v1"] = (
        ADAPTIVE_EXECUTION_FALLBACK_SERIALIZATION_VERSION
    )
    fallback_engine_implemented: Literal[True] = True
    fallback_policy_applied: Literal[True] = True
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False


class AdaptiveExecutionManualAction(BaseModel):
    """HITL/manual-only surface for intentionally deferred runtime actions."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    action_id: str = Field(min_length=1, max_length=180)
    action_kind: ManualActionKind
    target_provider_id: ProviderId | None = None
    target_runtime_kind: LocalRuntimeKind | None = None
    recommended_model: str = Field(min_length=1, max_length=180)
    suggested_source_category: SuggestedSourceCategory
    reason_summary: str = Field(min_length=1, max_length=320)
    manual_action_required: Literal[True] = True
    hitl_required: Literal[True] = True
    execution_blocked_until_resolved: Literal[True] = True
    serialization_version: Literal["adaptive_execution_manual_action.v1"] = (
        ADAPTIVE_EXECUTION_MANUAL_ACTION_SERIALIZATION_VERSION
    )
    manual_action_surface_implemented: Literal[True] = True
    automatic_model_download_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    runtime_installation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False


class AdaptiveExecutionEscalationRule(BaseModel):
    """Policy-driven escalation rule used by the controlled decision engine."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    rule_id: str = Field(min_length=1, max_length=180)
    trigger_reason: str = Field(min_length=1, max_length=120)
    from_strategy_kind: AdaptiveExecutionStrategyKind
    to_strategy_summary: str = Field(min_length=1, max_length=260)
    escalation_summary: str = Field(min_length=1, max_length=320)
    execution_mode_impact: str = Field(min_length=1, max_length=260)
    hitl_required: bool
    safety_constraints: tuple[str, ...] = Field(
        default=_SAFETY_CONSTRAINTS,
        min_length=1,
        max_length=12,
    )
    serialization_version: Literal["adaptive_execution_escalation_rule.v1"] = (
        ADAPTIVE_EXECUTION_ESCALATION_RULE_SERIALIZATION_VERSION
    )
    escalation_policy_implemented: Literal[True] = True
    policy_driven_escalation_implemented: Literal[True] = True
    escalation_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    automatic_api_key_assumption_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False


class ControlledAdaptiveExecutionOption(BaseModel):
    """One actionable-but-guarded execution option."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    option_id: str = Field(min_length=1, max_length=180)
    status: AdaptiveExecutionOptionStatus
    strategy_kind: AdaptiveExecutionStrategyKind
    strategy_id: str = Field(min_length=1, max_length=180)
    policy_direction: HybridRoutingPolicyDirection | None = None
    concrete_strategy_id: str = Field(min_length=1, max_length=180)
    execution_mode_id: ExecutionModeId
    effective_execution_mode_id: ExecutionModeId
    provider_model_path: tuple[AdaptiveExecutionPathStep, ...] = Field(
        min_length=1,
        max_length=4,
    )
    fallback_option_id: str | None = Field(default=None, max_length=180)
    required_hitl_gates: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    unavailable_reason_codes: tuple[UnavailableReasonCode, ...] = Field(
        default_factory=tuple,
        max_length=9,
    )
    blocked_reasons: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    safety_constraints: tuple[str, ...] = Field(
        default=_SAFETY_CONSTRAINTS,
        min_length=1,
        max_length=12,
    )
    execution_readiness_status: ExecutionReadinessStatus
    execution_allowed_now: bool
    execution_requires_user_confirmation: bool
    execution_blocked: bool
    estimated_quality: EstimatedQualityBand
    estimated_cost: EstimatedCostBand
    estimated_latency: EstimatedLatencyBand
    confidence_score: float = Field(ge=0.0, le=1.0)
    adaptive_policy_score: int = Field(ge=0, le=500)
    simulation: AdaptiveExecutionSimulation
    fallback: AdaptiveExecutionFallbackDecision
    suggested_action: str = Field(min_length=1, max_length=320)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=10)
    controlled_policy_option_implemented: Literal[True] = True
    policy_application_implemented: Literal[True] = True
    execution_policy_application_implemented: Literal[True] = True
    availability_aware_execution_implemented: Literal[True] = True
    provider_model_path_selection_implemented: Literal[True] = True
    fallback_engine_implemented: Literal[True] = True
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    automatic_api_key_assumption_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False

    @model_validator(mode="after")
    def _option_readiness_is_consistent(self) -> Self:
        if self.execution_allowed_now and (
            self.execution_blocked
            or self.execution_requires_user_confirmation
            or self.required_hitl_gates
        ):
            raise ValueError("allowed options cannot be blocked or require confirmation")
        if self.execution_readiness_status == "ready_now" and not self.execution_allowed_now:
            raise ValueError("ready_now options must be allowed")
        if self.execution_readiness_status == "blocked" and not self.execution_blocked:
            raise ValueError("blocked readiness must set execution_blocked")
        if self.simulation.provider_model_path != self.provider_model_path:
            raise ValueError("simulation provider path must match option")
        return self


class ControlledAdaptiveExecutionDecision(BaseModel):
    """Final controlled adaptive execution policy decision."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    decision_id: str = Field(min_length=1, max_length=180)
    task_type: TaskRoutingType
    requested_execution_mode_id: ExecutionModeId
    selected_execution_mode_id: ExecutionModeId
    mode_transition: AdaptiveExecutionModeTransition
    recommended_option_id: str = Field(min_length=1, max_length=180)
    selected_option_id: str | None = Field(default=None, max_length=180)
    recommended_strategy_id: str = Field(min_length=1, max_length=180)
    selected_strategy_id: str | None = Field(default=None, max_length=180)
    selected_provider_model_path: tuple[AdaptiveExecutionPathStep, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    fallback_provider_model_path: tuple[AdaptiveExecutionPathStep, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    fallback_option_id: str | None = Field(default=None, max_length=180)
    required_hitl_gates: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    unavailable_reason_codes: tuple[UnavailableReasonCode, ...] = Field(
        default_factory=tuple,
        max_length=9,
    )
    blocked_reasons: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    safety_constraints: tuple[str, ...] = Field(
        default=_SAFETY_CONSTRAINTS,
        min_length=1,
        max_length=12,
    )
    execution_readiness_status: ExecutionReadinessStatus
    execution_allowed_now: bool
    execution_requires_user_confirmation: bool
    execution_blocked: bool
    fallback: AdaptiveExecutionFallbackDecision
    suggested_action: str = Field(min_length=1, max_length=320)
    serialization_version: Literal["adaptive_execution_policy_decision.v1"] = (
        ADAPTIVE_EXECUTION_DECISION_SERIALIZATION_VERSION
    )
    actionable_decision_implemented: Literal[True] = True
    policy_application_implemented: Literal[True] = True
    execution_policy_application_implemented: Literal[True] = True
    provider_model_path_selection_implemented: Literal[True] = True
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False

    @model_validator(mode="after")
    def _mode_semantics_hold(self) -> Self:
        if self.requested_execution_mode_id == "manual_mode":
            if self.selected_option_id is not None or self.execution_allowed_now:
                raise ValueError("manual mode cannot auto-select or allow execution")
            if not self.execution_requires_user_confirmation:
                raise ValueError("manual mode requires explicit user selection")
        if self.requested_execution_mode_id == "auto_mode":
            if self.required_hitl_gates and self.selected_execution_mode_id != "assisted_mode":
                raise ValueError("unsafe auto decisions must downgrade to assisted")
            if self.execution_allowed_now and self.selected_execution_mode_id != "auto_mode":
                raise ValueError("allowed auto decisions must remain auto mode")
        return self


class ControlledAdaptiveExecutionPlan(BaseModel):
    """Controlled V5.5 adaptive execution policy result."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["adaptive_execution_policy_engine"] = (
        "adaptive_execution_policy_engine"
    )
    serialization_version: Literal["adaptive_execution_policy_plan.v1"] = (
        ADAPTIVE_EXECUTION_POLICY_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ADAPTIVE_EXECUTION_POLICY_AUTHORITY_BOUNDARY,
        max_length=2000,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    source_hybrid_workflow_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_strategy_selection_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    requested_execution_mode_id: ExecutionModeId
    selected_execution_mode_id: ExecutionModeId
    task_types: tuple[TaskRoutingType, ...] = Field(min_length=12, max_length=12)
    hybrid_policy_directions: tuple[HybridRoutingPolicyDirection, ...] = Field(
        min_length=4,
        max_length=4,
    )
    hybrid_policies: tuple[ControlledHybridExecutionPolicy, ...] = Field(
        min_length=4,
        max_length=4,
    )
    execution_options: tuple[ControlledAdaptiveExecutionOption, ...] = Field(
        min_length=5,
        max_length=5,
    )
    option_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    simulations: tuple[AdaptiveExecutionSimulation, ...] = Field(
        min_length=5,
        max_length=5,
    )
    fallback_decisions: tuple[AdaptiveExecutionFallbackDecision, ...] = Field(
        min_length=5,
        max_length=5,
    )
    manual_actions: tuple[AdaptiveExecutionManualAction, ...] = Field(
        min_length=4,
        max_length=4,
    )
    escalation_rules: tuple[AdaptiveExecutionEscalationRule, ...] = Field(
        min_length=6,
        max_length=6,
    )
    selected_decision: ControlledAdaptiveExecutionDecision
    recommended_option_id: str = Field(min_length=1, max_length=180)
    selected_option_id: str | None = Field(default=None, max_length=180)
    fallback_option_id: str | None = Field(default=None, max_length=180)
    candidate_count: int = Field(ge=5, le=5)
    simulation_count: int = Field(ge=5, le=5)
    fallback_decision_count: int = Field(ge=5, le=5)
    hybrid_policy_count: int = Field(ge=4, le=4)
    task_coverage_count: int = Field(ge=12, le=12)
    manual_action_count: int = Field(ge=4, le=4)
    escalation_rule_count: int = Field(ge=6, le=6)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_CONTROLLED_POLICY_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=24,
    )
    controlled_adaptive_execution_engine_implemented: Literal[True] = True
    actionable_execution_decision_implemented: Literal[True] = True
    policy_application_implemented: Literal[True] = True
    execution_policy_application_implemented: Literal[True] = True
    execution_simulation_implemented: Literal[True] = True
    availability_aware_execution_implemented: Literal[True] = True
    intelligent_fallback_engine_implemented: Literal[True] = True
    adaptive_escalation_policy_implemented: Literal[True] = True
    hybrid_execution_policy_implemented: Literal[True] = True
    task_aware_execution_engine_implemented: Literal[True] = True
    execution_modes_implemented: Literal[True] = True
    hitl_manual_action_surfaces_implemented: Literal[True] = True
    controlled_policy_only: Literal[True] = True
    advisory_only: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    runtime_installation_implemented: Literal[False] = False
    automatic_api_key_assumption_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False

    @model_validator(mode="after")
    def _plan_matches_children(self) -> Self:
        if self.task_types != _REQUIRED_TASK_TYPES:
            raise ValueError("task_types must cover required task taxonomy")
        if self.hybrid_policy_directions != _REQUIRED_HYBRID_DIRECTIONS:
            raise ValueError("hybrid policies must cover all directions")
        if self.option_ids != tuple(option.option_id for option in self.execution_options):
            raise ValueError("option_ids must match options")
        if self.candidate_count != len(self.execution_options):
            raise ValueError("candidate_count must match options")
        if self.simulation_count != len(self.simulations):
            raise ValueError("simulation_count must match simulations")
        if self.fallback_decision_count != len(self.fallback_decisions):
            raise ValueError("fallback_decision_count must match fallbacks")
        if self.hybrid_policy_count != len(self.hybrid_policies):
            raise ValueError("hybrid_policy_count must match policies")
        if self.task_coverage_count != len(self.task_types):
            raise ValueError("task_coverage_count must match task_types")
        if self.manual_action_count != len(self.manual_actions):
            raise ValueError("manual_action_count must match manual actions")
        if self.escalation_rule_count != len(self.escalation_rules):
            raise ValueError("escalation_rule_count must match escalation rules")
        if self.recommended_option_id != self.selected_decision.recommended_option_id:
            raise ValueError("recommended option must match decision")
        if self.selected_option_id != self.selected_decision.selected_option_id:
            raise ValueError("selected option must match decision")
        if self.fallback_option_id != self.selected_decision.fallback_option_id:
            raise ValueError("fallback option must match decision")
        return self


def adaptive_execution_availability_context(
    *,
    settings: Settings | None = None,
    configured_provider_ids: tuple[ProviderId, ...] | None = None,
    supported_provider_ids: tuple[ProviderId, ...] = ("openai",),
    confirmed_local_runtime_kinds: tuple[LocalRuntimeKind, ...] = (),
    installed_local_model_labels: tuple[str, ...] = (),
    safe_auto_cost_bands: tuple[EstimatedCostBand, ...] = ("low", "medium"),
    safe_auto_risk_bands: tuple[RoutingRiskBand, ...] = ("low",),
    safe_auto_latency_bands: tuple[EstimatedLatencyBand, ...] = ("fast", "moderate"),
) -> AdaptiveExecutionAvailabilityContext:
    """Build explicit availability context without probing providers or runtimes."""

    provider_ids = configured_provider_ids
    if provider_ids is None:
        provider_ids = ("openai",) if settings and settings.has_openai_api_key else ()
    return AdaptiveExecutionAvailabilityContext(
        configured_provider_ids=provider_ids,
        supported_provider_ids=supported_provider_ids,
        confirmed_local_runtime_kinds=confirmed_local_runtime_kinds,
        installed_local_model_labels=installed_local_model_labels,
        safe_auto_cost_bands=safe_auto_cost_bands,
        safe_auto_risk_bands=safe_auto_risk_bands,
        safe_auto_latency_bands=safe_auto_latency_bands,
    )


def evaluate_adaptive_execution_policy(
    *,
    task_type: TaskRoutingType | str = "creative_coding",
    route: RouteName | str = RouteName.GENERATE,
    execution_mode_id: ExecutionModeId | str | None = None,
    settings: Settings | None = None,
    availability_context: AdaptiveExecutionAvailabilityContext | None = None,
    hybrid_workflow: HybridWorkflowOptimizationPlan | None = None,
    strategy_selection: AdaptiveExecutionStrategySelectionPlan | None = None,
) -> ControlledAdaptiveExecutionPlan:
    """Return a controlled adaptive execution decision without executing work."""

    route_name = _resolve_route(route)
    normalized_task_type = str(task_type).strip()
    if normalized_task_type not in _REQUIRED_TASK_TYPES:
        raise ValueError("task_type must be a supported V5.5 task type")
    context = availability_context or adaptive_execution_availability_context(
        settings=settings,
    )
    hybrid_plan = hybrid_workflow or optimize_hybrid_workflow(
        task_type=normalized_task_type,
        route=route_name,
        execution_mode_id=execution_mode_id,
    )
    strategy_plan = strategy_selection or select_dynamic_execution_strategy(
        task_type=normalized_task_type,
        route=route_name,
        execution_mode_id=execution_mode_id,
        hybrid_workflow=hybrid_plan,
    )
    task_registry = task_aware_routing_registry()
    task_decision = _task_decision(normalized_task_type, task_registry.decisions)
    requested_mode = _requested_mode(execution_mode_id, task_decision.execution_mode_id)
    mode_registry = routing_execution_mode_registry()
    if requested_mode not in mode_registry.execution_mode_ids:
        raise ValueError("execution_mode_id must be a known execution mode")

    hybrid_policies = _controlled_hybrid_policies(task_decision)
    options = _execution_options(
        route_name=route_name,
        task_decision=task_decision,
        requested_mode=requested_mode,
        context=context,
        hybrid_plan=hybrid_plan,
        strategy_plan=strategy_plan,
    )
    options = _attach_fallbacks(options)
    recommended = _recommended_option(options)
    selected = _selected_option(recommended, options, requested_mode)
    fallback_option = _fallback_option_for_decision(recommended, options)
    decision = _decision_from_options(
        task_decision=task_decision,
        requested_mode=requested_mode,
        recommended=recommended,
        selected=selected,
        fallback_option=fallback_option,
    )
    return ControlledAdaptiveExecutionPlan(
        route_name=route_name,
        task_type=task_decision.task_type,
        source_hybrid_workflow_serialization_version=hybrid_plan.serialization_version,
        source_strategy_selection_serialization_version=strategy_plan.serialization_version,
        requested_execution_mode_id=requested_mode,
        selected_execution_mode_id=decision.selected_execution_mode_id,
        task_types=task_registry.task_types,
        hybrid_policy_directions=tuple(policy.direction for policy in hybrid_policies),
        hybrid_policies=hybrid_policies,
        execution_options=options,
        option_ids=tuple(option.option_id for option in options),
        simulations=tuple(option.simulation for option in options),
        fallback_decisions=tuple(option.fallback for option in options),
        manual_actions=_manual_actions(task_decision),
        escalation_rules=_escalation_rules(),
        selected_decision=decision,
        recommended_option_id=decision.recommended_option_id,
        selected_option_id=decision.selected_option_id,
        fallback_option_id=decision.fallback_option_id,
        candidate_count=len(options),
        simulation_count=len(options),
        fallback_decision_count=len(options),
        hybrid_policy_count=len(hybrid_policies),
        task_coverage_count=len(task_registry.task_types),
        manual_action_count=4,
        escalation_rule_count=6,
    )


def simulate_adaptive_execution(
    *,
    task_type: TaskRoutingType | str = "creative_coding",
    route: RouteName | str = RouteName.GENERATE,
    execution_mode_id: ExecutionModeId | str | None = None,
    settings: Settings | None = None,
    availability_context: AdaptiveExecutionAvailabilityContext | None = None,
) -> tuple[AdaptiveExecutionSimulation, ...]:
    """Return deterministic pre-run simulations without calling providers."""

    plan = evaluate_adaptive_execution_policy(
        task_type=task_type,
        route=route,
        execution_mode_id=execution_mode_id,
        settings=settings,
        availability_context=availability_context,
    )
    return plan.simulations


def adaptive_execution_option_by_id(
    option_id: str,
    plan: ControlledAdaptiveExecutionPlan | None = None,
) -> ControlledAdaptiveExecutionOption | None:
    """Return one controlled adaptive execution option."""

    source_plan = plan or evaluate_adaptive_execution_policy()
    normalized_option_id = str(option_id).strip()
    for option in source_plan.execution_options:
        if option.option_id == normalized_option_id:
            return option
    return None


def adaptive_execution_options_for_readiness(
    readiness: ExecutionReadinessStatus,
    plan: ControlledAdaptiveExecutionPlan | None = None,
) -> tuple[ControlledAdaptiveExecutionOption, ...]:
    """Return controlled options by readiness status."""

    source_plan = plan or evaluate_adaptive_execution_policy()
    return tuple(
        option
        for option in source_plan.execution_options
        if option.execution_readiness_status == readiness
    )


def _execution_options(
    *,
    route_name: RouteName,
    task_decision: TaskAwareRoutingDecision,
    requested_mode: ExecutionModeId,
    context: AdaptiveExecutionAvailabilityContext,
    hybrid_plan: HybridWorkflowOptimizationPlan,
    strategy_plan: AdaptiveExecutionStrategySelectionPlan,
) -> tuple[ControlledAdaptiveExecutionOption, ...]:
    direct = _direct_option(
        route_name=route_name,
        task_decision=task_decision,
        requested_mode=requested_mode,
        context=context,
        strategy_plan=strategy_plan,
    )
    hybrid_options = tuple(
        _hybrid_option(
            route_name=route_name,
            task_decision=task_decision,
            requested_mode=requested_mode,
            context=context,
            candidate=candidate,
            strategy_plan=strategy_plan,
        )
        for candidate in hybrid_plan.candidates
    )
    return (direct, *hybrid_options)


def _direct_option(
    *,
    route_name: RouteName,
    task_decision: TaskAwareRoutingDecision,
    requested_mode: ExecutionModeId,
    context: AdaptiveExecutionAvailabilityContext,
    strategy_plan: AdaptiveExecutionStrategySelectionPlan,
) -> ControlledAdaptiveExecutionOption:
    provider_sequence: tuple[ProviderId, ...] = ("openai",)
    path = _provider_model_path(
        provider_sequence=provider_sequence,
        model_profile_sequence=(task_decision.recommended_model_profile_id,),
        requirements=task_decision.capability_requirements,
        context=context,
    )
    reasons = _path_reasons(path, _task_policy_reasons(task_decision))
    blocked = _policy_blocked_reasons(
        mode=requested_mode,
        cost=task_decision.estimated_cost,
        latency=task_decision.estimated_latency,
        quality=task_decision.estimated_quality,
        risk=task_decision.risk_band,
        context=context,
    )
    return _option(
        route_name=route_name,
        task_type=task_decision.task_type,
        option_id="adaptive_execution_option::direct_task_provider",
        strategy_kind="direct_provider_path",
        strategy_id=strategy_plan.selected_strategy_id,
        concrete_strategy_id=f"direct_provider::{task_decision.task_type}",
        policy_direction=None,
        requested_mode=requested_mode,
        provider_model_path=path,
        quality=task_decision.estimated_quality,
        cost=task_decision.estimated_cost,
        latency=task_decision.estimated_latency,
        confidence=task_decision.confidence_score,
        risk=task_decision.risk_band,
        workflow_path_candidate_id="task_aware_direct_provider_path",
        unavailable_reason_codes=reasons,
        blocked_reasons=blocked,
    )


def _hybrid_option(
    *,
    route_name: RouteName,
    task_decision: TaskAwareRoutingDecision,
    requested_mode: ExecutionModeId,
    context: AdaptiveExecutionAvailabilityContext,
    candidate: HybridWorkflowOptimizationCandidate,
    strategy_plan: AdaptiveExecutionStrategySelectionPlan,
) -> ControlledAdaptiveExecutionOption:
    path = _provider_model_path(
        provider_sequence=candidate.provider_sequence,
        model_profile_sequence=candidate.model_profile_sequence,
        requirements=task_decision.capability_requirements,
        context=context,
    )
    reasons = _path_reasons(path, _task_policy_reasons(task_decision))
    blocked = _policy_blocked_reasons(
        mode=requested_mode,
        cost=candidate.estimated_cost,
        latency=candidate.estimated_latency,
        quality=candidate.estimated_quality,
        risk=candidate.risk_band,
        context=context,
    )
    return _option(
        route_name=route_name,
        task_type=task_decision.task_type,
        option_id=f"adaptive_execution_option::{candidate.policy_direction}",
        strategy_kind="hybrid_policy_path",
        strategy_id=strategy_plan.selected_strategy_id,
        concrete_strategy_id=_concrete_strategy_id(candidate.policy_direction),
        policy_direction=candidate.policy_direction,
        requested_mode=requested_mode,
        provider_model_path=path,
        quality=candidate.estimated_quality,
        cost=candidate.estimated_cost,
        latency=candidate.estimated_latency,
        confidence=candidate.confidence_score,
        risk=candidate.risk_band,
        workflow_path_candidate_id=candidate.workflow_path_candidate_id,
        unavailable_reason_codes=reasons,
        blocked_reasons=blocked,
    )


def _option(
    *,
    route_name: RouteName,
    task_type: TaskRoutingType,
    option_id: str,
    strategy_kind: AdaptiveExecutionStrategyKind,
    strategy_id: str,
    concrete_strategy_id: str,
    policy_direction: HybridRoutingPolicyDirection | None,
    requested_mode: ExecutionModeId,
    provider_model_path: tuple[AdaptiveExecutionPathStep, ...],
    quality: EstimatedQualityBand,
    cost: EstimatedCostBand,
    latency: EstimatedLatencyBand,
    confidence: float,
    risk: RoutingRiskBand,
    workflow_path_candidate_id: str,
    unavailable_reason_codes: tuple[UnavailableReasonCode, ...],
    blocked_reasons: tuple[str, ...],
) -> ControlledAdaptiveExecutionOption:
    hitl = _hitl_gates(
        mode=requested_mode,
        unavailable_reason_codes=unavailable_reason_codes,
        blocked_reasons=blocked_reasons,
        risk=risk,
    )
    readiness, allowed, confirmation, blocked = _readiness(
        mode=requested_mode,
        unavailable_reason_codes=unavailable_reason_codes,
        blocked_reasons=blocked_reasons,
        hitl_gates=hitl,
    )
    score = _policy_score(quality, cost, latency, confidence, risk, unavailable_reason_codes, blocked_reasons)
    simulation = AdaptiveExecutionSimulation(
        simulation_id=f"adaptive_execution_simulation::{option_id.split('::')[-1]}",
        route_name=route_name,
        task_type=task_type,
        strategy_kind=strategy_kind,
        strategy_id=strategy_id,
        policy_direction=policy_direction,
        concrete_strategy_id=concrete_strategy_id,
        workflow_path_candidate_id=workflow_path_candidate_id,
        provider_model_path=provider_model_path,
        fallback_option_id=None,
        estimated_quality=quality,
        estimated_cost=cost,
        estimated_latency=latency,
        token_resource_posture=_token_posture(cost, latency),
        runtime_resource_posture=_resource_posture(provider_model_path),
        confidence_score=confidence,
        required_hitl_gates=hitl,
        unavailable_reason_codes=unavailable_reason_codes,
        blocked_reasons=blocked_reasons,
    )
    fallback = _fallback_decision(
        primary_option_id=option_id,
        fallback_option_id=None,
        reason_code=_fallback_reason(unavailable_reason_codes, blocked_reasons),
        primary_path=provider_model_path,
        fallback_path=(),
        mode=requested_mode,
    )
    return ControlledAdaptiveExecutionOption(
        option_id=option_id,
        status="blocked" if blocked else "fallback",
        strategy_kind=strategy_kind,
        strategy_id=strategy_id,
        policy_direction=policy_direction,
        concrete_strategy_id=concrete_strategy_id,
        execution_mode_id=requested_mode,
        effective_execution_mode_id=_effective_mode(requested_mode, hitl, blocked),
        provider_model_path=provider_model_path,
        fallback_option_id=None,
        required_hitl_gates=hitl,
        unavailable_reason_codes=unavailable_reason_codes,
        blocked_reasons=blocked_reasons,
        execution_readiness_status=readiness,
        execution_allowed_now=allowed,
        execution_requires_user_confirmation=confirmation,
        execution_blocked=blocked,
        estimated_quality=quality,
        estimated_cost=cost,
        estimated_latency=latency,
        confidence_score=confidence,
        adaptive_policy_score=score,
        simulation=simulation,
        fallback=fallback,
        suggested_action=_suggested_action(unavailable_reason_codes, blocked_reasons),
        evidence=(
            f"strategy_kind:{strategy_kind}",
            f"requested_mode:{requested_mode}",
            f"path_steps:{len(provider_model_path)}",
            f"quality:{quality}",
            f"cost:{cost}",
            f"latency:{latency}",
        ),
    )


def _attach_fallbacks(
    options: tuple[ControlledAdaptiveExecutionOption, ...],
) -> tuple[ControlledAdaptiveExecutionOption, ...]:
    ready = tuple(option for option in options if option.execution_allowed_now)
    best_ready = max(ready, key=lambda option: option.adaptive_policy_score) if ready else None
    fallback_id = best_ready.option_id if best_ready is not None else None
    updated: list[ControlledAdaptiveExecutionOption] = []
    for option in options:
        target_id = None if option.option_id == fallback_id else fallback_id
        target = best_ready if target_id else None
        fallback = _fallback_decision(
            primary_option_id=option.option_id,
            fallback_option_id=target_id,
            reason_code=_fallback_reason(option.unavailable_reason_codes, option.blocked_reasons),
            primary_path=option.provider_model_path,
            fallback_path=target.provider_model_path if target else (),
            mode=option.execution_mode_id,
        )
        simulation = option.simulation.model_copy(update={"fallback_option_id": target_id})
        status: AdaptiveExecutionOptionStatus = (
            "selected"
            if option.execution_allowed_now and option.option_id == fallback_id
            else "blocked"
            if option.execution_blocked
            else "fallback"
        )
        updated.append(
            option.model_copy(
                update={
                    "status": status,
                    "fallback_option_id": target_id,
                    "simulation": simulation,
                    "fallback": fallback,
                }
            )
        )
    recommended = _recommended_option(tuple(updated))
    return tuple(
        option.model_copy(update={"status": "recommended"})
        if option.option_id == recommended.option_id and option.status != "selected"
        else option
        for option in updated
    )


def _decision_from_options(
    *,
    task_decision: TaskAwareRoutingDecision,
    requested_mode: ExecutionModeId,
    recommended: ControlledAdaptiveExecutionOption,
    selected: ControlledAdaptiveExecutionOption | None,
    fallback_option: ControlledAdaptiveExecutionOption | None,
) -> ControlledAdaptiveExecutionDecision:
    selected_option = selected if requested_mode != "manual_mode" else None
    fallback_path = fallback_option.provider_model_path if fallback_option else ()
    option_for_state = selected_option or recommended
    selected_mode = _effective_mode(
        requested_mode,
        option_for_state.required_hitl_gates,
        option_for_state.execution_blocked,
    )
    transition = _mode_transition(requested_mode, selected_mode, selected_option, option_for_state)
    return ControlledAdaptiveExecutionDecision(
        decision_id=f"adaptive_execution_decision::{task_decision.task_type}",
        task_type=task_decision.task_type,
        requested_execution_mode_id=requested_mode,
        selected_execution_mode_id=selected_mode,
        mode_transition=transition,
        recommended_option_id=recommended.option_id,
        selected_option_id=selected_option.option_id if selected_option else None,
        recommended_strategy_id=recommended.strategy_id,
        selected_strategy_id=selected_option.strategy_id if selected_option else None,
        selected_provider_model_path=(
            selected_option.provider_model_path if selected_option else ()
        ),
        fallback_provider_model_path=fallback_path,
        fallback_option_id=fallback_option.option_id if fallback_option else None,
        required_hitl_gates=option_for_state.required_hitl_gates,
        unavailable_reason_codes=option_for_state.unavailable_reason_codes,
        blocked_reasons=option_for_state.blocked_reasons,
        execution_readiness_status=option_for_state.execution_readiness_status,
        execution_allowed_now=(
            bool(selected_option) and option_for_state.execution_allowed_now
        ),
        execution_requires_user_confirmation=(
            requested_mode == "manual_mode"
            or option_for_state.execution_requires_user_confirmation
        ),
        execution_blocked=option_for_state.execution_blocked,
        fallback=option_for_state.fallback,
        suggested_action=_decision_action(requested_mode, option_for_state),
    )


def _provider_model_path(
    *,
    provider_sequence: tuple[ProviderId, ...],
    model_profile_sequence: tuple[str, ...],
    requirements: tuple[RoutingCapabilityFamily, ...],
    context: AdaptiveExecutionAvailabilityContext,
) -> tuple[AdaptiveExecutionPathStep, ...]:
    profiles = routing_provider_profile_registry().provider_profiles
    profile_by_provider = {profile.provider_id: profile for profile in profiles}
    return tuple(
        _path_step(
            index=index,
            provider_id=provider,
            model_profile_id=model_profile_sequence[index],
            requirements=requirements,
            context=context,
            profile=profile_by_provider[provider],
        )
        for index, provider in enumerate(provider_sequence)
    )


def _path_step(
    *,
    index: int,
    provider_id: ProviderId,
    model_profile_id: str,
    requirements: tuple[RoutingCapabilityFamily, ...],
    context: AdaptiveExecutionAvailabilityContext,
    profile: object,
) -> AdaptiveExecutionPathStep:
    category = "local" if provider_id == "local" else "cloud"
    provider_supported = provider_id in context.supported_provider_ids
    credential_configured = provider_id == "local" or provider_id in context.configured_provider_ids
    local_runtime_confirmed = (
        provider_id != "local" or bool(context.confirmed_local_runtime_kinds)
    )
    local_model_confirmed = provider_id != "local" or bool(context.installed_local_model_labels)
    capability_supported = set(requirements).issubset(
        set(getattr(profile, "supported_capability_families"))
    )
    reasons: list[UnavailableReasonCode] = []
    if not provider_supported:
        reasons.append("provider_unsupported")
    if provider_id != "local" and not credential_configured:
        reasons.append("missing_api_key")
    if provider_id == "local":
        if not local_runtime_confirmed:
            reasons.append("local_runtime_unavailable")
        if not local_model_confirmed:
            reasons.append("local_model_not_installed")
        if not local_runtime_confirmed:
            reasons.append("insufficient_local_resources")
    if not capability_supported:
        reasons.append("missing_modality_support")
    if reasons:
        reasons.append("hitl_required")
    deduped = _dedupe_reason_codes(tuple(reasons))
    return AdaptiveExecutionPathStep(
        step_id=f"adaptive_execution_path_step::{index}::{provider_id}",
        provider_id=provider_id,
        provider_category=category,
        model_profile_id=model_profile_id,
        surface=category,
        credential_configured=credential_configured,
        provider_supported=provider_supported,
        local_runtime_confirmed=local_runtime_confirmed,
        local_model_confirmed=local_model_confirmed,
        capability_supported=capability_supported,
        unavailable_reason_codes=deduped,
        manual_action_ids=_manual_action_ids(deduped),
        immediate_execution_ready=not deduped,
    )


def _controlled_hybrid_policies(
    task_decision: TaskAwareRoutingDecision,
) -> tuple[ControlledHybridExecutionPolicy, ...]:
    source_policies = advisory_hybrid_routing_policy_registry().policies
    return tuple(
        ControlledHybridExecutionPolicy(
            policy_id=f"controlled_hybrid_policy::{policy.direction}",
            direction=policy.direction,
            concrete_strategy_ids=_concrete_strategy_ids(policy.direction),
            when_applies=_when_applies(policy.direction),
            required_capabilities=task_decision.capability_requirements,
            provider_requirements=_provider_requirements(policy.direction),
            model_requirements=_model_requirements(policy.direction),
            availability_constraints=policy.availability_constraints,
            fallback_behavior=policy.fallback_logic,
            cost_quality_latency_tradeoff=policy.cost_quality_latency_tradeoff,
            hitl_requirements=policy.hitl_requirements,
            execution_readiness_policy=_readiness_policy(policy.direction),
        )
        for policy in source_policies
    )


def _manual_actions(
    task_decision: TaskAwareRoutingDecision,
) -> tuple[AdaptiveExecutionManualAction, ...]:
    return (
        AdaptiveExecutionManualAction(
            action_id="adaptive_execution_manual_action::local_model_download",
            action_kind="local_model_download",
            target_provider_id="local",
            target_runtime_kind="ollama",
            recommended_model=task_decision.fallback_model_profile_id,
            suggested_source_category="Ollama",
            reason_summary=(
                "Local model inventory does not confirm installation; user must "
                "install or select a local model manually before local execution."
            ),
        ),
        AdaptiveExecutionManualAction(
            action_id="adaptive_execution_manual_action::provider_provisioning",
            action_kind="provider_provisioning",
            target_provider_id="openai",
            recommended_model=task_decision.recommended_model_profile_id,
            suggested_source_category="Provider console",
            reason_summary=(
                "Provider credentials or account configuration are missing; user "
                "must configure provider access manually."
            ),
        ),
        AdaptiveExecutionManualAction(
            action_id="adaptive_execution_manual_action::runtime_installation",
            action_kind="runtime_installation",
            target_provider_id="local",
            target_runtime_kind="ollama",
            recommended_model=task_decision.fallback_model_profile_id,
            suggested_source_category="Runtime vendor documentation",
            reason_summary=(
                "Local runtime readiness is unconfirmed; user must install and "
                "start the runtime manually."
            ),
        ),
        AdaptiveExecutionManualAction(
            action_id="adaptive_execution_manual_action::runtime_evolution_review",
            action_kind="runtime_evolution_review",
            recommended_model="Runtime Pack",
            suggested_source_category="Runtime Evolution Review",
            reason_summary=(
                "Runtime Evolution remains review-only and requires HITL before "
                "any runtime methodology or workflow change."
            ),
        ),
    )


def _escalation_rules() -> tuple[AdaptiveExecutionEscalationRule, ...]:
    return (
        _escalation_rule(
            "local_draft_insufficient",
            "direct_provider_path",
            "Escalate local draft or exploration to cloud synthesis after HITL.",
            "Local draft insufficient -> cloud synthesis.",
            "Assisted confirmation required before provider/model change.",
            True,
        ),
        _escalation_rule(
            "low_expected_quality",
            "direct_provider_path",
            "Escalate low-cost model to higher-quality model profile.",
            "Low expected quality -> higher quality model.",
            "Assisted/HITL gate required before quality/cost change.",
            True,
        ),
        _escalation_rule(
            "high_latency_policy",
            "hybrid_policy_path",
            "Prefer fast draft policy or ask for assisted confirmation.",
            "Fast draft insufficient -> quality mode only with HITL.",
            "Auto mode downgrades to Assisted when latency is unsafe.",
            True,
        ),
        _escalation_rule(
            "local_runtime_unavailable",
            "hybrid_policy_path",
            "Use configured cloud fallback when local runtime is unavailable.",
            "Missing local runtime -> cloud fallback.",
            "Execution blocked until cloud fallback is confirmed or safe for Auto.",
            True,
        ),
        _escalation_rule(
            "provider_unsupported",
            "hybrid_policy_path",
            "Use supported cloud provider or local fallback if confirmed.",
            "Unavailable cloud provider -> alternative provider/local fallback.",
            "Provider-changing decisions require HITL unless already safe.",
            True,
        ),
        _escalation_rule(
            "high_cost_policy",
            "direct_provider_path",
            "Downgrade Auto to Assisted before high-cost strategy use.",
            "High-cost strategy -> Assisted/HITL gate.",
            "Execution waits for user confirmation.",
            True,
        ),
    )


def _escalation_rule(
    trigger: str,
    strategy_kind: AdaptiveExecutionStrategyKind,
    to_strategy: str,
    summary: str,
    mode_impact: str,
    hitl_required: bool,
) -> AdaptiveExecutionEscalationRule:
    return AdaptiveExecutionEscalationRule(
        rule_id=f"adaptive_execution_escalation::{trigger}",
        trigger_reason=trigger,
        from_strategy_kind=strategy_kind,
        to_strategy_summary=to_strategy,
        escalation_summary=summary,
        execution_mode_impact=mode_impact,
        hitl_required=hitl_required,
    )


def _task_decision(
    task_type: str,
    decisions: tuple[TaskAwareRoutingDecision, ...],
) -> TaskAwareRoutingDecision:
    for decision in decisions:
        if decision.task_type == task_type:
            return decision
    raise ValueError("task_type must be present in task-aware routing registry")


def _requested_mode(
    requested: ExecutionModeId | str | None,
    default: ExecutionModeId,
) -> ExecutionModeId:
    return str(requested or default).strip()  # type: ignore[return-value]


def _recommended_option(
    options: tuple[ControlledAdaptiveExecutionOption, ...],
) -> ControlledAdaptiveExecutionOption:
    allowed = tuple(option for option in options if option.execution_allowed_now)
    if allowed:
        return max(allowed, key=lambda option: option.adaptive_policy_score)
    return max(options, key=lambda option: option.adaptive_policy_score)


def _selected_option(
    recommended: ControlledAdaptiveExecutionOption,
    options: tuple[ControlledAdaptiveExecutionOption, ...],
    mode: ExecutionModeId,
) -> ControlledAdaptiveExecutionOption | None:
    if mode == "manual_mode":
        return None
    if mode == "auto_mode" and recommended.execution_allowed_now:
        return recommended
    if mode == "assisted_mode" and not recommended.execution_blocked:
        return recommended
    if mode == "auto_mode":
        ready = tuple(option for option in options if option.execution_allowed_now)
        return max(ready, key=lambda option: option.adaptive_policy_score) if ready else None
    return None


def _fallback_option_for_decision(
    recommended: ControlledAdaptiveExecutionOption,
    options: tuple[ControlledAdaptiveExecutionOption, ...],
) -> ControlledAdaptiveExecutionOption | None:
    if recommended.fallback_option_id is None:
        return None
    for option in options:
        if option.option_id == recommended.fallback_option_id:
            return option
    return None


def _readiness(
    *,
    mode: ExecutionModeId,
    unavailable_reason_codes: tuple[UnavailableReasonCode, ...],
    blocked_reasons: tuple[str, ...],
    hitl_gates: tuple[str, ...],
) -> tuple[ExecutionReadinessStatus, bool, bool, bool]:
    if unavailable_reason_codes or blocked_reasons:
        return "blocked", False, True, True
    if mode == "auto_mode" and not hitl_gates:
        return "ready_now", True, False, False
    return "requires_confirmation", False, True, False


def _effective_mode(
    mode: ExecutionModeId,
    hitl_gates: tuple[str, ...],
    blocked: bool,
) -> ExecutionModeId:
    if mode == "auto_mode" and (hitl_gates or blocked):
        return "assisted_mode"
    return mode


def _mode_transition(
    requested: ExecutionModeId,
    selected_mode: ExecutionModeId,
    selected_option: ControlledAdaptiveExecutionOption | None,
    state_option: ControlledAdaptiveExecutionOption,
) -> AdaptiveExecutionModeTransition:
    if state_option.execution_blocked:
        return "blocked"
    if requested == "manual_mode":
        return "manual_selection_required"
    if requested == "assisted_mode":
        return "assisted_confirmation_required"
    if requested == "auto_mode" and selected_mode == "auto_mode" and selected_option:
        return "auto_selected"
    return "auto_downgraded_to_assisted"


def _path_reasons(
    path: tuple[AdaptiveExecutionPathStep, ...],
    source: tuple[UnavailableReasonCode, ...],
) -> tuple[UnavailableReasonCode, ...]:
    return _dedupe_reason_codes(
        tuple(reason for step in path for reason in step.unavailable_reason_codes)
        + source
    )


def _task_policy_reasons(
    task_decision: TaskAwareRoutingDecision,
) -> tuple[UnavailableReasonCode, ...]:
    policy_reasons = tuple(
        reason
        for reason in task_decision.unavailable_reason_codes
        if reason
        in {
            "missing_modality_support",
            "cost_policy_blocked",
            "latency_policy_blocked",
        }
    )
    if policy_reasons and "hitl_required" not in policy_reasons:
        return (*policy_reasons, "hitl_required")
    return policy_reasons


def _policy_blocked_reasons(
    *,
    mode: ExecutionModeId,
    cost: EstimatedCostBand,
    latency: EstimatedLatencyBand,
    quality: EstimatedQualityBand,
    risk: RoutingRiskBand,
    context: AdaptiveExecutionAvailabilityContext,
) -> tuple[str, ...]:
    blocked: list[str] = []
    if mode == "auto_mode" and cost not in context.safe_auto_cost_bands:
        blocked.append("high_cost_policy")
    if mode == "auto_mode" and latency not in context.safe_auto_latency_bands:
        blocked.append("high_latency_policy")
    if quality == "low":
        blocked.append("low_expected_quality")
    if risk not in context.safe_auto_risk_bands and mode == "auto_mode":
        blocked.append("high_risk_execution_path")
    return tuple(blocked)


def _hitl_gates(
    *,
    mode: ExecutionModeId,
    unavailable_reason_codes: tuple[UnavailableReasonCode, ...],
    blocked_reasons: tuple[str, ...],
    risk: RoutingRiskBand,
) -> tuple[str, ...]:
    gates: list[str] = []
    if mode == "manual_mode":
        gates.append("manual_provider_model_selection_required")
    if unavailable_reason_codes:
        gates.append("hitl_before_unavailable_provider_or_model")
    if "missing_api_key" in unavailable_reason_codes:
        gates.append("hitl_before_provider_provisioning")
    if "local_runtime_unavailable" in unavailable_reason_codes:
        gates.append("hitl_before_runtime_installation")
    if "local_model_not_installed" in unavailable_reason_codes:
        gates.append("hitl_before_local_model_download")
    if "high_cost_policy" in blocked_reasons:
        gates.append("hitl_before_expensive_execution")
    if "high_latency_policy" in blocked_reasons:
        gates.append("hitl_before_high_latency_execution")
    if risk == "high" or "high_risk_execution_path" in blocked_reasons:
        gates.append("hitl_before_high_risk_execution")
    if unavailable_reason_codes or blocked_reasons:
        gates.append("hitl_required")
    return tuple(dict.fromkeys(gates))


def _fallback_decision(
    *,
    primary_option_id: str,
    fallback_option_id: str | None,
    reason_code: str,
    primary_path: tuple[AdaptiveExecutionPathStep, ...],
    fallback_path: tuple[AdaptiveExecutionPathStep, ...],
    mode: ExecutionModeId,
) -> AdaptiveExecutionFallbackDecision:
    return AdaptiveExecutionFallbackDecision(
        fallback_id=f"adaptive_execution_fallback::{primary_option_id.split('::')[-1]}",
        primary_option_id=primary_option_id,
        fallback_option_id=fallback_option_id,
        reason_code=reason_code,
        primary_path_summary=_path_summary(primary_path),
        fallback_path_summary=(
            _path_summary(fallback_path)
            if fallback_path
            else "No immediately executable fallback path is available."
        ),
        reason_summary=_fallback_reason_summary(reason_code),
        tradeoff_summary=_fallback_tradeoff(reason_code),
        execution_mode_impact=_fallback_mode_impact(mode, fallback_option_id),
        suggested_action=_fallback_suggested_action(reason_code),
        hitl_required=reason_code != "ready",
        fallback_available=fallback_option_id is not None,
    )


def _manual_action_ids(
    reasons: tuple[UnavailableReasonCode, ...],
) -> tuple[str, ...]:
    action_ids: list[str] = []
    if "local_model_not_installed" in reasons:
        action_ids.append("adaptive_execution_manual_action::local_model_download")
    if "missing_api_key" in reasons or "provider_unsupported" in reasons:
        action_ids.append("adaptive_execution_manual_action::provider_provisioning")
    if "local_runtime_unavailable" in reasons:
        action_ids.append("adaptive_execution_manual_action::runtime_installation")
    return tuple(action_ids)


def _fallback_reason(
    reasons: tuple[UnavailableReasonCode, ...],
    blocked: tuple[str, ...],
) -> str:
    for reason in (
        "missing_api_key",
        "provider_unsupported",
        "missing_modality_support",
        "local_runtime_unavailable",
        "local_model_not_installed",
        "insufficient_local_resources",
    ):
        if reason in reasons:
            return reason
    for reason in (
        "high_cost_policy",
        "high_latency_policy",
        "low_expected_quality",
        "high_risk_execution_path",
    ):
        if reason in blocked:
            return reason
    return "ready"


def _fallback_reason_summary(reason_code: str) -> str:
    summaries = {
        "missing_api_key": "Primary provider credentials are missing or unverified.",
        "provider_unsupported": "Primary provider adapter or capability is unavailable.",
        "missing_modality_support": "Required modality support is not confirmed.",
        "local_runtime_unavailable": "Local runtime readiness is not confirmed.",
        "local_model_not_installed": "Local model installation is not confirmed.",
        "insufficient_local_resources": "Local hardware fit has not been confirmed.",
        "high_cost_policy": "Estimated cost exceeds safe Auto policy.",
        "high_latency_policy": "Estimated latency exceeds safe Auto policy.",
        "low_expected_quality": "Expected quality is too low for immediate use.",
        "high_risk_execution_path": "Risk posture requires human review.",
        "ready": "Primary path is ready under the requested policy.",
    }
    return summaries[reason_code]


def _fallback_tradeoff(reason_code: str) -> str:
    if reason_code in {"missing_api_key", "provider_unsupported"}:
        return "Fallback may use a configured provider with lower quality or narrower capability."
    if reason_code.startswith("local_") or reason_code == "insufficient_local_resources":
        return "Cloud fallback can avoid local setup but may increase credential and cost requirements."
    if reason_code == "high_cost_policy":
        return "Lower-cost fallback may reduce quality or context capacity."
    if reason_code == "high_latency_policy":
        return "Faster fallback may reduce quality or refinement depth."
    if reason_code == "low_expected_quality":
        return "Higher-quality fallback may increase cost or latency."
    if reason_code == "high_risk_execution_path":
        return "Safer fallback may require manual confirmation and reduced autonomy."
    return "No fallback tradeoff is required."


def _fallback_mode_impact(mode: ExecutionModeId, fallback_option_id: str | None) -> str:
    if mode == "manual_mode":
        return "Manual mode requires user-selected provider/model path."
    if fallback_option_id is None:
        return "Execution remains blocked until a safe path is configured or confirmed."
    if mode == "auto_mode":
        return "Auto mode can use fallback only when it is fully safe; otherwise it downgrades to Assisted."
    return "Assisted mode can present fallback for confirmation."


def _fallback_suggested_action(reason_code: str) -> str:
    actions = {
        "missing_api_key": "Configure provider credentials manually, then re-evaluate.",
        "provider_unsupported": "Choose a supported provider or approve provider setup work separately.",
        "missing_modality_support": "Select a model profile with confirmed modality support.",
        "local_runtime_unavailable": "Install/start the local runtime manually or use a cloud fallback.",
        "local_model_not_installed": "Install/select the local model manually; no download is automatic.",
        "insufficient_local_resources": "Confirm hardware fit manually or choose a cloud fallback.",
        "high_cost_policy": "Confirm the higher-cost route or choose a lower-cost fallback.",
        "high_latency_policy": "Confirm slower route or choose a fast draft fallback.",
        "low_expected_quality": "Escalate to a higher-quality model with confirmation.",
        "high_risk_execution_path": "Require HITL before proceeding.",
        "ready": "Proceed only through the explicit execution contract.",
    }
    return actions[reason_code]


def _policy_score(
    quality: EstimatedQualityBand,
    cost: EstimatedCostBand,
    latency: EstimatedLatencyBand,
    confidence: float,
    risk: RoutingRiskBand,
    reasons: tuple[UnavailableReasonCode, ...],
    blocked: tuple[str, ...],
) -> int:
    quality_points = {"low": 20, "medium": 45, "high": 70, "maximum": 90}[quality]
    cost_points = {"low": 80, "medium": 55, "high": 25}[cost]
    latency_points = {"fast": 70, "moderate": 45, "slow": 15}[latency]
    risk_penalty = {"low": 0, "medium": 30, "high": 70}[risk]
    reason_penalty = min(160, len(reasons) * 24 + len(blocked) * 32)
    return max(
        0,
        min(
            500,
            int(quality_points + cost_points + latency_points + confidence * 100 - risk_penalty - reason_penalty),
        ),
    )


def _token_posture(
    cost: EstimatedCostBand,
    latency: EstimatedLatencyBand,
) -> TokenResourcePosture:
    if cost == "high" or latency == "slow":
        return "high_token_pressure"
    if cost == "medium" or latency == "moderate":
        return "moderate_token_pressure"
    return "low_token_pressure"


def _resource_posture(
    path: tuple[AdaptiveExecutionPathStep, ...],
) -> RuntimeResourcePosture:
    providers = tuple(step.provider_id for step in path)
    if all(provider == "local" for provider in providers):
        return "local_runtime_required"
    if any(provider == "local" for provider in providers):
        return "hybrid_runtime_required"
    if len(set(providers)) > 1:
        return "multi_provider_cloud_required"
    return "cloud_only"


def _path_summary(path: tuple[AdaptiveExecutionPathStep, ...]) -> str:
    return " -> ".join(
        f"{step.provider_id}/{step.model_profile_id}" for step in path
    )


def _suggested_action(
    reasons: tuple[UnavailableReasonCode, ...],
    blocked: tuple[str, ...],
) -> str:
    reason = _fallback_reason(reasons, blocked)
    return _fallback_suggested_action(reason)


def _decision_action(
    mode: ExecutionModeId,
    option: ControlledAdaptiveExecutionOption,
) -> str:
    if mode == "manual_mode":
        return "Ask the user to choose an explicit provider/model path before execution."
    if option.execution_allowed_now:
        return "Execution is allowed by controlled policy; use the explicit execution contract."
    if option.execution_blocked:
        return option.suggested_action
    return "Request user confirmation before applying the recommended path."


def _concrete_strategy_ids(
    direction: HybridRoutingPolicyDirection,
) -> tuple[str, ...]:
    strategies = {
        "local_to_cloud": (
            "local_draft_to_cloud_final",
            "local_exploration_to_cloud_synthesis",
        ),
        "cloud_to_local": (
            "cloud_reasoning_to_local_variants",
            "cloud_primary_to_local_fallback",
        ),
        "cloud_to_cloud": ("cloud_provider_a_to_cloud_provider_b_fallback",),
        "local_to_local": ("local_model_a_to_local_model_b_fallback",),
    }
    return strategies[direction]


def _concrete_strategy_id(direction: HybridRoutingPolicyDirection) -> str:
    return _concrete_strategy_ids(direction)[0]


def _when_applies(direction: HybridRoutingPolicyDirection) -> str:
    mapping = {
        "local_to_cloud": "Use when local drafting/exploration can reduce cost before cloud final quality.",
        "cloud_to_local": "Use when cloud reasoning should seed local variants or local fallback work.",
        "cloud_to_cloud": "Use when multiple cloud providers are candidates for quality, modality, or fallback comparison.",
        "local_to_local": "Use when user-managed local runtimes/models can satisfy privacy or cost constraints.",
    }
    return mapping[direction]


def _provider_requirements(direction: HybridRoutingPolicyDirection) -> tuple[str, ...]:
    mapping = {
        "local_to_cloud": ("confirmed local runtime/model", "configured cloud provider credential"),
        "cloud_to_local": ("configured cloud provider credential", "confirmed local runtime/model"),
        "cloud_to_cloud": ("supported cloud provider adapters", "configured cloud provider credentials"),
        "local_to_local": ("confirmed local runtimes", "confirmed installed local models"),
    }
    return mapping[direction]


def _model_requirements(direction: HybridRoutingPolicyDirection) -> tuple[str, ...]:
    mapping = {
        "local_to_cloud": ("local draft model", "cloud synthesis model"),
        "cloud_to_local": ("cloud reasoning model", "local variant model"),
        "cloud_to_cloud": ("primary cloud model", "fallback cloud model"),
        "local_to_local": ("primary local model", "fallback local model"),
    }
    return mapping[direction]


def _readiness_policy(direction: HybridRoutingPolicyDirection) -> str:
    mapping = {
        "local_to_cloud": "Ready only when local runtime/model and cloud credential are confirmed.",
        "cloud_to_local": "Ready only when cloud credential and local runtime/model are confirmed.",
        "cloud_to_cloud": "Ready only when every cloud provider is supported and credentialed.",
        "local_to_local": "Ready only when local runtime, installed model, and resource fit are confirmed.",
    }
    return mapping[direction]


def _dedupe_reason_codes(
    reasons: tuple[UnavailableReasonCode, ...],
) -> tuple[UnavailableReasonCode, ...]:
    deduped: list[UnavailableReasonCode] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return tuple(deduped)


def _resolve_route(route: RouteName | str) -> RouteName:
    if isinstance(route, RouteName):
        return route
    return RouteName(str(route).strip())
