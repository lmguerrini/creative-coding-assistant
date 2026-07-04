"""V5.6 production release final optimization metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_execution_policy_engine import (
    AdaptiveExecutionAvailabilityContext,
    ControlledAdaptiveExecutionPlan,
    adaptive_execution_availability_context,
    evaluate_adaptive_execution_policy,
)
from creative_coding_assistant.orchestration.execution_path_optimization import (
    ExecutionPathOptimizationPlan,
    plan_execution_path_optimization,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    ModelRoutingIntelligenceRegistry,
    ProviderAvailabilityRegistry,
    ProviderId,
    TaskRoutingType,
    model_routing_intelligence_registry,
    provider_availability_registry,
)
from creative_coding_assistant.orchestration.system_health_monitoring import (
    SystemHealthMonitoring,
    build_system_health_monitoring,
)
from creative_coding_assistant.orchestration.workflow_explainability_dashboard import (
    WorkflowExplainabilityDashboard,
    build_workflow_explainability_dashboard,
)

ProductionOptimizationDomain = Literal[
    "provider_configuration_review",
    "execution_safety_review",
    "decision_explainability_review",
    "failure_determinism_review",
    "demo_workflow_readiness",
]
ProductionOptimizationStatus = Literal["ready", "guarded"]

PRODUCTION_OPTIMIZATION_RECORD_SERIALIZATION_VERSION = (
    "production_release_optimization_record.v1"
)
PRODUCTION_FINAL_OPTIMIZATION_PLAN_SERIALIZATION_VERSION = (
    "production_release_final_optimization_plan.v1"
)
PRODUCTION_FINAL_OPTIMIZATION_AUTHORITY_BOUNDARY = (
    "V5.6 Production Release final optimization metadata reviews the "
    "existing V5 execution, routing, observability, explainability, safety, "
    "and demo-readiness surfaces for release posture only; it does not "
    "introduce core architecture, mutate provider or model routing, execute "
    "providers, probe local runtimes, download models, provision providers, "
    "install runtimes, infer API keys, emit HITL requests, execute workflows, "
    "write storage, modify generated output, merge, push, tag, or apply "
    "Runtime Evolution."
)

_REQUIRED_DOMAINS: tuple[ProductionOptimizationDomain, ...] = (
    "provider_configuration_review",
    "execution_safety_review",
    "decision_explainability_review",
    "failure_determinism_review",
    "demo_workflow_readiness",
)
_REQUIRED_EXPLANATION_FIELDS = (
    "selected_provider",
    "selected_model",
    "execution_mode",
    "execution_strategy",
    "quality_estimate",
    "cost_estimate",
    "latency_estimate",
    "fallback_strategy",
    "escalation_reason",
)
_DEMO_WORKFLOW_STEPS = (
    "task",
    "routing_intelligence",
    "adaptive_execution_policy",
    "execution_simulation",
    "generation",
    "artifact",
    "explanation",
    "final_output",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "core_architecture_expansion",
    "provider_configuration_mutation",
    "environment_variable_mutation",
    "api_key_assumption",
    "provider_or_model_routing_mutation",
    "provider_execution",
    "automatic_provider_switching",
    "automatic_model_switching",
    "automatic_model_download",
    "provider_provisioning",
    "local_runtime_probe",
    "local_model_inventory_scan",
    "runtime_auto_installation",
    "hitl_request_emission",
    "workflow_execution",
    "workflow_control",
    "workflow_graph_mutation",
    "telemetry_emission",
    "alert_emission",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
    "merge_push_tag_operation",
)


class ProductionOptimizationRecord(BaseModel):
    """One V5.6 production optimization review record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    record_id: str = Field(min_length=1, max_length=180)
    domain: ProductionOptimizationDomain
    status: ProductionOptimizationStatus
    readiness_score: int = Field(ge=0, le=100)
    source_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_serialization_versions: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    guarded_reason_codes: tuple[str, ...] = Field(default_factory=tuple, max_length=24)
    release_actions: tuple[str, ...] = Field(min_length=1, max_length=8)
    release_blocker: bool
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    production_optimization_record_implemented: Literal[True] = True
    provider_configuration_mutation_implemented: Literal[False] = False
    environment_configuration_mutation_implemented: Literal[False] = False
    api_key_assumption_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    runtime_installation_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    merge_push_tag_implemented: Literal[False] = False
    serialization_version: Literal["production_release_optimization_record.v1"] = (
        PRODUCTION_OPTIMIZATION_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _record_is_consistent(self) -> Self:
        if self.record_id != f"production_optimization::{self.domain}":
            raise ValueError("record_id must match domain")
        if len(self.source_surface_ids) != len(self.source_serialization_versions):
            raise ValueError("source ids and serialization versions must align")
        if len(set(self.source_surface_ids)) != len(self.source_surface_ids):
            raise ValueError("source_surface_ids must be unique")
        if self.status != _status_for(self.guarded_reason_codes, self.readiness_score):
            raise ValueError("status must match guarded reasons and readiness score")
        if self.release_blocker and self.status != "guarded":
            raise ValueError("release blockers must be guarded")
        return self


class ProductionReleaseFinalOptimizationPlan(BaseModel):
    """Final V5.6 optimization posture over existing V5 metadata surfaces."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["production_release_final_optimization"] = (
        "production_release_final_optimization"
    )
    serialization_version: Literal["production_release_final_optimization_plan.v1"] = (
        PRODUCTION_FINAL_OPTIMIZATION_PLAN_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PRODUCTION_FINAL_OPTIMIZATION_AUTHORITY_BOUNDARY,
        max_length=2400,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    requested_execution_mode_id: ExecutionModeId
    selected_execution_mode_id: ExecutionModeId
    source_execution_path_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_routing_intelligence_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_provider_availability_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_system_health_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_workflow_explainability_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_adaptive_execution_policy_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    provider_ids: tuple[ProviderId, ...] = Field(min_length=4, max_length=4)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    explanation_fields: tuple[str, ...] = Field(min_length=9, max_length=9)
    demo_workflow_steps: tuple[str, ...] = Field(min_length=8, max_length=8)
    records: tuple[ProductionOptimizationRecord, ...] = Field(
        min_length=5,
        max_length=5,
    )
    record_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    domain_ids: tuple[ProductionOptimizationDomain, ...] = Field(
        min_length=5,
        max_length=5,
    )
    ready_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    guarded_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    release_blocker_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    record_count: int = Field(ge=5, le=5)
    guarded_record_count: int = Field(ge=0, le=5)
    release_blocker_count: int = Field(ge=0, le=5)
    production_optimization_status: ProductionOptimizationStatus
    selected_decision_id: str = Field(min_length=1, max_length=180)
    selected_option_id: str | None = Field(default=None, max_length=180)
    recommended_option_id: str = Field(min_length=1, max_length=180)
    fallback_option_id: str | None = Field(default=None, max_length=180)
    required_hitl_gates: tuple[str, ...] = Field(default_factory=tuple, max_length=24)
    unavailable_reason_codes: tuple[str, ...] = Field(
        default_factory=tuple, max_length=24
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=28,
    )
    final_optimization_implemented: Literal[True] = True
    production_configuration_review_implemented: Literal[True] = True
    production_safety_review_implemented: Literal[True] = True
    production_explainability_review_implemented: Literal[True] = True
    production_failure_review_implemented: Literal[True] = True
    demo_readiness_review_implemented: Literal[True] = True
    provider_configuration_mutation_implemented: Literal[False] = False
    environment_configuration_mutation_implemented: Literal[False] = False
    api_key_assumption_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    provider_provisioning_implemented: Literal[False] = False
    runtime_installation_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    workflow_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    telemetry_emission_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    merge_push_tag_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _plan_matches_records(self) -> Self:
        derived_record_ids = tuple(record.record_id for record in self.records)
        derived_domain_ids = tuple(record.domain for record in self.records)
        if self.record_ids != derived_record_ids:
            raise ValueError("record_ids must match records")
        if self.domain_ids != derived_domain_ids:
            raise ValueError("domain_ids must match records")
        if self.domain_ids != _REQUIRED_DOMAINS:
            raise ValueError("domain_ids must cover required production domains")
        if self.ready_record_ids != _record_ids_for_status(self.records, "ready"):
            raise ValueError("ready_record_ids must match records")
        if self.guarded_record_ids != _record_ids_for_status(self.records, "guarded"):
            raise ValueError("guarded_record_ids must match records")
        if self.release_blocker_ids != tuple(
            record.record_id for record in self.records if record.release_blocker
        ):
            raise ValueError("release_blocker_ids must match records")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.guarded_record_count != len(self.guarded_record_ids):
            raise ValueError("guarded_record_count must match records")
        if self.release_blocker_count != len(self.release_blocker_ids):
            raise ValueError("release_blocker_count must match records")
        if self.production_optimization_status != _plan_status(self.records):
            raise ValueError("production_optimization_status must match records")
        if self.explanation_fields != _REQUIRED_EXPLANATION_FIELDS:
            raise ValueError("explanation_fields must cover production UX")
        if self.demo_workflow_steps != _DEMO_WORKFLOW_STEPS:
            raise ValueError("demo_workflow_steps must cover end-to-end demo")
        return self


def build_production_release_final_optimization(
    *,
    task_type: TaskRoutingType | str = "creative_coding",
    route: RouteName | str = RouteName.GENERATE,
    execution_mode_id: ExecutionModeId | str = "assisted_mode",
    availability_context: AdaptiveExecutionAvailabilityContext | None = None,
    execution_path: ExecutionPathOptimizationPlan | None = None,
    routing_intelligence: ModelRoutingIntelligenceRegistry | None = None,
    provider_availability: ProviderAvailabilityRegistry | None = None,
    system_health: SystemHealthMonitoring | None = None,
    workflow_explainability: WorkflowExplainabilityDashboard | None = None,
    adaptive_execution_policy: ControlledAdaptiveExecutionPlan | None = None,
) -> ProductionReleaseFinalOptimizationPlan:
    """Build final production optimization metadata without executing work."""

    path_source = execution_path or plan_execution_path_optimization()
    routing_source = routing_intelligence or model_routing_intelligence_registry()
    availability_source = provider_availability or provider_availability_registry()
    health_source = system_health or build_system_health_monitoring()
    explainability_source = (
        workflow_explainability or build_workflow_explainability_dashboard()
    )
    context = availability_context or adaptive_execution_availability_context()
    policy_source = adaptive_execution_policy or evaluate_adaptive_execution_policy(
        task_type=task_type,
        route=route,
        execution_mode_id=execution_mode_id,
        availability_context=context,
    )
    decision = policy_source.selected_decision
    guarded_reasons = _unique(
        (*decision.unavailable_reason_codes, *decision.required_hitl_gates)
    )
    records = _records(
        path_source=path_source,
        routing_source=routing_source,
        availability_source=availability_source,
        health_source=health_source,
        explainability_source=explainability_source,
        policy_source=policy_source,
        guarded_reasons=guarded_reasons,
    )
    return ProductionReleaseFinalOptimizationPlan(
        route_name=policy_source.route_name,
        task_type=policy_source.task_type,
        requested_execution_mode_id=policy_source.requested_execution_mode_id,
        selected_execution_mode_id=policy_source.selected_execution_mode_id,
        source_execution_path_serialization_version=path_source.serialization_version,
        source_routing_intelligence_serialization_version=(
            routing_source.serialization_version
        ),
        source_provider_availability_serialization_version=(
            availability_source.serialization_version
        ),
        source_system_health_serialization_version=health_source.serialization_version,
        source_workflow_explainability_serialization_version=(
            explainability_source.serialization_version
        ),
        source_adaptive_execution_policy_serialization_version=(
            policy_source.serialization_version
        ),
        provider_ids=routing_source.provider_ids,
        execution_mode_ids=routing_source.execution_mode_ids,
        explanation_fields=_REQUIRED_EXPLANATION_FIELDS,
        demo_workflow_steps=_DEMO_WORKFLOW_STEPS,
        records=records,
        record_ids=tuple(record.record_id for record in records),
        domain_ids=tuple(record.domain for record in records),
        ready_record_ids=_record_ids_for_status(records, "ready"),
        guarded_record_ids=_record_ids_for_status(records, "guarded"),
        release_blocker_ids=tuple(
            record.record_id for record in records if record.release_blocker
        ),
        record_count=len(records),
        guarded_record_count=len(_record_ids_for_status(records, "guarded")),
        release_blocker_count=sum(1 for record in records if record.release_blocker),
        production_optimization_status=_plan_status(records),
        selected_decision_id=decision.decision_id,
        selected_option_id=decision.selected_option_id,
        recommended_option_id=decision.recommended_option_id,
        fallback_option_id=decision.fallback_option_id,
        required_hitl_gates=decision.required_hitl_gates,
        unavailable_reason_codes=decision.unavailable_reason_codes,
    )


def production_optimization_record_by_domain(
    domain: ProductionOptimizationDomain | str,
    plan: ProductionReleaseFinalOptimizationPlan | None = None,
) -> ProductionOptimizationRecord | None:
    """Return one production optimization record by domain."""

    normalized = str(domain).strip()
    source_plan = plan or build_production_release_final_optimization()
    for record in source_plan.records:
        if record.domain == normalized:
            return record
    return None


def production_optimization_records_for_status(
    status: ProductionOptimizationStatus,
    plan: ProductionReleaseFinalOptimizationPlan | None = None,
) -> tuple[ProductionOptimizationRecord, ...]:
    """Return production optimization records by status."""

    source_plan = plan or build_production_release_final_optimization()
    return tuple(record for record in source_plan.records if record.status == status)


def _records(
    *,
    path_source: ExecutionPathOptimizationPlan,
    routing_source: ModelRoutingIntelligenceRegistry,
    availability_source: ProviderAvailabilityRegistry,
    health_source: SystemHealthMonitoring,
    explainability_source: WorkflowExplainabilityDashboard,
    policy_source: ControlledAdaptiveExecutionPlan,
    guarded_reasons: tuple[str, ...],
) -> tuple[ProductionOptimizationRecord, ...]:
    decision = policy_source.selected_decision
    configuration_reasons = _unique(
        reason
        for reason in guarded_reasons
        if reason
        in {
            "missing_api_key",
            "local_runtime_unavailable",
            "local_model_not_installed",
            "hitl_before_provider_provisioning",
            "hitl_before_local_model_download",
            "hitl_before_runtime_installation",
        }
    )
    safety_reasons = _unique(
        reason for reason in guarded_reasons if reason not in configuration_reasons
    )
    demo_reasons = _unique(
        (*decision.required_hitl_gates, *decision.unavailable_reason_codes)
        if decision.execution_requires_user_confirmation
        else ()
    )
    return (
        _record(
            domain="provider_configuration_review",
            score=82 if configuration_reasons else 96,
            source_surface_ids=(
                routing_source.role,
                availability_source.role,
                policy_source.role,
            ),
            source_versions=(
                routing_source.serialization_version,
                availability_source.serialization_version,
                policy_source.serialization_version,
            ),
            evidence=(
                f"providers:{','.join(routing_source.provider_ids)}",
                f"unavailable_reasons:{len(availability_source.unavailable_reason_codes)}",
                "configuration_fail_safe_boundary:present",
            ),
            guarded_reason_codes=configuration_reasons,
            release_actions=(
                "Keep provider and API key requirements explicit.",
                "Fail closed when required provider configuration is missing.",
                "Surface missing local runtime and local model diagnostics.",
            ),
            release_blocker=False,
        ),
        _record(
            domain="execution_safety_review",
            score=86 if safety_reasons else 94,
            source_surface_ids=(routing_source.role, policy_source.role),
            source_versions=(
                routing_source.serialization_version,
                policy_source.serialization_version,
            ),
            evidence=(
                f"execution_modes:{','.join(routing_source.execution_mode_ids)}",
                f"manual_actions:{policy_source.manual_action_count}",
                f"escalation_rules:{policy_source.escalation_rule_count}",
            ),
            guarded_reason_codes=safety_reasons,
            release_actions=(
                "Preserve Manual, Assisted, and Auto mode semantics.",
                "Require HITL before unavailable, costly, or high-risk execution.",
                "Keep downloads, provisioning, and runtime installs manual.",
            ),
            release_blocker=False,
        ),
        _record(
            domain="decision_explainability_review",
            score=94,
            source_surface_ids=(explainability_source.role, policy_source.role),
            source_versions=(
                explainability_source.serialization_version,
                policy_source.serialization_version,
            ),
            evidence=(
                f"explainability_panels:{explainability_source.panel_count}",
                f"explanation_fields:{len(_REQUIRED_EXPLANATION_FIELDS)}",
                f"selected_decision:{decision.decision_id}",
            ),
            guarded_reason_codes=(),
            release_actions=(
                "Expose provider, model, mode, strategy, estimates, fallback, and escalation reason.",
                "Keep explanation generation separate from this metadata review.",
            ),
            release_blocker=False,
        ),
        _record(
            domain="failure_determinism_review",
            score=92,
            source_surface_ids=(
                health_source.role,
                availability_source.role,
                policy_source.role,
            ),
            source_versions=(
                health_source.serialization_version,
                availability_source.serialization_version,
                policy_source.serialization_version,
            ),
            evidence=(
                f"system_health_panels:{health_source.panel_count}",
                f"guarded_health_panels:{len(health_source.guarded_panel_ids)}",
                "provider_unavailable_paths:deterministic_metadata",
            ),
            guarded_reason_codes=(),
            release_actions=(
                "Keep provider, API key, runtime, model, capability, fallback, and HITL failures deterministic.",
                "Do not add automated remediation during production release hardening.",
            ),
            release_blocker=False,
        ),
        _record(
            domain="demo_workflow_readiness",
            score=84 if demo_reasons else 93,
            source_surface_ids=(
                path_source.role,
                explainability_source.role,
                policy_source.role,
            ),
            source_versions=(
                path_source.serialization_version,
                explainability_source.serialization_version,
                policy_source.serialization_version,
            ),
            evidence=(
                f"demo_steps:{len(_DEMO_WORKFLOW_STEPS)}",
                f"path_candidates:{path_source.candidate_count}",
                f"policy_simulations:{policy_source.simulation_count}",
            ),
            guarded_reason_codes=demo_reasons,
            release_actions=(
                "Demonstrate task to routing to policy to simulation to generation to artifact to explanation.",
                "Keep generation execution outside this optimization metadata surface.",
            ),
            release_blocker=False,
        ),
    )


def _record(
    *,
    domain: ProductionOptimizationDomain,
    score: int,
    source_surface_ids: tuple[str, ...],
    source_versions: tuple[str, ...],
    evidence: tuple[str, ...],
    guarded_reason_codes: tuple[str, ...],
    release_actions: tuple[str, ...],
    release_blocker: bool,
) -> ProductionOptimizationRecord:
    return ProductionOptimizationRecord(
        record_id=f"production_optimization::{domain}",
        domain=domain,
        status=_status_for(guarded_reason_codes, score),
        readiness_score=score,
        source_surface_ids=source_surface_ids,
        source_serialization_versions=source_versions,
        evidence=evidence,
        guarded_reason_codes=guarded_reason_codes,
        release_actions=release_actions,
        release_blocker=release_blocker,
    )


def _status_for(
    guarded_reason_codes: tuple[str, ...],
    readiness_score: int,
) -> ProductionOptimizationStatus:
    if guarded_reason_codes or readiness_score < 90:
        return "guarded"
    return "ready"


def _plan_status(
    records: tuple[ProductionOptimizationRecord, ...],
) -> ProductionOptimizationStatus:
    if any(record.status == "guarded" for record in records):
        return "guarded"
    return "ready"


def _record_ids_for_status(
    records: tuple[ProductionOptimizationRecord, ...],
    status: ProductionOptimizationStatus,
) -> tuple[str, ...]:
    return tuple(record.record_id for record in records if record.status == status)


def _unique(values: tuple[str, ...] | object) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return tuple(result)
