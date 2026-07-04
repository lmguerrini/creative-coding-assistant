"""V5.4 advisory routing diagnostics metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.hybrid_routing import HybridRoutingPlan, route_hybrid_model_request
from creative_coding_assistant.orchestration.local_cloud_routing import LocalCloudRoutingPlan, route_local_vs_cloud
from creative_coding_assistant.orchestration.model_router import ModelRoutingPlan, route_model_request
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ModelRoutingIntelligenceRegistry,
    ProviderAvailabilityRegistry,
    RoutingProviderProfileRegistry,
    RoutingSafetyContractRegistry,
    TaskAwareRoutingRegistry,
    model_routing_intelligence_registry,
    provider_availability_registry,
    routing_provider_profile_registry,
    routing_safety_contract_registry,
    task_aware_routing_registry,
)

RoutingDiagnosticPanelKind = Literal[
    "provider_profile_coverage",
    "availability_metadata",
    "task_routing_policy",
    "model_route_recommendation",
    "local_cloud_posture",
    "hybrid_route_posture",
    "safety_contracts",
    "intelligence_summary",
]
RoutingDiagnosticStatus = Literal["ready", "guarded"]

ROUTING_DIAGNOSTIC_PANEL_SERIALIZATION_VERSION = "routing_diagnostic_panel.v1"
ROUTING_DIAGNOSTICS_SERIALIZATION_VERSION = "routing_diagnostics.v1"
ROUTING_DIAGNOSTICS_AUTHORITY_BOUNDARY = (
    "The V5.4 Routing Diagnostics surface converts existing V5.2 routing "
    "provider profiles, provider availability metadata, task-aware routing, "
    "model route recommendations, local/cloud comparison, hybrid route "
    "recommendations, safety contracts, and routing intelligence metadata "
    "into read-only routing diagnostics only; it does not apply routes, "
    "switch providers or models, select providers or models, execute "
    "providers, discover or download local models, assume API keys, emit HITL "
    "requests, enforce budgets, control workflows, trigger retries, mutate "
    "prompts, write storage, or modify generated output."
)

_SOURCE_SURFACES = (
    "routing_provider_profile_registry",
    "provider_availability_registry",
    "task_aware_routing_registry",
    "model_routing_plan",
    "local_cloud_routing_plan",
    "hybrid_routing_plan",
    "routing_safety_contract_registry",
    "model_routing_intelligence_registry",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "routing_application",
    "provider_or_model_routing",
    "provider_execution",
    "configured_provider_switching",
    "configured_model_switching",
    "automatic_provider_selection",
    "automatic_model_selection",
    "local_model_discovery",
    "local_model_download",
    "automatic_api_key_assumption",
    "hybrid_routing_application",
    "budget_enforcement",
    "human_input_request_emission",
    "workflow_control",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class RoutingDiagnosticPanel(BaseModel):
    """One read-only V5.4 routing diagnostics panel."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    panel_id: str = Field(min_length=1, max_length=180)
    panel_kind: RoutingDiagnosticPanelKind
    status: RoutingDiagnosticStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=120)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=200)
    routing_signal_count: int = Field(ge=0, le=1000)
    guardrail_signal_count: int = Field(ge=0, le=240)
    applied_route_count: None = None
    provider_call_count: None = None
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    routing_diagnostic_panel_implemented: Literal[True] = True
    routing_application_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    provider_switching_implemented: Literal[False] = False
    model_switching_implemented: Literal[False] = False
    automatic_provider_selection_implemented: Literal[False] = False
    automatic_model_selection_implemented: Literal[False] = False
    local_model_discovery_implemented: Literal[False] = False
    local_model_download_implemented: Literal[False] = False
    automatic_api_key_assumption_implemented: Literal[False] = False
    hybrid_routing_application_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["routing_diagnostic_panel.v1"] = (
        ROUTING_DIAGNOSTIC_PANEL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _panel_matches_contract(self) -> Self:
        if self.panel_id != f"routing_diagnostics::{self.panel_kind}":
            raise ValueError("panel_id must match panel_kind")
        if self.applied_route_count is not None:
            raise ValueError("applied_route_count must remain unset")
        if self.provider_call_count is not None:
            raise ValueError("provider_call_count must remain unset")
        if self.guardrail_signal_count > self.routing_signal_count:
            raise ValueError("guardrail_signal_count must fit routing_signal_count")
        if self.status != _status_for_guardrails(self.guardrail_signal_count):
            raise ValueError("status must match guardrail_signal_count")
        return self


class RoutingDiagnostics(BaseModel):
    """Read-only V5.4 routing diagnostics over V5.2 routing metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["routing_diagnostics"] = "routing_diagnostics"
    serialization_version: Literal["routing_diagnostics.v1"] = (
        ROUTING_DIAGNOSTICS_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ROUTING_DIAGNOSTICS_AUTHORITY_BOUNDARY,
        max_length=1900,
    )
    source_provider_profile_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_provider_availability_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_task_routing_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_model_routing_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_local_cloud_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_hybrid_routing_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_safety_contract_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_intelligence_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_surfaces: tuple[str, ...] = Field(min_length=8, max_length=8)
    route_name: RouteName
    panels: tuple[RoutingDiagnosticPanel, ...] = Field(min_length=1, max_length=8)
    panel_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    ready_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    guarded_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    panel_count: int = Field(ge=1, le=8)
    routing_signal_count: int = Field(ge=0, le=2000)
    guardrail_signal_count: int = Field(ge=0, le=480)
    applied_route_count: None = None
    provider_call_count: None = None
    routing_diagnostics_status: RoutingDiagnosticStatus
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    routing_diagnostics_implemented: Literal[True] = True
    routing_application_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    provider_switching_implemented: Literal[False] = False
    model_switching_implemented: Literal[False] = False
    automatic_provider_selection_implemented: Literal[False] = False
    automatic_model_selection_implemented: Literal[False] = False
    local_model_discovery_implemented: Literal[False] = False
    local_model_download_implemented: Literal[False] = False
    automatic_api_key_assumption_implemented: Literal[False] = False
    hybrid_routing_application_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    hitl_request_emission_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _diagnostics_matches_panels(self) -> Self:
        derived_panel_ids = tuple(panel.panel_id for panel in self.panels)
        if len(set(derived_panel_ids)) != len(derived_panel_ids):
            raise ValueError("panel_ids must be unique")
        if self.panel_ids != derived_panel_ids:
            raise ValueError("panel_ids must match panels")
        if self.panel_count != len(self.panels):
            raise ValueError("panel_count must match panels")
        if self.ready_panel_ids != _panel_ids_for_status(self.panels, "ready"):
            raise ValueError("ready_panel_ids must match panels")
        if self.guarded_panel_ids != _panel_ids_for_status(self.panels, "guarded"):
            raise ValueError("guarded_panel_ids must match panels")
        if self.routing_signal_count != sum(
            panel.routing_signal_count for panel in self.panels
        ):
            raise ValueError("routing_signal_count must match panels")
        if self.guardrail_signal_count != sum(
            panel.guardrail_signal_count for panel in self.panels
        ):
            raise ValueError("guardrail_signal_count must match panels")
        if self.applied_route_count is not None:
            raise ValueError("applied_route_count must remain unset")
        if self.provider_call_count is not None:
            raise ValueError("provider_call_count must remain unset")
        if self.routing_diagnostics_status != _diagnostics_status(self.panels):
            raise ValueError("routing_diagnostics_status must match panels")
        if self.source_surfaces != _SOURCE_SURFACES:
            raise ValueError("source_surfaces must match routing diagnostic sources")
        return self


def build_routing_diagnostics(
    *,
    route: RouteName | str = RouteName.GENERATE,
    provider_profiles: RoutingProviderProfileRegistry | None = None,
    provider_availability: ProviderAvailabilityRegistry | None = None,
    task_routing: TaskAwareRoutingRegistry | None = None,
    model_routing: ModelRoutingPlan | None = None,
    local_cloud_routing: LocalCloudRoutingPlan | None = None,
    hybrid_routing: HybridRoutingPlan | None = None,
    safety_contracts: RoutingSafetyContractRegistry | None = None,
    routing_intelligence: ModelRoutingIntelligenceRegistry | None = None,
) -> RoutingDiagnostics:
    """Build read-only routing diagnostics without applying routes."""

    route_name = RouteName(str(route))
    providers = provider_profiles or routing_provider_profile_registry()
    availability = provider_availability or provider_availability_registry()
    tasks = task_routing or task_aware_routing_registry()
    model = model_routing or route_model_request(route=route_name)
    local_cloud = local_cloud_routing or route_local_vs_cloud(model_routing=model)
    hybrid = hybrid_routing or route_hybrid_model_request(
        local_cloud_routing=local_cloud
    )
    safety = safety_contracts or routing_safety_contract_registry()
    intelligence = routing_intelligence or model_routing_intelligence_registry()
    panels = (
        _provider_profiles_panel(providers),
        _availability_panel(availability),
        _task_routing_panel(tasks),
        _model_routing_panel(model),
        _local_cloud_panel(local_cloud),
        _hybrid_routing_panel(hybrid),
        _safety_contracts_panel(safety),
        _intelligence_panel(intelligence),
    )

    return RoutingDiagnostics(
        source_provider_profile_serialization_version=providers.serialization_version,
        source_provider_availability_serialization_version=(
            availability.serialization_version
        ),
        source_task_routing_serialization_version=tasks.serialization_version,
        source_model_routing_serialization_version=model.serialization_version,
        source_local_cloud_serialization_version=local_cloud.serialization_version,
        source_hybrid_routing_serialization_version=hybrid.serialization_version,
        source_safety_contract_serialization_version=safety.serialization_version,
        source_intelligence_serialization_version=intelligence.serialization_version,
        source_surfaces=_SOURCE_SURFACES,
        route_name=route_name,
        panels=panels,
        panel_ids=tuple(panel.panel_id for panel in panels),
        ready_panel_ids=_panel_ids_for_status(panels, "ready"),
        guarded_panel_ids=_panel_ids_for_status(panels, "guarded"),
        panel_count=len(panels),
        routing_signal_count=sum(panel.routing_signal_count for panel in panels),
        guardrail_signal_count=sum(panel.guardrail_signal_count for panel in panels),
        routing_diagnostics_status=_diagnostics_status(panels),
        advisory_actions=_diagnostics_actions(panels),
    )


def routing_diagnostic_panel_by_id(
    panel_id: str,
    diagnostics: RoutingDiagnostics | None = None,
) -> RoutingDiagnosticPanel | None:
    """Return one routing diagnostic panel without applying route behavior."""

    source_diagnostics = diagnostics or build_routing_diagnostics()
    for panel in source_diagnostics.panels:
        if panel.panel_id == panel_id:
            return panel
    return None


def routing_diagnostic_panels_for_status(
    status: RoutingDiagnosticStatus,
    diagnostics: RoutingDiagnostics | None = None,
) -> tuple[RoutingDiagnosticPanel, ...]:
    """Return routing diagnostic panels by status without route application."""

    source_diagnostics = diagnostics or build_routing_diagnostics()
    return tuple(panel for panel in source_diagnostics.panels if panel.status == status)


def _provider_profiles_panel(
    registry: RoutingProviderProfileRegistry,
) -> RoutingDiagnosticPanel:
    guardrails = len(registry.blocked_runtime_behaviors)
    return RoutingDiagnosticPanel(
        panel_id="routing_diagnostics::provider_profile_coverage",
        panel_kind="provider_profile_coverage",
        status=_status_for_guardrails(guardrails),
        source_id="routing_provider_profile_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=tuple(registry.provider_ids),
        routing_signal_count=(
            registry.provider_count + len(registry.extension_points) + guardrails
        ),
        guardrail_signal_count=guardrails,
        evidence=(
            f"providers:{registry.provider_count}",
            f"extension_points:{len(registry.extension_points)}",
            f"runtime_adapter_required:{registry.runtime_adapter_required_for_execution}",
        ),
        advisory_actions=(
            "Display provider profile coverage without selecting providers.",
            "Keep provider execution, switching, downloads, and API-key assumptions disabled.",
        ),
    )


def _availability_panel(
    registry: ProviderAvailabilityRegistry,
) -> RoutingDiagnosticPanel:
    guardrails = len(registry.unavailable_reason_codes)
    return RoutingDiagnosticPanel(
        panel_id="routing_diagnostics::availability_metadata",
        panel_kind="availability_metadata",
        status=_status_for_guardrails(guardrails),
        source_id="provider_availability_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=tuple(registry.provider_ids)
        + tuple(registry.unavailable_reason_codes),
        routing_signal_count=registry.decision_count,
        guardrail_signal_count=guardrails,
        evidence=(
            f"availability_records:{registry.decision_count}",
            f"providers:{len(registry.provider_ids)}",
            f"unavailable_reasons:{len(registry.unavailable_reason_codes)}",
        ),
        advisory_actions=(
            "Display availability metadata without probing providers or runtimes.",
            "Keep local discovery, model download, provider execution, and switching disabled.",
        ),
    )


def _task_routing_panel(registry: TaskAwareRoutingRegistry) -> RoutingDiagnosticPanel:
    guardrails = registry.hitl_required_decision_count
    return RoutingDiagnosticPanel(
        panel_id="routing_diagnostics::task_routing_policy",
        panel_kind="task_routing_policy",
        status=_status_for_guardrails(guardrails),
        source_id="task_aware_routing_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=registry.decision_ids,
        routing_signal_count=registry.decision_count + len(registry.task_types),
        guardrail_signal_count=guardrails,
        evidence=(
            f"task_routing_decisions:{registry.decision_count}",
            f"task_types:{len(registry.task_types)}",
            f"hitl_required:{registry.hitl_required_decision_count}",
        ),
        advisory_actions=(
            "Display task-aware routing policy without applying routes.",
            "Keep provider/model switching, HITL emission, and execution disabled.",
        ),
    )


def _model_routing_panel(plan: ModelRoutingPlan) -> RoutingDiagnosticPanel:
    guardrails = len(plan.fallback_candidate_ids)
    return RoutingDiagnosticPanel(
        panel_id="routing_diagnostics::model_route_recommendation",
        panel_kind="model_route_recommendation",
        status=_status_for_guardrails(guardrails),
        source_id="model_routing_plan",
        source_serialization_version=plan.serialization_version,
        source_item_ids=plan.candidate_ids,
        routing_signal_count=plan.candidate_count,
        guardrail_signal_count=guardrails,
        evidence=(
            f"route:{plan.route_name.value}",
            f"confidence:{plan.recommendation_confidence}",
            f"fallbacks:{len(plan.fallback_candidate_ids)}",
        ),
        advisory_actions=(
            "Display model-route recommendations without selecting a model.",
            "Keep provider/model routing, cost/quality optimization, and execution disabled.",
        ),
    )


def _local_cloud_panel(plan: LocalCloudRoutingPlan) -> RoutingDiagnosticPanel:
    guardrails = len(plan.fallback_decision_ids)
    return RoutingDiagnosticPanel(
        panel_id="routing_diagnostics::local_cloud_posture",
        panel_kind="local_cloud_posture",
        status=_status_for_guardrails(guardrails),
        source_id="local_cloud_routing_plan",
        source_serialization_version=plan.serialization_version,
        source_item_ids=plan.decision_ids,
        routing_signal_count=plan.decision_count,
        guardrail_signal_count=guardrails,
        evidence=(
            f"posture:{plan.recommended_routing_posture}",
            f"lane:{plan.recommended_routing_lane}",
            f"confidence:{plan.routing_confidence}",
        ),
        advisory_actions=(
            "Display local/cloud posture without calling providers.",
            "Keep discovery, provider execution, hybrid routing, and switching disabled.",
        ),
    )


def _hybrid_routing_panel(plan: HybridRoutingPlan) -> RoutingDiagnosticPanel:
    guardrails = len(plan.fallback_decision_ids)
    return RoutingDiagnosticPanel(
        panel_id="routing_diagnostics::hybrid_route_posture",
        panel_kind="hybrid_route_posture",
        status=_status_for_guardrails(guardrails),
        source_id="hybrid_routing_plan",
        source_serialization_version=plan.serialization_version,
        source_item_ids=plan.decision_ids,
        routing_signal_count=plan.decision_count,
        guardrail_signal_count=guardrails,
        evidence=(
            f"mode:{plan.recommended_hybrid_mode}",
            f"confidence:{plan.routing_confidence}",
            f"fallbacks:{len(plan.fallback_decision_ids)}",
        ),
        advisory_actions=(
            "Display hybrid routing posture without applying hybrid routes.",
            "Keep workflow execution, provider output merging, and routing disabled.",
        ),
    )


def _safety_contracts_panel(
    registry: RoutingSafetyContractRegistry,
) -> RoutingDiagnosticPanel:
    guardrails = len(registry.safety_boundaries)
    return RoutingDiagnosticPanel(
        panel_id="routing_diagnostics::safety_contracts",
        panel_kind="safety_contracts",
        status=_status_for_guardrails(guardrails),
        source_id="routing_safety_contract_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=registry.contract_ids,
        routing_signal_count=registry.decision_count + len(registry.safety_boundaries),
        guardrail_signal_count=guardrails,
        evidence=(
            f"safety_contracts:{registry.decision_count}",
            f"safety_boundaries:{len(registry.safety_boundaries)}",
            f"hitl_request_emitted:{registry.hitl_request_emitted}",
        ),
        advisory_actions=(
            "Display routing safety contracts without emitting HITL requests.",
            "Keep provider switching, model download, provider execution, and output mutation disabled.",
        ),
    )


def _intelligence_panel(
    registry: ModelRoutingIntelligenceRegistry,
) -> RoutingDiagnosticPanel:
    guardrails = len(registry.unavailable_reason_codes)
    return RoutingDiagnosticPanel(
        panel_id="routing_diagnostics::intelligence_summary",
        panel_kind="intelligence_summary",
        status=_status_for_guardrails(guardrails),
        source_id="model_routing_intelligence_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=(
            tuple(registry.provider_ids)
            + tuple(registry.task_types)
            + tuple(registry.execution_mode_ids)
            + tuple(registry.hybrid_policy_directions)
            + tuple(registry.unavailable_reason_codes)
        ),
        routing_signal_count=registry.decision_count,
        guardrail_signal_count=guardrails,
        evidence=(
            f"routing_decisions:{registry.decision_count}",
            f"providers:{len(registry.provider_ids)}",
            f"unavailable_reasons:{len(registry.unavailable_reason_codes)}",
        ),
        advisory_actions=(
            "Display aggregate routing intelligence without applying routing.",
            "Keep provider execution, automatic switching, HITL emission, and storage disabled.",
        ),
    )


def _panel_ids_for_status(
    panels: tuple[RoutingDiagnosticPanel, ...],
    status: RoutingDiagnosticStatus,
) -> tuple[str, ...]:
    return tuple(panel.panel_id for panel in panels if panel.status == status)


def _status_for_guardrails(guardrail_count: int) -> RoutingDiagnosticStatus:
    if guardrail_count:
        return "guarded"
    return "ready"


def _diagnostics_status(
    panels: tuple[RoutingDiagnosticPanel, ...],
) -> RoutingDiagnosticStatus:
    if _panel_ids_for_status(panels, "guarded"):
        return "guarded"
    return "ready"


def _diagnostics_actions(
    panels: tuple[RoutingDiagnosticPanel, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose routing diagnostics panels as read-only observability metadata.",
        "Preserve routing application, provider execution, switching, download, "
        "HITL, workflow, storage, and output boundaries.",
    ]
    if _panel_ids_for_status(panels, "guarded"):
        actions.append(
            "Keep guarded routing diagnostic panels detached from route application."
        )
    return tuple(actions)
