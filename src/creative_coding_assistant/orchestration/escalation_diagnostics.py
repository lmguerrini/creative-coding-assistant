"""V5.4 advisory escalation diagnostics metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .agent_escalation_signals import (
    AgentEscalationSignalRegistry,
    agent_escalation_signal_registry,
)
from .escalation_policy import EscalationPolicyRegistry, escalation_policy_registry
from .escalation_policy_audit import (
    EscalationPolicyAuditRegistry,
    escalation_policy_audit_registry,
)
from .hybrid_agentic_workflow import (
    AdaptiveMultiAgentEscalationRegistry,
    EscalationGateRegistry,
    EscalationTraceRegistry,
    HitlEscalationGateRegistry,
    adaptive_multi_agent_escalation_registry,
    escalation_gate_registry,
    escalation_trace_registry,
    hitl_escalation_gate_registry,
)

EscalationDiagnosticPanelKind = Literal[
    "policy_rules",
    "signal_thresholds",
    "policy_audit",
    "escalation_gates",
    "trace_contexts",
    "hitl_gates",
    "adaptive_boundary",
]
EscalationDiagnosticStatus = Literal["ready", "guarded"]

ESCALATION_DIAGNOSTIC_PANEL_SERIALIZATION_VERSION = (
    "escalation_diagnostic_panel.v1"
)
ESCALATION_DIAGNOSTICS_SERIALIZATION_VERSION = "escalation_diagnostics.v1"
ESCALATION_DIAGNOSTICS_AUTHORITY_BOUNDARY = (
    "The V5.4 Escalation Diagnostics surface converts existing escalation "
    "policy, escalation signal, policy audit, gate, trace, HITL gate, and "
    "adaptive escalation metadata into read-only escalation diagnostics only; "
    "it does not evaluate policies, trigger escalation, execute escalation, "
    "approve gates, request human review, capture or emit traces, orchestrate "
    "agents, invoke agents, select runtimes, route providers or models, "
    "control workflows, trigger retries, write memory or storage, or modify "
    "generated output."
)

_SOURCE_SURFACES = (
    "escalation_policy_registry",
    "agent_escalation_signal_registry",
    "escalation_policy_audit_registry",
    "escalation_gate_registry",
    "escalation_trace_registry",
    "hitl_escalation_gate_registry",
    "adaptive_multi_agent_escalation_registry",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "policy_evaluation",
    "escalation_triggering",
    "escalation_execution",
    "gate_evaluation",
    "escalation_approval",
    "human_review_request",
    "hitl_triggering",
    "trace_capture",
    "trace_emission",
    "multi_agent_orchestration",
    "agent_invocation",
    "runtime_selection",
    "provider_or_model_routing",
    "workflow_control",
    "retry_or_refinement_triggering",
    "memory_write",
    "persistent_storage_write",
    "generated_output_modification",
)


class EscalationDiagnosticPanel(BaseModel):
    """One read-only V5.4 escalation diagnostics panel."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    panel_id: str = Field(min_length=1, max_length=180)
    panel_kind: EscalationDiagnosticPanelKind
    status: EscalationDiagnosticStatus
    source_id: str = Field(min_length=1, max_length=180)
    source_serialization_version: str = Field(min_length=1, max_length=120)
    source_item_ids: tuple[str, ...] = Field(min_length=1, max_length=160)
    escalation_signal_count: int = Field(ge=0, le=1000)
    guardrail_signal_count: int = Field(ge=0, le=240)
    triggered_escalation_count: None = None
    hitl_request_count: None = None
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    escalation_diagnostic_panel_implemented: Literal[True] = True
    policy_evaluation_implemented: Literal[False] = False
    escalation_triggering_implemented: Literal[False] = False
    escalation_execution_implemented: Literal[False] = False
    gate_evaluation_implemented: Literal[False] = False
    escalation_approval_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    hitl_triggering_implemented: Literal[False] = False
    trace_capture_implemented: Literal[False] = False
    trace_emission_implemented: Literal[False] = False
    adaptation_evaluation_implemented: Literal[False] = False
    multi_agent_orchestration_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["escalation_diagnostic_panel.v1"] = (
        ESCALATION_DIAGNOSTIC_PANEL_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _panel_matches_contract(self) -> Self:
        if self.panel_id != f"escalation_diagnostics::{self.panel_kind}":
            raise ValueError("panel_id must match panel_kind")
        if self.triggered_escalation_count is not None:
            raise ValueError("triggered_escalation_count must remain unset")
        if self.hitl_request_count is not None:
            raise ValueError("hitl_request_count must remain unset")
        if self.guardrail_signal_count > self.escalation_signal_count:
            raise ValueError(
                "guardrail_signal_count must fit escalation_signal_count"
            )
        if self.status != _status_for_guardrails(self.guardrail_signal_count):
            raise ValueError("status must match guardrail_signal_count")
        return self


class EscalationDiagnostics(BaseModel):
    """Read-only V5.4 escalation diagnostics over passive escalation metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["escalation_diagnostics"] = "escalation_diagnostics"
    serialization_version: Literal["escalation_diagnostics.v1"] = (
        ESCALATION_DIAGNOSTICS_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ESCALATION_DIAGNOSTICS_AUTHORITY_BOUNDARY,
        max_length=1900,
    )
    source_policy_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_signal_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_policy_audit_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_gate_serialization_version: str = Field(min_length=1, max_length=120)
    source_trace_serialization_version: str = Field(min_length=1, max_length=120)
    source_hitl_gate_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_adaptive_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    source_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    panels: tuple[EscalationDiagnosticPanel, ...] = Field(
        min_length=1,
        max_length=8,
    )
    panel_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    ready_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    guarded_panel_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    panel_count: int = Field(ge=1, le=8)
    escalation_signal_count: int = Field(ge=0, le=2000)
    guardrail_signal_count: int = Field(ge=0, le=480)
    triggered_escalation_count: None = None
    hitl_request_count: None = None
    escalation_diagnostics_status: EscalationDiagnosticStatus
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=20,
    )
    escalation_diagnostics_implemented: Literal[True] = True
    policy_evaluation_implemented: Literal[False] = False
    escalation_triggering_implemented: Literal[False] = False
    escalation_execution_implemented: Literal[False] = False
    gate_evaluation_implemented: Literal[False] = False
    escalation_approval_implemented: Literal[False] = False
    human_review_request_implemented: Literal[False] = False
    hitl_triggering_implemented: Literal[False] = False
    trace_capture_implemented: Literal[False] = False
    trace_emission_implemented: Literal[False] = False
    adaptation_evaluation_implemented: Literal[False] = False
    multi_agent_orchestration_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
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
        if self.escalation_signal_count != sum(
            panel.escalation_signal_count for panel in self.panels
        ):
            raise ValueError("escalation_signal_count must match panels")
        if self.guardrail_signal_count != sum(
            panel.guardrail_signal_count for panel in self.panels
        ):
            raise ValueError("guardrail_signal_count must match panels")
        if self.triggered_escalation_count is not None:
            raise ValueError("triggered_escalation_count must remain unset")
        if self.hitl_request_count is not None:
            raise ValueError("hitl_request_count must remain unset")
        if self.escalation_diagnostics_status != _diagnostics_status(self.panels):
            raise ValueError("escalation_diagnostics_status must match panels")
        if self.source_surfaces != _SOURCE_SURFACES:
            raise ValueError(
                "source_surfaces must match escalation diagnostic sources"
            )
        return self


def build_escalation_diagnostics(
    *,
    policy: EscalationPolicyRegistry | None = None,
    signals: AgentEscalationSignalRegistry | None = None,
    policy_audit: EscalationPolicyAuditRegistry | None = None,
    gates: EscalationGateRegistry | None = None,
    traces: EscalationTraceRegistry | None = None,
    hitl_gates: HitlEscalationGateRegistry | None = None,
    adaptive: AdaptiveMultiAgentEscalationRegistry | None = None,
) -> EscalationDiagnostics:
    """Build read-only escalation diagnostics without triggering escalation."""

    policy_source = policy or escalation_policy_registry()
    signal_source = signals or agent_escalation_signal_registry()
    audit_source = policy_audit or escalation_policy_audit_registry()
    gate_source = gates or escalation_gate_registry()
    trace_source = traces or escalation_trace_registry()
    hitl_source = hitl_gates or hitl_escalation_gate_registry()
    adaptive_source = adaptive or adaptive_multi_agent_escalation_registry()
    panels = (
        _policy_panel(policy_source),
        _signals_panel(signal_source),
        _policy_audit_panel(audit_source),
        _gates_panel(gate_source),
        _traces_panel(trace_source),
        _hitl_gates_panel(hitl_source),
        _adaptive_panel(adaptive_source),
    )

    return EscalationDiagnostics(
        source_policy_serialization_version=policy_source.serialization_version,
        source_signal_serialization_version=signal_source.serialization_version,
        source_policy_audit_serialization_version=audit_source.serialization_version,
        source_gate_serialization_version=gate_source.serialization_version,
        source_trace_serialization_version=trace_source.serialization_version,
        source_hitl_gate_serialization_version=hitl_source.serialization_version,
        source_adaptive_serialization_version=adaptive_source.serialization_version,
        source_surfaces=_SOURCE_SURFACES,
        panels=panels,
        panel_ids=tuple(panel.panel_id for panel in panels),
        ready_panel_ids=_panel_ids_for_status(panels, "ready"),
        guarded_panel_ids=_panel_ids_for_status(panels, "guarded"),
        panel_count=len(panels),
        escalation_signal_count=sum(
            panel.escalation_signal_count for panel in panels
        ),
        guardrail_signal_count=sum(panel.guardrail_signal_count for panel in panels),
        escalation_diagnostics_status=_diagnostics_status(panels),
        advisory_actions=_diagnostics_actions(panels),
    )


def escalation_diagnostic_panel_by_id(
    panel_id: str,
    diagnostics: EscalationDiagnostics | None = None,
) -> EscalationDiagnosticPanel | None:
    """Return one escalation diagnostic panel without runtime behavior."""

    source_diagnostics = diagnostics or build_escalation_diagnostics()
    for panel in source_diagnostics.panels:
        if panel.panel_id == panel_id:
            return panel
    return None


def escalation_diagnostic_panels_for_status(
    status: EscalationDiagnosticStatus,
    diagnostics: EscalationDiagnostics | None = None,
) -> tuple[EscalationDiagnosticPanel, ...]:
    """Return escalation diagnostic panels by status without evaluation."""

    source_diagnostics = diagnostics or build_escalation_diagnostics()
    return tuple(panel for panel in source_diagnostics.panels if panel.status == status)


def _policy_panel(registry: EscalationPolicyRegistry) -> EscalationDiagnosticPanel:
    blocked = tuple(
        dict.fromkeys(
            behavior
            for rule in registry.rules
            for behavior in rule.blocked_runtime_behaviors
        )
    )
    signal_count = registry.rule_count + sum(len(rule.trigger_signals) for rule in registry.rules)
    signal_count += sum(len(rule.evidence_sources) for rule in registry.rules)
    return EscalationDiagnosticPanel(
        panel_id="escalation_diagnostics::policy_rules",
        panel_kind="policy_rules",
        status=_status_for_guardrails(len(blocked)),
        source_id="escalation_policy_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=registry.rule_ids,
        escalation_signal_count=signal_count,
        guardrail_signal_count=len(blocked),
        evidence=(
            f"policy_rules:{registry.rule_count}",
            f"source_registries:{len(registry.source_contract_registries)}",
            f"blocked_behaviors:{len(blocked)}",
        ),
        advisory_actions=(
            "Display escalation policy rules without evaluating policy.",
            "Keep escalation triggering, routing, retries, agents, and artifacts disabled.",
        ),
    )


def _signals_panel(
    registry: AgentEscalationSignalRegistry,
) -> EscalationDiagnosticPanel:
    signal_links = sum(len(signal.policy_rule_ids) for signal in registry.signals)
    guardrails = len(registry.blocked_runtime_behaviors)
    return EscalationDiagnosticPanel(
        panel_id="escalation_diagnostics::signal_thresholds",
        panel_kind="signal_thresholds",
        status=_status_for_guardrails(guardrails),
        source_id="agent_escalation_signal_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=registry.signal_ids,
        escalation_signal_count=len(registry.signal_ids) + len(registry.categories) + signal_links,
        guardrail_signal_count=guardrails,
        evidence=(
            f"signals:{len(registry.signal_ids)}",
            f"categories:{len(registry.categories)}",
            f"policy_links:{signal_links}",
        ),
        advisory_actions=(
            "Display escalation signal thresholds without performing escalation.",
            "Keep automatic HITL triggering, voting, workflow control, and routing disabled.",
        ),
    )


def _policy_audit_panel(
    registry: EscalationPolicyAuditRegistry,
) -> EscalationDiagnosticPanel:
    guardrails = len(registry.passive_boundary_flags)
    return EscalationDiagnosticPanel(
        panel_id="escalation_diagnostics::policy_audit",
        panel_kind="policy_audit",
        status=_status_for_guardrails(guardrails),
        source_id="escalation_policy_audit_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=registry.rule_ids,
        escalation_signal_count=(
            registry.audit_count
            + len(registry.validated_policy_surfaces)
            + len(registry.audited_registry_refs)
        ),
        guardrail_signal_count=guardrails,
        evidence=(
            f"audit_records:{registry.audit_count}",
            f"audited_registries:{len(registry.audited_registry_refs)}",
            f"no_missing_coverage:{registry.no_missing_coverage}",
        ),
        advisory_actions=(
            "Display escalation policy audit coverage without active audit behavior.",
            "Keep policy evaluation, escalation triggering, routing, and memory writes disabled.",
        ),
    )


def _gates_panel(registry: EscalationGateRegistry) -> EscalationDiagnosticPanel:
    guardrails = len(registry.blocked_runtime_behaviors)
    return EscalationDiagnosticPanel(
        panel_id="escalation_diagnostics::escalation_gates",
        panel_kind="escalation_gates",
        status=_status_for_guardrails(guardrails),
        source_id="escalation_gate_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=registry.gate_ids,
        escalation_signal_count=(
            registry.gate_count + len(registry.gate_kinds) + len(registry.condition_ids)
        ),
        guardrail_signal_count=guardrails,
        evidence=(
            f"gates:{registry.gate_count}",
            f"gate_kinds:{len(registry.gate_kinds)}",
            f"conditions:{len(registry.condition_ids)}",
        ),
        advisory_actions=(
            "Display escalation gate metadata without evaluating gates.",
            "Keep gate approval, workflow control, agent invocation, and retries disabled.",
        ),
    )


def _traces_panel(registry: EscalationTraceRegistry) -> EscalationDiagnosticPanel:
    guardrails = len(registry.blocked_runtime_behaviors)
    return EscalationDiagnosticPanel(
        panel_id="escalation_diagnostics::trace_contexts",
        panel_kind="trace_contexts",
        status=_status_for_guardrails(guardrails),
        source_id="escalation_trace_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=registry.trace_profile_ids,
        escalation_signal_count=(
            registry.profile_count
            + len(registry.escalation_signal_ids)
            + len(registry.gate_ids)
        ),
        guardrail_signal_count=guardrails,
        evidence=(
            f"trace_profiles:{registry.profile_count}",
            f"signals:{len(registry.escalation_signal_ids)}",
            f"gates:{len(registry.gate_ids)}",
        ),
        advisory_actions=(
            "Display escalation trace context without capturing or emitting traces.",
            "Keep trace capture, escalation execution, memory writes, and workflow control disabled.",
        ),
    )


def _hitl_gates_panel(
    registry: HitlEscalationGateRegistry,
) -> EscalationDiagnosticPanel:
    guardrails = len(registry.blocked_runtime_behaviors)
    return EscalationDiagnosticPanel(
        panel_id="escalation_diagnostics::hitl_gates",
        panel_kind="hitl_gates",
        status=_status_for_guardrails(guardrails),
        source_id="hitl_escalation_gate_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=registry.hitl_gate_profile_ids,
        escalation_signal_count=(
            registry.profile_count
            + len(registry.hitl_postures)
            + len(registry.escalation_signal_ids)
        ),
        guardrail_signal_count=guardrails,
        evidence=(
            f"hitl_gate_profiles:{registry.profile_count}",
            f"hitl_postures:{len(registry.hitl_postures)}",
            f"signals:{len(registry.escalation_signal_ids)}",
        ),
        advisory_actions=(
            "Display HITL gate posture without triggering human review.",
            "Keep human review requests, gate approval, agent invocation, and workflow control disabled.",
        ),
    )


def _adaptive_panel(
    registry: AdaptiveMultiAgentEscalationRegistry,
) -> EscalationDiagnosticPanel:
    guardrails = len(registry.blocked_runtime_behaviors)
    return EscalationDiagnosticPanel(
        panel_id="escalation_diagnostics::adaptive_boundary",
        panel_kind="adaptive_boundary",
        status=_status_for_guardrails(guardrails),
        source_id="adaptive_multi_agent_escalation_registry",
        source_serialization_version=registry.serialization_version,
        source_item_ids=registry.adaptive_profile_ids,
        escalation_signal_count=(
            registry.profile_count
            + len(registry.adaptive_postures)
            + len(registry.escalation_signal_ids)
            + len(registry.adaptive_evidence_surfaces)
        ),
        guardrail_signal_count=guardrails,
        evidence=(
            f"adaptive_profiles:{registry.profile_count}",
            f"adaptive_postures:{len(registry.adaptive_postures)}",
            f"signals:{len(registry.escalation_signal_ids)}",
        ),
        advisory_actions=(
            "Display adaptive escalation readiness without orchestration.",
            "Keep adaptation evaluation, escalation execution, multi-agent orchestration, and routing disabled.",
        ),
    )


def _panel_ids_for_status(
    panels: tuple[EscalationDiagnosticPanel, ...],
    status: EscalationDiagnosticStatus,
) -> tuple[str, ...]:
    return tuple(panel.panel_id for panel in panels if panel.status == status)


def _status_for_guardrails(guardrail_count: int) -> EscalationDiagnosticStatus:
    if guardrail_count:
        return "guarded"
    return "ready"


def _diagnostics_status(
    panels: tuple[EscalationDiagnosticPanel, ...],
) -> EscalationDiagnosticStatus:
    if _panel_ids_for_status(panels, "guarded"):
        return "guarded"
    return "ready"


def _diagnostics_actions(
    panels: tuple[EscalationDiagnosticPanel, ...],
) -> tuple[str, ...]:
    actions = [
        "Expose escalation diagnostics panels as read-only observability metadata.",
        "Preserve policy evaluation, escalation triggering, HITL, trace, agent, "
        "routing, workflow, memory, storage, and output boundaries.",
    ]
    if _panel_ids_for_status(panels, "guarded"):
        actions.append(
            "Keep guarded escalation diagnostic panels detached from runtime escalation."
        )
    return tuple(actions)
