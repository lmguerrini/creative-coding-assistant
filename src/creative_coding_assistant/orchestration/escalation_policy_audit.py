"""Passive V4.6 escalation policy audit metadata."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_escalation_signals import (
    agent_escalation_signal_registry,
)
from creative_coding_assistant.orchestration.escalation_policy import (
    ESCALATION_POLICY_REGISTRY,
    ESCALATION_POLICY_RULES,
    EscalationPolicyRule,
)
from creative_coding_assistant.orchestration.hybrid_agentic_workflow import (
    adaptive_multi_agent_escalation_registry,
    ambiguity_escalation_registry,
    conditional_multi_agent_escalation_registry,
    creative_escalation_policy_registry,
    escalation_gate_registry,
    escalation_trace_registry,
    hitl_escalation_gate_registry,
    hybrid_agentic_workflow_registry,
    quality_escalation_registry,
    reflection_escalation_registry,
    risk_escalation_registry,
)

EscalationPolicyAuditStage = Literal["v4_6_escalation_policy_hardening"]
EscalationPolicyAuditStatus = Literal["pass"]

ESCALATION_POLICY_AUDIT_SERIALIZATION_VERSION = "escalation_policy_audit.v1"
ESCALATION_POLICY_AUDIT_REGISTRY_SERIALIZATION_VERSION = (
    "escalation_policy_audit_registry.v1"
)
ESCALATION_POLICY_AUDIT_REGISTRY_AUTHORITY_BOUNDARY = (
    "V4.6 escalation policy audit metadata checks passive escalation policy "
    "rules, source contract registry coverage, trigger and evidence surfaces, "
    "downstream advisory registry references, and blocked runtime behaviors "
    "only; it does not evaluate policy, trigger escalation, route providers "
    "or models, select runtimes, invoke agents, retry work, write memory, or "
    "modify generated output."
)

_AUDITED_REGISTRY_REFS = (
    "escalation_policy_registry",
    "agent_escalation_signal_registry",
    "conditional_multi_agent_escalation_registry",
    "escalation_gate_registry",
    "creative_escalation_policy_registry",
    "reflection_escalation_registry",
    "escalation_trace_registry",
    "hitl_escalation_gate_registry",
    "ambiguity_escalation_registry",
    "risk_escalation_registry",
    "quality_escalation_registry",
    "adaptive_multi_agent_escalation_registry",
    "hybrid_agentic_workflow_registry",
)
_VALIDATED_POLICY_SURFACES = (
    "authority_boundary",
    "policy_stage",
    "source_contract_registries",
    "trigger_signals",
    "evidence_sources",
    "advisory_outcome",
    "blocked_runtime_behaviors",
)
_PASSIVE_BOUNDARY_FLAGS = (
    "policy_evaluation_blocked",
    "escalation_triggering_blocked",
    "agent_invocation_blocked",
    "provider_model_routing_blocked",
    "runtime_selection_blocked",
    "retry_triggering_blocked",
    "generated_output_mutation_blocked",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "policy_evaluation",
    "escalation_triggering",
    "agent_invocation",
    "provider_or_model_routing",
    "runtime_selection",
    "workflow_control",
    "retry_or_refinement_triggering",
    "memory_write",
    "generated_output_modification",
)
_AUDIT_FINDINGS = (
    "policy_rule_metadata_confirmed",
    "source_contract_registry_refs_confirmed",
    "trigger_signal_surface_confirmed",
    "evidence_source_surface_confirmed",
    "downstream_registry_reference_confirmed",
    "runtime_behavior_blocks_confirmed",
)


class EscalationPolicyAuditRecord(BaseModel):
    """One passive V4.6 audit record for an escalation policy rule."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    rule_id: str = Field(min_length=1, max_length=80)
    policy_stage: str = Field(min_length=1, max_length=80)
    rule_serialization_version: str = Field(min_length=1, max_length=80)
    audit_stage: EscalationPolicyAuditStage = "v4_6_escalation_policy_hardening"
    audited_registry_refs: tuple[str, ...] = Field(min_length=13, max_length=13)
    rule_source_contract_registries: tuple[str, ...] = Field(
        min_length=1,
        max_length=6,
    )
    downstream_registry_refs: tuple[str, ...] = Field(min_length=1, max_length=13)
    trigger_signals: tuple[str, ...] = Field(min_length=1, max_length=12)
    evidence_sources: tuple[str, ...] = Field(min_length=1, max_length=12)
    validated_policy_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=7, max_length=7)
    audit_findings: tuple[str, ...] = Field(min_length=6, max_length=6)
    missing_coverage_items: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    rule_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    audit_status: EscalationPolicyAuditStatus = "pass"
    metadata_only_declared: Literal[True] = True
    policy_evaluation_implemented: Literal[False] = False
    escalation_triggering_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["escalation_policy_audit.v1"] = (
        ESCALATION_POLICY_AUDIT_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class EscalationPolicyAuditRegistry(BaseModel):
    """Stable passive V4.6 audit registry for escalation policy rules."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["escalation_policy_audit_registry"] = (
        "escalation_policy_audit_registry"
    )
    serialization_version: Literal["escalation_policy_audit_registry.v1"] = (
        ESCALATION_POLICY_AUDIT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ESCALATION_POLICY_AUDIT_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    audit_stage: EscalationPolicyAuditStage = "v4_6_escalation_policy_hardening"
    audit_records: tuple[EscalationPolicyAuditRecord, ...] = Field(
        min_length=5,
        max_length=5,
    )
    rule_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    audit_count: int = Field(ge=5, le=5)
    source_policy_registry: Literal["escalation_policy_registry"] = (
        "escalation_policy_registry"
    )
    audited_registry_refs: tuple[str, ...] = Field(min_length=13, max_length=13)
    source_contract_registries: tuple[str, ...] = Field(min_length=4, max_length=4)
    validated_policy_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    all_policy_rules_covered: Literal[True] = True
    no_missing_coverage: Literal[True] = True
    active_runtime_audit_implemented: Literal[False] = False
    policy_evaluation_implemented: Literal[False] = False
    escalation_triggering_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_audit_records(self) -> Self:
        derived_rule_ids = tuple(record.rule_id for record in self.audit_records)
        if len(set(derived_rule_ids)) != len(derived_rule_ids):
            raise ValueError("rule_ids must be unique")
        if self.rule_ids != derived_rule_ids:
            raise ValueError("rule_ids must match audit records")
        if self.audit_count != len(self.audit_records):
            raise ValueError("audit_count must match audit records")
        for record in self.audit_records:
            if record.audited_registry_refs != self.audited_registry_refs:
                raise ValueError("audited_registry_refs must match registry")
            if record.validated_policy_surfaces != self.validated_policy_surfaces:
                raise ValueError("validated_policy_surfaces must match registry")
            if record.passive_boundary_flags != self.passive_boundary_flags:
                raise ValueError("passive_boundary_flags must match registry")
            if not set(record.rule_source_contract_registries).issubset(
                self.source_contract_registries
            ):
                raise ValueError("rule source registries must be known")
            if record.missing_coverage_items:
                raise ValueError("audit records must not contain missing coverage")
        return self


def escalation_policy_audit_registry() -> EscalationPolicyAuditRegistry:
    """Return passive V4.6 escalation policy audit metadata."""

    return ESCALATION_POLICY_AUDIT_REGISTRY


def escalation_policy_audit_by_rule_id(
    rule_id: str,
    registry: EscalationPolicyAuditRegistry | None = None,
) -> EscalationPolicyAuditRecord | None:
    """Return one passive escalation policy audit record by rule id."""

    source_registry = registry or ESCALATION_POLICY_AUDIT_REGISTRY
    for record in source_registry.audit_records:
        if record.rule_id == rule_id:
            return record
    return None


def escalation_policy_audits_for_downstream_registry(
    downstream_registry_ref: str,
    registry: EscalationPolicyAuditRegistry | None = None,
) -> tuple[EscalationPolicyAuditRecord, ...]:
    """Return passive policy audit records linked to one downstream registry."""

    source_registry = registry or ESCALATION_POLICY_AUDIT_REGISTRY
    normalized_ref = str(downstream_registry_ref).strip()
    return tuple(
        record
        for record in source_registry.audit_records
        if normalized_ref in record.downstream_registry_refs
    )


def _contains_value(value: Any, needle: str) -> bool:
    if isinstance(value, str):
        return value == needle
    if isinstance(value, dict):
        return any(_contains_value(item, needle) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_value(item, needle) for item in value)
    return False


def _downstream_registry_refs(rule_id: str) -> tuple[str, ...]:
    registries = (
        ("agent_escalation_signal_registry", agent_escalation_signal_registry()),
        (
            "conditional_multi_agent_escalation_registry",
            conditional_multi_agent_escalation_registry(),
        ),
        ("escalation_gate_registry", escalation_gate_registry()),
        ("creative_escalation_policy_registry", creative_escalation_policy_registry()),
        ("reflection_escalation_registry", reflection_escalation_registry()),
        ("escalation_trace_registry", escalation_trace_registry()),
        ("hitl_escalation_gate_registry", hitl_escalation_gate_registry()),
        ("ambiguity_escalation_registry", ambiguity_escalation_registry()),
        ("risk_escalation_registry", risk_escalation_registry()),
        ("quality_escalation_registry", quality_escalation_registry()),
        (
            "adaptive_multi_agent_escalation_registry",
            adaptive_multi_agent_escalation_registry(),
        ),
        ("hybrid_agentic_workflow_registry", hybrid_agentic_workflow_registry()),
    )
    return tuple(
        registry_ref
        for registry_ref, registry in registries
        if _contains_value(registry.model_dump(mode="json"), rule_id)
    )


def _missing_coverage_items(
    rule: EscalationPolicyRule,
    downstream_registry_refs: tuple[str, ...],
) -> tuple[str, ...]:
    missing: list[str] = []
    if rule.rule_id not in ESCALATION_POLICY_REGISTRY.rule_ids:
        missing.append("policy_registry_missing_rule")
    if not rule.source_contract_registries:
        missing.append("source_contract_registries_missing")
    if not set(rule.source_contract_registries).issubset(
        ESCALATION_POLICY_REGISTRY.source_contract_registries
    ):
        missing.append("unknown_source_contract_registry")
    if not rule.trigger_signals:
        missing.append("trigger_signals_missing")
    if not rule.evidence_sources:
        missing.append("evidence_sources_missing")
    if not downstream_registry_refs:
        missing.append("downstream_policy_reference_missing")
    if "does not evaluate policy" not in rule.authority_boundary:
        missing.append("authority_boundary_policy_evaluation_block_missing")
    for blocked_behavior in (
        "provider_or_model_routing",
        "runtime_selection",
        "retry_or_refinement_triggering",
        "agent_invocation",
        "generated_output_modification",
    ):
        if blocked_behavior not in rule.blocked_runtime_behaviors:
            missing.append(f"{blocked_behavior}_block_missing")
    return tuple(missing)


def _audit_record(rule: EscalationPolicyRule) -> EscalationPolicyAuditRecord:
    downstream_refs = _downstream_registry_refs(rule.rule_id)
    return EscalationPolicyAuditRecord(
        rule_id=rule.rule_id,
        policy_stage=rule.policy_stage,
        rule_serialization_version=rule.serialization_version,
        audited_registry_refs=_AUDITED_REGISTRY_REFS,
        rule_source_contract_registries=rule.source_contract_registries,
        downstream_registry_refs=downstream_refs,
        trigger_signals=rule.trigger_signals,
        evidence_sources=rule.evidence_sources,
        validated_policy_surfaces=_VALIDATED_POLICY_SURFACES,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
        audit_findings=_AUDIT_FINDINGS,
        missing_coverage_items=_missing_coverage_items(rule, downstream_refs),
        rule_blocked_runtime_behaviors=rule.blocked_runtime_behaviors,
    )


ESCALATION_POLICY_AUDIT_RECORDS = tuple(
    _audit_record(rule) for rule in ESCALATION_POLICY_RULES
)
ESCALATION_POLICY_AUDIT_REGISTRY = EscalationPolicyAuditRegistry(
    audit_records=ESCALATION_POLICY_AUDIT_RECORDS,
    rule_ids=tuple(record.rule_id for record in ESCALATION_POLICY_AUDIT_RECORDS),
    audit_count=len(ESCALATION_POLICY_AUDIT_RECORDS),
    audited_registry_refs=_AUDITED_REGISTRY_REFS,
    source_contract_registries=ESCALATION_POLICY_REGISTRY.source_contract_registries,
    validated_policy_surfaces=_VALIDATED_POLICY_SURFACES,
    passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
)
