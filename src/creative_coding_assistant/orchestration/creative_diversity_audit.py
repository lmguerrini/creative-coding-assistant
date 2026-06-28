"""Passive V4.6 creative diversity audit metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_escalation_signals import (
    agent_escalation_signal_registry,
)
from creative_coding_assistant.orchestration.hybrid_agentic_workflow import (
    CREATIVE_EXPLORATION_BUDGET_REGISTRY,
    CreativeExplorationBudgetProfile,
    decision_provenance_registry,
    escalation_trace_registry,
)

CreativeDiversityAuditStage = Literal["v4_6_creative_diversity_hardening"]
CreativeDiversityAuditStatus = Literal["pass"]

CREATIVE_DIVERSITY_AUDIT_RECORD_SERIALIZATION_VERSION = (
    "creative_diversity_audit_record.v1"
)
CREATIVE_DIVERSITY_AUDIT_REGISTRY_SERIALIZATION_VERSION = (
    "creative_diversity_audit_registry.v1"
)
CREATIVE_DIVERSITY_AUDIT_REGISTRY_AUTHORITY_BOUNDARY = (
    "V4.6 creative diversity audit metadata checks passive exploration "
    "budget posture coverage, advisory variant bounds, source trace and "
    "provenance alignment, escalation signal references, source registry "
    "coverage, metadata-only declarations, and runtime diversity blocks only; "
    "it does not enforce budgets, generate variants, trigger refinement, "
    "route by cost, invoke agents, route providers or models, control "
    "workflow transitions, trigger retries, or modify generated output."
)

_VALIDATED_DIVERSITY_SURFACES = (
    "budget_profile_identity",
    "topic_alignment",
    "posture_sequence",
    "advisory_variant_bounds",
    "source_trace_alignment",
    "source_provenance_alignment",
    "escalation_signal_alignment",
)
_PASSIVE_BOUNDARY_FLAGS = (
    "budget_enforcement_blocked",
    "variant_generation_blocked",
    "refinement_triggering_blocked",
    "cost_routing_blocked",
    "agent_invocation_blocked",
    "provider_model_routing_blocked",
    "generated_output_mutation_blocked",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "budget_enforcement",
    "variant_generation",
    "refinement_triggering",
    "cost_routing",
    "agent_invocation",
    "provider_or_model_routing",
    "workflow_control",
    "retry_triggering",
    "generated_output_modification",
)
_AUDIT_FINDINGS = (
    "exploration_budget_profile_coverage_confirmed",
    "diversity_posture_sequence_confirmed",
    "advisory_variant_bounds_confirmed",
    "source_trace_alignment_confirmed",
    "provenance_and_signal_alignment_confirmed",
    "runtime_diversity_blocks_confirmed",
)
_REQUIRED_PROFILE_BLOCKS = (
    "budget_enforcement",
    "variant_generation",
    "refinement_triggering",
    "cost_routing",
    "agent_invocation",
    "provider_or_model_routing",
    "generated_output_modification",
)


class CreativeDiversityAuditRecord(BaseModel):
    """One passive V4.6 audit record for an exploration budget profile."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    budget_profile_id: str = Field(min_length=1, max_length=150)
    topic_id: str = Field(min_length=1, max_length=120)
    audit_stage: CreativeDiversityAuditStage = (
        "v4_6_creative_diversity_hardening"
    )
    budget_serialization_version: str = Field(min_length=1, max_length=120)
    budget_posture: str = Field(min_length=1, max_length=80)
    max_advisory_variants: int = Field(ge=0, le=3)
    max_advisory_refinement_passes: int = Field(ge=0, le=3)
    cost_pressure_signal: str = Field(min_length=1, max_length=120)
    source_trace_profile_id: str = Field(min_length=1, max_length=160)
    source_provenance_profile_id: str = Field(min_length=1, max_length=160)
    source_escalation_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    budget_dimensions: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    validated_diversity_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=7, max_length=7)
    audit_findings: tuple[str, ...] = Field(min_length=6, max_length=6)
    missing_coverage_items: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    profile_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    audit_status: CreativeDiversityAuditStatus = "pass"
    metadata_only_declared: Literal[True] = True
    budget_enforcement_implemented: Literal[False] = False
    variant_generation_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    cost_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["creative_diversity_audit_record.v1"] = (
        CREATIVE_DIVERSITY_AUDIT_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class CreativeDiversityAuditRegistry(BaseModel):
    """Stable passive V4.6 audit registry for creative diversity metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_diversity_audit_registry"] = (
        "creative_diversity_audit_registry"
    )
    serialization_version: Literal["creative_diversity_audit_registry.v1"] = (
        CREATIVE_DIVERSITY_AUDIT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CREATIVE_DIVERSITY_AUDIT_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    audit_stage: CreativeDiversityAuditStage = (
        "v4_6_creative_diversity_hardening"
    )
    audit_records: tuple[CreativeDiversityAuditRecord, ...] = Field(
        min_length=4,
        max_length=4,
    )
    budget_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    topic_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    budget_postures: tuple[str, ...] = Field(min_length=4, max_length=4)
    audit_count: int = Field(ge=4, le=4)
    source_creative_exploration_registry: Literal[
        "creative_exploration_budget_registry"
    ] = "creative_exploration_budget_registry"
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    trace_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    provenance_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    escalation_signal_ids: tuple[str, ...] = Field(min_length=7, max_length=7)
    validated_diversity_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    all_budget_profiles_covered: Literal[True] = True
    posture_sequence_confirmed: Literal[True] = True
    no_missing_coverage: Literal[True] = True
    active_diversity_generation_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    variant_generation_implemented: Literal[False] = False
    refinement_triggering_implemented: Literal[False] = False
    cost_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_audit_records(self) -> Self:
        derived_profile_ids = tuple(
            record.budget_profile_id for record in self.audit_records
        )
        derived_topic_ids = tuple(record.topic_id for record in self.audit_records)
        derived_postures = tuple(
            record.budget_posture for record in self.audit_records
        )
        if len(set(derived_profile_ids)) != len(derived_profile_ids):
            raise ValueError("budget_profile_ids must be unique")
        if self.budget_profile_ids != derived_profile_ids:
            raise ValueError("budget_profile_ids must match audit records")
        if self.topic_ids != derived_topic_ids:
            raise ValueError("topic_ids must match audit records")
        if self.budget_postures != derived_postures:
            raise ValueError("budget_postures must match audit records")
        if self.audit_count != len(self.audit_records):
            raise ValueError("audit_count must match audit records")
        for record in self.audit_records:
            if record.source_registries != self.source_registries:
                raise ValueError("source_registries must match registry")
            if record.validated_diversity_surfaces != (
                self.validated_diversity_surfaces
            ):
                raise ValueError("validated_diversity_surfaces must match registry")
            if record.passive_boundary_flags != self.passive_boundary_flags:
                raise ValueError("passive_boundary_flags must match registry")
            if record.source_trace_profile_id not in self.trace_profile_ids:
                raise ValueError("source_trace_profile_id must be known")
            if record.source_provenance_profile_id not in self.provenance_profile_ids:
                raise ValueError("source_provenance_profile_id must be known")
            if not set(record.source_escalation_signal_ids).issubset(
                self.escalation_signal_ids
            ):
                raise ValueError("source_escalation_signal_ids must be known")
            if record.missing_coverage_items:
                raise ValueError("audit records must not contain missing coverage")
        return self


def creative_diversity_audit_registry() -> CreativeDiversityAuditRegistry:
    """Return passive V4.6 creative diversity audit metadata."""

    return CREATIVE_DIVERSITY_AUDIT_REGISTRY


def creative_diversity_audit_by_profile_id(
    budget_profile_id: str,
    registry: CreativeDiversityAuditRegistry | None = None,
) -> CreativeDiversityAuditRecord | None:
    """Return one passive diversity audit record by budget profile id."""

    source_registry = registry or CREATIVE_DIVERSITY_AUDIT_REGISTRY
    for record in source_registry.audit_records:
        if record.budget_profile_id == budget_profile_id:
            return record
    return None


def creative_diversity_audits_for_posture(
    budget_posture: str,
    registry: CreativeDiversityAuditRegistry | None = None,
) -> tuple[CreativeDiversityAuditRecord, ...]:
    """Return passive diversity audit records for one budget posture."""

    source_registry = registry or CREATIVE_DIVERSITY_AUDIT_REGISTRY
    normalized_posture = str(budget_posture).strip()
    return tuple(
        record
        for record in source_registry.audit_records
        if record.budget_posture == normalized_posture
    )


def creative_diversity_audits_for_source_registry(
    source_registry_ref: str,
    registry: CreativeDiversityAuditRegistry | None = None,
) -> tuple[CreativeDiversityAuditRecord, ...]:
    """Return passive diversity audit records referencing one source registry."""

    source_registry = registry or CREATIVE_DIVERSITY_AUDIT_REGISTRY
    normalized_ref = str(source_registry_ref).strip()
    return tuple(
        record
        for record in source_registry.audit_records
        if normalized_ref in record.source_registries
    )


def _missing_coverage_items(
    profile: CreativeExplorationBudgetProfile,
) -> tuple[str, ...]:
    missing: list[str] = []
    budget_registry = CREATIVE_EXPLORATION_BUDGET_REGISTRY
    trace_registry = escalation_trace_registry()
    provenance_registry = decision_provenance_registry()
    signal_registry = agent_escalation_signal_registry()
    if profile.budget_profile_id not in budget_registry.budget_profile_ids:
        missing.append("budget_profile_missing")
    if profile.topic_id not in budget_registry.topic_ids:
        missing.append("topic_missing")
    if profile.budget_posture not in budget_registry.budget_postures:
        missing.append("budget_posture_missing")
    if profile.source_registries != budget_registry.source_registries:
        missing.append("source_registries_mismatch")
    if profile.source_trace_profile_id not in trace_registry.trace_profile_ids:
        missing.append("source_trace_profile_missing")
    if profile.source_provenance_profile_id not in (
        provenance_registry.provenance_profile_ids
    ):
        missing.append("source_provenance_profile_missing")
    if not set(profile.source_escalation_signal_ids).issubset(
        signal_registry.signal_ids
    ):
        missing.append("source_escalation_signal_missing")
    if not profile.budget_dimensions:
        missing.append("budget_dimensions_missing")
    if not profile.advisory_outputs:
        missing.append("advisory_outputs_missing")
    for blocked_behavior in _REQUIRED_PROFILE_BLOCKS:
        if blocked_behavior not in profile.blocked_runtime_behaviors:
            missing.append(f"{blocked_behavior}_block_missing")
    if profile.budget_enforcement_implemented:
        missing.append("budget_enforcement_enabled")
    if profile.variant_generation_implemented:
        missing.append("variant_generation_enabled")
    if profile.refinement_triggering_implemented:
        missing.append("refinement_triggering_enabled")
    if profile.cost_routing_implemented:
        missing.append("cost_routing_enabled")
    if profile.agent_invocation_implemented:
        missing.append("agent_invocation_enabled")
    if profile.workflow_control_implemented:
        missing.append("workflow_control_enabled")
    if profile.retry_triggering_implemented:
        missing.append("retry_triggering_enabled")
    if profile.generated_output_mutation_implemented:
        missing.append("generated_output_mutation_enabled")
    if not profile.metadata_only:
        missing.append("metadata_only_declaration_missing")
    return tuple(missing)


def _audit_record(
    profile: CreativeExplorationBudgetProfile,
) -> CreativeDiversityAuditRecord:
    return CreativeDiversityAuditRecord(
        budget_profile_id=profile.budget_profile_id,
        topic_id=profile.topic_id,
        budget_serialization_version=profile.serialization_version,
        budget_posture=profile.budget_posture,
        max_advisory_variants=profile.max_advisory_variants,
        max_advisory_refinement_passes=profile.max_advisory_refinement_passes,
        cost_pressure_signal=profile.cost_pressure_signal,
        source_trace_profile_id=profile.source_trace_profile_id,
        source_provenance_profile_id=profile.source_provenance_profile_id,
        source_escalation_signal_ids=profile.source_escalation_signal_ids,
        source_registries=profile.source_registries,
        budget_dimensions=profile.budget_dimensions,
        advisory_outputs=profile.advisory_outputs,
        validated_diversity_surfaces=_VALIDATED_DIVERSITY_SURFACES,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
        audit_findings=_AUDIT_FINDINGS,
        missing_coverage_items=_missing_coverage_items(profile),
        profile_blocked_runtime_behaviors=profile.blocked_runtime_behaviors,
        metadata_only_declared=profile.metadata_only,
    )


CREATIVE_DIVERSITY_AUDIT_RECORDS = tuple(
    _audit_record(profile)
    for profile in CREATIVE_EXPLORATION_BUDGET_REGISTRY.budget_profiles
)
CREATIVE_DIVERSITY_AUDIT_REGISTRY = CreativeDiversityAuditRegistry(
    audit_records=CREATIVE_DIVERSITY_AUDIT_RECORDS,
    budget_profile_ids=tuple(
        record.budget_profile_id for record in CREATIVE_DIVERSITY_AUDIT_RECORDS
    ),
    topic_ids=tuple(record.topic_id for record in CREATIVE_DIVERSITY_AUDIT_RECORDS),
    budget_postures=tuple(
        record.budget_posture for record in CREATIVE_DIVERSITY_AUDIT_RECORDS
    ),
    audit_count=len(CREATIVE_DIVERSITY_AUDIT_RECORDS),
    source_registries=CREATIVE_EXPLORATION_BUDGET_REGISTRY.source_registries,
    trace_profile_ids=CREATIVE_EXPLORATION_BUDGET_REGISTRY.trace_profile_ids,
    provenance_profile_ids=(
        CREATIVE_EXPLORATION_BUDGET_REGISTRY.provenance_profile_ids
    ),
    escalation_signal_ids=CREATIVE_EXPLORATION_BUDGET_REGISTRY.escalation_signal_ids,
    validated_diversity_surfaces=_VALIDATED_DIVERSITY_SURFACES,
    passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
)
