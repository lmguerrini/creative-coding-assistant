"""Passive V4.6 hybrid workflow audit metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_capabilities import (
    agent_capability_registry,
)
from creative_coding_assistant.orchestration.escalation_policy import (
    escalation_policy_registry,
)
from creative_coding_assistant.orchestration.hybrid_agentic_workflow import (
    HYBRID_AGENTIC_WORKFLOW_REGISTRY,
    HYBRID_AGENTIC_WORKFLOW_STAGES,
    HybridAgenticWorkflowStage,
)
from creative_coding_assistant.orchestration.workflow_graph import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
)

HybridWorkflowAuditStage = Literal["v4_6_hybrid_workflow_hardening"]
HybridWorkflowAuditStatus = Literal["pass"]

HYBRID_WORKFLOW_AUDIT_SERIALIZATION_VERSION = "hybrid_workflow_audit.v1"
HYBRID_WORKFLOW_AUDIT_REGISTRY_SERIALIZATION_VERSION = (
    "hybrid_workflow_audit_registry.v1"
)
HYBRID_WORKFLOW_AUDIT_REGISTRY_AUTHORITY_BOUNDARY = (
    "V4.6 hybrid workflow audit metadata checks passive V4.3 hybrid workflow "
    "stage coverage, source registry alignment, V3 workflow node references, "
    "future capability references, escalation policy references, advisory "
    "outputs, and blocked runtime behaviors only; it does not change workflow "
    "graph order, execute agents, route providers or models, select runtimes, "
    "trigger retries, execute artifacts, or modify generated output."
)

_VALIDATED_STAGE_SURFACES = (
    "authority_boundary",
    "v3_workflow_nodes",
    "future_capability_ids",
    "escalation_rule_ids",
    "source_metadata_registries",
    "advisory_outputs",
    "blocked_runtime_behaviors",
)
_PASSIVE_BOUNDARY_FLAGS = (
    "workflow_order_mutation_blocked",
    "agent_execution_blocked",
    "provider_model_routing_blocked",
    "runtime_selection_blocked",
    "retry_triggering_blocked",
    "artifact_execution_blocked",
    "generated_output_mutation_blocked",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "workflow_graph_order_change",
    "agent_execution",
    "provider_or_model_routing",
    "runtime_selection",
    "retry_or_refinement_triggering",
    "artifact_execution",
    "generated_output_modification",
)
_AUDIT_FINDINGS = (
    "hybrid_stage_metadata_confirmed",
    "source_registry_alignment_confirmed",
    "v3_workflow_node_refs_confirmed",
    "future_capability_refs_confirmed",
    "escalation_policy_refs_confirmed",
    "runtime_behavior_blocks_confirmed",
)


class HybridWorkflowAuditRecord(BaseModel):
    """One passive V4.6 audit record for a hybrid workflow stage."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    stage_id: str = Field(min_length=1, max_length=80)
    stage_name: str = Field(min_length=1, max_length=140)
    stage_serialization_version: str = Field(min_length=1, max_length=80)
    audit_stage: HybridWorkflowAuditStage = "v4_6_hybrid_workflow_hardening"
    v3_workflow_nodes: tuple[str, ...] = Field(min_length=1, max_length=8)
    future_capability_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    escalation_rule_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_metadata_registries: tuple[str, ...] = Field(
        min_length=43,
        max_length=43,
    )
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=12)
    validated_stage_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=7, max_length=7)
    audit_findings: tuple[str, ...] = Field(min_length=6, max_length=6)
    missing_coverage_items: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    stage_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    audit_status: HybridWorkflowAuditStatus = "pass"
    metadata_only_declared: Literal[True] = True
    workflow_order_mutation_implemented: Literal[False] = False
    agent_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    artifact_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["hybrid_workflow_audit.v1"] = (
        HYBRID_WORKFLOW_AUDIT_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class HybridWorkflowAuditRegistry(BaseModel):
    """Stable passive V4.6 audit registry for hybrid workflow readiness."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["hybrid_workflow_audit_registry"] = (
        "hybrid_workflow_audit_registry"
    )
    serialization_version: Literal["hybrid_workflow_audit_registry.v1"] = (
        HYBRID_WORKFLOW_AUDIT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=HYBRID_WORKFLOW_AUDIT_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    audit_stage: HybridWorkflowAuditStage = "v4_6_hybrid_workflow_hardening"
    audit_records: tuple[HybridWorkflowAuditRecord, ...] = Field(
        min_length=5,
        max_length=5,
    )
    stage_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    audit_count: int = Field(ge=5, le=5)
    source_hybrid_workflow_registry: Literal["hybrid_agentic_workflow_registry"] = (
        "hybrid_agentic_workflow_registry"
    )
    source_metadata_registries: tuple[str, ...] = Field(
        min_length=43,
        max_length=43,
    )
    validated_stage_surfaces: tuple[str, ...] = Field(min_length=7, max_length=7)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=7, max_length=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    all_stages_covered: Literal[True] = True
    all_sources_covered: Literal[True] = True
    no_missing_coverage: Literal[True] = True
    active_runtime_audit_implemented: Literal[False] = False
    workflow_order_mutation_implemented: Literal[False] = False
    agent_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_audit_records(self) -> Self:
        derived_stage_ids = tuple(record.stage_id for record in self.audit_records)
        if len(set(derived_stage_ids)) != len(derived_stage_ids):
            raise ValueError("stage_ids must be unique")
        if self.stage_ids != derived_stage_ids:
            raise ValueError("stage_ids must match audit records")
        if self.audit_count != len(self.audit_records):
            raise ValueError("audit_count must match audit records")
        for record in self.audit_records:
            if record.source_metadata_registries != self.source_metadata_registries:
                raise ValueError("source_metadata_registries must match registry")
            if record.validated_stage_surfaces != self.validated_stage_surfaces:
                raise ValueError("validated_stage_surfaces must match registry")
            if record.passive_boundary_flags != self.passive_boundary_flags:
                raise ValueError("passive_boundary_flags must match registry")
            if record.missing_coverage_items:
                raise ValueError("audit records must not contain missing coverage")
        return self


def hybrid_workflow_audit_registry() -> HybridWorkflowAuditRegistry:
    """Return passive V4.6 hybrid workflow audit metadata."""

    return HYBRID_WORKFLOW_AUDIT_REGISTRY


def hybrid_workflow_audit_by_stage_id(
    stage_id: str,
    registry: HybridWorkflowAuditRegistry | None = None,
) -> HybridWorkflowAuditRecord | None:
    """Return one passive hybrid workflow audit record by stage id."""

    source_registry = registry or HYBRID_WORKFLOW_AUDIT_REGISTRY
    for record in source_registry.audit_records:
        if record.stage_id == stage_id:
            return record
    return None


def hybrid_workflow_audits_for_source_registry(
    source_registry_ref: str,
    registry: HybridWorkflowAuditRegistry | None = None,
) -> tuple[HybridWorkflowAuditRecord, ...]:
    """Return passive hybrid audit records referencing one source registry."""

    source_registry = registry or HYBRID_WORKFLOW_AUDIT_REGISTRY
    normalized_ref = str(source_registry_ref).strip()
    return tuple(
        record
        for record in source_registry.audit_records
        if normalized_ref in record.source_metadata_registries
    )


def _missing_coverage_items(stage: HybridAgenticWorkflowStage) -> tuple[str, ...]:
    missing: list[str] = []
    if stage.stage_id not in HYBRID_AGENTIC_WORKFLOW_REGISTRY.stage_ids:
        missing.append("hybrid_workflow_registry_missing_stage")
    if (
        stage.source_metadata_registries
        != HYBRID_AGENTIC_WORKFLOW_REGISTRY.source_metadata_registries
    ):
        missing.append("source_metadata_registries_mismatch")
    if not set(stage.v3_workflow_nodes).issubset(ASSISTANT_WORKFLOW_NODE_ORDER):
        missing.append("unknown_v3_workflow_node")
    if not set(stage.future_capability_ids).issubset(
        agent_capability_registry().capability_ids
    ):
        missing.append("unknown_future_capability")
    if not set(stage.escalation_rule_ids).issubset(
        escalation_policy_registry().rule_ids
    ):
        missing.append("unknown_escalation_policy_rule")
    if not stage.advisory_outputs:
        missing.append("advisory_outputs_missing")
    if "provider_or_model_routing" not in stage.blocked_runtime_behaviors:
        missing.append("provider_model_routing_block_missing")
    if "runtime_selection" not in stage.blocked_runtime_behaviors:
        missing.append("runtime_selection_block_missing")
    if "retry_or_refinement_triggering" not in stage.blocked_runtime_behaviors:
        missing.append("retry_triggering_block_missing")
    if "generated_output_modification" not in stage.blocked_runtime_behaviors:
        missing.append("generated_output_mutation_block_missing")
    return tuple(missing)


def _audit_record(stage: HybridAgenticWorkflowStage) -> HybridWorkflowAuditRecord:
    return HybridWorkflowAuditRecord(
        stage_id=stage.stage_id,
        stage_name=stage.stage_name,
        stage_serialization_version=stage.serialization_version,
        v3_workflow_nodes=stage.v3_workflow_nodes,
        future_capability_ids=stage.future_capability_ids,
        escalation_rule_ids=stage.escalation_rule_ids,
        source_metadata_registries=stage.source_metadata_registries,
        advisory_outputs=stage.advisory_outputs,
        validated_stage_surfaces=_VALIDATED_STAGE_SURFACES,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
        audit_findings=_AUDIT_FINDINGS,
        missing_coverage_items=_missing_coverage_items(stage),
        stage_blocked_runtime_behaviors=stage.blocked_runtime_behaviors,
    )


HYBRID_WORKFLOW_AUDIT_RECORDS = tuple(
    _audit_record(stage) for stage in HYBRID_AGENTIC_WORKFLOW_STAGES
)
HYBRID_WORKFLOW_AUDIT_REGISTRY = HybridWorkflowAuditRegistry(
    audit_records=HYBRID_WORKFLOW_AUDIT_RECORDS,
    stage_ids=tuple(record.stage_id for record in HYBRID_WORKFLOW_AUDIT_RECORDS),
    audit_count=len(HYBRID_WORKFLOW_AUDIT_RECORDS),
    source_metadata_registries=(
        HYBRID_AGENTIC_WORKFLOW_REGISTRY.source_metadata_registries
    ),
    validated_stage_surfaces=_VALIDATED_STAGE_SURFACES,
    passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
)
