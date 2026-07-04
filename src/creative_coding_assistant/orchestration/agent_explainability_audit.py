"""Passive V4.6 agent explainability audit metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_contract_audit import (
    agent_contract_audit_registry,
)
from creative_coding_assistant.orchestration.agent_contracts import (
    AGENT_CONTRACT_REGISTRY,
    AgentContract,
)
from creative_coding_assistant.orchestration.hybrid_agentic_workflow import (
    decision_provenance_registry,
    escalation_trace_registry,
)

AgentExplainabilityAuditStage = Literal["v4_6_agent_explainability_hardening"]
AgentExplainabilityAuditStatus = Literal["pass"]

AGENT_EXPLAINABILITY_AUDIT_RECORD_SERIALIZATION_VERSION = (
    "agent_explainability_audit_record.v1"
)
AGENT_EXPLAINABILITY_AUDIT_REGISTRY_SERIALIZATION_VERSION = (
    "agent_explainability_audit_registry.v1"
)
AGENT_EXPLAINABILITY_AUDIT_REGISTRY_AUTHORITY_BOUNDARY = (
    "V4.6 agent explainability audit metadata checks passive agent contract "
    "explanation surfaces, produced metadata, produced signals, output "
    "summary contracts, source registry references, memory reference sources, "
    "contract audit alignment, and passive provenance and trace registry "
    "coverage only; it does not generate explanations, invoke agents, route "
    "providers or models, select runtimes, capture traces, write memory, "
    "trigger retries, or modify generated output."
)

_VALIDATED_EXPLAINABILITY_SURFACES = (
    "produced_metadata",
    "produced_signals",
    "produced_outputs",
    "source_registry_refs",
    "memory_reference_sources",
    "contract_audit_alignment",
    "provenance_trace_registry_refs",
    "blocked_runtime_behaviors",
)
_PASSIVE_BOUNDARY_FLAGS = (
    "explanation_generation_blocked",
    "agent_invocation_blocked",
    "provider_model_routing_blocked",
    "runtime_selection_blocked",
    "trace_capture_blocked",
    "memory_write_blocked",
    "retry_triggering_blocked",
    "generated_output_mutation_blocked",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "explanation_generation",
    "agent_invocation",
    "provider_or_model_routing",
    "runtime_selection",
    "trace_capture",
    "memory_write",
    "retry_or_refinement_triggering",
    "generated_output_modification",
)
_AUDIT_FINDINGS = (
    "agent_contract_explainability_metadata_confirmed",
    "signal_surface_confirmed",
    "output_summary_contracts_confirmed",
    "source_registry_refs_confirmed",
    "memory_reference_sources_confirmed",
    "provenance_trace_registry_refs_confirmed",
    "runtime_behavior_blocks_confirmed",
)
_REQUIRED_CONTRACT_BLOCKS = (
    "agent_invocation",
    "provider_or_model_routing",
    "runtime_selection",
    "retry_or_refinement_triggering",
    "generated_output_modification",
)


class AgentExplainabilityAuditRecord(BaseModel):
    """One passive V4.6 explainability audit record for an agent contract."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=80)
    role_id: str = Field(min_length=1, max_length=80)
    audit_stage: AgentExplainabilityAuditStage = "v4_6_agent_explainability_hardening"
    contract_serialization_version: str = Field(min_length=1, max_length=80)
    contract_source_registry_refs: tuple[str, ...] = Field(min_length=1, max_length=24)
    memory_reference_sources: tuple[str, ...] = Field(min_length=1, max_length=12)
    explanation_metadata_keys: tuple[str, ...] = Field(min_length=1, max_length=18)
    explanation_signal_keys: tuple[str, ...] = Field(min_length=1, max_length=18)
    explanation_output_contracts: tuple[str, ...] = Field(min_length=1, max_length=16)
    decision_provenance_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    escalation_trace_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    validated_explainability_surfaces: tuple[str, ...] = Field(
        min_length=8,
        max_length=8,
    )
    passive_boundary_flags: tuple[str, ...] = Field(min_length=8, max_length=8)
    audit_findings: tuple[str, ...] = Field(min_length=7, max_length=7)
    missing_coverage_items: tuple[str, ...] = Field(
        default_factory=tuple, max_length=16
    )
    contract_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=16,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    audit_status: AgentExplainabilityAuditStatus = "pass"
    metadata_only_declared: Literal[True] = True
    provenance_memory_reference_present: Literal[True] = True
    active_explanation_generation_implemented: Literal[False] = False
    active_agent_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    trace_capture_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["agent_explainability_audit_record.v1"] = (
        AGENT_EXPLAINABILITY_AUDIT_RECORD_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentExplainabilityAuditRegistry(BaseModel):
    """Stable passive V4.6 audit registry for agent explainability metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_explainability_audit_registry"] = (
        "agent_explainability_audit_registry"
    )
    serialization_version: Literal["agent_explainability_audit_registry.v1"] = (
        AGENT_EXPLAINABILITY_AUDIT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_EXPLAINABILITY_AUDIT_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    audit_stage: AgentExplainabilityAuditStage = "v4_6_agent_explainability_hardening"
    audit_records: tuple[AgentExplainabilityAuditRecord, ...] = Field(
        min_length=12,
        max_length=12,
    )
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    audit_count: int = Field(ge=12, le=12)
    source_contract_registry: Literal["agent_contract_registry"] = (
        "agent_contract_registry"
    )
    source_contract_audit_registry: Literal["agent_contract_audit_registry"] = (
        "agent_contract_audit_registry"
    )
    source_registry_refs: tuple[str, ...] = Field(min_length=24, max_length=24)
    memory_reference_sources: tuple[str, ...] = Field(min_length=6, max_length=6)
    decision_provenance_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    escalation_trace_profile_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    validated_explainability_surfaces: tuple[str, ...] = Field(
        min_length=8,
        max_length=8,
    )
    passive_boundary_flags: tuple[str, ...] = Field(min_length=8, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    all_agent_contracts_covered: Literal[True] = True
    all_records_provenance_referenced: Literal[True] = True
    no_missing_coverage: Literal[True] = True
    active_explanation_generation_implemented: Literal[False] = False
    active_agent_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    trace_capture_implemented: Literal[False] = False
    memory_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_audit_records(self) -> Self:
        derived_agent_ids = tuple(record.agent_id for record in self.audit_records)
        if len(set(derived_agent_ids)) != len(derived_agent_ids):
            raise ValueError("agent_ids must be unique")
        if self.agent_ids != derived_agent_ids:
            raise ValueError("agent_ids must match audit records")
        if self.audit_count != len(self.audit_records):
            raise ValueError("audit_count must match audit records")

        contract_audit_agents = set(agent_contract_audit_registry().agent_ids)
        for record in self.audit_records:
            if record.agent_id not in contract_audit_agents:
                raise ValueError("agent contract audit must cover explainability agent")
            if not set(record.contract_source_registry_refs).issubset(
                self.source_registry_refs
            ):
                raise ValueError("contract_source_registry_refs must be known")
            if not set(record.memory_reference_sources).issubset(
                self.memory_reference_sources
            ):
                raise ValueError("memory_reference_sources must be known")
            if record.decision_provenance_profile_ids != (
                self.decision_provenance_profile_ids
            ):
                raise ValueError("decision_provenance_profile_ids must match registry")
            if record.escalation_trace_profile_ids != self.escalation_trace_profile_ids:
                raise ValueError("escalation_trace_profile_ids must match registry")
            if record.validated_explainability_surfaces != (
                self.validated_explainability_surfaces
            ):
                raise ValueError(
                    "validated_explainability_surfaces must match registry"
                )
            if record.passive_boundary_flags != self.passive_boundary_flags:
                raise ValueError("passive_boundary_flags must match registry")
            if not record.provenance_memory_reference_present:
                raise ValueError("records must reference provenance metadata")
            if record.missing_coverage_items:
                raise ValueError("audit records must not contain missing coverage")
        return self


def agent_explainability_audit_registry() -> AgentExplainabilityAuditRegistry:
    """Return passive V4.6 agent explainability audit metadata."""

    return AGENT_EXPLAINABILITY_AUDIT_REGISTRY


def agent_explainability_audit_by_agent_id(
    agent_id: str,
    registry: AgentExplainabilityAuditRegistry | None = None,
) -> AgentExplainabilityAuditRecord | None:
    """Return one passive explainability audit record by agent id."""

    source_registry = registry or AGENT_EXPLAINABILITY_AUDIT_REGISTRY
    for record in source_registry.audit_records:
        if record.agent_id == agent_id:
            return record
    return None


def agent_explainability_audits_for_source_registry(
    source_registry_ref: str,
    registry: AgentExplainabilityAuditRegistry | None = None,
) -> tuple[AgentExplainabilityAuditRecord, ...]:
    """Return passive explainability audit records for one source registry."""

    source_registry = registry or AGENT_EXPLAINABILITY_AUDIT_REGISTRY
    normalized_ref = str(source_registry_ref).strip()
    return tuple(
        record
        for record in source_registry.audit_records
        if normalized_ref in record.contract_source_registry_refs
    )


def agent_explainability_audits_for_memory_source(
    memory_source: str,
    registry: AgentExplainabilityAuditRegistry | None = None,
) -> tuple[AgentExplainabilityAuditRecord, ...]:
    """Return passive explainability audit records for one memory source."""

    source_registry = registry or AGENT_EXPLAINABILITY_AUDIT_REGISTRY
    normalized_source = str(memory_source).strip()
    return tuple(
        record
        for record in source_registry.audit_records
        if normalized_source in record.memory_reference_sources
    )


def _missing_coverage_items(contract: AgentContract) -> tuple[str, ...]:
    missing: list[str] = []
    if contract.agent_id not in agent_contract_audit_registry().agent_ids:
        missing.append("agent_contract_audit_missing")
    if not contract.produced_metadata:
        missing.append("produced_metadata_missing")
    if not contract.produced_signals:
        missing.append("produced_signals_missing")
    if not contract.produced_outputs:
        missing.append("produced_outputs_missing")
    if not contract.source_contract_registries:
        missing.append("source_contract_registry_refs_missing")
    if not contract.memory_access.allowed_memory_sources:
        missing.append("memory_reference_sources_missing")
    if "provenance_metadata" not in contract.memory_access.allowed_memory_sources:
        missing.append("provenance_memory_reference_missing")
    for blocked_behavior in _REQUIRED_CONTRACT_BLOCKS:
        if blocked_behavior not in contract.blocked_runtime_behaviors:
            missing.append(f"{blocked_behavior}_block_missing")
    if not contract.metadata_only:
        missing.append("metadata_only_declaration_missing")
    if contract.memory_access.writes_runtime_memory:
        missing.append("runtime_memory_write_enabled")
    if contract.memory_access.creates_memory_store:
        missing.append("memory_store_creation_enabled")
    if contract.estimated_cost_metadata.external_provider_calls:
        missing.append("external_provider_calls_enabled")
    if contract.estimated_latency_metadata.network_required:
        missing.append("network_latency_enabled")
    return tuple(missing)


def _audit_record(contract: AgentContract) -> AgentExplainabilityAuditRecord:
    return AgentExplainabilityAuditRecord(
        agent_id=contract.agent_id,
        role_id=contract.role_id,
        contract_serialization_version=contract.serialization_version,
        contract_source_registry_refs=contract.source_contract_registries,
        memory_reference_sources=contract.memory_access.allowed_memory_sources,
        explanation_metadata_keys=contract.produced_metadata,
        explanation_signal_keys=contract.produced_signals,
        explanation_output_contracts=contract.produced_outputs,
        decision_provenance_profile_ids=(
            decision_provenance_registry().provenance_profile_ids
        ),
        escalation_trace_profile_ids=escalation_trace_registry().trace_profile_ids,
        validated_explainability_surfaces=_VALIDATED_EXPLAINABILITY_SURFACES,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
        audit_findings=_AUDIT_FINDINGS,
        missing_coverage_items=_missing_coverage_items(contract),
        contract_blocked_runtime_behaviors=contract.blocked_runtime_behaviors,
        metadata_only_declared=contract.metadata_only,
        provenance_memory_reference_present=(
            "provenance_metadata" in contract.memory_access.allowed_memory_sources
        ),
    )


AGENT_EXPLAINABILITY_AUDIT_RECORDS = tuple(
    _audit_record(contract) for contract in AGENT_CONTRACT_REGISTRY.contracts
)
AGENT_EXPLAINABILITY_AUDIT_REGISTRY = AgentExplainabilityAuditRegistry(
    audit_records=AGENT_EXPLAINABILITY_AUDIT_RECORDS,
    agent_ids=tuple(record.agent_id for record in AGENT_EXPLAINABILITY_AUDIT_RECORDS),
    audit_count=len(AGENT_EXPLAINABILITY_AUDIT_RECORDS),
    source_registry_refs=tuple(
        dict.fromkeys(
            source_ref
            for contract in AGENT_CONTRACT_REGISTRY.contracts
            for source_ref in contract.source_contract_registries
        )
    ),
    memory_reference_sources=tuple(
        dict.fromkeys(
            memory_source
            for contract in AGENT_CONTRACT_REGISTRY.contracts
            for memory_source in contract.memory_access.allowed_memory_sources
        )
    ),
    decision_provenance_profile_ids=decision_provenance_registry().provenance_profile_ids,
    escalation_trace_profile_ids=escalation_trace_registry().trace_profile_ids,
    validated_explainability_surfaces=_VALIDATED_EXPLAINABILITY_SURFACES,
    passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
)
