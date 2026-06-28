"""Passive V4.6 agent contract audit metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_boundaries import (
    AGENT_BOUNDARY_REGISTRY,
)
from creative_coding_assistant.orchestration.agent_contracts import (
    AGENT_CONTRACT_REGISTRY,
    AGENT_CONTRACTS,
    AgentContract,
)
from creative_coding_assistant.orchestration.agent_memory_contracts import (
    AGENT_MEMORY_CONTRACT_REGISTRY,
)
from creative_coding_assistant.orchestration.agent_metadata import (
    AGENT_METADATA_REGISTRY,
)
from creative_coding_assistant.orchestration.agent_roles import AGENT_ROLE_REGISTRY

AgentContractAuditStage = Literal["v4_6_agent_contract_hardening"]
AgentContractAuditStatus = Literal["pass"]

AGENT_CONTRACT_AUDIT_SERIALIZATION_VERSION = "agent_contract_audit.v1"
AGENT_CONTRACT_AUDIT_REGISTRY_SERIALIZATION_VERSION = (
    "agent_contract_audit_registry.v1"
)
AGENT_CONTRACT_AUDIT_REGISTRY_AUTHORITY_BOUNDARY = (
    "V4.6 agent contract audit metadata checks passive V4.1 agent contract "
    "coverage, source registry alignment, authority boundaries, memory "
    "posture, cost and latency hints, and blocked runtime behaviors only; it "
    "does not execute agents, route providers or models, select runtimes, "
    "trigger retries, control workflows, write memory, or modify generated "
    "outputs."
)

_AUDITED_REGISTRY_REFS = (
    "agent_contract_registry",
    "agent_role_registry",
    "agent_boundary_registry",
    "agent_metadata_registry",
    "agent_memory_contract_registry",
)
_VALIDATED_CONTRACT_SURFACES = (
    "authority_boundary",
    "allowed_actions",
    "prohibited_actions",
    "required_inputs",
    "optional_inputs",
    "produced_outputs",
    "produced_metadata",
    "produced_signals",
    "memory_access",
    "estimated_cost_metadata",
    "estimated_latency_metadata",
    "blocked_runtime_behaviors",
)
_PASSIVE_BOUNDARY_FLAGS = (
    "metadata_only_declared",
    "runtime_memory_reads_blocked",
    "runtime_memory_writes_blocked",
    "memory_store_creation_blocked",
    "external_provider_calls_blocked",
    "network_latency_blocked",
    "provider_model_routing_blocked",
    "generated_output_mutation_blocked",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "audit_runtime_execution",
    "agent_invocation",
    "provider_or_model_routing",
    "runtime_selection",
    "workflow_control",
    "retry_or_refinement_triggering",
    "memory_mutation",
    "generated_output_modification",
)
_AUDIT_FINDINGS = (
    "contract_metadata_only_confirmed",
    "role_registry_alignment_confirmed",
    "boundary_registry_alignment_confirmed",
    "metadata_registry_alignment_confirmed",
    "memory_contract_registry_alignment_confirmed",
    "cost_latency_metadata_confirmed",
    "runtime_behavior_blocks_confirmed",
)


class AgentContractAuditRecord(BaseModel):
    """One passive V4.6 audit record for a V4.1 agent contract."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=80)
    role_id: str = Field(min_length=1, max_length=80)
    contract_serialization_version: str = Field(min_length=1, max_length=80)
    audit_stage: AgentContractAuditStage = "v4_6_agent_contract_hardening"
    audited_registry_refs: tuple[str, ...] = Field(min_length=5, max_length=5)
    contract_source_registry_refs: tuple[str, ...] = Field(max_length=12)
    validated_contract_surfaces: tuple[str, ...] = Field(
        min_length=12,
        max_length=12,
    )
    passive_boundary_flags: tuple[str, ...] = Field(min_length=8, max_length=8)
    audit_findings: tuple[str, ...] = Field(min_length=7, max_length=7)
    missing_coverage_items: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    contract_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=16,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    audit_status: AgentContractAuditStatus = "pass"
    metadata_only_declared: Literal[True] = True
    active_agent_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["agent_contract_audit.v1"] = (
        AGENT_CONTRACT_AUDIT_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentContractAuditRegistry(BaseModel):
    """Stable passive V4.6 audit registry for all agent contracts."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_contract_audit_registry"] = "agent_contract_audit_registry"
    serialization_version: Literal["agent_contract_audit_registry.v1"] = (
        AGENT_CONTRACT_AUDIT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_CONTRACT_AUDIT_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    audit_stage: AgentContractAuditStage = "v4_6_agent_contract_hardening"
    audit_records: tuple[AgentContractAuditRecord, ...] = Field(
        min_length=12,
        max_length=12,
    )
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    audit_count: int = Field(ge=12, le=12)
    source_contract_registry: Literal["agent_contract_registry"] = (
        "agent_contract_registry"
    )
    audited_registry_refs: tuple[str, ...] = Field(min_length=5, max_length=5)
    validated_contract_surfaces: tuple[str, ...] = Field(
        min_length=12,
        max_length=12,
    )
    passive_boundary_flags: tuple[str, ...] = Field(min_length=8, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    all_contracts_covered: Literal[True] = True
    no_missing_coverage: Literal[True] = True
    active_runtime_audit_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
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
        for record in self.audit_records:
            if record.audited_registry_refs != self.audited_registry_refs:
                raise ValueError("audited_registry_refs must match registry")
            if record.validated_contract_surfaces != self.validated_contract_surfaces:
                raise ValueError("validated_contract_surfaces must match registry")
            if record.passive_boundary_flags != self.passive_boundary_flags:
                raise ValueError("passive_boundary_flags must match registry")
            if record.missing_coverage_items:
                raise ValueError("audit records must not contain missing coverage")
            if not record.metadata_only_declared:
                raise ValueError("audit records must declare metadata only")
        return self


def agent_contract_audit_registry() -> AgentContractAuditRegistry:
    """Return passive V4.6 agent contract audit metadata."""

    return AGENT_CONTRACT_AUDIT_REGISTRY


def agent_contract_audit_by_agent_id(
    agent_id: str,
    registry: AgentContractAuditRegistry | None = None,
) -> AgentContractAuditRecord | None:
    """Return one passive contract audit record without executing agents."""

    source_registry = registry or AGENT_CONTRACT_AUDIT_REGISTRY
    for record in source_registry.audit_records:
        if record.agent_id == agent_id:
            return record
    return None


def agent_contract_audits_for_registry_ref(
    registry_ref: str,
    registry: AgentContractAuditRegistry | None = None,
) -> tuple[AgentContractAuditRecord, ...]:
    """Return passive audit records referencing one source registry."""

    source_registry = registry or AGENT_CONTRACT_AUDIT_REGISTRY
    normalized_ref = str(registry_ref).strip()
    return tuple(
        record
        for record in source_registry.audit_records
        if normalized_ref in record.audited_registry_refs
    )


def _registry_contains_agent(registry: object, agent_id: str) -> bool:
    return agent_id in tuple(getattr(registry, "agent_ids", ()))


def _missing_coverage_items(contract: AgentContract) -> tuple[str, ...]:
    missing: list[str] = []
    if not contract.metadata_only:
        missing.append("contract_metadata_only_missing")
    if not _registry_contains_agent(AGENT_ROLE_REGISTRY, contract.agent_id):
        missing.append("role_registry_missing_agent")
    if contract.role_id not in AGENT_ROLE_REGISTRY.role_ids:
        missing.append("role_registry_missing_role")
    if not _registry_contains_agent(AGENT_BOUNDARY_REGISTRY, contract.agent_id):
        missing.append("boundary_registry_missing_agent")
    if not _registry_contains_agent(AGENT_METADATA_REGISTRY, contract.agent_id):
        missing.append("metadata_registry_missing_agent")
    if not _registry_contains_agent(AGENT_MEMORY_CONTRACT_REGISTRY, contract.agent_id):
        missing.append("memory_contract_registry_missing_agent")
    if not contract.source_contract_registries:
        missing.append("contract_source_registry_refs_missing")
    if not contract.memory_access.metadata_only:
        missing.append("memory_access_metadata_only_missing")
    if contract.memory_access.reads_runtime_memory:
        missing.append("runtime_memory_read_enabled")
    if contract.memory_access.writes_runtime_memory:
        missing.append("runtime_memory_write_enabled")
    if contract.memory_access.creates_memory_store:
        missing.append("memory_store_creation_enabled")
    if contract.estimated_cost_metadata.external_provider_calls:
        missing.append("external_provider_calls_enabled")
    if contract.estimated_latency_metadata.network_required:
        missing.append("network_latency_enabled")
    if "provider_or_model_routing" not in contract.blocked_runtime_behaviors:
        missing.append("provider_model_routing_block_missing")
    if "generated_output_modification" not in contract.blocked_runtime_behaviors:
        missing.append("generated_output_mutation_block_missing")
    return tuple(missing)


def _audit_record(contract: AgentContract) -> AgentContractAuditRecord:
    return AgentContractAuditRecord(
        agent_id=contract.agent_id,
        role_id=contract.role_id,
        contract_serialization_version=contract.serialization_version,
        audited_registry_refs=_AUDITED_REGISTRY_REFS,
        contract_source_registry_refs=contract.source_contract_registries,
        validated_contract_surfaces=_VALIDATED_CONTRACT_SURFACES,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
        audit_findings=_AUDIT_FINDINGS,
        missing_coverage_items=_missing_coverage_items(contract),
        contract_blocked_runtime_behaviors=contract.blocked_runtime_behaviors,
        metadata_only_declared=contract.metadata_only,
    )


AGENT_CONTRACT_AUDIT_RECORDS = tuple(
    _audit_record(contract) for contract in AGENT_CONTRACTS
)
AGENT_CONTRACT_AUDIT_REGISTRY = AgentContractAuditRegistry(
    audit_records=AGENT_CONTRACT_AUDIT_RECORDS,
    agent_ids=tuple(record.agent_id for record in AGENT_CONTRACT_AUDIT_RECORDS),
    audit_count=len(AGENT_CONTRACT_AUDIT_RECORDS),
    audited_registry_refs=_AUDITED_REGISTRY_REFS,
    validated_contract_surfaces=_VALIDATED_CONTRACT_SURFACES,
    passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
)
