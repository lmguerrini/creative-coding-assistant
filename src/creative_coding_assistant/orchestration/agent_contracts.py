"""Passive V4.1 agent contract foundation metadata."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

AgentContractCategory = Literal["multi_agent_core"]
AgentContractStage = Literal["v4_1_contract_foundation"]
AgentContractCacheability = Literal[
    "deterministic_static_metadata",
    "deterministic_with_upstream_metadata",
]
AgentContractCostClass = Literal["none", "low", "medium"]
AgentContractLatencyClass = Literal["none", "low", "medium"]
AgentMemoryAccessMode = Literal[
    "no_runtime_memory",
    "metadata_reference_only",
]

AGENT_CONTRACT_SERIALIZATION_VERSION = "agent_contract.v1"
AGENT_CONTRACT_REGISTRY_SERIALIZATION_VERSION = "agent_contract_registry.v1"
AGENT_CONTRACT_REGISTRY_AUTHORITY_BOUNDARY = (
    "V4.1 agent contracts describe passive agent identities, roles, "
    "capabilities, boundaries, metadata inputs and outputs, memory access "
    "posture, cost hints, latency hints, and future orchestration hooks only; "
    "they do not create agents, invoke agents, route work, call providers, "
    "select models or runtimes, coordinate collaboration, trigger retries, "
    "write memory, execute artifacts, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "agent_instantiation",
    "agent_invocation",
    "dynamic_agent_routing",
    "multi_agent_orchestration",
    "provider_or_model_routing",
    "runtime_selection",
    "workflow_control",
    "retry_or_refinement_triggering",
    "memory_storage_or_mutation",
    "artifact_execution",
    "generated_output_modification",
)

_FORBIDDEN_MEMORY_OPERATIONS = (
    "memory_write",
    "memory_store_creation",
    "blackboard_memory_mutation",
    "shared_context_mutation",
)


class AgentMemoryAccessContract(BaseModel):
    """Metadata-only description of an agent contract's memory posture."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    access_mode: AgentMemoryAccessMode = "metadata_reference_only"
    allowed_memory_sources: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    forbidden_memory_operations: tuple[str, ...] = Field(
        default=_FORBIDDEN_MEMORY_OPERATIONS,
        min_length=1,
        max_length=12,
    )
    retention_policy: str = Field(
        default=(
            "Agent contracts may reference existing metadata but do not retain "
            "or write memory."
        ),
        min_length=1,
        max_length=260,
    )
    reads_runtime_memory: Literal[False] = False
    writes_runtime_memory: Literal[False] = False
    creates_memory_store: Literal[False] = False
    metadata_only: Literal[True] = True


class AgentContractCostMetadata(BaseModel):
    """Static estimated cost metadata for a passive agent contract."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    relative_cost: AgentContractCostClass = "none"
    external_provider_calls: Literal[False] = False
    cost_basis: str = Field(min_length=1, max_length=260)
    cache_sensitivity: str = Field(min_length=1, max_length=260)


class AgentContractLatencyMetadata(BaseModel):
    """Static estimated latency metadata for a passive agent contract."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    relative_latency: AgentContractLatencyClass = "none"
    network_required: Literal[False] = False
    latency_basis: str = Field(min_length=1, max_length=260)
    blocking_inputs: tuple[str, ...] = Field(default_factory=tuple, max_length=16)


class AgentContract(BaseModel):
    """Common metadata contract surface for a future V4.1 agent."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=80)
    agent_name: str = Field(min_length=1, max_length=140)
    agent_version: str = Field(min_length=1, max_length=24)
    agent_category: AgentContractCategory = "multi_agent_core"
    contract_stage: AgentContractStage = "v4_1_contract_foundation"
    role_id: str = Field(min_length=1, max_length=80)
    role_name: str = Field(min_length=1, max_length=140)
    role_purpose: str = Field(min_length=1, max_length=260)
    authority_boundary: str = Field(min_length=1, max_length=900)
    allowed_actions: tuple[str, ...] = Field(min_length=1, max_length=16)
    prohibited_actions: tuple[str, ...] = Field(min_length=1, max_length=16)
    capabilities: tuple[str, ...] = Field(min_length=1, max_length=16)
    required_inputs: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    optional_inputs: tuple[str, ...] = Field(default_factory=tuple, max_length=24)
    produced_outputs: tuple[str, ...] = Field(min_length=1, max_length=16)
    produced_metadata: tuple[str, ...] = Field(min_length=1, max_length=18)
    produced_signals: tuple[str, ...] = Field(min_length=1, max_length=18)
    memory_access: AgentMemoryAccessContract
    cacheability: AgentContractCacheability = "deterministic_static_metadata"
    estimated_cost_metadata: AgentContractCostMetadata
    estimated_latency_metadata: AgentContractLatencyMetadata
    future_orchestration_hooks: tuple[str, ...] = Field(min_length=1, max_length=12)
    source_contract_registries: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    serialization_version: Literal["agent_contract.v1"] = (
        AGENT_CONTRACT_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentContractRegistry(BaseModel):
    """Stable registry envelope for passive V4.1 agent contracts."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_contract_registry"] = "agent_contract_registry"
    agent_category: AgentContractCategory = "multi_agent_core"
    serialization_version: Literal["agent_contract_registry.v1"] = (
        AGENT_CONTRACT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_CONTRACT_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    contracts: tuple[AgentContract, ...] = Field(default_factory=tuple, max_length=32)
    agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=32)
    contract_count: int = Field(ge=0, le=32)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_contracts(self) -> Self:
        derived_agent_ids = tuple(contract.agent_id for contract in self.contracts)
        if len(set(derived_agent_ids)) != len(derived_agent_ids):
            raise ValueError("agent_ids must be unique")
        if self.agent_ids != derived_agent_ids:
            raise ValueError("agent_ids must match contracts")
        if self.contract_count != len(self.contracts):
            raise ValueError("contract_count must match contracts")
        return self


def build_agent_contract_registry(
    contracts: Iterable[AgentContract],
) -> AgentContractRegistry:
    """Build a passive registry envelope without creating executable agents."""

    contract_tuple = tuple(contracts)
    return AgentContractRegistry(
        contracts=contract_tuple,
        agent_ids=tuple(contract.agent_id for contract in contract_tuple),
        contract_count=len(contract_tuple),
    )


def agent_contract_registry() -> AgentContractRegistry:
    """Return the static V4.1 agent contract foundation registry."""

    return AGENT_CONTRACT_REGISTRY


def agent_contract_by_id(
    agent_id: str,
    registry: AgentContractRegistry | None = None,
) -> AgentContract | None:
    """Return one passive contract by id without invoking or routing agents."""

    source_registry = registry or AGENT_CONTRACT_REGISTRY
    for contract in source_registry.contracts:
        if contract.agent_id == agent_id:
            return contract
    return None


AGENT_CONTRACTS: tuple[AgentContract, ...] = ()
AGENT_CONTRACT_REGISTRY = build_agent_contract_registry(AGENT_CONTRACTS)
