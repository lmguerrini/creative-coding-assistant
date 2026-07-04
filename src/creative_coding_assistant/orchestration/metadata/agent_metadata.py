"""Passive V4.1 agent operational metadata layer."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_contracts import AGENT_CONTRACTS

AgentParallelizationSupport = Literal[
    "parallel_after_required_inputs",
    "requires_ordered_upstream_metadata",
]
AgentFutureReadiness = Literal["future_orchestration_metadata_ready"]

AGENT_METADATA_SERIALIZATION_VERSION = "agent_metadata.v1"
AGENT_METADATA_REGISTRY_SERIALIZATION_VERSION = "agent_metadata_registry.v1"
AGENT_METADATA_REGISTRY_AUTHORITY_BOUNDARY = (
    "Agent metadata describes advisory cacheability, parallelization support, "
    "estimated cost, estimated latency, observability, auditability, and "
    "future orchestration readiness only; it does not implement caching, "
    "parallel execution, cost or latency routing, provider routing, runtime "
    "selection, retries, workflow control, or generated output changes."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "cache_implementation",
    "parallel_execution",
    "cost_or_latency_routing",
    "provider_or_model_routing",
    "runtime_selection",
    "retry_or_refinement_triggering",
    "workflow_control",
    "agent_execution",
    "generated_output_modification",
)

_OBSERVABILITY_SURFACES = (
    "agent_id",
    "role_id",
    "required_inputs",
    "produced_outputs",
    "blocked_runtime_behaviors",
)

_AUDITABILITY_SURFACES = (
    "authority_boundary",
    "allowed_actions",
    "prohibited_actions",
    "source_contract_registries",
    "future_orchestration_hooks",
)


class AgentOperationalMetadata(BaseModel):
    """Advisory operational metadata for one passive V4.1 agent contract."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=80)
    role_id: str = Field(min_length=1, max_length=80)
    cacheability: str = Field(min_length=1, max_length=80)
    parallelization_support: AgentParallelizationSupport
    estimated_cost_class: str = Field(min_length=1, max_length=40)
    estimated_cost_basis: str = Field(min_length=1, max_length=260)
    estimated_latency_class: str = Field(min_length=1, max_length=40)
    estimated_latency_basis: str = Field(min_length=1, max_length=260)
    observability_surfaces: tuple[str, ...] = Field(min_length=5, max_length=5)
    auditability_surfaces: tuple[str, ...] = Field(min_length=5, max_length=5)
    future_orchestration_readiness: AgentFutureReadiness = (
        "future_orchestration_metadata_ready"
    )
    advisory_only: Literal[True] = True
    caching_implemented: Literal[False] = False
    parallel_execution_implemented: Literal[False] = False
    cost_latency_routing_implemented: Literal[False] = False
    serialization_version: Literal["agent_metadata.v1"] = (
        AGENT_METADATA_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentMetadataRegistry(BaseModel):
    """Stable advisory metadata registry for all passive V4.1 agents."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_metadata_registry"] = "agent_metadata_registry"
    serialization_version: Literal["agent_metadata_registry.v1"] = (
        AGENT_METADATA_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_METADATA_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    metadata: tuple[AgentOperationalMetadata, ...] = Field(
        min_length=12,
        max_length=12,
    )
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    metadata_count: int = Field(ge=12, le=12)
    source_contract_registry: Literal["agent_contract_registry"] = (
        "agent_contract_registry"
    )
    observability_surfaces: tuple[str, ...] = Field(min_length=5, max_length=5)
    auditability_surfaces: tuple[str, ...] = Field(min_length=5, max_length=5)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    advisory_only: Literal[True] = True
    caching_implemented: Literal[False] = False
    parallel_execution_implemented: Literal[False] = False
    cost_latency_routing_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_metadata(self) -> Self:
        derived_agent_ids = tuple(item.agent_id for item in self.metadata)
        if len(set(derived_agent_ids)) != len(derived_agent_ids):
            raise ValueError("agent_ids must be unique")
        if self.agent_ids != derived_agent_ids:
            raise ValueError("agent_ids must match metadata")
        if self.metadata_count != len(self.metadata):
            raise ValueError("metadata_count must match metadata")
        for item in self.metadata:
            if item.observability_surfaces != self.observability_surfaces:
                raise ValueError("observability_surfaces must match registry")
            if item.auditability_surfaces != self.auditability_surfaces:
                raise ValueError("auditability_surfaces must match registry")
        return self


def agent_metadata_registry() -> AgentMetadataRegistry:
    """Return the static advisory V4.1 agent metadata registry."""

    return AGENT_METADATA_REGISTRY


def agent_metadata_by_agent_id(
    agent_id: str,
) -> AgentOperationalMetadata | None:
    """Return one advisory metadata entry without routing or execution."""

    for item in AGENT_METADATA:
        if item.agent_id == agent_id:
            return item
    return None


def _metadata(agent_id: str) -> AgentOperationalMetadata:
    contract = next(
        contract for contract in AGENT_CONTRACTS if contract.agent_id == agent_id
    )
    return AgentOperationalMetadata(
        agent_id=agent_id,
        role_id=contract.role_id,
        cacheability=contract.cacheability,
        parallelization_support="parallel_after_required_inputs",
        estimated_cost_class=contract.estimated_cost_metadata.relative_cost,
        estimated_cost_basis=contract.estimated_cost_metadata.cost_basis,
        estimated_latency_class=contract.estimated_latency_metadata.relative_latency,
        estimated_latency_basis=contract.estimated_latency_metadata.latency_basis,
        observability_surfaces=_OBSERVABILITY_SURFACES,
        auditability_surfaces=_AUDITABILITY_SURFACES,
    )


AGENT_METADATA = tuple(_metadata(contract.agent_id) for contract in AGENT_CONTRACTS)
AGENT_METADATA_REGISTRY = AgentMetadataRegistry(
    metadata=AGENT_METADATA,
    agent_ids=tuple(item.agent_id for item in AGENT_METADATA),
    metadata_count=len(AGENT_METADATA),
    observability_surfaces=_OBSERVABILITY_SURFACES,
    auditability_surfaces=_AUDITABILITY_SURFACES,
)
