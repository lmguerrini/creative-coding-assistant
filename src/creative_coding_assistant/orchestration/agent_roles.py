"""Passive V4.1 agent role registry metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_contracts import AGENT_CONTRACTS
from creative_coding_assistant.orchestration.agent_identities import (
    AGENT_IDENTITIES,
    AgentCapabilityClass,
    AgentRoleFamily,
)

AGENT_ROLE_SERIALIZATION_VERSION = "agent_role.v1"
AGENT_ROLE_REGISTRY_SERIALIZATION_VERSION = "agent_role_registry.v1"
AGENT_ROLE_REGISTRY_AUTHORITY_BOUNDARY = (
    "Agent role registry metadata records the complete V4.1 passive role "
    "order, family grouping, capability family metadata, and contract "
    "relationships only; it does not execute agents, route tasks dynamically, "
    "implement orchestration, call providers, select runtimes, trigger "
    "retries, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "agent_execution",
    "dynamic_task_routing",
    "multi_agent_orchestration",
    "provider_or_model_routing",
    "runtime_selection",
    "retry_or_refinement_triggering",
    "workflow_control",
    "artifact_execution",
    "generated_output_modification",
)


class AgentRoleMetadata(BaseModel):
    """Static role metadata for one passive V4.1 agent role."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role_id: str = Field(min_length=1, max_length=80)
    role_name: str = Field(min_length=1, max_length=140)
    agent_id: str = Field(min_length=1, max_length=80)
    agent_name: str = Field(min_length=1, max_length=140)
    role_order: int = Field(ge=1, le=32)
    role_family: AgentRoleFamily
    capability_family: AgentCapabilityClass
    purpose: str = Field(min_length=1, max_length=260)
    contract_serialization_version: str = Field(min_length=1, max_length=80)
    source_contract_registries: tuple[str, ...] = Field(max_length=12)
    produced_outputs: tuple[str, ...] = Field(min_length=1, max_length=16)
    future_orchestration_hooks: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    serialization_version: Literal["agent_role.v1"] = (
        AGENT_ROLE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentRoleRegistry(BaseModel):
    """Stable registry of all passive V4.1 agent roles."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_role_registry"] = "agent_role_registry"
    serialization_version: Literal["agent_role_registry.v1"] = (
        AGENT_ROLE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=AGENT_ROLE_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    roles: tuple[AgentRoleMetadata, ...] = Field(min_length=12, max_length=12)
    role_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    role_count: int = Field(ge=12, le=12)
    role_families: tuple[str, ...] = Field(min_length=11, max_length=11)
    capability_families: tuple[str, ...] = Field(min_length=12, max_length=12)
    source_identity_registry: Literal["agent_identity_registry"] = (
        "agent_identity_registry"
    )
    source_contract_registry: Literal["agent_contract_registry"] = (
        "agent_contract_registry"
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_roles(self) -> Self:
        derived_role_ids = tuple(role.role_id for role in self.roles)
        derived_agent_ids = tuple(role.agent_id for role in self.roles)
        if len(set(derived_role_ids)) != len(derived_role_ids):
            raise ValueError("role_ids must be unique")
        if len(set(derived_agent_ids)) != len(derived_agent_ids):
            raise ValueError("agent_ids must be unique")
        if self.role_ids != derived_role_ids:
            raise ValueError("role_ids must match roles")
        if self.agent_ids != derived_agent_ids:
            raise ValueError("agent_ids must match roles")
        if self.role_count != len(self.roles):
            raise ValueError("role_count must match roles")
        role_families = tuple(dict.fromkeys(role.role_family for role in self.roles))
        capability_families = tuple(
            dict.fromkeys(role.capability_family for role in self.roles)
        )
        if self.role_families != role_families:
            raise ValueError("role_families must match roles")
        if self.capability_families != capability_families:
            raise ValueError("capability_families must match roles")
        return self


def agent_role_registry() -> AgentRoleRegistry:
    """Return the static passive V4.1 agent role registry."""

    return AGENT_ROLE_REGISTRY


def agent_role_by_id(role_id: str) -> AgentRoleMetadata | None:
    """Return one role metadata entry without routing or invoking agents."""

    for role in AGENT_ROLES:
        if role.role_id == role_id:
            return role
    return None


def agent_roles_by_family(role_family: str) -> tuple[AgentRoleMetadata, ...]:
    """Return roles in one role family without changing runtime behavior."""

    return tuple(role for role in AGENT_ROLES if role.role_family == role_family)


def agent_roles_by_capability_family(
    capability_family: str,
) -> tuple[AgentRoleMetadata, ...]:
    """Return roles in one capability family without changing runtime behavior."""

    return tuple(
        role for role in AGENT_ROLES if role.capability_family == capability_family
    )


def _role(index: int, agent_id: str) -> AgentRoleMetadata:
    contract = next(
        contract for contract in AGENT_CONTRACTS if contract.agent_id == agent_id
    )
    identity = next(
        identity for identity in AGENT_IDENTITIES if identity.agent_id == agent_id
    )
    return AgentRoleMetadata(
        role_id=contract.role_id,
        role_name=contract.role_name,
        agent_id=agent_id,
        agent_name=contract.agent_name,
        role_order=index,
        role_family=identity.role_family,
        capability_family=identity.capability_class,
        purpose=identity.purpose,
        contract_serialization_version=contract.serialization_version,
        source_contract_registries=contract.source_contract_registries,
        produced_outputs=contract.produced_outputs,
        future_orchestration_hooks=contract.future_orchestration_hooks,
    )


AGENT_ROLES = tuple(
    _role(index, contract.agent_id)
    for index, contract in enumerate(AGENT_CONTRACTS, start=1)
)
AGENT_ROLE_REGISTRY = AgentRoleRegistry(
    roles=AGENT_ROLES,
    role_ids=tuple(role.role_id for role in AGENT_ROLES),
    agent_ids=tuple(role.agent_id for role in AGENT_ROLES),
    role_count=len(AGENT_ROLES),
    role_families=tuple(dict.fromkeys(role.role_family for role in AGENT_ROLES)),
    capability_families=tuple(
        dict.fromkeys(role.capability_family for role in AGENT_ROLES)
    ),
)
