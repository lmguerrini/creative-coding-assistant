"""Passive V4.2 blackboard memory boundary contracts."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_identities import AGENT_IDENTITIES
from creative_coding_assistant.orchestration.agent_memory_contracts import (
    AGENT_MEMORY_CONTRACTS,
    AgentMemoryContract,
)

BlackboardMemoryStage = Literal["v4_2_blackboard_memory_contract"]
BlackboardAccessMode = Literal[
    "future_metadata_only",
    "metadata_reference_only",
]
BlackboardPersistenceMode = Literal["not_persisted"]
BlackboardStorageBoundary = Literal["no_storage_backend"]

BLACKBOARD_MEMORY_CHANNEL_SERIALIZATION_VERSION = "blackboard_memory_channel.v1"
BLACKBOARD_AGENT_PERMISSION_SERIALIZATION_VERSION = "blackboard_agent_permission.v1"
BLACKBOARD_MEMORY_REGISTRY_SERIALIZATION_VERSION = "blackboard_memory_registry.v1"
BLACKBOARD_MEMORY_REGISTRY_AUTHORITY_BOUNDARY = (
    "Blackboard memory contracts describe planned shared-context metadata "
    "channels, future agent permissions, persistence flags, storage boundaries, "
    "and non-implemented runtime access only; they do not implement "
    "persistence, create storage backends, read runtime blackboard state, write "
    "runtime blackboard state, mutate memory, change retrieval behavior, "
    "control workflows, invoke agents, route providers or models, or modify "
    "generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "blackboard_persistence",
    "runtime_blackboard_materialization",
    "runtime_blackboard_mutation",
    "storage_backend_creation",
    "database_schema_creation",
    "memory_record_writes",
    "retrieval_side_effects",
    "shared_context_view_materialization",
    "agent_invocation",
    "workflow_control",
    "provider_or_model_routing",
    "generated_output_modification",
)

_CHANNEL_BOUNDARY = (
    "This blackboard channel is a planned shared-context metadata contract "
    "only. It may be inspected for future handoff semantics but does not "
    "persist records, create storage, read runtime blackboard state, write "
    "runtime blackboard state, mutate memory, or materialize shared context."
)

_PERMISSION_BOUNDARY = (
    "This permission contract describes future metadata access only. It does "
    "not read or write runtime blackboard state, persist memory records, "
    "create storage, invoke agents, or alter workflow behavior."
)


class BlackboardMemoryChannelContract(BaseModel):
    """Metadata-only blackboard channel contract for future shared context."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    channel_id: str = Field(min_length=1, max_length=120)
    channel_name: str = Field(min_length=1, max_length=160)
    owner_agent_id: str = Field(min_length=1, max_length=80)
    owner_role_family: str = Field(min_length=1, max_length=80)
    memory_stage: BlackboardMemoryStage = "v4_2_blackboard_memory_contract"
    permitted_writer_agent_ids: tuple[str, ...] = Field(min_length=1, max_length=1)
    permitted_reader_agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    permitted_reference_agent_ids: tuple[str, ...] = Field(
        min_length=12,
        max_length=12,
    )
    metadata_keys: tuple[str, ...] = Field(min_length=1, max_length=12)
    persistence_mode: BlackboardPersistenceMode = "not_persisted"
    storage_boundary: BlackboardStorageBoundary = "no_storage_backend"
    persistence_policy: str = Field(min_length=1, max_length=260)
    authority_boundary: str = Field(
        default=_CHANNEL_BOUNDARY,
        min_length=1,
        max_length=900,
    )
    source_memory_contract_id: str = Field(min_length=1, max_length=120)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    persistence_implemented: Literal[False] = False
    storage_backend_implemented: Literal[False] = False
    database_schema_implemented: Literal[False] = False
    runtime_read_implemented: Literal[False] = False
    runtime_write_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    serialization_version: Literal["blackboard_memory_channel.v1"] = (
        BLACKBOARD_MEMORY_CHANNEL_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class BlackboardAgentPermissionContract(BaseModel):
    """Metadata-only future blackboard permissions for one passive agent."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=80)
    permission_id: str = Field(min_length=1, max_length=140)
    memory_stage: BlackboardMemoryStage = "v4_2_blackboard_memory_contract"
    read_access: BlackboardAccessMode = "future_metadata_only"
    write_access: BlackboardAccessMode = "future_metadata_only"
    reference_access: BlackboardAccessMode = "metadata_reference_only"
    readable_channel_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    writable_channel_ids: tuple[str, ...] = Field(min_length=1, max_length=1)
    referenceable_channel_ids: tuple[str, ...] = Field(
        min_length=12,
        max_length=12,
    )
    writable_metadata_keys: tuple[str, ...] = Field(min_length=1, max_length=12)
    permission_boundary: str = Field(
        default=_PERMISSION_BOUNDARY,
        min_length=1,
        max_length=900,
    )
    source_memory_contract_id: str = Field(min_length=1, max_length=120)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    reads_runtime_blackboard: Literal[False] = False
    writes_runtime_blackboard: Literal[False] = False
    persists_blackboard_records: Literal[False] = False
    creates_storage_backend: Literal[False] = False
    mutates_shared_context: Literal[False] = False
    serialization_version: Literal["blackboard_agent_permission.v1"] = (
        BLACKBOARD_AGENT_PERMISSION_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class BlackboardMemoryRegistry(BaseModel):
    """Stable registry for passive V4.2 blackboard memory contracts."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["blackboard_memory_registry"] = "blackboard_memory_registry"
    serialization_version: Literal["blackboard_memory_registry.v1"] = (
        BLACKBOARD_MEMORY_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=BLACKBOARD_MEMORY_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    channels: tuple[BlackboardMemoryChannelContract, ...] = Field(
        min_length=12,
        max_length=12,
    )
    permissions: tuple[BlackboardAgentPermissionContract, ...] = Field(
        min_length=12,
        max_length=12,
    )
    channel_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    channel_count: int = Field(ge=12, le=12)
    permission_count: int = Field(ge=12, le=12)
    source_registries: tuple[str, ...] = Field(min_length=2, max_length=2)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    persistence_implemented: Literal[False] = False
    storage_backend_implemented: Literal[False] = False
    database_schema_implemented: Literal[False] = False
    runtime_read_implemented: Literal[False] = False
    runtime_write_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_channels_and_permissions(self) -> Self:
        derived_channel_ids = tuple(channel.channel_id for channel in self.channels)
        derived_agent_ids = tuple(
            permission.agent_id for permission in self.permissions
        )
        if len(set(derived_channel_ids)) != len(derived_channel_ids):
            raise ValueError("channel_ids must be unique")
        if len(set(derived_agent_ids)) != len(derived_agent_ids):
            raise ValueError("agent_ids must be unique")
        if self.channel_ids != derived_channel_ids:
            raise ValueError("channel_ids must match channels")
        if self.agent_ids != derived_agent_ids:
            raise ValueError("agent_ids must match permissions")
        if self.channel_count != len(self.channels):
            raise ValueError("channel_count must match channels")
        if self.permission_count != len(self.permissions):
            raise ValueError("permission_count must match permissions")

        known_channels = set(self.channel_ids)
        channel_by_owner = {
            channel.owner_agent_id: channel for channel in self.channels
        }
        for permission in self.permissions:
            if tuple(permission.readable_channel_ids) != self.channel_ids:
                raise ValueError("readable_channel_ids must match registry channels")
            if tuple(permission.referenceable_channel_ids) != self.channel_ids:
                raise ValueError(
                    "referenceable_channel_ids must match registry channels"
                )
            if not set(permission.writable_channel_ids).issubset(known_channels):
                raise ValueError("writable_channel_ids must be known channels")
            owner_channel = channel_by_owner.get(permission.agent_id)
            if owner_channel is None:
                raise ValueError("each permission must match one owner channel")
            if permission.writable_channel_ids != (owner_channel.channel_id,):
                raise ValueError("writable_channel_ids must match owner channel")
            if permission.writable_metadata_keys != owner_channel.metadata_keys:
                raise ValueError("writable_metadata_keys must match owner channel")
        return self


def blackboard_memory_registry() -> BlackboardMemoryRegistry:
    """Return passive V4.2 blackboard memory contracts."""

    return BLACKBOARD_MEMORY_REGISTRY


def blackboard_channel_by_id(
    channel_id: str,
    registry: BlackboardMemoryRegistry | None = None,
) -> BlackboardMemoryChannelContract | None:
    """Return one channel contract without reading or writing memory."""

    source_registry = registry or BLACKBOARD_MEMORY_REGISTRY
    for channel in source_registry.channels:
        if channel.channel_id == channel_id:
            return channel
    return None


def blackboard_permissions_by_agent_id(
    agent_id: str,
    registry: BlackboardMemoryRegistry | None = None,
) -> BlackboardAgentPermissionContract | None:
    """Return one permission contract without materializing blackboard state."""

    source_registry = registry or BLACKBOARD_MEMORY_REGISTRY
    for permission in source_registry.permissions:
        if permission.agent_id == agent_id:
            return permission
    return None


def blackboard_channels_for_agent(
    agent_id: str,
    registry: BlackboardMemoryRegistry | None = None,
) -> tuple[BlackboardMemoryChannelContract, ...]:
    """Return channels the agent may reference as future metadata only."""

    permission = blackboard_permissions_by_agent_id(agent_id, registry)
    if permission is None:
        return ()
    source_registry = registry or BLACKBOARD_MEMORY_REGISTRY
    channel_by_id = {
        channel.channel_id: channel for channel in source_registry.channels
    }
    return tuple(
        channel_by_id[channel_id]
        for channel_id in permission.referenceable_channel_ids
        if channel_id in channel_by_id
    )


def _future_blackboard_metadata(
    memory_contract: AgentMemoryContract,
) -> tuple[str, ...]:
    future_blackboard_surface = next(
        surface
        for surface in memory_contract.surfaces
        if surface.surface == "future_blackboard"
    )
    return future_blackboard_surface.writable_metadata


def _channel_id(agent_id: str) -> str:
    return f"{agent_id}_blackboard_channel"


def _channel(memory_contract: AgentMemoryContract) -> BlackboardMemoryChannelContract:
    identity = next(
        identity
        for identity in AGENT_IDENTITIES
        if identity.agent_id == memory_contract.agent_id
    )
    agent_ids = tuple(contract.agent_id for contract in AGENT_MEMORY_CONTRACTS)
    return BlackboardMemoryChannelContract(
        channel_id=_channel_id(memory_contract.agent_id),
        channel_name=f"{identity.agent_name} Blackboard Metadata Channel",
        owner_agent_id=memory_contract.agent_id,
        owner_role_family=identity.role_family,
        permitted_writer_agent_ids=(memory_contract.agent_id,),
        permitted_reader_agent_ids=agent_ids,
        permitted_reference_agent_ids=agent_ids,
        metadata_keys=_future_blackboard_metadata(memory_contract),
        persistence_policy=(
            "Blackboard channel data is not persisted in V4.2; this contract "
            "only names future shared-context metadata keys."
        ),
        source_memory_contract_id=memory_contract.memory_contract_id,
    )


def _permission(
    memory_contract: AgentMemoryContract,
    channel_ids: tuple[str, ...],
) -> BlackboardAgentPermissionContract:
    return BlackboardAgentPermissionContract(
        agent_id=memory_contract.agent_id,
        permission_id=f"{memory_contract.agent_id}_blackboard_permission",
        readable_channel_ids=channel_ids,
        writable_channel_ids=(_channel_id(memory_contract.agent_id),),
        referenceable_channel_ids=channel_ids,
        writable_metadata_keys=_future_blackboard_metadata(memory_contract),
        source_memory_contract_id=memory_contract.memory_contract_id,
    )


BLACKBOARD_MEMORY_CHANNELS = tuple(
    _channel(memory_contract) for memory_contract in AGENT_MEMORY_CONTRACTS
)
_BLACKBOARD_CHANNEL_IDS = tuple(
    channel.channel_id for channel in BLACKBOARD_MEMORY_CHANNELS
)
BLACKBOARD_AGENT_PERMISSIONS = tuple(
    _permission(memory_contract, _BLACKBOARD_CHANNEL_IDS)
    for memory_contract in AGENT_MEMORY_CONTRACTS
)
BLACKBOARD_MEMORY_REGISTRY = BlackboardMemoryRegistry(
    channels=BLACKBOARD_MEMORY_CHANNELS,
    permissions=BLACKBOARD_AGENT_PERMISSIONS,
    channel_ids=_BLACKBOARD_CHANNEL_IDS,
    agent_ids=tuple(permission.agent_id for permission in BLACKBOARD_AGENT_PERMISSIONS),
    channel_count=len(BLACKBOARD_MEMORY_CHANNELS),
    permission_count=len(BLACKBOARD_AGENT_PERMISSIONS),
    source_registries=(
        "agent_memory_contract_registry",
        "agent_identity_registry",
    ),
)
