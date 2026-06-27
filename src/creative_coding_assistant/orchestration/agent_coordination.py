"""Passive V4.2 agent coordination contract metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_parallel_scheduling import (
    PARALLEL_SCHEDULING_GROUPS,
)

CoordinationEventType = Literal[
    "coordination_checkpoint_declared",
    "handoff_metadata_available",
    "coordination_risk_flagged",
    "human_review_signal_declared",
]

COORDINATION_RESPONSIBILITY_SERIALIZATION_VERSION = (
    "coordination_responsibility.v1"
)
COORDINATION_HANDOFF_CHANNEL_SERIALIZATION_VERSION = (
    "coordination_handoff_channel.v1"
)
COORDINATION_EVENT_SERIALIZATION_VERSION = "coordination_event.v1"
COORDINATION_REGISTRY_SERIALIZATION_VERSION = "coordination_registry.v1"
COORDINATION_REGISTRY_AUTHORITY_BOUNDARY = (
    "Agent coordination contracts describe coordinator responsibilities, "
    "handoff channels, coordination event metadata, and collaboration "
    "boundaries only; they do not implement live coordination, trigger agent "
    "actions, mutate outputs, schedule work, route providers or models, change "
    "workflow control, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "live_coordination",
    "agent_action_triggering",
    "output_mutation",
    "parallel_execution",
    "workflow_control",
    "provider_or_model_routing",
    "retry_or_refinement_triggering",
    "generated_output_modification",
)

_EVENT_TYPES: tuple[CoordinationEventType, ...] = (
    "coordination_checkpoint_declared",
    "handoff_metadata_available",
    "coordination_risk_flagged",
    "human_review_signal_declared",
)

_EVENT_PAYLOAD_FIELDS = (
    "event_type",
    "stage_id",
    "source_group_id",
    "target_group_id",
    "metadata_keys",
    "boundary_notes",
)


class CoordinationResponsibilityContract(BaseModel):
    """Passive coordinator responsibility metadata for one stage group."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    coordinator_id: str = Field(min_length=1, max_length=140)
    stage_id: str = Field(min_length=1, max_length=80)
    group_id: str = Field(min_length=1, max_length=120)
    responsible_agent_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    responsibilities: tuple[str, ...] = Field(min_length=1, max_length=8)
    required_inputs: tuple[str, ...] = Field(min_length=1, max_length=12)
    produced_handoff_channel_ids: tuple[str, ...] = Field(max_length=6)
    coordination_boundary: str = Field(min_length=1, max_length=600)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    live_coordination_implemented: Literal[False] = False
    agent_actions_triggered: Literal[False] = False
    output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["coordination_responsibility.v1"] = (
        COORDINATION_RESPONSIBILITY_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class CoordinationHandoffChannelContract(BaseModel):
    """Passive handoff channel metadata between scheduling groups."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    handoff_channel_id: str = Field(min_length=1, max_length=160)
    source_group_id: str = Field(min_length=1, max_length=120)
    target_group_id: str = Field(min_length=1, max_length=120)
    source_agent_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    target_agent_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    coordination_event_types: tuple[CoordinationEventType, ...] = Field(
        min_length=1,
        max_length=4,
    )
    payload_metadata_keys: tuple[str, ...] = Field(min_length=1, max_length=12)
    handoff_boundary: str = Field(min_length=1, max_length=600)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    live_coordination_implemented: Literal[False] = False
    agent_actions_triggered: Literal[False] = False
    output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["coordination_handoff_channel.v1"] = (
        COORDINATION_HANDOFF_CHANNEL_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class CoordinationEventContract(BaseModel):
    """Passive coordination event schema metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    event_type: CoordinationEventType
    event_order: int = Field(ge=1, le=8)
    payload_fields: tuple[str, ...] = Field(min_length=1, max_length=12)
    emitted_by_handoff_channel_ids: tuple[str, ...] = Field(max_length=8)
    event_boundary: str = Field(min_length=1, max_length=600)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    event_emission_implemented: Literal[False] = False
    agent_actions_triggered: Literal[False] = False
    output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["coordination_event.v1"] = (
        COORDINATION_EVENT_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentCoordinationRegistry(BaseModel):
    """Stable passive V4.2 agent coordination registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_coordination_registry"] = "agent_coordination_registry"
    serialization_version: Literal["coordination_registry.v1"] = (
        COORDINATION_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=COORDINATION_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    responsibilities: tuple[CoordinationResponsibilityContract, ...] = Field(
        min_length=6,
        max_length=6,
    )
    handoff_channels: tuple[CoordinationHandoffChannelContract, ...] = Field(
        min_length=5,
        max_length=5,
    )
    event_contracts: tuple[CoordinationEventContract, ...] = Field(
        min_length=4,
        max_length=4,
    )
    coordinator_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    handoff_channel_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    event_types: tuple[CoordinationEventType, ...] = Field(min_length=4, max_length=4)
    source_registries: tuple[str, ...] = Field(min_length=2, max_length=2)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    live_coordination_implemented: Literal[False] = False
    agent_actions_triggered: Literal[False] = False
    output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_contracts(self) -> Self:
        derived_coordinator_ids = tuple(
            responsibility.coordinator_id
            for responsibility in self.responsibilities
        )
        derived_channel_ids = tuple(
            channel.handoff_channel_id for channel in self.handoff_channels
        )
        derived_event_types = tuple(event.event_type for event in self.event_contracts)
        if self.coordinator_ids != derived_coordinator_ids:
            raise ValueError("coordinator_ids must match responsibilities")
        if self.handoff_channel_ids != derived_channel_ids:
            raise ValueError("handoff_channel_ids must match handoff_channels")
        if self.event_types != derived_event_types:
            raise ValueError("event_types must match event_contracts")
        if len(set(derived_coordinator_ids)) != len(derived_coordinator_ids):
            raise ValueError("coordinator_ids must be unique")
        if len(set(derived_channel_ids)) != len(derived_channel_ids):
            raise ValueError("handoff_channel_ids must be unique")
        if derived_event_types != _EVENT_TYPES:
            raise ValueError("event_contracts must preserve deterministic event order")
        group_ids = tuple(group.group_id for group in PARALLEL_SCHEDULING_GROUPS)
        group_index = {group_id: index for index, group_id in enumerate(group_ids)}
        for channel in self.handoff_channels:
            if channel.source_group_id not in group_index:
                raise ValueError("source_group_id must be a known scheduling group")
            if channel.target_group_id not in group_index:
                raise ValueError("target_group_id must be a known scheduling group")
            if group_index[channel.source_group_id] >= group_index[channel.target_group_id]:
                raise ValueError("handoff channels must move downstream")
        return self


def agent_coordination_registry() -> AgentCoordinationRegistry:
    """Return passive V4.2 coordination contract metadata."""

    return AGENT_COORDINATION_REGISTRY


def coordination_responsibility_by_id(
    coordinator_id: str,
    registry: AgentCoordinationRegistry | None = None,
) -> CoordinationResponsibilityContract | None:
    """Return one responsibility without live coordination."""

    source_registry = registry or AGENT_COORDINATION_REGISTRY
    for responsibility in source_registry.responsibilities:
        if responsibility.coordinator_id == coordinator_id:
            return responsibility
    return None


def coordination_handoff_channel_by_id(
    handoff_channel_id: str,
    registry: AgentCoordinationRegistry | None = None,
) -> CoordinationHandoffChannelContract | None:
    """Return one handoff channel without triggering agents."""

    source_registry = registry or AGENT_COORDINATION_REGISTRY
    for channel in source_registry.handoff_channels:
        if channel.handoff_channel_id == handoff_channel_id:
            return channel
    return None


def coordination_event_contract_by_type(
    event_type: CoordinationEventType,
    registry: AgentCoordinationRegistry | None = None,
) -> CoordinationEventContract | None:
    """Return one event contract without emitting coordination events."""

    source_registry = registry or AGENT_COORDINATION_REGISTRY
    for event_contract in source_registry.event_contracts:
        if event_contract.event_type == event_type:
            return event_contract
    return None


def _handoff_channel_id(source_group_id: str, target_group_id: str) -> str:
    return f"coordination_handoff::{source_group_id}->{target_group_id}"


def _responsibilities() -> tuple[CoordinationResponsibilityContract, ...]:
    produced_channel_ids_by_group = {
        group.group_id: tuple(
            _handoff_channel_id(group.group_id, downstream_group_id)
            for downstream_group_id in group.downstream_group_ids
        )
        for group in PARALLEL_SCHEDULING_GROUPS
    }
    return tuple(
        CoordinationResponsibilityContract(
            coordinator_id=f"coordinator::{group.stage_id}",
            stage_id=group.stage_id,
            group_id=group.group_id,
            responsible_agent_ids=group.agent_ids,
            responsibilities=(
                "collect_group_metadata",
                "declare_handoff_readiness",
                "surface_coordination_risks",
            ),
            required_inputs=(
                "parallel_scheduling_group_metadata",
                "agent_dependency_graph_metadata",
                "shared_context_view_metadata",
            ),
            produced_handoff_channel_ids=produced_channel_ids_by_group[group.group_id],
            coordination_boundary=(
                "Coordinator responsibility metadata is advisory only; it does "
                "not coordinate live agents, trigger actions, schedule work, "
                "or mutate outputs."
            ),
        )
        for group in PARALLEL_SCHEDULING_GROUPS
    )


def _handoff_channels() -> tuple[CoordinationHandoffChannelContract, ...]:
    group_by_id = {group.group_id: group for group in PARALLEL_SCHEDULING_GROUPS}
    channels: list[CoordinationHandoffChannelContract] = []
    for group in PARALLEL_SCHEDULING_GROUPS:
        for downstream_group_id in group.downstream_group_ids:
            downstream_group = group_by_id[downstream_group_id]
            channels.append(
                CoordinationHandoffChannelContract(
                    handoff_channel_id=_handoff_channel_id(
                        group.group_id,
                        downstream_group_id,
                    ),
                    source_group_id=group.group_id,
                    target_group_id=downstream_group_id,
                    source_agent_ids=group.agent_ids,
                    target_agent_ids=downstream_group.agent_ids,
                    coordination_event_types=_EVENT_TYPES,
                    payload_metadata_keys=(
                        "source_group_id",
                        "target_group_id",
                        "source_agent_ids",
                        "target_agent_ids",
                        "handoff_metadata_keys",
                    ),
                    handoff_boundary=(
                        "Handoff channel metadata names future coordination "
                        "payloads only; it does not emit events, invoke agents, "
                        "or mutate outputs."
                    ),
                )
            )
    return tuple(channels)


def _event_contracts(
    handoff_channels: tuple[CoordinationHandoffChannelContract, ...],
) -> tuple[CoordinationEventContract, ...]:
    handoff_channel_ids = tuple(channel.handoff_channel_id for channel in handoff_channels)
    return tuple(
        CoordinationEventContract(
            event_type=event_type,
            event_order=index,
            payload_fields=_EVENT_PAYLOAD_FIELDS,
            emitted_by_handoff_channel_ids=handoff_channel_ids,
            event_boundary=(
                "Coordination event metadata is a deterministic schema only; "
                "it does not emit runtime events, trigger agent actions, or "
                "mutate generated output."
            ),
        )
        for index, event_type in enumerate(_EVENT_TYPES, start=1)
    )


COORDINATION_HANDOFF_CHANNELS = _handoff_channels()
COORDINATION_RESPONSIBILITIES = _responsibilities()
COORDINATION_EVENT_CONTRACTS = _event_contracts(COORDINATION_HANDOFF_CHANNELS)
AGENT_COORDINATION_REGISTRY = AgentCoordinationRegistry(
    responsibilities=COORDINATION_RESPONSIBILITIES,
    handoff_channels=COORDINATION_HANDOFF_CHANNELS,
    event_contracts=COORDINATION_EVENT_CONTRACTS,
    coordinator_ids=tuple(
        responsibility.coordinator_id
        for responsibility in COORDINATION_RESPONSIBILITIES
    ),
    handoff_channel_ids=tuple(
        channel.handoff_channel_id for channel in COORDINATION_HANDOFF_CHANNELS
    ),
    event_types=tuple(event.event_type for event in COORDINATION_EVENT_CONTRACTS),
    source_registries=(
        "parallel_scheduling_registry",
        "agent_dependency_graph_registry",
    ),
)
