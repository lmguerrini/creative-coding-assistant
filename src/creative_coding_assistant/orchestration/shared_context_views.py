"""Passive V4.2 per-agent shared context view metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_memory_contracts import (
    AGENT_MEMORY_CONTRACTS,
)
from creative_coding_assistant.orchestration.blackboard_memory import (
    BLACKBOARD_MEMORY_CHANNELS,
)

SharedContextViewStage = Literal["v4_2_shared_context_view"]
SharedContextViewAccessMode = Literal["scoped_metadata_view"]

SHARED_CONTEXT_VIEW_SERIALIZATION_VERSION = "shared_context_view.v1"
SHARED_CONTEXT_VIEW_REGISTRY_SERIALIZATION_VERSION = (
    "shared_context_view_registry.v1"
)
SHARED_CONTEXT_VIEW_REGISTRY_AUTHORITY_BOUNDARY = (
    "Shared context view metadata describes per-agent scoped visibility over "
    "existing memory contract surfaces and planned blackboard channels only; "
    "it does not expose unrestricted global state, implement runtime memory, "
    "materialize shared context, mutate context, invoke agents, route "
    "providers or models, control workflows, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "unrestricted_global_state_access",
    "runtime_memory_materialization",
    "shared_context_mutation",
    "blackboard_state_reads",
    "blackboard_state_writes",
    "storage_backend_creation",
    "agent_invocation",
    "workflow_control",
    "provider_or_model_routing",
    "generated_output_modification",
)

_VIEW_BOUNDARY = (
    "This view is scoped metadata only. It names visible and hidden context "
    "surfaces for future orchestration but does not read runtime memory, expose "
    "unrestricted shared state, materialize context, or mutate context."
)

_VIEW_CHANNELS_BY_AGENT_ID: dict[str, tuple[str, ...]] = {
    "planner_agent": (
        "planner_agent_blackboard_channel",
        "research_agent_blackboard_channel",
        "artifact_agent_blackboard_channel",
        "critic_agent_blackboard_channel",
        "final_synthesizer_agent_blackboard_channel",
    ),
    "research_agent": (
        "planner_agent_blackboard_channel",
        "research_agent_blackboard_channel",
        "narrative_symbolic_agent_blackboard_channel",
        "final_synthesizer_agent_blackboard_channel",
    ),
    "style_agent": (
        "style_agent_blackboard_channel",
        "art_direction_agent_blackboard_channel",
        "aesthetic_critic_agent_blackboard_channel",
        "creative_curator_agent_blackboard_channel",
        "final_synthesizer_agent_blackboard_channel",
    ),
    "runtime_agent": (
        "planner_agent_blackboard_channel",
        "runtime_agent_blackboard_channel",
        "artifact_agent_blackboard_channel",
        "critic_agent_blackboard_channel",
        "final_synthesizer_agent_blackboard_channel",
    ),
    "artifact_agent": (
        "planner_agent_blackboard_channel",
        "runtime_agent_blackboard_channel",
        "artifact_agent_blackboard_channel",
        "refiner_agent_blackboard_channel",
        "final_synthesizer_agent_blackboard_channel",
    ),
    "art_direction_agent": (
        "style_agent_blackboard_channel",
        "art_direction_agent_blackboard_channel",
        "narrative_symbolic_agent_blackboard_channel",
        "creative_curator_agent_blackboard_channel",
        "final_synthesizer_agent_blackboard_channel",
    ),
    "aesthetic_critic_agent": (
        "style_agent_blackboard_channel",
        "art_direction_agent_blackboard_channel",
        "aesthetic_critic_agent_blackboard_channel",
        "critic_agent_blackboard_channel",
        "final_synthesizer_agent_blackboard_channel",
    ),
    "narrative_symbolic_agent": (
        "planner_agent_blackboard_channel",
        "style_agent_blackboard_channel",
        "art_direction_agent_blackboard_channel",
        "narrative_symbolic_agent_blackboard_channel",
        "final_synthesizer_agent_blackboard_channel",
    ),
    "creative_curator_agent": (
        "style_agent_blackboard_channel",
        "art_direction_agent_blackboard_channel",
        "aesthetic_critic_agent_blackboard_channel",
        "creative_curator_agent_blackboard_channel",
        "final_synthesizer_agent_blackboard_channel",
    ),
    "critic_agent": (
        "planner_agent_blackboard_channel",
        "artifact_agent_blackboard_channel",
        "aesthetic_critic_agent_blackboard_channel",
        "critic_agent_blackboard_channel",
        "refiner_agent_blackboard_channel",
        "final_synthesizer_agent_blackboard_channel",
    ),
    "refiner_agent": (
        "artifact_agent_blackboard_channel",
        "critic_agent_blackboard_channel",
        "refiner_agent_blackboard_channel",
        "final_synthesizer_agent_blackboard_channel",
    ),
    "final_synthesizer_agent": (
        "planner_agent_blackboard_channel",
        "research_agent_blackboard_channel",
        "artifact_agent_blackboard_channel",
        "critic_agent_blackboard_channel",
        "refiner_agent_blackboard_channel",
        "final_synthesizer_agent_blackboard_channel",
    ),
}


class SharedContextViewContract(BaseModel):
    """Metadata-only scoped shared context view for one passive agent."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=80)
    view_id: str = Field(min_length=1, max_length=140)
    view_stage: SharedContextViewStage = "v4_2_shared_context_view"
    access_mode: SharedContextViewAccessMode = "scoped_metadata_view"
    visible_memory_surfaces: tuple[str, ...] = Field(max_length=5)
    visible_blackboard_channel_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    hidden_blackboard_channel_ids: tuple[str, ...] = Field(max_length=12)
    visible_metadata_keys: tuple[str, ...] = Field(min_length=1, max_length=36)
    source_memory_contract_id: str = Field(min_length=1, max_length=120)
    source_blackboard_registry: Literal["blackboard_memory_registry"] = (
        "blackboard_memory_registry"
    )
    view_boundary: str = Field(default=_VIEW_BOUNDARY, min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    unrestricted_global_state_exposed: Literal[False] = False
    runtime_memory_implemented: Literal[False] = False
    context_materialization_implemented: Literal[False] = False
    context_mutation_implemented: Literal[False] = False
    storage_backend_implemented: Literal[False] = False
    serialization_version: Literal["shared_context_view.v1"] = (
        SHARED_CONTEXT_VIEW_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class SharedContextViewRegistry(BaseModel):
    """Stable registry of passive V4.2 scoped shared context views."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["shared_context_view_registry"] = "shared_context_view_registry"
    serialization_version: Literal["shared_context_view_registry.v1"] = (
        SHARED_CONTEXT_VIEW_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=SHARED_CONTEXT_VIEW_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    views: tuple[SharedContextViewContract, ...] = Field(
        min_length=12,
        max_length=12,
    )
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    view_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    blackboard_channel_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    view_count: int = Field(ge=12, le=12)
    source_registries: tuple[str, ...] = Field(min_length=2, max_length=2)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    unrestricted_global_state_exposed: Literal[False] = False
    runtime_memory_implemented: Literal[False] = False
    context_materialization_implemented: Literal[False] = False
    context_mutation_implemented: Literal[False] = False
    storage_backend_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_views(self) -> Self:
        derived_agent_ids = tuple(view.agent_id for view in self.views)
        derived_view_ids = tuple(view.view_id for view in self.views)
        if len(set(derived_agent_ids)) != len(derived_agent_ids):
            raise ValueError("agent_ids must be unique")
        if len(set(derived_view_ids)) != len(derived_view_ids):
            raise ValueError("view_ids must be unique")
        if self.agent_ids != derived_agent_ids:
            raise ValueError("agent_ids must match views")
        if self.view_ids != derived_view_ids:
            raise ValueError("view_ids must match views")
        if self.view_count != len(self.views):
            raise ValueError("view_count must match views")

        known_channels = set(self.blackboard_channel_ids)
        for view in self.views:
            visible = set(view.visible_blackboard_channel_ids)
            hidden = set(view.hidden_blackboard_channel_ids)
            if not visible.issubset(known_channels):
                raise ValueError("visible channels must be known blackboard channels")
            if visible == known_channels:
                raise ValueError("shared context views must not expose every channel")
            if hidden != known_channels - visible:
                raise ValueError("hidden channels must complement visible channels")
        return self


def shared_context_view_registry() -> SharedContextViewRegistry:
    """Return passive V4.2 shared context view metadata."""

    return SHARED_CONTEXT_VIEW_REGISTRY


def shared_context_view_by_agent_id(
    agent_id: str,
    registry: SharedContextViewRegistry | None = None,
) -> SharedContextViewContract | None:
    """Return one scoped view without materializing runtime context."""

    source_registry = registry or SHARED_CONTEXT_VIEW_REGISTRY
    for view in source_registry.views:
        if view.agent_id == agent_id:
            return view
    return None


def shared_context_view_by_id(
    view_id: str,
    registry: SharedContextViewRegistry | None = None,
) -> SharedContextViewContract | None:
    """Return one shared context view by id without exposing global state."""

    source_registry = registry or SHARED_CONTEXT_VIEW_REGISTRY
    for view in source_registry.views:
        if view.view_id == view_id:
            return view
    return None


def _visible_metadata_keys(channel_ids: tuple[str, ...]) -> tuple[str, ...]:
    channel_by_id = {
        channel.channel_id: channel for channel in BLACKBOARD_MEMORY_CHANNELS
    }
    keys: list[str] = []
    for channel_id in channel_ids:
        keys.extend(channel_by_id[channel_id].metadata_keys)
    return tuple(dict.fromkeys(keys))


def _view(memory_contract_id: str, agent_id: str) -> SharedContextViewContract:
    memory_contract = next(
        contract for contract in AGENT_MEMORY_CONTRACTS if contract.agent_id == agent_id
    )
    blackboard_channel_ids = tuple(
        channel.channel_id for channel in BLACKBOARD_MEMORY_CHANNELS
    )
    visible_channel_ids = _VIEW_CHANNELS_BY_AGENT_ID[agent_id]
    return SharedContextViewContract(
        agent_id=agent_id,
        view_id=f"{agent_id}_shared_context_view",
        visible_memory_surfaces=memory_contract.readable_memory_surfaces,
        visible_blackboard_channel_ids=visible_channel_ids,
        hidden_blackboard_channel_ids=tuple(
            channel_id
            for channel_id in blackboard_channel_ids
            if channel_id not in visible_channel_ids
        ),
        visible_metadata_keys=_visible_metadata_keys(visible_channel_ids),
        source_memory_contract_id=memory_contract_id,
    )


SHARED_CONTEXT_VIEWS = tuple(
    _view(memory_contract.memory_contract_id, memory_contract.agent_id)
    for memory_contract in AGENT_MEMORY_CONTRACTS
)
SHARED_CONTEXT_VIEW_REGISTRY = SharedContextViewRegistry(
    views=SHARED_CONTEXT_VIEWS,
    agent_ids=tuple(view.agent_id for view in SHARED_CONTEXT_VIEWS),
    view_ids=tuple(view.view_id for view in SHARED_CONTEXT_VIEWS),
    blackboard_channel_ids=tuple(
        channel.channel_id for channel in BLACKBOARD_MEMORY_CHANNELS
    ),
    view_count=len(SHARED_CONTEXT_VIEWS),
    source_registries=(
        "agent_memory_contract_registry",
        "blackboard_memory_registry",
    ),
)
