"""Passive V4.2 parallel scheduling metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_dependency_graph import (
    agent_dependency_graph_registry,
)

SchedulingHint = Literal[
    "parallel_after_upstream_dependencies",
    "single_agent_after_upstream_dependencies",
]

PARALLEL_SCHEDULING_GROUP_SERIALIZATION_VERSION = "parallel_scheduling_group.v1"
PARALLEL_SCHEDULING_REGISTRY_SERIALIZATION_VERSION = "parallel_scheduling_registry.v1"
PARALLEL_SCHEDULING_REGISTRY_AUTHORITY_BOUNDARY = (
    "Parallel scheduling metadata describes future concurrency groups, "
    "blocking relationships, scheduling hints, and safety flags only; it does "
    "not run tasks in parallel, alter async behavior, change workflow timing, "
    "schedule graph execution, invoke agents, route providers or models, or "
    "modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "parallel_task_execution",
    "async_behavior_change",
    "workflow_timing_change",
    "graph_scheduler_execution",
    "agent_invocation",
    "provider_or_model_routing",
    "retry_or_refinement_triggering",
    "generated_output_modification",
)

_SAFETY_FLAGS = (
    "requires_completed_upstream_metadata",
    "requires_scoped_context_view",
    "metadata_only_parallelism",
    "no_async_execution_hook",
)

_GROUPS_BY_STAGE: dict[str, tuple[str, ...]] = {
    "foundational_context": ("planner_agent", "research_agent"),
    "domain_context": (
        "style_agent",
        "art_direction_agent",
        "narrative_symbolic_agent",
    ),
    "execution_context": ("runtime_agent", "artifact_agent"),
    "quality_review": (
        "aesthetic_critic_agent",
        "creative_curator_agent",
        "critic_agent",
    ),
    "refinement_context": ("refiner_agent",),
    "final_synthesis": ("final_synthesizer_agent",),
}


class ParallelSchedulingGroup(BaseModel):
    """Passive future concurrency group metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    group_id: str = Field(min_length=1, max_length=120)
    stage_id: str = Field(min_length=1, max_length=80)
    agent_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    scheduling_hint: SchedulingHint
    max_parallel_agents: int = Field(ge=1, le=6)
    blocking_group_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    downstream_group_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    safety_flags: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_dependency_nodes: tuple[str, ...] = Field(min_length=1, max_length=18)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    parallel_execution_implemented: Literal[False] = False
    async_behavior_changed: Literal[False] = False
    workflow_timing_changed: Literal[False] = False
    scheduler_runtime_hook_implemented: Literal[False] = False
    serialization_version: Literal["parallel_scheduling_group.v1"] = (
        PARALLEL_SCHEDULING_GROUP_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class ParallelSchedulingRegistry(BaseModel):
    """Stable passive V4.2 parallel scheduling metadata registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["parallel_scheduling_registry"] = "parallel_scheduling_registry"
    serialization_version: Literal["parallel_scheduling_registry.v1"] = (
        PARALLEL_SCHEDULING_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PARALLEL_SCHEDULING_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    groups: tuple[ParallelSchedulingGroup, ...] = Field(min_length=6, max_length=6)
    group_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    group_count: int = Field(ge=6, le=6)
    source_dependency_graph: Literal["agent_dependency_graph_registry"] = (
        "agent_dependency_graph_registry"
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    parallel_execution_implemented: Literal[False] = False
    async_behavior_changed: Literal[False] = False
    workflow_timing_changed: Literal[False] = False
    scheduler_runtime_hook_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_groups(self) -> Self:
        derived_group_ids = tuple(group.group_id for group in self.groups)
        derived_agent_ids = tuple(
            agent_id for group in self.groups for agent_id in group.agent_ids
        )
        if len(set(derived_group_ids)) != len(derived_group_ids):
            raise ValueError("group_ids must be unique")
        if len(set(derived_agent_ids)) != len(derived_agent_ids):
            raise ValueError("agent_ids must be unique")
        if self.group_ids != derived_group_ids:
            raise ValueError("group_ids must match groups")
        if self.agent_ids != derived_agent_ids:
            raise ValueError("agent_ids must match groups")
        if self.group_count != len(self.groups):
            raise ValueError("group_count must match groups")

        group_index = {group.group_id: index for index, group in enumerate(self.groups)}
        known_group_ids = set(self.group_ids)
        for group in self.groups:
            for blocking_group_id in group.blocking_group_ids:
                if blocking_group_id not in known_group_ids:
                    raise ValueError("blocking_group_ids must be known groups")
                if group_index[blocking_group_id] >= group_index[group.group_id]:
                    raise ValueError("blocking relationships must be acyclic")
            for downstream_group_id in group.downstream_group_ids:
                if downstream_group_id not in known_group_ids:
                    raise ValueError("downstream_group_ids must be known groups")
                if group_index[downstream_group_id] <= group_index[group.group_id]:
                    raise ValueError("downstream relationships must be acyclic")
            if group.max_parallel_agents != len(group.agent_ids):
                raise ValueError("max_parallel_agents must match group size")
        return self


def parallel_scheduling_registry() -> ParallelSchedulingRegistry:
    """Return passive V4.2 parallel scheduling metadata."""

    return PARALLEL_SCHEDULING_REGISTRY


def parallel_scheduling_group_by_id(
    group_id: str,
    registry: ParallelSchedulingRegistry | None = None,
) -> ParallelSchedulingGroup | None:
    """Return one scheduling group without running work in parallel."""

    source_registry = registry or PARALLEL_SCHEDULING_REGISTRY
    for group in source_registry.groups:
        if group.group_id == group_id:
            return group
    return None


def parallel_scheduling_group_for_agent(
    agent_id: str,
    registry: ParallelSchedulingRegistry | None = None,
) -> ParallelSchedulingGroup | None:
    """Return the future scheduling group for an agent without scheduling."""

    source_registry = registry or PARALLEL_SCHEDULING_REGISTRY
    for group in source_registry.groups:
        if agent_id in group.agent_ids:
            return group
    return None


def _group_id(stage_id: str) -> str:
    return f"parallel_group::{stage_id}"


def _source_dependency_nodes(
    agent_ids: tuple[str, ...], stage_id: str
) -> tuple[str, ...]:
    return (
        f"stage::{stage_id}",
        *(f"context_view::{agent_id}" for agent_id in agent_ids),
        *(f"agent::{agent_id}" for agent_id in agent_ids),
    )


def _groups() -> tuple[ParallelSchedulingGroup, ...]:
    dependency_graph = agent_dependency_graph_registry()
    groups: list[ParallelSchedulingGroup] = []
    stage_order = dependency_graph.stage_order
    for index, stage_id in enumerate(stage_order):
        agent_ids = _GROUPS_BY_STAGE[stage_id]
        group_id = _group_id(stage_id)
        groups.append(
            ParallelSchedulingGroup(
                group_id=group_id,
                stage_id=stage_id,
                agent_ids=agent_ids,
                scheduling_hint=(
                    "parallel_after_upstream_dependencies"
                    if len(agent_ids) > 1
                    else "single_agent_after_upstream_dependencies"
                ),
                max_parallel_agents=len(agent_ids),
                blocking_group_ids=(
                    (_group_id(stage_order[index - 1]),) if index > 0 else ()
                ),
                downstream_group_ids=(
                    (_group_id(stage_order[index + 1]),)
                    if index < len(stage_order) - 1
                    else ()
                ),
                safety_flags=_SAFETY_FLAGS,
                source_dependency_nodes=_source_dependency_nodes(agent_ids, stage_id),
            )
        )
    return tuple(groups)


PARALLEL_SCHEDULING_GROUPS = _groups()
PARALLEL_SCHEDULING_REGISTRY = ParallelSchedulingRegistry(
    groups=PARALLEL_SCHEDULING_GROUPS,
    group_ids=tuple(group.group_id for group in PARALLEL_SCHEDULING_GROUPS),
    agent_ids=tuple(
        agent_id for group in PARALLEL_SCHEDULING_GROUPS for agent_id in group.agent_ids
    ),
    group_count=len(PARALLEL_SCHEDULING_GROUPS),
)
