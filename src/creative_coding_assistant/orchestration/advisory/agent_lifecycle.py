"""Passive V4.2 agent lifecycle metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_contracts import AGENT_CONTRACTS

AgentLifecycleState = Literal[
    "planned",
    "active",
    "waiting",
    "completed",
    "skipped",
    "failed",
    "blocked",
    "reviewed",
]

LIFECYCLE_PROFILE_SERIALIZATION_VERSION = "agent_lifecycle_profile.v1"
LIFECYCLE_TRANSITION_SERIALIZATION_VERSION = "agent_lifecycle_transition.v1"
LIFECYCLE_REGISTRY_SERIALIZATION_VERSION = "agent_lifecycle_registry.v1"
LIFECYCLE_REGISTRY_AUTHORITY_BOUNDARY = (
    "Agent lifecycle metadata describes planned, active, waiting, completed, "
    "skipped, failed, blocked, and reviewed states plus allowed transition "
    "definitions only; it does not implement a runtime lifecycle engine, run "
    "state transitions, change workflow state, invoke agents, route providers "
    "or models, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "runtime_lifecycle_engine",
    "state_transition_execution",
    "workflow_state_change",
    "agent_invocation",
    "provider_or_model_routing",
    "generated_output_modification",
)

_LIFECYCLE_STATES: tuple[AgentLifecycleState, ...] = (
    "planned",
    "active",
    "waiting",
    "completed",
    "skipped",
    "failed",
    "blocked",
    "reviewed",
)

_TRANSITION_SPECS: tuple[tuple[AgentLifecycleState, AgentLifecycleState, str], ...] = (
    ("planned", "active", "agent_metadata_ready"),
    ("active", "waiting", "upstream_dependency_pending"),
    ("waiting", "active", "upstream_dependency_metadata_available"),
    ("active", "completed", "agent_metadata_complete"),
    ("active", "skipped", "agent_metadata_not_required"),
    ("active", "failed", "agent_metadata_failed"),
    ("active", "blocked", "agent_metadata_blocked"),
    ("completed", "reviewed", "agent_metadata_reviewed"),
    ("failed", "reviewed", "failure_metadata_reviewed"),
    ("blocked", "reviewed", "blocker_metadata_reviewed"),
)


class AgentLifecycleTransition(BaseModel):
    """Static lifecycle transition metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    transition_id: str = Field(min_length=1, max_length=120)
    from_state: AgentLifecycleState
    to_state: AgentLifecycleState
    transition_event: str = Field(min_length=1, max_length=120)
    transition_boundary: str = Field(min_length=1, max_length=600)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    transition_execution_implemented: Literal[False] = False
    workflow_state_change_implemented: Literal[False] = False
    runtime_lifecycle_engine_implemented: Literal[False] = False
    serialization_version: Literal["agent_lifecycle_transition.v1"] = (
        LIFECYCLE_TRANSITION_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentLifecycleProfile(BaseModel):
    """Passive lifecycle metadata for one agent."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=80)
    lifecycle_profile_id: str = Field(min_length=1, max_length=140)
    initial_state: Literal["planned"] = "planned"
    allowed_states: tuple[AgentLifecycleState, ...] = Field(min_length=8, max_length=8)
    terminal_states: tuple[AgentLifecycleState, ...] = Field(min_length=5, max_length=5)
    transition_ids: tuple[str, ...] = Field(min_length=1, max_length=16)
    source_contract_registry: Literal["agent_contract_registry"] = (
        "agent_contract_registry"
    )
    lifecycle_boundary: str = Field(min_length=1, max_length=600)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    transition_execution_implemented: Literal[False] = False
    workflow_state_change_implemented: Literal[False] = False
    runtime_lifecycle_engine_implemented: Literal[False] = False
    serialization_version: Literal["agent_lifecycle_profile.v1"] = (
        LIFECYCLE_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentLifecycleRegistry(BaseModel):
    """Stable passive V4.2 agent lifecycle registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_lifecycle_registry"] = "agent_lifecycle_registry"
    serialization_version: Literal["agent_lifecycle_registry.v1"] = (
        LIFECYCLE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=LIFECYCLE_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    states: tuple[AgentLifecycleState, ...] = Field(min_length=8, max_length=8)
    transitions: tuple[AgentLifecycleTransition, ...] = Field(
        min_length=10,
        max_length=10,
    )
    profiles: tuple[AgentLifecycleProfile, ...] = Field(min_length=12, max_length=12)
    transition_ids: tuple[str, ...] = Field(min_length=10, max_length=10)
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    profile_count: int = Field(ge=12, le=12)
    source_registries: tuple[str, ...] = Field(min_length=1, max_length=3)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    transition_execution_implemented: Literal[False] = False
    workflow_state_change_implemented: Literal[False] = False
    runtime_lifecycle_engine_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_lifecycle_metadata(self) -> Self:
        if self.states != _LIFECYCLE_STATES:
            raise ValueError("states must match lifecycle state order")
        derived_transition_ids = tuple(
            transition.transition_id for transition in self.transitions
        )
        derived_agent_ids = tuple(profile.agent_id for profile in self.profiles)
        if self.transition_ids != derived_transition_ids:
            raise ValueError("transition_ids must match transitions")
        if self.agent_ids != derived_agent_ids:
            raise ValueError("agent_ids must match profiles")
        if self.profile_count != len(self.profiles):
            raise ValueError("profile_count must match profiles")
        transition_id_set = set(self.transition_ids)
        for profile in self.profiles:
            if profile.allowed_states != self.states:
                raise ValueError("profile allowed_states must match registry states")
            if not set(profile.transition_ids).issubset(transition_id_set):
                raise ValueError("profile transition_ids must be known transitions")
        return self


def agent_lifecycle_registry() -> AgentLifecycleRegistry:
    """Return passive V4.2 lifecycle metadata."""

    return AGENT_LIFECYCLE_REGISTRY


def agent_lifecycle_profile_by_agent_id(
    agent_id: str,
    registry: AgentLifecycleRegistry | None = None,
) -> AgentLifecycleProfile | None:
    """Return one lifecycle profile without running transitions."""

    source_registry = registry or AGENT_LIFECYCLE_REGISTRY
    for profile in source_registry.profiles:
        if profile.agent_id == agent_id:
            return profile
    return None


def agent_lifecycle_transition_by_id(
    transition_id: str,
    registry: AgentLifecycleRegistry | None = None,
) -> AgentLifecycleTransition | None:
    """Return one transition definition without changing state."""

    source_registry = registry or AGENT_LIFECYCLE_REGISTRY
    for transition in source_registry.transitions:
        if transition.transition_id == transition_id:
            return transition
    return None


def _transition(
    from_state: AgentLifecycleState,
    to_state: AgentLifecycleState,
    event: str,
) -> AgentLifecycleTransition:
    return AgentLifecycleTransition(
        transition_id=f"lifecycle::{from_state}->{to_state}",
        from_state=from_state,
        to_state=to_state,
        transition_event=event,
        transition_boundary=(
            "Lifecycle transition metadata is declarative only; it does not "
            "run transitions, change workflow state, invoke agents, or mutate "
            "generated output."
        ),
    )


LIFECYCLE_TRANSITIONS = tuple(
    _transition(from_state, to_state, event)
    for from_state, to_state, event in _TRANSITION_SPECS
)
_LIFECYCLE_TRANSITION_IDS = tuple(
    transition.transition_id for transition in LIFECYCLE_TRANSITIONS
)


def _profile(agent_id: str) -> AgentLifecycleProfile:
    return AgentLifecycleProfile(
        agent_id=agent_id,
        lifecycle_profile_id=f"{agent_id}_lifecycle_profile",
        allowed_states=_LIFECYCLE_STATES,
        terminal_states=("completed", "skipped", "failed", "blocked", "reviewed"),
        transition_ids=_LIFECYCLE_TRANSITION_IDS,
        lifecycle_boundary=(
            "Lifecycle profile metadata is passive only; it does not create "
            "agents, run state transitions, change workflow state, or execute "
            "orchestration."
        ),
    )


AGENT_LIFECYCLE_PROFILES = tuple(
    _profile(contract.agent_id) for contract in AGENT_CONTRACTS
)
AGENT_LIFECYCLE_REGISTRY = AgentLifecycleRegistry(
    states=_LIFECYCLE_STATES,
    transitions=LIFECYCLE_TRANSITIONS,
    profiles=AGENT_LIFECYCLE_PROFILES,
    transition_ids=_LIFECYCLE_TRANSITION_IDS,
    agent_ids=tuple(profile.agent_id for profile in AGENT_LIFECYCLE_PROFILES),
    profile_count=len(AGENT_LIFECYCLE_PROFILES),
    source_registries=("agent_contract_registry",),
)
