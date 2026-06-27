"""Passive V4.2 agent state synchronization metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_lifecycle import (
    AGENT_LIFECYCLE_REGISTRY,
    AgentLifecycleState,
)
from creative_coding_assistant.orchestration.shared_context_views import (
    SHARED_CONTEXT_VIEW_REGISTRY,
)

SyncCheckpointStage = Literal[
    "pre_activation",
    "active_context_review",
    "waiting_dependency_review",
    "completion_review",
    "post_terminal_review",
]
ConsistencyConstraintCategory = Literal[
    "lifecycle_context_alignment",
    "context_scope",
    "stale_state_warning",
    "conflict_surface_declaration",
]
StaleStateWarningCategory = Literal[
    "dependency_wait",
    "active_context_age",
    "blocked_conflict",
    "terminal_late_change",
]
ConflictSurfaceCategory = Literal[
    "lifecycle_context",
    "blackboard_visibility",
    "checkpoint_order",
    "stale_warning",
]

STATE_SYNC_CHECKPOINT_SERIALIZATION_VERSION = "agent_state_sync_checkpoint.v1"
STATE_SYNC_CONSTRAINT_SERIALIZATION_VERSION = "agent_state_sync_constraint.v1"
STATE_SYNC_STALE_WARNING_SERIALIZATION_VERSION = "agent_state_stale_warning.v1"
STATE_SYNC_CONFLICT_SURFACE_SERIALIZATION_VERSION = (
    "agent_state_conflict_surface.v1"
)
STATE_SYNC_PROFILE_SERIALIZATION_VERSION = "agent_state_sync_profile.v1"
STATE_SYNC_REGISTRY_SERIALIZATION_VERSION = "agent_state_sync_registry.v1"
STATE_SYNC_REGISTRY_AUTHORITY_BOUNDARY = (
    "Agent state synchronization metadata describes sync checkpoints, "
    "consistency constraints, stale-state warnings, and conflict surfaces "
    "between passive lifecycle profiles and shared context views only; it "
    "does not synchronize runtime state, mutate blackboard state, persist "
    "storage changes, resolve conflicts, invoke agents, control workflows, "
    "route providers or models, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "runtime_state_synchronization",
    "blackboard_mutation",
    "storage_mutation",
    "conflict_resolution",
    "stale_state_detection_execution",
    "agent_invocation",
    "workflow_control",
    "provider_or_model_routing",
    "generated_output_modification",
)

_CHECKPOINT_SPECS: tuple[
    tuple[
        str,
        SyncCheckpointStage,
        tuple[AgentLifecycleState, ...],
        tuple[str, ...],
    ],
    ...,
] = (
    (
        "state_sync_pre_activation_checkpoint",
        "pre_activation",
        ("planned",),
        ("lifecycle_state_context_view_alignment", "visible_context_scope_constraint"),
    ),
    (
        "state_sync_active_context_checkpoint",
        "active_context_review",
        ("active",),
        ("lifecycle_state_context_view_alignment", "stale_warning_declaration"),
    ),
    (
        "state_sync_waiting_dependency_checkpoint",
        "waiting_dependency_review",
        ("waiting",),
        ("stale_warning_declaration", "conflict_surface_declaration"),
    ),
    (
        "state_sync_completion_checkpoint",
        "completion_review",
        ("completed", "skipped"),
        ("lifecycle_state_context_view_alignment", "visible_context_scope_constraint"),
    ),
    (
        "state_sync_post_terminal_checkpoint",
        "post_terminal_review",
        ("failed", "blocked", "reviewed"),
        ("conflict_surface_declaration", "stale_warning_declaration"),
    ),
)

_CONSTRAINT_SPECS: tuple[
    tuple[str, ConsistencyConstraintCategory, tuple[str, ...]],
    ...,
] = (
    (
        "lifecycle_state_context_view_alignment",
        "lifecycle_context_alignment",
        ("lifecycle_profile_id", "context_view_id", "allowed_lifecycle_states"),
    ),
    (
        "visible_context_scope_constraint",
        "context_scope",
        ("visible_blackboard_channel_ids", "hidden_blackboard_channel_ids"),
    ),
    (
        "stale_warning_declaration",
        "stale_state_warning",
        ("stale_warning_ids", "affected_lifecycle_states"),
    ),
    (
        "conflict_surface_declaration",
        "conflict_surface_declaration",
        ("conflict_surface_ids", "source_registry_ids"),
    ),
)

_STALE_WARNING_SPECS: tuple[
    tuple[
        str,
        StaleStateWarningCategory,
        tuple[AgentLifecycleState, ...],
        tuple[str, ...],
    ],
    ...,
] = (
    (
        "waiting_dependency_stale_warning",
        "dependency_wait",
        ("waiting",),
        ("upstream_dependency_pending", "blocking_inputs"),
    ),
    (
        "active_context_age_warning",
        "active_context_age",
        ("active",),
        ("visible_metadata_keys", "context_review_age"),
    ),
    (
        "blocked_conflict_stale_warning",
        "blocked_conflict",
        ("blocked",),
        ("conflict_surface_ids", "unresolved_risks"),
    ),
    (
        "terminal_late_change_warning",
        "terminal_late_change",
        ("completed", "skipped", "failed", "reviewed"),
        ("terminal_state", "late_context_change"),
    ),
)

_CONFLICT_SURFACE_SPECS: tuple[
    tuple[str, ConflictSurfaceCategory, tuple[str, ...]],
    ...,
] = (
    (
        "lifecycle_context_conflict_surface",
        "lifecycle_context",
        ("agent_lifecycle_registry", "shared_context_view_registry"),
    ),
    (
        "blackboard_visibility_conflict_surface",
        "blackboard_visibility",
        ("shared_context_view_registry", "blackboard_memory_registry"),
    ),
    (
        "checkpoint_order_conflict_surface",
        "checkpoint_order",
        ("agent_lifecycle_registry", "agent_dependency_graph_registry"),
    ),
    (
        "stale_warning_conflict_surface",
        "stale_warning",
        ("agent_state_synchronization_registry", "agent_escalation_signal_registry"),
    ),
)


class AgentStateSyncCheckpoint(BaseModel):
    """Static checkpoint metadata for future state/context synchronization."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    checkpoint_id: str = Field(min_length=1, max_length=140)
    checkpoint_stage: SyncCheckpointStage
    applicable_lifecycle_states: tuple[AgentLifecycleState, ...] = Field(
        min_length=1,
        max_length=4,
    )
    consistency_constraint_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    checkpoint_boundary: str = Field(min_length=1, max_length=700)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    runtime_synchronization_implemented: Literal[False] = False
    blackboard_mutation_implemented: Literal[False] = False
    conflict_resolution_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    serialization_version: Literal["agent_state_sync_checkpoint.v1"] = (
        STATE_SYNC_CHECKPOINT_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentStateConsistencyConstraint(BaseModel):
    """Passive consistency constraint metadata for state/context alignment."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    constraint_id: str = Field(min_length=1, max_length=140)
    category: ConsistencyConstraintCategory
    compared_metadata_fields: tuple[str, ...] = Field(min_length=1, max_length=8)
    constraint_boundary: str = Field(min_length=1, max_length=700)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    runtime_synchronization_implemented: Literal[False] = False
    blackboard_mutation_implemented: Literal[False] = False
    conflict_resolution_implemented: Literal[False] = False
    serialization_version: Literal["agent_state_sync_constraint.v1"] = (
        STATE_SYNC_CONSTRAINT_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentStateStaleWarning(BaseModel):
    """Advisory stale-state warning metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    warning_id: str = Field(min_length=1, max_length=140)
    category: StaleStateWarningCategory
    affected_lifecycle_states: tuple[AgentLifecycleState, ...] = Field(
        min_length=1,
        max_length=4,
    )
    advisory_evidence_keys: tuple[str, ...] = Field(min_length=1, max_length=8)
    warning_boundary: str = Field(min_length=1, max_length=700)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    stale_state_detection_implemented: Literal[False] = False
    runtime_synchronization_implemented: Literal[False] = False
    blackboard_mutation_implemented: Literal[False] = False
    serialization_version: Literal["agent_state_stale_warning.v1"] = (
        STATE_SYNC_STALE_WARNING_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentStateConflictSurface(BaseModel):
    """Passive conflict surface metadata without resolution behavior."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    conflict_surface_id: str = Field(min_length=1, max_length=140)
    category: ConflictSurfaceCategory
    source_registry_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    conflict_boundary: str = Field(min_length=1, max_length=700)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    conflict_detection_implemented: Literal[False] = False
    conflict_resolution_implemented: Literal[False] = False
    blackboard_mutation_implemented: Literal[False] = False
    serialization_version: Literal["agent_state_conflict_surface.v1"] = (
        STATE_SYNC_CONFLICT_SURFACE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentStateSyncProfile(BaseModel):
    """Per-agent passive state/context synchronization profile."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=80)
    sync_profile_id: str = Field(min_length=1, max_length=140)
    source_lifecycle_profile_id: str = Field(min_length=1, max_length=140)
    source_context_view_id: str = Field(min_length=1, max_length=140)
    visible_blackboard_channel_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    sync_checkpoint_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    consistency_constraint_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    stale_warning_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    conflict_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    sync_boundary: str = Field(min_length=1, max_length=800)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    runtime_synchronization_implemented: Literal[False] = False
    blackboard_mutation_implemented: Literal[False] = False
    conflict_resolution_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    serialization_version: Literal["agent_state_sync_profile.v1"] = (
        STATE_SYNC_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class AgentStateSynchronizationRegistry(BaseModel):
    """Stable passive registry for agent state synchronization metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["agent_state_synchronization_registry"] = (
        "agent_state_synchronization_registry"
    )
    serialization_version: Literal["agent_state_sync_registry.v1"] = (
        STATE_SYNC_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=STATE_SYNC_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    profiles: tuple[AgentStateSyncProfile, ...] = Field(min_length=12, max_length=12)
    checkpoints: tuple[AgentStateSyncCheckpoint, ...] = Field(
        min_length=5,
        max_length=5,
    )
    constraints: tuple[AgentStateConsistencyConstraint, ...] = Field(
        min_length=4,
        max_length=4,
    )
    stale_warnings: tuple[AgentStateStaleWarning, ...] = Field(
        min_length=4,
        max_length=4,
    )
    conflict_surfaces: tuple[AgentStateConflictSurface, ...] = Field(
        min_length=4,
        max_length=4,
    )
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    profile_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    checkpoint_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    constraint_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    stale_warning_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    conflict_surface_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    profile_count: int = Field(ge=12, le=12)
    source_registries: tuple[str, ...] = Field(min_length=3, max_length=5)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    runtime_synchronization_implemented: Literal[False] = False
    blackboard_mutation_implemented: Literal[False] = False
    conflict_resolution_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_sync_metadata(self) -> Self:
        derived_agent_ids = tuple(profile.agent_id for profile in self.profiles)
        derived_profile_ids = tuple(profile.sync_profile_id for profile in self.profiles)
        derived_checkpoint_ids = tuple(
            checkpoint.checkpoint_id for checkpoint in self.checkpoints
        )
        derived_constraint_ids = tuple(
            constraint.constraint_id for constraint in self.constraints
        )
        derived_stale_warning_ids = tuple(
            warning.warning_id for warning in self.stale_warnings
        )
        derived_conflict_surface_ids = tuple(
            surface.conflict_surface_id for surface in self.conflict_surfaces
        )
        if self.agent_ids != derived_agent_ids:
            raise ValueError("agent_ids must match profiles")
        if self.profile_ids != derived_profile_ids:
            raise ValueError("profile_ids must match profiles")
        if self.checkpoint_ids != derived_checkpoint_ids:
            raise ValueError("checkpoint_ids must match checkpoints")
        if self.constraint_ids != derived_constraint_ids:
            raise ValueError("constraint_ids must match constraints")
        if self.stale_warning_ids != derived_stale_warning_ids:
            raise ValueError("stale_warning_ids must match stale warnings")
        if self.conflict_surface_ids != derived_conflict_surface_ids:
            raise ValueError("conflict_surface_ids must match conflict surfaces")
        if self.profile_count != len(self.profiles):
            raise ValueError("profile_count must match profiles")

        checkpoint_ids = set(self.checkpoint_ids)
        constraint_ids = set(self.constraint_ids)
        stale_warning_ids = set(self.stale_warning_ids)
        conflict_surface_ids = set(self.conflict_surface_ids)
        for checkpoint in self.checkpoints:
            if not set(checkpoint.consistency_constraint_ids).issubset(
                constraint_ids
            ):
                raise ValueError("checkpoint constraints must be known constraints")
        for profile in self.profiles:
            if not set(profile.sync_checkpoint_ids).issubset(checkpoint_ids):
                raise ValueError("profile checkpoints must be known checkpoints")
            if not set(profile.consistency_constraint_ids).issubset(constraint_ids):
                raise ValueError("profile constraints must be known constraints")
            if not set(profile.stale_warning_ids).issubset(stale_warning_ids):
                raise ValueError("profile stale warnings must be known warnings")
            if not set(profile.conflict_surface_ids).issubset(conflict_surface_ids):
                raise ValueError("profile conflict surfaces must be known surfaces")
        return self


def agent_state_synchronization_registry() -> AgentStateSynchronizationRegistry:
    """Return passive V4.2 state synchronization metadata."""

    return AGENT_STATE_SYNCHRONIZATION_REGISTRY


def agent_state_sync_profile_by_agent_id(
    agent_id: str,
    registry: AgentStateSynchronizationRegistry | None = None,
) -> AgentStateSyncProfile | None:
    """Return one sync profile without synchronizing runtime state."""

    source_registry = registry or AGENT_STATE_SYNCHRONIZATION_REGISTRY
    for profile in source_registry.profiles:
        if profile.agent_id == agent_id:
            return profile
    return None


def agent_state_sync_checkpoint_by_id(
    checkpoint_id: str,
    registry: AgentStateSynchronizationRegistry | None = None,
) -> AgentStateSyncCheckpoint | None:
    """Return one checkpoint without running synchronization."""

    source_registry = registry or AGENT_STATE_SYNCHRONIZATION_REGISTRY
    for checkpoint in source_registry.checkpoints:
        if checkpoint.checkpoint_id == checkpoint_id:
            return checkpoint
    return None


def agent_state_conflict_surface_by_id(
    conflict_surface_id: str,
    registry: AgentStateSynchronizationRegistry | None = None,
) -> AgentStateConflictSurface | None:
    """Return one conflict surface without resolving conflicts."""

    source_registry = registry or AGENT_STATE_SYNCHRONIZATION_REGISTRY
    for conflict_surface in source_registry.conflict_surfaces:
        if conflict_surface.conflict_surface_id == conflict_surface_id:
            return conflict_surface
    return None


def _checkpoint(
    spec: tuple[
        str,
        SyncCheckpointStage,
        tuple[AgentLifecycleState, ...],
        tuple[str, ...],
    ],
) -> AgentStateSyncCheckpoint:
    checkpoint_id, stage, states, constraints = spec
    return AgentStateSyncCheckpoint(
        checkpoint_id=checkpoint_id,
        checkpoint_stage=stage,
        applicable_lifecycle_states=states,
        consistency_constraint_ids=constraints,
        checkpoint_boundary=(
            "State synchronization checkpoints are declarative metadata only; "
            "they do not synchronize runtime state, mutate blackboard state, "
            "persist storage changes, or resolve conflicts."
        ),
    )


def _constraint(
    spec: tuple[str, ConsistencyConstraintCategory, tuple[str, ...]],
) -> AgentStateConsistencyConstraint:
    constraint_id, category, fields = spec
    return AgentStateConsistencyConstraint(
        constraint_id=constraint_id,
        category=category,
        compared_metadata_fields=fields,
        constraint_boundary=(
            "Consistency constraints describe future comparison surfaces only; "
            "they do not inspect runtime state, update shared context, mutate "
            "blackboard channels, or resolve conflicts."
        ),
    )


def _stale_warning(
    spec: tuple[
        str,
        StaleStateWarningCategory,
        tuple[AgentLifecycleState, ...],
        tuple[str, ...],
    ],
) -> AgentStateStaleWarning:
    warning_id, category, states, evidence_keys = spec
    return AgentStateStaleWarning(
        warning_id=warning_id,
        category=category,
        affected_lifecycle_states=states,
        advisory_evidence_keys=evidence_keys,
        warning_boundary=(
            "Stale-state warnings are advisory metadata only; they do not run "
            "stale-state detection, synchronize state, mutate shared context, "
            "or change workflow control."
        ),
    )


def _conflict_surface(
    spec: tuple[str, ConflictSurfaceCategory, tuple[str, ...]],
) -> AgentStateConflictSurface:
    surface_id, category, source_registries = spec
    return AgentStateConflictSurface(
        conflict_surface_id=surface_id,
        category=category,
        source_registry_ids=source_registries,
        conflict_boundary=(
            "Conflict surfaces identify possible future inconsistencies only; "
            "they do not detect conflicts, resolve conflicts, mutate "
            "blackboard state, or change workflow control."
        ),
    )


STATE_SYNC_CHECKPOINTS = tuple(_checkpoint(spec) for spec in _CHECKPOINT_SPECS)
STATE_SYNC_CONSTRAINTS = tuple(_constraint(spec) for spec in _CONSTRAINT_SPECS)
STATE_SYNC_STALE_WARNINGS = tuple(_stale_warning(spec) for spec in _STALE_WARNING_SPECS)
STATE_SYNC_CONFLICT_SURFACES = tuple(
    _conflict_surface(spec) for spec in _CONFLICT_SURFACE_SPECS
)
_CHECKPOINT_IDS = tuple(checkpoint.checkpoint_id for checkpoint in STATE_SYNC_CHECKPOINTS)
_CONSTRAINT_IDS = tuple(constraint.constraint_id for constraint in STATE_SYNC_CONSTRAINTS)
_STALE_WARNING_IDS = tuple(warning.warning_id for warning in STATE_SYNC_STALE_WARNINGS)
_CONFLICT_SURFACE_IDS = tuple(
    surface.conflict_surface_id for surface in STATE_SYNC_CONFLICT_SURFACES
)


def _profile(agent_id: str) -> AgentStateSyncProfile:
    lifecycle_profile = next(
        profile
        for profile in AGENT_LIFECYCLE_REGISTRY.profiles
        if profile.agent_id == agent_id
    )
    context_view = next(
        view
        for view in SHARED_CONTEXT_VIEW_REGISTRY.views
        if view.agent_id == agent_id
    )
    return AgentStateSyncProfile(
        agent_id=agent_id,
        sync_profile_id=f"{agent_id}_state_sync_profile",
        source_lifecycle_profile_id=lifecycle_profile.lifecycle_profile_id,
        source_context_view_id=context_view.view_id,
        visible_blackboard_channel_ids=context_view.visible_blackboard_channel_ids,
        sync_checkpoint_ids=_CHECKPOINT_IDS,
        consistency_constraint_ids=_CONSTRAINT_IDS,
        stale_warning_ids=_STALE_WARNING_IDS,
        conflict_surface_ids=_CONFLICT_SURFACE_IDS,
        sync_boundary=(
            "Agent state synchronization profiles map lifecycle metadata to "
            "scoped shared context view metadata only; they do not synchronize "
            "runtime state, mutate blackboard state, persist storage changes, "
            "or resolve conflicts."
        ),
    )


AGENT_STATE_SYNC_PROFILES = tuple(
    _profile(profile.agent_id) for profile in AGENT_LIFECYCLE_REGISTRY.profiles
)
AGENT_STATE_SYNCHRONIZATION_REGISTRY = AgentStateSynchronizationRegistry(
    profiles=AGENT_STATE_SYNC_PROFILES,
    checkpoints=STATE_SYNC_CHECKPOINTS,
    constraints=STATE_SYNC_CONSTRAINTS,
    stale_warnings=STATE_SYNC_STALE_WARNINGS,
    conflict_surfaces=STATE_SYNC_CONFLICT_SURFACES,
    agent_ids=tuple(profile.agent_id for profile in AGENT_STATE_SYNC_PROFILES),
    profile_ids=tuple(
        profile.sync_profile_id for profile in AGENT_STATE_SYNC_PROFILES
    ),
    checkpoint_ids=_CHECKPOINT_IDS,
    constraint_ids=_CONSTRAINT_IDS,
    stale_warning_ids=_STALE_WARNING_IDS,
    conflict_surface_ids=_CONFLICT_SURFACE_IDS,
    profile_count=len(AGENT_STATE_SYNC_PROFILES),
    source_registries=(
        "agent_lifecycle_registry",
        "shared_context_view_registry",
        "blackboard_memory_registry",
    ),
)
