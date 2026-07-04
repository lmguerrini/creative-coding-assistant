"""V6.6 Cognitive Blackboard metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.blackboard_memory import (
    BlackboardMemoryRegistry,
    blackboard_memory_registry,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
    CognitiveOSCapability,
    CognitiveOSLayer,
    CognitiveOSPosture,
)
from creative_coding_assistant.orchestration.cognitive_router import (
    CognitiveRouterPlan,
    build_cognitive_router,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)

COGNITIVE_BLACKBOARD_SERIALIZATION_VERSION = "cognitive_blackboard.v1"
COGNITIVE_BLACKBOARD_ROADMAP_ITEM = "Cognitive Blackboard"
COGNITIVE_BLACKBOARD_AUTHORITY_BOUNDARY = (
    "V6.6 Cognitive Blackboard projects cognitive route decisions into "
    "read-only blackboard entries over passive blackboard channel contracts "
    "for route-to-blackboard continuity, capability context visibility, "
    "agent channel ownership, governance checkpoint memory, explanation "
    "continuity, and HITL readiness. It exposes blackboard metadata only; "
    "it does not read runtime blackboard state, write runtime blackboard "
    "state, persist records, materialize shared context, mutate memory, "
    "invoke agents, route requests, change prompts, retrieval, storage, "
    "generated output, runtime state, or apply Runtime Evolution."
)
COGNITIVE_BLACKBOARD_DIMENSIONS = (
    "route-to-blackboard continuity",
    "capability context visibility",
    "agent channel ownership",
    "governance checkpoint memory",
    "HITL blackboard boundary",
)


class CognitiveBlackboardEntry(BaseModel):
    """One read-only Cognitive OS blackboard entry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    blackboard_entry_id: str = Field(min_length=1, max_length=190)
    route_decision_id: str = Field(min_length=1, max_length=190)
    plan_id: str = Field(min_length=1, max_length=190)
    schedule_id: str = Field(min_length=1, max_length=190)
    emergence_id: str = Field(min_length=1, max_length=190)
    identity_id: str = Field(min_length=1, max_length=190)
    cognition_id: str = Field(min_length=1, max_length=190)
    governance_id: str = Field(min_length=1, max_length=190)
    planning_id: str = Field(min_length=1, max_length=170)
    reasoning_id: str = Field(min_length=1, max_length=170)
    profile_id: str = Field(min_length=1, max_length=170)
    state_id: str = Field(min_length=1, max_length=160)
    capability_id: str = Field(min_length=1, max_length=80)
    capability_name: CognitiveOSCapability
    cognitive_layer: CognitiveOSLayer
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    visible_blackboard_channel_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    visible_blackboard_channel_count: int = Field(ge=0, le=12)
    referenceable_blackboard_channel_ids: tuple[str, ...] = Field(
        min_length=12,
        max_length=12,
    )
    blackboard_rank: int = Field(ge=1, le=6)
    dependency_depth: int = Field(ge=0, le=5)
    blackboard_dimensions: tuple[str, ...] = Field(min_length=5, max_length=5)
    routing_posture: CognitiveOSPosture
    blackboard_posture: CognitiveOSPosture
    runtime_blackboard_access_authorized: Literal[False] = False
    blackboard_summary: str = Field(min_length=1, max_length=640)
    dependency_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _entry_matches_sources_and_boundary(self) -> Self:
        expected_entry_id = f"cognitive_blackboard::{self.capability_id}"
        if self.blackboard_entry_id != expected_entry_id:
            raise ValueError("blackboard_entry_id must match capability_id")
        expected_route_id = f"cognitive_router::{self.capability_id}"
        if self.route_decision_id != expected_route_id:
            raise ValueError("route_decision_id must match capability_id")
        expected_plan_id = f"cognitive_planner::{self.capability_id}"
        if self.plan_id != expected_plan_id:
            raise ValueError("plan_id must match capability_id")
        expected_schedule_id = f"cognitive_scheduler::{self.capability_id}"
        if self.schedule_id != expected_schedule_id:
            raise ValueError("schedule_id must match capability_id")
        expected_emergence_id = f"emergent_creativity::{self.capability_id}"
        if self.emergence_id != expected_emergence_id:
            raise ValueError("emergence_id must match capability_id")
        expected_identity_id = f"creative_identity::{self.capability_id}"
        if self.identity_id != expected_identity_id:
            raise ValueError("identity_id must match capability_id")
        expected_cognition_id = f"creative_cognition::{self.capability_id}"
        if self.cognition_id != expected_cognition_id:
            raise ValueError("cognition_id must match capability_id")
        expected_governance_id = f"cognitive_governance::{self.capability_id}"
        if self.governance_id != expected_governance_id:
            raise ValueError("governance_id must match capability_id")
        expected_planning_id = f"meta_planning::{self.capability_id}"
        if self.planning_id != expected_planning_id:
            raise ValueError("planning_id must match capability_id")
        expected_reasoning_id = f"meta_reasoning::{self.capability_id}"
        if self.reasoning_id != expected_reasoning_id:
            raise ValueError("reasoning_id must match capability_id")
        expected_profile_id = f"cognitive_profile::{self.capability_id}"
        if self.profile_id != expected_profile_id:
            raise ValueError("profile_id must match capability_id")
        expected_state_id = f"cognitive_state::{self.capability_id}"
        if self.state_id != expected_state_id:
            raise ValueError("state_id must match capability_id")
        if self.visible_blackboard_channel_count != len(
            self.visible_blackboard_channel_ids
        ):
            raise ValueError("visible_blackboard_channel_count must match channels")
        if self.blackboard_dimensions != COGNITIVE_BLACKBOARD_DIMENSIONS:
            raise ValueError("blackboard_dimensions must match V6.6 blackboard")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class CognitiveBlackboardPlan(BaseModel):
    """Read-only cognitive blackboard over route decisions."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cognitive_blackboard"] = "cognitive_blackboard"
    serialization_version: Literal["cognitive_blackboard.v1"] = (
        COGNITIVE_BLACKBOARD_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=COGNITIVE_BLACKBOARD_AUTHORITY_BOUNDARY,
        max_length=2300,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    cognitive_router_role: Literal["cognitive_router"]
    cognitive_router_serialization_version: Literal["cognitive_router.v1"]
    cognitive_planner_role: Literal["cognitive_planner"]
    cognitive_scheduler_role: Literal["cognitive_scheduler"]
    blackboard_memory_registry_role: Literal["blackboard_memory_registry"]
    blackboard_memory_registry_serialization_version: Literal[
        "blackboard_memory_registry.v1"
    ]
    source_blackboard_channel_ids: tuple[str, ...] = Field(
        min_length=12,
        max_length=12,
    )
    source_blackboard_channel_count: int = Field(ge=12, le=12)
    source_blackboard_permission_ids: tuple[str, ...] = Field(
        min_length=12,
        max_length=12,
    )
    source_blackboard_permission_count: int = Field(ge=12, le=12)
    layer_order: tuple[CognitiveOSLayer, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_count: int = Field(ge=6, le=6)
    source_route_decision_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_route_decision_count: int = Field(ge=6, le=6)
    source_plan_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_plan_count: int = Field(ge=6, le=6)
    source_schedule_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_schedule_count: int = Field(ge=6, le=6)
    source_emergence_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_emergence_count: int = Field(ge=6, le=6)
    blackboard_entries: tuple[CognitiveBlackboardEntry, ...] = Field(
        min_length=6,
        max_length=6,
    )
    blackboard_entry_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    candidate_blackboard_entry_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    review_required_blackboard_entry_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    guarded_blackboard_entry_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    blackboard_entry_count: int = Field(ge=6, le=6)
    candidate_blackboard_entry_count: int = Field(ge=0, le=6)
    review_required_blackboard_entry_count: int = Field(ge=0, le=6)
    guarded_blackboard_entry_count: int = Field(ge=0, le=6)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    cognitive_blackboard_implemented: Literal[True] = True
    cognitive_router_integrated: Literal[True] = True
    blackboard_memory_registry_integrated: Literal[True] = True
    blackboard_entry_contract_implemented: Literal[True] = True
    blackboard_dependency_traceability_implemented: Literal[True] = True
    blackboard_governance_contract_implemented: Literal[True] = True
    blackboard_explainability_contract_implemented: Literal[True] = True
    runtime_blackboard_read_implemented: Literal[False] = False
    runtime_blackboard_write_implemented: Literal[False] = False
    blackboard_persistence_implemented: Literal[False] = False
    blackboard_mutation_implemented: Literal[False] = False
    shared_context_materialization_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    routing_application_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    materialized_blackboard_entry_ids: tuple[str, ...] = Field(default_factory=tuple)
    read_blackboard_channel_ids: tuple[str, ...] = Field(default_factory=tuple)
    written_blackboard_channel_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _blackboard_matches_sources(self) -> Self:
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_count != len(self.capability_ids):
            raise ValueError("capability_count must match capability ids")
        if self.source_blackboard_channel_count != len(
            self.source_blackboard_channel_ids
        ):
            raise ValueError("source_blackboard_channel_count must match channels")
        if self.source_blackboard_permission_count != len(
            self.source_blackboard_permission_ids
        ):
            raise ValueError("source_blackboard_permission_count must match ids")
        if self.source_route_decision_count != len(self.source_route_decision_ids):
            raise ValueError("source_route_decision_count must match route ids")
        if self.source_plan_count != len(self.source_plan_ids):
            raise ValueError("source_plan_count must match plan ids")
        if self.source_schedule_count != len(self.source_schedule_ids):
            raise ValueError("source_schedule_count must match schedule ids")
        if self.source_emergence_count != len(self.source_emergence_ids):
            raise ValueError("source_emergence_count must match emergence ids")
        if self.blackboard_entry_ids != tuple(
            entry.blackboard_entry_id for entry in self.blackboard_entries
        ):
            raise ValueError("blackboard_entry_ids must match entries")
        if self.blackboard_entry_count != len(self.blackboard_entries):
            raise ValueError("blackboard_entry_count must match entries")
        if len(set(self.blackboard_entry_ids)) != len(self.blackboard_entry_ids):
            raise ValueError("blackboard_entry_ids must be unique")
        if self.candidate_blackboard_entry_ids != _blackboard_ids_for_posture(
            self.blackboard_entries,
            "candidate",
        ):
            raise ValueError("candidate_blackboard_entry_ids must match entries")
        if self.review_required_blackboard_entry_ids != _blackboard_ids_for_posture(
            self.blackboard_entries,
            "review_required",
        ):
            raise ValueError("review_required_blackboard_entry_ids must match entries")
        if self.guarded_blackboard_entry_ids != _blackboard_ids_for_posture(
            self.blackboard_entries,
            "guarded",
        ):
            raise ValueError("guarded_blackboard_entry_ids must match entries")
        if self.candidate_blackboard_entry_count != len(
            self.candidate_blackboard_entry_ids
        ):
            raise ValueError("candidate_blackboard_entry_count must match ids")
        if self.review_required_blackboard_entry_count != len(
            self.review_required_blackboard_entry_ids
        ):
            raise ValueError("review_required_blackboard_entry_count must match ids")
        if self.guarded_blackboard_entry_count != len(
            self.guarded_blackboard_entry_ids
        ):
            raise ValueError("guarded_blackboard_entry_count must match ids")

        declared_capabilities = set(self.capability_ids)
        declared_route_decisions = set(self.source_route_decision_ids)
        declared_plans = set(self.source_plan_ids)
        declared_schedules = set(self.source_schedule_ids)
        declared_emergence = set(self.source_emergence_ids)
        declared_agents = set(self.linked_agent_ids)
        declared_channels = set(self.source_blackboard_channel_ids)
        for entry in self.blackboard_entries:
            if entry.capability_id not in declared_capabilities:
                raise ValueError("entry capability_id must be declared")
            if entry.route_decision_id not in declared_route_decisions:
                raise ValueError("entry route_decision_id must be declared")
            if entry.plan_id not in declared_plans:
                raise ValueError("entry plan_id must be declared")
            if entry.schedule_id not in declared_schedules:
                raise ValueError("entry schedule_id must be declared")
            if entry.emergence_id not in declared_emergence:
                raise ValueError("entry emergence_id must be declared")
            if not set(entry.linked_agent_ids).issubset(declared_agents):
                raise ValueError("entry linked_agent_ids must be declared")
            if not set(entry.visible_blackboard_channel_ids).issubset(
                declared_channels
            ):
                raise ValueError("entry visible channels must be declared")
            if entry.referenceable_blackboard_channel_ids != (
                self.source_blackboard_channel_ids
            ):
                raise ValueError("entry referenceable channels must match registry")
        if self.covered_roadmap_items != (COGNITIVE_BLACKBOARD_ROADMAP_ITEM,):
            raise ValueError("covered_roadmap_items must be Task 20 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.materialized_blackboard_entry_ids,
                self.read_blackboard_channel_ids,
                self.written_blackboard_channel_ids,
                self.emitted_hitl_request_ids,
            )
        ):
            raise ValueError(
                "blackboard materialization, reads, writes, and HITL ids must be empty",
            )
        if not all(entry.advisory_only for entry in self.blackboard_entries):
            raise ValueError("all cognitive blackboard entries must be advisory only")
        return self


def build_cognitive_blackboard(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    cognitive_router: CognitiveRouterPlan | None = None,
    blackboard_registry: BlackboardMemoryRegistry | None = None,
) -> CognitiveBlackboardPlan:
    """Build read-only cognitive blackboard metadata."""

    router = cognitive_router or build_cognitive_router(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    registry = blackboard_registry or blackboard_memory_registry()
    entries = _cognitive_blackboard_entries(router, registry)
    return CognitiveBlackboardPlan(
        route_name=router.route_name,
        task_type=router.task_type,
        execution_mode_ids=router.execution_mode_ids,
        cognitive_router_role=router.role,
        cognitive_router_serialization_version=router.serialization_version,
        cognitive_planner_role=router.cognitive_planner_role,
        cognitive_scheduler_role=router.cognitive_scheduler_role,
        blackboard_memory_registry_role=registry.role,
        blackboard_memory_registry_serialization_version=registry.serialization_version,
        source_blackboard_channel_ids=registry.channel_ids,
        source_blackboard_channel_count=registry.channel_count,
        source_blackboard_permission_ids=tuple(
            permission.permission_id for permission in registry.permissions
        ),
        source_blackboard_permission_count=registry.permission_count,
        layer_order=router.layer_order,
        capabilities=router.capabilities,
        capability_ids=router.capability_ids,
        capability_count=router.capability_count,
        source_route_decision_ids=router.route_decision_ids,
        source_route_decision_count=router.route_decision_count,
        source_plan_ids=router.source_plan_ids,
        source_plan_count=router.source_plan_count,
        source_schedule_ids=router.source_schedule_ids,
        source_schedule_count=router.source_schedule_count,
        source_emergence_ids=router.source_emergence_ids,
        source_emergence_count=router.source_emergence_count,
        blackboard_entries=entries,
        blackboard_entry_ids=tuple(entry.blackboard_entry_id for entry in entries),
        candidate_blackboard_entry_ids=_blackboard_ids_for_posture(
            entries,
            "candidate",
        ),
        review_required_blackboard_entry_ids=_blackboard_ids_for_posture(
            entries,
            "review_required",
        ),
        guarded_blackboard_entry_ids=_blackboard_ids_for_posture(
            entries,
            "guarded",
        ),
        blackboard_entry_count=len(entries),
        candidate_blackboard_entry_count=len(
            _blackboard_ids_for_posture(entries, "candidate")
        ),
        review_required_blackboard_entry_count=len(
            _blackboard_ids_for_posture(entries, "review_required")
        ),
        guarded_blackboard_entry_count=len(
            _blackboard_ids_for_posture(entries, "guarded")
        ),
        linked_agent_ids=router.linked_agent_ids,
        covered_roadmap_items=(COGNITIVE_BLACKBOARD_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture=router.graph_posture,
    )


def cognitive_blackboard_entry_by_id(
    blackboard_entry_id: str,
    blackboard: CognitiveBlackboardPlan | None = None,
) -> CognitiveBlackboardEntry | None:
    """Return one cognitive blackboard entry without reading state."""

    source_blackboard = blackboard or build_cognitive_blackboard()
    for entry in source_blackboard.blackboard_entries:
        if entry.blackboard_entry_id == blackboard_entry_id:
            return entry
    return None


def cognitive_blackboard_entries_for_layer(
    cognitive_layer: CognitiveOSLayer,
    blackboard: CognitiveBlackboardPlan | None = None,
) -> tuple[CognitiveBlackboardEntry, ...]:
    """Return cognitive blackboard entries for one Cognitive OS layer."""

    source_blackboard = blackboard or build_cognitive_blackboard()
    return tuple(
        entry
        for entry in source_blackboard.blackboard_entries
        if entry.cognitive_layer == cognitive_layer
    )


def cognitive_blackboard_entries_for_agent(
    agent_id: str,
    blackboard: CognitiveBlackboardPlan | None = None,
) -> tuple[CognitiveBlackboardEntry, ...]:
    """Return cognitive blackboard entries linked to one agent."""

    source_blackboard = blackboard or build_cognitive_blackboard()
    return tuple(
        entry
        for entry in source_blackboard.blackboard_entries
        if agent_id in entry.linked_agent_ids
    )


def cognitive_blackboard_entries_for_channel(
    channel_id: str,
    blackboard: CognitiveBlackboardPlan | None = None,
) -> tuple[CognitiveBlackboardEntry, ...]:
    """Return entries that cite a passive blackboard channel."""

    source_blackboard = blackboard or build_cognitive_blackboard()
    return tuple(
        entry
        for entry in source_blackboard.blackboard_entries
        if channel_id in entry.visible_blackboard_channel_ids
    )


def cognitive_blackboard_entries_for_posture(
    posture: CognitiveOSPosture,
    blackboard: CognitiveBlackboardPlan | None = None,
) -> tuple[CognitiveBlackboardEntry, ...]:
    """Return cognitive blackboard entries by posture without applying them."""

    source_blackboard = blackboard or build_cognitive_blackboard()
    return tuple(
        entry
        for entry in source_blackboard.blackboard_entries
        if entry.blackboard_posture == posture
    )


def _cognitive_blackboard_entries(
    router: CognitiveRouterPlan,
    registry: BlackboardMemoryRegistry,
) -> tuple[CognitiveBlackboardEntry, ...]:
    return tuple(
        CognitiveBlackboardEntry(
            blackboard_entry_id=f"cognitive_blackboard::{decision.capability_id}",
            route_decision_id=decision.route_decision_id,
            plan_id=decision.plan_id,
            schedule_id=decision.schedule_id,
            emergence_id=decision.emergence_id,
            identity_id=decision.identity_id,
            cognition_id=decision.cognition_id,
            governance_id=decision.governance_id,
            planning_id=decision.planning_id,
            reasoning_id=decision.reasoning_id,
            profile_id=decision.profile_id,
            state_id=decision.state_id,
            capability_id=decision.capability_id,
            capability_name=decision.capability_name,
            cognitive_layer=decision.cognitive_layer,
            linked_agent_ids=decision.linked_agent_ids,
            visible_blackboard_channel_ids=_channels_for_agents(
                decision.linked_agent_ids,
                registry,
            ),
            visible_blackboard_channel_count=len(
                _channels_for_agents(decision.linked_agent_ids, registry)
            ),
            referenceable_blackboard_channel_ids=registry.channel_ids,
            blackboard_rank=decision.route_rank,
            dependency_depth=decision.dependency_depth,
            blackboard_dimensions=COGNITIVE_BLACKBOARD_DIMENSIONS,
            routing_posture=decision.routing_posture,
            blackboard_posture=decision.routing_posture,
            blackboard_summary=(
                f"Read-only cognitive blackboard entry for "
                f"{decision.capability_name}; maps {decision.route_decision_id} "
                "to passive channel visibility, ownership, governance, "
                "explanation, and HITL metadata without runtime blackboard access."
            ),
            dependency_contracts=(
                "cognitive blackboard entry follows cognitive route decision",
                f"cognitive route decision:{decision.route_decision_id}",
                f"blackboard memory registry:{registry.serialization_version}",
            ),
            governance_contracts=(
                "cognitive blackboard does not read runtime blackboard state",
                "cognitive blackboard does not write or persist records",
                "HITL required before any blackboard-driven behavior",
            ),
            explanation_contracts=(
                "cognitive blackboard cites router and channel contracts",
                "cognitive blackboard preserves capability and agent ownership",
                "cognitive blackboard explains why no state is materialized",
            ),
            blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        for decision in router.route_decisions
    )


def _channels_for_agents(
    agent_ids: tuple[str, ...],
    registry: BlackboardMemoryRegistry,
) -> tuple[str, ...]:
    known_channels = set(registry.channel_ids)
    return tuple(
        channel_id
        for channel_id in (f"{agent_id}_blackboard_channel" for agent_id in agent_ids)
        if channel_id in known_channels
    )


def _blackboard_ids_for_posture(
    entries: tuple[CognitiveBlackboardEntry, ...],
    posture: CognitiveOSPosture,
) -> tuple[str, ...]:
    return tuple(
        entry.blackboard_entry_id
        for entry in entries
        if entry.blackboard_posture == posture
    )
