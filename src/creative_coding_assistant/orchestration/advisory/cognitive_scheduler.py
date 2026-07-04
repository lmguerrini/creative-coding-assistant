"""V6.6 Cognitive Scheduler metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
    CognitiveOSCapability,
    CognitiveOSLayer,
    CognitiveOSPosture,
)
from creative_coding_assistant.orchestration.emergent_creativity_layer import (
    EmergentCreativityLayerPlan,
    build_emergent_creativity_layer,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)

COGNITIVE_SCHEDULER_SERIALIZATION_VERSION = "cognitive_scheduler.v1"
COGNITIVE_SCHEDULER_ROADMAP_ITEM = "Cognitive Scheduler"
COGNITIVE_SCHEDULER_AUTHORITY_BOUNDARY = (
    "V6.6 Cognitive Scheduler projects emergent creativity, identity, "
    "governance, and meta-planning signals into read-only cognitive schedule "
    "slots for capability attention ordering, dependency-aware review, "
    "governance checkpoint sequencing, explanation continuity, and HITL "
    "readiness. It exposes scheduler metadata only; it does not schedule "
    "runtime tasks, create async work, invoke agents, mutate workflows, "
    "route providers or models, change prompts, memory, retrieval, storage, "
    "generated output, runtime state, or apply Runtime Evolution."
)
COGNITIVE_SCHEDULING_DIMENSIONS = (
    "capability attention ordering",
    "dependency-aware review window",
    "governance checkpoint sequencing",
    "explanation continuity",
    "HITL readiness window",
)


class CognitiveScheduleSlot(BaseModel):
    """One read-only Cognitive OS schedule slot."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

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
    schedule_rank: int = Field(ge=1, le=6)
    dependency_depth: int = Field(ge=0, le=5)
    upstream_schedule_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=1)
    downstream_schedule_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=1,
    )
    scheduling_dimensions: tuple[str, ...] = Field(min_length=5, max_length=5)
    emergence_posture: CognitiveOSPosture
    scheduling_posture: CognitiveOSPosture
    execution_schedule_authorized: Literal[False] = False
    schedule_summary: str = Field(min_length=1, max_length=560)
    dependency_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _slot_matches_sources_and_boundary(self) -> Self:
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
        if self.scheduling_dimensions != COGNITIVE_SCHEDULING_DIMENSIONS:
            raise ValueError("scheduling_dimensions must match V6.6 scheduler")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class CognitiveSchedulerPlan(BaseModel):
    """Read-only cognitive scheduler over emergent creativity signals."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cognitive_scheduler"] = "cognitive_scheduler"
    serialization_version: Literal["cognitive_scheduler.v1"] = (
        COGNITIVE_SCHEDULER_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=COGNITIVE_SCHEDULER_AUTHORITY_BOUNDARY,
        max_length=2200,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    emergent_creativity_layer_role: Literal["emergent_creativity_layer"]
    emergent_creativity_layer_serialization_version: Literal[
        "emergent_creativity_layer.v1"
    ]
    creative_identity_layer_role: Literal["creative_identity_layer"]
    creative_cognition_layer_role: Literal["creative_cognition_layer"]
    cognitive_governance_layer_role: Literal["cognitive_governance_layer"]
    meta_planning_layer_role: Literal["meta_planning_layer"] = "meta_planning_layer"
    layer_order: tuple[CognitiveOSLayer, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_count: int = Field(ge=6, le=6)
    source_emergence_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_emergence_count: int = Field(ge=6, le=6)
    source_identity_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_identity_count: int = Field(ge=6, le=6)
    source_cognition_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_cognition_count: int = Field(ge=6, le=6)
    source_governance_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_governance_count: int = Field(ge=6, le=6)
    source_planning_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_planning_count: int = Field(ge=6, le=6)
    source_reasoning_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_reasoning_count: int = Field(ge=6, le=6)
    source_profile_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_profile_count: int = Field(ge=6, le=6)
    source_state_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_state_count: int = Field(ge=6, le=6)
    schedule_slots: tuple[CognitiveScheduleSlot, ...] = Field(
        min_length=6,
        max_length=6,
    )
    schedule_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    candidate_schedule_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    review_required_schedule_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    guarded_schedule_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    schedule_count: int = Field(ge=6, le=6)
    candidate_schedule_count: int = Field(ge=0, le=6)
    review_required_schedule_count: int = Field(ge=0, le=6)
    guarded_schedule_count: int = Field(ge=0, le=6)
    max_dependency_depth: int = Field(ge=0, le=5)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    cognitive_scheduler_implemented: Literal[True] = True
    emergent_creativity_layer_integrated: Literal[True] = True
    schedule_slot_contract_implemented: Literal[True] = True
    schedule_dependency_traceability_implemented: Literal[True] = True
    schedule_governance_contract_implemented: Literal[True] = True
    schedule_explainability_contract_implemented: Literal[True] = True
    runtime_scheduling_implemented: Literal[False] = False
    autonomous_workflow_scheduling_implemented: Literal[False] = False
    parallel_execution_implemented: Literal[False] = False
    async_execution_implemented: Literal[False] = False
    workflow_timing_mutation_implemented: Literal[False] = False
    workflow_graph_mutation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    memory_mutation_implemented: Literal[False] = False
    retrieval_mutation_implemented: Literal[False] = False
    storage_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    executed_schedule_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_schedule_ids: tuple[str, ...] = Field(default_factory=tuple)
    routed_schedule_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _scheduler_matches_sources(self) -> Self:
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_count != len(self.capability_ids):
            raise ValueError("capability_count must match capability ids")
        if self.source_emergence_count != len(self.source_emergence_ids):
            raise ValueError("source_emergence_count must match emergence ids")
        if self.source_identity_count != len(self.source_identity_ids):
            raise ValueError("source_identity_count must match identity ids")
        if self.source_cognition_count != len(self.source_cognition_ids):
            raise ValueError("source_cognition_count must match cognition ids")
        if self.source_governance_count != len(self.source_governance_ids):
            raise ValueError("source_governance_count must match governance ids")
        if self.source_planning_count != len(self.source_planning_ids):
            raise ValueError("source_planning_count must match planning ids")
        if self.source_reasoning_count != len(self.source_reasoning_ids):
            raise ValueError("source_reasoning_count must match reasoning ids")
        if self.source_profile_count != len(self.source_profile_ids):
            raise ValueError("source_profile_count must match profile ids")
        if self.source_state_count != len(self.source_state_ids):
            raise ValueError("source_state_count must match state ids")
        if self.schedule_ids != tuple(slot.schedule_id for slot in self.schedule_slots):
            raise ValueError("schedule_ids must match slots")
        if self.schedule_count != len(self.schedule_slots):
            raise ValueError("schedule_count must match slots")
        if len(set(self.schedule_ids)) != len(self.schedule_ids):
            raise ValueError("schedule_ids must be unique")
        expected_ranks = tuple(range(1, len(self.schedule_slots) + 1))
        if tuple(slot.schedule_rank for slot in self.schedule_slots) != expected_ranks:
            raise ValueError("schedule ranks must match slot order")
        if self.candidate_schedule_ids != _schedule_ids_for_posture(
            self.schedule_slots,
            "candidate",
        ):
            raise ValueError("candidate_schedule_ids must match slots")
        if self.review_required_schedule_ids != _schedule_ids_for_posture(
            self.schedule_slots,
            "review_required",
        ):
            raise ValueError("review_required_schedule_ids must match slots")
        if self.guarded_schedule_ids != _schedule_ids_for_posture(
            self.schedule_slots,
            "guarded",
        ):
            raise ValueError("guarded_schedule_ids must match slots")
        if self.candidate_schedule_count != len(self.candidate_schedule_ids):
            raise ValueError("candidate_schedule_count must match ids")
        if self.review_required_schedule_count != len(
            self.review_required_schedule_ids
        ):
            raise ValueError("review_required_schedule_count must match ids")
        if self.guarded_schedule_count != len(self.guarded_schedule_ids):
            raise ValueError("guarded_schedule_count must match ids")
        expected_max_depth = max(slot.dependency_depth for slot in self.schedule_slots)
        if self.max_dependency_depth != expected_max_depth:
            raise ValueError("max_dependency_depth must match slots")

        declared_slots = {
            slot.schedule_id: index for index, slot in enumerate(self.schedule_slots)
        }
        declared_capabilities = set(self.capability_ids)
        declared_emergence = set(self.source_emergence_ids)
        declared_identities = set(self.source_identity_ids)
        declared_cognition = set(self.source_cognition_ids)
        declared_governance = set(self.source_governance_ids)
        declared_planning = set(self.source_planning_ids)
        declared_reasoning = set(self.source_reasoning_ids)
        declared_profiles = set(self.source_profile_ids)
        declared_states = set(self.source_state_ids)
        declared_agents = set(self.linked_agent_ids)
        for slot in self.schedule_slots:
            if slot.capability_id not in declared_capabilities:
                raise ValueError("slot capability_id must be declared")
            if slot.emergence_id not in declared_emergence:
                raise ValueError("slot emergence_id must be declared")
            if slot.identity_id not in declared_identities:
                raise ValueError("slot identity_id must be declared")
            if slot.cognition_id not in declared_cognition:
                raise ValueError("slot cognition_id must be declared")
            if slot.governance_id not in declared_governance:
                raise ValueError("slot governance_id must be declared")
            if slot.planning_id not in declared_planning:
                raise ValueError("slot planning_id must be declared")
            if slot.reasoning_id not in declared_reasoning:
                raise ValueError("slot reasoning_id must be declared")
            if slot.profile_id not in declared_profiles:
                raise ValueError("slot profile_id must be declared")
            if slot.state_id not in declared_states:
                raise ValueError("slot state_id must be declared")
            if not set(slot.linked_agent_ids).issubset(declared_agents):
                raise ValueError("slot linked_agent_ids must be declared")
            for upstream_id in slot.upstream_schedule_ids:
                if upstream_id not in declared_slots:
                    raise ValueError("upstream_schedule_ids must be known slots")
                if declared_slots[upstream_id] >= declared_slots[slot.schedule_id]:
                    raise ValueError("upstream schedules must precede slot")
            for downstream_id in slot.downstream_schedule_ids:
                if downstream_id not in declared_slots:
                    raise ValueError("downstream_schedule_ids must be known slots")
                if declared_slots[downstream_id] <= declared_slots[slot.schedule_id]:
                    raise ValueError("downstream schedules must follow slot")
        if self.covered_roadmap_items != (COGNITIVE_SCHEDULER_ROADMAP_ITEM,):
            raise ValueError("covered_roadmap_items must be Task 17 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.executed_schedule_ids,
                self.mutated_schedule_ids,
                self.routed_schedule_ids,
                self.emitted_hitl_request_ids,
            )
        ):
            raise ValueError(
                "schedule execution, mutation, routing, and HITL ids must be empty",
            )
        if not all(slot.advisory_only for slot in self.schedule_slots):
            raise ValueError("all cognitive schedule slots must be advisory only")
        return self


def build_cognitive_scheduler(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    emergent_creativity_layer: EmergentCreativityLayerPlan | None = None,
) -> CognitiveSchedulerPlan:
    """Build read-only cognitive scheduler metadata."""

    emergence = emergent_creativity_layer or build_emergent_creativity_layer(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    slots = _cognitive_schedule_slots(emergence)
    return CognitiveSchedulerPlan(
        route_name=emergence.route_name,
        task_type=emergence.task_type,
        execution_mode_ids=emergence.execution_mode_ids,
        emergent_creativity_layer_role=emergence.role,
        emergent_creativity_layer_serialization_version=(
            emergence.serialization_version
        ),
        creative_identity_layer_role=emergence.creative_identity_layer_role,
        creative_cognition_layer_role=emergence.creative_cognition_layer_role,
        cognitive_governance_layer_role=emergence.cognitive_governance_layer_role,
        layer_order=emergence.layer_order,
        capabilities=emergence.capabilities,
        capability_ids=emergence.capability_ids,
        capability_count=emergence.capability_count,
        source_emergence_ids=emergence.emergence_ids,
        source_emergence_count=emergence.emergence_count,
        source_identity_ids=emergence.source_identity_ids,
        source_identity_count=emergence.source_identity_count,
        source_cognition_ids=emergence.source_cognition_ids,
        source_cognition_count=emergence.source_cognition_count,
        source_governance_ids=emergence.source_governance_ids,
        source_governance_count=emergence.source_governance_count,
        source_planning_ids=emergence.source_planning_ids,
        source_planning_count=emergence.source_planning_count,
        source_reasoning_ids=emergence.source_reasoning_ids,
        source_reasoning_count=emergence.source_reasoning_count,
        source_profile_ids=emergence.source_profile_ids,
        source_profile_count=emergence.source_profile_count,
        source_state_ids=emergence.source_state_ids,
        source_state_count=emergence.source_state_count,
        schedule_slots=slots,
        schedule_ids=tuple(slot.schedule_id for slot in slots),
        candidate_schedule_ids=_schedule_ids_for_posture(slots, "candidate"),
        review_required_schedule_ids=_schedule_ids_for_posture(
            slots,
            "review_required",
        ),
        guarded_schedule_ids=_schedule_ids_for_posture(slots, "guarded"),
        schedule_count=len(slots),
        candidate_schedule_count=len(_schedule_ids_for_posture(slots, "candidate")),
        review_required_schedule_count=len(
            _schedule_ids_for_posture(slots, "review_required")
        ),
        guarded_schedule_count=len(_schedule_ids_for_posture(slots, "guarded")),
        max_dependency_depth=max(slot.dependency_depth for slot in slots),
        linked_agent_ids=emergence.linked_agent_ids,
        covered_roadmap_items=(COGNITIVE_SCHEDULER_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture=emergence.graph_posture,
    )


def cognitive_schedule_slot_by_id(
    schedule_id: str,
    scheduler: CognitiveSchedulerPlan | None = None,
) -> CognitiveScheduleSlot | None:
    """Return one cognitive schedule slot without scheduling it."""

    source_scheduler = scheduler or build_cognitive_scheduler()
    for slot in source_scheduler.schedule_slots:
        if slot.schedule_id == schedule_id:
            return slot
    return None


def cognitive_schedule_slots_for_layer(
    cognitive_layer: CognitiveOSLayer,
    scheduler: CognitiveSchedulerPlan | None = None,
) -> tuple[CognitiveScheduleSlot, ...]:
    """Return cognitive schedule slots for one Cognitive OS layer."""

    source_scheduler = scheduler or build_cognitive_scheduler()
    return tuple(
        slot
        for slot in source_scheduler.schedule_slots
        if slot.cognitive_layer == cognitive_layer
    )


def cognitive_schedule_slots_for_agent(
    agent_id: str,
    scheduler: CognitiveSchedulerPlan | None = None,
) -> tuple[CognitiveScheduleSlot, ...]:
    """Return cognitive schedule slots linked to one agent."""

    source_scheduler = scheduler or build_cognitive_scheduler()
    return tuple(
        slot
        for slot in source_scheduler.schedule_slots
        if agent_id in slot.linked_agent_ids
    )


def cognitive_schedule_slots_for_posture(
    posture: CognitiveOSPosture,
    scheduler: CognitiveSchedulerPlan | None = None,
) -> tuple[CognitiveScheduleSlot, ...]:
    """Return cognitive schedule slots by posture without applying schedules."""

    source_scheduler = scheduler or build_cognitive_scheduler()
    return tuple(
        slot
        for slot in source_scheduler.schedule_slots
        if slot.scheduling_posture == posture
    )


def _cognitive_schedule_slots(
    emergence_layer: EmergentCreativityLayerPlan,
) -> tuple[CognitiveScheduleSlot, ...]:
    schedule_ids = tuple(
        f"cognitive_scheduler::{signal.capability_id}"
        for signal in emergence_layer.emergent_creativity_signals
    )
    slots: list[CognitiveScheduleSlot] = []
    for index, signal in enumerate(emergence_layer.emergent_creativity_signals):
        upstream_ids = (schedule_ids[index - 1],) if index else ()
        downstream_ids = (
            (schedule_ids[index + 1],)
            if index < len(emergence_layer.emergent_creativity_signals) - 1
            else ()
        )
        slots.append(
            CognitiveScheduleSlot(
                schedule_id=schedule_ids[index],
                emergence_id=signal.emergence_id,
                identity_id=signal.identity_id,
                cognition_id=signal.cognition_id,
                governance_id=signal.governance_id,
                planning_id=signal.planning_id,
                reasoning_id=signal.reasoning_id,
                profile_id=signal.profile_id,
                state_id=signal.state_id,
                capability_id=signal.capability_id,
                capability_name=signal.capability_name,
                cognitive_layer=signal.cognitive_layer,
                linked_agent_ids=signal.linked_agent_ids,
                schedule_rank=index + 1,
                dependency_depth=index,
                upstream_schedule_ids=upstream_ids,
                downstream_schedule_ids=downstream_ids,
                scheduling_dimensions=COGNITIVE_SCHEDULING_DIMENSIONS,
                emergence_posture=signal.emergence_posture,
                scheduling_posture=signal.emergence_posture,
                schedule_summary=(
                    f"Read-only cognitive schedule slot for "
                    f"{signal.capability_name}; orders "
                    f"{signal.emergence_id} within the Cognitive OS layer "
                    "sequence while preserving dependency, governance, "
                    "explanation, and HITL boundaries without runtime scheduling."
                ),
                dependency_contracts=(
                    "cognitive schedule slot follows emergent creativity signal",
                    f"emergent creativity signal:{signal.emergence_id}",
                    f"meta-planning projection:{signal.planning_id}",
                ),
                governance_contracts=(
                    "cognitive scheduler does not schedule runtime tasks",
                    "cognitive scheduler preserves workflow timing and graph order",
                    "HITL required before any scheduler-driven behavior",
                ),
                explanation_contracts=(
                    "cognitive scheduler cites emergence and planning sources",
                    "cognitive scheduler preserves capability and layer ownership",
                    "cognitive scheduler explains why no schedule is applied",
                ),
                blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
            )
        )
    return tuple(slots)


def _schedule_ids_for_posture(
    slots: tuple[CognitiveScheduleSlot, ...],
    posture: CognitiveOSPosture,
) -> tuple[str, ...]:
    return tuple(
        slot.schedule_id for slot in slots if slot.scheduling_posture == posture
    )
