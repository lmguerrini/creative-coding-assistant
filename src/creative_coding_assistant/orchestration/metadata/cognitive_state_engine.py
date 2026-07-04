"""V6.6 Cognitive State Engine metadata."""

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
from creative_coding_assistant.orchestration.cross_system_optimization_layer import (
    CrossSystemOptimizationLayerPlan,
    build_cross_system_optimization_layer,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)

COGNITIVE_STATE_ENGINE_SERIALIZATION_VERSION = "cognitive_state_engine.v1"
COGNITIVE_STATE_ENGINE_ROADMAP_ITEM = "Cognitive State Engine"
COGNITIVE_STATE_ENGINE_AUTHORITY_BOUNDARY = (
    "V6.6 Cognitive State Engine snapshots cross-system optimization, "
    "learning, capability, layer, and agent posture as read-only Cognitive OS "
    "state metadata. It exposes state dependencies, ownership, "
    "explainability, governance, and HITL readiness for inspection only; it "
    "does not persist cognitive state, mutate state, route agents, route "
    "providers or models, execute providers, control workflows, mutate "
    "generated output, emit HITL requests, apply HITL decisions, or apply "
    "Runtime Evolution."
)


class CognitiveStateSnapshot(BaseModel):
    """One read-only Cognitive OS state snapshot."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    state_id: str = Field(min_length=1, max_length=160)
    capability_id: str = Field(min_length=1, max_length=80)
    capability_name: CognitiveOSCapability
    cognitive_layer: CognitiveOSLayer
    source_optimization_signal_id: str = Field(min_length=1, max_length=180)
    source_learning_signal_id: str = Field(min_length=1, max_length=180)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    source_optimization_proposal_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_optimization_proposal_count: int = Field(ge=5, le=5)
    optimization_posture: CognitiveOSPosture
    state_posture: CognitiveOSPosture
    state_summary: str = Field(min_length=1, max_length=420)
    dependency_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _snapshot_matches_sources_and_boundary(self) -> Self:
        expected_state_id = f"cognitive_state::{self.capability_id}"
        if self.state_id != expected_state_id:
            raise ValueError("state_id must match capability_id")
        expected_optimization_id = f"cross_system_optimization::{self.capability_id}"
        if self.source_optimization_signal_id != expected_optimization_id:
            raise ValueError("source_optimization_signal_id must match capability")
        expected_learning_id = f"cross_system_learning::{self.capability_id}"
        if self.source_learning_signal_id != expected_learning_id:
            raise ValueError("source_learning_signal_id must match capability")
        if self.source_optimization_proposal_count != len(
            self.source_optimization_proposal_ids
        ):
            raise ValueError(
                "source_optimization_proposal_count must match proposals",
            )
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class CognitiveStateEnginePlan(BaseModel):
    """Read-only Cognitive OS state snapshot layer."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cognitive_state_engine"] = "cognitive_state_engine"
    serialization_version: Literal["cognitive_state_engine.v1"] = (
        COGNITIVE_STATE_ENGINE_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=COGNITIVE_STATE_ENGINE_AUTHORITY_BOUNDARY,
        max_length=1800,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    optimization_layer_role: Literal["cross_system_optimization_layer"]
    optimization_layer_serialization_version: Literal[
        "cross_system_optimization_layer.v1"
    ]
    learning_layer_role: Literal["cross_system_learning_layer"]
    layer_order: tuple[CognitiveOSLayer, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_count: int = Field(ge=6, le=6)
    source_optimization_signal_ids: tuple[str, ...] = Field(
        min_length=6,
        max_length=6,
    )
    source_optimization_signal_count: int = Field(ge=6, le=6)
    source_learning_signal_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_learning_signal_count: int = Field(ge=6, le=6)
    source_optimization_proposal_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_optimization_proposal_count: int = Field(ge=5, le=5)
    state_snapshots: tuple[CognitiveStateSnapshot, ...] = Field(
        min_length=6,
        max_length=6,
    )
    state_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    state_count: int = Field(ge=6, le=6)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    cognitive_state_engine_implemented: Literal[True] = True
    cross_system_optimization_layer_integrated: Literal[True] = True
    state_snapshot_contract_implemented: Literal[True] = True
    state_dependency_traceability_implemented: Literal[True] = True
    state_governance_contract_implemented: Literal[True] = True
    state_explainability_contract_implemented: Literal[True] = True
    state_persistence_implemented: Literal[False] = False
    state_mutation_implemented: Literal[False] = False
    stateful_agent_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    persisted_state_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_state_ids: tuple[str, ...] = Field(default_factory=tuple)
    routed_state_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _state_engine_matches_sources(self) -> Self:
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_count != len(self.capability_ids):
            raise ValueError("capability_count must match capability ids")
        if self.source_optimization_signal_count != len(
            self.source_optimization_signal_ids
        ):
            raise ValueError("source_optimization_signal_count must match signals")
        if self.source_learning_signal_count != len(self.source_learning_signal_ids):
            raise ValueError("source_learning_signal_count must match signals")
        if self.source_optimization_proposal_count != len(
            self.source_optimization_proposal_ids
        ):
            raise ValueError("source_optimization_proposal_count must match proposals")
        if self.state_ids != tuple(
            snapshot.state_id for snapshot in self.state_snapshots
        ):
            raise ValueError("state_ids must match state snapshots")
        if self.state_count != len(self.state_snapshots):
            raise ValueError("state_count must match state snapshots")
        if len(set(self.state_ids)) != len(self.state_ids):
            raise ValueError("state_ids must be unique")
        declared_capabilities = set(self.capability_ids)
        declared_optimization_signals = set(self.source_optimization_signal_ids)
        declared_learning_signals = set(self.source_learning_signal_ids)
        declared_agents = set(self.linked_agent_ids)
        for snapshot in self.state_snapshots:
            if snapshot.capability_id not in declared_capabilities:
                raise ValueError("snapshot capability_id must be declared")
            if (
                snapshot.source_optimization_signal_id
                not in declared_optimization_signals
            ):
                raise ValueError("snapshot optimization signal must be declared")
            if snapshot.source_learning_signal_id not in declared_learning_signals:
                raise ValueError("snapshot learning signal must be declared")
            if (
                snapshot.source_optimization_proposal_ids
                != self.source_optimization_proposal_ids
            ):
                raise ValueError("snapshot proposal ids must match engine")
            if not set(snapshot.linked_agent_ids).issubset(declared_agents):
                raise ValueError("snapshot linked_agent_ids must be declared")
        if self.covered_roadmap_items != (COGNITIVE_STATE_ENGINE_ROADMAP_ITEM,):
            raise ValueError("covered_roadmap_items must be Task 9 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.persisted_state_ids,
                self.mutated_state_ids,
                self.routed_state_ids,
                self.emitted_hitl_request_ids,
            )
        ):
            raise ValueError(
                "state persistence, mutation, routing, and HITL ids must be empty",
            )
        if not all(snapshot.advisory_only for snapshot in self.state_snapshots):
            raise ValueError("all state snapshots must be advisory only")
        return self


def build_cognitive_state_engine(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    optimization_layer: CrossSystemOptimizationLayerPlan | None = None,
) -> CognitiveStateEnginePlan:
    """Build read-only Cognitive OS state metadata."""

    optimization = optimization_layer or build_cross_system_optimization_layer(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    snapshots = _state_snapshots(optimization)
    return CognitiveStateEnginePlan(
        route_name=optimization.route_name,
        task_type=optimization.task_type,
        execution_mode_ids=optimization.execution_mode_ids,
        optimization_layer_role=optimization.role,
        optimization_layer_serialization_version=optimization.serialization_version,
        learning_layer_role=optimization.learning_layer_role,
        layer_order=optimization.layer_order,
        capabilities=optimization.capabilities,
        capability_ids=optimization.capability_ids,
        capability_count=optimization.capability_count,
        source_optimization_signal_ids=optimization.optimization_signal_ids,
        source_optimization_signal_count=optimization.optimization_signal_count,
        source_learning_signal_ids=optimization.source_learning_signal_ids,
        source_learning_signal_count=optimization.source_learning_signal_count,
        source_optimization_proposal_ids=optimization.source_optimization_proposal_ids,
        source_optimization_proposal_count=(
            optimization.source_optimization_proposal_count
        ),
        state_snapshots=snapshots,
        state_ids=tuple(snapshot.state_id for snapshot in snapshots),
        state_count=len(snapshots),
        linked_agent_ids=optimization.linked_agent_ids,
        covered_roadmap_items=(COGNITIVE_STATE_ENGINE_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture=optimization.graph_posture,
    )


def cognitive_state_snapshot_by_id(
    state_id: str,
    engine: CognitiveStateEnginePlan | None = None,
) -> CognitiveStateSnapshot | None:
    """Return one cognitive state snapshot without persisting or applying it."""

    source_engine = engine or build_cognitive_state_engine()
    for snapshot in source_engine.state_snapshots:
        if snapshot.state_id == state_id:
            return snapshot
    return None


def cognitive_state_snapshots_for_layer(
    cognitive_layer: CognitiveOSLayer,
    engine: CognitiveStateEnginePlan | None = None,
) -> tuple[CognitiveStateSnapshot, ...]:
    """Return read-only cognitive state snapshots for one layer."""

    source_engine = engine or build_cognitive_state_engine()
    return tuple(
        snapshot
        for snapshot in source_engine.state_snapshots
        if snapshot.cognitive_layer == cognitive_layer
    )


def cognitive_state_snapshots_for_agent(
    agent_id: str,
    engine: CognitiveStateEnginePlan | None = None,
) -> tuple[CognitiveStateSnapshot, ...]:
    """Return read-only cognitive state snapshots linked to one agent."""

    source_engine = engine or build_cognitive_state_engine()
    return tuple(
        snapshot
        for snapshot in source_engine.state_snapshots
        if agent_id in snapshot.linked_agent_ids
    )


def _state_snapshots(
    optimization_layer: CrossSystemOptimizationLayerPlan,
) -> tuple[CognitiveStateSnapshot, ...]:
    return tuple(
        CognitiveStateSnapshot(
            state_id=f"cognitive_state::{signal.capability_id}",
            capability_id=signal.capability_id,
            capability_name=signal.capability_name,
            cognitive_layer=signal.cognitive_layer,
            source_optimization_signal_id=signal.optimization_signal_id,
            source_learning_signal_id=signal.learning_signal_id,
            linked_agent_ids=signal.linked_agent_ids,
            source_optimization_proposal_ids=(signal.source_optimization_proposal_ids),
            source_optimization_proposal_count=(
                signal.source_optimization_proposal_count
            ),
            optimization_posture=signal.optimization_posture,
            state_posture=signal.optimization_posture,
            state_summary=(
                f"Read-only cognitive state snapshot for "
                f"{signal.capability_name}; traces {signal.learning_signal_id} "
                f"and {signal.optimization_signal_id} without persistence, "
                "mutation, routing, or execution authority."
            ),
            dependency_contracts=(
                "state snapshot follows cross-system optimization signal",
                f"optimization signal:{signal.optimization_signal_id}",
                f"learning signal:{signal.learning_signal_id}",
            ),
            governance_contracts=(
                "state snapshot does not authorize state persistence",
                "state snapshot does not authorize state mutation or routing",
                "HITL required before any stateful behavioral application",
            ),
            explanation_contracts=(
                "state snapshot cites optimization and learning sources",
                "state snapshot preserves capability and layer ownership",
                "state snapshot explains why no runtime behavior is applied",
            ),
            blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        for signal in optimization_layer.optimization_signals
    )
