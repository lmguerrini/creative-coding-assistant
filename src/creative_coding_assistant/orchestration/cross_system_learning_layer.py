"""V6.6 Cross-System Learning Layer metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.adaptive_learning_engine import (
    AdaptiveLearningPlan,
    evaluate_adaptive_learning_engine,
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
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)
from creative_coding_assistant.orchestration.unified_capability_registry import (
    UnifiedCapabilityRegistryPlan,
    build_unified_capability_registry,
)

CROSS_SYSTEM_LEARNING_LAYER_SERIALIZATION_VERSION = (
    "cross_system_learning_layer.v1"
)
CROSS_SYSTEM_LEARNING_LAYER_ROADMAP_ITEM = "Cross-System Learning Layer"
CROSS_SYSTEM_LEARNING_LAYER_AUTHORITY_BOUNDARY = (
    "V6.6 Cross-System Learning Layer projects V6.1 adaptive learning "
    "signals across the Unified Capability Registry so every Cognitive OS "
    "capability can expose source-owned learning posture, dependency traces, "
    "explainability, governance, and HITL readiness. It does not persist "
    "learning memory, apply feedback, mutate learning policies, activate "
    "capabilities, route agents, route providers or models, execute "
    "providers, control workflows, mutate generated output, emit HITL "
    "requests, apply HITL decisions, or apply Runtime Evolution."
)


class CrossSystemLearningSignal(BaseModel):
    """One cross-system learning projection for a Cognitive OS capability."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    signal_id: str = Field(min_length=1, max_length=160)
    capability_id: str = Field(min_length=1, max_length=80)
    capability_name: CognitiveOSCapability
    cognitive_layer: CognitiveOSLayer
    knowledge_node_id: str = Field(min_length=1, max_length=150)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    source_adaptive_signal_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    source_adaptive_signal_count: int = Field(ge=5, le=5)
    learning_posture: CognitiveOSPosture
    learning_priority_score: int = Field(ge=0, le=1_000)
    dependency_contracts: tuple[str, ...] = Field(min_length=2, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=2, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=2, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_capability_and_boundary(self) -> Self:
        expected_signal_id = f"cross_system_learning::{self.capability_id}"
        if self.signal_id != expected_signal_id:
            raise ValueError("signal_id must match capability_id")
        if self.source_adaptive_signal_count != len(self.source_adaptive_signal_ids):
            raise ValueError("source_adaptive_signal_count must match signals")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class CrossSystemLearningLayerPlan(BaseModel):
    """Advisory cross-system learning layer over Cognitive OS capabilities."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cross_system_learning_layer"] = "cross_system_learning_layer"
    serialization_version: Literal["cross_system_learning_layer.v1"] = (
        CROSS_SYSTEM_LEARNING_LAYER_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CROSS_SYSTEM_LEARNING_LAYER_AUTHORITY_BOUNDARY,
        max_length=1800,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    capability_registry_role: Literal["unified_capability_registry"]
    capability_registry_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    adaptive_learning_role: Literal["adaptive_learning_engine"]
    adaptive_learning_serialization_version: Literal["adaptive_learning_plan.v1"]
    layer_order: tuple[CognitiveOSLayer, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_count: int = Field(ge=6, le=6)
    source_adaptive_signal_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    source_adaptive_signal_count: int = Field(ge=5, le=5)
    adaptive_hitl_required_signal_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    adaptive_hitl_required_signal_count: int = Field(ge=5, le=5)
    learning_signals: tuple[CrossSystemLearningSignal, ...] = Field(
        min_length=6,
        max_length=6,
    )
    learning_signal_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    learning_signal_count: int = Field(ge=6, le=6)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    cross_system_learning_layer_implemented: Literal[True] = True
    unified_capability_registry_integrated: Literal[True] = True
    adaptive_learning_engine_integrated: Literal[True] = True
    cross_capability_learning_traceability_implemented: Literal[True] = True
    learning_governance_contract_implemented: Literal[True] = True
    learning_explainability_contract_implemented: Literal[True] = True
    learning_memory_persistence_implemented: Literal[False] = False
    learning_feedback_application_implemented: Literal[False] = False
    learning_policy_mutation_implemented: Literal[False] = False
    capability_activation_implemented: Literal[False] = False
    agent_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    persisted_learning_signal_ids: tuple[str, ...] = Field(default_factory=tuple)
    applied_learning_signal_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_learning_policy_ids: tuple[str, ...] = Field(default_factory=tuple)
    activated_capability_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _learning_layer_matches_sources(self) -> Self:
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_count != len(self.capability_ids):
            raise ValueError("capability_count must match capability ids")
        if self.source_adaptive_signal_count != len(self.source_adaptive_signal_ids):
            raise ValueError("source_adaptive_signal_count must match signals")
        if self.adaptive_hitl_required_signal_count != len(
            self.adaptive_hitl_required_signal_ids
        ):
            raise ValueError("adaptive_hitl_required_signal_count must match signals")
        if self.learning_signal_ids != tuple(
            signal.signal_id for signal in self.learning_signals
        ):
            raise ValueError("learning_signal_ids must match learning signals")
        if self.learning_signal_count != len(self.learning_signals):
            raise ValueError("learning_signal_count must match learning signals")
        if len(set(self.learning_signal_ids)) != len(self.learning_signal_ids):
            raise ValueError("learning_signal_ids must be unique")
        declared_capabilities = set(self.capability_ids)
        declared_agents = set(self.linked_agent_ids)
        for signal in self.learning_signals:
            if signal.capability_id not in declared_capabilities:
                raise ValueError("signal capability_id must be declared")
            if signal.source_adaptive_signal_ids != self.source_adaptive_signal_ids:
                raise ValueError("signal adaptive ids must match layer")
            if not set(signal.linked_agent_ids).issubset(declared_agents):
                raise ValueError("signal linked_agent_ids must be declared")
        if self.covered_roadmap_items != (CROSS_SYSTEM_LEARNING_LAYER_ROADMAP_ITEM,):
            raise ValueError("covered_roadmap_items must be Task 7 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.persisted_learning_signal_ids,
                self.applied_learning_signal_ids,
                self.mutated_learning_policy_ids,
                self.activated_capability_ids,
                self.emitted_hitl_request_ids,
            )
        ):
            raise ValueError(
                "learning persistence, application, mutation, and HITL ids "
                "must be empty",
            )
        if not all(signal.advisory_only for signal in self.learning_signals):
            raise ValueError("all learning signals must be advisory only")
        return self


def build_cross_system_learning_layer(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    capability_registry: UnifiedCapabilityRegistryPlan | None = None,
    adaptive_learning: AdaptiveLearningPlan | None = None,
) -> CrossSystemLearningLayerPlan:
    """Build advisory cross-system learning metadata."""

    capabilities = capability_registry or build_unified_capability_registry(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    adaptive = adaptive_learning or evaluate_adaptive_learning_engine(
        route=capabilities.route_name,
        task_type=capabilities.task_type,
        execution_mode_id=execution_mode_id,
    )
    signals = _learning_signals(capabilities, adaptive)
    return CrossSystemLearningLayerPlan(
        route_name=capabilities.route_name,
        task_type=capabilities.task_type,
        execution_mode_ids=capabilities.execution_mode_ids,
        capability_registry_role=capabilities.role,
        capability_registry_serialization_version=capabilities.serialization_version,
        adaptive_learning_role=adaptive.role,
        adaptive_learning_serialization_version=adaptive.serialization_version,
        layer_order=capabilities.layer_order,
        capabilities=capabilities.capabilities,
        capability_ids=capabilities.capability_ids,
        capability_count=capabilities.capability_count,
        source_adaptive_signal_ids=adaptive.signal_ids,
        source_adaptive_signal_count=adaptive.signal_count,
        adaptive_hitl_required_signal_ids=adaptive.hitl_required_signal_ids,
        adaptive_hitl_required_signal_count=adaptive.hitl_required_signal_count,
        learning_signals=signals,
        learning_signal_ids=tuple(signal.signal_id for signal in signals),
        learning_signal_count=len(signals),
        linked_agent_ids=capabilities.agent_ids,
        covered_roadmap_items=(CROSS_SYSTEM_LEARNING_LAYER_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture="guarded",
    )


def cross_system_learning_signal_by_id(
    signal_id: str,
    layer: CrossSystemLearningLayerPlan | None = None,
) -> CrossSystemLearningSignal | None:
    """Return one cross-system learning signal without applying it."""

    source_layer = layer or build_cross_system_learning_layer()
    for signal in source_layer.learning_signals:
        if signal.signal_id == signal_id:
            return signal
    return None


def cross_system_learning_signals_for_layer(
    cognitive_layer: CognitiveOSLayer,
    layer: CrossSystemLearningLayerPlan | None = None,
) -> tuple[CrossSystemLearningSignal, ...]:
    """Return cross-system learning signals for one cognitive layer."""

    source_layer = layer or build_cross_system_learning_layer()
    return tuple(
        signal
        for signal in source_layer.learning_signals
        if signal.cognitive_layer == cognitive_layer
    )


def cross_system_learning_signals_for_agent(
    agent_id: str,
    layer: CrossSystemLearningLayerPlan | None = None,
) -> tuple[CrossSystemLearningSignal, ...]:
    """Return cross-system learning signals linked to one passive agent id."""

    source_layer = layer or build_cross_system_learning_layer()
    return tuple(
        signal
        for signal in source_layer.learning_signals
        if agent_id in signal.linked_agent_ids
    )


def _learning_signals(
    capability_registry: UnifiedCapabilityRegistryPlan,
    adaptive_learning: AdaptiveLearningPlan,
) -> tuple[CrossSystemLearningSignal, ...]:
    return tuple(
        CrossSystemLearningSignal(
            signal_id=f"cross_system_learning::{entry.capability_id}",
            capability_id=entry.capability_id,
            capability_name=entry.capability_name,
            cognitive_layer=entry.cognitive_layer,
            knowledge_node_id=entry.knowledge_node_id,
            linked_agent_ids=entry.linked_agent_ids,
            source_adaptive_signal_ids=adaptive_learning.signal_ids,
            source_adaptive_signal_count=adaptive_learning.signal_count,
            learning_posture=adaptive_learning.overall_learning_posture,
            learning_priority_score=adaptive_learning.overall_learning_priority_score,
            dependency_contracts=(
                "learning signal follows unified capability registry entry",
                f"capability:{entry.capability_id}",
            ),
            governance_contracts=(
                "learning signal does not authorize feedback application",
                "HITL required before any learning-driven behavior",
            ),
            explanation_contracts=(
                "learning signal cites V6.1 adaptive learning signals",
                "capability registry explains cross-system placement",
            ),
            blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        for entry in capability_registry.capability_entries
    )
