"""V6.6 Cross-System Optimization Layer metadata."""

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
from creative_coding_assistant.orchestration.cross_system_learning_layer import (
    CrossSystemLearningLayerPlan,
    build_cross_system_learning_layer,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)

from .autonomous_optimization_suggestions import (
    AutonomousOptimizationSuggestionsPlan,
    build_autonomous_optimization_suggestions,
)

CROSS_SYSTEM_OPTIMIZATION_LAYER_SERIALIZATION_VERSION = (
    "cross_system_optimization_layer.v1"
)
CROSS_SYSTEM_OPTIMIZATION_LAYER_ROADMAP_ITEM = "Cross-System Optimization Layer"
CROSS_SYSTEM_OPTIMIZATION_LAYER_AUTHORITY_BOUNDARY = (
    "V6.6 Cross-System Optimization Layer projects V6.5 autonomous "
    "optimization suggestions across cross-system learning signals and "
    "Cognitive OS capabilities. It exposes optimization dependencies, "
    "proposal provenance, explainability, governance, and HITL posture as "
    "inspectable metadata only; it does not apply optimization suggestions, "
    "apply evolution proposals, mutate policies, activate capabilities, route "
    "agents, route providers or models, execute providers, control workflows, "
    "mutate generated output, emit HITL requests, apply HITL decisions, or "
    "apply Runtime Evolution."
)


class CrossSystemOptimizationSignal(BaseModel):
    """One optimization projection for a cross-system learning signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    optimization_signal_id: str = Field(min_length=1, max_length=180)
    learning_signal_id: str = Field(min_length=1, max_length=180)
    capability_id: str = Field(min_length=1, max_length=80)
    capability_name: CognitiveOSCapability
    cognitive_layer: CognitiveOSLayer
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    source_optimization_proposal_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_optimization_proposal_count: int = Field(ge=5, le=5)
    optimization_posture: CognitiveOSPosture
    optimization_rank_score: int = Field(ge=0, le=1_000)
    dependency_contracts: tuple[str, ...] = Field(min_length=2, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=2, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=2, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _signal_matches_learning_source(self) -> Self:
        expected = f"cross_system_optimization::{self.capability_id}"
        if self.optimization_signal_id != expected:
            raise ValueError("optimization_signal_id must match capability_id")
        if self.source_optimization_proposal_count != len(
            self.source_optimization_proposal_ids
        ):
            raise ValueError(
                "source_optimization_proposal_count must match proposals",
            )
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class CrossSystemOptimizationLayerPlan(BaseModel):
    """Advisory cross-system optimization layer over learning signals."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cross_system_optimization_layer"] = "cross_system_optimization_layer"
    serialization_version: Literal["cross_system_optimization_layer.v1"] = (
        CROSS_SYSTEM_OPTIMIZATION_LAYER_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CROSS_SYSTEM_OPTIMIZATION_LAYER_AUTHORITY_BOUNDARY,
        max_length=1900,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    learning_layer_role: Literal["cross_system_learning_layer"]
    learning_layer_serialization_version: Literal["cross_system_learning_layer.v1"]
    optimization_source_role: Literal["autonomous_optimization_suggestions"]
    optimization_source_serialization_version: str = Field(min_length=1, max_length=120)
    layer_order: tuple[CognitiveOSLayer, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_count: int = Field(ge=6, le=6)
    source_learning_signal_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_learning_signal_count: int = Field(ge=6, le=6)
    source_optimization_proposal_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_optimization_proposal_count: int = Field(ge=5, le=5)
    optimization_hitl_required_proposal_ids: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    optimization_hitl_required_proposal_count: int = Field(ge=5, le=5)
    optimization_signals: tuple[CrossSystemOptimizationSignal, ...] = Field(
        min_length=6,
        max_length=6,
    )
    optimization_signal_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    optimization_signal_count: int = Field(ge=6, le=6)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    cross_system_optimization_layer_implemented: Literal[True] = True
    cross_system_learning_layer_integrated: Literal[True] = True
    autonomous_optimization_suggestions_integrated: Literal[True] = True
    optimization_dependency_traceability_implemented: Literal[True] = True
    optimization_governance_contract_implemented: Literal[True] = True
    optimization_explainability_contract_implemented: Literal[True] = True
    optimization_application_implemented: Literal[False] = False
    evolution_proposal_application_implemented: Literal[False] = False
    optimization_policy_mutation_implemented: Literal[False] = False
    capability_activation_implemented: Literal[False] = False
    agent_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    applied_optimization_signal_ids: tuple[str, ...] = Field(default_factory=tuple)
    applied_evolution_proposal_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_optimization_policy_ids: tuple[str, ...] = Field(default_factory=tuple)
    activated_capability_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _optimization_layer_matches_sources(self) -> Self:
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_count != len(self.capability_ids):
            raise ValueError("capability_count must match capability ids")
        if self.source_learning_signal_count != len(self.source_learning_signal_ids):
            raise ValueError("source_learning_signal_count must match signals")
        if self.source_optimization_proposal_count != len(
            self.source_optimization_proposal_ids
        ):
            raise ValueError("source_optimization_proposal_count must match proposals")
        if self.optimization_hitl_required_proposal_count != len(
            self.optimization_hitl_required_proposal_ids
        ):
            raise ValueError("optimization HITL proposal count must match proposals")
        if self.optimization_signal_ids != tuple(
            signal.optimization_signal_id for signal in self.optimization_signals
        ):
            raise ValueError("optimization_signal_ids must match signals")
        if self.optimization_signal_count != len(self.optimization_signals):
            raise ValueError("optimization_signal_count must match signals")
        if len(set(self.optimization_signal_ids)) != len(self.optimization_signal_ids):
            raise ValueError("optimization_signal_ids must be unique")
        declared_capabilities = set(self.capability_ids)
        declared_learning_signals = set(self.source_learning_signal_ids)
        declared_agents = set(self.linked_agent_ids)
        for signal in self.optimization_signals:
            if signal.capability_id not in declared_capabilities:
                raise ValueError("signal capability_id must be declared")
            if signal.learning_signal_id not in declared_learning_signals:
                raise ValueError("signal learning_signal_id must be declared")
            if (
                signal.source_optimization_proposal_ids
                != self.source_optimization_proposal_ids
            ):
                raise ValueError("signal proposal ids must match layer")
            if not set(signal.linked_agent_ids).issubset(declared_agents):
                raise ValueError("signal linked_agent_ids must be declared")
        if self.covered_roadmap_items != (
            CROSS_SYSTEM_OPTIMIZATION_LAYER_ROADMAP_ITEM,
        ):
            raise ValueError("covered_roadmap_items must be Task 8 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.applied_optimization_signal_ids,
                self.applied_evolution_proposal_ids,
                self.mutated_optimization_policy_ids,
                self.activated_capability_ids,
                self.emitted_hitl_request_ids,
            )
        ):
            raise ValueError(
                "optimization application, proposal application, mutation, "
                "and HITL ids must be empty",
            )
        if not all(signal.advisory_only for signal in self.optimization_signals):
            raise ValueError("all optimization signals must be advisory only")
        return self


def build_cross_system_optimization_layer(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    learning_layer: CrossSystemLearningLayerPlan | None = None,
    optimization_source: AutonomousOptimizationSuggestionsPlan | None = None,
) -> CrossSystemOptimizationLayerPlan:
    """Build advisory cross-system optimization metadata."""

    learning = learning_layer or build_cross_system_learning_layer(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    optimization = optimization_source or build_autonomous_optimization_suggestions(
        route=learning.route_name,
        task_type=learning.task_type,
        execution_mode_id=execution_mode_id,
    )
    signals = _optimization_signals(learning, optimization)
    return CrossSystemOptimizationLayerPlan(
        route_name=learning.route_name,
        task_type=learning.task_type,
        execution_mode_ids=learning.execution_mode_ids,
        learning_layer_role=learning.role,
        learning_layer_serialization_version=learning.serialization_version,
        optimization_source_role=optimization.role,
        optimization_source_serialization_version=optimization.serialization_version,
        layer_order=learning.layer_order,
        capabilities=learning.capabilities,
        capability_ids=learning.capability_ids,
        capability_count=learning.capability_count,
        source_learning_signal_ids=learning.learning_signal_ids,
        source_learning_signal_count=learning.learning_signal_count,
        source_optimization_proposal_ids=optimization.proposal_ids,
        source_optimization_proposal_count=optimization.proposal_count,
        optimization_hitl_required_proposal_ids=(
            optimization.hitl_required_proposal_ids
        ),
        optimization_hitl_required_proposal_count=(
            optimization.hitl_required_proposal_count
        ),
        optimization_signals=signals,
        optimization_signal_ids=tuple(
            signal.optimization_signal_id for signal in signals
        ),
        optimization_signal_count=len(signals),
        linked_agent_ids=learning.linked_agent_ids,
        covered_roadmap_items=(CROSS_SYSTEM_OPTIMIZATION_LAYER_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture="guarded",
    )


def cross_system_optimization_signal_by_id(
    optimization_signal_id: str,
    layer: CrossSystemOptimizationLayerPlan | None = None,
) -> CrossSystemOptimizationSignal | None:
    """Return one optimization signal without applying it."""

    source_layer = layer or build_cross_system_optimization_layer()
    for signal in source_layer.optimization_signals:
        if signal.optimization_signal_id == optimization_signal_id:
            return signal
    return None


def cross_system_optimization_signals_for_layer(
    cognitive_layer: CognitiveOSLayer,
    layer: CrossSystemOptimizationLayerPlan | None = None,
) -> tuple[CrossSystemOptimizationSignal, ...]:
    """Return optimization signals for one cognitive layer."""

    source_layer = layer or build_cross_system_optimization_layer()
    return tuple(
        signal
        for signal in source_layer.optimization_signals
        if signal.cognitive_layer == cognitive_layer
    )


def cross_system_optimization_signals_for_agent(
    agent_id: str,
    layer: CrossSystemOptimizationLayerPlan | None = None,
) -> tuple[CrossSystemOptimizationSignal, ...]:
    """Return optimization signals linked to one passive agent id."""

    source_layer = layer or build_cross_system_optimization_layer()
    return tuple(
        signal
        for signal in source_layer.optimization_signals
        if agent_id in signal.linked_agent_ids
    )


def _optimization_signals(
    learning_layer: CrossSystemLearningLayerPlan,
    optimization_source: AutonomousOptimizationSuggestionsPlan,
) -> tuple[CrossSystemOptimizationSignal, ...]:
    return tuple(
        CrossSystemOptimizationSignal(
            optimization_signal_id=(
                f"cross_system_optimization::{learning_signal.capability_id}"
            ),
            learning_signal_id=learning_signal.signal_id,
            capability_id=learning_signal.capability_id,
            capability_name=learning_signal.capability_name,
            cognitive_layer=learning_signal.cognitive_layer,
            linked_agent_ids=learning_signal.linked_agent_ids,
            source_optimization_proposal_ids=optimization_source.proposal_ids,
            source_optimization_proposal_count=optimization_source.proposal_count,
            optimization_posture=optimization_source.overall_evolution_posture,
            optimization_rank_score=optimization_source.overall_proposal_rank_score,
            dependency_contracts=(
                "optimization signal follows cross-system learning signal",
                f"learning signal:{learning_signal.signal_id}",
            ),
            governance_contracts=(
                "optimization signal does not authorize optimization application",
                "HITL required before any optimization-driven behavior",
            ),
            explanation_contracts=(
                "optimization signal cites V6.5 optimization proposals",
                "learning layer explains cross-system capability placement",
            ),
            blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        for learning_signal in learning_layer.learning_signals
    )
