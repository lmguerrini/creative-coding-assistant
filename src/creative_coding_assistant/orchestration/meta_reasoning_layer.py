"""V6.6 Meta-Reasoning Layer metadata."""

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
from creative_coding_assistant.orchestration.cognitive_profile_engine import (
    CognitiveProfileEnginePlan,
    build_cognitive_profile_engine,
)
from creative_coding_assistant.orchestration.routing import RouteName
from creative_coding_assistant.orchestration.routing_intelligence import (
    ExecutionModeId,
    TaskRoutingType,
)

META_REASONING_LAYER_SERIALIZATION_VERSION = "meta_reasoning_layer.v1"
META_REASONING_LAYER_ROADMAP_ITEM = "Meta-Reasoning Layer"
META_REASONING_LAYER_AUTHORITY_BOUNDARY = (
    "V6.6 Meta-Reasoning Layer assesses Cognitive OS profiles for reasoning "
    "coherence, dependency traceability, governance posture, explainability, "
    "and HITL readiness. It exposes reasoning assessment metadata for "
    "inspection only; it does not execute reasoning decisions, mutate "
    "reasoning chains, select agents, route providers or models, execute "
    "providers, control workflows, mutate generated output, emit HITL "
    "requests, apply HITL decisions, or apply Runtime Evolution."
)
META_REASONING_FOCUSES = (
    "profile coherence",
    "dependency trace quality",
    "governance posture",
    "explanation coverage",
    "HITL readiness",
)


class MetaReasoningAssessment(BaseModel):
    """One read-only meta-reasoning assessment for a cognitive profile."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    reasoning_id: str = Field(min_length=1, max_length=170)
    profile_id: str = Field(min_length=1, max_length=170)
    state_id: str = Field(min_length=1, max_length=160)
    capability_id: str = Field(min_length=1, max_length=80)
    capability_name: CognitiveOSCapability
    cognitive_layer: CognitiveOSLayer
    source_optimization_signal_id: str = Field(min_length=1, max_length=180)
    source_learning_signal_id: str = Field(min_length=1, max_length=180)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    reasoning_focuses: tuple[str, ...] = Field(min_length=5, max_length=5)
    profile_posture: CognitiveOSPosture
    reasoning_posture: CognitiveOSPosture
    reasoning_summary: str = Field(min_length=1, max_length=460)
    dependency_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    governance_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    explanation_contracts: tuple[str, ...] = Field(min_length=3, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _assessment_matches_sources_and_boundary(self) -> Self:
        expected_reasoning_id = f"meta_reasoning::{self.capability_id}"
        if self.reasoning_id != expected_reasoning_id:
            raise ValueError("reasoning_id must match capability_id")
        expected_profile_id = f"cognitive_profile::{self.capability_id}"
        if self.profile_id != expected_profile_id:
            raise ValueError("profile_id must match capability_id")
        expected_state_id = f"cognitive_state::{self.capability_id}"
        if self.state_id != expected_state_id:
            raise ValueError("state_id must match capability_id")
        expected_optimization_id = f"cross_system_optimization::{self.capability_id}"
        if self.source_optimization_signal_id != expected_optimization_id:
            raise ValueError("source_optimization_signal_id must match capability")
        expected_learning_id = f"cross_system_learning::{self.capability_id}"
        if self.source_learning_signal_id != expected_learning_id:
            raise ValueError("source_learning_signal_id must match capability")
        if self.reasoning_focuses != META_REASONING_FOCUSES:
            raise ValueError("reasoning_focuses must match V6.6 meta contract")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        return self


class MetaReasoningLayerPlan(BaseModel):
    """Read-only meta-reasoning layer over Cognitive OS profiles."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["meta_reasoning_layer"] = "meta_reasoning_layer"
    serialization_version: Literal["meta_reasoning_layer.v1"] = (
        META_REASONING_LAYER_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=META_REASONING_LAYER_AUTHORITY_BOUNDARY,
        max_length=1900,
    )
    route_name: RouteName
    task_type: TaskRoutingType
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    profile_engine_role: Literal["cognitive_profile_engine"]
    profile_engine_serialization_version: Literal["cognitive_profile_engine.v1"]
    state_engine_role: Literal["cognitive_state_engine"]
    layer_order: tuple[CognitiveOSLayer, ...] = Field(min_length=6, max_length=6)
    capabilities: tuple[CognitiveOSCapability, ...] = Field(min_length=6, max_length=6)
    capability_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    capability_count: int = Field(ge=6, le=6)
    source_profile_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_profile_count: int = Field(ge=6, le=6)
    source_state_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_state_count: int = Field(ge=6, le=6)
    source_optimization_signal_ids: tuple[str, ...] = Field(
        min_length=6,
        max_length=6,
    )
    source_optimization_signal_count: int = Field(ge=6, le=6)
    source_learning_signal_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    source_learning_signal_count: int = Field(ge=6, le=6)
    reasoning_assessments: tuple[MetaReasoningAssessment, ...] = Field(
        min_length=6,
        max_length=6,
    )
    reasoning_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    reasoning_count: int = Field(ge=6, le=6)
    linked_agent_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    covered_roadmap_items: tuple[str, ...] = Field(min_length=1, max_length=1)
    covered_roadmap_item_count: int = Field(ge=1, le=1)
    cross_cutting_contracts: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=12, max_length=12)
    graph_posture: CognitiveOSPosture
    meta_reasoning_layer_implemented: Literal[True] = True
    cognitive_profile_engine_integrated: Literal[True] = True
    reasoning_assessment_contract_implemented: Literal[True] = True
    reasoning_dependency_traceability_implemented: Literal[True] = True
    reasoning_governance_contract_implemented: Literal[True] = True
    reasoning_explainability_contract_implemented: Literal[True] = True
    autonomous_reasoning_execution_implemented: Literal[False] = False
    reasoning_chain_mutation_implemented: Literal[False] = False
    decision_authority_implemented: Literal[False] = False
    agent_routing_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    executed_reasoning_ids: tuple[str, ...] = Field(default_factory=tuple)
    mutated_reasoning_chain_ids: tuple[str, ...] = Field(default_factory=tuple)
    adopted_decision_ids: tuple[str, ...] = Field(default_factory=tuple)
    emitted_hitl_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _meta_reasoning_matches_sources(self) -> Self:
        if self.layer_order != COGNITIVE_OS_LAYER_ORDER:
            raise ValueError("layer_order must match V6.6 cognitive order")
        if self.capabilities != COGNITIVE_OS_CAPABILITIES:
            raise ValueError("capabilities must match V6.1 through V6.6")
        if self.capability_count != len(self.capability_ids):
            raise ValueError("capability_count must match capability ids")
        if self.source_profile_count != len(self.source_profile_ids):
            raise ValueError("source_profile_count must match profile ids")
        if self.source_state_count != len(self.source_state_ids):
            raise ValueError("source_state_count must match state ids")
        if self.source_optimization_signal_count != len(
            self.source_optimization_signal_ids
        ):
            raise ValueError("source_optimization_signal_count must match signals")
        if self.source_learning_signal_count != len(self.source_learning_signal_ids):
            raise ValueError("source_learning_signal_count must match signals")
        if self.reasoning_ids != tuple(
            assessment.reasoning_id for assessment in self.reasoning_assessments
        ):
            raise ValueError("reasoning_ids must match assessments")
        if self.reasoning_count != len(self.reasoning_assessments):
            raise ValueError("reasoning_count must match assessments")
        if len(set(self.reasoning_ids)) != len(self.reasoning_ids):
            raise ValueError("reasoning_ids must be unique")
        declared_capabilities = set(self.capability_ids)
        declared_profiles = set(self.source_profile_ids)
        declared_states = set(self.source_state_ids)
        declared_optimization_signals = set(self.source_optimization_signal_ids)
        declared_learning_signals = set(self.source_learning_signal_ids)
        declared_agents = set(self.linked_agent_ids)
        for assessment in self.reasoning_assessments:
            if assessment.capability_id not in declared_capabilities:
                raise ValueError("assessment capability_id must be declared")
            if assessment.profile_id not in declared_profiles:
                raise ValueError("assessment profile_id must be declared")
            if assessment.state_id not in declared_states:
                raise ValueError("assessment state_id must be declared")
            if (
                assessment.source_optimization_signal_id
                not in declared_optimization_signals
            ):
                raise ValueError("assessment optimization signal must be declared")
            if assessment.source_learning_signal_id not in declared_learning_signals:
                raise ValueError("assessment learning signal must be declared")
            if not set(assessment.linked_agent_ids).issubset(declared_agents):
                raise ValueError("assessment linked_agent_ids must be declared")
        if self.covered_roadmap_items != (META_REASONING_LAYER_ROADMAP_ITEM,):
            raise ValueError("covered_roadmap_items must be Task 11 only")
        if self.covered_roadmap_item_count != len(self.covered_roadmap_items):
            raise ValueError("covered_roadmap_item_count must match roadmap")
        if self.cross_cutting_contracts != COGNITIVE_OS_CONTRACTS:
            raise ValueError("cross_cutting_contracts must match V6.6 contracts")
        if self.blocked_runtime_behaviors != COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS:
            raise ValueError("blocked_runtime_behaviors must match V6.6 boundary")
        if any(
            (
                self.executed_reasoning_ids,
                self.mutated_reasoning_chain_ids,
                self.adopted_decision_ids,
                self.emitted_hitl_request_ids,
            )
        ):
            raise ValueError(
                "reasoning execution, mutation, decisions, and HITL ids must be empty",
            )
        if not all(
            assessment.advisory_only for assessment in self.reasoning_assessments
        ):
            raise ValueError("all meta-reasoning assessments must be advisory only")
        return self


def build_meta_reasoning_layer(
    *,
    route: RouteName | str = RouteName.GENERATE,
    task_type: TaskRoutingType | str = "reasoning",
    execution_mode_id: ExecutionModeId | str | None = None,
    profile_engine: CognitiveProfileEnginePlan | None = None,
) -> MetaReasoningLayerPlan:
    """Build read-only meta-reasoning metadata."""

    profiles = profile_engine or build_cognitive_profile_engine(
        route=route,
        task_type=task_type,
        execution_mode_id=execution_mode_id,
    )
    assessments = _meta_reasoning_assessments(profiles)
    return MetaReasoningLayerPlan(
        route_name=profiles.route_name,
        task_type=profiles.task_type,
        execution_mode_ids=profiles.execution_mode_ids,
        profile_engine_role=profiles.role,
        profile_engine_serialization_version=profiles.serialization_version,
        state_engine_role=profiles.state_engine_role,
        layer_order=profiles.layer_order,
        capabilities=profiles.capabilities,
        capability_ids=profiles.capability_ids,
        capability_count=profiles.capability_count,
        source_profile_ids=profiles.profile_ids,
        source_profile_count=profiles.profile_count,
        source_state_ids=profiles.source_state_ids,
        source_state_count=profiles.source_state_count,
        source_optimization_signal_ids=profiles.source_optimization_signal_ids,
        source_optimization_signal_count=profiles.source_optimization_signal_count,
        source_learning_signal_ids=profiles.source_learning_signal_ids,
        source_learning_signal_count=profiles.source_learning_signal_count,
        reasoning_assessments=assessments,
        reasoning_ids=tuple(assessment.reasoning_id for assessment in assessments),
        reasoning_count=len(assessments),
        linked_agent_ids=profiles.linked_agent_ids,
        covered_roadmap_items=(META_REASONING_LAYER_ROADMAP_ITEM,),
        covered_roadmap_item_count=1,
        cross_cutting_contracts=COGNITIVE_OS_CONTRACTS,
        blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        graph_posture=profiles.graph_posture,
    )


def meta_reasoning_assessment_by_id(
    reasoning_id: str,
    layer: MetaReasoningLayerPlan | None = None,
) -> MetaReasoningAssessment | None:
    """Return one meta-reasoning assessment without executing it."""

    source_layer = layer or build_meta_reasoning_layer()
    for assessment in source_layer.reasoning_assessments:
        if assessment.reasoning_id == reasoning_id:
            return assessment
    return None


def meta_reasoning_assessments_for_layer(
    cognitive_layer: CognitiveOSLayer,
    layer: MetaReasoningLayerPlan | None = None,
) -> tuple[MetaReasoningAssessment, ...]:
    """Return meta-reasoning assessments for one Cognitive OS layer."""

    source_layer = layer or build_meta_reasoning_layer()
    return tuple(
        assessment
        for assessment in source_layer.reasoning_assessments
        if assessment.cognitive_layer == cognitive_layer
    )


def meta_reasoning_assessments_for_agent(
    agent_id: str,
    layer: MetaReasoningLayerPlan | None = None,
) -> tuple[MetaReasoningAssessment, ...]:
    """Return meta-reasoning assessments linked to one agent."""

    source_layer = layer or build_meta_reasoning_layer()
    return tuple(
        assessment
        for assessment in source_layer.reasoning_assessments
        if agent_id in assessment.linked_agent_ids
    )


def _meta_reasoning_assessments(
    profile_engine: CognitiveProfileEnginePlan,
) -> tuple[MetaReasoningAssessment, ...]:
    return tuple(
        MetaReasoningAssessment(
            reasoning_id=f"meta_reasoning::{profile.capability_id}",
            profile_id=profile.profile_id,
            state_id=profile.state_id,
            capability_id=profile.capability_id,
            capability_name=profile.capability_name,
            cognitive_layer=profile.cognitive_layer,
            source_optimization_signal_id=profile.source_optimization_signal_id,
            source_learning_signal_id=profile.source_learning_signal_id,
            linked_agent_ids=profile.linked_agent_ids,
            reasoning_focuses=META_REASONING_FOCUSES,
            profile_posture=profile.profile_posture,
            reasoning_posture=profile.profile_posture,
            reasoning_summary=(
                f"Read-only meta-reasoning assessment for "
                f"{profile.capability_name}; evaluates {profile.profile_id} "
                "for coherence, dependency trace, governance, explanation, "
                "and HITL readiness without decision authority."
            ),
            dependency_contracts=(
                "meta-reasoning assessment follows cognitive profile",
                f"cognitive profile:{profile.profile_id}",
                f"state snapshot:{profile.state_id}",
            ),
            governance_contracts=(
                "meta-reasoning does not execute reasoning decisions",
                "meta-reasoning does not mutate reasoning chains",
                "HITL required before any reasoning-driven behavior",
            ),
            explanation_contracts=(
                "meta-reasoning cites profile and state sources",
                "meta-reasoning preserves capability and layer ownership",
                "meta-reasoning explains why no decision is applied",
            ),
            blocked_runtime_behaviors=COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        for profile in profile_engine.cognitive_profiles
    )
