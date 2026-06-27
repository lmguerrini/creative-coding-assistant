"""Passive hybrid agentic workflow metadata for V4.3 preparation."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

BackboneModePhase = Literal[
    "context_intake",
    "planning_reasoning",
    "generation_artifact",
    "review_refinement",
    "terminal_guardrail",
]
ConditionalEscalationCategory = Literal[
    "ambiguity",
    "risk",
    "runtime",
    "quality",
    "hitl",
]
SpecialistLoopCategory = Literal[
    "planning",
    "artifact",
    "runtime",
    "evaluation",
    "synthesis",
]
EscalationGateKind = Literal[
    "backbone_entry",
    "evidence_completeness",
    "specialist_loop_boundary",
    "human_review_visibility",
    "return_handoff",
]
CreativeEscalationPolicyCategory = Literal[
    "concept",
    "aesthetic",
    "runtime",
    "quality",
    "synthesis",
]

V3_BACKBONE_MODE_ID = "v3_backbone_mode"
V3_BACKBONE_MODE_NODE_SERIALIZATION_VERSION = "v3_backbone_mode_node.v1"
V3_BACKBONE_MODE_REGISTRY_SERIALIZATION_VERSION = "v3_backbone_mode_registry.v1"
CONDITIONAL_ESCALATION_CONDITION_SERIALIZATION_VERSION = (
    "conditional_multi_agent_escalation_condition.v1"
)
CONDITIONAL_ESCALATION_REGISTRY_SERIALIZATION_VERSION = (
    "conditional_multi_agent_escalation_registry.v1"
)
SPECIALIST_AGENT_LOOP_SERIALIZATION_VERSION = "specialist_agent_loop.v1"
SPECIALIST_AGENT_LOOP_REGISTRY_SERIALIZATION_VERSION = (
    "specialist_agent_loop_registry.v1"
)
ESCALATION_GATE_SERIALIZATION_VERSION = "escalation_gate.v1"
ESCALATION_GATE_REGISTRY_SERIALIZATION_VERSION = "escalation_gate_registry.v1"
CREATIVE_ESCALATION_POLICY_RULE_SERIALIZATION_VERSION = (
    "creative_escalation_policy_rule.v1"
)
CREATIVE_ESCALATION_POLICY_REGISTRY_SERIALIZATION_VERSION = (
    "creative_escalation_policy_registry.v1"
)
HYBRID_WORKFLOW_STAGE_SERIALIZATION_VERSION = "hybrid_workflow_stage.v1"
HYBRID_WORKFLOW_REGISTRY_SERIALIZATION_VERSION = "hybrid_workflow_registry.v1"
V3_BACKBONE_MODE_AUTHORITY_BOUNDARY = (
    "V3 Backbone Mode metadata declares the current deterministic V3 workflow "
    "graph as the active backbone for V4.3 hybrid workflow readiness only; it "
    "does not change workflow graph order, perform multi-agent escalation, "
    "invoke agents, route providers or models, select runtimes, trigger "
    "retries, mutate prompts, write memory, execute artifacts, or modify "
    "generated output."
)
CONDITIONAL_ESCALATION_AUTHORITY_BOUNDARY = (
    "Conditional multi-agent escalation metadata describes advisory candidate "
    "conditions that could prepare future escalation context from the V3 "
    "backbone; it does not evaluate conditions, invoke agents, route providers "
    "or models, select runtimes, control workflow transitions, trigger retries, "
    "execute voting, write memory, or modify generated output."
)
SPECIALIST_AGENT_LOOP_AUTHORITY_BOUNDARY = (
    "Specialist agent loop metadata describes bounded future loop candidates "
    "for known passive agent contracts only; it does not execute loops, invoke "
    "agents, coordinate multi-agent work, route providers or models, select "
    "runtimes, control workflow transitions, trigger retries, write memory, or "
    "modify generated output."
)
ESCALATION_GATE_AUTHORITY_BOUNDARY = (
    "Escalation gate metadata describes passive advisory gates across the V3 "
    "backbone, conditional escalation candidates, and specialist loop profiles "
    "only; it does not evaluate gates, approve escalation, invoke agents, "
    "control workflow transitions, route providers or models, trigger retries, "
    "execute artifacts, write memory, or modify generated output."
)
CREATIVE_ESCALATION_POLICY_AUTHORITY_BOUNDARY = (
    "Creative escalation policy metadata describes passive creative-domain "
    "escalation rules tied to advisory gates and specialist loop candidates "
    "only; it does not evaluate creative policy, approve escalation, invoke "
    "agents, route providers or models, control workflow transitions, trigger "
    "retries, execute artifacts, write memory, or modify generated output."
)
HYBRID_WORKFLOW_REGISTRY_AUTHORITY_BOUNDARY = (
    "Hybrid agentic workflow metadata maps current V3 workflow nodes to future "
    "agent capability and escalation policy readiness only; it does not change "
    "workflow graph order, create agents, route providers or models, select "
    "runtimes, trigger retries, execute artifacts, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "workflow_graph_mutation",
    "provider_or_model_routing",
    "runtime_selection",
    "retry_or_refinement_triggering",
    "agent_invocation",
    "artifact_execution",
    "generated_output_modification",
)
_V3_BACKBONE_MODE_BLOCKED_RUNTIME_BEHAVIORS = (
    "workflow_graph_mutation",
    "provider_or_model_routing",
    "runtime_selection",
    "retry_or_refinement_triggering",
    "multi_agent_escalation_execution",
    "agent_invocation",
    "prompt_mutation",
    "memory_write",
    "generated_output_modification",
)
_V3_BACKBONE_MODE_SOURCE_REGISTRIES = (
    "assistant_workflow_node_order",
    "workflow_step_order",
    "artifact_engine_contract_registry",
    "evaluation_engine_contract_registry",
    "workstation_engine_contract_registry",
    "hybrid_agentic_workflow_registry",
)
_CONDITIONAL_ESCALATION_BLOCKED_RUNTIME_BEHAVIORS = (
    "condition_evaluation",
    "multi_agent_execution",
    "provider_or_model_routing",
    "runtime_selection",
    "workflow_control",
    "retry_or_refinement_triggering",
    "agent_invocation",
    "voting_execution",
    "memory_write",
    "generated_output_modification",
)
_CONDITIONAL_ESCALATION_SOURCE_REGISTRIES = (
    "v3_backbone_mode_registry",
    "agent_capability_registry",
    "escalation_policy_registry",
    "agent_escalation_signal_registry",
    "hybrid_agentic_workflow_registry",
)
_SPECIALIST_AGENT_LOOP_BLOCKED_RUNTIME_BEHAVIORS = (
    "loop_execution",
    "agent_invocation",
    "multi_agent_orchestration",
    "provider_or_model_routing",
    "runtime_selection",
    "workflow_control",
    "retry_or_refinement_triggering",
    "memory_write",
    "generated_output_modification",
)
_SPECIALIST_AGENT_LOOP_SOURCE_REGISTRIES = (
    "agent_contract_registry",
    "conditional_multi_agent_escalation_registry",
    "v3_backbone_mode_registry",
    "hybrid_agentic_workflow_registry",
)
_ESCALATION_GATE_BLOCKED_RUNTIME_BEHAVIORS = (
    "gate_evaluation",
    "escalation_approval",
    "agent_invocation",
    "multi_agent_orchestration",
    "provider_or_model_routing",
    "runtime_selection",
    "workflow_control",
    "retry_or_refinement_triggering",
    "artifact_execution",
    "memory_write",
    "generated_output_modification",
)
_ESCALATION_GATE_SOURCE_REGISTRIES = (
    "v3_backbone_mode_registry",
    "conditional_multi_agent_escalation_registry",
    "specialist_agent_loop_registry",
    "escalation_policy_registry",
    "hybrid_agentic_workflow_registry",
)
_CREATIVE_ESCALATION_POLICY_BLOCKED_RUNTIME_BEHAVIORS = (
    "creative_policy_evaluation",
    "escalation_approval",
    "gate_evaluation",
    "agent_invocation",
    "provider_or_model_routing",
    "runtime_selection",
    "workflow_control",
    "retry_or_refinement_triggering",
    "artifact_execution",
    "memory_write",
    "generated_output_modification",
)
_CREATIVE_ESCALATION_POLICY_SOURCE_REGISTRIES = (
    "escalation_gate_registry",
    "specialist_agent_loop_registry",
    "conditional_multi_agent_escalation_registry",
    "artifact_engine_contract_registry",
    "evaluation_engine_contract_registry",
    "hybrid_agentic_workflow_registry",
)
_V3_BACKBONE_MODE_PHASE_IDS: tuple[BackboneModePhase, ...] = (
    "context_intake",
    "planning_reasoning",
    "generation_artifact",
    "review_refinement",
    "terminal_guardrail",
)
_CONDITIONAL_ESCALATION_CATEGORIES: tuple[ConditionalEscalationCategory, ...] = (
    "ambiguity",
    "risk",
    "runtime",
    "quality",
    "hitl",
)
_SPECIALIST_AGENT_LOOP_CATEGORIES: tuple[SpecialistLoopCategory, ...] = (
    "planning",
    "artifact",
    "runtime",
    "evaluation",
    "synthesis",
)
_ESCALATION_GATE_KINDS: tuple[EscalationGateKind, ...] = (
    "backbone_entry",
    "evidence_completeness",
    "specialist_loop_boundary",
    "human_review_visibility",
    "return_handoff",
)
_CREATIVE_ESCALATION_POLICY_CATEGORIES: tuple[
    CreativeEscalationPolicyCategory, ...
] = (
    "concept",
    "aesthetic",
    "runtime",
    "quality",
    "synthesis",
)
_KNOWN_SPECIALIST_AGENT_IDS = (
    "planner_agent",
    "research_agent",
    "style_agent",
    "runtime_agent",
    "artifact_agent",
    "art_direction_agent",
    "aesthetic_critic_agent",
    "narrative_symbolic_agent",
    "creative_curator_agent",
    "critic_agent",
    "refiner_agent",
    "final_synthesizer_agent",
)
_KNOWN_CONDITIONAL_ESCALATION_CAPABILITY_IDS = (
    "v4_planner_agent",
    "v4_artifact_agent",
    "v4_runtime_agent",
    "v4_agent_router",
    "v4_agentic_studio",
    "adaptive_multi_agent_escalation",
)
_KNOWN_CONDITIONAL_ESCALATION_POLICY_RULE_IDS = (
    "missing_information_review",
    "artifact_risk_review",
    "runtime_incompatibility_review",
    "evaluation_confidence_review",
    "future_agent_escalation_readiness",
)
_KNOWN_CONDITIONAL_ESCALATION_SIGNAL_IDS = (
    "confidence_escalation_signal",
    "risk_escalation_signal",
    "ambiguity_escalation_signal",
    "cost_escalation_signal",
    "latency_escalation_signal",
    "quality_escalation_signal",
    "hitl_escalation_signal",
)


class V3BackboneModeNodeProfile(BaseModel):
    """Metadata-only profile for one preserved V3 workflow backbone node."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    mode_id: Literal["v3_backbone_mode"] = V3_BACKBONE_MODE_ID
    node_id: str = Field(min_length=1, max_length=80)
    phase: BackboneModePhase
    active_runtime_owner: Literal["v3_workflow_graph"] = "v3_workflow_graph"
    preserved_surfaces: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_V3_BACKBONE_MODE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    backbone_runtime_active: Literal[True] = True
    workflow_order_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    multi_agent_escalation_executed: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["v3_backbone_mode_node.v1"] = (
        V3_BACKBONE_MODE_NODE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class V3BackboneModeRegistry(BaseModel):
    """Stable passive registry declaring the preserved V3 workflow backbone."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["v3_backbone_mode_registry"] = "v3_backbone_mode_registry"
    mode_id: Literal["v3_backbone_mode"] = V3_BACKBONE_MODE_ID
    serialization_version: Literal["v3_backbone_mode_registry.v1"] = (
        V3_BACKBONE_MODE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=V3_BACKBONE_MODE_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    node_profiles: tuple[V3BackboneModeNodeProfile, ...] = Field(
        min_length=18,
        max_length=18,
    )
    node_ids: tuple[str, ...] = Field(min_length=18, max_length=18)
    preserved_workflow_order: tuple[str, ...] = Field(min_length=18, max_length=18)
    phase_ids: tuple[BackboneModePhase, ...] = Field(min_length=5, max_length=5)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    node_count: int = Field(ge=18, le=18)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_V3_BACKBONE_MODE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    backbone_runtime_active: Literal[True] = True
    workflow_order_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    multi_agent_escalation_executed: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_backbone_profiles(self) -> Self:
        derived_node_ids = tuple(profile.node_id for profile in self.node_profiles)
        derived_phase_ids = tuple(
            dict.fromkeys(profile.phase for profile in self.node_profiles)
        )
        if self.node_ids != derived_node_ids:
            raise ValueError("node_ids must match node_profiles")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("node_ids must be unique")
        if self.preserved_workflow_order != self.node_ids:
            raise ValueError("preserved_workflow_order must match node_ids")
        if self.phase_ids != derived_phase_ids:
            raise ValueError("phase_ids must match node profile phases")
        if self.node_count != len(self.node_profiles):
            raise ValueError("node_count must match node_profiles")

        source_registries = set(self.source_registries)
        profile_source_registries = {
            source_registry
            for profile in self.node_profiles
            for source_registry in profile.source_registries
        }
        if source_registries != profile_source_registries:
            raise ValueError("source_registries must match node profile sources")
        for profile in self.node_profiles:
            if profile.source_registries != self.source_registries:
                raise ValueError("node profile sources must match registry sources")
            if profile.workflow_order_mutation_implemented:
                raise ValueError("V3 backbone profiles must not mutate workflow order")
            if profile.multi_agent_escalation_executed:
                raise ValueError("V3 backbone profiles must not execute escalation")
        return self


def v3_backbone_mode_registry() -> V3BackboneModeRegistry:
    """Return passive V3 backbone mode metadata without changing workflow behavior."""

    return V3_BACKBONE_MODE_REGISTRY


def v3_backbone_mode_profile_by_node_id(
    node_id: str,
    registry: V3BackboneModeRegistry | None = None,
) -> V3BackboneModeNodeProfile | None:
    """Return one V3 backbone node profile without executing workflow changes."""

    source_registry = registry or V3_BACKBONE_MODE_REGISTRY
    for profile in source_registry.node_profiles:
        if profile.node_id == node_id:
            return profile
    return None


class ConditionalMultiAgentEscalationCondition(BaseModel):
    """Passive condition metadata for future multi-agent escalation candidates."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    condition_id: str = Field(min_length=1, max_length=120)
    condition_name: str = Field(min_length=1, max_length=160)
    category: ConditionalEscalationCategory
    backbone_phase: BackboneModePhase
    source_node_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    source_registries: tuple[str, ...] = Field(min_length=5, max_length=5)
    capability_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    policy_rule_ids: tuple[str, ...] = Field(min_length=1, max_length=3)
    escalation_signal_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_CONDITIONAL_ESCALATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    condition_evaluation_implemented: Literal[False] = False
    multi_agent_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal[
        "conditional_multi_agent_escalation_condition.v1"
    ] = CONDITIONAL_ESCALATION_CONDITION_SERIALIZATION_VERSION
    metadata_only: Literal[True] = True


class ConditionalMultiAgentEscalationRegistry(BaseModel):
    """Stable passive registry for conditional multi-agent escalation metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["conditional_multi_agent_escalation_registry"] = (
        "conditional_multi_agent_escalation_registry"
    )
    serialization_version: Literal[
        "conditional_multi_agent_escalation_registry.v1"
    ] = CONDITIONAL_ESCALATION_REGISTRY_SERIALIZATION_VERSION
    authority_boundary: str = Field(
        default=CONDITIONAL_ESCALATION_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    conditions: tuple[ConditionalMultiAgentEscalationCondition, ...] = Field(
        min_length=5,
        max_length=5,
    )
    condition_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    categories: tuple[ConditionalEscalationCategory, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_registries: tuple[str, ...] = Field(min_length=5, max_length=5)
    backbone_node_ids: tuple[str, ...] = Field(min_length=18, max_length=18)
    condition_count: int = Field(ge=5, le=5)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_CONDITIONAL_ESCALATION_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    condition_evaluation_implemented: Literal[False] = False
    multi_agent_execution_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_conditional_escalation_metadata(self) -> Self:
        derived_condition_ids = tuple(
            condition.condition_id for condition in self.conditions
        )
        derived_categories = tuple(condition.category for condition in self.conditions)
        if self.condition_ids != derived_condition_ids:
            raise ValueError("condition_ids must match conditions")
        if len(set(self.condition_ids)) != len(self.condition_ids):
            raise ValueError("condition_ids must be unique")
        if self.categories != derived_categories:
            raise ValueError("categories must match conditions")
        if self.condition_count != len(self.conditions):
            raise ValueError("condition_count must match conditions")

        source_registries = set(self.source_registries)
        condition_source_registries = {
            source_registry
            for condition in self.conditions
            for source_registry in condition.source_registries
        }
        if source_registries != condition_source_registries:
            raise ValueError("source_registries must match condition sources")

        known_nodes = set(self.backbone_node_ids)
        known_capabilities = set(_KNOWN_CONDITIONAL_ESCALATION_CAPABILITY_IDS)
        known_policies = set(_KNOWN_CONDITIONAL_ESCALATION_POLICY_RULE_IDS)
        known_signals = set(_KNOWN_CONDITIONAL_ESCALATION_SIGNAL_IDS)
        for condition in self.conditions:
            if condition.source_registries != self.source_registries:
                raise ValueError("condition sources must match registry sources")
            if not set(condition.source_node_ids).issubset(known_nodes):
                raise ValueError("condition source nodes must be V3 backbone nodes")
            if not set(condition.capability_ids).issubset(known_capabilities):
                raise ValueError("condition capabilities must be known metadata")
            if not set(condition.policy_rule_ids).issubset(known_policies):
                raise ValueError("condition policy rules must be known metadata")
            if not set(condition.escalation_signal_ids).issubset(known_signals):
                raise ValueError("condition signals must be known metadata")
            if condition.condition_evaluation_implemented:
                raise ValueError("conditions must remain unevaluated metadata")
            if condition.multi_agent_execution_implemented:
                raise ValueError("conditions must not execute multi-agent workflow")
        return self


def conditional_multi_agent_escalation_registry() -> (
    ConditionalMultiAgentEscalationRegistry
):
    """Return passive conditional escalation metadata without invoking agents."""

    return CONDITIONAL_MULTI_AGENT_ESCALATION_REGISTRY


def conditional_multi_agent_escalation_condition_by_id(
    condition_id: str,
    registry: ConditionalMultiAgentEscalationRegistry | None = None,
) -> ConditionalMultiAgentEscalationCondition | None:
    """Return one escalation condition without evaluating it."""

    source_registry = registry or CONDITIONAL_MULTI_AGENT_ESCALATION_REGISTRY
    for condition in source_registry.conditions:
        if condition.condition_id == condition_id:
            return condition
    return None


class SpecialistAgentLoopProfile(BaseModel):
    """Passive profile for one future specialist-agent loop candidate."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    loop_id: str = Field(min_length=1, max_length=120)
    loop_name: str = Field(min_length=1, max_length=160)
    category: SpecialistLoopCategory
    specialist_agent_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    source_condition_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_node_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_registries: tuple[str, ...] = Field(min_length=4, max_length=4)
    loop_inputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    max_advisory_passes: int = Field(ge=1, le=3)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_SPECIALIST_AGENT_LOOP_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    loop_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    multi_agent_orchestration_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["specialist_agent_loop.v1"] = (
        SPECIALIST_AGENT_LOOP_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class SpecialistAgentLoopRegistry(BaseModel):
    """Stable passive registry for specialist-agent loop metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["specialist_agent_loop_registry"] = (
        "specialist_agent_loop_registry"
    )
    serialization_version: Literal["specialist_agent_loop_registry.v1"] = (
        SPECIALIST_AGENT_LOOP_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=SPECIALIST_AGENT_LOOP_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    loops: tuple[SpecialistAgentLoopProfile, ...] = Field(
        min_length=5,
        max_length=5,
    )
    loop_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    categories: tuple[SpecialistLoopCategory, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_registries: tuple[str, ...] = Field(min_length=4, max_length=4)
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    condition_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    backbone_node_ids: tuple[str, ...] = Field(min_length=18, max_length=18)
    loop_count: int = Field(ge=5, le=5)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_SPECIALIST_AGENT_LOOP_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    loop_execution_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    multi_agent_orchestration_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_specialist_loop_metadata(self) -> Self:
        derived_loop_ids = tuple(loop.loop_id for loop in self.loops)
        derived_categories = tuple(loop.category for loop in self.loops)
        if self.loop_ids != derived_loop_ids:
            raise ValueError("loop_ids must match loops")
        if len(set(self.loop_ids)) != len(self.loop_ids):
            raise ValueError("loop_ids must be unique")
        if self.categories != derived_categories:
            raise ValueError("categories must match loops")
        if self.loop_count != len(self.loops):
            raise ValueError("loop_count must match loops")

        source_registries = set(self.source_registries)
        loop_source_registries = {
            source_registry
            for loop in self.loops
            for source_registry in loop.source_registries
        }
        if source_registries != loop_source_registries:
            raise ValueError("source_registries must match loop sources")

        known_agents = set(self.agent_ids)
        known_conditions = set(self.condition_ids)
        known_nodes = set(self.backbone_node_ids)
        for loop in self.loops:
            if loop.source_registries != self.source_registries:
                raise ValueError("loop sources must match registry sources")
            if not set(loop.specialist_agent_ids).issubset(known_agents):
                raise ValueError("loop agents must be known passive agents")
            if not set(loop.source_condition_ids).issubset(known_conditions):
                raise ValueError("loop conditions must be known metadata")
            if not set(loop.source_node_ids).issubset(known_nodes):
                raise ValueError("loop source nodes must be V3 backbone nodes")
            if loop.loop_execution_implemented:
                raise ValueError("specialist loops must not execute")
            if loop.agent_invocation_implemented:
                raise ValueError("specialist loops must not invoke agents")
        return self


def specialist_agent_loop_registry() -> SpecialistAgentLoopRegistry:
    """Return passive specialist loop metadata without executing loops."""

    return SPECIALIST_AGENT_LOOP_REGISTRY


def specialist_agent_loop_by_id(
    loop_id: str,
    registry: SpecialistAgentLoopRegistry | None = None,
) -> SpecialistAgentLoopProfile | None:
    """Return one specialist loop profile without invoking agents."""

    source_registry = registry or SPECIALIST_AGENT_LOOP_REGISTRY
    for loop in source_registry.loops:
        if loop.loop_id == loop_id:
            return loop
    return None


class EscalationGateProfile(BaseModel):
    """Passive advisory gate metadata for future escalation readiness."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    gate_id: str = Field(min_length=1, max_length=120)
    gate_name: str = Field(min_length=1, max_length=160)
    gate_kind: EscalationGateKind
    source_condition_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_loop_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    source_registries: tuple[str, ...] = Field(min_length=5, max_length=5)
    required_passive_inputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_decision_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_ESCALATION_GATE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    gate_evaluation_implemented: Literal[False] = False
    escalation_approval_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["escalation_gate.v1"] = (
        ESCALATION_GATE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class EscalationGateRegistry(BaseModel):
    """Stable passive registry for V4.3 escalation gate metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["escalation_gate_registry"] = "escalation_gate_registry"
    serialization_version: Literal["escalation_gate_registry.v1"] = (
        ESCALATION_GATE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=ESCALATION_GATE_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    gates: tuple[EscalationGateProfile, ...] = Field(min_length=5, max_length=5)
    gate_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    gate_kinds: tuple[EscalationGateKind, ...] = Field(min_length=5, max_length=5)
    source_registries: tuple[str, ...] = Field(min_length=5, max_length=5)
    condition_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    loop_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    gate_count: int = Field(ge=5, le=5)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_ESCALATION_GATE_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    gate_evaluation_implemented: Literal[False] = False
    escalation_approval_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_escalation_gate_metadata(self) -> Self:
        derived_gate_ids = tuple(gate.gate_id for gate in self.gates)
        derived_gate_kinds = tuple(gate.gate_kind for gate in self.gates)
        if self.gate_ids != derived_gate_ids:
            raise ValueError("gate_ids must match gates")
        if len(set(self.gate_ids)) != len(self.gate_ids):
            raise ValueError("gate_ids must be unique")
        if self.gate_kinds != derived_gate_kinds:
            raise ValueError("gate_kinds must match gates")
        if self.gate_count != len(self.gates):
            raise ValueError("gate_count must match gates")

        source_registries = set(self.source_registries)
        gate_source_registries = {
            source_registry
            for gate in self.gates
            for source_registry in gate.source_registries
        }
        if source_registries != gate_source_registries:
            raise ValueError("source_registries must match gate sources")

        known_conditions = set(self.condition_ids)
        known_loops = set(self.loop_ids)
        for gate in self.gates:
            if gate.source_registries != self.source_registries:
                raise ValueError("gate sources must match registry sources")
            if not set(gate.source_condition_ids).issubset(known_conditions):
                raise ValueError("gate conditions must be known metadata")
            if not set(gate.source_loop_ids).issubset(known_loops):
                raise ValueError("gate loops must be known metadata")
            if gate.gate_evaluation_implemented:
                raise ValueError("escalation gates must not evaluate")
            if gate.escalation_approval_implemented:
                raise ValueError("escalation gates must not approve escalation")
        return self


def escalation_gate_registry() -> EscalationGateRegistry:
    """Return passive escalation gate metadata without evaluating gates."""

    return ESCALATION_GATE_REGISTRY


def escalation_gate_by_id(
    gate_id: str,
    registry: EscalationGateRegistry | None = None,
) -> EscalationGateProfile | None:
    """Return one escalation gate profile without evaluating it."""

    source_registry = registry or ESCALATION_GATE_REGISTRY
    for gate in source_registry.gates:
        if gate.gate_id == gate_id:
            return gate
    return None


class CreativeEscalationPolicyRule(BaseModel):
    """Passive creative-domain escalation policy rule metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    policy_id: str = Field(min_length=1, max_length=120)
    policy_name: str = Field(min_length=1, max_length=160)
    category: CreativeEscalationPolicyCategory
    source_gate_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_loop_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    creative_signal_sources: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_policy_outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(min_length=1, max_length=900)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_CREATIVE_ESCALATION_POLICY_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    creative_policy_evaluation_implemented: Literal[False] = False
    escalation_approval_implemented: Literal[False] = False
    gate_evaluation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["creative_escalation_policy_rule.v1"] = (
        CREATIVE_ESCALATION_POLICY_RULE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class CreativeEscalationPolicyRegistry(BaseModel):
    """Stable passive registry for creative escalation policy metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_escalation_policy_registry"] = (
        "creative_escalation_policy_registry"
    )
    serialization_version: Literal["creative_escalation_policy_registry.v1"] = (
        CREATIVE_ESCALATION_POLICY_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CREATIVE_ESCALATION_POLICY_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    policies: tuple[CreativeEscalationPolicyRule, ...] = Field(
        min_length=5,
        max_length=5,
    )
    policy_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    categories: tuple[CreativeEscalationPolicyCategory, ...] = Field(
        min_length=5,
        max_length=5,
    )
    source_registries: tuple[str, ...] = Field(min_length=6, max_length=6)
    gate_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    loop_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    policy_count: int = Field(ge=5, le=5)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_CREATIVE_ESCALATION_POLICY_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    creative_policy_evaluation_implemented: Literal[False] = False
    escalation_approval_implemented: Literal[False] = False
    gate_evaluation_implemented: Literal[False] = False
    agent_invocation_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_creative_policy_metadata(self) -> Self:
        derived_policy_ids = tuple(policy.policy_id for policy in self.policies)
        derived_categories = tuple(policy.category for policy in self.policies)
        if self.policy_ids != derived_policy_ids:
            raise ValueError("policy_ids must match policies")
        if len(set(self.policy_ids)) != len(self.policy_ids):
            raise ValueError("policy_ids must be unique")
        if self.categories != derived_categories:
            raise ValueError("categories must match policies")
        if self.policy_count != len(self.policies):
            raise ValueError("policy_count must match policies")

        source_registries = set(self.source_registries)
        policy_source_registries = {
            source_registry
            for policy in self.policies
            for source_registry in policy.source_registries
        }
        if source_registries != policy_source_registries:
            raise ValueError("source_registries must match policy sources")

        known_gates = set(self.gate_ids)
        known_loops = set(self.loop_ids)
        for policy in self.policies:
            if policy.source_registries != self.source_registries:
                raise ValueError("policy sources must match registry sources")
            if not set(policy.source_gate_ids).issubset(known_gates):
                raise ValueError("policy gates must be known metadata")
            if not set(policy.source_loop_ids).issubset(known_loops):
                raise ValueError("policy loops must be known metadata")
            if policy.creative_policy_evaluation_implemented:
                raise ValueError("creative policies must not evaluate policy")
            if policy.escalation_approval_implemented:
                raise ValueError("creative policies must not approve escalation")
        return self


def creative_escalation_policy_registry() -> CreativeEscalationPolicyRegistry:
    """Return passive creative escalation policy metadata."""

    return CREATIVE_ESCALATION_POLICY_REGISTRY


def creative_escalation_policy_by_id(
    policy_id: str,
    registry: CreativeEscalationPolicyRegistry | None = None,
) -> CreativeEscalationPolicyRule | None:
    """Return one creative policy rule without evaluating it."""

    source_registry = registry or CREATIVE_ESCALATION_POLICY_REGISTRY
    for policy in source_registry.policies:
        if policy.policy_id == policy_id:
            return policy
    return None


class HybridAgenticWorkflowStage(BaseModel):
    """Metadata-only future hybrid workflow readiness stage."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    stage_id: str = Field(min_length=1, max_length=80)
    stage_name: str = Field(min_length=1, max_length=140)
    authority_boundary: str = Field(min_length=1, max_length=900)
    v3_workflow_nodes: tuple[str, ...] = Field(min_length=1, max_length=8)
    future_capability_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    escalation_rule_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_metadata_registries: tuple[str, ...] = Field(min_length=1, max_length=6)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(min_length=1, max_length=12)
    serialization_version: Literal["hybrid_workflow_stage.v1"] = (
        HYBRID_WORKFLOW_STAGE_SERIALIZATION_VERSION
    )


class HybridAgenticWorkflowRegistry(BaseModel):
    """Stable metadata registry for future hybrid agentic workflow readiness."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["hybrid_agentic_workflow_registry"] = (
        "hybrid_agentic_workflow_registry"
    )
    serialization_version: Literal["hybrid_workflow_registry.v1"] = (
        HYBRID_WORKFLOW_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=HYBRID_WORKFLOW_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    stages: tuple[HybridAgenticWorkflowStage, ...] = Field(
        min_length=5,
        max_length=5,
    )
    stage_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    stage_count: int = Field(ge=5, le=5)
    source_metadata_registries: tuple[str, ...] = Field(min_length=5, max_length=5)
    metadata_only: Literal[True] = True


def hybrid_agentic_workflow_registry() -> HybridAgenticWorkflowRegistry:
    """Return the static future hybrid workflow readiness registry."""

    return HYBRID_AGENTIC_WORKFLOW_REGISTRY


def hybrid_agentic_workflow_stage_by_id(
    stage_id: str,
) -> HybridAgenticWorkflowStage | None:
    """Return one hybrid workflow readiness stage without changing behavior."""

    for stage in HYBRID_AGENTIC_WORKFLOW_STAGES:
        if stage.stage_id == stage_id:
            return stage
    return None


def _backbone_profile(
    *,
    node_id: str,
    phase: BackboneModePhase,
    preserved_surfaces: tuple[str, ...],
) -> V3BackboneModeNodeProfile:
    return V3BackboneModeNodeProfile(
        node_id=node_id,
        phase=phase,
        preserved_surfaces=preserved_surfaces,
        source_registries=_V3_BACKBONE_MODE_SOURCE_REGISTRIES,
        authority_boundary=(
            "This node remains part of the deterministic V3 workflow backbone; "
            "its metadata does not change workflow order, perform multi-agent "
            "escalation, invoke agents, route providers or models, mutate "
            "prompts, write memory, or modify generated output."
        ),
    )


def _conditional_escalation_condition(
    *,
    condition_id: str,
    condition_name: str,
    category: ConditionalEscalationCategory,
    backbone_phase: BackboneModePhase,
    source_node_ids: tuple[str, ...],
    capability_ids: tuple[str, ...],
    policy_rule_ids: tuple[str, ...],
    escalation_signal_ids: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> ConditionalMultiAgentEscalationCondition:
    return ConditionalMultiAgentEscalationCondition(
        condition_id=condition_id,
        condition_name=condition_name,
        category=category,
        backbone_phase=backbone_phase,
        source_node_ids=source_node_ids,
        source_registries=_CONDITIONAL_ESCALATION_SOURCE_REGISTRIES,
        capability_ids=capability_ids,
        policy_rule_ids=policy_rule_ids,
        escalation_signal_ids=escalation_signal_ids,
        advisory_outputs=advisory_outputs,
        authority_boundary=(
            "This condition is advisory escalation metadata only; it does not "
            "evaluate conditions, invoke agents, route providers or models, "
            "control workflow transitions, trigger retries, execute voting, "
            "write memory, or modify generated output."
        ),
    )


def _specialist_agent_loop(
    *,
    loop_id: str,
    loop_name: str,
    category: SpecialistLoopCategory,
    specialist_agent_ids: tuple[str, ...],
    source_condition_ids: tuple[str, ...],
    source_node_ids: tuple[str, ...],
    loop_inputs: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
    max_advisory_passes: int,
) -> SpecialistAgentLoopProfile:
    return SpecialistAgentLoopProfile(
        loop_id=loop_id,
        loop_name=loop_name,
        category=category,
        specialist_agent_ids=specialist_agent_ids,
        source_condition_ids=source_condition_ids,
        source_node_ids=source_node_ids,
        source_registries=_SPECIALIST_AGENT_LOOP_SOURCE_REGISTRIES,
        loop_inputs=loop_inputs,
        advisory_outputs=advisory_outputs,
        max_advisory_passes=max_advisory_passes,
        authority_boundary=(
            "This specialist loop is advisory metadata only; it does not "
            "execute loops, invoke agents, coordinate multi-agent work, route "
            "providers or models, control workflow transitions, trigger "
            "retries, write memory, or modify generated output."
        ),
    )


def _escalation_gate(
    *,
    gate_id: str,
    gate_name: str,
    gate_kind: EscalationGateKind,
    source_condition_ids: tuple[str, ...],
    source_loop_ids: tuple[str, ...],
    required_passive_inputs: tuple[str, ...],
    advisory_decision_outputs: tuple[str, ...],
) -> EscalationGateProfile:
    return EscalationGateProfile(
        gate_id=gate_id,
        gate_name=gate_name,
        gate_kind=gate_kind,
        source_condition_ids=source_condition_ids,
        source_loop_ids=source_loop_ids,
        source_registries=_ESCALATION_GATE_SOURCE_REGISTRIES,
        required_passive_inputs=required_passive_inputs,
        advisory_decision_outputs=advisory_decision_outputs,
        authority_boundary=(
            "This gate is advisory metadata only; it does not evaluate gates, "
            "approve escalation, invoke agents, route providers or models, "
            "control workflow transitions, trigger retries, execute artifacts, "
            "write memory, or modify generated output."
        ),
    )


def _creative_escalation_policy(
    *,
    policy_id: str,
    policy_name: str,
    category: CreativeEscalationPolicyCategory,
    source_gate_ids: tuple[str, ...],
    source_loop_ids: tuple[str, ...],
    creative_signal_sources: tuple[str, ...],
    advisory_policy_outputs: tuple[str, ...],
) -> CreativeEscalationPolicyRule:
    return CreativeEscalationPolicyRule(
        policy_id=policy_id,
        policy_name=policy_name,
        category=category,
        source_gate_ids=source_gate_ids,
        source_loop_ids=source_loop_ids,
        source_registries=_CREATIVE_ESCALATION_POLICY_SOURCE_REGISTRIES,
        creative_signal_sources=creative_signal_sources,
        advisory_policy_outputs=advisory_policy_outputs,
        authority_boundary=(
            "This creative escalation policy is advisory metadata only; it "
            "does not evaluate creative policy, approve escalation, evaluate "
            "gates, invoke agents, control workflow transitions, trigger "
            "retries, execute artifacts, write memory, or modify generated "
            "output."
        ),
    )


def _stage(
    *,
    stage_id: str,
    stage_name: str,
    v3_workflow_nodes: tuple[str, ...],
    future_capability_ids: tuple[str, ...],
    escalation_rule_ids: tuple[str, ...],
    advisory_outputs: tuple[str, ...],
) -> HybridAgenticWorkflowStage:
    return HybridAgenticWorkflowStage(
        stage_id=stage_id,
        stage_name=stage_name,
        authority_boundary=(
            "This stage is future hybrid workflow readiness metadata only; it "
            "does not change V3 workflow graph order, create agents, route "
            "providers or models, select runtimes, trigger retries, execute "
            "artifacts, or modify generated output."
        ),
        v3_workflow_nodes=v3_workflow_nodes,
        future_capability_ids=future_capability_ids,
        escalation_rule_ids=escalation_rule_ids,
        source_metadata_registries=(
            "agent_capability_registry",
            "escalation_policy_registry",
            "artifact_engine_contract_registry",
            "evaluation_engine_contract_registry",
            "workstation_engine_contract_registry",
        ),
        advisory_outputs=advisory_outputs,
        blocked_runtime_behaviors=_BLOCKED_RUNTIME_BEHAVIORS,
    )


V3_BACKBONE_MODE_NODE_PROFILES = (
    _backbone_profile(
        node_id="intake",
        phase="context_intake",
        preserved_surfaces=("request_received_event", "workflow_state"),
    ),
    _backbone_profile(
        node_id="routing",
        phase="context_intake",
        preserved_surfaces=("route_decision", "workflow_transition"),
    ),
    _backbone_profile(
        node_id="memory",
        phase="context_intake",
        preserved_surfaces=("memory_context", "project_memory_context"),
    ),
    _backbone_profile(
        node_id="retrieval",
        phase="context_intake",
        preserved_surfaces=("retrieval_context", "knowledge_chunks"),
    ),
    _backbone_profile(
        node_id="context_assembly",
        phase="context_intake",
        preserved_surfaces=("assembled_context", "context_summary"),
    ),
    _backbone_profile(
        node_id="prompt_input",
        phase="planning_reasoning",
        preserved_surfaces=("prompt_input", "operator_request_shape"),
    ),
    _backbone_profile(
        node_id="planning",
        phase="planning_reasoning",
        preserved_surfaces=("creative_plan", "planning_metadata"),
    ),
    _backbone_profile(
        node_id="director",
        phase="planning_reasoning",
        preserved_surfaces=("director_brief", "creative_direction"),
    ),
    _backbone_profile(
        node_id="reasoning",
        phase="planning_reasoning",
        preserved_surfaces=("creative_reasoning", "reasoning_evidence"),
    ),
    _backbone_profile(
        node_id="prompt_rendering",
        phase="planning_reasoning",
        preserved_surfaces=("rendered_prompt", "prompt_sections"),
    ),
    _backbone_profile(
        node_id="generation",
        phase="generation_artifact",
        preserved_surfaces=("generation_stream", "provider_generation_request"),
    ),
    _backbone_profile(
        node_id="artifact_extraction",
        phase="generation_artifact",
        preserved_surfaces=("workflow_artifacts", "artifact_metadata"),
    ),
    _backbone_profile(
        node_id="preview_preparation",
        phase="generation_artifact",
        preserved_surfaces=("preview_results", "preview_runtime_metadata"),
    ),
    _backbone_profile(
        node_id="artifact_critique",
        phase="generation_artifact",
        preserved_surfaces=("artifact_critique", "quality_observations"),
    ),
    _backbone_profile(
        node_id="review",
        phase="review_refinement",
        preserved_surfaces=("workflow_review", "hitl_recommendation"),
    ),
    _backbone_profile(
        node_id="refinement",
        phase="review_refinement",
        preserved_surfaces=("refinement_history", "refinement_decision"),
    ),
    _backbone_profile(
        node_id="finalization",
        phase="terminal_guardrail",
        preserved_surfaces=("final_workflow_event", "final_answer"),
    ),
    _backbone_profile(
        node_id="failure",
        phase="terminal_guardrail",
        preserved_surfaces=("failure_info", "failure_answer"),
    ),
)
V3_BACKBONE_MODE_REGISTRY = V3BackboneModeRegistry(
    node_profiles=V3_BACKBONE_MODE_NODE_PROFILES,
    node_ids=tuple(profile.node_id for profile in V3_BACKBONE_MODE_NODE_PROFILES),
    preserved_workflow_order=tuple(
        profile.node_id for profile in V3_BACKBONE_MODE_NODE_PROFILES
    ),
    phase_ids=_V3_BACKBONE_MODE_PHASE_IDS,
    source_registries=_V3_BACKBONE_MODE_SOURCE_REGISTRIES,
    node_count=len(V3_BACKBONE_MODE_NODE_PROFILES),
)
CONDITIONAL_MULTI_AGENT_ESCALATION_CONDITIONS = (
    _conditional_escalation_condition(
        condition_id="planning_ambiguity_multi_agent_candidate",
        condition_name="Planning Ambiguity Multi-Agent Candidate",
        category="ambiguity",
        backbone_phase="planning_reasoning",
        source_node_ids=("prompt_input", "planning", "reasoning"),
        capability_ids=(
            "v4_planner_agent",
            "adaptive_multi_agent_escalation",
        ),
        policy_rule_ids=(
            "missing_information_review",
            "future_agent_escalation_readiness",
        ),
        escalation_signal_ids=(
            "ambiguity_escalation_signal",
            "hitl_escalation_signal",
        ),
        advisory_outputs=(
            "planning_escalation_context_packet",
            "unresolved_question_summary",
            "candidate_planner_handoff",
        ),
    ),
    _conditional_escalation_condition(
        condition_id="artifact_risk_multi_agent_candidate",
        condition_name="Artifact Risk Multi-Agent Candidate",
        category="risk",
        backbone_phase="generation_artifact",
        source_node_ids=(
            "generation",
            "artifact_extraction",
            "artifact_critique",
        ),
        capability_ids=(
            "v4_artifact_agent",
            "adaptive_multi_agent_escalation",
        ),
        policy_rule_ids=(
            "artifact_risk_review",
            "future_agent_escalation_readiness",
        ),
        escalation_signal_ids=(
            "risk_escalation_signal",
            "quality_escalation_signal",
        ),
        advisory_outputs=(
            "artifact_risk_context_packet",
            "implementation_risk_summary",
            "candidate_artifact_handoff",
        ),
    ),
    _conditional_escalation_condition(
        condition_id="runtime_fit_multi_agent_candidate",
        condition_name="Runtime Fit Multi-Agent Candidate",
        category="runtime",
        backbone_phase="generation_artifact",
        source_node_ids=(
            "generation",
            "artifact_extraction",
            "preview_preparation",
        ),
        capability_ids=(
            "v4_runtime_agent",
            "v4_agent_router",
            "adaptive_multi_agent_escalation",
        ),
        policy_rule_ids=(
            "runtime_incompatibility_review",
            "future_agent_escalation_readiness",
        ),
        escalation_signal_ids=(
            "risk_escalation_signal",
            "latency_escalation_signal",
        ),
        advisory_outputs=(
            "runtime_fit_context_packet",
            "compatibility_gap_summary",
            "candidate_runtime_handoff",
        ),
    ),
    _conditional_escalation_condition(
        condition_id="evaluation_confidence_multi_agent_candidate",
        condition_name="Evaluation Confidence Multi-Agent Candidate",
        category="quality",
        backbone_phase="review_refinement",
        source_node_ids=("review", "refinement"),
        capability_ids=(
            "v4_agentic_studio",
            "adaptive_multi_agent_escalation",
        ),
        policy_rule_ids=(
            "evaluation_confidence_review",
            "future_agent_escalation_readiness",
        ),
        escalation_signal_ids=(
            "confidence_escalation_signal",
            "quality_escalation_signal",
            "hitl_escalation_signal",
        ),
        advisory_outputs=(
            "evaluation_escalation_context_packet",
            "quality_uncertainty_summary",
            "candidate_studio_handoff",
        ),
    ),
    _conditional_escalation_condition(
        condition_id="terminal_guardrail_multi_agent_candidate",
        condition_name="Terminal Guardrail Multi-Agent Candidate",
        category="hitl",
        backbone_phase="terminal_guardrail",
        source_node_ids=("finalization", "failure"),
        capability_ids=(
            "v4_agent_router",
            "adaptive_multi_agent_escalation",
        ),
        policy_rule_ids=("future_agent_escalation_readiness",),
        escalation_signal_ids=(
            "risk_escalation_signal",
            "hitl_escalation_signal",
        ),
        advisory_outputs=(
            "completion_guardrail_context_packet",
            "failure_review_posture",
            "candidate_router_handoff",
        ),
    ),
)
CONDITIONAL_MULTI_AGENT_ESCALATION_REGISTRY = (
    ConditionalMultiAgentEscalationRegistry(
        conditions=CONDITIONAL_MULTI_AGENT_ESCALATION_CONDITIONS,
        condition_ids=tuple(
            condition.condition_id
            for condition in CONDITIONAL_MULTI_AGENT_ESCALATION_CONDITIONS
        ),
        categories=tuple(
            condition.category
            for condition in CONDITIONAL_MULTI_AGENT_ESCALATION_CONDITIONS
        ),
        source_registries=_CONDITIONAL_ESCALATION_SOURCE_REGISTRIES,
        backbone_node_ids=V3_BACKBONE_MODE_REGISTRY.node_ids,
        condition_count=len(CONDITIONAL_MULTI_AGENT_ESCALATION_CONDITIONS),
    )
)
SPECIALIST_AGENT_LOOPS = (
    _specialist_agent_loop(
        loop_id="planning_specialist_agent_loop",
        loop_name="Planning Specialist Agent Loop",
        category="planning",
        specialist_agent_ids=("planner_agent", "research_agent"),
        source_condition_ids=("planning_ambiguity_multi_agent_candidate",),
        source_node_ids=("prompt_input", "planning", "reasoning"),
        loop_inputs=(
            "planning_escalation_context_packet",
            "unresolved_question_summary",
        ),
        advisory_outputs=(
            "planning_loop_notes",
            "research_gap_summary",
            "planner_handoff_recommendation",
        ),
        max_advisory_passes=2,
    ),
    _specialist_agent_loop(
        loop_id="artifact_specialist_agent_loop",
        loop_name="Artifact Specialist Agent Loop",
        category="artifact",
        specialist_agent_ids=(
            "artifact_agent",
            "art_direction_agent",
            "style_agent",
        ),
        source_condition_ids=("artifact_risk_multi_agent_candidate",),
        source_node_ids=(
            "generation",
            "artifact_extraction",
            "artifact_critique",
        ),
        loop_inputs=(
            "artifact_risk_context_packet",
            "implementation_risk_summary",
        ),
        advisory_outputs=(
            "artifact_loop_notes",
            "art_direction_review_summary",
            "style_consistency_handoff",
        ),
        max_advisory_passes=2,
    ),
    _specialist_agent_loop(
        loop_id="runtime_specialist_agent_loop",
        loop_name="Runtime Specialist Agent Loop",
        category="runtime",
        specialist_agent_ids=("runtime_agent", "artifact_agent"),
        source_condition_ids=("runtime_fit_multi_agent_candidate",),
        source_node_ids=(
            "generation",
            "artifact_extraction",
            "preview_preparation",
        ),
        loop_inputs=(
            "runtime_fit_context_packet",
            "compatibility_gap_summary",
        ),
        advisory_outputs=(
            "runtime_loop_notes",
            "compatibility_review_summary",
            "runtime_handoff_recommendation",
        ),
        max_advisory_passes=2,
    ),
    _specialist_agent_loop(
        loop_id="evaluation_specialist_agent_loop",
        loop_name="Evaluation Specialist Agent Loop",
        category="evaluation",
        specialist_agent_ids=(
            "critic_agent",
            "aesthetic_critic_agent",
            "creative_curator_agent",
            "refiner_agent",
        ),
        source_condition_ids=("evaluation_confidence_multi_agent_candidate",),
        source_node_ids=("review", "refinement"),
        loop_inputs=(
            "evaluation_escalation_context_packet",
            "quality_uncertainty_summary",
        ),
        advisory_outputs=(
            "evaluation_loop_notes",
            "critic_disagreement_summary",
            "refinement_handoff_recommendation",
        ),
        max_advisory_passes=3,
    ),
    _specialist_agent_loop(
        loop_id="synthesis_specialist_agent_loop",
        loop_name="Synthesis Specialist Agent Loop",
        category="synthesis",
        specialist_agent_ids=(
            "final_synthesizer_agent",
            "narrative_symbolic_agent",
            "creative_curator_agent",
        ),
        source_condition_ids=("terminal_guardrail_multi_agent_candidate",),
        source_node_ids=("finalization", "failure"),
        loop_inputs=(
            "completion_guardrail_context_packet",
            "failure_review_posture",
        ),
        advisory_outputs=(
            "synthesis_loop_notes",
            "terminal_handoff_summary",
            "final_synthesis_recommendation",
        ),
        max_advisory_passes=1,
    ),
)
SPECIALIST_AGENT_LOOP_REGISTRY = SpecialistAgentLoopRegistry(
    loops=SPECIALIST_AGENT_LOOPS,
    loop_ids=tuple(loop.loop_id for loop in SPECIALIST_AGENT_LOOPS),
    categories=tuple(loop.category for loop in SPECIALIST_AGENT_LOOPS),
    source_registries=_SPECIALIST_AGENT_LOOP_SOURCE_REGISTRIES,
    agent_ids=_KNOWN_SPECIALIST_AGENT_IDS,
    condition_ids=CONDITIONAL_MULTI_AGENT_ESCALATION_REGISTRY.condition_ids,
    backbone_node_ids=V3_BACKBONE_MODE_REGISTRY.node_ids,
    loop_count=len(SPECIALIST_AGENT_LOOPS),
)
ESCALATION_GATES = (
    _escalation_gate(
        gate_id="backbone_entry_escalation_gate",
        gate_name="Backbone Entry Escalation Gate",
        gate_kind="backbone_entry",
        source_condition_ids=(
            "planning_ambiguity_multi_agent_candidate",
            "artifact_risk_multi_agent_candidate",
            "runtime_fit_multi_agent_candidate",
        ),
        source_loop_ids=(),
        required_passive_inputs=(
            "v3_backbone_mode_registry",
            "conditional_escalation_conditions",
        ),
        advisory_decision_outputs=(
            "backbone_entry_gate_notes",
            "candidate_condition_summary",
        ),
    ),
    _escalation_gate(
        gate_id="evidence_completeness_escalation_gate",
        gate_name="Evidence Completeness Escalation Gate",
        gate_kind="evidence_completeness",
        source_condition_ids=CONDITIONAL_MULTI_AGENT_ESCALATION_REGISTRY.condition_ids,
        source_loop_ids=(),
        required_passive_inputs=(
            "condition_source_registries",
            "policy_rule_references",
            "escalation_signal_references",
        ),
        advisory_decision_outputs=(
            "evidence_completeness_notes",
            "missing_metadata_summary",
        ),
    ),
    _escalation_gate(
        gate_id="specialist_loop_boundary_gate",
        gate_name="Specialist Loop Boundary Gate",
        gate_kind="specialist_loop_boundary",
        source_condition_ids=CONDITIONAL_MULTI_AGENT_ESCALATION_REGISTRY.condition_ids,
        source_loop_ids=SPECIALIST_AGENT_LOOP_REGISTRY.loop_ids,
        required_passive_inputs=(
            "specialist_agent_loop_registry",
            "agent_contract_registry",
            "loop_pass_limits",
        ),
        advisory_decision_outputs=(
            "loop_boundary_notes",
            "specialist_loop_candidate_summary",
        ),
    ),
    _escalation_gate(
        gate_id="human_review_visibility_gate",
        gate_name="Human Review Visibility Gate",
        gate_kind="human_review_visibility",
        source_condition_ids=(
            "planning_ambiguity_multi_agent_candidate",
            "evaluation_confidence_multi_agent_candidate",
            "terminal_guardrail_multi_agent_candidate",
        ),
        source_loop_ids=(
            "planning_specialist_agent_loop",
            "evaluation_specialist_agent_loop",
            "synthesis_specialist_agent_loop",
        ),
        required_passive_inputs=(
            "hitl_escalation_signal",
            "human_review_posture",
            "operator_review_surface",
        ),
        advisory_decision_outputs=(
            "human_review_visibility_notes",
            "hitl_surface_summary",
        ),
    ),
    _escalation_gate(
        gate_id="return_handoff_escalation_gate",
        gate_name="Return Handoff Escalation Gate",
        gate_kind="return_handoff",
        source_condition_ids=CONDITIONAL_MULTI_AGENT_ESCALATION_REGISTRY.condition_ids,
        source_loop_ids=SPECIALIST_AGENT_LOOP_REGISTRY.loop_ids,
        required_passive_inputs=(
            "v3_backbone_mode_registry",
            "specialist_loop_advisory_outputs",
            "final_handoff_summary",
        ),
        advisory_decision_outputs=(
            "return_handoff_gate_notes",
            "backbone_rejoin_summary",
        ),
    ),
)
ESCALATION_GATE_REGISTRY = EscalationGateRegistry(
    gates=ESCALATION_GATES,
    gate_ids=tuple(gate.gate_id for gate in ESCALATION_GATES),
    gate_kinds=tuple(gate.gate_kind for gate in ESCALATION_GATES),
    source_registries=_ESCALATION_GATE_SOURCE_REGISTRIES,
    condition_ids=CONDITIONAL_MULTI_AGENT_ESCALATION_REGISTRY.condition_ids,
    loop_ids=SPECIALIST_AGENT_LOOP_REGISTRY.loop_ids,
    gate_count=len(ESCALATION_GATES),
)
CREATIVE_ESCALATION_POLICIES = (
    _creative_escalation_policy(
        policy_id="concept_ambiguity_creative_escalation_policy",
        policy_name="Concept Ambiguity Creative Escalation Policy",
        category="concept",
        source_gate_ids=(
            "backbone_entry_escalation_gate",
            "evidence_completeness_escalation_gate",
        ),
        source_loop_ids=("planning_specialist_agent_loop",),
        creative_signal_sources=(
            "planning_gap_summary",
            "missing_information",
            "creative_intent",
        ),
        advisory_policy_outputs=(
            "concept_escalation_policy_notes",
            "planning_clarity_review_summary",
        ),
    ),
    _creative_escalation_policy(
        policy_id="aesthetic_risk_creative_escalation_policy",
        policy_name="Aesthetic Risk Creative Escalation Policy",
        category="aesthetic",
        source_gate_ids=(
            "specialist_loop_boundary_gate",
            "human_review_visibility_gate",
        ),
        source_loop_ids=("artifact_specialist_agent_loop",),
        creative_signal_sources=(
            "artifact_risk_summary",
            "style_consistency_handoff",
            "aesthetic_critic_signals",
        ),
        advisory_policy_outputs=(
            "aesthetic_escalation_policy_notes",
            "artifact_style_review_summary",
        ),
    ),
    _creative_escalation_policy(
        policy_id="runtime_fit_creative_escalation_policy",
        policy_name="Runtime Fit Creative Escalation Policy",
        category="runtime",
        source_gate_ids=(
            "evidence_completeness_escalation_gate",
            "specialist_loop_boundary_gate",
        ),
        source_loop_ids=("runtime_specialist_agent_loop",),
        creative_signal_sources=(
            "runtime_fit_context_packet",
            "compatibility_gap_summary",
            "preview_runtime_metadata",
        ),
        advisory_policy_outputs=(
            "runtime_escalation_policy_notes",
            "compatibility_policy_summary",
        ),
    ),
    _creative_escalation_policy(
        policy_id="quality_uncertainty_creative_escalation_policy",
        policy_name="Quality Uncertainty Creative Escalation Policy",
        category="quality",
        source_gate_ids=(
            "human_review_visibility_gate",
            "return_handoff_escalation_gate",
        ),
        source_loop_ids=("evaluation_specialist_agent_loop",),
        creative_signal_sources=(
            "quality_uncertainty_summary",
            "creative_confidence",
            "evaluation_reports",
        ),
        advisory_policy_outputs=(
            "quality_escalation_policy_notes",
            "evaluation_confidence_review_summary",
        ),
    ),
    _creative_escalation_policy(
        policy_id="terminal_synthesis_creative_escalation_policy",
        policy_name="Terminal Synthesis Creative Escalation Policy",
        category="synthesis",
        source_gate_ids=(
            "human_review_visibility_gate",
            "return_handoff_escalation_gate",
        ),
        source_loop_ids=("synthesis_specialist_agent_loop",),
        creative_signal_sources=(
            "completion_guardrail_context_packet",
            "failure_review_posture",
            "final_handoff_summary",
        ),
        advisory_policy_outputs=(
            "synthesis_escalation_policy_notes",
            "terminal_creative_review_summary",
        ),
    ),
)
CREATIVE_ESCALATION_POLICY_REGISTRY = CreativeEscalationPolicyRegistry(
    policies=CREATIVE_ESCALATION_POLICIES,
    policy_ids=tuple(policy.policy_id for policy in CREATIVE_ESCALATION_POLICIES),
    categories=tuple(policy.category for policy in CREATIVE_ESCALATION_POLICIES),
    source_registries=_CREATIVE_ESCALATION_POLICY_SOURCE_REGISTRIES,
    gate_ids=ESCALATION_GATE_REGISTRY.gate_ids,
    loop_ids=SPECIALIST_AGENT_LOOP_REGISTRY.loop_ids,
    policy_count=len(CREATIVE_ESCALATION_POLICIES),
)

HYBRID_AGENTIC_WORKFLOW_STAGES = (
    _stage(
        stage_id="intake_routing_context_readiness",
        stage_name="Intake Routing Context Readiness",
        v3_workflow_nodes=(
            "intake",
            "routing",
            "memory",
            "retrieval",
            "context_assembly",
        ),
        future_capability_ids=("v4_agent_router",),
        escalation_rule_ids=("missing_information_review",),
        advisory_outputs=(
            "routing_context_packet",
            "retrieval_gap_summary",
            "context_handoff_notes",
        ),
    ),
    _stage(
        stage_id="planning_reasoning_readiness",
        stage_name="Planning Reasoning Readiness",
        v3_workflow_nodes=(
            "prompt_input",
            "planning",
            "director",
            "reasoning",
            "prompt_rendering",
        ),
        future_capability_ids=("v4_planner_agent", "v4_agentic_studio"),
        escalation_rule_ids=(
            "missing_information_review",
            "evaluation_confidence_review",
        ),
        advisory_outputs=(
            "planning_context_packet",
            "reasoning_review_notes",
            "prompt_handoff_summary",
        ),
    ),
    _stage(
        stage_id="generation_artifact_readiness",
        stage_name="Generation Artifact Readiness",
        v3_workflow_nodes=(
            "generation",
            "artifact_extraction",
            "preview_preparation",
            "artifact_critique",
        ),
        future_capability_ids=("v4_artifact_agent", "v4_runtime_agent"),
        escalation_rule_ids=(
            "artifact_risk_review",
            "runtime_incompatibility_review",
        ),
        advisory_outputs=(
            "artifact_context_packet",
            "runtime_fit_notes",
            "preview_readiness_summary",
        ),
    ),
    _stage(
        stage_id="review_refinement_readiness",
        stage_name="Review Refinement Readiness",
        v3_workflow_nodes=("review", "refinement"),
        future_capability_ids=(
            "v4_agentic_studio",
            "adaptive_multi_agent_escalation",
        ),
        escalation_rule_ids=(
            "evaluation_confidence_review",
            "future_agent_escalation_readiness",
        ),
        advisory_outputs=(
            "review_context_packet",
            "refinement_candidate_summary",
            "human_review_posture",
        ),
    ),
    _stage(
        stage_id="completion_guardrail_readiness",
        stage_name="Completion Guardrail Readiness",
        v3_workflow_nodes=("finalization", "failure"),
        future_capability_ids=(
            "v4_agent_router",
            "adaptive_multi_agent_escalation",
        ),
        escalation_rule_ids=("future_agent_escalation_readiness",),
        advisory_outputs=(
            "completion_context_packet",
            "failure_guardrail_notes",
            "final_handoff_summary",
        ),
    ),
)

HYBRID_AGENTIC_WORKFLOW_REGISTRY = HybridAgenticWorkflowRegistry(
    stages=HYBRID_AGENTIC_WORKFLOW_STAGES,
    stage_ids=tuple(stage.stage_id for stage in HYBRID_AGENTIC_WORKFLOW_STAGES),
    stage_count=len(HYBRID_AGENTIC_WORKFLOW_STAGES),
    source_metadata_registries=(
        "agent_capability_registry",
        "escalation_policy_registry",
        "artifact_engine_contract_registry",
        "evaluation_engine_contract_registry",
        "workstation_engine_contract_registry",
    ),
)
