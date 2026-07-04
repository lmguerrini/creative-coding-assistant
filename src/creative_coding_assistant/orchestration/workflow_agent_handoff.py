"""Passive V4.2 workflow-to-agent handoff metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.agent_roles import AGENT_ROLE_REGISTRY
from creative_coding_assistant.orchestration.shared_context_views import (
    SHARED_CONTEXT_VIEW_REGISTRY,
)
from creative_coding_assistant.orchestration.workflow import WORKFLOW_STEP_ORDER

WorkflowHandoffSurface = Literal[
    "planning",
    "artifact",
    "evaluation",
    "provenance",
    "finalization",
]
WorkflowHandoffPayloadExposure = Literal["metadata_reference_only"]

WORKFLOW_AGENT_HANDOFF_CONTRACT_SERIALIZATION_VERSION = (
    "workflow_agent_handoff_contract.v1"
)
WORKFLOW_AGENT_HANDOFF_PROFILE_SERIALIZATION_VERSION = (
    "workflow_agent_handoff_profile.v1"
)
WORKFLOW_AGENT_HANDOFF_REGISTRY_SERIALIZATION_VERSION = (
    "workflow_agent_handoff_registry.v1"
)
WORKFLOW_AGENT_HANDOFF_REGISTRY_AUTHORITY_BOUNDARY = (
    "Workflow-to-agent handoff metadata maps existing V3 planning, artifact, "
    "evaluation, provenance, and finalization workflow surfaces to passive V4 "
    "agent roles only; it does not change the workflow graph, alter prompts, "
    "execute agents, perform runtime handoffs, route providers or models, "
    "trigger retries, mutate storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "workflow_graph_change",
    "prompt_alteration",
    "agent_execution",
    "runtime_handoff_execution",
    "provider_or_model_routing",
    "retry_or_refinement_triggering",
    "storage_mutation",
    "generated_output_modification",
)

_HANDOFF_SPECS: tuple[
    tuple[
        str,
        WorkflowHandoffSurface,
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
        str,
    ],
    ...,
] = (
    (
        "planning_surface_agent_handoff",
        "planning",
        ("planning", "director", "reasoning"),
        (
            "creative_intent",
            "creative_hierarchy",
            "creative_strategy",
            "creative_techniques",
            "creative_plan",
            "creative_constraints",
            "creative_constraint_priorities",
            "runtime_capabilities",
            "creative_tradeoffs",
            "creative_quality_prediction",
            "symbolic_narrative",
            "creative_composition",
            "procedural_structure",
            "generative_structure",
            "semantic_motif",
            "emotional_consistency",
            "cross_modality",
            "audio_visual_scene",
            "creative_director",
            "creative_reasoning",
        ),
        (
            "planner_agent",
            "research_agent",
            "style_agent",
            "runtime_agent",
            "art_direction_agent",
            "narrative_symbolic_agent",
        ),
        "V3 planning metadata can be referenced by planning, source, style, "
        "runtime, art direction, and narrative agents.",
    ),
    (
        "artifact_surface_agent_handoff",
        "artifact",
        (
            "artifact_extraction",
            "preview_preparation",
            "artifact_critique",
            "refinement",
        ),
        (
            "artifact_plan",
            "artifact_dependency_graph",
            "runtime_compatibility",
            "artifact_capability_matrix",
            "multi_artifact_strategy",
            "artifact_critic",
            "artifact_refiner",
            "artifact_intelligence_synthesis",
            "artifact_merge_planner",
            "artifact_export_intelligence",
            "artifact_engine_contracts",
            "artifacts",
            "preview_results",
            "artifact_critique_summary",
            "refinement_passes",
        ),
        (
            "runtime_agent",
            "artifact_agent",
            "critic_agent",
            "refiner_agent",
            "final_synthesizer_agent",
        ),
        "V3 artifact metadata can be referenced by runtime, artifact, review, "
        "refinement, and final synthesis agents.",
    ),
    (
        "evaluation_surface_agent_handoff",
        "evaluation",
        ("planning", "artifact_critique", "review", "finalization"),
        (
            "creative_critic",
            "self_evaluation",
            "creative_improvement_planner",
            "reflection_loop",
            "creative_confidence",
            "creative_score",
            "consistency_validation",
            "evaluation_report",
            "evaluation_engine_contracts",
            "artifact_critique_summary",
            "review_result",
        ),
        (
            "aesthetic_critic_agent",
            "creative_curator_agent",
            "critic_agent",
            "refiner_agent",
            "final_synthesizer_agent",
        ),
        "V3 evaluation metadata can be referenced by aesthetic review, "
        "curation, critique, refinement, and final synthesis agents.",
    ),
    (
        "provenance_surface_agent_handoff",
        "provenance",
        (
            "routing",
            "memory",
            "retrieval",
            "context_assembly",
            "artifact_extraction",
            "preview_preparation",
        ),
        (
            "route_decision",
            "memory_context",
            "retrieval_context",
            "assembled_context",
            "artifact_engine_contracts",
            "evaluation_engine_contracts",
            "artifacts",
            "preview_results",
            "refinement_passes",
        ),
        (
            "research_agent",
            "critic_agent",
            "final_synthesizer_agent",
        ),
        "V3 provenance metadata can be referenced by source context, critique, "
        "and final synthesis agents.",
    ),
    (
        "finalization_surface_agent_handoff",
        "finalization",
        ("review", "finalization"),
        (
            "final_answer",
            "review_result",
            "creative_director",
            "creative_reasoning",
            "evaluation_report",
            "artifacts",
            "artifact_critique_summary",
        ),
        (
            "planner_agent",
            "critic_agent",
            "refiner_agent",
            "final_synthesizer_agent",
        ),
        "V3 finalization metadata can be referenced by planning, critique, "
        "refinement, and final synthesis agents.",
    ),
)


class WorkflowAgentHandoffContract(BaseModel):
    """Passive contract mapping one V3 workflow surface to V4 agents."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    handoff_id: str = Field(min_length=1, max_length=140)
    surface: WorkflowHandoffSurface
    source_workflow_steps: tuple[str, ...] = Field(min_length=1, max_length=8)
    source_state_fields: tuple[str, ...] = Field(min_length=1, max_length=32)
    target_agent_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    target_role_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    payload_exposure: WorkflowHandoffPayloadExposure = "metadata_reference_only"
    handoff_intent: str = Field(min_length=1, max_length=500)
    handoff_boundary: str = Field(min_length=1, max_length=800)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    workflow_graph_change_implemented: Literal[False] = False
    prompt_alteration_implemented: Literal[False] = False
    agent_execution_implemented: Literal[False] = False
    runtime_handoff_implemented: Literal[False] = False
    serialization_version: Literal["workflow_agent_handoff_contract.v1"] = (
        WORKFLOW_AGENT_HANDOFF_CONTRACT_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class WorkflowAgentHandoffProfile(BaseModel):
    """Per-agent passive handoff profile."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=80)
    role_id: str = Field(min_length=1, max_length=80)
    handoff_profile_id: str = Field(min_length=1, max_length=140)
    accepted_handoff_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    accepted_surfaces: tuple[WorkflowHandoffSurface, ...] = Field(
        min_length=1,
        max_length=5,
    )
    accepted_state_fields: tuple[str, ...] = Field(min_length=1, max_length=64)
    source_context_view_id: str = Field(min_length=1, max_length=140)
    profile_boundary: str = Field(min_length=1, max_length=800)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    workflow_graph_change_implemented: Literal[False] = False
    prompt_alteration_implemented: Literal[False] = False
    agent_execution_implemented: Literal[False] = False
    runtime_handoff_implemented: Literal[False] = False
    serialization_version: Literal["workflow_agent_handoff_profile.v1"] = (
        WORKFLOW_AGENT_HANDOFF_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class WorkflowAgentHandoffRegistry(BaseModel):
    """Stable passive workflow-to-agent handoff registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["workflow_agent_handoff_registry"] = "workflow_agent_handoff_registry"
    serialization_version: Literal["workflow_agent_handoff_registry.v1"] = (
        WORKFLOW_AGENT_HANDOFF_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=WORKFLOW_AGENT_HANDOFF_REGISTRY_AUTHORITY_BOUNDARY,
        max_length=1000,
    )
    handoffs: tuple[WorkflowAgentHandoffContract, ...] = Field(
        min_length=5,
        max_length=5,
    )
    profiles: tuple[WorkflowAgentHandoffProfile, ...] = Field(
        min_length=12,
        max_length=12,
    )
    handoff_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    surfaces: tuple[WorkflowHandoffSurface, ...] = Field(min_length=5, max_length=5)
    agent_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    profile_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    workflow_step_ids: tuple[str, ...] = Field(min_length=17, max_length=17)
    handoff_count: int = Field(ge=5, le=5)
    profile_count: int = Field(ge=12, le=12)
    source_registries: tuple[str, ...] = Field(min_length=2, max_length=4)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    workflow_graph_change_implemented: Literal[False] = False
    prompt_alteration_implemented: Literal[False] = False
    agent_execution_implemented: Literal[False] = False
    runtime_handoff_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_handoff_metadata(self) -> Self:
        derived_handoff_ids = tuple(handoff.handoff_id for handoff in self.handoffs)
        derived_surfaces = tuple(handoff.surface for handoff in self.handoffs)
        derived_agent_ids = tuple(profile.agent_id for profile in self.profiles)
        derived_profile_ids = tuple(
            profile.handoff_profile_id for profile in self.profiles
        )
        if self.handoff_ids != derived_handoff_ids:
            raise ValueError("handoff_ids must match handoffs")
        if self.surfaces != derived_surfaces:
            raise ValueError("surfaces must match handoffs")
        if self.agent_ids != derived_agent_ids:
            raise ValueError("agent_ids must match profiles")
        if self.profile_ids != derived_profile_ids:
            raise ValueError("profile_ids must match profiles")
        if self.handoff_count != len(self.handoffs):
            raise ValueError("handoff_count must match handoffs")
        if self.profile_count != len(self.profiles):
            raise ValueError("profile_count must match profiles")

        known_workflow_steps = set(self.workflow_step_ids)
        known_handoff_ids = set(self.handoff_ids)
        known_agents = set(self.agent_ids)
        for handoff in self.handoffs:
            if not set(handoff.source_workflow_steps).issubset(known_workflow_steps):
                raise ValueError("source_workflow_steps must be known workflow steps")
            if not set(handoff.target_agent_ids).issubset(known_agents):
                raise ValueError("target_agent_ids must be known agents")
        for profile in self.profiles:
            if not set(profile.accepted_handoff_ids).issubset(known_handoff_ids):
                raise ValueError("profile handoffs must be known handoffs")
        return self


def workflow_agent_handoff_registry() -> WorkflowAgentHandoffRegistry:
    """Return passive V4.2 workflow-to-agent handoff metadata."""

    return WORKFLOW_AGENT_HANDOFF_REGISTRY


def workflow_agent_handoff_by_id(
    handoff_id: str,
    registry: WorkflowAgentHandoffRegistry | None = None,
) -> WorkflowAgentHandoffContract | None:
    """Return one handoff contract without executing a handoff."""

    source_registry = registry or WORKFLOW_AGENT_HANDOFF_REGISTRY
    for handoff in source_registry.handoffs:
        if handoff.handoff_id == handoff_id:
            return handoff
    return None


def workflow_agent_handoff_profile_by_agent_id(
    agent_id: str,
    registry: WorkflowAgentHandoffRegistry | None = None,
) -> WorkflowAgentHandoffProfile | None:
    """Return one agent handoff profile without invoking agents."""

    source_registry = registry or WORKFLOW_AGENT_HANDOFF_REGISTRY
    for profile in source_registry.profiles:
        if profile.agent_id == agent_id:
            return profile
    return None


def workflow_agent_handoffs_for_surface(
    surface: WorkflowHandoffSurface,
    registry: WorkflowAgentHandoffRegistry | None = None,
) -> tuple[WorkflowAgentHandoffContract, ...]:
    """Return handoff metadata for one V3 surface without changing workflows."""

    source_registry = registry or WORKFLOW_AGENT_HANDOFF_REGISTRY
    return tuple(
        handoff for handoff in source_registry.handoffs if handoff.surface == surface
    )


def _role_ids_for_agent_ids(agent_ids: tuple[str, ...]) -> tuple[str, ...]:
    role_by_agent_id = {
        role.agent_id: role.role_id for role in AGENT_ROLE_REGISTRY.roles
    }
    return tuple(role_by_agent_id[agent_id] for agent_id in agent_ids)


def _handoff(
    spec: tuple[
        str,
        WorkflowHandoffSurface,
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
        str,
    ],
) -> WorkflowAgentHandoffContract:
    handoff_id, surface, workflow_steps, state_fields, target_agents, intent = spec
    return WorkflowAgentHandoffContract(
        handoff_id=handoff_id,
        surface=surface,
        source_workflow_steps=workflow_steps,
        source_state_fields=state_fields,
        target_agent_ids=target_agents,
        target_role_ids=_role_ids_for_agent_ids(target_agents),
        handoff_intent=intent,
        handoff_boundary=(
            "Workflow handoff contracts are metadata references only; they do "
            "not change graph nodes, alter prompts, execute agents, perform "
            "runtime handoffs, or mutate generated output."
        ),
    )


WORKFLOW_AGENT_HANDOFFS = tuple(_handoff(spec) for spec in _HANDOFF_SPECS)


def _profile(agent_id: str) -> WorkflowAgentHandoffProfile:
    role = next(role for role in AGENT_ROLE_REGISTRY.roles if role.agent_id == agent_id)
    context_view = next(
        view for view in SHARED_CONTEXT_VIEW_REGISTRY.views if view.agent_id == agent_id
    )
    accepted_handoffs = tuple(
        handoff
        for handoff in WORKFLOW_AGENT_HANDOFFS
        if agent_id in handoff.target_agent_ids
    )
    return WorkflowAgentHandoffProfile(
        agent_id=agent_id,
        role_id=role.role_id,
        handoff_profile_id=f"{agent_id}_workflow_handoff_profile",
        accepted_handoff_ids=tuple(handoff.handoff_id for handoff in accepted_handoffs),
        accepted_surfaces=tuple(handoff.surface for handoff in accepted_handoffs),
        accepted_state_fields=tuple(
            dict.fromkeys(
                state_field
                for handoff in accepted_handoffs
                for state_field in handoff.source_state_fields
            )
        ),
        source_context_view_id=context_view.view_id,
        profile_boundary=(
            "Agent handoff profiles are passive metadata references only; they "
            "do not execute agents, alter prompts, perform runtime handoffs, "
            "or change workflow behavior."
        ),
    )


WORKFLOW_AGENT_HANDOFF_PROFILES = tuple(
    _profile(role.agent_id) for role in AGENT_ROLE_REGISTRY.roles
)
WORKFLOW_AGENT_HANDOFF_REGISTRY = WorkflowAgentHandoffRegistry(
    handoffs=WORKFLOW_AGENT_HANDOFFS,
    profiles=WORKFLOW_AGENT_HANDOFF_PROFILES,
    handoff_ids=tuple(handoff.handoff_id for handoff in WORKFLOW_AGENT_HANDOFFS),
    surfaces=tuple(handoff.surface for handoff in WORKFLOW_AGENT_HANDOFFS),
    agent_ids=tuple(profile.agent_id for profile in WORKFLOW_AGENT_HANDOFF_PROFILES),
    profile_ids=tuple(
        profile.handoff_profile_id for profile in WORKFLOW_AGENT_HANDOFF_PROFILES
    ),
    workflow_step_ids=tuple(step.value for step in WORKFLOW_STEP_ORDER),
    handoff_count=len(WORKFLOW_AGENT_HANDOFFS),
    profile_count=len(WORKFLOW_AGENT_HANDOFF_PROFILES),
    source_registries=(
        "workflow_state",
        "agent_role_registry",
        "shared_context_view_registry",
    ),
)
