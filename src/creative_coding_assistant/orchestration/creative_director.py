"""Bounded Creative Assistant Director guidance for V3 workflows."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration.artifact_critique import (
    ArtifactCritiqueSummary,
)
from creative_coding_assistant.orchestration.clarification import ClarificationRequest
from creative_coding_assistant.orchestration.creative_composition import (
    CreativeCompositionPlan,
)
from creative_coding_assistant.orchestration.creative_constraint_priorities import (
    CreativeConstraintPrioritization,
)
from creative_coding_assistant.orchestration.creative_constraints import (
    CreativeConstraintSolution,
)
from creative_coding_assistant.orchestration.creative_director_signals import (
    build_director_brief_payload,
)
from creative_coding_assistant.orchestration.creative_hierarchy import (
    CreativeHierarchyPlan,
)
from creative_coding_assistant.orchestration.creative_intent import (
    CreativeIntentDecomposition,
)
from creative_coding_assistant.orchestration.creative_planning import (
    CreativeExecutionPlan,
)
from creative_coding_assistant.orchestration.creative_quality_prediction import (
    CreativeQualityPrediction,
)
from creative_coding_assistant.orchestration.creative_strategy import (
    CreativeStrategyProfile,
)
from creative_coding_assistant.orchestration.creative_technique import (
    CreativeTechniqueProfile,
)
from creative_coding_assistant.orchestration.creative_tradeoffs import (
    CreativeTradeoffProfile,
)
from creative_coding_assistant.orchestration.creative_translation import (
    CreativeTranslation,
)
from creative_coding_assistant.orchestration.cross_modality import (
    CrossModalityCompositionProfile,
)
from creative_coding_assistant.orchestration.emotional_consistency import (
    EmotionalConsistencyProfile,
)
from creative_coding_assistant.orchestration.generative_structure import (
    GenerativeStructureBlueprint,
)
from creative_coding_assistant.orchestration.procedural_structure import (
    ProceduralStructurePlan,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityProfile,
)
from creative_coding_assistant.orchestration.semantic_motif import SemanticMotifSystem
from creative_coding_assistant.orchestration.symbolic_narrative import (
    SymbolicNarrativePlan,
)
from creative_coding_assistant.orchestration.workflow_review import WorkflowReviewResult

AmbiguityLevel = Literal["low", "medium", "high"]
RetrievalPosture = Literal["not_requested", "useful", "available"]

AUTHORITY_BOUNDARY = (
    "The user remains the Creative Director; the assistant provides bounded "
    "decision support and asks for HITL input when meaningful creative choices "
    "are ambiguous."
)


class CreativeAssistantDirectorBrief(BaseModel):
    """Inspectable Director metadata derived from existing workflow signals."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_assistant_director"] = "creative_assistant_director"
    creative_brief: str = Field(min_length=1, max_length=360)
    ambiguity_level: AmbiguityLevel
    ambiguity_signals: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    retrieval_posture: RetrievalPosture
    modality_direction: str | None = Field(default=None, max_length=120)
    runtime_direction: str | None = Field(default=None, max_length=160)
    planning_focus: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    critique_focus: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    refinement_focus: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    next_actions: tuple[str, ...] = Field(min_length=1, max_length=6)
    hitl_required: bool = False
    hitl_reason: str | None = Field(default=None, max_length=280)
    authority_boundary: str = Field(default=AUTHORITY_BOUNDARY, max_length=360)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=10)


def derive_creative_assistant_director_brief(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None = None,
    creative_intent: CreativeIntentDecomposition | None = None,
    creative_hierarchy: CreativeHierarchyPlan | None = None,
    creative_plan: CreativeExecutionPlan | None = None,
    creative_strategy: CreativeStrategyProfile | None = None,
    creative_techniques: CreativeTechniqueProfile | None = None,
    creative_constraints: CreativeConstraintSolution | None = None,
    creative_constraint_priorities: CreativeConstraintPrioritization | None = None,
    runtime_capabilities: RuntimeCapabilityProfile | None = None,
    creative_tradeoffs: CreativeTradeoffProfile | None = None,
    creative_quality_prediction: CreativeQualityPrediction | None = None,
    symbolic_narrative: SymbolicNarrativePlan | None = None,
    creative_composition: CreativeCompositionPlan | None = None,
    procedural_structure: ProceduralStructurePlan | None = None,
    generative_structure: GenerativeStructureBlueprint | None = None,
    semantic_motif: SemanticMotifSystem | None = None,
    emotional_consistency: EmotionalConsistencyProfile | None = None,
    cross_modality: CrossModalityCompositionProfile | None = None,
    clarification: ClarificationRequest | None = None,
    retrieval_chunk_count: int = 0,
    artifact_critique_summary: ArtifactCritiqueSummary | None = None,
    review_result: WorkflowReviewResult | None = None,
    refinement_count: int = 0,
) -> CreativeAssistantDirectorBrief:
    """Compose bounded Director guidance from deterministic workflow outputs."""

    return CreativeAssistantDirectorBrief(
        **build_director_brief_payload(
            request=request,
            route_decision=route_decision,
            creative_translation=creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_strategy=creative_strategy,
            creative_techniques=creative_techniques,
            creative_plan=creative_plan,
            creative_constraints=creative_constraints,
            creative_constraint_priorities=creative_constraint_priorities,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=creative_tradeoffs,
            creative_quality_prediction=creative_quality_prediction,
            symbolic_narrative=symbolic_narrative,
            creative_composition=creative_composition,
            procedural_structure=procedural_structure,
            generative_structure=generative_structure,
            semantic_motif=semantic_motif,
            emotional_consistency=emotional_consistency,
            cross_modality=cross_modality,
            retrieval_chunk_count=retrieval_chunk_count,
            clarification=clarification,
            artifact_critique_summary=artifact_critique_summary,
            review_result=review_result,
            refinement_count=refinement_count,
        )
    )


def creative_assistant_director_prompt_lines(
    brief: CreativeAssistantDirectorBrief,
) -> tuple[str, ...]:
    """Render compact Director guidance into provider prompt instructions."""

    lines = [
        f"Authority boundary: {brief.authority_boundary}",
        f"Creative brief: {brief.creative_brief}",
        f"Ambiguity level: {brief.ambiguity_level}.",
        f"Retrieval posture: {brief.retrieval_posture}.",
    ]
    if brief.modality_direction is not None:
        lines.append(f"Modality direction: {brief.modality_direction}.")
    if brief.runtime_direction is not None:
        lines.append(f"Runtime direction: {brief.runtime_direction}.")
    lines.extend(f"Planning focus: {item}" for item in brief.planning_focus)
    lines.extend(f"Critique focus: {item}" for item in brief.critique_focus)
    lines.extend(f"Refinement focus: {item}" for item in brief.refinement_focus)
    lines.extend(f"Next action: {item}" for item in brief.next_actions)
    return tuple(lines[:18])
