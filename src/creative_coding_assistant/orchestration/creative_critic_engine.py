"""Bounded Creative Critic Engine for V3.4 evaluation metadata."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration._metadata_utils import (
    _clip,
    _contains_any,
    _dedupe,
)
from creative_coding_assistant.orchestration.artifact_capability_matrix import (
    ArtifactCapabilityMatrix,
)
from creative_coding_assistant.orchestration.artifact_critic import (
    ArtifactCriticProfile,
)
from creative_coding_assistant.orchestration.artifact_dependency_graph import (
    ArtifactDependencyGraph,
)
from creative_coding_assistant.orchestration.artifact_export_intelligence import (
    ArtifactExportIntelligenceProfile,
)
from creative_coding_assistant.orchestration.artifact_intelligence_synthesis import (
    ArtifactIntelligenceSynthesisProfile,
)
from creative_coding_assistant.orchestration.artifact_merge_planner import (
    ArtifactMergePlannerProfile,
)
from creative_coding_assistant.orchestration.artifact_planner import ArtifactPlan
from creative_coding_assistant.orchestration.artifact_refiner import (
    ArtifactRefinerProfile,
)
from creative_coding_assistant.orchestration.artifacts import WorkflowArtifact
from creative_coding_assistant.orchestration.audio_visual_scene import (
    AudioVisualSceneProfile,
)
from creative_coding_assistant.orchestration.creative_composition import (
    CreativeCompositionPlan,
)
from creative_coding_assistant.orchestration.creative_constraint_priorities import (
    CreativeConstraintPrioritization,
)
from creative_coding_assistant.orchestration.creative_constraints import (
    CreativeConstraintSolution,
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
from creative_coding_assistant.orchestration.multi_artifact_strategy import (
    MultiArtifactStrategy,
)
from creative_coding_assistant.orchestration.procedural_structure import (
    ProceduralStructurePlan,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityProfile,
)
from creative_coding_assistant.orchestration.runtime_compatibility import (
    RuntimeCompatibilityProfile,
)
from creative_coding_assistant.orchestration.semantic_motif import SemanticMotifSystem
from creative_coding_assistant.orchestration.symbolic_narrative import (
    SymbolicNarrativePlan,
)

CreativeCriticRiskAssessment = Literal["low", "medium", "high", "blocked"]

CREATIVE_CRITIC_AUTHORITY_BOUNDARY = (
    "The Creative Critic Engine evaluates creative and artifact metadata only; "
    "it may identify strengths, weaknesses, risks, unsupported assumptions, "
    "missing information, HITL questions, and improvement opportunities, but "
    "it does not modify artifacts, reject outputs, select runtimes, route "
    "providers or models, change previews, execute artifacts, trigger retries, "
    "trigger refinement, repair runtime behavior, implement Studio Mode, "
    "implement HoloMind, or implement future V4/V5 execution systems."
)


class CreativeCriticProfile(BaseModel):
    """Inspectable metadata-only critique across creative workflow signals."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_critic_engine"] = "creative_critic_engine"
    critic_confidence: float = Field(ge=0, le=1)
    critique_summary: str = Field(min_length=1, max_length=620)
    creative_strengths: tuple[str, ...] = Field(min_length=1, max_length=10)
    creative_weaknesses: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    concept_quality: float = Field(ge=0, le=1)
    execution_quality: float = Field(ge=0, le=1)
    artifact_quality: float = Field(ge=0, le=1)
    coherence_quality: float = Field(ge=0, le=1)
    runtime_fit_quality: float = Field(ge=0, le=1)
    originality_quality: float = Field(ge=0, le=1)
    clarity_quality: float = Field(ge=0, le=1)
    feasibility_quality: float = Field(ge=0, le=1)
    risk_assessment: CreativeCriticRiskAssessment
    missing_information: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    unsupported_assumptions: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=10,
    )
    improvement_opportunities: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=10,
    )
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=CREATIVE_CRITIC_AUTHORITY_BOUNDARY,
        max_length=980,
    )
    evidence: tuple[str, ...] = Field(min_length=1, max_length=16)


def derive_creative_critic_profile(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None = None,
    creative_intent: CreativeIntentDecomposition | None = None,
    creative_hierarchy: CreativeHierarchyPlan | None = None,
    creative_plan: CreativeExecutionPlan | None = None,
    creative_constraints: CreativeConstraintSolution | None = None,
    creative_constraint_priorities: CreativeConstraintPrioritization | None = None,
    creative_strategy: CreativeStrategyProfile | None = None,
    creative_techniques: CreativeTechniqueProfile | None = None,
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
    audio_visual_scene: AudioVisualSceneProfile | None = None,
    artifact_plan: ArtifactPlan | None = None,
    artifact_dependency_graph: ArtifactDependencyGraph | None = None,
    runtime_compatibility: RuntimeCompatibilityProfile | None = None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None = None,
    multi_artifact_strategy: MultiArtifactStrategy | None = None,
    artifact_critic: ArtifactCriticProfile | None = None,
    artifact_refiner: ArtifactRefinerProfile | None = None,
    artifact_intelligence_synthesis: (
        ArtifactIntelligenceSynthesisProfile | None
    ) = None,
    artifact_merge_planner: ArtifactMergePlannerProfile | None = None,
    artifact_export_intelligence: ArtifactExportIntelligenceProfile | None = None,
    generated_response: str | None = None,
    artifacts: Sequence[WorkflowArtifact] = (),
) -> CreativeCriticProfile:
    """Evaluate available creative metadata without altering workflow behavior."""

    qualities = _quality_scores(
        request=request,
        route_decision=route_decision,
        creative_translation=creative_translation,
        creative_intent=creative_intent,
        creative_hierarchy=creative_hierarchy,
        creative_plan=creative_plan,
        creative_constraints=creative_constraints,
        creative_constraint_priorities=creative_constraint_priorities,
        creative_strategy=creative_strategy,
        creative_techniques=creative_techniques,
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
        audio_visual_scene=audio_visual_scene,
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_critic=artifact_critic,
        artifact_refiner=artifact_refiner,
        artifact_intelligence_synthesis=artifact_intelligence_synthesis,
        artifact_merge_planner=artifact_merge_planner,
        artifact_export_intelligence=artifact_export_intelligence,
        generated_response=generated_response,
        artifacts=artifacts,
    )
    missing = _missing_information(
        route_decision=route_decision,
        creative_translation=creative_translation,
        creative_intent=creative_intent,
        creative_hierarchy=creative_hierarchy,
        creative_plan=creative_plan,
        creative_strategy=creative_strategy,
        creative_techniques=creative_techniques,
        runtime_capabilities=runtime_capabilities,
        creative_quality_prediction=creative_quality_prediction,
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_critic=artifact_critic,
        generated_response=generated_response,
        artifacts=artifacts,
    )
    unsupported = _unsupported_assumptions(
        creative_plan=creative_plan,
        creative_constraints=creative_constraints,
        runtime_capabilities=runtime_capabilities,
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        artifact_critic=artifact_critic,
        artifacts=artifacts,
    )
    weaknesses = _creative_weaknesses(
        qualities=qualities,
        missing_information=missing,
        unsupported_assumptions=unsupported,
        creative_quality_prediction=creative_quality_prediction,
        artifact_critic=artifact_critic,
    )
    strengths = _creative_strengths(
        qualities=qualities,
        creative_intent=creative_intent,
        creative_hierarchy=creative_hierarchy,
        creative_strategy=creative_strategy,
        creative_techniques=creative_techniques,
        creative_quality_prediction=creative_quality_prediction,
        artifact_plan=artifact_plan,
        artifact_critic=artifact_critic,
        artifacts=artifacts,
    )
    risk = _risk_assessment(
        qualities=qualities,
        weaknesses=weaknesses,
        missing_information=missing,
        unsupported_assumptions=unsupported,
        artifact_critic=artifact_critic,
        runtime_compatibility=runtime_compatibility,
    )
    opportunities = _improvement_opportunities(
        risk=risk,
        qualities=qualities,
        missing_information=missing,
        unsupported_assumptions=unsupported,
        creative_quality_prediction=creative_quality_prediction,
        artifact_critic=artifact_critic,
        artifact_refiner=artifact_refiner,
        artifact_intelligence_synthesis=artifact_intelligence_synthesis,
    )

    return CreativeCriticProfile(
        critic_confidence=_critic_confidence(
            missing_information=missing,
            route_decision=route_decision,
            creative_plan=creative_plan,
            creative_strategy=creative_strategy,
            creative_techniques=creative_techniques,
            creative_quality_prediction=creative_quality_prediction,
            artifact_plan=artifact_plan,
            artifact_critic=artifact_critic,
            generated_response=generated_response,
            artifacts=artifacts,
        ),
        critique_summary=_critique_summary(
            risk=risk,
            qualities=qualities,
            strengths=strengths,
            weaknesses=weaknesses,
        ),
        creative_strengths=strengths,
        creative_weaknesses=weaknesses,
        concept_quality=qualities["concept"],
        execution_quality=qualities["execution"],
        artifact_quality=qualities["artifact"],
        coherence_quality=qualities["coherence"],
        runtime_fit_quality=qualities["runtime_fit"],
        originality_quality=qualities["originality"],
        clarity_quality=qualities["clarity"],
        feasibility_quality=qualities["feasibility"],
        risk_assessment=risk,
        missing_information=missing,
        unsupported_assumptions=unsupported,
        improvement_opportunities=opportunities,
        hitl_questions=_hitl_questions(
            risk=risk,
            missing_information=missing,
            unsupported_assumptions=unsupported,
            creative_quality_prediction=creative_quality_prediction,
            artifact_critic=artifact_critic,
        ),
        prompt_guidance=_prompt_guidance(
            risk=risk,
            opportunities=opportunities,
        ),
        evidence=_evidence(
            request=request,
            route_decision=route_decision,
            creative_strategy=creative_strategy,
            creative_techniques=creative_techniques,
            creative_quality_prediction=creative_quality_prediction,
            artifact_plan=artifact_plan,
            artifact_critic=artifact_critic,
            generated_response=generated_response,
            artifacts=artifacts,
        ),
    )


def creative_critic_prompt_lines(profile: CreativeCriticProfile) -> tuple[str, ...]:
    """Render Creative Critic metadata as compact prompt guidance."""

    lines = [
        f"Authority boundary: {profile.authority_boundary}",
        f"Creative critic confidence: {profile.critic_confidence:.2f}.",
        f"Creative critic risk assessment: {profile.risk_assessment}.",
        f"Creative critic summary: {profile.critique_summary}",
        (
            "Creative quality scores: "
            f"concept {profile.concept_quality:.2f}; "
            f"execution {profile.execution_quality:.2f}; "
            f"artifact {profile.artifact_quality:.2f}; "
            f"coherence {profile.coherence_quality:.2f}; "
            f"runtime fit {profile.runtime_fit_quality:.2f}; "
            f"originality {profile.originality_quality:.2f}; "
            f"clarity {profile.clarity_quality:.2f}; "
            f"feasibility {profile.feasibility_quality:.2f}."
        ),
    ]
    lines.extend(f"Creative critic strength: {item}" for item in profile.creative_strengths)
    lines.extend(f"Creative critic weakness: {item}" for item in profile.creative_weaknesses)
    lines.extend(
        f"Creative critic missing information: {item}"
        for item in profile.missing_information
    )
    lines.extend(
        f"Creative critic unsupported assumption: {item}"
        for item in profile.unsupported_assumptions
    )
    lines.extend(
        f"Creative critic improvement opportunity: {item}"
        for item in profile.improvement_opportunities
    )
    lines.extend(f"Creative critic HITL question: {item}" for item in profile.hitl_questions)
    lines.extend(f"Creative critic guidance: {item}" for item in profile.prompt_guidance)
    return tuple(lines[:64])


def _quality_scores(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    creative_constraint_priorities: CreativeConstraintPrioritization | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
    creative_quality_prediction: CreativeQualityPrediction | None,
    symbolic_narrative: SymbolicNarrativePlan | None,
    creative_composition: CreativeCompositionPlan | None,
    procedural_structure: ProceduralStructurePlan | None,
    generative_structure: GenerativeStructureBlueprint | None,
    semantic_motif: SemanticMotifSystem | None,
    emotional_consistency: EmotionalConsistencyProfile | None,
    cross_modality: CrossModalityCompositionProfile | None,
    audio_visual_scene: AudioVisualSceneProfile | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
    artifact_intelligence_synthesis: ArtifactIntelligenceSynthesisProfile | None,
    artifact_merge_planner: ArtifactMergePlannerProfile | None,
    artifact_export_intelligence: ArtifactExportIntelligenceProfile | None,
    generated_response: str | None,
    artifacts: Sequence[WorkflowArtifact],
) -> dict[str, float]:
    query = request.query.lower()
    artifact_scores = [item.quality_score for item in artifacts if item.quality_score is not None]
    artifact_average = sum(artifact_scores) / len(artifact_scores) if artifact_scores else None
    readiness = (
        creative_quality_prediction.readiness_score / 100
        if creative_quality_prediction is not None
        else None
    )
    artifact_risk = artifact_critic.risk_assessment if artifact_critic is not None else None

    concept = _score(
        0.34,
        positives=(
            creative_translation,
            creative_intent,
            creative_hierarchy,
            creative_strategy,
            symbolic_narrative,
            semantic_motif,
            creative_composition,
        ),
        bonus=(readiness or 0) * 0.12,
        penalties=(0.08 if _contains_any(query, ("vague", "anything", "whatever")) else 0),
    )
    execution = _score(
        0.32,
        positives=(
            creative_plan,
            creative_techniques,
            creative_constraints,
            creative_constraint_priorities,
            runtime_capabilities,
            creative_tradeoffs,
            procedural_structure,
            generative_structure,
        ),
        bonus=(readiness or 0) * 0.10,
        penalties=_risk_penalty(creative_quality_prediction),
    )
    artifact = _score(
        0.30,
        positives=(
            artifact_plan,
            artifact_dependency_graph,
            runtime_compatibility,
            artifact_capability_matrix,
            multi_artifact_strategy,
            artifact_critic,
            artifact_refiner,
            artifact_intelligence_synthesis,
            artifact_merge_planner,
            artifact_export_intelligence,
        ),
        bonus=(artifact_average or 0) * 0.16 + min(len(artifacts), 3) * 0.03,
        penalties=_artifact_risk_penalty(artifact_risk),
    )
    coherence = _score(
        0.34,
        positives=(
            creative_hierarchy,
            creative_constraint_priorities,
            symbolic_narrative,
            creative_composition,
            semantic_motif,
            emotional_consistency,
            cross_modality,
            audio_visual_scene,
            artifact_dependency_graph,
            multi_artifact_strategy,
        ),
        bonus=(readiness or 0) * 0.08,
        penalties=_sequence_penalty(
            getattr(creative_quality_prediction, "missing_information", ())
        ),
    )
    runtime_fit = _score(
        0.30,
        positives=(
            route_decision,
            runtime_capabilities,
            runtime_compatibility,
            artifact_capability_matrix,
            creative_plan,
        ),
        bonus=0.08 if artifacts and any(item.preview_eligible for item in artifacts) else 0,
        penalties=(
            _sequence_penalty(getattr(runtime_compatibility, "unsupported_runtimes", ())) * 0.45
            + _sequence_penalty(getattr(runtime_compatibility, "implementation_risks", ())) * 0.6
            + _sequence_penalty(getattr(artifact_critic, "runtime_concerns", ())) * 0.6
        ),
    )
    originality = _score(
        0.36,
        positives=(
            creative_strategy,
            creative_techniques,
            symbolic_narrative,
            semantic_motif,
            generative_structure,
            emotional_consistency,
            cross_modality,
            audio_visual_scene,
        ),
        bonus=0.05 if _contains_any(query, ("novel", "surprising", "unusual", "experimental")) else 0,
        penalties=0.08 if _contains_any(query, ("simple", "basic", "minimal")) else 0,
    )
    clarity = _score(
        0.34,
        positives=(
            creative_translation,
            creative_intent,
            creative_hierarchy,
            creative_plan,
            creative_constraints,
            artifact_plan,
            generated_response,
        ),
        bonus=min(len(request.query), 240) / 240 * 0.08,
        penalties=_sequence_penalty(
            (
                *getattr(creative_quality_prediction, "missing_information", ()),
                *getattr(artifact_critic, "missing_information", ()),
            )
        ),
    )
    feasibility = _score(
        0.32,
        positives=(
            route_decision,
            creative_plan,
            creative_constraints,
            runtime_capabilities,
            runtime_compatibility,
            artifact_capability_matrix,
            artifact_refiner,
            artifact_intelligence_synthesis,
        ),
        bonus=0.08 if generated_response else 0,
        penalties=(
            _risk_penalty(creative_quality_prediction)
            + _artifact_risk_penalty(artifact_risk)
            + _sequence_penalty(getattr(runtime_compatibility, "implementation_risks", ())) * 0.6
        ),
    )
    return {
        "concept": concept,
        "execution": execution,
        "artifact": artifact,
        "coherence": coherence,
        "runtime_fit": runtime_fit,
        "originality": originality,
        "clarity": clarity,
        "feasibility": feasibility,
    }


def _score(
    base: float,
    *,
    positives: Sequence[object | None],
    bonus: float = 0,
    penalties: float = 0,
) -> float:
    present = sum(item is not None for item in positives)
    return _clamp(base + present * 0.055 + bonus - penalties)


def _risk_penalty(prediction: CreativeQualityPrediction | None) -> float:
    if prediction is None:
        return 0
    if prediction.predicted_quality_level == "blocked":
        return 0.24
    if prediction.predicted_quality_level == "risky":
        return 0.16
    if prediction.predicted_quality_level == "ambiguous":
        return 0.10
    return 0


def _artifact_risk_penalty(risk: str | None) -> float:
    if risk == "blocked":
        return 0.24
    if risk == "high":
        return 0.18
    if risk == "medium":
        return 0.09
    return 0


def _sequence_penalty(values: Sequence[object]) -> float:
    return min(len(values), 4) * 0.035


def _clamp(value: float) -> float:
    return round(max(0.05, min(0.98, value)), 2)


def _missing_information(
    *,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    creative_quality_prediction: CreativeQualityPrediction | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
    generated_response: str | None,
    artifacts: Sequence[WorkflowArtifact],
) -> tuple[str, ...]:
    missing: list[str] = []
    if route_decision is None:
        missing.append("Route decision is unavailable for critic scope.")
    if creative_translation is None:
        missing.append("Creative translation is unavailable.")
    if creative_intent is None:
        missing.append("Creative intent decomposition is unavailable.")
    if creative_hierarchy is None:
        missing.append("Creative hierarchy plan is unavailable.")
    if creative_strategy is None:
        missing.append("Creative strategy metadata is unavailable.")
    if creative_techniques is None:
        missing.append("Creative technique metadata is unavailable.")
    if creative_plan is None:
        missing.append("Creative execution plan is unavailable.")
    if runtime_capabilities is None:
        missing.append("Runtime capability profile is unavailable.")
    if creative_quality_prediction is not None:
        missing.extend(creative_quality_prediction.missing_information[:3])
    if artifact_plan is None:
        missing.append("Artifact plan is unavailable for artifact-aware critique.")
    if artifact_dependency_graph is None:
        missing.append("Artifact dependency graph is unavailable.")
    if runtime_compatibility is None:
        missing.append("Runtime compatibility metadata is unavailable.")
    if artifact_capability_matrix is None:
        missing.append("Artifact capability matrix is unavailable.")
    if multi_artifact_strategy is None:
        missing.append("Multi-artifact strategy is unavailable.")
    if artifact_critic is not None:
        missing.extend(artifact_critic.missing_information[:3])
    if generated_response is None and not artifacts:
        missing.append(
            "Generated response and artifacts are not available; critique is pre-generation."
        )
    return _dedupe(missing)[:10]


def _unsupported_assumptions(
    *,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifacts: Sequence[WorkflowArtifact],
) -> tuple[str, ...]:
    assumptions: list[str] = [
        "Creative Critic findings are advisory metadata and must not modify, reject, refine, retry, route, preview, or execute anything."
    ]
    if creative_plan is not None and not creative_plan.runtime_available:
        assumptions.append("Live runtime availability is not guaranteed by the plan.")
    if creative_constraints is not None and creative_constraints.runtime_fit != "supported":
        assumptions.append(
            f"Runtime fit is {creative_constraints.runtime_fit}; generation should not assume full runtime support."
        )
    if runtime_capabilities is not None:
        for candidate in runtime_capabilities.candidate_runtimes[:2]:
            assumptions.extend(candidate.risks[:1])
    if runtime_compatibility is not None:
        assumptions.extend(
            f"Unsupported runtime remains non-viable metadata: {item}."
            for item in runtime_compatibility.unsupported_runtimes[:3]
        )
    if artifact_capability_matrix is not None:
        assumptions.extend(artifact_capability_matrix.unsupported_or_risky_capabilities[:3])
    if artifact_critic is not None:
        assumptions.extend(artifact_critic.unsupported_assumptions[:3])
    for artifact in artifacts[:3]:
        if artifact.runtime and artifact.preview_eligible is False:
            assumptions.append(
                f"Artifact {artifact.id} declares runtime {artifact.runtime} without preview eligibility."
            )
    return _dedupe(assumptions)[:10]


def _creative_strengths(
    *,
    qualities: dict[str, float],
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_quality_prediction: CreativeQualityPrediction | None,
    artifact_plan: ArtifactPlan | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifacts: Sequence[WorkflowArtifact],
) -> tuple[str, ...]:
    strengths: list[str] = []
    if creative_intent is not None:
        strengths.append(f"Intent is decomposed around {creative_intent.primary_expression}.")
    if creative_hierarchy is not None:
        strengths.append(
            "Hierarchy gives critic-visible priorities: "
            + ", ".join(
                item.dimension for item in creative_hierarchy.primary_creative_priorities[:3]
            )
            + "."
        )
    if creative_strategy is not None:
        strengths.append(
            f"Strategy {creative_strategy.primary_strategy} carries confidence {creative_strategy.confidence:.2f}."
        )
    if creative_techniques is not None:
        strengths.append(
            f"Technique {creative_techniques.primary_technique} has {creative_techniques.compatibility} strategy compatibility."
        )
    if creative_quality_prediction is not None and creative_quality_prediction.readiness_score >= 70:
        strengths.append(
            f"Quality predictor estimates {creative_quality_prediction.predicted_quality_level} readiness at {creative_quality_prediction.readiness_score}/100."
        )
    if artifact_plan is not None:
        strengths.append(
            f"Artifact plan is critic-visible: {artifact_plan.artifact_type} / {artifact_plan.artifact_family}."
        )
    if artifact_critic is not None and artifact_critic.strengths:
        strengths.append(f"Artifact critic strength: {artifact_critic.strengths[0]}")
    if artifacts:
        strengths.append(f"Artifact-aware critique can inspect {len(artifacts)} generated artifact(s).")
    for label, score in qualities.items():
        if score >= 0.78:
            strengths.append(f"{label.replace('_', ' ').title()} quality is strong ({score:.2f}).")
    return _dedupe(strengths)[:10] or (
        "Creative Critic has enough request context for bounded metadata-only assessment.",
    )


def _creative_weaknesses(
    *,
    qualities: dict[str, float],
    missing_information: tuple[str, ...],
    unsupported_assumptions: tuple[str, ...],
    creative_quality_prediction: CreativeQualityPrediction | None,
    artifact_critic: ArtifactCriticProfile | None,
) -> tuple[str, ...]:
    weaknesses: list[str] = []
    for label, score in qualities.items():
        if score < 0.58:
            weaknesses.append(f"{label.replace('_', ' ').title()} quality is weak ({score:.2f}).")
    if missing_information:
        weaknesses.append(f"Critique is constrained by {len(missing_information)} missing information signal(s).")
    if unsupported_assumptions:
        weaknesses.append(
            f"Critique depends on {len(unsupported_assumptions)} unsupported-assumption guardrail(s)."
        )
    if creative_quality_prediction is not None:
        weaknesses.extend(
            f"Quality predictor weak signal: {item.summary}"
            for item in creative_quality_prediction.weakest_quality_signals[:2]
        )
        weaknesses.extend(creative_quality_prediction.quality_risks[:2])
    if artifact_critic is not None:
        weaknesses.extend(artifact_critic.weaknesses[:2])
        weaknesses.extend(artifact_critic.runtime_concerns[:1])
    return _dedupe(weaknesses)[:10]


def _risk_assessment(
    *,
    qualities: dict[str, float],
    weaknesses: tuple[str, ...],
    missing_information: tuple[str, ...],
    unsupported_assumptions: tuple[str, ...],
    artifact_critic: ArtifactCriticProfile | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
) -> CreativeCriticRiskAssessment:
    core_missing = any(
        item.startswith(
            (
                "Route decision",
                "Creative translation",
                "Creative intent",
                "Creative hierarchy",
                "Creative strategy",
                "Creative technique",
                "Creative execution plan",
                "Runtime capability",
                "Artifact plan",
            )
        )
        for item in missing_information
    )
    if (core_missing and len(missing_information) >= 4) or min(qualities.values()) < 0.20:
        return "blocked"
    if artifact_critic is not None and artifact_critic.risk_assessment == "blocked":
        return "blocked"
    high_risk_inputs = (
        artifact_critic is not None and artifact_critic.risk_assessment == "high"
    )
    if (
        high_risk_inputs
        or min(qualities.values()) < 0.40
        or (len(weaknesses) >= 8 and len(missing_information) >= 4)
    ):
        return "high"
    if (
        sum(qualities.values()) / len(qualities) < 0.70
        or len(weaknesses) >= 2
        or len(missing_information) >= 2
        or len(unsupported_assumptions) >= 3
    ):
        return "medium"
    return "low"


def _improvement_opportunities(
    *,
    risk: CreativeCriticRiskAssessment,
    qualities: dict[str, float],
    missing_information: tuple[str, ...],
    unsupported_assumptions: tuple[str, ...],
    creative_quality_prediction: CreativeQualityPrediction | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
    artifact_intelligence_synthesis: ArtifactIntelligenceSynthesisProfile | None,
) -> tuple[str, ...]:
    opportunities: list[str] = []
    for label, score in sorted(qualities.items(), key=lambda item: item[1])[:3]:
        opportunities.append(
            f"Improve {label.replace('_', ' ')} clarity before expanding scope ({score:.2f})."
        )
    if missing_information:
        opportunities.append(f"Resolve or caveat missing critic input: {missing_information[0]}")
    if unsupported_assumptions:
        opportunities.append(f"Preserve unsupported-assumption caveat: {unsupported_assumptions[0]}")
    if creative_quality_prediction is not None:
        opportunities.extend(creative_quality_prediction.prompt_guidance[:2])
    if artifact_critic is not None:
        opportunities.extend(artifact_critic.improvement_opportunities[:2])
    if artifact_refiner is not None:
        opportunities.extend(artifact_refiner.priority_improvements[:1])
    if artifact_intelligence_synthesis is not None:
        opportunities.append(artifact_intelligence_synthesis.recommended_strategy_summary)
    if risk in {"high", "blocked"}:
        opportunities.append(
            "Keep the critique visible as advisory guidance and ask HITL before scope expansion."
        )
    return _dedupe(opportunities)[:10]


def _hitl_questions(
    *,
    risk: CreativeCriticRiskAssessment,
    missing_information: tuple[str, ...],
    unsupported_assumptions: tuple[str, ...],
    creative_quality_prediction: CreativeQualityPrediction | None,
    artifact_critic: ArtifactCriticProfile | None,
) -> tuple[str, ...]:
    questions: list[str] = []
    if risk in {"high", "blocked"}:
        questions.append(f"Should generation proceed with Creative Critic risk {risk}?")
    if missing_information:
        questions.append(f"Should this missing critic input be resolved: {missing_information[0]}")
    if unsupported_assumptions:
        questions.append(
            f"Should this unsupported assumption be treated as a caveat: {unsupported_assumptions[0]}"
        )
    if creative_quality_prediction is not None:
        questions.extend(creative_quality_prediction.hitl_questions[:2])
    if artifact_critic is not None:
        questions.extend(artifact_critic.hitl_questions[:2])
    return _dedupe(questions)[:8]


def _prompt_guidance(
    *,
    risk: CreativeCriticRiskAssessment,
    opportunities: tuple[str, ...],
) -> tuple[str, ...]:
    guidance = [
        "Use Creative Critic output as metadata-only critique, not as artifact modification or rejection.",
        "Preserve user intent and prior planning metadata before applying critic caveats.",
        (
            "Do not trigger retries, refinement, runtime selection, provider routing, "
            "preview changes, execution, runtime repair, Studio Mode, or HoloMind."
        ),
    ]
    if risk in {"high", "blocked"}:
        guidance.append("Surface critic risk and HITL questions before expanding implementation scope.")
    guidance.extend(opportunities[:3])
    return _dedupe(guidance)[:8]


def _critic_confidence(
    *,
    missing_information: tuple[str, ...],
    route_decision: RouteDecision | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_quality_prediction: CreativeQualityPrediction | None,
    artifact_plan: ArtifactPlan | None,
    artifact_critic: ArtifactCriticProfile | None,
    generated_response: str | None,
    artifacts: Sequence[WorkflowArtifact],
) -> float:
    present = sum(
        item is not None
        for item in (
            route_decision,
            creative_plan,
            creative_strategy,
            creative_techniques,
            creative_quality_prediction,
            artifact_plan,
            artifact_critic,
            generated_response,
        )
    ) + min(len(artifacts), 2)
    return _clamp(0.26 + present * 0.07 - min(len(missing_information), 8) * 0.035)


def _critique_summary(
    *,
    risk: CreativeCriticRiskAssessment,
    qualities: dict[str, float],
    strengths: tuple[str, ...],
    weaknesses: tuple[str, ...],
) -> str:
    lowest_label, lowest_score = min(qualities.items(), key=lambda item: item[1])
    highest_label, highest_score = max(qualities.items(), key=lambda item: item[1])
    return _clip(
        (
            f"Creative critique risk is {risk}. Strongest area is "
            f"{highest_label.replace('_', ' ')} ({highest_score:.2f}); weakest "
            f"area is {lowest_label.replace('_', ' ')} ({lowest_score:.2f}). "
            f"{len(strengths)} strength signal(s) and {len(weaknesses)} weakness "
            "signal(s) are available as advisory metadata only."
        ),
        620,
    )


def _evidence(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_quality_prediction: CreativeQualityPrediction | None,
    artifact_plan: ArtifactPlan | None,
    artifact_critic: ArtifactCriticProfile | None,
    generated_response: str | None,
    artifacts: Sequence[WorkflowArtifact],
) -> tuple[str, ...]:
    evidence = [f"Request: {_clip(request.query, 220)}"]
    if route_decision is not None:
        domains = ", ".join(domain.value for domain in route_decision.domains) or "none"
        evidence.append(f"Route: {route_decision.route.value}; domains {domains}.")
    if creative_strategy is not None:
        evidence.append(
            f"Strategy: {creative_strategy.primary_strategy}; confidence {creative_strategy.confidence:.2f}."
        )
    if creative_techniques is not None:
        evidence.append(
            f"Technique: {creative_techniques.primary_technique}; compatibility {creative_techniques.compatibility}."
        )
    if creative_quality_prediction is not None:
        evidence.append(
            f"Quality predictor: {creative_quality_prediction.predicted_quality_level}; readiness {creative_quality_prediction.readiness_score}/100."
        )
    if artifact_plan is not None:
        evidence.append(f"Artifact plan: {artifact_plan.artifact_type}; {artifact_plan.artifact_family}.")
    if artifact_critic is not None:
        evidence.append(
            f"Artifact critic: {artifact_critic.risk_assessment} risk; {artifact_critic.critique_confidence:.2f} confidence."
        )
    if generated_response is not None:
        evidence.append(f"Generated response available: {len(generated_response)} characters.")
    if artifacts:
        evidence.append(
            "Artifacts available: "
            + ", ".join(f"{artifact.id}:{artifact.language}" for artifact in artifacts[:5])
            + "."
        )
    evidence.append("Authority boundary verified: metadata-only critique.")
    return _dedupe(evidence)[:16]
