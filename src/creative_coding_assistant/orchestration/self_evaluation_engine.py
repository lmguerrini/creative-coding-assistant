"""Metadata-only Self Evaluation Engine for V3.4 creative evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration._metadata_utils import (
    _clamp_score,
    _clip,
    _contains_any,
    _dedupe,
    _score,
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
from creative_coding_assistant.orchestration.creative_critic_engine import (
    CreativeCriticProfile,
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

SelfEvaluationCompleteness = Literal[
    "complete",
    "mostly_complete",
    "partial",
    "blocked",
]
SelfEvaluationRisk = Literal["low", "medium", "high"]
SelfEvaluationAmbiguity = Literal["low", "medium", "high"]

SELF_EVALUATION_AUTHORITY_BOUNDARY = (
    "The Self Evaluation Engine assesses request alignment, coherence, "
    "completeness, risks, gaps, and improvement opportunities as metadata "
    "only; it does not modify outputs, choose final answers, reject outputs, "
    "execute artifacts, select runtimes, route providers or models, change "
    "previews, trigger retries, trigger refinement, run reflection loops, "
    "repair runtime behavior, implement Studio Mode, HoloMind, V4 agents, or "
    "V5 execution optimization."
)


class SelfEvaluationProfile(BaseModel):
    """Inspectable metadata-only assessment of workflow response alignment."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["self_evaluation_engine"] = "self_evaluation_engine"
    self_evaluation_confidence: float = Field(ge=0, le=1)
    evaluation_summary: str = Field(min_length=1, max_length=680)
    request_alignment: float = Field(ge=0, le=1)
    intent_alignment: float = Field(ge=0, le=1)
    constraint_alignment: float = Field(ge=0, le=1)
    artifact_alignment: float = Field(ge=0, le=1)
    runtime_alignment: float = Field(ge=0, le=1)
    creative_coherence: float = Field(ge=0, le=1)
    technical_coherence: float = Field(ge=0, le=1)
    completeness_assessment: SelfEvaluationCompleteness
    ambiguity_assessment: SelfEvaluationAmbiguity
    hallucination_risk: SelfEvaluationRisk
    overreach_risk: SelfEvaluationRisk
    underdelivery_risk: SelfEvaluationRisk
    missing_information: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    unsupported_assumptions: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=10,
    )
    quality_gaps: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    improvement_opportunities: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=10,
    )
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=SELF_EVALUATION_AUTHORITY_BOUNDARY,
        max_length=1180,
    )
    evidence: tuple[str, ...] = Field(min_length=1, max_length=16)


def derive_self_evaluation_profile(
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
    creative_critic: CreativeCriticProfile | None = None,
    generated_response: str | None = None,
    artifacts: Sequence[WorkflowArtifact] = (),
) -> SelfEvaluationProfile:
    """Assess alignment and quality gaps without changing workflow behavior."""

    scores = _alignment_scores(
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
        creative_critic=creative_critic,
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
        artifact_plan=artifact_plan,
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        creative_critic=creative_critic,
        generated_response=generated_response,
        artifacts=artifacts,
    )
    unsupported = _unsupported_assumptions(
        creative_plan=creative_plan,
        creative_constraints=creative_constraints,
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        artifact_critic=artifact_critic,
        creative_critic=creative_critic,
        generated_response=generated_response,
        artifacts=artifacts,
    )
    quality_gaps = _quality_gaps(
        scores=scores,
        missing_information=missing,
        unsupported_assumptions=unsupported,
        creative_quality_prediction=creative_quality_prediction,
        artifact_critic=artifact_critic,
        creative_critic=creative_critic,
    )
    completeness = _completeness_assessment(
        scores=scores,
        missing_information=missing,
        generated_response=generated_response,
        artifacts=artifacts,
    )
    ambiguity = _ambiguity_assessment(
        missing_information=missing,
        quality_gaps=quality_gaps,
        creative_quality_prediction=creative_quality_prediction,
        creative_critic=creative_critic,
    )
    hallucination = _hallucination_risk(
        unsupported_assumptions=unsupported,
        generated_response=generated_response,
        runtime_compatibility=runtime_compatibility,
        creative_critic=creative_critic,
    )
    overreach = _overreach_risk(
        request=request,
        generated_response=generated_response,
        creative_critic=creative_critic,
        artifacts=artifacts,
    )
    underdelivery = _underdelivery_risk(
        request=request,
        completeness=completeness,
        scores=scores,
        generated_response=generated_response,
        artifacts=artifacts,
    )
    opportunities = _improvement_opportunities(
        scores=scores,
        completeness=completeness,
        hallucination_risk=hallucination,
        overreach_risk=overreach,
        underdelivery_risk=underdelivery,
        missing_information=missing,
        unsupported_assumptions=unsupported,
        quality_gaps=quality_gaps,
        creative_critic=creative_critic,
    )

    return SelfEvaluationProfile(
        self_evaluation_confidence=_self_evaluation_confidence(
            missing_information=missing,
            route_decision=route_decision,
            creative_plan=creative_plan,
            creative_critic=creative_critic,
            generated_response=generated_response,
            artifacts=artifacts,
        ),
        evaluation_summary=_evaluation_summary(
            scores=scores,
            completeness=completeness,
            hallucination_risk=hallucination,
            overreach_risk=overreach,
            underdelivery_risk=underdelivery,
            quality_gaps=quality_gaps,
        ),
        request_alignment=scores["request"],
        intent_alignment=scores["intent"],
        constraint_alignment=scores["constraint"],
        artifact_alignment=scores["artifact"],
        runtime_alignment=scores["runtime"],
        creative_coherence=scores["creative_coherence"],
        technical_coherence=scores["technical_coherence"],
        completeness_assessment=completeness,
        ambiguity_assessment=ambiguity,
        hallucination_risk=hallucination,
        overreach_risk=overreach,
        underdelivery_risk=underdelivery,
        missing_information=missing,
        unsupported_assumptions=unsupported,
        quality_gaps=quality_gaps,
        improvement_opportunities=opportunities,
        hitl_questions=_hitl_questions(
            completeness=completeness,
            ambiguity_assessment=ambiguity,
            hallucination_risk=hallucination,
            overreach_risk=overreach,
            underdelivery_risk=underdelivery,
            missing_information=missing,
            unsupported_assumptions=unsupported,
            creative_critic=creative_critic,
        ),
        prompt_guidance=_prompt_guidance(
            completeness=completeness,
            hallucination_risk=hallucination,
            overreach_risk=overreach,
            underdelivery_risk=underdelivery,
            opportunities=opportunities,
        ),
        evidence=_evidence(
            request=request,
            route_decision=route_decision,
            creative_intent=creative_intent,
            creative_plan=creative_plan,
            creative_quality_prediction=creative_quality_prediction,
            artifact_plan=artifact_plan,
            artifact_critic=artifact_critic,
            creative_critic=creative_critic,
            generated_response=generated_response,
            artifacts=artifacts,
        ),
    )


def self_evaluation_prompt_lines(
    profile: SelfEvaluationProfile,
) -> tuple[str, ...]:
    """Render Self Evaluation metadata as compact advisory prompt guidance."""

    lines = [
        f"Authority boundary: {profile.authority_boundary}",
        f"Self evaluation confidence: {profile.self_evaluation_confidence:.2f}.",
        f"Self evaluation summary: {profile.evaluation_summary}",
        (
            "Alignment scores: "
            f"request {profile.request_alignment:.2f}; "
            f"intent {profile.intent_alignment:.2f}; "
            f"constraints {profile.constraint_alignment:.2f}; "
            f"artifact {profile.artifact_alignment:.2f}; "
            f"runtime {profile.runtime_alignment:.2f}; "
            f"creative coherence {profile.creative_coherence:.2f}; "
            f"technical coherence {profile.technical_coherence:.2f}."
        ),
        (
            "Self evaluation status: "
            f"completeness {profile.completeness_assessment}; "
            f"ambiguity {profile.ambiguity_assessment}; "
            f"hallucination risk {profile.hallucination_risk}; "
            f"overreach risk {profile.overreach_risk}; "
            f"underdelivery risk {profile.underdelivery_risk}."
        ),
    ]
    lines.extend(
        f"Self evaluation missing information: {item}"
        for item in profile.missing_information
    )
    lines.extend(
        f"Self evaluation unsupported assumption: {item}"
        for item in profile.unsupported_assumptions
    )
    lines.extend(
        f"Self evaluation quality gap: {item}" for item in profile.quality_gaps
    )
    lines.extend(
        f"Self evaluation improvement opportunity: {item}"
        for item in profile.improvement_opportunities
    )
    lines.extend(
        f"Self evaluation HITL question: {item}" for item in profile.hitl_questions
    )
    lines.extend(
        f"Self evaluation guidance: {item}" for item in profile.prompt_guidance
    )
    return tuple(lines[:64])


def _alignment_scores(
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
    creative_critic: CreativeCriticProfile | None,
    generated_response: str | None,
    artifacts: Sequence[WorkflowArtifact],
) -> dict[str, float]:
    response_coverage = _request_keyword_coverage(request.query, generated_response)
    readiness = (
        creative_quality_prediction.readiness_score / 100
        if creative_quality_prediction is not None
        else 0
    )
    critic_average = _critic_average(creative_critic)
    artifact_scores = [
        artifact.quality_score
        for artifact in artifacts
        if artifact.quality_score is not None
    ]
    artifact_average = (
        sum(artifact_scores) / len(artifact_scores) if artifact_scores else 0
    )
    critic_risk_penalty = _critic_risk_penalty(creative_critic)

    request_alignment = _score(
        0.34,
        positives=(
            route_decision,
            creative_translation,
            creative_plan,
            creative_critic,
        ),
        bonus=response_coverage * 0.28 + readiness * 0.06,
        penalties=critic_risk_penalty,
    )
    intent_alignment = _score(
        0.32,
        positives=(
            creative_translation,
            creative_intent,
            creative_hierarchy,
            creative_strategy,
            creative_techniques,
            symbolic_narrative,
            semantic_motif,
        ),
        bonus=response_coverage * 0.12 + critic_average * 0.08,
        penalties=critic_risk_penalty * 0.7,
    )
    constraint_alignment = _score(
        0.30,
        positives=(
            creative_constraints,
            creative_constraint_priorities,
            creative_tradeoffs,
            creative_quality_prediction,
            creative_critic,
        ),
        bonus=readiness * 0.08,
        penalties=(
            _sequence_penalty(getattr(creative_quality_prediction, "quality_risks", ()))
            + critic_risk_penalty
        ),
    )
    artifact_alignment = _score(
        0.28,
        positives=(
            artifact_plan,
            artifact_dependency_graph,
            artifact_capability_matrix,
            multi_artifact_strategy,
            artifact_critic,
            artifact_refiner,
            artifact_intelligence_synthesis,
            artifact_merge_planner,
            artifact_export_intelligence,
        ),
        bonus=artifact_average * 0.14 + min(len(artifacts), 3) * 0.04,
        penalties=_artifact_risk_penalty(artifact_critic),
    )
    runtime_alignment = _score(
        0.30,
        positives=(
            route_decision,
            runtime_capabilities,
            runtime_compatibility,
            artifact_capability_matrix,
            creative_plan,
        ),
        bonus=0.08 if any(artifact.preview_eligible for artifact in artifacts) else 0,
        penalties=(
            _sequence_penalty(
                getattr(runtime_compatibility, "unsupported_runtimes", ())
            )
            + _sequence_penalty(
                getattr(runtime_compatibility, "implementation_risks", ())
            )
        ),
    )
    creative_coherence = _score(
        0.33,
        positives=(
            creative_hierarchy,
            creative_strategy,
            creative_techniques,
            symbolic_narrative,
            creative_composition,
            generative_structure,
            semantic_motif,
            emotional_consistency,
            cross_modality,
            audio_visual_scene,
        ),
        bonus=critic_average * 0.10 + readiness * 0.06,
        penalties=critic_risk_penalty * 0.7,
    )
    technical_coherence = _score(
        0.31,
        positives=(
            creative_plan,
            creative_constraints,
            runtime_capabilities,
            procedural_structure,
            artifact_plan,
            artifact_dependency_graph,
            runtime_compatibility,
            artifact_capability_matrix,
            artifact_refiner,
            artifact_intelligence_synthesis,
        ),
        bonus=artifact_average * 0.08,
        penalties=_artifact_risk_penalty(artifact_critic)
        + _sequence_penalty(getattr(runtime_compatibility, "implementation_risks", ())),
    )
    return {
        "request": request_alignment,
        "intent": intent_alignment,
        "constraint": constraint_alignment,
        "artifact": artifact_alignment,
        "runtime": runtime_alignment,
        "creative_coherence": creative_coherence,
        "technical_coherence": technical_coherence,
    }


def _sequence_penalty(values: Sequence[object]) -> float:
    return min(len(values), 4) * 0.035


def _critic_average(profile: CreativeCriticProfile | None) -> float:
    if profile is None:
        return 0
    values = (
        profile.concept_quality,
        profile.execution_quality,
        profile.artifact_quality,
        profile.coherence_quality,
        profile.runtime_fit_quality,
        profile.originality_quality,
        profile.clarity_quality,
        profile.feasibility_quality,
    )
    return sum(values) / len(values)


def _critic_risk_penalty(profile: CreativeCriticProfile | None) -> float:
    if profile is None:
        return 0
    if profile.risk_assessment == "blocked":
        return 0.18
    if profile.risk_assessment == "high":
        return 0.13
    if profile.risk_assessment == "medium":
        return 0.06
    return 0


def _artifact_risk_penalty(profile: ArtifactCriticProfile | None) -> float:
    if profile is None:
        return 0
    if profile.risk_assessment == "blocked":
        return 0.20
    if profile.risk_assessment == "high":
        return 0.15
    if profile.risk_assessment == "medium":
        return 0.07
    return 0


def _request_keyword_coverage(query: str, generated_response: str | None) -> float:
    if not generated_response:
        return 0
    query_terms = _keywords(query)
    if not query_terms:
        return 0
    response = generated_response.lower()
    matched = sum(1 for term in query_terms if term in response)
    return matched / len(query_terms)


def _keywords(value: str) -> tuple[str, ...]:
    stopwords = {
        "about",
        "after",
        "also",
        "and",
        "are",
        "but",
        "can",
        "code",
        "for",
        "from",
        "generate",
        "give",
        "have",
        "into",
        "make",
        "need",
        "please",
        "should",
        "that",
        "the",
        "this",
        "with",
        "you",
    }
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in value)
    terms = [
        term for term in cleaned.split() if len(term) >= 4 and term not in stopwords
    ]
    return _dedupe(terms, clip_limit=None)[:12]


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
    artifact_plan: ArtifactPlan | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    creative_critic: CreativeCriticProfile | None,
    generated_response: str | None,
    artifacts: Sequence[WorkflowArtifact],
) -> tuple[str, ...]:
    missing: list[str] = []
    if route_decision is None:
        missing.append("Route decision is unavailable for self evaluation.")
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
    if artifact_plan is None:
        missing.append("Artifact plan is unavailable for artifact alignment.")
    if runtime_compatibility is None:
        missing.append("Runtime compatibility metadata is unavailable.")
    if artifact_capability_matrix is None:
        missing.append("Artifact capability matrix is unavailable.")
    if creative_critic is None:
        missing.append("Creative Critic metadata is unavailable.")
    else:
        missing.extend(creative_critic.missing_information[:3])
    if generated_response is None:
        missing.append(
            "Generated response text is not available for response alignment."
        )
    if generated_response is not None and not generated_response.strip():
        missing.append("Generated response text is empty.")
    if generated_response is not None and _contains_any(
        generated_response,
        ("[todo]", "placeholder", "lorem ipsum"),
    ):
        missing.append("Generated response appears to contain placeholder content.")
    if (
        not artifacts
        and generated_response is not None
        and _query_requests_artifact_context(generated_response)
    ):
        missing.append(
            "Generated response references artifacts but no artifacts are available."
        )
    return _dedupe(missing)[:10]


def _query_requests_artifact_context(value: str) -> bool:
    return _contains_any(value, ("artifact", "sketch", "component", "shader", "canvas"))


def _unsupported_assumptions(
    *,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    artifact_critic: ArtifactCriticProfile | None,
    creative_critic: CreativeCriticProfile | None,
    generated_response: str | None,
    artifacts: Sequence[WorkflowArtifact],
) -> tuple[str, ...]:
    assumptions: list[str] = [
        "Self Evaluation findings are advisory metadata and must not modify, reject, refine, retry, route, preview, execute, or select anything."  # noqa: E501
    ]
    if creative_plan is not None and not creative_plan.runtime_available:
        assumptions.append("Runtime availability remains unverified by the plan.")
    if (
        creative_constraints is not None
        and creative_constraints.runtime_fit != "supported"
    ):
        assumptions.append(
            f"Runtime fit is {creative_constraints.runtime_fit}; response should not imply full runtime support."
        )
    if runtime_compatibility is not None:
        assumptions.extend(
            f"Unsupported runtime must remain caveated: {item}."
            for item in runtime_compatibility.unsupported_runtimes[:3]
        )
    if artifact_capability_matrix is not None:
        assumptions.extend(
            artifact_capability_matrix.unsupported_or_risky_capabilities[:3]
        )
    if artifact_critic is not None:
        assumptions.extend(artifact_critic.unsupported_assumptions[:3])
    if creative_critic is not None:
        assumptions.extend(creative_critic.unsupported_assumptions[:3])
    if generated_response is not None and _contains_any(
        generated_response,
        ("guaranteed", "tested", "production-ready", "runs everywhere", "no caveats"),
    ):
        assumptions.append(
            "Generated response uses certainty language that is not supported by evaluation metadata."
        )
    for artifact in artifacts[:3]:
        if artifact.runtime and artifact.preview_eligible is False:
            assumptions.append(
                f"Artifact {artifact.id} declares runtime {artifact.runtime} without preview eligibility."
            )
    return _dedupe(assumptions)[:10]


def _quality_gaps(
    *,
    scores: dict[str, float],
    missing_information: tuple[str, ...],
    unsupported_assumptions: tuple[str, ...],
    creative_quality_prediction: CreativeQualityPrediction | None,
    artifact_critic: ArtifactCriticProfile | None,
    creative_critic: CreativeCriticProfile | None,
) -> tuple[str, ...]:
    gaps: list[str] = []
    for label, score in scores.items():
        if score < 0.58:
            gaps.append(
                f"{label.replace('_', ' ').title()} is below evaluation threshold ({score:.2f})."
            )
    if missing_information:
        gaps.append(
            f"Self evaluation is constrained by {len(missing_information)} missing information signal(s)."
        )
    if unsupported_assumptions:
        gaps.append(
            f"Self evaluation found {len(unsupported_assumptions)} unsupported-assumption guardrail(s)."
        )
    if creative_quality_prediction is not None:
        gaps.extend(creative_quality_prediction.quality_risks[:2])
    if artifact_critic is not None:
        gaps.extend(artifact_critic.weaknesses[:2])
    if creative_critic is not None:
        gaps.extend(creative_critic.creative_weaknesses[:2])
    return _dedupe(gaps)[:10]


def _completeness_assessment(
    *,
    scores: dict[str, float],
    missing_information: tuple[str, ...],
    generated_response: str | None,
    artifacts: Sequence[WorkflowArtifact],
) -> SelfEvaluationCompleteness:
    average = sum(scores.values()) / len(scores)
    core_missing = any(
        item.startswith(
            (
                "Route decision",
                "Creative translation",
                "Creative intent",
                "Creative execution plan",
                "Generated response text is empty",
            )
        )
        for item in missing_information
    )
    if core_missing and len(missing_information) >= 5:
        return "blocked"
    if (
        average >= 0.78
        and generated_response
        and (artifacts or len(generated_response) > 240)
    ):
        return "complete"
    if average >= 0.60 and generated_response:
        return "mostly_complete"
    if average >= 0.48:
        return "partial"
    return "blocked"


def _ambiguity_assessment(
    *,
    missing_information: tuple[str, ...],
    quality_gaps: tuple[str, ...],
    creative_quality_prediction: CreativeQualityPrediction | None,
    creative_critic: CreativeCriticProfile | None,
) -> SelfEvaluationAmbiguity:
    if (
        len(missing_information) >= 5
        or len(quality_gaps) >= 7
        or (
            creative_quality_prediction is not None
            and creative_quality_prediction.predicted_quality_level
            in {"ambiguous", "blocked"}
        )
        or (
            creative_critic is not None
            and creative_critic.risk_assessment in {"high", "blocked"}
        )
    ):
        return "high"
    if len(missing_information) >= 2 or len(quality_gaps) >= 3:
        return "medium"
    return "low"


def _hallucination_risk(
    *,
    unsupported_assumptions: tuple[str, ...],
    generated_response: str | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    creative_critic: CreativeCriticProfile | None,
) -> SelfEvaluationRisk:
    certainty_claim = generated_response is not None and _contains_any(
        generated_response,
        (
            "guaranteed",
            "tested",
            "production-ready",
            "runs everywhere",
            "all runtimes",
            "no caveats",
        ),
    )
    unsupported_runtime = runtime_compatibility is not None and bool(
        runtime_compatibility.unsupported_runtimes
    )
    critic_risky = creative_critic is not None and creative_critic.risk_assessment in {
        "high",
        "blocked",
    }
    if certainty_claim and (unsupported_runtime or len(unsupported_assumptions) >= 3):
        return "high"
    if certainty_claim or critic_risky or len(unsupported_assumptions) >= 4:
        return "medium"
    return "low"


def _overreach_risk(
    *,
    request: AssistantRequest,
    generated_response: str | None,
    creative_critic: CreativeCriticProfile | None,
    artifacts: Sequence[WorkflowArtifact],
) -> SelfEvaluationRisk:
    if generated_response is None:
        return "low"
    response = generated_response.lower()
    scope_expansion = _contains_any(
        response,
        (
            "full application",
            "complete platform",
            "production deployment",
            "multi-agent",
            "autonomous repair",
            "provider routing",
        ),
    )
    simple_request = _contains_any(request.query, ("simple", "minimal", "explain"))
    critic_risky = creative_critic is not None and creative_critic.risk_assessment in {
        "high",
        "blocked",
    }
    if scope_expansion and (simple_request or critic_risky):
        return "high"
    if scope_expansion or len(artifacts) > 4:
        return "medium"
    return "low"


def _underdelivery_risk(
    *,
    request: AssistantRequest,
    completeness: SelfEvaluationCompleteness,
    scores: dict[str, float],
    generated_response: str | None,
    artifacts: Sequence[WorkflowArtifact],
) -> SelfEvaluationRisk:
    query_needs_artifact = _contains_any(
        request.query,
        ("generate", "build", "create", "sketch", "shader", "component", "artifact"),
    )
    if completeness == "blocked" or (
        query_needs_artifact and generated_response is not None and not artifacts
    ):
        return "high"
    if completeness == "partial" or min(scores.values()) < 0.48:
        return "medium"
    return "low"


def _improvement_opportunities(
    *,
    scores: dict[str, float],
    completeness: SelfEvaluationCompleteness,
    hallucination_risk: SelfEvaluationRisk,
    overreach_risk: SelfEvaluationRisk,
    underdelivery_risk: SelfEvaluationRisk,
    missing_information: tuple[str, ...],
    unsupported_assumptions: tuple[str, ...],
    quality_gaps: tuple[str, ...],
    creative_critic: CreativeCriticProfile | None,
) -> tuple[str, ...]:
    opportunities: list[str] = []
    for label, score in sorted(scores.items(), key=lambda item: item[1])[:3]:
        opportunities.append(
            f"Improve {label.replace('_', ' ')} evidence before expanding scope ({score:.2f})."
        )
    if completeness in {"partial", "blocked"} and missing_information:
        opportunities.append(
            f"Resolve or caveat missing self-evaluation input: {missing_information[0]}"
        )
    if hallucination_risk != "low" and unsupported_assumptions:
        opportunities.append(
            f"Reduce certainty around unsupported assumption: {unsupported_assumptions[0]}"
        )
    if overreach_risk != "low":
        opportunities.append(
            "Trim unsupported scope expansion and keep output aligned to the request."
        )
    if underdelivery_risk != "low":
        opportunities.append(
            "Make the expected deliverable explicit or ask for HITL clarification."
        )
    if creative_critic is not None:
        opportunities.extend(creative_critic.improvement_opportunities[:2])
    opportunities.extend(quality_gaps[:2])
    return _dedupe(opportunities)[:10]


def _hitl_questions(
    *,
    completeness: SelfEvaluationCompleteness,
    ambiguity_assessment: SelfEvaluationAmbiguity,
    hallucination_risk: SelfEvaluationRisk,
    overreach_risk: SelfEvaluationRisk,
    underdelivery_risk: SelfEvaluationRisk,
    missing_information: tuple[str, ...],
    unsupported_assumptions: tuple[str, ...],
    creative_critic: CreativeCriticProfile | None,
) -> tuple[str, ...]:
    questions: list[str] = []
    if completeness in {"partial", "blocked"}:
        questions.append(
            f"Should missing self-evaluation inputs be resolved before relying on {completeness} output?"
        )
    if ambiguity_assessment == "high" and missing_information:
        questions.append(f"Should this ambiguity be resolved: {missing_information[0]}")
    if hallucination_risk == "high" and unsupported_assumptions:
        questions.append(
            f"Should this unsupported assumption be softened: {unsupported_assumptions[0]}"
        )
    if overreach_risk == "high":
        questions.append("Should scope be narrowed to avoid overreach?")
    if underdelivery_risk == "high":
        questions.append("Should the deliverable be clarified to avoid underdelivery?")
    if creative_critic is not None:
        questions.extend(creative_critic.hitl_questions[:2])
    return _dedupe(questions)[:8]


def _prompt_guidance(
    *,
    completeness: SelfEvaluationCompleteness,
    hallucination_risk: SelfEvaluationRisk,
    overreach_risk: SelfEvaluationRisk,
    underdelivery_risk: SelfEvaluationRisk,
    opportunities: tuple[str, ...],
) -> tuple[str, ...]:
    guidance = [
        "Use Self Evaluation output as metadata-only assessment, not as output modification, rejection, retry, refinement, routing, preview, or execution behavior.",  # noqa: E501
        "Preserve the user's request and existing planning metadata before applying evaluation caveats.",
        "Do not implement reflection loops, automatic improvement, runtime repair, provider routing, Studio Mode, HoloMind, V4 agents, or V5 optimization.",  # noqa: E501
    ]
    if completeness in {"partial", "blocked"}:
        guidance.append(
            "Surface missing evaluation inputs as caveats rather than silently filling gaps."
        )
    if hallucination_risk != "low":
        guidance.append(
            "Avoid certainty claims that are unsupported by workflow evidence."
        )
    if overreach_risk != "low":
        guidance.append("Keep scope bounded to the requested deliverable.")
    if underdelivery_risk != "low":
        guidance.append(
            "Make any incomplete deliverable explicit and ask HITL if needed."
        )
    guidance.extend(opportunities[:3])
    return _dedupe(guidance)[:8]


def _self_evaluation_confidence(
    *,
    missing_information: tuple[str, ...],
    route_decision: RouteDecision | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_critic: CreativeCriticProfile | None,
    generated_response: str | None,
    artifacts: Sequence[WorkflowArtifact],
) -> float:
    present = sum(
        item is not None
        for item in (
            route_decision,
            creative_plan,
            creative_critic,
            generated_response,
        )
    ) + min(len(artifacts), 2)
    return _clamp_score(
        0.28 + present * 0.09 - min(len(missing_information), 8) * 0.035
    )


def _evaluation_summary(
    *,
    scores: dict[str, float],
    completeness: SelfEvaluationCompleteness,
    hallucination_risk: SelfEvaluationRisk,
    overreach_risk: SelfEvaluationRisk,
    underdelivery_risk: SelfEvaluationRisk,
    quality_gaps: tuple[str, ...],
) -> str:
    lowest_label, lowest_score = min(scores.items(), key=lambda item: item[1])
    highest_label, highest_score = max(scores.items(), key=lambda item: item[1])
    return _clip(
        (
            f"Self evaluation is {completeness}. Strongest signal is "
            f"{highest_label.replace('_', ' ')} ({highest_score:.2f}); weakest "
            f"signal is {lowest_label.replace('_', ' ')} ({lowest_score:.2f}). "
            f"Risks are hallucination {hallucination_risk}, overreach "
            f"{overreach_risk}, and underdelivery {underdelivery_risk}, with "
            f"{len(quality_gaps)} quality gap(s) captured as advisory metadata."
        ),
        680,
    )


def _evidence(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_quality_prediction: CreativeQualityPrediction | None,
    artifact_plan: ArtifactPlan | None,
    artifact_critic: ArtifactCriticProfile | None,
    creative_critic: CreativeCriticProfile | None,
    generated_response: str | None,
    artifacts: Sequence[WorkflowArtifact],
) -> tuple[str, ...]:
    evidence = [f"Request: {_clip(request.query, 220)}"]
    if route_decision is not None:
        domains = ", ".join(domain.value for domain in route_decision.domains) or "none"
        evidence.append(f"Route: {route_decision.route.value}; domains {domains}.")
    if creative_intent is not None:
        evidence.append(f"Intent: {_clip(creative_intent.primary_expression, 180)}.")
    if creative_plan is not None:
        evidence.append(f"Plan: {_clip(creative_plan.generation_strategy, 220)}")
    if creative_quality_prediction is not None:
        evidence.append(
            f"Quality predictor: {creative_quality_prediction.predicted_quality_level}; readiness {creative_quality_prediction.readiness_score}/100."  # noqa: E501
        )
    if artifact_plan is not None:
        evidence.append(
            f"Artifact plan: {artifact_plan.artifact_type}; {artifact_plan.artifact_family}."
        )
    if artifact_critic is not None:
        evidence.append(
            f"Artifact critic: {artifact_critic.risk_assessment} risk; {artifact_critic.critique_confidence:.2f} confidence."  # noqa: E501
        )
    if creative_critic is not None:
        evidence.append(
            f"Creative critic: {creative_critic.risk_assessment} risk; {creative_critic.critic_confidence:.2f} confidence."  # noqa: E501
        )
    if generated_response is not None:
        evidence.append(
            f"Generated response available: {len(generated_response)} characters."
        )
    if artifacts:
        evidence.append(
            "Artifacts available: "
            + ", ".join(
                f"{artifact.id}:{artifact.language}" for artifact in artifacts[:5]
            )
            + "."
        )
    evidence.append("Authority boundary verified: metadata-only self evaluation.")
    return _dedupe(evidence)[:16]
