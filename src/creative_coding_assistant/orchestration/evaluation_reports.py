"""Metadata-only Evaluation Reports capability for V3.4 evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration._metadata_utils import _clip, _dedupe
from creative_coding_assistant.orchestration.consistency_validation_engine import (
    ConsistencyValidationProfile,
)
from creative_coding_assistant.orchestration.creative_confidence_engine import (
    CreativeConfidenceProfile,
    ExpectedHumanReviewNeed,
)
from creative_coding_assistant.orchestration.creative_critic_engine import (
    CreativeCriticProfile,
)
from creative_coding_assistant.orchestration.creative_improvement_planner import (
    CreativeImprovementPlannerProfile,
)
from creative_coding_assistant.orchestration.creative_score_engine import (
    CreativeScoreProfile,
)
from creative_coding_assistant.orchestration.reflection_loop_engine import (
    ReflectionLoopProfile,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.self_evaluation_engine import (
    SelfEvaluationProfile,
)

EvaluationReportSource = Literal[
    "creative_critic",
    "self_evaluation",
    "creative_improvement_planner",
    "reflection_loop",
    "creative_confidence",
    "creative_score",
    "consistency_validation",
]
EvaluationReportDependencyStatus = Literal["available", "missing"]

EVALUATION_REPORT_AUTHORITY_BOUNDARY = (
    "The Evaluation Reports capability summarizes, explains, traces, and "
    "exposes provenance across existing V3.4 evaluation metadata only; it "
    "does not modify outputs, regenerate responses, execute artifacts, "
    "trigger refinement, trigger retries, change routing, select runtimes, "
    "alter previews, invoke future V4 agents, or perform future inspector, "
    "dashboard, studio, optimization, or learning behavior."
)


class EvaluationTraceEntry(BaseModel):
    """One ordered provenance step in the evaluation report."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    step: int = Field(ge=1)
    source: EvaluationReportSource
    role: str = Field(min_length=1, max_length=120)
    contribution: str = Field(min_length=1, max_length=420)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=4)


class EvaluationProvenanceEntry(BaseModel):
    """Source-level provenance for report consumers."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source: EvaluationReportSource
    role: str = Field(min_length=1, max_length=120)
    summary: str = Field(min_length=1, max_length=420)
    confidence: float | None = Field(default=None, ge=0, le=1)
    hitl_recommendation: ExpectedHumanReviewNeed | None = None


class EvaluationDependency(BaseModel):
    """Dependency availability for the report pipeline."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source: EvaluationReportSource
    status: EvaluationReportDependencyStatus
    required: bool = True
    note: str = Field(min_length=1, max_length=260)


class EvaluationEvidenceLink(BaseModel):
    """Compact evidence link used by the report."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source: EvaluationReportSource
    claim: str = Field(min_length=1, max_length=320)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=4)


class EvaluationReportProfile(BaseModel):
    """Unified V3.4 evaluation report derived from prior metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["evaluation_reports"] = "evaluation_reports"
    serialization_version: Literal["v1"] = "v1"
    executive_summary: str = Field(min_length=1, max_length=720)
    quality_summary: str = Field(min_length=1, max_length=620)
    confidence_summary: str = Field(min_length=1, max_length=620)
    consistency_summary: str = Field(min_length=1, max_length=620)
    improvement_summary: str = Field(min_length=1, max_length=620)
    score_summary: str = Field(min_length=1, max_length=620)
    strengths: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    weaknesses: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    risks: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    recommendations: tuple[str, ...] = Field(min_length=1, max_length=10)
    hitl_recommendation: ExpectedHumanReviewNeed
    evaluation_trace: tuple[EvaluationTraceEntry, ...] = Field(
        min_length=1,
        max_length=12,
    )
    evaluation_provenance: tuple[EvaluationProvenanceEntry, ...] = Field(
        min_length=1,
        max_length=12,
    )
    evaluation_explainability: tuple[str, ...] = Field(min_length=1, max_length=10)
    evaluation_dependencies: tuple[EvaluationDependency, ...] = Field(
        min_length=7,
        max_length=7,
    )
    evidence_chain: tuple[EvaluationEvidenceLink, ...] = Field(
        min_length=1,
        max_length=16,
    )
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=EVALUATION_REPORT_AUTHORITY_BOUNDARY,
        max_length=1120,
    )


def derive_evaluation_report_profile(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    creative_score: CreativeScoreProfile | None,
    consistency_validation: ConsistencyValidationProfile | None,
    planning_metadata: Sequence[object] = (),
) -> EvaluationReportProfile:
    """Aggregate V3.4 evaluation metadata into one advisory report."""

    dependencies = _evaluation_dependencies(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=creative_improvement_planner,
        reflection_loop=reflection_loop,
        creative_confidence=creative_confidence,
        creative_score=creative_score,
        consistency_validation=consistency_validation,
    )
    hitl = _hitl_recommendation(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=creative_improvement_planner,
        reflection_loop=reflection_loop,
        creative_confidence=creative_confidence,
        creative_score=creative_score,
        consistency_validation=consistency_validation,
        dependencies=dependencies,
    )
    strengths = _strengths(
        creative_critic=creative_critic,
        creative_confidence=creative_confidence,
        creative_score=creative_score,
    )
    weaknesses = _weaknesses(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_confidence=creative_confidence,
        creative_score=creative_score,
        consistency_validation=consistency_validation,
    )
    risks = _risks(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        reflection_loop=reflection_loop,
        creative_confidence=creative_confidence,
        creative_score=creative_score,
        consistency_validation=consistency_validation,
        dependencies=dependencies,
    )
    recommendations = _recommendations(
        creative_improvement_planner=creative_improvement_planner,
        reflection_loop=reflection_loop,
        creative_confidence=creative_confidence,
        creative_score=creative_score,
        consistency_validation=consistency_validation,
        hitl=hitl,
    )
    trace = _evaluation_trace(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=creative_improvement_planner,
        reflection_loop=reflection_loop,
        creative_confidence=creative_confidence,
        creative_score=creative_score,
        consistency_validation=consistency_validation,
    )
    provenance = _evaluation_provenance(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=creative_improvement_planner,
        reflection_loop=reflection_loop,
        creative_confidence=creative_confidence,
        creative_score=creative_score,
        consistency_validation=consistency_validation,
    )
    evidence = _evidence_chain(
        request=request,
        route_decision=route_decision,
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        creative_improvement_planner=creative_improvement_planner,
        reflection_loop=reflection_loop,
        creative_confidence=creative_confidence,
        creative_score=creative_score,
        consistency_validation=consistency_validation,
        planning_metadata=planning_metadata,
    )

    return EvaluationReportProfile(
        executive_summary=_executive_summary(
            creative_score=creative_score,
            creative_confidence=creative_confidence,
            consistency_validation=consistency_validation,
            hitl=hitl,
        ),
        quality_summary=_quality_summary(
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
        ),
        confidence_summary=_confidence_summary(creative_confidence),
        consistency_summary=_consistency_summary(consistency_validation),
        improvement_summary=_improvement_summary(
            creative_improvement_planner,
            reflection_loop,
        ),
        score_summary=_score_summary(creative_score),
        strengths=strengths,
        weaknesses=weaknesses,
        risks=risks,
        recommendations=recommendations,
        hitl_recommendation=hitl,
        evaluation_trace=trace,
        evaluation_provenance=provenance,
        evaluation_explainability=_evaluation_explainability(
            hitl=hitl,
            dependencies=dependencies,
            creative_score=creative_score,
            creative_confidence=creative_confidence,
            consistency_validation=consistency_validation,
        ),
        evaluation_dependencies=dependencies,
        evidence_chain=evidence,
        prompt_guidance=_prompt_guidance(hitl=hitl),
    )


def evaluation_report_prompt_lines(
    profile: EvaluationReportProfile,
) -> tuple[str, ...]:
    """Render evaluation report metadata as compact prompt guidance."""

    lines = [
        f"Authority boundary: {profile.authority_boundary}",
        f"Serialization version: {profile.serialization_version}.",
        f"Executive summary: {profile.executive_summary}",
        f"Quality summary: {profile.quality_summary}",
        f"Confidence summary: {profile.confidence_summary}",
        f"Consistency summary: {profile.consistency_summary}",
        f"Improvement summary: {profile.improvement_summary}",
        f"Score summary: {profile.score_summary}",
        f"HITL recommendation: {profile.hitl_recommendation}.",
    ]
    lines.extend(f"Evaluation strength: {item}" for item in profile.strengths)
    lines.extend(f"Evaluation weakness: {item}" for item in profile.weaknesses)
    lines.extend(f"Evaluation risk: {item}" for item in profile.risks)
    lines.extend(f"Evaluation recommendation: {item}" for item in profile.recommendations)
    lines.extend(
        (
            "Evaluation trace: "
            f"{item.step}. {item.source}; {item.contribution}"
        )
        for item in profile.evaluation_trace
    )
    lines.extend(
        f"Evaluation provenance: {item.source}; {item.summary}"
        for item in profile.evaluation_provenance
    )
    lines.extend(
        f"Evaluation explainability: {item}"
        for item in profile.evaluation_explainability
    )
    lines.extend(
        (
            "Evaluation dependency: "
            f"{item.source}; {item.status}; {item.note}"
        )
        for item in profile.evaluation_dependencies
    )
    lines.extend(
        f"Evaluation evidence: {item.source}; {item.claim}"
        for item in profile.evidence_chain[:8]
    )
    lines.extend(f"Evaluation prompt guidance: {item}" for item in profile.prompt_guidance)
    return tuple(lines[:72])


def _executive_summary(
    *,
    creative_score: CreativeScoreProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    consistency_validation: ConsistencyValidationProfile | None,
    hitl: ExpectedHumanReviewNeed,
) -> str:
    score = (
        f"{creative_score.score_band} ({creative_score.overall_creative_score:.1f}/100)"
        if creative_score is not None
        else "unscored"
    )
    confidence = (
        f"{creative_confidence.confidence_level} confidence"
        if creative_confidence is not None
        else "unknown confidence"
    )
    consistency = (
        f"{consistency_validation.consistency_status} consistency"
        if consistency_validation is not None
        else "unvalidated consistency"
    )
    return _clip(
        (
            "Evaluation report aggregates V3.4 metadata into an advisory "
            f"{score} assessment with {confidence}, {consistency}, and "
            f"{hitl} HITL recommendation. It summarizes only existing "
            "evaluation metadata and cannot change execution behavior."
        ),
        720,
    )


def _quality_summary(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
) -> str:
    parts: list[str] = []
    if creative_critic is not None:
        parts.append(
            f"critic risk {creative_critic.risk_assessment} with "
            f"{creative_critic.critic_confidence:.2f} confidence"
        )
    if self_evaluation is not None:
        parts.append(
            f"self-evaluation {self_evaluation.completeness_assessment} "
            f"and {self_evaluation.ambiguity_assessment} ambiguity"
        )
    return _clip(
        "Quality summary combines " + "; ".join(parts) + "."
        if parts
        else "Quality summary has insufficient critique metadata.",
        620,
    )


def _confidence_summary(
    creative_confidence: CreativeConfidenceProfile | None,
) -> str:
    if creative_confidence is None:
        return "Confidence summary is unavailable because Creative Confidence metadata is missing."
    return _clip(creative_confidence.confidence_summary, 620)


def _consistency_summary(
    consistency_validation: ConsistencyValidationProfile | None,
) -> str:
    if consistency_validation is None:
        return "Consistency summary is unavailable because Consistency Validation metadata is missing."
    return _clip(consistency_validation.consistency_summary, 620)


def _improvement_summary(
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
) -> str:
    parts: list[str] = []
    if creative_improvement_planner is not None:
        parts.append(creative_improvement_planner.improvement_summary)
    if reflection_loop is not None:
        parts.append(reflection_loop.reflection_summary)
    return _clip(
        " ".join(parts)
        if parts
        else "Improvement summary is unavailable because improvement and reflection metadata are missing.",
        620,
    )


def _score_summary(creative_score: CreativeScoreProfile | None) -> str:
    if creative_score is None:
        return "Score summary is unavailable because Creative Score metadata is missing."
    return _clip(creative_score.score_summary, 620)


def _strengths(
    *,
    creative_critic: CreativeCriticProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    creative_score: CreativeScoreProfile | None,
) -> tuple[str, ...]:
    strengths: list[str] = []
    if creative_critic is not None:
        strengths.extend(creative_critic.creative_strengths[:4])
    if creative_confidence is not None:
        strengths.extend(creative_confidence.confidence_strengths[:3])
    if creative_score is not None:
        strengths.extend(creative_score.strengths[:3])
        strengths.extend(creative_score.positive_contributions[:2])
    return _dedupe(strengths, clip_limit=320)[:10]


def _weaknesses(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    creative_score: CreativeScoreProfile | None,
    consistency_validation: ConsistencyValidationProfile | None,
) -> tuple[str, ...]:
    weaknesses: list[str] = []
    if creative_critic is not None:
        weaknesses.extend(creative_critic.creative_weaknesses[:3])
    if self_evaluation is not None:
        weaknesses.extend(self_evaluation.quality_gaps[:3])
    if creative_confidence is not None:
        weaknesses.extend(creative_confidence.confidence_weaknesses[:2])
    if creative_score is not None:
        weaknesses.extend(creative_score.weaknesses[:2])
        weaknesses.extend(creative_score.negative_contributions[:2])
    if consistency_validation is not None:
        weaknesses.extend(consistency_validation.detected_conflicts[:2])
    return _dedupe(weaknesses, clip_limit=320)[:10]


def _risks(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    creative_score: CreativeScoreProfile | None,
    consistency_validation: ConsistencyValidationProfile | None,
    dependencies: tuple[EvaluationDependency, ...],
) -> tuple[str, ...]:
    risks: list[str] = []
    if creative_critic is not None:
        risks.append(f"Creative Critic risk assessment is {creative_critic.risk_assessment}.")
        risks.extend(creative_critic.unsupported_assumptions[:2])
        risks.extend(creative_critic.missing_information[:2])
    if self_evaluation is not None:
        risks.append(f"Self-evaluation hallucination risk is {self_evaluation.hallucination_risk}.")
        risks.append(f"Self-evaluation underdelivery risk is {self_evaluation.underdelivery_risk}.")
        risks.extend(self_evaluation.unsupported_assumptions[:2])
    if reflection_loop is not None and reflection_loop.reflection_required:
        risks.append(f"Reflection Loop reports {reflection_loop.reflection_priority} reflection pressure.")
    if creative_confidence is not None:
        risks.extend(creative_confidence.confidence_uncertainties[:3])
    if creative_score is not None:
        risks.extend(creative_score.negative_contributions[:2])
    if consistency_validation is not None:
        risks.extend(consistency_validation.detected_conflicts[:3])
        risks.extend(consistency_validation.unsupported_conclusions[:2])
    risks.extend(
        f"{item.source} dependency is missing."
        for item in dependencies
        if item.status == "missing"
    )
    return _dedupe(risks, clip_limit=320)[:10]


def _recommendations(
    *,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    creative_score: CreativeScoreProfile | None,
    consistency_validation: ConsistencyValidationProfile | None,
    hitl: ExpectedHumanReviewNeed,
) -> tuple[str, ...]:
    recommendations: list[str] = []
    if consistency_validation is not None:
        recommendations.extend(consistency_validation.prompt_guidance[:2])
    if creative_score is not None:
        recommendations.extend(creative_score.prompt_guidance[:2])
    if creative_confidence is not None:
        recommendations.extend(creative_confidence.prompt_guidance[:2])
    if creative_improvement_planner is not None:
        recommendations.extend(
            creative_improvement_planner.highest_impact_opportunities[:3]
        )
    if reflection_loop is not None:
        recommendations.extend(reflection_loop.prompt_guidance[:2])
    if hitl in {"optional", "recommended", "required"}:
        recommendations.append(
            f"Surface {hitl} human review before treating evaluation conclusions as settled."
        )
    recommendations.append(
        "Keep Evaluation Reports metadata advisory; do not trigger retries, refinement, routing, runtime changes, previews, or future agents."
    )
    return _dedupe(recommendations, clip_limit=360)[:10]


def _evaluation_trace(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    creative_score: CreativeScoreProfile | None,
    consistency_validation: ConsistencyValidationProfile | None,
) -> tuple[EvaluationTraceEntry, ...]:
    entries: list[EvaluationTraceEntry] = []
    for source, role, contribution, evidence in (
        (
            "creative_critic",
            "Creative Critic Engine",
            creative_critic.critique_summary if creative_critic else None,
            creative_critic.evidence[:3] if creative_critic else (),
        ),
        (
            "self_evaluation",
            "Self Evaluation Engine",
            self_evaluation.evaluation_summary if self_evaluation else None,
            self_evaluation.evidence[:3] if self_evaluation else (),
        ),
        (
            "creative_improvement_planner",
            "Creative Improvement Planner",
            (
                creative_improvement_planner.improvement_summary
                if creative_improvement_planner
                else None
            ),
            (
                creative_improvement_planner.evidence[:3]
                if creative_improvement_planner
                else ()
            ),
        ),
        (
            "reflection_loop",
            "Reflection Loop Engine",
            reflection_loop.reflection_summary if reflection_loop else None,
            reflection_loop.evidence[:3] if reflection_loop else (),
        ),
        (
            "creative_confidence",
            "Creative Confidence Engine",
            creative_confidence.confidence_summary if creative_confidence else None,
            (
                creative_confidence.confidence_evidence[:3]
                if creative_confidence
                else ()
            ),
        ),
        (
            "creative_score",
            "Creative Score Engine",
            creative_score.score_summary if creative_score else None,
            creative_score.score_evidence[:3] if creative_score else (),
        ),
        (
            "consistency_validation",
            "Consistency Validation Engine",
            (
                consistency_validation.consistency_summary
                if consistency_validation
                else None
            ),
            consistency_validation.evidence[:3] if consistency_validation else (),
        ),
    ):
        if contribution is None:
            continue
        entries.append(
            EvaluationTraceEntry(
                step=len(entries) + 1,
                source=source,
                role=role,
                contribution=_clip(contribution, 420),
                evidence=evidence,
            )
        )
    return tuple(entries) or (
        EvaluationTraceEntry(
            step=1,
            source="creative_critic",
            role="Evaluation Reports fallback",
            contribution="No V3.4 evaluation metadata was available for tracing.",
            evidence=("Evaluation report generated from missing dependency state.",),
        ),
    )


def _evaluation_provenance(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    creative_score: CreativeScoreProfile | None,
    consistency_validation: ConsistencyValidationProfile | None,
) -> tuple[EvaluationProvenanceEntry, ...]:
    entries: list[EvaluationProvenanceEntry] = []
    if creative_critic is not None:
        entries.append(
            EvaluationProvenanceEntry(
                source="creative_critic",
                role=creative_critic.role,
                summary=creative_critic.critique_summary,
                confidence=creative_critic.critic_confidence,
            )
        )
    if self_evaluation is not None:
        entries.append(
            EvaluationProvenanceEntry(
                source="self_evaluation",
                role=self_evaluation.role,
                summary=self_evaluation.evaluation_summary,
                confidence=self_evaluation.self_evaluation_confidence,
            )
        )
    if creative_improvement_planner is not None:
        entries.append(
            EvaluationProvenanceEntry(
                source="creative_improvement_planner",
                role=creative_improvement_planner.role,
                summary=creative_improvement_planner.improvement_summary,
                confidence=creative_improvement_planner.confidence,
            )
        )
    if reflection_loop is not None:
        entries.append(
            EvaluationProvenanceEntry(
                source="reflection_loop",
                role=reflection_loop.role,
                summary=reflection_loop.reflection_summary,
                confidence=reflection_loop.reflection_confidence,
                hitl_recommendation=reflection_loop.hitl_recommendation,
            )
        )
    if creative_confidence is not None:
        entries.append(
            EvaluationProvenanceEntry(
                source="creative_confidence",
                role=creative_confidence.role,
                summary=creative_confidence.confidence_summary,
                confidence=creative_confidence.confidence_score,
                hitl_recommendation=creative_confidence.hitl_recommendation,
            )
        )
    if creative_score is not None:
        entries.append(
            EvaluationProvenanceEntry(
                source="creative_score",
                role=creative_score.role,
                summary=creative_score.score_summary,
                confidence=creative_score.confidence_weight,
                hitl_recommendation=creative_score.hitl_recommendation,
            )
        )
    if consistency_validation is not None:
        entries.append(
            EvaluationProvenanceEntry(
                source="consistency_validation",
                role=consistency_validation.role,
                summary=consistency_validation.consistency_summary,
                hitl_recommendation=consistency_validation.hitl_recommendation,
            )
        )
    return tuple(entries) or (
        EvaluationProvenanceEntry(
            source="creative_critic",
            role="missing",
            summary="No evaluation provenance was available.",
        ),
    )


def _evaluation_explainability(
    *,
    hitl: ExpectedHumanReviewNeed,
    dependencies: tuple[EvaluationDependency, ...],
    creative_score: CreativeScoreProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    consistency_validation: ConsistencyValidationProfile | None,
) -> tuple[str, ...]:
    explanations = [
        (
            "Report ordering follows V3.4 evaluation pipeline: critic -> "
            "self-evaluation -> planner -> reflection -> confidence -> "
            "score -> consistency validation -> evaluation report."
        ),
        (
            "HITL recommendation is the maximum advisory review need across "
            "confidence, score, consistency, and dependency availability."
        ),
        (
            "Evaluation report is metadata-only and cannot alter generated "
            "outputs or workflow control."
        ),
    ]
    if creative_score is not None:
        explanations.append(f"Score explainability: {creative_score.score_explainability}")
    if creative_confidence is not None:
        explanations.append(
            f"Confidence contribution uses {len(creative_confidence.confidence_components)} component(s)."
        )
    if consistency_validation is not None:
        explanations.append(
            "Consistency contribution uses "
            f"{len(consistency_validation.detected_conflicts)} conflict signal(s) "
            f"and {consistency_validation.evaluation_integrity} integrity."
        )
    if any(item.status == "missing" for item in dependencies):
        explanations.append("Missing dependencies lower report completeness but do not trigger workflow changes.")
    explanations.append(f"Final report HITL posture is {hitl}.")
    return _dedupe(explanations, clip_limit=520)[:10]


def _evaluation_dependencies(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    creative_score: CreativeScoreProfile | None,
    consistency_validation: ConsistencyValidationProfile | None,
) -> tuple[EvaluationDependency, ...]:
    values: tuple[tuple[EvaluationReportSource, object | None], ...] = (
        ("creative_critic", creative_critic),
        ("self_evaluation", self_evaluation),
        ("creative_improvement_planner", creative_improvement_planner),
        ("reflection_loop", reflection_loop),
        ("creative_confidence", creative_confidence),
        ("creative_score", creative_score),
        ("consistency_validation", consistency_validation),
    )
    return tuple(
        EvaluationDependency(
            source=source,
            status="available" if value is not None else "missing",
            note=(
                f"{source} metadata is available for report aggregation."
                if value is not None
                else f"{source} metadata is missing from report aggregation."
            ),
        )
        for source, value in values
    )


def _evidence_chain(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    creative_score: CreativeScoreProfile | None,
    consistency_validation: ConsistencyValidationProfile | None,
    planning_metadata: Sequence[object],
) -> tuple[EvaluationEvidenceLink, ...]:
    links: list[EvaluationEvidenceLink] = [
        EvaluationEvidenceLink(
            source="creative_critic",
            claim=f"Request inspected: {_clip(request.query, 180)}",
            evidence=(
                f"Route {route_decision.route.value}."
                if route_decision is not None
                else "Route metadata unavailable.",
                f"Planning metadata objects: {len(planning_metadata)}.",
            ),
        )
    ]
    if creative_critic is not None:
        links.append(
            EvaluationEvidenceLink(
                source="creative_critic",
                claim=f"Critic risk is {creative_critic.risk_assessment}.",
                evidence=creative_critic.evidence[:3],
            )
        )
    if self_evaluation is not None:
        links.append(
            EvaluationEvidenceLink(
                source="self_evaluation",
                claim=f"Self evaluation is {self_evaluation.completeness_assessment}.",
                evidence=self_evaluation.evidence[:3],
            )
        )
    if creative_improvement_planner is not None:
        links.append(
            EvaluationEvidenceLink(
                source="creative_improvement_planner",
                claim=(
                    f"{len(creative_improvement_planner.improvement_priorities)} "
                    "improvement priority signal(s)."
                ),
                evidence=creative_improvement_planner.evidence[:3],
            )
        )
    if reflection_loop is not None:
        links.append(
            EvaluationEvidenceLink(
                source="reflection_loop",
                claim=f"Reflection priority is {reflection_loop.reflection_priority}.",
                evidence=reflection_loop.evidence[:3],
            )
        )
    if creative_confidence is not None:
        links.append(
            EvaluationEvidenceLink(
                source="creative_confidence",
                claim=f"Confidence level is {creative_confidence.confidence_level}.",
                evidence=creative_confidence.confidence_evidence[:3],
            )
        )
    if creative_score is not None:
        links.append(
            EvaluationEvidenceLink(
                source="creative_score",
                claim=f"Score band is {creative_score.score_band}.",
                evidence=creative_score.score_evidence[:3],
            )
        )
    if consistency_validation is not None:
        links.append(
            EvaluationEvidenceLink(
                source="consistency_validation",
                claim=f"Consistency status is {consistency_validation.consistency_status}.",
                evidence=consistency_validation.evidence[:3],
            )
        )
    return tuple(links[:16])


def _hitl_recommendation(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    creative_improvement_planner: CreativeImprovementPlannerProfile | None,
    reflection_loop: ReflectionLoopProfile | None,
    creative_confidence: CreativeConfidenceProfile | None,
    creative_score: CreativeScoreProfile | None,
    consistency_validation: ConsistencyValidationProfile | None,
    dependencies: tuple[EvaluationDependency, ...],
) -> ExpectedHumanReviewNeed:
    levels = {"not_needed": 0, "optional": 1, "recommended": 2, "required": 3}
    values: list[ExpectedHumanReviewNeed] = []
    if creative_critic is not None and creative_critic.hitl_questions:
        values.append("recommended")
    if self_evaluation is not None and self_evaluation.hitl_questions:
        values.append("recommended")
    if creative_improvement_planner is not None and creative_improvement_planner.hitl_questions:
        values.append("optional")
    if reflection_loop is not None:
        values.append(reflection_loop.hitl_recommendation)
    if creative_confidence is not None:
        values.append(creative_confidence.hitl_recommendation)
    if creative_score is not None:
        values.append(creative_score.hitl_recommendation)
    if consistency_validation is not None:
        values.append(consistency_validation.hitl_recommendation)
    if any(item.status == "missing" for item in dependencies):
        values.append("optional")
    if not values:
        return "optional"
    return max(values, key=lambda value: levels[value])


def _prompt_guidance(*, hitl: ExpectedHumanReviewNeed) -> tuple[str, ...]:
    guidance = [
        "Use Evaluation Reports metadata as advisory evaluation context only.",
        (
            "Do not modify outputs, execute artifacts, retry, refine, route, "
            "select runtime, alter previews, or invoke future agents from "
            "evaluation report metadata."
        ),
        (
            "Preserve evaluation trace, provenance, explainability, "
            "dependencies, and evidence as inspectable metadata."
        ),
    ]
    if hitl in {"optional", "recommended", "required"}:
        guidance.append(
            f"Surface {hitl} human review when report metadata affects trust."
        )
    return tuple(guidance)
