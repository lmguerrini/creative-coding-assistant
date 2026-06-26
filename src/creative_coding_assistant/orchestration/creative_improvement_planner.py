"""Metadata-only Creative Improvement Planner for V3.4 evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration._metadata_utils import _clip, _dedupe
from creative_coding_assistant.orchestration.artifacts import WorkflowArtifact
from creative_coding_assistant.orchestration.creative_critic_engine import (
    CreativeCriticProfile,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.self_evaluation_engine import (
    SelfEvaluationProfile,
)

ImprovementPriorityLevel = Literal["critical", "high", "medium", "low"]
ImprovementImpact = Literal["high", "medium", "low"]
ImprovementRisk = Literal["low", "medium", "high"]
ImprovementSource = Literal[
    "creative_critic",
    "self_evaluation",
    "artifact_context",
    "workflow_context",
]

CREATIVE_IMPROVEMENT_PLANNER_AUTHORITY_BOUNDARY = (
    "The Creative Improvement Planner converts Creative Critic and Self "
    "Evaluation metadata into advisory improvement guidance only; it does not "
    "modify prompts, edit artifacts, regenerate outputs, call providers, "
    "retry models, select runtimes, route providers or models, alter previews, "
    "change workflow control, trigger loops, execute artifacts, invoke V4 "
    "agents, or perform future refinement behavior."
)


class CreativeImprovementPriority(BaseModel):
    """One bounded improvement priority with source evidence."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    priority_id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=220)
    priority: ImprovementPriorityLevel
    impact: ImprovementImpact
    risk: ImprovementRisk
    source: ImprovementSource
    rationale: str = Field(min_length=1, max_length=360)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=4)


class CreativeImprovementPlannerProfile(BaseModel):
    """Inspectable metadata-only improvement plan for evaluation signals."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_improvement_planner"] = (
        "creative_improvement_planner"
    )
    serialization_version: Literal["v1"] = "v1"
    confidence: float = Field(ge=0, le=1)
    improvement_summary: str = Field(min_length=1, max_length=620)
    improvement_priorities: tuple[CreativeImprovementPriority, ...] = Field(
        min_length=1,
        max_length=8,
    )
    highest_impact_opportunities: tuple[str, ...] = Field(min_length=1, max_length=8)
    low_risk_improvements: tuple[str, ...] = Field(min_length=1, max_length=8)
    experimental_improvements: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    trade_off_recommendations: tuple[str, ...] = Field(min_length=1, max_length=8)
    improvement_rationale: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=16)
    future_refinement_candidates: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=CREATIVE_IMPROVEMENT_PLANNER_AUTHORITY_BOUNDARY,
        max_length=980,
    )


def derive_creative_improvement_planner_profile(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    generated_response: str | None = None,
    artifacts: Sequence[WorkflowArtifact] = (),
) -> CreativeImprovementPlannerProfile:
    """Convert critique and self-evaluation signals into advisory priorities."""

    priorities = _improvement_priorities(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        generated_response=generated_response,
        artifacts=artifacts,
    )
    high_impact = _highest_impact_opportunities(priorities)
    low_risk = _low_risk_improvements(
        priorities=priorities,
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
    )
    experimental = _experimental_improvements(
        request=request,
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
    )
    tradeoffs = _trade_off_recommendations(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        priorities=priorities,
    )
    rationale = _improvement_rationale(
        creative_critic=creative_critic,
        self_evaluation=self_evaluation,
        priorities=priorities,
    )

    return CreativeImprovementPlannerProfile(
        confidence=_confidence(
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            generated_response=generated_response,
            artifacts=artifacts,
        ),
        improvement_summary=_summary(
            priorities=priorities,
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
        ),
        improvement_priorities=priorities,
        highest_impact_opportunities=high_impact,
        low_risk_improvements=low_risk,
        experimental_improvements=experimental,
        trade_off_recommendations=tradeoffs,
        improvement_rationale=rationale,
        evidence=_evidence(
            request=request,
            route_decision=route_decision,
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            generated_response=generated_response,
            artifacts=artifacts,
        ),
        future_refinement_candidates=_future_refinement_candidates(
            priorities=priorities,
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
        ),
        hitl_questions=_hitl_questions(
            creative_critic=creative_critic,
            self_evaluation=self_evaluation,
            priorities=priorities,
        ),
        prompt_guidance=_prompt_guidance(priorities=priorities),
    )


def creative_improvement_planner_prompt_lines(
    profile: CreativeImprovementPlannerProfile,
) -> tuple[str, ...]:
    """Render Creative Improvement Planner metadata as compact prompt guidance."""

    lines = [
        f"Authority boundary: {profile.authority_boundary}",
        f"Serialization version: {profile.serialization_version}.",
        f"Improvement planner confidence: {profile.confidence:.2f}.",
        f"Improvement summary: {profile.improvement_summary}",
    ]
    lines.extend(
        (
            "Improvement priority: "
            f"{item.priority_id}; {item.priority}; {item.impact} impact; "
            f"{item.risk} risk; {item.title}"
        )
        for item in profile.improvement_priorities
    )
    lines.extend(
        f"Highest-impact opportunity: {item}"
        for item in profile.highest_impact_opportunities
    )
    lines.extend(f"Low-risk improvement: {item}" for item in profile.low_risk_improvements)
    lines.extend(
        f"Experimental improvement: {item}"
        for item in profile.experimental_improvements
    )
    lines.extend(
        f"Trade-off recommendation: {item}"
        for item in profile.trade_off_recommendations
    )
    lines.extend(f"Improvement rationale: {item}" for item in profile.improvement_rationale)
    lines.extend(
        f"Future refinement candidate: {item}"
        for item in profile.future_refinement_candidates
    )
    lines.extend(f"Improvement planner HITL question: {item}" for item in profile.hitl_questions)
    lines.extend(f"Improvement planner guidance: {item}" for item in profile.prompt_guidance)
    return tuple(lines[:64])


def _improvement_priorities(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    generated_response: str | None,
    artifacts: Sequence[WorkflowArtifact],
) -> tuple[CreativeImprovementPriority, ...]:
    candidates: list[CreativeImprovementPriority] = []
    if self_evaluation is not None:
        for index, item in enumerate(self_evaluation.quality_gaps[:3], start=1):
            candidates.append(
                _priority(
                    priority_id=f"self_eval_gap_{index}",
                    title=item,
                    source="self_evaluation",
                    priority=_priority_level(self_evaluation, creative_critic),
                    impact="high" if index == 1 else "medium",
                    risk="low",
                    rationale=(
                        "Self Evaluation identified this as a quality gap; "
                        "treat it as guidance without changing workflow control."
                    ),
                    evidence=(self_evaluation.evaluation_summary,),
                )
            )
        for index, item in enumerate(
            self_evaluation.improvement_opportunities[:2],
            start=1,
        ):
            candidates.append(
                _priority(
                    priority_id=f"self_eval_opportunity_{index}",
                    title=item,
                    source="self_evaluation",
                    priority="high" if index == 1 else "medium",
                    impact="high" if index == 1 else "medium",
                    risk="low" if self_evaluation.overreach_risk == "low" else "medium",
                    rationale="Self Evaluation surfaced this improvement direction.",
                    evidence=(self_evaluation.evaluation_summary,),
                )
            )
    if creative_critic is not None:
        for index, item in enumerate(creative_critic.improvement_opportunities[:2], start=1):
            candidates.append(
                _priority(
                    priority_id=f"creative_critic_opportunity_{index}",
                    title=item,
                    source="creative_critic",
                    priority=(
                        "critical"
                        if creative_critic.risk_assessment in {"high", "blocked"}
                        and index == 1
                        else "high"
                    ),
                    impact="high" if index == 1 else "medium",
                    risk="medium" if creative_critic.risk_assessment in {"high", "blocked"} else "low",
                    rationale="Creative Critic recommends this as advisory improvement guidance.",
                    evidence=(creative_critic.critique_summary,),
                )
            )
        for index, item in enumerate(creative_critic.creative_weaknesses[:2], start=1):
            candidates.append(
                _priority(
                    priority_id=f"creative_critic_weakness_{index}",
                    title=item,
                    source="creative_critic",
                    priority="high" if index == 1 else "medium",
                    impact="medium",
                    risk="low",
                    rationale="Creative Critic weakness should be addressed before scope expansion.",
                    evidence=(creative_critic.critique_summary,),
                )
            )
    if generated_response is not None and not artifacts:
        candidates.append(
            _priority(
                priority_id="artifact_context_missing",
                title="Clarify generated response caveats when no artifact metadata is available.",
                source="artifact_context",
                priority="medium",
                impact="medium",
                risk="low",
                rationale="Generated text is available but artifact evidence is absent.",
                evidence=(f"Generated response length: {len(generated_response)} characters.",),
            )
        )
    return tuple(candidates[:8]) or (
        _priority(
            priority_id="preserve_current_alignment",
            title="Preserve current alignment and avoid expanding scope.",
            source="workflow_context",
            priority="low",
            impact="low",
            risk="low",
            rationale="No critic or self-evaluation improvement signals are available.",
            evidence=("Improvement planner has only request-level context.",),
        ),
    )


def _priority(
    *,
    priority_id: str,
    title: str,
    source: ImprovementSource,
    priority: ImprovementPriorityLevel,
    impact: ImprovementImpact,
    risk: ImprovementRisk,
    rationale: str,
    evidence: tuple[str, ...],
) -> CreativeImprovementPriority:
    return CreativeImprovementPriority(
        priority_id=priority_id,
        title=_clip(title, 220),
        priority=priority,
        impact=impact,
        risk=risk,
        source=source,
        rationale=_clip(rationale, 360),
        evidence=_dedupe(evidence)[:4],
    )


def _priority_level(
    self_evaluation: SelfEvaluationProfile,
    creative_critic: CreativeCriticProfile | None,
) -> ImprovementPriorityLevel:
    if (
        self_evaluation.completeness_assessment == "blocked"
        or self_evaluation.hallucination_risk == "high"
        or self_evaluation.underdelivery_risk == "high"
        or (
            creative_critic is not None
            and creative_critic.risk_assessment in {"high", "blocked"}
        )
    ):
        return "critical"
    if (
        self_evaluation.ambiguity_assessment == "high"
        or self_evaluation.overreach_risk == "high"
    ):
        return "high"
    return "medium"


def _highest_impact_opportunities(
    priorities: tuple[CreativeImprovementPriority, ...],
) -> tuple[str, ...]:
    ranked = sorted(
        priorities,
        key=lambda item: (
            {"critical": 0, "high": 1, "medium": 2, "low": 3}[item.priority],
            {"high": 0, "medium": 1, "low": 2}[item.impact],
        ),
    )
    return _dedupe(item.title for item in ranked[:4]) or (
        "Preserve current direction before considering new improvements.",
    )


def _low_risk_improvements(
    *,
    priorities: tuple[CreativeImprovementPriority, ...],
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
) -> tuple[str, ...]:
    improvements = [item.title for item in priorities if item.risk == "low"]
    if self_evaluation is not None:
        improvements.extend(self_evaluation.prompt_guidance[:2])
    if creative_critic is not None:
        improvements.extend(creative_critic.prompt_guidance[:1])
    return _dedupe(improvements)[:8] or (
        "Keep improvement guidance as caveated metadata without changing execution.",
    )


def _experimental_improvements(
    *,
    request: AssistantRequest,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
) -> tuple[str, ...]:
    experimental: list[str] = []
    if creative_critic is not None and creative_critic.originality_quality >= 0.72:
        experimental.append(
            "Consider one optional creative variation only after core alignment is preserved."
        )
    if self_evaluation is not None and self_evaluation.overreach_risk == "low":
        experimental.append(
            "Explore a bounded enhancement path as future metadata, not automatic refinement."
        )
    if "experimental" in request.query.lower():
        experimental.append(
            "Keep experimental improvements explicitly optional and HITL-visible."
        )
    return _dedupe(experimental)[:6]


def _trade_off_recommendations(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    priorities: tuple[CreativeImprovementPriority, ...],
) -> tuple[str, ...]:
    recommendations: list[str] = []
    if self_evaluation is not None:
        recommendations.append(
            "Prioritize request and intent alignment before adding new creative scope."
        )
        if self_evaluation.hallucination_risk != "low":
            recommendations.append(
                "Trade certainty language for explicit caveats around unsupported assumptions."
            )
        if self_evaluation.underdelivery_risk != "low":
            recommendations.append(
                "Favor deliverable clarity over additional experimental detail."
            )
    if creative_critic is not None and creative_critic.risk_assessment in {"high", "blocked"}:
        recommendations.append(
            "Treat high critic risk as a HITL-visible caveat, not an automatic retry trigger."
        )
    if any(item.risk == "medium" for item in priorities):
        recommendations.append(
            "Sequence medium-risk improvements after low-risk alignment fixes."
        )
    return _dedupe(recommendations)[:8] or (
        "Prefer low-risk clarification over broad scope expansion.",
    )


def _improvement_rationale(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    priorities: tuple[CreativeImprovementPriority, ...],
) -> tuple[str, ...]:
    rationale = [
        f"{len(priorities)} improvement priority signal(s) were derived without changing execution behavior."
    ]
    if creative_critic is not None:
        rationale.append(
            f"Creative Critic contributes {creative_critic.risk_assessment} risk and {creative_critic.critic_confidence:.2f} confidence."
        )
    if self_evaluation is not None:
        rationale.append(
            f"Self Evaluation contributes {self_evaluation.completeness_assessment} completeness and {self_evaluation.self_evaluation_confidence:.2f} confidence."
        )
    rationale.append("Improvement guidance remains advisory metadata only.")
    return _dedupe(rationale)[:8]


def _confidence(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    generated_response: str | None,
    artifacts: Sequence[WorkflowArtifact],
) -> float:
    values: list[float] = []
    if creative_critic is not None:
        values.append(creative_critic.critic_confidence)
    if self_evaluation is not None:
        values.append(self_evaluation.self_evaluation_confidence)
    if generated_response:
        values.append(0.66)
    if artifacts:
        values.append(min(0.78, 0.58 + len(artifacts) * 0.05))
    if not values:
        return 0.28
    return round(max(0.05, min(0.98, sum(values) / len(values))), 2)


def _summary(
    *,
    priorities: tuple[CreativeImprovementPriority, ...],
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
) -> str:
    top = priorities[0]
    critic_risk = (
        creative_critic.risk_assessment if creative_critic is not None else "unavailable"
    )
    completeness = (
        self_evaluation.completeness_assessment
        if self_evaluation is not None
        else "unavailable"
    )
    return _clip(
        (
            f"Creative Improvement Planner identified {len(priorities)} advisory "
            f"priority signal(s). Top priority is {top.priority} / {top.impact} "
            f"impact: {top.title}. Critic risk is {critic_risk}; self-evaluation "
            f"completeness is {completeness}. No workflow control or artifact "
            "behavior is changed."
        ),
        620,
    )


def _evidence(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    generated_response: str | None,
    artifacts: Sequence[WorkflowArtifact],
) -> tuple[str, ...]:
    evidence = [f"Request: {_clip(request.query, 220)}"]
    if route_decision is not None:
        domains = ", ".join(domain.value for domain in route_decision.domains) or "none"
        evidence.append(f"Route: {route_decision.route.value}; domains {domains}.")
    if creative_critic is not None:
        evidence.append(
            f"Creative critic: {creative_critic.risk_assessment} risk; {creative_critic.critic_confidence:.2f} confidence."
        )
    if self_evaluation is not None:
        evidence.append(
            f"Self evaluation: {self_evaluation.completeness_assessment}; {self_evaluation.self_evaluation_confidence:.2f} confidence."
        )
    if generated_response is not None:
        evidence.append(f"Generated response available: {len(generated_response)} characters.")
    if artifacts:
        evidence.append(
            "Artifacts available: "
            + ", ".join(f"{artifact.id}:{artifact.language}" for artifact in artifacts[:5])
            + "."
        )
    evidence.append("Authority boundary verified: metadata-only improvement planning.")
    return _dedupe(evidence)[:16]


def _future_refinement_candidates(
    *,
    priorities: tuple[CreativeImprovementPriority, ...],
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
) -> tuple[str, ...]:
    candidates = [
        f"Future candidate from {item.priority_id}: {item.title}"
        for item in priorities[:3]
        if item.priority in {"critical", "high"}
    ]
    if creative_critic is not None:
        candidates.extend(creative_critic.improvement_opportunities[:1])
    if self_evaluation is not None:
        candidates.extend(self_evaluation.improvement_opportunities[:1])
    return _dedupe(candidates)[:8]


def _hitl_questions(
    *,
    creative_critic: CreativeCriticProfile | None,
    self_evaluation: SelfEvaluationProfile | None,
    priorities: tuple[CreativeImprovementPriority, ...],
) -> tuple[str, ...]:
    questions: list[str] = []
    if any(item.priority == "critical" for item in priorities):
        questions.append("Should critical improvement priorities be resolved before scope expansion?")
    if creative_critic is not None:
        questions.extend(creative_critic.hitl_questions[:2])
    if self_evaluation is not None:
        questions.extend(self_evaluation.hitl_questions[:2])
    return _dedupe(questions)[:8]


def _prompt_guidance(
    *,
    priorities: tuple[CreativeImprovementPriority, ...],
) -> tuple[str, ...]:
    guidance = [
        "Use Creative Improvement Planner metadata as advisory guidance only, not as automatic refinement or output modification.",
        "Apply low-risk alignment and caveat improvements before experimental improvements.",
        "Do not trigger retries, model calls, provider routing, runtime selection, preview changes, workflow loops, artifact edits, or V4 agents.",
    ]
    if any(item.priority == "critical" for item in priorities):
        guidance.append("Surface critical improvement priorities and HITL questions before expanding scope.")
    guidance.extend(item.title for item in priorities[:3])
    return _dedupe(guidance)[:8]
