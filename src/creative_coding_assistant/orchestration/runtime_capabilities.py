"""Bounded Runtime Capability Reasoner for Creative Intelligence workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest, CreativeCodingDomain
from creative_coding_assistant.orchestration.creative_constraints import (
    ConstraintPressure,
    CreativeConstraintSolution,
)
from creative_coding_assistant.orchestration.creative_planning import (
    CreativeExecutionPlan,
)
from creative_coding_assistant.orchestration.creative_strategy import (
    CreativeStrategyId,
    CreativeStrategyProfile,
)
from creative_coding_assistant.orchestration.creative_technique import (
    CreativeTechniqueId,
    CreativeTechniqueProfile,
    TechniquePressure,
)
from creative_coding_assistant.orchestration.creative_translation import (
    CreativeOutputModality,
    CreativeTranslation,
)
from creative_coding_assistant.orchestration.routing import RouteDecision

RuntimeCapabilityId = Literal[
    "p5_js",
    "three_js",
    "react_three_fiber",
    "glsl",
    "hydra",
    "tone_js",
    "gsap",
    "svg",
    "canvas",
]
RuntimeCapabilityFit = Literal["strong", "moderate", "weak"]
RuntimeCapabilityComplexity = Literal["low", "medium", "high"]
RuntimePreviewSupport = Literal[
    "backend_preview_supported",
    "workstation_preview_bounded",
    "code_only",
]

RUNTIME_REASONER_AUTHORITY_BOUNDARY = (
    "The Runtime Capability Reasoner evaluates runtime fit for inspection only; "
    "it does not auto-select runtimes, route providers or models, choose "
    "renderers, create execution profiles, change preview behavior, or run "
    "runtime repair."
)


class RuntimeCapabilityCandidate(BaseModel):
    """One supported runtime evaluated against the current creative context."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    runtime: RuntimeCapabilityId
    label: str = Field(min_length=1, max_length=80)
    suitability: RuntimeCapabilityFit
    confidence: float = Field(ge=0, le=1)
    strategy_alignment: RuntimeCapabilityFit
    technique_compatibility: RuntimeCapabilityFit
    output_goal_fit: RuntimeCapabilityFit
    implementation_complexity: RuntimeCapabilityComplexity
    performance_pressure: ConstraintPressure
    preview_support: RuntimePreviewSupport
    strengths: tuple[str, ...] = Field(min_length=1, max_length=5)
    limitations: tuple[str, ...] = Field(min_length=1, max_length=5)
    risks: tuple[str, ...] = Field(min_length=1, max_length=5)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=5)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=8)


class RuntimeCapabilityProfile(BaseModel):
    """Inspectable runtime capability metadata derived before generation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["runtime_capability_reasoner"] = "runtime_capability_reasoner"
    output_goal: str = Field(min_length=1, max_length=360)
    likely_candidates: tuple[RuntimeCapabilityId, ...] = Field(
        min_length=1,
        max_length=3,
    )
    candidate_runtimes: tuple[RuntimeCapabilityCandidate, ...] = Field(
        min_length=1,
        max_length=9,
    )
    strategy_context: str | None = Field(default=None, max_length=180)
    technique_context: str | None = Field(default=None, max_length=180)
    constraint_context: str | None = Field(default=None, max_length=220)
    hitl_advisable: bool = False
    hitl_reason: str | None = Field(default=None, max_length=280)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=RUNTIME_REASONER_AUTHORITY_BOUNDARY,
        max_length=420,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=10)


def derive_runtime_capability_profile(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
) -> RuntimeCapabilityProfile:
    """Evaluate runtime capabilities without selecting or changing a runtime."""

    domains = _effective_domains(request, route_decision)
    output_goal = _output_goal(request, creative_translation, creative_plan)
    modality = _output_modality(creative_translation, creative_plan)
    normalized = _runtime_text(
        request=request,
        creative_translation=creative_translation,
        creative_strategy=creative_strategy,
        creative_techniques=creative_techniques,
        creative_plan=creative_plan,
    )
    scored = sorted(
        (
            _score_runtime(
                signal,
                normalized=normalized,
                domains=domains,
                modality=modality,
                request=request,
                creative_translation=creative_translation,
                creative_strategy=creative_strategy,
                creative_techniques=creative_techniques,
                creative_plan=creative_plan,
                creative_constraints=creative_constraints,
            )
            for signal in _RUNTIME_SIGNALS
        ),
        key=lambda item: (item.score, item.signal.runtime),
        reverse=True,
    )
    candidates = tuple(
        _candidate_from_scored(
            scored_runtime,
            creative_techniques=creative_techniques,
            creative_plan=creative_plan,
            creative_constraints=creative_constraints,
        )
        for scored_runtime in scored
    )
    likely_candidates = tuple(item.runtime for item in candidates[:3])
    hitl_reason = _hitl_reason(
        scored=scored,
        candidates=candidates,
        domains=domains,
        creative_plan=creative_plan,
        creative_constraints=creative_constraints,
    )
    return RuntimeCapabilityProfile(
        output_goal=output_goal,
        likely_candidates=likely_candidates,
        candidate_runtimes=candidates,
        strategy_context=_strategy_context(creative_strategy),
        technique_context=_technique_context(creative_techniques),
        constraint_context=_constraint_context(creative_constraints),
        hitl_advisable=hitl_reason is not None,
        hitl_reason=hitl_reason,
        prompt_guidance=_profile_prompt_guidance(
            likely_candidates=likely_candidates,
            candidates=candidates,
            hitl_reason=hitl_reason,
        ),
        evidence=_profile_evidence(
            request=request,
            route_decision=route_decision,
            domains=domains,
            creative_strategy=creative_strategy,
            creative_techniques=creative_techniques,
            creative_plan=creative_plan,
            creative_constraints=creative_constraints,
            scored=scored,
        ),
    )


def runtime_capability_prompt_lines(
    profile: RuntimeCapabilityProfile,
) -> tuple[str, ...]:
    """Render runtime capability metadata as compact prompt guidance."""

    labels = _candidate_labels(profile)
    lines = [
        f"Authority boundary: {profile.authority_boundary}",
        f"Output goal: {profile.output_goal}",
        "Likely runtime candidates (non-binding): " + ", ".join(labels) + ".",
    ]
    if profile.strategy_context is not None:
        lines.append(f"Strategy context: {profile.strategy_context}")
    if profile.technique_context is not None:
        lines.append(f"Technique context: {profile.technique_context}")
    if profile.constraint_context is not None:
        lines.append(f"Constraint context: {profile.constraint_context}")
    if profile.hitl_advisable and profile.hitl_reason is not None:
        lines.append(f"HITL advisory: {profile.hitl_reason}")
    lines.extend(f"Runtime guidance: {item}" for item in profile.prompt_guidance)
    for candidate in profile.candidate_runtimes[:3]:
        lines.append(
            "Candidate runtime: "
            f"{candidate.label} ({candidate.runtime}); "
            f"suitability {candidate.suitability}; "
            f"strategy {candidate.strategy_alignment}; "
            f"technique {candidate.technique_compatibility}; "
            f"output {candidate.output_goal_fit}; "
            f"complexity {candidate.implementation_complexity}; "
            f"performance {candidate.performance_pressure}; "
            f"preview {candidate.preview_support}."
        )
        lines.append(f"Candidate strength: {candidate.strengths[0]}")
        lines.append(f"Candidate limitation: {candidate.limitations[0]}")
        lines.append(f"Candidate risk: {candidate.risks[0]}")
    return tuple(lines[:24])


@dataclass(frozen=True)
class _RuntimeSignal:
    runtime: RuntimeCapabilityId
    label: str
    domains: tuple[CreativeCodingDomain, ...]
    tokens: tuple[str, ...]
    strategy_affinities: tuple[CreativeStrategyId, ...]
    technique_affinities: tuple[CreativeTechniqueId, ...]
    modality_affinities: tuple[CreativeOutputModality, ...]
    implementation_complexity: RuntimeCapabilityComplexity
    performance_pressure: ConstraintPressure
    preview_support: RuntimePreviewSupport
    strengths: tuple[str, ...]
    limitations: tuple[str, ...]
    risks: tuple[str, ...]
    prompt_guidance: tuple[str, ...]


@dataclass(frozen=True)
class _ScoredRuntime:
    signal: _RuntimeSignal
    score: int
    matched_signals: tuple[str, ...]
    strategy_alignment: RuntimeCapabilityFit
    technique_compatibility: RuntimeCapabilityFit
    output_goal_fit: RuntimeCapabilityFit


def _score_runtime(
    signal: _RuntimeSignal,
    *,
    normalized: str,
    domains: tuple[CreativeCodingDomain, ...],
    modality: CreativeOutputModality | None,
    request: AssistantRequest,
    creative_translation: CreativeTranslation | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
) -> _ScoredRuntime:
    matched: list[str] = []
    score = 0
    for domain in domains:
        if domain in signal.domains:
            score += 5
            matched.append(f"domain:{domain.value}")
        elif domain in _RELATED_DOMAINS.get(signal.runtime, ()):
            score += 2
            matched.append(f"related-domain:{domain.value}")
    for token in signal.tokens:
        if _keyword_matches(normalized, token):
            score += 2
            matched.append(token)
    if creative_translation is not None:
        score += _translation_score(signal, creative_translation, matched)
    if _plan_runtime_matches(signal, creative_plan):
        score += 4
        matched.append("planning runtime signal")
    strategy_alignment = _strategy_alignment(signal, creative_strategy)
    if strategy_alignment == "strong":
        score += 3
        if creative_strategy is not None:
            matched.append(f"strategy:{creative_strategy.primary_strategy}")
    elif strategy_alignment == "moderate":
        score += 1
    technique_compatibility = _technique_compatibility(signal, creative_techniques)
    if technique_compatibility == "strong":
        score += 4
        if creative_techniques is not None:
            matched.append(f"technique:{creative_techniques.primary_technique}")
    elif technique_compatibility == "moderate":
        score += 1
    output_goal_fit = _output_goal_fit(signal, modality)
    if output_goal_fit == "strong":
        score += 2
    elif output_goal_fit == "moderate":
        score += 1
    if _constraints_pressure_bonus(signal, creative_constraints):
        score += 1
        matched.append("constraint pressure fit")
    if score == 0 and signal.runtime == "p5_js":
        score = 1
        matched.append("default bounded visual runtime")
    return _ScoredRuntime(
        signal=signal,
        score=score,
        matched_signals=tuple(_dedupe_text(matched))[:8],
        strategy_alignment=strategy_alignment,
        technique_compatibility=technique_compatibility,
        output_goal_fit=output_goal_fit,
    )


def _candidate_from_scored(
    scored: _ScoredRuntime,
    *,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
) -> RuntimeCapabilityCandidate:
    signal = scored.signal
    complexity = _implementation_complexity(
        signal.implementation_complexity,
        creative_plan,
        creative_constraints,
    )
    performance_pressure = _performance_pressure(
        signal.performance_pressure,
        creative_techniques,
        creative_constraints,
    )
    return RuntimeCapabilityCandidate(
        runtime=signal.runtime,
        label=signal.label,
        suitability=_suitability(scored),
        confidence=_confidence(scored.score),
        strategy_alignment=scored.strategy_alignment,
        technique_compatibility=scored.technique_compatibility,
        output_goal_fit=scored.output_goal_fit,
        implementation_complexity=complexity,
        performance_pressure=performance_pressure,
        preview_support=signal.preview_support,
        strengths=_strengths(signal, scored),
        limitations=_limitations(signal, creative_plan),
        risks=_risks(signal, performance_pressure, creative_constraints),
        prompt_guidance=signal.prompt_guidance,
        evidence=_candidate_evidence(scored),
    )


def _translation_score(
    signal: _RuntimeSignal,
    translation: CreativeTranslation,
    matched: list[str],
) -> int:
    score = 0
    for label in translation.runtime_recommendations:
        if _runtime_label_matches(signal, label):
            score += 3
            matched.append(f"translation:{label}")
    for value in (
        *translation.symbolic_references,
        *translation.geometric_references,
        *translation.musical_references,
        *translation.mood_atmosphere,
        *translation.movement_language,
        *translation.structure_direction,
    ):
        normalized = value.lower()
        if any(token in normalized for token in signal.tokens):
            score += 1
            matched.append(value)
    if (
        translation.audio_reactive is not None
        and signal.runtime in {"tone_js", "hydra", "p5_js"}
    ):
        score += 2
        matched.append("audio-reactive output")
    if (
        translation.sacred_geometry is not None
        and signal.runtime in {"p5_js", "svg", "canvas", "glsl"}
    ):
        score += 2
        matched.append("sacred geometry output")
    return score


def _strategy_alignment(
    signal: _RuntimeSignal,
    creative_strategy: CreativeStrategyProfile | None,
) -> RuntimeCapabilityFit:
    if creative_strategy is None:
        return "moderate"
    if creative_strategy.primary_strategy in signal.strategy_affinities:
        return "strong"
    if creative_strategy.alternative_strategies and any(
        item.strategy in signal.strategy_affinities
        for item in creative_strategy.alternative_strategies
    ):
        return "moderate"
    return "weak"


def _technique_compatibility(
    signal: _RuntimeSignal,
    creative_techniques: CreativeTechniqueProfile | None,
) -> RuntimeCapabilityFit:
    if creative_techniques is None:
        return "moderate"
    if creative_techniques.primary_technique in signal.technique_affinities:
        return "strong"
    if creative_techniques.alternative_techniques and any(
        item.technique in signal.technique_affinities
        for item in creative_techniques.alternative_techniques
    ):
        return "moderate"
    return "weak"


def _output_goal_fit(
    signal: _RuntimeSignal,
    modality: CreativeOutputModality | None,
) -> RuntimeCapabilityFit:
    if modality is None:
        return "moderate"
    if modality in signal.modality_affinities:
        return "strong"
    if (
        modality is CreativeOutputModality.AUDIOVISUAL
        and (
            CreativeOutputModality.VISUAL in signal.modality_affinities
            or CreativeOutputModality.AUDIO in signal.modality_affinities
        )
    ):
        return "moderate"
    return "weak"


def _plan_runtime_matches(
    signal: _RuntimeSignal,
    creative_plan: CreativeExecutionPlan | None,
) -> bool:
    if creative_plan is None or creative_plan.recommended_runtime is None:
        return False
    runtime = creative_plan.recommended_runtime
    if runtime == "p5":
        return signal.runtime == "p5_js"
    if runtime == "glsl":
        return signal.runtime == "glsl"
    if runtime == "three":
        return signal.runtime in {"three_js", "react_three_fiber"}
    return _runtime_label_matches(signal, runtime)


def _runtime_label_matches(signal: _RuntimeSignal, value: str) -> bool:
    normalized = value.strip().lower().replace("-", "_").replace(".", "_")
    return (
        signal.runtime in normalized
        or signal.label.lower() in value.lower()
        or any(token.replace(" ", "_") in normalized for token in signal.tokens)
    )


def _constraints_pressure_bonus(
    signal: _RuntimeSignal,
    creative_constraints: CreativeConstraintSolution | None,
) -> bool:
    if creative_constraints is None:
        return False
    if (
        creative_constraints.performance_pressure == "high"
        and signal.performance_pressure == "low"
    ):
        return True
    return (
        creative_constraints.runtime_fit == "code_only"
        and signal.preview_support != "backend_preview_supported"
    )


def _implementation_complexity(
    base: RuntimeCapabilityComplexity,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
) -> RuntimeCapabilityComplexity:
    if creative_plan is not None and creative_plan.expected_complexity == "high":
        return "high"
    if (
        creative_constraints is not None
        and creative_constraints.complexity_pressure == "high"
    ):
        return "high"
    return base


def _performance_pressure(
    base: ConstraintPressure,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_constraints: CreativeConstraintSolution | None,
) -> ConstraintPressure:
    pressure = base
    if creative_techniques is not None:
        pressure = _max_pressure(pressure, creative_techniques.performance_pressure)
    if creative_constraints is not None:
        pressure = _max_pressure(pressure, creative_constraints.performance_pressure)
    return pressure


def _max_pressure(
    current: ConstraintPressure,
    candidate: ConstraintPressure | TechniquePressure,
) -> ConstraintPressure:
    order = {"low": 0, "medium": 1, "high": 2}
    return candidate if order[candidate] > order[current] else current


def _suitability(scored: _ScoredRuntime) -> RuntimeCapabilityFit:
    if scored.output_goal_fit == "weak" and scored.score < 8:
        return "weak"
    if scored.score >= 12:
        return "strong"
    if scored.score >= 6:
        return "moderate"
    return "weak"


def _confidence(score: int) -> float:
    return min(0.95, max(0.3, round(0.35 + score * 0.05, 2)))


def _strengths(
    signal: _RuntimeSignal,
    scored: _ScoredRuntime,
) -> tuple[str, ...]:
    strengths = list(signal.strengths)
    if scored.strategy_alignment == "strong":
        strengths.insert(0, "Aligns strongly with the selected creative strategy.")
    if scored.technique_compatibility == "strong":
        strengths.insert(0, "Fits the selected creative technique.")
    return _dedupe_text(strengths)[:5]


def _limitations(
    signal: _RuntimeSignal,
    creative_plan: CreativeExecutionPlan | None,
) -> tuple[str, ...]:
    limitations = list(signal.limitations)
    if (
        creative_plan is not None
        and creative_plan.recommended_runtime is not None
        and not _plan_runtime_matches(signal, creative_plan)
    ):
        limitations.append(
            "Differs from the current planning runtime signal; treat as "
            "comparison only."
        )
    return _dedupe_text(limitations)[:5]


def _risks(
    signal: _RuntimeSignal,
    performance_pressure: ConstraintPressure,
    creative_constraints: CreativeConstraintSolution | None,
) -> tuple[str, ...]:
    risks = list(signal.risks)
    if performance_pressure == "high":
        risks.append("High performance pressure requires bounded effect scope.")
    if (
        creative_constraints is not None
        and creative_constraints.hitl_advisable
        and creative_constraints.hitl_reason is not None
    ):
        risks.append(creative_constraints.hitl_reason)
    risks.append("Do not treat this capability assessment as runtime selection.")
    return _dedupe_text(risks)[:5]


def _candidate_evidence(scored: _ScoredRuntime) -> tuple[str, ...]:
    evidence = [f"Capability score: {scored.score}."]
    if scored.matched_signals:
        evidence.append("Matched signals: " + ", ".join(scored.matched_signals) + ".")
    return tuple(evidence[:8])


def _hitl_reason(
    *,
    scored: list[_ScoredRuntime],
    candidates: tuple[RuntimeCapabilityCandidate, ...],
    domains: tuple[CreativeCodingDomain, ...],
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
) -> str | None:
    if creative_constraints is not None and creative_constraints.hitl_advisable:
        return creative_constraints.hitl_reason or (
            "Runtime-related constraints are risky enough to surface to the user."
        )
    if candidates[0].suitability == "weak":
        return "No supported runtime has strong fit for the current output goal."
    if len(scored) > 1 and scored[0].score - scored[1].score <= 2:
        return (
            "Runtime capability fit is close across multiple candidates; ask the "
            "user before changing runtime direction."
        )
    if not domains and (
        creative_plan is None or creative_plan.recommended_runtime is None
    ):
        return "Runtime/domain direction is inferred rather than user-selected."
    return None


def _profile_prompt_guidance(
    *,
    likely_candidates: tuple[RuntimeCapabilityId, ...],
    candidates: tuple[RuntimeCapabilityCandidate, ...],
    hitl_reason: str | None,
) -> tuple[str, ...]:
    labels = _candidate_labels_for_ids(likely_candidates, candidates)
    guidance = [
        "Use runtime capability metadata to explain trade-offs, not to change "
        "the selected runtime.",
        "Keep generated code aligned with the route and existing planning contract.",
        "Likely candidates are non-binding: " + ", ".join(labels) + ".",
    ]
    if hitl_reason is not None:
        guidance.append(hitl_reason)
    guidance.extend(candidates[0].prompt_guidance[:2])
    return _dedupe_text(guidance)[:8]


def _profile_evidence(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    domains: tuple[CreativeCodingDomain, ...],
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    scored: list[_ScoredRuntime],
) -> tuple[str, ...]:
    evidence = [f"Mode: {request.mode.value}."]
    if route_decision is not None:
        evidence.append(f"Route selected: {route_decision.route.value}.")
    if domains:
        evidence.append(
            "Domains: " + ", ".join(domain.value for domain in domains) + "."
        )
    if creative_strategy is not None:
        evidence.append(f"Creative strategy: {creative_strategy.primary_strategy}.")
    if creative_techniques is not None:
        evidence.append(
            f"Creative technique: {creative_techniques.primary_technique}."
        )
    if creative_plan is not None:
        evidence.append(
            f"Planning runtime signal: {creative_plan.recommended_runtime}."
        )
        evidence.append(f"Output modality: {creative_plan.output_modality.value}.")
    if creative_constraints is not None:
        evidence.append(f"Constraint runtime fit: {creative_constraints.runtime_fit}.")
    evidence.append(
        "Top runtime scores: "
        + ", ".join(f"{item.signal.runtime}={item.score}" for item in scored[:4])
        + "."
    )
    return _dedupe_text(evidence)[:10]


def _candidate_labels(profile: RuntimeCapabilityProfile) -> tuple[str, ...]:
    return _candidate_labels_for_ids(
        profile.likely_candidates,
        profile.candidate_runtimes,
    )


def _candidate_labels_for_ids(
    ids: tuple[RuntimeCapabilityId, ...],
    candidates: tuple[RuntimeCapabilityCandidate, ...],
) -> tuple[str, ...]:
    by_runtime = {candidate.runtime: candidate.label for candidate in candidates}
    return tuple(by_runtime[item] for item in ids if item in by_runtime)


def _strategy_context(
    creative_strategy: CreativeStrategyProfile | None,
) -> str | None:
    if creative_strategy is None:
        return None
    return (
        f"{creative_strategy.primary_strategy} "
        f"with confidence {creative_strategy.confidence:.2f}."
    )


def _technique_context(
    creative_techniques: CreativeTechniqueProfile | None,
) -> str | None:
    if creative_techniques is None:
        return None
    return (
        f"{creative_techniques.primary_technique} "
        f"with {creative_techniques.performance_pressure} performance pressure."
    )


def _constraint_context(
    creative_constraints: CreativeConstraintSolution | None,
) -> str | None:
    if creative_constraints is None:
        return None
    return (
        f"Runtime fit {creative_constraints.runtime_fit}; "
        f"complexity {creative_constraints.complexity_pressure}; "
        f"performance {creative_constraints.performance_pressure}; "
        f"HITL {creative_constraints.hitl_advisable}."
    )


def _output_goal(
    request: AssistantRequest,
    creative_translation: CreativeTranslation | None,
    creative_plan: CreativeExecutionPlan | None,
) -> str:
    if creative_plan is not None:
        return creative_plan.generation_strategy
    if creative_translation is not None:
        return creative_translation.creative_intent
    return _compact(request.query)[:360]


def _output_modality(
    creative_translation: CreativeTranslation | None,
    creative_plan: CreativeExecutionPlan | None,
) -> CreativeOutputModality | None:
    if creative_plan is not None:
        return creative_plan.output_modality
    if creative_translation is not None:
        return creative_translation.output_modality
    return None


def _runtime_text(
    *,
    request: AssistantRequest,
    creative_translation: CreativeTranslation | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_plan: CreativeExecutionPlan | None,
) -> str:
    parts = [request.query]
    if creative_translation is not None:
        parts.extend(
            (
                creative_translation.creative_intent,
                " ".join(creative_translation.runtime_recommendations),
                " ".join(creative_translation.musical_references),
                " ".join(creative_translation.mood_atmosphere),
                " ".join(creative_translation.movement_language),
                " ".join(creative_translation.structure_direction),
            )
        )
    if creative_strategy is not None:
        parts.extend(
            (
                creative_strategy.primary_strategy,
                " ".join(creative_strategy.creative_goals),
                " ".join(creative_strategy.strategy_directives),
            )
        )
    if creative_techniques is not None:
        parts.extend(
            (
                creative_techniques.primary_technique,
                " ".join(creative_techniques.implementation_notes),
                " ".join(creative_techniques.artistic_suitability),
            )
        )
    if creative_plan is not None:
        parts.extend(
            (
                creative_plan.recommended_runtime or "",
                creative_plan.recommended_renderer_id or "",
                creative_plan.runtime_support_summary,
                creative_plan.generation_strategy,
            )
        )
    return _compact(" ".join(parts)).lower()


def _effective_domains(
    request: AssistantRequest,
    route_decision: RouteDecision | None,
) -> tuple[CreativeCodingDomain, ...]:
    if request.domains:
        return request.domains
    if request.domain is not None:
        return (request.domain,)
    if route_decision is not None and route_decision.domains:
        return route_decision.domains
    if request.artifact_refinement and request.artifact_refinement.domain:
        return (request.artifact_refinement.domain,)
    return ()


def _keyword_matches(normalized: str, keyword: str) -> bool:
    return re.search(rf"\b{re.escape(keyword)}\b", normalized) is not None


def _compact(value: str) -> str:
    return " ".join(value.strip().split())


def _dedupe_text(values: list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        cleaned = _compact(value)
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return tuple(normalized)


_RELATED_DOMAINS: dict[RuntimeCapabilityId, tuple[CreativeCodingDomain, ...]] = {
    "three_js": (CreativeCodingDomain.REACT_THREE_FIBER,),
    "react_three_fiber": (CreativeCodingDomain.THREE_JS,),
    "p5_js": (CreativeCodingDomain.P5_SOUND, CreativeCodingDomain.CANVAS_2D),
    "canvas": (CreativeCodingDomain.P5_JS,),
    "tone_js": (CreativeCodingDomain.WEB_AUDIO_API, CreativeCodingDomain.P5_SOUND),
}

_RUNTIME_SIGNALS: tuple[_RuntimeSignal, ...] = (
    _RuntimeSignal(
        runtime="p5_js",
        label="p5.js",
        domains=(CreativeCodingDomain.P5_JS,),
        tokens=("p5", "p5.js", "sketch", "draw loop", "canvas", "2d"),
        strategy_affinities=(
            "particle_cosmology",
            "field_dynamics",
            "sacred_geometry",
            "fractal_growth",
            "cellular_evolution",
            "minimal_generative_systems",
        ),
        technique_affinities=(
            "particle_systems",
            "boids",
            "cellular_automata",
            "noise_fields",
            "recursive_geometry",
            "fractal_recursion",
            "feedback_systems",
            "audio_reactive_mappings",
        ),
        modality_affinities=(
            CreativeOutputModality.VISUAL,
            CreativeOutputModality.AUDIOVISUAL,
        ),
        implementation_complexity="low",
        performance_pressure="medium",
        preview_support="backend_preview_supported",
        strengths=(
            "Fast path for approachable 2D generative sketches.",
            "Good fit for particles, fields, interactive drawing, and visual studies.",
        ),
        limitations=(
            "Less natural for deep 3D scenes or shader-first composition.",
            "Audio use still needs explicit browser-safe input boundaries.",
        ),
        risks=(
            "Large particle counts or dense pixels can pressure browser frame rate.",
        ),
        prompt_guidance=(
            "Use p5.js capability when the output should be a self-contained sketch.",
            "Keep setup(), draw(), and browser-safe globals clear if p5.js "
            "remains in scope.",
        ),
    ),
    _RuntimeSignal(
        runtime="three_js",
        label="Three.js",
        domains=(CreativeCodingDomain.THREE_JS,),
        tokens=("three", "three.js", "webgl", "3d", "scene", "mesh", "camera"),
        strategy_affinities=("particle_cosmology", "field_dynamics", "sacred_geometry"),
        technique_affinities=(
            "particle_systems",
            "boids",
            "sdf",
            "signed_distance_composition",
            "recursive_geometry",
        ),
        modality_affinities=(CreativeOutputModality.VISUAL,),
        implementation_complexity="medium",
        performance_pressure="high",
        preview_support="backend_preview_supported",
        strengths=(
            "Strong for 3D scenes, camera movement, materials, and spatial particles.",
            "Supports richer spatial composition than 2D sketch runtimes.",
        ),
        limitations=(
            "Requires scene, camera, renderer, animation-loop, and resource "
            "discipline.",
            "Less concise for simple 2D symbolic compositions.",
        ),
        risks=("Complex materials and large geometries can raise performance risk.",),
        prompt_guidance=(
            "Use Three.js capability for browser-oriented 3D scene structure.",
            "Prefer direct Three.js only when React wrappers are not requested.",
        ),
    ),
    _RuntimeSignal(
        runtime="react_three_fiber",
        label="React Three Fiber",
        domains=(CreativeCodingDomain.REACT_THREE_FIBER,),
        tokens=("react three fiber", "r3f", "jsx", "tsx", "useframe", "hooks"),
        strategy_affinities=("particle_cosmology", "field_dynamics", "sacred_geometry"),
        technique_affinities=(
            "particle_systems",
            "boids",
            "sdf",
            "signed_distance_composition",
            "recursive_geometry",
        ),
        modality_affinities=(CreativeOutputModality.VISUAL,),
        implementation_complexity="high",
        performance_pressure="high",
        preview_support="backend_preview_supported",
        strengths=(
            "Strong when the user needs React component composition around Three.js.",
            "Good fit for stateful interactive 3D sketches in a React app.",
        ),
        limitations=(
            "Adds React component and hook complexity over plain Three.js.",
            "Not ideal unless React, JSX, TSX, or Canvas components are requested.",
        ),
        risks=("Mixing imperative Three.js escapes with React state can be fragile.",),
        prompt_guidance=(
            "Use React Three Fiber capability only when React component form is "
            "in scope.",
            "Keep Canvas component structure and useFrame behavior explicit.",
        ),
    ),
    _RuntimeSignal(
        runtime="glsl",
        label="GLSL",
        domains=(CreativeCodingDomain.GLSL,),
        tokens=("glsl", "shader", "fragment", "uniform", "sdf", "raymarch"),
        strategy_affinities=("field_dynamics", "sacred_geometry", "fractal_growth"),
        technique_affinities=(
            "reaction_diffusion",
            "noise_fields",
            "sdf",
            "signed_distance_composition",
            "fractal_recursion",
            "cellular_automata",
            "feedback_systems",
        ),
        modality_affinities=(CreativeOutputModality.VISUAL,),
        implementation_complexity="high",
        performance_pressure="high",
        preview_support="backend_preview_supported",
        strengths=(
            "Strong for shader fields, distance functions, procedural texture, "
            "and glow.",
            "Useful when visual behavior should emerge from compact "
            "mathematical rules.",
        ),
        limitations=(
            "Less suitable for DOM, UI, or high-level scene graph requirements.",
            "Requires shader-safe uniforms and fragment-stage assumptions.",
        ),
        risks=("High iteration counts or raymarch loops can exceed browser budgets.",),
        prompt_guidance=(
            "Use GLSL capability for fragment-shader style procedural visuals.",
            "Keep uniform names and iteration bounds explicit.",
        ),
    ),
    _RuntimeSignal(
        runtime="hydra",
        label="Hydra",
        domains=(CreativeCodingDomain.HYDRA,),
        tokens=("hydra", "osc", "feedback", "video synth", "live coded"),
        strategy_affinities=("field_dynamics", "recursive_emergence"),
        technique_affinities=(
            "feedback_systems",
            "noise_fields",
            "audio_reactive_mappings",
        ),
        modality_affinities=(
            CreativeOutputModality.VISUAL,
            CreativeOutputModality.AUDIOVISUAL,
        ),
        implementation_complexity="medium",
        performance_pressure="medium",
        preview_support="workstation_preview_bounded",
        strengths=(
            "Strong for live-coded video synthesis, feedback, oscillators, and "
            "modulation.",
            "Useful for audiovisual texture and iterative performance sketches.",
        ),
        limitations=(
            "Less suitable for conventional object scenes or full app structure.",
            "Backend planning does not currently select Hydra as a generation runtime.",
        ),
        risks=("Feedback-heavy patches can become visually unstable quickly.",),
        prompt_guidance=(
            "Use Hydra capability as comparison guidance when live video "
            "synthesis is requested.",
            "Keep feedback and modulation chains bounded and readable.",
        ),
    ),
    _RuntimeSignal(
        runtime="tone_js",
        label="Tone.js",
        domains=(CreativeCodingDomain.TONE_JS,),
        tokens=("tone", "tone.js", "synth", "sequencer", "rhythm", "audio"),
        strategy_affinities=("field_dynamics", "minimal_generative_systems"),
        technique_affinities=("audio_reactive_mappings", "feedback_systems"),
        modality_affinities=(
            CreativeOutputModality.AUDIO,
            CreativeOutputModality.AUDIOVISUAL,
        ),
        implementation_complexity="medium",
        performance_pressure="medium",
        preview_support="workstation_preview_bounded",
        strengths=(
            "Strong for browser-based synthesis, timing, sequencing, and "
            "musical structure.",
            "Pairs well with audiovisual briefs that need explicit sonic behavior.",
        ),
        limitations=(
            "Does not solve visual rendering by itself.",
            "Backend planning treats Tone.js generation as code-only unless "
            "routed separately.",
        ),
        risks=(
            "Audio context start, user gesture, and timing boundaries must stay "
            "explicit.",
        ),
        prompt_guidance=(
            "Use Tone.js capability when sound generation is central to the brief.",
            "Keep audio start behavior, transport timing, and visual bridge "
            "assumptions explicit.",
        ),
    ),
    _RuntimeSignal(
        runtime="gsap",
        label="GSAP",
        domains=(CreativeCodingDomain.GSAP,),
        tokens=("gsap", "timeline", "tween", "motion path", "animation"),
        strategy_affinities=("minimal_generative_systems", "sacred_geometry"),
        technique_affinities=("recursive_geometry", "feedback_systems"),
        modality_affinities=(CreativeOutputModality.VISUAL,),
        implementation_complexity="low",
        performance_pressure="medium",
        preview_support="workstation_preview_bounded",
        strengths=(
            "Strong for timeline-driven animation, staged motion, and DOM/SVG "
            "transitions.",
            "Good fit when choreography matters more than simulation.",
        ),
        limitations=(
            "Less suitable for heavy particle simulation or shader fields.",
            "Plugin-specific GSAP features may exceed bounded preview assumptions.",
        ),
        risks=("Plugin or DOM assumptions can make output less portable.",),
        prompt_guidance=(
            "Use GSAP capability for choreographed animation and explicit timelines.",
            "Avoid unsupported plugin assumptions unless the user requested them.",
        ),
    ),
    _RuntimeSignal(
        runtime="svg",
        label="SVG",
        domains=(),
        tokens=("svg", "vector", "path", "viewbox", "symbol", "icon", "diagram"),
        strategy_affinities=(
            "sacred_geometry",
            "minimal_generative_systems",
            "fractal_growth",
        ),
        technique_affinities=(
            "recursive_geometry",
            "fractal_recursion",
            "voronoi",
        ),
        modality_affinities=(CreativeOutputModality.VISUAL,),
        implementation_complexity="low",
        performance_pressure="low",
        preview_support="workstation_preview_bounded",
        strengths=(
            "Strong for crisp vector geometry, symbolic structures, and static "
            "compositions.",
            "Fits precise shape systems and lightweight browser presentation.",
        ),
        limitations=(
            "Not an existing backend CreativeCodingDomain value.",
            "Limited for dense simulation, shader effects, and arbitrary scripting.",
        ),
        risks=(
            "Unsafe SVG markup, external assets, or event handlers must be "
            "avoided.",
        ),
        prompt_guidance=(
            "Use SVG capability for self-contained vector output only when the "
            "brief supports it.",
            "Keep markup sanitized and avoid scriptable or remote SVG constructs.",
        ),
    ),
    _RuntimeSignal(
        runtime="canvas",
        label="Canvas",
        domains=(CreativeCodingDomain.CANVAS_2D,),
        tokens=("canvas", "canvas2d", "2d context", "ctx", "raster"),
        strategy_affinities=(
            "particle_cosmology",
            "field_dynamics",
            "cellular_evolution",
            "minimal_generative_systems",
        ),
        technique_affinities=(
            "particle_systems",
            "boids",
            "cellular_automata",
            "noise_fields",
            "feedback_systems",
            "fractal_recursion",
        ),
        modality_affinities=(CreativeOutputModality.VISUAL,),
        implementation_complexity="medium",
        performance_pressure="medium",
        preview_support="workstation_preview_bounded",
        strengths=(
            "Strong for direct 2D raster control and lightweight custom loops.",
            "Useful when p5 abstractions are unnecessary or not requested.",
        ),
        limitations=(
            "Requires more manual setup than p5.js.",
            "Backend planning does not currently select Canvas as a generation "
            "runtime.",
        ),
        risks=("Manual resize, animation-loop, and cleanup behavior can be missed.",),
        prompt_guidance=(
            "Use Canvas capability when direct 2D context control is valuable.",
            "Keep resize, animation loop, and drawing state boundaries explicit.",
        ),
    ),
)
