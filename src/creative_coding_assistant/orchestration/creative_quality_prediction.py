"""Pre-generation Creative Quality Predictor for V3 workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
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
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityCandidate,
    RuntimeCapabilityProfile,
)

CreativeQualityLevel = Literal[
    "strong",
    "promising",
    "ambiguous",
    "risky",
    "blocked",
]
CreativeQualityDimension = Literal[
    "intent_clarity",
    "symbolic_coherence",
    "narrative_coherence",
    "emotional_coherence",
    "geometric_formal_clarity",
    "technique_suitability",
    "runtime_suitability",
    "tradeoff_balance",
    "constraint_alignment",
    "implementation_feasibility",
    "previewability",
    "performance_risk",
    "originality_potential",
    "aesthetic_coherence_potential",
]

QUALITY_PREDICTOR_AUTHORITY_BOUNDARY = (
    "The Creative Quality Predictor estimates pre-generation readiness only; "
    "it does not critique generated artifacts, rank or select artifacts, "
    "auto-select runtimes, route providers or models, change preview/runtime "
    "behavior, generate code, or run autonomous repair loops."
)

_AMBIGUITY_TOKENS = frozenset(
    {
        "maybe",
        "something",
        "whatever",
        "vibe",
        "vibes",
        "cool",
        "interesting",
        "nice",
        "stuff",
        "thing",
        "profound",
        "cinematic",
    }
)
_GEOMETRY_TOKENS = frozenset(
    {
        "circle",
        "circular",
        "grid",
        "lattice",
        "mandala",
        "radial",
        "ring",
        "spiral",
        "symmetry",
        "triangle",
        "voronoi",
        "sacred",
        "geometry",
    }
)
_LIGHT_COLOR_TOKENS = frozenset(
    {
        "blue",
        "gold",
        "palette",
        "color",
        "colour",
        "glow",
        "luminous",
        "light",
        "shadow",
        "contrast",
        "material",
    }
)
_CONFLICT_TOKENS = frozenset(
    {
        "dense",
        "spectacle",
        "complex",
        "audio",
        "interactive",
        "interaction",
        "mobile",
        "60",
        "fps",
        "browser-safe",
    }
)


class CreativeQualitySignal(BaseModel):
    """One scored pre-generation quality signal."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    dimension: CreativeQualityDimension
    score: int = Field(ge=0, le=10)
    summary: str = Field(min_length=1, max_length=260)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=6)


class CreativeQualityPrediction(BaseModel):
    """Inspectable prediction of plan readiness before generation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_quality_predictor"] = "creative_quality_predictor"
    predicted_quality_level: CreativeQualityLevel
    confidence: float = Field(ge=0, le=1)
    readiness_score: int = Field(ge=0, le=100)
    strongest_quality_signals: tuple[CreativeQualitySignal, ...] = Field(
        min_length=1,
        max_length=6,
    )
    weakest_quality_signals: tuple[CreativeQualitySignal, ...] = Field(
        min_length=1,
        max_length=6,
    )
    quality_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    missing_information: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    likely_failure_modes: tuple[str, ...] = Field(min_length=1, max_length=8)
    suggested_improvements: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=QUALITY_PREDICTOR_AUTHORITY_BOUNDARY,
        max_length=520,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=12)


@dataclass(frozen=True)
class _QualityContext:
    request: AssistantRequest
    route_decision: RouteDecision | None
    creative_translation: CreativeTranslation | None
    creative_intent: CreativeIntentDecomposition | None
    creative_hierarchy: CreativeHierarchyPlan | None
    creative_plan: CreativeExecutionPlan | None
    creative_constraints: CreativeConstraintSolution | None
    creative_constraint_priorities: CreativeConstraintPrioritization | None
    creative_strategy: CreativeStrategyProfile | None
    creative_techniques: CreativeTechniqueProfile | None
    runtime_capabilities: RuntimeCapabilityProfile | None
    creative_tradeoffs: CreativeTradeoffProfile | None
    normalized_text: str
    tokens: frozenset[str]


def derive_creative_quality_prediction(
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
) -> CreativeQualityPrediction:
    """Predict pre-generation quality readiness from existing metadata only."""

    context = _context(
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
    )
    signals = (
        _intent_clarity_signal(context),
        _symbolic_coherence_signal(context),
        _narrative_coherence_signal(context),
        _emotional_coherence_signal(context),
        _geometric_formal_signal(context),
        _technique_suitability_signal(context),
        _runtime_suitability_signal(context),
        _tradeoff_balance_signal(context),
        _constraint_alignment_signal(context),
        _implementation_feasibility_signal(context),
        _previewability_signal(context),
        _performance_risk_signal(context),
        _originality_potential_signal(context),
        _aesthetic_coherence_signal(context),
    )
    missing_information = _missing_information(context, signals)
    quality_risks = _quality_risks(context, signals)
    readiness = _readiness_score(
        signals,
        missing_information=missing_information,
        quality_risks=quality_risks,
    )
    quality_level = _quality_level(readiness)
    likely_failure_modes = _likely_failure_modes(
        signals,
        missing_information=missing_information,
        quality_risks=quality_risks,
    )
    improvements = _suggested_improvements(
        signals,
        missing_information=missing_information,
        quality_risks=quality_risks,
        creative_tradeoffs=creative_tradeoffs,
    )
    hitl_questions = _hitl_questions(
        readiness_score=readiness,
        missing_information=missing_information,
        quality_risks=quality_risks,
        context=context,
    )
    strongest = tuple(
        sorted(signals, key=lambda item: (item.score, item.dimension), reverse=True)[:6]
    )
    weakest = tuple(sorted(signals, key=lambda item: (item.score, item.dimension))[:6])
    return CreativeQualityPrediction(
        predicted_quality_level=quality_level,
        confidence=_confidence(
            context,
            readiness_score=readiness,
            missing_information=missing_information,
            quality_risks=quality_risks,
        ),
        readiness_score=readiness,
        strongest_quality_signals=strongest,
        weakest_quality_signals=weakest,
        quality_risks=quality_risks,
        missing_information=missing_information,
        likely_failure_modes=likely_failure_modes,
        suggested_improvements=improvements,
        hitl_questions=hitl_questions,
        prompt_guidance=_prompt_guidance(
            readiness_score=readiness,
            quality_level=quality_level,
            weakest_signals=weakest,
            hitl_questions=hitl_questions,
        ),
        evidence=_evidence(context, readiness_score=readiness),
    )


def creative_quality_prediction_prompt_lines(
    prediction: CreativeQualityPrediction,
) -> tuple[str, ...]:
    """Render pre-generation quality metadata as compact prompt guidance."""

    lines = [
        f"Authority boundary: {prediction.authority_boundary}",
        (
            "Predicted quality: "
            f"{prediction.predicted_quality_level}; readiness "
            f"{prediction.readiness_score}/100; confidence "
            f"{prediction.confidence:.2f}."
        ),
    ]
    for signal in prediction.strongest_quality_signals[:3]:
        lines.append(
            "Strong quality signal: "
            f"{signal.dimension} scored {signal.score}/10; {signal.summary}"
        )
    for signal in prediction.weakest_quality_signals[:4]:
        lines.append(
            "Weak quality signal: "
            f"{signal.dimension} scored {signal.score}/10; {signal.summary}"
        )
    lines.extend(f"Quality risk: {item}" for item in prediction.quality_risks[:4])
    lines.extend(
        f"Missing information: {item}" for item in prediction.missing_information[:4]
    )
    lines.extend(
        f"Likely failure mode: {item}" for item in prediction.likely_failure_modes[:4]
    )
    lines.extend(
        f"Suggested improvement: {item}"
        for item in prediction.suggested_improvements[:4]
    )
    lines.extend(
        f"HITL quality question: {item}" for item in prediction.hitl_questions[:3]
    )
    lines.extend(
        f"Quality predictor guidance: {item}" for item in prediction.prompt_guidance[:5]
    )
    return tuple(lines[:32])


def _context(
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
) -> _QualityContext:
    text_parts = [
        request.query,
        creative_translation.creative_intent if creative_translation else "",
        creative_plan.generation_strategy if creative_plan else "",
        creative_strategy.rationale if creative_strategy else "",
        creative_techniques.rationale if creative_techniques else "",
    ]
    normalized = _normalize_text(" ".join(text_parts))
    return _QualityContext(
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
        normalized_text=normalized,
        tokens=frozenset(re.findall(r"[a-z0-9_.+#-]+", normalized)),
    )


def _intent_clarity_signal(context: _QualityContext) -> CreativeQualitySignal:
    score = 5
    evidence: list[str] = []
    intent = context.creative_intent
    if intent is not None:
        active = [
            item.name
            for item in intent.atomic_dimensions
            if item.explicitness in {"explicit", "inferred"}
        ]
        score += min(len(active), 4)
        evidence.append(f"active intent dimensions:{len(active)}")
        if intent.unresolved_intent_gaps:
            score -= min(len(intent.unresolved_intent_gaps) * 2, 5)
            evidence.append(f"intent gaps:{len(intent.unresolved_intent_gaps)}")
        else:
            score += 1
            evidence.append("no unresolved intent gaps")
    if len(context.request.query.split()) >= 10:
        score += 1
        evidence.append("request has concrete detail")
    ambiguity = _ambiguity_count(context)
    if ambiguity:
        score -= min(ambiguity * 2, 5)
        evidence.append(f"ambiguous wording:{ambiguity}")
    return _signal(
        "intent_clarity",
        score,
        "Intent is clear enough to guide generation."
        if score >= 7
        else "Intent needs sharper subject, modality, or experiential detail.",
        evidence,
    )


def _symbolic_coherence_signal(context: _QualityContext) -> CreativeQualitySignal:
    score = 6
    evidence: list[str] = []
    if _intent_active(context, "symbolic"):
        score += 2
        evidence.append("symbolic intent active")
    if context.creative_strategy and context.creative_strategy.symbolic_alignment:
        score += 1
        evidence.append("strategy symbolic alignment")
    if _protected_category(context, "symbolic_fidelity"):
        score += 1
        evidence.append("symbolic fidelity protected")
        if not _intent_active(context, "geometric") and not _has_geometry(context):
            score -= 4
            evidence.append("protected symbolism lacks formal specificity")
    if not _intent_active(context, "symbolic") and _has_any(
        context,
        {"symbolic", "sacred", "ritual", "myth", "profane", "profound"},
    ):
        score -= 1
        evidence.append("symbolic wording is broad")
    return _signal(
        "symbolic_coherence",
        score,
        "Symbolic direction has usable coherence."
        if score >= 7
        else "Symbolic direction may remain shallow without concrete form rules.",
        evidence,
    )


def _narrative_coherence_signal(context: _QualityContext) -> CreativeQualitySignal:
    score = 6
    evidence: list[str] = []
    if _intent_active(context, "narrative"):
        score += 2
        evidence.append("narrative intent active")
    elif _intent_active(context, "climax_transformation"):
        score += 1
        evidence.append("transformation arc implies narrative movement")
    else:
        evidence.append("no required narrative arc")
    if _has_any(context, {"journey", "story", "arc", "transformation"}):
        score += 1
    if _ambiguity_count(context) >= 2:
        score -= 1
    return _signal(
        "narrative_coherence",
        score,
        "Narrative demand is bounded and coherent."
        if score >= 7
        else "Narrative arc is optional or underspecified.",
        evidence,
    )


def _emotional_coherence_signal(context: _QualityContext) -> CreativeQualitySignal:
    score = 5
    evidence: list[str] = []
    if _intent_active(context, "emotional"):
        score += 3
        evidence.append("emotional intent active")
    if context.creative_intent and any(
        "Emotional tone" in item
        for item in context.creative_intent.unresolved_intent_gaps
    ):
        score -= 3
        evidence.append("emotional tone gap")
    if _has_any(context, {"awe", "calm", "tranquil", "ritual", "cinematic"}):
        score += 1
    return _signal(
        "emotional_coherence",
        score,
        "Emotional tone can anchor aesthetic choices."
        if score >= 7
        else "Emotional tone needs a clearer target to avoid generic mood.",
        evidence,
    )


def _geometric_formal_signal(context: _QualityContext) -> CreativeQualitySignal:
    score = 5
    evidence: list[str] = []
    if _intent_active(context, "geometric"):
        score += 3
        evidence.append("geometric intent active")
    if _has_geometry(context):
        score += 2
        evidence.append("geometry/form keywords present")
    if _protected_category(context, "symbolic_fidelity") and not _has_geometry(context):
        score -= 3
        evidence.append("symbolic protection lacks explicit geometry")
    return _signal(
        "geometric_formal_clarity",
        score,
        "Formal structure is specific enough to guide composition."
        if score >= 7
        else "Formal structure needs clearer geometry, layout, or composition.",
        evidence,
    )


def _technique_suitability_signal(context: _QualityContext) -> CreativeQualitySignal:
    technique = context.creative_techniques
    score = 5
    evidence: list[str] = []
    if technique is not None:
        score += round(technique.confidence * 3)
        evidence.append(f"technique confidence:{technique.confidence:.2f}")
        score += {"strong": 2, "moderate": 1, "weak": -2}[technique.compatibility]
        evidence.append(f"strategy compatibility:{technique.compatibility}")
        if technique.complexity_pressure == "high":
            score -= 1
        if technique.performance_pressure == "high":
            score -= 1
    return _signal(
        "technique_suitability",
        score,
        "Selected technique fits the strategy and output goal."
        if score >= 7
        else "Technique choice may need simplification or stronger alignment.",
        evidence,
    )


def _runtime_suitability_signal(context: _QualityContext) -> CreativeQualitySignal:
    candidate = _top_runtime(context.runtime_capabilities)
    score = 5
    evidence: list[str] = []
    if candidate is not None:
        score += {"strong": 3, "moderate": 1, "weak": -2}[candidate.suitability]
        score += round(candidate.confidence * 2)
        evidence.append(f"top runtime:{candidate.runtime}")
        evidence.append(f"suitability:{candidate.suitability}")
        if candidate.preview_support == "code_only":
            score -= 1
            evidence.append("code-only preview")
    if context.creative_constraints is not None:
        runtime_fit = context.creative_constraints.runtime_fit
        score += {"supported": 1, "code_only": -2, "undetermined": -3}[runtime_fit]
        evidence.append(f"constraint runtime fit:{runtime_fit}")
    return _signal(
        "runtime_suitability",
        score,
        "Inspected runtime support is suitable for generation."
        if score >= 7
        else "Runtime support may limit output quality or inspectability.",
        evidence,
    )


def _tradeoff_balance_signal(context: _QualityContext) -> CreativeQualitySignal:
    score = 7
    evidence: list[str] = []
    tradeoffs = context.creative_tradeoffs
    if tradeoffs is not None:
        severity_penalty = {"info": 0, "watch": 0, "risk": 1, "blocking": 3}
        penalty = max(
            severity_penalty[item.severity] for item in tradeoffs.primary_tradeoffs
        )
        score -= penalty
        evidence.append(f"max trade-off penalty:{penalty}")
        if tradeoffs.hitl_advisable:
            score -= 1
            evidence.append("trade-off HITL advisable")
    priorities = context.creative_constraint_priorities
    if priorities is not None and priorities.conflict_relationships:
        score -= 1
        evidence.append("priority conflicts present")
    return _signal(
        "tradeoff_balance",
        score,
        "Trade-offs look balanced enough for bounded generation."
        if score >= 7
        else "Trade-offs need clearer sacrifice or mitigation rules.",
        evidence,
    )


def _constraint_alignment_signal(context: _QualityContext) -> CreativeQualitySignal:
    score = 7
    evidence: list[str] = []
    constraints = context.creative_constraints
    priorities = context.creative_constraint_priorities
    if constraints is not None:
        if constraints.runtime_fit == "supported":
            score += 1
        if constraints.conflicts:
            score -= min(len(constraints.conflicts), 2)
            evidence.append(f"solver conflicts:{len(constraints.conflicts)}")
        if constraints.hitl_advisable:
            score -= 1
            evidence.append("solver HITL advisable")
    if priorities is not None:
        score += min(len(priorities.non_negotiable_constraints), 1)
        if priorities.conflict_relationships:
            score -= 1
        evidence.append(f"non-negotiable:{len(priorities.non_negotiable_constraints)}")
    return _signal(
        "constraint_alignment",
        score,
        "Constraint priorities are aligned with the plan."
        if score >= 7
        else "Constraint alignment may require explicit relaxation choices.",
        evidence,
    )


def _implementation_feasibility_signal(
    context: _QualityContext,
) -> CreativeQualitySignal:
    score = 7
    evidence: list[str] = []
    plan = context.creative_plan
    if plan is not None:
        score += {"low": 1, "medium": 0, "high": -1}[plan.expected_complexity]
        score += {"ready": 1, "partial": -1, "blocked": -4}[plan.export_readiness]
        if not plan.runtime_available:
            score -= 2
        evidence.append(f"plan complexity:{plan.expected_complexity}")
        evidence.append(f"export readiness:{plan.export_readiness}")
    if context.creative_constraints is not None:
        pressure = context.creative_constraints.complexity_pressure
        score += {"low": 1, "medium": 0, "high": -1}[pressure]
        evidence.append(f"complexity pressure:{pressure}")
    return _signal(
        "implementation_feasibility",
        score,
        "Implementation scope appears feasible."
        if score >= 7
        else "Implementation scope may need reduction before generation.",
        evidence,
    )


def _previewability_signal(context: _QualityContext) -> CreativeQualitySignal:
    score = 5
    evidence: list[str] = []
    plan = context.creative_plan
    if plan is not None:
        if plan.runtime_available:
            score += 2
            evidence.append("plan runtime available")
        else:
            score -= 2
            evidence.append("no plan runtime available")
    candidate = _top_runtime(context.runtime_capabilities)
    if candidate is not None:
        score += {
            "backend_preview_supported": 2,
            "workstation_preview_bounded": 1,
            "code_only": -2,
        }[candidate.preview_support]
        evidence.append(f"preview support:{candidate.preview_support}")
    if _protected_category(context, "previewability"):
        score += 1
    if _sacrificial_category(context, "previewability"):
        score -= 1
        evidence.append("previewability marked sacrificial")
    return _signal(
        "previewability",
        score,
        "Preview path is inspectable enough for this plan."
        if score >= 7
        else "Previewability may be limited or explicitly sacrificial.",
        evidence,
    )


def _performance_risk_signal(context: _QualityContext) -> CreativeQualitySignal:
    score = 8
    evidence: list[str] = []
    if context.creative_constraints is not None:
        pressure = context.creative_constraints.performance_pressure
        score += {"low": 2, "medium": 0, "high": -2}[pressure]
        evidence.append(f"solver performance:{pressure}")
    if context.creative_techniques is not None:
        pressure = context.creative_techniques.performance_pressure
        score += {"low": 1, "medium": 0, "high": -1}[pressure]
        evidence.append(f"technique performance:{pressure}")
    if context.creative_tradeoffs and context.creative_tradeoffs.performance_concerns:
        score -= min(len(context.creative_tradeoffs.performance_concerns), 2)
        evidence.append("performance trade-off concerns")
    if _has_any(context, {"60", "fps", "mobile", "dense", "spectacle"}):
        score -= 1 if _has_conflict_pressure(context) else 0
        evidence.append("performance-sensitive wording")
    return _signal(
        "performance_risk",
        score,
        "Performance risk appears controlled."
        if score >= 7
        else "Performance pressure could reduce creative quality.",
        evidence,
    )


def _originality_potential_signal(context: _QualityContext) -> CreativeQualitySignal:
    score = 5
    evidence: list[str] = []
    if context.creative_strategy is not None:
        strategy = context.creative_strategy.primary_strategy
        if strategy != "minimal_generative_systems":
            score += 2
        score += round(context.creative_strategy.confidence * 2)
        evidence.append(f"strategy:{strategy}")
    if _intent_active(context, "symbolic") or _intent_active(context, "emotional"):
        score += 1
    if _ambiguity_count(context) >= 2:
        score -= 2
    return _signal(
        "originality_potential",
        score,
        "The concept has enough specificity for original expression."
        if score >= 7
        else "Originality may collapse into generic effects without refinement.",
        evidence,
    )


def _aesthetic_coherence_signal(context: _QualityContext) -> CreativeQualitySignal:
    score = 5
    evidence: list[str] = []
    if _intent_active(context, "light_color") or _has_light_color(context):
        score += 2
        evidence.append("light/color direction present")
    if _intent_active(context, "emotional"):
        score += 1
        evidence.append("emotional anchor present")
    if _intent_active(context, "geometric") or _has_geometry(context):
        score += 1
        evidence.append("formal anchor present")
    if context.creative_translation and context.creative_translation.visual_style:
        score += 1
        evidence.append("visual style guidance present")
    if _ambiguity_count(context) >= 2:
        score -= 2
    return _signal(
        "aesthetic_coherence_potential",
        score,
        "Aesthetic cues can cohere around visible anchors."
        if score >= 7
        else "Aesthetic direction needs stronger palette, form, or mood anchors.",
        evidence,
    )


def _missing_information(
    context: _QualityContext,
    signals: tuple[CreativeQualitySignal, ...],
) -> tuple[str, ...]:
    missing: list[str] = []
    if context.creative_intent is not None:
        missing.extend(context.creative_intent.unresolved_intent_gaps[:4])
    if _protected_category(context, "symbolic_fidelity") and not _has_geometry(context):
        missing.append(
            "Symbolic fidelity is protected, but visual/geometric specificity is "
            "not explicit."
        )
    if _intent_active(context, "interaction") and not _intent_active(
        context,
        "climax_transformation",
    ):
        missing.append("Interaction is present, but the state-change rule is unclear.")
    if _visual_plan(context) and not _has_light_color(context):
        missing.append("Palette, lighting, or material direction is not explicit.")
    if _score_for(signals, "intent_clarity") <= 5:
        missing.append("Core subject, motif, or experiential target needs refinement.")
    return _dedupe(missing)[:8]


def _quality_risks(
    context: _QualityContext,
    signals: tuple[CreativeQualitySignal, ...],
) -> tuple[str, ...]:
    risks: list[str] = []
    for signal in signals:
        if signal.score <= 3:
            risks.append(f"{signal.dimension} is weak: {signal.summary}")
    performance_score = _score_for(signals, "performance_risk")
    runtime_score = _score_for(signals, "runtime_suitability")
    tradeoff_score = _score_for(signals, "tradeoff_balance")
    constraint_score = _score_for(signals, "constraint_alignment")
    if performance_score <= 3:
        risks.append("High performance pressure may force visual simplification.")
    if runtime_score <= 3:
        risks.append("Runtime suitability is weak enough to threaten output quality.")
    if tradeoff_score <= 3:
        risks.append("Trade-offs may conflict without a clear sacrifice order.")
    if context.creative_tradeoffs is not None and (
        performance_score <= 4 or tradeoff_score <= 4 or _has_conflict_pressure(context)
    ):
        risks.extend(context.creative_tradeoffs.runtime_risks[:2])
        risks.extend(context.creative_tradeoffs.fidelity_risks[:2])
    if context.creative_constraints is not None and (
        constraint_score <= 4 or performance_score <= 4
    ):
        risks.extend(context.creative_constraints.conflicts[:2])
    return _dedupe(risks)[:8]


def _readiness_score(
    signals: tuple[CreativeQualitySignal, ...],
    *,
    missing_information: tuple[str, ...],
    quality_risks: tuple[str, ...],
) -> int:
    weights = {
        "intent_clarity": 1.25,
        "constraint_alignment": 1.15,
        "implementation_feasibility": 1.15,
        "runtime_suitability": 1.1,
        "technique_suitability": 1.05,
        "tradeoff_balance": 1.05,
        "performance_risk": 1.05,
    }
    weighted_total = 0.0
    total_weight = 0.0
    for signal in signals:
        weight = weights.get(signal.dimension, 1.0)
        weighted_total += signal.score * weight
        total_weight += weight
    readiness = round((weighted_total / total_weight) * 10)
    readiness -= min(len(missing_information) * 3, 12)
    readiness -= min(len(quality_risks), 8)
    return max(0, min(100, readiness))


def _quality_level(readiness_score: int) -> CreativeQualityLevel:
    if readiness_score >= 82:
        return "strong"
    if readiness_score >= 68:
        return "promising"
    if readiness_score >= 52:
        return "ambiguous"
    if readiness_score >= 35:
        return "risky"
    return "blocked"


def _likely_failure_modes(
    signals: tuple[CreativeQualitySignal, ...],
    *,
    missing_information: tuple[str, ...],
    quality_risks: tuple[str, ...],
) -> tuple[str, ...]:
    modes: list[str] = []
    weak = [item for item in signals if item.score <= 5]
    for signal in weak[:4]:
        modes.append(_failure_mode_for(signal.dimension))
    if missing_information:
        modes.append(
            "Generation may invent missing details instead of preserving intent."
        )
    if quality_risks:
        modes.append(
            "The output may overfit one pressure and weaken the creative brief."
        )
    if not modes:
        modes.append(
            "No high-likelihood failure mode is visible before generation; "
            "continue to normal review after output."
        )
    return _dedupe(modes)[:8]


def _suggested_improvements(
    signals: tuple[CreativeQualitySignal, ...],
    *,
    missing_information: tuple[str, ...],
    quality_risks: tuple[str, ...],
    creative_tradeoffs: CreativeTradeoffProfile | None,
) -> tuple[str, ...]:
    improvements: list[str] = []
    for signal in sorted(signals, key=lambda item: item.score):
        if signal.score <= 6:
            improvements.append(_improvement_for(signal.dimension))
    for item in missing_information[:2]:
        improvements.append(f"Clarify before generation: {item}")
    if creative_tradeoffs is not None:
        improvements.append(creative_tradeoffs.primary_tradeoffs[0].mitigation)
    if quality_risks and not improvements:
        improvements.append("Reduce the highest-risk pressure before adding features.")
    return _dedupe(improvements)[:8]


def _hitl_questions(
    *,
    readiness_score: int,
    missing_information: tuple[str, ...],
    quality_risks: tuple[str, ...],
    context: _QualityContext,
) -> tuple[str, ...]:
    if readiness_score >= 68:
        return ()
    questions: list[str] = []
    if context.creative_constraint_priorities is not None:
        questions.extend(context.creative_constraint_priorities.hitl_questions[:2])
    if context.creative_hierarchy is not None:
        questions.extend(context.creative_hierarchy.hitl_questions[:2])
    for item in missing_information[:3]:
        questions.append(_question_for_missing_information(item))
    if _score_sensitive_quality_risk(quality_risks, "performance"):
        questions.append(
            "Should performance constraints reduce visual density before generation?"
        )
    if _score_sensitive_quality_risk(quality_risks, "runtime"):
        questions.append("Should runtime/preview inspectability lead the design?")
    return _dedupe(questions)[:6]


def _prompt_guidance(
    *,
    readiness_score: int,
    quality_level: CreativeQualityLevel,
    weakest_signals: tuple[CreativeQualitySignal, ...],
    hitl_questions: tuple[str, ...],
) -> tuple[str, ...]:
    guidance = [
        "Treat this as pre-generation readiness guidance, not artifact critique.",
        (
            f"Predicted quality is {quality_level}; readiness score is "
            f"{readiness_score}/100."
        ),
    ]
    if hitl_questions:
        guidance.append("Ask or resolve quality HITL questions before expanding scope.")
    elif readiness_score >= 68:
        guidance.append("Proceed with bounded generation using the strongest signals.")
    else:
        guidance.append(
            "Refine the weakest signals before adding more techniques or effects."
        )
    guidance.extend(
        f"Protect against weak signal: {item.dimension}."
        for item in weakest_signals[:3]
        if item.score <= 6
    )
    return tuple(guidance[:8])


def _confidence(
    context: _QualityContext,
    *,
    readiness_score: int,
    missing_information: tuple[str, ...],
    quality_risks: tuple[str, ...],
) -> float:
    evidence_count = sum(
        item is not None
        for item in (
            context.creative_translation,
            context.creative_intent,
            context.creative_hierarchy,
            context.creative_plan,
            context.creative_constraints,
            context.creative_constraint_priorities,
            context.creative_strategy,
            context.creative_techniques,
            context.runtime_capabilities,
            context.creative_tradeoffs,
        )
    )
    confidence = 0.42 + min(evidence_count, 10) * 0.045
    if readiness_score >= 68:
        confidence += 0.05
    confidence -= min(len(missing_information), 5) * 0.035
    confidence -= min(len(quality_risks), 5) * 0.025
    return round(max(0.25, min(0.92, confidence)), 2)


def _evidence(
    context: _QualityContext,
    *,
    readiness_score: int,
) -> tuple[str, ...]:
    evidence = [f"Readiness score: {readiness_score}/100."]
    if context.creative_intent is not None:
        evidence.append(
            f"Intent gaps: {len(context.creative_intent.unresolved_intent_gaps)}."
        )
    if context.creative_hierarchy is not None:
        evidence.append(
            "Hierarchy confidence: "
            f"{context.creative_hierarchy.hierarchy_confidence:.2f}."
        )
    if context.creative_strategy is not None:
        evidence.append(
            f"Strategy confidence: {context.creative_strategy.confidence:.2f}."
        )
    if context.creative_techniques is not None:
        evidence.append(
            f"Technique compatibility: {context.creative_techniques.compatibility}."
        )
    if context.creative_constraints is not None:
        evidence.append(
            "Constraint pressures: "
            f"complexity {context.creative_constraints.complexity_pressure}, "
            f"performance {context.creative_constraints.performance_pressure}."
        )
    if context.creative_constraint_priorities is not None:
        evidence.append(
            "Constraint priority conflicts: "
            f"{len(context.creative_constraint_priorities.conflict_relationships)}."
        )
    top = _top_runtime(context.runtime_capabilities)
    if top is not None:
        evidence.append(f"Top runtime candidate: {top.runtime} ({top.suitability}).")
    if context.creative_tradeoffs is not None:
        evidence.append(
            f"Trade-off HITL advisable: {context.creative_tradeoffs.hitl_advisable}."
        )
    return tuple(evidence[:12])


def _signal(
    dimension: CreativeQualityDimension,
    score: int,
    summary: str,
    evidence: list[str],
) -> CreativeQualitySignal:
    return CreativeQualitySignal(
        dimension=dimension,
        score=max(0, min(10, score)),
        summary=summary,
        evidence=tuple(_dedupe(evidence))[:6],
    )


def _score_for(
    signals: tuple[CreativeQualitySignal, ...],
    dimension: CreativeQualityDimension,
) -> int:
    for signal in signals:
        if signal.dimension == dimension:
            return signal.score
    return 0


def _top_runtime(
    profile: RuntimeCapabilityProfile | None,
) -> RuntimeCapabilityCandidate | None:
    if profile is None or not profile.candidate_runtimes:
        return None
    return profile.candidate_runtimes[0]


def _intent_active(context: _QualityContext, name: str) -> bool:
    if context.creative_intent is None:
        return False
    for dimension in context.creative_intent.atomic_dimensions:
        if dimension.name == name and dimension.explicitness != "absent":
            return True
    return False


def _protected_category(context: _QualityContext, category: str) -> bool:
    priorities = context.creative_constraint_priorities
    if priorities is None:
        return False
    protected = (
        *priorities.non_negotiable_constraints,
        *priorities.high_priority_constraints,
    )
    return any(item.category == category for item in protected)


def _sacrificial_category(context: _QualityContext, category: str) -> bool:
    priorities = context.creative_constraint_priorities
    if priorities is None:
        return False
    return any(item.category == category for item in priorities.sacrificial_constraints)


def _has_any(context: _QualityContext, tokens: set[str]) -> bool:
    return bool(context.tokens.intersection(tokens))


def _has_geometry(context: _QualityContext) -> bool:
    if context.tokens.intersection(_GEOMETRY_TOKENS):
        return True
    translation = context.creative_translation
    return bool(translation and translation.geometric_references)


def _has_light_color(context: _QualityContext) -> bool:
    if context.tokens.intersection(_LIGHT_COLOR_TOKENS):
        return True
    translation = context.creative_translation
    return bool(translation and translation.color_material_direction)


def _visual_plan(context: _QualityContext) -> bool:
    plan = context.creative_plan
    if plan is None:
        return True
    return plan.output_modality.value in {"visual", "audiovisual"}


def _ambiguity_count(context: _QualityContext) -> int:
    return len(context.tokens.intersection(_AMBIGUITY_TOKENS))


def _has_conflict_pressure(context: _QualityContext) -> bool:
    return len(context.tokens.intersection(_CONFLICT_TOKENS)) >= 3


def _failure_mode_for(dimension: CreativeQualityDimension) -> str:
    return {
        "intent_clarity": (
            "Generated output may feel generic because intent is underspecified."
        ),
        "symbolic_coherence": (
            "Symbolic claims may appear as labels instead of visible form."
        ),
        "narrative_coherence": (
            "Any implied story may be absent or visually incoherent."
        ),
        "emotional_coherence": (
            "Mood may default to generic spectacle rather than a clear tone."
        ),
        "geometric_formal_clarity": "Composition may lack a readable formal structure.",
        "technique_suitability": (
            "Technique may dominate the concept instead of serving it."
        ),
        "runtime_suitability": (
            "Runtime limitations may force unsupported or brittle output."
        ),
        "tradeoff_balance": (
            "Unresolved trade-offs may produce contradictory implementation choices."
        ),
        "constraint_alignment": (
            "Protected constraints may be weakened without explanation."
        ),
        "implementation_feasibility": "Scope may exceed the bounded generation budget.",
        "previewability": (
            "The output may be hard to inspect in the current preview path."
        ),
        "performance_risk": (
            "Performance pressure may cause dropped detail or sluggish motion."
        ),
        "originality_potential": "The result may resemble a stock generative effect.",
        "aesthetic_coherence_potential": (
            "Palette, form, and mood may not cohere visually."
        ),
    }[dimension]


def _improvement_for(dimension: CreativeQualityDimension) -> str:
    return {
        "intent_clarity": (
            "Name the subject, motif, output modality, and desired experience."
        ),
        "symbolic_coherence": (
            "Tie symbols to visible geometry, behavior, or transformation."
        ),
        "narrative_coherence": (
            "Define whether a story arc or state change should exist."
        ),
        "emotional_coherence": "Choose one dominant emotional tone before generation.",
        "geometric_formal_clarity": (
            "Specify composition, geometry, symmetry, or spatial rules."
        ),
        "technique_suitability": (
            "Simplify or realign the technique with the selected strategy."
        ),
        "runtime_suitability": (
            "Keep runtime assumptions bounded and explicitly inspectable."
        ),
        "tradeoff_balance": "State what can be sacrificed if constraints conflict.",
        "constraint_alignment": "Confirm non-negotiable and relaxable constraints.",
        "implementation_feasibility": (
            "Reduce scope, candidate count, or effect density."
        ),
        "previewability": "Prefer a previewable version before non-previewable polish.",
        "performance_risk": "Cap density, animation cost, and interaction complexity.",
        "originality_potential": "Add a distinctive rule, metaphor, or transformation.",
        "aesthetic_coherence_potential": (
            "Define palette, contrast, material, and focal hierarchy."
        ),
    }[dimension]


def _question_for_missing_information(item: str) -> str:
    lowered = item.lower()
    if "palette" in lowered or "lighting" in lowered or "material" in lowered:
        return "What palette, lighting, or material quality should lead the piece?"
    if "symbolic" in lowered or "geometric" in lowered:
        return "What visible geometry or form should carry the symbolic fidelity?"
    if "interaction" in lowered or "state-change" in lowered:
        return "What should visibly change when interaction occurs?"
    if "emotional" in lowered:
        return "What emotional tone should the generated piece prioritize?"
    if "subject" in lowered or "motif" in lowered:
        return "What subject or motif should anchor the generated piece?"
    return f"Should this be clarified before generation: {item}"


def _score_sensitive_quality_risk(
    quality_risks: tuple[str, ...],
    needle: str,
) -> bool:
    return any(needle in item.lower() for item in quality_risks)


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def _dedupe(values: list[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    for value in values:
        cleaned = " ".join(str(value).split())
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return tuple(deduped)
