"""Bounded Creative Constraint Prioritizer for V3 workflows."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration.creative_constraints import (
    ConstraintSeverity,
    CreativeConstraintSolution,
)
from creative_coding_assistant.orchestration.creative_hierarchy import (
    CreativeHierarchyPlan,
    HierarchyDimension,
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
    TradeoffSeverity,
)
from creative_coding_assistant.orchestration.creative_translation import (
    CreativeOutputModality,
    CreativeTranslation,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityProfile,
)

ConstraintPriorityCategory = Literal[
    "symbolic_fidelity",
    "narrative_fidelity",
    "emotional_fidelity",
    "geometric_fidelity",
    "visual_quality",
    "motion_quality",
    "audio_quality",
    "runtime_safety",
    "previewability",
    "performance",
    "implementation_simplicity",
    "cost_sensitivity",
    "interaction_complexity",
    "maintainability",
]
ConstraintPriorityLevel = Literal[
    "non_negotiable",
    "high_priority",
    "flexible",
    "relaxable",
    "sacrificial",
]
ConstraintPrioritySource = Literal[
    "explicit",
    "hierarchy",
    "solver",
    "runtime",
    "tradeoff",
    "coherence",
]

PRIORITIZER_AUTHORITY_BOUNDARY = (
    "The Creative Constraint Prioritizer ranks and negotiates constraint "
    "importance for inspection only; it does not auto-select runtimes, route "
    "providers or models, change preview behavior, run repair loops, or make "
    "final creative decisions."
)


class CreativeConstraintPriority(BaseModel):
    """One ranked constraint category and its negotiation posture."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    category: ConstraintPriorityCategory
    priority_level: ConstraintPriorityLevel
    rank: int = Field(ge=1, le=14)
    priority_score: int = Field(ge=0, le=14)
    source: ConstraintPrioritySource
    rationale: str = Field(min_length=1, max_length=300)
    negotiation_guidance: str = Field(min_length=1, max_length=300)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=6)


class CreativeConstraintPriorityConflict(BaseModel):
    """A priority conflict between two constraint categories."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    protected_category: ConstraintPriorityCategory
    competing_category: ConstraintPriorityCategory
    severity: ConstraintSeverity
    summary: str = Field(min_length=1, max_length=280)
    negotiation_note: str = Field(min_length=1, max_length=300)
    hitl_recommended: bool = False


class CreativeConstraintPrioritization(BaseModel):
    """Inspectable constraint-priority metadata derived after constraint solving."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_constraint_prioritizer"] = "creative_constraint_prioritizer"
    non_negotiable_constraints: tuple[CreativeConstraintPriority, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    high_priority_constraints: tuple[CreativeConstraintPriority, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    flexible_constraints: tuple[CreativeConstraintPriority, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    relaxable_constraints: tuple[CreativeConstraintPriority, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    sacrificial_constraints: tuple[CreativeConstraintPriority, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    priority_rationale: tuple[str, ...] = Field(min_length=1, max_length=8)
    negotiation_notes: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    conflict_relationships: tuple[CreativeConstraintPriorityConflict, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=PRIORITIZER_AUTHORITY_BOUNDARY,
        max_length=460,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=10)


def derive_creative_constraint_priorities(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
) -> CreativeConstraintPrioritization:
    """Rank constraint importance without changing execution behavior."""

    scores = _initial_scores()
    text = " ".join(request.query.lower().split())
    _score_text(scores, text)
    _score_intent(scores, creative_intent)
    _score_translation(scores, creative_translation)
    _score_hierarchy(scores, creative_hierarchy)
    _score_solver(scores, creative_constraints)
    _score_plan(scores, creative_plan)
    _score_strategy(scores, creative_strategy)
    _score_techniques(scores, creative_techniques)
    _score_runtime(scores, runtime_capabilities)
    _score_tradeoffs(scores, creative_tradeoffs)
    priorities = _rank_priorities(scores)
    conflicts = _conflicts(
        priorities=priorities,
        creative_constraints=creative_constraints,
        creative_hierarchy=creative_hierarchy,
        creative_tradeoffs=creative_tradeoffs,
    )
    return CreativeConstraintPrioritization(
        non_negotiable_constraints=_priorities_for(
            priorities,
            "non_negotiable",
        ),
        high_priority_constraints=_priorities_for(priorities, "high_priority"),
        flexible_constraints=_priorities_for(priorities, "flexible"),
        relaxable_constraints=_priorities_for(priorities, "relaxable"),
        sacrificial_constraints=_priorities_for(priorities, "sacrificial"),
        priority_rationale=_priority_rationale(priorities),
        negotiation_notes=_negotiation_notes(priorities, conflicts),
        conflict_relationships=conflicts,
        hitl_questions=_hitl_questions(
            priorities,
            conflicts,
            creative_hierarchy,
            creative_constraints,
            creative_tradeoffs,
        ),
        prompt_guidance=_prompt_guidance(priorities, conflicts),
        evidence=_evidence(
            request=request,
            route_decision=route_decision,
            creative_hierarchy=creative_hierarchy,
            creative_constraints=creative_constraints,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=creative_tradeoffs,
        ),
    )


def creative_constraint_priorities_prompt_lines(
    prioritization: CreativeConstraintPrioritization,
) -> tuple[str, ...]:
    """Render prioritization metadata as compact provider prompt guidance."""

    lines = [
        f"Authority boundary: {prioritization.authority_boundary}",
    ]
    lines.extend(
        f"Non-negotiable constraint: {item.category}; {item.negotiation_guidance}"
        for item in prioritization.non_negotiable_constraints
    )
    lines.extend(
        f"High-priority constraint: {item.category}; {item.negotiation_guidance}"
        for item in prioritization.high_priority_constraints[:4]
    )
    lines.extend(
        f"Flexible constraint: {item.category}; {item.negotiation_guidance}"
        for item in prioritization.flexible_constraints[:4]
    )
    lines.extend(
        f"Relaxable constraint: {item.category}; {item.negotiation_guidance}"
        for item in prioritization.relaxable_constraints[:4]
    )
    lines.extend(
        f"Sacrificial constraint: {item.category}; {item.negotiation_guidance}"
        for item in prioritization.sacrificial_constraints[:4]
    )
    for conflict in prioritization.conflict_relationships[:4]:
        lines.append(
            "Constraint priority conflict: "
            f"{conflict.protected_category} vs {conflict.competing_category}; "
            f"{conflict.negotiation_note}"
        )
    lines.extend(
        f"HITL constraint question: {item}" for item in prioritization.hitl_questions
    )
    lines.extend(
        f"Constraint priority guidance: {item}"
        for item in prioritization.prompt_guidance
    )
    return tuple(lines[:28])


@dataclass
class _PriorityScore:
    category: ConstraintPriorityCategory
    score: int = 0
    evidence: list[str] = field(default_factory=list)
    source_weights: dict[ConstraintPrioritySource, int] = field(default_factory=dict)
    protected: bool = False
    relaxable_hint: bool = False
    sacrificial_hint: bool = False


def _initial_scores() -> dict[ConstraintPriorityCategory, _PriorityScore]:
    return {category: _PriorityScore(category) for category in _CATEGORY_ORDER}


def _bump(
    scores: dict[ConstraintPriorityCategory, _PriorityScore],
    category: ConstraintPriorityCategory,
    amount: int,
    source: ConstraintPrioritySource,
    evidence: str,
    *,
    protected: bool = False,
    relaxable: bool = False,
    sacrificial: bool = False,
) -> None:
    item = scores[category]
    item.score = min(14, item.score + amount)
    item.evidence.append(evidence)
    item.source_weights[source] = item.source_weights.get(source, 0) + amount
    item.protected = item.protected or protected
    item.relaxable_hint = item.relaxable_hint or relaxable
    item.sacrificial_hint = item.sacrificial_hint or sacrificial


def _score_text(
    scores: dict[ConstraintPriorityCategory, _PriorityScore],
    text: str,
) -> None:
    for category, pattern in _CATEGORY_PATTERNS.items():
        if not pattern.search(text):
            continue
        explicit_emphasis = _has_nearby_hint(
            text,
            pattern,
            _NON_NEGOTIABLE_PATTERN,
            window=32,
        )
        relaxable = _has_nearby_hint(
            text,
            pattern,
            _RELAXABLE_PATTERN,
            window=40,
        )
        sacrificial = _has_nearby_hint(
            text,
            pattern,
            _SACRIFICIAL_PATTERN,
            window=24,
        )
        _bump(
            scores,
            category,
            4,
            "explicit",
            f"keyword:{category}",
            protected=explicit_emphasis,
            relaxable=relaxable,
            sacrificial=sacrificial,
        )
        if explicit_emphasis:
            _bump(
                scores,
                category,
                2,
                "explicit",
                f"emphasis:{category}",
                protected=True,
            )


def _score_intent(
    scores: dict[ConstraintPriorityCategory, _PriorityScore],
    creative_intent: CreativeIntentDecomposition | None,
) -> None:
    if creative_intent is None:
        return
    for dimension in creative_intent.atomic_dimensions:
        if dimension.explicitness == "absent":
            continue
        category = _INTENT_CATEGORY[dimension.name]
        amount = 3 if dimension.explicitness == "explicit" else 2
        _bump(
            scores,
            category,
            amount,
            "coherence",
            f"intent:{dimension.name}",
        )


def _score_translation(
    scores: dict[ConstraintPriorityCategory, _PriorityScore],
    translation: CreativeTranslation | None,
) -> None:
    if translation is None:
        return
    if translation.output_modality is CreativeOutputModality.AUDIO:
        _bump(scores, "audio_quality", 3, "coherence", "audio modality")
    elif translation.output_modality is CreativeOutputModality.AUDIOVISUAL:
        _bump(scores, "audio_quality", 2, "coherence", "audiovisual modality")
        _bump(scores, "visual_quality", 2, "coherence", "audiovisual modality")
    if translation.visual_style or translation.shader_presets:
        _bump(scores, "visual_quality", 2, "coherence", "visual style")
    if translation.generation_constraints:
        _bump(
            scores,
            "runtime_safety",
            2,
            "coherence",
            "generation constraints",
        )


def _score_hierarchy(
    scores: dict[ConstraintPriorityCategory, _PriorityScore],
    hierarchy: CreativeHierarchyPlan | None,
) -> None:
    if hierarchy is None:
        return
    non_negotiable = {
        _HIERARCHY_CATEGORY[dimension]
        for dimension in hierarchy.non_negotiable_dimensions
    }
    for priority in hierarchy.primary_creative_priorities:
        category = _HIERARCHY_CATEGORY[priority.dimension]
        _bump(
            scores,
            category,
            4,
            "hierarchy",
            f"hierarchy primary:{priority.dimension}",
            protected=category in non_negotiable,
        )
    for priority in hierarchy.secondary_creative_priorities:
        _bump(
            scores,
            _HIERARCHY_CATEGORY[priority.dimension],
            2,
            "hierarchy",
            f"hierarchy secondary:{priority.dimension}",
        )
    for dimension in hierarchy.flexible_dimensions:
        _bump(
            scores,
            _HIERARCHY_CATEGORY[dimension],
            1,
            "hierarchy",
            f"hierarchy flexible:{dimension}",
            relaxable=True,
        )


def _score_solver(
    scores: dict[ConstraintPriorityCategory, _PriorityScore],
    constraints: CreativeConstraintSolution | None,
) -> None:
    if constraints is None:
        return
    if constraints.safety_pressure == "high":
        _bump(
            scores,
            "runtime_safety",
            5,
            "solver",
            "solver safety high",
            protected=True,
        )
    elif constraints.safety_pressure == "medium":
        _bump(scores, "runtime_safety", 3, "solver", "solver safety medium")
    if constraints.performance_pressure == "high":
        _bump(
            scores,
            "performance",
            5,
            "solver",
            "solver performance high",
            protected=True,
        )
    elif constraints.performance_pressure == "medium":
        _bump(scores, "performance", 3, "solver", "solver performance medium")
    if constraints.complexity_pressure == "high":
        _bump(scores, "implementation_simplicity", 3, "solver", "complexity high")
        _bump(scores, "maintainability", 2, "solver", "complexity high")
    if constraints.cost_pressure == "high":
        _bump(scores, "cost_sensitivity", 4, "solver", "cost high")
    elif constraints.cost_pressure == "medium":
        _bump(scores, "cost_sensitivity", 2, "solver", "cost medium")
    if constraints.runtime_fit == "code_only":
        _bump(scores, "previewability", 4, "solver", "code-only runtime fit")
    for constraint in constraints.active_constraints:
        category = _SOLVER_AXIS_CATEGORY.get(constraint.axis)
        if category is None:
            continue
        _bump(
            scores,
            category,
            _severity_weight(constraint.severity),
            "solver",
            f"solver:{constraint.axis}:{constraint.severity}",
            protected=constraint.severity == "blocking",
        )
    for conflict in constraints.conflicts:
        _score_conflict_text(scores, conflict, "solver")


def _score_plan(
    scores: dict[ConstraintPriorityCategory, _PriorityScore],
    plan: CreativeExecutionPlan | None,
) -> None:
    if plan is None:
        return
    if plan.runtime_available:
        _bump(scores, "previewability", 2, "coherence", "runtime available")
    else:
        _bump(scores, "previewability", 3, "coherence", "runtime unavailable")
    if plan.expected_complexity == "high":
        _bump(
            scores,
            "implementation_simplicity",
            3,
            "coherence",
            "plan high complexity",
        )
        _bump(scores, "maintainability", 2, "coherence", "plan high complexity")
    if plan.estimated_token_cost >= 6200:
        _bump(scores, "cost_sensitivity", 3, "coherence", "plan high token cost")


def _score_strategy(
    scores: dict[ConstraintPriorityCategory, _PriorityScore],
    strategy: CreativeStrategyProfile | None,
) -> None:
    if strategy is None:
        return
    if strategy.primary_strategy == "sacred_geometry":
        _bump(scores, "symbolic_fidelity", 2, "coherence", "strategy sacred geometry")
        _bump(scores, "geometric_fidelity", 2, "coherence", "strategy sacred geometry")
    elif strategy.primary_strategy in {"particle_cosmology", "field_dynamics"}:
        evidence = f"strategy {strategy.primary_strategy}"
        _bump(scores, "motion_quality", 2, "coherence", evidence)
        _bump(scores, "visual_quality", 2, "coherence", evidence)
    elif strategy.primary_strategy == "minimal_generative_systems":
        _bump(scores, "implementation_simplicity", 2, "coherence", "strategy minimal")


def _score_techniques(
    scores: dict[ConstraintPriorityCategory, _PriorityScore],
    techniques: CreativeTechniqueProfile | None,
) -> None:
    if techniques is None:
        return
    if techniques.primary_technique == "audio_reactive_mappings":
        _bump(scores, "audio_quality", 3, "coherence", "audio reactive technique")
    if techniques.primary_technique in {"particle_systems", "feedback_systems"}:
        _bump(
            scores,
            "motion_quality",
            2,
            "coherence",
            f"technique {techniques.primary_technique}",
        )
    if techniques.performance_pressure == "high":
        _bump(scores, "performance", 3, "coherence", "technique performance high")
    if techniques.complexity_pressure == "high":
        _bump(
            scores,
            "implementation_simplicity",
            2,
            "coherence",
            "technique complexity high",
        )
        _bump(scores, "maintainability", 2, "coherence", "technique complexity high")


def _score_runtime(
    scores: dict[ConstraintPriorityCategory, _PriorityScore],
    runtime: RuntimeCapabilityProfile | None,
) -> None:
    if runtime is None:
        return
    for candidate in runtime.candidate_runtimes[:2]:
        if candidate.performance_pressure == "high":
            _bump(
                scores,
                "performance",
                2,
                "runtime",
                f"{candidate.runtime} performance high",
            )
        if candidate.preview_support == "code_only":
            _bump(
                scores,
                "previewability",
                2,
                "runtime",
                f"{candidate.runtime} code-only",
            )
        if candidate.implementation_complexity == "high":
            _bump(
                scores,
                "maintainability",
                2,
                "runtime",
                f"{candidate.runtime} complexity high",
            )
    if runtime.hitl_advisable:
        _bump(scores, "runtime_safety", 2, "runtime", "runtime HITL advisable")


def _score_tradeoffs(
    scores: dict[ConstraintPriorityCategory, _PriorityScore],
    tradeoffs: CreativeTradeoffProfile | None,
) -> None:
    if tradeoffs is None:
        return
    for tradeoff in tradeoffs.primary_tradeoffs[:4]:
        for axis in (tradeoff.source_axis, tradeoff.target_axis):
            category = _TRADEOFF_AXIS_CATEGORY.get(axis)
            if category is not None:
                _bump(
                    scores,
                    category,
                    _tradeoff_weight(tradeoff.severity),
                    "tradeoff",
                    f"tradeoff:{axis}:{tradeoff.severity}",
                    protected=tradeoff.severity == "blocking",
                )
    for concern in tradeoffs.performance_concerns:
        _bump(scores, "performance", 2, "tradeoff", concern)
    for risk in tradeoffs.complexity_risks:
        _bump(scores, "implementation_simplicity", 1, "tradeoff", risk)
        _bump(scores, "maintainability", 1, "tradeoff", risk)
    for risk in tradeoffs.fidelity_risks:
        _score_conflict_text(scores, risk, "tradeoff")


def _rank_priorities(
    scores: dict[ConstraintPriorityCategory, _PriorityScore],
) -> tuple[CreativeConstraintPriority, ...]:
    active = [
        item
        for item in scores.values()
        if item.score > 0 or item.relaxable_hint or item.sacrificial_hint
    ]
    if not active:
        active = [scores["implementation_simplicity"]]
        active[0].score = 1
        active[0].evidence.append("default bounded implementation path")
        active[0].source_weights["coherence"] = 1
    ranked_scores = sorted(
        active,
        key=lambda item: (item.score, -_CATEGORY_INDEX[item.category]),
        reverse=True,
    )
    levels = {item.category: _priority_level(item) for item in ranked_scores}
    _ensure_sacrificial_level(ranked_scores, levels)
    return tuple(
        _priority(item, level=levels[item.category], rank=index + 1)
        for index, item in enumerate(ranked_scores)
    )


def _priority_level(item: _PriorityScore) -> ConstraintPriorityLevel:
    if item.sacrificial_hint:
        return "sacrificial"
    if item.protected:
        return "non_negotiable"
    if item.relaxable_hint and (
        item.score <= 6 or _dominant_source(item) == "explicit"
    ):
        return "relaxable"
    if item.score >= 7:
        return "high_priority"
    if item.score >= 3:
        return "flexible"
    return "relaxable"


def _ensure_sacrificial_level(
    ranked_scores: list[_PriorityScore],
    levels: dict[ConstraintPriorityCategory, ConstraintPriorityLevel],
) -> None:
    if any(level == "sacrificial" for level in levels.values()):
        return
    for item in reversed(ranked_scores):
        if item.category in _SACRIFICIAL_CANDIDATES and not item.protected:
            if item.score <= 5 or item.relaxable_hint:
                levels[item.category] = "sacrificial"
                return


def _priority(
    score: _PriorityScore,
    *,
    level: ConstraintPriorityLevel,
    rank: int,
) -> CreativeConstraintPriority:
    return CreativeConstraintPriority(
        category=score.category,
        priority_level=level,
        rank=rank,
        priority_score=score.score,
        source=_dominant_source(score),
        rationale=_rationale(score, level),
        negotiation_guidance=_negotiation_guidance(score.category, level),
        evidence=tuple(_dedupe(score.evidence))[:6],
    )


def _dominant_source(score: _PriorityScore) -> ConstraintPrioritySource:
    if not score.source_weights:
        return "coherence"
    return max(
        score.source_weights,
        key=lambda source: (score.source_weights[source], _SOURCE_ORDER[source]),
    )


def _priorities_for(
    priorities: tuple[CreativeConstraintPriority, ...],
    level: ConstraintPriorityLevel,
) -> tuple[CreativeConstraintPriority, ...]:
    return tuple(item for item in priorities if item.priority_level == level)[:6]


def _conflicts(
    *,
    priorities: tuple[CreativeConstraintPriority, ...],
    creative_constraints: CreativeConstraintSolution | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
) -> tuple[CreativeConstraintPriorityConflict, ...]:
    by_category = {item.category: item for item in priorities}
    conflicts: list[CreativeConstraintPriorityConflict] = []
    _append_conflict(conflicts, by_category, "visual_quality", "performance")
    _append_conflict(
        conflicts,
        by_category,
        "symbolic_fidelity",
        "implementation_simplicity",
    )
    _append_conflict(
        conflicts,
        by_category,
        "emotional_fidelity",
        "performance",
    )
    _append_conflict(
        conflicts,
        by_category,
        "interaction_complexity",
        "runtime_safety",
    )
    _append_conflict(conflicts, by_category, "audio_quality", "previewability")
    if creative_constraints is not None:
        for conflict in creative_constraints.conflicts[:2]:
            pair = _conflict_pair_from_text(conflict)
            if pair is not None:
                _append_conflict(conflicts, by_category, pair[0], pair[1])
    if creative_hierarchy is not None:
        for conflict in creative_hierarchy.priority_conflicts[:2]:
            pair = _conflict_pair_from_text(conflict)
            if pair is not None:
                _append_conflict(conflicts, by_category, pair[0], pair[1])
    if creative_tradeoffs is not None:
        for tradeoff in creative_tradeoffs.primary_tradeoffs[:3]:
            source = _TRADEOFF_AXIS_CATEGORY.get(tradeoff.source_axis)
            target = _TRADEOFF_AXIS_CATEGORY.get(tradeoff.target_axis)
            if source is not None and target is not None:
                _append_conflict(
                    conflicts,
                    by_category,
                    source,
                    target,
                    severity=_constraint_severity(tradeoff.severity),
                )
    return tuple(_dedupe_conflicts(conflicts))[:8]


def _append_conflict(
    conflicts: list[CreativeConstraintPriorityConflict],
    priorities: dict[ConstraintPriorityCategory, CreativeConstraintPriority],
    protected: ConstraintPriorityCategory,
    competing: ConstraintPriorityCategory,
    *,
    severity: ConstraintSeverity | None = None,
) -> None:
    left = priorities.get(protected)
    right = priorities.get(competing)
    if left is None or right is None:
        return
    if min(left.priority_score, right.priority_score) < 4:
        return
    if _level_rank(left.priority_level) < _level_rank(right.priority_level):
        left, right = right, left
    conflicts.append(
        CreativeConstraintPriorityConflict(
            protected_category=left.category,
            competing_category=right.category,
            severity=severity or _conflict_severity(left, right),
            summary=(
                f"{left.category} has stronger priority than {right.category}, "
                "but both affect feasibility."
            ),
            negotiation_note=(
                f"Protect {left.category} first; relax {right.category} only "
                "after making the trade-off explicit."
            ),
            hitl_recommended=(
                left.priority_level == "non_negotiable"
                and right.priority_level in {"non_negotiable", "high_priority"}
            ),
        )
    )


def _priority_rationale(
    priorities: tuple[CreativeConstraintPriority, ...],
) -> tuple[str, ...]:
    rationale = [
        "Constraint priorities are ranked from user emphasis, hierarchy, "
        "solver pressure, runtime capability, and trade-off evidence."
    ]
    protected = _priorities_for(priorities, "non_negotiable")
    if protected:
        rationale.append(
            "Non-negotiable constraints: "
            + ", ".join(item.category for item in protected)
            + "."
        )
    sacrificial = _priorities_for(priorities, "sacrificial")
    if sacrificial:
        rationale.append(
            "Sacrificial constraints may be reduced first: "
            + ", ".join(item.category for item in sacrificial)
            + "."
        )
    rationale.extend(item.rationale for item in priorities[:4])
    return tuple(rationale[:8])


def _negotiation_notes(
    priorities: tuple[CreativeConstraintPriority, ...],
    conflicts: tuple[CreativeConstraintPriorityConflict, ...],
) -> tuple[str, ...]:
    notes = [
        item.negotiation_guidance
        for item in priorities
        if item.priority_level in {"non_negotiable", "sacrificial", "relaxable"}
    ]
    notes.extend(item.negotiation_note for item in conflicts[:3])
    return tuple(_dedupe(notes))[:8]


def _hitl_questions(
    priorities: tuple[CreativeConstraintPriority, ...],
    conflicts: tuple[CreativeConstraintPriorityConflict, ...],
    hierarchy: CreativeHierarchyPlan | None,
    constraints: CreativeConstraintSolution | None,
    tradeoffs: CreativeTradeoffProfile | None,
) -> tuple[str, ...]:
    questions: list[str] = []
    questions.extend(
        f"May {item.competing_category} be relaxed to protect "
        f"{item.protected_category}?"
        for item in conflicts
        if item.hitl_recommended
    )
    if hierarchy is not None:
        questions.extend(hierarchy.hitl_questions[:2])
    if constraints is not None and constraints.hitl_advisable:
        questions.append(
            constraints.hitl_reason
            or "Which constraint should take priority before generation?"
        )
    if tradeoffs is not None and tradeoffs.hitl_advisable:
        questions.append(
            tradeoffs.hitl_reason
            or "Which trade-off boundary should be protected first?"
        )
    if not questions and _has_close_top_scores(priorities):
        first, second = priorities[:2]
        questions.append(
            f"Should {first.category} or {second.category} lead if trade-offs tighten?"
        )
    return tuple(_dedupe(questions))[:6]


def _prompt_guidance(
    priorities: tuple[CreativeConstraintPriority, ...],
    conflicts: tuple[CreativeConstraintPriorityConflict, ...],
) -> tuple[str, ...]:
    guidance = [
        "Use constraint priorities as negotiation guidance, not feature additions.",
        "Protect non-negotiable constraints before optimizing flexible ones.",
    ]
    sacrificial = _priorities_for(priorities, "sacrificial")
    if sacrificial:
        guidance.append(
            "Reduce sacrificial constraints first: "
            + ", ".join(item.category for item in sacrificial[:3])
            + "."
        )
    relaxable = _priorities_for(priorities, "relaxable")
    if relaxable:
        guidance.append(
            "Relaxable constraints need explicit explanation before weakening: "
            + ", ".join(item.category for item in relaxable[:3])
            + "."
        )
    if conflicts:
        guidance.append("State constraint priority conflicts before expanding scope.")
    guidance.append(
        "Do not treat constraint priority as runtime, provider, or model selection."
    )
    return tuple(guidance[:8])


def _evidence(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
) -> tuple[str, ...]:
    evidence = [f"Mode: {request.mode.value}."]
    if route_decision is not None:
        evidence.append(f"Route selected: {route_decision.route.value}.")
    if creative_hierarchy is not None:
        evidence.append(
            "Hierarchy non-negotiables: "
            + ", ".join(creative_hierarchy.non_negotiable_dimensions)
            + "."
        )
    if creative_constraints is not None:
        evidence.append(
            f"Solver pressures: complexity {creative_constraints.complexity_pressure}, "
            f"performance {creative_constraints.performance_pressure}, "
            f"safety {creative_constraints.safety_pressure}, "
            f"cost {creative_constraints.cost_pressure}."
        )
    if runtime_capabilities is not None:
        evidence.append(
            "Runtime candidates: "
            + ", ".join(runtime_capabilities.likely_candidates)
            + "."
        )
    if creative_tradeoffs is not None:
        evidence.append(
            f"Trade-off count: {len(creative_tradeoffs.primary_tradeoffs)}."
        )
    return tuple(_dedupe(evidence))[:10]


def _rationale(
    score: _PriorityScore,
    level: ConstraintPriorityLevel,
) -> str:
    label = score.category.replace("_", " ")
    return (
        f"{label} is {level.replace('_', ' ')} because score "
        f"{score.score} is supported by {_dominant_source(score)} evidence."
    )


def _negotiation_guidance(
    category: ConstraintPriorityCategory,
    level: ConstraintPriorityLevel,
) -> str:
    label = category.replace("_", " ")
    if level == "non_negotiable":
        return f"Do not relax {label} without explicit user confirmation."
    if level == "high_priority":
        return f"Protect {label} unless a non-negotiable constraint conflicts."
    if level == "flexible":
        return f"Adjust {label} when needed to keep the output feasible."
    if level == "relaxable":
        return f"Relax {label} before weakening higher-priority constraints."
    return f"Sacrifice {label} first if feasibility or safety requires reduction."


def _score_conflict_text(
    scores: dict[ConstraintPriorityCategory, _PriorityScore],
    text: str,
    source: ConstraintPrioritySource,
) -> None:
    normalized = text.lower()
    for category, pattern in _CATEGORY_PATTERNS.items():
        if pattern.search(normalized):
            _bump(scores, category, 2, source, f"conflict:{category}")


def _has_nearby_hint(
    text: str,
    category_pattern: re.Pattern[str],
    hint_pattern: re.Pattern[str],
    *,
    window: int = 48,
) -> bool:
    for match in category_pattern.finditer(text):
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        if hint_pattern.search(text[start:end]):
            return True
    return False


def _priority_text(
    request: AssistantRequest,
    creative_translation: CreativeTranslation | None,
    creative_intent: CreativeIntentDecomposition | None,
) -> str:
    parts = [request.query]
    if creative_translation is not None:
        parts.extend(
            [
                creative_translation.creative_intent,
                " ".join(creative_translation.generation_constraints),
            ]
        )
    if creative_intent is not None:
        parts.extend(
            [
                creative_intent.primary_expression,
                creative_intent.experiential_goal,
            ]
        )
    return " ".join(" ".join(parts).lower().split())


def _severity_weight(severity: ConstraintSeverity) -> int:
    return {"info": 1, "watch": 2, "risk": 3, "blocking": 5}[severity]


def _tradeoff_weight(severity: TradeoffSeverity) -> int:
    return {"info": 1, "watch": 2, "risk": 3, "blocking": 5}[severity]


def _constraint_severity(severity: TradeoffSeverity) -> ConstraintSeverity:
    return severity


def _conflict_severity(
    left: CreativeConstraintPriority,
    right: CreativeConstraintPriority,
) -> ConstraintSeverity:
    if left.priority_level == "non_negotiable" and right.priority_level in {
        "non_negotiable",
        "high_priority",
    }:
        return "risk"
    if min(left.priority_score, right.priority_score) >= 7:
        return "watch"
    return "info"


def _level_rank(level: ConstraintPriorityLevel) -> int:
    return {
        "sacrificial": 0,
        "relaxable": 1,
        "flexible": 2,
        "high_priority": 3,
        "non_negotiable": 4,
    }[level]


def _has_close_top_scores(
    priorities: tuple[CreativeConstraintPriority, ...],
) -> bool:
    return (
        len(priorities) > 1
        and priorities[0].priority_score == priorities[1].priority_score
    )


def _conflict_pair_from_text(
    text: str,
) -> tuple[ConstraintPriorityCategory, ConstraintPriorityCategory] | None:
    normalized = text.lower()
    matched = [
        category
        for category, pattern in _CATEGORY_PATTERNS.items()
        if pattern.search(normalized)
    ]
    if len(matched) >= 2:
        return matched[0], matched[1]
    if "visual" in normalized and "performance" in normalized:
        return "visual_quality", "performance"
    if "complexity" in normalized and "performance" in normalized:
        return "implementation_simplicity", "performance"
    if "audio" in normalized and "visual" in normalized:
        return "audio_quality", "visual_quality"
    return None


def _dedupe(values: Sequence[object]) -> tuple:
    deduped: list[object] = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return tuple(deduped)


def _dedupe_conflicts(
    conflicts: Sequence[CreativeConstraintPriorityConflict],
) -> tuple[CreativeConstraintPriorityConflict, ...]:
    deduped: list[CreativeConstraintPriorityConflict] = []
    seen: set[tuple[str, str]] = set()
    for conflict in conflicts:
        key = (conflict.protected_category, conflict.competing_category)
        if key not in seen:
            seen.add(key)
            deduped.append(conflict)
    return tuple(deduped)


_CATEGORY_ORDER: tuple[ConstraintPriorityCategory, ...] = (
    "symbolic_fidelity",
    "narrative_fidelity",
    "emotional_fidelity",
    "geometric_fidelity",
    "visual_quality",
    "motion_quality",
    "audio_quality",
    "runtime_safety",
    "previewability",
    "performance",
    "implementation_simplicity",
    "cost_sensitivity",
    "interaction_complexity",
    "maintainability",
)
_CATEGORY_INDEX = {category: index for index, category in enumerate(_CATEGORY_ORDER)}
_SOURCE_ORDER: dict[ConstraintPrioritySource, int] = {
    "explicit": 5,
    "hierarchy": 4,
    "solver": 3,
    "runtime": 2,
    "tradeoff": 1,
    "coherence": 0,
}
_SACRIFICIAL_CANDIDATES: set[ConstraintPriorityCategory] = {
    "interaction_complexity",
    "cost_sensitivity",
    "implementation_simplicity",
    "maintainability",
    "previewability",
}
_CATEGORY_PATTERNS: dict[ConstraintPriorityCategory, re.Pattern[str]] = {
    "symbolic_fidelity": re.compile(
        r"\b(?:symbol|symbolic|ritual|myth|archetype|sacred|initiatic)\b"
    ),
    "narrative_fidelity": re.compile(
        r"\b(?:story|narrative|journey|arc|transformation|rebirth|"
        r"death-and-rebirth)\b"
    ),
    "emotional_fidelity": re.compile(
        r"\b(?:emotion|emotional|feeling|mood|awe|grief|calm|tension|"
        r"catharsis)\b"
    ),
    "geometric_fidelity": re.compile(
        r"\b(?:geometry|geometric|mandala|spiral|fractal|grid)\b"
    ),
    "visual_quality": re.compile(
        r"\b(?:visual|cinematic|luminous|glow|palette|spectacle|beautiful|"
        r"polished)\b"
    ),
    "motion_quality": re.compile(
        r"\b(?:motion|movement|pulse|morph|animation|flow|drift)\b"
    ),
    "audio_quality": re.compile(r"\b(?:audio|sound|music|beat|sonic|synth)\b"),
    "runtime_safety": re.compile(
        r"\b(?:safe|browser-safe|accessible|no autoplay|guardrail)\b"
    ),
    "previewability": re.compile(
        r"\b(?:preview|live preview|workstation|renderable)\b"
    ),
    "performance": re.compile(r"\b(?:performance|60\s?fps|fps|mobile|realtime|fast)\b"),
    "implementation_simplicity": re.compile(
        r"\b(?:simple|minimal|lightweight|small|straightforward)\b"
    ),
    "cost_sensitivity": re.compile(r"\b(?:cost|token|cheap|budget|efficient)\b"),
    "interaction_complexity": re.compile(
        r"\b(?:interaction|interactive|mouse|touch|controls|input)\b"
    ),
    "maintainability": re.compile(
        r"\b(?:maintainable|readable|clean|modular|debuggable)\b"
    ),
}
_NON_NEGOTIABLE_PATTERN = re.compile(
    r"\b(?:must|non-negotiable|required|highest priority|most important|"
    r"prioriti[sz]e)\b"
)
_RELAXABLE_PATTERN = re.compile(
    r"\b(?:optional|flexible|secondary|if needed|can relax|can simplify)\b"
)
_SACRIFICIAL_PATTERN = re.compile(
    r"\b(?:sacrifice|sacrificed|drop|dropped|omit|remove|can skip|nice to have)\b"
)
_INTENT_CATEGORY = {
    "narrative": "narrative_fidelity",
    "symbolic": "symbolic_fidelity",
    "emotional": "emotional_fidelity",
    "geometric": "geometric_fidelity",
    "motion": "motion_quality",
    "rhythm": "motion_quality",
    "light_color": "visual_quality",
    "audio": "audio_quality",
    "interaction": "interaction_complexity",
    "climax_transformation": "narrative_fidelity",
}
_HIERARCHY_CATEGORY: dict[HierarchyDimension, ConstraintPriorityCategory] = {
    "symbolism": "symbolic_fidelity",
    "narrative": "narrative_fidelity",
    "emotion": "emotional_fidelity",
    "geometry": "geometric_fidelity",
    "motion": "motion_quality",
    "rhythm": "motion_quality",
    "light_color": "visual_quality",
    "audio": "audio_quality",
    "interaction": "interaction_complexity",
    "visual_impact": "visual_quality",
    "performance": "performance",
    "simplicity": "implementation_simplicity",
    "complexity": "maintainability",
    "runtime_safety": "runtime_safety",
    "experiential_depth": "emotional_fidelity",
}
_SOLVER_AXIS_CATEGORY = {
    "intent": "symbolic_fidelity",
    "modality": "visual_quality",
    "runtime": "runtime_safety",
    "safety": "runtime_safety",
    "performance": "performance",
    "complexity": "implementation_simplicity",
    "cost": "cost_sensitivity",
    "hitl": "runtime_safety",
    "output_goal": "visual_quality",
}
_TRADEOFF_AXIS_CATEGORY = {
    "creative_expressiveness": "visual_quality",
    "concept_fidelity": "symbolic_fidelity",
    "implementation_complexity": "implementation_simplicity",
    "performance": "performance",
    "runtime_support": "runtime_safety",
    "previewability": "previewability",
    "cost_sensitivity": "cost_sensitivity",
    "safety": "runtime_safety",
    "maintainability": "maintainability",
    "hitl": "runtime_safety",
}
