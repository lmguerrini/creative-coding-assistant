"""Bounded Creative Trade-off Explorer for Creative Intelligence workflows."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration.creative_constraints import (
    CreativeConstraintSolution,
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
from creative_coding_assistant.orchestration.creative_translation import (
    CreativeTranslation,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityCandidate,
    RuntimeCapabilityProfile,
)

TradeoffAxis = Literal[
    "creative_expressiveness",
    "concept_fidelity",
    "implementation_complexity",
    "performance",
    "runtime_support",
    "previewability",
    "cost_sensitivity",
    "safety",
    "maintainability",
    "hitl",
]
TradeoffSeverity = Literal["info", "watch", "risk", "blocking"]
TradeoffPressure = Literal["low", "medium", "high"]

TRADEOFF_EXPLORER_AUTHORITY_BOUNDARY = (
    "The Creative Trade-off Explorer structures consequences and discussion "
    "points only; it does not select final artifacts, auto-select runtimes, "
    "route providers or models, create execution profiles, change preview "
    "behavior, choose renderers, run runtime repair, or replace the Creative "
    "Director."
)


class CreativeTradeoff(BaseModel):
    """One bounded trade-off between creative and technical directions."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source_axis: TradeoffAxis
    target_axis: TradeoffAxis
    severity: TradeoffSeverity
    summary: str = Field(min_length=1, max_length=280)
    creative_benefit: str = Field(min_length=1, max_length=280)
    technical_cost: str = Field(min_length=1, max_length=280)
    runtime_implication: str = Field(min_length=1, max_length=280)
    mitigation: str = Field(min_length=1, max_length=280)
    director_discussion_point: str = Field(min_length=1, max_length=280)
    hitl_recommended: bool = False
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=6)


class CreativeTradeoffProfile(BaseModel):
    """Inspectable trade-off metadata derived before provider generation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_tradeoff_explorer"] = "creative_tradeoff_explorer"
    output_goal: str = Field(min_length=1, max_length=360)
    primary_tradeoffs: tuple[CreativeTradeoff, ...] = Field(
        min_length=1,
        max_length=8,
    )
    creative_benefits: tuple[str, ...] = Field(min_length=1, max_length=8)
    technical_costs: tuple[str, ...] = Field(min_length=1, max_length=8)
    runtime_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    performance_concerns: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    complexity_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    fidelity_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    cost_sensitivity: TradeoffPressure
    safety_concerns: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    maintainability_concerns: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    hitl_advisable: bool = False
    hitl_reason: str | None = Field(default=None, max_length=280)
    director_discussion_points: tuple[str, ...] = Field(min_length=1, max_length=8)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=TRADEOFF_EXPLORER_AUTHORITY_BOUNDARY,
        max_length=480,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=10)


def derive_creative_tradeoff_profile(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
) -> CreativeTradeoffProfile:
    """Explore creative/technical trade-offs without selecting an outcome."""

    output_goal = _output_goal(request, creative_translation, creative_plan)
    top_runtime = _top_runtime(runtime_capabilities)
    tradeoffs = _primary_tradeoffs(
        creative_strategy=creative_strategy,
        creative_techniques=creative_techniques,
        creative_plan=creative_plan,
        creative_constraints=creative_constraints,
        runtime_capabilities=runtime_capabilities,
        top_runtime=top_runtime,
    )
    creative_benefits = _creative_benefits(
        creative_strategy=creative_strategy,
        creative_techniques=creative_techniques,
        runtime_capabilities=runtime_capabilities,
    )
    technical_costs = _technical_costs(
        creative_techniques=creative_techniques,
        creative_plan=creative_plan,
        creative_constraints=creative_constraints,
        top_runtime=top_runtime,
    )
    runtime_risks = _runtime_risks(runtime_capabilities)
    performance_concerns = _performance_concerns(
        creative_techniques,
        creative_constraints,
        runtime_capabilities,
    )
    complexity_risks = _complexity_risks(
        creative_techniques,
        creative_plan,
        creative_constraints,
    )
    fidelity_risks = _fidelity_risks(
        creative_strategy,
        creative_techniques,
        runtime_capabilities,
    )
    safety_concerns = _safety_concerns(creative_constraints)
    maintainability_concerns = _maintainability_concerns(
        creative_techniques,
        runtime_capabilities,
    )
    hitl_reason = _hitl_reason(
        tradeoffs=tradeoffs,
        creative_constraints=creative_constraints,
        runtime_capabilities=runtime_capabilities,
    )

    return CreativeTradeoffProfile(
        output_goal=output_goal,
        primary_tradeoffs=tradeoffs,
        creative_benefits=creative_benefits,
        technical_costs=technical_costs,
        runtime_risks=runtime_risks,
        performance_concerns=performance_concerns,
        complexity_risks=complexity_risks,
        fidelity_risks=fidelity_risks,
        cost_sensitivity=_cost_sensitivity(creative_constraints),
        safety_concerns=safety_concerns,
        maintainability_concerns=maintainability_concerns,
        hitl_advisable=hitl_reason is not None,
        hitl_reason=hitl_reason,
        director_discussion_points=_director_discussion_points(
            tradeoffs,
            hitl_reason,
        ),
        prompt_guidance=_prompt_guidance(tradeoffs, hitl_reason),
        evidence=_evidence(
            request=request,
            route_decision=route_decision,
            creative_strategy=creative_strategy,
            creative_techniques=creative_techniques,
            creative_plan=creative_plan,
            creative_constraints=creative_constraints,
            runtime_capabilities=runtime_capabilities,
        ),
    )


def creative_tradeoff_prompt_lines(
    profile: CreativeTradeoffProfile,
) -> tuple[str, ...]:
    """Render trade-off metadata as compact provider prompt guidance."""

    lines = [
        f"Authority boundary: {profile.authority_boundary}",
        f"Output goal: {profile.output_goal}",
        f"Cost sensitivity: {profile.cost_sensitivity}.",
    ]
    if profile.hitl_advisable and profile.hitl_reason is not None:
        lines.append(f"HITL advisory: {profile.hitl_reason}")
    lines.extend(f"Trade-off guidance: {item}" for item in profile.prompt_guidance)
    for tradeoff in profile.primary_tradeoffs[:4]:
        lines.append(
            "Primary trade-off: "
            f"{tradeoff.source_axis} vs {tradeoff.target_axis}; "
            f"{tradeoff.severity}; {tradeoff.summary}"
        )
        lines.append(f"Creative benefit: {tradeoff.creative_benefit}")
        lines.append(f"Technical cost: {tradeoff.technical_cost}")
        lines.append(f"Runtime implication: {tradeoff.runtime_implication}")
        lines.append(f"Mitigation: {tradeoff.mitigation}")
    lines.extend(
        f"Director discussion point: {item}"
        for item in profile.director_discussion_points[:4]
    )
    return tuple(lines[:28])


def _primary_tradeoffs(
    *,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    top_runtime: RuntimeCapabilityCandidate | None,
) -> tuple[CreativeTradeoff, ...]:
    tradeoffs: list[CreativeTradeoff] = []

    if creative_strategy is not None or creative_techniques is not None:
        tradeoffs.append(
            CreativeTradeoff(
                source_axis="creative_expressiveness",
                target_axis="implementation_complexity",
                severity=_severity_for_complexity(
                    creative_plan,
                    creative_constraints,
                    creative_techniques,
                ),
                summary=(
                    "Expressive strategy and technique choices can increase "
                    "implementation scope."
                ),
                creative_benefit=_expressive_benefit(
                    creative_strategy,
                    creative_techniques,
                ),
                technical_cost=_complexity_cost(
                    creative_plan,
                    creative_techniques,
                    top_runtime,
                ),
                runtime_implication=_runtime_implication(top_runtime),
                mitigation=(
                    "Keep the selected strategy visible while bounding the "
                    "number of interacting systems."
                ),
                director_discussion_point=(
                    "Should the output prioritize expressive richness or a "
                    "simpler implementation path?"
                ),
                hitl_recommended=_pressure_is_high(
                    creative_constraints.complexity_pressure
                    if creative_constraints is not None
                    else None
                ),
                evidence=_tradeoff_evidence(
                    creative_strategy,
                    creative_techniques,
                    top_runtime,
                ),
            )
        )

    if runtime_capabilities is not None:
        tradeoffs.append(
            CreativeTradeoff(
                source_axis="runtime_support",
                target_axis="concept_fidelity",
                severity=_severity_for_runtime(runtime_capabilities, top_runtime),
                summary=(
                    "A runtime may be easier to support while preserving only "
                    "part of the concept."
                ),
                creative_benefit=_runtime_benefit(top_runtime),
                technical_cost=_runtime_cost(runtime_capabilities, top_runtime),
                runtime_implication=(
                    _runtime_implication(top_runtime)
                    + " Treat likely candidates as comparison metadata only."
                ),
                mitigation=(
                    "Do not switch runtimes automatically; explain runtime "
                    "fit and preserve the current planning contract."
                ),
                director_discussion_point=(
                    "Does the user want strongest concept fidelity, or the "
                    "lowest-risk supported runtime path?"
                ),
                hitl_recommended=runtime_capabilities.hitl_advisable,
                evidence=runtime_capabilities.evidence[:6],
            )
        )

    if _has_performance_pressure(creative_techniques, creative_constraints):
        tradeoffs.append(
            CreativeTradeoff(
                source_axis="creative_expressiveness",
                target_axis="performance",
                severity="risk",
                summary=(
                    "Dense motion, simulation, or reactive behavior improves "
                    "expressiveness but can reduce frame stability."
                ),
                creative_benefit=(
                    "Keeps the output energetic, responsive, and visually rich."
                ),
                technical_cost=(
                    "Requires bounded counts, iteration limits, and clear "
                    "fallback behavior."
                ),
                runtime_implication=_runtime_implication(top_runtime),
                mitigation=(
                    "Prefer fewer legible systems before adding secondary "
                    "effects or variations."
                ),
                director_discussion_point=(
                    "How much performance headroom should be reserved before "
                    "adding expressive density?"
                ),
                hitl_recommended=False,
                evidence=_performance_evidence(
                    creative_techniques,
                    creative_constraints,
                    runtime_capabilities,
                ),
            )
        )

    if creative_constraints is not None and (
        creative_constraints.safety_pressure != "low"
        or creative_constraints.cost_pressure != "low"
    ):
        tradeoffs.append(
            CreativeTradeoff(
                source_axis="safety",
                target_axis="creative_expressiveness",
                severity=_severity_for_pressure(
                    max(
                        _pressure_rank(creative_constraints.safety_pressure),
                        _pressure_rank(creative_constraints.cost_pressure),
                    )
                ),
                summary=(
                    "Safety or cost pressure can require narrower generation "
                    "scope than the most expressive direction."
                ),
                creative_benefit=(
                    "A narrower scope keeps user intent reviewable and safer."
                ),
                technical_cost=(
                    "Some visual or conceptual ambition may need to be deferred."
                ),
                runtime_implication=(
                    "Keep runtime guidance conservative and avoid unsupported "
                    "claims."
                ),
                mitigation=(
                    "Surface the constraint explicitly and ask for HITL input "
                    "when the compromise affects intent."
                ),
                director_discussion_point=(
                    "Which safety or cost boundary should constrain the next "
                    "generation step?"
                ),
                hitl_recommended=creative_constraints.hitl_advisable,
                evidence=creative_constraints.evidence[:6],
            )
        )

    if not tradeoffs:
        tradeoffs.append(
            CreativeTradeoff(
                source_axis="maintainability",
                target_axis="creative_expressiveness",
                severity="info",
                summary=(
                    "The current context has low pressure; preserve a readable "
                    "implementation before adding novelty."
                ),
                creative_benefit="Supports a clear creative direction.",
                technical_cost="Requires restraint rather than extra systems.",
                runtime_implication="No runtime capability conflict is visible.",
                mitigation="Keep the output compact and explainable.",
                director_discussion_point=(
                    "Should the next response stay simple or add a controlled "
                    "creative variation?"
                ),
                evidence=("Default low-pressure trade-off.",),
            )
        )

    return tuple(tradeoffs[:8])


def _creative_benefits(
    *,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
) -> tuple[str, ...]:
    benefits: list[str] = []
    if creative_strategy is not None:
        benefits.extend(creative_strategy.creative_goals[:3])
    if creative_techniques is not None:
        benefits.extend(creative_techniques.artistic_suitability[:3])
    if runtime_capabilities is not None:
        benefits.extend(runtime_capabilities.candidate_runtimes[0].strengths[:2])
    if not benefits:
        benefits.append("Preserves the user's creative intent.")
    return _dedupe_text(benefits)[:8]


def _technical_costs(
    *,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    top_runtime: RuntimeCapabilityCandidate | None,
) -> tuple[str, ...]:
    costs: list[str] = []
    if creative_techniques is not None:
        costs.extend(creative_techniques.implementation_notes[:2])
    if creative_plan is not None:
        costs.append(f"Expected complexity: {creative_plan.expected_complexity}.")
        costs.append(f"Estimated token cost: {creative_plan.estimated_token_cost}.")
    if creative_constraints is not None:
        costs.append(
            f"Constraint complexity pressure: "
            f"{creative_constraints.complexity_pressure}."
        )
    if top_runtime is not None:
        costs.extend(top_runtime.limitations[:2])
    if not costs:
        costs.append("No major technical cost signal is available.")
    return _dedupe_text(costs)[:8]


def _runtime_risks(
    runtime_capabilities: RuntimeCapabilityProfile | None,
) -> tuple[str, ...]:
    if runtime_capabilities is None:
        return ()
    risks: list[str] = []
    for candidate in runtime_capabilities.candidate_runtimes[:3]:
        risks.extend(candidate.risks[:2])
    return _dedupe_text(risks)[:8]


def _performance_concerns(
    creative_techniques: CreativeTechniqueProfile | None,
    creative_constraints: CreativeConstraintSolution | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
) -> tuple[str, ...]:
    concerns: list[str] = []
    if (
        creative_techniques is not None
        and creative_techniques.performance_pressure != "low"
    ):
        concerns.append(
            f"Technique performance pressure: "
            f"{creative_techniques.performance_pressure}."
        )
    if (
        creative_constraints is not None
        and creative_constraints.performance_pressure != "low"
    ):
        concerns.append(
            f"Constraint performance pressure: "
            f"{creative_constraints.performance_pressure}."
        )
    if runtime_capabilities is not None:
        for candidate in runtime_capabilities.candidate_runtimes[:3]:
            if candidate.performance_pressure != "low":
                concerns.append(
                    f"{candidate.label} performance pressure: "
                    f"{candidate.performance_pressure}."
                )
    return _dedupe_text(concerns)[:8]


def _complexity_risks(
    creative_techniques: CreativeTechniqueProfile | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
) -> tuple[str, ...]:
    risks: list[str] = []
    if creative_plan is not None and creative_plan.expected_complexity != "low":
        risks.append(f"Plan complexity is {creative_plan.expected_complexity}.")
    if (
        creative_techniques is not None
        and creative_techniques.complexity_pressure != "low"
    ):
        risks.append(
            f"Technique complexity pressure is "
            f"{creative_techniques.complexity_pressure}."
        )
    if (
        creative_constraints is not None
        and creative_constraints.complexity_pressure != "low"
    ):
        risks.append(
            f"Constraint complexity pressure is "
            f"{creative_constraints.complexity_pressure}."
        )
    return _dedupe_text(risks)[:8]


def _fidelity_risks(
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
) -> tuple[str, ...]:
    risks: list[str] = []
    if creative_strategy is not None and creative_strategy.confidence < 0.6:
        risks.append("Strategy confidence is moderate; preserve user intent visibly.")
    if (
        creative_techniques is not None
        and creative_techniques.compatibility != "strong"
    ):
        risks.append(
            "Technique compatibility is not strong; verify concept alignment."
        )
    if runtime_capabilities is not None:
        top = runtime_capabilities.candidate_runtimes[0]
        if top.output_goal_fit != "strong" or top.suitability != "strong":
            risks.append(
                f"{top.label} may not fully preserve the output goal."
            )
    return _dedupe_text(risks)[:8]


def _safety_concerns(
    creative_constraints: CreativeConstraintSolution | None,
) -> tuple[str, ...]:
    if (
        creative_constraints is None
        or creative_constraints.safety_pressure == "low"
    ):
        return ()
    return tuple(
        item.summary
        for item in creative_constraints.active_constraints
        if item.axis == "safety"
    )[:8]


def _maintainability_concerns(
    creative_techniques: CreativeTechniqueProfile | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
) -> tuple[str, ...]:
    concerns: list[str] = []
    if creative_techniques is not None:
        concerns.append(
            f"Keep {creative_techniques.primary_technique} behavior readable."
        )
    if runtime_capabilities is not None:
        for candidate in runtime_capabilities.candidate_runtimes[:2]:
            if candidate.implementation_complexity != "low":
                concerns.append(
                    f"{candidate.label} requires explicit structure and limits."
                )
    if not concerns:
        concerns.append("Keep the implementation compact and explainable.")
    return _dedupe_text(concerns)[:8]


def _hitl_reason(
    *,
    tradeoffs: tuple[CreativeTradeoff, ...],
    creative_constraints: CreativeConstraintSolution | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
) -> str | None:
    if creative_constraints is not None and creative_constraints.hitl_advisable:
        return creative_constraints.hitl_reason or (
            "Constraint solver indicates a user-visible trade-off."
        )
    if runtime_capabilities is not None and runtime_capabilities.hitl_advisable:
        return runtime_capabilities.hitl_reason or (
            "Runtime capability fit is ambiguous enough for user confirmation."
        )
    if any(
        tradeoff.severity in {"risk", "blocking"} and tradeoff.hitl_recommended
        for tradeoff in tradeoffs
    ):
        return "A high-impact trade-off should be confirmed before generation."
    return None


def _director_discussion_points(
    tradeoffs: tuple[CreativeTradeoff, ...],
    hitl_reason: str | None,
) -> tuple[str, ...]:
    points = [item.director_discussion_point for item in tradeoffs]
    if hitl_reason is not None:
        points.insert(0, hitl_reason)
    return _dedupe_text(points)[:8]


def _prompt_guidance(
    tradeoffs: tuple[CreativeTradeoff, ...],
    hitl_reason: str | None,
) -> tuple[str, ...]:
    guidance = [
        "Use trade-off metadata to explain consequences, not to select an outcome.",
        "Preserve current planning, runtime, provider, and preview contracts.",
    ]
    if hitl_reason is not None:
        guidance.append(hitl_reason)
    guidance.extend(item.mitigation for item in tradeoffs[:3])
    return _dedupe_text(guidance)[:8]


def _evidence(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
) -> tuple[str, ...]:
    evidence = [f"Mode: {request.mode.value}."]
    if route_decision is not None:
        evidence.append(f"Route selected: {route_decision.route.value}.")
    if creative_strategy is not None:
        evidence.append(f"Creative strategy: {creative_strategy.primary_strategy}.")
    if creative_techniques is not None:
        evidence.append(
            f"Creative technique: {creative_techniques.primary_technique}."
        )
    if creative_plan is not None:
        evidence.append(f"Output goal: {creative_plan.generation_strategy}")
        evidence.append(f"Plan complexity: {creative_plan.expected_complexity}.")
    if creative_constraints is not None:
        evidence.append(
            f"Constraint pressures: complexity "
            f"{creative_constraints.complexity_pressure}, performance "
            f"{creative_constraints.performance_pressure}, cost "
            f"{creative_constraints.cost_pressure}."
        )
    if runtime_capabilities is not None:
        evidence.append(
            "Runtime candidates: "
            + ", ".join(runtime_capabilities.likely_candidates)
            + "."
        )
    return _dedupe_text(evidence)[:10]


def _output_goal(
    request: AssistantRequest,
    creative_translation: CreativeTranslation | None,
    creative_plan: CreativeExecutionPlan | None,
) -> str:
    if creative_plan is not None:
        return creative_plan.generation_strategy
    if creative_translation is not None:
        return creative_translation.creative_intent
    return " ".join(request.query.split())[:360]


def _top_runtime(
    runtime_capabilities: RuntimeCapabilityProfile | None,
) -> RuntimeCapabilityCandidate | None:
    if runtime_capabilities is None:
        return None
    return runtime_capabilities.candidate_runtimes[0]


def _expressive_benefit(
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
) -> str:
    if creative_strategy is not None and creative_techniques is not None:
        return (
            f"{creative_strategy.primary_strategy} with "
            f"{creative_techniques.primary_technique} can preserve a distinct "
            "creative direction."
        )
    if creative_strategy is not None:
        return f"{creative_strategy.primary_strategy} clarifies creative intent."
    if creative_techniques is not None:
        return f"{creative_techniques.primary_technique} adds concrete form."
    return "Expressive choices can make the output more distinctive."


def _complexity_cost(
    creative_plan: CreativeExecutionPlan | None,
    creative_techniques: CreativeTechniqueProfile | None,
    top_runtime: RuntimeCapabilityCandidate | None,
) -> str:
    parts: list[str] = []
    if creative_plan is not None:
        parts.append(f"plan complexity {creative_plan.expected_complexity}")
    if creative_techniques is not None:
        parts.append(f"technique complexity {creative_techniques.complexity_pressure}")
    if top_runtime is not None:
        parts.append(
            f"{top_runtime.label} complexity "
            f"{top_runtime.implementation_complexity}"
        )
    if not parts:
        return "Complexity cost is low but still requires readable structure."
    return "Requires managing " + ", ".join(parts) + "."


def _runtime_implication(
    top_runtime: RuntimeCapabilityCandidate | None,
) -> str:
    if top_runtime is None:
        return "Runtime implications are undetermined from available metadata."
    return (
        f"{top_runtime.label} has {top_runtime.suitability} suitability and "
        f"{top_runtime.preview_support} preview support."
    )


def _runtime_benefit(
    top_runtime: RuntimeCapabilityCandidate | None,
) -> str:
    if top_runtime is None:
        return "Runtime comparison can clarify implementation direction."
    return top_runtime.strengths[0]


def _runtime_cost(
    runtime_capabilities: RuntimeCapabilityProfile | None,
    top_runtime: RuntimeCapabilityCandidate | None,
) -> str:
    if top_runtime is None:
        return "No candidate runtime is available for comparison."
    cost = top_runtime.limitations[0]
    if runtime_capabilities is not None and runtime_capabilities.hitl_advisable:
        cost += " Runtime fit may need user confirmation."
    return cost


def _tradeoff_evidence(
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    top_runtime: RuntimeCapabilityCandidate | None,
) -> tuple[str, ...]:
    evidence: list[str] = []
    if creative_strategy is not None:
        evidence.append(f"Strategy: {creative_strategy.primary_strategy}.")
    if creative_techniques is not None:
        evidence.append(f"Technique: {creative_techniques.primary_technique}.")
    if top_runtime is not None:
        evidence.append(f"Runtime candidate: {top_runtime.runtime}.")
    return tuple(evidence[:6])


def _performance_evidence(
    creative_techniques: CreativeTechniqueProfile | None,
    creative_constraints: CreativeConstraintSolution | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
) -> tuple[str, ...]:
    evidence: list[str] = []
    if creative_techniques is not None:
        evidence.append(
            f"Technique performance: {creative_techniques.performance_pressure}."
        )
    if creative_constraints is not None:
        evidence.append(
            f"Constraint performance: {creative_constraints.performance_pressure}."
        )
    if runtime_capabilities is not None:
        top = runtime_capabilities.candidate_runtimes[0]
        evidence.append(f"Runtime performance: {top.performance_pressure}.")
    return tuple(evidence[:6])


def _has_performance_pressure(
    creative_techniques: CreativeTechniqueProfile | None,
    creative_constraints: CreativeConstraintSolution | None,
) -> bool:
    return (
        creative_techniques is not None
        and creative_techniques.performance_pressure == "high"
    ) or (
        creative_constraints is not None
        and creative_constraints.performance_pressure == "high"
    )


def _severity_for_complexity(
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    creative_techniques: CreativeTechniqueProfile | None,
) -> TradeoffSeverity:
    pressure = 0
    if creative_plan is not None:
        pressure = max(pressure, _pressure_rank(creative_plan.expected_complexity))
    if creative_constraints is not None:
        pressure = max(
            pressure,
            _pressure_rank(creative_constraints.complexity_pressure),
        )
    if creative_techniques is not None:
        pressure = max(
            pressure,
            _pressure_rank(creative_techniques.complexity_pressure),
        )
    return _severity_for_pressure(pressure)


def _severity_for_runtime(
    runtime_capabilities: RuntimeCapabilityProfile,
    top_runtime: RuntimeCapabilityCandidate | None,
) -> TradeoffSeverity:
    if runtime_capabilities.hitl_advisable:
        return "risk"
    if top_runtime is None:
        return "watch"
    if top_runtime.suitability == "weak":
        return "risk"
    if top_runtime.suitability == "moderate":
        return "watch"
    return "info"


def _severity_for_pressure(pressure: int) -> TradeoffSeverity:
    if pressure >= 3:
        return "blocking"
    if pressure >= 2:
        return "risk"
    if pressure == 1:
        return "watch"
    return "info"


def _pressure_rank(value: str | None) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(value or "low", 0)


def _pressure_is_high(value: str | None) -> bool:
    return value == "high"


def _cost_sensitivity(
    creative_constraints: CreativeConstraintSolution | None,
) -> TradeoffPressure:
    if creative_constraints is None:
        return "low"
    return creative_constraints.cost_pressure


def _dedupe_text(values: list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        cleaned = " ".join(value.strip().split())
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return tuple(normalized)
