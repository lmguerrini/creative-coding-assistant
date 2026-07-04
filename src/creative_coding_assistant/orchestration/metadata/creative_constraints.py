"""Bounded creative constraint solving for V3 workflows."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest, CreativeCodingDomain
from creative_coding_assistant.orchestration.clarification import ClarificationRequest
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
from creative_coding_assistant.orchestration.creative_translation import (
    CreativeOutputModality,
    CreativeTranslation,
)
from creative_coding_assistant.orchestration.routing import RouteDecision

ConstraintAxis = Literal[
    "intent",
    "modality",
    "runtime",
    "safety",
    "performance",
    "complexity",
    "cost",
    "hitl",
    "output_goal",
]
ConstraintSeverity = Literal["info", "watch", "risk", "blocking"]
ConstraintPressure = Literal["low", "medium", "high"]
RuntimeFit = Literal["supported", "code_only", "undetermined"]

SOLVER_AUTHORITY_BOUNDARY = (
    "The Creative Constraint Solver structures trade-offs for inspection; it "
    "does not make autonomous creative decisions or override the user."
)


class CreativeConstraint(BaseModel):
    """One inspectable constraint active for the current assistant run."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    axis: ConstraintAxis
    severity: ConstraintSeverity
    summary: str = Field(min_length=1, max_length=240)
    recommendation: str = Field(min_length=1, max_length=280)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=6)


class CreativeConstraintTradeoff(BaseModel):
    """A bounded trade-off between two constraint axes."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source_axis: ConstraintAxis
    target_axis: ConstraintAxis
    severity: ConstraintSeverity
    summary: str = Field(min_length=1, max_length=260)
    recommendation: str = Field(min_length=1, max_length=300)


class CreativeConstraintSolution(BaseModel):
    """Structured trade-off metadata derived from existing workflow signals."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_constraint_solver"] = "creative_constraint_solver"
    intent_summary: str = Field(min_length=1, max_length=280)
    output_goal: str = Field(min_length=1, max_length=360)
    modality: str | None = Field(default=None, max_length=80)
    runtime_fit: RuntimeFit
    recommended_runtime: str | None = Field(default=None, max_length=80)
    complexity_pressure: ConstraintPressure
    safety_pressure: ConstraintPressure
    performance_pressure: ConstraintPressure
    cost_pressure: ConstraintPressure
    hitl_advisable: bool = False
    hitl_reason: str | None = Field(default=None, max_length=280)
    active_constraints: tuple[CreativeConstraint, ...] = Field(
        min_length=1,
        max_length=12,
    )
    tradeoffs: tuple[CreativeConstraintTradeoff, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    conflicts: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=SOLVER_AUTHORITY_BOUNDARY,
        max_length=320,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=10)


def derive_creative_constraint_solution(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_intent: CreativeIntentDecomposition | None = None,
    creative_hierarchy: CreativeHierarchyPlan | None = None,
    creative_translation: CreativeTranslation | None = None,
    creative_plan: CreativeExecutionPlan | None = None,
    creative_strategy: CreativeStrategyProfile | None = None,
    creative_techniques: CreativeTechniqueProfile | None = None,
    clarification: ClarificationRequest | None = None,
    retrieval_chunk_count: int = 0,
) -> CreativeConstraintSolution:
    """Solve current creative constraints with deterministic, inspectable rules."""

    domains = _effective_domains(request, route_decision)
    runtime_fit = _runtime_fit(creative_plan, domains)
    complexity_pressure = _complexity_pressure(request, creative_plan, domains)
    cost_pressure = _cost_pressure(creative_plan, retrieval_chunk_count)
    safety_pressure = _safety_pressure(request, creative_translation)
    performance_pressure = _performance_pressure(
        request=request,
        creative_plan=creative_plan,
        complexity_pressure=complexity_pressure,
    )
    hitl_reason = _hitl_reason(
        clarification=clarification,
        creative_intent=creative_intent,
        creative_hierarchy=creative_hierarchy,
        route_decision=route_decision,
        creative_plan=creative_plan,
        runtime_fit=runtime_fit,
        complexity_pressure=complexity_pressure,
    )
    hitl_advisable = hitl_reason is not None
    intent_summary = _intent_summary(request, creative_intent, creative_translation)
    output_goal = _output_goal(request, creative_plan)

    active_constraints = _active_constraints(
        intent_summary=intent_summary,
        output_goal=output_goal,
        request=request,
        creative_intent=creative_intent,
        creative_hierarchy=creative_hierarchy,
        creative_translation=creative_translation,
        creative_strategy=creative_strategy,
        creative_techniques=creative_techniques,
        creative_plan=creative_plan,
        runtime_fit=runtime_fit,
        safety_pressure=safety_pressure,
        performance_pressure=performance_pressure,
        complexity_pressure=complexity_pressure,
        cost_pressure=cost_pressure,
        hitl_reason=hitl_reason,
        domains=domains,
    )
    conflicts = _conflicts(
        clarification=clarification,
        creative_plan=creative_plan,
        creative_intent=creative_intent,
        creative_hierarchy=creative_hierarchy,
        creative_translation=creative_translation,
        runtime_fit=runtime_fit,
        performance_pressure=performance_pressure,
        complexity_pressure=complexity_pressure,
    )
    return CreativeConstraintSolution(
        intent_summary=intent_summary,
        output_goal=output_goal,
        modality=_modality_label(creative_translation, creative_plan),
        runtime_fit=runtime_fit,
        recommended_runtime=(
            creative_plan.recommended_runtime if creative_plan is not None else None
        ),
        complexity_pressure=complexity_pressure,
        safety_pressure=safety_pressure,
        performance_pressure=performance_pressure,
        cost_pressure=cost_pressure,
        hitl_advisable=hitl_advisable,
        hitl_reason=hitl_reason,
        active_constraints=active_constraints,
        tradeoffs=_tradeoffs(
            runtime_fit=runtime_fit,
            safety_pressure=safety_pressure,
            performance_pressure=performance_pressure,
            complexity_pressure=complexity_pressure,
            cost_pressure=cost_pressure,
            hitl_advisable=hitl_advisable,
        ),
        conflicts=conflicts,
        prompt_guidance=_prompt_guidance(
            creative_plan=creative_plan,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            runtime_fit=runtime_fit,
            complexity_pressure=complexity_pressure,
            cost_pressure=cost_pressure,
            safety_pressure=safety_pressure,
            hitl_advisable=hitl_advisable,
            creative_strategy=creative_strategy,
            creative_techniques=creative_techniques,
        ),
        evidence=_evidence(
            request=request,
            route_decision=route_decision,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_translation=creative_translation,
            creative_strategy=creative_strategy,
            creative_techniques=creative_techniques,
            creative_plan=creative_plan,
            retrieval_chunk_count=retrieval_chunk_count,
            clarification=clarification,
            domains=domains,
        ),
    )


def creative_constraint_solution_prompt_lines(
    solution: CreativeConstraintSolution,
) -> tuple[str, ...]:
    """Render solver metadata into compact provider prompt guidance."""

    lines = [
        f"Authority boundary: {solution.authority_boundary}",
        f"Intent summary: {solution.intent_summary}",
        f"Output goal: {solution.output_goal}",
        f"Runtime fit: {solution.runtime_fit}.",
        f"Complexity pressure: {solution.complexity_pressure}.",
        f"Performance pressure: {solution.performance_pressure}.",
        f"Cost pressure: {solution.cost_pressure}.",
        f"Safety pressure: {solution.safety_pressure}.",
    ]
    if solution.recommended_runtime is not None:
        lines.append(f"Recommended runtime: {solution.recommended_runtime}.")
    if solution.hitl_advisable and solution.hitl_reason is not None:
        lines.append(f"HITL advisory: {solution.hitl_reason}")
    lines.extend(f"Constraint guidance: {item}" for item in solution.prompt_guidance)
    lines.extend(f"Trade-off: {item.summary}" for item in solution.tradeoffs[:3])
    lines.extend(f"Conflict: {item}" for item in solution.conflicts[:3])
    return tuple(lines[:20])


def _intent_summary(
    request: AssistantRequest,
    creative_intent: CreativeIntentDecomposition | None,
    creative_translation: CreativeTranslation | None,
) -> str:
    if creative_intent is not None:
        return creative_intent.primary_expression
    if creative_translation is not None:
        return creative_translation.creative_intent
    return _compact(request.query)[:280]


def _output_goal(
    request: AssistantRequest,
    creative_plan: CreativeExecutionPlan | None,
) -> str:
    if creative_plan is not None:
        return creative_plan.generation_strategy
    action = "Refine" if request.artifact_refinement is not None else "Generate"
    return f"{action} a bounded creative-coding response for the user request."


def _modality_label(
    creative_translation: CreativeTranslation | None,
    creative_plan: CreativeExecutionPlan | None,
) -> str | None:
    if creative_plan is not None:
        return creative_plan.output_modality.value
    if creative_translation is not None and creative_translation.output_modality:
        return creative_translation.output_modality.value
    return None


def _runtime_fit(
    creative_plan: CreativeExecutionPlan | None,
    domains: tuple[CreativeCodingDomain, ...],
) -> RuntimeFit:
    if creative_plan is None:
        return "undetermined"
    if creative_plan.runtime_available:
        return "supported"
    if domains or creative_plan.recommended_runtime is None:
        return "code_only"
    return "undetermined"


def _complexity_pressure(
    request: AssistantRequest,
    creative_plan: CreativeExecutionPlan | None,
    domains: tuple[CreativeCodingDomain, ...],
) -> ConstraintPressure:
    if creative_plan is not None:
        return creative_plan.expected_complexity
    score = int(bool(request.attachments)) + int(len(domains) > 1)
    score += int(len(request.query) > 220)
    if score >= 2:
        return "medium"
    return "low"


def _cost_pressure(
    creative_plan: CreativeExecutionPlan | None,
    retrieval_chunk_count: int,
) -> ConstraintPressure:
    if creative_plan is None:
        return "medium" if retrieval_chunk_count >= 3 else "low"
    if creative_plan.estimated_token_cost >= 6200:
        return "high"
    if creative_plan.estimated_token_cost >= 3400 or retrieval_chunk_count >= 4:
        return "medium"
    return "low"


def _safety_pressure(
    request: AssistantRequest,
    creative_translation: CreativeTranslation | None,
) -> ConstraintPressure:
    score = 0
    normalized = request.query.lower()
    if request.attachments:
        score += 1
    if _SAFETY_PATTERN.search(normalized):
        score += 2
    if creative_translation is not None:
        score += len(creative_translation.generation_constraints) > 0
        if creative_translation.reference_fusion is not None:
            score += len(creative_translation.reference_fusion.safety_constraints)
    if score >= 3:
        return "high"
    if score >= 1:
        return "medium"
    return "low"


def _performance_pressure(
    *,
    request: AssistantRequest,
    creative_plan: CreativeExecutionPlan | None,
    complexity_pressure: ConstraintPressure,
) -> ConstraintPressure:
    score = 0
    if complexity_pressure == "high":
        score += 2
    elif complexity_pressure == "medium":
        score += 1
    if _PERFORMANCE_PATTERN.search(request.query.lower()):
        score += 2
    if creative_plan is not None and creative_plan.candidate_count > 1:
        score += 1
    if score >= 3:
        return "high"
    if score >= 1:
        return "medium"
    return "low"


def _hitl_reason(
    *,
    clarification: ClarificationRequest | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    route_decision: RouteDecision | None,
    creative_plan: CreativeExecutionPlan | None,
    runtime_fit: RuntimeFit,
    complexity_pressure: ConstraintPressure,
) -> str | None:
    if clarification is not None:
        return clarification.summary
    if creative_hierarchy is not None and creative_hierarchy.hitl_questions:
        return creative_hierarchy.hitl_questions[0]
    if creative_intent is not None and creative_intent.unresolved_intent_gaps:
        return creative_intent.unresolved_intent_gaps[0]
    if route_decision is not None and len(route_decision.domains) > 2:
        return "Multiple domains are active; confirm which runtime should lead."
    if runtime_fit == "code_only" and creative_plan is not None:
        return "Requested scope has no current live preview runtime support."
    if complexity_pressure == "high" and creative_plan is not None:
        return "High complexity may need scope reduction or a split output."
    return None


def _active_constraints(
    *,
    intent_summary: str,
    output_goal: str,
    request: AssistantRequest,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_translation: CreativeTranslation | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_plan: CreativeExecutionPlan | None,
    runtime_fit: RuntimeFit,
    safety_pressure: ConstraintPressure,
    performance_pressure: ConstraintPressure,
    complexity_pressure: ConstraintPressure,
    cost_pressure: ConstraintPressure,
    hitl_reason: str | None,
    domains: tuple[CreativeCodingDomain, ...],
) -> tuple[CreativeConstraint, ...]:
    constraints = [
        CreativeConstraint(
            axis="intent",
            severity="info",
            summary=f"Preserve creative intent: {intent_summary}",
            recommendation=_intent_recommendation(creative_strategy),
            evidence=_constraint_evidence(
                creative_intent,
                creative_translation,
                creative_strategy,
                "intent",
            ),
        ),
        CreativeConstraint(
            axis="output_goal",
            severity="info",
            summary=_clip(output_goal, 240),
            recommendation="Keep the response aligned to the planned output goal.",
            evidence=(request.mode.value,),
        ),
        _runtime_constraint(creative_plan, runtime_fit, domains),
        CreativeConstraint(
            axis="complexity",
            severity=_severity_for_pressure(complexity_pressure),
            summary=f"Implementation complexity pressure is {complexity_pressure}.",
            recommendation=_complexity_recommendation(complexity_pressure),
            evidence=_plan_evidence(creative_plan, "expected_complexity"),
        ),
        CreativeConstraint(
            axis="performance",
            severity=_severity_for_pressure(performance_pressure),
            summary=f"Performance pressure is {performance_pressure}.",
            recommendation=_performance_recommendation(performance_pressure),
            evidence=_plan_evidence(creative_plan, "candidate_count"),
        ),
        CreativeConstraint(
            axis="cost",
            severity=_severity_for_pressure(cost_pressure),
            summary=f"Estimated cost pressure is {cost_pressure}.",
            recommendation=_cost_recommendation(cost_pressure),
            evidence=_plan_evidence(creative_plan, "estimated_token_cost"),
        ),
        CreativeConstraint(
            axis="safety",
            severity=_severity_for_pressure(safety_pressure),
            summary=f"Safety pressure is {safety_pressure}.",
            recommendation=_safety_recommendation(safety_pressure),
            evidence=_safety_evidence(request, creative_translation),
        ),
    ]
    if hitl_reason is not None:
        constraints.append(
            CreativeConstraint(
                axis="hitl",
                severity="watch",
                summary="Human-in-the-loop input is advisable.",
                recommendation=hitl_reason,
                evidence=(hitl_reason,),
            )
        )
    if creative_intent is not None and creative_intent.unresolved_intent_gaps:
        constraints.append(
            CreativeConstraint(
                axis="hitl",
                severity="watch",
                summary="Creative intent has unresolved decomposition gaps.",
                recommendation=creative_intent.unresolved_intent_gaps[0],
                evidence=creative_intent.hitl_questions[:3],
            )
        )
    if creative_hierarchy is not None:
        constraints.append(
            CreativeConstraint(
                axis="intent",
                severity="info",
                summary=_hierarchy_constraint_summary(creative_hierarchy),
                recommendation=creative_hierarchy.prompt_guidance[0],
                evidence=creative_hierarchy.priority_rationale[:3],
            )
        )
    if creative_strategy is not None:
        constraints.append(
            CreativeConstraint(
                axis="intent",
                severity="info",
                summary=(f"Creative strategy: {creative_strategy.primary_strategy}."),
                recommendation=(
                    "Use strategy as artistic direction only, not runtime or "
                    "technique selection."
                ),
                evidence=creative_strategy.strategy_directives[:3],
            )
        )
    if creative_techniques is not None:
        constraints.append(
            CreativeConstraint(
                axis="complexity",
                severity=_severity_for_pressure(
                    creative_techniques.complexity_pressure
                ),
                summary=(
                    f"Creative technique: {creative_techniques.primary_technique}."
                ),
                recommendation=(
                    "Use technique guidance only where it supports the selected "
                    "strategy and constraints."
                ),
                evidence=creative_techniques.implementation_notes[:3],
            )
        )
    return tuple(constraints[:12])


def _runtime_constraint(
    creative_plan: CreativeExecutionPlan | None,
    runtime_fit: RuntimeFit,
    domains: tuple[CreativeCodingDomain, ...],
) -> CreativeConstraint:
    if creative_plan is not None and creative_plan.runtime_available:
        return CreativeConstraint(
            axis="runtime",
            severity="info",
            summary=creative_plan.runtime_support_summary,
            recommendation=(
                f"Use {creative_plan.recommended_runtime} through "
                f"{creative_plan.recommended_renderer_id}."
            ),
            evidence=_plan_evidence(creative_plan, "recommended_runtime"),
        )
    domain_label = ", ".join(domain.value for domain in domains) or "unspecified"
    return CreativeConstraint(
        axis="runtime",
        severity="risk" if domains else "watch",
        summary=f"Runtime fit is {runtime_fit} for {domain_label}.",
        recommendation=(
            "Keep output code-only and do not claim live preview readiness."
            if runtime_fit == "code_only"
            else "Keep runtime assumptions explicit in the generated response."
        ),
        evidence=(domain_label,),
    )


def _hierarchy_constraint_summary(
    creative_hierarchy: CreativeHierarchyPlan,
) -> str:
    if creative_hierarchy.non_negotiable_dimensions:
        return _clip(
            "Creative hierarchy non-negotiables: "
            + ", ".join(creative_hierarchy.non_negotiable_dimensions)
            + ".",
            240,
        )
    return "Creative hierarchy priorities are advisory."


def _tradeoffs(
    *,
    runtime_fit: RuntimeFit,
    safety_pressure: ConstraintPressure,
    performance_pressure: ConstraintPressure,
    complexity_pressure: ConstraintPressure,
    cost_pressure: ConstraintPressure,
    hitl_advisable: bool,
) -> tuple[CreativeConstraintTradeoff, ...]:
    tradeoffs: list[CreativeConstraintTradeoff] = []
    if runtime_fit == "code_only":
        tradeoffs.append(
            CreativeConstraintTradeoff(
                source_axis="intent",
                target_axis="runtime",
                severity="risk",
                summary="Creative intent may exceed current live preview support.",
                recommendation=(
                    "Preserve the concept while labeling output as code-only."
                ),
            )
        )
    if complexity_pressure == "high" or performance_pressure == "high":
        tradeoffs.append(
            CreativeConstraintTradeoff(
                source_axis="complexity",
                target_axis="performance",
                severity="risk",
                summary="Dense implementation goals can threaten runtime performance.",
                recommendation="Reduce particle counts, passes, or candidates first.",
            )
        )
    if cost_pressure == "high":
        tradeoffs.append(
            CreativeConstraintTradeoff(
                source_axis="output_goal",
                target_axis="cost",
                severity="watch",
                summary="Broad output goals can increase token and refinement cost.",
                recommendation="Prefer one coherent artifact before extra variations.",
            )
        )
    if safety_pressure in {"medium", "high"}:
        tradeoffs.append(
            CreativeConstraintTradeoff(
                source_axis="intent",
                target_axis="safety",
                severity="watch",
                summary="Aesthetic goals must remain inside safety guardrails.",
                recommendation="Adapt sensitive inputs without identifying people.",
            )
        )
    if hitl_advisable:
        tradeoffs.append(
            CreativeConstraintTradeoff(
                source_axis="output_goal",
                target_axis="hitl",
                severity="watch",
                summary="Proceeding immediately may hide an important creative choice.",
                recommendation="Surface the ambiguity before generating when possible.",
            )
        )
    return tuple(tradeoffs[:8])


def _conflicts(
    *,
    clarification: ClarificationRequest | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_translation: CreativeTranslation | None,
    runtime_fit: RuntimeFit,
    performance_pressure: ConstraintPressure,
    complexity_pressure: ConstraintPressure,
) -> tuple[str, ...]:
    conflicts: list[str] = []
    if clarification is not None:
        conflicts.append(f"Clarification pending: {clarification.reason.value}.")
    if creative_intent is not None and creative_intent.unresolved_intent_gaps:
        conflicts.append(
            "Intent ambiguity: " + creative_intent.unresolved_intent_gaps[0]
        )
    if creative_hierarchy is not None:
        conflicts.extend(creative_hierarchy.priority_conflicts[:2])
    if runtime_fit == "code_only":
        conflicts.append(
            "Requested scope does not map to current live preview support."
        )
    if complexity_pressure == "high" and performance_pressure == "high":
        conflicts.append("High complexity conflicts with stable runtime performance.")
    if (
        creative_translation is not None
        and creative_translation.audio_reactive is not None
        and creative_plan is not None
        and creative_plan.output_modality == CreativeOutputModality.VISUAL
    ):
        conflicts.append(
            "Audio-reactive intent needs explicit mapping inside a visual output."
        )
    return tuple(_dedupe_text(conflicts))[:6]


def _prompt_guidance(
    *,
    creative_plan: CreativeExecutionPlan | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    runtime_fit: RuntimeFit,
    complexity_pressure: ConstraintPressure,
    cost_pressure: ConstraintPressure,
    safety_pressure: ConstraintPressure,
    hitl_advisable: bool,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
) -> tuple[str, ...]:
    guidance = ["Preserve the user's creative intent while making constraints visible."]
    if creative_intent is not None:
        guidance.append("Treat decomposed intent dimensions as separate constraints.")
    if creative_hierarchy is not None:
        guidance.append("Preserve non-negotiable hierarchy dimensions first.")
    if creative_strategy is not None:
        guidance.append(
            f"Preserve {creative_strategy.primary_strategy} as high-level strategy."
        )
    if creative_techniques is not None:
        guidance.append(
            "Use "
            f"{creative_techniques.primary_technique} "
            "as bounded technique guidance."
        )
    if creative_plan is not None and creative_plan.recommended_runtime is not None:
        guidance.append(f"Target {creative_plan.recommended_runtime} output.")
    if runtime_fit == "code_only":
        guidance.append("Do not claim live preview readiness for this output.")
    if complexity_pressure == "high":
        guidance.append("Reduce scope before adding extra candidates or effects.")
    if cost_pressure == "high":
        guidance.append("Prefer the smallest complete artifact that satisfies intent.")
    if safety_pressure in {"medium", "high"}:
        guidance.append("Keep safety constraints explicit in the generated answer.")
    if hitl_advisable:
        guidance.append("Surface the unresolved trade-off to the user.")
    return tuple(_dedupe_text(guidance))[:8]


def _evidence(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_translation: CreativeTranslation | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_plan: CreativeExecutionPlan | None,
    retrieval_chunk_count: int,
    clarification: ClarificationRequest | None,
    domains: tuple[CreativeCodingDomain, ...],
) -> tuple[str, ...]:
    evidence = [f"Mode: {request.mode.value}."]
    if route_decision is not None:
        evidence.append(f"Route selected: {route_decision.route.value}.")
    if domains:
        evidence.append(
            "Domains: " + ", ".join(domain.value for domain in domains) + "."
        )
    if creative_translation is not None:
        evidence.append(f"Intent: {creative_translation.creative_intent}.")
    if creative_intent is not None:
        evidence.append(f"Intent gaps: {len(creative_intent.unresolved_intent_gaps)}.")
    if creative_hierarchy is not None:
        evidence.append(
            f"Hierarchy confidence: {creative_hierarchy.hierarchy_confidence:.2f}."
        )
    if creative_strategy is not None:
        evidence.append(f"Creative strategy: {creative_strategy.primary_strategy}.")
    if creative_techniques is not None:
        evidence.append(f"Creative technique: {creative_techniques.primary_technique}.")
    if creative_plan is not None:
        evidence.append(f"Runtime available: {creative_plan.runtime_available}.")
        evidence.append(f"Complexity: {creative_plan.expected_complexity}.")
        evidence.append(f"Estimated token cost: {creative_plan.estimated_token_cost}.")
    if retrieval_chunk_count:
        evidence.append(f"Retrieval chunks: {retrieval_chunk_count}.")
    if clarification is not None:
        evidence.append(f"HITL reason: {clarification.reason.value}.")
    return tuple(_dedupe_text(evidence))[:10]


def _effective_domains(
    request: AssistantRequest,
    route_decision: RouteDecision | None,
) -> tuple[CreativeCodingDomain, ...]:
    domains = request.domains
    if domains:
        return domains
    if request.domain is not None:
        return (request.domain,)
    if route_decision is not None and route_decision.domains:
        return route_decision.domains
    if request.artifact_refinement and request.artifact_refinement.domain:
        return (request.artifact_refinement.domain,)
    return ()


def _severity_for_pressure(pressure: ConstraintPressure) -> ConstraintSeverity:
    if pressure == "high":
        return "risk"
    if pressure == "medium":
        return "watch"
    return "info"


def _complexity_recommendation(pressure: ConstraintPressure) -> str:
    if pressure == "high":
        return "Split or simplify the request before expanding artifact scope."
    if pressure == "medium":
        return "Keep the implementation focused and avoid unnecessary variations."
    return "Keep the implementation direct."


def _performance_recommendation(pressure: ConstraintPressure) -> str:
    if pressure == "high":
        return "Favor efficient loops, bounded geometry, and conservative effects."
    if pressure == "medium":
        return "Mention likely performance-sensitive choices in the output."
    return "No special performance trade-off is active."


def _cost_recommendation(pressure: ConstraintPressure) -> str:
    if pressure == "high":
        return "Avoid extra candidates and long explanatory detours."
    if pressure == "medium":
        return "Keep prompt scope bounded to the current creative goal."
    return "Cost pressure is low for the planned response."


def _safety_recommendation(pressure: ConstraintPressure) -> str:
    if pressure == "high":
        return "Prioritize safety constraints over aesthetic literalism."
    if pressure == "medium":
        return "Preserve safety guardrails while translating the aesthetic."
    return "Apply normal global safety guardrails."


def _plan_evidence(
    creative_plan: CreativeExecutionPlan | None,
    field_name: str,
) -> tuple[str, ...]:
    if creative_plan is None:
        return ()
    value = getattr(creative_plan, field_name, None)
    return (f"{field_name}: {value}",) if value is not None else ()


def _constraint_evidence(
    creative_intent: CreativeIntentDecomposition | None,
    creative_translation: CreativeTranslation | None,
    creative_strategy: CreativeStrategyProfile | None,
    fallback: str,
) -> tuple[str, ...]:
    evidence: list[str] = []
    if creative_intent is not None:
        evidence.append(creative_intent.primary_expression)
    elif creative_translation is None:
        evidence.append(fallback)
    else:
        evidence.append(creative_translation.creative_intent)
    if creative_strategy is not None:
        evidence.append(f"Strategy: {creative_strategy.primary_strategy}.")
    return tuple(_dedupe_text(evidence))[:6]


def _intent_recommendation(
    creative_strategy: CreativeStrategyProfile | None,
) -> str:
    if creative_strategy is None:
        return "Treat the translated intent as the source of truth."
    return "Treat the translated intent and high-level strategy as source of truth."


def _safety_evidence(
    request: AssistantRequest,
    creative_translation: CreativeTranslation | None,
) -> tuple[str, ...]:
    evidence: list[str] = []
    if request.attachments:
        evidence.append(f"Image references: {len(request.attachments)}.")
    if creative_translation is not None:
        evidence.extend(creative_translation.generation_constraints[:3])
        if creative_translation.reference_fusion is not None:
            evidence.extend(
                creative_translation.reference_fusion.safety_constraints[:3]
            )
    return tuple(_dedupe_text(evidence))[:6]


def _compact(value: str) -> str:
    return " ".join(value.strip().split())


def _clip(value: str, limit: int) -> str:
    normalized = _compact(value)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "."


def _dedupe_text(values: list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        cleaned = _compact(value)
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return tuple(normalized)


_SAFETY_PATTERN = re.compile(
    r"\b(?:autoplay|camera|face|flash|flashing|flicker|identify|microphone|"
    r"person|photo|seizure|strobe|upload|webcam)\b"
)
_PERFORMANCE_PATTERN = re.compile(
    r"\b(?:60\s?fps|dense|mobile|many|performance|realtime|real-time|"
    r"thousands?|webgl)\b"
)
