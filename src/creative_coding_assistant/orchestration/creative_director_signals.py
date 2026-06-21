"""Signal composition helpers for Creative Assistant Director metadata."""

from __future__ import annotations

from creative_coding_assistant.contracts import AssistantRequest, CreativeCodingDomain
from creative_coding_assistant.orchestration.artifact_critique import (
    ArtifactCritiqueSummary,
)
from creative_coding_assistant.orchestration.clarification import ClarificationRequest
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
from creative_coding_assistant.orchestration.routing import (
    RouteCapability,
    RouteDecision,
)
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityProfile,
)
from creative_coding_assistant.orchestration.workflow_review import (
    WorkflowReviewOutcome,
    WorkflowReviewResult,
)


def build_director_brief_payload(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    clarification: ClarificationRequest | None,
    retrieval_chunk_count: int,
    artifact_critique_summary: ArtifactCritiqueSummary | None,
    review_result: WorkflowReviewResult | None,
    refinement_count: int,
) -> dict[str, object]:
    retrieval_posture = _retrieval_posture(route_decision, retrieval_chunk_count)
    ambiguity_signals = _ambiguity_signals(
        route_decision=route_decision,
        creative_plan=creative_plan,
        clarification=clarification,
    )

    return {
        "creative_brief": _creative_brief(request, creative_translation),
        "ambiguity_level": _ambiguity_level(clarification, ambiguity_signals),
        "ambiguity_signals": ambiguity_signals,
        "retrieval_posture": retrieval_posture,
        "modality_direction": (
            creative_plan.output_modality.value if creative_plan is not None else None
        ),
        "runtime_direction": _runtime_direction(creative_plan),
        "planning_focus": _planning_focus(
            creative_plan,
            creative_strategy,
            creative_techniques,
            creative_constraints,
            runtime_capabilities,
        ),
        "critique_focus": _critique_focus(
            creative_plan=creative_plan,
            creative_strategy=creative_strategy,
            creative_techniques=creative_techniques,
            creative_constraints=creative_constraints,
            runtime_capabilities=runtime_capabilities,
            artifact_critique_summary=artifact_critique_summary,
            review_result=review_result,
        ),
        "refinement_focus": _refinement_focus(
            artifact_critique_summary=artifact_critique_summary,
            review_result=review_result,
            refinement_count=refinement_count,
        ),
        "next_actions": _next_actions(
            clarification=clarification,
            creative_plan=creative_plan,
            creative_constraints=creative_constraints,
            review_result=review_result,
            retrieval_posture=retrieval_posture,
        ),
        "hitl_required": clarification is not None,
        "hitl_reason": clarification.summary if clarification is not None else None,
        "evidence": _evidence(
            request=request,
            route_decision=route_decision,
            creative_translation=creative_translation,
            creative_strategy=creative_strategy,
            creative_techniques=creative_techniques,
            creative_plan=creative_plan,
            creative_constraints=creative_constraints,
            runtime_capabilities=runtime_capabilities,
            retrieval_chunk_count=retrieval_chunk_count,
            clarification=clarification,
            artifact_critique_summary=artifact_critique_summary,
            review_result=review_result,
            refinement_count=refinement_count,
        ),
    }


def _creative_brief(
    request: AssistantRequest,
    creative_translation: CreativeTranslation | None,
) -> str:
    if creative_translation is not None:
        return creative_translation.creative_intent
    return " ".join(request.query.split())[:360]


def _ambiguity_level(
    clarification: ClarificationRequest | None,
    ambiguity_signals: tuple[str, ...],
) -> str:
    if clarification is not None:
        return "high"
    if len(ambiguity_signals) >= 2:
        return "medium"
    return "low"


def _ambiguity_signals(
    *,
    route_decision: RouteDecision | None,
    creative_plan: CreativeExecutionPlan | None,
    clarification: ClarificationRequest | None,
) -> tuple[str, ...]:
    signals: list[str] = []
    if clarification is not None:
        signals.append(f"Clarification required: {clarification.reason.value}.")
    if route_decision is not None and len(route_decision.domains) > 1:
        signals.append("Multiple effective domains require explicit bridging.")
    if route_decision is not None and not route_decision.domains:
        signals.append("Runtime/domain direction is inferred rather than selected.")
    if creative_plan is not None and not creative_plan.runtime_available:
        signals.append("No live preview runtime is available for the selected scope.")
    return tuple(signals[:8])


def _retrieval_posture(
    route_decision: RouteDecision | None,
    retrieval_chunk_count: int,
) -> str:
    if retrieval_chunk_count > 0:
        return "available"
    if (
        route_decision is not None
        and RouteCapability.OFFICIAL_DOCS in route_decision.capabilities
    ):
        return "useful"
    return "not_requested"


def _runtime_direction(plan: CreativeExecutionPlan | None) -> str | None:
    if plan is None:
        return None
    if plan.recommended_runtime is not None:
        return plan.recommended_runtime
    return plan.runtime_support_summary


def _planning_focus(
    plan: CreativeExecutionPlan | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_constraints: CreativeConstraintSolution | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
) -> tuple[str, ...]:
    focus: list[str] = []
    if creative_strategy is not None:
        focus.append(f"High-level strategy: {creative_strategy.primary_strategy}.")
    if creative_techniques is not None:
        focus.append(
            f"Primary technique: {creative_techniques.primary_technique}."
        )
    if runtime_capabilities is not None:
        focus.append(
            "Runtime capability candidates: "
            + ", ".join(runtime_capabilities.likely_candidates)
            + "."
        )
        focus.extend(runtime_capabilities.prompt_guidance[:2])
    if creative_strategy is not None:
        focus.extend(creative_strategy.strategy_directives[:2])
    if creative_techniques is not None:
        focus.extend(creative_techniques.implementation_notes[:2])
    if creative_constraints is not None:
        focus.extend(creative_constraints.prompt_guidance[:2])
    if plan is None:
        focus.extend(
            (
                "Preserve the user's creative brief as the source of truth.",
                "Keep guidance bounded to the selected route and domains.",
            )
        )
        return _dedupe_text(focus)[:6]
    focus.extend([plan.generation_strategy, *plan.plan_steps[:3]])
    if plan.constraints:
        focus.append(f"Primary constraint: {plan.constraints[0]}")
    return _dedupe_text(focus)[:6]


def _critique_focus(
    *,
    creative_plan: CreativeExecutionPlan | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_constraints: CreativeConstraintSolution | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    artifact_critique_summary: ArtifactCritiqueSummary | None,
    review_result: WorkflowReviewResult | None,
) -> tuple[str, ...]:
    focus: list[str] = []
    if creative_plan is not None:
        focus.append(
            "Check output against runtime support, domain scope, and plan constraints."
        )
    if creative_strategy is not None:
        focus.append(f"Strategy rationale: {creative_strategy.rationale}")
    if creative_techniques is not None:
        focus.append(f"Technique rationale: {creative_techniques.rationale}")
    if creative_constraints is not None:
        focus.extend(creative_constraints.conflicts[:2])
        focus.extend(
            tradeoff.summary for tradeoff in creative_constraints.tradeoffs[:2]
        )
    if runtime_capabilities is not None:
        focus.append(
            "Runtime capability reasoner is non-binding; verify output stays "
            "inside selected route/runtime contract."
        )
        focus.extend(runtime_capabilities.candidate_runtimes[0].risks[:2])
    if artifact_critique_summary is not None:
        focus.append(
            "Recommended artifact: "
            f"{artifact_critique_summary.recommended_artifact_title or 'none'}."
        )
        focus.append(
            f"Artifact average score: {artifact_critique_summary.average_score:.2f}."
        )
    if review_result is not None:
        focus.append(f"Workflow review outcome: {review_result.outcome.value}.")
        focus.append(review_result.rationale)
    if not focus:
        focus.append("Review generated output before finalization.")
    return _dedupe_text(focus)[:6]


def _refinement_focus(
    *,
    artifact_critique_summary: ArtifactCritiqueSummary | None,
    review_result: WorkflowReviewResult | None,
    refinement_count: int,
) -> tuple[str, ...]:
    focus: list[str] = []
    if artifact_critique_summary and artifact_critique_summary.refinement_required:
        focus.extend(artifact_critique_summary.refinement_reasons)
        if artifact_critique_summary.refinement_guidance:
            focus.append(artifact_critique_summary.refinement_guidance)
    if (
        review_result is not None
        and review_result.outcome is WorkflowReviewOutcome.NEEDS_REFINEMENT
    ):
        focus.extend(review_result.reasons)
    if refinement_count > 0:
        focus.append(f"Completed refinement pass count: {refinement_count}.")
    if not focus:
        focus.append("Use bounded refinement only when review signals a concrete gap.")
    return _dedupe_text(focus)[:6]


def _next_actions(
    *,
    clarification: ClarificationRequest | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    review_result: WorkflowReviewResult | None,
    retrieval_posture: str,
) -> tuple[str, ...]:
    if clarification is not None:
        return ("Ask the listed HITL clarification before generation.",)
    if creative_constraints is not None and creative_constraints.hitl_advisable:
        return (
            creative_constraints.hitl_reason
            or "Surface the unresolved constraint trade-off before generation.",
        )
    if (
        review_result is not None
        and review_result.outcome is WorkflowReviewOutcome.NEEDS_REFINEMENT
    ):
        return ("Prepare bounded refinement guidance for the next generation pass.",)
    actions = [
        "Render the prompt and continue through the deterministic workflow."
        if creative_plan is not None
        else "Continue with the available workflow context.",
    ]
    if retrieval_posture in {"available", "useful"}:
        actions.append("Use official KB context when it is available and relevant.")
    actions.append("Keep final creative choices visible to the user.")
    return tuple(actions[:6])


def _evidence(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    retrieval_chunk_count: int,
    clarification: ClarificationRequest | None,
    artifact_critique_summary: ArtifactCritiqueSummary | None,
    review_result: WorkflowReviewResult | None,
    refinement_count: int,
) -> tuple[str, ...]:
    evidence = [f"Mode: {request.mode.value}."]
    if route_decision is not None:
        evidence.append(f"Route selected: {route_decision.route.value}.")
        domains = route_decision.domains or _request_domains(request)
        if domains:
            evidence.append(
                "Domains: " + ", ".join(domain.value for domain in domains) + "."
            )
    if creative_translation is not None:
        evidence.append(f"Creative intent: {creative_translation.creative_intent}.")
    if creative_strategy is not None:
        evidence.append(f"Creative strategy: {creative_strategy.primary_strategy}.")
        evidence.append(f"Strategy confidence: {creative_strategy.confidence:.2f}.")
    if creative_techniques is not None:
        evidence.append(
            f"Creative technique: {creative_techniques.primary_technique}."
        )
        evidence.append(f"Technique confidence: {creative_techniques.confidence:.2f}.")
    if creative_plan is not None:
        evidence.append(f"Plan complexity: {creative_plan.expected_complexity}.")
        evidence.append(f"Export readiness: {creative_plan.export_readiness}.")
    if creative_constraints is not None:
        evidence.append(
            "Constraint solver: "
            f"{len(creative_constraints.active_constraints)} active constraint(s)."
        )
        evidence.append(
            f"Runtime fit: {creative_constraints.runtime_fit}."
        )
    if runtime_capabilities is not None:
        evidence.append(
            "Runtime capability candidates: "
            + ", ".join(runtime_capabilities.likely_candidates)
            + "."
        )
        evidence.append(
            f"Runtime capability HITL: {runtime_capabilities.hitl_advisable}."
        )
    if retrieval_chunk_count:
        evidence.append(f"Retrieval chunks: {retrieval_chunk_count}.")
    if clarification is not None:
        evidence.append(f"HITL reason: {clarification.reason.value}.")
    if artifact_critique_summary is not None:
        evidence.append(
            f"Artifact critique average: {artifact_critique_summary.average_score:.2f}."
        )
    if review_result is not None:
        evidence.append(f"Review outcome: {review_result.outcome.value}.")
    if refinement_count:
        evidence.append(f"Refinement count: {refinement_count}.")
    return _dedupe_text(evidence)[:10]


def _request_domains(
    request: AssistantRequest,
) -> tuple[CreativeCodingDomain, ...]:
    if request.domains:
        return request.domains
    if request.domain is not None:
        return (request.domain,)
    return ()


def _dedupe_text(values: list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        cleaned = " ".join(value.strip().split())
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return tuple(normalized)
