"""Signal synthesis helpers for Creative Reasoning Engine."""

from __future__ import annotations

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration.creative_constraint_priorities import (
    CreativeConstraintPrioritization,
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
from creative_coding_assistant.orchestration.creative_reasoning_contracts import (
    CreativeReasoningStep,
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
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityCandidate,
    RuntimeCapabilityProfile,
)


def build_recommended_direction(
    *,
    request: AssistantRequest,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_translation: CreativeTranslation | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
    creative_constraint_priorities: CreativeConstraintPrioritization | None,
) -> str:
    intent = (
        creative_intent.primary_expression
        if creative_intent is not None
        else _translation_or_request_intent(creative_translation, request)
    )
    output_goal = (
        creative_plan.generation_strategy
        if creative_plan is not None
        else "Produce a bounded creative-coding response."
    )
    direction = (
        f"Recommend {_strategy_label(creative_strategy)} via "
        f"{_technique_label(creative_techniques)} because it protects "
        f"'{_clip(intent, 70)}'. Prioritize "
        f"{_clip(_hierarchy_label(creative_hierarchy), 70)}. "
        f"Protect constraints: "
        f"{_clip(_constraint_priority_label(creative_constraint_priorities), 70)}. "
        f"Fit the output goal: {_clip(output_goal, 90)} "
        f"Use inspected runtime guidance: "
        f"{_clip(_runtime_label(runtime_capabilities, creative_plan), 70)}. "
        f"Bound the trade-off: {_clip(_tradeoff_summary(creative_tradeoffs), 100)}"
    )
    return _clip(direction, 500)


def build_reasoning_path(
    *,
    direction: str,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_plan: CreativeExecutionPlan | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
    creative_constraint_priorities: CreativeConstraintPrioritization | None,
) -> tuple[CreativeReasoningStep, ...]:
    strategy = _strategy_label(creative_strategy)
    technique = _technique_label(creative_techniques)
    return (
        CreativeReasoningStep(
            stage="strategy",
            claim=f"Use {strategy} as the conceptual spine.",
            because=(
                _strategy_reason(
                    creative_strategy,
                    creative_intent,
                    creative_hierarchy,
                )
                if creative_strategy is not None
                else "No strategy profile is available, so user intent leads."
            ),
            implications=("Keep one visible creative idea in focus.",),
        ),
        CreativeReasoningStep(
            stage="technique",
            claim=f"Translate that strategy through {technique}.",
            because=_technique_reason(creative_techniques, strategy),
            implications=("Technique choices should serve the strategy.",),
        ),
        CreativeReasoningStep(
            stage="runtime",
            claim=(
                "Shape implementation around inspected capability: "
                f"{_runtime_label(runtime_capabilities, creative_plan)}."
            ),
            because=_runtime_reason(runtime_capabilities, creative_plan),
            implications=("Runtime guidance remains non-binding.",),
        ),
        CreativeReasoningStep(
            stage="tradeoff",
            claim=(
                "Manage the main consequence: "
                f"{_tradeoff_summary(creative_tradeoffs)}"
            ),
            because=_tradeoff_reason(
                creative_tradeoffs,
                creative_constraint_priorities,
            ),
            implications=(
                "Prefer bounded implementation over feature growth.",
                _constraint_priority_implication(creative_constraint_priorities),
            ),
        ),
        CreativeReasoningStep(
            stage="recommendation",
            claim=direction,
            because=(
                "Strategy, technique, runtime capability, and trade-off "
                "signals converge on the same bounded direction."
            ),
            implications=("Use this as the prompt spine before generation.",),
        ),
    )


def _top_runtime(
    profile: RuntimeCapabilityProfile | None,
) -> RuntimeCapabilityCandidate | None:
    if profile is None or not profile.candidate_runtimes:
        return None
    return profile.candidate_runtimes[0]


def _strategy_label(profile: CreativeStrategyProfile | None) -> str:
    return profile.primary_strategy if profile is not None else "bounded creative"


def _translation_or_request_intent(
    creative_translation: CreativeTranslation | None,
    request: AssistantRequest,
) -> str:
    if creative_translation is not None:
        return creative_translation.creative_intent
    return request.query


def _hierarchy_label(profile: CreativeHierarchyPlan | None) -> str:
    if profile is None:
        return "the current creative hierarchy"
    return ", ".join(item.dimension for item in profile.primary_creative_priorities)


def _constraint_priority_label(
    profile: CreativeConstraintPrioritization | None,
) -> str:
    if profile is None or not profile.non_negotiable_constraints:
        return "the current constraint priority order"
    return ", ".join(item.category for item in profile.non_negotiable_constraints)


def _constraint_priority_implication(
    profile: CreativeConstraintPrioritization | None,
) -> str:
    if profile is None or not profile.sacrificial_constraints:
        return "Constraint priority remains advisory."
    sacrificed = ", ".join(
        item.category for item in profile.sacrificial_constraints[:2]
    )
    return f"Relax {sacrificed} before protected constraints."


def _technique_label(profile: CreativeTechniqueProfile | None) -> str:
    return profile.primary_technique if profile is not None else "minimal viable"


def _runtime_label(
    profile: RuntimeCapabilityProfile | None,
    plan: CreativeExecutionPlan | None,
) -> str:
    top = _top_runtime(profile)
    if top is not None:
        return f"{top.label} ({top.suitability} inspected fit)"
    if plan is not None and plan.recommended_runtime is not None:
        return f"{plan.recommended_runtime} from the existing execution plan"
    return "the inspected runtime capability context"


def _tradeoff_summary(profile: CreativeTradeoffProfile | None) -> str:
    if profile is None:
        return "preserve intent while keeping implementation scope bounded."
    tradeoff = profile.primary_tradeoffs[0]
    return f"{tradeoff.source_axis} vs {tradeoff.target_axis}: {tradeoff.summary}"


def _technique_reason(
    profile: CreativeTechniqueProfile | None,
    strategy: str,
) -> str:
    if profile is None:
        return "No technique profile is available, so stay minimal."
    return (
        f"{profile.rationale} This connects the technique to {strategy} "
        f"with {profile.compatibility} compatibility."
    )


def _strategy_reason(
    profile: CreativeStrategyProfile,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
) -> str:
    details: list[str] = [profile.rationale]
    if creative_intent is not None:
        details.append(
            "Decomposed intent substrate: "
            f"{_clip(creative_intent.primary_expression, 120)}."
        )
    if creative_hierarchy is not None:
        details.append(
            "Hierarchy priorities: "
            f"{_clip(_hierarchy_label(creative_hierarchy), 120)}."
        )
    return (
        " ".join(details)
    )


def _runtime_reason(
    profile: RuntimeCapabilityProfile | None,
    plan: CreativeExecutionPlan | None,
) -> str:
    top = _top_runtime(profile)
    if top is not None:
        return (
            f"{top.label} shows {top.suitability} suitability, "
            f"{top.technique_compatibility} technique compatibility, and "
            f"{top.preview_support} preview support."
        )
    if plan is not None:
        return plan.runtime_support_summary
    return "No runtime capability profile is available, so avoid runtime claims."


def _tradeoff_reason(
    profile: CreativeTradeoffProfile | None,
    priorities: CreativeConstraintPrioritization | None,
) -> str:
    if profile is None:
        return "No trade-off profile is available; stay conservative."
    tradeoff = profile.primary_tradeoffs[0]
    reason = (
        f"The creative benefit is '{tradeoff.creative_benefit}' while the "
        f"technical cost is '{tradeoff.technical_cost}'."
    )
    if priorities is not None and priorities.negotiation_notes:
        reason += f" Constraint priority says: {priorities.negotiation_notes[0]}"
    return reason


def _clip(value: str, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "."
