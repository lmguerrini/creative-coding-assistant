"""Bounded Creative Reasoning Engine public API."""

from __future__ import annotations

from collections.abc import Mapping

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration.creative_constraints import (
    CreativeConstraintSolution,
)
from creative_coding_assistant.orchestration.creative_director import (
    CreativeAssistantDirectorBrief,
)
from creative_coding_assistant.orchestration.creative_planning import (
    CreativeExecutionPlan,
)
from creative_coding_assistant.orchestration.creative_reasoning_contracts import (
    CREATIVE_REASONING_AUTHORITY_BOUNDARY,
    CreativeReasoningEvidence,
    CreativeReasoningResult,
    CreativeReasoningStep,
    CreativeRejectedAlternative,
)
from creative_coding_assistant.orchestration.creative_reasoning_evidence import (
    build_evidence_chain,
)
from creative_coding_assistant.orchestration.creative_reasoning_signals import (
    build_reasoning_path,
    build_recommended_direction,
)
from creative_coding_assistant.orchestration.creative_reasoning_support import (
    build_hitl_questions,
    build_implementation_guidance,
    build_prompt_guidance,
    build_rejected_alternatives,
    build_strongest_signals,
    build_unresolved_decisions,
    normalize_future_knowledge_context,
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
    RuntimeCapabilityProfile,
)


def derive_creative_reasoning_result(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_director: CreativeAssistantDirectorBrief | None,
    creative_constraints: CreativeConstraintSolution | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
    future_knowledge_context: Mapping[str, object] | None = None,
) -> CreativeReasoningResult:
    """Synthesize prior Creative Intelligence outputs into one decision brief."""

    direction = build_recommended_direction(
        request=request,
        creative_translation=creative_translation,
        creative_plan=creative_plan,
        creative_strategy=creative_strategy,
        creative_techniques=creative_techniques,
        runtime_capabilities=runtime_capabilities,
        creative_tradeoffs=creative_tradeoffs,
    )
    unresolved = build_unresolved_decisions(
        creative_director=creative_director,
        creative_constraints=creative_constraints,
        runtime_capabilities=runtime_capabilities,
        creative_tradeoffs=creative_tradeoffs,
        creative_strategy=creative_strategy,
        creative_techniques=creative_techniques,
    )
    return CreativeReasoningResult(
        recommended_creative_direction=direction,
        reasoning_path=build_reasoning_path(
            direction=direction,
            creative_strategy=creative_strategy,
            creative_techniques=creative_techniques,
            creative_plan=creative_plan,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=creative_tradeoffs,
        ),
        evidence_chain=build_evidence_chain(
            request=request,
            route_decision=route_decision,
            creative_translation=creative_translation,
            creative_plan=creative_plan,
            creative_director=creative_director,
            creative_constraints=creative_constraints,
            creative_strategy=creative_strategy,
            creative_techniques=creative_techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=creative_tradeoffs,
        ),
        strongest_supporting_signals=build_strongest_signals(
            creative_director=creative_director,
            creative_constraints=creative_constraints,
            creative_strategy=creative_strategy,
            creative_techniques=creative_techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=creative_tradeoffs,
        ),
        rejected_alternatives=build_rejected_alternatives(
            creative_strategy=creative_strategy,
            creative_techniques=creative_techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=creative_tradeoffs,
        ),
        unresolved_decisions=unresolved,
        implementation_guidance=build_implementation_guidance(
            creative_plan=creative_plan,
            creative_constraints=creative_constraints,
            creative_techniques=creative_techniques,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=creative_tradeoffs,
        ),
        prompt_guidance=build_prompt_guidance(unresolved),
        hitl_questions=build_hitl_questions(unresolved),
        future_knowledge_context=normalize_future_knowledge_context(
            future_knowledge_context
        ),
    )


def creative_reasoning_prompt_lines(
    result: CreativeReasoningResult,
) -> tuple[str, ...]:
    """Render the reasoning brief as compact provider prompt guidance."""

    lines = [
        f"Authority boundary: {result.authority_boundary}",
        f"Reasoning recommendation: {result.recommended_creative_direction}",
    ]
    for step in result.reasoning_path:
        lines.append(
            f"Reasoning path ({step.stage}): {step.claim} Because {step.because}"
        )
        lines.extend(f"Reasoning implication: {item}" for item in step.implications[:2])
    lines.extend(
        f"Supporting signal: {item}"
        for item in result.strongest_supporting_signals[:5]
    )
    lines.extend(
        f"Rejected alternative: {item.alternative}; {item.reason}"
        for item in result.rejected_alternatives[:3]
    )
    lines.extend(
        f"Unresolved decision: {item}" for item in result.unresolved_decisions[:4]
    )
    lines.extend(
        f"Implementation guidance: {item}"
        for item in result.implementation_guidance[:5]
    )
    lines.extend(f"Prompt guidance: {item}" for item in result.prompt_guidance[:5])
    lines.extend(f"HITL question: {item}" for item in result.hitl_questions[:3])
    return tuple(lines[:32])


__all__ = [
    "CREATIVE_REASONING_AUTHORITY_BOUNDARY",
    "CreativeReasoningEvidence",
    "CreativeReasoningResult",
    "CreativeReasoningStep",
    "CreativeRejectedAlternative",
    "creative_reasoning_prompt_lines",
    "derive_creative_reasoning_result",
]
