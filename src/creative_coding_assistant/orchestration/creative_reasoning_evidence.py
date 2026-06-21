"""Evidence-chain construction for Creative Reasoning Engine."""

from __future__ import annotations

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
    CreativeReasoningEvidence,
)
from creative_coding_assistant.orchestration.creative_reasoning_signals import (
    _tradeoff_summary,
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


def build_evidence_chain(
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
) -> tuple[CreativeReasoningEvidence, ...]:
    evidence = [
        CreativeReasoningEvidence(
            source="request",
            signal=request.query[:240],
            interpretation="The recommendation must preserve stated intent.",
        )
    ]
    if route_decision is not None:
        domains = ", ".join(item.value for item in route_decision.domains) or "none"
        evidence.append(
            CreativeReasoningEvidence(
                source="planning",
                signal=f"Route {route_decision.route.value}; domains {domains}.",
                interpretation="Reasoning must stay inside route and domain scope.",
            )
        )
    if creative_translation is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="translation",
                signal=creative_translation.creative_intent,
                interpretation="Creative translation supplies intent to protect.",
            )
        )
    _append_strategy_evidence(evidence, creative_strategy)
    _append_technique_evidence(evidence, creative_techniques)
    if creative_plan is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="planning",
                signal=creative_plan.generation_strategy,
                interpretation="The output goal constrains executable scope.",
            )
        )
    _append_constraint_evidence(evidence, creative_constraints)
    if runtime_capabilities is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="runtime_capability",
                signal=", ".join(runtime_capabilities.likely_candidates),
                interpretation=(
                    "Runtime evidence informs feasibility without selecting runtime."
                ),
            )
        )
    if creative_tradeoffs is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="tradeoff_explorer",
                signal=_tradeoff_summary(creative_tradeoffs),
                interpretation="Trade-off evidence explains the bounded stance.",
            )
        )
    if creative_director is not None:
        evidence.append(
            CreativeReasoningEvidence(
                source="director",
                signal=creative_director.creative_brief,
                interpretation="Director guidance frames brief and HITL posture.",
            )
        )
    return tuple(evidence[:10])


def _append_strategy_evidence(
    evidence: list[CreativeReasoningEvidence],
    profile: CreativeStrategyProfile | None,
) -> None:
    if profile is None:
        return
    evidence.append(
        CreativeReasoningEvidence(
            source="creative_strategy",
            signal=f"{profile.primary_strategy} confidence {profile.confidence:.2f}.",
            interpretation=profile.rationale,
        )
    )


def _append_technique_evidence(
    evidence: list[CreativeReasoningEvidence],
    profile: CreativeTechniqueProfile | None,
) -> None:
    if profile is None:
        return
    evidence.append(
        CreativeReasoningEvidence(
            source="creative_technique",
            signal=(
                f"{profile.primary_technique} "
                f"compatibility {profile.compatibility}."
            ),
            interpretation="Technique shows how strategy becomes behavior.",
        )
    )


def _append_constraint_evidence(
    evidence: list[CreativeReasoningEvidence],
    profile: CreativeConstraintSolution | None,
) -> None:
    if profile is None:
        return
    evidence.append(
        CreativeReasoningEvidence(
            source="constraint_solver",
            signal=(
                f"complexity {profile.complexity_pressure}; "
                f"performance {profile.performance_pressure}; "
                f"safety {profile.safety_pressure}."
            ),
            interpretation="Constraint pressures bound the recommendation.",
        )
    )
