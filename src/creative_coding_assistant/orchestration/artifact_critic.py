"""Bounded Artifact Critic for V3.3 planning metadata."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration._metadata_utils import (
    _contains_any,
    _dedupe,
)
from creative_coding_assistant.orchestration.artifact_capability_matrix import (
    ArtifactCapabilityMatrix,
)
from creative_coding_assistant.orchestration.artifact_dependency_graph import (
    ArtifactDependencyGraph,
)
from creative_coding_assistant.orchestration.artifact_planner import ArtifactPlan
from creative_coding_assistant.orchestration.multi_artifact_strategy import (
    MultiArtifactStrategy,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.runtime_compatibility import (
    RuntimeCompatibilityProfile,
)

ArtifactCriticRiskAssessment = Literal["low", "medium", "high", "blocked"]

ARTIFACT_CRITIC_AUTHORITY_BOUNDARY = (
    "The Artifact Critic evaluates planning metadata only; it may identify "
    "issues and suggest improvements, but it does not modify artifacts, "
    "choose a strategy, reject a strategy, refine a strategy, merge artifacts, "
    "execute artifacts or runtimes, select runtimes, change routing, change "
    "previews, activate workflows, trigger retries, route providers or models, "
    "implement Artifact Refiner, implement Artifact Merge Planner, implement "
    "Artifact Export Intelligence, implement V4 multi-agent behavior, or "
    "implement V5 execution optimization."
)


class ArtifactCriticProfile(BaseModel):
    """Inspectable metadata-only critique of planned artifact metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["artifact_critic"] = "artifact_critic"
    critique_confidence: float = Field(ge=0, le=1)
    critique_summary: str = Field(min_length=1, max_length=520)
    strengths: tuple[str, ...] = Field(min_length=1, max_length=10)
    weaknesses: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    capability_gaps: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    dependency_concerns: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    runtime_concerns: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    scalability_concerns: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    maintainability_concerns: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    complexity_concerns: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    risk_assessment: ArtifactCriticRiskAssessment
    unsupported_assumptions: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    missing_information: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    open_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    improvement_opportunities: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=10,
    )
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=ARTIFACT_CRITIC_AUTHORITY_BOUNDARY,
        max_length=900,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=12)


def derive_artifact_critic_profile(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
) -> ArtifactCriticProfile:
    """Critique artifact planning metadata without changing workflow behavior."""

    strengths = _strengths(
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
    )
    capability_gaps = _capability_gaps(artifact_capability_matrix)
    dependency_concerns = _dependency_concerns(artifact_dependency_graph)
    runtime_concerns = _runtime_concerns(runtime_compatibility)
    scalability_concerns = _scalability_concerns(
        artifact_plan=artifact_plan,
        artifact_capability_matrix=artifact_capability_matrix,
        runtime_compatibility=runtime_compatibility,
    )
    maintainability_concerns = _maintainability_concerns(
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        multi_artifact_strategy=multi_artifact_strategy,
    )
    complexity_concerns = _complexity_concerns(
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_compatibility=runtime_compatibility,
        multi_artifact_strategy=multi_artifact_strategy,
    )
    unsupported = _unsupported_assumptions(
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
    )
    missing = _missing_information(
        route_decision=route_decision,
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
    )
    weaknesses = _weaknesses(
        capability_gaps=capability_gaps,
        dependency_concerns=dependency_concerns,
        runtime_concerns=runtime_concerns,
        scalability_concerns=scalability_concerns,
        maintainability_concerns=maintainability_concerns,
        complexity_concerns=complexity_concerns,
        unsupported_assumptions=unsupported,
        missing_information=missing,
    )
    risk = _risk_assessment(
        weaknesses=weaknesses,
        missing=missing,
        dependency_concerns=dependency_concerns,
        runtime_concerns=runtime_concerns,
        unsupported_assumptions=unsupported,
    )
    open_questions = _open_questions(
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
        missing=missing,
    )
    improvement_opportunities = _improvement_opportunities(
        risk=risk,
        capability_gaps=capability_gaps,
        dependency_concerns=dependency_concerns,
        runtime_concerns=runtime_concerns,
        multi_artifact_strategy=multi_artifact_strategy,
    )
    return ArtifactCriticProfile(
        critique_confidence=_critique_confidence(
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            multi_artifact_strategy=multi_artifact_strategy,
            missing=missing,
        ),
        critique_summary=_critique_summary(
            risk=risk,
            strengths=strengths,
            weaknesses=weaknesses,
        ),
        strengths=strengths,
        weaknesses=weaknesses,
        capability_gaps=capability_gaps,
        dependency_concerns=dependency_concerns,
        runtime_concerns=runtime_concerns,
        scalability_concerns=scalability_concerns,
        maintainability_concerns=maintainability_concerns,
        complexity_concerns=complexity_concerns,
        risk_assessment=risk,
        unsupported_assumptions=unsupported,
        missing_information=missing,
        open_questions=open_questions,
        hitl_questions=_hitl_questions(
            open_questions=open_questions,
            risk=risk,
            missing=missing,
        ),
        improvement_opportunities=improvement_opportunities,
        prompt_guidance=_prompt_guidance(
            risk=risk,
            improvement_opportunities=improvement_opportunities,
        ),
        evidence=_evidence(
            request=request,
            route_decision=route_decision,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            multi_artifact_strategy=multi_artifact_strategy,
        ),
    )


def artifact_critic_prompt_lines(
    profile: ArtifactCriticProfile,
) -> tuple[str, ...]:
    """Render Artifact Critic metadata as compact prompt guidance."""

    lines = [
        f"Authority boundary: {profile.authority_boundary}",
        f"Critique confidence: {profile.critique_confidence:.2f}.",
        f"Critique risk assessment: {profile.risk_assessment}.",
        f"Critique summary: {profile.critique_summary}",
    ]
    lines.extend(f"Artifact critic strength: {item}" for item in profile.strengths)
    lines.extend(f"Artifact critic weakness: {item}" for item in profile.weaknesses)
    lines.extend(f"Capability gap: {item}" for item in profile.capability_gaps)
    lines.extend(
        f"Dependency concern: {item}" for item in profile.dependency_concerns
    )
    lines.extend(f"Runtime concern: {item}" for item in profile.runtime_concerns)
    lines.extend(
        f"Scalability concern: {item}" for item in profile.scalability_concerns
    )
    lines.extend(
        f"Maintainability concern: {item}"
        for item in profile.maintainability_concerns
    )
    lines.extend(
        f"Complexity concern: {item}" for item in profile.complexity_concerns
    )
    lines.extend(
        f"Unsupported assumption: {item}"
        for item in profile.unsupported_assumptions
    )
    lines.extend(
        f"Missing critic information: {item}" for item in profile.missing_information
    )
    lines.extend(f"Open critic question: {item}" for item in profile.open_questions)
    lines.extend(f"HITL critic question: {item}" for item in profile.hitl_questions)
    lines.extend(
        f"Improvement opportunity: {item}"
        for item in profile.improvement_opportunities
    )
    lines.extend(
        f"Artifact critic guidance: {item}" for item in profile.prompt_guidance
    )
    return tuple(lines[:64])


def _strengths(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
) -> tuple[str, ...]:
    strengths: list[str] = []
    if artifact_plan is not None:
        strengths.append(
            f"Artifact plan declares {artifact_plan.artifact_type} / "
            f"{artifact_plan.artifact_family} with "
            f"{len(artifact_plan.required_components)} required component(s)."
        )
    if artifact_dependency_graph is not None:
        strengths.append(
            "Dependency graph exposes "
            f"{len(artifact_dependency_graph.artifact_nodes)} node(s) and "
            f"{len(artifact_dependency_graph.dependency_edges)} edge(s)."
        )
    if runtime_compatibility is not None and runtime_compatibility.preferred_runtimes:
        strengths.append(
            "Runtime compatibility has preferred runtime metadata: "
            + ", ".join(runtime_compatibility.preferred_runtimes)
            + "."
        )
    if (
        artifact_capability_matrix is not None
        and artifact_capability_matrix.strongest_targets
    ):
        strengths.append(
            "Capability matrix identifies strongest targets: "
            + ", ".join(artifact_capability_matrix.strongest_targets)
            + "."
        )
    if multi_artifact_strategy is not None:
        strengths.append(
            "Multi-artifact strategy orders "
            f"{len(multi_artifact_strategy.artifact_sequence)} step(s) with "
            f"{len(multi_artifact_strategy.supporting_artifacts)} supporting role(s)."
        )
    strengths.append("Critique remains metadata-only and non-executing.")
    return _dedupe(strengths)[:10]


def _capability_gaps(
    matrix: ArtifactCapabilityMatrix | None,
) -> tuple[str, ...]:
    if matrix is None:
        return ("Artifact Capability Matrix metadata is unavailable.",)
    gaps: list[str] = []
    gaps.extend(matrix.unsupported_or_risky_capabilities[:4])
    gaps.extend(matrix.target_weaknesses[:3])
    if matrix.artifact_fit in {"weak", "unsupported"}:
        gaps.append(f"Overall artifact fit is {matrix.artifact_fit}.")
    if matrix.export_fit in {"weak", "unsupported"}:
        gaps.append(f"Export fit is {matrix.export_fit}; avoid export claims.")
    return _dedupe(gaps)[:10]


def _dependency_concerns(
    graph: ArtifactDependencyGraph | None,
) -> tuple[str, ...]:
    if graph is None:
        return ("Artifact Dependency Graph metadata is unavailable.",)
    concerns: list[str] = []
    concerns.extend(graph.blocking_dependencies[:3])
    concerns.extend(graph.dependency_conflicts[:3])
    concerns.extend(graph.missing_dependency_risks[:3])
    if len(graph.dependency_edges) > 8:
        concerns.append("Dependency graph has many edges; keep handoffs inspectable.")
    return _dedupe(concerns)[:10]


def _runtime_concerns(
    profile: RuntimeCompatibilityProfile | None,
) -> tuple[str, ...]:
    if profile is None:
        return ("Runtime Compatibility Engine metadata is unavailable.",)
    concerns: list[str] = []
    if profile.unsupported_runtimes:
        concerns.append(
            "Unsupported runtimes should not be treated as viable targets: "
            + ", ".join(profile.unsupported_runtimes[:4])
            + "."
        )
    concerns.extend(profile.implementation_risks[:3])
    concerns.extend(profile.runtime_limitations[:3])
    concerns.extend(profile.missing_runtime_information[:2])
    return _dedupe(concerns)[:10]


def _scalability_concerns(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
) -> tuple[str, ...]:
    source_items: list[str] = []
    if artifact_plan is not None:
        source_items.extend(artifact_plan.implementation_risks)
        source_items.extend(artifact_plan.runtime_requirements)
    if artifact_capability_matrix is not None:
        source_items.extend(artifact_capability_matrix.target_weaknesses)
        source_items.extend(artifact_capability_matrix.capability_risks)
    if runtime_compatibility is not None:
        source_items.extend(runtime_compatibility.implementation_risks)
    concerns = [
        item
        for item in source_items
        if _contains_any(
            item,
            ("scale", "performance", "frame", "particle", "dense", "large"),
        )
    ]
    return _dedupe(concerns)[:8]


def _maintainability_concerns(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
) -> tuple[str, ...]:
    concerns: list[str] = []
    if artifact_plan is not None and len(artifact_plan.required_components) > 6:
        concerns.append(
            "Many required components may reduce maintainability without clear "
            "sections."
        )
    if (
        artifact_dependency_graph is not None
        and len(artifact_dependency_graph.artifact_nodes) > 10
    ):
        concerns.append(
            "Many dependency nodes may require clearer metadata grouping."
        )
    if (
        multi_artifact_strategy is not None
        and len(multi_artifact_strategy.supporting_artifacts) > 3
    ):
        concerns.append(
            "Multiple supporting artifacts may require strict section labels."
        )
    if artifact_plan is not None:
        concerns.extend(
            item
            for item in artifact_plan.implementation_risks
            if _contains_any(item, ("maintain", "inspect", "complex", "boilerplate"))
        )
    return _dedupe(concerns)[:8]


def _complexity_concerns(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
) -> tuple[str, ...]:
    concerns: list[str] = []
    if (
        runtime_compatibility is not None
        and runtime_compatibility.expected_implementation_complexity == "high"
    ):
        concerns.append(
            "Runtime compatibility indicates high implementation complexity."
        )
    if artifact_plan is not None:
        concerns.extend(
            item
            for item in artifact_plan.implementation_risks
            if _contains_any(item, ("complex", "scope", "boilerplate", "3d"))
        )
    if (
        artifact_dependency_graph is not None
        and artifact_dependency_graph.blocking_dependencies
    ):
        concerns.append("Blocking dependencies raise planning complexity.")
    if (
        multi_artifact_strategy is not None
        and multi_artifact_strategy.combination_mode == "separated_parallel_sections"
    ):
        concerns.append("Separated parallel sections require explicit ordering.")
    return _dedupe(concerns)[:8]


def _unsupported_assumptions(
    *,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
) -> tuple[str, ...]:
    assumptions: list[str] = [
        "Artifact Critic findings are advisory and must not reject or refine strategy."
    ]
    if runtime_compatibility is not None and runtime_compatibility.unsupported_runtimes:
        assumptions.append(
            "Unsupported runtimes are caveats, not alternate selected targets: "
            + ", ".join(runtime_compatibility.unsupported_runtimes[:3])
            + "."
        )
    if artifact_capability_matrix is not None:
        assumptions.extend(
            f"Capability caveat remains non-executing: {item}"
            for item in artifact_capability_matrix.unsupported_or_risky_capabilities[:2]
        )
    if multi_artifact_strategy is not None:
        assumptions.append(
            "Supporting artifact roles must not be treated as generated variants."
        )
    return _dedupe(assumptions)[:8]


def _missing_information(
    *,
    route_decision: RouteDecision | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
) -> tuple[str, ...]:
    missing: list[str] = []
    if route_decision is None or not route_decision.domains:
        missing.append("Route/domain metadata is inferred or unavailable.")
    if artifact_plan is None:
        missing.append("Artifact Plan metadata is unavailable.")
    else:
        missing.extend(artifact_plan.missing_information[:3])
    if artifact_dependency_graph is None:
        missing.append("Artifact Dependency Graph metadata is unavailable.")
    else:
        missing.extend(artifact_dependency_graph.missing_dependency_risks[:2])
    if runtime_compatibility is None:
        missing.append("Runtime Compatibility Engine metadata is unavailable.")
    else:
        missing.extend(runtime_compatibility.missing_runtime_information[:2])
    if artifact_capability_matrix is None:
        missing.append("Artifact Capability Matrix metadata is unavailable.")
    else:
        missing.extend(artifact_capability_matrix.missing_capability_information[:2])
    if multi_artifact_strategy is None:
        missing.append("Multi-Artifact Strategy metadata is unavailable.")
    else:
        missing.extend(multi_artifact_strategy.missing_information[:2])
    return _dedupe(missing)[:10]


def _weaknesses(
    *,
    capability_gaps: tuple[str, ...],
    dependency_concerns: tuple[str, ...],
    runtime_concerns: tuple[str, ...],
    scalability_concerns: tuple[str, ...],
    maintainability_concerns: tuple[str, ...],
    complexity_concerns: tuple[str, ...],
    unsupported_assumptions: tuple[str, ...],
    missing_information: tuple[str, ...],
) -> tuple[str, ...]:
    weakness_sources = [
        *capability_gaps[:2],
        *dependency_concerns[:2],
        *runtime_concerns[:2],
        *scalability_concerns[:1],
        *maintainability_concerns[:1],
        *complexity_concerns[:1],
        *unsupported_assumptions[:1],
        *missing_information[:2],
    ]
    return _dedupe(weakness_sources)[:10]


def _risk_assessment(
    *,
    weaknesses: tuple[str, ...],
    missing: tuple[str, ...],
    dependency_concerns: tuple[str, ...],
    runtime_concerns: tuple[str, ...],
    unsupported_assumptions: tuple[str, ...],
) -> ArtifactCriticRiskAssessment:
    if any("unavailable" in item.lower() for item in missing):
        return "blocked"
    high_signal_count = sum(
        1
        for item in (*dependency_concerns, *runtime_concerns, *unsupported_assumptions)
        if _contains_any(item, ("blocking", "unsupported", "unavailable"))
    )
    if high_signal_count >= 3 or len(weaknesses) >= 8:
        return "high"
    if high_signal_count >= 1 or len(weaknesses) >= 4:
        return "medium"
    return "low"


def _open_questions(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    missing: tuple[str, ...],
) -> tuple[str, ...]:
    questions: list[str] = []
    if artifact_plan is not None:
        questions.extend(artifact_plan.hitl_questions[:2])
    if artifact_dependency_graph is not None:
        questions.extend(artifact_dependency_graph.hitl_questions[:2])
    if runtime_compatibility is not None:
        questions.extend(runtime_compatibility.hitl_questions[:2])
    if artifact_capability_matrix is not None:
        questions.extend(artifact_capability_matrix.hitl_questions[:2])
    if multi_artifact_strategy is not None:
        questions.extend(multi_artifact_strategy.hitl_questions[:2])
    questions.extend(
        f"Should this missing planning metadata be resolved: {item}"
        for item in missing[:3]
    )
    return _dedupe(questions)[:10]


def _hitl_questions(
    *,
    open_questions: tuple[str, ...],
    risk: ArtifactCriticRiskAssessment,
    missing: tuple[str, ...],
) -> tuple[str, ...]:
    questions: list[str] = []
    if risk in {"high", "blocked"}:
        questions.append(
            f"Should generation wait because Artifact Critic risk is {risk}?"
        )
    questions.extend(open_questions[:4])
    questions.extend(
        f"Should this Artifact Critic missing input be resolved: {item}"
        for item in missing[:2]
    )
    return _dedupe(questions)[:8]


def _improvement_opportunities(
    *,
    risk: ArtifactCriticRiskAssessment,
    capability_gaps: tuple[str, ...],
    dependency_concerns: tuple[str, ...],
    runtime_concerns: tuple[str, ...],
    multi_artifact_strategy: MultiArtifactStrategy | None,
) -> tuple[str, ...]:
    opportunities = [
        "Use critic findings as visible caveats in prompt guidance, not as edits.",
    ]
    if capability_gaps:
        opportunities.append("Clarify target capability limits before generation.")
    if dependency_concerns:
        opportunities.append(
            "Keep dependency assumptions explicit in response structure."
        )
    if runtime_concerns:
        opportunities.append(
            "Caveat unsupported runtimes without selecting alternatives."
        )
    if multi_artifact_strategy is not None and multi_artifact_strategy.risk_areas:
        opportunities.append(
            "Use multi-artifact risks to label support sections without adding outputs."
        )
    if risk in {"high", "blocked"}:
        opportunities.append("Surface HITL questions before expanding artifact scope.")
    return _dedupe(opportunities)[:10]


def _prompt_guidance(
    *,
    risk: ArtifactCriticRiskAssessment,
    improvement_opportunities: tuple[str, ...],
) -> tuple[str, ...]:
    guidance = [
        "Use Artifact Critic output as metadata-only critique of planning signals.",
        f"Treat critic risk assessment as advisory: {risk}.",
        (
            "Do not modify, refine, merge, execute, reject, or select artifacts "
            "from Artifact Critic output."
        ),
    ]
    guidance.extend(improvement_opportunities[:3])
    return _dedupe(guidance)[:8]


def _critique_confidence(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    missing: tuple[str, ...],
) -> float:
    available = sum(
        item is not None
        for item in (
            artifact_plan,
            artifact_dependency_graph,
            runtime_compatibility,
            artifact_capability_matrix,
            multi_artifact_strategy,
        )
    )
    confidence = 0.35 + (available * 0.11) - (len(missing) * 0.025)
    return max(0.0, min(0.96, round(confidence, 2)))


def _critique_summary(
    *,
    risk: ArtifactCriticRiskAssessment,
    strengths: tuple[str, ...],
    weaknesses: tuple[str, ...],
) -> str:
    if weaknesses:
        return (
            f"Artifact planning critique risk is {risk}; "
            f"{len(strengths)} strength(s) and {len(weaknesses)} concern(s) "
            "were found in planning metadata."
        )
    return (
        f"Artifact planning critique risk is {risk}; available metadata is "
        "coherent with no blocking planning concern detected."
    )


def _evidence(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
) -> tuple[str, ...]:
    evidence = [f"Mode: {request.mode.value}."]
    if route_decision is not None:
        evidence.append(f"Route selected: {route_decision.route.value}.")
    if artifact_plan is not None:
        evidence.append(
            f"Artifact plan: {artifact_plan.artifact_type}; "
            f"{artifact_plan.artifact_family}."
        )
    if artifact_dependency_graph is not None:
        evidence.append(
            "Dependency graph: "
            f"{len(artifact_dependency_graph.artifact_nodes)} nodes; "
            f"{len(artifact_dependency_graph.dependency_edges)} edges."
        )
    if runtime_compatibility is not None:
        evidence.append(
            "Runtime compatibility: "
            f"{len(runtime_compatibility.compatible_runtimes)} compatible; "
            f"{len(runtime_compatibility.unsupported_runtimes)} unsupported."
        )
    if artifact_capability_matrix is not None:
        evidence.append(
            "Capability matrix: "
            f"{len(artifact_capability_matrix.capability_profiles)} profiles."
        )
    if multi_artifact_strategy is not None:
        evidence.append(
            "Multi-artifact strategy: "
            f"{len(multi_artifact_strategy.supporting_artifacts)} supporting."
        )
    return _dedupe(evidence)[:12]
