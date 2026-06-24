"""Bounded Artifact Refiner for V3.3 planning metadata."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration.artifact_capability_matrix import (
    ArtifactCapabilityMatrix,
)
from creative_coding_assistant.orchestration.artifact_critic import (
    ArtifactCriticProfile,
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

ArtifactRefinementFocus = Literal[
    "capability",
    "dependency",
    "runtime",
    "scalability",
    "maintainability",
    "complexity",
    "risk",
    "metadata",
]

ARTIFACT_REFINER_AUTHORITY_BOUNDARY = (
    "The Artifact Refiner derives refinement intelligence from planning "
    "metadata only; it may recommend and prioritize improvements, but it does "
    "not modify artifacts, choose a final implementation, trigger execution, "
    "alter workflow behavior, perform automatic refinement, merge artifacts, "
    "export artifacts, select runtimes, change routing, change previews, "
    "trigger workflows, trigger retries, route providers or models, implement "
    "Artifact Merge Planner, implement Artifact Export Intelligence, implement "
    "V4 multi-agent behavior, or implement V5 execution optimization."
)


class ArtifactRefinerProfile(BaseModel):
    """Inspectable metadata-only refinement intelligence for planned artifacts."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["artifact_refiner"] = "artifact_refiner"
    refinement_confidence: float = Field(ge=0, le=1)
    refinement_summary: str = Field(min_length=1, max_length=520)
    recommended_improvements: tuple[str, ...] = Field(min_length=1, max_length=12)
    priority_improvements: tuple[str, ...] = Field(min_length=1, max_length=8)
    capability_improvements: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=10,
    )
    dependency_improvements: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=10,
    )
    runtime_improvements: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    scalability_improvements: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    maintainability_improvements: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    complexity_reductions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    risk_reductions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    refinement_candidates: tuple[str, ...] = Field(min_length=1, max_length=10)
    implementation_suggestions: tuple[str, ...] = Field(min_length=1, max_length=10)
    alternative_refinement_paths: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=ARTIFACT_REFINER_AUTHORITY_BOUNDARY,
        max_length=960,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=12)


def derive_artifact_refiner_profile(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
) -> ArtifactRefinerProfile:
    """Derive refinement intelligence without changing workflow behavior."""

    missing = _missing_sources(
        route_decision=route_decision,
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_critic=artifact_critic,
    )
    capability = _capability_improvements(
        artifact_capability_matrix=artifact_capability_matrix,
        artifact_critic=artifact_critic,
    )
    dependency = _dependency_improvements(
        artifact_dependency_graph=artifact_dependency_graph,
        artifact_critic=artifact_critic,
    )
    runtime = _runtime_improvements(
        runtime_compatibility=runtime_compatibility,
        artifact_critic=artifact_critic,
    )
    scalability = _scalability_improvements(
        artifact_plan=artifact_plan,
        artifact_capability_matrix=artifact_capability_matrix,
        runtime_compatibility=runtime_compatibility,
        artifact_critic=artifact_critic,
    )
    maintainability = _maintainability_improvements(
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_critic=artifact_critic,
    )
    complexity = _complexity_reductions(
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_compatibility=runtime_compatibility,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_critic=artifact_critic,
    )
    risk = _risk_reductions(
        artifact_critic=artifact_critic,
        dependency_improvements=dependency,
        runtime_improvements=runtime,
        missing=missing,
    )
    recommended = _recommended_improvements(
        capability_improvements=capability,
        dependency_improvements=dependency,
        runtime_improvements=runtime,
        scalability_improvements=scalability,
        maintainability_improvements=maintainability,
        complexity_reductions=complexity,
        risk_reductions=risk,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_critic=artifact_critic,
        missing=missing,
    )
    priority = _priority_improvements(
        recommended=recommended,
        missing=missing,
        artifact_critic=artifact_critic,
        dependency_improvements=dependency,
        runtime_improvements=runtime,
        capability_improvements=capability,
    )
    candidates = _refinement_candidates(
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_critic=artifact_critic,
    )
    implementation = _implementation_suggestions(
        priority_improvements=priority,
        artifact_critic=artifact_critic,
    )
    alternatives = _alternative_refinement_paths(
        capability_improvements=capability,
        dependency_improvements=dependency,
        runtime_improvements=runtime,
        complexity_reductions=complexity,
        risk_reductions=risk,
    )
    return ArtifactRefinerProfile(
        refinement_confidence=_refinement_confidence(
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            multi_artifact_strategy=multi_artifact_strategy,
            artifact_critic=artifact_critic,
            missing=missing,
        ),
        refinement_summary=_refinement_summary(
            recommended=recommended,
            priority=priority,
            artifact_critic=artifact_critic,
            missing=missing,
        ),
        recommended_improvements=recommended,
        priority_improvements=priority,
        capability_improvements=capability,
        dependency_improvements=dependency,
        runtime_improvements=runtime,
        scalability_improvements=scalability,
        maintainability_improvements=maintainability,
        complexity_reductions=complexity,
        risk_reductions=risk,
        refinement_candidates=candidates,
        implementation_suggestions=implementation,
        alternative_refinement_paths=alternatives,
        hitl_questions=_hitl_questions(
            priority_improvements=priority,
            artifact_critic=artifact_critic,
            missing=missing,
        ),
        prompt_guidance=_prompt_guidance(priority, artifact_critic),
        evidence=_evidence(
            request=request,
            route_decision=route_decision,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            multi_artifact_strategy=multi_artifact_strategy,
            artifact_critic=artifact_critic,
        ),
    )


def artifact_refiner_prompt_lines(
    profile: ArtifactRefinerProfile,
) -> tuple[str, ...]:
    """Render Artifact Refiner metadata as compact prompt guidance."""

    lines = [
        f"Authority boundary: {profile.authority_boundary}",
        f"Refinement confidence: {profile.refinement_confidence:.2f}.",
        f"Refinement summary: {profile.refinement_summary}",
    ]
    lines.extend(
        f"Recommended refinement improvement: {item}"
        for item in profile.recommended_improvements
    )
    lines.extend(
        f"Priority refinement improvement: {item}"
        for item in profile.priority_improvements
    )
    lines.extend(
        f"Capability refinement improvement: {item}"
        for item in profile.capability_improvements
    )
    lines.extend(
        f"Dependency refinement improvement: {item}"
        for item in profile.dependency_improvements
    )
    lines.extend(
        f"Runtime refinement improvement: {item}"
        for item in profile.runtime_improvements
    )
    lines.extend(
        f"Scalability refinement improvement: {item}"
        for item in profile.scalability_improvements
    )
    lines.extend(
        f"Maintainability refinement improvement: {item}"
        for item in profile.maintainability_improvements
    )
    lines.extend(
        f"Complexity reduction: {item}" for item in profile.complexity_reductions
    )
    lines.extend(f"Risk reduction: {item}" for item in profile.risk_reductions)
    lines.extend(
        f"Refinement candidate: {item}" for item in profile.refinement_candidates
    )
    lines.extend(
        f"Implementation suggestion: {item}"
        for item in profile.implementation_suggestions
    )
    lines.extend(
        f"Alternative refinement path: {item}"
        for item in profile.alternative_refinement_paths
    )
    lines.extend(f"HITL refiner question: {item}" for item in profile.hitl_questions)
    lines.extend(
        f"Artifact refiner guidance: {item}" for item in profile.prompt_guidance
    )
    return tuple(lines[:72])


def _capability_improvements(
    *,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    artifact_critic: ArtifactCriticProfile | None,
) -> tuple[str, ...]:
    improvements: list[str] = []
    if artifact_capability_matrix is None:
        improvements.append("Attach capability matrix metadata before refinement.")
    else:
        improvements.extend(
            f"Clarify capability limitation: {item}"
            for item in artifact_capability_matrix.unsupported_or_risky_capabilities[:3]
        )
        improvements.extend(
            f"Preserve target caveat: {item}"
            for item in artifact_capability_matrix.target_weaknesses[:3]
        )
        if artifact_capability_matrix.artifact_fit in {"weak", "unsupported"}:
            improvements.append(
                "Narrow artifact scope around supported target capabilities."
            )
        if artifact_capability_matrix.export_fit in {"weak", "unsupported"}:
            improvements.append("Avoid export claims in refinement suggestions.")
    if artifact_critic is not None:
        improvements.extend(
            f"Address critic capability gap: {item}"
            for item in artifact_critic.capability_gaps[:3]
        )
    return _dedupe(improvements)[:10]


def _dependency_improvements(
    *,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    artifact_critic: ArtifactCriticProfile | None,
) -> tuple[str, ...]:
    improvements: list[str] = []
    if artifact_dependency_graph is None:
        improvements.append("Attach dependency graph metadata before refinement.")
    else:
        improvements.extend(
            f"Resolve blocking dependency assumption: {item}"
            for item in artifact_dependency_graph.blocking_dependencies[:3]
        )
        improvements.extend(
            f"Separate conflicting dependency assumption: {item}"
            for item in artifact_dependency_graph.dependency_conflicts[:3]
        )
        improvements.extend(
            f"Document missing dependency risk: {item}"
            for item in artifact_dependency_graph.missing_dependency_risks[:3]
        )
    if artifact_critic is not None:
        improvements.extend(
            f"Address critic dependency concern: {item}"
            for item in artifact_critic.dependency_concerns[:3]
        )
    return _dedupe(improvements)[:10]


def _runtime_improvements(
    *,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_critic: ArtifactCriticProfile | None,
) -> tuple[str, ...]:
    improvements: list[str] = []
    if runtime_compatibility is None:
        improvements.append(
            "Attach runtime compatibility metadata before refinement."
        )
    else:
        if runtime_compatibility.unsupported_runtimes:
            improvements.append(
                "Caveat unsupported runtimes without selecting alternatives: "
                + ", ".join(runtime_compatibility.unsupported_runtimes[:4])
                + "."
            )
        improvements.extend(
            f"Preserve runtime limitation: {item}"
            for item in runtime_compatibility.runtime_limitations[:3]
        )
        improvements.extend(
            f"Reduce runtime implementation risk: {item}"
            for item in runtime_compatibility.implementation_risks[:3]
        )
        improvements.extend(
            f"Clarify missing runtime detail: {item}"
            for item in runtime_compatibility.missing_runtime_information[:2]
        )
    if artifact_critic is not None:
        improvements.extend(
            f"Address critic runtime concern: {item}"
            for item in artifact_critic.runtime_concerns[:3]
        )
    return _dedupe(improvements)[:10]


def _scalability_improvements(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_critic: ArtifactCriticProfile | None,
) -> tuple[str, ...]:
    source_items: list[str] = []
    if artifact_plan is not None:
        source_items.extend(artifact_plan.implementation_risks)
        source_items.extend(artifact_plan.runtime_requirements)
    if artifact_capability_matrix is not None:
        source_items.extend(artifact_capability_matrix.capability_risks)
        source_items.extend(artifact_capability_matrix.target_weaknesses)
    if runtime_compatibility is not None:
        source_items.extend(runtime_compatibility.implementation_risks)
    if artifact_critic is not None:
        source_items.extend(artifact_critic.scalability_concerns)
    improvements = [
        f"Add bounded-scope caveat for scalability signal: {item}"
        for item in source_items
        if _contains_any(
            item,
            ("scale", "performance", "frame", "particle", "dense", "large"),
        )
    ]
    return _dedupe(improvements)[:8]


def _maintainability_improvements(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
) -> tuple[str, ...]:
    improvements: list[str] = []
    if artifact_plan is not None and len(artifact_plan.required_components) > 6:
        improvements.append("Group required components into named response sections.")
    if (
        artifact_dependency_graph is not None
        and len(artifact_dependency_graph.artifact_nodes) > 10
    ):
        improvements.append("Group dependency nodes before describing handoffs.")
    if (
        multi_artifact_strategy is not None
        and len(multi_artifact_strategy.supporting_artifacts) > 3
    ):
        improvements.append("Keep supporting artifacts separated with strict labels.")
    if artifact_plan is not None:
        improvements.extend(
            f"Improve inspectability for implementation risk: {item}"
            for item in artifact_plan.implementation_risks
            if _contains_any(item, ("maintain", "inspect", "complex", "boilerplate"))
        )
    if artifact_critic is not None:
        improvements.extend(
            f"Address critic maintainability concern: {item}"
            for item in artifact_critic.maintainability_concerns[:3]
        )
    return _dedupe(improvements)[:8]


def _complexity_reductions(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
) -> tuple[str, ...]:
    reductions: list[str] = []
    if (
        runtime_compatibility is not None
        and runtime_compatibility.expected_implementation_complexity == "high"
    ):
        reductions.append("Reduce implementation scope before adding optional details.")
    if artifact_plan is not None:
        reductions.extend(
            f"Keep complex plan element bounded: {item}"
            for item in artifact_plan.implementation_risks
            if _contains_any(item, ("complex", "scope", "boilerplate", "3d"))
        )
    if (
        artifact_dependency_graph is not None
        and artifact_dependency_graph.blocking_dependencies
    ):
        reductions.append("Resolve blocking dependencies before adding new sections.")
    if (
        multi_artifact_strategy is not None
        and multi_artifact_strategy.combination_mode == "separated_parallel_sections"
    ):
        reductions.append("Favor explicit ordering over parallel expansion.")
    if artifact_critic is not None:
        reductions.extend(
            f"Reduce critic complexity concern: {item}"
            for item in artifact_critic.complexity_concerns[:3]
        )
    return _dedupe(reductions)[:8]


def _risk_reductions(
    *,
    artifact_critic: ArtifactCriticProfile | None,
    dependency_improvements: tuple[str, ...],
    runtime_improvements: tuple[str, ...],
    missing: tuple[str, ...],
) -> tuple[str, ...]:
    reductions = [
        "Preserve refinement advice as metadata-only guidance, not an edit."
    ]
    if artifact_critic is None:
        reductions.append("Wait for critic metadata before prioritizing risk.")
    else:
        reductions.append(
            f"Treat Artifact Critic risk as advisory: "
            f"{artifact_critic.risk_assessment}."
        )
        reductions.extend(
            f"Reduce critic weakness: {item}" for item in artifact_critic.weaknesses[:3]
        )
    if dependency_improvements:
        reductions.append("Resolve dependency risks before expanding artifact scope.")
    if runtime_improvements:
        reductions.append("Retain runtime caveats without selecting runtimes.")
    if missing:
        reductions.append("Resolve missing metadata before relying on refinements.")
    return _dedupe(reductions)[:8]


def _recommended_improvements(
    *,
    capability_improvements: tuple[str, ...],
    dependency_improvements: tuple[str, ...],
    runtime_improvements: tuple[str, ...],
    scalability_improvements: tuple[str, ...],
    maintainability_improvements: tuple[str, ...],
    complexity_reductions: tuple[str, ...],
    risk_reductions: tuple[str, ...],
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
    missing: tuple[str, ...],
) -> tuple[str, ...]:
    improvements: list[str] = []
    improvements.extend(risk_reductions[:2])
    improvements.extend(capability_improvements[:2])
    improvements.extend(dependency_improvements[:2])
    improvements.extend(runtime_improvements[:2])
    improvements.extend(scalability_improvements[:1])
    improvements.extend(maintainability_improvements[:1])
    improvements.extend(complexity_reductions[:1])
    if multi_artifact_strategy is not None:
        improvements.extend(multi_artifact_strategy.risk_areas[:2])
    if artifact_critic is not None:
        improvements.extend(artifact_critic.improvement_opportunities[:2])
    improvements.extend(
        f"Resolve missing refinement input: {item}" for item in missing[:2]
    )
    if not improvements:
        improvements.append(
            "Preserve current artifact plan and surface existing metadata caveats."
        )
    return _dedupe(improvements)[:12]


def _priority_improvements(
    *,
    recommended: tuple[str, ...],
    missing: tuple[str, ...],
    artifact_critic: ArtifactCriticProfile | None,
    dependency_improvements: tuple[str, ...],
    runtime_improvements: tuple[str, ...],
    capability_improvements: tuple[str, ...],
) -> tuple[str, ...]:
    priority: list[str] = []
    if missing:
        priority.append("Resolve unavailable refinement inputs first.")
    if artifact_critic is not None and artifact_critic.risk_assessment in {
        "high",
        "blocked",
    }:
        priority.append("Prioritize Artifact Critic risk reductions.")
    priority.extend(dependency_improvements[:2])
    priority.extend(runtime_improvements[:1])
    priority.extend(capability_improvements[:1])
    priority.extend(recommended[:3])
    return _dedupe(priority)[:8] or recommended[:1]


def _refinement_candidates(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
) -> tuple[str, ...]:
    candidates: list[str] = []
    if artifact_plan is not None:
        candidates.append(
            f"Primary candidate: tighten {artifact_plan.artifact_family} "
            "response structure without modifying the artifact."
        )
    if artifact_dependency_graph is not None and (
        artifact_dependency_graph.blocking_dependencies
        or artifact_dependency_graph.dependency_conflicts
    ):
        candidates.append("Dependency candidate: clarify assumptions and handoffs.")
    if runtime_compatibility is not None and runtime_compatibility.unsupported_runtimes:
        candidates.append("Runtime candidate: add caveats for unsupported runtimes.")
    if artifact_capability_matrix is not None and (
        artifact_capability_matrix.unsupported_or_risky_capabilities
        or artifact_capability_matrix.target_weaknesses
    ):
        candidates.append("Capability candidate: narrow unsupported target claims.")
    if multi_artifact_strategy is not None and multi_artifact_strategy.risk_areas:
        candidates.append("Strategy candidate: label support sections more clearly.")
    if artifact_critic is not None and artifact_critic.weaknesses:
        candidates.append("Critic candidate: address top weakness signals.")
    if not candidates:
        candidates.append("Conservative candidate: preserve current plan boundaries.")
    return _dedupe(candidates)[:10]


def _implementation_suggestions(
    *,
    priority_improvements: tuple[str, ...],
    artifact_critic: ArtifactCriticProfile | None,
) -> tuple[str, ...]:
    suggestions = [
        "Label every refinement as advisory metadata, not an artifact edit.",
        "Keep suggested changes inspectable and scoped to response guidance.",
    ]
    suggestions.extend(
        f"Priority implementation suggestion: {item}"
        for item in priority_improvements[:3]
    )
    if artifact_critic is not None:
        suggestions.extend(artifact_critic.prompt_guidance[:2])
    return _dedupe(suggestions)[:10]


def _alternative_refinement_paths(
    *,
    capability_improvements: tuple[str, ...],
    dependency_improvements: tuple[str, ...],
    runtime_improvements: tuple[str, ...],
    complexity_reductions: tuple[str, ...],
    risk_reductions: tuple[str, ...],
) -> tuple[str, ...]:
    paths: list[str] = []
    if capability_improvements:
        paths.append("Capability-first path: clarify target limits first.")
    if dependency_improvements:
        paths.append("Dependency-first path: resolve handoffs and conflicts first.")
    if runtime_improvements:
        paths.append("Runtime-first path: document runtime caveats first.")
    if complexity_reductions:
        paths.append("Complexity-first path: reduce scope before adding detail.")
    if risk_reductions:
        paths.append("Risk-first path: surface advisory risks before expansion.")
    if not paths:
        paths.append("Conservative path: preserve current plan and caveats.")
    return _dedupe(paths)[:8]


def _hitl_questions(
    *,
    priority_improvements: tuple[str, ...],
    artifact_critic: ArtifactCriticProfile | None,
    missing: tuple[str, ...],
) -> tuple[str, ...]:
    questions: list[str] = []
    if artifact_critic is not None:
        questions.extend(artifact_critic.hitl_questions[:3])
    if missing:
        questions.append("Should missing refinement metadata be resolved first?")
    if priority_improvements:
        questions.append("Which advisory refinement should be prioritized first?")
    return _dedupe(questions)[:8]


def _prompt_guidance(
    priority_improvements: tuple[str, ...],
    artifact_critic: ArtifactCriticProfile | None,
) -> tuple[str, ...]:
    guidance = [
        "Use Artifact Refiner output as metadata-only refinement intelligence.",
        (
            "Do not modify artifacts, choose final implementation, execute, "
            "merge, export, select runtime, route, preview, or retry."
        ),
    ]
    if artifact_critic is not None:
        guidance.append(
            "Treat Artifact Critic findings as advisory input to refinement "
            "priorities only."
        )
    guidance.extend(priority_improvements[:3])
    return _dedupe(guidance)[:8]


def _refinement_confidence(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
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
            artifact_critic,
        )
    )
    confidence = 0.3 + (available * 0.095) - (len(missing) * 0.025)
    if artifact_critic is not None and artifact_critic.risk_assessment == "blocked":
        confidence -= 0.12
    if artifact_critic is not None and artifact_critic.risk_assessment == "high":
        confidence -= 0.06
    return max(0.0, min(0.96, round(confidence, 2)))


def _refinement_summary(
    *,
    recommended: tuple[str, ...],
    priority: tuple[str, ...],
    artifact_critic: ArtifactCriticProfile | None,
    missing: tuple[str, ...],
) -> str:
    critic_clause = (
        f"critic risk {artifact_critic.risk_assessment}"
        if artifact_critic is not None
        else "critic metadata unavailable"
    )
    missing_clause = f"; {len(missing)} missing input(s)" if missing else ""
    return (
        "Artifact refinement intelligence is advisory only with "
        f"{len(recommended)} recommended improvement(s), "
        f"{len(priority)} priority item(s), {critic_clause}{missing_clause}."
    )


def _missing_sources(
    *,
    route_decision: RouteDecision | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
) -> tuple[str, ...]:
    missing: list[str] = []
    if route_decision is None or not route_decision.domains:
        missing.append("Route/domain metadata is inferred or unavailable.")
    if artifact_plan is None:
        missing.append("Artifact Plan metadata is unavailable.")
    if artifact_dependency_graph is None:
        missing.append("Artifact Dependency Graph metadata is unavailable.")
    if runtime_compatibility is None:
        missing.append("Runtime Compatibility Engine metadata is unavailable.")
    if artifact_capability_matrix is None:
        missing.append("Artifact Capability Matrix metadata is unavailable.")
    if multi_artifact_strategy is None:
        missing.append("Multi-Artifact Strategy metadata is unavailable.")
    if artifact_critic is None:
        missing.append("Artifact Critic metadata is unavailable.")
    return _dedupe(missing)[:10]


def _evidence(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
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
    if artifact_critic is not None:
        evidence.append(
            "Artifact critic: "
            f"{artifact_critic.risk_assessment} risk; "
            f"{len(artifact_critic.weaknesses)} weakness signals."
        )
    return _dedupe(evidence)[:12]


def _contains_any(value: str, needles: tuple[str, ...]) -> bool:
    lowered = value.lower()
    return any(needle in lowered for needle in needles)


def _dedupe(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    deduped: list[str] = []
    for value in values:
        cleaned = _clip(value)
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return tuple(deduped)


def _clip(value: str, limit: int = 360) -> str:
    normalized = " ".join(value.strip().split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "."
