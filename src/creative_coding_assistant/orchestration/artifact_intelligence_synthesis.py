"""Bounded Artifact Intelligence Synthesis for V3.3 planning metadata."""

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
from creative_coding_assistant.orchestration.artifact_refiner import (
    ArtifactRefinerProfile,
)
from creative_coding_assistant.orchestration.multi_artifact_strategy import (
    MultiArtifactStrategy,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.runtime_compatibility import (
    RuntimeCompatibilityProfile,
)

ArtifactImplementationReadiness = Literal[
    "ready",
    "needs_caveats",
    "needs_hitl",
    "blocked",
]
ArtifactImplementationComplexity = Literal["low", "medium", "high"]
ArtifactImplementationRisk = Literal["low", "medium", "high", "blocked"]
ArtifactImplementationPriority = Literal["critical", "high", "medium", "low"]

ARTIFACT_INTELLIGENCE_SYNTHESIS_AUTHORITY_BOUNDARY = (
    "The Artifact Intelligence Synthesis capability summarizes, ranks, and "
    "recommends across existing artifact planning metadata only; it does not "
    "modify artifacts, execute decisions, auto-select runtimes, choose "
    "providers or models, change routing, change previews, merge artifacts, "
    "export artifacts, trigger workflows, trigger retries, trigger "
    "escalation behavior, perform runtime repair, or implement V4 or V5 "
    "agent, routing, escalation, execution optimization, or production "
    "intelligence systems."
)


class ArtifactIntelligenceSynthesisProfile(BaseModel):
    """Inspectable metadata-only synthesis over V3.3 artifact intelligence."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["artifact_intelligence_synthesis"] = (
        "artifact_intelligence_synthesis"
    )
    synthesis_confidence: float = Field(ge=0, le=1)
    synthesis_summary: str = Field(min_length=1, max_length=520)
    recommended_artifact_path: str = Field(min_length=1, max_length=420)
    recommended_strategy_summary: str = Field(min_length=1, max_length=420)
    recommended_runtime_direction: str = Field(min_length=1, max_length=420)
    major_strengths: tuple[str, ...] = Field(min_length=1, max_length=10)
    major_weaknesses: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    major_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    dependency_overview: str = Field(min_length=1, max_length=420)
    capability_overview: str = Field(min_length=1, max_length=420)
    refinement_overview: str = Field(min_length=1, max_length=420)
    critique_overview: str = Field(min_length=1, max_length=420)
    implementation_readiness: ArtifactImplementationReadiness
    implementation_complexity: ArtifactImplementationComplexity
    implementation_risk: ArtifactImplementationRisk
    implementation_priority: ArtifactImplementationPriority
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=ARTIFACT_INTELLIGENCE_SYNTHESIS_AUTHORITY_BOUNDARY,
        max_length=1100,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=12)


def derive_artifact_intelligence_synthesis_profile(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
) -> ArtifactIntelligenceSynthesisProfile:
    """Synthesize artifact intelligence without changing workflow behavior."""

    missing = _missing_sources(
        route_decision=route_decision,
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_critic=artifact_critic,
        artifact_refiner=artifact_refiner,
    )
    strengths = _major_strengths(
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_critic=artifact_critic,
        artifact_refiner=artifact_refiner,
    )
    weaknesses = _major_weaknesses(
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        artifact_critic=artifact_critic,
        artifact_refiner=artifact_refiner,
        missing=missing,
    )
    risks = _major_risks(
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_critic=artifact_critic,
        artifact_refiner=artifact_refiner,
        missing=missing,
    )
    hitl_questions = _hitl_questions(
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_critic=artifact_critic,
        artifact_refiner=artifact_refiner,
        risks=risks,
        missing=missing,
    )
    readiness = _implementation_readiness(
        artifact_critic=artifact_critic,
        weaknesses=weaknesses,
        risks=risks,
        hitl_questions=hitl_questions,
        missing=missing,
    )
    complexity = _implementation_complexity(
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_compatibility=runtime_compatibility,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_critic=artifact_critic,
        artifact_refiner=artifact_refiner,
    )
    risk = _implementation_risk(
        readiness=readiness,
        artifact_critic=artifact_critic,
        risks=risks,
        weaknesses=weaknesses,
    )
    priority = _implementation_priority(readiness=readiness, risk=risk)
    return ArtifactIntelligenceSynthesisProfile(
        synthesis_confidence=_synthesis_confidence(
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            multi_artifact_strategy=multi_artifact_strategy,
            artifact_critic=artifact_critic,
            artifact_refiner=artifact_refiner,
            missing=missing,
            risks=risks,
        ),
        synthesis_summary=_synthesis_summary(
            readiness=readiness,
            risk=risk,
            priority=priority,
            strengths=strengths,
            weaknesses=weaknesses,
            risks=risks,
        ),
        recommended_artifact_path=_recommended_artifact_path(
            artifact_plan=artifact_plan,
            multi_artifact_strategy=multi_artifact_strategy,
        ),
        recommended_strategy_summary=_recommended_strategy_summary(
            multi_artifact_strategy=multi_artifact_strategy,
            artifact_critic=artifact_critic,
            artifact_refiner=artifact_refiner,
        ),
        recommended_runtime_direction=_recommended_runtime_direction(
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
        ),
        major_strengths=strengths,
        major_weaknesses=weaknesses,
        major_risks=risks,
        dependency_overview=_dependency_overview(artifact_dependency_graph),
        capability_overview=_capability_overview(artifact_capability_matrix),
        refinement_overview=_refinement_overview(artifact_refiner),
        critique_overview=_critique_overview(artifact_critic),
        implementation_readiness=readiness,
        implementation_complexity=complexity,
        implementation_risk=risk,
        implementation_priority=priority,
        hitl_questions=hitl_questions,
        prompt_guidance=_prompt_guidance(
            readiness=readiness,
            risk=risk,
            priority=priority,
        ),
        evidence=_evidence(
            request=request,
            route_decision=route_decision,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            multi_artifact_strategy=multi_artifact_strategy,
            artifact_critic=artifact_critic,
            artifact_refiner=artifact_refiner,
        ),
    )


def artifact_intelligence_synthesis_prompt_lines(
    profile: ArtifactIntelligenceSynthesisProfile,
) -> tuple[str, ...]:
    """Render synthesis metadata as compact prompt guidance."""

    lines = [
        f"Authority boundary: {profile.authority_boundary}",
        f"Synthesis confidence: {profile.synthesis_confidence:.2f}.",
        f"Synthesis summary: {profile.synthesis_summary}",
        f"Recommended artifact path: {profile.recommended_artifact_path}",
        f"Recommended strategy summary: {profile.recommended_strategy_summary}",
        f"Recommended runtime direction: {profile.recommended_runtime_direction}",
        f"Dependency overview: {profile.dependency_overview}",
        f"Capability overview: {profile.capability_overview}",
        f"Refinement overview: {profile.refinement_overview}",
        f"Critique overview: {profile.critique_overview}",
        f"Implementation readiness: {profile.implementation_readiness}.",
        f"Implementation complexity: {profile.implementation_complexity}.",
        f"Implementation risk: {profile.implementation_risk}.",
        f"Implementation priority: {profile.implementation_priority}.",
    ]
    lines.extend(
        f"Major synthesis strength: {item}" for item in profile.major_strengths
    )
    lines.extend(
        f"Major synthesis weakness: {item}" for item in profile.major_weaknesses
    )
    lines.extend(f"Major synthesis risk: {item}" for item in profile.major_risks)
    lines.extend(
        f"HITL synthesis question: {item}" for item in profile.hitl_questions
    )
    lines.extend(
        f"Artifact intelligence synthesis guidance: {item}"
        for item in profile.prompt_guidance
    )
    return tuple(lines[:72])


def _missing_sources(
    *,
    route_decision: RouteDecision | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
) -> tuple[str, ...]:
    missing: list[str] = []
    for label, value in (
        ("route decision", route_decision),
        ("artifact plan", artifact_plan),
        ("artifact dependency graph", artifact_dependency_graph),
        ("runtime compatibility", runtime_compatibility),
        ("artifact capability matrix", artifact_capability_matrix),
        ("multi-artifact strategy", multi_artifact_strategy),
        ("artifact critic", artifact_critic),
        ("artifact refiner", artifact_refiner),
    ):
        if value is None:
            missing.append(label)
    return tuple(missing)


def _major_strengths(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
) -> tuple[str, ...]:
    strengths: list[str] = []
    if artifact_plan is not None:
        strengths.append(
            f"Artifact shape is declared as {artifact_plan.artifact_type} / "
            f"{artifact_plan.artifact_family}."
        )
    if artifact_dependency_graph is not None:
        strengths.append(
            f"Dependency graph exposes {len(artifact_dependency_graph.artifact_nodes)} "
            f"nodes and {len(artifact_dependency_graph.dependency_edges)} edges."
        )
    if runtime_compatibility is not None:
        preferred = ", ".join(runtime_compatibility.preferred_runtimes) or "none"
        strengths.append(
            f"Runtime compatibility provides advisory targets: {preferred}."
        )
    if artifact_capability_matrix is not None:
        strongest = ", ".join(artifact_capability_matrix.strongest_targets) or "none"
        strengths.append(f"Capability matrix strongest targets are {strongest}.")
    if multi_artifact_strategy is not None:
        strengths.append(
            "Multi-artifact strategy identifies "
            f"{multi_artifact_strategy.primary_artifact.artifact_id} as primary "
            f"with {len(multi_artifact_strategy.supporting_artifacts)} supporting "
            "artifacts."
        )
    if artifact_critic is not None:
        strengths.extend(artifact_critic.strengths[:2])
    if artifact_refiner is not None:
        strengths.append(
            f"Refiner supplies {len(artifact_refiner.priority_improvements)} "
            "priority improvements."
        )
    strengths.append(
        "Synthesis remains metadata-only and does not execute, route, preview, "
        "merge, export, retry, or select runtimes."
    )
    return _dedupe(strengths)[:10]


def _major_weaknesses(
    *,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
    missing: tuple[str, ...],
) -> tuple[str, ...]:
    weaknesses: list[str] = []
    if artifact_dependency_graph is not None:
        weaknesses.extend(artifact_dependency_graph.dependency_conflicts[:2])
        weaknesses.extend(artifact_dependency_graph.missing_dependency_risks[:2])
    if runtime_compatibility is not None:
        weaknesses.extend(runtime_compatibility.runtime_limitations[:2])
        weaknesses.extend(runtime_compatibility.missing_runtime_information[:1])
    if artifact_capability_matrix is not None:
        weaknesses.extend(artifact_capability_matrix.target_weaknesses[:2])
        weaknesses.extend(artifact_capability_matrix.missing_capability_information[:1])
    if artifact_critic is not None:
        weaknesses.extend(artifact_critic.weaknesses[:3])
        weaknesses.extend(artifact_critic.unsupported_assumptions[:1])
    if artifact_refiner is not None:
        weaknesses.extend(
            f"Priority improvement remains unresolved: {item}"
            for item in artifact_refiner.priority_improvements[:2]
        )
    weaknesses.extend(f"Missing synthesis input: {item}." for item in missing[:3])
    return _dedupe(weaknesses)[:10]


def _major_risks(
    *,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
    missing: tuple[str, ...],
) -> tuple[str, ...]:
    risks: list[str] = []
    if artifact_critic is not None and artifact_critic.risk_assessment in {
        "high",
        "blocked",
    }:
        risks.append(f"Artifact Critic risk is {artifact_critic.risk_assessment}.")
    if artifact_dependency_graph is not None:
        risks.extend(artifact_dependency_graph.blocking_dependencies[:3])
        risks.extend(artifact_dependency_graph.dependency_conflicts[:2])
    if runtime_compatibility is not None:
        risks.extend(
            f"Unsupported runtime remains advisory only: {item}."
            for item in runtime_compatibility.unsupported_runtimes[:3]
        )
        risks.extend(runtime_compatibility.implementation_risks[:2])
    if artifact_capability_matrix is not None:
        risks.extend(artifact_capability_matrix.unsupported_or_risky_capabilities[:3])
        risks.extend(artifact_capability_matrix.capability_risks[:2])
    if multi_artifact_strategy is not None:
        risks.extend(multi_artifact_strategy.risk_areas[:2])
    if artifact_refiner is not None:
        risks.extend(artifact_refiner.risk_reductions[:2])
    if missing:
        risks.append(
            "Synthesis confidence is limited by unavailable metadata: "
            + ", ".join(missing[:4])
            + "."
        )
    return _dedupe(risks)[:10]


def _dependency_overview(
    graph: ArtifactDependencyGraph | None,
) -> str:
    if graph is None:
        return (
            "Dependency graph metadata is unavailable; treat dependency claims "
            "as unresolved."
        )
    return (
        f"{len(graph.artifact_nodes)} nodes, {len(graph.dependency_edges)} edges, "
        f"{len(graph.blocking_dependencies)} blocking dependencies, "
        f"{len(graph.dependency_conflicts)} conflicts, and downstream consumers "
        f"{', '.join(graph.downstream_consumers[:4]) or 'none'}."
    )


def _capability_overview(
    matrix: ArtifactCapabilityMatrix | None,
) -> str:
    if matrix is None:
        return (
            "Capability matrix metadata is unavailable; avoid target capability "
            "claims."
        )
    strongest = ", ".join(matrix.strongest_targets) or "none"
    weakest = ", ".join(matrix.weakest_targets) or "none"
    return (
        f"Strongest targets: {strongest}; weakest targets: {weakest}; "
        f"{len(matrix.unsupported_or_risky_capabilities)} unsupported/risky "
        f"capabilities; artifact fit {matrix.artifact_fit}."
    )


def _refinement_overview(
    profile: ArtifactRefinerProfile | None,
) -> str:
    if profile is None:
        return (
            "Artifact Refiner metadata is unavailable; do not prioritize "
            "refinements."
        )
    priority = (
        profile.priority_improvements[0]
        if profile.priority_improvements
        else "none"
    )
    return (
        f"Refiner confidence {profile.refinement_confidence:.2f}; "
        f"{len(profile.priority_improvements)} priority improvements; "
        f"{len(profile.refinement_candidates)} candidates; top priority: {priority}"
    )


def _critique_overview(
    profile: ArtifactCriticProfile | None,
) -> str:
    if profile is None:
        return "Artifact Critic metadata is unavailable; critique risk is unresolved."
    return (
        f"Critic risk {profile.risk_assessment}; confidence "
        f"{profile.critique_confidence:.2f}; {len(profile.weaknesses)} weaknesses; "
        f"{len(profile.open_questions)} open questions."
    )


def _recommended_artifact_path(
    *,
    artifact_plan: ArtifactPlan | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
) -> str:
    if multi_artifact_strategy is not None:
        supporting = ", ".join(
            artifact.artifact_id
            for artifact in multi_artifact_strategy.supporting_artifacts[:4]
        )
        suffix = f" with supporting artifacts {supporting}" if supporting else ""
        return (
            f"Lead with {multi_artifact_strategy.primary_artifact.artifact_id}"
            f"{suffix}; keep sections separated and advisory."
        )
    if artifact_plan is not None:
        return (
            f"Preserve a single {artifact_plan.artifact_type} artifact path for "
            f"{artifact_plan.artifact_family} with metadata caveats."
        )
    return (
        "Preserve the current request as a single advisory response until artifact "
        "metadata is available."
    )


def _recommended_strategy_summary(
    *,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
) -> str:
    parts: list[str] = []
    if multi_artifact_strategy is not None:
        parts.append(
            f"Use {multi_artifact_strategy.combination_mode} with "
            f"{len(multi_artifact_strategy.artifact_sequence)} ordered steps"
        )
    else:
        parts.append("Use the available artifact plan as the strategy source")
    if artifact_critic is not None:
        parts.append(f"treat critic risk as {artifact_critic.risk_assessment}")
    if artifact_refiner is not None:
        parts.append(
            f"surface {len(artifact_refiner.priority_improvements)} priority "
            "refinement advisories"
        )
    return "; ".join(parts) + "."


def _recommended_runtime_direction(
    *,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
) -> str:
    parts: list[str] = []
    if runtime_compatibility is not None:
        preferred = ", ".join(runtime_compatibility.preferred_runtimes)
        compatible = ", ".join(runtime_compatibility.compatible_runtimes[:4])
        if preferred:
            parts.append(
                "Document preferred runtime metadata as advisory only: "
                f"{preferred}"
            )
        elif compatible:
            parts.append(
                "Document compatible runtime metadata as advisory only: "
                f"{compatible}"
            )
        if runtime_compatibility.unsupported_runtimes:
            parts.append(
                "Caveat unsupported runtimes without selecting alternatives: "
                + ", ".join(runtime_compatibility.unsupported_runtimes[:4])
            )
    else:
        parts.append("Runtime compatibility metadata is unavailable")
    if artifact_capability_matrix is not None:
        strongest = ", ".join(artifact_capability_matrix.strongest_targets)
        if strongest:
            parts.append(f"use capability matrix caveats for {strongest}")
    return "; ".join(parts) + "."


def _implementation_readiness(
    *,
    artifact_critic: ArtifactCriticProfile | None,
    weaknesses: tuple[str, ...],
    risks: tuple[str, ...],
    hitl_questions: tuple[str, ...],
    missing: tuple[str, ...],
) -> ArtifactImplementationReadiness:
    if artifact_critic is not None and artifact_critic.risk_assessment == "blocked":
        return "blocked"
    if len(missing) >= 5:
        return "blocked"
    if hitl_questions or (
        artifact_critic is not None and artifact_critic.risk_assessment == "high"
    ):
        return "needs_hitl"
    if risks or weaknesses or missing:
        return "needs_caveats"
    return "ready"


def _implementation_complexity(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
) -> ArtifactImplementationComplexity:
    if (
        runtime_compatibility is not None
        and runtime_compatibility.expected_implementation_complexity == "high"
    ):
        return "high"
    if artifact_dependency_graph is not None and (
        artifact_dependency_graph.blocking_dependencies
        or len(artifact_dependency_graph.artifact_nodes) > 10
    ):
        return "high"
    if artifact_plan is not None and len(artifact_plan.required_components) > 7:
        return "high"
    if (
        multi_artifact_strategy is not None
        and len(multi_artifact_strategy.supporting_artifacts) > 3
    ):
        return "high"
    if artifact_critic is not None and artifact_critic.complexity_concerns:
        return "medium"
    if artifact_refiner is not None and artifact_refiner.complexity_reductions:
        return "medium"
    if (
        runtime_compatibility is not None
        and runtime_compatibility.expected_implementation_complexity == "medium"
    ):
        return "medium"
    return "low"


def _implementation_risk(
    *,
    readiness: ArtifactImplementationReadiness,
    artifact_critic: ArtifactCriticProfile | None,
    risks: tuple[str, ...],
    weaknesses: tuple[str, ...],
) -> ArtifactImplementationRisk:
    if readiness == "blocked":
        return "blocked"
    if artifact_critic is not None and artifact_critic.risk_assessment in {
        "high",
        "blocked",
    }:
        return "high"
    if len(risks) >= 4:
        return "high"
    if risks or len(weaknesses) >= 4:
        return "medium"
    return "low"


def _implementation_priority(
    *,
    readiness: ArtifactImplementationReadiness,
    risk: ArtifactImplementationRisk,
) -> ArtifactImplementationPriority:
    if readiness == "blocked" or risk == "blocked":
        return "critical"
    if risk == "high" or readiness == "needs_hitl":
        return "high"
    if risk == "medium" or readiness == "needs_caveats":
        return "medium"
    return "low"


def _hitl_questions(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
    risks: tuple[str, ...],
    missing: tuple[str, ...],
) -> tuple[str, ...]:
    questions: list[str] = []
    if artifact_refiner is not None:
        questions.extend(artifact_refiner.hitl_questions[:2])
    if artifact_critic is not None:
        questions.extend(artifact_critic.hitl_questions[:2])
    for profile in (
        artifact_plan,
        artifact_dependency_graph,
        runtime_compatibility,
        artifact_capability_matrix,
        multi_artifact_strategy,
    ):
        if profile is not None:
            questions.extend(getattr(profile, "hitl_questions", ())[:1])
    if risks:
        questions.append("Should synthesis risks be resolved before generation?")
    if missing:
        questions.append("Should missing artifact intelligence inputs block synthesis?")
    return _dedupe(questions)[:8]


def _prompt_guidance(
    *,
    readiness: ArtifactImplementationReadiness,
    risk: ArtifactImplementationRisk,
    priority: ArtifactImplementationPriority,
) -> tuple[str, ...]:
    return (
        "Use Artifact Intelligence Synthesis as metadata-only prompt guidance.",
        (
            "Summarize the recommended artifact path, strategy, runtime "
            "direction, strengths, weaknesses, risks, and HITL questions "
            "without executing decisions."
        ),
        (
            "Do not modify artifacts, auto-select runtimes, route providers or "
            "models, change previews, merge, export, trigger workflows, "
            "trigger retries, trigger escalation, or perform runtime repair."
        ),
        (
            "Expose synthesis fields as downstream-readable metadata only; do "
            "not activate V4 or V5 agent, routing, escalation, execution "
            "optimization, or production intelligence systems."
        ),
        (
            f"Treat readiness {readiness}, risk {risk}, and priority {priority} "
            "as advisory labels only."
        ),
    )


def _synthesis_confidence(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
    missing: tuple[str, ...],
    risks: tuple[str, ...],
) -> float:
    available_count = sum(
        item is not None
        for item in (
            artifact_plan,
            artifact_dependency_graph,
            runtime_compatibility,
            artifact_capability_matrix,
            multi_artifact_strategy,
            artifact_critic,
            artifact_refiner,
        )
    )
    confidence = 0.28 + (available_count * 0.09)
    if artifact_critic is not None:
        confidence += artifact_critic.critique_confidence * 0.08
    if artifact_refiner is not None:
        confidence += artifact_refiner.refinement_confidence * 0.08
    confidence -= min(len(missing) * 0.06, 0.24)
    confidence -= min(len(risks) * 0.015, 0.12)
    return round(min(max(confidence, 0.0), 1.0), 2)


def _synthesis_summary(
    *,
    readiness: ArtifactImplementationReadiness,
    risk: ArtifactImplementationRisk,
    priority: ArtifactImplementationPriority,
    strengths: tuple[str, ...],
    weaknesses: tuple[str, ...],
    risks: tuple[str, ...],
) -> str:
    return (
        f"Artifact intelligence synthesis reports {readiness} readiness, "
        f"{risk} implementation risk, and {priority} implementation priority "
        f"across {len(strengths)} strengths, {len(weaknesses)} weaknesses, and "
        f"{len(risks)} risks as metadata-only guidance."
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
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
) -> tuple[str, ...]:
    evidence = [f"Request mode: {request.mode.value}."]
    if route_decision is not None:
        evidence.append(f"Route: {route_decision.route.value}.")
    if artifact_plan is not None:
        evidence.append(
            f"Artifact plan: {artifact_plan.artifact_type}; "
            f"{artifact_plan.artifact_family}."
        )
    if artifact_dependency_graph is not None:
        evidence.append(
            "Dependency graph: "
            f"{len(artifact_dependency_graph.artifact_nodes)} nodes; "
            f"{len(artifact_dependency_graph.blocking_dependencies)} blocking."
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
            f"{len(artifact_capability_matrix.capability_profiles)} profiles; "
            f"{len(artifact_capability_matrix.unsupported_or_risky_capabilities)} "
            "unsupported/risky."
        )
    if multi_artifact_strategy is not None:
        evidence.append(
            "Multi-artifact strategy: "
            f"{len(multi_artifact_strategy.supporting_artifacts)} supporting; "
            f"{multi_artifact_strategy.combination_mode}."
        )
    if artifact_critic is not None:
        evidence.append(
            f"Artifact critic: {artifact_critic.risk_assessment} risk; "
            f"{artifact_critic.critique_confidence:.2f} confidence."
        )
    if artifact_refiner is not None:
        evidence.append(
            "Artifact refiner: "
            f"{artifact_refiner.refinement_confidence:.2f} confidence; "
            f"{len(artifact_refiner.priority_improvements)} priority."
        )
    return _dedupe(evidence)[:12]


def _dedupe(values: list[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    for value in values:
        normalized = " ".join(value.split())
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return tuple(deduped)
