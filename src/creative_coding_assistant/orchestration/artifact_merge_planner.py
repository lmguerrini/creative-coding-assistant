"""Bounded Artifact Merge Planner for V3.3 planning metadata."""

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
from creative_coding_assistant.orchestration.artifact_intelligence_synthesis import (
    ArtifactIntelligenceSynthesisProfile,
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

ArtifactMergeStrategy = Literal[
    "single_artifact_no_merge",
    "primary_with_supporting_sections",
    "separated_advisory_sections",
    "defer_merge_preserve_separation",
]

ARTIFACT_MERGE_PLANNER_AUTHORITY_BOUNDARY = (
    "The Artifact Merge Planner recommends merge and composition strategy "
    "from existing artifact metadata only; it may identify boundaries, join "
    "points, separation points, integration order, alternatives, rejected "
    "paths, risks, and HITL questions, but it does not merge artifacts, "
    "alter artifacts, choose a final implementation, execute artifacts or "
    "runtimes, export artifacts, select runtimes, change routing, change "
    "previews, trigger workflows, trigger retries, trigger escalation "
    "behavior, implement V4 agent routing or escalation, implement V5 "
    "execution optimization, or implement V6 Blueprint Export workflows."
)


class ArtifactMergePlannerProfile(BaseModel):
    """Inspectable metadata-only merge planning for planned artifacts."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["artifact_merge_planner"] = "artifact_merge_planner"
    merge_confidence: float = Field(ge=0, le=1)
    merge_summary: str = Field(min_length=1, max_length=520)
    merge_strategy: ArtifactMergeStrategy
    composition_strategy: str = Field(min_length=1, max_length=420)
    artifact_boundaries: tuple[str, ...] = Field(min_length=1, max_length=10)
    artifact_join_points: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    artifact_separation_points: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=10,
    )
    integration_order: tuple[str, ...] = Field(min_length=1, max_length=10)
    composition_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    dependency_merge_risks: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=10,
    )
    runtime_merge_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    capability_merge_risks: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=10,
    )
    recommended_merge_path: str = Field(min_length=1, max_length=420)
    alternative_merge_paths: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    rejected_merge_paths: tuple[str, ...] = Field(min_length=1, max_length=8)
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=ARTIFACT_MERGE_PLANNER_AUTHORITY_BOUNDARY,
        max_length=1100,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=12)


def derive_artifact_merge_planner_profile(
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
    artifact_intelligence_synthesis: ArtifactIntelligenceSynthesisProfile | None,
) -> ArtifactMergePlannerProfile:
    """Plan merge/composition metadata without merging or changing workflow."""

    missing = _missing_sources(
        route_decision=route_decision,
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_critic=artifact_critic,
        artifact_refiner=artifact_refiner,
        artifact_intelligence_synthesis=artifact_intelligence_synthesis,
    )
    boundaries = _artifact_boundaries(
        artifact_plan=artifact_plan,
        multi_artifact_strategy=multi_artifact_strategy,
    )
    join_points = _artifact_join_points(
        artifact_dependency_graph=artifact_dependency_graph,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_refiner=artifact_refiner,
        artifact_intelligence_synthesis=artifact_intelligence_synthesis,
    )
    separation_points = _artifact_separation_points(
        artifact_dependency_graph=artifact_dependency_graph,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_critic=artifact_critic,
        artifact_intelligence_synthesis=artifact_intelligence_synthesis,
    )
    integration_order = _integration_order(
        artifact_plan=artifact_plan,
        multi_artifact_strategy=multi_artifact_strategy,
    )
    dependency_risks = _dependency_merge_risks(
        artifact_dependency_graph=artifact_dependency_graph,
        artifact_critic=artifact_critic,
    )
    runtime_risks = _runtime_merge_risks(
        runtime_compatibility=runtime_compatibility,
        artifact_critic=artifact_critic,
    )
    capability_risks = _capability_merge_risks(
        artifact_capability_matrix=artifact_capability_matrix,
        artifact_critic=artifact_critic,
    )
    composition_risks = _composition_risks(
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_critic=artifact_critic,
        artifact_refiner=artifact_refiner,
        artifact_intelligence_synthesis=artifact_intelligence_synthesis,
        dependency_merge_risks=dependency_risks,
        runtime_merge_risks=runtime_risks,
        capability_merge_risks=capability_risks,
        missing=missing,
    )
    merge_strategy = _merge_strategy(
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_dependency_graph=artifact_dependency_graph,
        artifact_critic=artifact_critic,
        artifact_intelligence_synthesis=artifact_intelligence_synthesis,
        composition_risks=composition_risks,
    )
    recommended = _recommended_merge_path(
        merge_strategy=merge_strategy,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_intelligence_synthesis=artifact_intelligence_synthesis,
    )
    alternatives = _alternative_merge_paths(
        merge_strategy=merge_strategy,
        multi_artifact_strategy=multi_artifact_strategy,
        separation_points=separation_points,
    )
    rejected = _rejected_merge_paths(
        merge_strategy=merge_strategy,
        runtime_compatibility=runtime_compatibility,
        composition_risks=composition_risks,
    )
    return ArtifactMergePlannerProfile(
        merge_confidence=_merge_confidence(
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            multi_artifact_strategy=multi_artifact_strategy,
            artifact_critic=artifact_critic,
            artifact_refiner=artifact_refiner,
            artifact_intelligence_synthesis=artifact_intelligence_synthesis,
            missing=missing,
            composition_risks=composition_risks,
        ),
        merge_summary=_merge_summary(
            merge_strategy=merge_strategy,
            boundaries=boundaries,
            join_points=join_points,
            separation_points=separation_points,
            composition_risks=composition_risks,
        ),
        merge_strategy=merge_strategy,
        composition_strategy=_composition_strategy(
            merge_strategy=merge_strategy,
            artifact_plan=artifact_plan,
            multi_artifact_strategy=multi_artifact_strategy,
        ),
        artifact_boundaries=boundaries,
        artifact_join_points=join_points,
        artifact_separation_points=separation_points,
        integration_order=integration_order,
        composition_risks=composition_risks,
        dependency_merge_risks=dependency_risks,
        runtime_merge_risks=runtime_risks,
        capability_merge_risks=capability_risks,
        recommended_merge_path=recommended,
        alternative_merge_paths=alternatives,
        rejected_merge_paths=rejected,
        hitl_questions=_hitl_questions(
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            multi_artifact_strategy=multi_artifact_strategy,
            artifact_critic=artifact_critic,
            artifact_refiner=artifact_refiner,
            artifact_intelligence_synthesis=artifact_intelligence_synthesis,
            composition_risks=composition_risks,
            missing=missing,
        ),
        prompt_guidance=_prompt_guidance(merge_strategy),
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
            artifact_intelligence_synthesis=artifact_intelligence_synthesis,
        ),
    )


def artifact_merge_planner_prompt_lines(
    profile: ArtifactMergePlannerProfile,
) -> tuple[str, ...]:
    """Render Artifact Merge Planner metadata as compact prompt guidance."""

    lines = [
        f"Authority boundary: {profile.authority_boundary}",
        f"Merge confidence: {profile.merge_confidence:.2f}.",
        f"Merge strategy: {profile.merge_strategy}.",
        f"Merge summary: {profile.merge_summary}",
        f"Composition strategy: {profile.composition_strategy}",
        f"Recommended merge path: {profile.recommended_merge_path}",
    ]
    lines.extend(f"Artifact boundary: {item}" for item in profile.artifact_boundaries)
    lines.extend(
        f"Artifact join point: {item}" for item in profile.artifact_join_points
    )
    lines.extend(
        f"Artifact separation point: {item}"
        for item in profile.artifact_separation_points
    )
    lines.extend(f"Integration order: {item}" for item in profile.integration_order)
    lines.extend(f"Composition risk: {item}" for item in profile.composition_risks)
    lines.extend(
        f"Dependency merge risk: {item}" for item in profile.dependency_merge_risks
    )
    lines.extend(f"Runtime merge risk: {item}" for item in profile.runtime_merge_risks)
    lines.extend(
        f"Capability merge risk: {item}" for item in profile.capability_merge_risks
    )
    lines.extend(
        f"Alternative merge path: {item}" for item in profile.alternative_merge_paths
    )
    lines.extend(
        f"Rejected merge path: {item}" for item in profile.rejected_merge_paths
    )
    lines.extend(f"HITL merge question: {item}" for item in profile.hitl_questions)
    lines.extend(
        f"Artifact merge planner guidance: {item}" for item in profile.prompt_guidance
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
    artifact_intelligence_synthesis: ArtifactIntelligenceSynthesisProfile | None,
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
        ("artifact intelligence synthesis", artifact_intelligence_synthesis),
    ):
        if value is None:
            missing.append(label)
    return tuple(missing)


def _artifact_boundaries(
    *,
    artifact_plan: ArtifactPlan | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
) -> tuple[str, ...]:
    boundaries: list[str] = []
    if multi_artifact_strategy is not None:
        primary = multi_artifact_strategy.primary_artifact
        boundaries.append(
            f"Primary boundary: {primary.artifact_id} remains the lead "
            f"{primary.artifact_type} artifact."
        )
        boundaries.extend(
            f"Supporting boundary: {artifact.artifact_id} remains "
            f"{artifact.role} / {artifact.artifact_type}."
            for artifact in multi_artifact_strategy.supporting_artifacts[:6]
        )
    elif artifact_plan is not None:
        boundaries.append(
            f"Single boundary: preserve {artifact_plan.artifact_type} / "
            f"{artifact_plan.artifact_family} as one artifact path."
        )
    else:
        boundaries.append("Fallback boundary: no artifact metadata is available.")
    return _dedupe(boundaries)[:10]


def _artifact_join_points(
    *,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_refiner: ArtifactRefinerProfile | None,
    artifact_intelligence_synthesis: ArtifactIntelligenceSynthesisProfile | None,
) -> tuple[str, ...]:
    joins: list[str] = []
    if multi_artifact_strategy is not None:
        joins.extend(multi_artifact_strategy.artifact_handoff_points[:4])
        joins.extend(
            f"{step.artifact_id}: {step.rationale}"
            for step in multi_artifact_strategy.artifact_sequence[:3]
            if step.depends_on
        )
    if artifact_dependency_graph is not None:
        joins.extend(artifact_dependency_graph.prompt_facing_dependencies[:2])
        joins.extend(artifact_dependency_graph.runtime_facing_dependencies[:1])
    if artifact_refiner is not None:
        joins.extend(artifact_refiner.implementation_suggestions[:1])
    if artifact_intelligence_synthesis is not None:
        joins.append(artifact_intelligence_synthesis.recommended_artifact_path)
    if not joins:
        joins.append("No merge join point is recommended for a single artifact path.")
    return _dedupe(joins)[:10]


def _artifact_separation_points(
    *,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_intelligence_synthesis: ArtifactIntelligenceSynthesisProfile | None,
) -> tuple[str, ...]:
    points: list[str] = []
    if multi_artifact_strategy is not None:
        points.extend(multi_artifact_strategy.artifact_separation_strategy[:4])
    if artifact_dependency_graph is not None:
        points.extend(
            f"Separate dependency conflict: {item}"
            for item in artifact_dependency_graph.dependency_conflicts[:3]
        )
        points.extend(
            f"Preserve blocking boundary: {item}"
            for item in artifact_dependency_graph.blocking_dependencies[:2]
        )
    if artifact_critic is not None:
        points.extend(
            f"Separate critic weakness: {item}"
            for item in artifact_critic.weaknesses[:2]
        )
    if artifact_intelligence_synthesis is not None:
        points.extend(
            f"Separate synthesis risk: {item}"
            for item in artifact_intelligence_synthesis.major_risks[:2]
        )
    if not points:
        points.append(
            "Keep artifact metadata separate from generated artifact content."
        )
    return _dedupe(points)[:10]


def _integration_order(
    *,
    artifact_plan: ArtifactPlan | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
) -> tuple[str, ...]:
    if multi_artifact_strategy is not None:
        return tuple(
            f"{step.order}. {step.artifact_id}: {step.action}"
            for step in multi_artifact_strategy.artifact_sequence[:10]
        )
    if artifact_plan is not None:
        return (
            f"1. Primary {artifact_plan.artifact_type} / "
            f"{artifact_plan.artifact_family}",
        )
    return ("1. Preserve request-level response until artifact metadata is available.",)


def _dependency_merge_risks(
    *,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    artifact_critic: ArtifactCriticProfile | None,
) -> tuple[str, ...]:
    risks: list[str] = []
    if artifact_dependency_graph is None:
        risks.append("Dependency merge risks are unresolved without a graph.")
    else:
        risks.extend(artifact_dependency_graph.blocking_dependencies[:3])
        risks.extend(artifact_dependency_graph.dependency_conflicts[:3])
        risks.extend(artifact_dependency_graph.missing_dependency_risks[:2])
    if artifact_critic is not None:
        risks.extend(artifact_critic.dependency_concerns[:2])
    return _dedupe(risks)[:10]


def _runtime_merge_risks(
    *,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_critic: ArtifactCriticProfile | None,
) -> tuple[str, ...]:
    risks: list[str] = []
    if runtime_compatibility is None:
        risks.append(
            "Runtime merge risks are unresolved without compatibility metadata."
        )
    else:
        risks.extend(
            f"Unsupported runtime must not be merged into path: {runtime}."
            for runtime in runtime_compatibility.unsupported_runtimes[:4]
        )
        risks.extend(runtime_compatibility.implementation_risks[:3])
        risks.extend(runtime_compatibility.runtime_limitations[:2])
    if artifact_critic is not None:
        risks.extend(artifact_critic.runtime_concerns[:2])
    return _dedupe(risks)[:10]


def _capability_merge_risks(
    *,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    artifact_critic: ArtifactCriticProfile | None,
) -> tuple[str, ...]:
    risks: list[str] = []
    if artifact_capability_matrix is None:
        risks.append("Capability merge risks are unresolved without matrix metadata.")
    else:
        risks.extend(artifact_capability_matrix.unsupported_or_risky_capabilities[:4])
        risks.extend(artifact_capability_matrix.capability_risks[:3])
        risks.extend(artifact_capability_matrix.target_weaknesses[:2])
    if artifact_critic is not None:
        risks.extend(artifact_critic.capability_gaps[:2])
    return _dedupe(risks)[:10]


def _composition_risks(
    *,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
    artifact_intelligence_synthesis: ArtifactIntelligenceSynthesisProfile | None,
    dependency_merge_risks: tuple[str, ...],
    runtime_merge_risks: tuple[str, ...],
    capability_merge_risks: tuple[str, ...],
    missing: tuple[str, ...],
) -> tuple[str, ...]:
    risks: list[str] = []
    if multi_artifact_strategy is not None:
        risks.extend(multi_artifact_strategy.risk_areas[:3])
    if artifact_critic is not None and artifact_critic.risk_assessment in {
        "high",
        "blocked",
    }:
        risks.append(f"Artifact Critic risk is {artifact_critic.risk_assessment}.")
    if artifact_refiner is not None:
        risks.extend(artifact_refiner.risk_reductions[:2])
    if artifact_intelligence_synthesis is not None:
        risks.extend(artifact_intelligence_synthesis.major_risks[:3])
        if artifact_intelligence_synthesis.implementation_risk in {"high", "blocked"}:
            risks.append(
                "Synthesis implementation risk constrains merge planning: "
                f"{artifact_intelligence_synthesis.implementation_risk}."
            )
    if dependency_merge_risks:
        risks.append("Dependency risks may make direct artifact merging unsafe.")
    if runtime_merge_risks:
        risks.append("Runtime risks may require runtime notes to remain separate.")
    if capability_merge_risks:
        risks.append("Capability risks may require capability caveats to stay visible.")
    if missing:
        risks.append(
            "Merge confidence is limited by unavailable metadata: "
            + ", ".join(missing[:4])
            + "."
        )
    return _dedupe(risks)[:10]


def _merge_strategy(
    *,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_intelligence_synthesis: ArtifactIntelligenceSynthesisProfile | None,
    composition_risks: tuple[str, ...],
) -> ArtifactMergeStrategy:
    if (
        multi_artifact_strategy is None
        or not multi_artifact_strategy.supporting_artifacts
    ):
        return "single_artifact_no_merge"
    if artifact_dependency_graph is not None and (
        artifact_dependency_graph.blocking_dependencies
        or artifact_dependency_graph.dependency_conflicts
    ):
        return "defer_merge_preserve_separation"
    if artifact_critic is not None and artifact_critic.risk_assessment in {
        "high",
        "blocked",
    }:
        return "defer_merge_preserve_separation"
    if (
        artifact_intelligence_synthesis is not None
        and artifact_intelligence_synthesis.implementation_risk in {"high", "blocked"}
    ):
        return "defer_merge_preserve_separation"
    if len(composition_risks) >= 4:
        return "separated_advisory_sections"
    if multi_artifact_strategy.combination_mode == "primary_with_supporting_sections":
        return "primary_with_supporting_sections"
    return "separated_advisory_sections"


def _composition_strategy(
    *,
    merge_strategy: ArtifactMergeStrategy,
    artifact_plan: ArtifactPlan | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
) -> str:
    if merge_strategy == "single_artifact_no_merge":
        if artifact_plan is None:
            return "Keep response as a single advisory artifact until metadata exists."
        return (
            f"Keep {artifact_plan.artifact_family} output as the only artifact "
            "and append merge caveats as metadata guidance."
        )
    if merge_strategy == "primary_with_supporting_sections":
        return (
            "Compose primary artifact first, then attach supporting sections "
            "using explicit labels and advisory boundaries."
        )
    if merge_strategy == "defer_merge_preserve_separation":
        return (
            "Defer merge; preserve artifact, runtime, capability, and critique "
            "sections as separated advisory metadata."
        )
    if multi_artifact_strategy is not None:
        return (
            f"Use {multi_artifact_strategy.combination_mode} with separated "
            "sections and explicit handoff notes."
        )
    return "Use separated advisory sections with no artifact merge."


def _recommended_merge_path(
    *,
    merge_strategy: ArtifactMergeStrategy,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_intelligence_synthesis: ArtifactIntelligenceSynthesisProfile | None,
) -> str:
    if merge_strategy == "single_artifact_no_merge":
        return "Do not merge; preserve a single artifact path with metadata caveats."
    if merge_strategy == "defer_merge_preserve_separation":
        return "Defer merge and keep artifact boundaries visible until risks resolve."
    if artifact_intelligence_synthesis is not None:
        return (
            "Follow synthesis path as advisory merge guidance: "
            f"{artifact_intelligence_synthesis.recommended_artifact_path}"
        )
    if multi_artifact_strategy is not None:
        return (
            "Merge only at labeled section boundaries after "
            f"{multi_artifact_strategy.primary_artifact.artifact_id}."
        )
    return "Use advisory section composition without artifact merging."


def _alternative_merge_paths(
    *,
    merge_strategy: ArtifactMergeStrategy,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    separation_points: tuple[str, ...],
) -> tuple[str, ...]:
    alternatives: list[str] = []
    if merge_strategy != "single_artifact_no_merge":
        alternatives.append("Alternative: preserve all artifacts as separate sections.")
    if merge_strategy != "primary_with_supporting_sections":
        alternatives.append(
            "Alternative: attach support notes after the primary artifact."
        )
    if multi_artifact_strategy is not None:
        alternatives.extend(multi_artifact_strategy.artifact_combination_strategy[:2])
    if separation_points:
        alternatives.append("Alternative: defer joins at identified separation points.")
    return _dedupe(alternatives)[:8]


def _rejected_merge_paths(
    *,
    merge_strategy: ArtifactMergeStrategy,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    composition_risks: tuple[str, ...],
) -> tuple[str, ...]:
    rejected = [
        "Reject automatic artifact merging because this planner is metadata-only.",
        "Reject artifact modification because merge planning cannot alter artifacts.",
    ]
    if merge_strategy == "defer_merge_preserve_separation":
        rejected.append("Reject direct merge while conflicts or high risks remain.")
    if runtime_compatibility is not None and runtime_compatibility.unsupported_runtimes:
        rejected.append("Reject merging unsupported runtime paths into the artifact.")
    if composition_risks:
        rejected.append("Reject hidden composition because risks must remain visible.")
    return _dedupe(rejected)[:8]


def _hitl_questions(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
    artifact_intelligence_synthesis: ArtifactIntelligenceSynthesisProfile | None,
    composition_risks: tuple[str, ...],
    missing: tuple[str, ...],
) -> tuple[str, ...]:
    questions: list[str] = []
    for profile in (
        artifact_plan,
        artifact_dependency_graph,
        runtime_compatibility,
        artifact_capability_matrix,
        multi_artifact_strategy,
        artifact_critic,
        artifact_refiner,
        artifact_intelligence_synthesis,
    ):
        if profile is not None:
            questions.extend(getattr(profile, "hitl_questions", ())[:1])
    if composition_risks:
        questions.append(
            "Should merge planning preserve separation until risks resolve?"
        )
    if missing:
        questions.append("Should missing merge metadata block composition guidance?")
    return _dedupe(questions)[:8]


def _prompt_guidance(
    merge_strategy: ArtifactMergeStrategy,
) -> tuple[str, ...]:
    return (
        "Use Artifact Merge Planner output as metadata-only merge guidance.",
        (
            "Respect artifact boundaries, join points, separation points, "
            "integration order, rejected paths, and HITL questions without "
            "performing a merge."
        ),
        (
            "Do not merge artifacts, modify artifacts, choose final "
            "implementation, execute runtimes, export, select runtimes, route "
            "providers or models, change previews, trigger workflows, trigger "
            "retries, or escalate autonomously."
        ),
        (
            "Expose merge planner fields as downstream-readable metadata only; "
            "do not implement V4, V5, or V6 systems."
        ),
        f"Treat merge strategy {merge_strategy} as advisory only.",
    )


def _merge_confidence(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
    artifact_intelligence_synthesis: ArtifactIntelligenceSynthesisProfile | None,
    missing: tuple[str, ...],
    composition_risks: tuple[str, ...],
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
            artifact_intelligence_synthesis,
        )
    )
    confidence = 0.24 + (available_count * 0.075)
    if artifact_critic is not None:
        confidence += artifact_critic.critique_confidence * 0.05
    if artifact_refiner is not None:
        confidence += artifact_refiner.refinement_confidence * 0.05
    if artifact_intelligence_synthesis is not None:
        confidence += artifact_intelligence_synthesis.synthesis_confidence * 0.08
    confidence -= min(len(missing) * 0.045, 0.2)
    confidence -= min(len(composition_risks) * 0.012, 0.12)
    return round(min(max(confidence, 0.0), 1.0), 2)


def _merge_summary(
    *,
    merge_strategy: ArtifactMergeStrategy,
    boundaries: tuple[str, ...],
    join_points: tuple[str, ...],
    separation_points: tuple[str, ...],
    composition_risks: tuple[str, ...],
) -> str:
    return (
        f"Artifact merge planning recommends {merge_strategy} across "
        f"{len(boundaries)} boundaries, {len(join_points)} join points, "
        f"{len(separation_points)} separation points, and "
        f"{len(composition_risks)} composition risks as metadata-only guidance."
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
    artifact_intelligence_synthesis: ArtifactIntelligenceSynthesisProfile | None,
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
            f"{len(artifact_dependency_graph.dependency_conflicts)} conflicts."
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
    if artifact_intelligence_synthesis is not None:
        evidence.append(
            "Artifact intelligence synthesis: "
            f"{artifact_intelligence_synthesis.implementation_readiness} "
            "readiness; "
            f"{artifact_intelligence_synthesis.implementation_risk} risk."
        )
    return _dedupe(evidence)[:12]


def _dedupe(values: list[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    for value in values:
        normalized = " ".join(value.split())
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return tuple(deduped)
