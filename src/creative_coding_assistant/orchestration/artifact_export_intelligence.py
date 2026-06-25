"""Bounded Artifact Export Intelligence for V3.3 planning metadata."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration._metadata_utils import _dedupe
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
from creative_coding_assistant.orchestration.artifact_merge_planner import (
    ArtifactMergePlannerProfile,
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

ArtifactExportReadiness = Literal[
    "ready_with_caveats",
    "needs_handoff_metadata",
    "blocked_by_missing_metadata",
    "defer_export",
]

ARTIFACT_EXPORT_INTELLIGENCE_AUTHORITY_BOUNDARY = (
    "The Artifact Export Intelligence engine recommends export targets, "
    "formats, requirements, constraints, risks, package notes, portability, "
    "interoperability, documentation needs, and downstream handoffs from "
    "existing metadata only; it does not export files, write files, generate "
    "packages, modify artifacts, merge artifacts, execute artifacts or "
    "runtimes, select final runtimes, deploy, change routing, change previews, "
    "trigger workflows, trigger retries, trigger escalation behavior, "
    "implement V4 agent routing or escalation, implement V5 execution "
    "optimization, implement V6 Blueprint Export, or implement HOLOiVERSE / "
    "HoloGenesis production export workflows."
)


class ArtifactExportIntelligenceProfile(BaseModel):
    """Inspectable metadata-only export intelligence for planned artifacts."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["artifact_export_intelligence"] = "artifact_export_intelligence"
    export_confidence: float = Field(ge=0, le=1)
    export_summary: str = Field(min_length=1, max_length=520)
    export_targets: tuple[str, ...] = Field(min_length=1, max_length=8)
    preferred_export_target: str = Field(min_length=1, max_length=120)
    export_format_recommendations: tuple[str, ...] = Field(
        min_length=1,
        max_length=10,
    )
    export_readiness: ArtifactExportReadiness
    export_requirements: tuple[str, ...] = Field(min_length=1, max_length=10)
    export_constraints: tuple[str, ...] = Field(min_length=1, max_length=10)
    export_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    runtime_export_notes: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    artifact_package_notes: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=10,
    )
    portability_notes: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    interoperability_notes: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    documentation_requirements: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    downstream_tool_handoffs: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    rejected_export_paths: tuple[str, ...] = Field(min_length=1, max_length=8)
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=ARTIFACT_EXPORT_INTELLIGENCE_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=12)


def derive_artifact_export_intelligence_profile(
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
    artifact_merge_planner: ArtifactMergePlannerProfile | None,
) -> ArtifactExportIntelligenceProfile:
    """Plan export intelligence metadata without exporting or writing files."""

    missing = _missing_sources(
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_critic=artifact_critic,
        artifact_refiner=artifact_refiner,
        artifact_intelligence_synthesis=artifact_intelligence_synthesis,
        artifact_merge_planner=artifact_merge_planner,
    )
    targets = _export_targets(
        artifact_plan=artifact_plan,
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_merge_planner=artifact_merge_planner,
    )
    preferred = _preferred_export_target(
        targets=targets,
        artifact_plan=artifact_plan,
        multi_artifact_strategy=multi_artifact_strategy,
        artifact_merge_planner=artifact_merge_planner,
        missing=missing,
    )
    risks = _export_risks(
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        artifact_critic=artifact_critic,
        artifact_refiner=artifact_refiner,
        artifact_intelligence_synthesis=artifact_intelligence_synthesis,
        artifact_merge_planner=artifact_merge_planner,
        missing=missing,
    )
    readiness = _export_readiness(
        runtime_compatibility=runtime_compatibility,
        artifact_capability_matrix=artifact_capability_matrix,
        artifact_critic=artifact_critic,
        artifact_intelligence_synthesis=artifact_intelligence_synthesis,
        artifact_merge_planner=artifact_merge_planner,
        missing=missing,
        risks=risks,
    )
    return ArtifactExportIntelligenceProfile(
        export_confidence=_export_confidence(
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            artifact_critic=artifact_critic,
            artifact_refiner=artifact_refiner,
            artifact_intelligence_synthesis=artifact_intelligence_synthesis,
            artifact_merge_planner=artifact_merge_planner,
            missing=missing,
            risks=risks,
        ),
        export_summary=_export_summary(
            readiness=readiness,
            preferred_export_target=preferred,
            targets=targets,
            risks=risks,
        ),
        export_targets=targets,
        preferred_export_target=preferred,
        export_format_recommendations=_export_format_recommendations(
            artifact_plan=artifact_plan,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            multi_artifact_strategy=multi_artifact_strategy,
            artifact_merge_planner=artifact_merge_planner,
        ),
        export_readiness=readiness,
        export_requirements=_export_requirements(
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_compatibility=runtime_compatibility,
            artifact_merge_planner=artifact_merge_planner,
        ),
        export_constraints=_export_constraints(
            artifact_capability_matrix=artifact_capability_matrix,
            artifact_merge_planner=artifact_merge_planner,
            missing=missing,
        ),
        export_risks=risks,
        runtime_export_notes=_runtime_export_notes(
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
        ),
        artifact_package_notes=_artifact_package_notes(
            artifact_plan=artifact_plan,
            multi_artifact_strategy=multi_artifact_strategy,
            artifact_merge_planner=artifact_merge_planner,
        ),
        portability_notes=_portability_notes(
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
        ),
        interoperability_notes=_interoperability_notes(
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            artifact_dependency_graph=artifact_dependency_graph,
        ),
        documentation_requirements=_documentation_requirements(
            artifact_plan=artifact_plan,
            runtime_compatibility=runtime_compatibility,
            artifact_merge_planner=artifact_merge_planner,
        ),
        downstream_tool_handoffs=_downstream_tool_handoffs(
            artifact_dependency_graph=artifact_dependency_graph,
            multi_artifact_strategy=multi_artifact_strategy,
            artifact_merge_planner=artifact_merge_planner,
        ),
        rejected_export_paths=_rejected_export_paths(
            readiness=readiness,
            runtime_compatibility=runtime_compatibility,
            artifact_merge_planner=artifact_merge_planner,
            risks=risks,
        ),
        hitl_questions=_hitl_questions(
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            runtime_compatibility=runtime_compatibility,
            artifact_capability_matrix=artifact_capability_matrix,
            multi_artifact_strategy=multi_artifact_strategy,
            artifact_critic=artifact_critic,
            artifact_refiner=artifact_refiner,
            artifact_intelligence_synthesis=artifact_intelligence_synthesis,
            artifact_merge_planner=artifact_merge_planner,
            missing=missing,
            readiness=readiness,
        ),
        prompt_guidance=_prompt_guidance(readiness),
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
            artifact_merge_planner=artifact_merge_planner,
        ),
    )


def artifact_export_intelligence_prompt_lines(
    profile: ArtifactExportIntelligenceProfile,
) -> tuple[str, ...]:
    """Render Artifact Export Intelligence as compact prompt guidance."""

    lines = [
        f"Authority boundary: {profile.authority_boundary}",
        f"Export confidence: {profile.export_confidence:.2f}.",
        f"Export readiness: {profile.export_readiness}.",
        f"Export summary: {profile.export_summary}",
        f"Preferred export target: {profile.preferred_export_target}.",
    ]
    lines.extend(f"Export target: {item}" for item in profile.export_targets)
    lines.extend(
        f"Export format recommendation: {item}"
        for item in profile.export_format_recommendations
    )
    lines.extend(f"Export requirement: {item}" for item in profile.export_requirements)
    lines.extend(f"Export constraint: {item}" for item in profile.export_constraints)
    lines.extend(f"Export risk: {item}" for item in profile.export_risks)
    lines.extend(
        f"Runtime export note: {item}" for item in profile.runtime_export_notes
    )
    lines.extend(
        f"Artifact package note: {item}" for item in profile.artifact_package_notes
    )
    lines.extend(f"Portability note: {item}" for item in profile.portability_notes)
    lines.extend(
        f"Interoperability note: {item}"
        for item in profile.interoperability_notes
    )
    lines.extend(
        f"Documentation requirement: {item}"
        for item in profile.documentation_requirements
    )
    lines.extend(
        f"Downstream tool handoff: {item}"
        for item in profile.downstream_tool_handoffs
    )
    lines.extend(
        f"Rejected export path: {item}" for item in profile.rejected_export_paths
    )
    lines.extend(f"HITL export question: {item}" for item in profile.hitl_questions)
    lines.extend(
        f"Artifact export intelligence guidance: {item}"
        for item in profile.prompt_guidance
    )
    return tuple(lines[:80])


def _missing_sources(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
    artifact_intelligence_synthesis: ArtifactIntelligenceSynthesisProfile | None,
    artifact_merge_planner: ArtifactMergePlannerProfile | None,
) -> tuple[str, ...]:
    missing: list[str] = []
    for label, value in (
        ("artifact plan", artifact_plan),
        ("artifact dependency graph", artifact_dependency_graph),
        ("runtime compatibility", runtime_compatibility),
        ("artifact capability matrix", artifact_capability_matrix),
        ("multi-artifact strategy", multi_artifact_strategy),
        ("artifact critic", artifact_critic),
        ("artifact refiner", artifact_refiner),
        ("artifact intelligence synthesis", artifact_intelligence_synthesis),
        ("artifact merge planner", artifact_merge_planner),
    ):
        if value is None:
            missing.append(label)
    return tuple(missing)


def _export_targets(
    *,
    artifact_plan: ArtifactPlan | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_merge_planner: ArtifactMergePlannerProfile | None,
) -> tuple[str, ...]:
    targets = ["inline_response"]
    if artifact_plan is not None:
        if artifact_plan.artifact_type in {"runnable_code", "debug_patch"}:
            targets.append("single_source_artifact")
        if artifact_plan.artifact_type in {"design_spec", "explanation"}:
            targets.append("documentation_bundle")
        if artifact_plan.artifact_family in {
            "p5_sketch",
            "three_scene",
            "react_three_fiber_scene",
            "hydra_patch",
            "tone_sketch",
            "canvas_sketch",
            "audiovisual_scene",
        }:
            targets.append("runtime_project_bundle")
    if multi_artifact_strategy is not None and (
        multi_artifact_strategy.supporting_artifacts
    ):
        targets.append("multi_artifact_package")
    if artifact_merge_planner is not None and (
        artifact_merge_planner.artifact_separation_points
        or artifact_merge_planner.merge_strategy == "defer_merge_preserve_separation"
    ):
        targets.append("separated_metadata_handoff")
    if artifact_capability_matrix is not None and (
        artifact_capability_matrix.export_fit == "strong"
        or artifact_capability_matrix.portability_fit == "strong"
    ):
        targets.append("portable_asset_handoff")
    if runtime_compatibility is not None and runtime_compatibility.interoperability in {
        "high",
        "medium",
    }:
        targets.append("interoperability_handoff")
    return _dedupe(targets, clip_limit=None)[:8]


def _preferred_export_target(
    *,
    targets: tuple[str, ...],
    artifact_plan: ArtifactPlan | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_merge_planner: ArtifactMergePlannerProfile | None,
    missing: tuple[str, ...],
) -> str:
    if "artifact plan" in missing or "artifact merge planner" in missing:
        return "defer_export_until_metadata_complete"
    if artifact_merge_planner is not None and (
        artifact_merge_planner.merge_strategy == "defer_merge_preserve_separation"
    ):
        return "separated_metadata_handoff"
    if multi_artifact_strategy is not None and (
        multi_artifact_strategy.supporting_artifacts
        and "multi_artifact_package" in targets
    ):
        return "multi_artifact_package"
    if artifact_plan is not None and artifact_plan.artifact_type == "runnable_code":
        return "single_source_artifact"
    return targets[0]


def _export_format_recommendations(
    *,
    artifact_plan: ArtifactPlan | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_merge_planner: ArtifactMergePlannerProfile | None,
) -> tuple[str, ...]:
    recommendations: list[str] = []
    if artifact_plan is not None:
        recommendations.append(
            f"Represent {artifact_plan.artifact_family} as advisory "
            f"{artifact_plan.artifact_type} export metadata."
        )
        recommendations.extend(
            f"Include expected output section: {item}"
            for item in artifact_plan.expected_output_structure[:3]
        )
    if runtime_compatibility is not None:
        recommendations.extend(
            f"Future exporter should preserve runtime note for {runtime}."
            for runtime in runtime_compatibility.preferred_runtimes[:3]
        )
    if artifact_capability_matrix is not None:
        recommendations.append(
            f"Carry export fit {artifact_capability_matrix.export_fit} and "
            f"portability fit {artifact_capability_matrix.portability_fit}."
        )
    if multi_artifact_strategy is not None:
        recommendations.extend(multi_artifact_strategy.artifact_combination_strategy[:2])
    if artifact_merge_planner is not None:
        recommendations.append(artifact_merge_planner.recommended_merge_path)
    if not recommendations:
        recommendations.append(
            "Use inline response metadata until export shape exists."
        )
    return _dedupe(recommendations, clip_limit=None)[:10]


def _export_requirements(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_merge_planner: ArtifactMergePlannerProfile | None,
) -> tuple[str, ...]:
    requirements: list[str] = []
    if artifact_plan is not None:
        requirements.extend(artifact_plan.required_components[:4])
        requirements.extend(artifact_plan.runtime_requirements[:3])
    if artifact_dependency_graph is not None:
        requirements.extend(artifact_dependency_graph.required_upstream_metadata[:3])
    if runtime_compatibility is not None:
        requirements.extend(runtime_compatibility.runtime_requirements[:3])
    if artifact_merge_planner is not None:
        requirements.extend(artifact_merge_planner.integration_order[:2])
    if not requirements:
        requirements.append("Export requirements are unresolved without metadata.")
    return _dedupe(requirements, clip_limit=None)[:10]


def _export_constraints(
    *,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    artifact_merge_planner: ArtifactMergePlannerProfile | None,
    missing: tuple[str, ...],
) -> tuple[str, ...]:
    constraints = [
        "Export Intelligence is metadata-only and cannot write files or packages.",
        "Future exporters must preserve artifact boundaries and user-visible caveats.",
    ]
    if artifact_capability_matrix is not None:
        constraints.extend(artifact_capability_matrix.unsupported_or_risky_capabilities[:3])
    if artifact_merge_planner is not None:
        constraints.extend(artifact_merge_planner.artifact_separation_points[:3])
    if missing:
        constraints.append(
            "Export planning is constrained by missing metadata: "
            + ", ".join(missing[:4])
            + "."
        )
    return _dedupe(constraints, clip_limit=None)[:10]


def _export_risks(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
    artifact_intelligence_synthesis: ArtifactIntelligenceSynthesisProfile | None,
    artifact_merge_planner: ArtifactMergePlannerProfile | None,
    missing: tuple[str, ...],
) -> tuple[str, ...]:
    risks: list[str] = []
    if artifact_plan is not None:
        risks.extend(artifact_plan.implementation_risks[:2])
        risks.extend(artifact_plan.missing_information[:2])
    if artifact_dependency_graph is not None:
        risks.extend(artifact_dependency_graph.blocking_dependencies[:2])
        risks.extend(artifact_dependency_graph.dependency_conflicts[:2])
        risks.extend(artifact_dependency_graph.missing_dependency_risks[:1])
    if runtime_compatibility is not None:
        risks.extend(runtime_compatibility.implementation_risks[:2])
        risks.extend(
            f"Unsupported runtime cannot be exported directly: {runtime}."
            for runtime in runtime_compatibility.unsupported_runtimes[:2]
        )
    if artifact_capability_matrix is not None:
        risks.extend(artifact_capability_matrix.capability_risks[:2])
        risks.extend(artifact_capability_matrix.target_weaknesses[:1])
    if artifact_critic is not None:
        risks.extend(artifact_critic.weaknesses[:2])
        risks.extend(artifact_critic.runtime_concerns[:1])
    if artifact_refiner is not None:
        risks.extend(artifact_refiner.risk_reductions[:2])
    if artifact_intelligence_synthesis is not None:
        risks.extend(artifact_intelligence_synthesis.major_risks[:2])
    if artifact_merge_planner is not None:
        risks.extend(artifact_merge_planner.composition_risks[:2])
        risks.extend(artifact_merge_planner.runtime_merge_risks[:1])
    if missing:
        risks.append(
            "Export confidence is limited by unavailable metadata: "
            + ", ".join(missing[:4])
            + "."
        )
    return _dedupe(risks, clip_limit=None)[:10]


def _export_readiness(
    *,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_intelligence_synthesis: ArtifactIntelligenceSynthesisProfile | None,
    artifact_merge_planner: ArtifactMergePlannerProfile | None,
    missing: tuple[str, ...],
    risks: tuple[str, ...],
) -> ArtifactExportReadiness:
    if missing:
        return "blocked_by_missing_metadata"
    if artifact_merge_planner is not None and (
        artifact_merge_planner.merge_strategy == "defer_merge_preserve_separation"
    ):
        return "defer_export"
    if artifact_critic is not None and artifact_critic.risk_assessment in {
        "high",
        "blocked",
    }:
        return "defer_export"
    if (
        artifact_intelligence_synthesis is not None
        and artifact_intelligence_synthesis.implementation_risk in {"high", "blocked"}
    ):
        return "defer_export"
    if runtime_compatibility is None or artifact_capability_matrix is None:
        return "needs_handoff_metadata"
    if artifact_capability_matrix.export_fit in {"weak", "unsupported"}:
        return "needs_handoff_metadata"
    if runtime_compatibility.portability == "low" or len(risks) >= 6:
        return "needs_handoff_metadata"
    return "ready_with_caveats"


def _runtime_export_notes(
    *,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
) -> tuple[str, ...]:
    notes: list[str] = []
    if runtime_compatibility is not None:
        notes.append(
            "Preferred runtimes are advisory export metadata: "
            + ", ".join(runtime_compatibility.preferred_runtimes or ("none",))
            + "."
        )
        notes.append(f"Runtime portability: {runtime_compatibility.portability}.")
        notes.append(
            f"Runtime interoperability: {runtime_compatibility.interoperability}."
        )
        notes.extend(runtime_compatibility.runtime_limitations[:2])
    if artifact_capability_matrix is not None:
        notes.append(
            "Strongest export targets from capability matrix: "
            + ", ".join(artifact_capability_matrix.strongest_targets or ("none",))
            + "."
        )
    return _dedupe(notes, clip_limit=None)[:10]


def _artifact_package_notes(
    *,
    artifact_plan: ArtifactPlan | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_merge_planner: ArtifactMergePlannerProfile | None,
) -> tuple[str, ...]:
    notes: list[str] = []
    if artifact_plan is not None:
        notes.append(
            f"Package note: preserve {artifact_plan.artifact_type} / "
            f"{artifact_plan.artifact_family} as declared metadata."
        )
    if multi_artifact_strategy is not None:
        notes.append(
            "Package primary artifact first: "
            f"{multi_artifact_strategy.primary_artifact.artifact_id}."
        )
        notes.extend(
            f"Supporting package section: {artifact.artifact_id}."
            for artifact in multi_artifact_strategy.supporting_artifacts[:4]
        )
    if artifact_merge_planner is not None:
        notes.extend(artifact_merge_planner.artifact_boundaries[:3])
        notes.extend(artifact_merge_planner.artifact_join_points[:2])
    if not notes:
        notes.append("Package shape is unresolved until artifact metadata exists.")
    return _dedupe(notes, clip_limit=None)[:10]


def _portability_notes(
    *,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
) -> tuple[str, ...]:
    notes: list[str] = []
    if runtime_compatibility is not None:
        notes.append(f"Runtime portability is {runtime_compatibility.portability}.")
    if artifact_capability_matrix is not None:
        notes.append(
            "Capability portability fit is "
            f"{artifact_capability_matrix.portability_fit}."
        )
        notes.extend(artifact_capability_matrix.target_weaknesses[:2])
    if not notes:
        notes.append(
            "Portability is unresolved without runtime and capability metadata."
        )
    return _dedupe(notes, clip_limit=None)[:8]


def _interoperability_notes(
    *,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
) -> tuple[str, ...]:
    notes: list[str] = []
    if runtime_compatibility is not None:
        notes.append(
            f"Runtime interoperability is {runtime_compatibility.interoperability}."
        )
        notes.extend(runtime_compatibility.dependency_compatibility[:2])
    if artifact_capability_matrix is not None:
        notes.append(
            "Capability interoperability fit is "
            f"{artifact_capability_matrix.interoperability_fit}."
        )
    if artifact_dependency_graph is not None:
        notes.extend(artifact_dependency_graph.downstream_consumers[:3])
    if not notes:
        notes.append("Interoperability is unresolved without dependency metadata.")
    return _dedupe(notes, clip_limit=None)[:8]


def _documentation_requirements(
    *,
    artifact_plan: ArtifactPlan | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_merge_planner: ArtifactMergePlannerProfile | None,
) -> tuple[str, ...]:
    requirements = [
        "Document that export intelligence is advisory metadata only.",
        "Document unsupported or rejected export paths before any future export.",
    ]
    if artifact_plan is not None:
        requirements.extend(
            f"Document expected output: {item}"
            for item in artifact_plan.expected_output_structure[:2]
        )
    if runtime_compatibility is not None:
        requirements.extend(
            f"Document runtime limitation: {item}"
            for item in runtime_compatibility.runtime_limitations[:2]
        )
    if artifact_merge_planner is not None:
        requirements.extend(
            f"Document artifact boundary: {item}"
            for item in artifact_merge_planner.artifact_boundaries[:2]
        )
    return _dedupe(requirements, clip_limit=None)[:8]


def _downstream_tool_handoffs(
    *,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    multi_artifact_strategy: MultiArtifactStrategy | None,
    artifact_merge_planner: ArtifactMergePlannerProfile | None,
) -> tuple[str, ...]:
    handoffs: list[str] = []
    if artifact_dependency_graph is not None:
        handoffs.extend(
            f"Downstream consumer handoff: {item}"
            for item in artifact_dependency_graph.downstream_consumers[:4]
        )
    if multi_artifact_strategy is not None:
        handoffs.extend(multi_artifact_strategy.artifact_handoff_points[:3])
    if artifact_merge_planner is not None:
        handoffs.extend(artifact_merge_planner.artifact_join_points[:2])
    handoffs.append(
        "Future export workflows must consume this metadata explicitly; this "
        "workflow does not trigger export."
    )
    return _dedupe(handoffs, clip_limit=None)[:8]


def _rejected_export_paths(
    *,
    readiness: ArtifactExportReadiness,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_merge_planner: ArtifactMergePlannerProfile | None,
    risks: tuple[str, ...],
) -> tuple[str, ...]:
    rejected = [
        "Reject direct file export because this engine is metadata-only.",
        "Reject package generation because export intelligence cannot write files.",
        "Reject runtime auto-selection because export intelligence is advisory.",
    ]
    if readiness in {"blocked_by_missing_metadata", "defer_export"}:
        rejected.append("Reject production export until blocking metadata resolves.")
    if runtime_compatibility is not None and runtime_compatibility.unsupported_runtimes:
        rejected.append("Reject unsupported runtime export paths.")
    if artifact_merge_planner is not None:
        rejected.extend(artifact_merge_planner.rejected_merge_paths[:1])
    if risks:
        rejected.append(
            "Reject hidden export assumptions because risks must be visible."
        )
    return _dedupe(rejected, clip_limit=None)[:8]


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
    artifact_merge_planner: ArtifactMergePlannerProfile | None,
    missing: tuple[str, ...],
    readiness: ArtifactExportReadiness,
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
        artifact_merge_planner,
    ):
        if profile is not None:
            questions.extend(getattr(profile, "hitl_questions", ())[:1])
    if missing:
        questions.append("Should missing export metadata block export guidance?")
    if readiness in {"blocked_by_missing_metadata", "defer_export"}:
        questions.append("Should export remain deferred until risks are resolved?")
    return _dedupe(questions, clip_limit=None)[:8]


def _prompt_guidance(
    readiness: ArtifactExportReadiness,
) -> tuple[str, ...]:
    return (
        "Use Artifact Export Intelligence as metadata-only export guidance.",
        (
            "Respect export targets, requirements, constraints, risks, "
            "documentation needs, rejected paths, and downstream handoffs "
            "without exporting files."
        ),
        (
            "Do not write files, generate packages, modify artifacts, merge "
            "artifacts, execute runtimes, select final runtimes, deploy, route "
            "providers or models, change previews, trigger workflows, trigger "
            "retries, or escalate autonomously."
        ),
        (
            "Expose export intelligence fields as downstream-readable metadata "
            "only; do not implement V4, V5, V6, HOLOiVERSE, or HoloGenesis."
        ),
        f"Treat export readiness {readiness} as advisory only.",
    )


def _export_confidence(
    *,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    artifact_capability_matrix: ArtifactCapabilityMatrix | None,
    artifact_critic: ArtifactCriticProfile | None,
    artifact_refiner: ArtifactRefinerProfile | None,
    artifact_intelligence_synthesis: ArtifactIntelligenceSynthesisProfile | None,
    artifact_merge_planner: ArtifactMergePlannerProfile | None,
    missing: tuple[str, ...],
    risks: tuple[str, ...],
) -> float:
    available = sum(
        item is not None
        for item in (
            runtime_compatibility,
            artifact_capability_matrix,
            artifact_critic,
            artifact_refiner,
            artifact_intelligence_synthesis,
            artifact_merge_planner,
        )
    )
    confidence = 0.28 + available * 0.075
    if artifact_capability_matrix is not None:
        if artifact_capability_matrix.export_fit == "strong":
            confidence += 0.1
        elif artifact_capability_matrix.export_fit == "moderate":
            confidence += 0.05
        else:
            confidence -= 0.08
    if artifact_critic is not None:
        confidence += artifact_critic.critique_confidence * 0.04
    if artifact_refiner is not None:
        confidence += artifact_refiner.refinement_confidence * 0.04
    if artifact_intelligence_synthesis is not None:
        confidence += artifact_intelligence_synthesis.synthesis_confidence * 0.06
    if artifact_merge_planner is not None:
        confidence += artifact_merge_planner.merge_confidence * 0.05
    confidence -= min(len(missing) * 0.045, 0.24)
    confidence -= min(len(risks) * 0.01, 0.1)
    return round(min(max(confidence, 0.0), 1.0), 2)


def _export_summary(
    *,
    readiness: ArtifactExportReadiness,
    preferred_export_target: str,
    targets: tuple[str, ...],
    risks: tuple[str, ...],
) -> str:
    return (
        f"Artifact export intelligence reports {readiness} readiness with "
        f"{preferred_export_target} preferred across {len(targets)} target "
        f"options and {len(risks)} visible export risks as metadata-only guidance."
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
    artifact_merge_planner: ArtifactMergePlannerProfile | None,
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
            "Dependency consumers: "
            f"{len(artifact_dependency_graph.downstream_consumers)}."
        )
    if runtime_compatibility is not None:
        evidence.append(
            "Runtime export basis: "
            f"{runtime_compatibility.portability} portability; "
            f"{runtime_compatibility.interoperability} interoperability."
        )
    if artifact_capability_matrix is not None:
        evidence.append(
            f"Capability export fit: {artifact_capability_matrix.export_fit}."
        )
    if multi_artifact_strategy is not None:
        evidence.append(
            f"Multi-artifact support count: "
            f"{len(multi_artifact_strategy.supporting_artifacts)}."
        )
    if artifact_critic is not None:
        evidence.append(
            f"Artifact critic risk: {artifact_critic.risk_assessment}."
        )
    if artifact_refiner is not None:
        evidence.append(
            f"Artifact refiner confidence: "
            f"{artifact_refiner.refinement_confidence:.2f}."
        )
    if artifact_intelligence_synthesis is not None:
        evidence.append(
            "Artifact synthesis readiness: "
            f"{artifact_intelligence_synthesis.implementation_readiness}."
        )
    if artifact_merge_planner is not None:
        evidence.append(
            f"Artifact merge strategy: {artifact_merge_planner.merge_strategy}."
        )
    return _dedupe(evidence, clip_limit=None)[:12]
