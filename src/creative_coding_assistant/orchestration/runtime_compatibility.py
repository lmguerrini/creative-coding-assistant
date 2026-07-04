"""Bounded Runtime Compatibility Engine for V3.3 workflows."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration._metadata_utils import _dedupe
from creative_coding_assistant.orchestration.artifact_dependency_graph import (
    ArtifactDependencyGraph,
)
from creative_coding_assistant.orchestration.artifact_planner import (
    ArtifactFamily,
    ArtifactPlan,
)
from creative_coding_assistant.orchestration.creative_constraints import (
    CreativeConstraintSolution,
)
from creative_coding_assistant.orchestration.creative_planning import (
    CreativeExecutionPlan,
)
from creative_coding_assistant.orchestration.creative_tradeoffs import (
    CreativeTradeoffProfile,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityCandidate,
    RuntimeCapabilityComplexity,
    RuntimeCapabilityId,
    RuntimeCapabilityProfile,
)

RuntimeCompatibilityLevel = Literal[
    "compatible",
    "partially_compatible",
    "unsupported",
]
RuntimePortability = Literal["high", "medium", "low"]
RuntimeInteroperability = Literal["high", "medium", "low"]

RUNTIME_COMPATIBILITY_AUTHORITY_BOUNDARY = (
    "The Runtime Compatibility Engine evaluates runtime compatibility for "
    "planned artifacts as inspectable metadata only; it does not execute "
    "runtimes, auto-select runtimes, route providers or models, choose "
    "renderers, change preview behavior, repair runtimes, implement "
    "Multi-Artifact Strategy, implement Artifact Critic, implement Artifact "
    "Refiner, implement V4 multi-agent behavior, or implement V5 execution "
    "optimization."
)

_RUNTIME_LABELS: dict[RuntimeCapabilityId, str] = {
    "p5_js": "p5.js",
    "three_js": "Three.js",
    "react_three_fiber": "React Three Fiber",
    "glsl": "GLSL",
    "hydra": "Hydra",
    "tone_js": "Tone.js",
    "gsap": "GSAP",
    "svg": "SVG",
    "canvas": "Canvas 2D",
}

_RUNTIME_COMPLEXITY: dict[RuntimeCapabilityId, RuntimeCapabilityComplexity] = {
    "p5_js": "low",
    "three_js": "medium",
    "react_three_fiber": "high",
    "glsl": "high",
    "hydra": "medium",
    "tone_js": "medium",
    "gsap": "medium",
    "svg": "low",
    "canvas": "low",
}

_FAMILY_RUNTIME_FIT: dict[
    ArtifactFamily,
    tuple[tuple[RuntimeCapabilityId, ...], tuple[RuntimeCapabilityId, ...]],
] = {
    "p5_sketch": (("p5_js",), ("canvas", "svg")),
    "three_scene": (("three_js",), ("react_three_fiber",)),
    "react_three_fiber_scene": (("react_three_fiber",), ("three_js",)),
    "glsl_shader": (("glsl",), ("three_js", "react_three_fiber")),
    "hydra_patch": (("hydra",), ("glsl",)),
    "tone_sketch": (("tone_js",), ("p5_js",)),
    "canvas_sketch": (("canvas",), ("p5_js", "svg")),
    "audiovisual_scene": (("p5_js", "hydra", "tone_js"), ("canvas", "glsl")),
    "generative_artifact": (
        ("p5_js", "canvas", "glsl"),
        ("three_js", "react_three_fiber", "svg"),
    ),
    "multimodal_reference_artifact": (("p5_js", "canvas"), ("three_js", "svg")),
    "creative_coding_response": (
        ("p5_js", "canvas"),
        ("three_js", "glsl", "svg"),
    ),
}


class RuntimeCompatibilityConfidence(BaseModel):
    """Confidence for one runtime compatibility assessment."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    runtime: RuntimeCapabilityId
    label: str = Field(min_length=1, max_length=80)
    confidence: float = Field(ge=0, le=1)


class RuntimeCompatibilityAssessment(BaseModel):
    """Compatibility assessment for one supported runtime."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    runtime: RuntimeCapabilityId
    label: str = Field(min_length=1, max_length=80)
    compatibility: RuntimeCompatibilityLevel
    confidence: float = Field(ge=0, le=1)
    compatibility_reasons: tuple[str, ...] = Field(min_length=1, max_length=8)
    runtime_requirements: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    runtime_limitations: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    dependency_compatibility: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    expected_implementation_complexity: RuntimeCapabilityComplexity
    portability: RuntimePortability
    interoperability: RuntimeInteroperability
    implementation_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=10)


class RuntimeCompatibilityProfile(BaseModel):
    """Inspectable metadata-only runtime compatibility profile."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["runtime_compatibility_engine"] = "runtime_compatibility_engine"
    compatible_runtimes: tuple[RuntimeCapabilityId, ...] = Field(
        default_factory=tuple,
        max_length=9,
    )
    unsupported_runtimes: tuple[RuntimeCapabilityId, ...] = Field(
        default_factory=tuple,
        max_length=9,
    )
    preferred_runtimes: tuple[RuntimeCapabilityId, ...] = Field(
        default_factory=tuple,
        max_length=3,
    )
    runtime_confidence: tuple[RuntimeCompatibilityConfidence, ...] = Field(
        default_factory=tuple,
        max_length=9,
    )
    compatibility_assessments: tuple[RuntimeCompatibilityAssessment, ...] = Field(
        min_length=1,
        max_length=9,
    )
    runtime_requirements: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    runtime_limitations: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    dependency_compatibility: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=10,
    )
    expected_implementation_complexity: RuntimeCapabilityComplexity
    portability: RuntimePortability
    interoperability: RuntimeInteroperability
    missing_runtime_information: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=10,
    )
    implementation_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=RUNTIME_COMPATIBILITY_AUTHORITY_BOUNDARY,
        max_length=760,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=12)


def derive_runtime_compatibility_profile(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_capabilities: RuntimeCapabilityProfile | None = None,
    creative_plan: CreativeExecutionPlan | None = None,
    creative_constraints: CreativeConstraintSolution | None = None,
    creative_tradeoffs: CreativeTradeoffProfile | None = None,
) -> RuntimeCompatibilityProfile:
    """Evaluate runtime compatibility without changing runtime behavior."""

    candidates = _runtime_candidates(runtime_capabilities)
    assessments = tuple(
        _assessment(
            candidate=candidate,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            creative_plan=creative_plan,
            creative_constraints=creative_constraints,
        )
        for candidate in candidates
    )
    assessments = tuple(
        sorted(
            assessments,
            key=lambda item: (
                _compatibility_rank(item.compatibility),
                item.confidence,
                item.runtime,
            ),
            reverse=True,
        )
    )
    compatible = tuple(
        item.runtime
        for item in assessments
        if item.compatibility in {"compatible", "partially_compatible"}
    )
    unsupported = tuple(
        item.runtime for item in assessments if item.compatibility == "unsupported"
    )
    preferred = tuple(
        item.runtime for item in assessments if item.compatibility == "compatible"
    )[:3]
    if not preferred:
        preferred = compatible[:3]
    missing = _missing_runtime_information(
        route_decision=route_decision,
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_capabilities=runtime_capabilities,
        creative_plan=creative_plan,
    )
    risks = _implementation_risks(
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        creative_tradeoffs=creative_tradeoffs,
        assessments=assessments,
    )
    guidance = _profile_prompt_guidance(preferred, missing)
    top = assessments[0]
    return RuntimeCompatibilityProfile(
        compatible_runtimes=compatible,
        unsupported_runtimes=unsupported,
        preferred_runtimes=preferred,
        runtime_confidence=tuple(
            RuntimeCompatibilityConfidence(
                runtime=item.runtime,
                label=item.label,
                confidence=item.confidence,
            )
            for item in assessments
        ),
        compatibility_assessments=assessments,
        runtime_requirements=_runtime_requirements(
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
            creative_plan=creative_plan,
        ),
        runtime_limitations=_runtime_limitations(assessments),
        dependency_compatibility=_dependency_compatibility(
            artifact_dependency_graph=artifact_dependency_graph,
            assessments=assessments,
        ),
        expected_implementation_complexity=top.expected_implementation_complexity,
        portability=top.portability,
        interoperability=top.interoperability,
        missing_runtime_information=missing,
        implementation_risks=risks,
        hitl_questions=_hitl_questions(missing, risks, unsupported),
        prompt_guidance=guidance,
        evidence=_profile_evidence(
            request=request,
            route_decision=route_decision,
            artifact_plan=artifact_plan,
            runtime_capabilities=runtime_capabilities,
            assessments=assessments,
        ),
    )


def runtime_compatibility_prompt_lines(
    profile: RuntimeCompatibilityProfile,
) -> tuple[str, ...]:
    """Render runtime compatibility metadata as compact prompt guidance."""

    lines = [
        f"Authority boundary: {profile.authority_boundary}",
        (
            "Compatible runtimes (metadata only): "
            + _runtime_list(profile.compatible_runtimes)
            + "."
        ),
        (
            "Preferred runtimes (non-binding): "
            + _runtime_list(profile.preferred_runtimes)
            + "."
        ),
        "Unsupported runtimes: " + _runtime_list(profile.unsupported_runtimes) + ".",
        (
            "Expected implementation complexity: "
            f"{profile.expected_implementation_complexity}."
        ),
        f"Portability: {profile.portability}.",
        f"Interoperability: {profile.interoperability}.",
    ]
    lines.extend(
        f"Runtime confidence: {item.label} {item.confidence:.2f}"
        for item in profile.runtime_confidence[:4]
    )
    lines.extend(
        f"Runtime requirement: {item}" for item in profile.runtime_requirements
    )
    lines.extend(f"Runtime limitation: {item}" for item in profile.runtime_limitations)
    lines.extend(
        f"Dependency compatibility: {item}" for item in profile.dependency_compatibility
    )
    lines.extend(
        f"Missing runtime information: {item}"
        for item in profile.missing_runtime_information
    )
    lines.extend(
        f"Runtime implementation risk: {item}" for item in profile.implementation_risks
    )
    lines.extend(f"HITL runtime question: {item}" for item in profile.hitl_questions)
    lines.extend(
        f"Runtime compatibility guidance: {item}" for item in profile.prompt_guidance
    )
    for assessment in profile.compatibility_assessments[:3]:
        lines.append(
            "Runtime compatibility assessment: "
            f"{assessment.label} ({assessment.runtime}) is "
            f"{assessment.compatibility} at {assessment.confidence:.2f}; "
            f"complexity {assessment.expected_implementation_complexity}; "
            f"portability {assessment.portability}; "
            f"interoperability {assessment.interoperability}."
        )
        lines.append(f"Compatibility reason: {assessment.compatibility_reasons[0]}")
    return tuple(lines[:56])


def _runtime_candidates(
    runtime_capabilities: RuntimeCapabilityProfile | None,
) -> tuple[RuntimeCapabilityCandidate | RuntimeCapabilityId, ...]:
    if runtime_capabilities is not None and runtime_capabilities.candidate_runtimes:
        return runtime_capabilities.candidate_runtimes
    return tuple(_RUNTIME_LABELS)


def _assessment(
    *,
    candidate: RuntimeCapabilityCandidate | RuntimeCapabilityId,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
) -> RuntimeCompatibilityAssessment:
    runtime = (
        candidate.runtime
        if isinstance(candidate, RuntimeCapabilityCandidate)
        else candidate
    )
    label = (
        candidate.label
        if isinstance(candidate, RuntimeCapabilityCandidate)
        else _RUNTIME_LABELS[runtime]
    )
    family = artifact_plan.artifact_family if artifact_plan is not None else None
    primary, secondary = _runtime_fit_for_family(family)
    compatibility = _compatibility_level(
        runtime=runtime,
        primary=primary,
        secondary=secondary,
        candidate=candidate,
        artifact_plan=artifact_plan,
    )
    confidence = _compatibility_confidence(
        runtime=runtime,
        compatibility=compatibility,
        candidate=candidate,
        primary=primary,
        secondary=secondary,
    )
    complexity = _assessment_complexity(
        runtime=runtime,
        candidate=candidate,
        creative_plan=creative_plan,
        creative_constraints=creative_constraints,
    )
    portability = _portability(runtime, compatibility)
    interoperability = _interoperability(
        runtime=runtime,
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
    )
    return RuntimeCompatibilityAssessment(
        runtime=runtime,
        label=label,
        compatibility=compatibility,
        confidence=confidence,
        compatibility_reasons=_compatibility_reasons(
            runtime=runtime,
            family=family,
            compatibility=compatibility,
            primary=primary,
            secondary=secondary,
            candidate=candidate,
        ),
        runtime_requirements=_assessment_requirements(
            runtime=runtime,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
        ),
        runtime_limitations=_assessment_limitations(candidate, compatibility),
        dependency_compatibility=_assessment_dependency_compatibility(
            runtime=runtime,
            compatibility=compatibility,
            artifact_dependency_graph=artifact_dependency_graph,
        ),
        expected_implementation_complexity=complexity,
        portability=portability,
        interoperability=interoperability,
        implementation_risks=_assessment_risks(candidate, compatibility),
        prompt_guidance=_assessment_prompt_guidance(
            runtime=runtime,
            compatibility=compatibility,
        ),
        evidence=_assessment_evidence(candidate, primary, secondary),
    )


def _compatibility_level(
    *,
    runtime: RuntimeCapabilityId,
    primary: tuple[RuntimeCapabilityId, ...],
    secondary: tuple[RuntimeCapabilityId, ...],
    candidate: RuntimeCapabilityCandidate | RuntimeCapabilityId,
    artifact_plan: ArtifactPlan | None,
) -> RuntimeCompatibilityLevel:
    if runtime in primary:
        return "compatible"
    if runtime in secondary:
        return "partially_compatible"
    if artifact_plan is None:
        if isinstance(candidate, RuntimeCapabilityCandidate):
            return (
                "partially_compatible"
                if candidate.suitability != "weak"
                else "unsupported"
            )
        return (
            "partially_compatible" if runtime in {"p5_js", "canvas"} else "unsupported"
        )
    if _generic_artifact_family(artifact_plan.artifact_family):
        if isinstance(candidate, RuntimeCapabilityCandidate):
            return (
                "partially_compatible"
                if candidate.suitability in {"strong", "moderate"}
                else "unsupported"
            )
        return "partially_compatible"
    return "unsupported"


def _compatibility_confidence(
    *,
    runtime: RuntimeCapabilityId,
    compatibility: RuntimeCompatibilityLevel,
    candidate: RuntimeCapabilityCandidate | RuntimeCapabilityId,
    primary: tuple[RuntimeCapabilityId, ...],
    secondary: tuple[RuntimeCapabilityId, ...],
) -> float:
    base = (
        candidate.confidence
        if isinstance(candidate, RuntimeCapabilityCandidate)
        else 0.45
    )
    if compatibility == "compatible":
        base += 0.08 if runtime in primary else 0.02
    elif compatibility == "partially_compatible":
        base += 0.02 if runtime in secondary else -0.04
    else:
        base -= 0.18
    return max(0.1, min(0.98, round(base, 2)))


def _assessment_complexity(
    *,
    runtime: RuntimeCapabilityId,
    candidate: RuntimeCapabilityCandidate | RuntimeCapabilityId,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
) -> RuntimeCapabilityComplexity:
    complexity = (
        candidate.implementation_complexity
        if isinstance(candidate, RuntimeCapabilityCandidate)
        else _RUNTIME_COMPLEXITY[runtime]
    )
    if creative_plan is not None and creative_plan.expected_complexity == "high":
        return "high"
    if (
        creative_constraints is not None
        and creative_constraints.complexity_pressure == "high"
    ):
        return "high"
    return complexity


def _compatibility_reasons(
    *,
    runtime: RuntimeCapabilityId,
    family: ArtifactFamily | None,
    compatibility: RuntimeCompatibilityLevel,
    primary: tuple[RuntimeCapabilityId, ...],
    secondary: tuple[RuntimeCapabilityId, ...],
    candidate: RuntimeCapabilityCandidate | RuntimeCapabilityId,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if family is None:
        reasons.append("Artifact family is unavailable; compatibility is inferred.")
    elif runtime in primary:
        reasons.append(f"{_RUNTIME_LABELS[runtime]} directly supports {family}.")
    elif runtime in secondary:
        reasons.append(f"{_RUNTIME_LABELS[runtime]} can partially support {family}.")
    else:
        reasons.append(f"{_RUNTIME_LABELS[runtime]} is not a direct fit for {family}.")
    if isinstance(candidate, RuntimeCapabilityCandidate):
        reasons.append(
            f"Runtime Capability Reasoner suitability is {candidate.suitability}."
        )
        reasons.extend(candidate.strengths[:2])
    reasons.append(f"Compatibility result: {compatibility}.")
    return _dedupe(reasons)[:8]


def _assessment_requirements(
    *,
    runtime: RuntimeCapabilityId,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
) -> tuple[str, ...]:
    requirements: list[str] = [f"Keep compatibility assessment scoped to {runtime}."]
    if artifact_plan is not None:
        requirements.extend(artifact_plan.runtime_requirements[:3])
    if artifact_dependency_graph is not None:
        requirements.extend(artifact_dependency_graph.runtime_facing_dependencies[:3])
    return _dedupe(requirements)[:8]


def _assessment_limitations(
    candidate: RuntimeCapabilityCandidate | RuntimeCapabilityId,
    compatibility: RuntimeCompatibilityLevel,
) -> tuple[str, ...]:
    limitations: list[str] = []
    if isinstance(candidate, RuntimeCapabilityCandidate):
        limitations.extend(candidate.limitations[:4])
    if compatibility == "unsupported":
        limitations.append(
            "Unsupported runtime should not be used as an output target."
        )
    elif compatibility == "partially_compatible":
        limitations.append("Partial compatibility requires explicit caveats in output.")
    limitations.append("Compatibility metadata must not change runtime execution.")
    return _dedupe(limitations)[:8]


def _assessment_dependency_compatibility(
    *,
    runtime: RuntimeCapabilityId,
    compatibility: RuntimeCompatibilityLevel,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
) -> tuple[str, ...]:
    if artifact_dependency_graph is None:
        return ("Dependency graph unavailable; runtime dependency fit is inferred.",)
    dependencies = list(artifact_dependency_graph.runtime_facing_dependencies[:3])
    if not dependencies:
        dependencies.append("No explicit runtime-facing dependency is declared.")
    status = (
        "satisfies"
        if compatibility == "compatible"
        else "partially satisfies"
        if compatibility == "partially_compatible"
        else "does not satisfy"
    )
    return tuple(
        f"{_RUNTIME_LABELS[runtime]} {status}: {dependency}"
        for dependency in dependencies
    )[:8]


def _assessment_risks(
    candidate: RuntimeCapabilityCandidate | RuntimeCapabilityId,
    compatibility: RuntimeCompatibilityLevel,
) -> tuple[str, ...]:
    risks: list[str] = []
    if isinstance(candidate, RuntimeCapabilityCandidate):
        risks.extend(candidate.risks[:3])
    if compatibility == "unsupported":
        risks.append(
            "Unsupported compatibility can mislead generation if treated as selection."
        )
    risks.append("Do not use compatibility metadata to auto-select runtimes.")
    return _dedupe(risks)[:8]


def _assessment_prompt_guidance(
    *,
    runtime: RuntimeCapabilityId,
    compatibility: RuntimeCompatibilityLevel,
) -> tuple[str, ...]:
    guidance = [
        f"Treat {_RUNTIME_LABELS[runtime]} compatibility as metadata only.",
        (
            "Do not change runtime, renderer, provider, or preview behavior "
            "from this profile."
        ),
    ]
    if compatibility == "unsupported":
        guidance.append("Mention unsupported runtime limitations if relevant.")
    elif compatibility == "partially_compatible":
        guidance.append("Name partial compatibility caveats before implementation.")
    else:
        guidance.append("Use compatible runtime notes only to explain feasibility.")
    return tuple(guidance[:8])


def _assessment_evidence(
    candidate: RuntimeCapabilityCandidate | RuntimeCapabilityId,
    primary: tuple[RuntimeCapabilityId, ...],
    secondary: tuple[RuntimeCapabilityId, ...],
) -> tuple[str, ...]:
    runtime = (
        candidate.runtime
        if isinstance(candidate, RuntimeCapabilityCandidate)
        else candidate
    )
    evidence = [
        "Primary family runtimes: " + ", ".join(primary or ("none",)) + ".",
        "Secondary family runtimes: " + ", ".join(secondary or ("none",)) + ".",
    ]
    if isinstance(candidate, RuntimeCapabilityCandidate):
        evidence.append(f"Candidate confidence: {candidate.confidence:.2f}.")
        evidence.extend(candidate.evidence[:2])
    evidence.append(f"Runtime evaluated: {runtime}.")
    return _dedupe(evidence)[:10]


def _runtime_requirements(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    creative_plan: CreativeExecutionPlan | None,
) -> tuple[str, ...]:
    requirements: list[str] = []
    if artifact_plan is not None:
        requirements.extend(artifact_plan.runtime_requirements)
    if artifact_dependency_graph is not None:
        requirements.extend(artifact_dependency_graph.runtime_facing_dependencies)
    if creative_plan is not None and creative_plan.recommended_runtime is not None:
        requirements.append(
            f"Existing planning runtime hint: {creative_plan.recommended_runtime}."
        )
    return _dedupe(requirements)[:10]


def _runtime_limitations(
    assessments: tuple[RuntimeCompatibilityAssessment, ...],
) -> tuple[str, ...]:
    limitations: list[str] = []
    for assessment in assessments[:4]:
        limitations.extend(
            f"{assessment.label}: {item}" for item in assessment.runtime_limitations[:2]
        )
    return _dedupe(limitations)[:10]


def _dependency_compatibility(
    *,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    assessments: tuple[RuntimeCompatibilityAssessment, ...],
) -> tuple[str, ...]:
    dependencies: list[str] = []
    if artifact_dependency_graph is not None:
        dependencies.extend(artifact_dependency_graph.runtime_facing_dependencies[:4])
        dependencies.extend(artifact_dependency_graph.dependency_conflicts[:2])
    if assessments:
        dependencies.append(
            f"Top runtime dependency fit: {assessments[0].label} "
            f"{assessments[0].compatibility}."
        )
    return _dedupe(dependencies)[:10]


def _missing_runtime_information(
    *,
    route_decision: RouteDecision | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    creative_plan: CreativeExecutionPlan | None,
) -> tuple[str, ...]:
    missing: list[str] = []
    if route_decision is None or not route_decision.domains:
        missing.append("Route/domain metadata is inferred or unavailable.")
    if runtime_capabilities is None:
        missing.append("Runtime Capability Reasoner metadata is unavailable.")
    if artifact_plan is None:
        missing.append("Artifact Plan metadata is unavailable.")
    else:
        missing.extend(
            item
            for item in artifact_plan.missing_information
            if "runtime" in item.lower() or "domain" in item.lower()
        )
    if artifact_dependency_graph is None:
        missing.append("Artifact Dependency Graph metadata is unavailable.")
    elif not artifact_dependency_graph.runtime_facing_dependencies:
        missing.append("No runtime-facing dependency was declared by the graph.")
    if creative_plan is None or creative_plan.recommended_runtime is None:
        missing.append("Creative plan does not provide an explicit runtime hint.")
    return _dedupe(missing)[:10]


def _implementation_risks(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
    assessments: tuple[RuntimeCompatibilityAssessment, ...],
) -> tuple[str, ...]:
    risks: list[str] = []
    if artifact_plan is not None:
        risks.extend(artifact_plan.implementation_risks[:3])
    if artifact_dependency_graph is not None:
        risks.extend(artifact_dependency_graph.dependency_conflicts[:3])
        risks.extend(artifact_dependency_graph.missing_dependency_risks[:2])
    if creative_tradeoffs is not None:
        risks.extend(creative_tradeoffs.runtime_risks[:3])
    for assessment in assessments[:3]:
        risks.extend(
            f"{assessment.label}: {item}"
            for item in assessment.implementation_risks[:2]
        )
    return _dedupe(risks)[:10]


def _hitl_questions(
    missing: tuple[str, ...],
    risks: tuple[str, ...],
    unsupported: tuple[RuntimeCapabilityId, ...],
) -> tuple[str, ...]:
    questions = [
        f"Should we resolve this missing runtime compatibility input: {item}"
        for item in missing[:3]
    ]
    questions.extend(
        f"Should this runtime compatibility risk constrain generation: {item}"
        for item in risks[:3]
    )
    if unsupported:
        questions.append(
            "Should unsupported runtimes be explicitly excluded from the response: "
            + _runtime_list(unsupported[:3])
            + "?"
        )
    return tuple(questions[:8])


def _profile_prompt_guidance(
    preferred: tuple[RuntimeCapabilityId, ...],
    missing: tuple[str, ...],
) -> tuple[str, ...]:
    guidance = [
        "Use Runtime Compatibility Engine output as compatibility metadata only.",
        (
            "Do not use compatibility metadata to auto-select runtimes, route "
            "providers or models, choose renderers, execute code, repair "
            "runtimes, or change preview behavior."
        ),
        "Treat preferred runtimes as non-binding feasibility guidance: "
        + _runtime_list(preferred)
        + ".",
        (
            "When compatibility is partial or unsupported, state the limitation "
            "instead of implying execution support."
        ),
    ]
    if missing:
        guidance.append("Surface missing runtime information before expanding scope.")
    return tuple(guidance[:8])


def _profile_evidence(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    artifact_plan: ArtifactPlan | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    assessments: tuple[RuntimeCompatibilityAssessment, ...],
) -> tuple[str, ...]:
    evidence = [f"Mode: {request.mode.value}."]
    if route_decision is not None:
        evidence.append(f"Route selected: {route_decision.route.value}.")
        if route_decision.domains:
            evidence.append(
                "Domains: "
                + ", ".join(domain.value for domain in route_decision.domains)
                + "."
            )
    if artifact_plan is not None:
        evidence.append(
            f"Artifact: {artifact_plan.artifact_type}; {artifact_plan.artifact_family}."
        )
    if runtime_capabilities is not None:
        evidence.append(
            "Upstream runtime candidates: "
            + ", ".join(runtime_capabilities.likely_candidates)
            + "."
        )
    evidence.append(
        "Compatibility order: "
        + ", ".join(
            f"{item.runtime}:{item.compatibility}:{item.confidence:.2f}"
            for item in assessments[:4]
        )
        + "."
    )
    return _dedupe(evidence)[:12]


def _runtime_fit_for_family(
    family: ArtifactFamily | None,
) -> tuple[tuple[RuntimeCapabilityId, ...], tuple[RuntimeCapabilityId, ...]]:
    if family is None:
        return (("p5_js", "canvas"), ("three_js", "glsl", "svg"))
    return _FAMILY_RUNTIME_FIT[family]


def _generic_artifact_family(family: ArtifactFamily) -> bool:
    return family in {
        "generative_artifact",
        "multimodal_reference_artifact",
        "creative_coding_response",
    }


def _portability(
    runtime: RuntimeCapabilityId,
    compatibility: RuntimeCompatibilityLevel,
) -> RuntimePortability:
    if compatibility == "unsupported":
        return "low"
    if runtime in {"p5_js", "canvas", "svg"}:
        return "high"
    if runtime in {"three_js", "tone_js", "gsap"}:
        return "medium"
    return "low"


def _interoperability(
    *,
    runtime: RuntimeCapabilityId,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
) -> RuntimeInteroperability:
    if runtime in {"p5_js", "canvas", "svg", "gsap"}:
        base: RuntimeInteroperability = "high"
    elif runtime in {"three_js", "react_three_fiber", "tone_js"}:
        base = "medium"
    else:
        base = "low"
    if artifact_plan is not None and artifact_plan.artifact_family == "glsl_shader":
        return "medium" if runtime in {"three_js", "react_three_fiber"} else base
    if (
        artifact_dependency_graph is not None
        and artifact_dependency_graph.dependency_conflicts
        and base == "high"
    ):
        return "medium"
    return base


def _compatibility_rank(level: RuntimeCompatibilityLevel) -> int:
    return {
        "compatible": 3,
        "partially_compatible": 2,
        "unsupported": 1,
    }[level]


def _runtime_list(runtimes: tuple[RuntimeCapabilityId, ...]) -> str:
    if not runtimes:
        return "none"
    return ", ".join(_RUNTIME_LABELS[item] for item in runtimes)
