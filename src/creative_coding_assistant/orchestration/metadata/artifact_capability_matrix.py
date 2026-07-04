"""Bounded Artifact Capability Matrix for V3.3 workflows."""

from __future__ import annotations

from dataclasses import dataclass
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
from creative_coding_assistant.orchestration.creative_strategy import (
    CreativeStrategyProfile,
)
from creative_coding_assistant.orchestration.creative_technique import (
    CreativeTechniqueProfile,
)
from creative_coding_assistant.orchestration.creative_tradeoffs import (
    CreativeTradeoffProfile,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityCandidate,
    RuntimeCapabilityId,
    RuntimeCapabilityProfile,
)
from creative_coding_assistant.orchestration.runtime_compatibility import (
    RuntimeCompatibilityAssessment,
    RuntimeCompatibilityLevel,
    RuntimeCompatibilityProfile,
)

ArtifactCapabilityFit = Literal["strong", "moderate", "weak", "unsupported"]

ARTIFACT_CAPABILITY_MATRIX_AUTHORITY_BOUNDARY = (
    "The Artifact Capability Matrix describes runtime and artifact target "
    "capabilities as inspectable planning metadata only; it does not "
    "auto-select runtimes, change execution behavior, route providers or "
    "models, choose renderers, change preview behavior, implement "
    "Multi-Artifact Strategy, implement Artifact Critic, implement Artifact "
    "Refiner, implement Artifact Merge Planner, implement Artifact Export "
    "Intelligence, implement V4 multi-agent behavior, or implement V5 "
    "execution optimization."
)

_FIT_SCORE: dict[ArtifactCapabilityFit, int] = {
    "strong": 3,
    "moderate": 2,
    "weak": 1,
    "unsupported": 0,
}


class ArtifactCapabilityConfidence(BaseModel):
    """Confidence for one target capability profile."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    target: RuntimeCapabilityId
    label: str = Field(min_length=1, max_length=80)
    confidence: float = Field(ge=0, le=1)


class ArtifactCapabilityProfile(BaseModel):
    """Capability profile for one runtime or artifact target."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    target: RuntimeCapabilityId
    label: str = Field(min_length=1, max_length=80)
    capability_confidence: float = Field(ge=0, le=1)
    capability_reasons: tuple[str, ...] = Field(min_length=1, max_length=8)
    strengths: tuple[str, ...] = Field(min_length=1, max_length=8)
    weaknesses: tuple[str, ...] = Field(min_length=1, max_length=8)
    unsupported_capabilities: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    risky_capabilities: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    artifact_fit: ArtifactCapabilityFit
    creative_fit: ArtifactCapabilityFit
    generative_fit: ArtifactCapabilityFit
    interaction_fit: ArtifactCapabilityFit
    audiovisual_fit: ArtifactCapabilityFit
    export_fit: ArtifactCapabilityFit
    interoperability_fit: ArtifactCapabilityFit
    portability_fit: ArtifactCapabilityFit
    capability_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=10)


class ArtifactCapabilityMatrix(BaseModel):
    """Inspectable metadata-only capability matrix for planned artifacts."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["artifact_capability_matrix"] = "artifact_capability_matrix"
    capability_profiles: tuple[ArtifactCapabilityProfile, ...] = Field(
        min_length=1,
        max_length=9,
    )
    strongest_targets: tuple[RuntimeCapabilityId, ...] = Field(
        default_factory=tuple,
        max_length=3,
    )
    weakest_targets: tuple[RuntimeCapabilityId, ...] = Field(
        default_factory=tuple,
        max_length=3,
    )
    target_strengths: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    target_weaknesses: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    unsupported_or_risky_capabilities: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=12,
    )
    capability_confidence: tuple[ArtifactCapabilityConfidence, ...] = Field(
        default_factory=tuple,
        max_length=9,
    )
    artifact_fit: ArtifactCapabilityFit
    creative_fit: ArtifactCapabilityFit
    generative_fit: ArtifactCapabilityFit
    interaction_fit: ArtifactCapabilityFit
    audiovisual_fit: ArtifactCapabilityFit
    export_fit: ArtifactCapabilityFit
    interoperability_fit: ArtifactCapabilityFit
    portability_fit: ArtifactCapabilityFit
    missing_capability_information: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=10,
    )
    capability_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=ARTIFACT_CAPABILITY_MATRIX_AUTHORITY_BOUNDARY,
        max_length=820,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=12)


def derive_artifact_capability_matrix(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_capabilities: RuntimeCapabilityProfile | None = None,
    runtime_compatibility: RuntimeCompatibilityProfile | None = None,
    creative_plan: CreativeExecutionPlan | None = None,
    creative_constraints: CreativeConstraintSolution | None = None,
    creative_strategy: CreativeStrategyProfile | None = None,
    creative_techniques: CreativeTechniqueProfile | None = None,
    creative_tradeoffs: CreativeTradeoffProfile | None = None,
) -> ArtifactCapabilityMatrix:
    """Derive target capability metadata without changing execution behavior."""

    candidates = _candidate_map(runtime_capabilities)
    assessments = _compatibility_map(runtime_compatibility)
    targets = _matrix_targets(
        runtime_capabilities=runtime_capabilities,
        runtime_compatibility=runtime_compatibility,
    )
    profiles = tuple(
        sorted(
            (
                _capability_profile(
                    target=target,
                    candidate=candidates.get(target),
                    compatibility=assessments.get(target),
                    artifact_plan=artifact_plan,
                    artifact_dependency_graph=artifact_dependency_graph,
                    creative_plan=creative_plan,
                    creative_constraints=creative_constraints,
                    creative_strategy=creative_strategy,
                    creative_techniques=creative_techniques,
                )
                for target in targets
            ),
            key=lambda item: (
                _profile_score(item),
                item.capability_confidence,
                item.target,
            ),
            reverse=True,
        )
    )
    missing = _missing_capability_information(
        route_decision=route_decision,
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_capabilities=runtime_capabilities,
        runtime_compatibility=runtime_compatibility,
        creative_plan=creative_plan,
    )
    risks = _capability_risks(
        artifact_plan=artifact_plan,
        artifact_dependency_graph=artifact_dependency_graph,
        runtime_compatibility=runtime_compatibility,
        creative_tradeoffs=creative_tradeoffs,
        profiles=profiles,
    )
    strongest = tuple(
        profile.target for profile in profiles if profile.artifact_fit != "unsupported"
    )[:3]
    weakest = tuple(profile.target for profile in reversed(profiles))[:3]
    top = profiles[0]
    return ArtifactCapabilityMatrix(
        capability_profiles=profiles,
        strongest_targets=strongest,
        weakest_targets=weakest,
        target_strengths=_target_strengths(profiles),
        target_weaknesses=_target_weaknesses(profiles),
        unsupported_or_risky_capabilities=_unsupported_or_risky_capabilities(
            profiles,
        ),
        capability_confidence=tuple(
            ArtifactCapabilityConfidence(
                target=profile.target,
                label=profile.label,
                confidence=profile.capability_confidence,
            )
            for profile in profiles
        ),
        artifact_fit=top.artifact_fit,
        creative_fit=top.creative_fit,
        generative_fit=top.generative_fit,
        interaction_fit=top.interaction_fit,
        audiovisual_fit=top.audiovisual_fit,
        export_fit=top.export_fit,
        interoperability_fit=top.interoperability_fit,
        portability_fit=top.portability_fit,
        missing_capability_information=missing,
        capability_risks=risks,
        hitl_questions=_hitl_questions(
            missing=missing,
            risks=risks,
            profiles=profiles,
        ),
        prompt_guidance=_profile_prompt_guidance(strongest, missing),
        evidence=_matrix_evidence(
            request=request,
            route_decision=route_decision,
            artifact_plan=artifact_plan,
            runtime_capabilities=runtime_capabilities,
            runtime_compatibility=runtime_compatibility,
            profiles=profiles,
        ),
    )


def artifact_capability_matrix_prompt_lines(
    matrix: ArtifactCapabilityMatrix,
) -> tuple[str, ...]:
    """Render capability matrix metadata as compact prompt guidance."""

    lines = [
        f"Authority boundary: {matrix.authority_boundary}",
        "Strongest targets (non-binding): " + _target_list(matrix.strongest_targets),
        "Weakest targets: " + _target_list(matrix.weakest_targets),
        f"Artifact fit: {matrix.artifact_fit}.",
        f"Creative fit: {matrix.creative_fit}.",
        f"Generative fit: {matrix.generative_fit}.",
        f"Interaction fit: {matrix.interaction_fit}.",
        f"Audiovisual fit: {matrix.audiovisual_fit}.",
        f"Export fit: {matrix.export_fit}.",
        f"Interoperability fit: {matrix.interoperability_fit}.",
        f"Portability fit: {matrix.portability_fit}.",
    ]
    lines.extend(f"Target strength: {item}" for item in matrix.target_strengths[:5])
    lines.extend(f"Target weakness: {item}" for item in matrix.target_weaknesses[:5])
    lines.extend(
        f"Unsupported or risky capability: {item}"
        for item in matrix.unsupported_or_risky_capabilities[:5]
    )
    lines.extend(
        f"Missing capability information: {item}"
        for item in matrix.missing_capability_information
    )
    lines.extend(f"Capability risk: {item}" for item in matrix.capability_risks)
    lines.extend(f"HITL capability question: {item}" for item in matrix.hitl_questions)
    lines.extend(
        f"Capability matrix guidance: {item}" for item in matrix.prompt_guidance
    )
    for profile in matrix.capability_profiles[:3]:
        lines.append(
            "Capability profile: "
            f"{profile.label} ({profile.target}); "
            f"artifact {profile.artifact_fit}; "
            f"creative {profile.creative_fit}; "
            f"generative {profile.generative_fit}; "
            f"interaction {profile.interaction_fit}; "
            f"audiovisual {profile.audiovisual_fit}; "
            f"export {profile.export_fit}; "
            f"interoperability {profile.interoperability_fit}; "
            f"portability {profile.portability_fit}; "
            f"confidence {profile.capability_confidence:.2f}."
        )
        lines.append(f"Capability reason: {profile.capability_reasons[0]}")
        lines.append(f"Capability strength: {profile.strengths[0]}")
        lines.append(f"Capability weakness: {profile.weaknesses[0]}")
    return tuple(lines[:64])


@dataclass(frozen=True)
class _TargetCapabilityMetadata:
    target: RuntimeCapabilityId
    label: str
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    unsupported: tuple[str, ...]
    risks: tuple[str, ...]
    generative_fit: ArtifactCapabilityFit
    interaction_fit: ArtifactCapabilityFit
    audiovisual_fit: ArtifactCapabilityFit
    export_fit: ArtifactCapabilityFit
    interoperability_fit: ArtifactCapabilityFit
    portability_fit: ArtifactCapabilityFit


_TARGET_CAPABILITIES: dict[RuntimeCapabilityId, _TargetCapabilityMetadata] = {
    "p5_js": _TargetCapabilityMetadata(
        target="p5_js",
        label="p5.js",
        strengths=(
            "Fast iteration for sketches, particles, geometry, and interaction.",
            "Simple setup/draw lifecycle maps well to single-file runnable artifacts.",
        ),
        weaknesses=(
            "Deep 3D scenes require simplification or another target.",
            "Large particle counts can pressure frame rate.",
        ),
        unsupported=("Native shader pipelines require additional scaffolding.",),
        risks=("Dense animation can become CPU-bound without caps.",),
        generative_fit="strong",
        interaction_fit="strong",
        audiovisual_fit="strong",
        export_fit="moderate",
        interoperability_fit="strong",
        portability_fit="strong",
    ),
    "three_js": _TargetCapabilityMetadata(
        target="three_js",
        label="Three.js",
        strengths=(
            "Strong fit for 3D scenes, cameras, materials, and lighting.",
            "Can host shader and geometry systems in browser runtimes.",
        ),
        weaknesses=(
            "Requires more boilerplate than sketch-oriented runtimes.",
            "Complex scene graphs increase implementation burden.",
        ),
        unsupported=("Audio sequencing is not native without companion libraries.",),
        risks=("Scene complexity can exceed bounded generation scope.",),
        generative_fit="moderate",
        interaction_fit="strong",
        audiovisual_fit="moderate",
        export_fit="moderate",
        interoperability_fit="moderate",
        portability_fit="moderate",
    ),
    "react_three_fiber": _TargetCapabilityMetadata(
        target="react_three_fiber",
        label="React Three Fiber",
        strengths=(
            "Good fit for React-hosted 3D component structure.",
            "Supports declarative composition around Three.js scenes.",
        ),
        weaknesses=(
            "Higher setup complexity than direct Three.js or p5.js.",
            "Less suitable when the expected output is a single sketch file.",
        ),
        unsupported=("Non-React sketch artifacts should avoid this target.",),
        risks=("React integration can add dependencies outside a bounded artifact.",),
        generative_fit="moderate",
        interaction_fit="strong",
        audiovisual_fit="moderate",
        export_fit="moderate",
        interoperability_fit="moderate",
        portability_fit="weak",
    ),
    "glsl": _TargetCapabilityMetadata(
        target="glsl",
        label="GLSL",
        strengths=(
            "Strong fit for fragment shaders, fields, and visual texture logic.",
            "Excellent for dense pixel-level generative visuals.",
        ),
        weaknesses=(
            "Requires host runtime scaffolding to run in a browser.",
            "Interaction and UI structure are indirect.",
        ),
        unsupported=("Standalone application structure is unsupported by GLSL alone.",),
        risks=("Shader-only output can mismatch requests for full sketches.",),
        generative_fit="strong",
        interaction_fit="weak",
        audiovisual_fit="moderate",
        export_fit="weak",
        interoperability_fit="weak",
        portability_fit="weak",
    ),
    "hydra": _TargetCapabilityMetadata(
        target="hydra",
        label="Hydra",
        strengths=(
            "Strong live-code target for video synthesis and feedback patterns.",
            "Good for audiovisual and signal-reactive visual language.",
        ),
        weaknesses=(
            "Less suitable for general-purpose sketch structure.",
            "Output portability depends on a Hydra host.",
        ),
        unsupported=("General DOM/UI application structure is unsupported.",),
        risks=("Hydra syntax can be too narrow for broad artifact requests.",),
        generative_fit="moderate",
        interaction_fit="moderate",
        audiovisual_fit="strong",
        export_fit="weak",
        interoperability_fit="weak",
        portability_fit="weak",
    ),
    "tone_js": _TargetCapabilityMetadata(
        target="tone_js",
        label="Tone.js",
        strengths=(
            "Strong fit for synthesis, sequencing, and audio timing.",
            "Useful companion target for audiovisual planning.",
        ),
        weaknesses=(
            "Visual artifact structure requires a separate visual runtime.",
            "Browser audio policies can require explicit interaction.",
        ),
        unsupported=("Standalone visual sketches are unsupported by Tone.js alone.",),
        risks=("Audio-only capability can mismatch visual-first requests.",),
        generative_fit="moderate",
        interaction_fit="moderate",
        audiovisual_fit="strong",
        export_fit="weak",
        interoperability_fit="moderate",
        portability_fit="moderate",
    ),
    "gsap": _TargetCapabilityMetadata(
        target="gsap",
        label="GSAP",
        strengths=(
            "Strong fit for timeline animation and UI motion choreography.",
            "Works well as a companion to DOM, SVG, and canvas artifacts.",
        ),
        weaknesses=(
            "Does not define the visual artifact by itself.",
            "Creative coding sketches need a rendering target alongside GSAP.",
        ),
        unsupported=("Standalone shader, audio, or sketch output is unsupported.",),
        risks=("Animation timeline logic can distract from primary artifact code.",),
        generative_fit="weak",
        interaction_fit="moderate",
        audiovisual_fit="weak",
        export_fit="moderate",
        interoperability_fit="strong",
        portability_fit="moderate",
    ),
    "svg": _TargetCapabilityMetadata(
        target="svg",
        label="SVG",
        strengths=(
            "Strong fit for crisp vector composition and exportable geometry.",
            "Highly portable for static or lightly animated artifacts.",
        ),
        weaknesses=(
            "Dense pixel effects and high particle counts are a poor fit.",
            "Complex real-time animation can become cumbersome.",
        ),
        unsupported=("Audio synthesis and shader effects are unsupported.",),
        risks=("Overusing SVG for dynamic simulation can reduce performance.",),
        generative_fit="moderate",
        interaction_fit="moderate",
        audiovisual_fit="weak",
        export_fit="strong",
        interoperability_fit="strong",
        portability_fit="strong",
    ),
    "canvas": _TargetCapabilityMetadata(
        target="canvas",
        label="Canvas 2D",
        strengths=(
            "Strong browser target for custom 2D drawing and generative visuals.",
            "Portable fallback for sketch-like output without p5.js dependency.",
        ),
        weaknesses=(
            "Requires manual lifecycle, input, and state management.",
            "Less ergonomic than p5.js for rapid creative sketch output.",
        ),
        unsupported=("Native 3D and audio synthesis require separate systems.",),
        risks=("Manual rendering loops can become hard to inspect if overbuilt.",),
        generative_fit="strong",
        interaction_fit="strong",
        audiovisual_fit="moderate",
        export_fit="moderate",
        interoperability_fit="strong",
        portability_fit="strong",
    ),
}

_FAMILY_TARGETS: dict[
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


def _matrix_targets(
    *,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
) -> tuple[RuntimeCapabilityId, ...]:
    targets: list[RuntimeCapabilityId] = []
    if runtime_compatibility is not None:
        targets.extend(
            assessment.runtime
            for assessment in runtime_compatibility.compatibility_assessments
        )
    if runtime_capabilities is not None:
        targets.extend(
            candidate.runtime for candidate in runtime_capabilities.candidate_runtimes
        )
    if not targets:
        targets.extend(_TARGET_CAPABILITIES)
    return _dedupe_targets(targets)


def _candidate_map(
    runtime_capabilities: RuntimeCapabilityProfile | None,
) -> dict[RuntimeCapabilityId, RuntimeCapabilityCandidate]:
    if runtime_capabilities is None:
        return {}
    return {
        candidate.runtime: candidate
        for candidate in runtime_capabilities.candidate_runtimes
    }


def _compatibility_map(
    runtime_compatibility: RuntimeCompatibilityProfile | None,
) -> dict[RuntimeCapabilityId, RuntimeCompatibilityAssessment]:
    if runtime_compatibility is None:
        return {}
    return {
        assessment.runtime: assessment
        for assessment in runtime_compatibility.compatibility_assessments
    }


def _capability_profile(
    *,
    target: RuntimeCapabilityId,
    candidate: RuntimeCapabilityCandidate | None,
    compatibility: RuntimeCompatibilityAssessment | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
) -> ArtifactCapabilityProfile:
    metadata = _TARGET_CAPABILITIES[target]
    artifact_fit = _artifact_fit(target, compatibility, artifact_plan)
    creative_fit = _creative_fit(candidate, creative_strategy)
    generative_fit = _generative_fit(metadata, target, artifact_plan, creative_plan)
    interaction_fit = metadata.interaction_fit
    audiovisual_fit = _audiovisual_fit(metadata, artifact_plan)
    export_fit = _export_fit(metadata, artifact_plan)
    interoperability_fit = _compatibility_fit(
        compatibility.interoperability if compatibility is not None else None,
        metadata.interoperability_fit,
    )
    portability_fit = _compatibility_fit(
        compatibility.portability if compatibility is not None else None,
        metadata.portability_fit,
    )
    confidence = _capability_confidence(
        compatibility=compatibility,
        candidate=candidate,
        fits=(
            artifact_fit,
            creative_fit,
            generative_fit,
            interaction_fit,
            audiovisual_fit,
            export_fit,
            interoperability_fit,
            portability_fit,
        ),
    )
    return ArtifactCapabilityProfile(
        target=target,
        label=metadata.label,
        capability_confidence=confidence,
        capability_reasons=_capability_reasons(
            target=target,
            artifact_fit=artifact_fit,
            compatibility=compatibility,
            candidate=candidate,
            artifact_plan=artifact_plan,
            artifact_dependency_graph=artifact_dependency_graph,
        ),
        strengths=_strengths(metadata, candidate, compatibility),
        weaknesses=_weaknesses(metadata, candidate, compatibility),
        unsupported_capabilities=_unsupported_capabilities(
            metadata=metadata,
            artifact_fit=artifact_fit,
            artifact_plan=artifact_plan,
            compatibility=compatibility,
        ),
        risky_capabilities=_risky_capabilities(
            metadata=metadata,
            candidate=candidate,
            compatibility=compatibility,
            creative_constraints=creative_constraints,
            creative_techniques=creative_techniques,
        ),
        artifact_fit=artifact_fit,
        creative_fit=creative_fit,
        generative_fit=generative_fit,
        interaction_fit=interaction_fit,
        audiovisual_fit=audiovisual_fit,
        export_fit=export_fit,
        interoperability_fit=interoperability_fit,
        portability_fit=portability_fit,
        capability_risks=_capability_profile_risks(
            metadata=metadata,
            candidate=candidate,
            compatibility=compatibility,
        ),
        prompt_guidance=_capability_profile_guidance(
            metadata=metadata,
            artifact_fit=artifact_fit,
        ),
        evidence=_capability_profile_evidence(
            target=target,
            artifact_plan=artifact_plan,
            compatibility=compatibility,
            candidate=candidate,
        ),
    )


def _artifact_fit(
    target: RuntimeCapabilityId,
    compatibility: RuntimeCompatibilityAssessment | None,
    artifact_plan: ArtifactPlan | None,
) -> ArtifactCapabilityFit:
    if compatibility is not None:
        return _compatibility_level_to_fit(compatibility.compatibility)
    if artifact_plan is None:
        return "moderate" if target in {"p5_js", "canvas"} else "weak"
    primary, secondary = _FAMILY_TARGETS[artifact_plan.artifact_family]
    if target in primary:
        return "strong"
    if target in secondary:
        return "moderate"
    return "unsupported"


def _creative_fit(
    candidate: RuntimeCapabilityCandidate | None,
    creative_strategy: CreativeStrategyProfile | None,
) -> ArtifactCapabilityFit:
    if candidate is not None:
        return _runtime_fit_to_capability_fit(candidate.strategy_alignment)
    if creative_strategy is None:
        return "moderate"
    return "moderate" if creative_strategy.confidence >= 0.55 else "weak"


def _generative_fit(
    metadata: _TargetCapabilityMetadata,
    target: RuntimeCapabilityId,
    artifact_plan: ArtifactPlan | None,
    creative_plan: CreativeExecutionPlan | None,
) -> ArtifactCapabilityFit:
    if artifact_plan is not None:
        primary, secondary = _FAMILY_TARGETS[artifact_plan.artifact_family]
        if target in primary and metadata.generative_fit != "unsupported":
            return "strong"
        if target in secondary and metadata.generative_fit == "strong":
            return "moderate"
    if creative_plan is not None and creative_plan.expected_complexity == "high":
        return _lower_fit(metadata.generative_fit)
    return metadata.generative_fit


def _audiovisual_fit(
    metadata: _TargetCapabilityMetadata,
    artifact_plan: ArtifactPlan | None,
) -> ArtifactCapabilityFit:
    if artifact_plan is not None and artifact_plan.artifact_family in {
        "audiovisual_scene",
        "tone_sketch",
        "hydra_patch",
    }:
        return metadata.audiovisual_fit
    if metadata.audiovisual_fit == "strong":
        return "moderate"
    return metadata.audiovisual_fit


def _export_fit(
    metadata: _TargetCapabilityMetadata,
    artifact_plan: ArtifactPlan | None,
) -> ArtifactCapabilityFit:
    if artifact_plan is not None and artifact_plan.artifact_type == "design_spec":
        return "strong" if metadata.target == "svg" else metadata.export_fit
    return metadata.export_fit


def _compatibility_fit(
    compatibility_value: str | None,
    fallback: ArtifactCapabilityFit,
) -> ArtifactCapabilityFit:
    if compatibility_value == "high":
        return "strong"
    if compatibility_value == "medium":
        return "moderate"
    if compatibility_value == "low":
        return "weak"
    return fallback


def _capability_confidence(
    *,
    compatibility: RuntimeCompatibilityAssessment | None,
    candidate: RuntimeCapabilityCandidate | None,
    fits: tuple[ArtifactCapabilityFit, ...],
) -> float:
    base = 0.45
    if compatibility is not None:
        base = compatibility.confidence
    elif candidate is not None:
        base = candidate.confidence
    fit_score = sum(_FIT_SCORE[item] for item in fits) / (len(fits) * 3)
    confidence = (base * 0.68) + (fit_score * 0.32)
    if "unsupported" in fits:
        confidence -= 0.08
    return max(0.1, min(0.98, round(confidence, 2)))


def _capability_reasons(
    *,
    target: RuntimeCapabilityId,
    artifact_fit: ArtifactCapabilityFit,
    compatibility: RuntimeCompatibilityAssessment | None,
    candidate: RuntimeCapabilityCandidate | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
) -> tuple[str, ...]:
    reasons = [
        (
            f"{_TARGET_CAPABILITIES[target].label} artifact capability fit "
            f"is {artifact_fit}."
        )
    ]
    if artifact_plan is not None:
        reasons.append(f"Planned artifact family: {artifact_plan.artifact_family}.")
    if compatibility is not None:
        reasons.append(
            "Runtime Compatibility Engine assessed target as "
            f"{compatibility.compatibility}."
        )
        reasons.extend(compatibility.compatibility_reasons[:2])
    if candidate is not None:
        reasons.append(
            f"Runtime Capability Reasoner suitability is {candidate.suitability}."
        )
    if artifact_dependency_graph is not None:
        reasons.extend(artifact_dependency_graph.runtime_facing_dependencies[:2])
    return _dedupe(reasons)[:8]


def _strengths(
    metadata: _TargetCapabilityMetadata,
    candidate: RuntimeCapabilityCandidate | None,
    compatibility: RuntimeCompatibilityAssessment | None,
) -> tuple[str, ...]:
    strengths = list(metadata.strengths)
    if candidate is not None:
        strengths.extend(candidate.strengths[:2])
    if compatibility is not None and compatibility.compatibility == "compatible":
        strengths.extend(compatibility.compatibility_reasons[:1])
    return _dedupe(strengths)[:8]


def _weaknesses(
    metadata: _TargetCapabilityMetadata,
    candidate: RuntimeCapabilityCandidate | None,
    compatibility: RuntimeCompatibilityAssessment | None,
) -> tuple[str, ...]:
    weaknesses = list(metadata.weaknesses)
    if candidate is not None:
        weaknesses.extend(candidate.limitations[:2])
    if compatibility is not None:
        weaknesses.extend(compatibility.runtime_limitations[:2])
    return _dedupe(weaknesses)[:8]


def _unsupported_capabilities(
    *,
    metadata: _TargetCapabilityMetadata,
    artifact_fit: ArtifactCapabilityFit,
    artifact_plan: ArtifactPlan | None,
    compatibility: RuntimeCompatibilityAssessment | None,
) -> tuple[str, ...]:
    unsupported = list(metadata.unsupported)
    if artifact_fit == "unsupported" and artifact_plan is not None:
        unsupported.append(
            f"{metadata.label} is unsuitable for {artifact_plan.artifact_family}."
        )
    if compatibility is not None and compatibility.compatibility == "unsupported":
        unsupported.append("Runtime compatibility marked this target unsupported.")
    return _dedupe(unsupported)[:8]


def _risky_capabilities(
    *,
    metadata: _TargetCapabilityMetadata,
    candidate: RuntimeCapabilityCandidate | None,
    compatibility: RuntimeCompatibilityAssessment | None,
    creative_constraints: CreativeConstraintSolution | None,
    creative_techniques: CreativeTechniqueProfile | None,
) -> tuple[str, ...]:
    risks = list(metadata.risks)
    if candidate is not None:
        risks.extend(candidate.risks[:2])
    if compatibility is not None:
        risks.extend(compatibility.implementation_risks[:2])
    if (
        creative_constraints is not None
        and creative_constraints.performance_pressure == "high"
    ):
        risks.append("High performance pressure makes target capabilities riskier.")
    if (
        creative_techniques is not None
        and creative_techniques.performance_pressure == "high"
    ):
        risks.append("Selected technique has high performance pressure.")
    return _dedupe(risks)[:8]


def _capability_profile_risks(
    *,
    metadata: _TargetCapabilityMetadata,
    candidate: RuntimeCapabilityCandidate | None,
    compatibility: RuntimeCompatibilityAssessment | None,
) -> tuple[str, ...]:
    risks = list(metadata.risks)
    if candidate is not None:
        risks.extend(candidate.risks[:2])
    if compatibility is not None:
        risks.extend(compatibility.implementation_risks[:2])
    risks.append("Do not use capability metadata to auto-select targets.")
    return _dedupe(risks)[:8]


def _capability_profile_guidance(
    *,
    metadata: _TargetCapabilityMetadata,
    artifact_fit: ArtifactCapabilityFit,
) -> tuple[str, ...]:
    guidance = [
        f"Use {metadata.label} capability notes as planning metadata only.",
        "Do not change runtime execution, provider routing, or preview behavior.",
    ]
    if artifact_fit == "unsupported":
        guidance.append("State unsupported target capability instead of using it.")
    elif artifact_fit == "weak":
        guidance.append("Treat weak capability as a caveat, not a blocker.")
    else:
        guidance.append("Use fit dimensions to explain implementation trade-offs.")
    return tuple(guidance[:8])


def _capability_profile_evidence(
    *,
    target: RuntimeCapabilityId,
    artifact_plan: ArtifactPlan | None,
    compatibility: RuntimeCompatibilityAssessment | None,
    candidate: RuntimeCapabilityCandidate | None,
) -> tuple[str, ...]:
    evidence = [f"Target evaluated: {target}."]
    if artifact_plan is not None:
        evidence.append(
            "Artifact context: "
            f"{artifact_plan.artifact_type}; {artifact_plan.artifact_family}."
        )
    if compatibility is not None:
        evidence.append(
            "Compatibility source: "
            f"{compatibility.compatibility} at {compatibility.confidence:.2f}."
        )
    if candidate is not None:
        evidence.append(
            "Runtime candidate source: "
            f"{candidate.suitability} at {candidate.confidence:.2f}."
        )
    return _dedupe(evidence)[:10]


def _missing_capability_information(
    *,
    route_decision: RouteDecision | None,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    creative_plan: CreativeExecutionPlan | None,
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
    if runtime_capabilities is None:
        missing.append("Runtime Capability Reasoner metadata is unavailable.")
    if runtime_compatibility is None:
        missing.append("Runtime Compatibility Engine metadata is unavailable.")
    if creative_plan is None or creative_plan.recommended_runtime is None:
        missing.append("Creative plan does not provide an explicit runtime hint.")
    return _dedupe(missing)[:10]


def _capability_risks(
    *,
    artifact_plan: ArtifactPlan | None,
    artifact_dependency_graph: ArtifactDependencyGraph | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
    profiles: tuple[ArtifactCapabilityProfile, ...],
) -> tuple[str, ...]:
    risks: list[str] = []
    if artifact_plan is not None:
        risks.extend(artifact_plan.implementation_risks[:3])
    if artifact_dependency_graph is not None:
        risks.extend(artifact_dependency_graph.dependency_conflicts[:3])
        risks.extend(artifact_dependency_graph.missing_dependency_risks[:2])
    if runtime_compatibility is not None:
        risks.extend(runtime_compatibility.implementation_risks[:3])
    if creative_tradeoffs is not None:
        risks.extend(creative_tradeoffs.runtime_risks[:2])
        risks.extend(creative_tradeoffs.complexity_risks[:2])
    for profile in profiles[:3]:
        risks.extend(
            f"{profile.label}: {item}" for item in profile.capability_risks[:2]
        )
    return _dedupe(risks)[:10]


def _hitl_questions(
    *,
    missing: tuple[str, ...],
    risks: tuple[str, ...],
    profiles: tuple[ArtifactCapabilityProfile, ...],
) -> tuple[str, ...]:
    questions = [
        f"Should we resolve this missing artifact capability input: {item}"
        for item in missing[:3]
    ]
    questions.extend(
        f"Should this target capability risk constrain generation: {item}"
        for item in risks[:3]
    )
    weak_targets = tuple(
        profile.label
        for profile in profiles
        if profile.artifact_fit in {"weak", "unsupported"}
    )[:3]
    if weak_targets:
        questions.append(
            "Should weak or unsupported targets be explicitly de-emphasized: "
            + ", ".join(weak_targets)
            + "?"
        )
    return tuple(questions[:8])


def _profile_prompt_guidance(
    strongest_targets: tuple[RuntimeCapabilityId, ...],
    missing: tuple[str, ...],
) -> tuple[str, ...]:
    guidance = [
        "Use Artifact Capability Matrix output as target capability metadata only.",
        (
            "Do not use capability metadata to auto-select runtimes, route "
            "providers or models, choose renderers, execute code, repair "
            "runtimes, or change preview behavior."
        ),
        "Treat strongest targets as non-binding capability context: "
        + _target_list(strongest_targets),
        (
            "When a capability is weak, risky, or unsupported, state the caveat "
            "instead of implying implementation support."
        ),
    ]
    if missing:
        guidance.append(
            "Surface missing capability information before expanding scope."
        )
    return tuple(guidance[:8])


def _matrix_evidence(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    artifact_plan: ArtifactPlan | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    runtime_compatibility: RuntimeCompatibilityProfile | None,
    profiles: tuple[ArtifactCapabilityProfile, ...],
) -> tuple[str, ...]:
    evidence = [f"Mode: {request.mode.value}."]
    if route_decision is not None:
        evidence.append(f"Route selected: {route_decision.route.value}.")
    if artifact_plan is not None:
        evidence.append(
            f"Artifact: {artifact_plan.artifact_type}; {artifact_plan.artifact_family}."
        )
    if runtime_capabilities is not None:
        evidence.append(
            "Runtime Capability Reasoner candidates: "
            + ", ".join(runtime_capabilities.likely_candidates)
            + "."
        )
    if runtime_compatibility is not None:
        evidence.append(
            "Runtime Compatibility preferred: "
            + ", ".join(runtime_compatibility.preferred_runtimes)
            + "."
        )
    evidence.append(
        "Capability order: "
        + ", ".join(
            f"{profile.target}:{profile.artifact_fit}:"
            f"{profile.capability_confidence:.2f}"
            for profile in profiles[:4]
        )
        + "."
    )
    return _dedupe(evidence)[:12]


def _target_strengths(
    profiles: tuple[ArtifactCapabilityProfile, ...],
) -> tuple[str, ...]:
    strengths: list[str] = []
    for profile in profiles[:4]:
        strengths.extend(f"{profile.label}: {item}" for item in profile.strengths[:2])
    return _dedupe(strengths)[:10]


def _target_weaknesses(
    profiles: tuple[ArtifactCapabilityProfile, ...],
) -> tuple[str, ...]:
    weaknesses: list[str] = []
    for profile in profiles[:4]:
        weaknesses.extend(f"{profile.label}: {item}" for item in profile.weaknesses[:2])
    return _dedupe(weaknesses)[:10]


def _unsupported_or_risky_capabilities(
    profiles: tuple[ArtifactCapabilityProfile, ...],
) -> tuple[str, ...]:
    capabilities: list[str] = []
    for profile in profiles:
        capabilities.extend(
            f"{profile.label}: {item}" for item in profile.unsupported_capabilities[:2]
        )
        capabilities.extend(
            f"{profile.label}: {item}" for item in profile.risky_capabilities[:2]
        )
    return _dedupe(capabilities)[:12]


def _compatibility_level_to_fit(
    compatibility: RuntimeCompatibilityLevel,
) -> ArtifactCapabilityFit:
    if compatibility == "compatible":
        return "strong"
    if compatibility == "partially_compatible":
        return "moderate"
    return "unsupported"


def _runtime_fit_to_capability_fit(value: str) -> ArtifactCapabilityFit:
    if value == "strong":
        return "strong"
    if value == "moderate":
        return "moderate"
    return "weak"


def _lower_fit(value: ArtifactCapabilityFit) -> ArtifactCapabilityFit:
    if value == "strong":
        return "moderate"
    if value == "moderate":
        return "weak"
    return value


def _profile_score(profile: ArtifactCapabilityProfile) -> int:
    return (
        _FIT_SCORE[profile.artifact_fit] * 3
        + _FIT_SCORE[profile.creative_fit]
        + _FIT_SCORE[profile.generative_fit]
        + _FIT_SCORE[profile.interaction_fit]
        + _FIT_SCORE[profile.audiovisual_fit]
        + _FIT_SCORE[profile.export_fit]
        + _FIT_SCORE[profile.interoperability_fit]
        + _FIT_SCORE[profile.portability_fit]
    )


def _target_list(targets: tuple[RuntimeCapabilityId, ...]) -> str:
    if not targets:
        return "none"
    return ", ".join(_TARGET_CAPABILITIES[target].label for target in targets) + "."


def _dedupe_targets(
    values: list[RuntimeCapabilityId],
) -> tuple[RuntimeCapabilityId, ...]:
    deduped: list[RuntimeCapabilityId] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped[:9])
