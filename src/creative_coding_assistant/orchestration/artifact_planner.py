"""Bounded Artifact Planner for V3.3 workflows."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration._metadata_utils import _clip, _dedupe
from creative_coding_assistant.orchestration.audio_visual_scene import (
    AudioVisualSceneProfile,
)
from creative_coding_assistant.orchestration.creative_composition import (
    CreativeCompositionPlan,
)
from creative_coding_assistant.orchestration.creative_constraint_priorities import (
    CreativeConstraintPrioritization,
)
from creative_coding_assistant.orchestration.creative_constraints import (
    CreativeConstraintSolution,
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
from creative_coding_assistant.orchestration.creative_quality_prediction import (
    CreativeQualityPrediction,
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
from creative_coding_assistant.orchestration.cross_modality import (
    CrossModalityCompositionProfile,
)
from creative_coding_assistant.orchestration.emotional_consistency import (
    EmotionalConsistencyProfile,
)
from creative_coding_assistant.orchestration.generative_structure import (
    GenerativeStructureBlueprint,
)
from creative_coding_assistant.orchestration.procedural_structure import (
    ProceduralStructurePlan,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityProfile,
)
from creative_coding_assistant.orchestration.semantic_motif import SemanticMotifSystem
from creative_coding_assistant.orchestration.symbolic_narrative import (
    SymbolicNarrativePlan,
)

ArtifactType = Literal[
    "runnable_code",
    "design_spec",
    "explanation",
    "debug_patch",
    "review_report",
    "refinement_patch",
    "preview_request",
]
ArtifactFamily = Literal[
    "p5_sketch",
    "three_scene",
    "react_three_fiber_scene",
    "glsl_shader",
    "hydra_patch",
    "tone_sketch",
    "canvas_sketch",
    "audiovisual_scene",
    "generative_artifact",
    "multimodal_reference_artifact",
    "creative_coding_response",
]

ARTIFACT_PLANNER_AUTHORITY_BOUNDARY = (
    "The Artifact Planner structures intended artifact shape, dependencies, "
    "requirements, risks, missing information, HITL questions, and prompt "
    "guidance as inspectable metadata only; it does not select generated "
    "artifacts, critique artifacts, refine artifacts, execute runtimes, route "
    "providers or models, change preview behavior, repair runtimes, implement "
    "V4 multi-agent behavior, implement Studio Mode, or implement HoloMind."
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9_.+#-]+")


class ArtifactPlan(BaseModel):
    """Inspectable artifact-shape metadata derived before generation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["artifact_planner"] = "artifact_planner"
    primary_artifact_intent: str = Field(min_length=1, max_length=420)
    artifact_type: ArtifactType
    artifact_family: ArtifactFamily
    required_components: tuple[str, ...] = Field(min_length=1, max_length=10)
    runtime_requirements: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    creative_dependencies: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    generative_dependencies: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=10,
    )
    expected_output_structure: tuple[str, ...] = Field(min_length=1, max_length=8)
    implementation_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    missing_information: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=ARTIFACT_PLANNER_AUTHORITY_BOUNDARY,
        max_length=760,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=12)


def derive_artifact_plan(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None = None,
    creative_intent: CreativeIntentDecomposition | None = None,
    creative_hierarchy: CreativeHierarchyPlan | None = None,
    creative_plan: CreativeExecutionPlan | None = None,
    creative_constraints: CreativeConstraintSolution | None = None,
    creative_constraint_priorities: CreativeConstraintPrioritization | None = None,
    creative_strategy: CreativeStrategyProfile | None = None,
    creative_techniques: CreativeTechniqueProfile | None = None,
    runtime_capabilities: RuntimeCapabilityProfile | None = None,
    creative_tradeoffs: CreativeTradeoffProfile | None = None,
    creative_quality_prediction: CreativeQualityPrediction | None = None,
    symbolic_narrative: SymbolicNarrativePlan | None = None,
    creative_composition: CreativeCompositionPlan | None = None,
    procedural_structure: ProceduralStructurePlan | None = None,
    generative_structure: GenerativeStructureBlueprint | None = None,
    semantic_motif: SemanticMotifSystem | None = None,
    emotional_consistency: EmotionalConsistencyProfile | None = None,
    cross_modality: CrossModalityCompositionProfile | None = None,
    audio_visual_scene: AudioVisualSceneProfile | None = None,
) -> ArtifactPlan:
    """Derive artifact planning metadata without changing runtime behavior."""

    artifact_type = _artifact_type(request)
    family = _artifact_family(
        request=request,
        route_decision=route_decision,
        creative_translation=creative_translation,
        creative_plan=creative_plan,
        runtime_capabilities=runtime_capabilities,
        audio_visual_scene=audio_visual_scene,
    )
    missing = _missing_information(
        request=request,
        route_decision=route_decision,
        creative_intent=creative_intent,
        creative_hierarchy=creative_hierarchy,
        creative_plan=creative_plan,
        creative_quality_prediction=creative_quality_prediction,
        symbolic_narrative=symbolic_narrative,
        creative_composition=creative_composition,
        procedural_structure=procedural_structure,
        generative_structure=generative_structure,
        semantic_motif=semantic_motif,
        emotional_consistency=emotional_consistency,
        cross_modality=cross_modality,
        audio_visual_scene=audio_visual_scene,
    )
    risks = _implementation_risks(
        creative_plan=creative_plan,
        creative_constraints=creative_constraints,
        creative_tradeoffs=creative_tradeoffs,
        creative_quality_prediction=creative_quality_prediction,
        procedural_structure=procedural_structure,
        generative_structure=generative_structure,
        semantic_motif=semantic_motif,
        emotional_consistency=emotional_consistency,
        cross_modality=cross_modality,
        audio_visual_scene=audio_visual_scene,
    )
    return ArtifactPlan(
        primary_artifact_intent=_primary_artifact_intent(
            request=request,
            creative_translation=creative_translation,
            creative_intent=creative_intent,
            creative_plan=creative_plan,
        ),
        artifact_type=artifact_type,
        artifact_family=family,
        required_components=_required_components(
            artifact_type=artifact_type,
            family=family,
            creative_plan=creative_plan,
            creative_composition=creative_composition,
            procedural_structure=procedural_structure,
            generative_structure=generative_structure,
            semantic_motif=semantic_motif,
            audio_visual_scene=audio_visual_scene,
        ),
        runtime_requirements=_runtime_requirements(
            family=family,
            creative_plan=creative_plan,
            runtime_capabilities=runtime_capabilities,
            creative_constraints=creative_constraints,
        ),
        creative_dependencies=_creative_dependencies(
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_strategy=creative_strategy,
            creative_techniques=creative_techniques,
            creative_constraints=creative_constraints,
            creative_constraint_priorities=creative_constraint_priorities,
            creative_quality_prediction=creative_quality_prediction,
            symbolic_narrative=symbolic_narrative,
            creative_composition=creative_composition,
        ),
        generative_dependencies=_generative_dependencies(
            procedural_structure=procedural_structure,
            generative_structure=generative_structure,
            semantic_motif=semantic_motif,
            emotional_consistency=emotional_consistency,
            cross_modality=cross_modality,
            audio_visual_scene=audio_visual_scene,
        ),
        expected_output_structure=_expected_output_structure(
            artifact_type=artifact_type,
            family=family,
            request=request,
            creative_plan=creative_plan,
        ),
        implementation_risks=risks,
        missing_information=missing,
        hitl_questions=_hitl_questions(missing, risks),
        prompt_guidance=_prompt_guidance(
            artifact_type=artifact_type,
            family=family,
            creative_plan=creative_plan,
        ),
        evidence=_evidence(
            request=request,
            route_decision=route_decision,
            creative_translation=creative_translation,
            creative_plan=creative_plan,
            runtime_capabilities=runtime_capabilities,
            family=family,
        ),
    )


def artifact_plan_prompt_lines(plan: ArtifactPlan) -> tuple[str, ...]:
    """Render artifact planning metadata as compact prompt guidance."""

    lines = [
        f"Authority boundary: {plan.authority_boundary}",
        f"Primary artifact intent: {plan.primary_artifact_intent}",
        f"Artifact type: {plan.artifact_type}.",
        f"Artifact family: {plan.artifact_family}.",
    ]
    lines.extend(
        f"Required artifact component: {item}"
        for item in plan.required_components
    )
    lines.extend(
        f"Runtime-facing artifact requirement: {item}"
        for item in plan.runtime_requirements
    )
    lines.extend(
        f"Creative artifact dependency: {item}"
        for item in plan.creative_dependencies
    )
    lines.extend(
        f"Generative artifact dependency: {item}"
        for item in plan.generative_dependencies
    )
    lines.extend(
        f"Expected artifact output structure: {item}"
        for item in plan.expected_output_structure
    )
    lines.extend(
        f"Artifact implementation risk: {item}"
        for item in plan.implementation_risks
    )
    lines.extend(
        f"Missing artifact information: {item}"
        for item in plan.missing_information
    )
    lines.extend(f"HITL artifact question: {item}" for item in plan.hitl_questions)
    lines.extend(f"Artifact planner guidance: {item}" for item in plan.prompt_guidance)
    return tuple(lines[:48])


def _artifact_type(request: AssistantRequest) -> ArtifactType:
    if request.artifact_refinement is not None:
        return "refinement_patch"
    if request.mode is AssistantMode.DESIGN:
        return "design_spec"
    if request.mode is AssistantMode.EXPLAIN:
        return "explanation"
    if request.mode is AssistantMode.DEBUG:
        return "debug_patch"
    if request.mode is AssistantMode.REVIEW:
        return "review_report"
    if request.mode is AssistantMode.PREVIEW:
        return "preview_request"
    return "runnable_code"


def _artifact_family(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
    creative_plan: CreativeExecutionPlan | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    audio_visual_scene: AudioVisualSceneProfile | None,
) -> ArtifactFamily:
    modality = (
        creative_translation.output_modality.value
        if creative_translation is not None
        and creative_translation.output_modality is not None
        else creative_plan.output_modality.value
        if creative_plan is not None
        else None
    )
    if modality == "audiovisual":
        return "audiovisual_scene"
    domains = _effective_domains(request, route_decision)
    if CreativeCodingDomain.P5_JS in domains:
        return "p5_sketch"
    if CreativeCodingDomain.THREE_JS in domains:
        return "three_scene"
    if CreativeCodingDomain.REACT_THREE_FIBER in domains:
        return "react_three_fiber_scene"
    if (
        CreativeCodingDomain.GLSL in domains
        or CreativeCodingDomain.SHADERTOY in domains
    ):
        return "glsl_shader"
    if CreativeCodingDomain.HYDRA in domains:
        return "hydra_patch"
    if (
        CreativeCodingDomain.TONE_JS in domains
        or CreativeCodingDomain.WEB_AUDIO_API in domains
        or CreativeCodingDomain.P5_SOUND in domains
    ):
        return "tone_sketch"
    if CreativeCodingDomain.CANVAS_2D in domains:
        return "canvas_sketch"
    top_runtime = (
        runtime_capabilities.likely_candidates[0]
        if runtime_capabilities is not None and runtime_capabilities.likely_candidates
        else None
    )
    if top_runtime == "p5_js":
        return "p5_sketch"
    if top_runtime in {"three_js", "react_three_fiber"}:
        return "three_scene"
    if top_runtime == "glsl":
        return "glsl_shader"
    if top_runtime == "hydra":
        return "hydra_patch"
    if top_runtime == "tone_js":
        return "tone_sketch"
    if top_runtime == "canvas":
        return "canvas_sketch"
    if creative_plan is not None and creative_plan.recommended_runtime == "p5":
        return "p5_sketch"
    if _tokens(request.query) & {"generative", "procedural", "system"}:
        return "generative_artifact"
    if request.attachments:
        return "multimodal_reference_artifact"
    return "creative_coding_response"


def _primary_artifact_intent(
    *,
    request: AssistantRequest,
    creative_translation: CreativeTranslation | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_plan: CreativeExecutionPlan | None,
) -> str:
    if request.artifact_refinement is not None:
        return _clip(
            (
                f"Refine '{request.artifact_refinement.title}' according to: "
                f"{request.artifact_refinement.instruction}"
            ),
            420,
        )
    if creative_intent is not None:
        return creative_intent.primary_expression
    if creative_translation is not None:
        return creative_translation.creative_intent
    if creative_plan is not None:
        return creative_plan.generation_strategy
    return _clip(request.query, 420)


def _required_components(
    *,
    artifact_type: ArtifactType,
    family: ArtifactFamily,
    creative_plan: CreativeExecutionPlan | None,
    creative_composition: CreativeCompositionPlan | None,
    procedural_structure: ProceduralStructurePlan | None,
    generative_structure: GenerativeStructureBlueprint | None,
    semantic_motif: SemanticMotifSystem | None,
    audio_visual_scene: AudioVisualSceneProfile | None,
) -> tuple[str, ...]:
    components = ["One clearly labeled primary artifact."]
    if artifact_type in {"runnable_code", "refinement_patch", "debug_patch"}:
        components.append("A fenced code block with an explicit language tag.")
    components.extend(_family_components(family))
    if creative_plan is not None and creative_plan.recommended_runtime is not None:
        components.append(
            f"Implementation compatible with {creative_plan.recommended_runtime}."
        )
    if creative_composition is not None:
        components.append(
            f"Visible composition spine: {creative_composition.composition_pattern}."
        )
    if procedural_structure is not None:
        family_name = procedural_structure.primary_structure.family
        components.append(
            f"Primary procedural family: {family_name}."
        )
    if generative_structure is not None:
        components.append(
            f"Named generative module set: {generative_structure.blueprint_name}."
        )
    if semantic_motif is not None:
        components.append(
            "Primary motif set: "
            + ", ".join(motif.motif_id for motif in semantic_motif.primary_motifs)
            + "."
        )
    if audio_visual_scene is not None:
        components.append(
            f"Scene arc with {len(audio_visual_scene.scene_phases)} planned phases."
        )
    return _dedupe(components)[:10]


def _family_components(family: ArtifactFamily) -> tuple[str, ...]:
    if family == "p5_sketch":
        return ("p5.js setup/draw lifecycle.", "Canvas-safe animation loop.")
    if family == "three_scene":
        return ("Scene, camera, renderer, and animation lifecycle.",)
    if family == "react_three_fiber_scene":
        return ("React component scene structure with bounded hooks.",)
    if family == "glsl_shader":
        return ("Shader entrypoint, uniforms, and visual output mapping.",)
    if family == "hydra_patch":
        return ("Hydra signal chain with readable modulation stages.",)
    if family == "tone_sketch":
        return ("Audio graph, transport timing, and safe start controls.",)
    if family == "canvas_sketch":
        return ("Canvas setup, draw loop, and resize-safe dimensions.",)
    if family == "audiovisual_scene":
        return ("Visual scene plus bounded audio/rhythm guidance.",)
    if family == "generative_artifact":
        return ("Seeded structure, parameters, and evolution rules.",)
    if family == "multimodal_reference_artifact":
        return (
            "Reference-aware visual translation without embedding source payloads.",
        )
    return ("Concise response structure aligned to the selected route.",)


def _runtime_requirements(
    *,
    family: ArtifactFamily,
    creative_plan: CreativeExecutionPlan | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    creative_constraints: CreativeConstraintSolution | None,
) -> tuple[str, ...]:
    requirements: list[str] = []
    if creative_plan is not None:
        if creative_plan.recommended_runtime is not None:
            requirements.append(
                f"Respect existing runtime hint: {creative_plan.recommended_runtime}."
            )
        if creative_plan.recommended_renderer_id is not None:
            renderer_id = creative_plan.recommended_renderer_id
            requirements.append(
                f"Keep renderer compatibility with {renderer_id}."
            )
        if creative_plan.recommended_preview_target is not None:
            preview_target = creative_plan.recommended_preview_target
            requirements.append(
                f"Keep preview target compatible with {preview_target}."
            )
        requirements.append(creative_plan.runtime_support_summary)
    if runtime_capabilities is not None:
        requirements.append(
            "Use inspected runtime candidates as non-binding feasibility notes: "
            + ", ".join(runtime_capabilities.likely_candidates)
            + "."
        )
        requirements.extend(runtime_capabilities.prompt_guidance[:2])
    if creative_constraints is not None:
        requirements.extend(creative_constraints.prompt_guidance[:2])
    if family == "audiovisual_scene":
        requirements.append(
            "Treat audio/rhythm plans as design guidance unless runtime "
            "explicitly supports audio."
        )
    return _dedupe(requirements)[:10]


def _creative_dependencies(
    *,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_constraints: CreativeConstraintSolution | None,
    creative_constraint_priorities: CreativeConstraintPrioritization | None,
    creative_quality_prediction: CreativeQualityPrediction | None,
    symbolic_narrative: SymbolicNarrativePlan | None,
    creative_composition: CreativeCompositionPlan | None,
) -> tuple[str, ...]:
    dependencies: list[str] = []
    if creative_intent is not None:
        dependencies.append(f"Intent: {creative_intent.primary_expression}.")
    if creative_hierarchy is not None:
        dependencies.append(
            "Hierarchy: "
            + ", ".join(
                item.dimension
                for item in creative_hierarchy.primary_creative_priorities[:3]
            )
            + "."
        )
    if creative_strategy is not None:
        dependencies.append(f"Strategy: {creative_strategy.primary_strategy}.")
    if creative_techniques is not None:
        dependencies.append(f"Technique: {creative_techniques.primary_technique}.")
    if creative_constraints is not None:
        complexity_pressure = creative_constraints.complexity_pressure
        dependencies.append(
            f"Constraint pressures: complexity {complexity_pressure}; "
            f"performance {creative_constraints.performance_pressure}."
        )
    if creative_constraint_priorities is not None:
        dependencies.append(
            "Protected constraints: "
            + ", ".join(
                item.category
                for item in (
                    creative_constraint_priorities.non_negotiable_constraints
                    or creative_constraint_priorities.high_priority_constraints
                )[:3]
            )
            + "."
        )
    if creative_quality_prediction is not None:
        dependencies.append(
            "Quality readiness: "
            f"{creative_quality_prediction.predicted_quality_level} "
            f"({creative_quality_prediction.readiness_score}/100)."
        )
    if symbolic_narrative is not None:
        dependencies.append(
            f"Narrative arc: {symbolic_narrative.narrative_archetype}."
        )
    if creative_composition is not None:
        dependencies.append(
            f"Composition: {creative_composition.composition_pattern}."
        )
    return _dedupe(dependencies)[:10]


def _generative_dependencies(
    *,
    procedural_structure: ProceduralStructurePlan | None,
    generative_structure: GenerativeStructureBlueprint | None,
    semantic_motif: SemanticMotifSystem | None,
    emotional_consistency: EmotionalConsistencyProfile | None,
    cross_modality: CrossModalityCompositionProfile | None,
    audio_visual_scene: AudioVisualSceneProfile | None,
) -> tuple[str, ...]:
    dependencies: list[str] = []
    if procedural_structure is not None:
        dependencies.append(
            f"Procedural structure: {procedural_structure.primary_structure.family}."
        )
    if generative_structure is not None:
        dependencies.append(
            f"Generative blueprint: {generative_structure.generative_architecture}."
        )
    if semantic_motif is not None:
        dependencies.append(
            "Semantic motifs: "
            + ", ".join(motif.motif_id for motif in semantic_motif.primary_motifs)
            + "."
        )
    if emotional_consistency is not None:
        dependencies.append(
            f"Emotion: {emotional_consistency.primary_emotional_tone}."
        )
    if cross_modality is not None:
        dependencies.append(
            f"Cross-modality: {cross_modality.modality_pattern}."
        )
    if audio_visual_scene is not None:
        dependencies.append(
            f"Audio-visual scene: {audio_visual_scene.scene_pattern}."
        )
    return _dedupe(dependencies)[:10]


def _expected_output_structure(
    *,
    artifact_type: ArtifactType,
    family: ArtifactFamily,
    request: AssistantRequest,
    creative_plan: CreativeExecutionPlan | None,
) -> tuple[str, ...]:
    if request.artifact_refinement is not None:
        return (
            "Return only the refined selected artifact unless the user asks "
            "for a new set.",
            "Keep the selected artifact language/runtime contract unless the "
            "instruction requires a compatible change.",
            "Place refined runnable code in one fenced code block.",
        )
    if artifact_type == "runnable_code":
        structure = [
            "Lead with the primary runnable artifact.",
            "Use a fenced code block with an explicit filename or language tag.",
            "Keep setup/run notes outside code fences and concise.",
        ]
        if creative_plan is not None and creative_plan.candidate_count > 1:
            candidate_count = creative_plan.candidate_count
            structure.append(
                f"Provide no more than {candidate_count} clearly labeled candidates."
            )
        if family == "audiovisual_scene":
            structure.append(
                "Separate audio/rhythm guidance from required visual code "
                "when audio runtime is uncertain."
            )
        return tuple(structure)
    if artifact_type == "design_spec":
        return (
            "Lead with a compact artifact specification.",
            "List implementation sections in dependency order.",
            "Keep open questions explicit.",
        )
    return (
        "Answer in the route-appropriate format.",
        "Keep artifact assumptions inspectable.",
        "Do not claim runtime output was executed or previewed.",
    )


def _implementation_risks(
    *,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
    creative_quality_prediction: CreativeQualityPrediction | None,
    procedural_structure: ProceduralStructurePlan | None,
    generative_structure: GenerativeStructureBlueprint | None,
    semantic_motif: SemanticMotifSystem | None,
    emotional_consistency: EmotionalConsistencyProfile | None,
    cross_modality: CrossModalityCompositionProfile | None,
    audio_visual_scene: AudioVisualSceneProfile | None,
) -> tuple[str, ...]:
    risks: list[str] = []
    if creative_plan is not None and creative_plan.expected_complexity == "high":
        risks.append("High expected complexity can obscure the primary artifact.")
    if creative_constraints is not None:
        risks.extend(creative_constraints.conflicts[:2])
        risks.extend(
            tradeoff.summary for tradeoff in creative_constraints.tradeoffs[:2]
        )
    if creative_tradeoffs is not None:
        risks.extend(creative_tradeoffs.runtime_risks[:2])
        risks.extend(creative_tradeoffs.performance_concerns[:2])
    if creative_quality_prediction is not None:
        risks.extend(creative_quality_prediction.quality_risks[:2])
        risks.extend(creative_quality_prediction.likely_failure_modes[:2])
    if procedural_structure is not None:
        risks.extend(procedural_structure.implementation_risks[:2])
        risks.extend(procedural_structure.performance_risks[:1])
    if generative_structure is not None:
        risks.extend(generative_structure.runtime_implementation_guidance[:2])
        risks.extend(generative_structure.performance_safeguards[:1])
    if semantic_motif is not None:
        risks.extend(semantic_motif.coherence_risks[:1])
        risks.extend(semantic_motif.overuse_risks[:1])
    if emotional_consistency is not None:
        risks.extend(emotional_consistency.mismatch_risks[:1])
        risks.extend(emotional_consistency.flattening_risks[:1])
    if cross_modality is not None:
        risks.extend(cross_modality.modality_conflicts[:1])
        risks.extend(cross_modality.overload_risks[:1])
    if audio_visual_scene is not None:
        risks.extend(audio_visual_scene.scene_risks[:1])
        risks.extend(audio_visual_scene.pacing_risks[:1])
        risks.extend(audio_visual_scene.overload_risks[:1])
    return _dedupe(risks)[:10]


def _missing_information(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_quality_prediction: CreativeQualityPrediction | None,
    symbolic_narrative: SymbolicNarrativePlan | None,
    creative_composition: CreativeCompositionPlan | None,
    procedural_structure: ProceduralStructurePlan | None,
    generative_structure: GenerativeStructureBlueprint | None,
    semantic_motif: SemanticMotifSystem | None,
    emotional_consistency: EmotionalConsistencyProfile | None,
    cross_modality: CrossModalityCompositionProfile | None,
    audio_visual_scene: AudioVisualSceneProfile | None,
) -> tuple[str, ...]:
    missing: list[str] = []
    if creative_intent is not None:
        missing.extend(creative_intent.unresolved_intent_gaps[:2])
    if creative_hierarchy is not None:
        missing.extend(creative_hierarchy.priority_conflicts[:2])
    if creative_quality_prediction is not None:
        missing.extend(creative_quality_prediction.missing_information[:3])
    if symbolic_narrative is not None:
        missing.extend(symbolic_narrative.unresolved_narrative_gaps[:2])
    if creative_composition is not None:
        missing.extend(creative_composition.unresolved_composition_gaps[:2])
    if procedural_structure is not None:
        missing.extend(procedural_structure.unresolved_procedural_gaps[:2])
    if generative_structure is not None:
        missing.extend(generative_structure.unresolved_implementation_gaps[:2])
    if semantic_motif is not None:
        missing.extend(semantic_motif.unresolved_motif_gaps[:2])
    if emotional_consistency is not None:
        missing.extend(emotional_consistency.unresolved_emotional_gaps[:2])
    if cross_modality is not None:
        missing.extend(cross_modality.unresolved_modality_gaps[:2])
    if audio_visual_scene is not None:
        missing.extend(audio_visual_scene.unresolved_scene_gaps[:2])
    if (
        creative_plan is None
        or creative_plan.recommended_runtime is None
        and not _effective_domains(request, route_decision)
    ):
        missing.append("Target runtime/domain is inferred rather than explicit.")
    return _dedupe(missing)[:10]


def _hitl_questions(
    missing_information: tuple[str, ...],
    implementation_risks: tuple[str, ...],
) -> tuple[str, ...]:
    questions = [
        f"Should we resolve this artifact planning gap before generation: {item}"
        for item in missing_information[:4]
    ]
    questions.extend(
        f"Should this artifact risk be constrained before generation: {item}"
        for item in implementation_risks[:2]
    )
    return tuple(questions[:8])


def _prompt_guidance(
    *,
    artifact_type: ArtifactType,
    family: ArtifactFamily,
    creative_plan: CreativeExecutionPlan | None,
) -> tuple[str, ...]:
    guidance = [
        "Use the Artifact Planner as artifact-shape guidance only.",
        "Satisfy required artifact components before adding secondary effects.",
        (
            "Keep artifact assumptions visible in code labels, comments, "
            "or concise notes."
        ),
        (
            "Do not treat the Artifact Planner as artifact selection, critique, "
            "runtime execution, provider routing, or preview behavior."
        ),
    ]
    if artifact_type in {"runnable_code", "refinement_patch", "debug_patch"}:
        guidance.append("Return runnable code in fenced blocks and keep prose outside.")
    if family == "audiovisual_scene":
        guidance.append(
            "Preserve audio-visual timing as design guidance unless audio "
            "runtime support is explicit."
        )
    if creative_plan is not None and creative_plan.export_readiness != "ready":
        guidance.append(
            "Mention export or runtime limitations instead of implying full "
            "readiness."
        )
    return tuple(guidance[:8])


def _evidence(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
    creative_plan: CreativeExecutionPlan | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    family: ArtifactFamily,
) -> tuple[str, ...]:
    evidence = [f"Mode: {request.mode.value}.", f"Artifact family: {family}."]
    if route_decision is not None:
        evidence.append(f"Route: {route_decision.route.value}.")
        if route_decision.domains:
            evidence.append(
                "Domains: "
                + ", ".join(domain.value for domain in route_decision.domains)
                + "."
            )
    if creative_translation is not None:
        evidence.append(f"Translated intent: {creative_translation.creative_intent}.")
    if creative_plan is not None:
        evidence.append(f"Plan runtime: {creative_plan.recommended_runtime or 'none'}.")
        evidence.append(f"Plan complexity: {creative_plan.expected_complexity}.")
    if runtime_capabilities is not None:
        evidence.append(
            "Inspected runtime candidates: "
            + ", ".join(runtime_capabilities.likely_candidates)
            + "."
        )
    return _dedupe(evidence)[:12]


def _effective_domains(
    request: AssistantRequest,
    route_decision: RouteDecision | None,
) -> tuple[CreativeCodingDomain, ...]:
    if route_decision is not None and route_decision.domains:
        return route_decision.domains
    if request.domains:
        return request.domains
    if request.domain is not None:
        return (request.domain,)
    return ()


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_PATTERN.findall(text.lower()))
