"""Bounded Audio-Visual Scene System for V3.2 workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
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
    CrossModalityChannel,
    CrossModalityCompositionProfile,
)
from creative_coding_assistant.orchestration.emotional_consistency import (
    EmotionalConsistencyProfile,
    EmotionalTone,
)
from creative_coding_assistant.orchestration.generative_structure import (
    GenerativeEvolutionPhase,
    GenerativeModuleKind,
    GenerativeStructureBlueprint,
)
from creative_coding_assistant.orchestration.procedural_structure import (
    ProceduralStructurePlan,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityProfile,
)
from creative_coding_assistant.orchestration.semantic_motif import (
    SemanticMotifId,
    SemanticMotifSystem,
)
from creative_coding_assistant.orchestration.symbolic_narrative import (
    NarrativePhaseName,
    SymbolicNarrativePlan,
)

AudioVisualScenePattern = Literal[
    "seed_to_expansion",
    "descent_to_return",
    "fragmentation_to_reintegration",
    "threshold_crossing",
    "spiral_ascent",
    "chaos_to_order",
    "void_to_emergence",
    "contraction_to_release",
    "ritual_opening_to_climax",
    "wave_build_and_collapse",
    "constellation_activation",
    "mirror_inversion",
    "pulse_escalation",
    "calm_expansion_after_rupture",
]
AudioVisualCueType = Literal[
    "visual",
    "motion",
    "audio",
    "rhythm",
    "camera",
    "motif",
    "emotion",
    "procedural",
    "synchronization",
]

AUDIO_VISUAL_SCENE_AUTHORITY_BOUNDARY = (
    "The Audio-Visual Scene System organizes scene phases, cues, transitions, "
    "climax, resolution, and timing guidance as inspectable design metadata "
    "only; it does not generate executable code, generate audio, render "
    "visuals, auto-select runtimes, route providers or models, change preview "
    "behavior, implement runtime repair, implement V4 multi-agent runtime, "
    "implement HoloMind, or implement autonomous loops."
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9_.+#-]+")
_AUDIO_TOKENS = frozenset(
    {
        "audio",
        "audiovisual",
        "beat",
        "bpm",
        "drone",
        "music",
        "pulse",
        "rhythm",
        "sound",
        "sonic",
        "tempo",
        "tone",
    }
)
_CAMERA_TOKENS = frozenset(
    {
        "3d",
        "camera",
        "cinematic",
        "depth",
        "orbit",
        "perspective",
        "scene",
        "shot",
        "viewpoint",
    }
)
_DENSE_TOKENS = frozenset(
    {
        "chaotic",
        "complex",
        "dense",
        "flashing",
        "intense",
        "loud",
        "many",
        "maximal",
        "overwhelming",
        "particles",
        "strobe",
        "swarm",
        "turbulent",
    }
)
_AMBIGUOUS_SCENE_TOKENS = frozenset(
    {
        "audiovisual",
        "cinematic",
        "maybe",
        "scene",
        "something",
        "story",
        "sync",
        "timing",
        "vibe",
    }
)
_FRAGMENTATION_TOKENS = frozenset(
    {"dissolve", "dissolves", "fragment", "fragments", "particles", "rupture"}
)
_REINTEGRATION_TOKENS = frozenset(
    {"phoenix", "reassemble", "reassembly", "reform", "reforms", "reintegration"}
)
_RITUAL_TOKENS = frozenset({"mandala", "ritual", "sacred", "solemn", "temple"})
_WAVE_TOKENS = frozenset({"collapse", "tide", "wave", "waves"})
_PULSE_TOKENS = frozenset({"beat", "bpm", "pulse", "tempo"})


class AudioVisualScenePhase(BaseModel):
    """One phase in the planned audio-visual scene arc."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    phase: NarrativePhaseName
    title: str = Field(min_length=1, max_length=140)
    scene_function: str = Field(min_length=1, max_length=360)
    visual_state: str = Field(min_length=1, max_length=360)
    motion_state: str = Field(min_length=1, max_length=340)
    audio_state: str | None = Field(default=None, max_length=320)
    rhythm_state: str = Field(min_length=1, max_length=320)
    camera_state: str | None = Field(default=None, max_length=300)
    motif_state: str = Field(min_length=1, max_length=320)
    emotional_state: str = Field(min_length=1, max_length=320)
    procedural_state: str = Field(min_length=1, max_length=340)
    cue_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    transition_out: str = Field(min_length=1, max_length=320)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=8)


class AudioVisualSceneCue(BaseModel):
    """One cue used to coordinate scene timing."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    cue_id: str = Field(min_length=1, max_length=80)
    phase: NarrativePhaseName
    cue_type: AudioVisualCueType
    description: str = Field(min_length=1, max_length=340)
    timing: str = Field(min_length=1, max_length=260)
    modalities: tuple[CrossModalityChannel, ...] = Field(min_length=1, max_length=6)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=6)


class AudioVisualSceneTransition(BaseModel):
    """Transition between two scene phases."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    from_phase: NarrativePhaseName
    to_phase: NarrativePhaseName
    transition: str = Field(min_length=1, max_length=340)
    visual_motion_guidance: str = Field(min_length=1, max_length=320)
    audio_rhythm_guidance: str | None = Field(default=None, max_length=300)
    continuity_guidance: str = Field(min_length=1, max_length=320)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=6)


class AudioVisualFallbackSceneStrategy(BaseModel):
    """Lower-risk scene structure when pacing or modality scope tightens."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    fallback_pattern: AudioVisualScenePattern
    preserved_phases: tuple[NarrativePhaseName, ...] = Field(
        min_length=3,
        max_length=5,
    )
    reduced_elements: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    simplification_strategy: str = Field(min_length=1, max_length=340)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=5)


class AudioVisualSceneProfile(BaseModel):
    """Inspectable audio-visual scene metadata derived before generation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["audio_visual_scene_system"] = "audio_visual_scene_system"
    scene_pattern: AudioVisualScenePattern
    scene_arc: str = Field(min_length=1, max_length=480)
    scene_phases: tuple[AudioVisualScenePhase, ...] = Field(
        min_length=5,
        max_length=5,
    )
    opening_scene: AudioVisualScenePhase
    development_scene: AudioVisualScenePhase
    threshold_scene: AudioVisualScenePhase
    climax_scene: AudioVisualScenePhase
    resolution_scene: AudioVisualScenePhase
    cue_plan: tuple[AudioVisualSceneCue, ...] = Field(min_length=5, max_length=24)
    transition_plan: tuple[AudioVisualSceneTransition, ...] = Field(
        min_length=4,
        max_length=4,
    )
    climax_strategy: str = Field(min_length=1, max_length=420)
    resolution_strategy: str = Field(min_length=1, max_length=420)
    visual_timing_plan: tuple[str, ...] = Field(min_length=1, max_length=8)
    motion_timing_plan: tuple[str, ...] = Field(min_length=1, max_length=8)
    audio_timing_plan: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    rhythm_timing_plan: tuple[str, ...] = Field(min_length=1, max_length=8)
    camera_timing_plan: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    motif_timing_plan: tuple[str, ...] = Field(min_length=1, max_length=8)
    emotional_timing_plan: tuple[str, ...] = Field(min_length=1, max_length=8)
    procedural_timing_plan: tuple[str, ...] = Field(min_length=1, max_length=8)
    synchronization_checkpoints: tuple[str, ...] = Field(min_length=1, max_length=8)
    scene_contrast_plan: tuple[str, ...] = Field(min_length=1, max_length=8)
    scene_continuity_plan: tuple[str, ...] = Field(min_length=1, max_length=8)
    scene_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    pacing_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    overload_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    fallback_scene_strategy: AudioVisualFallbackSceneStrategy
    unresolved_scene_gaps: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=AUDIO_VISUAL_SCENE_AUTHORITY_BOUNDARY,
        max_length=760,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=14)


@dataclass(frozen=True)
class _AudioVisualSceneContext:
    request: AssistantRequest
    route_decision: RouteDecision | None
    creative_translation: CreativeTranslation | None
    creative_intent: CreativeIntentDecomposition | None
    creative_hierarchy: CreativeHierarchyPlan | None
    creative_plan: CreativeExecutionPlan | None
    creative_constraints: CreativeConstraintSolution | None
    creative_constraint_priorities: CreativeConstraintPrioritization | None
    creative_strategy: CreativeStrategyProfile | None
    creative_techniques: CreativeTechniqueProfile | None
    runtime_capabilities: RuntimeCapabilityProfile | None
    creative_tradeoffs: CreativeTradeoffProfile | None
    creative_quality_prediction: CreativeQualityPrediction | None
    symbolic_narrative: SymbolicNarrativePlan | None
    creative_composition: CreativeCompositionPlan | None
    procedural_structure: ProceduralStructurePlan | None
    generative_structure: GenerativeStructureBlueprint | None
    semantic_motif: SemanticMotifSystem | None
    emotional_consistency: EmotionalConsistencyProfile | None
    cross_modality: CrossModalityCompositionProfile | None
    text: str
    request_tokens: frozenset[str]
    tokens: frozenset[str]


def derive_audio_visual_scene_profile(
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
) -> AudioVisualSceneProfile:
    """Derive scene metadata without generating media or changing runtime behavior."""

    context = _context(
        request=request,
        route_decision=route_decision,
        creative_translation=creative_translation,
        creative_intent=creative_intent,
        creative_hierarchy=creative_hierarchy,
        creative_plan=creative_plan,
        creative_constraints=creative_constraints,
        creative_constraint_priorities=creative_constraint_priorities,
        creative_strategy=creative_strategy,
        creative_techniques=creative_techniques,
        runtime_capabilities=runtime_capabilities,
        creative_tradeoffs=creative_tradeoffs,
        creative_quality_prediction=creative_quality_prediction,
        symbolic_narrative=symbolic_narrative,
        creative_composition=creative_composition,
        procedural_structure=procedural_structure,
        generative_structure=generative_structure,
        semantic_motif=semantic_motif,
        emotional_consistency=emotional_consistency,
        cross_modality=cross_modality,
    )
    audio_relevant = _audio_relevant(context)
    camera_relevant = _camera_relevant(context)
    pattern = _scene_pattern(context)
    phases = _scene_phases(
        context,
        pattern=pattern,
        audio_relevant=audio_relevant,
        camera_relevant=camera_relevant,
    )
    opening, development, threshold, climax, resolution = phases
    cue_plan = _cue_plan(
        context,
        phases=phases,
        audio_relevant=audio_relevant,
        camera_relevant=camera_relevant,
    )
    transition_plan = _transition_plan(
        phases,
        audio_relevant=audio_relevant,
        context=context,
    )
    scene_risks = _scene_risks(context, pattern=pattern)
    pacing_risks = _pacing_risks(
        context,
        audio_relevant=audio_relevant,
        camera_relevant=camera_relevant,
    )
    overload_risks = _overload_risks(
        context,
        audio_relevant=audio_relevant,
        camera_relevant=camera_relevant,
    )
    unresolved = _unresolved_gaps(
        context,
        audio_relevant=audio_relevant,
        camera_relevant=camera_relevant,
    )
    return AudioVisualSceneProfile(
        scene_pattern=pattern,
        scene_arc=_scene_arc(pattern, context),
        scene_phases=phases,
        opening_scene=opening,
        development_scene=development,
        threshold_scene=threshold,
        climax_scene=climax,
        resolution_scene=resolution,
        cue_plan=cue_plan,
        transition_plan=transition_plan,
        climax_strategy=_climax_strategy(context, climax, audio_relevant),
        resolution_strategy=_resolution_strategy(context, resolution),
        visual_timing_plan=_visual_timing_plan(phases),
        motion_timing_plan=_motion_timing_plan(phases),
        audio_timing_plan=_audio_timing_plan(phases) if audio_relevant else (),
        rhythm_timing_plan=_rhythm_timing_plan(phases),
        camera_timing_plan=_camera_timing_plan(phases) if camera_relevant else (),
        motif_timing_plan=_motif_timing_plan(phases),
        emotional_timing_plan=_emotional_timing_plan(phases),
        procedural_timing_plan=_procedural_timing_plan(phases),
        synchronization_checkpoints=_synchronization_checkpoints(
            context,
            phases=phases,
            audio_relevant=audio_relevant,
            camera_relevant=camera_relevant,
        ),
        scene_contrast_plan=_scene_contrast_plan(
            context,
            pattern=pattern,
            audio_relevant=audio_relevant,
            camera_relevant=camera_relevant,
        ),
        scene_continuity_plan=_scene_continuity_plan(context, phases),
        scene_risks=scene_risks,
        pacing_risks=pacing_risks,
        overload_risks=overload_risks,
        fallback_scene_strategy=_fallback_scene_strategy(
            pattern=pattern,
            audio_relevant=audio_relevant,
            camera_relevant=camera_relevant,
        ),
        unresolved_scene_gaps=unresolved,
        hitl_questions=_hitl_questions(
            unresolved,
            scene_risks,
            pacing_risks,
            overload_risks,
            audio_relevant=audio_relevant,
            camera_relevant=camera_relevant,
        ),
        prompt_guidance=_prompt_guidance(
            pattern=pattern,
            audio_relevant=audio_relevant,
            camera_relevant=camera_relevant,
        ),
        evidence=_evidence(
            context,
            pattern=pattern,
            audio_relevant=audio_relevant,
            camera_relevant=camera_relevant,
        ),
    )


def audio_visual_scene_prompt_lines(
    profile: AudioVisualSceneProfile,
) -> tuple[str, ...]:
    """Render scene metadata as compact prompt guidance."""

    lines = [
        f"Authority boundary: {profile.authority_boundary}",
        f"Scene pattern: {profile.scene_pattern}.",
        f"Scene arc: {profile.scene_arc}",
    ]
    lines.extend(
        "Scene phase: "
        f"{phase.phase}; {phase.title}; visual={phase.visual_state}; "
        f"motion={phase.motion_state}"
        for phase in profile.scene_phases
    )
    lines.extend(
        "Scene cue: "
        f"{cue.cue_id}; {cue.phase}; {cue.cue_type}; {cue.timing}; "
        f"{cue.description}"
        for cue in profile.cue_plan[:12]
    )
    lines.extend(
        "Scene transition: "
        f"{transition.from_phase} -> {transition.to_phase}; "
        f"{transition.transition}"
        for transition in profile.transition_plan
    )
    lines.append(f"Climax strategy: {profile.climax_strategy}")
    lines.append(f"Resolution strategy: {profile.resolution_strategy}")
    lines.extend(f"Visual timing: {item}" for item in profile.visual_timing_plan)
    lines.extend(f"Motion timing: {item}" for item in profile.motion_timing_plan)
    lines.extend(f"Audio timing: {item}" for item in profile.audio_timing_plan)
    lines.extend(f"Rhythm timing: {item}" for item in profile.rhythm_timing_plan)
    lines.extend(f"Camera timing: {item}" for item in profile.camera_timing_plan)
    lines.extend(f"Motif timing: {item}" for item in profile.motif_timing_plan)
    lines.extend(f"Emotional timing: {item}" for item in profile.emotional_timing_plan)
    lines.extend(
        f"Procedural timing: {item}" for item in profile.procedural_timing_plan
    )
    lines.extend(
        f"Synchronization checkpoint: {item}"
        for item in profile.synchronization_checkpoints
    )
    lines.extend(f"Scene contrast: {item}" for item in profile.scene_contrast_plan)
    lines.extend(f"Scene continuity: {item}" for item in profile.scene_continuity_plan)
    lines.extend(f"Scene risk: {item}" for item in profile.scene_risks)
    lines.extend(f"Pacing risk: {item}" for item in profile.pacing_risks)
    lines.extend(f"Scene overload risk: {item}" for item in profile.overload_risks)
    lines.append(
        "Fallback scene strategy: "
        f"{profile.fallback_scene_strategy.fallback_pattern}; "
        f"{profile.fallback_scene_strategy.simplification_strategy}"
    )
    lines.extend(
        f"Unresolved scene gap: {item}" for item in profile.unresolved_scene_gaps
    )
    lines.extend(f"HITL scene question: {item}" for item in profile.hitl_questions)
    lines.extend(
        f"Audio-visual scene guidance: {item}" for item in profile.prompt_guidance
    )
    return tuple(lines[:58])


def _context(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    creative_constraint_priorities: CreativeConstraintPrioritization | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
    creative_quality_prediction: CreativeQualityPrediction | None,
    symbolic_narrative: SymbolicNarrativePlan | None,
    creative_composition: CreativeCompositionPlan | None,
    procedural_structure: ProceduralStructurePlan | None,
    generative_structure: GenerativeStructureBlueprint | None,
    semantic_motif: SemanticMotifSystem | None,
    emotional_consistency: EmotionalConsistencyProfile | None,
    cross_modality: CrossModalityCompositionProfile | None,
) -> _AudioVisualSceneContext:
    request_tokens = _tokens(request.query)
    parts = [request.query]
    if creative_translation is not None:
        parts.extend(
            [
                creative_translation.creative_intent,
                " ".join(creative_translation.musical_references),
                " ".join(creative_translation.movement_language),
                " ".join(creative_translation.mood_atmosphere),
                " ".join(creative_translation.structure_direction),
            ]
        )
        if creative_translation.output_modality is not None:
            parts.append(creative_translation.output_modality.value)
    if creative_intent is not None:
        parts.append(creative_intent.primary_expression)
    if creative_hierarchy is not None:
        parts.extend(
            item.dimension for item in creative_hierarchy.primary_creative_priorities
        )
    if creative_plan is not None:
        parts.extend(
            [creative_plan.output_modality.value, creative_plan.generation_strategy]
        )
    if symbolic_narrative is not None:
        parts.extend(
            [symbolic_narrative.narrative_archetype, symbolic_narrative.symbolic_arc]
        )
        parts.extend(phase.title for phase in symbolic_narrative.phases)
        parts.extend(symbolic_narrative.visual_progression)
        parts.extend(symbolic_narrative.motion_progression)
        parts.extend(symbolic_narrative.audio_progression)
    if creative_composition is not None:
        parts.extend(
            [
                creative_composition.composition_pattern,
                creative_composition.rhythm_plan,
                creative_composition.camera_viewpoint_guidance or "",
                " ".join(creative_composition.audiovisual_composition_notes),
            ]
        )
    if procedural_structure is not None:
        parts.extend(
            [
                procedural_structure.primary_structure.family,
                procedural_structure.temporal_structure_plan,
                procedural_structure.audiovisual_structure_plan or "",
            ]
        )
    if generative_structure is not None:
        parts.extend(module.kind for module in generative_structure.procedural_modules)
        parts.extend(rule.phase for rule in generative_structure.evolution_rules)
        parts.extend(hook.signal for hook in generative_structure.audiovisual_hooks)
    if semantic_motif is not None:
        parts.extend(motif.motif_id for motif in semantic_motif.primary_motifs)
    if emotional_consistency is not None:
        parts.extend(
            [
                emotional_consistency.primary_emotional_tone,
                *emotional_consistency.secondary_emotional_tones,
                " ".join(emotional_consistency.emotional_arc),
                " ".join(emotional_consistency.motion_rhythm_guidance),
                " ".join(emotional_consistency.audiovisual_guidance),
            ]
        )
    if cross_modality is not None:
        parts.extend(
            [
                cross_modality.modality_pattern,
                cross_modality.primary_modality,
                " ".join(cross_modality.supporting_modalities),
                " ".join(cross_modality.modality_synchronization_plan),
                " ".join(cross_modality.contrast_balance_plan),
            ]
        )
    text = " ".join(item for item in parts if item)
    return _AudioVisualSceneContext(
        request=request,
        route_decision=route_decision,
        creative_translation=creative_translation,
        creative_intent=creative_intent,
        creative_hierarchy=creative_hierarchy,
        creative_plan=creative_plan,
        creative_constraints=creative_constraints,
        creative_constraint_priorities=creative_constraint_priorities,
        creative_strategy=creative_strategy,
        creative_techniques=creative_techniques,
        runtime_capabilities=runtime_capabilities,
        creative_tradeoffs=creative_tradeoffs,
        creative_quality_prediction=creative_quality_prediction,
        symbolic_narrative=symbolic_narrative,
        creative_composition=creative_composition,
        procedural_structure=procedural_structure,
        generative_structure=generative_structure,
        semantic_motif=semantic_motif,
        emotional_consistency=emotional_consistency,
        cross_modality=cross_modality,
        text=text,
        request_tokens=request_tokens,
        tokens=_tokens(text),
    )


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_PATTERN.findall(text.lower()))


def _audio_relevant(context: _AudioVisualSceneContext) -> bool:
    if context.request_tokens & _AUDIO_TOKENS:
        return True
    if (
        context.creative_translation is not None
        and context.creative_translation.output_modality is not None
        and context.creative_translation.output_modality.value
        in {"audio", "audiovisual"}
    ):
        return True
    if (
        context.symbolic_narrative is not None
        and context.symbolic_narrative.audio_progression
    ):
        return True
    if context.cross_modality is not None:
        return bool(
            context.cross_modality.audio_role
            or context.cross_modality.visual_to_audio_mapping
            or context.cross_modality.audio_to_motion_mapping
            or "audio" in context.cross_modality.supporting_modalities
        )
    return False


def _camera_relevant(context: _AudioVisualSceneContext) -> bool:
    if context.request_tokens & _CAMERA_TOKENS:
        return True
    if context.cross_modality is not None:
        return bool(
            context.cross_modality.camera_viewpoint_role
            or "camera" in context.cross_modality.supporting_modalities
        )
    if (
        context.creative_composition is not None
        and context.creative_composition.camera_viewpoint_guidance
    ):
        return True
    return False


def _scene_pattern(context: _AudioVisualSceneContext) -> AudioVisualScenePattern:
    archetype = (
        context.symbolic_narrative.narrative_archetype
        if context.symbolic_narrative is not None
        else None
    )
    cross_pattern = (
        context.cross_modality.modality_pattern
        if context.cross_modality is not None
        else None
    )
    if archetype in {
        "death_and_rebirth",
        "dissolution_and_reintegration",
        "fragmentation_and_recomposition",
    }:
        return "fragmentation_to_reintegration"
    if archetype == "threshold_crossing":
        return "threshold_crossing"
    if archetype == "descent_and_return":
        return "descent_to_return"
    if archetype in {"ascent", "spiral_transformation"}:
        return "spiral_ascent"
    if archetype == "expansion_from_seed_to_cosmos":
        return "seed_to_expansion"
    if archetype == "emergence_from_chaos":
        return "chaos_to_order"
    if (
        context.request_tokens & _FRAGMENTATION_TOKENS
        and context.request_tokens & _REINTEGRATION_TOKENS
    ):
        return "fragmentation_to_reintegration"
    if context.request_tokens & _RITUAL_TOKENS:
        return "ritual_opening_to_climax"
    if context.request_tokens & _WAVE_TOKENS:
        return "wave_build_and_collapse"
    if (
        context.request_tokens & _PULSE_TOKENS
        or cross_pattern == "audio_reactive_composition"
    ):
        return "pulse_escalation"
    if cross_pattern == "fragmentation_reassembly_visual_motion_layers":
        return "fragmentation_to_reintegration"
    if cross_pattern == "ritual_pulse_geometry_synchronization":
        return "ritual_opening_to_climax"
    if cross_pattern == "motion_led_transformation":
        return "contraction_to_release"
    if _tone(context) in {"rupture", "release"}:
        return "calm_expansion_after_rupture"
    return "seed_to_expansion"


def _scene_arc(
    pattern: AudioVisualScenePattern,
    context: _AudioVisualSceneContext,
) -> str:
    if context.symbolic_narrative is not None:
        return _clip(
            (
                f"{context.symbolic_narrative.symbolic_arc} Shape it as "
                f"{pattern} with explicit opening, development, threshold, "
                "climax, and resolution timing."
            ),
            480,
        )
    return _SCENE_ARCS[pattern]


def _scene_phases(
    context: _AudioVisualSceneContext,
    *,
    pattern: AudioVisualScenePattern,
    audio_relevant: bool,
    camera_relevant: bool,
) -> tuple[AudioVisualScenePhase, ...]:
    if context.symbolic_narrative is not None:
        phase_sources = context.symbolic_narrative.phases
    else:
        phase_sources = ()
    phases: list[AudioVisualScenePhase] = []
    for phase_name in _PHASE_ORDER:
        template = _PHASE_TEMPLATES[pattern][phase_name]
        narrative_phase = next(
            (phase for phase in phase_sources if phase.phase == phase_name),
            None,
        )
        title = narrative_phase.title if narrative_phase is not None else template[0]
        scene_function = (
            narrative_phase.symbolic_function
            if narrative_phase is not None
            else template[1]
        )
        visual_state = (
            narrative_phase.visual_state if narrative_phase is not None else template[2]
        )
        motion_state = (
            narrative_phase.motion_state if narrative_phase is not None else template[3]
        )
        audio_state = (
            _audio_state(context, phase_name, narrative_phase, template[4])
            if audio_relevant
            else None
        )
        phases.append(
            AudioVisualScenePhase(
                phase=phase_name,
                title=_clip(title, 140),
                scene_function=_clip(scene_function, 360),
                visual_state=_clip(visual_state, 360),
                motion_state=_clip(motion_state, 340),
                audio_state=audio_state,
                rhythm_state=_rhythm_state(context, phase_name),
                camera_state=(
                    _camera_state(context, phase_name) if camera_relevant else None
                ),
                motif_state=_motif_state(context, phase_name),
                emotional_state=_emotional_state(context, phase_name, narrative_phase),
                procedural_state=_procedural_state(context, phase_name),
                cue_ids=_cue_ids(phase_name, audio_relevant, camera_relevant),
                transition_out=_transition_out(pattern, phase_name),
                evidence=_phase_evidence(context, phase_name),
            )
        )
    return tuple(phases)  # type: ignore[return-value]


def _audio_state(
    context: _AudioVisualSceneContext,
    phase: NarrativePhaseName,
    narrative_phase: object | None,
    fallback: str,
) -> str:
    audio_state = getattr(narrative_phase, "audio_state", None)
    if audio_state:
        return _clip(str(audio_state), 320)
    if context.cross_modality is not None and context.cross_modality.audio_role:
        if phase == "threshold":
            return "Reduce audio density or use silence as a threshold cue."
        if phase == "climax":
            return (
                "Use pulse acceleration or brighter sonic emphasis as design guidance."
            )
        return _clip(context.cross_modality.audio_role, 320)
    return fallback


def _rhythm_state(
    context: _AudioVisualSceneContext,
    phase: NarrativePhaseName,
) -> str:
    if context.creative_composition is not None:
        rhythm = context.creative_composition.rhythm_plan
    else:
        rhythm = "Use measured phase timing with clear rests."
    if phase == "climax":
        return _clip(f"Intensify rhythm while preserving rests: {rhythm}", 320)
    if phase == "resolution":
        return _clip(
            f"Relax rhythm into a readable loop or stable cadence: {rhythm}", 320
        )
    return _clip(rhythm, 320)


def _camera_state(
    context: _AudioVisualSceneContext,
    phase: NarrativePhaseName,
) -> str:
    base = (
        context.cross_modality.camera_viewpoint_role
        if context.cross_modality is not None
        and context.cross_modality.camera_viewpoint_role
        else (
            context.creative_composition.camera_viewpoint_guidance
            if context.creative_composition is not None
            else None
        )
    )
    if phase in {"threshold", "climax"}:
        return _clip(
            f"Reserve active camera/viewpoint emphasis for {phase}: {base or 'shift viewpoint only at the scene hinge.'}",  # noqa: E501
            300,
        )
    return _clip(base or "Keep camera/viewpoint stable for scene continuity.", 300)


def _motif_state(
    context: _AudioVisualSceneContext,
    phase: NarrativePhaseName,
) -> str:
    motifs = _primary_motifs(context)
    if not motifs:
        return "Use a small recurring motif as a continuity marker."
    if phase == "opening":
        return f"Introduce motif seed(s): {', '.join(motifs[:3])}."
    if phase == "climax":
        return f"Transform motif(s) at peak intensity: {', '.join(motifs[:3])}."
    if phase == "resolution":
        return f"Resolve motif(s) into a stable final form: {', '.join(motifs[:3])}."
    return f"Repeat and vary motif(s) without overexplaining them: {', '.join(motifs[:3])}."


def _emotional_state(
    context: _AudioVisualSceneContext,
    phase: NarrativePhaseName,
    narrative_phase: object | None,
) -> str:
    if context.emotional_consistency is not None:
        for mapping in context.emotional_consistency.emotional_phase_mapping:
            if mapping.phase == phase:
                return _clip(
                    f"{mapping.tone} at {mapping.intensity} intensity; {mapping.guidance}",
                    320,
                )
        return _clip(
            f"Maintain {context.emotional_consistency.primary_emotional_tone} as the phase coherence target.",
            320,
        )
    state = getattr(narrative_phase, "emotional_state", None)
    return _clip(str(state or "Use the phase to clarify the emotional arc."), 320)


def _procedural_state(
    context: _AudioVisualSceneContext,
    phase: NarrativePhaseName,
) -> str:
    evolution = _evolution_for_phase(context, phase)
    if evolution is not None:
        return (
            f"Use generative evolution phase {evolution} to carry the scene "
            f"through {', '.join(_module_kinds(context)[:4]) or 'the active modules'}."
        )
    if context.procedural_structure is not None:
        return (
            f"Use {context.procedural_structure.primary_structure.family} as "
            f"the procedural basis for {phase}."
        )
    return "Keep procedural behavior tied to the scene phase instead of free-running."


def _evolution_for_phase(
    context: _AudioVisualSceneContext,
    phase: NarrativePhaseName,
) -> GenerativeEvolutionPhase | None:
    if context.generative_structure is None:
        return None
    preferred: dict[NarrativePhaseName, tuple[GenerativeEvolutionPhase, ...]] = {
        "opening": ("seed", "growth"),
        "development": ("growth", "fragmentation"),
        "threshold": ("threshold", "fragmentation"),
        "climax": ("reassembly", "fragmentation"),
        "resolution": ("stabilization", "loop"),
    }
    available = [rule.phase for rule in context.generative_structure.evolution_rules]
    for candidate in preferred[phase]:
        if candidate in available:
            return candidate
    return available[0] if available else None


def _cue_ids(
    phase: NarrativePhaseName,
    audio_relevant: bool,
    camera_relevant: bool,
) -> tuple[str, ...]:
    cue_ids = [
        f"{phase}_visual",
        f"{phase}_motion",
        f"{phase}_rhythm",
        f"{phase}_emotion",
        f"{phase}_procedural",
    ]
    if audio_relevant:
        cue_ids.append(f"{phase}_audio")
    if camera_relevant and phase in {"threshold", "climax"}:
        cue_ids.append(f"{phase}_camera")
    return tuple(cue_ids[:8])


def _transition_out(
    pattern: AudioVisualScenePattern,
    phase: NarrativePhaseName,
) -> str:
    if phase == "resolution":
        return (
            "Hold a stable ending or seamless loop without adding new scene material."
        )
    return _TRANSITION_OUT[pattern][phase]


def _phase_evidence(
    context: _AudioVisualSceneContext,
    phase: NarrativePhaseName,
) -> tuple[str, ...]:
    evidence = [f"Scene phase: {phase}."]
    if context.symbolic_narrative is not None:
        evidence.append(
            f"Narrative archetype: {context.symbolic_narrative.narrative_archetype}."
        )
    if context.cross_modality is not None:
        evidence.append(f"Cross-modality: {context.cross_modality.modality_pattern}.")
    if context.emotional_consistency is not None:
        evidence.append(
            f"Emotion: {context.emotional_consistency.primary_emotional_tone}."
        )
    return tuple(evidence[:8])


def _cue_plan(
    context: _AudioVisualSceneContext,
    *,
    phases: tuple[AudioVisualScenePhase, ...],
    audio_relevant: bool,
    camera_relevant: bool,
) -> tuple[AudioVisualSceneCue, ...]:
    cues: list[AudioVisualSceneCue] = []
    for phase in phases:
        cues.append(
            AudioVisualSceneCue(
                cue_id=f"{phase.phase}_visual",
                phase=phase.phase,
                cue_type="visual",
                description=phase.visual_state,
                timing=f"Lead {phase.phase} with visual state before secondary cues.",
                modalities=("visual_structure",),
                evidence=phase.evidence[:2],
            )
        )
        cues.append(
            AudioVisualSceneCue(
                cue_id=f"{phase.phase}_motion",
                phase=phase.phase,
                cue_type="motion",
                description=phase.motion_state,
                timing=f"Let motion clarify the {phase.phase} transition.",
                modalities=("motion", "structure"),
                evidence=phase.evidence[:2],
            )
        )
        if audio_relevant and phase.audio_state is not None:
            cues.append(
                AudioVisualSceneCue(
                    cue_id=f"{phase.phase}_audio",
                    phase=phase.phase,
                    cue_type="audio",
                    description=phase.audio_state,
                    timing=f"Use audio as timing guidance for {phase.phase}; do not claim generated audio.",
                    modalities=("audio", "rhythm", "motion"),
                    evidence=("Audio relevance detected.",),
                )
            )
        if camera_relevant and phase.phase in {"threshold", "climax"}:
            cues.append(
                AudioVisualSceneCue(
                    cue_id=f"{phase.phase}_camera",
                    phase=phase.phase,
                    cue_type="camera",
                    description=phase.camera_state
                    or "Use viewpoint only at the scene hinge.",
                    timing=f"Reserve camera/viewpoint change for {phase.phase}.",
                    modalities=("camera", "visual_structure", "motion"),
                    evidence=("Camera relevance detected.",),
                )
            )
        cues.append(
            AudioVisualSceneCue(
                cue_id=f"{phase.phase}_sync",
                phase=phase.phase,
                cue_type="synchronization",
                description=_sync_description(context, phase),
                timing=f"Checkpoint after {phase.phase} before entering the next phase.",
                modalities=_sync_modalities(audio_relevant, camera_relevant),
                evidence=("Scene synchronization checkpoint.",),
            )
        )
    return tuple(cues[:24])


def _sync_description(
    context: _AudioVisualSceneContext,
    phase: AudioVisualScenePhase,
) -> str:
    if (
        context.cross_modality is not None
        and context.cross_modality.modality_synchronization_plan
    ):
        return _clip(context.cross_modality.modality_synchronization_plan[0], 340)
    return f"Align visual, motion, rhythm, motif, and emotional cues for {phase.phase}."


def _sync_modalities(
    audio_relevant: bool,
    camera_relevant: bool,
) -> tuple[CrossModalityChannel, ...]:
    modalities: list[CrossModalityChannel] = [
        "visual_structure",
        "motion",
        "rhythm",
        "structure",
        "motif",
        "emotion",
    ]
    if audio_relevant:
        modalities.insert(2, "audio")
    if camera_relevant:
        modalities.insert(3, "camera")
    return tuple(modalities[:6])


def _transition_plan(
    phases: tuple[AudioVisualScenePhase, ...],
    *,
    audio_relevant: bool,
    context: _AudioVisualSceneContext,
) -> tuple[AudioVisualSceneTransition, ...]:
    transitions: list[AudioVisualSceneTransition] = []
    for current, following in zip(phases, phases[1:], strict=False):
        transitions.append(
            AudioVisualSceneTransition(
                from_phase=current.phase,
                to_phase=following.phase,
                transition=current.transition_out,
                visual_motion_guidance=(
                    f"Carry {current.title} into {following.title} by changing "
                    "visual density before motion intensity."
                ),
                audio_rhythm_guidance=(
                    f"Use rhythm or audio density to mark {current.phase} -> {following.phase}."
                    if audio_relevant
                    else None
                ),
                continuity_guidance=_continuity_guidance(context, current, following),
                evidence=(f"{current.phase} to {following.phase}.",),
            )
        )
    return tuple(transitions)  # type: ignore[return-value]


def _continuity_guidance(
    context: _AudioVisualSceneContext,
    current: AudioVisualScenePhase,
    following: AudioVisualScenePhase,
) -> str:
    motifs = _primary_motifs(context)
    if motifs:
        return (
            f"Keep {', '.join(motifs[:2])} visible enough to connect "
            f"{current.phase} and {following.phase}."
        )
    return f"Preserve a visual or rhythmic marker from {current.phase} into {following.phase}."


def _climax_strategy(
    context: _AudioVisualSceneContext,
    climax: AudioVisualScenePhase,
    audio_relevant: bool,
) -> str:
    audio = (
        " Synchronize pulse or silence with motion acceleration as design guidance."
        if audio_relevant
        else ""
    )
    if (
        context.cross_modality is not None
        and context.cross_modality.audio_to_motion_mapping
    ):
        audio = " " + context.cross_modality.audio_to_motion_mapping[0].mapping
    return _clip(
        (
            f"Make {climax.title} the only peak-density scene: {climax.visual_state} "
            f"{climax.motion_state}{audio}"
        ),
        420,
    )


def _resolution_strategy(
    context: _AudioVisualSceneContext,
    resolution: AudioVisualScenePhase,
) -> str:
    tone = _tone(context)
    tone_clause = f" Resolve toward {tone} coherence." if tone else ""
    return _clip(
        (
            f"After climax, reduce density and stabilize motif, motion, and "
            f"procedural behavior: {resolution.visual_state}{tone_clause}"
        ),
        420,
    )


def _visual_timing_plan(
    phases: tuple[AudioVisualScenePhase, ...],
) -> tuple[str, ...]:
    return tuple(f"{phase.phase}: {phase.visual_state}" for phase in phases)[:8]


def _motion_timing_plan(
    phases: tuple[AudioVisualScenePhase, ...],
) -> tuple[str, ...]:
    return tuple(f"{phase.phase}: {phase.motion_state}" for phase in phases)[:8]


def _audio_timing_plan(
    phases: tuple[AudioVisualScenePhase, ...],
) -> tuple[str, ...]:
    return tuple(
        f"{phase.phase}: {phase.audio_state}"
        for phase in phases
        if phase.audio_state is not None
    )[:8]


def _rhythm_timing_plan(
    phases: tuple[AudioVisualScenePhase, ...],
) -> tuple[str, ...]:
    return tuple(f"{phase.phase}: {phase.rhythm_state}" for phase in phases)[:8]


def _camera_timing_plan(
    phases: tuple[AudioVisualScenePhase, ...],
) -> tuple[str, ...]:
    return tuple(
        f"{phase.phase}: {phase.camera_state}"
        for phase in phases
        if phase.camera_state is not None
    )[:8]


def _motif_timing_plan(
    phases: tuple[AudioVisualScenePhase, ...],
) -> tuple[str, ...]:
    return tuple(f"{phase.phase}: {phase.motif_state}" for phase in phases)[:8]


def _emotional_timing_plan(
    phases: tuple[AudioVisualScenePhase, ...],
) -> tuple[str, ...]:
    return tuple(f"{phase.phase}: {phase.emotional_state}" for phase in phases)[:8]


def _procedural_timing_plan(
    phases: tuple[AudioVisualScenePhase, ...],
) -> tuple[str, ...]:
    return tuple(f"{phase.phase}: {phase.procedural_state}" for phase in phases)[:8]


def _synchronization_checkpoints(
    context: _AudioVisualSceneContext,
    *,
    phases: tuple[AudioVisualScenePhase, ...],
    audio_relevant: bool,
    camera_relevant: bool,
) -> tuple[str, ...]:
    checkpoints = [
        (
            f"{phase.phase}: align {', '.join(_sync_modalities(audio_relevant, camera_relevant)[:4])} "
            f"before transition to the next scene."
        )
        for phase in phases
    ]
    if context.cross_modality is not None:
        checkpoints.append(
            "Cross-modality checkpoint: "
            + context.cross_modality.modality_synchronization_plan[0]
        )
    return tuple(checkpoints[:8])


def _scene_contrast_plan(
    context: _AudioVisualSceneContext,
    *,
    pattern: AudioVisualScenePattern,
    audio_relevant: bool,
    camera_relevant: bool,
) -> tuple[str, ...]:
    plan = [
        f"Use {pattern} to create clear low, build, threshold, peak, and release contrast.",
        "Keep threshold quieter than development and climax.",
        "Make climax the only maximum-density scene.",
    ]
    if audio_relevant:
        plan.append(
            "Contrast audio or rhythm density against visual density instead of maximizing both constantly."
        )
    if camera_relevant:
        plan.append("Use camera/viewpoint contrast only at threshold or climax.")
    if context.emotional_consistency is not None:
        plan.append(
            "Use emotional phase intensity to decide when scenes should soften."
        )
    return tuple(plan[:8])


def _scene_continuity_plan(
    context: _AudioVisualSceneContext,
    phases: tuple[AudioVisualScenePhase, ...],
) -> tuple[str, ...]:
    motifs = _primary_motifs(context)
    plan = [
        "Carry one visual anchor, one motion rule, and one rhythm rule through all scenes.",
        "Use transitions to transform existing scene material instead of introducing unrelated material.",
    ]
    if motifs:
        plan.append(f"Preserve motif continuity through {', '.join(motifs[:3])}.")
    if context.generative_structure is not None:
        plan.append(
            f"Keep scene continuity attached to {context.generative_structure.blueprint_name}."
        )
    plan.append(f"Resolve from {phases[0].title} to {phases[-1].title} visibly.")
    return tuple(plan[:8])


def _scene_risks(
    context: _AudioVisualSceneContext,
    *,
    pattern: AudioVisualScenePattern,
) -> tuple[str, ...]:
    risks: list[str] = []
    if context.request_tokens & _AMBIGUOUS_SCENE_TOKENS:
        risks.append(
            "Scene structure is broad; the lead phase emphasis may need HITL confirmation."
        )
    if context.symbolic_narrative is None:
        risks.append("No symbolic narrative metadata is available to order scenes.")
    if (
        pattern in {"ritual_opening_to_climax", "threshold_crossing"}
        and "playful" in context.request_tokens
    ):
        risks.append(
            "Playful cues may weaken solemn scene pacing unless separated by phase."
        )
    if context.cross_modality is not None:
        risks.extend(context.cross_modality.modality_conflicts[:2])
    return tuple(risks[:8])


def _pacing_risks(
    context: _AudioVisualSceneContext,
    *,
    audio_relevant: bool,
    camera_relevant: bool,
) -> tuple[str, ...]:
    risks: list[str] = []
    if context.request_tokens & _DENSE_TOKENS:
        risks.append(
            "Dense or intense cues can collapse development, threshold, and climax into one flat peak."
        )
    if audio_relevant and context.request_tokens & {"loud", "intense"}:
        risks.append(
            "Loud audio timing can overpower threshold stillness or resolution calm."
        )
    if camera_relevant:
        risks.append(
            "Camera movement can make pacing feel busy if used outside threshold or climax."
        )
    if (
        context.emotional_consistency is not None
        and context.emotional_consistency.flattening_risks
    ):
        risks.extend(context.emotional_consistency.flattening_risks[:1])
    return tuple(risks[:8])


def _overload_risks(
    context: _AudioVisualSceneContext,
    *,
    audio_relevant: bool,
    camera_relevant: bool,
) -> tuple[str, ...]:
    risks: list[str] = []
    if context.cross_modality is not None:
        risks.extend(context.cross_modality.overload_risks[:3])
    if _module_kinds(context) and len(_module_kinds(context)) >= 4:
        risks.append("Too many module behaviors at once can obscure the scene phase.")
    if audio_relevant and camera_relevant:
        risks.append("Audio, camera, and dense motion should not peak in every scene.")
    return tuple(dict.fromkeys(risks))[:8]


def _unresolved_gaps(
    context: _AudioVisualSceneContext,
    *,
    audio_relevant: bool,
    camera_relevant: bool,
) -> tuple[str, ...]:
    gaps: list[str] = []
    if context.request_tokens & _AMBIGUOUS_SCENE_TOKENS:
        gaps.append(
            "Scene language is broad; confirm desired pacing if precision matters."
        )
    if (
        audio_relevant
        and context.cross_modality is not None
        and not context.cross_modality.audio_to_motion_mapping
    ):
        gaps.append("Audio is relevant, but audio-to-motion timing is unspecified.")
    if camera_relevant and (
        context.cross_modality is None
        or context.cross_modality.camera_viewpoint_role is None
    ):
        gaps.append(
            "Camera/viewpoint timing is relevant, but no camera role is specified."
        )
    if context.symbolic_narrative is None:
        gaps.append("Scene phases lack a symbolic narrative source.")
    return tuple(gaps[:8])


def _fallback_scene_strategy(
    *,
    pattern: AudioVisualScenePattern,
    audio_relevant: bool,
    camera_relevant: bool,
) -> AudioVisualFallbackSceneStrategy:
    reduced = []
    if audio_relevant:
        reduced.append("audio timing")
    if camera_relevant:
        reduced.append("camera/viewpoint timing")
    reduced.extend(["secondary cues", "extra peak effects"])
    fallback_pattern: AudioVisualScenePattern = (
        "seed_to_expansion"
        if pattern not in {"seed_to_expansion", "threshold_crossing"}
        else "threshold_crossing"
    )
    return AudioVisualFallbackSceneStrategy(
        fallback_pattern=fallback_pattern,
        preserved_phases=("opening", "threshold", "climax", "resolution"),
        reduced_elements=tuple(reduced[:8]),
        simplification_strategy=(
            "Preserve the five-phase scene arc, but reduce optional audio, "
            "camera, and secondary modulation cues before removing motif or "
            "emotional continuity."
        ),
        prompt_guidance=(
            "If scene scope is too broad, keep opening, threshold, climax, and resolution legible.",
            "Reduce optional audio/camera timing before changing the primary visual-motion arc.",
            "Treat scene timing as design guidance, not generated media.",
        ),
    )


def _hitl_questions(
    unresolved: tuple[str, ...],
    scene_risks: tuple[str, ...],
    pacing_risks: tuple[str, ...],
    overload_risks: tuple[str, ...],
    *,
    audio_relevant: bool,
    camera_relevant: bool,
) -> tuple[str, ...]:
    questions: list[str] = []
    if unresolved or scene_risks:
        questions.append(
            "Should the scene arc prioritize narrative clarity, audiovisual intensity, or symbolic ambiguity?"
        )
    if audio_relevant:
        questions.append(
            "Should audio timing drive scene transitions, or only support visual rhythm?"
        )
    if camera_relevant:
        questions.append(
            "Should camera/viewpoint shifts happen only at threshold and climax?"
        )
    if pacing_risks or overload_risks:
        questions.append(
            "Should the climax maximize density, or preserve readability with a quieter threshold?"
        )
    return tuple(questions[:6])


def _prompt_guidance(
    *,
    pattern: AudioVisualScenePattern,
    audio_relevant: bool,
    camera_relevant: bool,
) -> tuple[str, ...]:
    guidance = [
        f"Use {pattern} as the bounded audio-visual scene arc.",
        "Keep opening, development, threshold, climax, and resolution distinct.",
        "Treat scene cues and timing plans as design guidance, not generated media.",
        "Make climax the strongest synchronized scene and resolution the stabilizing scene.",
    ]
    if audio_relevant:
        guidance.append(
            "Include audio timing only as prompt guidance unless the final artifact explicitly implements audio."
        )
    if camera_relevant:
        guidance.append(
            "Use camera/viewpoint timing only when the selected output scope supports it."
        )
    guidance.append(
        "Do not auto-select runtimes, route providers, change preview behavior, or add runtime repair."
    )
    return tuple(guidance[:8])


def _evidence(
    context: _AudioVisualSceneContext,
    *,
    pattern: AudioVisualScenePattern,
    audio_relevant: bool,
    camera_relevant: bool,
) -> tuple[str, ...]:
    evidence = [
        f"Scene pattern: {pattern}.",
        f"Audio relevance: {audio_relevant}.",
        f"Camera relevance: {camera_relevant}.",
    ]
    if context.symbolic_narrative is not None:
        evidence.append(f"Narrative: {context.symbolic_narrative.narrative_archetype}.")
    if context.creative_composition is not None:
        evidence.append(
            f"Composition: {context.creative_composition.composition_pattern}."
        )
    if context.procedural_structure is not None:
        evidence.append(
            f"Procedural structure: {context.procedural_structure.primary_structure.family}."
        )
    if context.generative_structure is not None:
        evidence.append(
            f"Generative architecture: {context.generative_structure.generative_architecture}."
        )
    if context.semantic_motif is not None:
        evidence.append(f"Motifs: {', '.join(_primary_motifs(context))}.")
    if context.emotional_consistency is not None:
        evidence.append(
            f"Emotion: {context.emotional_consistency.primary_emotional_tone}."
        )
    if context.cross_modality is not None:
        evidence.append(f"Cross-modality: {context.cross_modality.modality_pattern}.")
    return tuple(evidence[:14])


def _primary_motifs(
    context: _AudioVisualSceneContext,
) -> tuple[SemanticMotifId, ...]:
    if context.semantic_motif is None:
        return ()
    return tuple(motif.motif_id for motif in context.semantic_motif.primary_motifs)


def _module_kinds(
    context: _AudioVisualSceneContext,
) -> tuple[GenerativeModuleKind, ...]:
    if context.generative_structure is None:
        return ()
    return tuple(
        module.kind for module in context.generative_structure.procedural_modules
    )


def _tone(context: _AudioVisualSceneContext) -> EmotionalTone | None:
    if context.emotional_consistency is None:
        return None
    return context.emotional_consistency.primary_emotional_tone


def _clip(value: str, limit: int) -> str:
    normalized = " ".join(value.strip().split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "."


_PHASE_ORDER: tuple[NarrativePhaseName, ...] = (
    "opening",
    "development",
    "threshold",
    "climax",
    "resolution",
)

_SCENE_ARCS: dict[AudioVisualScenePattern, str] = {
    "seed_to_expansion": "Begin from a single seed, expand structure and motion, cross a threshold of scale, peak in full form, then resolve into stable breadth.",  # noqa: E501
    "descent_to_return": "Descend into darker or denser material, cross a low threshold, return through a peak reveal, and resolve with restored order.",  # noqa: E501
    "fragmentation_to_reintegration": "Open with a coherent form, fragment into turbulent pieces, hold sparse threshold stillness, reassemble at climax, and resolve as integrated geometry.",  # noqa: E501
    "threshold_crossing": "Approach a boundary, complicate the crossing, pause at the liminal hinge, reveal the changed state, and stabilize after crossing.",  # noqa: E501
    "spiral_ascent": "Open with a small spiral, expand upward through layered rotations, cross a vertical threshold, peak in ascent, and resolve as elevated order.",  # noqa: E501
    "chaos_to_order": "Begin in unstable noise, develop turbulence, identify a threshold rule, crystallize order at climax, and resolve into coherent structure.",  # noqa: E501
    "void_to_emergence": "Open in sparse void, introduce faint signals, cross from absence to form, peak in emergence, and resolve as quiet presence.",  # noqa: E501
    "contraction_to_release": "Contract visual and motion energy, increase pressure, pause at compression, release at climax, and resolve into calm expansion.",  # noqa: E501
    "ritual_opening_to_climax": "Open ceremonially, layer repeated cues, cross a solemn threshold, peak in synchronized ritual intensity, and resolve with measured calm.",  # noqa: E501
    "wave_build_and_collapse": "Open with a small wave, build amplitude, hold crest tension, collapse at climax, and resolve through receding rhythm.",  # noqa: E501
    "constellation_activation": "Open with isolated points, activate relations, cross into network coherence, peak as a constellation, and resolve as stable connected light.",  # noqa: E501
    "mirror_inversion": "Open with symmetry, destabilize reflection, cross an inversion threshold, peak in mirrored reversal, and resolve into balanced duality.",  # noqa: E501
    "pulse_escalation": "Open with a restrained pulse, increase cue density, hold a syncopated threshold, peak in pulse acceleration, and resolve into slower cadence.",  # noqa: E501
    "calm_expansion_after_rupture": "Open after rupture, contain scattered motion, cross into calmer alignment, peak in gentle expansion, and resolve as quiet integration.",  # noqa: E501
}

_PHASE_TEMPLATES: dict[
    AudioVisualScenePattern,
    dict[NarrativePhaseName, tuple[str, str, str, str, str]],
] = {
    "seed_to_expansion": {
        "opening": (
            "Seed",
            "Establish the origin point.",
            "Low-density seed point or small geometry.",
            "Minimal drift or breathing motion.",
            "Near silence or a sparse pulse.",
        ),
        "development": (
            "Expansion",
            "Grow the visible system.",
            "Increasing rings, branches, or fields.",
            "Outward expansion with controlled acceleration.",
            "Pulse density gradually increases.",
        ),
        "threshold": (
            "Scale Shift",
            "Mark the change from local to full scale.",
            "A sparse gap or scale jump.",
            "Brief pause before expansion resumes.",
            "Reduce audio density before the scale shift.",
        ),
        "climax": (
            "Full Form",
            "Reveal the expanded system.",
            "Maximum legible breadth and luminosity.",
            "Largest coherent motion range.",
            "Pulse reaches strongest emphasis.",
        ),
        "resolution": (
            "Stable Breadth",
            "Settle into readable scale.",
            "Balanced full-field geometry.",
            "Calmer loop or slow expansion.",
            "Return to slower cadence.",
        ),
    },
    "fragmentation_to_reintegration": {
        "opening": (
            "Whole Form",
            "Show the form before rupture.",
            "Coherent luminous form in low-to-medium density.",
            "Slow contraction or orbit.",
            "Restrained low pulse or drone.",
        ),
        "development": (
            "Fragmentation",
            "Break the form into active fragments.",
            "Particles, shards, or broken contours spread outward.",
            "Turbulent scatter with readable trajectories.",
            "Pulse or noise density rises.",
        ),
        "threshold": (
            "Sparse Stillness",
            "Create a liminal pause before reassembly.",
            "Low-density field with visible fragments suspended.",
            "Motion slows or briefly freezes.",
            "Audio drops to silence, drone, or thin pulse.",
        ),
        "climax": (
            "Reassembly",
            "Synchronize return into a transformed form.",
            "Fragments spiral or converge into integrated geometry.",
            "Rapid but readable reassembly acceleration.",
            "Pulse acceleration supports the convergence.",
        ),
        "resolution": (
            "Integrated Geometry",
            "Stabilize the transformed whole.",
            "Luminous coherent geometry with reduced density.",
            "Calm expansion or orbit.",
            "Cadence relaxes after reassembly.",
        ),
    },
    "threshold_crossing": {
        "opening": (
            "Approach",
            "Establish the boundary.",
            "A visible gate, edge, or central threshold.",
            "Slow approach or orbit toward the boundary.",
            "Sparse pulse marks approach.",
        ),
        "development": (
            "Pressure",
            "Complicate the crossing.",
            "Layered density around the boundary.",
            "Motion compresses toward the hinge.",
            "Pulse thickens or syncopates.",
        ),
        "threshold": (
            "Crossing",
            "Hold the liminal hinge.",
            "Minimal field, clear boundary, reduced detail.",
            "Brief stillness or slowed crossing.",
            "Silence or restrained tone marks the hinge.",
        ),
        "climax": (
            "Reveal",
            "Show the changed state after crossing.",
            "Bright reveal or reorganized structure.",
            "Motion opens beyond the boundary.",
            "Pulse returns with stronger clarity.",
        ),
        "resolution": (
            "Afterimage",
            "Stabilize the new state.",
            "Resolved boundary transformed into stable form.",
            "Motion eases into loop or rest.",
            "Cadence lowers after reveal.",
        ),
    },
    "ritual_opening_to_climax": {
        "opening": (
            "Invocation",
            "Introduce ceremonial order.",
            "Symmetric or radial anchor with restrained light.",
            "Measured repetition begins.",
            "Low ceremonial pulse or drone.",
        ),
        "development": (
            "Layering",
            "Accumulate repeated ritual cues.",
            "More rings, motifs, or repeated figures.",
            "Pulsed expansion with deliberate rests.",
            "Pulse layers increase without clutter.",
        ),
        "threshold": (
            "Pause",
            "Create solemn stillness before the peak.",
            "Sparse radial field or dimmed mandala.",
            "Motion nearly stops.",
            "Audio thins or pauses.",
        ),
        "climax": (
            "Ritual Peak",
            "Synchronize motif, rhythm, and light.",
            "Bright radial or mandala peak.",
            "Measured expansion reaches maximum intensity.",
            "Pulse reaches strongest synchronized emphasis.",
        ),
        "resolution": (
            "Closing",
            "Return to ceremonial calm.",
            "Simplified stable geometry.",
            "Slow symmetrical settling.",
            "Cadence returns to low pulse.",
        ),
    },
    "pulse_escalation": {
        "opening": (
            "Pulse Seed",
            "Establish the timing source.",
            "Small repeated visual pulse.",
            "Low-amplitude oscillation.",
            "Sparse pulse.",
        ),
        "development": (
            "Pulse Build",
            "Increase timing density.",
            "Growing repeated forms.",
            "Motion follows pulse acceleration.",
            "Pulse rate or density increases.",
        ),
        "threshold": (
            "Sync Break",
            "Create a rhythmic gap.",
            "Reduced visual density.",
            "Motion pauses or desynchronizes briefly.",
            "Silence or off-beat cue.",
        ),
        "climax": (
            "Pulse Peak",
            "Synchronize strongest pulse with visual event.",
            "Maximum pulse-linked brightness or density.",
            "Fastest readable motion.",
            "Strongest pulse emphasis.",
        ),
        "resolution": (
            "Cadence Release",
            "Relax timing after peak.",
            "Lower-density repeated forms.",
            "Motion eases to slower cadence.",
            "Pulse slows or softens.",
        ),
    },
}

for _pattern in (
    "descent_to_return",
    "spiral_ascent",
    "chaos_to_order",
    "void_to_emergence",
    "contraction_to_release",
    "wave_build_and_collapse",
    "constellation_activation",
    "mirror_inversion",
    "calm_expansion_after_rupture",
):
    _PHASE_TEMPLATES[_pattern] = _PHASE_TEMPLATES["seed_to_expansion"]

_TRANSITION_OUT: dict[AudioVisualScenePattern, dict[NarrativePhaseName, str]] = {
    pattern: {
        "opening": "Increase density gradually while preserving the opening anchor.",
        "development": "Move from build-up into a clear threshold pause or hinge.",
        "threshold": "Release the held threshold into one synchronized climax cue.",
        "climax": "Reduce density and stabilize the dominant motif after the peak.",
        "resolution": "Hold a stable ending or seamless loop.",
    }
    for pattern in _SCENE_ARCS
}

_TRANSITION_OUT["fragmentation_to_reintegration"] = {
    "opening": "Let coherent form contract or fracture into controlled fragmentation.",
    "development": "Thin the fragmented field into threshold stillness.",
    "threshold": "Start reassembly from the sparse field into synchronized convergence.",
    "climax": "Ease reassembled geometry into a calmer integrated state.",
    "resolution": "Hold integrated geometry as a stable ending or loop.",
}
_TRANSITION_OUT["threshold_crossing"] = {
    "opening": "Move approach energy into pressure at the boundary.",
    "development": "Reduce pressure into a clear liminal crossing pause.",
    "threshold": "Open the crossed state with one reveal cue.",
    "climax": "Let the reveal settle into the afterimage.",
    "resolution": "Hold the afterimage as stable form.",
}

__all__ = [
    "AUDIO_VISUAL_SCENE_AUTHORITY_BOUNDARY",
    "AudioVisualCueType",
    "AudioVisualFallbackSceneStrategy",
    "AudioVisualSceneCue",
    "AudioVisualScenePattern",
    "AudioVisualScenePhase",
    "AudioVisualSceneProfile",
    "AudioVisualSceneTransition",
    "audio_visual_scene_prompt_lines",
    "derive_audio_visual_scene_profile",
]
