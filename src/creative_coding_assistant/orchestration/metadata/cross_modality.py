"""Bounded Cross-Modality Composer for V3.2 workflows."""

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
from creative_coding_assistant.orchestration.emotional_consistency import (
    EmotionalConsistencyProfile,
    EmotionalTone,
)
from creative_coding_assistant.orchestration.generative_structure import (
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

CrossModalityChannel = Literal[
    "visual_structure",
    "motion",
    "audio",
    "rhythm",
    "camera",
    "structure",
    "motif",
    "emotion",
    "interaction",
]
CrossModalityPattern = Literal[
    "visual_led_composition",
    "audio_reactive_composition",
    "motion_led_transformation",
    "rhythm_led_scene_evolution",
    "camera_led_immersion",
    "motif_led_symbolic_recurrence",
    "structure_led_procedural_evolution",
    "emotion_led_modulation",
    "balanced_audiovisual_composition",
    "minimal_visual_strong_sonic_cueing",
    "dense_visual_restrained_audio",
    "ritual_pulse_geometry_synchronization",
    "fragmentation_reassembly_visual_motion_layers",
]

CROSS_MODALITY_AUTHORITY_BOUNDARY = (
    "The Cross-Modality Composer organizes visual, motion, audio, rhythm, "
    "camera, structure, motif, and emotional signals as inspectable design "
    "metadata only; it does not generate executable code, generate audio, "
    "render visuals, auto-select runtimes, route providers or models, change "
    "preview behavior, implement runtime repair, implement V4 multi-agent "
    "runtime, implement HoloMind, or implement an Audio-Visual Scene System."
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
        "reactive",
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
        "spatial",
        "viewpoint",
    }
)
_DENSE_TOKENS = frozenset(
    {
        "complex",
        "dense",
        "many",
        "maximal",
        "overwhelming",
        "particles",
        "swarm",
        "turbulent",
    }
)
_LOUD_AUDIO_TOKENS = frozenset({"intense", "loud", "noisy", "thunder", "wall"})
_AMBIGUOUS_MODALITY_TOKENS = frozenset(
    {
        "audiovisual",
        "crossmodal",
        "cross-modality",
        "maybe",
        "multimodal",
        "something",
        "sync",
        "vibe",
    }
)
_FRAGMENTATION_TOKENS = frozenset(
    {"dissolve", "dissolves", "fragment", "fragments", "particles", "rupture"}
)
_REASSEMBLY_TOKENS = frozenset(
    {"phoenix", "reassemble", "reassembly", "reform", "reforms", "reintegration"}
)
_RITUAL_TOKENS = frozenset({"mandala", "pulse", "ritual", "sacred", "solemn", "temple"})
_ROLE_MAX_LENGTH = 340
_COMMON_ROLE_MAX_LENGTH = 320
_CAMERA_ROLE_MAX_LENGTH = 300
_TEMPORAL_CUE_MAX_LENGTH = 260
_TEMPORAL_TIMING_MAX_LENGTH = 280


class CrossModalityRole(BaseModel):
    """One modality's role in the composition hierarchy."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    modality: CrossModalityChannel
    role: str = Field(min_length=1, max_length=340)
    priority: Literal["primary", "secondary", "supporting", "fallback"]
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=6)


class CrossModalityMapping(BaseModel):
    """Design mapping from one modality signal to another."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source_modality: CrossModalityChannel
    target_modality: CrossModalityChannel
    mapping: str = Field(min_length=1, max_length=340)
    cues: tuple[str, ...] = Field(min_length=1, max_length=8)
    motif_id: SemanticMotifId | None = None
    emotional_tone: EmotionalTone | None = None
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=6)


class CrossModalityTemporalCue(BaseModel):
    """One temporal synchronization cue across modalities."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    phase: NarrativePhaseName
    cue: str = Field(min_length=1, max_length=260)
    modalities: tuple[CrossModalityChannel, ...] = Field(min_length=2, max_length=6)
    timing_guidance: str = Field(min_length=1, max_length=280)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=6)


class CrossModalityFallbackStrategy(BaseModel):
    """Lower-risk multimodal plan when scope or runtime support tightens."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    fallback_pattern: CrossModalityPattern
    preserved_modalities: tuple[CrossModalityChannel, ...] = Field(
        min_length=2,
        max_length=6,
    )
    reduced_modalities: tuple[CrossModalityChannel, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    simplification_strategy: str = Field(min_length=1, max_length=340)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=5)


class CrossModalityCompositionProfile(BaseModel):
    """Inspectable cross-modality composition metadata derived before generation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["cross_modality_composer"] = "cross_modality_composer"
    modality_pattern: CrossModalityPattern
    primary_modality: CrossModalityChannel
    supporting_modalities: tuple[CrossModalityChannel, ...] = Field(
        min_length=1,
        max_length=8,
    )
    modality_hierarchy: tuple[CrossModalityRole, ...] = Field(
        min_length=3,
        max_length=9,
    )
    visual_role: str = Field(min_length=1, max_length=320)
    motion_role: str = Field(min_length=1, max_length=320)
    audio_role: str | None = Field(default=None, max_length=320)
    rhythm_role: str = Field(min_length=1, max_length=320)
    camera_viewpoint_role: str | None = Field(default=None, max_length=300)
    structure_role: str = Field(min_length=1, max_length=340)
    motif_role: str = Field(min_length=1, max_length=320)
    emotion_role: str = Field(min_length=1, max_length=320)
    modality_synchronization_plan: tuple[str, ...] = Field(
        min_length=1,
        max_length=8,
    )
    visual_to_audio_mapping: tuple[CrossModalityMapping, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    audio_to_motion_mapping: tuple[CrossModalityMapping, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    motion_to_structure_mapping: tuple[CrossModalityMapping, ...] = Field(
        min_length=1,
        max_length=6,
    )
    motif_to_modality_mapping: tuple[CrossModalityMapping, ...] = Field(
        min_length=1,
        max_length=8,
    )
    emotional_to_modality_mapping: tuple[CrossModalityMapping, ...] = Field(
        min_length=1,
        max_length=8,
    )
    temporal_cue_plan: tuple[CrossModalityTemporalCue, ...] = Field(
        min_length=1,
        max_length=6,
    )
    contrast_balance_plan: tuple[str, ...] = Field(min_length=1, max_length=8)
    modality_conflicts: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    overload_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    underuse_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    fallback_multimodal_strategy: CrossModalityFallbackStrategy
    unresolved_modality_gaps: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=CROSS_MODALITY_AUTHORITY_BOUNDARY,
        max_length=760,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=14)


@dataclass(frozen=True)
class _CrossModalityContext:
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
    text: str
    request_tokens: frozenset[str]
    tokens: frozenset[str]


def derive_cross_modality_composition_profile(
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
) -> CrossModalityCompositionProfile:
    """Derive cross-modality metadata without selecting runtime or rendering output."""

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
    )
    audio_relevant = _audio_relevant(context)
    camera_relevant = _camera_relevant(context)
    pattern = _modality_pattern(
        context,
        audio_relevant=audio_relevant,
        camera_relevant=camera_relevant,
    )
    primary = _primary_modality(
        pattern,
        audio_relevant=audio_relevant,
        camera_relevant=camera_relevant,
    )
    supporting = _supporting_modalities(
        primary,
        audio_relevant=audio_relevant,
        camera_relevant=camera_relevant,
    )
    visual_role = _clip_text(_visual_role(context, pattern), _COMMON_ROLE_MAX_LENGTH)
    motion_role = _clip_text(_motion_role(context), _COMMON_ROLE_MAX_LENGTH)
    audio_role = _clip_optional_text(
        _audio_role(context, audio_relevant=audio_relevant),
        _COMMON_ROLE_MAX_LENGTH,
    )
    rhythm_role = _clip_text(
        _rhythm_role(context, audio_relevant=audio_relevant),
        _COMMON_ROLE_MAX_LENGTH,
    )
    camera_role = _clip_optional_text(
        _camera_role(context, camera_relevant=camera_relevant),
        _CAMERA_ROLE_MAX_LENGTH,
    )
    structure_role = _clip_text(_structure_role(context), _ROLE_MAX_LENGTH)
    motif_role = _clip_text(_motif_role(context), _COMMON_ROLE_MAX_LENGTH)
    emotion_role = _clip_text(_emotion_role(context), _COMMON_ROLE_MAX_LENGTH)
    modality_conflicts = _modality_conflicts(
        context,
        audio_relevant=audio_relevant,
        camera_relevant=camera_relevant,
    )
    overload = _overload_risks(
        context,
        audio_relevant=audio_relevant,
        camera_relevant=camera_relevant,
    )
    underuse = _underuse_risks(
        context,
        audio_relevant=audio_relevant,
        camera_relevant=camera_relevant,
    )
    unresolved = _unresolved_gaps(
        context,
        audio_relevant=audio_relevant,
        camera_relevant=camera_relevant,
    )
    return CrossModalityCompositionProfile(
        modality_pattern=pattern,
        primary_modality=primary,
        supporting_modalities=supporting,
        modality_hierarchy=_modality_hierarchy(
            primary=primary,
            supporting=supporting,
            visual_role=visual_role,
            motion_role=motion_role,
            audio_role=audio_role,
            rhythm_role=rhythm_role,
            camera_role=camera_role,
            structure_role=structure_role,
            motif_role=motif_role,
            emotion_role=emotion_role,
            context=context,
        ),
        visual_role=visual_role,
        motion_role=motion_role,
        audio_role=audio_role,
        rhythm_role=rhythm_role,
        camera_viewpoint_role=camera_role,
        structure_role=structure_role,
        motif_role=motif_role,
        emotion_role=emotion_role,
        modality_synchronization_plan=_synchronization_plan(
            context,
            pattern=pattern,
            audio_relevant=audio_relevant,
            camera_relevant=camera_relevant,
        ),
        visual_to_audio_mapping=_visual_to_audio_mappings(
            context,
            audio_relevant=audio_relevant,
        ),
        audio_to_motion_mapping=_audio_to_motion_mappings(
            context,
            audio_relevant=audio_relevant,
        ),
        motion_to_structure_mapping=_motion_to_structure_mappings(context),
        motif_to_modality_mapping=_motif_to_modality_mappings(
            context,
            audio_relevant=audio_relevant,
        ),
        emotional_to_modality_mapping=_emotional_to_modality_mappings(
            context,
            audio_relevant=audio_relevant,
        ),
        temporal_cue_plan=_temporal_cues(
            context,
            audio_relevant=audio_relevant,
            camera_relevant=camera_relevant,
        ),
        contrast_balance_plan=_contrast_balance_plan(
            context,
            pattern=pattern,
            audio_relevant=audio_relevant,
            camera_relevant=camera_relevant,
        ),
        modality_conflicts=modality_conflicts,
        overload_risks=overload,
        underuse_risks=underuse,
        fallback_multimodal_strategy=_fallback_strategy(
            pattern=pattern,
            primary=primary,
            supporting=supporting,
            audio_relevant=audio_relevant,
            camera_relevant=camera_relevant,
        ),
        unresolved_modality_gaps=unresolved,
        hitl_questions=_hitl_questions(
            unresolved,
            modality_conflicts,
            overload,
            audio_relevant=audio_relevant,
            camera_relevant=camera_relevant,
        ),
        prompt_guidance=_prompt_guidance(
            pattern=pattern,
            primary=primary,
            audio_relevant=audio_relevant,
            camera_relevant=camera_relevant,
        ),
        evidence=_evidence(
            context,
            pattern=pattern,
            primary=primary,
            supporting=supporting,
            audio_relevant=audio_relevant,
            camera_relevant=camera_relevant,
        ),
    )


def cross_modality_prompt_lines(
    profile: CrossModalityCompositionProfile,
) -> tuple[str, ...]:
    """Render cross-modality metadata as compact prompt guidance."""

    lines = [
        f"Authority boundary: {profile.authority_boundary}",
        f"Modality pattern: {profile.modality_pattern}.",
        f"Primary modality: {profile.primary_modality}.",
        "Supporting modalities: " + ", ".join(profile.supporting_modalities) + ".",
        f"Visual role: {profile.visual_role}",
        f"Motion role: {profile.motion_role}",
        f"Rhythm role: {profile.rhythm_role}",
        f"Structure role: {profile.structure_role}",
        f"Motif role: {profile.motif_role}",
        f"Emotion role: {profile.emotion_role}",
    ]
    if profile.audio_role:
        lines.append(f"Audio role: {profile.audio_role}")
    if profile.camera_viewpoint_role:
        lines.append(f"Camera/viewpoint role: {profile.camera_viewpoint_role}")
    lines.extend(
        f"Modality hierarchy: {item.modality}; {item.priority}; {item.role}"
        for item in profile.modality_hierarchy[:6]
    )
    lines.extend(
        f"Synchronization plan: {item}"
        for item in profile.modality_synchronization_plan
    )
    lines.extend(
        _mapping_lines("Visual-to-audio mapping", profile.visual_to_audio_mapping)
    )
    lines.extend(
        _mapping_lines("Audio-to-motion mapping", profile.audio_to_motion_mapping)
    )
    lines.extend(
        _mapping_lines(
            "Motion-to-structure mapping",
            profile.motion_to_structure_mapping,
        )
    )
    lines.extend(
        _mapping_lines(
            "Motif-to-modality mapping",
            profile.motif_to_modality_mapping,
        )
    )
    lines.extend(
        _mapping_lines(
            "Emotional-to-modality mapping",
            profile.emotional_to_modality_mapping,
        )
    )
    lines.extend(
        "Temporal cue: "
        f"{item.phase}; {', '.join(item.modalities)}; {item.timing_guidance}"
        for item in profile.temporal_cue_plan
    )
    lines.extend(f"Contrast/balance: {item}" for item in profile.contrast_balance_plan)
    lines.extend(f"Modality conflict: {item}" for item in profile.modality_conflicts)
    lines.extend(f"Modality overload risk: {item}" for item in profile.overload_risks)
    lines.extend(f"Modality underuse risk: {item}" for item in profile.underuse_risks)
    lines.append(
        "Fallback multimodal strategy: "
        f"{profile.fallback_multimodal_strategy.fallback_pattern}; "
        f"{profile.fallback_multimodal_strategy.simplification_strategy}"
    )
    lines.extend(
        f"Unresolved modality gap: {item}" for item in profile.unresolved_modality_gaps
    )
    lines.extend(f"HITL modality question: {item}" for item in profile.hitl_questions)
    lines.extend(f"Cross-modality guidance: {item}" for item in profile.prompt_guidance)
    return tuple(lines[:52])


def _mapping_lines(
    label: str,
    mappings: tuple[CrossModalityMapping, ...],
) -> tuple[str, ...]:
    return tuple(
        f"{label}: {item.source_modality} -> {item.target_modality}; {item.mapping}"
        for item in mappings
    )


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
) -> _CrossModalityContext:
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
    if creative_composition is not None:
        parts.extend(
            [
                creative_composition.composition_pattern,
                creative_composition.primary_focal_point,
                creative_composition.rhythm_plan,
                creative_composition.density_plan,
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
        parts.extend(hook.signal for hook in generative_structure.audiovisual_hooks)
        parts.extend(rule.phase for rule in generative_structure.evolution_rules)
    if symbolic_narrative is not None:
        parts.extend(symbolic_narrative.visual_progression)
        parts.extend(symbolic_narrative.motion_progression)
        parts.extend(symbolic_narrative.audio_progression)
    if semantic_motif is not None:
        parts.extend(motif.motif_id for motif in semantic_motif.primary_motifs)
    if emotional_consistency is not None:
        parts.extend(
            [
                emotional_consistency.primary_emotional_tone,
                *emotional_consistency.secondary_emotional_tones,
                " ".join(emotional_consistency.motion_rhythm_guidance),
                " ".join(emotional_consistency.audiovisual_guidance),
            ]
        )
    text = " ".join(item for item in parts if item)
    return _CrossModalityContext(
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
        text=text,
        request_tokens=request_tokens,
        tokens=_tokens(text),
    )


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_PATTERN.findall(text.lower()))


def _audio_relevant(context: _CrossModalityContext) -> bool:
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
        context.creative_plan is not None
        and context.creative_plan.output_modality.value
        in {
            "audio",
            "audiovisual",
        }
    ):
        return True
    if (
        context.symbolic_narrative is not None
        and context.symbolic_narrative.audio_progression
    ):
        return True
    if (
        context.creative_composition is not None
        and context.creative_composition.audiovisual_composition_notes
    ):
        return True
    if (
        context.procedural_structure is not None
        and context.procedural_structure.audiovisual_structure_plan
    ):
        return True
    if context.generative_structure is not None:
        return bool(context.generative_structure.audiovisual_hooks) or any(
            module.kind == "audio_reactive_modulation_layer"
            for module in context.generative_structure.procedural_modules
        )
    return False


def _camera_relevant(context: _CrossModalityContext) -> bool:
    if context.request_tokens & _CAMERA_TOKENS:
        return True
    if (
        context.creative_composition is not None
        and context.creative_composition.camera_viewpoint_guidance
    ):
        return True
    if context.generative_structure is not None:
        return any(
            module.kind == "camera_motion_path_hook"
            for module in context.generative_structure.procedural_modules
        )
    return False


def _modality_pattern(
    context: _CrossModalityContext,
    *,
    audio_relevant: bool,
    camera_relevant: bool,
) -> CrossModalityPattern:
    module_kinds = _module_kinds(context)
    if (
        (context.request_tokens & _FRAGMENTATION_TOKENS)
        and (context.request_tokens & _REASSEMBLY_TOKENS)
    ) or {
        "particle_emitter",
        "geometry_reassembly_layer",
    }.issubset(module_kinds):
        return "fragmentation_reassembly_visual_motion_layers"
    if (context.request_tokens & _RITUAL_TOKENS) and audio_relevant:
        return "ritual_pulse_geometry_synchronization"
    if audio_relevant and (
        "audio_reactive_modulation_layer" in module_kinds
        or "reactive" in context.request_tokens
    ):
        return "audio_reactive_composition"
    if audio_relevant and context.request_tokens & _DENSE_TOKENS:
        return "dense_visual_restrained_audio"
    if audio_relevant:
        return "balanced_audiovisual_composition"
    if camera_relevant:
        return "camera_led_immersion"
    if (
        context.emotional_consistency is not None
        and context.emotional_consistency.primary_emotional_tone
        in {"dissolution", "transformation", "rupture", "integration"}
    ):
        return "motion_led_transformation"
    if (
        context.semantic_motif is not None
        and len(context.semantic_motif.primary_motifs) >= 2
    ):
        return "motif_led_symbolic_recurrence"
    if context.generative_structure is not None:
        return "structure_led_procedural_evolution"
    return "visual_led_composition"


def _module_kinds(context: _CrossModalityContext) -> set[GenerativeModuleKind]:
    if context.generative_structure is None:
        return set()
    return {module.kind for module in context.generative_structure.procedural_modules}


def _primary_modality(
    pattern: CrossModalityPattern,
    *,
    audio_relevant: bool,
    camera_relevant: bool,
) -> CrossModalityChannel:
    if pattern == "audio_reactive_composition":
        return "audio"
    if pattern == "motion_led_transformation":
        return "motion"
    if pattern == "camera_led_immersion" and camera_relevant:
        return "camera"
    if pattern == "motif_led_symbolic_recurrence":
        return "motif"
    if pattern == "emotion_led_modulation":
        return "emotion"
    if pattern == "structure_led_procedural_evolution":
        return "structure"
    if pattern == "minimal_visual_strong_sonic_cueing" and audio_relevant:
        return "audio"
    return "visual_structure"


def _supporting_modalities(
    primary: CrossModalityChannel,
    *,
    audio_relevant: bool,
    camera_relevant: bool,
) -> tuple[CrossModalityChannel, ...]:
    ordered: list[CrossModalityChannel] = [
        "visual_structure",
        "motion",
        "rhythm",
        "structure",
        "motif",
        "emotion",
    ]
    if audio_relevant:
        ordered.insert(2, "audio")
    if camera_relevant:
        ordered.insert(3, "camera")
    if primary in ordered:
        ordered.remove(primary)
    return tuple(ordered)


def _visual_role(
    context: _CrossModalityContext,
    pattern: CrossModalityPattern,
) -> str:
    if context.creative_composition is not None:
        focal = context.creative_composition.primary_focal_point
        pattern_label = context.creative_composition.composition_pattern
        return (
            f"Carry {pattern_label} through {focal}, keeping visual structure "
            "legible before secondary modality effects."
        )
    if pattern == "fragmentation_reassembly_visual_motion_layers":
        return "Show rupture, particle dispersal, and reassembly as the visible multimodal anchor."
    return "Provide the primary visible form, density, and spatial anchor for other modalities."


def _motion_role(context: _CrossModalityContext) -> str:
    if (
        context.emotional_consistency is not None
        and context.emotional_consistency.motion_rhythm_guidance
    ):
        return context.emotional_consistency.motion_rhythm_guidance[0]
    if (
        context.symbolic_narrative is not None
        and context.symbolic_narrative.motion_progression
    ):
        return context.symbolic_narrative.motion_progression[0]
    return "Translate the composition into readable phase changes, not constant motion."


def _audio_role(
    context: _CrossModalityContext,
    *,
    audio_relevant: bool,
) -> str | None:
    if not audio_relevant:
        return None
    if (
        context.emotional_consistency is not None
        and context.emotional_consistency.audiovisual_guidance
    ):
        return context.emotional_consistency.audiovisual_guidance[0]
    if (
        context.symbolic_narrative is not None
        and context.symbolic_narrative.audio_progression
    ):
        return context.symbolic_narrative.audio_progression[0]
    return (
        "Use audio as a design cue for pulse, density, and thresholds; do not "
        "claim actual audio generation unless the final artifact implements it."
    )


def _rhythm_role(
    context: _CrossModalityContext,
    *,
    audio_relevant: bool,
) -> str:
    if context.creative_composition is not None:
        base = context.creative_composition.rhythm_plan
        if audio_relevant:
            return f"Coordinate visual rhythm with requested audio pacing: {base}"
        return base
    return "Use rhythm to stagger visual, motion, motif, and emotional beats."


def _camera_role(
    context: _CrossModalityContext,
    *,
    camera_relevant: bool,
) -> str | None:
    if not camera_relevant:
        return None
    if (
        context.creative_composition is not None
        and context.creative_composition.camera_viewpoint_guidance
    ):
        return context.creative_composition.camera_viewpoint_guidance
    return "Use viewpoint shifts sparingly as scene emphasis, not as a separate feature layer."


def _structure_role(context: _CrossModalityContext) -> str:
    if context.generative_structure is not None:
        return (
            f"Bind modalities to {context.generative_structure.blueprint_name} "
            f"using {context.generative_structure.generative_architecture}."
        )
    if context.procedural_structure is not None:
        return (
            f"Use {context.procedural_structure.primary_structure.family} as the "
            "structural carrier for modality changes."
        )
    return "Keep modality changes attached to one explicit procedural structure."


def _motif_role(context: _CrossModalityContext) -> str:
    motifs = _primary_motifs(context)
    if motifs:
        return (
            "Let recurring motifs bridge visual, motion, rhythm, and emotional "
            f"cues: {', '.join(motifs)}."
        )
    return "Use motifs only as recurring design metaphors, not hidden claims."


def _emotion_role(context: _CrossModalityContext) -> str:
    if context.emotional_consistency is not None:
        return (
            f"Use {context.emotional_consistency.primary_emotional_tone} as the "
            "modality modulation target while preserving secondary tone contrast."
        )
    return "Use emotion as a coherence check across modalities, not a factual claim."


def _primary_motifs(context: _CrossModalityContext) -> tuple[SemanticMotifId, ...]:
    if context.semantic_motif is None:
        return ()
    return tuple(motif.motif_id for motif in context.semantic_motif.primary_motifs)


def _modality_hierarchy(
    *,
    primary: CrossModalityChannel,
    supporting: tuple[CrossModalityChannel, ...],
    visual_role: str,
    motion_role: str,
    audio_role: str | None,
    rhythm_role: str,
    camera_role: str | None,
    structure_role: str,
    motif_role: str,
    emotion_role: str,
    context: _CrossModalityContext,
) -> tuple[CrossModalityRole, ...]:
    roles: dict[CrossModalityChannel, str] = {
        "visual_structure": visual_role,
        "motion": motion_role,
        "rhythm": rhythm_role,
        "structure": structure_role,
        "motif": motif_role,
        "emotion": emotion_role,
    }
    if audio_role is not None:
        roles["audio"] = audio_role
    if camera_role is not None:
        roles["camera"] = camera_role
    hierarchy = [
        CrossModalityRole(
            modality=primary,
            role=_clip_text(
                roles.get(primary, "Lead cross-modality coherence."),
                _ROLE_MAX_LENGTH,
            ),
            priority="primary",
            evidence=("Primary modality selected from composer pattern.",),
        )
    ]
    for index, modality in enumerate(supporting):
        hierarchy.append(
            CrossModalityRole(
                modality=modality,
                role=_clip_text(
                    roles.get(modality, "Support the primary modality."),
                    _ROLE_MAX_LENGTH,
                ),
                priority="secondary" if index < 3 else "supporting",
                evidence=_role_evidence(modality, context),
            )
        )
    return tuple(hierarchy[:9])


def _role_evidence(
    modality: CrossModalityChannel,
    context: _CrossModalityContext,
) -> tuple[str, ...]:
    if modality == "audio" and _audio_relevant(context):
        return ("Audio requested or inferred from audiovisual metadata.",)
    if modality == "camera" and _camera_relevant(context):
        return ("Camera/viewpoint requested or inferred from composition metadata.",)
    if modality == "motif" and context.semantic_motif is not None:
        return ("Semantic Motif Engine available.",)
    if modality == "emotion" and context.emotional_consistency is not None:
        return ("Emotional Consistency Engine available.",)
    if modality == "structure" and context.generative_structure is not None:
        return ("Generative Structure Engine available.",)
    return ("Derived from existing planning metadata.",)


def _synchronization_plan(
    context: _CrossModalityContext,
    *,
    pattern: CrossModalityPattern,
    audio_relevant: bool,
    camera_relevant: bool,
) -> tuple[str, ...]:
    plan = [
        f"Use {pattern} as the cross-modality ordering principle.",
        "Synchronize phase changes before adding decorative effects.",
        "Let structure and motif changes create visible cue points for motion.",
    ]
    if audio_relevant:
        plan.append(
            "Map audio pulse or silence to visual density and motion acceleration as prompt guidance only."
        )
    if camera_relevant:
        plan.append(
            "Trigger camera/viewpoint changes only at major narrative or structural thresholds."
        )
    if (
        context.emotional_consistency is not None
        and context.emotional_consistency.emotional_phase_mapping
    ):
        plan.append("Align modality intensity with emotional phase mapping.")
    return tuple(plan[:8])


def _visual_to_audio_mappings(
    context: _CrossModalityContext,
    *,
    audio_relevant: bool,
) -> tuple[CrossModalityMapping, ...]:
    if not audio_relevant:
        return ()
    cues = ["visual density", "brightness", "phase threshold"]
    if context.creative_composition is not None:
        cues.append(context.creative_composition.composition_pattern)
    return (
        CrossModalityMapping(
            source_modality="visual_structure",
            target_modality="audio",
            mapping=(
                "Treat increases in density, brightness, or reassembly clarity "
                "as cues for stronger pulse or brighter tone in prompt guidance."
            ),
            cues=tuple(cues[:8]),
            evidence=("Audio relevance detected.",),
        ),
    )


def _audio_to_motion_mappings(
    context: _CrossModalityContext,
    *,
    audio_relevant: bool,
) -> tuple[CrossModalityMapping, ...]:
    if not audio_relevant:
        return ()
    cues = ["pulse", "drone", "silence", "attack"]
    if context.generative_structure is not None:
        cues.extend(
            hook.signal for hook in context.generative_structure.audiovisual_hooks[:2]
        )
    return (
        CrossModalityMapping(
            source_modality="audio",
            target_modality="motion",
            mapping=(
                "Use pulse, amplitude, or silence as advisory timing for particle "
                "speed, orbit cadence, threshold pause, and reassembly easing."
            ),
            cues=tuple(cues[:8]),
            evidence=("Audio-to-motion mapping remains design guidance.",),
        ),
    )


def _motion_to_structure_mappings(
    context: _CrossModalityContext,
) -> tuple[CrossModalityMapping, ...]:
    module_cues = _module_kind_labels(context)
    evolution = _evolution_phase_labels(context)
    return (
        CrossModalityMapping(
            source_modality="motion",
            target_modality="structure",
            mapping=(
                "Let motion phases expose the procedural structure: growth, "
                "rupture, threshold stillness, and reassembly should correspond "
                "to named modules or procedural families."
            ),
            cues=tuple((*module_cues[:4], *evolution[:4])) or ("phase motion",),
            evidence=("Motion must remain attached to structure.",),
        ),
    )


def _motif_to_modality_mappings(
    context: _CrossModalityContext,
    *,
    audio_relevant: bool,
) -> tuple[CrossModalityMapping, ...]:
    motifs = _primary_motifs(context) or ("center",)
    mappings: list[CrossModalityMapping] = []
    for motif in motifs[:4]:
        target: CrossModalityChannel = (
            "audio" if audio_relevant and motif == "pulse" else "visual_structure"
        )
        if motif in {"fragmentation", "reintegration", "orbit", "wave"}:
            target = "motion"
        mappings.append(
            CrossModalityMapping(
                source_modality="motif",
                target_modality=target,
                mapping=(
                    f"Use motif '{motif}' as a recurring cue across visual form, "
                    "motion timing, rhythm, and emotional emphasis."
                ),
                cues=(motif, target, "recurrence"),
                motif_id=motif,
                evidence=("Semantic motif metadata mapped to modality role.",),
            )
        )
    return tuple(mappings)


def _emotional_to_modality_mappings(
    context: _CrossModalityContext,
    *,
    audio_relevant: bool,
) -> tuple[CrossModalityMapping, ...]:
    tone = (
        context.emotional_consistency.primary_emotional_tone
        if context.emotional_consistency is not None
        else "transformation"
    )
    mappings = [
        CrossModalityMapping(
            source_modality="emotion",
            target_modality="visual_structure",
            mapping=(
                f"Use {tone} to shape color, density, and contrast so visual "
                "structure carries the emotional intent."
            ),
            cues=(tone, "color", "density"),
            emotional_tone=tone,
            evidence=("Emotional consistency mapped to visual structure.",),
        ),
        CrossModalityMapping(
            source_modality="emotion",
            target_modality="motion",
            mapping=(
                f"Use {tone} to pace motion changes; avoid a separate motion mood "
                "that conflicts with the emotional hierarchy."
            ),
            cues=(tone, "motion pacing", "threshold"),
            emotional_tone=tone,
            evidence=("Emotional consistency mapped to motion rhythm.",),
        ),
    ]
    if audio_relevant:
        mappings.append(
            CrossModalityMapping(
                source_modality="emotion",
                target_modality="audio",
                mapping=(
                    f"Use {tone} as a cue for sonic restraint or emphasis in "
                    "prompt guidance without claiming audio generation."
                ),
                cues=(tone, "sonic restraint", "pulse"),
                emotional_tone=tone,
                evidence=("Audio relevance detected.",),
            )
        )
    return tuple(mappings[:8])


def _temporal_cues(
    context: _CrossModalityContext,
    *,
    audio_relevant: bool,
    camera_relevant: bool,
) -> tuple[CrossModalityTemporalCue, ...]:
    phases = (
        context.symbolic_narrative.phases
        if context.symbolic_narrative is not None
        else ()
    )
    modalities: list[CrossModalityChannel] = [
        "visual_structure",
        "motion",
        "rhythm",
        "structure",
    ]
    if audio_relevant:
        modalities.append("audio")
    if camera_relevant:
        modalities.append("camera")
    if not phases:
        return (
            CrossModalityTemporalCue(
                phase="opening",
                cue="Establish primary visual structure before secondary modality movement.",
                modalities=tuple(modalities[:6]),
                timing_guidance="Begin with a readable anchor, then add motion and rhythm in layers.",
                evidence=("No symbolic narrative phases available.",),
            ),
        )
    cues: list[CrossModalityTemporalCue] = []
    for phase in phases[:5]:
        phase_modalities = list(modalities)
        if phase.phase in {"threshold", "climax"}:
            phase_modalities.append("emotion")
        cues.append(
            CrossModalityTemporalCue(
                phase=phase.phase,
                cue=_clip_text(phase.title, _TEMPORAL_CUE_MAX_LENGTH),
                modalities=tuple(dict.fromkeys(phase_modalities[:6])),
                timing_guidance=_clip_text(
                    (
                        f"During {phase.phase}, coordinate visual state, motion "
                        "state, and rhythm before adding extra layers."
                    ),
                    _TEMPORAL_TIMING_MAX_LENGTH,
                ),
                evidence=(phase.symbolic_function,),
            )
        )
    return tuple(cues)


def _contrast_balance_plan(
    context: _CrossModalityContext,
    *,
    pattern: CrossModalityPattern,
    audio_relevant: bool,
    camera_relevant: bool,
) -> tuple[str, ...]:
    plan = [
        "Keep one leading modality per phase; use the others as reinforcement.",
        "Alternate dense visual moments with quieter rhythm or motion spacing.",
    ]
    if pattern == "dense_visual_restrained_audio":
        plan.append("Restrain sonic cues when visual density is high.")
    elif audio_relevant:
        plan.append(
            "Balance audio pulse against visual density so neither overwhelms the other."
        )
    if camera_relevant:
        plan.append(
            "Do not combine rapid camera movement with dense particle motion unless explicitly requested."
        )
    if context.emotional_consistency is not None:
        plan.append(
            "Use emotional tone hierarchy to decide which modality should soften first."
        )
    return tuple(plan[:8])


def _modality_conflicts(
    context: _CrossModalityContext,
    *,
    audio_relevant: bool,
    camera_relevant: bool,
) -> tuple[str, ...]:
    conflicts: list[str] = []
    if audio_relevant and not _has_audio_structure(context):
        conflicts.append(
            "Audio is requested or inferred, but current metadata only supports audio as design guidance."
        )
    if (
        camera_relevant
        and context.creative_plan is not None
        and context.creative_plan.output_modality.value == "audio"
    ):
        conflicts.append(
            "Camera/viewpoint cues conflict with an audio-led output goal."
        )
    if (context.request_tokens & _DENSE_TOKENS) and (
        context.request_tokens & _LOUD_AUDIO_TOKENS
    ):
        conflicts.append(
            "Dense visuals and loud/intense audio may compete for attention."
        )
    if "playful" in context.request_tokens and _tone(context) in {
        "dread",
        "ritual solemnity",
    }:
        conflicts.append(
            "Playful cues may conflict with solemn or dark emotional direction."
        )
    return tuple(conflicts[:8])


def _has_audio_structure(context: _CrossModalityContext) -> bool:
    if context.generative_structure is not None:
        if context.generative_structure.audiovisual_hooks:
            return True
        if any(
            module.kind == "audio_reactive_modulation_layer"
            for module in context.generative_structure.procedural_modules
        ):
            return True
    return bool(
        context.procedural_structure is not None
        and context.procedural_structure.audiovisual_structure_plan
    )


def _overload_risks(
    context: _CrossModalityContext,
    *,
    audio_relevant: bool,
    camera_relevant: bool,
) -> tuple[str, ...]:
    risks: list[str] = []
    if context.request_tokens & _DENSE_TOKENS:
        risks.append(
            "Dense visual systems can overload motion, rhythm, and motif readability."
        )
    if audio_relevant and context.request_tokens & _LOUD_AUDIO_TOKENS:
        risks.append(
            "Intense audio cues can overpower emotional pacing and visual hierarchy."
        )
    if camera_relevant and (context.request_tokens & _DENSE_TOKENS):
        risks.append(
            "Camera movement plus dense particle motion can reduce legibility."
        )
    if _module_kinds(context) & {
        "particle_emitter",
        "noise_modulation_layer",
        "wave_oscillator",
    }:
        risks.append(
            "Multiple modulation layers should expose controls and rests to avoid constant activity."
        )
    return tuple(risks[:8])


def _underuse_risks(
    context: _CrossModalityContext,
    *,
    audio_relevant: bool,
    camera_relevant: bool,
) -> tuple[str, ...]:
    risks: list[str] = []
    if not audio_relevant and "rhythm" in context.request_tokens:
        risks.append("Rhythm is requested but no explicit audio modality is active.")
    if audio_relevant and not _has_audio_structure(context):
        risks.append(
            "Audio may remain shallow unless mapped to motion or density as design guidance."
        )
    if camera_relevant and not (
        context.creative_composition is not None
        and context.creative_composition.camera_viewpoint_guidance
    ):
        risks.append(
            "Camera/viewpoint may remain generic without explicit phase triggers."
        )
    if context.semantic_motif is None:
        risks.append(
            "Motif coherence may be underused if no recurring motif metadata is available."
        )
    if context.request_tokens & _AMBIGUOUS_MODALITY_TOKENS:
        risks.append(
            "Broad multimodal phrasing can underuse one modality unless the lead/support hierarchy is explicit."
        )
    return tuple(risks[:8])


def _unresolved_gaps(
    context: _CrossModalityContext,
    *,
    audio_relevant: bool,
    camera_relevant: bool,
) -> tuple[str, ...]:
    gaps: list[str] = []
    if context.request_tokens & _AMBIGUOUS_MODALITY_TOKENS:
        gaps.append(
            "Multimodal intent is broad; confirm which modality should lead if precision matters."
        )
    if audio_relevant and not _has_audio_structure(context):
        gaps.append(
            "Audio is relevant, but implementation-level audio behavior remains unspecified."
        )
    if camera_relevant and (
        context.creative_composition is None
        or context.creative_composition.camera_viewpoint_guidance is None
    ):
        gaps.append(
            "Camera/viewpoint is relevant, but camera behavior is not specified."
        )
    if context.emotional_consistency is None:
        gaps.append(
            "Emotional-to-modality mapping lacks an Emotional Consistency profile."
        )
    return tuple(gaps[:8])


def _fallback_strategy(
    *,
    pattern: CrossModalityPattern,
    primary: CrossModalityChannel,
    supporting: tuple[CrossModalityChannel, ...],
    audio_relevant: bool,
    camera_relevant: bool,
) -> CrossModalityFallbackStrategy:
    reduced: list[CrossModalityChannel] = []
    if audio_relevant:
        reduced.append("audio")
    if camera_relevant:
        reduced.append("camera")
    preserved = tuple(
        dict.fromkeys(
            [
                primary,
                "visual_structure",
                "motion",
                "rhythm",
                *(
                    item
                    for item in supporting
                    if item in {"structure", "motif", "emotion"}
                ),
            ]
        )
    )[:6]
    return CrossModalityFallbackStrategy(
        fallback_pattern=(
            "visual_led_composition"
            if pattern
            not in {"visual_led_composition", "structure_led_procedural_evolution"}
            else "structure_led_procedural_evolution"
        ),
        preserved_modalities=preserved,
        reduced_modalities=tuple(reduced[:6]),
        simplification_strategy=(
            "Preserve visual structure, motion timing, motif recurrence, and "
            "emotional hierarchy; reduce audio/camera to optional prompt cues "
            "when runtime scope or clarity tightens."
        ),
        prompt_guidance=(
            "When multimodal scope is too broad, keep the visual-motion structure first.",
            "Treat audio and camera cues as optional guidance unless explicitly implemented.",
            "Do not add runtime or provider behavior to satisfy modality ambition.",
        ),
    )


def _hitl_questions(
    unresolved: tuple[str, ...],
    conflicts: tuple[str, ...],
    overload: tuple[str, ...],
    *,
    audio_relevant: bool,
    camera_relevant: bool,
) -> tuple[str, ...]:
    questions: list[str] = []
    if unresolved or conflicts:
        questions.append(
            "Which modality should lead if visual, motion, audio, and emotion compete?"
        )
    if audio_relevant:
        questions.append(
            "Should audio drive motion directly, or only support rhythm and mood?"
        )
    if camera_relevant:
        questions.append(
            "Should camera/viewpoint shifts be active, or should the composition stay stable?"
        )
    if overload:
        questions.append(
            "Should the design prioritize density or readability when modalities become crowded?"
        )
    return tuple(questions[:6])


def _prompt_guidance(
    *,
    pattern: CrossModalityPattern,
    primary: CrossModalityChannel,
    audio_relevant: bool,
    camera_relevant: bool,
) -> tuple[str, ...]:
    guidance = [
        f"Use {pattern} with {primary} as the leading modality.",
        "Treat cross-modality mappings as design guidance, not claims of generated audio or rendered output.",
        "Coordinate modality changes through named phases, modules, motifs, and emotional cues.",
        "Preserve readable visual-motion structure before adding secondary modality density.",
    ]
    if audio_relevant:
        guidance.append(
            "If code is generated, implement only supported audio behavior explicitly requested by the user."
        )
    if camera_relevant:
        guidance.append(
            "Keep camera/viewpoint guidance bounded to composition planning unless the target runtime supports it."
        )
    guidance.append(
        "Do not auto-select runtimes, route providers, repair runtime behavior, or change preview behavior."
    )
    return tuple(guidance[:8])


def _clip_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def _clip_optional_text(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    return _clip_text(value, limit)


def _evidence(
    context: _CrossModalityContext,
    *,
    pattern: CrossModalityPattern,
    primary: CrossModalityChannel,
    supporting: tuple[CrossModalityChannel, ...],
    audio_relevant: bool,
    camera_relevant: bool,
) -> tuple[str, ...]:
    evidence = [
        f"Pattern: {pattern}.",
        f"Primary modality: {primary}.",
        "Supporting modalities: " + ", ".join(supporting) + ".",
        f"Audio relevance: {audio_relevant}.",
        f"Camera relevance: {camera_relevant}.",
    ]
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
            f"Generative modules: {', '.join(_module_kind_labels(context)[:5])}."
        )
    if context.semantic_motif is not None:
        evidence.append(f"Motifs: {', '.join(_primary_motifs(context))}.")
    if context.emotional_consistency is not None:
        evidence.append(
            f"Emotion: {context.emotional_consistency.primary_emotional_tone}."
        )
    return tuple(evidence[:14])


def _module_kind_labels(context: _CrossModalityContext) -> tuple[str, ...]:
    return (
        tuple(module.kind for module in context.generative_structure.procedural_modules)
        if context.generative_structure is not None
        else ()
    )


def _evolution_phase_labels(context: _CrossModalityContext) -> tuple[str, ...]:
    return (
        tuple(rule.phase for rule in context.generative_structure.evolution_rules)
        if context.generative_structure is not None
        else ()
    )


def _tone(context: _CrossModalityContext) -> EmotionalTone | None:
    if context.emotional_consistency is None:
        return None
    return context.emotional_consistency.primary_emotional_tone


__all__ = [
    "CROSS_MODALITY_AUTHORITY_BOUNDARY",
    "CrossModalityChannel",
    "CrossModalityCompositionProfile",
    "CrossModalityFallbackStrategy",
    "CrossModalityMapping",
    "CrossModalityPattern",
    "CrossModalityRole",
    "CrossModalityTemporalCue",
    "cross_modality_prompt_lines",
    "derive_cross_modality_composition_profile",
]
