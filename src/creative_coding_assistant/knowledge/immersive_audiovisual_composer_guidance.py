"""Guidance builders for the V8.6 immersive audiovisual composer."""

from __future__ import annotations

from collections.abc import Sequence

from creative_coding_assistant.knowledge.immersive_audiovisual_composer_catalog import (
    PREVIEW_AUDIT_ROWS,
    UNSUPPORTED_COMPOSER_CLAIM_TOKENS,
)
from creative_coding_assistant.knowledge.immersive_audiovisual_composer_contracts import (
    ImmersiveArtisticDecision,
    ImmersiveAudienceJourneyPlan,
    ImmersiveComposerProvenance,
    ImmersiveComposerValidationFinding,
    ImmersiveComposerValidationSeverity,
    ImmersiveCompositionLayer,
    ImmersiveGeometryAnimationPlan,
    ImmersivePreviewImplementationStatus,
    ImmersivePreviewRuntimeAuditItem,
    ImmersiveSceneNode,
    ImmersiveSceneTransition,
    ImmersiveSpatialAudioPlan,
    ImmersiveVisualLanguagePlan,
)
from creative_coding_assistant.knowledge.mythopoetic_narrative_contracts import (
    MythopoeticNarrativeReport,
    MythopoeticNarrativeScene,
)
from creative_coding_assistant.knowledge.sacred_architecture_contracts import (
    SacredArchitectureReport,
)
from creative_coding_assistant.knowledge.sacred_geometry_contracts import SacredGeometryReport
from creative_coding_assistant.knowledge.symbolic_translation import SymbolicTranslationReport
from creative_coding_assistant.orchestration.audio_reactive import AudioReactiveGuidance
from creative_coding_assistant.orchestration.audio_visual_scene import AudioVisualSceneProfile
from creative_coding_assistant.orchestration.creative_translation import CreativeTranslation
from creative_coding_assistant.orchestration.shader_presets import ShaderPresetGuidance
from creative_coding_assistant.orchestration.style_profiles import StyleProfilePlan
from creative_coding_assistant.orchestration.visual_styles import VisualStyleGuidance


def build_immersive_scene_graph(
    *,
    source_query: str,
    creative_translation: CreativeTranslation,
    symbolic_translation: SymbolicTranslationReport,
    sacred_geometry: SacredGeometryReport,
    sacred_architecture: SacredArchitectureReport,
    mythopoetic_narrative: MythopoeticNarrativeReport,
    audio_visual_scene: AudioVisualSceneProfile,
) -> tuple[ImmersiveSceneNode, ...]:
    """Compose scene graph nodes from existing symbolic, geometry, architecture, and narrative outputs."""

    scenes = mythopoetic_narrative.scene_sequence[:5]
    geometry_patterns = sacred_geometry.pattern_guidance or ()
    architecture_patterns = sacred_architecture.pattern_guidance or ()
    symbolic_motifs = symbolic_translation.motif_mappings or ()
    audio_phases = audio_visual_scene.scene_phases
    preview_targets = _preview_targets(creative_translation)

    nodes: list[ImmersiveSceneNode] = []
    for index, scene in enumerate(scenes):
        geometry = geometry_patterns[index % len(geometry_patterns)] if geometry_patterns else None
        architecture = (
            architecture_patterns[index % len(architecture_patterns)] if architecture_patterns else None
        )
        motif = symbolic_motifs[index % len(symbolic_motifs)] if symbolic_motifs else None
        audio_phase = audio_phases[index % len(audio_phases)]
        node_id = f"scene::{scene.phase}"
        nodes.append(
            ImmersiveSceneNode(
                node_id=node_id,
                label=scene.title,
                layer_sequence=index,
                source_layers=(
                    ImmersiveCompositionLayer.NARRATIVE,
                    ImmersiveCompositionLayer.GEOMETRIC,
                    ImmersiveCompositionLayer.ARCHITECTURAL,
                    ImmersiveCompositionLayer.SYMBOLIC,
                    ImmersiveCompositionLayer.AUDIO,
                    ImmersiveCompositionLayer.AUDIENCE,
                    ImmersiveCompositionLayer.PREVIEW,
                ),
                source_surface_ids=_node_surface_ids(
                    scene=scene,
                    geometry_id=geometry.pattern_id if geometry else None,
                    architecture_id=architecture.pattern_id if architecture else None,
                    motif_id=motif.motif_id if motif else None,
                ),
                visual_language=_visual_language(scene, creative_translation, motif),
                sacred_lighting=_sacred_lighting(scene, geometry, architecture),
                symbolic_color=_symbolic_color(creative_translation, motif),
                geometry_driver=_geometry_driver(scene, geometry),
                animation_plan=_animation_plan(scene, geometry, audio_phase),
                particle_symbolism=_particle_symbolism(source_query, scene, motif),
                spatial_audio_role=_spatial_audio_role(scene, audio_phase),
                dramaturgical_function=scene.narrative_function,
                audience_function=_audience_function(scene, architecture),
                preview_targets=preview_targets,
                evidence=_node_evidence(scene, geometry, architecture, motif),
            )
        )

    return tuple(nodes)


def build_immersive_scene_transitions(
    *,
    scene_graph: Sequence[ImmersiveSceneNode],
    mythopoetic_narrative: MythopoeticNarrativeReport,
    audio_visual_scene: AudioVisualSceneProfile,
) -> tuple[ImmersiveSceneTransition, ...]:
    """Compose transitions between scene graph nodes."""

    transitions: list[ImmersiveSceneTransition] = []
    scene_transitions = audio_visual_scene.transition_plan
    narrative_scenes = mythopoetic_narrative.scene_sequence
    for index, (source, target) in enumerate(zip(scene_graph, scene_graph[1:], strict=False)):
        av_transition = scene_transitions[index % len(scene_transitions)]
        narrative_scene = narrative_scenes[min(index, len(narrative_scenes) - 1)]
        transitions.append(
            ImmersiveSceneTransition(
                transition_id=f"transition::{source.node_id}::{target.node_id}",
                from_node_id=source.node_id,
                to_node_id=target.node_id,
                transition_composer=av_transition.transition,
                ritual_timing=_ritual_timing(av_transition.audio_rhythm_guidance),
                temporal_dramaturgy=narrative_scene.transition_out,
                continuity_guidance=av_transition.continuity_guidance,
                evidence=(
                    f"V8.5 transition: {narrative_scene.transition_out}",
                    f"V3 audio-visual transition: {av_transition.transition}",
                ),
            )
        )
    return tuple(transitions)


def build_visual_language_plan(
    *,
    creative_translation: CreativeTranslation,
    symbolic_translation: SymbolicTranslationReport,
    sacred_geometry: SacredGeometryReport,
    sacred_architecture: SacredArchitectureReport,
    style_profiles: StyleProfilePlan,
    visual_style: VisualStyleGuidance | None,
    shader_presets: ShaderPresetGuidance | None,
) -> ImmersiveVisualLanguagePlan:
    """Build composed visual language, color, and lighting guidance."""

    visual_identity = _merge_text(
        tuple(style.value for style in visual_style.styles) if visual_style else (),
        creative_translation.mood_atmosphere,
        ("immersive symbolic scene language",),
        limit=10,
    )
    sacred_lighting = _merge_text(
        shader_presets.light_material_behavior if shader_presets else (),
        _geometry_light_guidance(sacred_geometry),
        _architecture_light_guidance(sacred_architecture),
        ("Use bounded luminous hierarchy without spiritual authority claims.",),
        limit=10,
    )
    symbolic_color = _merge_text(
        creative_translation.color_material_direction,
        tuple(
            guidance
            for motif in symbolic_translation.motif_mappings
            for guidance in motif.visual_guidance[:1]
        ),
        ("Tie color changes to visible symbolic and scene-state cues.",),
        limit=10,
    )
    shader_material = _merge_text(
        shader_presets.shader_structure if shader_presets else (),
        shader_presets.performance_constraints if shader_presets else (),
        ("Keep materials preview-safe and explainable in browser-internal runtimes.",),
        limit=10,
    )
    runtime_suitability = _merge_text(
        visual_style.runtime_suitability if visual_style else (),
        shader_presets.runtime_suitability if shader_presets else (),
        creative_translation.runtime_recommendations,
        ("Prefer browser-internal preview-compatible runtimes.",),
        limit=10,
    )

    return ImmersiveVisualLanguagePlan(
        visual_identity=visual_identity,
        sacred_lighting_guidance=sacred_lighting,
        symbolic_color_guidance=symbolic_color,
        shader_or_material_guidance=shader_material,
        style_profile_refs=style_profiles.profile_ids,
        runtime_suitability=runtime_suitability,
        boundary=(
            "Visual language is composed as generation guidance only; it does not "
            "apply style profiles, mutate artifacts, or execute preview rendering."
        ),
    )


def build_geometry_animation_plan(
    *,
    source_query: str,
    sacred_geometry: SacredGeometryReport,
    mythopoetic_narrative: MythopoeticNarrativeReport,
) -> ImmersiveGeometryAnimationPlan:
    """Build geometry animation, particle, quadrivium, and planetary guidance."""

    geometry_motion = _merge_text(
        tuple(
            item
            for pattern in sacred_geometry.pattern_guidance
            for item in pattern.motion_mappings[:2]
        ),
        tuple(
            _first(scene.motion_guidance, "")
            for scene in mythopoetic_narrative.scene_sequence[:3]
            if _first(scene.motion_guidance, "")
        ),
        ("Animate geometry through bounded phase changes tied to scene beats.",),
        limit=12,
    )
    harmonic_parameters = _merge_text(
        tuple(
            item
            for pattern in sacred_geometry.pattern_guidance
            for item in pattern.mathematical_parameters[:2]
        ),
        ("Expose ratio, radius, phase, and repetition as inspectable parameters.",),
        limit=12,
    )
    particle_symbolism = _merge_text(
        _particle_tokens(source_query),
        tuple(scene.symbolic_focus for scene in mythopoetic_narrative.scene_sequence[:3]),
        ("Use particles as symbolic traces, not as uncontrolled visual noise.",),
        limit=10,
    )

    return ImmersiveGeometryAnimationPlan(
        geometry_animation_guidance=geometry_motion,
        particle_symbolism_guidance=particle_symbolism,
        quadrivium_mapping=(
            "Number: keep ratios and counts explicit in scene parameters.",
            "Geometry: let V8.3 patterns drive form, symmetry, and spatial hierarchy.",
            "Music: map harmonic intervals and rhythm divisions to visual phases.",
            "Astronomy: use orbital cycles as metaphorical timing, not physical ephemerides.",
        ),
        planetary_motion_guidance=(
            "Use orbital, epicycle, or constellation-like paths when the request asks for planetary motion.",
            "Keep planetary motion stylized and deterministic inside browser preview constraints.",
            "Avoid claims of accurate astronomical simulation unless a dedicated ephemeris engine is added.",
        ),
        harmonic_parameters=harmonic_parameters,
        validation_notes=(
            "Quadrivium and planetary guidance are creative correspondences only.",
            "Geometry animation remains bounded guidance and does not execute preview runtime.",
        ),
    )


def build_spatial_audio_plan(
    *,
    creative_translation: CreativeTranslation,
    sacred_geometry: SacredGeometryReport,
    mythopoetic_narrative: MythopoeticNarrativeReport,
    audio_visual_scene: AudioVisualSceneProfile,
    audio_reactive: AudioReactiveGuidance | None,
) -> ImmersiveSpatialAudioPlan:
    """Build explicit-activation spatial audio and sacred music mapping guidance."""

    audio_runtime_candidates = _audio_runtime_candidates(creative_translation, audio_reactive)
    audio_guidance = _merge_text(
        tuple(
            _first(scene.audio_guidance, "")
            for scene in mythopoetic_narrative.scene_sequence[:4]
            if _first(scene.audio_guidance, "")
        ),
        audio_visual_scene.audio_timing_plan,
        ("Pan or layer audio cues according to audience path and scene focus.",),
        limit=10,
    )
    sacred_music = _merge_text(
        tuple(
            item
            for pattern in sacred_geometry.pattern_guidance
            for item in pattern.audio_harmonic_mappings[:2]
        ),
        creative_translation.musical_references,
        ("Map rhythm and drone changes to scene thresholds without ritual efficacy claims.",),
        limit=10,
    )
    sync = _merge_text(
        audio_visual_scene.synchronization_checkpoints,
        tuple(mapping.behavior for mapping in audio_reactive.mappings) if audio_reactive else (),
        ("Keep audio-reactive changes opt-in and inspectable.",),
        limit=10,
    )

    return ImmersiveSpatialAudioPlan(
        audio_runtime_candidates=audio_runtime_candidates,
        spatial_audio_guidance=audio_guidance,
        sacred_music_mapping=sacred_music,
        audiovisual_sync=sync,
        safety_constraints=(
            "Audio remains silent until explicit user activation.",
            "Do not autoplay, infer listener state, or claim therapeutic or ritual outcomes.",
            "Provide a visual-only fallback for demo reliability.",
        ),
    )


def build_audience_journey_plan(
    *,
    sacred_architecture: SacredArchitectureReport,
    mythopoetic_narrative: MythopoeticNarrativeReport,
    audio_visual_scene: AudioVisualSceneProfile,
) -> ImmersiveAudienceJourneyPlan:
    """Build installation, embodied experience, dramaturgy, and audience-flow guidance."""

    semantic_nodes = sacred_architecture.semantic_nodes
    scenes = mythopoetic_narrative.scene_sequence
    phases = audio_visual_scene.scene_phases
    return ImmersiveAudienceJourneyPlan(
        installation_flow=_merge_text(
            tuple(node.guidance for node in semantic_nodes[:4]),
            sacred_architecture.pattern_guidance[0].installation_guidance
            if sacred_architecture.pattern_guidance
            else (),
            ("Start with an entry threshold, reveal the focal field, and close with a return path.",),
            limit=10,
        ),
        audience_journey=_merge_text(
            tuple(scene.narrative_function for scene in scenes[:5]),
            ("Audience attention moves from orientation to threshold, climax, and integration.",),
            limit=10,
        ),
        embodied_experience=_merge_text(
            tuple(phase.emotional_state for phase in phases[:5]),
            ("Use scale, distance, rhythm, and light to imply embodied pacing.",),
            limit=10,
        ),
        spatial_dramaturgy=_merge_text(
            tuple(node.guidance for node in semantic_nodes[:5]),
            ("Spatial drama is built from entry, center, boundary, and sightline relationships.",),
            limit=10,
        ),
        temporal_dramaturgy=_merge_text(
            audio_visual_scene.visual_timing_plan,
            tuple(scene.transition_out for scene in scenes[:4]),
            ("Temporal drama follows bounded scene phases instead of autonomous workflow control.",),
            limit=10,
        ),
        emotional_resonance=_merge_text(
            tuple(scene.emotional_state for scene in scenes[:5]),
            mythopoetic_narrative.audience_communication,
            ("Treat emotional resonance as intended artistic effect, not measurement.",),
            limit=10,
        ),
        audience_flow_simulation=(
            "Qualitative flow: entry and threshold nodes should have the clearest affordances.",
            "Attention load: keep the climax dense but reserve visual rest zones before resolution.",
            "Fallback flow: collapse to opening, threshold, climax, and return when demo time is limited.",
        ),
        boundary=(
            "Audience flow simulation is a qualitative creative planning model only; "
            "it does not measure audiences, track users, or guarantee psychological outcomes."
        ),
    )


def build_artistic_decisions(
    *,
    scene_graph: Sequence[ImmersiveSceneNode],
    visual_language: ImmersiveVisualLanguagePlan,
    geometry_animation: ImmersiveGeometryAnimationPlan,
    spatial_audio: ImmersiveSpatialAudioPlan,
    audience_journey: ImmersiveAudienceJourneyPlan,
) -> tuple[ImmersiveArtisticDecision, ...]:
    """Build explainable decisions that cite reused source surfaces."""

    first_scene = scene_graph[0]
    last_scene = scene_graph[-1]
    return (
        ImmersiveArtisticDecision(
            decision_id="decision::scene_graph",
            decision=f"Use {len(scene_graph)} composed scene nodes from {first_scene.label} to {last_scene.label}.",
            rationale=(
                "V8.5 scene sequencing gives dramaturgical order while V8.3 and V8.4 "
                "supply geometry, lighting, topology, and audience path constraints."
            ),
            reused_surface_ids=(
                "v8_3_sacred_geometry_engine",
                "v8_4_sacred_architecture_engine",
                "v8_5_mythopoetic_engine",
            ),
            affected_layers=(
                ImmersiveCompositionLayer.NARRATIVE,
                ImmersiveCompositionLayer.GEOMETRIC,
                ImmersiveCompositionLayer.ARCHITECTURAL,
            ),
            evidence=(first_scene.dramaturgical_function, last_scene.dramaturgical_function),
        ),
        ImmersiveArtisticDecision(
            decision_id="decision::visual_language",
            decision="Bind sacred lighting, symbolic color, and style identity to the same scene states.",
            rationale=(
                "V8.2 symbolic motifs, V8.3 light mappings, and existing visual/style metadata "
                "are aligned so the visual language remains inspectable."
            ),
            reused_surface_ids=(
                "v8_2_symbolic_translation_engine",
                "v8_3_sacred_geometry_engine",
                "v6_style_profiles",
            ),
            affected_layers=(ImmersiveCompositionLayer.SYMBOLIC, ImmersiveCompositionLayer.VISUAL),
            evidence=(
                visual_language.visual_identity[0],
                visual_language.sacred_lighting_guidance[0],
            ),
        ),
        ImmersiveArtisticDecision(
            decision_id="decision::geometry_motion",
            decision="Use ratio-driven motion, particle traces, quadrivium mapping, and orbital timing.",
            rationale=(
                "V8.3 already owns geometry and harmonic mappings; V8.6 composes them into "
                "scene-level animation instead of duplicating geometry engines."
            ),
            reused_surface_ids=("v8_3_sacred_geometry_engine", "v8_5_mythopoetic_engine"),
            affected_layers=(ImmersiveCompositionLayer.GEOMETRIC, ImmersiveCompositionLayer.NARRATIVE),
            evidence=(
                geometry_animation.geometry_animation_guidance[0],
                geometry_animation.quadrivium_mapping[0],
            ),
        ),
        ImmersiveArtisticDecision(
            decision_id="decision::audio_audience",
            decision="Keep spatial audio opt-in while using audio cues to clarify audience journey beats.",
            rationale=(
                "Existing audio-reactive guidance and audio-visual scene timing support audiovisual "
                "composition without autoplay, playback execution, or workflow control."
            ),
            reused_surface_ids=("v3_audio_visual_scene", "v3_creative_translation"),
            affected_layers=(ImmersiveCompositionLayer.AUDIO, ImmersiveCompositionLayer.AUDIENCE),
            evidence=(
                spatial_audio.spatial_audio_guidance[0],
                audience_journey.audience_flow_simulation[0],
            ),
        ),
    )


def build_preview_runtime_audit() -> tuple[ImmersivePreviewRuntimeAuditItem, ...]:
    """Return the required V8.6 preview audit classifications."""

    return tuple(
        ImmersivePreviewRuntimeAuditItem(
            item=item,
            implementation_status=ImmersivePreviewImplementationStatus(status),
            reusable_for_v8_6=reusable,
            existing_behavior=existing_behavior,
            v8_6_action=v8_6_action,
            evidence_files=evidence_files,
        )
        for item, (status, reusable, existing_behavior, v8_6_action, evidence_files) in PREVIEW_AUDIT_ROWS.items()
    )


def build_composer_validation_findings(
    unsupported_claim_risks: Sequence[str],
) -> tuple[ImmersiveComposerValidationFinding, ...]:
    """Build deterministic validation findings for V8.6 report boundaries."""

    findings = [
        ImmersiveComposerValidationFinding(
            finding_id="validation::bounded_composition",
            severity=ImmersiveComposerValidationSeverity.INFO,
            summary="V8.6 emits typed composition guidance without executing artifacts or previews.",
            action="Use the report as planning and prompt context, not as proof of rendered output.",
        )
    ]
    if unsupported_claim_risks:
        findings.append(
            ImmersiveComposerValidationFinding(
                finding_id="validation::unsupported_runtime_claims",
                severity=ImmersiveComposerValidationSeverity.HITL_REQUIRED,
                summary="The request includes external, later-V8, or unsupported runtime claims.",
                action="Keep those claims deferred or ask HITL before expanding scope.",
            )
        )
    return tuple(findings)


def build_composer_provenance(
    *,
    source_query: str,
    style_profiles: StyleProfilePlan,
    preview_audit: Sequence[ImmersivePreviewRuntimeAuditItem],
    unsupported_claim_risks: Sequence[str],
) -> tuple[ImmersiveComposerProvenance, ...]:
    """Build top-level provenance records for V8.6 composition."""

    provenance = [
        ImmersiveComposerProvenance(
            provenance_id="v8_6::request",
            kind="request_signal",
            reference="source_query",
            summary=f"Request-visible composition intent: {_clip(source_query, 180)}",
            confidence_signal=0.72,
        ),
        ImmersiveComposerProvenance(
            provenance_id="v8_6::creative_translation",
            kind="v3_creative_translation",
            reference="v3_creative_translation",
            summary="Creative translation supplies modality, mood, movement, color, and runtime cues.",
            confidence_signal=0.78,
        ),
        ImmersiveComposerProvenance(
            provenance_id="v8_6::audio_visual_scene",
            kind="v3_audio_visual_scene",
            reference="audio_visual_scene_system",
            summary="Existing audio-visual scene metadata supplies phases, timing, cues, and fallbacks.",
            confidence_signal=0.76,
        ),
        ImmersiveComposerProvenance(
            provenance_id="v8_6::style_profiles",
            kind="v6_style_profiles",
            reference=", ".join(style_profiles.profile_ids[:3]),
            summary="Existing style profiles are reused as advisory style references only.",
            confidence_signal=0.68,
        ),
        ImmersiveComposerProvenance(
            provenance_id="v8_6::knowledge",
            kind="v8_1_creative_knowledge",
            reference="v8_1_creative_knowledge_distillation",
            summary="V8.1 contributes provenance, confidence, and audiovisual KB records.",
            confidence_signal=0.74,
        ),
        ImmersiveComposerProvenance(
            provenance_id="v8_6::symbolic",
            kind="v8_2_symbolic_translation",
            reference="v8_2_symbolic_translation_engine",
            summary="V8.2 contributes motifs, symbolic operations, color/meaning boundaries, and explainability.",
            confidence_signal=0.76,
        ),
        ImmersiveComposerProvenance(
            provenance_id="v8_6::geometry",
            kind="v8_3_sacred_geometry",
            reference="v8_3_sacred_geometry_engine",
            summary="V8.3 contributes geometry, harmonic, light, motion, and algorithmic guidance.",
            confidence_signal=0.78,
        ),
        ImmersiveComposerProvenance(
            provenance_id="v8_6::architecture",
            kind="v8_4_sacred_architecture",
            reference="v8_4_sacred_architecture_engine",
            summary="V8.4 contributes spatial topology, installation, threshold, and audience-flow guidance.",
            confidence_signal=0.78,
        ),
        ImmersiveComposerProvenance(
            provenance_id="v8_6::narrative",
            kind="v8_5_mythopoetic_narrative",
            reference="v8_5_mythopoetic_engine",
            summary="V8.5 contributes scene sequence, dramaturgy, emotional arc, and audience communication.",
            confidence_signal=0.8,
        ),
        ImmersiveComposerProvenance(
            provenance_id="v8_6::preview_audit",
            kind="preview_runtime_audit",
            reference=", ".join(item.item for item in preview_audit[:3]),
            summary="Preview audit confirms existing browser preview foundations are reused instead of duplicated.",
            confidence_signal=0.82,
        ),
    ]
    if unsupported_claim_risks:
        provenance.append(
            ImmersiveComposerProvenance(
                provenance_id="v8_6::safety_boundary",
                kind="safety_boundary",
                reference=_clip(", ".join(unsupported_claim_risks), 280),
                summary="Unsupported claims are preserved as boundary risks, not implemented behavior.",
                confidence_signal=0.52,
            )
        )
    return tuple(provenance)


def composer_unsupported_claim_risks(query: str) -> tuple[str, ...]:
    normalized = " ".join(query.lower().split())
    return tuple(
        f"Unsupported or deferred V8.6 request token: {token}."
        for token in sorted(UNSUPPORTED_COMPOSER_CLAIM_TOKENS)
        if token in normalized
    )


def composer_hitl_questions(unsupported_claim_risks: Sequence[str]) -> tuple[str, ...]:
    if not unsupported_claim_risks:
        return ()
    return (
        "Should deferred external integration or later-V8 behavior remain out of this V8.6 report?",
    )


def composition_audit_summary() -> tuple[str, ...]:
    return (
        "V8.1 contributes KB provenance, confidence, audiovisual reference records, and bounded knowledge evidence.",
        "V8.2 contributes symbolic motifs, symbolic-to-visual/audio operations, color language, and explainability.",
        "V8.3 contributes geometry patterns, harmonic ratios, light/audio mappings, and animation parameters.",
        "V8.4 contributes spatial topology, architectural thresholds, installation flow, and audience path guidance.",
        "V8.5 contributes narrative scene sequence, ritual timing, emotional arcs, "
        "temporal/spatial dramaturgy, and audience communication.",
    )


def _node_surface_ids(
    *,
    scene: MythopoeticNarrativeScene,
    geometry_id: str | None,
    architecture_id: str | None,
    motif_id: str | None,
) -> tuple[str, ...]:
    return _dedupe(
        (
            "v8_5_mythopoetic_engine",
            scene.scene_id,
            "v8_3_sacred_geometry_engine" if geometry_id else "",
            geometry_id or "",
            "v8_4_sacred_architecture_engine" if architecture_id else "",
            architecture_id or "",
            "v8_2_symbolic_translation_engine" if motif_id else "",
            motif_id or "",
            "v3_audio_visual_scene",
        )
    )


def _visual_language(
    scene: MythopoeticNarrativeScene,
    creative_translation: CreativeTranslation,
    motif: object | None,
) -> str:
    cue = scene.visual_guidance[0]
    mood = creative_translation.mood_atmosphere[0] if creative_translation.mood_atmosphere else "immersive"
    motif_label = getattr(motif, "motif_label", "symbolic focus")
    return f"{cue} Anchor the {mood} visual language around {motif_label}."


def _sacred_lighting(
    scene: MythopoeticNarrativeScene,
    geometry: object | None,
    architecture: object | None,
) -> str:
    geometry_light = _first(getattr(geometry, "color_light_mappings", ()), "")
    architecture_axis = _first(getattr(architecture, "axis_guidance", ()), "")
    if geometry_light and architecture_axis:
        return f"{geometry_light} Use architectural axis support: {architecture_axis}"
    return geometry_light or architecture_axis or f"Use luminous contrast to reveal {scene.symbolic_focus}."


def _symbolic_color(creative_translation: CreativeTranslation, motif: object | None) -> str:
    color = _first(creative_translation.color_material_direction, "controlled accent color")
    motif_guidance = _first(getattr(motif, "visual_guidance", ()), "visible symbolic cue")
    return f"Map {color} toward {motif_guidance} without claiming authoritative symbolism."


def _geometry_driver(scene: MythopoeticNarrativeScene, geometry: object | None) -> str:
    structure = _first(getattr(geometry, "structure_guidance", ()), "")
    parameters = ", ".join(getattr(geometry, "mathematical_parameters", ())[:2])
    if structure:
        return f"{structure} Parameter emphasis: {parameters or 'phase and repetition'}."
    return f"Use geometry as a visible driver for {scene.symbolic_focus}."


def _animation_plan(
    scene: MythopoeticNarrativeScene,
    geometry: object | None,
    audio_phase: object,
) -> str:
    motion = _first(getattr(geometry, "motion_mappings", ()), _first(scene.motion_guidance, "scene motion"))
    return f"{motion} Coordinate with scene rhythm: {audio_phase.rhythm_state}."


def _particle_symbolism(
    source_query: str,
    scene: MythopoeticNarrativeScene,
    motif: object | None,
) -> str:
    if "particle" in source_query.lower() or "swarm" in source_query.lower():
        return f"Use particles as traceable carriers of {scene.symbolic_focus}."
    motif_id = getattr(motif, "motif_id", "scene motif")
    return f"Use sparse particles only where they clarify {motif_id} and scene state."


def _spatial_audio_role(scene: MythopoeticNarrativeScene, audio_phase: object) -> str:
    audio = _first(scene.audio_guidance, getattr(audio_phase, "audio_state", "") or "silent timing cue")
    return f"{audio} Place the cue according to the audience path and keep playback opt-in."


def _audience_function(scene: MythopoeticNarrativeScene, architecture: object | None) -> str:
    installation = _first(getattr(architecture, "installation_guidance", ()), "")
    if installation:
        return f"{scene.narrative_function} Audience flow cue: {installation}"
    return f"{scene.narrative_function} Guide audience attention through a legible scene beat."


def _node_evidence(
    scene: MythopoeticNarrativeScene,
    geometry: object | None,
    architecture: object | None,
    motif: object | None,
) -> tuple[str, ...]:
    return _dedupe(
        (
            f"V8.5 scene: {scene.scene_id}",
            f"V8.3 geometry: {getattr(geometry, 'pattern_id', '')}",
            f"V8.4 architecture: {getattr(architecture, 'pattern_id', '')}",
            f"V8.2 motif: {getattr(motif, 'motif_id', '')}",
        )
    )


def _ritual_timing(audio_rhythm_guidance: str | None) -> str:
    if audio_rhythm_guidance:
        return audio_rhythm_guidance
    return "Use a measured threshold cue, one transition breath, and a visible settling beat."


def _preview_targets(creative_translation: CreativeTranslation) -> tuple[str, ...]:
    values = _merge_text(
        creative_translation.runtime_recommendations,
        ("Browser Preview Sandbox",),
        limit=8,
    )
    return tuple(value for value in values if value)


def _geometry_light_guidance(sacred_geometry: SacredGeometryReport) -> tuple[str, ...]:
    return tuple(
        item
        for pattern in sacred_geometry.pattern_guidance
        for item in pattern.color_light_mappings[:1]
    )


def _architecture_light_guidance(sacred_architecture: SacredArchitectureReport) -> tuple[str, ...]:
    return tuple(
        item
        for pattern in sacred_architecture.pattern_guidance
        for item in (*pattern.axis_guidance[:1], *pattern.installation_guidance[:1])
    )


def _particle_tokens(source_query: str) -> tuple[str, ...]:
    normalized = source_query.lower()
    if "swarm" in normalized:
        return ("Swarm particles should reveal collective motion without visual overload.",)
    if "particle" in normalized:
        return ("Particles should carry symbolic traces through each scene transition.",)
    return ()


def _audio_runtime_candidates(
    creative_translation: CreativeTranslation,
    audio_reactive: AudioReactiveGuidance | None,
) -> tuple[str, ...]:
    candidates = []
    if audio_reactive is not None and audio_reactive.audio_runtime:
        candidates.append(audio_reactive.audio_runtime)
    candidates.extend(
        runtime
        for runtime in creative_translation.runtime_recommendations
        if runtime in {"Tone.js", "Web Audio API", "p5.sound"}
    )
    candidates.append("Tone.js")
    return _dedupe(candidates)[:8]


def _merge_text(*groups: Sequence[str], limit: int) -> tuple[str, ...]:
    merged: list[str] = []
    for group in groups:
        for item in group:
            normalized = str(item).strip()
            if normalized and normalized not in merged:
                merged.append(normalized)
            if len(merged) >= limit:
                return tuple(merged)
    return tuple(merged)


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return tuple(result)


def _first(values: Sequence[str], fallback: str) -> str:
    for value in values:
        if str(value).strip():
            return str(value).strip()
    return fallback


def _clip(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


__all__ = [
    "build_artistic_decisions",
    "build_audience_journey_plan",
    "build_composer_provenance",
    "build_composer_validation_findings",
    "build_geometry_animation_plan",
    "build_immersive_scene_graph",
    "build_immersive_scene_transitions",
    "build_preview_runtime_audit",
    "build_spatial_audio_plan",
    "build_visual_language_plan",
    "composer_hitl_questions",
    "composer_unsupported_claim_risks",
    "composition_audit_summary",
]
