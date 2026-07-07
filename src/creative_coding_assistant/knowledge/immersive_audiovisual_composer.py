"""V8.6 bounded immersive audiovisual composer."""

from __future__ import annotations

from collections.abc import Sequence

from creative_coding_assistant.contracts import AssistantRequest, CreativeCodingDomain
from creative_coding_assistant.knowledge.creative_distillation import (
    CreativeKnowledgeDistillationReport,
    build_v8_1_creative_knowledge_distillation,
)
from creative_coding_assistant.knowledge.immersive_audiovisual_composer_catalog import (
    ROADMAP_CLASSIFICATION_ROWS,
)
from creative_coding_assistant.knowledge.immersive_audiovisual_composer_contracts import (
    ImmersiveAudiovisualComposerReport,
    ImmersiveComposerConfidence,
    ImmersiveComposerProvenance,
    ImmersiveCompositionRoadmapClassification,
    ImmersiveCompositionRoadmapItemAssessment,
    immersive_composer_confidence_band,
    immersive_composer_items_by_classification,
)
from creative_coding_assistant.knowledge.immersive_audiovisual_composer_guidance import (
    build_artistic_decisions,
    build_audience_journey_plan,
    build_composer_provenance,
    build_composer_validation_findings,
    build_geometry_animation_plan,
    build_immersive_scene_graph,
    build_immersive_scene_transitions,
    build_preview_runtime_audit,
    build_spatial_audio_plan,
    build_visual_language_plan,
    composer_hitl_questions,
    composer_unsupported_claim_risks,
    composition_audit_summary,
)
from creative_coding_assistant.knowledge.mythopoetic_narrative import (
    build_v8_5_mythopoetic_narrative_engine,
)
from creative_coding_assistant.knowledge.mythopoetic_narrative_contracts import (
    MythopoeticNarrativeReport,
)
from creative_coding_assistant.knowledge.sacred_architecture import (
    build_v8_4_sacred_architecture_engine,
)
from creative_coding_assistant.knowledge.sacred_architecture_contracts import (
    SacredArchitectureReport,
)
from creative_coding_assistant.knowledge.sacred_geometry import (
    build_v8_3_sacred_geometry_engine,
)
from creative_coding_assistant.knowledge.sacred_geometry_contracts import SacredGeometryReport
from creative_coding_assistant.knowledge.symbolic_translation import (
    SymbolicTranslationReport,
    build_v8_2_symbolic_translation_engine,
)
from creative_coding_assistant.orchestration.audio_visual_scene import (
    AudioVisualSceneProfile,
    derive_audio_visual_scene_profile,
)
from creative_coding_assistant.orchestration.creative_translation import (
    CreativeTranslation,
    derive_creative_translation,
)
from creative_coding_assistant.orchestration.shader_presets import ShaderPresetGuidance
from creative_coding_assistant.orchestration.style_profiles import (
    StyleProfilePlan,
    build_style_profiles,
)
from creative_coding_assistant.orchestration.visual_styles import VisualStyleGuidance


def build_v8_6_immersive_audiovisual_composer(
    query: str,
    *,
    domains: Sequence[CreativeCodingDomain] = (),
    creative_translation: CreativeTranslation | None = None,
    v8_1_distillation: CreativeKnowledgeDistillationReport | None = None,
    v8_2_symbolic_translation: SymbolicTranslationReport | None = None,
    v8_3_sacred_geometry: SacredGeometryReport | None = None,
    v8_4_sacred_architecture: SacredArchitectureReport | None = None,
    v8_5_mythopoetic_narrative: MythopoeticNarrativeReport | None = None,
    audio_visual_scene: AudioVisualSceneProfile | None = None,
    style_profiles: StyleProfilePlan | None = None,
    visual_style: VisualStyleGuidance | None = None,
    shader_presets: ShaderPresetGuidance | None = None,
) -> ImmersiveAudiovisualComposerReport:
    """Build a bounded V8.6 immersive composition report without runtime mutation."""

    translation = creative_translation or derive_creative_translation(query, domains=domains)
    distillation = v8_1_distillation or build_v8_1_creative_knowledge_distillation()
    symbolic = v8_2_symbolic_translation or build_v8_2_symbolic_translation_engine(
        query,
        domains=domains,
        creative_translation=translation,
        v8_1_distillation=distillation,
    )
    geometry = v8_3_sacred_geometry or build_v8_3_sacred_geometry_engine(
        query,
        domains=domains,
        creative_translation=translation,
        v8_1_distillation=distillation,
        v8_2_symbolic_translation=symbolic,
    )
    architecture = v8_4_sacred_architecture or build_v8_4_sacred_architecture_engine(
        query,
        domains=domains,
        creative_translation=translation,
        v8_1_distillation=distillation,
        v8_2_symbolic_translation=symbolic,
        v8_3_sacred_geometry=geometry,
    )
    narrative = v8_5_mythopoetic_narrative or build_v8_5_mythopoetic_narrative_engine(
        query,
        domains=domains,
        creative_translation=translation,
        v8_1_distillation=distillation,
        v8_2_symbolic_translation=symbolic,
        v8_3_sacred_geometry=geometry,
        v8_4_sacred_architecture=architecture,
    )
    request = AssistantRequest(query=query, domains=domains)
    av_scene = audio_visual_scene or derive_audio_visual_scene_profile(
        request=request,
        route_decision=None,
        creative_translation=translation,
    )
    styles = style_profiles or build_style_profiles()
    visual = visual_style or translation.visual_style
    shader = shader_presets or translation.shader_presets
    audio_reactive = translation.audio_reactive
    unsupported_claim_risks = composer_unsupported_claim_risks(query)
    preview_audit = build_preview_runtime_audit()
    scene_graph = build_immersive_scene_graph(
        source_query=query,
        creative_translation=translation,
        symbolic_translation=symbolic,
        sacred_geometry=geometry,
        sacred_architecture=architecture,
        mythopoetic_narrative=narrative,
        audio_visual_scene=av_scene,
    )
    scene_transitions = build_immersive_scene_transitions(
        scene_graph=scene_graph,
        mythopoetic_narrative=narrative,
        audio_visual_scene=av_scene,
    )
    visual_language = build_visual_language_plan(
        creative_translation=translation,
        symbolic_translation=symbolic,
        sacred_geometry=geometry,
        sacred_architecture=architecture,
        style_profiles=styles,
        visual_style=visual,
        shader_presets=shader,
    )
    geometry_animation = build_geometry_animation_plan(
        source_query=query,
        sacred_geometry=geometry,
        mythopoetic_narrative=narrative,
    )
    spatial_audio = build_spatial_audio_plan(
        creative_translation=translation,
        sacred_geometry=geometry,
        mythopoetic_narrative=narrative,
        audio_visual_scene=av_scene,
        audio_reactive=audio_reactive,
    )
    audience_journey = build_audience_journey_plan(
        sacred_architecture=architecture,
        mythopoetic_narrative=narrative,
        audio_visual_scene=av_scene,
    )
    artistic_decisions = build_artistic_decisions(
        scene_graph=scene_graph,
        visual_language=visual_language,
        geometry_animation=geometry_animation,
        spatial_audio=spatial_audio,
        audience_journey=audience_journey,
    )
    provenance = build_composer_provenance(
        source_query=query,
        style_profiles=styles,
        preview_audit=preview_audit,
        unsupported_claim_risks=unsupported_claim_risks,
    )
    validation_findings = build_composer_validation_findings(unsupported_claim_risks)
    roadmap = immersive_audiovisual_composer_roadmap_assessment()
    classified = immersive_composer_items_by_classification(roadmap)
    reused_surface_ids = _reused_surface_ids(
        narrative=narrative,
        style_profiles=styles,
        audio_visual_scene=av_scene,
    )
    confidence = _build_confidence(
        scene_graph=scene_graph,
        scene_transitions=scene_transitions,
        provenance=provenance,
        reused_surface_ids=reused_surface_ids,
        unsupported_claim_risks=unsupported_claim_risks,
    )

    return ImmersiveAudiovisualComposerReport(
        source_query=_clip(query, 860),
        reused_surface_ids=reused_surface_ids,
        composition_audit_summary=composition_audit_summary(),
        scene_graph=scene_graph,
        scene_transitions=scene_transitions,
        visual_language=visual_language,
        geometry_animation=geometry_animation,
        spatial_audio=spatial_audio,
        audience_journey=audience_journey,
        artistic_decisions=artistic_decisions,
        preview_audit=preview_audit,
        validation_findings=validation_findings,
        provenance=provenance,
        confidence=confidence,
        roadmap_assessment=roadmap,
        implemented_roadmap_items=classified[
            ImmersiveCompositionRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR
        ],
        reused_existing_roadmap_items=classified[
            ImmersiveCompositionRoadmapClassification.REUSED_EXISTING_RUNTIME
        ],
        partial_reusable_roadmap_items=classified[
            ImmersiveCompositionRoadmapClassification.PARTIAL_REUSABLE
        ],
        advisory_only_roadmap_items=classified[
            ImmersiveCompositionRoadmapClassification.ADVISORY_ONLY
        ],
        product_hitl_required_items=classified[
            ImmersiveCompositionRoadmapClassification.PRODUCT_HITL_REQUIRED
        ],
        later_v8_boundary_items=classified[
            ImmersiveCompositionRoadmapClassification.LATER_V8_BOUNDARY
        ],
        out_of_scope_unsupported_items=classified[
            ImmersiveCompositionRoadmapClassification.OUT_OF_SCOPE_UNSUPPORTED
        ],
        missing_roadmap_items=classified[ImmersiveCompositionRoadmapClassification.MISSING],
        unsupported_claim_risks=unsupported_claim_risks,
        hitl_questions=composer_hitl_questions(unsupported_claim_risks),
    )


def immersive_audiovisual_composer_prompt_lines(
    report: ImmersiveAudiovisualComposerReport,
) -> tuple[str, ...]:
    """Render compact provider-independent V8.6 composition guidance."""

    lines = [
        f"Immersive Composer boundary: {report.authority_boundary}",
        f"Immersive composer confidence: {report.confidence.band.value} {report.confidence.score:.2f}.",
    ]
    lines.extend(f"Composition audit: {item}" for item in report.composition_audit_summary)
    for node in report.scene_graph[:8]:
        lines.append(f"Scene graph node: {node.node_id}; {node.label}; {node.dramaturgical_function}")
        lines.append(f"{node.node_id} visual language: {node.visual_language}")
        lines.append(f"{node.node_id} lighting: {node.sacred_lighting}")
        lines.append(f"{node.node_id} geometry animation: {node.geometry_driver}; {node.animation_plan}")
        lines.append(f"{node.node_id} spatial audio: {node.spatial_audio_role}")
        lines.append(f"{node.node_id} audience function: {node.audience_function}")
    for transition in report.scene_transitions[:8]:
        lines.append(
            "Scene transition: "
            f"{transition.from_node_id} -> {transition.to_node_id}; "
            f"{transition.transition_composer}"
        )
    lines.extend(f"Visual language: {item}" for item in report.visual_language.visual_identity[:5])
    lines.extend(
        f"Sacred lighting: {item}" for item in report.visual_language.sacred_lighting_guidance[:5]
    )
    lines.extend(
        f"Symbolic color: {item}" for item in report.visual_language.symbolic_color_guidance[:5]
    )
    lines.extend(
        f"Geometry animation: {item}"
        for item in report.geometry_animation.geometry_animation_guidance[:6]
    )
    lines.extend(
        f"Particle symbolism: {item}"
        for item in report.geometry_animation.particle_symbolism_guidance[:4]
    )
    lines.extend(
        f"Quadrivium mapping: {item}" for item in report.geometry_animation.quadrivium_mapping
    )
    lines.extend(
        f"Spatial audio: {item}" for item in report.spatial_audio.spatial_audio_guidance[:6]
    )
    lines.extend(
        f"Sacred music mapping: {item}" for item in report.spatial_audio.sacred_music_mapping[:6]
    )
    lines.extend(
        f"Audience flow simulation: {item}"
        for item in report.audience_journey.audience_flow_simulation
    )
    lines.extend(
        f"Artistic decision: {item.decision}; {item.rationale}"
        for item in report.artistic_decisions[:6]
    )
    lines.extend(
        "Preview audit: "
        f"{item.item}; {item.implementation_status.value}; reusable={item.reusable_for_v8_6}"
        for item in report.preview_audit
    )
    lines.extend(
        f"Composer validation: {item.severity.value}; {item.summary}"
        for item in report.validation_findings
    )
    lines.extend(f"Unsupported composer claim risk: {item}" for item in report.unsupported_claim_risks)
    lines.extend(f"HITL composer question: {item}" for item in report.hitl_questions)
    return tuple(lines[:150])


def immersive_audiovisual_composer_roadmap_assessment() -> tuple[
    ImmersiveCompositionRoadmapItemAssessment,
    ...,
]:
    """Return the V8.6 roadmap reality-check assessment."""

    return tuple(
        ImmersiveCompositionRoadmapItemAssessment(
            item=item,
            classification=ImmersiveCompositionRoadmapClassification(classification),
            rationale=rationale,
            action_required_before_hitl=action_required,
            hitl_required=hitl_required,
        )
        for item, (classification, rationale, action_required, hitl_required) in (
            ROADMAP_CLASSIFICATION_ROWS.items()
        )
    )


def _reused_surface_ids(
    *,
    narrative: MythopoeticNarrativeReport,
    style_profiles: StyleProfilePlan,
    audio_visual_scene: AudioVisualSceneProfile,
) -> tuple[str, ...]:
    return _dedupe(
        (
            "v3_creative_translation",
            audio_visual_scene.role,
            "v6_style_profiles",
            *style_profiles.profile_ids[:5],
            "v8_1_creative_knowledge_distillation",
            "v8_2_symbolic_translation_engine",
            "v8_3_sacred_geometry_engine",
            "v8_4_sacred_architecture_engine",
            "v8_5_mythopoetic_engine",
            *narrative.reused_surface_ids,
            "browser_preview_sandbox",
            "multi_preview_comparison",
        )
    )[:24]


def _build_confidence(
    *,
    scene_graph: Sequence[object],
    scene_transitions: Sequence[object],
    provenance: Sequence[ImmersiveComposerProvenance],
    reused_surface_ids: Sequence[str],
    unsupported_claim_risks: Sequence[str],
) -> ImmersiveComposerConfidence:
    base = 0.48
    base += min(len(scene_graph), 5) * 0.04
    base += min(len(scene_transitions), 4) * 0.03
    base += min(len(reused_surface_ids), 12) * 0.015
    base += min(len(provenance), 10) * 0.012
    if unsupported_claim_risks:
        base -= 0.25
    score = max(0.0, min(0.96, round(base, 3)))
    return ImmersiveComposerConfidence(
        score=score,
        band=immersive_composer_confidence_band(score, guarded=bool(unsupported_claim_risks)),
        scene_node_count=len(scene_graph),
        transition_count=len(scene_transitions),
        evidence_count=sum(len(getattr(node, "evidence", ())) for node in scene_graph),
        provenance_count=len(provenance),
        reused_engine_ids=tuple(reused_surface_ids[:16]),
        caveats=tuple(unsupported_claim_risks[:10]),
    )


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return tuple(result)


def _clip(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


__all__ = [
    "build_v8_6_immersive_audiovisual_composer",
    "immersive_audiovisual_composer_prompt_lines",
    "immersive_audiovisual_composer_roadmap_assessment",
]
