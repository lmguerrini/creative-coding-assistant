"""Typed contracts for the V8.6 immersive audiovisual composer."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

V8_6_CAPABILITY_ID = "v8_6_immersive_composer"
V8_6_COMPOSER_SCOPE = (
    "Compose symbolic, geometric, architectural, narrative, audiovisual, "
    "preview, and audience-experience guidance into a structured immersive "
    "creative composition report."
)
V8_6_AUTHORITY_BOUNDARY = (
    "V8.6 Immersive Audiovisual Composer provides bounded, browser-internal "
    "composition guidance only. It composes existing V8.1 knowledge, V8.2 "
    "symbolic, V8.3 geometry, V8.4 architecture, V8.5 narrative, and existing "
    "preview/runtime metadata without mutating preview runtimes, executing "
    "artifacts, routing providers or models, writing storage, controlling "
    "workflows, implementing external DCC integrations, starting V8.7 or V8.8, "
    "or claiming HoloMind or HOLOiVERSE behavior."
)


class ImmersiveCompositionLayer(StrEnum):
    SYMBOLIC = "symbolic"
    GEOMETRIC = "geometric"
    ARCHITECTURAL = "architectural"
    NARRATIVE = "narrative"
    VISUAL = "visual"
    AUDIO = "audio"
    AUDIENCE = "audience"
    PREVIEW = "preview"


class ImmersiveCompositionRoadmapClassification(StrEnum):
    IMPLEMENTED_RUNTIME_BEHAVIOR = "implemented_runtime_behavior"
    REUSED_EXISTING_RUNTIME = "reused_existing_runtime"
    PARTIAL_REUSABLE = "partial_reusable"
    ADVISORY_ONLY = "advisory_only"
    PRODUCT_HITL_REQUIRED = "product_hitl_required"
    LATER_V8_BOUNDARY = "later_v8_boundary"
    OUT_OF_SCOPE_UNSUPPORTED = "out_of_scope_unsupported"
    MISSING = "missing"


class ImmersivePreviewImplementationStatus(StrEnum):
    ALREADY_IMPLEMENTED = "already_implemented"
    PARTIALLY_IMPLEMENTED = "partially_implemented"
    REUSABLE = "reusable"
    MISSING = "missing"


class ImmersiveComposerConfidenceBand(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    GUARDED = "guarded"


class ImmersiveComposerValidationSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    HITL_REQUIRED = "hitl_required"


class ImmersiveComposerProvenance(BaseModel):
    """Traceable source behind one V8.6 composition decision."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    provenance_id: str = Field(min_length=1, max_length=210)
    kind: Literal[
        "request_signal",
        "v3_creative_translation",
        "v3_audio_visual_scene",
        "v6_style_profiles",
        "v8_1_creative_knowledge",
        "v8_2_symbolic_translation",
        "v8_3_sacred_geometry",
        "v8_4_sacred_architecture",
        "v8_5_mythopoetic_narrative",
        "preview_runtime_audit",
        "bounded_composer_catalog",
        "safety_boundary",
    ]
    reference: str = Field(min_length=1, max_length=280)
    summary: str = Field(min_length=1, max_length=700)
    confidence_signal: float | None = Field(default=None, ge=0, le=1)


class ImmersiveSceneNode(BaseModel):
    """One node in the composed immersive scene graph."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    node_id: str = Field(min_length=1, max_length=150)
    label: str = Field(min_length=1, max_length=180)
    layer_sequence: int = Field(ge=0, le=24)
    source_layers: tuple[ImmersiveCompositionLayer, ...] = Field(min_length=1, max_length=8)
    source_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=18)
    visual_language: str = Field(min_length=1, max_length=620)
    sacred_lighting: str = Field(min_length=1, max_length=520)
    symbolic_color: str = Field(min_length=1, max_length=520)
    geometry_driver: str = Field(min_length=1, max_length=520)
    animation_plan: str = Field(min_length=1, max_length=560)
    particle_symbolism: str = Field(min_length=1, max_length=520)
    spatial_audio_role: str = Field(min_length=1, max_length=520)
    dramaturgical_function: str = Field(min_length=1, max_length=620)
    audience_function: str = Field(min_length=1, max_length=560)
    preview_targets: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=16)


class ImmersiveSceneTransition(BaseModel):
    """Transition between composed scene graph nodes."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    transition_id: str = Field(min_length=1, max_length=170)
    from_node_id: str = Field(min_length=1, max_length=150)
    to_node_id: str = Field(min_length=1, max_length=150)
    transition_composer: str = Field(min_length=1, max_length=560)
    ritual_timing: str = Field(min_length=1, max_length=520)
    temporal_dramaturgy: str = Field(min_length=1, max_length=560)
    continuity_guidance: str = Field(min_length=1, max_length=560)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)


class ImmersiveVisualLanguagePlan(BaseModel):
    """Visual language, lighting, and color plan for the composition."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    visual_identity: tuple[str, ...] = Field(min_length=1, max_length=10)
    sacred_lighting_guidance: tuple[str, ...] = Field(min_length=1, max_length=10)
    symbolic_color_guidance: tuple[str, ...] = Field(min_length=1, max_length=10)
    shader_or_material_guidance: tuple[str, ...] = Field(min_length=1, max_length=10)
    style_profile_refs: tuple[str, ...] = Field(min_length=1, max_length=8)
    runtime_suitability: tuple[str, ...] = Field(min_length=1, max_length=10)
    boundary: str = Field(min_length=1, max_length=620)


class ImmersiveGeometryAnimationPlan(BaseModel):
    """Geometric animation, particle, quadrivium, and planetary motion plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    geometry_animation_guidance: tuple[str, ...] = Field(min_length=1, max_length=12)
    particle_symbolism_guidance: tuple[str, ...] = Field(min_length=1, max_length=10)
    quadrivium_mapping: tuple[str, ...] = Field(min_length=1, max_length=8)
    planetary_motion_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    harmonic_parameters: tuple[str, ...] = Field(min_length=1, max_length=12)
    validation_notes: tuple[str, ...] = Field(min_length=1, max_length=8)


class ImmersiveSpatialAudioPlan(BaseModel):
    """Spatial audio and sacred music mapping plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    activation: Literal["explicit_user_gesture"] = "explicit_user_gesture"
    audio_runtime_candidates: tuple[str, ...] = Field(min_length=1, max_length=8)
    spatial_audio_guidance: tuple[str, ...] = Field(min_length=1, max_length=10)
    sacred_music_mapping: tuple[str, ...] = Field(min_length=1, max_length=10)
    audiovisual_sync: tuple[str, ...] = Field(min_length=1, max_length=10)
    safety_constraints: tuple[str, ...] = Field(min_length=1, max_length=8)


class ImmersiveAudienceJourneyPlan(BaseModel):
    """Embodied, spatial, temporal, emotional, and audience-flow plan."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    installation_flow: tuple[str, ...] = Field(min_length=1, max_length=10)
    audience_journey: tuple[str, ...] = Field(min_length=1, max_length=10)
    embodied_experience: tuple[str, ...] = Field(min_length=1, max_length=10)
    spatial_dramaturgy: tuple[str, ...] = Field(min_length=1, max_length=10)
    temporal_dramaturgy: tuple[str, ...] = Field(min_length=1, max_length=10)
    emotional_resonance: tuple[str, ...] = Field(min_length=1, max_length=10)
    audience_flow_simulation: tuple[str, ...] = Field(min_length=1, max_length=10)
    boundary: str = Field(min_length=1, max_length=640)


class ImmersiveArtisticDecision(BaseModel):
    """Explainable artistic decision produced by the composer."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    decision_id: str = Field(min_length=1, max_length=170)
    decision: str = Field(min_length=1, max_length=620)
    rationale: str = Field(min_length=1, max_length=760)
    reused_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    affected_layers: tuple[ImmersiveCompositionLayer, ...] = Field(min_length=1, max_length=8)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)


class ImmersivePreviewRuntimeAuditItem(BaseModel):
    """Audit classification for one preview-related roadmap item."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    item: str = Field(min_length=1)
    implementation_status: ImmersivePreviewImplementationStatus
    reusable_for_v8_6: bool
    existing_behavior: str = Field(min_length=1, max_length=760)
    v8_6_action: str = Field(min_length=1, max_length=620)
    evidence_files: tuple[str, ...] = Field(min_length=1, max_length=8)


class ImmersiveComposerValidationFinding(BaseModel):
    """Deterministic validation finding for V8.6 composition guidance."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    finding_id: str = Field(min_length=1, max_length=180)
    severity: ImmersiveComposerValidationSeverity
    summary: str = Field(min_length=1, max_length=560)
    action: str = Field(min_length=1, max_length=560)


class ImmersiveCompositionRoadmapItemAssessment(BaseModel):
    """Reality-check classification for one V8.6 roadmap item."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    item: str = Field(min_length=1)
    classification: ImmersiveCompositionRoadmapClassification
    rationale: str = Field(min_length=1, max_length=760)
    action_required_before_hitl: bool = False
    hitl_required: bool = False


class ImmersiveComposerConfidence(BaseModel):
    """Confidence posture for an immersive composer report."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    score: float = Field(ge=0, le=1)
    band: ImmersiveComposerConfidenceBand
    scene_node_count: int = Field(ge=0)
    transition_count: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    provenance_count: int = Field(ge=0)
    reused_engine_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    caveats: tuple[str, ...] = Field(default_factory=tuple, max_length=10)

    @model_validator(mode="after")
    def _band_matches_score(self) -> Self:
        if self.band != immersive_composer_confidence_band(self.score, guarded=bool(self.caveats)):
            raise ValueError("band must match score and caveat posture")
        return self


class ImmersiveAudiovisualComposerReport(BaseModel):
    """Top-level V8.6 immersive audiovisual composition report."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    capability_id: Literal["v8_6_immersive_composer"] = V8_6_CAPABILITY_ID
    composer_scope: str = Field(default=V8_6_COMPOSER_SCOPE, min_length=1)
    authority_boundary: str = Field(default=V8_6_AUTHORITY_BOUNDARY, min_length=1)
    source_query: str = Field(min_length=1, max_length=860)
    reused_surface_ids: tuple[str, ...] = Field(min_length=1, max_length=24)
    composition_audit_summary: tuple[str, ...] = Field(min_length=5, max_length=10)
    scene_graph: tuple[ImmersiveSceneNode, ...] = Field(min_length=3, max_length=12)
    scene_transitions: tuple[ImmersiveSceneTransition, ...] = Field(min_length=2, max_length=12)
    visual_language: ImmersiveVisualLanguagePlan
    geometry_animation: ImmersiveGeometryAnimationPlan
    spatial_audio: ImmersiveSpatialAudioPlan
    audience_journey: ImmersiveAudienceJourneyPlan
    artistic_decisions: tuple[ImmersiveArtisticDecision, ...] = Field(min_length=4, max_length=12)
    preview_audit: tuple[ImmersivePreviewRuntimeAuditItem, ...] = Field(min_length=9, max_length=12)
    validation_findings: tuple[ImmersiveComposerValidationFinding, ...] = Field(
        min_length=1,
        max_length=14,
    )
    provenance: tuple[ImmersiveComposerProvenance, ...] = Field(min_length=1, max_length=44)
    confidence: ImmersiveComposerConfidence
    roadmap_assessment: tuple[ImmersiveCompositionRoadmapItemAssessment, ...] = Field(
        min_length=1,
        max_length=48,
    )
    implemented_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    reused_existing_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    partial_reusable_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    product_hitl_required_items: tuple[str, ...] = Field(default_factory=tuple)
    later_v8_boundary_items: tuple[str, ...] = Field(default_factory=tuple)
    out_of_scope_unsupported_items: tuple[str, ...] = Field(default_factory=tuple)
    missing_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    unsupported_claim_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    scene_graph_composer_implemented: Literal[True] = True
    visual_language_engine_implemented: Literal[True] = True
    sacred_lighting_engine_implemented: Literal[True] = True
    symbolic_color_engine_implemented: Literal[True] = True
    geometry_animation_engine_implemented: Literal[True] = True
    particle_symbolism_engine_implemented: Literal[True] = True
    spatial_audio_planner_implemented: Literal[True] = True
    sacred_music_mapping_implemented: Literal[True] = True
    quadrivium_engine_implemented: Literal[True] = True
    planetary_motion_engine_implemented: Literal[True] = True
    ritual_timing_engine_implemented: Literal[True] = True
    transition_composer_implemented: Literal[True] = True
    multi_layer_scene_composition_implemented: Literal[True] = True
    installation_flow_engine_implemented: Literal[True] = True
    audience_journey_planner_implemented: Literal[True] = True
    explainable_artistic_decisions_implemented: Literal[True] = True
    creative_style_profiles_reused: Literal[True] = True
    embodied_experience_engine_implemented: Literal[True] = True
    spatial_dramaturgy_engine_implemented: Literal[True] = True
    temporal_dramaturgy_engine_implemented: Literal[True] = True
    emotional_resonance_engine_implemented: Literal[True] = True
    audience_flow_simulation_implemented: Literal[True] = True
    preview_runtime_audit_implemented: Literal[True] = True
    preview_runtime_mutation_implemented: Literal[False] = False
    preview_runtime_implementation_added: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    storage_write_implemented: Literal[False] = False
    external_dcc_integration_implemented: Literal[False] = False
    mcp_integration_implemented: Literal[False] = False
    v8_7_hologenesis_os_started: Literal[False] = False
    v8_8_demo_showcase_started: Literal[False] = False
    holomind_implemented: Literal[False] = False
    holoiverse_implemented: Literal[False] = False

    @model_validator(mode="after")
    def _report_matches_contract(self) -> Self:
        node_ids = {node.node_id for node in self.scene_graph}
        if len(node_ids) != len(self.scene_graph):
            raise ValueError("scene node ids must be unique")
        for transition in self.scene_transitions:
            if transition.from_node_id not in node_ids or transition.to_node_id not in node_ids:
                raise ValueError("scene transitions must reference known nodes")
        decision_ids = {decision.decision_id for decision in self.artistic_decisions}
        if len(decision_ids) != len(self.artistic_decisions):
            raise ValueError("artistic decision ids must be unique")
        preview_items = {item.item for item in self.preview_audit}
        if len(preview_items) != len(self.preview_audit):
            raise ValueError("preview audit items must be unique")
        classified = immersive_composer_items_by_classification(self.roadmap_assessment)
        if self.implemented_roadmap_items != classified[
            ImmersiveCompositionRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR
        ]:
            raise ValueError("implemented items must match roadmap assessment")
        if self.reused_existing_roadmap_items != classified[
            ImmersiveCompositionRoadmapClassification.REUSED_EXISTING_RUNTIME
        ]:
            raise ValueError("reused items must match roadmap assessment")
        if self.partial_reusable_roadmap_items != classified[
            ImmersiveCompositionRoadmapClassification.PARTIAL_REUSABLE
        ]:
            raise ValueError("partial reusable items must match roadmap assessment")
        if self.advisory_only_roadmap_items != classified[
            ImmersiveCompositionRoadmapClassification.ADVISORY_ONLY
        ]:
            raise ValueError("advisory-only items must match roadmap assessment")
        if self.product_hitl_required_items != classified[
            ImmersiveCompositionRoadmapClassification.PRODUCT_HITL_REQUIRED
        ]:
            raise ValueError("product HITL items must match roadmap assessment")
        if self.later_v8_boundary_items != classified[
            ImmersiveCompositionRoadmapClassification.LATER_V8_BOUNDARY
        ]:
            raise ValueError("later V8 items must match roadmap assessment")
        if self.out_of_scope_unsupported_items != classified[
            ImmersiveCompositionRoadmapClassification.OUT_OF_SCOPE_UNSUPPORTED
        ]:
            raise ValueError("out-of-scope unsupported items must match roadmap assessment")
        if self.missing_roadmap_items != classified[
            ImmersiveCompositionRoadmapClassification.MISSING
        ]:
            raise ValueError("missing items must match roadmap assessment")
        return self


def immersive_composer_items_by_classification(
    assessments: Sequence[ImmersiveCompositionRoadmapItemAssessment],
) -> dict[ImmersiveCompositionRoadmapClassification, tuple[str, ...]]:
    return {
        classification: tuple(item.item for item in assessments if item.classification == classification)
        for classification in ImmersiveCompositionRoadmapClassification
    }


def immersive_composer_confidence_band(
    score: float,
    *,
    guarded: bool,
) -> ImmersiveComposerConfidenceBand:
    if guarded:
        return ImmersiveComposerConfidenceBand.GUARDED
    if score >= 0.75:
        return ImmersiveComposerConfidenceBand.HIGH
    if score >= 0.5:
        return ImmersiveComposerConfidenceBand.MEDIUM
    return ImmersiveComposerConfidenceBand.LOW


__all__ = [
    "ImmersiveArtisticDecision",
    "ImmersiveAudiovisualComposerReport",
    "ImmersiveComposerConfidence",
    "ImmersiveComposerConfidenceBand",
    "ImmersiveComposerProvenance",
    "ImmersiveComposerValidationFinding",
    "ImmersiveComposerValidationSeverity",
    "ImmersiveCompositionLayer",
    "ImmersiveCompositionRoadmapClassification",
    "ImmersiveCompositionRoadmapItemAssessment",
    "ImmersiveGeometryAnimationPlan",
    "ImmersivePreviewImplementationStatus",
    "ImmersivePreviewRuntimeAuditItem",
    "ImmersiveSceneNode",
    "ImmersiveSceneTransition",
    "ImmersiveSpatialAudioPlan",
    "ImmersiveAudienceJourneyPlan",
    "ImmersiveVisualLanguagePlan",
    "V8_6_AUTHORITY_BOUNDARY",
    "V8_6_CAPABILITY_ID",
    "V8_6_COMPOSER_SCOPE",
    "immersive_composer_confidence_band",
    "immersive_composer_items_by_classification",
]
