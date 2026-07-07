"""Typed contracts for the V8.5 mythopoetic narrative engine."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

V8_5_CAPABILITY_ID = "v8_5_mythopoetic_engine"
V8_5_NARRATIVE_SCOPE = (
    "Translate mythopoetic, archetypal, symbolic, ritual, emotional, and "
    "experiential intent into bounded narrative guidance for creative coding, "
    "installation concepts, demo storytelling, and project framing."
)
V8_5_AUTHORITY_BOUNDARY = (
    "V8.5 Mythopoetic Narrative Engine provides bounded creative narrative "
    "guidance only. It maps user-visible narrative, symbolic, emotional, "
    "ritual, spatial, visual, motion, and audio cues to inspectable story "
    "structures without claiming religious authority, esoteric authority, "
    "psychological diagnosis, psychotherapy, ritual efficacy, V8.6 immersive "
    "composer behavior, V8.7 OS behavior, HoloMind, HOLOiVERSE, preview "
    "mutation, or external DCC integration."
)


class MythopoeticNarrativeFamily(StrEnum):
    TRANSFORMATION = "transformation"
    INITIATION = "initiation"
    HEROIC_CYCLE = "heroic_cycle"
    DESCENT_RETURN = "descent_return"
    EMERGENCE = "emergence"
    PROCESSION = "procession"
    CONTEMPLATION = "contemplation"
    REFLECTION = "reflection"
    INSTALLATION = "installation"
    DEMO_STORY = "demo_story"
    DIALOGUE = "dialogue"


class MythopoeticNarrativeOperationKind(StrEnum):
    NARRATIVE_STRUCTURE = "narrative_structure"
    SCENE_SEQUENCE = "scene_sequence"
    SYMBOLIC_DIALOGUE = "symbolic_dialogue"
    SYMBOLIC_TRANSITION = "symbolic_transition"
    EMOTIONAL_ARC = "emotional_arc"
    VISUAL_MAPPING = "visual_mapping"
    MOTION_MAPPING = "motion_mapping"
    AUDIO_MAPPING = "audio_mapping"
    SPATIAL_INSTALLATION_MAPPING = "spatial_installation_mapping"
    CREATIVE_BRIEF = "creative_brief"
    EXPLANATION = "explanation"
    AUDIENCE_COMMUNICATION = "audience_communication"
    VALIDATION = "validation"
    SAFETY_BOUNDARY = "safety_boundary"


class MythopoeticNarrativeGraphRole(StrEnum):
    ARCHETYPE = "archetype"
    SYMBOL = "symbol"
    PHASE = "phase"
    SPACE = "space"
    AUDIENCE = "audience"


class MythopoeticNarrativeGraphRelationship(StrEnum):
    TRANSFORMS_INTO = "transforms_into"
    THRESHOLD_TO = "threshold_to"
    REINFORCES = "reinforces"
    SPATIALLY_FRAMES = "spatially_frames"
    ADDRESSES_AUDIENCE = "addresses_audience"


class MythopoeticNarrativeRoadmapClassification(StrEnum):
    IMPLEMENTED_RUNTIME_BEHAVIOR = "implemented_runtime_behavior"
    REUSED_EXISTING_RUNTIME = "reused_existing_runtime"
    PARTIAL_REUSABLE = "partial_reusable"
    ADVISORY_ONLY = "advisory_only"
    PRODUCT_HITL_REQUIRED = "product_hitl_required"
    LATER_V8_BOUNDARY = "later_v8_boundary"
    OUT_OF_SCOPE_UNSUPPORTED = "out_of_scope_unsupported"
    MISSING = "missing"


class MythopoeticNarrativeConfidenceBand(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    GUARDED = "guarded"


class MythopoeticNarrativeValidationSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    HITL_REQUIRED = "hitl_required"


class MythopoeticNarrativeProvenance(BaseModel):
    """Traceable source behind one V8.5 narrative decision."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    provenance_id: str = Field(min_length=1, max_length=190)
    kind: Literal[
        "request_signal",
        "creative_translation",
        "v3_symbolic_narrative",
        "v8_1_creative_knowledge",
        "v8_2_symbolic_translation",
        "v8_3_sacred_geometry",
        "v8_4_sacred_architecture",
        "bounded_narrative_catalog",
        "safety_boundary",
    ]
    reference: str = Field(min_length=1, max_length=260)
    summary: str = Field(min_length=1, max_length=620)
    confidence_signal: float | None = Field(default=None, ge=0, le=1)


class MythopoeticNarrativeSymbolNode(BaseModel):
    """One node in the bounded narrative symbol graph."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    node_id: str = Field(min_length=1, max_length=140)
    label: str = Field(min_length=1, max_length=160)
    role: MythopoeticNarrativeGraphRole
    source_pattern_ids: tuple[str, ...] = Field(min_length=1, max_length=14)
    guidance: str = Field(min_length=1, max_length=460)


class MythopoeticNarrativeSymbolEdge(BaseModel):
    """One edge in the bounded narrative symbol graph."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    edge_id: str = Field(min_length=1, max_length=180)
    from_node_id: str = Field(min_length=1, max_length=140)
    to_node_id: str = Field(min_length=1, max_length=140)
    relationship: MythopoeticNarrativeGraphRelationship
    source_pattern_ids: tuple[str, ...] = Field(min_length=1, max_length=14)
    guidance: str = Field(min_length=1, max_length=520)


class MythopoeticNarrativeScene(BaseModel):
    """One deterministic scene or sequence beat for creative guidance."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    scene_id: str = Field(min_length=1, max_length=150)
    phase: Literal["opening", "call", "threshold", "ordeal", "integration", "return"]
    title: str = Field(min_length=1, max_length=160)
    source_pattern_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    narrative_function: str = Field(min_length=1, max_length=520)
    emotional_state: str = Field(min_length=1, max_length=360)
    symbolic_focus: str = Field(min_length=1, max_length=320)
    visual_guidance: tuple[str, ...] = Field(min_length=1, max_length=6)
    motion_guidance: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    audio_guidance: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    spatial_guidance: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    transition_out: str = Field(min_length=1, max_length=360)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)


class MythopoeticNarrativePatternGuidance(BaseModel):
    """Reusable generation contract for one mythopoetic narrative pattern."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    pattern_id: str = Field(min_length=1, max_length=110)
    label: str = Field(min_length=1, max_length=160)
    family: MythopoeticNarrativeFamily
    source_terms: tuple[str, ...] = Field(min_length=1, max_length=14)
    taxonomy_path: tuple[str, ...] = Field(min_length=2, max_length=7)
    narrative_intent: str = Field(min_length=1, max_length=620)
    archetypal_structure: tuple[str, ...] = Field(min_length=1, max_length=7)
    journey_arc: tuple[str, ...] = Field(min_length=1, max_length=8)
    ritual_structure: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    emotional_arc: tuple[str, ...] = Field(min_length=1, max_length=8)
    symbolic_transitions: tuple[str, ...] = Field(min_length=1, max_length=8)
    visual_mappings: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    motion_mappings: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    audio_mappings: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    spatial_installation_mappings: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    creative_brief_points: tuple[str, ...] = Field(min_length=1, max_length=8)
    explanation_points: tuple[str, ...] = Field(min_length=1, max_length=8)
    audience_communication: tuple[str, ...] = Field(min_length=1, max_length=8)
    boundary: str = Field(min_length=1, max_length=520)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=16)
    confidence_score: float = Field(ge=0, le=1)


class MythopoeticNarrativeOperationalGuidance(BaseModel):
    """Provider-independent operational narrative guidance."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    operation_id: str = Field(min_length=1, max_length=180)
    kind: MythopoeticNarrativeOperationKind
    source_pattern_ids: tuple[str, ...] = Field(min_length=1, max_length=18)
    guidance: tuple[str, ...] = Field(min_length=1, max_length=10)
    parameter_names: tuple[str, ...] = Field(default_factory=tuple, max_length=18)
    runtime_families: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    implementation_notes: tuple[str, ...] = Field(default_factory=tuple, max_length=9)
    constraints: tuple[str, ...] = Field(default_factory=tuple, max_length=10)


class MythopoeticNarrativeValidationFinding(BaseModel):
    """Deterministic validation finding for narrative guidance."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    finding_id: str = Field(min_length=1, max_length=180)
    severity: MythopoeticNarrativeValidationSeverity
    summary: str = Field(min_length=1, max_length=520)
    action: str = Field(min_length=1, max_length=520)


class MythopoeticNarrativeRoadmapItemAssessment(BaseModel):
    """Reality-check classification for one V8.5 roadmap item."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    item: str = Field(min_length=1)
    classification: MythopoeticNarrativeRoadmapClassification
    rationale: str = Field(min_length=1, max_length=620)
    action_required_before_hitl: bool = False
    hitl_required: bool = False


class MythopoeticNarrativeConfidence(BaseModel):
    """Confidence posture for a V8.5 narrative report."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    score: float = Field(ge=0, le=1)
    band: MythopoeticNarrativeConfidenceBand
    pattern_count: int = Field(ge=0)
    scene_count: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    provenance_count: int = Field(ge=0)
    v8_1_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    v8_2_motif_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    v8_3_pattern_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    v8_4_pattern_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    caveats: tuple[str, ...] = Field(default_factory=tuple, max_length=10)

    @model_validator(mode="after")
    def _band_matches_score(self) -> Self:
        if self.band != mythopoetic_narrative_confidence_band(self.score, guarded=bool(self.caveats)):
            raise ValueError("band must match score and caveat posture")
        return self


class MythopoeticNarrativeReport(BaseModel):
    """Top-level V8.5 mythopoetic narrative report."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    capability_id: Literal["v8_5_mythopoetic_engine"] = V8_5_CAPABILITY_ID
    narrative_scope: str = Field(default=V8_5_NARRATIVE_SCOPE, min_length=1)
    authority_boundary: str = Field(default=V8_5_AUTHORITY_BOUNDARY, min_length=1)
    source_query: str = Field(min_length=1, max_length=760)
    reused_surface_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    pattern_guidance: tuple[MythopoeticNarrativePatternGuidance, ...] = Field(min_length=1, max_length=16)
    symbol_nodes: tuple[MythopoeticNarrativeSymbolNode, ...] = Field(min_length=1, max_length=28)
    symbol_edges: tuple[MythopoeticNarrativeSymbolEdge, ...] = Field(default_factory=tuple, max_length=36)
    scene_sequence: tuple[MythopoeticNarrativeScene, ...] = Field(min_length=3, max_length=12)
    operational_guidance: tuple[MythopoeticNarrativeOperationalGuidance, ...] = Field(min_length=1, max_length=18)
    validation_findings: tuple[MythopoeticNarrativeValidationFinding, ...] = Field(min_length=1, max_length=14)
    creative_brief: tuple[str, ...] = Field(min_length=1, max_length=10)
    concept_explanation: tuple[str, ...] = Field(min_length=1, max_length=12)
    symbolic_dialogue_cues: tuple[str, ...] = Field(min_length=1, max_length=10)
    presentation_narrative: tuple[str, ...] = Field(min_length=1, max_length=10)
    demo_story: tuple[str, ...] = Field(min_length=1, max_length=10)
    audience_communication: tuple[str, ...] = Field(min_length=1, max_length=10)
    provenance: tuple[MythopoeticNarrativeProvenance, ...] = Field(min_length=1, max_length=36)
    confidence: MythopoeticNarrativeConfidence
    roadmap_assessment: tuple[MythopoeticNarrativeRoadmapItemAssessment, ...] = Field(
        min_length=1,
        max_length=40,
    )
    implemented_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    reused_existing_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    partial_reusable_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    product_hitl_required_items: tuple[str, ...] = Field(default_factory=tuple)
    later_v8_boundary_items: tuple[str, ...] = Field(default_factory=tuple)
    out_of_scope_unsupported_items: tuple[str, ...] = Field(default_factory=tuple)
    missing_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    interpretation_boundaries: tuple[str, ...] = Field(min_length=1, max_length=12)
    unsupported_claim_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    narrative_contracts_implemented: Literal[True] = True
    narrative_symbol_graph_implemented: Literal[True] = True
    archetypal_transformation_implemented: Literal[True] = True
    ritual_structure_implemented: Literal[True] = True
    emotional_arc_implemented: Literal[True] = True
    scene_sequence_generation_implemented: Literal[True] = True
    narrative_to_visual_motion_audio_spatial_guidance_implemented: Literal[True] = True
    creative_brief_generation_implemented: Literal[True] = True
    interpretability_implemented: Literal[True] = True
    provenance_confidence_integration_implemented: Literal[True] = True
    v8_2_symbolic_translation_reuse_implemented: Literal[True] = True
    v8_3_v8_4_geometry_architecture_reuse_implemented: Literal[True] = True
    immersive_audiovisual_composer_implemented: Literal[False] = False
    preview_runtime_mutation_implemented: Literal[False] = False
    external_dcc_integration_implemented: Literal[False] = False
    authoritative_religious_interpretation_implemented: Literal[False] = False
    authoritative_esoteric_interpretation_implemented: Literal[False] = False
    psychotherapy_or_diagnosis_implemented: Literal[False] = False
    ritual_efficacy_claims_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    storage_write_implemented: Literal[False] = False
    v8_6_immersive_composer_started: Literal[False] = False
    v8_7_hologenesis_os_started: Literal[False] = False
    v8_8_demo_showcase_started: Literal[False] = False
    holomind_implemented: Literal[False] = False
    holoiverse_implemented: Literal[False] = False

    @model_validator(mode="after")
    def _report_matches_contract(self) -> Self:
        pattern_ids = {item.pattern_id for item in self.pattern_guidance}
        if len(pattern_ids) != len(self.pattern_guidance):
            raise ValueError("pattern ids must be unique")
        node_ids = {item.node_id for item in self.symbol_nodes}
        if len(node_ids) != len(self.symbol_nodes):
            raise ValueError("symbol node ids must be unique")
        for item in (*self.scene_sequence, *self.operational_guidance):
            if not set(item.source_pattern_ids).issubset(pattern_ids):
                raise ValueError("scene and operational guidance must reference known patterns")
        for node in self.symbol_nodes:
            if not set(node.source_pattern_ids).issubset(pattern_ids):
                raise ValueError("symbol nodes must reference known patterns")
        for edge in self.symbol_edges:
            if edge.from_node_id not in node_ids or edge.to_node_id not in node_ids:
                raise ValueError("symbol edges must reference known nodes")
            if not set(edge.source_pattern_ids).issubset(pattern_ids):
                raise ValueError("symbol edges must reference known patterns")
        classified = mythopoetic_narrative_items_by_classification(self.roadmap_assessment)
        if self.implemented_roadmap_items != classified[
            MythopoeticNarrativeRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR
        ]:
            raise ValueError("implemented items must match roadmap assessment")
        if self.reused_existing_roadmap_items != classified[
            MythopoeticNarrativeRoadmapClassification.REUSED_EXISTING_RUNTIME
        ]:
            raise ValueError("reused items must match roadmap assessment")
        if self.partial_reusable_roadmap_items != classified[
            MythopoeticNarrativeRoadmapClassification.PARTIAL_REUSABLE
        ]:
            raise ValueError("partial reusable items must match roadmap assessment")
        if self.advisory_only_roadmap_items != classified[
            MythopoeticNarrativeRoadmapClassification.ADVISORY_ONLY
        ]:
            raise ValueError("advisory-only items must match roadmap assessment")
        if self.product_hitl_required_items != classified[
            MythopoeticNarrativeRoadmapClassification.PRODUCT_HITL_REQUIRED
        ]:
            raise ValueError("product HITL items must match roadmap assessment")
        if self.later_v8_boundary_items != classified[MythopoeticNarrativeRoadmapClassification.LATER_V8_BOUNDARY]:
            raise ValueError("later V8 items must match roadmap assessment")
        if self.out_of_scope_unsupported_items != classified[
            MythopoeticNarrativeRoadmapClassification.OUT_OF_SCOPE_UNSUPPORTED
        ]:
            raise ValueError("unsupported items must match roadmap assessment")
        if self.missing_roadmap_items != classified[MythopoeticNarrativeRoadmapClassification.MISSING]:
            raise ValueError("missing items must match roadmap assessment")
        return self


def mythopoetic_narrative_items_by_classification(
    assessments: Sequence[MythopoeticNarrativeRoadmapItemAssessment],
) -> dict[MythopoeticNarrativeRoadmapClassification, tuple[str, ...]]:
    return {
        classification: tuple(item.item for item in assessments if item.classification == classification)
        for classification in MythopoeticNarrativeRoadmapClassification
    }


def mythopoetic_narrative_confidence_band(
    score: float,
    *,
    guarded: bool,
) -> MythopoeticNarrativeConfidenceBand:
    if guarded:
        return MythopoeticNarrativeConfidenceBand.GUARDED
    if score >= 0.75:
        return MythopoeticNarrativeConfidenceBand.HIGH
    if score >= 0.5:
        return MythopoeticNarrativeConfidenceBand.MEDIUM
    return MythopoeticNarrativeConfidenceBand.LOW


__all__ = [
    "MythopoeticNarrativeConfidence",
    "MythopoeticNarrativeConfidenceBand",
    "MythopoeticNarrativeFamily",
    "MythopoeticNarrativeGraphRelationship",
    "MythopoeticNarrativeGraphRole",
    "MythopoeticNarrativeOperationKind",
    "MythopoeticNarrativeOperationalGuidance",
    "MythopoeticNarrativePatternGuidance",
    "MythopoeticNarrativeProvenance",
    "MythopoeticNarrativeReport",
    "MythopoeticNarrativeRoadmapClassification",
    "MythopoeticNarrativeRoadmapItemAssessment",
    "MythopoeticNarrativeScene",
    "MythopoeticNarrativeSymbolEdge",
    "MythopoeticNarrativeSymbolNode",
    "MythopoeticNarrativeValidationFinding",
    "MythopoeticNarrativeValidationSeverity",
    "V8_5_AUTHORITY_BOUNDARY",
    "V8_5_CAPABILITY_ID",
    "V8_5_NARRATIVE_SCOPE",
    "mythopoetic_narrative_confidence_band",
    "mythopoetic_narrative_items_by_classification",
]
