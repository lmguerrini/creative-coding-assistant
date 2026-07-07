"""Typed contracts for the V8.4 sacred architecture engine."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

V8_4_CAPABILITY_ID = "v8_4_sacred_architecture_engine"
V8_4_ARCHITECTURE_SCOPE = (
    "Translate architectural, spatial, proportional, geometric, symbolic, "
    "and installation-oriented intent into structured creative coding and "
    "spatial composition guidance."
)
V8_4_AUTHORITY_BOUNDARY = (
    "V8.4 Sacred Architecture and Reverse Engineering Engine provides bounded "
    "textual and spatial guidance only. It maps user-visible architectural "
    "cues to inspectable proportions, plan layouts, axes, thresholds, topology, "
    "symbolic-spatial mappings, and installation guidance without claiming "
    "image reconstruction, LIDAR interpretation, photogrammetry, real building "
    "survey, CAD/DCC integration, historical authority, V8.5 narrative behavior, "
    "V8.6 immersive composer behavior, HoloMind, or HOLOiVERSE."
)


class SacredArchitectureFamily(StrEnum):
    PROPORTION = "proportion"
    AXIAL = "axial"
    RADIAL = "radial"
    GRID = "grid"
    COURTYARD = "courtyard"
    THRESHOLD = "threshold"
    PROCESSIONAL = "processional"
    LIGHT_ORIENTATION = "light_orientation"
    LABYRINTHINE = "labyrinthine"
    INSTALLATION = "installation"
    TOPOLOGICAL = "topological"


class SacredArchitectureOperationKind(StrEnum):
    PROPORTION_GUIDANCE = "proportion_guidance"
    PLANIMETRY_LAYOUT = "planimetry_layout"
    AXIS_SYMMETRY = "axis_symmetry"
    THRESHOLD_PROCESSION = "threshold_procession"
    CENTER_PERIPHERY_TOPOLOGY = "center_periphery_topology"
    GEOMETRY_TO_ARCHITECTURE_MAPPING = "geometry_to_architecture_mapping"
    SYMBOLIC_TO_SPATIAL_MAPPING = "symbolic_to_spatial_mapping"
    SEMANTIC_GRAPH = "semantic_graph"
    INSTALLATION_PLANNING = "installation_planning"
    REVERSE_ENGINEERING_GUIDANCE = "reverse_engineering_guidance"
    VALIDATION = "validation"
    SAFETY_BOUNDARY = "safety_boundary"


class SacredArchitectureRoadmapClassification(StrEnum):
    IMPLEMENTED_RUNTIME_BEHAVIOR = "implemented_runtime_behavior"
    REUSED_EXISTING_RUNTIME = "reused_existing_runtime"
    PARTIAL_REUSABLE = "partial_reusable"
    ADVISORY_ONLY = "advisory_only"
    PRODUCT_HITL_REQUIRED = "product_hitl_required"
    LATER_V8_BOUNDARY = "later_v8_boundary"
    OUT_OF_SCOPE_UNSUPPORTED = "out_of_scope_unsupported"
    MISSING = "missing"


class SacredArchitectureConfidenceBand(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    GUARDED = "guarded"


class SacredArchitectureValidationSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    HITL_REQUIRED = "hitl_required"


class SacredArchitectureSemanticRole(StrEnum):
    ENTRY = "entry"
    THRESHOLD = "threshold"
    AXIS = "axis"
    CENTER = "center"
    PERIPHERY = "periphery"
    TRANSITION = "transition"
    GATHERING = "gathering"
    EXHIBIT = "exhibit"
    VOID = "void"
    BOUNDARY = "boundary"
    PROCESSION = "procession"
    LIGHT_SOURCE = "light_source"


class SacredArchitectureSemanticRelationship(StrEnum):
    PROCESSION = "procession"
    SYMMETRY_AXIS = "symmetry_axis"
    THRESHOLD_CROSSING = "threshold_crossing"
    CENTER_PERIPHERY = "center_periphery"
    SIGHTLINE = "sightline"
    ADJACENCY = "adjacency"
    CONTAINMENT = "containment"


class SacredArchitectureProvenance(BaseModel):
    """Traceable source behind one V8.4 architectural guidance decision."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    provenance_id: str = Field(min_length=1, max_length=180)
    kind: Literal[
        "request_signal",
        "creative_translation",
        "v8_1_creative_knowledge",
        "v8_2_symbolic_translation",
        "v8_3_sacred_geometry",
        "bounded_architecture_catalog",
        "safety_boundary",
    ]
    reference: str = Field(min_length=1, max_length=240)
    summary: str = Field(min_length=1, max_length=560)
    confidence_signal: float | None = Field(default=None, ge=0, le=1)


class SacredArchitecturePatternGuidance(BaseModel):
    """Reusable spatial composition contract for one architectural pattern."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    pattern_id: str = Field(min_length=1, max_length=100)
    label: str = Field(min_length=1, max_length=140)
    family: SacredArchitectureFamily
    source_terms: tuple[str, ...] = Field(min_length=1, max_length=14)
    taxonomy_path: tuple[str, ...] = Field(min_length=2, max_length=7)
    spatial_intent: str = Field(min_length=1, max_length=560)
    proportion_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    plan_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    axis_guidance: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    threshold_guidance: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    center_periphery_guidance: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    topology_guidance: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    geometry_mappings: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    symbolic_mappings: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    installation_guidance: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    reverse_engineering_cues: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    runtime_families: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    implementation_notes: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    boundary: str = Field(min_length=1, max_length=460)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=16)
    confidence_score: float = Field(ge=0, le=1)


class SacredArchitectureSemanticNode(BaseModel):
    """One node in a bounded architecture semantic topology graph."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    node_id: str = Field(min_length=1, max_length=120)
    label: str = Field(min_length=1, max_length=140)
    role: SacredArchitectureSemanticRole
    source_pattern_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    guidance: str = Field(min_length=1, max_length=360)


class SacredArchitectureSemanticEdge(BaseModel):
    """One relationship in the bounded architecture semantic topology graph."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    edge_id: str = Field(min_length=1, max_length=160)
    from_node_id: str = Field(min_length=1, max_length=120)
    to_node_id: str = Field(min_length=1, max_length=120)
    relationship: SacredArchitectureSemanticRelationship
    source_pattern_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    guidance: str = Field(min_length=1, max_length=420)


class SacredArchitectureOperationalGuidance(BaseModel):
    """Provider-independent operational guidance consumable by generation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    operation_id: str = Field(min_length=1, max_length=170)
    kind: SacredArchitectureOperationKind
    source_pattern_ids: tuple[str, ...] = Field(min_length=1, max_length=16)
    guidance: tuple[str, ...] = Field(min_length=1, max_length=10)
    parameter_names: tuple[str, ...] = Field(default_factory=tuple, max_length=18)
    runtime_families: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    implementation_notes: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    constraints: tuple[str, ...] = Field(default_factory=tuple, max_length=9)


class SacredArchitectureValidationFinding(BaseModel):
    """Deterministic validation finding for architecture guidance."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    finding_id: str = Field(min_length=1, max_length=170)
    severity: SacredArchitectureValidationSeverity
    summary: str = Field(min_length=1, max_length=460)
    action: str = Field(min_length=1, max_length=460)


class SacredArchitectureRoadmapItemAssessment(BaseModel):
    """Reality-check classification for one V8.4 roadmap item."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    item: str = Field(min_length=1)
    classification: SacredArchitectureRoadmapClassification
    rationale: str = Field(min_length=1, max_length=560)
    action_required_before_hitl: bool = False
    hitl_required: bool = False


class SacredArchitectureConfidence(BaseModel):
    """Confidence posture for a V8.4 architecture report."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    score: float = Field(ge=0, le=1)
    band: SacredArchitectureConfidenceBand
    pattern_count: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    provenance_count: int = Field(ge=0)
    v8_1_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    v8_2_motif_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    v8_3_pattern_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    caveats: tuple[str, ...] = Field(default_factory=tuple, max_length=9)

    @model_validator(mode="after")
    def _band_matches_score(self) -> Self:
        if self.band != sacred_architecture_confidence_band(self.score, guarded=bool(self.caveats)):
            raise ValueError("band must match score and caveat posture")
        return self


class SacredArchitectureReport(BaseModel):
    """Top-level V8.4 sacred architecture and reverse-engineering report."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    capability_id: Literal["v8_4_sacred_architecture_engine"] = V8_4_CAPABILITY_ID
    architecture_scope: str = Field(default=V8_4_ARCHITECTURE_SCOPE, min_length=1)
    authority_boundary: str = Field(default=V8_4_AUTHORITY_BOUNDARY, min_length=1)
    source_query: str = Field(min_length=1, max_length=680)
    reused_surface_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=14)
    pattern_guidance: tuple[SacredArchitecturePatternGuidance, ...] = Field(min_length=1, max_length=16)
    operational_guidance: tuple[SacredArchitectureOperationalGuidance, ...] = Field(min_length=1, max_length=14)
    semantic_nodes: tuple[SacredArchitectureSemanticNode, ...] = Field(min_length=1, max_length=18)
    semantic_edges: tuple[SacredArchitectureSemanticEdge, ...] = Field(default_factory=tuple, max_length=24)
    validation_findings: tuple[SacredArchitectureValidationFinding, ...] = Field(min_length=1, max_length=14)
    provenance: tuple[SacredArchitectureProvenance, ...] = Field(min_length=1, max_length=30)
    confidence: SacredArchitectureConfidence
    roadmap_assessment: tuple[SacredArchitectureRoadmapItemAssessment, ...] = Field(min_length=1, max_length=40)
    implemented_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    reused_existing_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    partial_reusable_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    product_hitl_required_items: tuple[str, ...] = Field(default_factory=tuple)
    later_v8_boundary_items: tuple[str, ...] = Field(default_factory=tuple)
    out_of_scope_unsupported_items: tuple[str, ...] = Field(default_factory=tuple)
    missing_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    interpretation_boundaries: tuple[str, ...] = Field(min_length=1, max_length=12)
    unsupported_claim_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=9)
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    architectural_proportion_guidance_implemented: Literal[True] = True
    floor_plan_reasoning_implemented: Literal[True] = True
    axis_symmetry_threshold_guidance_implemented: Literal[True] = True
    sacred_architecture_taxonomy_implemented: Literal[True] = True
    geometry_to_architecture_mapping_implemented: Literal[True] = True
    symbolic_to_spatial_mapping_implemented: Literal[True] = True
    installation_guidance_implemented: Literal[True] = True
    textual_reverse_engineering_implemented: Literal[True] = True
    architecture_pattern_recommendation_implemented: Literal[True] = True
    provenance_confidence_integration_implemented: Literal[True] = True
    image_based_reconstruction_implemented: Literal[False] = False
    lidar_interpretation_implemented: Literal[False] = False
    photogrammetry_implemented: Literal[False] = False
    cad_dcc_integration_implemented: Literal[False] = False
    actual_architectural_analysis_implemented: Literal[False] = False
    interactive_architecture_preview_implemented: Literal[False] = False
    preview_runtime_mutation_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    storage_write_implemented: Literal[False] = False
    v8_5_narrative_engine_started: Literal[False] = False
    v8_6_immersive_composer_started: Literal[False] = False
    holomind_implemented: Literal[False] = False
    holoiverse_implemented: Literal[False] = False

    @model_validator(mode="after")
    def _report_matches_contract(self) -> Self:
        pattern_ids = {item.pattern_id for item in self.pattern_guidance}
        if len(pattern_ids) != len(self.pattern_guidance):
            raise ValueError("pattern ids must be unique")
        node_ids = {item.node_id for item in self.semantic_nodes}
        if len(node_ids) != len(self.semantic_nodes):
            raise ValueError("semantic node ids must be unique")
        for item in self.operational_guidance:
            if not set(item.source_pattern_ids).issubset(pattern_ids):
                raise ValueError("operational guidance must reference known patterns")
        for node in self.semantic_nodes:
            if not set(node.source_pattern_ids).issubset(pattern_ids):
                raise ValueError("semantic nodes must reference known patterns")
        for edge in self.semantic_edges:
            if edge.from_node_id not in node_ids or edge.to_node_id not in node_ids:
                raise ValueError("semantic edges must reference known nodes")
            if not set(edge.source_pattern_ids).issubset(pattern_ids):
                raise ValueError("semantic edges must reference known patterns")
        classified = sacred_architecture_items_by_classification(self.roadmap_assessment)
        if self.implemented_roadmap_items != classified[
            SacredArchitectureRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR
        ]:
            raise ValueError("implemented items must match roadmap assessment")
        if self.reused_existing_roadmap_items != classified[
            SacredArchitectureRoadmapClassification.REUSED_EXISTING_RUNTIME
        ]:
            raise ValueError("reused items must match roadmap assessment")
        if self.partial_reusable_roadmap_items != classified[SacredArchitectureRoadmapClassification.PARTIAL_REUSABLE]:
            raise ValueError("partial reusable items must match roadmap assessment")
        if self.advisory_only_roadmap_items != classified[SacredArchitectureRoadmapClassification.ADVISORY_ONLY]:
            raise ValueError("advisory-only items must match roadmap assessment")
        if self.product_hitl_required_items != classified[
            SacredArchitectureRoadmapClassification.PRODUCT_HITL_REQUIRED
        ]:
            raise ValueError("product HITL items must match roadmap assessment")
        if self.later_v8_boundary_items != classified[SacredArchitectureRoadmapClassification.LATER_V8_BOUNDARY]:
            raise ValueError("later V8 items must match roadmap assessment")
        if self.out_of_scope_unsupported_items != classified[
            SacredArchitectureRoadmapClassification.OUT_OF_SCOPE_UNSUPPORTED
        ]:
            raise ValueError("unsupported items must match roadmap assessment")
        if self.missing_roadmap_items != classified[SacredArchitectureRoadmapClassification.MISSING]:
            raise ValueError("missing items must match roadmap assessment")
        return self


def sacred_architecture_items_by_classification(
    assessments: Sequence[SacredArchitectureRoadmapItemAssessment],
) -> dict[SacredArchitectureRoadmapClassification, tuple[str, ...]]:
    return {
        classification: tuple(item.item for item in assessments if item.classification == classification)
        for classification in SacredArchitectureRoadmapClassification
    }


def sacred_architecture_confidence_band(
    score: float,
    *,
    guarded: bool,
) -> SacredArchitectureConfidenceBand:
    if guarded:
        return SacredArchitectureConfidenceBand.GUARDED
    if score >= 0.75:
        return SacredArchitectureConfidenceBand.HIGH
    if score >= 0.5:
        return SacredArchitectureConfidenceBand.MEDIUM
    return SacredArchitectureConfidenceBand.LOW


__all__ = [
    "SacredArchitectureConfidence",
    "SacredArchitectureConfidenceBand",
    "SacredArchitectureFamily",
    "SacredArchitectureOperationKind",
    "SacredArchitectureOperationalGuidance",
    "SacredArchitecturePatternGuidance",
    "SacredArchitectureProvenance",
    "SacredArchitectureReport",
    "SacredArchitectureRoadmapClassification",
    "SacredArchitectureRoadmapItemAssessment",
    "SacredArchitectureSemanticEdge",
    "SacredArchitectureSemanticNode",
    "SacredArchitectureSemanticRelationship",
    "SacredArchitectureSemanticRole",
    "SacredArchitectureValidationFinding",
    "SacredArchitectureValidationSeverity",
    "V8_4_ARCHITECTURE_SCOPE",
    "V8_4_AUTHORITY_BOUNDARY",
    "V8_4_CAPABILITY_ID",
    "sacred_architecture_confidence_band",
    "sacred_architecture_items_by_classification",
]
