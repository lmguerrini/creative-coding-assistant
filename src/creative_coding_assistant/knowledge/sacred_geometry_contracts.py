"""Typed contracts for the V8.3 sacred geometry engine."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

V8_3_CAPABILITY_ID = "v8_3_sacred_geometry_engine"
V8_3_GEOMETRY_SCOPE = (
    "Translate geometric, mathematical, harmonic, recursive, fractal, "
    "morphogenetic, and proportion-based creative intent into structured "
    "creative coding guidance and reusable generation contracts."
)
V8_3_AUTHORITY_BOUNDARY = (
    "V8.3 Sacred Geometry and Sacred Mathematics Engine provides bounded "
    "creative and mathematical guidance only. It maps user-visible geometry "
    "and proportion cues to inspectable structures, algorithms, parameters, "
    "motion, light, and audio guidance without claiming metaphysical proof, "
    "tradition authority, ritual efficacy, HoloMind, HOLOiVERSE, external DCC "
    "integration, or V8.4 architecture/reverse-engineering behavior."
)


class SacredGeometryFamily(StrEnum):
    PROPORTION = "proportion"
    RADIAL = "radial"
    POLYGONAL = "polygonal"
    TESSELLATION = "tessellation"
    RECURSIVE = "recursive"
    FRACTAL = "fractal"
    GROWTH = "growth"
    FIELD = "field"
    MORPHOGENESIS = "morphogenesis"
    CELLULAR = "cellular"
    PARTICLE = "particle"
    THREE_DIMENSIONAL = "three_dimensional"
    CULTURAL_GEOMETRY = "cultural_geometry"
    COMPOSITIONAL_SPACE = "compositional_space"


class SacredGeometryOperationKind(StrEnum):
    STRUCTURE = "structure"
    PARAMETERIZATION = "parameterization"
    RECURSION = "recursion"
    MORPHOGENESIS = "morphogenesis"
    MOTION_MAPPING = "motion_mapping"
    COLOR_LIGHT_MAPPING = "color_light_mapping"
    AUDIO_HARMONIC_MAPPING = "audio_harmonic_mapping"
    ARCHITECTURAL_LAYOUT_MAPPING = "architectural_layout_mapping"
    RITUAL_PACING_MAPPING = "ritual_pacing_mapping"
    VALIDATION = "validation"
    SAFETY_BOUNDARY = "safety_boundary"


class SacredGeometryRoadmapClassification(StrEnum):
    IMPLEMENTED_RUNTIME_BEHAVIOR = "implemented_runtime_behavior"
    REUSED_EXISTING_RUNTIME = "reused_existing_runtime"
    PARTIAL_REUSABLE = "partial_reusable"
    ADVISORY_ONLY = "advisory_only"
    PRODUCT_HITL_REQUIRED = "product_hitl_required"
    LATER_V8_BOUNDARY = "later_v8_boundary"
    MISSING = "missing"


class SacredGeometryConfidenceBand(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    GUARDED = "guarded"


class SacredGeometryValidationSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    HITL_REQUIRED = "hitl_required"


class SacredGeometryProvenance(BaseModel):
    """Traceable source behind one V8.3 geometry guidance decision."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    provenance_id: str = Field(min_length=1, max_length=180)
    kind: Literal[
        "request_signal",
        "creative_translation",
        "v8_1_creative_knowledge",
        "v8_2_symbolic_translation",
        "bounded_geometry_catalog",
    ]
    reference: str = Field(min_length=1, max_length=240)
    summary: str = Field(min_length=1, max_length=520)
    confidence_signal: float | None = Field(default=None, ge=0, le=1)


class SacredGeometryPatternGuidance(BaseModel):
    """Reusable generation contract for one geometry or mathematics pattern."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    pattern_id: str = Field(min_length=1, max_length=100)
    label: str = Field(min_length=1, max_length=140)
    family: SacredGeometryFamily
    source_terms: tuple[str, ...] = Field(min_length=1, max_length=12)
    taxonomy_path: tuple[str, ...] = Field(min_length=2, max_length=6)
    creative_intent: str = Field(min_length=1, max_length=520)
    structure_guidance: tuple[str, ...] = Field(min_length=1, max_length=7)
    algorithm_recommendations: tuple[str, ...] = Field(min_length=1, max_length=7)
    mathematical_parameters: tuple[str, ...] = Field(min_length=1, max_length=10)
    motion_mappings: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    color_light_mappings: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    audio_harmonic_mappings: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    runtime_families: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    implementation_notes: tuple[str, ...] = Field(default_factory=tuple, max_length=7)
    boundary: str = Field(min_length=1, max_length=420)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=14)
    confidence_score: float = Field(ge=0, le=1)


class SacredGeometryOperationalGuidance(BaseModel):
    """Provider-independent operational guidance consumable by generation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    operation_id: str = Field(min_length=1, max_length=160)
    kind: SacredGeometryOperationKind
    source_pattern_ids: tuple[str, ...] = Field(min_length=1, max_length=16)
    guidance: tuple[str, ...] = Field(min_length=1, max_length=9)
    parameter_names: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    runtime_families: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    implementation_notes: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    constraints: tuple[str, ...] = Field(default_factory=tuple, max_length=8)


class SacredGeometryValidationFinding(BaseModel):
    """Deterministic validation finding for generated geometry guidance."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    finding_id: str = Field(min_length=1, max_length=160)
    severity: SacredGeometryValidationSeverity
    summary: str = Field(min_length=1, max_length=420)
    action: str = Field(min_length=1, max_length=420)


class SacredGeometryRoadmapItemAssessment(BaseModel):
    """Reality-check classification for one V8.3 roadmap item."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    item: str = Field(min_length=1)
    classification: SacredGeometryRoadmapClassification
    rationale: str = Field(min_length=1, max_length=520)
    action_required_before_v8_4: bool = False
    hitl_required: bool = False


class SacredGeometryConfidence(BaseModel):
    """Confidence posture for a V8.3 geometry report."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    score: float = Field(ge=0, le=1)
    band: SacredGeometryConfidenceBand
    pattern_count: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    provenance_count: int = Field(ge=0)
    v8_1_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    v8_2_motif_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    caveats: tuple[str, ...] = Field(default_factory=tuple, max_length=8)

    @model_validator(mode="after")
    def _band_matches_score(self) -> Self:
        if self.band != sacred_geometry_confidence_band(self.score, guarded=bool(self.caveats)):
            raise ValueError("band must match score and caveat posture")
        return self


class SacredGeometryReport(BaseModel):
    """Top-level V8.3 sacred geometry and sacred mathematics report."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    capability_id: Literal["v8_3_sacred_geometry_engine"] = V8_3_CAPABILITY_ID
    geometry_scope: str = Field(default=V8_3_GEOMETRY_SCOPE, min_length=1)
    authority_boundary: str = Field(default=V8_3_AUTHORITY_BOUNDARY, min_length=1)
    source_query: str = Field(min_length=1, max_length=620)
    reused_surface_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    pattern_guidance: tuple[SacredGeometryPatternGuidance, ...] = Field(min_length=1, max_length=16)
    operational_guidance: tuple[SacredGeometryOperationalGuidance, ...] = Field(min_length=1, max_length=14)
    validation_findings: tuple[SacredGeometryValidationFinding, ...] = Field(min_length=1, max_length=12)
    provenance: tuple[SacredGeometryProvenance, ...] = Field(min_length=1, max_length=28)
    confidence: SacredGeometryConfidence
    roadmap_assessment: tuple[SacredGeometryRoadmapItemAssessment, ...] = Field(min_length=1, max_length=40)
    implemented_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    reused_existing_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    partial_reusable_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    product_hitl_required_items: tuple[str, ...] = Field(default_factory=tuple)
    later_v8_boundary_items: tuple[str, ...] = Field(default_factory=tuple)
    missing_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    interpretation_boundaries: tuple[str, ...] = Field(min_length=1, max_length=10)
    unsupported_claim_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    geometry_contracts_implemented: Literal[True] = True
    mathematical_parameter_guidance_implemented: Literal[True] = True
    geometry_motion_light_audio_mapping_implemented: Literal[True] = True
    provenance_confidence_integration_implemented: Literal[True] = True
    preview_runtime_mutation_implemented: Literal[False] = False
    demo_asset_generation_implemented: Literal[False] = False
    external_dcc_integration_implemented: Literal[False] = False
    v8_4_architecture_engine_started: Literal[False] = False
    v8_5_narrative_engine_started: Literal[False] = False
    v8_6_immersive_composer_started: Literal[False] = False
    metaphysical_proof_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    storage_write_implemented: Literal[False] = False

    @model_validator(mode="after")
    def _report_matches_contract(self) -> Self:
        pattern_ids = {item.pattern_id for item in self.pattern_guidance}
        if len(pattern_ids) != len(self.pattern_guidance):
            raise ValueError("pattern ids must be unique")
        operation_ids = {item.operation_id for item in self.operational_guidance}
        if len(operation_ids) != len(self.operational_guidance):
            raise ValueError("operation ids must be unique")
        for item in self.operational_guidance:
            if not set(item.source_pattern_ids).issubset(pattern_ids):
                raise ValueError("operational guidance must reference known patterns")
        classified = sacred_geometry_items_by_classification(self.roadmap_assessment)
        if self.implemented_roadmap_items != classified[
            SacredGeometryRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR
        ]:
            raise ValueError("implemented items must match roadmap assessment")
        if self.reused_existing_roadmap_items != classified[
            SacredGeometryRoadmapClassification.REUSED_EXISTING_RUNTIME
        ]:
            raise ValueError("reused items must match roadmap assessment")
        if self.partial_reusable_roadmap_items != classified[SacredGeometryRoadmapClassification.PARTIAL_REUSABLE]:
            raise ValueError("partial reusable items must match roadmap assessment")
        if self.advisory_only_roadmap_items != classified[SacredGeometryRoadmapClassification.ADVISORY_ONLY]:
            raise ValueError("advisory-only items must match roadmap assessment")
        if self.product_hitl_required_items != classified[SacredGeometryRoadmapClassification.PRODUCT_HITL_REQUIRED]:
            raise ValueError("product HITL items must match roadmap assessment")
        if self.later_v8_boundary_items != classified[SacredGeometryRoadmapClassification.LATER_V8_BOUNDARY]:
            raise ValueError("later V8 items must match roadmap assessment")
        if self.missing_roadmap_items != classified[SacredGeometryRoadmapClassification.MISSING]:
            raise ValueError("missing items must match roadmap assessment")
        return self



def sacred_geometry_items_by_classification(
    assessments: Sequence[SacredGeometryRoadmapItemAssessment],
) -> dict[SacredGeometryRoadmapClassification, tuple[str, ...]]:
    return {
        classification: tuple(item.item for item in assessments if item.classification == classification)
        for classification in SacredGeometryRoadmapClassification
    }


def sacred_geometry_confidence_band(
    score: float,
    *,
    guarded: bool,
) -> SacredGeometryConfidenceBand:
    if guarded:
        return SacredGeometryConfidenceBand.GUARDED
    if score >= 0.75:
        return SacredGeometryConfidenceBand.HIGH
    if score >= 0.5:
        return SacredGeometryConfidenceBand.MEDIUM
    return SacredGeometryConfidenceBand.LOW


__all__ = [
    "SacredGeometryConfidence",
    "SacredGeometryConfidenceBand",
    "SacredGeometryFamily",
    "SacredGeometryOperationKind",
    "SacredGeometryOperationalGuidance",
    "SacredGeometryPatternGuidance",
    "SacredGeometryProvenance",
    "SacredGeometryReport",
    "SacredGeometryRoadmapClassification",
    "SacredGeometryRoadmapItemAssessment",
    "SacredGeometryValidationFinding",
    "SacredGeometryValidationSeverity",
    "V8_3_AUTHORITY_BOUNDARY",
    "V8_3_CAPABILITY_ID",
    "V8_3_GEOMETRY_SCOPE",
    "sacred_geometry_confidence_band",
    "sacred_geometry_items_by_classification",
]
