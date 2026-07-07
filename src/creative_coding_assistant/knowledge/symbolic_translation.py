"""V8.2 bounded symbolic translation contracts and guidance builders."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.knowledge.creative_distillation import (
    CreativeKnowledgeDistillationReport,
    CreativeKnowledgeRecord,
    build_v8_1_creative_knowledge_distillation,
)
from creative_coding_assistant.orchestration.creative_translation import (
    CreativeTranslation,
    derive_creative_translation,
)
from creative_coding_assistant.orchestration.semantic_motif import SemanticMotifSystem
from creative_coding_assistant.orchestration.symbolic_narrative import (
    SymbolicNarrativePlan,
)

V8_2_CAPABILITY_ID = "v8_2_symbolic_translation_engine"
V8_2_TRANSLATION_SCOPE = (
    "Translate symbolic, artistic, mythopoetic, geometric, ritual, aesthetic, "
    "and conceptual creative intent into structured creative coding guidance."
)
V8_2_AUTHORITY_BOUNDARY = (
    "V8.2 Symbolic Translation Engine is a creative interpretation and "
    "operational guidance surface only. It maps user-visible symbolic cues to "
    "design, motion, audio, runtime, and parameter guidance without claiming "
    "authoritative religious, historical, esoteric, HoloMind, or HOLOiVERSE "
    "interpretation."
)


class SymbolicRoadmapClassification(StrEnum):
    IMPLEMENTED_RUNTIME_BEHAVIOR = "implemented_runtime_behavior"
    PARTIAL_REUSABLE = "partial_reusable"
    METADATA_ONLY = "metadata_only"
    DOCS_ONLY = "docs_only"
    ADVISORY_ONLY = "advisory_only"
    MISSING = "missing"
    RISKY_HITL_REQUIRED = "risky_hitl_required"


class SymbolicIntentDomain(StrEnum):
    SYMBOLIC = "symbolic"
    ARTISTIC = "artistic"
    MYTHOPOETIC = "mythopoetic"
    GEOMETRIC = "geometric"
    RITUAL = "ritual"
    AESTHETIC = "aesthetic"
    CONCEPTUAL = "conceptual"


class SymbolicOperationKind(StrEnum):
    VISUAL_STRUCTURE = "visual_structure"
    MOTION_BEHAVIOR = "motion_behavior"
    AUDIO_MAPPING = "audio_mapping"
    RUNTIME_GUIDANCE = "runtime_guidance"
    PARAMETER_MAPPING = "parameter_mapping"
    COMPOSITION = "composition"
    SAFETY_BOUNDARY = "safety_boundary"


class SymbolicTranslationConfidenceBand(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    GUARDED = "guarded"


class SymbolicTranslationProvenance(BaseModel):
    """Traceable source behind one symbolic translation decision."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    provenance_id: str = Field(min_length=1, max_length=180)
    kind: Literal[
        "request_signal",
        "creative_translation",
        "semantic_motif",
        "symbolic_narrative",
        "v8_1_creative_knowledge",
        "bounded_dictionary",
    ]
    reference: str = Field(min_length=1, max_length=220)
    summary: str = Field(min_length=1, max_length=420)
    confidence_signal: float | None = Field(default=None, ge=0, le=1)


class SymbolicMotifIntentMapping(BaseModel):
    """One motif-to-creative-coding-intent mapping."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    motif_id: str = Field(min_length=1, max_length=80)
    motif_label: str = Field(min_length=1, max_length=120)
    source_terms: tuple[str, ...] = Field(min_length=1, max_length=10)
    intent_domains: tuple[SymbolicIntentDomain, ...] = Field(min_length=1, max_length=7)
    creative_coding_intent: str = Field(min_length=1, max_length=420)
    visual_guidance: tuple[str, ...] = Field(min_length=1, max_length=5)
    motion_guidance: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    audio_guidance: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    runtime_guidance: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    parameter_guidance: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    composition_guidance: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    interpretation_boundary: str = Field(min_length=1, max_length=420)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=12)
    confidence_score: float = Field(ge=0, le=1)


class SymbolicOperationalGuidance(BaseModel):
    """Operational guidance consumable by creative coding generation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    operation_id: str = Field(min_length=1, max_length=140)
    kind: SymbolicOperationKind
    source_motif_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    runtime_families: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    parameter_names: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    implementation_notes: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    constraints: tuple[str, ...] = Field(default_factory=tuple, max_length=8)


class SymbolicRoadmapItemAssessment(BaseModel):
    """Reality-check classification for one V8.2 roadmap item."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    item: str = Field(min_length=1)
    classification: SymbolicRoadmapClassification
    rationale: str = Field(min_length=1, max_length=420)
    hitl_required: bool = False


class SymbolicTranslationConfidence(BaseModel):
    """Confidence posture for a symbolic translation report."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    score: float = Field(ge=0, le=1)
    band: SymbolicTranslationConfidenceBand
    motif_count: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    provenance_count: int = Field(ge=0)
    v8_1_record_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    caveats: tuple[str, ...] = Field(default_factory=tuple, max_length=8)

    @model_validator(mode="after")
    def _band_matches_score(self) -> Self:
        if self.band != _confidence_band(self.score, guarded=bool(self.caveats)):
            raise ValueError("band must match score and caveat posture")
        return self


class SymbolicTranslationReport(BaseModel):
    """Top-level V8.2 symbolic translation report."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    capability_id: Literal["v8_2_symbolic_translation_engine"] = V8_2_CAPABILITY_ID
    translation_scope: str = Field(default=V8_2_TRANSLATION_SCOPE, min_length=1)
    authority_boundary: str = Field(default=V8_2_AUTHORITY_BOUNDARY, min_length=1)
    source_query: str = Field(min_length=1, max_length=520)
    reused_surface_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    motif_mappings: tuple[SymbolicMotifIntentMapping, ...] = Field(
        min_length=1,
        max_length=12,
    )
    operational_guidance: tuple[SymbolicOperationalGuidance, ...] = Field(
        min_length=1,
        max_length=16,
    )
    provenance: tuple[SymbolicTranslationProvenance, ...] = Field(
        min_length=1,
        max_length=24,
    )
    confidence: SymbolicTranslationConfidence
    roadmap_assessment: tuple[SymbolicRoadmapItemAssessment, ...] = Field(
        min_length=1,
        max_length=32,
    )
    implemented_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    partial_reusable_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    metadata_only_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    docs_only_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    advisory_only_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    missing_roadmap_items: tuple[str, ...] = Field(default_factory=tuple)
    risky_hitl_required_items: tuple[str, ...] = Field(default_factory=tuple)
    interpretation_boundaries: tuple[str, ...] = Field(min_length=1, max_length=10)
    unsupported_claim_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    symbolic_translation_contracts_implemented: Literal[True] = True
    symbolic_motif_mapping_implemented: Literal[True] = True
    operational_guidance_implemented: Literal[True] = True
    provenance_confidence_integration_implemented: Literal[True] = True
    live_workflow_runtime_integration_implemented: Literal[False] = False
    preview_runtime_mutation_implemented: Literal[False] = False
    authoritative_esoteric_interpretation_implemented: Literal[False] = False
    comparative_tradition_engine_implemented: Literal[False] = False
    mystical_correspondence_engine_implemented: Literal[False] = False
    hermeneutic_reasoning_engine_implemented: Literal[False] = False
    initiatory_consistency_validation_implemented: Literal[False] = False
    holomind_implemented: Literal[False] = False
    holoiverse_implemented: Literal[False] = False
    v8_3_sacred_geometry_engine_started: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    storage_write_implemented: Literal[False] = False

    @model_validator(mode="after")
    def _report_matches_contract(self) -> Self:
        motif_ids = {mapping.motif_id for mapping in self.motif_mappings}
        if len(motif_ids) != len(self.motif_mappings):
            raise ValueError("motif mapping ids must be unique")
        operation_ids = {item.operation_id for item in self.operational_guidance}
        if len(operation_ids) != len(self.operational_guidance):
            raise ValueError("operation ids must be unique")
        for item in self.operational_guidance:
            if not set(item.source_motif_ids).issubset(motif_ids):
                raise ValueError("operational guidance must reference motif mappings")
        classified = _items_by_classification(self.roadmap_assessment)
        if self.implemented_roadmap_items != classified[
            SymbolicRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR
        ]:
            raise ValueError("implemented items must match roadmap assessment")
        if self.partial_reusable_roadmap_items != classified[
            SymbolicRoadmapClassification.PARTIAL_REUSABLE
        ]:
            raise ValueError("partial reusable items must match roadmap assessment")
        if self.metadata_only_roadmap_items != classified[
            SymbolicRoadmapClassification.METADATA_ONLY
        ]:
            raise ValueError("metadata-only items must match roadmap assessment")
        if self.docs_only_roadmap_items != classified[SymbolicRoadmapClassification.DOCS_ONLY]:
            raise ValueError("docs-only items must match roadmap assessment")
        if self.advisory_only_roadmap_items != classified[
            SymbolicRoadmapClassification.ADVISORY_ONLY
        ]:
            raise ValueError("advisory-only items must match roadmap assessment")
        if self.missing_roadmap_items != classified[SymbolicRoadmapClassification.MISSING]:
            raise ValueError("missing items must match roadmap assessment")
        if self.risky_hitl_required_items != classified[
            SymbolicRoadmapClassification.RISKY_HITL_REQUIRED
        ]:
            raise ValueError("risky HITL items must match roadmap assessment")
        return self


@dataclass(frozen=True)
class _MotifBlueprint:
    label: str
    intent_domains: tuple[SymbolicIntentDomain, ...]
    creative_intent: str
    visual_guidance: tuple[str, ...]
    motion_guidance: tuple[str, ...]
    audio_guidance: tuple[str, ...]
    runtime_guidance: tuple[str, ...]
    parameter_guidance: tuple[str, ...]
    composition_guidance: tuple[str, ...]
    boundary: str


_TOKEN_PATTERN = re.compile(r"[a-z0-9_.+#'-]+")
_UNSUPPORTED_CLAIM_TOKENS = frozenset(
    {
        "absolute",
        "ancient",
        "chakra",
        "cosmic",
        "divine",
        "doctrine",
        "esoteric",
        "gnostic",
        "hidden",
        "prove",
        "sacred",
        "truth",
        "universal",
    }
)
_AMBIGUOUS_SYMBOLIC_TOKENS = frozenset(
    {"archetypal", "deep", "meaningful", "mystery", "mystical", "profound", "symbolic"}
)

_MOTIF_ALIASES: Mapping[str, tuple[str, ...]] = {
    "alchemy": ("transformation", "vessel"),
    "ascent": ("ascent",),
    "axis": ("axis",),
    "breath": ("breath",),
    "breathe": ("breath",),
    "circle": ("center", "circumference"),
    "constellation": ("constellation",),
    "dissolve": ("fragmentation",),
    "dissolving": ("fragmentation",),
    "ember": ("flame", "fragmentation"),
    "embers": ("flame", "fragmentation"),
    "eye": ("eye",),
    "flame": ("flame",),
    "gate": ("gate", "threshold"),
    "grid": ("grid",),
    "labyrinth": ("labyrinth", "threshold"),
    "lattice": ("lattice",),
    "mandala": ("mandala", "center"),
    "mirror": ("mirror",),
    "network": ("network",),
    "orbit": ("orbit",),
    "ouroboros": ("cycle", "transformation"),
    "pearl": ("pearl",),
    "phoenix": ("phoenix", "fragmentation", "reintegration", "flame"),
    "pulse": ("pulse",),
    "rebirth": ("reintegration", "phoenix"),
    "river": ("river",),
    "root": ("root",),
    "seed": ("seed",),
    "sigil": ("symbolic_mark",),
    "spiral": ("spiral",),
    "threshold": ("threshold", "gate"),
    "tree": ("tree", "root"),
    "void": ("void",),
    "wave": ("wave",),
    "waves": ("wave",),
}

_BLUEPRINTS: Mapping[str, _MotifBlueprint] = {
    "ascent": _MotifBlueprint(
        label="Ascent",
        intent_domains=(SymbolicIntentDomain.MYTHOPOETIC, SymbolicIntentDomain.CONCEPTUAL),
        creative_intent="Translate ascent into upward reveal, lightness, and increasing legibility.",
        visual_guidance=("Use vertical gradients, rising forms, or lifting camera paths.",),
        motion_guidance=("Move from low dense states toward higher open states.",),
        audio_guidance=("Raise register, brightness, or density gradually if audio is present.",),
        runtime_guidance=("p5.js", "Three.js", "Tone.js"),
        parameter_guidance=("ascent_progress", "vertical_offset", "brightness_curve"),
        composition_guidance=("Leave visible headroom so upward motion has space to resolve.",),
        boundary="Ascent is treated as a compositional direction, not a spiritual claim.",
    ),
    "axis": _MotifBlueprint(
        label="Axis",
        intent_domains=(SymbolicIntentDomain.GEOMETRIC, SymbolicIntentDomain.CONCEPTUAL),
        creative_intent="Translate axis into orientation, alignment, and relational order.",
        visual_guidance=("Use a clear central or diagonal guide that organizes secondary forms.",),
        motion_guidance=("Let motion rotate around or slide along the axis.",),
        audio_guidance=("Use the axis as a timing split between left/right or high/low layers.",),
        runtime_guidance=("p5.js", "GLSL", "Three.js"),
        parameter_guidance=("axis_angle", "axis_strength", "alignment_bias"),
        composition_guidance=("Keep the axis legible even when detail increases.",),
        boundary="Axis guidance is geometric and operational only.",
    ),
    "breath": _MotifBlueprint(
        label="Breath",
        intent_domains=(SymbolicIntentDomain.AESTHETIC, SymbolicIntentDomain.RITUAL),
        creative_intent="Translate breath into slow expansion, contraction, and perceptible pacing.",
        visual_guidance=("Scale or brighten forms with soft inhale/exhale envelopes.",),
        motion_guidance=("Use sinusoidal easing instead of abrupt cuts.",),
        audio_guidance=("Map amplitude or filter movement to a slow cyclic envelope.",),
        runtime_guidance=("p5.js", "GLSL", "Tone.js"),
        parameter_guidance=("breath_rate", "amplitude", "envelope_depth"),
        composition_guidance=("Preserve quiet intervals between motion peaks.",),
        boundary="Breath is a pacing metaphor, not a claim about ritual efficacy.",
    ),
    "center": _MotifBlueprint(
        label="Center",
        intent_domains=(SymbolicIntentDomain.GEOMETRIC, SymbolicIntentDomain.AESTHETIC),
        creative_intent="Translate center into a stable focal anchor for surrounding behavior.",
        visual_guidance=("Use a readable center point, ring, attractor, or light source.",),
        motion_guidance=("Let secondary elements orbit, converge, or pulse around the center.",),
        audio_guidance=("Use the center as a low-frequency or downbeat anchor if audio is present.",),
        runtime_guidance=("p5.js", "GLSL", "Three.js"),
        parameter_guidance=("center_x", "center_y", "attractor_strength", "ring_count"),
        composition_guidance=("Do not overcrowd the center; protect focal clarity.",),
        boundary="Center is interpreted as composition, not metaphysics.",
    ),
    "circumference": _MotifBlueprint(
        label="Circumference",
        intent_domains=(SymbolicIntentDomain.GEOMETRIC, SymbolicIntentDomain.AESTHETIC),
        creative_intent="Translate circumference into bounded loops, perimeter behavior, and edge rhythm.",
        visual_guidance=("Use rings, circular boundaries, or perimeter particles.",),
        motion_guidance=("Animate traversal around the perimeter with stable angular spacing.",),
        audio_guidance=("Map loop phase to rhythmic accents when audio is present.",),
        runtime_guidance=("p5.js", "GLSL"),
        parameter_guidance=("radius", "ring_count", "angular_phase"),
        composition_guidance=("Let the boundary frame inner transformations without closing them off visually.",),
        boundary="Circumference is a geometric boundary cue only.",
    ),
    "conceptual_field": _MotifBlueprint(
        label="Conceptual Field",
        intent_domains=(SymbolicIntentDomain.CONCEPTUAL, SymbolicIntentDomain.ARTISTIC),
        creative_intent="Translate broad symbolic intent into an inspectable field of visual decisions.",
        visual_guidance=("Choose one primary form language before adding secondary symbolism.",),
        motion_guidance=("Use restrained changes until the user supplies a clearer transformation.",),
        audio_guidance=("Keep audio supportive and abstract unless rhythmic intent is explicit.",),
        runtime_guidance=("p5.js", "GLSL"),
        parameter_guidance=("motif_density", "transition_progress", "contrast_level"),
        composition_guidance=("Prefer one strong motif over many weak symbolic references.",),
        boundary="Broad symbolic language requires user-authored meaning before stronger interpretation.",
    ),
    "constellation": _MotifBlueprint(
        label="Constellation",
        intent_domains=(SymbolicIntentDomain.AESTHETIC, SymbolicIntentDomain.CONCEPTUAL),
        creative_intent="Translate constellation into point networks, relation lines, and emergent grouping.",
        visual_guidance=("Use sparse nodes, lightweight edges, and variable brightness.",),
        motion_guidance=("Let nodes drift slowly while maintaining recognizable clusters.",),
        audio_guidance=("Map node activation to sparse plucks or tones if audio is present.",),
        runtime_guidance=("p5.js", "Three.js"),
        parameter_guidance=("node_count", "connection_radius", "cluster_strength"),
        composition_guidance=("Keep enough negative space for the relational pattern to read.",),
        boundary="Constellation is a visual relation system, not astrology or cosmology.",
    ),
    "cycle": _MotifBlueprint(
        label="Cycle",
        intent_domains=(SymbolicIntentDomain.CONCEPTUAL, SymbolicIntentDomain.MYTHOPOETIC),
        creative_intent="Translate cycle into recurrence, looping transformation, and return.",
        visual_guidance=("Show a form returning changed rather than simply looping identically.",),
        motion_guidance=("Use a loop with a visible phase shift or accumulated trace.",),
        audio_guidance=("Use cyclical rhythmic layers with a bounded variation rule.",),
        runtime_guidance=("p5.js", "GLSL", "Tone.js"),
        parameter_guidance=("cycle_phase", "loop_duration", "variation_amount"),
        composition_guidance=("Make the return point visible so the cycle is understandable.",),
        boundary="Cycle is treated as temporal structure, not doctrine.",
    ),
    "flame": _MotifBlueprint(
        label="Flame",
        intent_domains=(SymbolicIntentDomain.AESTHETIC, SymbolicIntentDomain.MYTHOPOETIC),
        creative_intent="Translate flame into energy, flicker, heat color, and transformation pressure.",
        visual_guidance=("Use warm gradients, noisy edges, sparks, or vertical flicker.",),
        motion_guidance=("Add nonuniform flicker and upward turbulence.",),
        audio_guidance=("Map brightness or spark density to noisy high-frequency accents.",),
        runtime_guidance=("p5.js", "GLSL", "Three.js"),
        parameter_guidance=("palette_shift", "noise_strength", "spark_density"),
        composition_guidance=("Use flame as an accent unless the request makes it primary.",),
        boundary="Flame remains an aesthetic energy cue, not a ritual assertion.",
    ),
    "fragmentation": _MotifBlueprint(
        label="Fragmentation",
        intent_domains=(SymbolicIntentDomain.MYTHOPOETIC, SymbolicIntentDomain.CONCEPTUAL),
        creative_intent="Translate fragmentation into separation, particles, shards, and loss of cohesion.",
        visual_guidance=("Break forms into particles, shards, or cells while preserving source trace.",),
        motion_guidance=("Use dispersal fields, turbulence, or outward drift.",),
        audio_guidance=("Use granular or noisy texture if audio is present.",),
        runtime_guidance=("p5.js", "Three.js", "GLSL"),
        parameter_guidance=("fragmentation_amount", "particle_count", "noise_strength"),
        composition_guidance=("Keep enough original silhouette for the transformation to remain legible.",),
        boundary="Fragmentation expresses a transformation state without claiming symbolic truth.",
    ),
    "gate": _MotifBlueprint(
        label="Gate",
        intent_domains=(SymbolicIntentDomain.RITUAL, SymbolicIntentDomain.CONCEPTUAL),
        creative_intent="Translate gate into a controllable boundary between visual states.",
        visual_guidance=("Use a portal, split plane, aperture, or threshold frame.",),
        motion_guidance=("Animate crossing, opening, or state interpolation.",),
        audio_guidance=("Mark crossing with a restrained shift in rhythm, filter, or density.",),
        runtime_guidance=("p5.js", "Three.js"),
        parameter_guidance=("gate_open", "transition_progress", "interaction_strength"),
        composition_guidance=("Make before and after states visually distinct.",),
        boundary="Gate is an interaction and state-transition metaphor only.",
    ),
    "grid": _MotifBlueprint(
        label="Grid",
        intent_domains=(SymbolicIntentDomain.GEOMETRIC, SymbolicIntentDomain.AESTHETIC),
        creative_intent="Translate grid into ordered sampling, repetition, and controlled variation.",
        visual_guidance=("Use rows, columns, cells, or modular tiles as the base structure.",),
        motion_guidance=("Let variation ripple across cells rather than randomizing all cells at once.",),
        audio_guidance=("Map grid rows, columns, or cells to sequenced rhythmic events.",),
        runtime_guidance=("p5.js", "GLSL"),
        parameter_guidance=("grid_resolution", "cell_size", "variation_amount"),
        composition_guidance=("Reserve contrast for cells that carry meaning or interaction.",),
        boundary="Grid is operational structure, not a universal symbolic code.",
    ),
    "labyrinth": _MotifBlueprint(
        label="Labyrinth",
        intent_domains=(SymbolicIntentDomain.MYTHOPOETIC, SymbolicIntentDomain.GEOMETRIC),
        creative_intent="Translate labyrinth into pathfinding, delay, and intentional traversal.",
        visual_guidance=("Draw winding paths, nested corridors, or line fields with a clear route.",),
        motion_guidance=("Move a camera, particle, or highlight through the path over time.",),
        audio_guidance=("Use call-and-response or delayed echoes to support traversal.",),
        runtime_guidance=("p5.js", "Three.js"),
        parameter_guidance=("path_complexity", "traversal_progress", "turn_density"),
        composition_guidance=("Avoid making the path so dense that the route becomes unreadable.",),
        boundary="Labyrinth is treated as journey and navigation, not an esoteric map.",
    ),
    "lattice": _MotifBlueprint(
        label="Lattice",
        intent_domains=(SymbolicIntentDomain.GEOMETRIC, SymbolicIntentDomain.CONCEPTUAL),
        creative_intent="Translate lattice into repeated relationships and structural interdependence.",
        visual_guidance=("Use repeated cells, linked nodes, or interlocking line systems.",),
        motion_guidance=("Animate local deformation while preserving global structure.",),
        audio_guidance=("Use interlocking rhythmic or harmonic layers if audio is present.",),
        runtime_guidance=("p5.js", "GLSL", "Three.js"),
        parameter_guidance=("grid_resolution", "connection_radius", "deformation_amount"),
        composition_guidance=("Keep lattice scale appropriate to the viewport.",),
        boundary="Lattice is a structural relation cue, not hidden correspondence.",
    ),
    "mandala": _MotifBlueprint(
        label="Mandala",
        intent_domains=(SymbolicIntentDomain.GEOMETRIC, SymbolicIntentDomain.RITUAL),
        creative_intent="Translate mandala into centered radial hierarchy and recurring motifs.",
        visual_guidance=("Build nested rings, radial segments, and a stable focal center.",),
        motion_guidance=("Use slow counter-rotation, pulsing rings, or phased radial reveals.",),
        audio_guidance=("Map rings or radial layers to distinct rhythm or frequency bands.",),
        runtime_guidance=("p5.js", "GLSL", "Tone.js"),
        parameter_guidance=("radial_symmetry", "ring_count", "rotation_step"),
        composition_guidance=("Keep segment count bounded so the structure stays readable.",),
        boundary="Mandala is treated as user-requested visual structure, not doctrine.",
    ),
    "mirror": _MotifBlueprint(
        label="Mirror",
        intent_domains=(SymbolicIntentDomain.CONCEPTUAL, SymbolicIntentDomain.GEOMETRIC),
        creative_intent="Translate mirror into reflection, duality, and reversible relationships.",
        visual_guidance=("Use symmetry axes, reflected layers, or paired forms.",),
        motion_guidance=("Let mirrored elements diverge subtly before returning to alignment.",),
        audio_guidance=("Use stereo or call-response gestures if audio is present.",),
        runtime_guidance=("p5.js", "GLSL", "Three.js"),
        parameter_guidance=("symmetry_axis", "reflection_blend", "phase_offset"),
        composition_guidance=("Make asymmetry intentional so it reads as expressive tension.",),
        boundary="Mirror guidance is formal and experiential, not psychological diagnosis.",
    ),
    "network": _MotifBlueprint(
        label="Network",
        intent_domains=(SymbolicIntentDomain.CONCEPTUAL, SymbolicIntentDomain.GEOMETRIC),
        creative_intent="Translate network into connection, flow, and dependency between elements.",
        visual_guidance=("Use nodes, weighted edges, and flow pulses through relationships.",),
        motion_guidance=("Animate propagation, attraction, or local clustering.",),
        audio_guidance=("Map active connections to rhythmic or tonal events.",),
        runtime_guidance=("p5.js", "Three.js"),
        parameter_guidance=("node_count", "connection_radius", "propagation_speed"),
        composition_guidance=("Use edge density sparingly to avoid visual noise.",),
        boundary="Network is a relational design structure, not a social or mystical claim.",
    ),
    "orbit": _MotifBlueprint(
        label="Orbit",
        intent_domains=(SymbolicIntentDomain.GEOMETRIC, SymbolicIntentDomain.AESTHETIC),
        creative_intent="Translate orbit into stable circular motion around anchors.",
        visual_guidance=("Use orbiting points, rings, cameras, or light paths.",),
        motion_guidance=("Keep orbital speed readable and vary phase offsets between layers.",),
        audio_guidance=("Map orbital phase to panning, filter sweeps, or accents.",),
        runtime_guidance=("p5.js", "Three.js", "GLSL"),
        parameter_guidance=("orbit_speed", "orbit_radius", "phase_offset"),
        composition_guidance=("Avoid too many simultaneous orbits unless hierarchy is clear.",),
        boundary="Orbit is a motion grammar, not a cosmological claim.",
    ),
    "pearl": _MotifBlueprint(
        label="Pearl",
        intent_domains=(SymbolicIntentDomain.AESTHETIC, SymbolicIntentDomain.CONCEPTUAL),
        creative_intent="Translate pearl into refinement, soft luminosity, and protected detail.",
        visual_guidance=("Use small luminous cores, nacre-like gradients, or soft specular highlights.",),
        motion_guidance=("Keep motion slow and reveal detail through glints or gentle rotation.",),
        audio_guidance=("Use sparse bright tones or soft bell-like accents if audio is present.",),
        runtime_guidance=("p5.js", "Three.js", "GLSL"),
        parameter_guidance=("specular_intensity", "core_radius", "glint_phase"),
        composition_guidance=("Use pearl cues as focal highlights rather than full-field noise.",),
        boundary="Pearl is an aesthetic material cue only.",
    ),
    "phoenix": _MotifBlueprint(
        label="Phoenix",
        intent_domains=(SymbolicIntentDomain.MYTHOPOETIC, SymbolicIntentDomain.RITUAL),
        creative_intent="Translate phoenix into a death, ember, and reassembly transformation arc.",
        visual_guidance=("Use dissolution into sparks followed by visible reassembly.",),
        motion_guidance=("Move from collapse or dispersion toward rising coherent motion.",),
        audio_guidance=("Use a low-to-bright build or muted-to-open texture if audio is present.",),
        runtime_guidance=("p5.js", "Three.js", "Tone.js"),
        parameter_guidance=("fragmentation_amount", "reassembly_speed", "spark_density"),
        composition_guidance=("Show before, dissolution, and reformed states distinctly.",),
        boundary="Phoenix is treated as a mythic creative metaphor, not a religious claim.",
    ),
    "pulse": _MotifBlueprint(
        label="Pulse",
        intent_domains=(SymbolicIntentDomain.AESTHETIC, SymbolicIntentDomain.RITUAL),
        creative_intent="Translate pulse into timed emphasis and repeating activation.",
        visual_guidance=("Use periodic brightness, scale, opacity, or density changes.",),
        motion_guidance=("Keep pulse timing coherent with visual or audio structure.",),
        audio_guidance=("Map pulse to amplitude envelope, beat accents, or filter movement.",),
        runtime_guidance=("p5.js", "GLSL", "Tone.js"),
        parameter_guidance=("pulse_rate", "amplitude", "frequency"),
        composition_guidance=("Use pulse to clarify rhythm, not to flash aggressively.",),
        boundary="Pulse is a temporal and sensory cue only.",
    ),
    "reintegration": _MotifBlueprint(
        label="Reintegration",
        intent_domains=(SymbolicIntentDomain.MYTHOPOETIC, SymbolicIntentDomain.CONCEPTUAL),
        creative_intent="Translate reintegration into reassembly, resolution, and recovered coherence.",
        visual_guidance=("Gather dispersed elements back into a recognizable structure.",),
        motion_guidance=("Use attractor fields, easing, or convergent paths.",),
        audio_guidance=("Reduce noise and restore tonal or rhythmic clarity.",),
        runtime_guidance=("p5.js", "Three.js", "GLSL"),
        parameter_guidance=("reassembly_speed", "attractor_strength", "coherence_level"),
        composition_guidance=("Make the final coherent state visually calmer than the fragmented state.",),
        boundary="Reintegration is an operational transformation state only.",
    ),
    "river": _MotifBlueprint(
        label="River",
        intent_domains=(SymbolicIntentDomain.AESTHETIC, SymbolicIntentDomain.CONCEPTUAL),
        creative_intent="Translate river into directional flow, continuity, and channeling.",
        visual_guidance=("Use streamlines, flow fields, or continuous bands.",),
        motion_guidance=("Move particles along coherent vector fields.",),
        audio_guidance=("Use flowing textures or gradual filter sweeps if audio is present.",),
        runtime_guidance=("p5.js", "GLSL"),
        parameter_guidance=("flow_strength", "noise_strength", "current_direction"),
        composition_guidance=("Let flow direction guide the viewer through the frame.",),
        boundary="River is a motion and continuity metaphor only.",
    ),
    "root": _MotifBlueprint(
        label="Root",
        intent_domains=(SymbolicIntentDomain.CONCEPTUAL, SymbolicIntentDomain.GEOMETRIC),
        creative_intent="Translate root into grounded branching and hidden support structure.",
        visual_guidance=("Use branching lines, underground networks, or anchored growth.",),
        motion_guidance=("Grow outward from stable anchors with branching constraints.",),
        audio_guidance=("Use low-frequency anchors or slow rhythmic foundations.",),
        runtime_guidance=("p5.js", "Three.js"),
        parameter_guidance=("branch_depth", "branch_angle", "growth_rate"),
        composition_guidance=("Keep roots visually connected to the main structure.",),
        boundary="Root is a structural metaphor, not biological simulation unless requested.",
    ),
    "seed": _MotifBlueprint(
        label="Seed",
        intent_domains=(SymbolicIntentDomain.CONCEPTUAL, SymbolicIntentDomain.MYTHOPOETIC),
        creative_intent="Translate seed into generative origin and controlled emergence.",
        visual_guidance=("Begin from a compact source that unfolds into structured complexity.",),
        motion_guidance=("Use growth, branching, expansion, or reveal over time.",),
        audio_guidance=("Start sparse and add layers as the visual system grows.",),
        runtime_guidance=("p5.js", "GLSL", "Tone.js"),
        parameter_guidance=("random_seed", "growth_rate", "recursion_depth"),
        composition_guidance=("Make the initial seed visible long enough to establish origin.",),
        boundary="Seed is a generative starting point, not a claim of life or creation.",
    ),
    "spiral": _MotifBlueprint(
        label="Spiral",
        intent_domains=(SymbolicIntentDomain.GEOMETRIC, SymbolicIntentDomain.MYTHOPOETIC),
        creative_intent="Translate spiral into progressive rotation, growth, and transformation.",
        visual_guidance=("Use polar paths, radius growth, or recursive spiral scaffolds.",),
        motion_guidance=("Animate phase, rotation, or radius changes along the curve.",),
        audio_guidance=("Map spiral radius or phase to tempo, filter, or density changes.",),
        runtime_guidance=("p5.js", "GLSL", "Three.js"),
        parameter_guidance=("spiral_tightness", "rotation_step", "recursion_depth"),
        composition_guidance=("Keep turn spacing legible and avoid unbounded recursion.",),
        boundary="Spiral is operational geometry and transformation language only.",
    ),
    "symbolic_mark": _MotifBlueprint(
        label="Symbolic Mark",
        intent_domains=(SymbolicIntentDomain.SYMBOLIC, SymbolicIntentDomain.AESTHETIC),
        creative_intent="Translate a symbolic mark into a user-authored visual glyph or motif anchor.",
        visual_guidance=("Use a crisp glyph-like form built from simple strokes or geometry.",),
        motion_guidance=("Animate reveal, tracing, or modulation without inventing meaning.",),
        audio_guidance=("Use a short accent only if audio is part of the request.",),
        runtime_guidance=("p5.js", "GLSL"),
        parameter_guidance=("stroke_weight", "trace_progress", "symbol_opacity"),
        composition_guidance=("Keep the mark readable and avoid decorative overload.",),
        boundary="Meaning must remain user-authored; the system only operationalizes the mark.",
    ),
    "threshold": _MotifBlueprint(
        label="Threshold",
        intent_domains=(SymbolicIntentDomain.RITUAL, SymbolicIntentDomain.CONCEPTUAL),
        creative_intent="Translate threshold into a visible transition between states.",
        visual_guidance=("Use doors, borders, split fields, apertures, or boundary lines.",),
        motion_guidance=("Let crossing or proximity trigger a state change when interaction is present.",),
        audio_guidance=("Use a restrained sonic shift at the transition point.",),
        runtime_guidance=("p5.js", "Three.js", "Tone.js"),
        parameter_guidance=("transition_progress", "boundary_width", "interaction_strength"),
        composition_guidance=("Make both sides of the threshold visually distinct.",),
        boundary="Threshold is a state-transition metaphor, not an initiatory claim.",
    ),
    "transformation": _MotifBlueprint(
        label="Transformation",
        intent_domains=(SymbolicIntentDomain.CONCEPTUAL, SymbolicIntentDomain.MYTHOPOETIC),
        creative_intent="Translate transformation into observable before, change, and after states.",
        visual_guidance=("Represent the transformation through form, density, palette, or topology change.",),
        motion_guidance=("Use staged interpolation instead of an instant swap.",),
        audio_guidance=("Shift texture or rhythm across the same phases when audio is present.",),
        runtime_guidance=("p5.js", "GLSL", "Three.js", "Tone.js"),
        parameter_guidance=("transition_progress", "morph_amount", "phase_index"),
        composition_guidance=("Preserve enough continuity for the viewer to track what changed.",),
        boundary="Transformation is creative structure, not proof of symbolic truth.",
    ),
    "tree": _MotifBlueprint(
        label="Tree",
        intent_domains=(SymbolicIntentDomain.CONCEPTUAL, SymbolicIntentDomain.GEOMETRIC),
        creative_intent="Translate tree into branching growth, hierarchy, and connected layers.",
        visual_guidance=("Use recursive branching, trunk-to-leaf hierarchy, or network growth.",),
        motion_guidance=("Grow branches over time with bounded recursion depth.",),
        audio_guidance=("Layer tones or rhythms by branch depth if audio is present.",),
        runtime_guidance=("p5.js", "Three.js"),
        parameter_guidance=("branch_depth", "branch_angle", "growth_rate"),
        composition_guidance=("Balance branch density against readable silhouette.",),
        boundary="Tree is a branching structure, not an authoritative symbolic system.",
    ),
    "vessel": _MotifBlueprint(
        label="Vessel",
        intent_domains=(SymbolicIntentDomain.CONCEPTUAL, SymbolicIntentDomain.AESTHETIC),
        creative_intent="Translate vessel into containment, transformation space, and material boundary.",
        visual_guidance=("Use a container form, bowl, field, shell, or bounded volume.",),
        motion_guidance=("Animate interior material separately from the containing edge.",),
        audio_guidance=("Use resonance or filtered texture if audio is present.",),
        runtime_guidance=("p5.js", "Three.js", "GLSL"),
        parameter_guidance=("container_radius", "surface_opacity", "internal_motion"),
        composition_guidance=("Separate container contour from inner contents.",),
        boundary="Vessel is a compositional container only.",
    ),
    "void": _MotifBlueprint(
        label="Void",
        intent_domains=(SymbolicIntentDomain.AESTHETIC, SymbolicIntentDomain.CONCEPTUAL),
        creative_intent="Translate void into negative space, absence, and perceptual contrast.",
        visual_guidance=("Use dark fields, cutouts, gaps, or quiet zones around active elements.",),
        motion_guidance=("Let motion pause, thin out, or orbit around absence.",),
        audio_guidance=("Use silence or sparse low texture if audio is present.",),
        runtime_guidance=("p5.js", "GLSL", "Three.js"),
        parameter_guidance=("density_cutoff", "opacity_floor", "negative_space_ratio"),
        composition_guidance=("Protect the empty region so it remains a visible design decision.",),
        boundary="Void is negative-space guidance, not metaphysical assertion.",
    ),
    "wave": _MotifBlueprint(
        label="Wave",
        intent_domains=(SymbolicIntentDomain.GEOMETRIC, SymbolicIntentDomain.AESTHETIC),
        creative_intent="Translate wave into oscillation, propagation, and interference.",
        visual_guidance=("Use sine fields, ripples, displacement, or banded propagation.",),
        motion_guidance=("Animate phase and amplitude with stable frequency relationships.",),
        audio_guidance=("Map waveform amplitude, frequency, or phase to visuals or sound layers.",),
        runtime_guidance=("p5.js", "GLSL", "Tone.js"),
        parameter_guidance=("amplitude", "frequency", "phase_offset"),
        composition_guidance=("Avoid uniform waves by varying scale or phase at clear hierarchy levels.",),
        boundary="Wave is a signal and motion grammar, not hidden correspondence.",
    ),
}

_ROADMAP_CLASSIFICATIONS: Mapping[str, tuple[SymbolicRoadmapClassification, str, bool]] = {
    "Symbolic Knowledge Graph": (
        SymbolicRoadmapClassification.METADATA_ONLY,
        "Represented as report provenance, motif relationships, and roadmap assessment metadata.",
        False,
    ),
    "Symbol Ontology": (
        SymbolicRoadmapClassification.PARTIAL_REUSABLE,
        "Bounded motif blueprint vocabulary maps symbols to design roles without claiming universality.",
        False,
    ),
    "Archetype Engine": (
        SymbolicRoadmapClassification.PARTIAL_REUSABLE,
        "Reuses V3 Symbolic Narrative Planner archetype metadata when supplied.",
        False,
    ),
    "Myth Engine": (
        SymbolicRoadmapClassification.PARTIAL_REUSABLE,
        "Mythic motifs such as phoenix are translated as creative metaphors only.",
        False,
    ),
    "Symbol Relationship Graph": (
        SymbolicRoadmapClassification.METADATA_ONLY,
        "Relationships are implicit in motif mappings and source evidence, not a graph runtime.",
        False,
    ),
    "Multi-layer Symbol Interpretation": (
        SymbolicRoadmapClassification.PARTIAL_REUSABLE,
        "Produces visual, motion, audio, runtime, composition, and parameter layers.",
        False,
    ),
    "Comparative Tradition Engine": (
        SymbolicRoadmapClassification.RISKY_HITL_REQUIRED,
        "Cross-tradition interpretation would require explicit scope, sources, and human review.",
        True,
    ),
    "Universal Symbol Dictionary": (
        SymbolicRoadmapClassification.RISKY_HITL_REQUIRED,
        "A universal dictionary would overclaim symbolic authority without scoped evidence.",
        True,
    ),
    "Symbol Evolution Engine": (
        SymbolicRoadmapClassification.MISSING,
        "No historical or temporal symbol-evolution runtime is implemented.",
        True,
    ),
    "Symbolic Grammar": (
        SymbolicRoadmapClassification.PARTIAL_REUSABLE,
        "Motif, operation, and guidance models provide a bounded operational grammar.",
        False,
    ),
    "Symbolic Language Parser": (
        SymbolicRoadmapClassification.PARTIAL_REUSABLE,
        "Deterministic token and alias parsing extracts supported motifs from request-visible text.",
        False,
    ),
    "Symbol Composition Engine": (
        SymbolicRoadmapClassification.PARTIAL_REUSABLE,
        "Combines motif mappings into operational guidance without generating artifacts.",
        False,
    ),
    "Symbolic Transformation Engine": (
        SymbolicRoadmapClassification.PARTIAL_REUSABLE,
        "Transformation motifs produce staged visual and parameter guidance only.",
        False,
    ),
    "Symbol Confidence Engine": (
        SymbolicRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR,
        "The report computes confidence from motif, evidence, provenance, and caveat signals.",
        False,
    ),
    "Symbolic Explainability": (
        SymbolicRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR,
        "Motif mappings preserve evidence, provenance, and interpretation boundaries.",
        False,
    ),
    "Mystical Correspondence Engine": (
        SymbolicRoadmapClassification.RISKY_HITL_REQUIRED,
        "Mystical correspondences are explicitly blocked unless scoped and reviewed.",
        True,
    ),
    "Symbolic Narrative Generator": (
        SymbolicRoadmapClassification.PARTIAL_REUSABLE,
        "Existing V3 symbolic narrative plans can feed V8.2 but no new narrative generator is added.",
        False,
    ),
    "Cross-Tradition Symbol Alignment": (
        SymbolicRoadmapClassification.RISKY_HITL_REQUIRED,
        "Cross-tradition alignment is out of scope without sources, consented framing, and HITL.",
        True,
    ),
    "Hermeneutic Reasoning Engine": (
        SymbolicRoadmapClassification.RISKY_HITL_REQUIRED,
        "Hermeneutic authority would exceed creative coding guidance boundaries.",
        True,
    ),
    "Initiatory Consistency Validation": (
        SymbolicRoadmapClassification.RISKY_HITL_REQUIRED,
        "Initiatory claims are not validated; the engine only reports creative transition coherence.",
        True,
    ),
    "Creative Concept Graph": (
        SymbolicRoadmapClassification.METADATA_ONLY,
        "Creative concept relationships are represented through motif and operation records.",
        False,
    ),
    "Cross-Domain Concept Translation": (
        SymbolicRoadmapClassification.PARTIAL_REUSABLE,
        "Visual, audio, runtime, and parameter guidance map concepts across coding domains.",
        False,
    ),
    "Creative Pattern Translation": (
        SymbolicRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR,
        "Supported symbolic patterns translate to concrete visual, motion, and coding guidance.",
        False,
    ),
    "Technical ↔ Artistic Translation": (
        SymbolicRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR,
        "Operational guidance converts artistic motifs into runtime and parameter decisions.",
        False,
    ),
    "Creative Analogy Engine": (
        SymbolicRoadmapClassification.PARTIAL_REUSABLE,
        "Analogies are bounded to motif-to-operation mappings and do not reason freely.",
        False,
    ),
    "Knowledge Distillation Integration": (
        SymbolicRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR,
        "V8.1 creative knowledge records are reused as provenance and confidence signals.",
        False,
    ),
    "Creative Explainability": (
        SymbolicRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR,
        "Prompt lines and report fields expose why guidance was produced and where it is bounded.",
        False,
    ),
    "Demo Concept Generation": (
        SymbolicRoadmapClassification.PARTIAL_REUSABLE,
        "The report generates demo-ready concept guidance but does not create frontend demo flows.",
        False,
    ),
}


def build_v8_2_symbolic_translation_engine(
    query: str,
    *,
    domains: Sequence[CreativeCodingDomain] = (),
    creative_translation: CreativeTranslation | None = None,
    semantic_motif: SemanticMotifSystem | None = None,
    symbolic_narrative: SymbolicNarrativePlan | None = None,
    v8_1_distillation: CreativeKnowledgeDistillationReport | None = None,
) -> SymbolicTranslationReport:
    """Build a bounded V8.2 symbolic translation report without runtime mutation."""

    translation = creative_translation or derive_creative_translation(
        query,
        domains=domains,
    )
    distillation = v8_1_distillation or build_v8_1_creative_knowledge_distillation()
    v8_1_records = _symbolic_v8_1_records(distillation)
    motif_sources = _collect_motif_sources(
        query=query,
        creative_translation=translation,
        semantic_motif=semantic_motif,
        symbolic_narrative=symbolic_narrative,
        v8_1_records=v8_1_records,
    )
    mappings = tuple(
        _build_mapping(motif_id, source_terms, evidence)
        for motif_id, (source_terms, evidence) in motif_sources.items()
    )
    risks = _unsupported_claim_risks(query, semantic_motif=semantic_motif)
    provenance = _build_provenance(
        query=query,
        creative_translation=translation,
        semantic_motif=semantic_motif,
        symbolic_narrative=symbolic_narrative,
        v8_1_records=v8_1_records,
    )
    confidence = _build_confidence(
        mappings=mappings,
        provenance=provenance,
        risks=risks,
        v8_1_records=v8_1_records,
    )
    roadmap = symbolic_translation_roadmap_assessment()
    classified = _items_by_classification(roadmap)
    return SymbolicTranslationReport(
        source_query=_clip(query, 520),
        reused_surface_ids=_reused_surface_ids(
            creative_translation=translation,
            semantic_motif=semantic_motif,
            symbolic_narrative=symbolic_narrative,
            v8_1_records=v8_1_records,
        ),
        motif_mappings=mappings,
        operational_guidance=_build_operational_guidance(mappings),
        provenance=provenance,
        confidence=confidence,
        roadmap_assessment=roadmap,
        implemented_roadmap_items=classified[
            SymbolicRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR
        ],
        partial_reusable_roadmap_items=classified[
            SymbolicRoadmapClassification.PARTIAL_REUSABLE
        ],
        metadata_only_roadmap_items=classified[
            SymbolicRoadmapClassification.METADATA_ONLY
        ],
        docs_only_roadmap_items=classified[SymbolicRoadmapClassification.DOCS_ONLY],
        advisory_only_roadmap_items=classified[
            SymbolicRoadmapClassification.ADVISORY_ONLY
        ],
        missing_roadmap_items=classified[SymbolicRoadmapClassification.MISSING],
        risky_hitl_required_items=classified[
            SymbolicRoadmapClassification.RISKY_HITL_REQUIRED
        ],
        interpretation_boundaries=_interpretation_boundaries(risks),
        unsupported_claim_risks=risks,
        hitl_questions=_hitl_questions(risks),
    )


def symbolic_translation_prompt_lines(
    report: SymbolicTranslationReport,
) -> tuple[str, ...]:
    """Render compact provider-independent symbolic translation guidance."""

    lines = [
        f"Symbolic Translation Engine boundary: {report.authority_boundary}",
        f"Symbolic translation confidence: {report.confidence.band.value} {report.confidence.score:.2f}.",
    ]
    for mapping in report.motif_mappings[:6]:
        lines.append(
            "Symbolic motif mapping: "
            f"{mapping.motif_id}; {mapping.creative_coding_intent}"
        )
        lines.extend(f"{mapping.motif_id} visual: {item}" for item in mapping.visual_guidance[:2])
        lines.extend(f"{mapping.motif_id} motion: {item}" for item in mapping.motion_guidance[:2])
        lines.extend(f"{mapping.motif_id} audio: {item}" for item in mapping.audio_guidance[:1])
        lines.append(f"{mapping.motif_id} boundary: {mapping.interpretation_boundary}")
    for guidance in report.operational_guidance[:8]:
        lines.append(
            "Symbolic operation: "
            f"{guidance.kind.value}; {', '.join(guidance.source_motif_ids)}; "
            f"{' '.join(guidance.guidance[:2])}"
        )
    lines.extend(f"Interpretation boundary: {item}" for item in report.interpretation_boundaries)
    lines.extend(f"Unsupported symbolic claim risk: {item}" for item in report.unsupported_claim_risks)
    lines.extend(f"HITL symbolic question: {item}" for item in report.hitl_questions)
    return tuple(lines[:60])


def symbolic_translation_roadmap_assessment() -> tuple[SymbolicRoadmapItemAssessment, ...]:
    """Return the V8.2 roadmap reality-check assessment."""

    return tuple(
        SymbolicRoadmapItemAssessment(
            item=item,
            classification=classification,
            rationale=rationale,
            hitl_required=hitl_required,
        )
        for item, (classification, rationale, hitl_required) in _ROADMAP_CLASSIFICATIONS.items()
    )


def detect_symbolic_motif_terms(query: str) -> tuple[str, ...]:
    """Return supported motif ids visible in request text."""

    tokens = _ordered_tokens(query)
    motif_ids: list[str] = []
    for token in tokens:
        motif_ids.extend(_MOTIF_ALIASES.get(token, ()))
    if not motif_ids and set(tokens).intersection(_AMBIGUOUS_SYMBOLIC_TOKENS):
        motif_ids.append("conceptual_field")
    return _dedupe(motif_ids)[:12]


def _collect_motif_sources(
    *,
    query: str,
    creative_translation: CreativeTranslation,
    semantic_motif: SemanticMotifSystem | None,
    symbolic_narrative: SymbolicNarrativePlan | None,
    v8_1_records: Sequence[CreativeKnowledgeRecord],
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    sources: dict[str, list[str]] = {}
    evidence: dict[str, list[str]] = {}

    for motif_id in detect_symbolic_motif_terms(query):
        _add_source(sources, evidence, motif_id, motif_id, f"Request-visible motif: {motif_id}.")

    for ref in (
        *creative_translation.symbolic_references,
        *creative_translation.geometric_references,
        *creative_translation.movement_language,
    ):
        for motif_id in _motifs_for_text(ref):
            _add_source(
                sources,
                evidence,
                motif_id,
                ref,
                f"Creative translation signal: {ref}.",
            )

    if semantic_motif is not None:
        for motif in (*semantic_motif.primary_motifs, *semantic_motif.secondary_motifs[:6]):
            _add_source(
                sources,
                evidence,
                str(motif.motif_id),
                str(motif.motif_id),
                f"Semantic motif system signal: {motif.motif_id}.",
            )

    if symbolic_narrative is not None:
        for motif_id in _narrative_motifs(symbolic_narrative):
            _add_source(
                sources,
                evidence,
                motif_id,
                symbolic_narrative.narrative_archetype,
                f"Symbolic narrative archetype: {symbolic_narrative.narrative_archetype}.",
            )

    for record in v8_1_records:
        for tag in (*record.technique_tags, *record.pattern_tags):
            for motif_id in _motifs_for_text(tag):
                _add_source(
                    sources,
                    evidence,
                    motif_id,
                    tag,
                    f"V8.1 distilled knowledge signal: {record.record_id}.",
                )

    if not sources:
        _add_source(
            sources,
            evidence,
            "conceptual_field",
            "symbolic intent",
            "Fallback for broad symbolic request without a supported motif.",
        )

    ordered = sorted(
        sources,
        key=lambda motif_id: (
            -len(evidence[motif_id]),
            list(_BLUEPRINTS).index(motif_id) if motif_id in _BLUEPRINTS else 999,
            motif_id,
        ),
    )[:10]
    return {
        motif_id: (_dedupe(sources[motif_id])[:10], _dedupe(evidence[motif_id])[:12])
        for motif_id in ordered
        if motif_id in _BLUEPRINTS
    }


def _build_mapping(
    motif_id: str,
    source_terms: tuple[str, ...],
    evidence: tuple[str, ...],
) -> SymbolicMotifIntentMapping:
    blueprint = _BLUEPRINTS[motif_id]
    score = min(0.96, 0.48 + 0.08 * len(source_terms) + 0.04 * len(evidence))
    return SymbolicMotifIntentMapping(
        motif_id=motif_id,
        motif_label=blueprint.label,
        source_terms=source_terms,
        intent_domains=blueprint.intent_domains,
        creative_coding_intent=blueprint.creative_intent,
        visual_guidance=blueprint.visual_guidance,
        motion_guidance=blueprint.motion_guidance,
        audio_guidance=blueprint.audio_guidance,
        runtime_guidance=blueprint.runtime_guidance,
        parameter_guidance=blueprint.parameter_guidance,
        composition_guidance=blueprint.composition_guidance,
        interpretation_boundary=blueprint.boundary,
        evidence=evidence,
        confidence_score=round(score, 2),
    )


def _build_operational_guidance(
    mappings: Sequence[SymbolicMotifIntentMapping],
) -> tuple[SymbolicOperationalGuidance, ...]:
    motif_ids = tuple(mapping.motif_id for mapping in mappings)
    visual = _collect_guidance(mappings, "visual_guidance")
    motion = _collect_guidance(mappings, "motion_guidance")
    audio = _collect_guidance(mappings, "audio_guidance")
    composition = _collect_guidance(mappings, "composition_guidance")
    runtimes = _dedupe(item for mapping in mappings for item in mapping.runtime_guidance)
    parameters = _dedupe(item for mapping in mappings for item in mapping.parameter_guidance)[:12]
    constraints = (
        "Treat symbolic language as user-authored creative direction.",
        "Do not add religious, historical, esoteric, HoloMind, or HOLOiVERSE claims.",
        "Keep mappings inspectable as visual, motion, audio, runtime, and parameter choices.",
    )
    operations = (
        SymbolicOperationalGuidance(
            operation_id="symbolic_translation::visual_structure",
            kind=SymbolicOperationKind.VISUAL_STRUCTURE,
            source_motif_ids=motif_ids,
            guidance=visual or ("Select one clear visual structure before adding symbolic layers.",),
            runtime_families=runtimes,
            parameter_names=parameters,
            implementation_notes=("Use motif mappings as scaffold guidance before style embellishment.",),
            constraints=constraints,
        ),
        SymbolicOperationalGuidance(
            operation_id="symbolic_translation::motion_behavior",
            kind=SymbolicOperationKind.MOTION_BEHAVIOR,
            source_motif_ids=motif_ids,
            guidance=motion or ("Use restrained temporal variation until transformation is explicit.",),
            runtime_families=runtimes,
            parameter_names=parameters,
            implementation_notes=("Tie motion changes to motif state changes.",),
            constraints=constraints,
        ),
        SymbolicOperationalGuidance(
            operation_id="symbolic_translation::audio_mapping",
            kind=SymbolicOperationKind.AUDIO_MAPPING,
            source_motif_ids=motif_ids,
            guidance=audio or ("Keep audio abstract and supportive unless the request names rhythm or sound.",),
            runtime_families=tuple(runtime for runtime in runtimes if runtime in {"Tone.js", "p5.js", "GLSL"}),
            parameter_names=tuple(
                parameter
                for parameter in parameters
                if parameter in {"amplitude", "frequency", "pulse_rate", "phase_offset", "breath_rate"}
            ),
            implementation_notes=("Keep browser audio behind explicit user interaction when audio is generated.",),
            constraints=constraints,
        ),
        SymbolicOperationalGuidance(
            operation_id="symbolic_translation::parameter_mapping",
            kind=SymbolicOperationKind.PARAMETER_MAPPING,
            source_motif_ids=motif_ids,
            guidance=("Expose motif-driving parameters with stable names and bounded ranges.",),
            runtime_families=runtimes,
            parameter_names=parameters,
            implementation_notes=("Use parameters as implementation handles, not as symbolic truth claims.",),
            constraints=constraints,
        ),
        SymbolicOperationalGuidance(
            operation_id="symbolic_translation::composition",
            kind=SymbolicOperationKind.COMPOSITION,
            source_motif_ids=motif_ids,
            guidance=composition or ("Preserve one dominant compositional read.",),
            runtime_families=runtimes,
            parameter_names=parameters,
            implementation_notes=("Balance motif density against legibility.",),
            constraints=constraints,
        ),
        SymbolicOperationalGuidance(
            operation_id="symbolic_translation::safety_boundary",
            kind=SymbolicOperationKind.SAFETY_BOUNDARY,
            source_motif_ids=motif_ids,
            guidance=tuple(mapping.interpretation_boundary for mapping in mappings[:8]),
            runtime_families=(),
            parameter_names=(),
            implementation_notes=("Ask HITL before crossing into tradition-specific interpretation.",),
            constraints=constraints,
        ),
    )
    return operations


def _build_provenance(
    *,
    query: str,
    creative_translation: CreativeTranslation,
    semantic_motif: SemanticMotifSystem | None,
    symbolic_narrative: SymbolicNarrativePlan | None,
    v8_1_records: Sequence[CreativeKnowledgeRecord],
) -> tuple[SymbolicTranslationProvenance, ...]:
    provenance = [
        SymbolicTranslationProvenance(
            provenance_id="symbolic_translation::request",
            kind="request_signal",
            reference="assistant_request.query",
            summary=_clip(query, 420),
            confidence_signal=None,
        ),
        SymbolicTranslationProvenance(
            provenance_id="symbolic_translation::creative_translation",
            kind="creative_translation",
            reference="orchestration.creative_translation",
            summary=(
                "Reused bounded creative translation signals for symbols, geometry, "
                "movement, modality, runtime, and constraints."
            ),
            confidence_signal=0.78,
        ),
        SymbolicTranslationProvenance(
            provenance_id="symbolic_translation::bounded_dictionary",
            kind="bounded_dictionary",
            reference="knowledge.symbolic_translation._BLUEPRINTS",
            summary="Used a scoped motif dictionary for operational creative coding mappings.",
            confidence_signal=0.72,
        ),
    ]
    if semantic_motif is not None:
        provenance.append(
            SymbolicTranslationProvenance(
                provenance_id="symbolic_translation::semantic_motif",
                kind="semantic_motif",
                reference="orchestration.semantic_motif",
                summary="Reused V3 Semantic Motif Engine hierarchy, recurrence, and claim-risk signals.",
                confidence_signal=0.8,
            )
        )
    if symbolic_narrative is not None:
        provenance.append(
            SymbolicTranslationProvenance(
                provenance_id="symbolic_translation::symbolic_narrative",
                kind="symbolic_narrative",
                reference="orchestration.symbolic_narrative",
                summary="Reused V3 Symbolic Narrative Planner archetype and phase metadata.",
                confidence_signal=0.78,
            )
        )
    provenance.extend(
        SymbolicTranslationProvenance(
            provenance_id=f"symbolic_translation::{record.record_id}",
            kind="v8_1_creative_knowledge",
            reference=record.record_id,
            summary=record.summary,
            confidence_signal=record.confidence.score,
        )
        for record in v8_1_records[:8]
    )
    return tuple(provenance[:24])


def _build_confidence(
    *,
    mappings: Sequence[SymbolicMotifIntentMapping],
    provenance: Sequence[SymbolicTranslationProvenance],
    risks: Sequence[str],
    v8_1_records: Sequence[CreativeKnowledgeRecord],
) -> SymbolicTranslationConfidence:
    evidence_count = sum(len(mapping.evidence) for mapping in mappings)
    provenance_bonus = min(0.18, 0.03 * len(provenance))
    motif_bonus = min(0.24, 0.04 * len(mappings))
    evidence_bonus = min(0.24, 0.02 * evidence_count)
    v8_1_bonus = min(0.12, 0.04 * len(v8_1_records))
    risk_penalty = min(0.24, 0.08 * len(risks))
    score = max(0.05, min(0.96, 0.32 + provenance_bonus + motif_bonus + evidence_bonus + v8_1_bonus - risk_penalty))
    caveats = tuple(risks[:4])
    return SymbolicTranslationConfidence(
        score=round(score, 2),
        band=_confidence_band(score, guarded=bool(caveats)),
        motif_count=len(mappings),
        evidence_count=evidence_count,
        provenance_count=len(provenance),
        v8_1_record_ids=tuple(record.record_id for record in v8_1_records[:10]),
        caveats=caveats,
    )


def _symbolic_v8_1_records(
    report: CreativeKnowledgeDistillationReport,
) -> tuple[CreativeKnowledgeRecord, ...]:
    tags = {"motif_translation", "operational_translation", "recursive_geometry", "morphogenesis_seed"}
    return tuple(
        record
        for record in report.records
        if tags.intersection(record.technique_tags)
        or tags.intersection(record.pattern_tags)
        or "symbolic" in record.title.lower()
        or "symbolic" in record.summary.lower()
    )


def _unsupported_claim_risks(
    query: str,
    *,
    semantic_motif: SemanticMotifSystem | None,
) -> tuple[str, ...]:
    tokens = _tokens(query)
    risks = [
        f"Request includes '{token}', which must remain user-authored creative framing."
        for token in sorted(tokens.intersection(_UNSUPPORTED_CLAIM_TOKENS))
    ]
    if semantic_motif is not None:
        risks.extend(semantic_motif.unsupported_symbolic_claims)
    return _dedupe(risks)[:8]


def _interpretation_boundaries(risks: Sequence[str]) -> tuple[str, ...]:
    boundaries = [
        "Use symbolic motifs as creative design inputs, not authoritative interpretation.",
        "Do not claim religious, historical, esoteric, HoloMind, or HOLOiVERSE truth.",
        "Do not start V8.3 Sacred Geometry Engine or mutate preview runtime.",
        "Ask HITL before comparative, tradition-specific, mystical, or initiatory claims.",
    ]
    if risks:
        boundaries.append("Unsupported claim risks require bounded wording or HITL review.")
    return tuple(boundaries)


def _hitl_questions(risks: Sequence[str]) -> tuple[str, ...]:
    if not risks:
        return ()
    return (
        "Which symbolic meanings are explicitly user-authored and safe to preserve as creative framing?",
        "Should tradition-specific or esoteric language be removed, quoted from the user, or reviewed by HITL?",
    )


def _reused_surface_ids(
    *,
    creative_translation: CreativeTranslation,
    semantic_motif: SemanticMotifSystem | None,
    symbolic_narrative: SymbolicNarrativePlan | None,
    v8_1_records: Sequence[CreativeKnowledgeRecord],
) -> tuple[str, ...]:
    del creative_translation
    surfaces = ["v3_creative_translation"]
    if semantic_motif is not None:
        surfaces.append("v3_semantic_motif")
    if symbolic_narrative is not None:
        surfaces.append("v3_symbolic_narrative")
    if v8_1_records:
        surfaces.append("v8_1_creative_knowledge_distillation")
    return tuple(surfaces)


def _narrative_motifs(plan: SymbolicNarrativePlan) -> tuple[str, ...]:
    mapping = {
        "death_and_rebirth": ("phoenix", "fragmentation", "reintegration"),
        "descent_and_return": ("threshold", "ascent"),
        "emergence_from_chaos": ("seed", "void", "reintegration"),
        "initiation": ("threshold", "gate"),
        "ascent": ("ascent",),
        "dissolution_and_reintegration": ("fragmentation", "reintegration"),
        "expansion_from_seed_to_cosmos": ("seed", "constellation"),
        "fragmentation_and_recomposition": ("fragmentation", "reintegration"),
        "threshold_crossing": ("threshold", "gate"),
        "spiral_transformation": ("spiral", "transformation"),
        "mirror_reflection_journey": ("mirror",),
        "dark_to_light_transformation": ("void", "flame", "transformation"),
        "symbolic_vignette": ("conceptual_field",),
    }
    return mapping.get(plan.narrative_archetype, ("conceptual_field",))


def _motifs_for_text(value: str) -> tuple[str, ...]:
    motif_ids: list[str] = []
    for token in _ordered_tokens(value):
        motif_ids.extend(_MOTIF_ALIASES.get(token, ()))
        if token in _BLUEPRINTS:
            motif_ids.append(token)
    return _dedupe(motif_ids)


def _add_source(
    sources: dict[str, list[str]],
    evidence: dict[str, list[str]],
    motif_id: str,
    source: str,
    evidence_line: str,
) -> None:
    if motif_id not in _BLUEPRINTS:
        return
    sources.setdefault(motif_id, []).append(source)
    evidence.setdefault(motif_id, []).append(evidence_line)


def _collect_guidance(
    mappings: Sequence[SymbolicMotifIntentMapping],
    field_name: str,
) -> tuple[str, ...]:
    return _dedupe(
        item
        for mapping in mappings
        for item in getattr(mapping, field_name)
    )[:8]


def _items_by_classification(
    assessments: Sequence[SymbolicRoadmapItemAssessment],
) -> dict[SymbolicRoadmapClassification, tuple[str, ...]]:
    return {
        classification: tuple(
            item.item for item in assessments if item.classification is classification
        )
        for classification in SymbolicRoadmapClassification
    }


def _confidence_band(
    score: float,
    *,
    guarded: bool,
) -> SymbolicTranslationConfidenceBand:
    if guarded:
        return SymbolicTranslationConfidenceBand.GUARDED
    if score >= 0.75:
        return SymbolicTranslationConfidenceBand.HIGH
    if score >= 0.5:
        return SymbolicTranslationConfidenceBand.MEDIUM
    return SymbolicTranslationConfidenceBand.LOW


def _tokens(value: str) -> frozenset[str]:
    return frozenset(_TOKEN_PATTERN.findall(_normalize(value)))


def _ordered_tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(_normalize(value)))


def _normalize(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").split())


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _clip(value: str, limit: int) -> str:
    normalized = " ".join(value.strip().split())
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "..."


__all__ = [
    "SymbolicIntentDomain",
    "SymbolicMotifIntentMapping",
    "SymbolicOperationKind",
    "SymbolicOperationalGuidance",
    "SymbolicRoadmapClassification",
    "SymbolicRoadmapItemAssessment",
    "SymbolicTranslationConfidence",
    "SymbolicTranslationConfidenceBand",
    "SymbolicTranslationProvenance",
    "SymbolicTranslationReport",
    "V8_2_AUTHORITY_BOUNDARY",
    "V8_2_CAPABILITY_ID",
    "V8_2_TRANSLATION_SCOPE",
    "build_v8_2_symbolic_translation_engine",
    "detect_symbolic_motif_terms",
    "symbolic_translation_prompt_lines",
    "symbolic_translation_roadmap_assessment",
]
