"""V8.5 bounded mythopoetic narrative guidance engine."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.knowledge.creative_distillation import (
    CreativeKnowledgeDistillationReport,
    CreativeKnowledgeRecord,
    build_v8_1_creative_knowledge_distillation,
)
from creative_coding_assistant.knowledge.mythopoetic_narrative_catalog import (
    ARCHITECTURE_TO_NARRATIVE,
    GEOMETRY_TO_NARRATIVE,
    NARRATIVE_ALIASES,
    PATTERN_ROWS,
    ROADMAP_CLASSIFICATION_ROWS,
    SYMBOL_TO_NARRATIVE,
    UNSUPPORTED_NARRATIVE_CLAIM_TOKENS,
)
from creative_coding_assistant.knowledge.mythopoetic_narrative_contracts import (
    MythopoeticNarrativeConfidence,
    MythopoeticNarrativeFamily,
    MythopoeticNarrativePatternGuidance,
    MythopoeticNarrativeProvenance,
    MythopoeticNarrativeReport,
    MythopoeticNarrativeRoadmapClassification,
    MythopoeticNarrativeRoadmapItemAssessment,
    mythopoetic_narrative_confidence_band,
    mythopoetic_narrative_items_by_classification,
)
from creative_coding_assistant.knowledge.mythopoetic_narrative_guidance import (
    build_mythopoetic_narrative_operational_guidance,
    build_mythopoetic_narrative_scene_sequence,
    build_mythopoetic_narrative_symbol_edges,
    build_mythopoetic_narrative_symbol_nodes,
    build_mythopoetic_narrative_validation_findings,
)
from creative_coding_assistant.knowledge.sacred_architecture import (
    SacredArchitectureReport,
    build_v8_4_sacred_architecture_engine,
)
from creative_coding_assistant.knowledge.sacred_geometry import (
    SacredGeometryReport,
    build_v8_3_sacred_geometry_engine,
)
from creative_coding_assistant.knowledge.symbolic_translation import (
    SymbolicTranslationReport,
    build_v8_2_symbolic_translation_engine,
)
from creative_coding_assistant.orchestration.creative_translation import (
    CreativeTranslation,
    derive_creative_translation,
)
from creative_coding_assistant.orchestration.symbolic_narrative import (
    SymbolicNarrativePlan,
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9_.+#'-]+")
_LATER_V8_PATTERNS = (
    (re.compile(r"\bimmersive composer\b"), "immersive composer"),
    (re.compile(r"\blive preview\b"), "live preview"),
    (re.compile(r"\bpreview runtime\b"), "preview runtime"),
    (re.compile(r"\bshowcase system\b"), "showcase system"),
    (re.compile(r"\bdemo mode\b"), "demo mode"),
    (re.compile(r"\bhologenesis os\b"), "hologenesis os"),
    (re.compile(r"\bholomind\b"), "holomind"),
    (re.compile(r"\bholoiverse\b"), "holoiverse"),
    (re.compile(r"\b(?:unity|unreal|blender|houdini|touchdesigner|dcc)\b"), "external DCC integration"),
)
_SYMBOLIC_NARRATIVE_TO_PATTERN = {
    "descent_and_return": ("descent_return",),
    "death_and_rebirth": ("death_rebirth_cycle",),
    "emergence_from_chaos": ("seed_emergence", "descent_return"),
    "initiation": ("threshold_initiation_arc",),
    "ascent": ("heroic_cycle", "threshold_initiation_arc"),
    "dissolution_and_reintegration": ("death_rebirth_cycle",),
    "expansion_from_seed_to_cosmos": ("seed_emergence",),
    "fragmentation_and_recomposition": ("death_rebirth_cycle",),
    "threshold_crossing": ("threshold_initiation_arc",),
    "spiral_transformation": ("archetypal_transformation", "threshold_initiation_arc"),
    "mirror_reflection_journey": ("mirror_revelation",),
    "dark_to_light_transformation": ("descent_return", "heroic_cycle"),
    "symbolic_vignette": ("demo_project_story",),
}


def build_v8_5_mythopoetic_narrative_engine(
    query: str,
    *,
    domains: Sequence[CreativeCodingDomain] = (),
    creative_translation: CreativeTranslation | None = None,
    symbolic_narrative: SymbolicNarrativePlan | None = None,
    v8_1_distillation: CreativeKnowledgeDistillationReport | None = None,
    v8_2_symbolic_translation: SymbolicTranslationReport | None = None,
    v8_3_sacred_geometry: SacredGeometryReport | None = None,
    v8_4_sacred_architecture: SacredArchitectureReport | None = None,
) -> MythopoeticNarrativeReport:
    """Build a bounded V8.5 mythopoetic narrative report without runtime mutation."""

    translation = creative_translation or derive_creative_translation(query, domains=domains)
    distillation = v8_1_distillation or build_v8_1_creative_knowledge_distillation()
    symbolic = v8_2_symbolic_translation or build_v8_2_symbolic_translation_engine(
        query,
        domains=domains,
        creative_translation=translation,
        semantic_motif=None,
        symbolic_narrative=symbolic_narrative,
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
    v8_1_records = _narrative_v8_1_records(distillation)
    pattern_sources = _collect_pattern_sources(
        query=query,
        creative_translation=translation,
        symbolic_narrative=symbolic_narrative,
        symbolic_translation=symbolic,
        sacred_geometry=geometry,
        sacred_architecture=architecture,
        v8_1_records=v8_1_records,
    )
    patterns = tuple(
        _build_pattern(pattern_id, source_terms, evidence)
        for pattern_id, (source_terms, evidence) in pattern_sources.items()
    )
    risks = _unsupported_claim_risks(query, symbolic_translation=symbolic, sacred_architecture=architecture)
    later_boundary_requests = _later_v8_boundary_requests(query)
    symbol_nodes = build_mythopoetic_narrative_symbol_nodes(patterns)
    roadmap = mythopoetic_narrative_roadmap_assessment()
    classified = mythopoetic_narrative_items_by_classification(roadmap)
    provenance = _build_provenance(
        query=query,
        creative_translation=translation,
        symbolic_narrative=symbolic_narrative,
        symbolic_translation=symbolic,
        sacred_geometry=geometry,
        sacred_architecture=architecture,
        v8_1_records=v8_1_records,
        risks=risks,
    )
    scenes = build_mythopoetic_narrative_scene_sequence(patterns)
    return MythopoeticNarrativeReport(
        source_query=_clip(query, 760),
        reused_surface_ids=_reused_surface_ids(
            creative_translation=translation,
            symbolic_narrative=symbolic_narrative,
            v8_1_records=v8_1_records,
            symbolic_translation=symbolic,
            sacred_geometry=geometry,
            sacred_architecture=architecture,
        ),
        pattern_guidance=patterns,
        symbol_nodes=symbol_nodes,
        symbol_edges=build_mythopoetic_narrative_symbol_edges(symbol_nodes, patterns),
        scene_sequence=scenes,
        operational_guidance=build_mythopoetic_narrative_operational_guidance(patterns, domains=domains),
        validation_findings=build_mythopoetic_narrative_validation_findings(
            patterns=patterns,
            risks=risks,
            later_boundary_requests=later_boundary_requests,
        ),
        creative_brief=_creative_brief(query=query, patterns=patterns, domains=domains),
        concept_explanation=_concept_explanation(patterns, symbolic, geometry, architecture),
        symbolic_dialogue_cues=_symbolic_dialogue_cues(patterns),
        presentation_narrative=_presentation_narrative(patterns),
        demo_story=_demo_story(patterns, scenes),
        audience_communication=_audience_communication(patterns),
        provenance=provenance,
        confidence=_build_confidence(
            patterns=patterns,
            scenes=scenes,
            provenance=provenance,
            risks=risks,
            v8_1_records=v8_1_records,
            symbolic_translation=symbolic,
            sacred_geometry=geometry,
            sacred_architecture=architecture,
        ),
        roadmap_assessment=roadmap,
        implemented_roadmap_items=classified[
            MythopoeticNarrativeRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR
        ],
        reused_existing_roadmap_items=classified[
            MythopoeticNarrativeRoadmapClassification.REUSED_EXISTING_RUNTIME
        ],
        partial_reusable_roadmap_items=classified[
            MythopoeticNarrativeRoadmapClassification.PARTIAL_REUSABLE
        ],
        advisory_only_roadmap_items=classified[MythopoeticNarrativeRoadmapClassification.ADVISORY_ONLY],
        product_hitl_required_items=classified[
            MythopoeticNarrativeRoadmapClassification.PRODUCT_HITL_REQUIRED
        ],
        later_v8_boundary_items=classified[MythopoeticNarrativeRoadmapClassification.LATER_V8_BOUNDARY],
        out_of_scope_unsupported_items=classified[
            MythopoeticNarrativeRoadmapClassification.OUT_OF_SCOPE_UNSUPPORTED
        ],
        missing_roadmap_items=classified[MythopoeticNarrativeRoadmapClassification.MISSING],
        interpretation_boundaries=_interpretation_boundaries(risks, later_boundary_requests),
        unsupported_claim_risks=risks,
        hitl_questions=_hitl_questions(risks, later_boundary_requests),
    )


def mythopoetic_narrative_prompt_lines(report: MythopoeticNarrativeReport) -> tuple[str, ...]:
    """Render compact provider-independent V8.5 narrative guidance."""

    lines = [
        f"Mythopoetic Narrative Engine boundary: {report.authority_boundary}",
        f"Mythopoetic confidence: {report.confidence.band.value} {report.confidence.score:.2f}.",
    ]
    for pattern in report.pattern_guidance[:8]:
        lines.append(f"Narrative pattern: {pattern.pattern_id}; {pattern.narrative_intent}")
        lines.extend(f"{pattern.pattern_id} archetype: {item}" for item in pattern.archetypal_structure[:2])
        lines.extend(f"{pattern.pattern_id} emotion: {item}" for item in pattern.emotional_arc[:2])
        lines.extend(f"{pattern.pattern_id} transition: {item}" for item in pattern.symbolic_transitions[:2])
        lines.extend(f"{pattern.pattern_id} visual: {item}" for item in pattern.visual_mappings[:1])
        lines.extend(f"{pattern.pattern_id} motion: {item}" for item in pattern.motion_mappings[:1])
        lines.extend(f"{pattern.pattern_id} audio: {item}" for item in pattern.audio_mappings[:1])
        lines.extend(f"{pattern.pattern_id} spatial: {item}" for item in pattern.spatial_installation_mappings[:1])
        lines.append(f"{pattern.pattern_id} boundary: {pattern.boundary}")
    for scene in report.scene_sequence[:8]:
        lines.append(f"Narrative scene: {scene.phase}; {scene.title}; {scene.narrative_function}")
    lines.extend(f"Creative brief: {item}" for item in report.creative_brief[:8])
    lines.extend(f"Concept explanation: {item}" for item in report.concept_explanation[:8])
    lines.extend(f"Symbolic dialogue cue: {item}" for item in report.symbolic_dialogue_cues[:6])
    lines.extend(f"Demo story: {item}" for item in report.demo_story[:6])
    lines.extend(f"Audience communication: {item}" for item in report.audience_communication[:6])
    lines.extend(f"Narrative validation: {item.severity.value}; {item.summary}" for item in report.validation_findings)
    lines.extend(f"Narrative boundary: {item}" for item in report.interpretation_boundaries)
    lines.extend(f"Unsupported narrative claim risk: {item}" for item in report.unsupported_claim_risks)
    lines.extend(f"HITL narrative question: {item}" for item in report.hitl_questions)
    return tuple(lines[:120])


def mythopoetic_narrative_roadmap_assessment() -> tuple[MythopoeticNarrativeRoadmapItemAssessment, ...]:
    """Return the V8.5 roadmap reality-check assessment."""

    return tuple(
        MythopoeticNarrativeRoadmapItemAssessment(
            item=item,
            classification=MythopoeticNarrativeRoadmapClassification(classification),
            rationale=rationale,
            action_required_before_hitl=action_required,
            hitl_required=hitl_required,
        )
        for item, (classification, rationale, action_required, hitl_required) in ROADMAP_CLASSIFICATION_ROWS.items()
    )


def detect_mythopoetic_narrative_terms(query: str) -> tuple[str, ...]:
    """Return supported V8.5 narrative pattern ids visible in request text."""

    pattern_ids: list[str] = []
    for token in _ordered_tokens(query):
        pattern_ids.extend(NARRATIVE_ALIASES.get(token, ()))
        if token in PATTERN_ROWS:
            pattern_ids.append(token)
    return _dedupe(pattern_ids)[:18]


def _collect_pattern_sources(
    *,
    query: str,
    creative_translation: CreativeTranslation,
    symbolic_narrative: SymbolicNarrativePlan | None,
    symbolic_translation: SymbolicTranslationReport,
    sacred_geometry: SacredGeometryReport,
    sacred_architecture: SacredArchitectureReport,
    v8_1_records: Sequence[CreativeKnowledgeRecord],
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    sources: dict[str, list[str]] = {}
    evidence: dict[str, list[str]] = {}

    for pattern_id in detect_mythopoetic_narrative_terms(query):
        _add_source(sources, evidence, pattern_id, pattern_id, f"Request-visible narrative cue: {pattern_id}.")

    for ref in (
        creative_translation.creative_intent,
        *creative_translation.symbolic_references,
        *creative_translation.mood_atmosphere,
        *creative_translation.movement_language,
        *creative_translation.structure_direction,
        *creative_translation.musical_references,
        *creative_translation.geometric_references,
    ):
        for pattern_id in _patterns_for_text(ref):
            _add_source(sources, evidence, pattern_id, ref, f"Creative translation narrative signal: {ref}.")

    if symbolic_narrative is not None:
        for pattern_id in _SYMBOLIC_NARRATIVE_TO_PATTERN.get(symbolic_narrative.narrative_archetype, ()):
            _add_source(
                sources,
                evidence,
                pattern_id,
                symbolic_narrative.narrative_archetype,
                f"V3 symbolic narrative archetype: {symbolic_narrative.narrative_archetype}.",
            )

    for mapping in symbolic_translation.motif_mappings:
        for pattern_id in SYMBOL_TO_NARRATIVE.get(mapping.motif_id, ()):
            _add_source(
                sources,
                evidence,
                pattern_id,
                mapping.motif_id,
                f"V8.2 symbolic motif maps to narrative: {mapping.motif_id}.",
            )
        for term in mapping.source_terms:
            for pattern_id in _patterns_for_text(term):
                _add_source(sources, evidence, pattern_id, term, f"V8.2 source term maps to narrative: {term}.")

    for pattern in sacred_geometry.pattern_guidance:
        for pattern_id in GEOMETRY_TO_NARRATIVE.get(pattern.pattern_id, ()):
            _add_source(
                sources,
                evidence,
                pattern_id,
                pattern.pattern_id,
                f"V8.3 geometry pattern maps to narrative: {pattern.pattern_id}.",
            )

    for pattern in sacred_architecture.pattern_guidance:
        for pattern_id in ARCHITECTURE_TO_NARRATIVE.get(pattern.pattern_id, ()):
            _add_source(
                sources,
                evidence,
                pattern_id,
                pattern.pattern_id,
                f"V8.4 architecture pattern maps to narrative: {pattern.pattern_id}.",
            )

    for record in v8_1_records:
        text = _record_text(record)
        for pattern_id in _patterns_for_text(text):
            _add_source(
                sources,
                evidence,
                pattern_id,
                record.record_id,
                f"V8.1 creative knowledge record supports narrative guidance: {record.record_id}.",
            )

    if not sources:
        _add_source(
            sources,
            evidence,
            "demo_project_story",
            "broad_creative_framing",
            "Fallback for broad creative/project framing without stronger narrative cues.",
        )

    return {
        pattern_id: (_dedupe(source_terms)[:14], _dedupe(evidence_items)[:16])
        for pattern_id, source_terms in sources.items()
        for evidence_items in (evidence[pattern_id],)
    }


def _build_pattern(
    pattern_id: str,
    source_terms: Sequence[str],
    evidence: Sequence[str],
) -> MythopoeticNarrativePatternGuidance:
    row = PATTERN_ROWS[pattern_id]
    confidence = min(0.94, 0.52 + 0.045 * len(source_terms) + 0.025 * len(evidence))
    if source_terms == ("broad_creative_framing",):
        confidence = 0.48
    return MythopoeticNarrativePatternGuidance(
        pattern_id=pattern_id,
        label=row.label,
        family=MythopoeticNarrativeFamily(row.family),
        source_terms=tuple(source_terms) or (pattern_id,),
        taxonomy_path=row.taxonomy_path,
        narrative_intent=row.narrative_intent,
        archetypal_structure=row.archetypal_structure,
        journey_arc=row.journey_arc,
        ritual_structure=row.ritual_structure,
        emotional_arc=row.emotional_arc,
        symbolic_transitions=row.symbolic_transitions,
        visual_mappings=row.visual_mappings,
        motion_mappings=row.motion_mappings,
        audio_mappings=row.audio_mappings,
        spatial_installation_mappings=row.spatial_installation_mappings,
        creative_brief_points=row.creative_brief_points,
        explanation_points=row.explanation_points,
        audience_communication=row.audience_communication,
        boundary=row.boundary,
        evidence=tuple(evidence) or (f"Bounded narrative catalog pattern: {pattern_id}.",),
        confidence_score=round(confidence, 3),
    )


def _build_provenance(
    *,
    query: str,
    creative_translation: CreativeTranslation,
    symbolic_narrative: SymbolicNarrativePlan | None,
    symbolic_translation: SymbolicTranslationReport,
    sacred_geometry: SacredGeometryReport,
    sacred_architecture: SacredArchitectureReport,
    v8_1_records: Sequence[CreativeKnowledgeRecord],
    risks: Sequence[str],
) -> tuple[MythopoeticNarrativeProvenance, ...]:
    provenance = [
        MythopoeticNarrativeProvenance(
            provenance_id="mythopoetic_narrative::request",
            kind="request_signal",
            reference="user_query",
            summary=f"Request-visible narrative guidance source: {_clip(query, 220)}",
        ),
        MythopoeticNarrativeProvenance(
            provenance_id="mythopoetic_narrative::creative_translation",
            kind="creative_translation",
            reference="v3_creative_translation",
            summary="V3 creative translation supplies intent, mood, motion, modality, and runtime cues.",
        ),
        MythopoeticNarrativeProvenance(
            provenance_id="mythopoetic_narrative::v8_2_symbolic_translation",
            kind="v8_2_symbolic_translation",
            reference=symbolic_translation.capability_id,
            summary="V8.2 symbolic motifs supply bounded symbolic-to-narrative signals.",
            confidence_signal=symbolic_translation.confidence.score,
        ),
        MythopoeticNarrativeProvenance(
            provenance_id="mythopoetic_narrative::v8_3_sacred_geometry",
            kind="v8_3_sacred_geometry",
            reference=sacred_geometry.capability_id,
            summary="V8.3 geometry guidance supplies visual, motion, audio, and structure signals.",
            confidence_signal=sacred_geometry.confidence.score,
        ),
        MythopoeticNarrativeProvenance(
            provenance_id="mythopoetic_narrative::v8_4_sacred_architecture",
            kind="v8_4_sacred_architecture",
            reference=sacred_architecture.capability_id,
            summary="V8.4 architecture guidance supplies spatial, procession, topology, and installation signals.",
            confidence_signal=sacred_architecture.confidence.score,
        ),
        MythopoeticNarrativeProvenance(
            provenance_id="mythopoetic_narrative::bounded_catalog",
            kind="bounded_narrative_catalog",
            reference="knowledge.mythopoetic_narrative_catalog.PATTERN_ROWS",
            summary="Static bounded narrative catalog maps supported arcs to creative coding guidance.",
        ),
    ]
    if symbolic_narrative is not None:
        provenance.append(
            MythopoeticNarrativeProvenance(
                provenance_id="mythopoetic_narrative::v3_symbolic_narrative",
                kind="v3_symbolic_narrative",
                reference=symbolic_narrative.narrative_archetype,
                summary="Existing V3 symbolic narrative plan is reused as archetypal arc evidence.",
            )
        )
    if risks:
        provenance.append(
            MythopoeticNarrativeProvenance(
                provenance_id="mythopoetic_narrative::safety_boundary",
                kind="safety_boundary",
                reference="v8_5_authority_boundary",
                summary="Unsupported claim-risk language is guarded by V8.5 interpretation boundaries.",
            )
        )
    for record in v8_1_records[:8]:
        provenance.append(
            MythopoeticNarrativeProvenance(
                provenance_id=f"mythopoetic_narrative::{record.record_id}",
                kind="v8_1_creative_knowledge",
                reference=record.record_id,
                summary=_clip(record.summary, 420),
                confidence_signal=record.confidence.score,
            )
        )
    _ = creative_translation
    return tuple(provenance)


def _build_confidence(
    *,
    patterns: Sequence[MythopoeticNarrativePatternGuidance],
    scenes: Sequence[object],
    provenance: Sequence[MythopoeticNarrativeProvenance],
    risks: Sequence[str],
    v8_1_records: Sequence[CreativeKnowledgeRecord],
    symbolic_translation: SymbolicTranslationReport,
    sacred_geometry: SacredGeometryReport,
    sacred_architecture: SacredArchitectureReport,
) -> MythopoeticNarrativeConfidence:
    evidence_count = sum(len(pattern.evidence) for pattern in patterns)
    score = (
        0.42
        + min(0.2, 0.035 * len(patterns))
        + min(0.12, 0.008 * evidence_count)
        + min(0.1, 0.006 * len(provenance))
        + min(0.08, 0.01 * len(symbolic_translation.motif_mappings))
        + min(0.08, 0.012 * len(sacred_geometry.pattern_guidance))
        + min(0.08, 0.012 * len(sacred_architecture.pattern_guidance))
    )
    if risks:
        score -= 0.22
    rounded_score = round(max(0.0, min(0.96, score)), 3)
    caveats = tuple(risks[:8])
    return MythopoeticNarrativeConfidence(
        score=rounded_score,
        band=mythopoetic_narrative_confidence_band(rounded_score, guarded=bool(caveats)),
        pattern_count=len(patterns),
        scene_count=len(scenes),
        evidence_count=evidence_count,
        provenance_count=len(provenance),
        v8_1_record_ids=tuple(record.record_id for record in v8_1_records[:10]),
        v8_2_motif_ids=tuple(mapping.motif_id for mapping in symbolic_translation.motif_mappings[:12]),
        v8_3_pattern_ids=tuple(pattern.pattern_id for pattern in sacred_geometry.pattern_guidance[:12]),
        v8_4_pattern_ids=tuple(pattern.pattern_id for pattern in sacred_architecture.pattern_guidance[:12]),
        caveats=caveats,
    )


def _creative_brief(
    *,
    query: str,
    patterns: Sequence[MythopoeticNarrativePatternGuidance],
    domains: Sequence[CreativeCodingDomain],
) -> tuple[str, ...]:
    domain_text = ", ".join(domain.value for domain in domains) if domains else "unspecified creative coding runtime"
    lines = [f"Intent: {_clip(query, 180)}", f"Target runtime context: {domain_text}."]
    lines.extend(_dedupe(item for pattern in patterns for item in pattern.creative_brief_points)[:8])
    return tuple(lines[:10])


def _concept_explanation(
    patterns: Sequence[MythopoeticNarrativePatternGuidance],
    symbolic_translation: SymbolicTranslationReport,
    sacred_geometry: SacredGeometryReport,
    sacred_architecture: SacredArchitectureReport,
) -> tuple[str, ...]:
    lines = list(_dedupe(item for pattern in patterns for item in pattern.explanation_points)[:8])
    lines.append(f"Symbolic reuse: {len(symbolic_translation.motif_mappings)} bounded V8.2 motif mappings.")
    lines.append(f"Geometry reuse: {len(sacred_geometry.pattern_guidance)} bounded V8.3 geometry patterns.")
    lines.append(f"Spatial reuse: {len(sacred_architecture.pattern_guidance)} bounded V8.4 architecture patterns.")
    return tuple(lines[:12])


def _symbolic_dialogue_cues(
    patterns: Sequence[MythopoeticNarrativePatternGuidance],
) -> tuple[str, ...]:
    if len(patterns) == 1:
        pattern = patterns[0]
        return (
            f"Let {pattern.label} speak through contrasting before/after visual states.",
            "Use symbolic dialogue as a compositional relation, not an authoritative voice.",
        )
    first, second = patterns[0], patterns[1]
    return (
        f"Stage {first.label} and {second.label} as two design voices.",
        "Let motion, density, or color answer between the voices before resolving.",
        "Keep all meanings user-authored and explainable.",
    )


def _presentation_narrative(
    patterns: Sequence[MythopoeticNarrativePatternGuidance],
) -> tuple[str, ...]:
    primary = patterns[0]
    return (
        f"Open with the core arc: {primary.label}.",
        f"Explain the intended journey: {' -> '.join(primary.archetypal_structure[:5])}.",
        "Show how symbolic, visual, motion, audio, and spatial cues support the same arc.",
        "Close by naming boundaries: creative interpretation only, no authority or therapeutic claim.",
    )


def _demo_story(
    patterns: Sequence[MythopoeticNarrativePatternGuidance],
    scenes: Sequence[object],
) -> tuple[str, ...]:
    return (
        f"Demo the narrative as {len(scenes)} ordered beats from opening to return.",
        f"Lead with {patterns[0].label} and use secondary patterns as supporting scenes.",
        "Point out one visual transition, one motion transition, and one spatial/audience cue.",
        "Disclose that V8.5 generates narrative guidance, not V8.6 audiovisual composition or V8.8 showcase UI.",
    )


def _audience_communication(
    patterns: Sequence[MythopoeticNarrativePatternGuidance],
) -> tuple[str, ...]:
    items = list(_dedupe(item for pattern in patterns for item in pattern.audience_communication))
    items.append("State intended audience experience as design intent, not guaranteed effect.")
    return tuple(items[:10])


def _narrative_v8_1_records(
    distillation: CreativeKnowledgeDistillationReport,
) -> tuple[CreativeKnowledgeRecord, ...]:
    selected = []
    signals = ("narrative", "story", "symbol", "ritual", "demo", "installation", "spatial", "emotion")
    for record in distillation.records:
        text = _record_text(record).lower()
        if any(signal in text for signal in signals):
            selected.append(record)
    return tuple(selected[:10])


def _record_text(record: CreativeKnowledgeRecord) -> str:
    return " ".join(
        (
            record.record_id,
            record.title,
            record.summary,
            *record.technique_tags,
            *record.workflow_steps,
            *record.pattern_tags,
            *record.taxonomy_path,
        )
    )


def _unsupported_claim_risks(
    query: str,
    *,
    symbolic_translation: SymbolicTranslationReport,
    sacred_architecture: SacredArchitectureReport,
) -> tuple[str, ...]:
    risks = [
        f"Unsupported narrative authority cue: {token}."
        for token in _ordered_tokens(query)
        if token in UNSUPPORTED_NARRATIVE_CLAIM_TOKENS
    ]
    risks.extend(symbolic_translation.unsupported_claim_risks)
    risks.extend(sacred_architecture.unsupported_claim_risks)
    return _dedupe(risks)[:10]


def _later_v8_boundary_requests(query: str) -> tuple[str, ...]:
    normalized = query.lower()
    requests = (
        f"Later-version boundary cue: {label}."
        for pattern, label in _LATER_V8_PATTERNS
        if pattern.search(normalized)
    )
    return tuple(requests)[:8]


def _interpretation_boundaries(
    risks: Sequence[str],
    later_boundary_requests: Sequence[str],
) -> tuple[str, ...]:
    boundaries = [
        "Use mythopoetic language as creative guidance, not religious, esoteric, or historical authority.",
        "Do not infer psychological diagnosis, psychotherapy, trauma treatment, or guaranteed audience outcomes.",
        "Keep ritual/processional language as pacing and structure, not efficacy or lineage.",
        "Keep output report-only: no preview mutation, workflow control, storage writes, or provider routing.",
        "Do not start V8.6 immersive composer, V8.7 OS, V8.8 showcase system, HoloMind, or HOLOiVERSE behavior.",
    ]
    if risks:
        boundaries.append("Unsupported claim-risk terms require cautious wording or HITL-scoped source decisions.")
    if later_boundary_requests:
        boundaries.append("Later-version cues may be described as narrative intent only, not implemented behavior.")
    return tuple(boundaries)


def _hitl_questions(
    risks: Sequence[str],
    later_boundary_requests: Sequence[str],
) -> tuple[str, ...]:
    questions: list[str] = []
    if risks:
        questions.append(
            "Should authoritative religious, esoteric, or psychological language be removed or explicitly sourced?"
        )
    if later_boundary_requests:
        questions.append("Should later V8 preview/composer/showcase/OS cues remain conceptual until their own gate?")
    return tuple(questions[:8])


def _reused_surface_ids(
    *,
    creative_translation: CreativeTranslation,
    symbolic_narrative: SymbolicNarrativePlan | None,
    v8_1_records: Sequence[CreativeKnowledgeRecord],
    symbolic_translation: SymbolicTranslationReport,
    sacred_geometry: SacredGeometryReport,
    sacred_architecture: SacredArchitectureReport,
) -> tuple[str, ...]:
    surfaces = [
        "v3_creative_translation",
        symbolic_translation.capability_id,
        sacred_geometry.capability_id,
        sacred_architecture.capability_id,
    ]
    if symbolic_narrative is not None:
        surfaces.append("v3_symbolic_narrative_planner")
    if v8_1_records:
        surfaces.append("v8_1_creative_knowledge_distillation")
    if creative_translation.output_modality is not None:
        surfaces.append(f"v3_output_modality::{creative_translation.output_modality.value}")
    return tuple(surfaces)


def _patterns_for_text(text: str) -> tuple[str, ...]:
    pattern_ids: list[str] = []
    for token in _ordered_tokens(text):
        pattern_ids.extend(NARRATIVE_ALIASES.get(token, ()))
        if token in PATTERN_ROWS:
            pattern_ids.append(token)
    return _dedupe(pattern_ids)[:12]


def _add_source(
    sources: dict[str, list[str]],
    evidence: dict[str, list[str]],
    pattern_id: str,
    source_term: str,
    evidence_line: str,
) -> None:
    if pattern_id not in PATTERN_ROWS:
        return
    sources.setdefault(pattern_id, [])
    evidence.setdefault(pattern_id, [])
    if source_term not in sources[pattern_id]:
        sources[pattern_id].append(_clip(source_term, 160))
    if evidence_line not in evidence[pattern_id]:
        evidence[pattern_id].append(_clip(evidence_line, 360))


def _ordered_tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).strip("'") for match in _TOKEN_PATTERN.finditer(text.lower()))


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return tuple(seen)


def _clip(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


__all__ = [
    "build_v8_5_mythopoetic_narrative_engine",
    "detect_mythopoetic_narrative_terms",
    "mythopoetic_narrative_prompt_lines",
    "mythopoetic_narrative_roadmap_assessment",
]
