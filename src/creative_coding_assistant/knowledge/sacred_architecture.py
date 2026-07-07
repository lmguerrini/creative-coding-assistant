"""V8.4 bounded sacred architecture and reverse-engineering guidance engine."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.knowledge.creative_distillation import (
    CreativeKnowledgeDistillationReport,
    CreativeKnowledgeRecord,
    build_v8_1_creative_knowledge_distillation,
)
from creative_coding_assistant.knowledge.sacred_architecture_catalog import (
    ARCHITECTURE_ALIASES,
    GEOMETRY_PATTERN_TO_ARCHITECTURE,
    PATTERN_ROWS,
    ROADMAP_CLASSIFICATION_ROWS,
    SYMBOL_TO_ARCHITECTURE,
    UNSUPPORTED_ARCHITECTURE_CLAIM_TOKENS,
)
from creative_coding_assistant.knowledge.sacred_architecture_contracts import (
    V8_4_CAPABILITY_ID,
    SacredArchitectureConfidence,
    SacredArchitectureFamily,
    SacredArchitecturePatternGuidance,
    SacredArchitectureProvenance,
    SacredArchitectureReport,
    SacredArchitectureRoadmapClassification,
    SacredArchitectureRoadmapItemAssessment,
    sacred_architecture_confidence_band,
    sacred_architecture_items_by_classification,
)
from creative_coding_assistant.knowledge.sacred_architecture_guidance import (
    build_sacred_architecture_operational_guidance,
    build_sacred_architecture_semantic_edges,
    build_sacred_architecture_semantic_nodes,
    build_sacred_architecture_validation_findings,
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


@dataclass(frozen=True)
class _ArchitectureBlueprint:
    label: str
    family: SacredArchitectureFamily
    taxonomy_path: tuple[str, ...]
    spatial_intent: str
    proportions: tuple[str, ...]
    plan: tuple[str, ...]
    axes: tuple[str, ...]
    thresholds: tuple[str, ...]
    center_periphery: tuple[str, ...]
    topology: tuple[str, ...]
    geometry_mappings: tuple[str, ...]
    symbolic_mappings: tuple[str, ...]
    installation: tuple[str, ...]
    reverse_engineering_cues: tuple[str, ...]
    parameters: tuple[str, ...]
    runtimes: tuple[str, ...]
    notes: tuple[str, ...]
    boundary: str


_TOKEN_PATTERN = re.compile(r"[a-z0-9_.+#'-]+")
_ARCHITECTURE_ALIASES = ARCHITECTURE_ALIASES
_BLUEPRINTS: dict[str, _ArchitectureBlueprint] = {
    pattern_id: _ArchitectureBlueprint(
        label=row.label,
        family=SacredArchitectureFamily(row.family),
        taxonomy_path=row.taxonomy_path,
        spatial_intent=row.spatial_intent,
        proportions=row.proportions,
        plan=row.plan,
        axes=row.axes,
        thresholds=row.thresholds,
        center_periphery=row.center_periphery,
        topology=row.topology,
        geometry_mappings=row.geometry_mappings,
        symbolic_mappings=row.symbolic_mappings,
        installation=row.installation,
        reverse_engineering_cues=row.reverse_engineering_cues,
        parameters=row.parameters,
        runtimes=row.runtimes,
        notes=row.notes,
        boundary=row.boundary,
    )
    for pattern_id, row in PATTERN_ROWS.items()
}
_ROADMAP_CLASSIFICATIONS: dict[str, tuple[SacredArchitectureRoadmapClassification, str, bool, bool]] = {
    item: (SacredArchitectureRoadmapClassification(classification), rationale, action_required, hitl_required)
    for item, (classification, rationale, action_required, hitl_required) in ROADMAP_CLASSIFICATION_ROWS.items()
}


def build_v8_4_sacred_architecture_engine(
    query: str,
    *,
    domains: Sequence[CreativeCodingDomain] = (),
    creative_translation: CreativeTranslation | None = None,
    v8_1_distillation: CreativeKnowledgeDistillationReport | None = None,
    v8_2_symbolic_translation: SymbolicTranslationReport | None = None,
    v8_3_sacred_geometry: SacredGeometryReport | None = None,
) -> SacredArchitectureReport:
    """Build a bounded V8.4 sacred architecture report without runtime mutation."""

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
    v8_1_records = _architecture_v8_1_records(distillation)
    pattern_sources = _collect_pattern_sources(
        query=query,
        creative_translation=translation,
        symbolic_translation=symbolic,
        sacred_geometry=geometry,
        v8_1_records=v8_1_records,
    )
    patterns = tuple(
        _build_pattern(pattern_id, source_terms, evidence)
        for pattern_id, (source_terms, evidence) in pattern_sources.items()
    )
    risks = _unsupported_claim_risks(query, symbolic_translation=symbolic, sacred_geometry=geometry)
    provenance = _build_provenance(
        query=query,
        creative_translation=translation,
        symbolic_translation=symbolic,
        sacred_geometry=geometry,
        v8_1_records=v8_1_records,
        risks=risks,
    )
    confidence = _build_confidence(
        patterns=patterns,
        provenance=provenance,
        risks=risks,
        v8_1_records=v8_1_records,
        symbolic_translation=symbolic,
        sacred_geometry=geometry,
    )
    roadmap = sacred_architecture_roadmap_assessment()
    classified = sacred_architecture_items_by_classification(roadmap)
    semantic_nodes = build_sacred_architecture_semantic_nodes(patterns)
    return SacredArchitectureReport(
        source_query=_clip(query, 680),
        reused_surface_ids=_reused_surface_ids(
            creative_translation=translation,
            v8_1_records=v8_1_records,
            symbolic_translation=symbolic,
            sacred_geometry=geometry,
        ),
        pattern_guidance=patterns,
        operational_guidance=build_sacred_architecture_operational_guidance(patterns, domains=domains),
        semantic_nodes=semantic_nodes,
        semantic_edges=build_sacred_architecture_semantic_edges(semantic_nodes, patterns),
        validation_findings=build_sacred_architecture_validation_findings(patterns=patterns, risks=risks),
        provenance=provenance,
        confidence=confidence,
        roadmap_assessment=roadmap,
        implemented_roadmap_items=classified[
            SacredArchitectureRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR
        ],
        reused_existing_roadmap_items=classified[SacredArchitectureRoadmapClassification.REUSED_EXISTING_RUNTIME],
        partial_reusable_roadmap_items=classified[SacredArchitectureRoadmapClassification.PARTIAL_REUSABLE],
        advisory_only_roadmap_items=classified[SacredArchitectureRoadmapClassification.ADVISORY_ONLY],
        product_hitl_required_items=classified[SacredArchitectureRoadmapClassification.PRODUCT_HITL_REQUIRED],
        later_v8_boundary_items=classified[SacredArchitectureRoadmapClassification.LATER_V8_BOUNDARY],
        out_of_scope_unsupported_items=classified[
            SacredArchitectureRoadmapClassification.OUT_OF_SCOPE_UNSUPPORTED
        ],
        missing_roadmap_items=classified[SacredArchitectureRoadmapClassification.MISSING],
        interpretation_boundaries=_interpretation_boundaries(risks),
        unsupported_claim_risks=risks,
        hitl_questions=_hitl_questions(risks),
    )


def sacred_architecture_prompt_lines(report: SacredArchitectureReport) -> tuple[str, ...]:
    """Render compact provider-independent architecture guidance."""

    lines = [
        f"Sacred Architecture Engine boundary: {report.authority_boundary}",
        f"Sacred architecture confidence: {report.confidence.band.value} {report.confidence.score:.2f}.",
    ]
    for pattern in report.pattern_guidance[:8]:
        lines.append(f"Architecture pattern: {pattern.pattern_id}; {pattern.spatial_intent}")
        lines.extend(f"{pattern.pattern_id} proportion: {item}" for item in pattern.proportion_guidance[:2])
        lines.extend(f"{pattern.pattern_id} plan: {item}" for item in pattern.plan_guidance[:2])
        lines.extend(f"{pattern.pattern_id} axis: {item}" for item in pattern.axis_guidance[:1])
        lines.extend(f"{pattern.pattern_id} threshold: {item}" for item in pattern.threshold_guidance[:1])
        lines.extend(f"{pattern.pattern_id} topology: {item}" for item in pattern.topology_guidance[:1])
        lines.extend(f"{pattern.pattern_id} installation: {item}" for item in pattern.installation_guidance[:1])
        lines.append(f"{pattern.pattern_id} boundary: {pattern.boundary}")
    for guidance in report.operational_guidance[:10]:
        lines.append(
            "Architecture operation: "
            f"{guidance.kind.value}; {', '.join(guidance.source_pattern_ids)}; "
            f"{' '.join(guidance.guidance[:2])}"
        )
    for node in report.semantic_nodes[:8]:
        lines.append(f"Architecture semantic node: {node.role.value}; {node.label}; {node.guidance}")
    lines.extend(
        f"Architecture validation: {item.severity.value}; {item.summary}"
        for item in report.validation_findings
    )
    lines.extend(f"Architecture boundary: {item}" for item in report.interpretation_boundaries)
    lines.extend(f"Unsupported architecture claim risk: {item}" for item in report.unsupported_claim_risks)
    lines.extend(f"HITL architecture question: {item}" for item in report.hitl_questions)
    return tuple(lines[:90])


def sacred_architecture_roadmap_assessment() -> tuple[SacredArchitectureRoadmapItemAssessment, ...]:
    """Return the V8.4 roadmap reality-check assessment."""

    return tuple(
        SacredArchitectureRoadmapItemAssessment(
            item=item,
            classification=classification,
            rationale=rationale,
            action_required_before_hitl=action_required,
            hitl_required=hitl_required,
        )
        for item, (classification, rationale, action_required, hitl_required) in _ROADMAP_CLASSIFICATIONS.items()
    )


def detect_sacred_architecture_terms(query: str) -> tuple[str, ...]:
    """Return supported V8.4 architecture pattern ids visible in request text."""

    pattern_ids: list[str] = []
    for token in _ordered_tokens(query):
        pattern_ids.extend(_ARCHITECTURE_ALIASES.get(token, ()))
        if token in _BLUEPRINTS:
            pattern_ids.append(token)
    return _dedupe(pattern_ids)[:16]


def _collect_pattern_sources(
    *,
    query: str,
    creative_translation: CreativeTranslation,
    symbolic_translation: SymbolicTranslationReport,
    sacred_geometry: SacredGeometryReport,
    v8_1_records: Sequence[CreativeKnowledgeRecord],
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    sources: dict[str, list[str]] = {}
    evidence: dict[str, list[str]] = {}

    for pattern_id in detect_sacred_architecture_terms(query):
        _add_source(sources, evidence, pattern_id, pattern_id, f"Request-visible architecture cue: {pattern_id}.")

    for ref in (
        creative_translation.creative_intent,
        *creative_translation.geometric_references,
        *creative_translation.symbolic_references,
        *creative_translation.movement_language,
        *creative_translation.structure_direction,
        *creative_translation.color_material_direction,
        *creative_translation.runtime_recommendations,
    ):
        for pattern_id in _patterns_for_text(ref):
            _add_source(sources, evidence, pattern_id, ref, f"Creative translation spatial signal: {ref}.")

    for mapping in symbolic_translation.motif_mappings:
        for pattern_id in SYMBOL_TO_ARCHITECTURE.get(mapping.motif_id, ()):
            _add_source(
                sources,
                evidence,
                pattern_id,
                mapping.motif_id,
                f"V8.2 symbolic-to-spatial signal: {mapping.motif_id}.",
            )
        for value in (*mapping.source_terms, *mapping.visual_guidance, *mapping.composition_guidance):
            for pattern_id in _patterns_for_text(value):
                _add_source(
                    sources,
                    evidence,
                    pattern_id,
                    value,
                    f"V8.2 symbolic guidance spatial signal: {mapping.motif_id}.",
                )

    for pattern in sacred_geometry.pattern_guidance:
        for pattern_id in GEOMETRY_PATTERN_TO_ARCHITECTURE.get(pattern.pattern_id, ()):
            _add_source(
                sources,
                evidence,
                pattern_id,
                pattern.pattern_id,
                f"V8.3 geometry-to-architecture signal: {pattern.pattern_id}.",
            )
        for value in (*pattern.source_terms, *pattern.structure_guidance, *pattern.mathematical_parameters):
            for pattern_id in _patterns_for_text(value):
                _add_source(
                    sources,
                    evidence,
                    pattern_id,
                    value,
                    f"V8.3 geometry guidance spatial signal: {pattern.pattern_id}.",
                )

    for record in v8_1_records:
        for value in (*record.technique_tags, *record.pattern_tags, record.title, record.summary):
            for pattern_id in _patterns_for_text(value):
                _add_source(
                    sources,
                    evidence,
                    pattern_id,
                    value,
                    f"V8.1 creative knowledge spatial signal: {record.record_id}.",
                )

    if not sources:
        _add_source(
            sources,
            evidence,
            "axis_threshold_procession",
            "architecture",
            "Fallback for broad architecture request without a more specific supported cue.",
        )

    ordered = sorted(
        sources,
        key=lambda pattern_id: (
            -len(evidence[pattern_id]),
            list(_BLUEPRINTS).index(pattern_id) if pattern_id in _BLUEPRINTS else 999,
            pattern_id,
        ),
    )[:12]
    return {
        pattern_id: (_dedupe(sources[pattern_id])[:14], _dedupe(evidence[pattern_id])[:16])
        for pattern_id in ordered
        if pattern_id in _BLUEPRINTS
    }


def _build_pattern(
    pattern_id: str,
    source_terms: tuple[str, ...],
    evidence: tuple[str, ...],
) -> SacredArchitecturePatternGuidance:
    blueprint = _BLUEPRINTS[pattern_id]
    score = min(0.97, 0.48 + 0.065 * len(source_terms) + 0.03 * len(evidence))
    return SacredArchitecturePatternGuidance(
        pattern_id=pattern_id,
        label=blueprint.label,
        family=blueprint.family,
        source_terms=source_terms,
        taxonomy_path=blueprint.taxonomy_path,
        spatial_intent=blueprint.spatial_intent,
        proportion_guidance=blueprint.proportions,
        plan_guidance=blueprint.plan,
        axis_guidance=blueprint.axes,
        threshold_guidance=blueprint.thresholds,
        center_periphery_guidance=blueprint.center_periphery,
        topology_guidance=blueprint.topology,
        geometry_mappings=blueprint.geometry_mappings,
        symbolic_mappings=blueprint.symbolic_mappings,
        installation_guidance=blueprint.installation,
        reverse_engineering_cues=blueprint.reverse_engineering_cues,
        runtime_families=blueprint.runtimes,
        implementation_notes=blueprint.notes,
        boundary=blueprint.boundary,
        evidence=evidence,
        confidence_score=round(score, 2),
    )


def _build_provenance(
    *,
    query: str,
    creative_translation: CreativeTranslation,
    symbolic_translation: SymbolicTranslationReport,
    sacred_geometry: SacredGeometryReport,
    v8_1_records: Sequence[CreativeKnowledgeRecord],
    risks: Sequence[str],
) -> tuple[SacredArchitectureProvenance, ...]:
    provenance = [
        SacredArchitectureProvenance(
            provenance_id="sacred_architecture::request",
            kind="request_signal",
            reference="assistant_request.query",
            summary=_clip(query, 560),
            confidence_signal=None,
        ),
        SacredArchitectureProvenance(
            provenance_id="sacred_architecture::creative_translation",
            kind="creative_translation",
            reference="orchestration.creative_translation",
            summary="Reused creative translation geometry, structure, movement, runtime, and aesthetic signals.",
            confidence_signal=0.78,
        ),
        SacredArchitectureProvenance(
            provenance_id="sacred_architecture::v8_2_symbolic_translation",
            kind="v8_2_symbolic_translation",
            reference=symbolic_translation.capability_id,
            summary="Reused bounded symbolic motif mappings as spatial role and threshold signals.",
            confidence_signal=symbolic_translation.confidence.score,
        ),
        SacredArchitectureProvenance(
            provenance_id="sacred_architecture::v8_3_sacred_geometry",
            kind="v8_3_sacred_geometry",
            reference=sacred_geometry.capability_id,
            summary="Reused sacred geometry patterns, ratios, temple-axis composition, and layout mappings.",
            confidence_signal=sacred_geometry.confidence.score,
        ),
        SacredArchitectureProvenance(
            provenance_id="sacred_architecture::bounded_catalog",
            kind="bounded_architecture_catalog",
            reference="knowledge.sacred_architecture_catalog.PATTERN_ROWS",
            summary="Used scoped sacred architecture and installation pattern rows with explicit safety boundaries.",
            confidence_signal=0.73,
        ),
    ]
    del creative_translation
    if risks:
        provenance.append(
            SacredArchitectureProvenance(
                provenance_id="sacred_architecture::safety_boundary",
                kind="safety_boundary",
                reference="v8_4_authority_boundary",
                summary="Unsupported reconstruction, survey, authority, or safety terms were downgraded to caveats.",
                confidence_signal=0.45,
            )
        )
    provenance.extend(
        SacredArchitectureProvenance(
            provenance_id=f"sacred_architecture::{record.record_id}",
            kind="v8_1_creative_knowledge",
            reference=record.record_id,
            summary=record.summary,
            confidence_signal=record.confidence.score,
        )
        for record in v8_1_records[:8]
    )
    return tuple(provenance[:30])


def _build_confidence(
    *,
    patterns: Sequence[SacredArchitecturePatternGuidance],
    provenance: Sequence[SacredArchitectureProvenance],
    risks: Sequence[str],
    v8_1_records: Sequence[CreativeKnowledgeRecord],
    symbolic_translation: SymbolicTranslationReport,
    sacred_geometry: SacredGeometryReport,
) -> SacredArchitectureConfidence:
    evidence_count = sum(len(pattern.evidence) for pattern in patterns)
    pattern_bonus = min(0.26, 0.042 * len(patterns))
    evidence_bonus = min(0.24, 0.015 * evidence_count)
    provenance_bonus = min(0.19, 0.022 * len(provenance))
    v8_1_bonus = min(0.09, 0.02 * len(v8_1_records))
    v8_2_bonus = min(0.1, 0.018 * len(symbolic_translation.motif_mappings))
    v8_3_bonus = min(0.12, 0.018 * len(sacred_geometry.pattern_guidance))
    risk_penalty = min(0.3, 0.08 * len(risks))
    score = max(
        0.05,
        min(
            0.97,
            0.28
            + pattern_bonus
            + evidence_bonus
            + provenance_bonus
            + v8_1_bonus
            + v8_2_bonus
            + v8_3_bonus
            - risk_penalty,
        ),
    )
    rounded_score = round(score, 2)
    return SacredArchitectureConfidence(
        score=rounded_score,
        band=sacred_architecture_confidence_band(rounded_score, guarded=bool(risks)),
        pattern_count=len(patterns),
        evidence_count=evidence_count,
        provenance_count=len(provenance),
        v8_1_record_ids=tuple(record.record_id for record in v8_1_records[:10]),
        v8_2_motif_ids=tuple(mapping.motif_id for mapping in symbolic_translation.motif_mappings[:12]),
        v8_3_pattern_ids=tuple(pattern.pattern_id for pattern in sacred_geometry.pattern_guidance[:12]),
        caveats=tuple(risks[:9]),
    )


def _architecture_v8_1_records(
    report: CreativeKnowledgeDistillationReport,
) -> tuple[CreativeKnowledgeRecord, ...]:
    tags = {
        "recursive_geometry",
        "motif_translation",
        "mandala_motif",
        "operational_translation",
        "morphogenesis_seed",
        "visual_accent_mapping",
        "kaleidoscopic_composition",
        "signal_to_motion",
    }
    return tuple(
        record
        for record in report.records
        if tags.intersection(record.technique_tags)
        or tags.intersection(record.pattern_tags)
        or "geometry" in record.summary.lower()
        or "visual" in record.summary.lower()
        or "composition" in record.summary.lower()
    )


def _unsupported_claim_risks(
    query: str,
    *,
    symbolic_translation: SymbolicTranslationReport,
    sacred_geometry: SacredGeometryReport,
) -> tuple[str, ...]:
    tokens = _tokens(query)
    risks = [
        f"Request includes '{token}', which must remain bounded and cannot be treated as implemented reconstruction."
        for token in sorted(tokens.intersection(UNSUPPORTED_ARCHITECTURE_CLAIM_TOKENS))
    ]
    risks.extend(symbolic_translation.unsupported_claim_risks)
    risks.extend(sacred_geometry.unsupported_claim_risks)
    return _dedupe(risks)[:9]


def _interpretation_boundaries(risks: Sequence[str]) -> tuple[str, ...]:
    boundaries = [
        "Use sacred architecture as creative spatial guidance, not historical, religious, or engineering authority.",
        (
            "Use reverse engineering only for textual/spatial-description hypotheses "
            "unless measured data exists elsewhere."
        ),
        (
            "Do not claim image-based reconstruction, LIDAR, photogrammetry, CAD/BIM, "
            "real survey, or safety certification."
        ),
        "Do not start V8.5 narrative, V8.6 immersive composer, HoloMind, HOLOiVERSE, or preview runtime mutation.",
    ]
    if risks:
        boundaries.append(
            "Unsupported reconstruction, authority, or sacred-meaning risks require bounded wording or HITL."
        )
    return tuple(boundaries)


def _hitl_questions(risks: Sequence[str]) -> tuple[str, ...]:
    questions = [
        "Should interactive architecture preview be scoped later with explicit product and validation requirements?",
    ]
    if risks:
        questions.extend(
            [
                "Which architectural observations are user-authored text versus claims requiring measured evidence?",
                "Should image, LIDAR, CAD, venue safety, or tradition-specific claims be removed or HITL-reviewed?",
            ]
        )
    return tuple(questions[:8])


def _reused_surface_ids(
    *,
    creative_translation: CreativeTranslation,
    v8_1_records: Sequence[CreativeKnowledgeRecord],
    symbolic_translation: SymbolicTranslationReport,
    sacred_geometry: SacredGeometryReport,
) -> tuple[str, ...]:
    del creative_translation
    surfaces = ["v3_creative_translation", symbolic_translation.capability_id, sacred_geometry.capability_id]
    if v8_1_records:
        surfaces.append("v8_1_creative_knowledge_distillation")
    return tuple(dict.fromkeys(surfaces))


def _patterns_for_text(value: str) -> tuple[str, ...]:
    pattern_ids: list[str] = []
    for token in _ordered_tokens(value):
        pattern_ids.extend(_ARCHITECTURE_ALIASES.get(token, ()))
        if token in _BLUEPRINTS:
            pattern_ids.append(token)
    return _dedupe(pattern_ids)


def _add_source(
    sources: dict[str, list[str]],
    evidence: dict[str, list[str]],
    pattern_id: str,
    source: str,
    evidence_line: str,
) -> None:
    if pattern_id not in _BLUEPRINTS:
        return
    sources.setdefault(pattern_id, []).append(source)
    evidence.setdefault(pattern_id, []).append(evidence_line)


def _tokens(value: str) -> frozenset[str]:
    return frozenset(_TOKEN_PATTERN.findall(_normalize(value)))


def _ordered_tokens(value: str) -> tuple[str, ...]:
    return _dedupe(_TOKEN_PATTERN.findall(_normalize(value)))


def _normalize(value: str) -> str:
    return value.lower().replace("/", " ").replace("_", " ")


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        normalized = value.strip()
        if normalized and normalized not in seen:
            seen.append(normalized)
    return tuple(seen)


def _clip(value: str, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "..."


__all__ = [
    "build_v8_4_sacred_architecture_engine",
    "detect_sacred_architecture_terms",
    "sacred_architecture_prompt_lines",
    "sacred_architecture_roadmap_assessment",
    "V8_4_CAPABILITY_ID",
]
