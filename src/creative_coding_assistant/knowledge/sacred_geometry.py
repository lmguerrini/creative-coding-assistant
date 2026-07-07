"""V8.3 bounded sacred geometry and sacred mathematics guidance engine."""

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
from creative_coding_assistant.knowledge.sacred_geometry_catalog import (
    BLUEPRINT_ROWS,
    DOMAIN_RUNTIME_NAMES,
    GEOMETRY_ALIASES,
    ROADMAP_CLASSIFICATION_ROWS,
    UNSUPPORTED_CLAIM_TOKENS,
)
from creative_coding_assistant.knowledge.sacred_geometry_contracts import (
    V8_3_AUTHORITY_BOUNDARY,
    V8_3_CAPABILITY_ID,
    V8_3_GEOMETRY_SCOPE,
    SacredGeometryConfidence,
    SacredGeometryConfidenceBand,
    SacredGeometryFamily,
    SacredGeometryOperationalGuidance,
    SacredGeometryOperationKind,
    SacredGeometryPatternGuidance,
    SacredGeometryProvenance,
    SacredGeometryReport,
    SacredGeometryRoadmapClassification,
    SacredGeometryRoadmapItemAssessment,
    SacredGeometryValidationFinding,
    SacredGeometryValidationSeverity,
    sacred_geometry_confidence_band,
    sacred_geometry_items_by_classification,
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
class _GeometryBlueprint:
    label: str
    family: SacredGeometryFamily
    taxonomy_path: tuple[str, ...]
    creative_intent: str
    structure: tuple[str, ...]
    algorithms: tuple[str, ...]
    parameters: tuple[str, ...]
    motion: tuple[str, ...]
    color_light: tuple[str, ...]
    audio: tuple[str, ...]
    runtimes: tuple[str, ...]
    notes: tuple[str, ...]
    boundary: str


_TOKEN_PATTERN = re.compile(r"[a-z0-9_.+#']+")
_GEOMETRY_ALIASES = GEOMETRY_ALIASES
_BLUEPRINTS: dict[str, _GeometryBlueprint] = {
    pattern_id: _GeometryBlueprint(
        label=row.label,
        family=SacredGeometryFamily(row.family),
        taxonomy_path=row.taxonomy_path,
        creative_intent=row.creative_intent,
        structure=row.structure,
        algorithms=row.algorithms,
        parameters=row.parameters,
        motion=row.motion,
        color_light=row.color_light,
        audio=row.audio,
        runtimes=row.runtimes,
        notes=row.notes,
        boundary=row.boundary,
    )
    for pattern_id, row in BLUEPRINT_ROWS.items()
}
_ROADMAP_CLASSIFICATIONS: dict[str, tuple[SacredGeometryRoadmapClassification, str, bool, bool]] = {
    item: (SacredGeometryRoadmapClassification(classification), rationale, action_required, hitl_required)
    for item, (classification, rationale, action_required, hitl_required) in ROADMAP_CLASSIFICATION_ROWS.items()
}


def build_v8_3_sacred_geometry_engine(
    query: str,
    *,
    domains: Sequence[CreativeCodingDomain] = (),
    creative_translation: CreativeTranslation | None = None,
    v8_1_distillation: CreativeKnowledgeDistillationReport | None = None,
    v8_2_symbolic_translation: SymbolicTranslationReport | None = None,
) -> SacredGeometryReport:
    """Build a bounded V8.3 sacred geometry report without runtime mutation."""

    translation = creative_translation or derive_creative_translation(query, domains=domains)
    distillation = v8_1_distillation or build_v8_1_creative_knowledge_distillation()
    symbolic = v8_2_symbolic_translation or build_v8_2_symbolic_translation_engine(
        query,
        domains=domains,
        creative_translation=translation,
        v8_1_distillation=distillation,
    )
    v8_1_records = _geometry_v8_1_records(distillation)
    pattern_sources = _collect_pattern_sources(
        query=query,
        creative_translation=translation,
        symbolic_translation=symbolic,
        v8_1_records=v8_1_records,
    )
    patterns = tuple(
        _build_pattern(pattern_id, source_terms, evidence)
        for pattern_id, (source_terms, evidence) in pattern_sources.items()
    )
    risks = _unsupported_claim_risks(query, symbolic_translation=symbolic)
    provenance = _build_provenance(
        query=query,
        creative_translation=translation,
        symbolic_translation=symbolic,
        v8_1_records=v8_1_records,
    )
    confidence = _build_confidence(
        patterns=patterns,
        provenance=provenance,
        risks=risks,
        v8_1_records=v8_1_records,
        symbolic_translation=symbolic,
    )
    roadmap = sacred_geometry_roadmap_assessment()
    classified = sacred_geometry_items_by_classification(roadmap)
    return SacredGeometryReport(
        source_query=_clip(query, 620),
        reused_surface_ids=_reused_surface_ids(
            creative_translation=translation,
            v8_1_records=v8_1_records,
            symbolic_translation=symbolic,
        ),
        pattern_guidance=patterns,
        operational_guidance=_build_operational_guidance(patterns, domains=domains),
        validation_findings=_build_validation_findings(patterns=patterns, risks=risks),
        provenance=provenance,
        confidence=confidence,
        roadmap_assessment=roadmap,
        implemented_roadmap_items=classified[
            SacredGeometryRoadmapClassification.IMPLEMENTED_RUNTIME_BEHAVIOR
        ],
        reused_existing_roadmap_items=classified[
            SacredGeometryRoadmapClassification.REUSED_EXISTING_RUNTIME
        ],
        partial_reusable_roadmap_items=classified[
            SacredGeometryRoadmapClassification.PARTIAL_REUSABLE
        ],
        advisory_only_roadmap_items=classified[SacredGeometryRoadmapClassification.ADVISORY_ONLY],
        product_hitl_required_items=classified[
            SacredGeometryRoadmapClassification.PRODUCT_HITL_REQUIRED
        ],
        later_v8_boundary_items=classified[SacredGeometryRoadmapClassification.LATER_V8_BOUNDARY],
        missing_roadmap_items=classified[SacredGeometryRoadmapClassification.MISSING],
        interpretation_boundaries=_interpretation_boundaries(risks),
        unsupported_claim_risks=risks,
        hitl_questions=_hitl_questions(risks),
    )


def sacred_geometry_prompt_lines(report: SacredGeometryReport) -> tuple[str, ...]:
    """Render compact provider-independent geometry guidance."""

    lines = [
        f"Sacred Geometry Engine boundary: {report.authority_boundary}",
        f"Sacred geometry confidence: {report.confidence.band.value} {report.confidence.score:.2f}.",
    ]
    for pattern in report.pattern_guidance[:8]:
        lines.append(f"Geometry pattern: {pattern.pattern_id}; {pattern.creative_intent}")
        lines.extend(f"{pattern.pattern_id} structure: {item}" for item in pattern.structure_guidance[:2])
        lines.extend(f"{pattern.pattern_id} algorithm: {item}" for item in pattern.algorithm_recommendations[:2])
        lines.extend(f"{pattern.pattern_id} parameter: {item}" for item in pattern.mathematical_parameters[:3])
        lines.extend(f"{pattern.pattern_id} motion: {item}" for item in pattern.motion_mappings[:1])
        lines.extend(f"{pattern.pattern_id} light: {item}" for item in pattern.color_light_mappings[:1])
        lines.extend(f"{pattern.pattern_id} audio: {item}" for item in pattern.audio_harmonic_mappings[:1])
        lines.append(f"{pattern.pattern_id} boundary: {pattern.boundary}")
    for guidance in report.operational_guidance[:10]:
        lines.append(
            "Geometry operation: "
            f"{guidance.kind.value}; {', '.join(guidance.source_pattern_ids)}; "
            f"{' '.join(guidance.guidance[:2])}"
        )
    lines.extend(f"Geometry validation: {item.severity.value}; {item.summary}" for item in report.validation_findings)
    lines.extend(f"Interpretation boundary: {item}" for item in report.interpretation_boundaries)
    lines.extend(f"Unsupported geometry claim risk: {item}" for item in report.unsupported_claim_risks)
    lines.extend(f"HITL geometry question: {item}" for item in report.hitl_questions)
    return tuple(lines[:80])


def sacred_geometry_roadmap_assessment() -> tuple[SacredGeometryRoadmapItemAssessment, ...]:
    """Return the V8.3 roadmap reality-check assessment."""

    return tuple(
        SacredGeometryRoadmapItemAssessment(
            item=item,
            classification=classification,
            rationale=rationale,
            action_required_before_v8_4=action_required,
            hitl_required=hitl_required,
        )
        for item, (classification, rationale, action_required, hitl_required) in _ROADMAP_CLASSIFICATIONS.items()
    )


def detect_sacred_geometry_terms(query: str) -> tuple[str, ...]:
    """Return supported V8.3 geometry pattern ids visible in request text."""

    pattern_ids: list[str] = []
    for token in _ordered_tokens(query):
        pattern_ids.extend(_GEOMETRY_ALIASES.get(token, ()))
        if token in _BLUEPRINTS:
            pattern_ids.append(token)
    return _dedupe(pattern_ids)[:16]


def _collect_pattern_sources(
    *,
    query: str,
    creative_translation: CreativeTranslation,
    symbolic_translation: SymbolicTranslationReport,
    v8_1_records: Sequence[CreativeKnowledgeRecord],
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    sources: dict[str, list[str]] = {}
    evidence: dict[str, list[str]] = {}

    for pattern_id in detect_sacred_geometry_terms(query):
        _add_source(sources, evidence, pattern_id, pattern_id, f"Request-visible geometry cue: {pattern_id}.")

    for ref in (
        *creative_translation.geometric_references,
        *creative_translation.symbolic_references,
        *creative_translation.movement_language,
        *creative_translation.color_material_direction,
        *creative_translation.structure_direction,
        *creative_translation.musical_references,
        *(creative_translation.runtime_recommendations),
    ):
        for pattern_id in _patterns_for_text(ref):
            _add_source(sources, evidence, pattern_id, ref, f"Creative translation geometry signal: {ref}.")

    for mapping in symbolic_translation.motif_mappings:
        for value in (
            mapping.motif_id,
            *mapping.source_terms,
            *mapping.visual_guidance,
            *mapping.motion_guidance,
            *mapping.parameter_guidance,
        ):
            for pattern_id in _patterns_for_text(value):
                _add_source(
                    sources,
                    evidence,
                    pattern_id,
                    value,
                    f"V8.2 symbolic translation signal: {mapping.motif_id}.",
                )

    for record in v8_1_records:
        for value in (*record.technique_tags, *record.pattern_tags, record.title, record.summary):
            for pattern_id in _patterns_for_text(value):
                _add_source(
                    sources,
                    evidence,
                    pattern_id,
                    value,
                    f"V8.1 distilled geometry knowledge signal: {record.record_id}.",
                )

    if not sources:
        _add_source(
            sources,
            evidence,
            "sacred_polygon_circle_grid",
            "geometry",
            "Fallback for broad geometry request without a more specific supported cue.",
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
        pattern_id: (_dedupe(sources[pattern_id])[:12], _dedupe(evidence[pattern_id])[:14])
        for pattern_id in ordered
        if pattern_id in _BLUEPRINTS
    }


def _build_pattern(
    pattern_id: str,
    source_terms: tuple[str, ...],
    evidence: tuple[str, ...],
) -> SacredGeometryPatternGuidance:
    blueprint = _BLUEPRINTS[pattern_id]
    score = min(0.97, 0.5 + 0.07 * len(source_terms) + 0.035 * len(evidence))
    return SacredGeometryPatternGuidance(
        pattern_id=pattern_id,
        label=blueprint.label,
        family=blueprint.family,
        source_terms=source_terms,
        taxonomy_path=blueprint.taxonomy_path,
        creative_intent=blueprint.creative_intent,
        structure_guidance=blueprint.structure,
        algorithm_recommendations=blueprint.algorithms,
        mathematical_parameters=blueprint.parameters,
        motion_mappings=blueprint.motion,
        color_light_mappings=blueprint.color_light,
        audio_harmonic_mappings=blueprint.audio,
        runtime_families=blueprint.runtimes,
        implementation_notes=blueprint.notes,
        boundary=blueprint.boundary,
        evidence=evidence,
        confidence_score=round(score, 2),
    )


def _build_operational_guidance(
    patterns: Sequence[SacredGeometryPatternGuidance],
    *,
    domains: Sequence[CreativeCodingDomain],
) -> tuple[SacredGeometryOperationalGuidance, ...]:
    pattern_ids = tuple(pattern.pattern_id for pattern in patterns)
    runtimes = _dedupe(
        (
            *_domain_runtime_names(domains),
            *(item for pattern in patterns for item in pattern.runtime_families),
        )
    )
    parameters = _dedupe(
        parameter.split(":", maxsplit=1)[0].strip()
        for pattern in patterns
        for parameter in pattern.mathematical_parameters
    )[:16]
    constraints = (
        "Keep sacred geometry language bounded to creative and mathematical guidance.",
        "Do not claim metaphysical proof, cultural authority, ritual efficacy, HoloMind, or HOLOiVERSE behavior.",
        "Do not mutate preview runtime, storage, provider routing, or workflow control from this report.",
    )
    return (
        SacredGeometryOperationalGuidance(
            operation_id="sacred_geometry::structure",
            kind=SacredGeometryOperationKind.STRUCTURE,
            source_pattern_ids=pattern_ids,
            guidance=_collect_guidance(patterns, "structure_guidance"),
            parameter_names=parameters,
            runtime_families=runtimes,
            implementation_notes=("Choose one primary geometry family before layering secondary motifs.",),
            constraints=constraints,
        ),
        SacredGeometryOperationalGuidance(
            operation_id="sacred_geometry::parameterization",
            kind=SacredGeometryOperationKind.PARAMETERIZATION,
            source_pattern_ids=pattern_ids,
            guidance=_collect_guidance(patterns, "mathematical_parameters"),
            parameter_names=parameters,
            runtime_families=runtimes,
            implementation_notes=("Expose parameters with bounded ranges and deterministic defaults.",),
            constraints=constraints,
        ),
        SacredGeometryOperationalGuidance(
            operation_id="sacred_geometry::recursion",
            kind=SacredGeometryOperationKind.RECURSION,
            source_pattern_ids=pattern_ids,
            guidance=_recursion_guidance(patterns),
            parameter_names=tuple(name for name in parameters if name in {"recursion_depth", "depth", "iterations"}),
            runtime_families=runtimes,
            implementation_notes=("Cap recursion, iteration, and segment counts before generation.",),
            constraints=constraints,
        ),
        SacredGeometryOperationalGuidance(
            operation_id="sacred_geometry::morphogenesis",
            kind=SacredGeometryOperationKind.MORPHOGENESIS,
            source_pattern_ids=pattern_ids,
            guidance=_morphogenesis_guidance(patterns),
            parameter_names=tuple(
                name
                for name in parameters
                if name in {"feed_rate", "kill_rate", "field_scale", "growth_rate", "state_count"}
            ),
            runtime_families=runtimes,
            implementation_notes=("Frame simulation-like systems as visual guidance unless execution is requested.",),
            constraints=constraints,
        ),
        SacredGeometryOperationalGuidance(
            operation_id="sacred_geometry::motion_mapping",
            kind=SacredGeometryOperationKind.MOTION_MAPPING,
            source_pattern_ids=pattern_ids,
            guidance=_collect_guidance(patterns, "motion_mappings"),
            parameter_names=parameters,
            runtime_families=runtimes,
            implementation_notes=("Tie motion to geometry state, phase, depth, or field values.",),
            constraints=constraints,
        ),
        SacredGeometryOperationalGuidance(
            operation_id="sacred_geometry::color_light_mapping",
            kind=SacredGeometryOperationKind.COLOR_LIGHT_MAPPING,
            source_pattern_ids=pattern_ids,
            guidance=_collect_guidance(patterns, "color_light_mappings"),
            parameter_names=parameters,
            runtime_families=runtimes,
            implementation_notes=("Use color and light as readable encodings of geometric state.",),
            constraints=constraints,
        ),
        SacredGeometryOperationalGuidance(
            operation_id="sacred_geometry::audio_harmonic_mapping",
            kind=SacredGeometryOperationKind.AUDIO_HARMONIC_MAPPING,
            source_pattern_ids=pattern_ids,
            guidance=_collect_guidance(patterns, "audio_harmonic_mappings"),
            parameter_names=parameters,
            runtime_families=tuple(runtime for runtime in runtimes if runtime in {"Tone.js", "p5.js", "Web Audio API"}),
            implementation_notes=("Keep audio optional and behind browser user-gesture requirements.",),
            constraints=constraints,
        ),
        SacredGeometryOperationalGuidance(
            operation_id="sacred_geometry::architectural_layout_mapping",
            kind=SacredGeometryOperationKind.ARCHITECTURAL_LAYOUT_MAPPING,
            source_pattern_ids=pattern_ids,
            guidance=(
                "Use axes, thresholds, grids, rings, and proportions as spatial layout hints only.",
                "Do not infer buildings, plans, historical systems, or V8.4 architecture behavior.",
            ),
            parameter_names=tuple(
                name
                for name in parameters
                if name in {"axis_angle", "threshold_count", "grid_columns"}
            ),
            runtime_families=runtimes,
            implementation_notes=("Keep this mapping as composition guidance before V8.4.",),
            constraints=constraints,
        ),
        SacredGeometryOperationalGuidance(
            operation_id="sacred_geometry::ritual_pacing_mapping",
            kind=SacredGeometryOperationKind.RITUAL_PACING_MAPPING,
            source_pattern_ids=pattern_ids,
            guidance=(
                "Map rings, thresholds, reveals, and cycles to user-authored pacing or ceremony-like sequence.",
                "Avoid claiming ritual efficacy or tradition-specific meaning without HITL-approved context.",
            ),
            parameter_names=tuple(
                name
                for name in parameters
                if name in {"ring_count", "cycle_phase", "threshold_count"}
            ),
            runtime_families=runtimes,
            implementation_notes=("Treat ritual language as pacing and interaction structure.",),
            constraints=constraints,
        ),
        SacredGeometryOperationalGuidance(
            operation_id="sacred_geometry::safety_boundary",
            kind=SacredGeometryOperationKind.SAFETY_BOUNDARY,
            source_pattern_ids=pattern_ids,
            guidance=tuple(pattern.boundary for pattern in patterns[:8]),
            parameter_names=(),
            runtime_families=(),
            implementation_notes=("Ask HITL before tradition authority, metaphysical proof, or preview asset scope.",),
            constraints=constraints,
        ),
    )


def _build_validation_findings(
    *,
    patterns: Sequence[SacredGeometryPatternGuidance],
    risks: Sequence[str],
) -> tuple[SacredGeometryValidationFinding, ...]:
    findings = [
        SacredGeometryValidationFinding(
            finding_id="sacred_geometry::validation::bounded_scope",
            severity=SacredGeometryValidationSeverity.INFO,
            summary="Geometry guidance is deterministic and report-only.",
            action="Use the report as generation guidance; do not treat it as workflow, storage, or preview mutation.",
        )
    ]
    heavy = {
        SacredGeometryFamily.FRACTAL,
        SacredGeometryFamily.FIELD,
        SacredGeometryFamily.MORPHOGENESIS,
        SacredGeometryFamily.CELLULAR,
        SacredGeometryFamily.PARTICLE,
    }
    if any(pattern.family in heavy for pattern in patterns):
        findings.append(
            SacredGeometryValidationFinding(
                finding_id="sacred_geometry::validation::runtime_budget",
                severity=SacredGeometryValidationSeverity.WARNING,
                summary="Selected geometry includes simulation or high-density generative systems.",
                action="Cap iterations, particle counts, grid sizes, and simulation steps before code generation.",
            )
        )
    if risks:
        findings.append(
            SacredGeometryValidationFinding(
                finding_id="sacred_geometry::validation::unsupported_claims",
                severity=SacredGeometryValidationSeverity.HITL_REQUIRED,
                summary="Request includes terms that could imply unsupported metaphysical or cultural authority.",
                action="Keep wording user-authored and bounded, or request HITL review before stronger claims.",
            )
        )
    findings.append(
        SacredGeometryValidationFinding(
            finding_id="sacred_geometry::validation::preview_assets",
            severity=SacredGeometryValidationSeverity.INFO,
            summary="Interactive preview and demo asset generation are not mutated by V8.3.",
            action="Use product/HITL scope before adding preview UI or generated demo assets.",
        )
    )
    return tuple(findings)


def _build_provenance(
    *,
    query: str,
    creative_translation: CreativeTranslation,
    symbolic_translation: SymbolicTranslationReport,
    v8_1_records: Sequence[CreativeKnowledgeRecord],
) -> tuple[SacredGeometryProvenance, ...]:
    provenance = [
        SacredGeometryProvenance(
            provenance_id="sacred_geometry::request",
            kind="request_signal",
            reference="assistant_request.query",
            summary=_clip(query, 520),
            confidence_signal=None,
        ),
        SacredGeometryProvenance(
            provenance_id="sacred_geometry::creative_translation",
            kind="creative_translation",
            reference="orchestration.creative_translation",
            summary="Reused creative translation geometry, modality, motion, and aesthetic signals.",
            confidence_signal=0.78,
        ),
        SacredGeometryProvenance(
            provenance_id="sacred_geometry::v8_2_symbolic_translation",
            kind="v8_2_symbolic_translation",
            reference=symbolic_translation.capability_id,
            summary="Reused bounded symbolic motif mappings as geometry intent signals.",
            confidence_signal=symbolic_translation.confidence.score,
        ),
        SacredGeometryProvenance(
            provenance_id="sacred_geometry::bounded_catalog",
            kind="bounded_geometry_catalog",
            reference="knowledge.sacred_geometry._BLUEPRINTS",
            summary="Used scoped geometry and mathematics catalog entries with explicit safety boundaries.",
            confidence_signal=0.74,
        ),
    ]
    del creative_translation
    provenance.extend(
        SacredGeometryProvenance(
            provenance_id=f"sacred_geometry::{record.record_id}",
            kind="v8_1_creative_knowledge",
            reference=record.record_id,
            summary=record.summary,
            confidence_signal=record.confidence.score,
        )
        for record in v8_1_records[:8]
    )
    return tuple(provenance[:28])


def _build_confidence(
    *,
    patterns: Sequence[SacredGeometryPatternGuidance],
    provenance: Sequence[SacredGeometryProvenance],
    risks: Sequence[str],
    v8_1_records: Sequence[CreativeKnowledgeRecord],
    symbolic_translation: SymbolicTranslationReport,
) -> SacredGeometryConfidence:
    evidence_count = sum(len(pattern.evidence) for pattern in patterns)
    pattern_bonus = min(0.28, 0.045 * len(patterns))
    evidence_bonus = min(0.22, 0.018 * evidence_count)
    provenance_bonus = min(0.2, 0.025 * len(provenance))
    v8_1_bonus = min(0.1, 0.025 * len(v8_1_records))
    v8_2_bonus = min(0.12, 0.02 * len(symbolic_translation.motif_mappings))
    risk_penalty = min(0.28, 0.08 * len(risks))
    score = max(
        0.05,
        min(
            0.97,
            0.31
            + pattern_bonus
            + evidence_bonus
            + provenance_bonus
            + v8_1_bonus
            + v8_2_bonus
            - risk_penalty,
        ),
    )
    return SacredGeometryConfidence(
        score=round(score, 2),
        band=sacred_geometry_confidence_band(score, guarded=bool(risks)),
        pattern_count=len(patterns),
        evidence_count=evidence_count,
        provenance_count=len(provenance),
        v8_1_record_ids=tuple(record.record_id for record in v8_1_records[:10]),
        v8_2_motif_ids=tuple(mapping.motif_id for mapping in symbolic_translation.motif_mappings[:12]),
        caveats=tuple(risks[:8]),
    )


def _geometry_v8_1_records(
    report: CreativeKnowledgeDistillationReport,
) -> tuple[CreativeKnowledgeRecord, ...]:
    tags = {
        "recursive_geometry",
        "mandala_motif",
        "morphogenesis_seed",
        "signal_to_motion",
        "audio_reactive_mappings",
        "kaleidoscopic_composition",
    }
    return tuple(
        record
        for record in report.records
        if tags.intersection(record.technique_tags)
        or tags.intersection(record.pattern_tags)
        or "geometry" in record.summary.lower()
        or "morphogenesis" in record.summary.lower()
    )


def _unsupported_claim_risks(
    query: str,
    *,
    symbolic_translation: SymbolicTranslationReport,
) -> tuple[str, ...]:
    tokens = _tokens(query)
    risks = [
        f"Request includes '{token}', which must remain bounded creative framing."
        for token in sorted(tokens.intersection(UNSUPPORTED_CLAIM_TOKENS))
    ]
    risks.extend(symbolic_translation.unsupported_claim_risks)
    return _dedupe(risks)[:8]


def _interpretation_boundaries(risks: Sequence[str]) -> tuple[str, ...]:
    boundaries = [
        "Use sacred geometry as creative and mathematical guidance, not metaphysical proof.",
        "Use cultural geometry references as user-authored style cues unless scoped sources and HITL review exist.",
        "Do not start V8.4 architecture, V8.5 narrative, V8.6 composer, HoloMind, or HOLOiVERSE behavior.",
        "Do not mutate preview runtime, storage, provider routing, external DCC integrations, or workflow control.",
    ]
    if risks:
        boundaries.append("Unsupported claim risks require bounded wording or HITL review.")
    return tuple(boundaries)


def _hitl_questions(risks: Sequence[str]) -> tuple[str, ...]:
    questions = [
        "Should interactive geometry preview or demo asset generation be scoped for a later product pass?",
    ]
    if risks:
        questions.extend(
            [
                "Which sacred or cultural meanings are explicitly user-authored and safe to preserve?",
                "Should metaphysical, tradition-specific, or ritual-efficacy language be removed or HITL-reviewed?",
            ]
        )
    return tuple(questions[:8])


def _reused_surface_ids(
    *,
    creative_translation: CreativeTranslation,
    v8_1_records: Sequence[CreativeKnowledgeRecord],
    symbolic_translation: SymbolicTranslationReport,
) -> tuple[str, ...]:
    del creative_translation
    surfaces = ["v3_creative_translation", symbolic_translation.capability_id]
    if v8_1_records:
        surfaces.append("v8_1_creative_knowledge_distillation")
    return tuple(dict.fromkeys(surfaces))


def _recursion_guidance(patterns: Sequence[SacredGeometryPatternGuidance]) -> tuple[str, ...]:
    values = _collect_guidance(patterns, "algorithm_recommendations")
    recursive = tuple(
        item
        for item in values
        if any(
            term in item.lower()
            for term in ("recursive", "fractal", "l-system", "iterated")
        )
    )
    return recursive or ("Use bounded iteration or repeated construction only where the selected geometry needs it.",)


def _morphogenesis_guidance(patterns: Sequence[SacredGeometryPatternGuidance]) -> tuple[str, ...]:
    morphogenetic = tuple(
        item
        for pattern in patterns
        if pattern.family
        in {
            SacredGeometryFamily.FIELD,
            SacredGeometryFamily.MORPHOGENESIS,
            SacredGeometryFamily.CELLULAR,
            SacredGeometryFamily.PARTICLE,
            SacredGeometryFamily.GROWTH,
        }
        for item in (*pattern.structure_guidance, *pattern.algorithm_recommendations)
    )
    return _dedupe(morphogenetic)[:8] or ("Treat morphogenesis as optional visual structure unless requested.",)


def _patterns_for_text(value: str) -> tuple[str, ...]:
    pattern_ids: list[str] = []
    for token in _ordered_tokens(value):
        pattern_ids.extend(_GEOMETRY_ALIASES.get(token, ()))
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


def _collect_guidance(
    patterns: Sequence[SacredGeometryPatternGuidance],
    field_name: str,
) -> tuple[str, ...]:
    return _dedupe(item for pattern in patterns for item in getattr(pattern, field_name))[:8]


def _domain_runtime_names(domains: Sequence[CreativeCodingDomain]) -> tuple[str, ...]:
    return tuple(
        DOMAIN_RUNTIME_NAMES[domain.value]
        for domain in domains
        if domain.value in DOMAIN_RUNTIME_NAMES
    )


def _tokens(value: str) -> frozenset[str]:
    return frozenset(_TOKEN_PATTERN.findall(_normalize(value)))


def _ordered_tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(_normalize(value)))


def _normalize(value: str) -> str:
    lowered = value.lower().replace("l-system", "lsystem")
    return " ".join(lowered.replace("-", " ").split())


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _clip(value: str, limit: int) -> str:
    normalized = " ".join(value.strip().split())
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "..."


__all__ = [
    "SacredGeometryConfidence", "SacredGeometryConfidenceBand", "SacredGeometryFamily",
    "SacredGeometryOperationKind", "SacredGeometryOperationalGuidance",
    "SacredGeometryPatternGuidance", "SacredGeometryProvenance", "SacredGeometryReport",
    "SacredGeometryRoadmapClassification", "SacredGeometryRoadmapItemAssessment",
    "SacredGeometryValidationFinding", "SacredGeometryValidationSeverity",
    "V8_3_AUTHORITY_BOUNDARY", "V8_3_CAPABILITY_ID", "V8_3_GEOMETRY_SCOPE",
    "build_v8_3_sacred_geometry_engine", "detect_sacred_geometry_terms",
    "sacred_geometry_prompt_lines", "sacred_geometry_roadmap_assessment",
]
