"""Operational guidance helpers for the V8.4 sacred architecture engine."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.knowledge.sacred_architecture_catalog import PATTERN_ROWS
from creative_coding_assistant.knowledge.sacred_architecture_contracts import (
    SacredArchitectureFamily,
    SacredArchitectureOperationalGuidance,
    SacredArchitectureOperationKind,
    SacredArchitecturePatternGuidance,
    SacredArchitectureSemanticEdge,
    SacredArchitectureSemanticNode,
    SacredArchitectureSemanticRelationship,
    SacredArchitectureSemanticRole,
    SacredArchitectureValidationFinding,
    SacredArchitectureValidationSeverity,
)
from creative_coding_assistant.knowledge.sacred_geometry_catalog import DOMAIN_RUNTIME_NAMES


def build_sacred_architecture_operational_guidance(
    patterns: Sequence[SacredArchitecturePatternGuidance],
    *,
    domains: Sequence[CreativeCodingDomain],
) -> tuple[SacredArchitectureOperationalGuidance, ...]:
    """Build provider-independent operations from selected architecture patterns."""

    pattern_ids = tuple(pattern.pattern_id for pattern in patterns)
    runtimes = _dedupe(
        (
            *_domain_runtime_names(domains),
            *(item for pattern in patterns for item in pattern.runtime_families),
        )
    )
    parameters = _dedupe(
        parameter
        for pattern in patterns
        for parameter in PATTERN_ROWS[pattern.pattern_id].parameters
    )[:18]
    constraints = (
        "Keep sacred architecture language bounded to creative spatial guidance.",
        "Do not claim image reconstruction, LIDAR, photogrammetry, CAD, real survey, or safety certification.",
        "Do not mutate preview runtime, storage, provider routing, workflow control, or later V8 capabilities.",
    )
    return (
        _operation(
            "sacred_architecture::proportion",
            SacredArchitectureOperationKind.PROPORTION_GUIDANCE,
            pattern_ids,
            _collect_guidance(patterns, "proportion_guidance"),
            parameters,
            runtimes,
            ("Expose ratio choices as tunable creative parameters.",),
            constraints,
        ),
        _operation(
            "sacred_architecture::planimetry",
            SacredArchitectureOperationKind.PLANIMETRY_LAYOUT,
            pattern_ids,
            _collect_guidance(patterns, "plan_guidance"),
            parameters,
            runtimes,
            ("Represent plans as zones, paths, nodes, and edges before ornament.",),
            constraints,
        ),
        _operation(
            "sacred_architecture::axis_symmetry",
            SacredArchitectureOperationKind.AXIS_SYMMETRY,
            pattern_ids,
            _collect_guidance(patterns, "axis_guidance"),
            parameters,
            runtimes,
            ("Make primary, secondary, radial, or local axes visible in generated structure.",),
            constraints,
        ),
        _operation(
            "sacred_architecture::threshold_procession",
            SacredArchitectureOperationKind.THRESHOLD_PROCESSION,
            pattern_ids,
            _collect_guidance(patterns, "threshold_guidance"),
            parameters,
            runtimes,
            ("Treat procession as spatial pacing, not narrative generation or ritual efficacy.",),
            constraints,
        ),
        _operation(
            "sacred_architecture::center_periphery_topology",
            SacredArchitectureOperationKind.CENTER_PERIPHERY_TOPOLOGY,
            pattern_ids,
            _collect_guidance(patterns, "center_periphery_guidance"),
            parameters,
            runtimes,
            ("Name centers, peripheries, voids, and boundaries explicitly.",),
            constraints,
        ),
        _operation(
            "sacred_architecture::geometry_mapping",
            SacredArchitectureOperationKind.GEOMETRY_TO_ARCHITECTURE_MAPPING,
            pattern_ids,
            tuple(
                f"Map V8.3 geometry pattern '{item}' into architecture layout guidance."
                for item in _geometry_ids(patterns)
            ),
            parameters,
            runtimes,
            ("Reuse V8.3 geometry ids as evidence; do not duplicate geometry engines.",),
            constraints,
        ),
        _operation(
            "sacred_architecture::symbolic_mapping",
            SacredArchitectureOperationKind.SYMBOLIC_TO_SPATIAL_MAPPING,
            pattern_ids,
            tuple(f"Map V8.2 symbolic motif '{item}' into spatial role guidance." for item in _symbol_ids(patterns)),
            parameters,
            runtimes,
            ("Reuse V8.2 motifs as symbolic cues without authoritative interpretation.",),
            constraints,
        ),
        _operation(
            "sacred_architecture::semantic_graph",
            SacredArchitectureOperationKind.SEMANTIC_GRAPH,
            pattern_ids,
            _collect_guidance(patterns, "topology_guidance"),
            parameters,
            runtimes,
            ("Use semantic nodes and edges as explainable topology, not a real-building graph.",),
            constraints,
        ),
        _operation(
            "sacred_architecture::installation_planning",
            SacredArchitectureOperationKind.INSTALLATION_PLANNING,
            pattern_ids,
            _collect_guidance(patterns, "installation_guidance"),
            parameters,
            runtimes,
            ("Include audience flow, sightline, dwell, clearance, and sensory-zone guidance where relevant.",),
            constraints,
        ),
        _operation(
            "sacred_architecture::reverse_engineering",
            SacredArchitectureOperationKind.REVERSE_ENGINEERING_GUIDANCE,
            pattern_ids,
            _collect_guidance(patterns, "reverse_engineering_cues"),
            parameters,
            runtimes,
            ("Treat reverse engineering as hypothesis from text unless separate measured data exists.",),
            constraints,
        ),
        _operation(
            "sacred_architecture::safety_boundary",
            SacredArchitectureOperationKind.SAFETY_BOUNDARY,
            pattern_ids,
            tuple(pattern.boundary for pattern in patterns[:8]),
            (),
            (),
            ("Ask HITL before stronger reconstruction, venue, safety, tradition, or preview claims.",),
            constraints,
        ),
    )


def build_sacred_architecture_semantic_nodes(
    patterns: Sequence[SacredArchitecturePatternGuidance],
) -> tuple[SacredArchitectureSemanticNode, ...]:
    """Build bounded semantic topology nodes for selected architecture patterns."""

    pattern_ids = tuple(pattern.pattern_id for pattern in patterns)
    nodes = [
        SacredArchitectureSemanticNode(
            node_id="architecture::entry",
            label="Entry",
            role=SacredArchitectureSemanticRole.ENTRY,
            source_pattern_ids=pattern_ids,
            guidance="Define where the viewer or participant first orients to the spatial system.",
        )
    ]
    if _has_guidance(patterns, "threshold_guidance"):
        nodes.append(
            SacredArchitectureSemanticNode(
                node_id="architecture::threshold",
                label="Threshold",
                role=SacredArchitectureSemanticRole.THRESHOLD,
                source_pattern_ids=pattern_ids,
                guidance="Mark gates, apertures, density changes, or sensory transitions as explicit crossings.",
            )
        )
    if _has_guidance(patterns, "axis_guidance"):
        nodes.append(
            SacredArchitectureSemanticNode(
                node_id="architecture::axis",
                label="Axis",
                role=SacredArchitectureSemanticRole.AXIS,
                source_pattern_ids=pattern_ids,
                guidance="Use alignment, sightline, bay rhythm, or light as the organizing axis.",
            )
        )
    if _has_family(patterns, SacredArchitectureFamily.RADIAL) or _mentions_center(patterns):
        nodes.append(
            SacredArchitectureSemanticNode(
                node_id="architecture::center",
                label="Center",
                role=SacredArchitectureSemanticRole.CENTER,
                source_pattern_ids=pattern_ids,
                guidance="Protect a focal center, void, work, chamber, or luminous anchor.",
            )
        )
    if _has_guidance(patterns, "center_periphery_guidance"):
        nodes.append(
            SacredArchitectureSemanticNode(
                node_id="architecture::periphery",
                label="Periphery",
                role=SacredArchitectureSemanticRole.PERIPHERY,
                source_pattern_ids=pattern_ids,
                guidance="Assign edge, perimeter, circulation, and support zones clear roles.",
            )
        )
    if _has_family(patterns, SacredArchitectureFamily.INSTALLATION):
        nodes.append(
            SacredArchitectureSemanticNode(
                node_id="architecture::exhibit",
                label="Exhibit Or Installation Node",
                role=SacredArchitectureSemanticRole.EXHIBIT,
                source_pattern_ids=pattern_ids,
                guidance="Name hero work, interaction core, projection zone, or distributed anchors.",
            )
        )
    if _has_family(patterns, SacredArchitectureFamily.LIGHT_ORIENTATION):
        nodes.append(
            SacredArchitectureSemanticNode(
                node_id="architecture::light",
                label="Light Source",
                role=SacredArchitectureSemanticRole.LIGHT_SOURCE,
                source_pattern_ids=pattern_ids,
                guidance="Use light as a bounded orientation and reveal system.",
            )
        )
    return tuple(nodes[:18])


def build_sacred_architecture_semantic_edges(
    nodes: Sequence[SacredArchitectureSemanticNode],
    patterns: Sequence[SacredArchitecturePatternGuidance],
) -> tuple[SacredArchitectureSemanticEdge, ...]:
    """Build bounded semantic topology edges for selected architecture patterns."""

    node_ids = {node.node_id for node in nodes}
    pattern_ids = tuple(pattern.pattern_id for pattern in patterns)
    edges: list[SacredArchitectureSemanticEdge] = []
    if {"architecture::entry", "architecture::threshold"}.issubset(node_ids):
        edges.append(
            _edge(
                "architecture::entry->threshold",
                "architecture::entry",
                "architecture::threshold",
                SacredArchitectureSemanticRelationship.THRESHOLD_CROSSING,
                pattern_ids,
                "Entry should lead into an explicit threshold crossing or state change.",
            )
        )
    if {"architecture::threshold", "architecture::axis"}.issubset(node_ids):
        edges.append(
            _edge(
                "architecture::threshold->axis",
                "architecture::threshold",
                "architecture::axis",
                SacredArchitectureSemanticRelationship.PROCESSION,
                pattern_ids,
                "Threshold crossing should clarify the next axis, path, or sightline.",
            )
        )
    if {"architecture::axis", "architecture::center"}.issubset(node_ids):
        edges.append(
            _edge(
                "architecture::axis->center",
                "architecture::axis",
                "architecture::center",
                SacredArchitectureSemanticRelationship.SIGHTLINE,
                pattern_ids,
                "Axis should resolve toward a center, focus, chamber, light source, or work.",
            )
        )
    if {"architecture::center", "architecture::periphery"}.issubset(node_ids):
        edges.append(
            _edge(
                "architecture::center->periphery",
                "architecture::center",
                "architecture::periphery",
                SacredArchitectureSemanticRelationship.CENTER_PERIPHERY,
                pattern_ids,
                "Center and periphery should have distinct visual, circulation, or sensory responsibilities.",
            )
        )
    if {"architecture::entry", "architecture::exhibit"}.issubset(node_ids):
        edges.append(
            _edge(
                "architecture::entry->exhibit",
                "architecture::entry",
                "architecture::exhibit",
                SacredArchitectureSemanticRelationship.PROCESSION,
                pattern_ids,
                "Audience path should move from orientation into exhibit or installation nodes.",
            )
        )
    if {"architecture::light", "architecture::center"}.issubset(node_ids):
        edges.append(
            _edge(
                "architecture::light->center",
                "architecture::light",
                "architecture::center",
                SacredArchitectureSemanticRelationship.SIGHTLINE,
                pattern_ids,
                "Light should reinforce the focal hierarchy without claiming historical analysis.",
            )
        )
    return tuple(edges[:24])


def build_sacred_architecture_validation_findings(
    *,
    patterns: Sequence[SacredArchitecturePatternGuidance],
    risks: Sequence[str],
) -> tuple[SacredArchitectureValidationFinding, ...]:
    """Build deterministic validation findings for architecture guidance."""

    findings = [
        SacredArchitectureValidationFinding(
            finding_id="sacred_architecture::validation::bounded_scope",
            severity=SacredArchitectureValidationSeverity.INFO,
            summary="Architecture guidance is deterministic, local, and report-only.",
            action=(
                "Use the report as generation guidance; do not treat it as survey, "
                "CAD, storage, or preview mutation."
            ),
        )
    ]
    if any(pattern.family is SacredArchitectureFamily.INSTALLATION for pattern in patterns):
        findings.append(
            SacredArchitectureValidationFinding(
                finding_id="sacred_architecture::validation::installation_boundary",
                severity=SacredArchitectureValidationSeverity.WARNING,
                summary="Installation guidance may imply physical space, clearance, audience flow, or safety concerns.",
                action=(
                    "Keep guidance conceptual unless a human provides venue dimensions, "
                    "safety constraints, and review."
                ),
            )
        )
    if risks:
        findings.append(
            SacredArchitectureValidationFinding(
                finding_id="sacred_architecture::validation::unsupported_reconstruction",
                severity=SacredArchitectureValidationSeverity.HITL_REQUIRED,
                summary=(
                    "Request includes terms that could imply unsupported reconstruction, "
                    "survey, or authority claims."
                ),
                action="Keep wording bounded or request HITL review before stronger architectural analysis claims.",
            )
        )
    findings.append(
        SacredArchitectureValidationFinding(
            finding_id="sacred_architecture::validation::preview_boundary",
            severity=SacredArchitectureValidationSeverity.INFO,
            summary="Interactive architecture preview, CAD export, and generated venue assets are not mutated by V8.4.",
            action="Use product/HITL scope before adding preview UI, DCC/CAD, or reconstruction features.",
        )
    )
    return tuple(findings)


def _operation(
    operation_id: str,
    kind: SacredArchitectureOperationKind,
    pattern_ids: tuple[str, ...],
    guidance: tuple[str, ...],
    parameters: tuple[str, ...],
    runtimes: tuple[str, ...],
    notes: tuple[str, ...],
    constraints: tuple[str, ...],
) -> SacredArchitectureOperationalGuidance:
    return SacredArchitectureOperationalGuidance(
        operation_id=operation_id,
        kind=kind,
        source_pattern_ids=pattern_ids,
        guidance=guidance or ("No additional architecture guidance beyond selected pattern boundaries.",),
        parameter_names=parameters,
        runtime_families=runtimes,
        implementation_notes=notes,
        constraints=constraints,
    )


def _edge(
    edge_id: str,
    from_node_id: str,
    to_node_id: str,
    relationship: SacredArchitectureSemanticRelationship,
    pattern_ids: tuple[str, ...],
    guidance: str,
) -> SacredArchitectureSemanticEdge:
    return SacredArchitectureSemanticEdge(
        edge_id=edge_id,
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        relationship=relationship,
        source_pattern_ids=pattern_ids,
        guidance=guidance,
    )


def _collect_guidance(
    patterns: Sequence[SacredArchitecturePatternGuidance],
    field_name: str,
) -> tuple[str, ...]:
    return _dedupe(item for pattern in patterns for item in getattr(pattern, field_name))[:10]


def _has_guidance(patterns: Sequence[SacredArchitecturePatternGuidance], field_name: str) -> bool:
    return any(getattr(pattern, field_name) for pattern in patterns)


def _has_family(patterns: Sequence[SacredArchitecturePatternGuidance], family: SacredArchitectureFamily) -> bool:
    return any(pattern.family is family for pattern in patterns)


def _mentions_center(patterns: Sequence[SacredArchitecturePatternGuidance]) -> bool:
    return any("center" in item.lower() for pattern in patterns for item in pattern.center_periphery_guidance)


def _geometry_ids(patterns: Sequence[SacredArchitecturePatternGuidance]) -> tuple[str, ...]:
    return _dedupe(item for pattern in patterns for item in pattern.geometry_mappings)[:10]


def _symbol_ids(patterns: Sequence[SacredArchitecturePatternGuidance]) -> tuple[str, ...]:
    return _dedupe(item for pattern in patterns for item in pattern.symbolic_mappings)[:10]


def _domain_runtime_names(domains: Sequence[CreativeCodingDomain]) -> tuple[str, ...]:
    return tuple(
        DOMAIN_RUNTIME_NAMES[domain.value]
        for domain in domains
        if domain.value in DOMAIN_RUNTIME_NAMES
    )


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        normalized = value.strip()
        if normalized and normalized not in seen:
            seen.append(normalized)
    return tuple(seen)


__all__ = [
    "build_sacred_architecture_operational_guidance",
    "build_sacred_architecture_semantic_edges",
    "build_sacred_architecture_semantic_nodes",
    "build_sacred_architecture_validation_findings",
]
