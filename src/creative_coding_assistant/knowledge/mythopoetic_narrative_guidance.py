"""Guidance helpers for the V8.5 mythopoetic narrative engine."""

from __future__ import annotations

import re
from collections.abc import Sequence

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.knowledge.mythopoetic_narrative_contracts import (
    MythopoeticNarrativeGraphRelationship,
    MythopoeticNarrativeGraphRole,
    MythopoeticNarrativeOperationalGuidance,
    MythopoeticNarrativeOperationKind,
    MythopoeticNarrativePatternGuidance,
    MythopoeticNarrativeScene,
    MythopoeticNarrativeSymbolEdge,
    MythopoeticNarrativeSymbolNode,
    MythopoeticNarrativeValidationFinding,
    MythopoeticNarrativeValidationSeverity,
)
from creative_coding_assistant.knowledge.sacred_geometry_catalog import DOMAIN_RUNTIME_NAMES


def build_mythopoetic_narrative_symbol_nodes(
    patterns: Sequence[MythopoeticNarrativePatternGuidance],
) -> tuple[MythopoeticNarrativeSymbolNode, ...]:
    """Build bounded narrative symbol graph nodes from selected patterns."""

    nodes: list[MythopoeticNarrativeSymbolNode] = []
    for pattern in patterns[:10]:
        nodes.append(
            MythopoeticNarrativeSymbolNode(
                node_id=_node_id(pattern.pattern_id, "archetype"),
                label=pattern.label,
                role=MythopoeticNarrativeGraphRole.ARCHETYPE,
                source_pattern_ids=(pattern.pattern_id,),
                guidance=pattern.narrative_intent,
            )
        )
        nodes.append(
            MythopoeticNarrativeSymbolNode(
                node_id=_node_id(pattern.pattern_id, "symbol"),
                label=_symbol_label(pattern),
                role=MythopoeticNarrativeGraphRole.SYMBOL,
                source_pattern_ids=(pattern.pattern_id,),
                guidance=pattern.symbolic_transitions[0],
            )
        )
    nodes.append(
        MythopoeticNarrativeSymbolNode(
            node_id="narrative_audience",
            label="Audience Experience",
            role=MythopoeticNarrativeGraphRole.AUDIENCE,
            source_pattern_ids=tuple(pattern.pattern_id for pattern in patterns[:10]),
            guidance="Audience experience is described as intended design posture, not measured transformation.",
        )
    )
    return tuple(nodes)


def build_mythopoetic_narrative_symbol_edges(
    nodes: Sequence[MythopoeticNarrativeSymbolNode],
    patterns: Sequence[MythopoeticNarrativePatternGuidance],
) -> tuple[MythopoeticNarrativeSymbolEdge, ...]:
    """Build bounded narrative symbol graph edges from selected patterns."""

    node_ids = {node.node_id for node in nodes}
    edges: list[MythopoeticNarrativeSymbolEdge] = []
    for pattern in patterns[:10]:
        archetype_id = _node_id(pattern.pattern_id, "archetype")
        symbol_id = _node_id(pattern.pattern_id, "symbol")
        if archetype_id in node_ids and symbol_id in node_ids:
            edges.append(
                MythopoeticNarrativeSymbolEdge(
                    edge_id=f"{archetype_id}__reinforces__{symbol_id}",
                    from_node_id=archetype_id,
                    to_node_id=symbol_id,
                    relationship=MythopoeticNarrativeGraphRelationship.REINFORCES,
                    source_pattern_ids=(pattern.pattern_id,),
                    guidance=f"{pattern.label} uses '{_symbol_label(pattern)}' as its primary symbolic focus.",
                )
            )
    for current, next_pattern in zip(patterns, patterns[1:], strict=False):
        from_node_id = _node_id(current.pattern_id, "symbol")
        to_node_id = _node_id(next_pattern.pattern_id, "archetype")
        if from_node_id in node_ids and to_node_id in node_ids:
            edges.append(
                MythopoeticNarrativeSymbolEdge(
                    edge_id=f"{from_node_id}__threshold_to__{to_node_id}",
                    from_node_id=from_node_id,
                    to_node_id=to_node_id,
                    relationship=MythopoeticNarrativeGraphRelationship.THRESHOLD_TO,
                    source_pattern_ids=(current.pattern_id, next_pattern.pattern_id),
                    guidance=(
                        f"Sequence {_symbol_label(current)} toward {next_pattern.label} "
                        "as a bounded symbolic transition."
                    ),
                )
            )
    if patterns:
        last_symbol = _node_id(patterns[min(len(patterns), 10) - 1].pattern_id, "symbol")
        if last_symbol in node_ids:
            edges.append(
                MythopoeticNarrativeSymbolEdge(
                    edge_id=f"{last_symbol}__addresses__narrative_audience",
                    from_node_id=last_symbol,
                    to_node_id="narrative_audience",
                    relationship=MythopoeticNarrativeGraphRelationship.ADDRESSES_AUDIENCE,
                    source_pattern_ids=tuple(pattern.pattern_id for pattern in patterns[:10]),
                    guidance="Close the narrative by explaining intended audience experience and safe boundaries.",
                )
            )
    return tuple(edges)


def build_mythopoetic_narrative_scene_sequence(
    patterns: Sequence[MythopoeticNarrativePatternGuidance],
) -> tuple[MythopoeticNarrativeScene, ...]:
    """Build a deterministic opening-to-return narrative sequence."""

    phase_rows = (
        ("opening", "Opening Signal", 0),
        ("call", "Call and Orientation", 1),
        ("threshold", "Threshold Crossing", 2),
        ("ordeal", "Intensified Encounter", 3),
        ("integration", "Integration and Recognition", 4),
        ("return", "Return and Communication", -1),
    )
    scenes: list[MythopoeticNarrativeScene] = []
    for index, (phase, title, arc_index) in enumerate(phase_rows):
        pattern = patterns[index % len(patterns)]
        journey = _at(pattern.journey_arc, arc_index)
        emotional = _at(pattern.emotional_arc, arc_index)
        transition = _at(pattern.symbolic_transitions, min(index, len(pattern.symbolic_transitions) - 1))
        scenes.append(
            MythopoeticNarrativeScene(
                scene_id=f"mythopoetic_scene::{phase}",
                phase=phase,  # type: ignore[arg-type]
                title=f"{title}: {pattern.label}",
                source_pattern_ids=(pattern.pattern_id,),
                narrative_function=journey,
                emotional_state=emotional,
                symbolic_focus=_symbol_label(pattern),
                visual_guidance=pattern.visual_mappings[:3],
                motion_guidance=pattern.motion_mappings[:3],
                audio_guidance=pattern.audio_mappings[:2],
                spatial_guidance=pattern.spatial_installation_mappings[:3],
                transition_out=transition,
                evidence=pattern.evidence[:8],
            )
        )
    return tuple(scenes)


def build_mythopoetic_narrative_operational_guidance(
    patterns: Sequence[MythopoeticNarrativePatternGuidance],
    *,
    domains: Sequence[CreativeCodingDomain] = (),
) -> tuple[MythopoeticNarrativeOperationalGuidance, ...]:
    """Build provider-independent operations from narrative patterns."""

    source_ids = tuple(pattern.pattern_id for pattern in patterns)
    runtimes = _runtime_families(domains)
    return (
        MythopoeticNarrativeOperationalGuidance(
            operation_id="mythopoetic_narrative::structure",
            kind=MythopoeticNarrativeOperationKind.NARRATIVE_STRUCTURE,
            source_pattern_ids=source_ids,
            guidance=_collect(patterns, "archetypal_structure", limit=7),
            parameter_names=("narrative_phase", "threshold_progress", "integration_level"),
            runtime_families=runtimes,
            implementation_notes=("Keep phases inspectable before adding visual density.",),
        ),
        MythopoeticNarrativeOperationalGuidance(
            operation_id="mythopoetic_narrative::scene_sequence",
            kind=MythopoeticNarrativeOperationKind.SCENE_SEQUENCE,
            source_pattern_ids=source_ids,
            guidance=_collect(patterns, "journey_arc", limit=8),
            parameter_names=("scene_index", "scene_duration", "transition_progress"),
            runtime_families=runtimes,
            constraints=("Do not create a live preview or immersive composer scene graph.",),
        ),
        MythopoeticNarrativeOperationalGuidance(
            operation_id="mythopoetic_narrative::symbolic_dialogue",
            kind=MythopoeticNarrativeOperationKind.SYMBOLIC_DIALOGUE,
            source_pattern_ids=source_ids,
            guidance=_symbolic_dialogue_guidance(patterns),
            parameter_names=("voice_a_weight", "voice_b_weight", "call_response_phase"),
            runtime_families=runtimes,
            constraints=("Keep symbolic voices user-authored and non-authoritative.",),
        ),
        MythopoeticNarrativeOperationalGuidance(
            operation_id="mythopoetic_narrative::symbolic_transition",
            kind=MythopoeticNarrativeOperationKind.SYMBOLIC_TRANSITION,
            source_pattern_ids=source_ids,
            guidance=_collect(patterns, "symbolic_transitions", limit=8),
            parameter_names=("transition_progress", "symbol_weight", "resolution_bias"),
            runtime_families=runtimes,
        ),
        MythopoeticNarrativeOperationalGuidance(
            operation_id="mythopoetic_narrative::emotional_arc",
            kind=MythopoeticNarrativeOperationKind.EMOTIONAL_ARC,
            source_pattern_ids=source_ids,
            guidance=_collect(patterns, "emotional_arc", limit=8),
            parameter_names=("emotional_intensity", "tension_level", "release_level"),
            runtime_families=runtimes,
            constraints=("Describe intended emotional pacing, not viewer diagnosis or therapy.",),
        ),
        MythopoeticNarrativeOperationalGuidance(
            operation_id="mythopoetic_narrative::visual_mapping",
            kind=MythopoeticNarrativeOperationKind.VISUAL_MAPPING,
            source_pattern_ids=source_ids,
            guidance=_collect(patterns, "visual_mappings", limit=8),
            parameter_names=("palette_shift", "density_level", "form_coherence"),
            runtime_families=runtimes,
        ),
        MythopoeticNarrativeOperationalGuidance(
            operation_id="mythopoetic_narrative::motion_mapping",
            kind=MythopoeticNarrativeOperationKind.MOTION_MAPPING,
            source_pattern_ids=source_ids,
            guidance=_collect(patterns, "motion_mappings", limit=8),
            parameter_names=("motion_intensity", "phase_offset", "path_progress"),
            runtime_families=runtimes,
        ),
        MythopoeticNarrativeOperationalGuidance(
            operation_id="mythopoetic_narrative::audio_mapping",
            kind=MythopoeticNarrativeOperationKind.AUDIO_MAPPING,
            source_pattern_ids=source_ids,
            guidance=_collect(patterns, "audio_mappings", limit=8),
            parameter_names=("audio_density", "filter_shift", "cadence_phase"),
            runtime_families=runtimes,
            constraints=("Require explicit user interaction before browser audio playback.",),
        ),
        MythopoeticNarrativeOperationalGuidance(
            operation_id="mythopoetic_narrative::spatial_installation_mapping",
            kind=MythopoeticNarrativeOperationKind.SPATIAL_INSTALLATION_MAPPING,
            source_pattern_ids=source_ids,
            guidance=_collect(patterns, "spatial_installation_mappings", limit=8),
            parameter_names=("entry_zone", "threshold_count", "dwell_time", "exit_cue"),
            runtime_families=runtimes,
            constraints=("Keep spatial guidance conceptual unless a product-scoped preview path exists.",),
        ),
        MythopoeticNarrativeOperationalGuidance(
            operation_id="mythopoetic_narrative::creative_brief",
            kind=MythopoeticNarrativeOperationKind.CREATIVE_BRIEF,
            source_pattern_ids=source_ids,
            guidance=_collect(patterns, "creative_brief_points", limit=8),
            runtime_families=runtimes,
        ),
        MythopoeticNarrativeOperationalGuidance(
            operation_id="mythopoetic_narrative::explanation",
            kind=MythopoeticNarrativeOperationKind.EXPLANATION,
            source_pattern_ids=source_ids,
            guidance=_collect(patterns, "explanation_points", limit=8),
            constraints=("Separate implemented guidance from future HoloGenesis or V8.6+ ideas.",),
        ),
        MythopoeticNarrativeOperationalGuidance(
            operation_id="mythopoetic_narrative::audience_communication",
            kind=MythopoeticNarrativeOperationKind.AUDIENCE_COMMUNICATION,
            source_pattern_ids=source_ids,
            guidance=_collect(patterns, "audience_communication", limit=8),
            constraints=("Avoid authoritative religious, esoteric, or psychological claims.",),
        ),
        MythopoeticNarrativeOperationalGuidance(
            operation_id="mythopoetic_narrative::validation",
            kind=MythopoeticNarrativeOperationKind.VALIDATION,
            source_pattern_ids=source_ids,
            guidance=("Validate evidence, boundaries, later-version scope, and unsupported claim risks.",),
        ),
        MythopoeticNarrativeOperationalGuidance(
            operation_id="mythopoetic_narrative::safety_boundary",
            kind=MythopoeticNarrativeOperationKind.SAFETY_BOUNDARY,
            source_pattern_ids=source_ids,
            guidance=tuple(pattern.boundary for pattern in patterns[:8]),
            constraints=(
                "No psychotherapy or diagnosis.",
                "No spiritual, religious, esoteric, or ritual authority.",
                "No V8.6 immersive audiovisual composer behavior.",
            ),
        ),
    )


def build_mythopoetic_narrative_validation_findings(
    *,
    patterns: Sequence[MythopoeticNarrativePatternGuidance],
    risks: Sequence[str],
    later_boundary_requests: Sequence[str],
) -> tuple[MythopoeticNarrativeValidationFinding, ...]:
    """Build deterministic validation findings for V8.5 narrative guidance."""

    findings = [
        MythopoeticNarrativeValidationFinding(
            finding_id="mythopoetic_narrative::validation::bounded_scope",
            severity=MythopoeticNarrativeValidationSeverity.INFO,
            summary="Narrative guidance is report-only and bounded to creative coding/project framing.",
            action="Use fields as planning guidance; do not treat them as doctrine, therapy, or runtime control.",
        )
    ]
    if len(patterns) > 8:
        findings.append(
            MythopoeticNarrativeValidationFinding(
                finding_id="mythopoetic_narrative::validation::pattern_density",
                severity=MythopoeticNarrativeValidationSeverity.WARNING,
                summary="Many narrative patterns were selected; generated concepts may need hierarchy.",
                action="Prioritize one primary arc and use secondary patterns as supporting scenes.",
            )
        )
    if risks:
        findings.append(
            MythopoeticNarrativeValidationFinding(
                finding_id="mythopoetic_narrative::validation::unsupported_claims",
                severity=MythopoeticNarrativeValidationSeverity.HITL_REQUIRED,
                summary="Request language includes authoritative religious, esoteric, or psychological claim risk.",
                action="Reframe as creative metaphor or ask HITL to provide scoped source/wording decisions.",
            )
        )
    if later_boundary_requests:
        findings.append(
            MythopoeticNarrativeValidationFinding(
                finding_id="mythopoetic_narrative::validation::later_v8_boundary",
                severity=MythopoeticNarrativeValidationSeverity.WARNING,
                summary="Request language touches V8.6+ preview/composer/showcase/OS territory.",
                action="Keep V8.5 output as narrative guidance only and defer implementation of later surfaces.",
            )
        )
    findings.append(
        MythopoeticNarrativeValidationFinding(
            finding_id="mythopoetic_narrative::validation::audience_boundary",
            severity=MythopoeticNarrativeValidationSeverity.INFO,
            summary="Audience transformation is represented as intended experience, not measured effect.",
            action="Use audience communication fields for framing; do not claim psychological outcomes.",
        )
    )
    return tuple(findings)


def _node_id(pattern_id: str, role: str) -> str:
    return f"narrative_{_safe_id(pattern_id)}_{role}"


def _symbol_label(pattern: MythopoeticNarrativePatternGuidance) -> str:
    return pattern.source_terms[0].replace("_", " ").title()


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")


def _at(values: Sequence[str], index: int) -> str:
    if not values:
        return "Hold the narrative beat clearly."
    if index < 0:
        return values[-1]
    return values[min(index, len(values) - 1)]


def _collect(
    patterns: Sequence[MythopoeticNarrativePatternGuidance],
    attr: str,
    *,
    limit: int,
) -> tuple[str, ...]:
    collected: list[str] = []
    for pattern in patterns:
        for item in getattr(pattern, attr):
            if item not in collected:
                collected.append(item)
            if len(collected) >= limit:
                return tuple(collected)
    return tuple(collected) or ("Keep narrative guidance bounded and inspectable.",)


def _symbolic_dialogue_guidance(
    patterns: Sequence[MythopoeticNarrativePatternGuidance],
) -> tuple[str, ...]:
    if len(patterns) == 1:
        pattern = patterns[0]
        return (
            f"Let {_symbol_label(pattern)} answer its own opposite through before/after states.",
            "Keep symbolic voice as a visible design relation, not an external authority.",
        )
    first, second = patterns[0], patterns[1]
    return (
        f"Stage {_symbol_label(first)} and {_symbol_label(second)} as contrasting visual states.",
        "Use call-response timing, mirrored movement, or alternating density to show exchange.",
        "Keep all symbolic dialogue user-authored and explainable.",
    )


def _runtime_families(domains: Sequence[CreativeCodingDomain]) -> tuple[str, ...]:
    names: list[str] = []
    for domain in domains:
        label = DOMAIN_RUNTIME_NAMES.get(domain.value) or domain.value
        if label not in names:
            names.append(label)
    return tuple(names[:10])


__all__ = [
    "build_mythopoetic_narrative_operational_guidance",
    "build_mythopoetic_narrative_scene_sequence",
    "build_mythopoetic_narrative_symbol_edges",
    "build_mythopoetic_narrative_symbol_nodes",
    "build_mythopoetic_narrative_validation_findings",
]
