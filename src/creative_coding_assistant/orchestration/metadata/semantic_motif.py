"""Bounded Semantic Motif Engine for V3.2 workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import AssistantRequest
from creative_coding_assistant.orchestration.creative_composition import (
    CreativeCompositionPlan,
)
from creative_coding_assistant.orchestration.creative_constraint_priorities import (
    CreativeConstraintPrioritization,
)
from creative_coding_assistant.orchestration.creative_constraints import (
    CreativeConstraintSolution,
)
from creative_coding_assistant.orchestration.creative_hierarchy import (
    CreativeHierarchyPlan,
)
from creative_coding_assistant.orchestration.creative_intent import (
    CreativeIntentDecomposition,
)
from creative_coding_assistant.orchestration.creative_planning import (
    CreativeExecutionPlan,
)
from creative_coding_assistant.orchestration.creative_quality_prediction import (
    CreativeQualityPrediction,
)
from creative_coding_assistant.orchestration.creative_strategy import (
    CreativeStrategyProfile,
)
from creative_coding_assistant.orchestration.creative_technique import (
    CreativeTechniqueProfile,
)
from creative_coding_assistant.orchestration.creative_tradeoffs import (
    CreativeTradeoffProfile,
)
from creative_coding_assistant.orchestration.creative_translation import (
    CreativeTranslation,
)
from creative_coding_assistant.orchestration.generative_structure import (
    GenerativeModuleKind,
    GenerativeStructureBlueprint,
)
from creative_coding_assistant.orchestration.procedural_structure import (
    ProceduralFamily,
    ProceduralStructurePlan,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.symbolic_narrative import (
    SymbolicNarrativePlan,
)

if TYPE_CHECKING:
    from creative_coding_assistant.orchestration.creative_reasoning import (
        CreativeReasoningResult,
    )

SemanticMotifId = Literal[
    "seed",
    "spiral",
    "threshold",
    "mirror",
    "void",
    "center",
    "circumference",
    "axis",
    "descent",
    "ascent",
    "fragmentation",
    "reintegration",
    "wave",
    "lattice",
    "network",
    "pearl",
    "flame",
    "root",
    "tree",
    "vessel",
    "mandala",
    "grid",
    "swarm",
    "orbit",
    "pulse",
    "breath",
    "gate",
    "eye",
    "river",
    "constellation",
]
SemanticMotifRole = Literal[
    "anchor",
    "threshold",
    "transformation",
    "connector",
    "counterpoint",
    "rhythm",
    "spatial_order",
    "material_signal",
    "fallback",
]
SemanticMotifHierarchyLevel = Literal[
    "primary",
    "secondary",
    "supporting",
    "fallback",
]

SEMANTIC_MOTIF_AUTHORITY_BOUNDARY = (
    "The Semantic Motif Engine organizes recurring symbolic motifs as "
    "inspectable design metadata only; it does not generate code, validate "
    "doctrine, make hidden-knowledge claims, auto-select runtimes, change "
    "preview behavior, route providers or models, run autonomous loops, "
    "implement V4 multi-agent runtime, or implement HoloMind."
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9_.+#-]+")
_AMBIGUOUS_MOTIF_TOKENS = frozenset(
    {
        "archetypal",
        "deep",
        "evocative",
        "meaningful",
        "maybe",
        "mystery",
        "mystical",
        "profound",
        "something",
        "symbolic",
        "vibe",
    }
)
_UNSUPPORTED_CLAIM_TOKENS = frozenset(
    {
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
    }
)


class SemanticMotif(BaseModel):
    """One recurring semantic motif and its bounded design role."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    motif_id: SemanticMotifId
    label: str = Field(min_length=1, max_length=80)
    role: SemanticMotifRole
    hierarchy_level: SemanticMotifHierarchyLevel
    rationale: str = Field(min_length=1, max_length=320)
    recurrence_guidance: tuple[str, ...] = Field(min_length=1, max_length=5)
    transformation_guidance: tuple[str, ...] = Field(min_length=1, max_length=5)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=8)


class SemanticMotifStructureMapping(BaseModel):
    """Mapping from motif to procedural/generative structure."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    motif_id: SemanticMotifId
    procedural_families: tuple[ProceduralFamily, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    generative_module_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    generative_module_kinds: tuple[GenerativeModuleKind, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    structural_behavior: str = Field(min_length=1, max_length=340)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=8)


class SemanticMotifCompositionMapping(BaseModel):
    """Mapping from motif to visual composition."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    motif_id: SemanticMotifId
    composition_role: str = Field(min_length=1, max_length=260)
    spatial_anchor: str = Field(min_length=1, max_length=260)
    rhythm_or_density_guidance: str = Field(min_length=1, max_length=260)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=8)


class SemanticMotifNarrativeMapping(BaseModel):
    """Mapping from motif to narrative phase or symbolic arc."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    motif_id: SemanticMotifId
    narrative_function: str = Field(min_length=1, max_length=300)
    phase_alignment: tuple[str, ...] = Field(min_length=1, max_length=5)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=8)


class SemanticMotifParameterMapping(BaseModel):
    """Mapping from motif to named generative parameters."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    motif_id: SemanticMotifId
    parameter_names: tuple[str, ...] = Field(min_length=1, max_length=8)
    parameter_guidance: str = Field(min_length=1, max_length=300)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=8)


class SemanticMotifFallbackPlan(BaseModel):
    """Lower-risk motif plan when symbolic scope is too broad or unclear."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    fallback_primary_motif: SemanticMotifId
    fallback_secondary_motifs: tuple[SemanticMotifId, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    simplification_strategy: str = Field(min_length=1, max_length=300)
    preserved_meaning: str = Field(min_length=1, max_length=260)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=5)


class SemanticMotifSystem(BaseModel):
    """Inspectable motif system derived before generation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["semantic_motif_engine"] = "semantic_motif_engine"
    motif_system_name: str = Field(min_length=1, max_length=180)
    primary_motifs: tuple[SemanticMotif, ...] = Field(min_length=1, max_length=3)
    secondary_motifs: tuple[SemanticMotif, ...] = Field(min_length=1, max_length=8)
    motif_hierarchy: tuple[str, ...] = Field(min_length=1, max_length=8)
    motif_recurrence_plan: tuple[str, ...] = Field(min_length=1, max_length=8)
    motif_transformation_plan: tuple[str, ...] = Field(min_length=1, max_length=8)
    motif_to_structure_mapping: tuple[SemanticMotifStructureMapping, ...] = Field(
        min_length=1,
        max_length=10,
    )
    motif_to_composition_mapping: tuple[SemanticMotifCompositionMapping, ...] = Field(
        min_length=1,
        max_length=10,
    )
    motif_to_narrative_mapping: tuple[SemanticMotifNarrativeMapping, ...] = Field(
        min_length=1,
        max_length=10,
    )
    motif_to_parameter_mapping: tuple[SemanticMotifParameterMapping, ...] = Field(
        min_length=1,
        max_length=10,
    )
    coherence_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    overuse_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    underuse_risks: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    unsupported_symbolic_claims: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    motif_fallback_plan: SemanticMotifFallbackPlan
    unresolved_motif_gaps: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=SEMANTIC_MOTIF_AUTHORITY_BOUNDARY,
        max_length=640,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=12)


@dataclass(frozen=True)
class _MotifContext:
    request: AssistantRequest
    route_decision: RouteDecision | None
    creative_translation: CreativeTranslation | None
    creative_intent: CreativeIntentDecomposition | None
    creative_hierarchy: CreativeHierarchyPlan | None
    creative_plan: CreativeExecutionPlan | None
    creative_constraints: CreativeConstraintSolution | None
    creative_constraint_priorities: CreativeConstraintPrioritization | None
    creative_strategy: CreativeStrategyProfile | None
    creative_techniques: CreativeTechniqueProfile | None
    creative_tradeoffs: CreativeTradeoffProfile | None
    creative_quality_prediction: CreativeQualityPrediction | None
    symbolic_narrative: SymbolicNarrativePlan | None
    creative_composition: CreativeCompositionPlan | None
    procedural_structure: ProceduralStructurePlan | None
    generative_structure: GenerativeStructureBlueprint | None
    creative_reasoning: CreativeReasoningResult | None
    text: str
    tokens: frozenset[str]


def derive_semantic_motif_system(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None = None,
    creative_intent: CreativeIntentDecomposition | None = None,
    creative_hierarchy: CreativeHierarchyPlan | None = None,
    creative_plan: CreativeExecutionPlan | None = None,
    creative_constraints: CreativeConstraintSolution | None = None,
    creative_constraint_priorities: CreativeConstraintPrioritization | None = None,
    creative_strategy: CreativeStrategyProfile | None = None,
    creative_techniques: CreativeTechniqueProfile | None = None,
    creative_tradeoffs: CreativeTradeoffProfile | None = None,
    creative_quality_prediction: CreativeQualityPrediction | None = None,
    symbolic_narrative: SymbolicNarrativePlan | None = None,
    creative_composition: CreativeCompositionPlan | None = None,
    procedural_structure: ProceduralStructurePlan | None = None,
    generative_structure: GenerativeStructureBlueprint | None = None,
    creative_reasoning: CreativeReasoningResult | None = None,
) -> SemanticMotifSystem:
    """Derive a bounded motif system without generating artifacts."""

    context = _context(
        request=request,
        route_decision=route_decision,
        creative_translation=creative_translation,
        creative_intent=creative_intent,
        creative_hierarchy=creative_hierarchy,
        creative_plan=creative_plan,
        creative_constraints=creative_constraints,
        creative_constraint_priorities=creative_constraint_priorities,
        creative_strategy=creative_strategy,
        creative_techniques=creative_techniques,
        creative_tradeoffs=creative_tradeoffs,
        creative_quality_prediction=creative_quality_prediction,
        symbolic_narrative=symbolic_narrative,
        creative_composition=creative_composition,
        procedural_structure=procedural_structure,
        generative_structure=generative_structure,
        creative_reasoning=creative_reasoning,
    )
    scored = _score_motifs(context)
    selected = _selected_motifs(scored)
    if "fragmentation" in selected and "reintegration" in selected:
        primary_ids = ("fragmentation", "reintegration")
        secondary_source = tuple(
            motif for motif in selected if motif not in primary_ids
        )
    else:
        primary_ids = tuple(selected[:2])
        secondary_source = tuple(selected[2:8])
    secondary_ids = secondary_source[:6] or _fallback_secondary(primary_ids)
    motifs = tuple(
        _motif(
            motif_id,
            level="primary" if motif_id in primary_ids else "secondary",
            context=context,
            evidence=scored[motif_id][1],
        )
        for motif_id in (*primary_ids, *secondary_ids)
    )
    primary = tuple(item for item in motifs if item.hierarchy_level == "primary")
    secondary = tuple(item for item in motifs if item.hierarchy_level == "secondary")
    unresolved = _unresolved_gaps(context, primary, secondary)
    unsupported_claims = _unsupported_symbolic_claims(context)
    return SemanticMotifSystem(
        motif_system_name=_motif_system_name(context, primary),
        primary_motifs=primary,
        secondary_motifs=secondary,
        motif_hierarchy=_motif_hierarchy(primary, secondary),
        motif_recurrence_plan=_recurrence_plan(primary, secondary, context),
        motif_transformation_plan=_transformation_plan(primary, secondary, context),
        motif_to_structure_mapping=_structure_mappings(motifs, context),
        motif_to_composition_mapping=_composition_mappings(motifs, context),
        motif_to_narrative_mapping=_narrative_mappings(motifs, context),
        motif_to_parameter_mapping=_parameter_mappings(motifs, context),
        coherence_risks=_coherence_risks(primary, secondary, context),
        overuse_risks=_overuse_risks(primary, context),
        underuse_risks=_underuse_risks(secondary, context),
        unsupported_symbolic_claims=unsupported_claims,
        motif_fallback_plan=_fallback_plan(primary, secondary, context),
        unresolved_motif_gaps=unresolved,
        hitl_questions=_hitl_questions(unresolved, unsupported_claims),
        prompt_guidance=_prompt_guidance(primary, secondary, unsupported_claims),
        evidence=_evidence(context, primary, secondary),
    )


def semantic_motif_prompt_lines(system: SemanticMotifSystem) -> tuple[str, ...]:
    """Render motif metadata as compact prompt guidance."""

    lines = [
        f"Authority boundary: {system.authority_boundary}",
        f"Semantic motif system: {system.motif_system_name}",
        "Primary motifs: "
        + ", ".join(motif.motif_id for motif in system.primary_motifs)
        + ".",
        "Secondary motifs: "
        + ", ".join(motif.motif_id for motif in system.secondary_motifs[:6])
        + ".",
    ]
    for motif in (*system.primary_motifs, *system.secondary_motifs[:4]):
        lines.append(
            "Motif role: "
            f"{motif.motif_id}; {motif.role}; {motif.hierarchy_level}; "
            f"{motif.rationale}"
        )
    lines.extend(f"Motif hierarchy: {item}" for item in system.motif_hierarchy[:6])
    lines.extend(
        f"Motif recurrence: {item}" for item in system.motif_recurrence_plan[:6]
    )
    lines.extend(
        f"Motif transformation: {item}" for item in system.motif_transformation_plan[:6]
    )
    lines.extend(
        f"Motif structure mapping: {item.motif_id}; {item.structural_behavior}"
        for item in system.motif_to_structure_mapping[:6]
    )
    lines.extend(
        "Motif composition mapping: "
        f"{item.motif_id}; {item.composition_role}; {item.spatial_anchor}"
        for item in system.motif_to_composition_mapping[:6]
    )
    lines.extend(
        "Motif narrative mapping: "
        f"{item.motif_id}; {item.narrative_function}; "
        f"{', '.join(item.phase_alignment)}"
        for item in system.motif_to_narrative_mapping[:6]
    )
    lines.extend(
        f"Motif parameter mapping: {item.motif_id}; {', '.join(item.parameter_names)}"
        for item in system.motif_to_parameter_mapping[:6]
    )
    lines.extend(f"Motif coherence risk: {item}" for item in system.coherence_risks)
    lines.extend(f"Motif overuse risk: {item}" for item in system.overuse_risks)
    lines.extend(f"Motif underuse risk: {item}" for item in system.underuse_risks)
    lines.extend(
        f"Unsupported symbolic claim risk: {item}"
        for item in system.unsupported_symbolic_claims
    )
    lines.append(
        "Motif fallback plan: "
        f"{system.motif_fallback_plan.fallback_primary_motif}; "
        f"{system.motif_fallback_plan.simplification_strategy}"
    )
    lines.extend(
        f"Unresolved motif gap: {item}" for item in system.unresolved_motif_gaps
    )
    lines.extend(f"HITL motif question: {item}" for item in system.hitl_questions)
    lines.extend(f"Motif prompt guidance: {item}" for item in system.prompt_guidance)
    return tuple(lines[:48])


def _context(
    *,
    request: AssistantRequest,
    route_decision: RouteDecision | None,
    creative_translation: CreativeTranslation | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_constraints: CreativeConstraintSolution | None,
    creative_constraint_priorities: CreativeConstraintPrioritization | None,
    creative_strategy: CreativeStrategyProfile | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
    creative_quality_prediction: CreativeQualityPrediction | None,
    symbolic_narrative: SymbolicNarrativePlan | None,
    creative_composition: CreativeCompositionPlan | None,
    procedural_structure: ProceduralStructurePlan | None,
    generative_structure: GenerativeStructureBlueprint | None,
    creative_reasoning: CreativeReasoningResult | None,
) -> _MotifContext:
    parts = [
        request.query,
        creative_translation.creative_intent if creative_translation else "",
        creative_intent.primary_expression if creative_intent else "",
        creative_strategy.primary_strategy if creative_strategy else "",
        creative_techniques.primary_technique if creative_techniques else "",
    ]
    if creative_translation is not None:
        parts.extend(creative_translation.structure_direction)
        parts.extend(creative_translation.movement_language)
        parts.extend(creative_translation.color_material_direction)
    if symbolic_narrative is not None:
        parts.extend(
            (
                symbolic_narrative.narrative_archetype,
                symbolic_narrative.symbolic_arc,
                symbolic_narrative.experiential_goal,
            )
        )
        for phase in symbolic_narrative.phases:
            parts.extend(
                (
                    phase.title,
                    phase.symbolic_function,
                    phase.visual_state,
                    phase.motion_state,
                    phase.audio_state or "",
                )
            )
    if creative_composition is not None:
        parts.extend(
            (
                creative_composition.composition_pattern,
                creative_composition.primary_focal_point,
                creative_composition.spatial_organization,
                creative_composition.rhythm_plan,
                creative_composition.balance_plan,
            )
        )
        parts.extend(creative_composition.secondary_focal_elements)
    if procedural_structure is not None:
        parts.extend(procedural_structure.recommended_families)
        parts.append(procedural_structure.combination_strategy)
        parts.append(procedural_structure.spatial_structure_plan)
        parts.append(procedural_structure.temporal_structure_plan)
    if generative_structure is not None:
        parts.append(generative_structure.blueprint_name)
        parts.append(generative_structure.generative_architecture)
        parts.append(generative_structure.spatial_evolution)
        parts.append(generative_structure.temporal_evolution)
        parts.extend(module.kind for module in generative_structure.procedural_modules)
        parts.extend(
            parameter.name for parameter in generative_structure.parameter_schema
        )
    if creative_reasoning is not None:
        parts.append(creative_reasoning.recommended_creative_direction)
    text = _normalize(" ".join(parts))
    return _MotifContext(
        request=request,
        route_decision=route_decision,
        creative_translation=creative_translation,
        creative_intent=creative_intent,
        creative_hierarchy=creative_hierarchy,
        creative_plan=creative_plan,
        creative_constraints=creative_constraints,
        creative_constraint_priorities=creative_constraint_priorities,
        creative_strategy=creative_strategy,
        creative_techniques=creative_techniques,
        creative_tradeoffs=creative_tradeoffs,
        creative_quality_prediction=creative_quality_prediction,
        symbolic_narrative=symbolic_narrative,
        creative_composition=creative_composition,
        procedural_structure=procedural_structure,
        generative_structure=generative_structure,
        creative_reasoning=creative_reasoning,
        text=text,
        tokens=frozenset(_TOKEN_PATTERN.findall(text)),
    )


def _score_motifs(
    context: _MotifContext,
) -> dict[SemanticMotifId, tuple[int, tuple[str, ...]]]:
    scores: dict[SemanticMotifId, int] = {motif: 0 for motif in _MOTIF_ORDER}
    evidence: dict[SemanticMotifId, list[str]] = {motif: [] for motif in _MOTIF_ORDER}
    request_tokens = frozenset(
        _TOKEN_PATTERN.findall(_normalize(context.request.query))
    )
    for motif in _MOTIF_ORDER:
        if motif in request_tokens:
            _add_score(scores, evidence, motif, 6, f"Explicit request token: {motif}.")
        elif motif in context.tokens:
            _add_score(scores, evidence, motif, 2, f"Upstream signal mentions {motif}.")
    for token, motifs in _TOKEN_MOTIF_ALIASES.items():
        if token in context.tokens:
            for motif in motifs:
                _add_score(scores, evidence, motif, 3, f"Alias signal: {token}.")
    if request_tokens.intersection(
        {
            "dissolve",
            "dissolves",
            "dissolving",
            "particles",
            "phoenix",
            "reassembling",
            "rebirth",
        }
    ):
        _add_score(
            scores,
            evidence,
            "fragmentation",
            6,
            "Explicit dissolution or particle transformation request.",
        )
        _add_score(
            scores,
            evidence,
            "reintegration",
            6,
            "Explicit reassembly or rebirth transformation request.",
        )
    if context.symbolic_narrative is not None:
        for motif in _NARRATIVE_MOTIFS.get(
            context.symbolic_narrative.narrative_archetype,
            (),
        ):
            _add_score(
                scores,
                evidence,
                motif,
                4,
                f"Narrative archetype: {context.symbolic_narrative.narrative_archetype}.",
            )
    if context.creative_composition is not None:
        for motif in _COMPOSITION_MOTIFS.get(
            context.creative_composition.composition_pattern,
            (),
        ):
            _add_score(
                scores,
                evidence,
                motif,
                3,
                f"Composition pattern: {context.creative_composition.composition_pattern}.",
            )
    if context.procedural_structure is not None:
        for family in context.procedural_structure.recommended_families:
            for motif in _PROCEDURAL_MOTIFS.get(family, ()):
                _add_score(scores, evidence, motif, 2, f"Procedural family: {family}.")
    if context.generative_structure is not None:
        for module in context.generative_structure.procedural_modules:
            for motif in _MODULE_MOTIFS.get(module.kind, ()):
                _add_score(
                    scores,
                    evidence,
                    motif,
                    2,
                    f"Generative module: {module.kind}.",
                )
        for parameter in context.generative_structure.parameter_schema:
            for motif in _PARAMETER_MOTIFS.get(parameter.name, ()):
                _add_score(
                    scores,
                    evidence,
                    motif,
                    1,
                    f"Generative parameter: {parameter.name}.",
                )
    return {
        motif: (scores[motif], tuple(evidence[motif][:8])) for motif in _MOTIF_ORDER
    }


def _selected_motifs(
    scored: dict[SemanticMotifId, tuple[int, tuple[str, ...]]],
) -> tuple[SemanticMotifId, ...]:
    selected = sorted(
        (motif for motif, (score, _) in scored.items() if score > 0),
        key=lambda motif: (-scored[motif][0], _MOTIF_ORDER.index(motif)),
    )
    if selected:
        return tuple(selected[:8])
    return ("seed", "center", "threshold", "wave")


def _fallback_secondary(
    primary_ids: tuple[SemanticMotifId, ...],
) -> tuple[SemanticMotifId, ...]:
    fallback = [
        motif
        for motif in ("center", "axis", "pulse", "seed")
        if motif not in primary_ids
    ]
    return tuple(fallback[:3])


def _motif(
    motif_id: SemanticMotifId,
    *,
    level: SemanticMotifHierarchyLevel,
    context: _MotifContext,
    evidence: tuple[str, ...],
) -> SemanticMotif:
    return SemanticMotif(
        motif_id=motif_id,
        label=_label(motif_id),
        role=_role_for_motif(motif_id),
        hierarchy_level=level,
        rationale=_rationale(motif_id, level, context),
        recurrence_guidance=_recurrence_guidance(motif_id, level),
        transformation_guidance=_transformation_guidance(motif_id, context),
        evidence=evidence or (f"{_label(motif_id)} selected as fallback motif.",),
    )


def _motif_system_name(
    context: _MotifContext,
    primary: tuple[SemanticMotif, ...],
) -> str:
    subject = (
        context.creative_intent.primary_expression
        if context.creative_intent is not None
        else context.request.query
    )
    motifs = " / ".join(motif.motif_id for motif in primary)
    return _clip(f"{motifs.title()} Motif System for {subject}", 180)


def _motif_hierarchy(
    primary: tuple[SemanticMotif, ...],
    secondary: tuple[SemanticMotif, ...],
) -> tuple[str, ...]:
    hierarchy = [
        f"Primary motif {index + 1}: {motif.motif_id} acts as {motif.role}."
        for index, motif in enumerate(primary)
    ]
    hierarchy.extend(
        f"Secondary motif: {motif.motif_id} supports {primary[0].motif_id}."
        for motif in secondary[:5]
    )
    return tuple(hierarchy[:8])


def _recurrence_plan(
    primary: tuple[SemanticMotif, ...],
    secondary: tuple[SemanticMotif, ...],
    context: _MotifContext,
) -> tuple[str, ...]:
    plan = [
        f"Repeat {motif.motif_id} at each major visual or narrative phase."
        for motif in primary
    ]
    plan.extend(
        f"Let {motif.motif_id} recur as a quieter secondary cue."
        for motif in secondary[:4]
    )
    if context.creative_composition is not None:
        plan.append(
            "Tie motif recurrence to composition rhythm: "
            f"{context.creative_composition.rhythm_plan}"
        )
    return tuple(_dedupe(plan))[:8]


def _transformation_plan(
    primary: tuple[SemanticMotif, ...],
    secondary: tuple[SemanticMotif, ...],
    context: _MotifContext,
) -> tuple[str, ...]:
    del secondary
    plan = [
        f"Transform {motif.motif_id} through scale, density, phase, or orientation."
        for motif in primary
    ]
    if context.generative_structure is not None:
        phases = ", ".join(
            rule.phase for rule in context.generative_structure.evolution_rules[:4]
        )
        plan.append(f"Align motif transformation with generative phases: {phases}.")
    if context.symbolic_narrative is not None:
        plan.append(
            "Align motif transformation with narrative arc: "
            f"{context.symbolic_narrative.narrative_archetype}."
        )
    return tuple(_dedupe(plan))[:8]


def _structure_mappings(
    motifs: tuple[SemanticMotif, ...],
    context: _MotifContext,
) -> tuple[SemanticMotifStructureMapping, ...]:
    return tuple(_structure_mapping(motif, context) for motif in motifs[:10])


def _structure_mapping(
    motif: SemanticMotif,
    context: _MotifContext,
) -> SemanticMotifStructureMapping:
    families = _families_for_motif(motif.motif_id, context)
    modules = _modules_for_motif(motif.motif_id, context)
    module_ids = tuple(module[0] for module in modules)
    module_kinds = tuple(module[1] for module in modules)
    return SemanticMotifStructureMapping(
        motif_id=motif.motif_id,
        procedural_families=families,
        generative_module_ids=module_ids,
        generative_module_kinds=module_kinds,
        structural_behavior=_structural_behavior(motif.motif_id),
        evidence=_structure_evidence(families, module_kinds),
    )


def _composition_mappings(
    motifs: tuple[SemanticMotif, ...],
    context: _MotifContext,
) -> tuple[SemanticMotifCompositionMapping, ...]:
    return tuple(_composition_mapping(motif, context) for motif in motifs[:10])


def _composition_mapping(
    motif: SemanticMotif,
    context: _MotifContext,
) -> SemanticMotifCompositionMapping:
    if context.creative_composition is not None:
        role = (
            f"Use {motif.motif_id} inside "
            f"{context.creative_composition.composition_pattern}."
        )
        anchor = context.creative_composition.primary_focal_point
        rhythm = context.creative_composition.rhythm_plan
        evidence = (
            f"Composition pattern: {context.creative_composition.composition_pattern}.",
        )
    else:
        role = f"Use {motif.motif_id} as a visible compositional cue."
        anchor = _default_spatial_anchor(motif.motif_id)
        rhythm = "Let recurrence stay visible but not dominant."
        evidence = ("No Creative Composition Planner metadata attached.",)
    return SemanticMotifCompositionMapping(
        motif_id=motif.motif_id,
        composition_role=_clip(role, 260),
        spatial_anchor=_clip(anchor, 260),
        rhythm_or_density_guidance=_clip(rhythm, 260),
        evidence=evidence,
    )


def _narrative_mappings(
    motifs: tuple[SemanticMotif, ...],
    context: _MotifContext,
) -> tuple[SemanticMotifNarrativeMapping, ...]:
    return tuple(_narrative_mapping(motif, context) for motif in motifs[:10])


def _narrative_mapping(
    motif: SemanticMotif,
    context: _MotifContext,
) -> SemanticMotifNarrativeMapping:
    phases = _phase_alignment(motif.motif_id)
    if context.symbolic_narrative is not None:
        function = (
            f"Use {motif.motif_id} to clarify "
            f"{context.symbolic_narrative.narrative_archetype} without "
            "asserting external doctrine."
        )
        evidence = (
            f"Narrative archetype: {context.symbolic_narrative.narrative_archetype}.",
        )
    else:
        function = f"Use {motif.motif_id} as a recurring visual metaphor."
        evidence = ("No Symbolic Narrative Planner metadata attached.",)
    return SemanticMotifNarrativeMapping(
        motif_id=motif.motif_id,
        narrative_function=_clip(function, 300),
        phase_alignment=phases,
        evidence=evidence,
    )


def _parameter_mappings(
    motifs: tuple[SemanticMotif, ...],
    context: _MotifContext,
) -> tuple[SemanticMotifParameterMapping, ...]:
    return tuple(_parameter_mapping(motif, context) for motif in motifs[:10])


def _parameter_mapping(
    motif: SemanticMotif,
    context: _MotifContext,
) -> SemanticMotifParameterMapping:
    parameters = _parameters_for_motif(motif.motif_id, context)
    return SemanticMotifParameterMapping(
        motif_id=motif.motif_id,
        parameter_names=parameters,
        parameter_guidance=(
            f"Use {', '.join(parameters)} to make {motif.motif_id} recur or "
            "transform without adding new runtime behavior."
        ),
        evidence=(f"Motif-to-parameter mapping for {motif.motif_id}.",),
    )


def _coherence_risks(
    primary: tuple[SemanticMotif, ...],
    secondary: tuple[SemanticMotif, ...],
    context: _MotifContext,
) -> tuple[str, ...]:
    risks: list[str] = []
    if len(secondary) > 5:
        risks.append("Too many secondary motifs may blur the motif hierarchy.")
    if context.symbolic_narrative is None:
        risks.append("Narrative phase mapping is inferred rather than planned.")
    if context.creative_composition is None:
        risks.append("Composition mapping is inferred rather than planned.")
    if primary:
        risks.append(
            f"Keep {primary[0].motif_id} visibly dominant over decorative motifs."
        )
    return tuple(_dedupe(risks))[:8]


def _overuse_risks(
    primary: tuple[SemanticMotif, ...],
    context: _MotifContext,
) -> tuple[str, ...]:
    risks = [
        f"Overusing {motif.motif_id} literally may flatten symbolic ambiguity."
        for motif in primary
    ]
    if context.tokens.intersection(_UNSUPPORTED_CLAIM_TOKENS):
        risks.append(
            "Do not turn motif language into factual spiritual, historical, "
            "or doctrinal claims."
        )
    return tuple(_dedupe(risks))[:8]


def _underuse_risks(
    secondary: tuple[SemanticMotif, ...],
    context: _MotifContext,
) -> tuple[str, ...]:
    del context
    return tuple(
        f"Underusing {motif.motif_id} may make the motif system feel disconnected."
        for motif in secondary[:3]
    )


def _unsupported_symbolic_claims(context: _MotifContext) -> tuple[str, ...]:
    claims: list[str] = []
    tokens = sorted(context.tokens.intersection(_UNSUPPORTED_CLAIM_TOKENS))
    if tokens:
        claims.append(
            "Treat symbolic terms as user-supplied design language, not factual "
            f"doctrine: {', '.join(tokens[:6])}."
        )
    if "prove" in context.tokens or "truth" in context.tokens:
        claims.append(
            "Do not claim the artwork proves a spiritual, historical, or "
            "cosmological truth."
        )
    return tuple(_dedupe(claims))[:8]


def _fallback_plan(
    primary: tuple[SemanticMotif, ...],
    secondary: tuple[SemanticMotif, ...],
    context: _MotifContext,
) -> SemanticMotifFallbackPlan:
    fallback_primary = primary[0].motif_id if primary else "seed"
    fallback_secondary = tuple(motif.motif_id for motif in secondary[:2])
    if not fallback_secondary:
        fallback_secondary = ("center", "pulse")
    return SemanticMotifFallbackPlan(
        fallback_primary_motif=fallback_primary,
        fallback_secondary_motifs=fallback_secondary,
        simplification_strategy=(
            f"Preserve {fallback_primary} and remove weak secondary motifs "
            "before changing structure or runtime."
        ),
        preserved_meaning=_fallback_meaning(context, fallback_primary),
        prompt_guidance=(
            "Use fewer motifs before adding new symbolic material.",
            "Keep fallback motifs as design metaphors, not doctrine.",
        ),
    )


def _unresolved_gaps(
    context: _MotifContext,
    primary: tuple[SemanticMotif, ...],
    secondary: tuple[SemanticMotif, ...],
) -> tuple[str, ...]:
    gaps: list[str] = []
    if context.tokens.intersection(_AMBIGUOUS_MOTIF_TOKENS):
        gaps.append("Motif language is abstract; confirm the primary motif emphasis.")
    if "maybe" in context.tokens and len(primary) + len(secondary) > 2:
        gaps.append("Multiple possible motifs compete; confirm which should lead.")
    if context.symbolic_narrative is None:
        gaps.append("No Symbolic Narrative Planner metadata is attached.")
    if context.generative_structure is None:
        gaps.append("No Generative Structure Engine metadata is attached.")
    if context.tokens.intersection(_UNSUPPORTED_CLAIM_TOKENS):
        gaps.append("Unsupported symbolic claims need user-authored framing.")
    return tuple(_dedupe(gaps))[:8]


def _hitl_questions(
    unresolved: tuple[str, ...],
    unsupported_claims: tuple[str, ...],
) -> tuple[str, ...]:
    questions: list[str] = []
    for gap in unresolved:
        lowered = gap.lower()
        if "primary motif" in lowered or "which should lead" in lowered:
            questions.append("Which motif should remain visually dominant?")
        elif "unsupported" in lowered or unsupported_claims:
            questions.append(
                "Which symbolic interpretation should remain user-authored?"
            )
        elif "narrative" in lowered:
            questions.append("Which narrative phase should carry the motif?")
        elif "generative" in lowered:
            questions.append("Which generated structure should express the motif?")
    if unsupported_claims and not questions:
        questions.append("Which symbolic interpretation should remain user-authored?")
    return tuple(_dedupe(questions))[:6]


def _prompt_guidance(
    primary: tuple[SemanticMotif, ...],
    secondary: tuple[SemanticMotif, ...],
    unsupported_claims: tuple[str, ...],
) -> tuple[str, ...]:
    guidance = [
        "Use the motif system as semantic guidance, not executable code.",
        "Make primary motifs recur before adding secondary motif detail.",
        "Map motifs to existing structure, composition, narrative, and parameters.",
        "Avoid unsupported symbolic claims; frame motifs as design metaphors.",
    ]
    if primary:
        guidance.append(f"Keep {primary[0].motif_id} as the clearest motif anchor.")
    if secondary:
        guidance.append(
            "Let secondary motifs support rather than compete with primary motifs."
        )
    if unsupported_claims:
        guidance.append("Ask HITL before asserting any symbolic interpretation.")
    return tuple(_dedupe(guidance))[:8]


def _evidence(
    context: _MotifContext,
    primary: tuple[SemanticMotif, ...],
    secondary: tuple[SemanticMotif, ...],
) -> tuple[str, ...]:
    evidence = [
        "Primary motifs: " + ", ".join(motif.motif_id for motif in primary) + ".",
        "Secondary motifs: "
        + ", ".join(motif.motif_id for motif in secondary[:6])
        + ".",
    ]
    if context.symbolic_narrative is not None:
        evidence.append(
            f"Narrative source: {context.symbolic_narrative.narrative_archetype}."
        )
    if context.creative_composition is not None:
        evidence.append(
            f"Composition source: {context.creative_composition.composition_pattern}."
        )
    if context.procedural_structure is not None:
        evidence.append(
            "Procedural source: "
            f"{context.procedural_structure.primary_structure.family}."
        )
    if context.generative_structure is not None:
        evidence.append(
            "Generative source: "
            f"{context.generative_structure.generative_architecture}."
        )
    return tuple(evidence[:12])


def _add_score(
    scores: dict[SemanticMotifId, int],
    evidence: dict[SemanticMotifId, list[str]],
    motif: SemanticMotifId,
    score: int,
    reason: str,
) -> None:
    scores[motif] += score
    evidence[motif].append(reason)


def _role_for_motif(motif: SemanticMotifId) -> SemanticMotifRole:
    if motif in {"threshold", "gate"}:
        return "threshold"
    if motif in {"fragmentation", "reintegration", "descent", "ascent"}:
        return "transformation"
    if motif in {"spiral", "wave", "orbit", "pulse", "breath", "river"}:
        return "rhythm"
    if motif in {"grid", "lattice", "network", "axis", "center", "circumference"}:
        return "spatial_order"
    if motif in {"mirror", "void", "eye"}:
        return "counterpoint"
    if motif in {"flame", "pearl", "vessel", "root", "tree"}:
        return "material_signal"
    if motif in {"constellation", "swarm"}:
        return "connector"
    return "anchor"


def _rationale(
    motif: SemanticMotifId,
    level: SemanticMotifHierarchyLevel,
    context: _MotifContext,
) -> str:
    source = "available request and planning signals"
    if context.symbolic_narrative is not None:
        source = f"{context.symbolic_narrative.narrative_archetype} narrative"
    return (
        f"Use {motif} as a {level} motif because it is supported by {source} "
        "and can recur without adding new runtime behavior."
    )


def _recurrence_guidance(
    motif: SemanticMotifId,
    level: SemanticMotifHierarchyLevel,
) -> tuple[str, ...]:
    intensity = "major phase" if level == "primary" else "supporting beat"
    return (
        f"Introduce {motif} early as a {intensity}.",
        f"Repeat {motif} through shape, motion, color, or timing.",
    )


def _transformation_guidance(
    motif: SemanticMotifId,
    context: _MotifContext,
) -> tuple[str, ...]:
    guidance = [f"Let {motif} transform through bounded parameter changes."]
    if context.generative_structure is not None:
        guidance.append(
            "Use existing generative evolution rules rather than new systems."
        )
    return tuple(guidance)


def _families_for_motif(
    motif: SemanticMotifId,
    context: _MotifContext,
) -> tuple[ProceduralFamily, ...]:
    families: list[ProceduralFamily] = []
    if context.procedural_structure is not None:
        for family in context.procedural_structure.recommended_families:
            if motif in _PROCEDURAL_MOTIFS.get(family, ()):
                families.append(family)
        if not families:
            families.append(context.procedural_structure.primary_structure.family)
    return tuple(_dedupe(families))[:5]


def _modules_for_motif(
    motif: SemanticMotifId,
    context: _MotifContext,
) -> tuple[tuple[str, GenerativeModuleKind], ...]:
    modules: list[tuple[str, GenerativeModuleKind]] = []
    if context.generative_structure is None:
        return ()
    for module in context.generative_structure.procedural_modules:
        if motif in _MODULE_MOTIFS.get(module.kind, ()):
            modules.append((module.module_id, module.kind))
    if not modules:
        modules.append(
            (
                context.generative_structure.procedural_modules[0].module_id,
                context.generative_structure.procedural_modules[0].kind,
            )
        )
    return tuple(modules[:8])


def _parameters_for_motif(
    motif: SemanticMotifId,
    context: _MotifContext,
) -> tuple[str, ...]:
    names: list[str] = []
    if context.generative_structure is not None:
        available = {
            item.name for item in context.generative_structure.parameter_schema
        }
        for name, motifs in _PARAMETER_MOTIFS.items():
            if motif in motifs and name in available:
                names.append(name)
        if not names:
            names.extend(
                name
                for name in context.generative_structure.control_parameters[:3]
                if name in available
            )
    if not names:
        names = ["global_scale", "time_phase"]
    return tuple(_dedupe(names))[:8]


def _structure_evidence(
    families: tuple[ProceduralFamily, ...],
    module_kinds: tuple[GenerativeModuleKind, ...],
) -> tuple[str, ...]:
    evidence: list[str] = []
    if families:
        evidence.append("Procedural families: " + ", ".join(families) + ".")
    if module_kinds:
        evidence.append("Generative modules: " + ", ".join(module_kinds) + ".")
    return tuple(evidence[:8])


def _structural_behavior(motif: SemanticMotifId) -> str:
    behavior = {
        "fragmentation": "Break form into bounded fragments or particles.",
        "reintegration": "Reassemble fragments into a readable final structure.",
        "spiral": "Map form along a recursive or radial spiral path.",
        "threshold": "Use a boundary, crossing, or state transition.",
        "gate": "Express threshold as a visible entry or aperture.",
        "mirror": "Use reflected or paired structures.",
        "grid": "Repeat motif through a bounded grid or tiling system.",
        "lattice": "Repeat motif through interlocking modular structure.",
        "network": "Express motif as nodes, links, or relational clusters.",
        "wave": "Use phase and amplitude as the motif's structure.",
        "pulse": "Use repeating temporal emphasis as the motif carrier.",
        "orbit": "Use circular motion around a focal path.",
        "mandala": "Use radial symmetry as the motif carrier.",
    }
    return behavior.get(motif, f"Carry {motif} through recurring structure.")


def _default_spatial_anchor(motif: SemanticMotifId) -> str:
    if motif in {"center", "seed", "mandala", "eye"}:
        return "Central focal region."
    if motif in {"threshold", "gate", "axis"}:
        return "Boundary line or crossing plane."
    if motif in {"circumference", "orbit", "spiral"}:
        return "Circular or radial path."
    return "Primary visible structure."


def _phase_alignment(motif: SemanticMotifId) -> tuple[str, ...]:
    if motif in {"seed", "void", "root"}:
        return ("opening",)
    if motif in {"descent", "fragmentation", "mirror"}:
        return ("development", "threshold")
    if motif in {"threshold", "gate", "eye"}:
        return ("threshold",)
    if motif in {"ascent", "reintegration", "flame", "tree"}:
        return ("climax", "resolution")
    return ("opening", "development", "resolution")


def _fallback_meaning(
    context: _MotifContext,
    fallback_primary: SemanticMotifId,
) -> str:
    if context.symbolic_narrative is not None:
        return (
            f"Preserve the {context.symbolic_narrative.narrative_archetype} "
            f"arc through {fallback_primary}."
        )
    return f"Preserve the user's intent through {fallback_primary}."


def _label(motif: SemanticMotifId) -> str:
    return motif.replace("_", " ").title()


def _clip(value: str, limit: int) -> str:
    normalized = " ".join(value.strip().split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "."


def _normalize(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").replace("_", " ").split())


def _dedupe(
    values: list[str] | list[SemanticMotifId] | list[ProceduralFamily],
) -> tuple:
    deduped: list[object] = []
    for value in values:
        cleaned = " ".join(str(value).strip().split())
        candidate = value if not isinstance(value, str) else cleaned
        if cleaned and candidate not in deduped:
            deduped.append(candidate)
    return tuple(deduped)


_MOTIF_ORDER: tuple[SemanticMotifId, ...] = (
    "seed",
    "spiral",
    "threshold",
    "mirror",
    "void",
    "center",
    "circumference",
    "axis",
    "descent",
    "ascent",
    "fragmentation",
    "reintegration",
    "wave",
    "lattice",
    "network",
    "pearl",
    "flame",
    "root",
    "tree",
    "vessel",
    "mandala",
    "grid",
    "swarm",
    "orbit",
    "pulse",
    "breath",
    "gate",
    "eye",
    "river",
    "constellation",
)

_TOKEN_MOTIF_ALIASES: dict[str, tuple[SemanticMotifId, ...]] = {
    "phoenix": ("fragmentation", "reintegration", "flame"),
    "rebirth": ("fragmentation", "reintegration", "seed"),
    "reassembly": ("reintegration",),
    "reassemble": ("reintegration",),
    "reassembling": ("reintegration",),
    "dissolve": ("fragmentation",),
    "dissolves": ("fragmentation",),
    "dissolving": ("fragmentation",),
    "ember": ("flame", "fragmentation"),
    "embers": ("flame", "fragmentation"),
    "radial": ("center", "circumference", "mandala"),
    "recursive": ("spiral", "seed"),
    "audio": ("pulse", "wave", "breath"),
    "rhythm": ("pulse", "wave"),
    "particles": ("fragmentation", "swarm"),
}

_NARRATIVE_MOTIFS: dict[str, tuple[SemanticMotifId, ...]] = {
    "descent_and_return": ("descent", "ascent", "root"),
    "death_and_rebirth": ("fragmentation", "reintegration", "flame", "seed"),
    "emergence_from_chaos": ("void", "seed", "center"),
    "initiation": ("threshold", "gate", "seed"),
    "ascent": ("ascent", "axis", "flame"),
    "dissolution_and_reintegration": (
        "fragmentation",
        "reintegration",
        "river",
    ),
    "expansion_from_seed_to_cosmos": (
        "seed",
        "center",
        "constellation",
    ),
    "fragmentation_and_recomposition": (
        "fragmentation",
        "reintegration",
        "lattice",
    ),
    "threshold_crossing": ("threshold", "gate", "axis"),
    "spiral_transformation": ("spiral", "orbit", "center"),
    "mirror_reflection_journey": ("mirror", "axis", "eye"),
    "dark_to_light_transformation": ("void", "flame", "ascent"),
    "symbolic_vignette": ("seed", "center", "threshold"),
}

_COMPOSITION_MOTIFS: dict[str, tuple[SemanticMotifId, ...]] = {
    "central_emergence": ("center", "seed"),
    "radial_expansion": ("center", "circumference", "mandala"),
    "spiral_composition": ("spiral", "orbit"),
    "layered_depth": ("descent", "ascent"),
    "field_composition": ("swarm", "constellation"),
    "threshold_composition": ("threshold", "gate", "axis"),
    "descent_ascent_composition": ("descent", "ascent", "root"),
    "fragmented_recomposition": ("fragmentation", "reintegration"),
    "mirrored_composition": ("mirror", "axis"),
    "orbiting_focal_structure": ("orbit", "center", "circumference"),
    "distributed_constellation": ("constellation", "network"),
    "minimal_void_and_form_composition": ("void", "center"),
}

_PROCEDURAL_MOTIFS: dict[ProceduralFamily, tuple[SemanticMotifId, ...]] = {
    "fractals": ("seed", "tree", "spiral"),
    "recursive_geometry": ("spiral", "seed", "mandala"),
    "l_systems": ("root", "tree", "seed"),
    "particle_systems": ("fragmentation", "swarm", "reintegration"),
    "boids": ("swarm", "network"),
    "cellular_automata": ("grid", "lattice"),
    "reaction_diffusion": ("wave", "lattice"),
    "voronoi_systems": ("network", "lattice"),
    "noise_fields": ("river", "wave"),
    "flow_fields": ("river", "wave", "swarm"),
    "signed_distance_fields": ("vessel", "void"),
    "polar_radial_systems": ("center", "circumference", "orbit"),
    "grid_systems": ("grid", "lattice"),
    "graph_network_systems": ("network", "constellation"),
    "swarm_systems": ("swarm", "network"),
    "wave_systems": ("wave", "pulse", "breath"),
    "harmonic_oscillators": ("pulse", "wave", "breath"),
    "modular_tiling": ("grid", "lattice"),
    "sacred_geometry_pattern_systems": ("mandala", "center", "axis"),
}

_MODULE_MOTIFS: dict[GenerativeModuleKind, tuple[SemanticMotifId, ...]] = {
    "seed_system": ("seed",),
    "recursive_module": ("spiral", "tree"),
    "particle_emitter": ("fragmentation", "swarm"),
    "force_field": ("river", "wave"),
    "attractor_field": ("center", "orbit"),
    "noise_modulation_layer": ("river", "wave"),
    "symmetry_transform": ("mirror", "axis", "mandala"),
    "tiling_layer": ("grid", "lattice"),
    "graph_network_layer": ("network", "constellation"),
    "cellular_grid_layer": ("grid", "lattice"),
    "wave_oscillator": ("wave", "pulse", "breath"),
    "geometry_reassembly_layer": ("reintegration", "vessel"),
    "color_modulation_layer": ("flame", "pearl"),
    "audio_reactive_modulation_layer": ("pulse", "breath", "wave"),
    "camera_motion_path_hook": ("orbit", "axis"),
}

_PARAMETER_MOTIFS: dict[str, tuple[SemanticMotifId, ...]] = {
    "random_seed": ("seed",),
    "global_scale": ("center",),
    "time_phase": ("pulse", "wave"),
    "recursion_depth": ("spiral", "tree"),
    "spiral_tightness": ("spiral", "orbit"),
    "scale_decay": ("descent", "ascent"),
    "rotation_step": ("orbit", "axis"),
    "particle_count": ("fragmentation", "swarm"),
    "max_particle_count": ("swarm",),
    "reassembly_speed": ("reintegration",),
    "fragmentation_amount": ("fragmentation",),
    "force_strength": ("river", "wave"),
    "attractor_strength": ("center", "orbit"),
    "noise_strength": ("river", "wave"),
    "radial_symmetry": ("mandala", "axis"),
    "ring_count": ("circumference", "mandala"),
    "orbit_speed": ("orbit", "pulse"),
    "grid_resolution": ("grid", "lattice"),
    "node_count": ("network", "constellation"),
    "connection_radius": ("network",),
    "amplitude": ("wave", "breath"),
    "frequency": ("pulse", "wave"),
    "palette_shift": ("flame", "pearl"),
    "audio_gain": ("pulse", "breath"),
    "interaction_strength": ("gate", "threshold"),
}

__all__ = [
    "SEMANTIC_MOTIF_AUTHORITY_BOUNDARY",
    "SemanticMotif",
    "SemanticMotifCompositionMapping",
    "SemanticMotifFallbackPlan",
    "SemanticMotifHierarchyLevel",
    "SemanticMotifId",
    "SemanticMotifNarrativeMapping",
    "SemanticMotifParameterMapping",
    "SemanticMotifRole",
    "SemanticMotifStructureMapping",
    "SemanticMotifSystem",
    "derive_semantic_motif_system",
    "semantic_motif_prompt_lines",
]
