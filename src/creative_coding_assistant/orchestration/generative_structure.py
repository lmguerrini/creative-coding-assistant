"""Bounded Generative Structure Engine for V3.2 workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

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
from creative_coding_assistant.orchestration.procedural_structure import (
    ProceduralFamily,
    ProceduralStructureChoice,
    ProceduralStructurePlan,
)
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityProfile,
)
from creative_coding_assistant.orchestration.symbolic_narrative import (
    SymbolicNarrativePlan,
)

GenerativeArchitecture = Literal[
    "recursive_modular_blueprint",
    "agent_field_blueprint",
    "grid_state_blueprint",
    "radial_pattern_blueprint",
    "network_relation_blueprint",
    "wave_modulation_blueprint",
    "minimal_parameter_blueprint",
]
GenerativeModuleKind = Literal[
    "seed_system",
    "recursive_module",
    "particle_emitter",
    "force_field",
    "attractor_field",
    "noise_modulation_layer",
    "symmetry_transform",
    "tiling_layer",
    "graph_network_layer",
    "cellular_grid_layer",
    "wave_oscillator",
    "geometry_reassembly_layer",
    "color_modulation_layer",
    "audio_reactive_modulation_layer",
    "camera_motion_path_hook",
]
GenerativeRelationshipType = Literal[
    "feeds",
    "modulates",
    "constrains",
    "emits",
    "attracts",
    "mirrors",
    "samples",
    "reassembles",
    "times",
    "fallback_for",
]
GenerativeParameterValueType = Literal[
    "integer",
    "float",
    "boolean",
    "vector",
    "color",
    "enum",
]
GenerativeParameterRole = Literal["control", "derived", "constraint"]
GenerativeEvolutionPhase = Literal[
    "seed",
    "growth",
    "fragmentation",
    "threshold",
    "reassembly",
    "stabilization",
    "loop",
]
GenerativeEvolutionTrigger = Literal[
    "time",
    "interaction",
    "audio",
    "parameter",
    "narrative_phase",
]
GenerativeHookType = Literal["interaction", "audiovisual"]

GENERATIVE_STRUCTURE_AUTHORITY_BOUNDARY = (
    "The Generative Structure Engine turns procedural metadata into an "
    "inspectable generative blueprint only; it does not generate executable "
    "code, execute artifacts, auto-select runtimes, change preview behavior, "
    "route providers or models, run autonomous loops, implement V4 multi-agent "
    "runtime, or implement HoloMind."
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9_.+#-]+")
_REASSEMBLY_TOKENS = frozenset(
    {
        "rebirth",
        "reassembly",
        "reassemble",
        "recompose",
        "recomposition",
        "reform",
        "reforms",
        "phoenix",
    }
)


class GenerativeModule(BaseModel):
    """One operational module in the generative blueprint."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    module_id: str = Field(min_length=1, max_length=80)
    kind: GenerativeModuleKind
    label: str = Field(min_length=1, max_length=140)
    source_family: ProceduralFamily | None = None
    purpose: str = Field(min_length=1, max_length=360)
    inputs: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    parameters: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    evolution_role: str = Field(min_length=1, max_length=280)
    implementation_notes: tuple[str, ...] = Field(min_length=1, max_length=6)
    safeguards: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=8)


class GenerativeModuleRelationship(BaseModel):
    """Directed relationship between two blueprint modules."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source_module_id: str = Field(min_length=1, max_length=80)
    target_module_id: str = Field(min_length=1, max_length=80)
    relationship_type: GenerativeRelationshipType
    description: str = Field(min_length=1, max_length=340)
    parameters: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=6)


class GenerativeParameter(BaseModel):
    """Inspectable parameter schema entry for later generation layers."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=120)
    value_type: GenerativeParameterValueType
    role: GenerativeParameterRole
    default_value: str = Field(min_length=1, max_length=120)
    bounds: str | None = Field(default=None, max_length=160)
    controlled_by: str | None = Field(default=None, max_length=120)
    target_modules: tuple[str, ...] = Field(min_length=1, max_length=8)
    rationale: str = Field(min_length=1, max_length=260)


class GenerativeEvolutionRule(BaseModel):
    """One rule for how modules evolve over time or interaction."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    phase: GenerativeEvolutionPhase
    trigger: GenerativeEvolutionTrigger
    rule: str = Field(min_length=1, max_length=360)
    affected_modules: tuple[str, ...] = Field(min_length=1, max_length=8)
    parameter_changes: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    safeguards: tuple[str, ...] = Field(default_factory=tuple, max_length=6)


class GenerativeStructureHook(BaseModel):
    """Optional interaction or audiovisual modulation hook."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    hook_id: str = Field(min_length=1, max_length=80)
    hook_type: GenerativeHookType
    signal: str = Field(min_length=1, max_length=160)
    target_modules: tuple[str, ...] = Field(min_length=1, max_length=8)
    parameter_mapping: tuple[str, ...] = Field(min_length=1, max_length=8)
    fallback_behavior: str = Field(min_length=1, max_length=240)


class GenerativeFallbackBlueprint(BaseModel):
    """Lower-risk blueprint fallback when complexity or ambiguity tightens."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=160)
    architecture: GenerativeArchitecture
    module_kinds: tuple[GenerativeModuleKind, ...] = Field(min_length=1, max_length=6)
    parameter_reductions: tuple[str, ...] = Field(min_length=1, max_length=8)
    reason: str = Field(min_length=1, max_length=320)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=6)


class GenerativeStructureBlueprint(BaseModel):
    """Operational generative blueprint derived before code generation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["generative_structure_engine"] = "generative_structure_engine"
    blueprint_name: str = Field(min_length=1, max_length=180)
    generative_architecture: GenerativeArchitecture
    procedural_modules: tuple[GenerativeModule, ...] = Field(
        min_length=2,
        max_length=12,
    )
    module_relationships: tuple[GenerativeModuleRelationship, ...] = Field(
        min_length=1,
        max_length=16,
    )
    parameter_schema: tuple[GenerativeParameter, ...] = Field(
        min_length=3,
        max_length=28,
    )
    control_parameters: tuple[str, ...] = Field(min_length=1, max_length=16)
    evolution_rules: tuple[GenerativeEvolutionRule, ...] = Field(
        min_length=3,
        max_length=10,
    )
    spatial_evolution: str = Field(min_length=1, max_length=420)
    temporal_evolution: str = Field(min_length=1, max_length=420)
    interaction_hooks: tuple[GenerativeStructureHook, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    audiovisual_hooks: tuple[GenerativeStructureHook, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    runtime_implementation_guidance: tuple[str, ...] = Field(
        min_length=1,
        max_length=8,
    )
    performance_safeguards: tuple[str, ...] = Field(min_length=1, max_length=8)
    fallback_blueprint: GenerativeFallbackBlueprint
    unresolved_implementation_gaps: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=8,
    )
    hitl_questions: tuple[str, ...] = Field(default_factory=tuple, max_length=6)
    prompt_guidance: tuple[str, ...] = Field(min_length=1, max_length=8)
    authority_boundary: str = Field(
        default=GENERATIVE_STRUCTURE_AUTHORITY_BOUNDARY,
        max_length=620,
    )
    evidence: tuple[str, ...] = Field(default_factory=tuple, max_length=12)


@dataclass(frozen=True)
class _GenerativeContext:
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
    runtime_capabilities: RuntimeCapabilityProfile | None
    creative_tradeoffs: CreativeTradeoffProfile | None
    creative_quality_prediction: CreativeQualityPrediction | None
    symbolic_narrative: SymbolicNarrativePlan | None
    creative_composition: CreativeCompositionPlan | None
    procedural_structure: ProceduralStructurePlan | None
    text: str
    tokens: frozenset[str]


@dataclass(frozen=True)
class _ParameterSpec:
    name: str
    label: str
    value_type: GenerativeParameterValueType
    role: GenerativeParameterRole
    default_value: str
    bounds: str | None
    rationale: str


def derive_generative_structure_blueprint(
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
    runtime_capabilities: RuntimeCapabilityProfile | None = None,
    creative_tradeoffs: CreativeTradeoffProfile | None = None,
    creative_quality_prediction: CreativeQualityPrediction | None = None,
    symbolic_narrative: SymbolicNarrativePlan | None = None,
    creative_composition: CreativeCompositionPlan | None = None,
    procedural_structure: ProceduralStructurePlan | None = None,
) -> GenerativeStructureBlueprint:
    """Derive an operational blueprint without generating executable code."""

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
        runtime_capabilities=runtime_capabilities,
        creative_tradeoffs=creative_tradeoffs,
        creative_quality_prediction=creative_quality_prediction,
        symbolic_narrative=symbolic_narrative,
        creative_composition=creative_composition,
        procedural_structure=procedural_structure,
    )
    primary = _primary_choice(context)
    architecture = _architecture_for_family(primary.family)
    modules = _modules(context, primary=primary)
    relationships = _relationships(context, modules)
    parameters = _parameters_for_modules(modules, context)
    unresolved = _unresolved_gaps(context, modules, parameters)
    return GenerativeStructureBlueprint(
        blueprint_name=_blueprint_name(context, primary),
        generative_architecture=architecture,
        procedural_modules=modules,
        module_relationships=relationships,
        parameter_schema=parameters,
        control_parameters=tuple(
            parameter.name for parameter in parameters if parameter.role == "control"
        )[:16],
        evolution_rules=_evolution_rules(context, modules),
        spatial_evolution=_spatial_evolution(context, primary),
        temporal_evolution=_temporal_evolution(context, primary),
        interaction_hooks=_interaction_hooks(context, modules),
        audiovisual_hooks=_audiovisual_hooks(context, modules),
        runtime_implementation_guidance=_runtime_guidance(context, modules),
        performance_safeguards=_performance_safeguards(context, modules),
        fallback_blueprint=_fallback_blueprint(context),
        unresolved_implementation_gaps=unresolved,
        hitl_questions=_hitl_questions(context, unresolved),
        prompt_guidance=_prompt_guidance(context, primary),
        evidence=_evidence(context, primary, modules),
    )


def generative_structure_prompt_lines(
    blueprint: GenerativeStructureBlueprint,
) -> tuple[str, ...]:
    """Render generative blueprint metadata as compact prompt guidance."""

    lines = [
        f"Authority boundary: {blueprint.authority_boundary}",
        f"Blueprint name: {blueprint.blueprint_name}",
        f"Generative architecture: {blueprint.generative_architecture}.",
        f"Spatial evolution: {blueprint.spatial_evolution}",
        f"Temporal evolution: {blueprint.temporal_evolution}",
    ]
    for module in blueprint.procedural_modules[:6]:
        lines.append(
            "Generative module: "
            f"{module.module_id}; {module.kind}; {module.purpose}"
        )
        if module.parameters:
            lines.append(
                "Module parameters: "
                f"{module.module_id}: {', '.join(module.parameters[:6])}."
            )
    for relationship in blueprint.module_relationships[:5]:
        lines.append(
            "Module relationship: "
            f"{relationship.source_module_id} "
            f"{relationship.relationship_type} "
            f"{relationship.target_module_id}; {relationship.description}"
        )
    lines.extend(
        "Generative parameter: "
        f"{item.name}; {item.value_type}; {item.role}; {item.bounds or 'unbounded'}"
        for item in blueprint.parameter_schema[:8]
    )
    lines.extend(
        f"Evolution rule ({item.phase}): {item.rule}"
        for item in blueprint.evolution_rules[:6]
    )
    lines.extend(
        f"Interaction hook: {item.signal}; {', '.join(item.parameter_mapping)}"
        for item in blueprint.interaction_hooks
    )
    lines.extend(
        f"Audiovisual hook: {item.signal}; {', '.join(item.parameter_mapping)}"
        for item in blueprint.audiovisual_hooks
    )
    lines.extend(
        f"Runtime implementation guidance: {item}"
        for item in blueprint.runtime_implementation_guidance
    )
    lines.extend(
        f"Performance safeguard: {item}"
        for item in blueprint.performance_safeguards
    )
    lines.append(
        "Fallback blueprint: "
        f"{blueprint.fallback_blueprint.name}; "
        f"{blueprint.fallback_blueprint.architecture}."
    )
    lines.extend(
        f"Unresolved generative gap: {item}"
        for item in blueprint.unresolved_implementation_gaps
    )
    lines.extend(
        f"HITL generative question: {item}" for item in blueprint.hitl_questions
    )
    lines.extend(
        f"Generative blueprint guidance: {item}" for item in blueprint.prompt_guidance
    )
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
    runtime_capabilities: RuntimeCapabilityProfile | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
    creative_quality_prediction: CreativeQualityPrediction | None,
    symbolic_narrative: SymbolicNarrativePlan | None,
    creative_composition: CreativeCompositionPlan | None,
    procedural_structure: ProceduralStructurePlan | None,
) -> _GenerativeContext:
    parts = [
        request.query,
        creative_translation.creative_intent if creative_translation else "",
        creative_intent.primary_expression if creative_intent else "",
        creative_strategy.primary_strategy if creative_strategy else "",
        creative_techniques.primary_technique if creative_techniques else "",
        symbolic_narrative.narrative_archetype if symbolic_narrative else "",
        symbolic_narrative.symbolic_arc if symbolic_narrative else "",
        creative_composition.composition_pattern if creative_composition else "",
    ]
    if creative_translation is not None:
        parts.extend(creative_translation.structure_direction)
        parts.extend(creative_translation.movement_language)
        parts.extend(creative_translation.color_material_direction)
    if procedural_structure is not None:
        parts.extend(procedural_structure.recommended_families)
        parts.append(procedural_structure.combination_strategy)
        parts.append(procedural_structure.spatial_structure_plan)
        parts.append(procedural_structure.temporal_structure_plan)
        if procedural_structure.interaction_structure_plan is not None:
            parts.append(procedural_structure.interaction_structure_plan)
        if procedural_structure.audiovisual_structure_plan is not None:
            parts.append(procedural_structure.audiovisual_structure_plan)
    if runtime_capabilities is not None:
        parts.extend(runtime_capabilities.likely_candidates)
    text = _normalize(" ".join(parts))
    return _GenerativeContext(
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
        runtime_capabilities=runtime_capabilities,
        creative_tradeoffs=creative_tradeoffs,
        creative_quality_prediction=creative_quality_prediction,
        symbolic_narrative=symbolic_narrative,
        creative_composition=creative_composition,
        procedural_structure=procedural_structure,
        text=text,
        tokens=frozenset(_TOKEN_PATTERN.findall(text)),
    )


def _primary_choice(context: _GenerativeContext) -> ProceduralStructureChoice:
    if context.procedural_structure is not None:
        return context.procedural_structure.primary_structure
    return ProceduralStructureChoice(
        family="grid_systems",
        label="Grid Systems",
        rationale="Fallback when no procedural structure plan is attached.",
        evidence=("Generative Structure Engine fallback.",),
    )


def _blueprint_name(
    context: _GenerativeContext,
    primary: ProceduralStructureChoice,
) -> str:
    subject = _subject_label(context)
    prefix = primary.label.replace("/", " ").replace("-", " ")
    return f"{prefix} Blueprint for {_clip(subject, 80)}"


def _modules(
    context: _GenerativeContext,
    *,
    primary: ProceduralStructureChoice,
) -> tuple[GenerativeModule, ...]:
    modules: list[GenerativeModule] = [_seed_module(context, primary)]
    choices = [primary]
    if context.procedural_structure is not None:
        choices.extend(context.procedural_structure.secondary_structures[:3])
    for index, choice in enumerate(choices):
        for kind in _module_kinds_for_family(choice.family):
            module = _module_from_choice(choice, kind, index)
            if module.module_id not in {item.module_id for item in modules}:
                modules.append(module)
    if _needs_reassembly(context) and not _has_kind(
        modules,
        "geometry_reassembly_layer",
    ):
        modules.append(_support_module("geometry_reassembly_layer", primary))
    if _has_color_cues(context):
        modules.append(_support_module("color_modulation_layer", primary))
    if context.procedural_structure is not None:
        if context.procedural_structure.audiovisual_structure_plan is not None:
            modules.append(_support_module("audio_reactive_modulation_layer", primary))
        if context.procedural_structure.interaction_structure_plan is not None:
            modules.append(_support_module("camera_motion_path_hook", primary))
    return tuple(modules[:12])


def _seed_module(
    context: _GenerativeContext,
    primary: ProceduralStructureChoice,
) -> GenerativeModule:
    return GenerativeModule(
        module_id="seed_system",
        kind="seed_system",
        label="Seed System",
        source_family=primary.family,
        purpose=(
            "Define deterministic origin values, coordinate frame, and shared "
            f"state for {_subject_label(context)}."
        ),
        inputs=("user intent", "procedural structure plan"),
        outputs=("seeded coordinate state", "global timing state"),
        parameters=("random_seed", "global_scale", "time_phase"),
        evolution_role="Initializes the blueprint before procedural modules evolve.",
        implementation_notes=(
            "Keep seeding deterministic and inspectable.",
            "Expose seed and scale as top-level controls only.",
        ),
        safeguards=("Avoid hidden random loops or autonomous repair behavior.",),
        evidence=("Required root module for operational generative blueprints.",),
    )


def _module_from_choice(
    choice: ProceduralStructureChoice,
    kind: GenerativeModuleKind,
    index: int,
) -> GenerativeModule:
    module_id = f"{_slug(kind)}_{index}"
    return GenerativeModule(
        module_id=module_id,
        kind=kind,
        label=_MODULE_LABELS[kind],
        source_family=choice.family,
        purpose=_purpose_for_kind(kind, choice),
        inputs=("seed_system", *_inputs_for_kind(kind)),
        outputs=_outputs_for_kind(kind),
        parameters=_parameter_names_for_kind(kind),
        evolution_role=_evolution_role_for_kind(kind),
        implementation_notes=_implementation_notes_for_kind(kind),
        safeguards=_safeguards_for_kind(kind),
        evidence=(f"Source procedural family: {choice.family}.", choice.rationale),
    )


def _support_module(
    kind: GenerativeModuleKind,
    primary: ProceduralStructureChoice,
) -> GenerativeModule:
    return GenerativeModule(
        module_id=_slug(kind),
        kind=kind,
        label=_MODULE_LABELS[kind],
        source_family=primary.family,
        purpose=_purpose_for_kind(kind, primary),
        inputs=("seed_system", *_inputs_for_kind(kind)),
        outputs=_outputs_for_kind(kind),
        parameters=_parameter_names_for_kind(kind),
        evolution_role=_evolution_role_for_kind(kind),
        implementation_notes=_implementation_notes_for_kind(kind),
        safeguards=_safeguards_for_kind(kind),
        evidence=(f"Support module inferred from {primary.family}.",),
    )


def _relationships(
    context: _GenerativeContext,
    modules: tuple[GenerativeModule, ...],
) -> tuple[GenerativeModuleRelationship, ...]:
    relationships: list[GenerativeModuleRelationship] = []
    for module in modules:
        if module.module_id == "seed_system":
            continue
        relationships.append(
            GenerativeModuleRelationship(
                source_module_id="seed_system",
                target_module_id=module.module_id,
                relationship_type="feeds",
                description=(
                    "Seeded coordinates and time phase initialize the module."
                ),
                parameters=("random_seed", "global_scale", "time_phase"),
                evidence=("All modules depend on deterministic seed state.",),
            )
        )
    recursive = _first_module_id(modules, "recursive_module")
    particle = _first_module_id(modules, "particle_emitter")
    if recursive and particle:
        relationships.append(
            GenerativeModuleRelationship(
                source_module_id=recursive,
                target_module_id=particle,
                relationship_type="attracts",
                description=(
                    "Particles inherit the recursive or spiral attractor path "
                    "before dissolving or reassembling."
                ),
                parameters=("spiral_tightness", "reassembly_speed"),
                evidence=("Recursive and particle modules coexist.",),
            )
        )
    force = _first_module_id(modules, "force_field")
    noise = _first_module_id(modules, "noise_modulation_layer")
    if noise and force:
        relationships.append(
            GenerativeModuleRelationship(
                source_module_id=noise,
                target_module_id=force,
                relationship_type="modulates",
                description="Noise changes the field direction and force strength.",
                parameters=("noise_scale", "noise_strength", "force_strength"),
                evidence=("Noise and force modules coexist.",),
            )
        )
    reassembly = _first_module_id(modules, "geometry_reassembly_layer")
    if reassembly and particle:
        relationships.append(
            GenerativeModuleRelationship(
                source_module_id=particle,
                target_module_id=reassembly,
                relationship_type="reassembles",
                description=(
                    "Particle state becomes the source material for the "
                    "reassembly layer."
                ),
                parameters=("fragmentation_amount", "reassembly_speed"),
                evidence=("Reassembly requested or inferred.",),
            )
        )
    audio = _first_module_id(modules, "audio_reactive_modulation_layer")
    if audio:
        target = _first_target_module(modules, exclude={audio, "seed_system"})
        relationships.append(
            GenerativeModuleRelationship(
                source_module_id=audio,
                target_module_id=target,
                relationship_type="times",
                description="Audio envelope modulates phase, density, or amplitude.",
                parameters=("audio_gain", "audio_smoothing"),
                evidence=("Audiovisual procedural plan is present.",),
            )
        )
    if context.procedural_structure is not None:
        fallback_family = context.procedural_structure.fallback_structure_options[0]
        relationships.append(
            GenerativeModuleRelationship(
                source_module_id=_first_target_module(modules),
                target_module_id="fallback_blueprint",
                relationship_type="fallback_for",
                description=(
                    f"Fallback uses {fallback_family.label} when complexity, "
                    "performance, or ambiguity tightens."
                ),
                parameters=("frame_budget_ms", "max_particle_count"),
                evidence=("Procedural Structure Planner supplied fallback options.",),
            )
        )
    return tuple(relationships[:16])


def _parameters_for_modules(
    modules: tuple[GenerativeModule, ...],
    context: _GenerativeContext,
) -> tuple[GenerativeParameter, ...]:
    targets: dict[str, list[str]] = {}
    for module in modules:
        for name in module.parameters:
            targets.setdefault(name, []).append(module.module_id)
    targets.setdefault("frame_budget_ms", ["seed_system"])
    if _has_kind(modules, "particle_emitter"):
        targets.setdefault(
            "max_particle_count",
            [_first_module_id(modules, "particle_emitter") or "seed_system"],
        )
    parameters: list[GenerativeParameter] = []
    for name, module_ids in targets.items():
        spec = _PARAMETERS.get(name, _default_parameter(name))
        parameters.append(
            GenerativeParameter(
                name=spec.name,
                label=spec.label,
                value_type=spec.value_type,
                role=spec.role,
                default_value=spec.default_value,
                bounds=spec.bounds,
                controlled_by=_controlled_by(name, context),
                target_modules=tuple(_dedupe(module_ids))[:8],
                rationale=spec.rationale,
            )
        )
    parameters.sort(key=lambda item: (item.role != "control", item.name))
    return tuple(parameters[:28])


def _evolution_rules(
    context: _GenerativeContext,
    modules: tuple[GenerativeModule, ...],
) -> tuple[GenerativeEvolutionRule, ...]:
    target_modules = tuple(module.module_id for module in modules if module.module_id)
    rules = [
        GenerativeEvolutionRule(
            phase="seed",
            trigger="time",
            rule="Initialize seed, scale, coordinate frame, and phase clock.",
            affected_modules=("seed_system",),
            parameter_changes=("random_seed fixed", "time_phase starts at 0"),
            safeguards=("No hidden randomness after initialization.",),
        ),
        GenerativeEvolutionRule(
            phase="growth",
            trigger="narrative_phase",
            rule=_growth_rule(context),
            affected_modules=target_modules[:6],
            parameter_changes=_growth_parameter_changes(modules),
            safeguards=("Clamp depth, count, and force before adding detail.",),
        ),
    ]
    if _needs_reassembly(context):
        rules.append(
            GenerativeEvolutionRule(
                phase="fragmentation",
                trigger="narrative_phase",
                rule="Break the primary form into particles, cells, or fragments.",
                affected_modules=target_modules[:6],
                parameter_changes=(
                    "fragmentation_amount increases",
                    "reassembly_speed stays low",
                ),
                safeguards=("Keep fragment counts under declared caps.",),
            )
        )
        rules.append(
            GenerativeEvolutionRule(
                phase="reassembly",
                trigger="narrative_phase",
                rule="Reintegrate fragments around the primary attractor path.",
                affected_modules=target_modules[:6],
                parameter_changes=(
                    "reassembly_speed increases",
                    "noise_strength decreases",
                ),
                safeguards=("Preserve the main silhouette or focal path.",),
            )
        )
    if _has_interaction(context):
        rules.append(
            GenerativeEvolutionRule(
                phase="threshold",
                trigger="interaction",
                rule="Map one user gesture to a bounded structural parameter.",
                affected_modules=target_modules[:5],
                parameter_changes=("interaction_strength changes within bounds",),
                safeguards=("Do not introduce new interaction modes automatically.",),
            )
        )
    if _has_audio(context):
        rules.append(
            GenerativeEvolutionRule(
                phase="loop",
                trigger="audio",
                rule="Use audio envelope to modulate phase, density, or amplitude.",
                affected_modules=target_modules[:5],
                parameter_changes=("audio_gain applies after smoothing",),
                safeguards=("Audio modulation remains optional and bounded.",),
            )
        )
    rules.append(
        GenerativeEvolutionRule(
            phase="stabilization",
            trigger="time",
            rule="Resolve toward a stable readable final hierarchy.",
            affected_modules=target_modules[:6],
            parameter_changes=("damping increases", "global_scale stabilizes"),
            safeguards=("Avoid autonomous loops or runtime repair behavior.",),
        )
    )
    return tuple(rules[:10])


def _interaction_hooks(
    context: _GenerativeContext,
    modules: tuple[GenerativeModule, ...],
) -> tuple[GenerativeStructureHook, ...]:
    if not _has_interaction(context):
        return ()
    target = _first_target_module(modules)
    return (
        GenerativeStructureHook(
            hook_id="interaction_control_hook",
            hook_type="interaction",
            signal=_interaction_signal(context),
            target_modules=(target,),
            parameter_mapping=(
                "pointer or gesture -> interaction_strength",
                "interaction_strength -> density, radius, or force",
            ),
            fallback_behavior="Use time-based evolution if interaction is unresolved.",
        ),
    )


def _audiovisual_hooks(
    context: _GenerativeContext,
    modules: tuple[GenerativeModule, ...],
) -> tuple[GenerativeStructureHook, ...]:
    if not _has_audio(context):
        return ()
    target = _first_target_module(modules)
    return (
        GenerativeStructureHook(
            hook_id="audiovisual_modulation_hook",
            hook_type="audiovisual",
            signal=_audio_signal(context),
            target_modules=(target,),
            parameter_mapping=(
                "audio envelope -> audio_gain",
                "audio_gain -> phase, amplitude, density, or color shift",
            ),
            fallback_behavior="Use a slow oscillator if no live audio signal exists.",
        ),
    )


def _fallback_blueprint(context: _GenerativeContext) -> GenerativeFallbackBlueprint:
    choice = None
    if context.procedural_structure is not None:
        choice = context.procedural_structure.fallback_structure_options[0]
    family = choice.family if choice is not None else "grid_systems"
    return GenerativeFallbackBlueprint(
        name=f"Bounded {_label_for_family(family)} Fallback",
        architecture=_architecture_for_family(family),
        module_kinds=("seed_system", *_module_kinds_for_family(family)[:2]),
        parameter_reductions=_fallback_reductions(family),
        reason=(
            "Use when performance pressure, implementation ambiguity, or "
            "runtime constraints make the primary blueprint too expensive."
        ),
        prompt_guidance=(
            "Keep the same creative intent while reducing module count.",
            "Prefer lower iteration depth, fewer agents, and simpler modulation.",
        ),
    )


def _runtime_guidance(
    context: _GenerativeContext,
    modules: tuple[GenerativeModule, ...],
) -> tuple[str, ...]:
    guidance: list[str] = []
    if context.runtime_capabilities is not None:
        guidance.append(
            "Treat inspected runtime candidates as feasibility guidance only: "
            f"{', '.join(context.runtime_capabilities.likely_candidates)}."
        )
        top = context.runtime_capabilities.candidate_runtimes[0]
        guidance.append(
            f"{top.label} suggests {top.implementation_complexity} "
            f"implementation complexity and {top.performance_pressure} "
            "performance pressure."
        )
    elif context.creative_plan is not None and context.creative_plan.recommended_runtime:
        guidance.append(
            "Existing creative plan names "
            f"{context.creative_plan.recommended_runtime}; do not treat this "
            "blueprint as runtime auto-selection."
        )
    else:
        guidance.append("Avoid runtime-specific implementation claims.")
    if _has_kind(modules, "particle_emitter"):
        guidance.append("Use bounded particle arrays and explicit lifecycle state.")
    if _has_kind(modules, "recursive_module"):
        guidance.append("Use explicit recursion depth caps or iterative expansion.")
    if _has_kind(modules, "cellular_grid_layer"):
        guidance.append("Keep grid resolution tied to frame budget.")
    return _dedupe(guidance)[:8]


def _performance_safeguards(
    context: _GenerativeContext,
    modules: tuple[GenerativeModule, ...],
) -> tuple[str, ...]:
    safeguards: list[str] = [
        "Expose frame_budget_ms as a constraint parameter.",
        "Prefer bounded loops and capped collections over autonomous evolution.",
    ]
    for module in modules:
        safeguards.extend(module.safeguards[:2])
    if context.procedural_structure is not None:
        safeguards.extend(context.procedural_structure.performance_risks[:2])
    if context.creative_tradeoffs is not None:
        safeguards.extend(context.creative_tradeoffs.performance_concerns[:2])
    return _dedupe(safeguards)[:8]


def _unresolved_gaps(
    context: _GenerativeContext,
    modules: tuple[GenerativeModule, ...],
    parameters: tuple[GenerativeParameter, ...],
) -> tuple[str, ...]:
    gaps: list[str] = []
    if context.procedural_structure is None:
        gaps.append("No Procedural Structure Planner metadata is attached.")
    elif context.procedural_structure.unresolved_procedural_gaps:
        gaps.extend(context.procedural_structure.unresolved_procedural_gaps[:3])
    if _has_interaction(context) and not _has_explicit_interaction(context):
        gaps.append("Interaction hook exists but the controlling gesture is unclear.")
    if _has_audio(context) and not _has_explicit_audio_timing(context):
        gaps.append("Audiovisual hook exists but beat, pulse, or tempo is unclear.")
    if not parameters:
        gaps.append("No parameter schema could be derived from modules.")
    if context.creative_quality_prediction is not None:
        gaps.extend(context.creative_quality_prediction.missing_information[:2])
    return _dedupe(gaps)[:8]


def _hitl_questions(
    context: _GenerativeContext,
    unresolved: tuple[str, ...],
) -> tuple[str, ...]:
    questions: list[str] = []
    if _has_interaction(context) and not _has_explicit_interaction(context):
        questions.append("Which gesture should control the generative structure?")
    if _has_audio(context):
        questions.append("Which beat, pulse, or audio feature should modulate it?")
    for gap in unresolved:
        lowered = gap.lower()
        if "gesture" in lowered or "interaction" in lowered:
            questions.append("Which gesture should control the generative structure?")
        elif "audio" in lowered or "beat" in lowered or "tempo" in lowered:
            questions.append("Which beat, pulse, or audio feature should modulate it?")
        elif "parameter" in lowered:
            questions.append("Which parameters should remain user-controllable?")
        elif "procedural" in lowered:
            questions.append("Which procedural module should be primary?")
    if context.procedural_structure is not None:
        questions.extend(context.procedural_structure.hitl_questions[:3])
    return _dedupe(questions)[:6]


def _prompt_guidance(
    context: _GenerativeContext,
    primary: ProceduralStructureChoice,
) -> tuple[str, ...]:
    guidance = [
        "Use the generative blueprint as structure guidance, not executable code.",
        f"Build the blueprint around {primary.label} before secondary modules.",
        "Name module state, parameters, and evolution rules explicitly.",
        "Keep runtime guidance non-binding and preserve existing preview behavior.",
    ]
    if context.procedural_structure is not None:
        guidance.extend(context.procedural_structure.prompt_guidance[:2])
    return tuple(_dedupe(guidance)[:8])


def _spatial_evolution(
    context: _GenerativeContext,
    primary: ProceduralStructureChoice,
) -> str:
    if context.procedural_structure is not None:
        base = context.procedural_structure.spatial_structure_plan
    else:
        base = f"Organize modules around {primary.label}."
    if context.creative_composition is not None:
        return _clip(
            f"{base} Anchor spatial evolution to "
            f"{context.creative_composition.composition_pattern} and preserve "
            f"{context.creative_composition.primary_focal_point}.",
            420,
        )
    return _clip(base, 420)


def _temporal_evolution(
    context: _GenerativeContext,
    primary: ProceduralStructureChoice,
) -> str:
    del primary
    if context.procedural_structure is not None:
        base = context.procedural_structure.temporal_structure_plan
    else:
        base = "Advance modules through seed, growth, stabilization, and loop."
    if context.symbolic_narrative is not None:
        return _clip(
            f"{base} Align temporal phases with "
            f"{context.symbolic_narrative.narrative_archetype}.",
            420,
        )
    return _clip(base, 420)


def _evidence(
    context: _GenerativeContext,
    primary: ProceduralStructureChoice,
    modules: tuple[GenerativeModule, ...],
) -> tuple[str, ...]:
    evidence = [
        f"Primary procedural family: {primary.family}.",
        "Module kinds: " + ", ".join(module.kind for module in modules[:8]) + ".",
    ]
    if context.procedural_structure is not None:
        evidence.append(
            "Procedural families: "
            + ", ".join(context.procedural_structure.recommended_families)
            + "."
        )
    if context.runtime_capabilities is not None:
        evidence.append(
            "Runtime candidates: "
            + ", ".join(context.runtime_capabilities.likely_candidates)
            + "."
        )
    if context.symbolic_narrative is not None:
        evidence.append(
            f"Narrative source: {context.symbolic_narrative.narrative_archetype}."
        )
    if context.creative_composition is not None:
        evidence.append(
            f"Composition source: {context.creative_composition.composition_pattern}."
        )
    return tuple(evidence[:12])


def _module_kinds_for_family(
    family: ProceduralFamily,
) -> tuple[GenerativeModuleKind, ...]:
    return _FAMILY_MODULES.get(family, ("tiling_layer",))


def _architecture_for_family(family: ProceduralFamily) -> GenerativeArchitecture:
    if family in {"fractals", "recursive_geometry", "l_systems"}:
        return "recursive_modular_blueprint"
    if family in {
        "particle_systems",
        "boids",
        "flow_fields",
        "noise_fields",
        "swarm_systems",
    }:
        return "agent_field_blueprint"
    if family in {"cellular_automata", "reaction_diffusion", "grid_systems"}:
        return "grid_state_blueprint"
    if family in {
        "polar_radial_systems",
        "sacred_geometry_pattern_systems",
        "modular_tiling",
    }:
        return "radial_pattern_blueprint"
    if family in {"graph_network_systems", "voronoi_systems"}:
        return "network_relation_blueprint"
    if family in {"wave_systems", "harmonic_oscillators"}:
        return "wave_modulation_blueprint"
    return "minimal_parameter_blueprint"


def _label_for_family(family: ProceduralFamily) -> str:
    return family.replace("_", " ").replace("systems", "system").title()


def _purpose_for_kind(
    kind: GenerativeModuleKind,
    choice: ProceduralStructureChoice,
) -> str:
    purposes = {
        "recursive_module": (
            f"Build {choice.label.lower()} through bounded depth, scale, and "
            "rotation rules."
        ),
        "particle_emitter": (
            "Emit bounded particles with lifecycle, attraction, dissolution, "
            "and optional reassembly state."
        ),
        "force_field": "Apply bounded vector forces to modules or particles.",
        "attractor_field": "Pull modules toward a readable focal path or origin.",
        "noise_modulation_layer": (
            "Modulate density, displacement, color, or force with coherent noise."
        ),
        "symmetry_transform": "Apply radial, mirrored, or orbital symmetry rules.",
        "tiling_layer": "Repeat modules with controlled variation and state changes.",
        "graph_network_layer": "Represent relations as nodes, links, and clusters.",
        "cellular_grid_layer": "Advance local cell states through bounded rules.",
        "wave_oscillator": "Drive phase, amplitude, and rhythm with oscillators.",
        "geometry_reassembly_layer": (
            "Recompose fragments or particles into the target symbolic structure."
        ),
        "color_modulation_layer": "Map state and phase to palette changes.",
        "audio_reactive_modulation_layer": (
            "Map audio features to bounded generative parameters."
        ),
        "camera_motion_path_hook": (
            "Expose a bounded viewpoint or motion path hook without changing runtime."
        ),
    }
    return purposes.get(kind, f"Support {choice.label.lower()} as a module.")


def _inputs_for_kind(kind: GenerativeModuleKind) -> tuple[str, ...]:
    if kind in {"particle_emitter", "force_field", "attractor_field"}:
        return ("time_phase", "primary structure state")
    if kind in {"audio_reactive_modulation_layer", "wave_oscillator"}:
        return ("time_phase", "optional audio envelope")
    if kind == "camera_motion_path_hook":
        return ("time_phase", "optional user gesture")
    return ("time_phase",)


def _outputs_for_kind(kind: GenerativeModuleKind) -> tuple[str, ...]:
    outputs = {
        "recursive_module": ("recursive geometry state",),
        "particle_emitter": ("particle state",),
        "force_field": ("force vectors",),
        "attractor_field": ("attractor path",),
        "noise_modulation_layer": ("noise modulation state",),
        "symmetry_transform": ("symmetry coordinates",),
        "tiling_layer": ("tile states",),
        "graph_network_layer": ("node and edge states",),
        "cellular_grid_layer": ("cell states",),
        "wave_oscillator": ("oscillator phase state",),
        "geometry_reassembly_layer": ("reassembled geometry state",),
        "color_modulation_layer": ("color phase state",),
        "audio_reactive_modulation_layer": ("audio modulation state",),
        "camera_motion_path_hook": ("view or motion path state",),
    }
    return outputs.get(kind, ("module state",))


def _parameter_names_for_kind(kind: GenerativeModuleKind) -> tuple[str, ...]:
    return _MODULE_PARAMETERS.get(kind, ())


def _evolution_role_for_kind(kind: GenerativeModuleKind) -> str:
    return _MODULE_EVOLUTION_ROLES.get(
        kind,
        "Participates in bounded module evolution.",
    )


def _implementation_notes_for_kind(kind: GenerativeModuleKind) -> tuple[str, ...]:
    return _MODULE_NOTES.get(kind, ("Keep module state explicit and bounded.",))


def _safeguards_for_kind(kind: GenerativeModuleKind) -> tuple[str, ...]:
    return _MODULE_SAFEGUARDS.get(kind, ("Clamp module parameters.",))


def _controlled_by(name: str, context: _GenerativeContext) -> str | None:
    if name.startswith("audio_") and _has_audio(context):
        return "audiovisual hook"
    if name == "interaction_strength" and _has_interaction(context):
        return "interaction hook"
    if name in {"time_phase", "orbit_speed", "frequency"}:
        return "temporal evolution"
    return None


def _default_parameter(name: str) -> _ParameterSpec:
    return _ParameterSpec(
        name=name,
        label=name.replace("_", " ").title(),
        value_type="float",
        role="control",
        default_value="1.0",
        bounds="0.0..1.0",
        rationale="Derived fallback parameter for an inferred module.",
    )


def _fallback_reductions(
    family: ProceduralFamily,
) -> tuple[str, ...]:
    reductions = [
        "Reduce active module count to seed plus one primary structure.",
        "Clamp frame_budget_ms before adding detail.",
    ]
    if family in {"particle_systems", "swarm_systems", "boids"}:
        reductions.append("Lower max_particle_count and disable trails.")
    if family in {"fractals", "recursive_geometry", "l_systems"}:
        reductions.append("Lower recursion_depth and freeze deep branches.")
    if family in {"cellular_automata", "reaction_diffusion", "grid_systems"}:
        reductions.append("Lower grid_resolution and update fewer cells.")
    return tuple(reductions[:8])


def _growth_rule(context: _GenerativeContext) -> str:
    if context.symbolic_narrative is not None:
        return (
            "Grow modules according to the symbolic phase order while keeping "
            "parameters visible."
        )
    return "Grow modules through bounded parameter changes over time."


def _growth_parameter_changes(
    modules: tuple[GenerativeModule, ...],
) -> tuple[str, ...]:
    changes: list[str] = []
    if _has_kind(modules, "recursive_module"):
        changes.append("recursion_depth increases within bounds")
    if _has_kind(modules, "particle_emitter"):
        changes.append("particle_count approaches max_particle_count")
    if _has_kind(modules, "symmetry_transform"):
        changes.append("ring_count and radial_symmetry shape structure")
    if _has_kind(modules, "wave_oscillator"):
        changes.append("amplitude and frequency shape motion")
    return tuple(changes[:6]) or ("global_scale changes within bounds",)


def _has_kind(
    modules: tuple[GenerativeModule, ...] | list[GenerativeModule],
    kind: GenerativeModuleKind,
) -> bool:
    return any(module.kind == kind for module in modules)


def _first_module_id(
    modules: tuple[GenerativeModule, ...],
    kind: GenerativeModuleKind,
) -> str | None:
    for module in modules:
        if module.kind == kind:
            return module.module_id
    return None


def _first_target_module(
    modules: tuple[GenerativeModule, ...],
    *,
    exclude: set[str] | None = None,
) -> str:
    excluded = exclude or {"seed_system"}
    for module in modules:
        if module.module_id not in excluded:
            return module.module_id
    return "seed_system"


def _needs_reassembly(context: _GenerativeContext) -> bool:
    return bool(context.tokens.intersection(_REASSEMBLY_TOKENS))


def _has_interaction(context: _GenerativeContext) -> bool:
    if context.procedural_structure is not None:
        return context.procedural_structure.interaction_structure_plan is not None
    return bool(context.tokens.intersection({"interactive", "gesture", "mouse"}))


def _has_audio(context: _GenerativeContext) -> bool:
    if context.procedural_structure is not None:
        return context.procedural_structure.audiovisual_structure_plan is not None
    return bool(context.tokens.intersection({"audio", "beat", "pulse", "tempo"}))


def _has_explicit_interaction(context: _GenerativeContext) -> bool:
    return bool(context.tokens.intersection({"click", "drag", "mouse", "touch"}))


def _has_explicit_audio_timing(context: _GenerativeContext) -> bool:
    return bool(context.tokens.intersection({"beat", "pulse", "rhythm", "tempo"}))


def _has_color_cues(context: _GenerativeContext) -> bool:
    if context.creative_translation is None:
        return False
    return bool(context.creative_translation.color_material_direction)


def _interaction_signal(context: _GenerativeContext) -> str:
    if _has_explicit_interaction(context):
        return "explicit pointer, click, drag, touch, or mouse gesture"
    return "unspecified interaction gesture"


def _audio_signal(context: _GenerativeContext) -> str:
    if _has_explicit_audio_timing(context):
        return "explicit beat, pulse, rhythm, or tempo signal"
    return "unspecified audio envelope"


def _subject_label(context: _GenerativeContext) -> str:
    if context.creative_intent is not None:
        return context.creative_intent.primary_expression
    if context.creative_translation is not None:
        return context.creative_translation.creative_intent
    return "the requested creative artifact"


def _clip(value: str, limit: int) -> str:
    normalized = " ".join(value.strip().split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "."


def _slug(value: str) -> str:
    return value.replace("/", "_").replace("-", "_")


def _normalize(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").replace("_", " ").split())


def _dedupe(values: list[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    for value in values:
        cleaned = " ".join(str(value).strip().split())
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return tuple(deduped)


_FAMILY_MODULES: dict[ProceduralFamily, tuple[GenerativeModuleKind, ...]] = {
    "fractals": ("recursive_module",),
    "recursive_geometry": ("recursive_module", "attractor_field"),
    "l_systems": ("recursive_module",),
    "particle_systems": ("particle_emitter", "attractor_field"),
    "boids": ("particle_emitter", "force_field"),
    "cellular_automata": ("cellular_grid_layer",),
    "reaction_diffusion": ("cellular_grid_layer", "noise_modulation_layer"),
    "voronoi_systems": ("graph_network_layer", "tiling_layer"),
    "noise_fields": ("noise_modulation_layer",),
    "flow_fields": ("force_field", "noise_modulation_layer"),
    "signed_distance_fields": ("geometry_reassembly_layer",),
    "polar_radial_systems": ("symmetry_transform", "attractor_field"),
    "grid_systems": ("cellular_grid_layer", "tiling_layer"),
    "graph_network_systems": ("graph_network_layer",),
    "swarm_systems": ("particle_emitter", "force_field"),
    "wave_systems": ("wave_oscillator",),
    "harmonic_oscillators": ("wave_oscillator",),
    "modular_tiling": ("tiling_layer",),
    "sacred_geometry_pattern_systems": ("symmetry_transform", "tiling_layer"),
}

_MODULE_LABELS: dict[GenerativeModuleKind, str] = {
    "seed_system": "Seed System",
    "recursive_module": "Recursive Module",
    "particle_emitter": "Particle Emitter",
    "force_field": "Force Field",
    "attractor_field": "Attractor Field",
    "noise_modulation_layer": "Noise Modulation Layer",
    "symmetry_transform": "Symmetry Transform",
    "tiling_layer": "Tiling Layer",
    "graph_network_layer": "Graph/Network Layer",
    "cellular_grid_layer": "Cellular Grid Layer",
    "wave_oscillator": "Wave Oscillator",
    "geometry_reassembly_layer": "Geometry Reassembly Layer",
    "color_modulation_layer": "Color Modulation Layer",
    "audio_reactive_modulation_layer": "Audio-Reactive Modulation Layer",
    "camera_motion_path_hook": "Camera/Motion Path Hook",
}

_MODULE_PARAMETERS: dict[GenerativeModuleKind, tuple[str, ...]] = {
    "seed_system": ("random_seed", "global_scale", "time_phase"),
    "recursive_module": (
        "recursion_depth",
        "spiral_tightness",
        "scale_decay",
        "rotation_step",
    ),
    "particle_emitter": (
        "particle_count",
        "max_particle_count",
        "lifespan",
        "emitter_radius",
        "reassembly_speed",
    ),
    "force_field": ("force_strength", "damping", "field_resolution"),
    "attractor_field": ("attractor_strength", "attractor_radius"),
    "noise_modulation_layer": (
        "noise_scale",
        "noise_strength",
        "turbulence_speed",
    ),
    "symmetry_transform": ("radial_symmetry", "ring_count", "orbit_speed"),
    "tiling_layer": ("tile_count", "variation_amount"),
    "graph_network_layer": ("node_count", "connection_radius"),
    "cellular_grid_layer": ("grid_resolution", "state_threshold"),
    "wave_oscillator": ("amplitude", "frequency", "damping"),
    "geometry_reassembly_layer": ("fragmentation_amount", "reassembly_speed"),
    "color_modulation_layer": ("palette_shift",),
    "audio_reactive_modulation_layer": ("audio_gain", "audio_smoothing"),
    "camera_motion_path_hook": ("camera_path_amount", "interaction_strength"),
}

_MODULE_EVOLUTION_ROLES: dict[GenerativeModuleKind, str] = {
    "recursive_module": "Controls recursive build-up and transformation depth.",
    "particle_emitter": "Controls dissolution, drift, and reassembly material.",
    "force_field": "Shapes motion through bounded vector forces.",
    "attractor_field": "Maintains the primary path, origin, or focal pull.",
    "noise_modulation_layer": "Adds controlled variation without changing topology.",
    "symmetry_transform": "Maintains radial, mirrored, or orbital order.",
    "tiling_layer": "Provides modular repetition and bounded variation.",
    "graph_network_layer": "Maintains relational structure between nodes.",
    "cellular_grid_layer": "Evolves local state through bounded neighbor rules.",
    "wave_oscillator": "Coordinates rhythm, phase, amplitude, and loop behavior.",
    "geometry_reassembly_layer": "Controls fragmentation and reintegration phases.",
    "color_modulation_layer": "Maps phase and state to palette changes.",
    "audio_reactive_modulation_layer": "Maps audio features to module parameters.",
    "camera_motion_path_hook": "Exposes bounded viewpoint or path modulation.",
}

_MODULE_NOTES: dict[GenerativeModuleKind, tuple[str, ...]] = {
    "recursive_module": ("Prefer iterative depth counters over unbounded recursion.",),
    "particle_emitter": ("Represent particle lifecycle explicitly.",),
    "force_field": ("Sample force values at bounded resolution.",),
    "attractor_field": ("Keep attractor count small and readable.",),
    "noise_modulation_layer": ("Limit octave count and name mapped parameters.",),
    "symmetry_transform": ("Keep symmetry count explicit and adjustable.",),
    "tiling_layer": ("Reuse module definitions rather than duplicating logic.",),
    "graph_network_layer": ("Avoid dense all-to-all links by default.",),
    "cellular_grid_layer": ("Separate state update from visual mapping.",),
    "wave_oscillator": ("Use named amplitude, frequency, phase, and damping.",),
    "geometry_reassembly_layer": ("Keep fragment identity or target mapping explicit.",),
    "color_modulation_layer": ("Map color changes to state or phase.",),
    "audio_reactive_modulation_layer": ("Smooth audio input before mapping.",),
    "camera_motion_path_hook": ("Keep camera/path behavior optional.",),
}

_MODULE_SAFEGUARDS: dict[GenerativeModuleKind, tuple[str, ...]] = {
    "recursive_module": ("Clamp recursion_depth.",),
    "particle_emitter": ("Clamp particle_count and max_particle_count.",),
    "force_field": ("Clamp field_resolution and force_strength.",),
    "attractor_field": ("Clamp attractor_strength.",),
    "noise_modulation_layer": ("Clamp noise_strength and octave count.",),
    "symmetry_transform": ("Clamp ring_count and radial_symmetry.",),
    "tiling_layer": ("Clamp tile_count.",),
    "graph_network_layer": ("Clamp node_count and connection radius.",),
    "cellular_grid_layer": ("Clamp grid_resolution.",),
    "wave_oscillator": ("Clamp amplitude and frequency.",),
    "geometry_reassembly_layer": ("Clamp fragmentation_amount.",),
    "color_modulation_layer": ("Clamp palette_shift.",),
    "audio_reactive_modulation_layer": ("Clamp audio_gain after smoothing.",),
    "camera_motion_path_hook": ("Clamp camera_path_amount.",),
}

_PARAMETERS: dict[str, _ParameterSpec] = {
    "random_seed": _ParameterSpec(
        "random_seed",
        "Random Seed",
        "integer",
        "control",
        "1",
        "0..99999",
        "Keeps stochastic variation deterministic.",
    ),
    "global_scale": _ParameterSpec(
        "global_scale",
        "Global Scale",
        "float",
        "control",
        "1.0",
        "0.1..4.0",
        "Controls overall blueprint scale.",
    ),
    "time_phase": _ParameterSpec(
        "time_phase",
        "Time Phase",
        "float",
        "derived",
        "0.0",
        "0.0..1.0 loop",
        "Coordinates temporal evolution.",
    ),
    "recursion_depth": _ParameterSpec(
        "recursion_depth",
        "Recursion Depth",
        "integer",
        "control",
        "5",
        "1..9",
        "Caps recursive or fractal expansion.",
    ),
    "spiral_tightness": _ParameterSpec(
        "spiral_tightness",
        "Spiral Tightness",
        "float",
        "control",
        "0.55",
        "0.0..1.0",
        "Controls curvature of recursive or radial paths.",
    ),
    "scale_decay": _ParameterSpec(
        "scale_decay",
        "Scale Decay",
        "float",
        "control",
        "0.72",
        "0.1..0.95",
        "Controls size reduction across levels.",
    ),
    "rotation_step": _ParameterSpec(
        "rotation_step",
        "Rotation Step",
        "float",
        "control",
        "0.28",
        "0.0..6.283",
        "Controls angular change between generated elements.",
    ),
    "particle_count": _ParameterSpec(
        "particle_count",
        "Particle Count",
        "integer",
        "control",
        "320",
        "0..max_particle_count",
        "Controls active particle density.",
    ),
    "max_particle_count": _ParameterSpec(
        "max_particle_count",
        "Max Particle Count",
        "integer",
        "constraint",
        "800",
        "50..2000",
        "Caps particle memory and update cost.",
    ),
    "lifespan": _ParameterSpec(
        "lifespan",
        "Particle Lifespan",
        "float",
        "control",
        "1.0",
        "0.1..5.0",
        "Controls particle lifecycle duration.",
    ),
    "emitter_radius": _ParameterSpec(
        "emitter_radius",
        "Emitter Radius",
        "float",
        "control",
        "0.4",
        "0.0..2.0",
        "Controls initial particle spread.",
    ),
    "reassembly_speed": _ParameterSpec(
        "reassembly_speed",
        "Reassembly Speed",
        "float",
        "control",
        "0.35",
        "0.0..1.0",
        "Controls how quickly fragments return to structure.",
    ),
    "force_strength": _ParameterSpec(
        "force_strength",
        "Force Strength",
        "float",
        "control",
        "0.5",
        "0.0..2.0",
        "Controls vector field force magnitude.",
    ),
    "damping": _ParameterSpec(
        "damping",
        "Damping",
        "float",
        "control",
        "0.88",
        "0.0..1.0",
        "Prevents runaway motion.",
    ),
    "field_resolution": _ParameterSpec(
        "field_resolution",
        "Field Resolution",
        "integer",
        "constraint",
        "48",
        "8..128",
        "Caps field sampling cost.",
    ),
    "attractor_strength": _ParameterSpec(
        "attractor_strength",
        "Attractor Strength",
        "float",
        "control",
        "0.6",
        "0.0..2.0",
        "Controls pull toward the primary path.",
    ),
    "attractor_radius": _ParameterSpec(
        "attractor_radius",
        "Attractor Radius",
        "float",
        "control",
        "0.5",
        "0.0..2.0",
        "Controls spatial reach of attraction.",
    ),
    "noise_scale": _ParameterSpec(
        "noise_scale",
        "Noise Scale",
        "float",
        "control",
        "0.02",
        "0.001..0.2",
        "Controls coherent noise sampling scale.",
    ),
    "noise_strength": _ParameterSpec(
        "noise_strength",
        "Noise Strength",
        "float",
        "control",
        "0.35",
        "0.0..1.0",
        "Controls amount of organic variation.",
    ),
    "turbulence_speed": _ParameterSpec(
        "turbulence_speed",
        "Turbulence Speed",
        "float",
        "control",
        "0.2",
        "0.0..2.0",
        "Controls temporal drift in noise fields.",
    ),
    "radial_symmetry": _ParameterSpec(
        "radial_symmetry",
        "Radial Symmetry",
        "integer",
        "control",
        "8",
        "1..24",
        "Controls repeated angular sectors.",
    ),
    "ring_count": _ParameterSpec(
        "ring_count",
        "Ring Count",
        "integer",
        "control",
        "5",
        "1..18",
        "Controls radial structural bands.",
    ),
    "orbit_speed": _ParameterSpec(
        "orbit_speed",
        "Orbit Speed",
        "float",
        "control",
        "0.2",
        "0.0..2.0",
        "Controls orbital temporal motion.",
    ),
    "tile_count": _ParameterSpec(
        "tile_count",
        "Tile Count",
        "integer",
        "control",
        "64",
        "1..512",
        "Controls modular repetition density.",
    ),
    "variation_amount": _ParameterSpec(
        "variation_amount",
        "Variation Amount",
        "float",
        "control",
        "0.25",
        "0.0..1.0",
        "Controls local module variation.",
    ),
    "node_count": _ParameterSpec(
        "node_count",
        "Node Count",
        "integer",
        "control",
        "80",
        "2..512",
        "Controls graph or network complexity.",
    ),
    "connection_radius": _ParameterSpec(
        "connection_radius",
        "Connection Radius",
        "float",
        "control",
        "0.25",
        "0.0..1.0",
        "Controls graph edge density.",
    ),
    "grid_resolution": _ParameterSpec(
        "grid_resolution",
        "Grid Resolution",
        "integer",
        "constraint",
        "64",
        "8..256",
        "Caps cellular grid cost.",
    ),
    "state_threshold": _ParameterSpec(
        "state_threshold",
        "State Threshold",
        "float",
        "control",
        "0.5",
        "0.0..1.0",
        "Controls cell or field activation.",
    ),
    "amplitude": _ParameterSpec(
        "amplitude",
        "Amplitude",
        "float",
        "control",
        "0.5",
        "0.0..2.0",
        "Controls oscillator strength.",
    ),
    "frequency": _ParameterSpec(
        "frequency",
        "Frequency",
        "float",
        "control",
        "1.0",
        "0.0..8.0",
        "Controls oscillation speed.",
    ),
    "fragmentation_amount": _ParameterSpec(
        "fragmentation_amount",
        "Fragmentation Amount",
        "float",
        "control",
        "0.4",
        "0.0..1.0",
        "Controls how much structure breaks apart.",
    ),
    "palette_shift": _ParameterSpec(
        "palette_shift",
        "Palette Shift",
        "float",
        "control",
        "0.2",
        "0.0..1.0",
        "Controls phase-based color movement.",
    ),
    "audio_gain": _ParameterSpec(
        "audio_gain",
        "Audio Gain",
        "float",
        "control",
        "0.4",
        "0.0..1.0",
        "Controls audio modulation strength.",
    ),
    "audio_smoothing": _ParameterSpec(
        "audio_smoothing",
        "Audio Smoothing",
        "float",
        "constraint",
        "0.8",
        "0.0..1.0",
        "Prevents noisy audio modulation.",
    ),
    "camera_path_amount": _ParameterSpec(
        "camera_path_amount",
        "Camera Path Amount",
        "float",
        "control",
        "0.15",
        "0.0..1.0",
        "Controls optional path or viewpoint modulation.",
    ),
    "interaction_strength": _ParameterSpec(
        "interaction_strength",
        "Interaction Strength",
        "float",
        "control",
        "0.0",
        "0.0..1.0",
        "Controls user gesture influence.",
    ),
    "frame_budget_ms": _ParameterSpec(
        "frame_budget_ms",
        "Frame Budget",
        "float",
        "constraint",
        "16.7",
        "8.0..33.3",
        "Keeps browser animation cost explicit.",
    ),
}

__all__ = [
    "GENERATIVE_STRUCTURE_AUTHORITY_BOUNDARY",
    "GenerativeArchitecture",
    "GenerativeEvolutionPhase",
    "GenerativeEvolutionRule",
    "GenerativeEvolutionTrigger",
    "GenerativeFallbackBlueprint",
    "GenerativeHookType",
    "GenerativeModule",
    "GenerativeModuleKind",
    "GenerativeModuleRelationship",
    "GenerativeParameter",
    "GenerativeParameterRole",
    "GenerativeParameterValueType",
    "GenerativeRelationshipType",
    "GenerativeStructureBlueprint",
    "GenerativeStructureHook",
    "derive_generative_structure_blueprint",
    "generative_structure_prompt_lines",
]
