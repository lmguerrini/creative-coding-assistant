"""V5.1 creative complexity analyzer for advisory creative pressure signals."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.creative_constraints import (
    CreativeConstraintSolution,
)
from creative_coding_assistant.orchestration.creative_hierarchy import (
    CreativeHierarchyPlan,
)
from creative_coding_assistant.orchestration.creative_intent import (
    CreativeIntentDecomposition,
    active_intent_dimension_names,
)
from creative_coding_assistant.orchestration.creative_planning import (
    CreativeExecutionPlan,
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
from creative_coding_assistant.orchestration.runtime_capabilities import (
    RuntimeCapabilityProfile,
)

CreativeComplexityFactorKind = Literal[
    "input_coverage",
    "translation_surface",
    "intent_density",
    "hierarchy_pressure",
    "plan_shape",
    "technique_pressure",
    "constraint_pressure",
    "runtime_pressure",
    "tradeoff_risk",
]
CreativeComplexityLevel = Literal["low", "medium", "high"]

CREATIVE_COMPLEXITY_FACTOR_SERIALIZATION_VERSION = "creative_complexity_factor.v1"
CREATIVE_COMPLEXITY_ANALYSIS_SERIALIZATION_VERSION = "creative_complexity_analysis.v1"
CREATIVE_COMPLEXITY_ANALYZER_AUTHORITY_BOUNDARY = (
    "Creative complexity analysis derives advisory creative pressure signals "
    "from existing translation, intent, hierarchy, plan, technique, constraint, "
    "runtime, and trade-off metadata only; it does not rewrite prompts, change "
    "creative outputs, choose runtimes, select providers or models, control "
    "workflow execution, trigger retries, write storage, or modify generated "
    "output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "prompt_rewrite",
    "creative_output_mutation",
    "runtime_selection",
    "provider_or_model_routing",
    "workflow_control",
    "retry_or_refinement_triggering",
    "artifact_execution",
    "preview_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class CreativeComplexityFactor(BaseModel):
    """One advisory factor in a creative complexity analysis."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    factor_id: str = Field(min_length=1, max_length=160)
    factor_kind: CreativeComplexityFactorKind
    source_id: str = Field(min_length=1, max_length=120)
    level: CreativeComplexityLevel
    score: int = Field(ge=0, le=200)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    prompt_rewrite_implemented: Literal[False] = False
    creative_output_mutation_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    artifact_execution_implemented: Literal[False] = False
    preview_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["creative_complexity_factor.v1"] = (
        CREATIVE_COMPLEXITY_FACTOR_SERIALIZATION_VERSION
    )
    analysis_only: Literal[True] = True


class CreativeComplexityAnalysis(BaseModel):
    """Bounded V5.1 analysis of creative-side complexity pressure."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["creative_complexity_analyzer"] = "creative_complexity_analyzer"
    serialization_version: Literal["creative_complexity_analysis.v1"] = (
        CREATIVE_COMPLEXITY_ANALYSIS_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=CREATIVE_COMPLEXITY_ANALYZER_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    source_metadata_roles: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    factors: tuple[CreativeComplexityFactor, ...] = Field(min_length=1, max_length=12)
    factor_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    active_intent_dimension_count: int = Field(ge=0, le=10)
    unresolved_gap_count: int = Field(ge=0, le=20)
    hierarchy_conflict_count: int = Field(ge=0, le=20)
    runtime_candidate_count: int = Field(ge=0, le=12)
    tradeoff_risk_count: int = Field(ge=0, le=40)
    creative_complexity_score: int = Field(ge=0, le=400)
    creative_complexity_level: CreativeComplexityLevel
    hitl_advisable: bool
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    creative_complexity_analysis_implemented: Literal[True] = True
    prompt_rewrite_implemented: Literal[False] = False
    creative_output_mutation_implemented: Literal[False] = False
    runtime_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    artifact_execution_implemented: Literal[False] = False
    preview_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    analysis_only: Literal[True] = True

    @model_validator(mode="after")
    def _analysis_matches_factors(self) -> Self:
        derived_factor_ids = tuple(factor.factor_id for factor in self.factors)
        if len(set(derived_factor_ids)) != len(derived_factor_ids):
            raise ValueError("factor_ids must be unique")
        if self.factor_ids != derived_factor_ids:
            raise ValueError("factor_ids must match factors")
        derived_score = sum(factor.score for factor in self.factors)
        if self.creative_complexity_score != derived_score:
            raise ValueError("creative_complexity_score must match factors")
        if self.creative_complexity_level != _complexity_level(derived_score):
            raise ValueError("creative_complexity_level must match score")
        if self.unresolved_gap_count > 0 and not self.hitl_advisable:
            raise ValueError("unresolved gaps require HITL advisability")
        return self


def analyze_creative_complexity(
    *,
    creative_translation: CreativeTranslation | None = None,
    creative_intent: CreativeIntentDecomposition | None = None,
    creative_hierarchy: CreativeHierarchyPlan | None = None,
    creative_plan: CreativeExecutionPlan | None = None,
    creative_techniques: CreativeTechniqueProfile | None = None,
    creative_constraints: CreativeConstraintSolution | None = None,
    runtime_capabilities: RuntimeCapabilityProfile | None = None,
    creative_tradeoffs: CreativeTradeoffProfile | None = None,
) -> CreativeComplexityAnalysis:
    """Return advisory creative complexity analysis without mutation."""

    factors = _factors(
        creative_translation=creative_translation,
        creative_intent=creative_intent,
        creative_hierarchy=creative_hierarchy,
        creative_plan=creative_plan,
        creative_techniques=creative_techniques,
        creative_constraints=creative_constraints,
        runtime_capabilities=runtime_capabilities,
        creative_tradeoffs=creative_tradeoffs,
    )
    score = sum(factor.score for factor in factors)
    active_dimensions = (
        len(active_intent_dimension_names(creative_intent))
        if creative_intent is not None
        else 0
    )
    unresolved_gaps = (
        len(creative_intent.unresolved_intent_gaps)
        if creative_intent is not None
        else 0
    )
    hierarchy_conflicts = (
        len(creative_hierarchy.priority_conflicts)
        if creative_hierarchy is not None
        else 0
    )
    runtime_candidates = (
        len(runtime_capabilities.candidate_runtimes)
        if runtime_capabilities is not None
        else 0
    )
    tradeoff_risks = _tradeoff_risk_count(creative_tradeoffs)
    hitl_advisable = any(
        (
            unresolved_gaps > 0,
            bool(creative_hierarchy and creative_hierarchy.hitl_questions),
            bool(creative_constraints and creative_constraints.hitl_advisable),
            bool(runtime_capabilities and runtime_capabilities.hitl_advisable),
            bool(creative_tradeoffs and creative_tradeoffs.hitl_advisable),
            any(factor.level == "high" for factor in factors),
        )
    )

    return CreativeComplexityAnalysis(
        source_metadata_roles=_source_metadata_roles(
            creative_translation=creative_translation,
            creative_intent=creative_intent,
            creative_hierarchy=creative_hierarchy,
            creative_plan=creative_plan,
            creative_techniques=creative_techniques,
            creative_constraints=creative_constraints,
            runtime_capabilities=runtime_capabilities,
            creative_tradeoffs=creative_tradeoffs,
        ),
        factors=factors,
        factor_ids=tuple(factor.factor_id for factor in factors),
        active_intent_dimension_count=active_dimensions,
        unresolved_gap_count=unresolved_gaps,
        hierarchy_conflict_count=hierarchy_conflicts,
        runtime_candidate_count=runtime_candidates,
        tradeoff_risk_count=tradeoff_risks,
        creative_complexity_score=score,
        creative_complexity_level=_complexity_level(score),
        hitl_advisable=hitl_advisable,
        advisory_actions=_analysis_actions(score, hitl_advisable),
    )


def creative_complexity_factor_by_id(
    factor_id: str,
    analysis: CreativeComplexityAnalysis | None = None,
) -> CreativeComplexityFactor | None:
    """Return one creative complexity factor without changing output."""

    source_analysis = analysis or analyze_creative_complexity()
    for factor in source_analysis.factors:
        if factor.factor_id == factor_id:
            return factor
    return None


def creative_complexity_factors_for_kind(
    factor_kind: CreativeComplexityFactorKind,
    analysis: CreativeComplexityAnalysis | None = None,
) -> tuple[CreativeComplexityFactor, ...]:
    """Return creative complexity factors by kind without runtime action."""

    source_analysis = analysis or analyze_creative_complexity()
    return tuple(
        factor
        for factor in source_analysis.factors
        if factor.factor_kind == factor_kind
    )


def _factors(
    *,
    creative_translation: CreativeTranslation | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_constraints: CreativeConstraintSolution | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
) -> tuple[CreativeComplexityFactor, ...]:
    factors: list[CreativeComplexityFactor] = []
    if creative_translation is not None:
        factors.append(_translation_surface_factor(creative_translation))
    if creative_intent is not None:
        factors.append(_intent_density_factor(creative_intent))
    if creative_hierarchy is not None:
        factors.append(_hierarchy_pressure_factor(creative_hierarchy))
    if creative_plan is not None:
        factors.append(_plan_shape_factor(creative_plan))
    if creative_techniques is not None:
        factors.append(_technique_pressure_factor(creative_techniques))
    if creative_constraints is not None:
        factors.append(_constraint_pressure_factor(creative_constraints))
    if runtime_capabilities is not None:
        factors.append(_runtime_pressure_factor(runtime_capabilities))
    if creative_tradeoffs is not None:
        factors.append(_tradeoff_risk_factor(creative_tradeoffs))
    if not factors:
        factors.append(_input_coverage_factor())
    return tuple(factors)


def _input_coverage_factor() -> CreativeComplexityFactor:
    return CreativeComplexityFactor(
        factor_id="factor::input_coverage",
        factor_kind="input_coverage",
        source_id="creative_complexity_inputs",
        level="low",
        score=1,
        evidence=("no_creative_metadata_supplied",),
        advisory_actions=("Collect creative metadata before optimization decisions.",),
    )


def _translation_surface_factor(
    translation: CreativeTranslation,
) -> CreativeComplexityFactor:
    surface_count = sum(
        (
            len(translation.symbolic_references),
            len(translation.geometric_references),
            len(translation.musical_references),
            len(translation.mood_atmosphere),
            len(translation.movement_language),
            len(translation.color_material_direction),
            len(translation.runtime_recommendations),
            len(translation.structure_direction),
            len(translation.generation_constraints),
            len(translation.refinement_targets),
        )
    )
    optional_guidance_count = sum(
        int(item is not None)
        for item in (
            translation.sacred_geometry,
            translation.shader_presets,
            translation.visual_style,
            translation.audio_reactive,
            translation.reference_fusion,
        )
    )
    score = surface_count + optional_guidance_count * 3
    return CreativeComplexityFactor(
        factor_id="factor::translation_surface",
        factor_kind="translation_surface",
        source_id="creative_translation",
        level=_factor_level(score),
        score=score,
        evidence=(
            f"surface_count:{surface_count}",
            f"optional_guidance:{optional_guidance_count}",
            f"constraints:{len(translation.generation_constraints)}",
        ),
        advisory_actions=("Use translation breadth as creative pressure metadata.",),
    )


def _intent_density_factor(
    intent: CreativeIntentDecomposition,
) -> CreativeComplexityFactor:
    active_dimensions = active_intent_dimension_names(intent)
    score = (
        len(active_dimensions) * 2
        + len(intent.unresolved_intent_gaps) * 4
        + len(intent.hitl_questions) * 2
    )
    return CreativeComplexityFactor(
        factor_id="factor::intent_density",
        factor_kind="intent_density",
        source_id=intent.role,
        level=_factor_level(score),
        score=score,
        evidence=(
            f"active_dimensions:{len(active_dimensions)}",
            f"unresolved_gaps:{len(intent.unresolved_intent_gaps)}",
            f"abstraction:{intent.abstraction_level}",
        ),
        advisory_actions=("Keep dense intent visible for context budgeting.",),
    )


def _hierarchy_pressure_factor(
    hierarchy: CreativeHierarchyPlan,
) -> CreativeComplexityFactor:
    score = (
        len(hierarchy.primary_creative_priorities) * 3
        + len(hierarchy.non_negotiable_dimensions) * 2
        + len(hierarchy.priority_conflicts) * 5
        + len(hierarchy.hitl_questions) * 3
    )
    return CreativeComplexityFactor(
        factor_id="factor::hierarchy_pressure",
        factor_kind="hierarchy_pressure",
        source_id=hierarchy.role,
        level=_factor_level(score),
        score=score,
        evidence=(
            f"primary:{len(hierarchy.primary_creative_priorities)}",
            f"non_negotiable:{len(hierarchy.non_negotiable_dimensions)}",
            f"conflicts:{len(hierarchy.priority_conflicts)}",
        ),
        advisory_actions=("Use hierarchy pressure without changing priorities.",),
    )


def _plan_shape_factor(plan: CreativeExecutionPlan) -> CreativeComplexityFactor:
    score = (
        plan.candidate_count * 3
        + plan.refinement_budget * 4
        + _pressure_rank(plan.expected_complexity) * 3
        + int(plan.estimated_token_cost >= 6200) * 4
    )
    return CreativeComplexityFactor(
        factor_id="factor::plan_shape",
        factor_kind="plan_shape",
        source_id="creative_execution_plan",
        level=_factor_level(score),
        score=score,
        evidence=(
            f"candidate_count:{plan.candidate_count}",
            f"refinement_budget:{plan.refinement_budget}",
            f"expected_complexity:{plan.expected_complexity}",
            f"tokens:{plan.estimated_token_cost}",
        ),
        advisory_actions=("Use plan shape as pressure metadata only.",),
    )


def _technique_pressure_factor(
    techniques: CreativeTechniqueProfile,
) -> CreativeComplexityFactor:
    score = (
        _pressure_rank(techniques.complexity_pressure) * 4
        + _pressure_rank(techniques.performance_pressure) * 3
        + len(techniques.alternative_techniques) * 2
        + len(techniques.technique_constraints)
    )
    return CreativeComplexityFactor(
        factor_id="factor::technique_pressure",
        factor_kind="technique_pressure",
        source_id=techniques.role,
        level=_factor_level(score),
        score=score,
        evidence=(
            f"primary:{techniques.primary_technique}",
            f"complexity:{techniques.complexity_pressure}",
            f"performance:{techniques.performance_pressure}",
        ),
        advisory_actions=("Expose technique pressure without selecting runtimes.",),
    )


def _constraint_pressure_factor(
    constraints: CreativeConstraintSolution,
) -> CreativeComplexityFactor:
    score = (
        _pressure_rank(constraints.complexity_pressure) * 4
        + _pressure_rank(constraints.safety_pressure) * 2
        + _pressure_rank(constraints.performance_pressure) * 3
        + _pressure_rank(constraints.cost_pressure) * 2
        + len(constraints.conflicts) * 4
        + int(constraints.hitl_advisable) * 4
    )
    return CreativeComplexityFactor(
        factor_id="factor::constraint_pressure",
        factor_kind="constraint_pressure",
        source_id=constraints.role,
        level=_factor_level(score),
        score=score,
        evidence=(
            f"complexity:{constraints.complexity_pressure}",
            f"performance:{constraints.performance_pressure}",
            f"cost:{constraints.cost_pressure}",
            f"conflicts:{len(constraints.conflicts)}",
        ),
        advisory_actions=("Use constraint pressure without enforcing changes.",),
    )


def _runtime_pressure_factor(
    runtime_capabilities: RuntimeCapabilityProfile,
) -> CreativeComplexityFactor:
    high_complexity = sum(
        1
        for candidate in runtime_capabilities.candidate_runtimes
        if candidate.implementation_complexity == "high"
    )
    medium_complexity = sum(
        1
        for candidate in runtime_capabilities.candidate_runtimes
        if candidate.implementation_complexity == "medium"
    )
    score = (
        high_complexity * 4
        + medium_complexity * 2
        + len(runtime_capabilities.likely_candidates)
        + int(runtime_capabilities.hitl_advisable) * 4
    )
    return CreativeComplexityFactor(
        factor_id="factor::runtime_pressure",
        factor_kind="runtime_pressure",
        source_id=runtime_capabilities.role,
        level=_factor_level(score),
        score=score,
        evidence=(
            f"candidates:{len(runtime_capabilities.candidate_runtimes)}",
            f"high_complexity:{high_complexity}",
            f"likely:{len(runtime_capabilities.likely_candidates)}",
        ),
        advisory_actions=("Use runtime pressure without selecting runtimes.",),
    )


def _tradeoff_risk_factor(
    tradeoffs: CreativeTradeoffProfile,
) -> CreativeComplexityFactor:
    risk_count = _tradeoff_risk_count(tradeoffs)
    score = (
        len(tradeoffs.complexity_risks) * 4
        + len(tradeoffs.performance_concerns) * 3
        + len(tradeoffs.runtime_risks) * 3
        + len(tradeoffs.maintainability_concerns) * 2
        + len(tradeoffs.fidelity_risks) * 2
        + int(tradeoffs.hitl_advisable) * 4
    )
    return CreativeComplexityFactor(
        factor_id="factor::tradeoff_risk",
        factor_kind="tradeoff_risk",
        source_id=tradeoffs.role,
        level=_factor_level(score),
        score=score,
        evidence=(
            f"risk_count:{risk_count}",
            f"complexity_risks:{len(tradeoffs.complexity_risks)}",
            f"cost_sensitivity:{tradeoffs.cost_sensitivity}",
        ),
        advisory_actions=("Keep trade-off risks visible without choosing outcomes.",),
    )


def _source_metadata_roles(
    *,
    creative_translation: CreativeTranslation | None,
    creative_intent: CreativeIntentDecomposition | None,
    creative_hierarchy: CreativeHierarchyPlan | None,
    creative_plan: CreativeExecutionPlan | None,
    creative_techniques: CreativeTechniqueProfile | None,
    creative_constraints: CreativeConstraintSolution | None,
    runtime_capabilities: RuntimeCapabilityProfile | None,
    creative_tradeoffs: CreativeTradeoffProfile | None,
) -> tuple[str, ...]:
    roles: list[str] = []
    if creative_translation is not None:
        roles.append("creative_translation")
    for item in (
        creative_intent,
        creative_hierarchy,
        creative_plan,
        creative_techniques,
        creative_constraints,
        runtime_capabilities,
        creative_tradeoffs,
    ):
        if item is None:
            continue
        roles.append(getattr(item, "role", item.__class__.__name__))
    return tuple(roles)


def _tradeoff_risk_count(tradeoffs: CreativeTradeoffProfile | None) -> int:
    if tradeoffs is None:
        return 0
    return sum(
        (
            len(tradeoffs.runtime_risks),
            len(tradeoffs.performance_concerns),
            len(tradeoffs.complexity_risks),
            len(tradeoffs.fidelity_risks),
            len(tradeoffs.safety_concerns),
            len(tradeoffs.maintainability_concerns),
        )
    )


def _pressure_rank(value: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(value, 1)


def _factor_level(score: int) -> CreativeComplexityLevel:
    if score < 10:
        return "low"
    if score < 24:
        return "medium"
    return "high"


def _complexity_level(score: int) -> CreativeComplexityLevel:
    if score < 32:
        return "low"
    if score < 84:
        return "medium"
    return "high"


def _analysis_actions(
    score: int,
    hitl_advisable: bool,
) -> tuple[str, ...]:
    actions = [
        "Expose creative complexity as advisory metadata only.",
        "Preserve generated output and routing boundaries.",
    ]
    if score >= 84:
        actions.append("Flag high creative pressure for later budget planning.")
    if hitl_advisable:
        actions.append("Surface HITL advisability without opening a gate.")
    return tuple(actions)
