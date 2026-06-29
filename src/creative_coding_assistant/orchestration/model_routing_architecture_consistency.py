"""Passive V5.2 model-routing architecture consistency metadata."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration.budget_policies import (
    evaluate_budget_policies,
)
from creative_coding_assistant.orchestration.cost_estimator import (
    estimate_routing_cost,
)
from creative_coding_assistant.orchestration.cost_prediction_engine import (
    predict_cost_for_route,
)
from creative_coding_assistant.orchestration.creative_consistency_predictor import (
    predict_creative_consistency,
)
from creative_coding_assistant.orchestration.creative_diversity_predictor import (
    predict_creative_diversity,
)
from creative_coding_assistant.orchestration.creative_quality_prediction import (
    derive_creative_quality_prediction,
)
from creative_coding_assistant.orchestration.execution_policy_engine import (
    evaluate_execution_policies,
)
from creative_coding_assistant.orchestration.hitl_budget_gate import (
    evaluate_hitl_budget_gate,
)
from creative_coding_assistant.orchestration.hybrid_routing import (
    route_hybrid_model_request,
)
from creative_coding_assistant.orchestration.local_cloud_routing import (
    route_local_vs_cloud,
)
from creative_coding_assistant.orchestration.model_capability_matrix import (
    build_model_capability_matrix,
)
from creative_coding_assistant.orchestration.model_recommendation_engine import (
    recommend_model_profile,
)
from creative_coding_assistant.orchestration.model_router import route_model_request
from creative_coding_assistant.orchestration.provider_capability_matrix import (
    build_provider_capability_matrix,
)
from creative_coding_assistant.orchestration.quality_cost_optimizer import (
    optimize_quality_cost,
)
from creative_coding_assistant.orchestration.quality_prediction_engine import (
    predict_quality_for_route,
)
from creative_coding_assistant.orchestration.routing import (
    RouteCapability,
    RouteDecision,
    RouteName,
)
from creative_coding_assistant.orchestration.routing_explainability import (
    explain_routing_decision,
)
from creative_coding_assistant.orchestration.runtime_recommendation_engine import (
    recommend_runtime_execution,
)

ModelRoutingArchitectureLayer = Literal[
    "routing_metadata_boundary",
    "optimization_budget_boundary",
    "runtime_policy_boundary",
    "capability_matrix_boundary",
    "prediction_metadata_boundary",
    "creative_prediction_boundary",
    "explainability_boundary",
]
ModelRoutingArchitectureStage = Literal["v5_2_architecture_consistency_pass"]
ModelRoutingArchitectureStatus = Literal["pass"]

MODEL_ROUTING_ARCHITECTURE_CONSISTENCY_RECORD_SERIALIZATION_VERSION = (
    "model_routing_architecture_consistency_record.v1"
)
MODEL_ROUTING_ARCHITECTURE_CONSISTENCY_REGISTRY_SERIALIZATION_VERSION = (
    "model_routing_architecture_consistency_registry.v1"
)
MODEL_ROUTING_ARCHITECTURE_CONSISTENCY_AUTHORITY_BOUNDARY = (
    "V5.2 model-routing architecture consistency metadata checks advisory "
    "surface coverage, serialization, route consistency, passive runtime "
    "boundaries, V4 compatibility, and version-level HITL/runtime evolution "
    "rules only; it does not apply routing, select or switch providers or "
    "models, execute providers, enforce budgets, emit HITL requests, control "
    "workflows, trigger retries, mutate prompts, write storage, or modify "
    "generated output."
)

_ROUTE_NAME = RouteName.GENERATE
_ARCHITECTURE_LAYERS: tuple[ModelRoutingArchitectureLayer, ...] = (
    "routing_metadata_boundary",
    "optimization_budget_boundary",
    "runtime_policy_boundary",
    "capability_matrix_boundary",
    "prediction_metadata_boundary",
    "creative_prediction_boundary",
    "explainability_boundary",
)
_VALIDATED_VERSION_RULES = (
    "v5_surface_role_declared",
    "serialization_version_declared",
    "advisory_metadata_only",
    "provider_model_routing_not_applied",
    "generated_output_mutation_blocked",
    "workflow_control_blocked",
    "runtime_evolution_not_applied",
    "v4_boundary_compatibility_confirmed",
    "human_gate_not_emitted",
)
_PASSIVE_BOUNDARY_FLAGS = (
    "provider_model_routing_blocked",
    "configured_model_switching_blocked",
    "provider_execution_blocked",
    "budget_enforcement_blocked",
    "hitl_emission_blocked",
    "workflow_control_blocked",
    "prompt_mutation_blocked",
    "storage_write_blocked",
    "generated_output_mutation_blocked",
    "runtime_evolution_blocked",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "provider_or_model_routing",
    "configured_model_switching",
    "provider_execution",
    "budget_enforcement",
    "human_input_request_emission",
    "workflow_control",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
    "runtime_evolution_application",
)
_ACTIVE_RUNTIME_FLAGS = (
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "workflow_control_implemented",
    "budget_enforcement_implemented",
    "hitl_request_emitted",
    "human_input_request_implemented",
    "routing_application_implemented",
    "runtime_recommendation_application_implemented",
    "execution_policy_application_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "storage_write_implemented",
    "generated_output_mutation_implemented",
)
_COUNT_FIELD_CANDIDATES = (
    "candidate_count",
    "decision_count",
    "scenario_count",
    "policy_count",
    "gate_count",
    "recommendation_count",
    "row_count",
    "prediction_count",
    "explanation_count",
    "quality_signal_count",
)
_SURFACE_LAYERS: tuple[tuple[str, ModelRoutingArchitectureLayer], ...] = (
    ("model_router", "routing_metadata_boundary"),
    ("local_cloud_routing", "routing_metadata_boundary"),
    ("hybrid_routing", "routing_metadata_boundary"),
    ("quality_cost_optimizer", "optimization_budget_boundary"),
    ("cost_estimator", "optimization_budget_boundary"),
    ("budget_policies", "optimization_budget_boundary"),
    ("hitl_budget_gate", "optimization_budget_boundary"),
    ("runtime_recommendation_engine", "runtime_policy_boundary"),
    ("execution_policy_engine", "runtime_policy_boundary"),
    ("model_recommendation_engine", "runtime_policy_boundary"),
    ("model_capability_matrix", "capability_matrix_boundary"),
    ("provider_capability_matrix", "capability_matrix_boundary"),
    ("quality_prediction_engine", "prediction_metadata_boundary"),
    ("cost_prediction_engine", "prediction_metadata_boundary"),
    ("creative_quality_predictor", "creative_prediction_boundary"),
    ("creative_diversity_predictor", "creative_prediction_boundary"),
    ("creative_consistency_predictor", "creative_prediction_boundary"),
    ("routing_explainability", "explainability_boundary"),
)
_SERIALIZATION_OVERRIDES = {
    "creative_quality_predictor": "creative_quality_prediction.v1",
}


class ModelRoutingArchitectureConsistencyRecord(BaseModel):
    """One passive V5.2 architecture consistency record."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    surface_id: str = Field(min_length=1, max_length=120)
    architecture_layer: ModelRoutingArchitectureLayer
    architecture_stage: ModelRoutingArchitectureStage = (
        "v5_2_architecture_consistency_pass"
    )
    source_role: str = Field(min_length=1, max_length=120)
    source_serialization_version: str = Field(min_length=1, max_length=120)
    source_route_name: RouteName | None = None
    source_count_field: str = Field(min_length=1, max_length=80)
    source_count: int = Field(ge=1, le=80)
    validated_version_rules: tuple[str, ...] = Field(min_length=9, max_length=9)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=10, max_length=10)
    source_blocked_runtime_behaviors: tuple[str, ...] = Field(
        min_length=1,
        max_length=32,
    )
    source_active_runtime_flags: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=24,
    )
    missing_coverage_items: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    source_metadata_only_declared: Literal[True] = True
    v5_architecture_consistency_confirmed: Literal[True] = True
    v4_boundary_compatibility_confirmed: Literal[True] = True
    version_runtime_rules_confirmed: Literal[True] = True
    architecture_consistency_status: ModelRoutingArchitectureStatus = "pass"
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    hitl_emission_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    serialization_version: Literal[
        "model_routing_architecture_consistency_record.v1"
    ] = MODEL_ROUTING_ARCHITECTURE_CONSISTENCY_RECORD_SERIALIZATION_VERSION
    advisory_only: Literal[True] = True


class ModelRoutingArchitectureConsistencyRegistry(BaseModel):
    """Passive V5.2 architecture consistency pass registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["model_routing_architecture_consistency_registry"] = (
        "model_routing_architecture_consistency_registry"
    )
    serialization_version: Literal[
        "model_routing_architecture_consistency_registry.v1"
    ] = MODEL_ROUTING_ARCHITECTURE_CONSISTENCY_REGISTRY_SERIALIZATION_VERSION
    authority_boundary: str = Field(
        default=MODEL_ROUTING_ARCHITECTURE_CONSISTENCY_AUTHORITY_BOUNDARY,
        max_length=1500,
    )
    architecture_stage: ModelRoutingArchitectureStage = (
        "v5_2_architecture_consistency_pass"
    )
    route_name: RouteName = _ROUTE_NAME
    records: tuple[ModelRoutingArchitectureConsistencyRecord, ...] = Field(
        min_length=18,
        max_length=18,
    )
    surface_ids: tuple[str, ...] = Field(min_length=18, max_length=18)
    record_count: int = Field(ge=18, le=18)
    architecture_layers: tuple[ModelRoutingArchitectureLayer, ...] = Field(
        min_length=7,
        max_length=7,
    )
    validated_version_rules: tuple[str, ...] = Field(min_length=9, max_length=9)
    passive_boundary_flags: tuple[str, ...] = Field(min_length=10, max_length=10)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=14,
    )
    all_surfaces_covered: Literal[True] = True
    route_consistency_confirmed: Literal[True] = True
    no_active_runtime_flags: Literal[True] = True
    no_missing_coverage: Literal[True] = True
    v4_boundaries_preserved: Literal[True] = True
    runtime_evolution_not_applied: Literal[True] = True
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    hitl_emission_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    runtime_evolution_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_records(self) -> Self:
        derived_surface_ids = tuple(record.surface_id for record in self.records)
        if len(set(derived_surface_ids)) != len(derived_surface_ids):
            raise ValueError("surface_ids must be unique")
        if self.surface_ids != derived_surface_ids:
            raise ValueError("surface_ids must match records")
        if self.surface_ids != _surface_ids():
            raise ValueError("surface_ids must match V5.2 surface order")
        if self.record_count != len(self.records):
            raise ValueError("record_count must match records")
        if self.architecture_layers != _ARCHITECTURE_LAYERS:
            raise ValueError("architecture_layers must match V5.2 layer order")
        if self.validated_version_rules != _VALIDATED_VERSION_RULES:
            raise ValueError("validated_version_rules must match registry")
        if self.passive_boundary_flags != _PASSIVE_BOUNDARY_FLAGS:
            raise ValueError("passive_boundary_flags must match registry")
        for record in self.records:
            if record.architecture_stage != self.architecture_stage:
                raise ValueError("architecture_stage must match registry")
            if record.validated_version_rules != self.validated_version_rules:
                raise ValueError("validated_version_rules must match registry")
            if record.passive_boundary_flags != self.passive_boundary_flags:
                raise ValueError("passive_boundary_flags must match registry")
            if record.source_route_name is not None and (
                record.source_route_name != self.route_name
            ):
                raise ValueError("source_route_name must match registry route")
            if record.source_active_runtime_flags:
                raise ValueError("records must not contain active runtime flags")
            if record.missing_coverage_items:
                raise ValueError("records must not contain missing coverage")
        return self


def model_routing_architecture_consistency_registry(
) -> ModelRoutingArchitectureConsistencyRegistry:
    """Return passive V5.2 model-routing architecture consistency metadata."""

    records = tuple(
        _record_from_source(surface_id, layer, source)
        for surface_id, layer, source in _source_specs()
    )
    return ModelRoutingArchitectureConsistencyRegistry(
        records=records,
        surface_ids=tuple(record.surface_id for record in records),
        record_count=len(records),
        architecture_layers=_ARCHITECTURE_LAYERS,
        validated_version_rules=_VALIDATED_VERSION_RULES,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
    )


def model_routing_architecture_consistency_by_surface(
    surface_id: str,
    registry: ModelRoutingArchitectureConsistencyRegistry | None = None,
) -> ModelRoutingArchitectureConsistencyRecord | None:
    """Return one V5.2 architecture record by surface id."""

    source_registry = registry or model_routing_architecture_consistency_registry()
    normalized_surface_id = str(surface_id).strip()
    for record in source_registry.records:
        if record.surface_id == normalized_surface_id:
            return record
    return None


def model_routing_architecture_consistency_records_for_layer(
    architecture_layer: str,
    registry: ModelRoutingArchitectureConsistencyRegistry | None = None,
) -> tuple[ModelRoutingArchitectureConsistencyRecord, ...]:
    """Return passive V5.2 architecture records for one layer."""

    source_registry = registry or model_routing_architecture_consistency_registry()
    normalized_layer = str(architecture_layer).strip()
    return tuple(
        record
        for record in source_registry.records
        if record.architecture_layer == normalized_layer
    )


def _source_specs(
) -> tuple[tuple[str, ModelRoutingArchitectureLayer, Any], ...]:
    sources = _source_objects()
    return tuple(
        (surface_id, layer, sources[surface_id])
        for surface_id, layer in _SURFACE_LAYERS
    )


def _source_objects() -> dict[str, Any]:
    model_routing = route_model_request(route=_ROUTE_NAME)
    local_cloud = route_local_vs_cloud(
        model_routing=model_routing,
        route=_ROUTE_NAME,
    )
    hybrid = route_hybrid_model_request(
        local_cloud_routing=local_cloud,
        model_routing=model_routing,
        route=_ROUTE_NAME,
    )
    quality_cost = optimize_quality_cost(hybrid_routing=hybrid, route=_ROUTE_NAME)
    cost_estimation = estimate_routing_cost(quality_cost_plan=quality_cost)
    budget_policies = evaluate_budget_policies(cost_estimation=cost_estimation)
    hitl_gate = evaluate_hitl_budget_gate(budget_policies=budget_policies)
    runtime_recommendation = recommend_runtime_execution(hitl_budget_gate=hitl_gate)
    execution_policy = evaluate_execution_policies(
        runtime_recommendations=runtime_recommendation
    )
    model_recommendation = recommend_model_profile(
        model_routing=model_routing,
        execution_policies=execution_policy,
    )
    quality_prediction = predict_quality_for_route(route=_ROUTE_NAME)
    cost_prediction = predict_cost_for_route(route=_ROUTE_NAME)
    creative_quality = _creative_quality_source()
    creative_diversity = predict_creative_diversity()
    creative_consistency = predict_creative_consistency(
        creative_quality_prediction=creative_quality,
    )
    explainability = explain_routing_decision(
        model_routing=model_routing,
        local_cloud_routing=local_cloud,
        hybrid_routing=hybrid,
        quality_prediction=quality_prediction,
        cost_prediction=cost_prediction,
        model_recommendation=model_recommendation,
    )

    return {
        "model_router": model_routing,
        "local_cloud_routing": local_cloud,
        "hybrid_routing": hybrid,
        "quality_cost_optimizer": quality_cost,
        "cost_estimator": cost_estimation,
        "budget_policies": budget_policies,
        "hitl_budget_gate": hitl_gate,
        "runtime_recommendation_engine": runtime_recommendation,
        "execution_policy_engine": execution_policy,
        "model_recommendation_engine": model_recommendation,
        "model_capability_matrix": build_model_capability_matrix(),
        "provider_capability_matrix": build_provider_capability_matrix(),
        "quality_prediction_engine": quality_prediction,
        "cost_prediction_engine": cost_prediction,
        "creative_quality_predictor": creative_quality,
        "creative_diversity_predictor": creative_diversity,
        "creative_consistency_predictor": creative_consistency,
        "routing_explainability": explainability,
    }


def _record_from_source(
    surface_id: str,
    layer: ModelRoutingArchitectureLayer,
    source: Any,
) -> ModelRoutingArchitectureConsistencyRecord:
    count_field, source_count = _source_count(source)
    serialization_version = _source_serialization(surface_id, source)
    active_flags = _active_runtime_flags(source)
    missing = _missing_coverage(
        source=source,
        serialization_version=serialization_version,
        source_count=source_count,
        active_flags=active_flags,
    )
    return ModelRoutingArchitectureConsistencyRecord(
        surface_id=surface_id,
        architecture_layer=layer,
        source_role=str(getattr(source, "role", surface_id)),
        source_serialization_version=serialization_version,
        source_route_name=getattr(source, "route_name", None),
        source_count_field=count_field,
        source_count=source_count,
        validated_version_rules=_VALIDATED_VERSION_RULES,
        passive_boundary_flags=_PASSIVE_BOUNDARY_FLAGS,
        source_blocked_runtime_behaviors=_source_blocked_runtime_behaviors(source),
        source_active_runtime_flags=active_flags,
        missing_coverage_items=missing,
        source_metadata_only_declared=_source_metadata_only_declared(source),
    )


def _creative_quality_source() -> Any:
    request = AssistantRequest(
        query="Generate V5.2 architecture consistency metadata.",
        mode=AssistantMode.GENERATE,
        domains=(CreativeCodingDomain.P5_JS,),
    )
    route = RouteDecision(
        route=_ROUTE_NAME,
        mode=AssistantMode.GENERATE,
        domain=CreativeCodingDomain.P5_JS,
        domains=(CreativeCodingDomain.P5_JS,),
        capabilities=(RouteCapability.TOOL_USE,),
    )
    return derive_creative_quality_prediction(
        request=request,
        route_decision=route,
        creative_translation=None,
        creative_intent=None,
        creative_hierarchy=None,
        creative_plan=None,
        creative_constraints=None,
        creative_constraint_priorities=None,
        creative_strategy=None,
        creative_techniques=None,
        runtime_capabilities=None,
        creative_tradeoffs=None,
    )


def _source_count(source: Any) -> tuple[str, int]:
    for count_field in _COUNT_FIELD_CANDIDATES:
        if hasattr(source, count_field):
            return count_field, int(getattr(source, count_field))
    if hasattr(source, "strongest_quality_signals") and hasattr(
        source,
        "weakest_quality_signals",
    ):
        return "quality_signal_count", len(
            source.strongest_quality_signals + source.weakest_quality_signals
        )
    raise ValueError("V5.2 architecture source must expose a count field")


def _source_serialization(surface_id: str, source: Any) -> str:
    if surface_id in _SERIALIZATION_OVERRIDES:
        return _SERIALIZATION_OVERRIDES[surface_id]
    return str(getattr(source, "serialization_version", ""))


def _source_blocked_runtime_behaviors(source: Any) -> tuple[str, ...]:
    blocked = getattr(source, "blocked_runtime_behaviors", None)
    if blocked:
        return tuple(str(item) for item in blocked)
    return _BLOCKED_RUNTIME_BEHAVIORS


def _source_metadata_only_declared(source: Any) -> bool:
    return bool(
        getattr(
            source,
            "advisory_only",
            getattr(source, "recommendation_only", getattr(source, "metadata_only", True)),
        )
    )


def _active_runtime_flags(source: Any) -> tuple[str, ...]:
    return tuple(
        flag for flag in _ACTIVE_RUNTIME_FLAGS if bool(getattr(source, flag, False))
    )


def _missing_coverage(
    *,
    source: Any,
    serialization_version: str,
    source_count: int,
    active_flags: tuple[str, ...],
) -> tuple[str, ...]:
    missing: list[str] = []
    if not getattr(source, "role", ""):
        missing.append("role_missing")
    if not serialization_version.endswith(".v1"):
        missing.append("serialization_version_missing")
    if source_count < 1:
        missing.append("source_count_missing")
    if not _source_blocked_runtime_behaviors(source):
        missing.append("blocked_runtime_behaviors_missing")
    if active_flags:
        missing.append("active_runtime_flags_present")
    if not _source_metadata_only_declared(source):
        missing.append("metadata_only_declaration_missing")
    return tuple(missing)


def _surface_ids() -> tuple[str, ...]:
    return tuple(surface_id for surface_id, _layer in _SURFACE_LAYERS)
