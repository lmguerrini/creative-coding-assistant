"""V5.2 advisory model capability matrix metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.hybrid_studio import (
    ModelProfile,
    ModelProfileKind,
    ModelProfileRegistry,
    model_profile_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

MODEL_CAPABILITY_MATRIX_ROW_SERIALIZATION_VERSION = (
    "model_capability_matrix_row.v1"
)
MODEL_CAPABILITY_MATRIX_SERIALIZATION_VERSION = "model_capability_matrix.v1"
MODEL_CAPABILITY_MATRIX_AUTHORITY_BOUNDARY = (
    "The V5.2 Model Capability Matrix projects existing passive model profile "
    "metadata into an inspectable capability matrix only; it does not score "
    "capabilities, predict quality or cost, select or switch providers or "
    "models, route providers or models, execute providers, apply execution "
    "policies, control workflows, trigger retries, mutate prompts, write "
    "storage, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "capability_scoring",
    "quality_prediction",
    "cost_prediction",
    "automatic_model_selection",
    "automatic_provider_selection",
    "configured_model_switching",
    "provider_or_model_routing",
    "provider_execution",
    "execution_policy_application",
    "workflow_control",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class ModelCapabilityMatrixRow(BaseModel):
    """One passive model capability matrix row."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    row_id: str = Field(min_length=1, max_length=180)
    source_model_profile_id: str = Field(min_length=1, max_length=120)
    profile_name: str = Field(min_length=1, max_length=160)
    model_profile_kind: ModelProfileKind
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    capability_dimensions: tuple[str, ...] = Field(min_length=1, max_length=10)
    provider_candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_local_surface_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    source_cloud_surface_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    profile_inputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    route_count: int = Field(ge=1, le=6)
    capability_dimension_count: int = Field(ge=1, le=10)
    provider_candidate_count: int = Field(ge=1, le=5)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    model_capability_matrix_implemented: Literal[True] = True
    capability_lookup_implemented: Literal[True] = True
    capability_scoring_implemented: Literal[False] = False
    quality_prediction_implemented: Literal[False] = False
    cost_prediction_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    automatic_model_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    execution_policy_application_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["model_capability_matrix_row.v1"] = (
        MODEL_CAPABILITY_MATRIX_ROW_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _row_counts_match_metadata(self) -> Self:
        if self.route_count != len(self.route_applicability):
            raise ValueError("route_count must match route_applicability")
        if self.capability_dimension_count != len(self.capability_dimensions):
            raise ValueError("capability_dimension_count must match dimensions")
        if self.provider_candidate_count != len(self.provider_candidate_ids):
            raise ValueError("provider_candidate_count must match provider candidates")
        return self


class ModelCapabilityMatrix(BaseModel):
    """Bounded V5.2 passive model capability matrix."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["model_capability_matrix"] = "model_capability_matrix"
    serialization_version: Literal["model_capability_matrix.v1"] = (
        MODEL_CAPABILITY_MATRIX_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=MODEL_CAPABILITY_MATRIX_AUTHORITY_BOUNDARY,
        max_length=1500,
    )
    source_model_profile_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    rows: tuple[ModelCapabilityMatrixRow, ...] = Field(min_length=1, max_length=12)
    row_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    model_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    route_names: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    capability_dimensions: tuple[str, ...] = Field(min_length=1, max_length=40)
    provider_candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    row_count: int = Field(ge=1, le=12)
    model_profile_count: int = Field(ge=1, le=12)
    route_count: int = Field(ge=1, le=6)
    capability_dimension_count: int = Field(ge=1, le=40)
    provider_candidate_count: int = Field(ge=1, le=12)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    model_capability_matrix_implemented: Literal[True] = True
    capability_lookup_implemented: Literal[True] = True
    capability_scoring_implemented: Literal[False] = False
    quality_prediction_implemented: Literal[False] = False
    cost_prediction_implemented: Literal[False] = False
    model_selection_implemented: Literal[False] = False
    automatic_model_selection_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    execution_policy_application_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _matrix_matches_rows(self) -> Self:
        derived_row_ids = tuple(row.row_id for row in self.rows)
        if len(set(derived_row_ids)) != len(derived_row_ids):
            raise ValueError("row_ids must be unique")
        if self.row_ids != derived_row_ids:
            raise ValueError("row_ids must match rows")
        if self.model_profile_ids != tuple(
            row.source_model_profile_id for row in self.rows
        ):
            raise ValueError("model_profile_ids must match rows")
        if self.row_count != len(self.rows):
            raise ValueError("row_count must match rows")
        if self.model_profile_count != len(self.model_profile_ids):
            raise ValueError("model_profile_count must match model_profile_ids")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.route_count != len(self.route_names):
            raise ValueError("route_count must match route_names")
        if self.capability_dimensions != _dedupe(
            dimension
            for row in self.rows
            for dimension in row.capability_dimensions
        ):
            raise ValueError("capability_dimensions must match rows")
        if self.provider_candidate_ids != _dedupe(
            provider
            for row in self.rows
            for provider in row.provider_candidate_ids
        ):
            raise ValueError("provider_candidate_ids must match rows")
        if self.capability_dimension_count != len(self.capability_dimensions):
            raise ValueError("capability_dimension_count must match dimensions")
        if self.provider_candidate_count != len(self.provider_candidate_ids):
            raise ValueError("provider_candidate_count must match providers")
        known_routes = set(self.route_names)
        for row in self.rows:
            if not set(row.route_applicability).issubset(known_routes):
                raise ValueError("row route_applicability must be known route names")
        return self


def build_model_capability_matrix(
    *,
    model_profiles: ModelProfileRegistry | None = None,
) -> ModelCapabilityMatrix:
    """Return passive model capability matrix metadata without scoring it."""

    registry = model_profiles or model_profile_registry()
    rows = tuple(_row_from_profile(profile) for profile in registry.model_profiles)
    capability_dimensions = _dedupe(
        dimension for row in rows for dimension in row.capability_dimensions
    )
    provider_candidate_ids = _dedupe(
        provider for row in rows for provider in row.provider_candidate_ids
    )

    return ModelCapabilityMatrix(
        source_model_profile_serialization_version=registry.serialization_version,
        rows=rows,
        row_ids=tuple(row.row_id for row in rows),
        model_profile_ids=tuple(row.source_model_profile_id for row in rows),
        route_names=tuple(RouteName),
        capability_dimensions=capability_dimensions,
        provider_candidate_ids=provider_candidate_ids,
        row_count=len(rows),
        model_profile_count=len(rows),
        route_count=len(tuple(RouteName)),
        capability_dimension_count=len(capability_dimensions),
        provider_candidate_count=len(provider_candidate_ids),
        advisory_actions=(
            "Present model capability rows for inspection only.",
            "Keep capability scoring, model selection, and provider routing disabled.",
        ),
    )


def model_capability_row_by_profile_id(
    model_profile_id: str,
    matrix: ModelCapabilityMatrix | None = None,
) -> ModelCapabilityMatrixRow | None:
    """Return one matrix row without selecting or scoring a model."""

    source_matrix = matrix or build_model_capability_matrix()
    for row in source_matrix.rows:
        if row.source_model_profile_id == model_profile_id:
            return row
    return None


def model_capability_rows_for_route(
    route: RouteName | str,
    matrix: ModelCapabilityMatrix | None = None,
) -> tuple[ModelCapabilityMatrixRow, ...]:
    """Return passive model capability rows applicable to one route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_matrix = matrix or build_model_capability_matrix()
    return tuple(
        row for row in source_matrix.rows if route_name in row.route_applicability
    )


def _row_from_profile(profile: ModelProfile) -> ModelCapabilityMatrixRow:
    return ModelCapabilityMatrixRow(
        row_id=f"model_capability::{profile.model_profile_id}",
        source_model_profile_id=profile.model_profile_id,
        profile_name=profile.profile_name,
        model_profile_kind=profile.model_profile_kind,
        route_applicability=profile.route_applicability,
        capability_dimensions=profile.capability_dimensions,
        provider_candidate_ids=profile.provider_candidate_ids,
        source_local_surface_ids=profile.source_local_surface_ids,
        source_cloud_surface_ids=profile.source_cloud_surface_ids,
        profile_inputs=profile.profile_inputs,
        advisory_outputs=profile.advisory_outputs,
        route_count=len(profile.route_applicability),
        capability_dimension_count=len(profile.capability_dimensions),
        provider_candidate_count=len(profile.provider_candidate_ids),
        evidence=(
            f"Derived from {profile.model_profile_id}.",
            f"Model profile kind: {profile.model_profile_kind}.",
            "Capability dimensions are passive profile metadata.",
        ),
        advisory_actions=(
            "Inspect capability dimensions without scoring them.",
            "Keep model selection and provider routing disabled.",
        ),
    )


def _dedupe(values: object) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))
