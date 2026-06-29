"""V5.2 advisory provider capability matrix metadata."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.hybrid_studio import (
    ProviderSelectionPosture,
    ProviderSelectionProfile,
    ProviderSelectionRegistry,
    provider_selection_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

PROVIDER_CAPABILITY_MATRIX_ROW_SERIALIZATION_VERSION = (
    "provider_capability_matrix_row.v1"
)
PROVIDER_CAPABILITY_MATRIX_SERIALIZATION_VERSION = "provider_capability_matrix.v1"
PROVIDER_CAPABILITY_MATRIX_AUTHORITY_BOUNDARY = (
    "The V5.2 Provider Capability Matrix projects existing passive provider "
    "selection metadata into an inspectable provider capability matrix only; "
    "it does not select providers, select or switch models, route providers "
    "or models, execute local or cloud providers, request human input, control "
    "workflows, trigger retries, mutate prompts, write storage, or modify "
    "generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "provider_selection_execution",
    "automatic_provider_selection",
    "automatic_model_selection",
    "configured_model_switching",
    "provider_or_model_routing",
    "local_provider_execution",
    "cloud_provider_execution",
    "human_input_request_emission",
    "workflow_control",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class ProviderCapabilityMatrixRow(BaseModel):
    """One passive provider capability matrix row."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    row_id: str = Field(min_length=1, max_length=180)
    source_provider_selection_profile_id: str = Field(min_length=1, max_length=140)
    profile_name: str = Field(min_length=1, max_length=160)
    provider_selection_posture: ProviderSelectionPosture
    provider_candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    source_local_surface_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    source_cloud_surface_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=4,
    )
    source_auto_mode_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    source_hitl_decision_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
    )
    route_applicability: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    selection_inputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    advisory_outputs: tuple[str, ...] = Field(min_length=1, max_length=10)
    provider_candidate_count: int = Field(ge=1, le=5)
    route_count: int = Field(ge=1, le=6)
    local_surface_count: int = Field(ge=0, le=4)
    cloud_surface_count: int = Field(ge=0, le=4)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    provider_capability_matrix_implemented: Literal[True] = True
    provider_capability_lookup_implemented: Literal[True] = True
    provider_selection_implemented: Literal[False] = False
    automatic_provider_selection_implemented: Literal[False] = False
    automatic_model_selection_implemented: Literal[False] = False
    model_switching_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    local_provider_execution_implemented: Literal[False] = False
    cloud_provider_execution_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["provider_capability_matrix_row.v1"] = (
        PROVIDER_CAPABILITY_MATRIX_ROW_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _row_counts_match_metadata(self) -> Self:
        if self.provider_candidate_count != len(self.provider_candidate_ids):
            raise ValueError("provider_candidate_count must match provider candidates")
        if self.route_count != len(self.route_applicability):
            raise ValueError("route_count must match route_applicability")
        if self.local_surface_count != len(self.source_local_surface_ids):
            raise ValueError("local_surface_count must match local surfaces")
        if self.cloud_surface_count != len(self.source_cloud_surface_ids):
            raise ValueError("cloud_surface_count must match cloud surfaces")
        return self


class ProviderCapabilityMatrix(BaseModel):
    """Bounded V5.2 passive provider capability matrix."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["provider_capability_matrix"] = "provider_capability_matrix"
    serialization_version: Literal["provider_capability_matrix.v1"] = (
        PROVIDER_CAPABILITY_MATRIX_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PROVIDER_CAPABILITY_MATRIX_AUTHORITY_BOUNDARY,
        max_length=1500,
    )
    source_provider_selection_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    rows: tuple[ProviderCapabilityMatrixRow, ...] = Field(
        min_length=1,
        max_length=12,
    )
    row_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    provider_selection_profile_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
    provider_selection_postures: tuple[ProviderSelectionPosture, ...] = Field(
        min_length=1,
        max_length=12,
    )
    provider_candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    route_names: tuple[RouteName, ...] = Field(min_length=1, max_length=6)
    row_count: int = Field(ge=1, le=12)
    provider_selection_profile_count: int = Field(ge=1, le=12)
    provider_selection_posture_count: int = Field(ge=1, le=12)
    provider_candidate_count: int = Field(ge=1, le=12)
    route_count: int = Field(ge=1, le=6)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    provider_capability_matrix_implemented: Literal[True] = True
    provider_capability_lookup_implemented: Literal[True] = True
    provider_selection_implemented: Literal[False] = False
    automatic_provider_selection_implemented: Literal[False] = False
    automatic_model_selection_implemented: Literal[False] = False
    model_switching_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    local_provider_execution_implemented: Literal[False] = False
    cloud_provider_execution_implemented: Literal[False] = False
    human_input_request_implemented: Literal[False] = False
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
        if self.provider_selection_profile_ids != tuple(
            row.source_provider_selection_profile_id for row in self.rows
        ):
            raise ValueError("provider_selection_profile_ids must match rows")
        if self.provider_selection_postures != tuple(
            row.provider_selection_posture for row in self.rows
        ):
            raise ValueError("provider_selection_postures must match rows")
        if self.provider_candidate_ids != _dedupe(
            provider for row in self.rows for provider in row.provider_candidate_ids
        ):
            raise ValueError("provider_candidate_ids must match rows")
        if self.row_count != len(self.rows):
            raise ValueError("row_count must match rows")
        if self.provider_selection_profile_count != len(
            self.provider_selection_profile_ids
        ):
            raise ValueError(
                "provider_selection_profile_count must match profile ids"
            )
        if self.provider_selection_posture_count != len(
            self.provider_selection_postures
        ):
            raise ValueError(
                "provider_selection_posture_count must match postures"
            )
        if self.provider_candidate_count != len(self.provider_candidate_ids):
            raise ValueError("provider_candidate_count must match providers")
        if self.route_names != tuple(RouteName):
            raise ValueError("route_names must match route enum order")
        if self.route_count != len(self.route_names):
            raise ValueError("route_count must match route_names")
        known_routes = set(self.route_names)
        for row in self.rows:
            if not set(row.route_applicability).issubset(known_routes):
                raise ValueError("row route_applicability must be known route names")
        return self


def build_provider_capability_matrix(
    *,
    provider_selection: ProviderSelectionRegistry | None = None,
) -> ProviderCapabilityMatrix:
    """Return passive provider capability matrix metadata without selecting."""

    registry = provider_selection or provider_selection_registry()
    rows = tuple(
        _row_from_provider_selection_profile(profile)
        for profile in registry.provider_selection_profiles
    )
    provider_candidate_ids = _dedupe(
        provider for row in rows for provider in row.provider_candidate_ids
    )

    return ProviderCapabilityMatrix(
        source_provider_selection_serialization_version=registry.serialization_version,
        rows=rows,
        row_ids=tuple(row.row_id for row in rows),
        provider_selection_profile_ids=tuple(
            row.source_provider_selection_profile_id for row in rows
        ),
        provider_selection_postures=tuple(
            row.provider_selection_posture for row in rows
        ),
        provider_candidate_ids=provider_candidate_ids,
        route_names=tuple(RouteName),
        row_count=len(rows),
        provider_selection_profile_count=len(rows),
        provider_selection_posture_count=len(
            tuple(row.provider_selection_posture for row in rows)
        ),
        provider_candidate_count=len(provider_candidate_ids),
        route_count=len(tuple(RouteName)),
        advisory_actions=(
            "Present provider capability rows for inspection only.",
            "Keep provider selection, model switching, and provider execution disabled.",
        ),
    )


def provider_capability_row_by_profile_id(
    provider_selection_profile_id: str,
    matrix: ProviderCapabilityMatrix | None = None,
) -> ProviderCapabilityMatrixRow | None:
    """Return one provider capability row without selecting providers."""

    source_matrix = matrix or build_provider_capability_matrix()
    for row in source_matrix.rows:
        if row.source_provider_selection_profile_id == provider_selection_profile_id:
            return row
    return None


def provider_capability_rows_for_provider(
    provider_candidate_id: str,
    matrix: ProviderCapabilityMatrix | None = None,
) -> tuple[ProviderCapabilityMatrixRow, ...]:
    """Return passive provider capability rows mentioning one provider."""

    provider_id = str(provider_candidate_id).strip()
    source_matrix = matrix or build_provider_capability_matrix()
    return tuple(
        row for row in source_matrix.rows if provider_id in row.provider_candidate_ids
    )


def provider_capability_rows_for_route(
    route: RouteName | str,
    matrix: ProviderCapabilityMatrix | None = None,
) -> tuple[ProviderCapabilityMatrixRow, ...]:
    """Return passive provider capability rows applicable to one route."""

    route_name = route if isinstance(route, RouteName) else RouteName(str(route))
    source_matrix = matrix or build_provider_capability_matrix()
    return tuple(
        row for row in source_matrix.rows if route_name in row.route_applicability
    )


def _row_from_provider_selection_profile(
    profile: ProviderSelectionProfile,
) -> ProviderCapabilityMatrixRow:
    return ProviderCapabilityMatrixRow(
        row_id=f"provider_capability::{profile.provider_selection_profile_id}",
        source_provider_selection_profile_id=profile.provider_selection_profile_id,
        profile_name=profile.profile_name,
        provider_selection_posture=profile.provider_selection_posture,
        provider_candidate_ids=profile.provider_candidate_ids,
        source_local_surface_ids=profile.source_local_surface_ids,
        source_cloud_surface_ids=profile.source_cloud_surface_ids,
        source_auto_mode_profile_ids=profile.source_auto_mode_profile_ids,
        source_hitl_decision_profile_ids=profile.source_hitl_decision_profile_ids,
        route_applicability=profile.route_applicability,
        selection_inputs=profile.selection_inputs,
        advisory_outputs=profile.advisory_outputs,
        provider_candidate_count=len(profile.provider_candidate_ids),
        route_count=len(profile.route_applicability),
        local_surface_count=len(profile.source_local_surface_ids),
        cloud_surface_count=len(profile.source_cloud_surface_ids),
        evidence=(
            f"Derived from {profile.provider_selection_profile_id}.",
            f"Provider selection posture: {profile.provider_selection_posture}.",
            "Provider capabilities are passive selection metadata.",
        ),
        advisory_actions=(
            "Inspect provider candidates without selecting them.",
            "Keep provider/model routing and provider execution disabled.",
        ),
    )


def _dedupe(values: object) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))
