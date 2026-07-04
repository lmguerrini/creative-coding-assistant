"""V5.2 passive routing intelligence architecture surfaces."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.hybrid_studio import (
    ModelProfileRegistry,
    model_profile_registry,
)

ProviderId = Literal["openai", "anthropic", "gemini", "local"]
ProviderFamily = Literal["openai", "anthropic", "gemini", "local"]
ProviderCategory = Literal["cloud", "local"]
CredentialRequirement = Literal[
    "api_key_required",
    "none_required",
    "user_managed_local_runtime",
]
ProviderModelFamily = Literal[
    "gpt",
    "claude",
    "gemini",
    "embedding",
    "local_chat",
    "local_code",
    "local_multimodal",
]
RoutingCapabilityFamily = Literal[
    "coding",
    "reasoning",
    "creative_coding",
    "creative_writing",
    "long_context_reasoning",
    "multimodal_understanding",
    "image_understanding",
    "tool_use",
    "structured_output",
    "fast_draft",
    "low_cost_execution",
    "maximum_quality_execution",
]
AvailabilityCheckPolicy = Literal[
    "settings_metadata_only",
    "environment_metadata_only",
    "local_runtime_metadata_only",
    "manual_operator_confirmation",
]
ProviderAvailabilityStatus = Literal[
    "available_by_configuration",
    "unavailable",
    "requires_hitl",
    "unknown_metadata_only",
]
UnavailableReasonCode = Literal[
    "missing_api_key",
    "provider_unsupported",
    "local_runtime_unavailable",
    "local_model_not_installed",
    "insufficient_local_resources",
    "missing_modality_support",
    "cost_policy_blocked",
    "latency_policy_blocked",
    "hitl_required",
]
LocalRuntimeKind = Literal[
    "ollama",
    "lm_studio",
    "llama_cpp",
    "local_transformers",
]
ExecutionModeId = Literal["manual_mode", "assisted_mode", "auto_mode"]
TaskRoutingType = Literal[
    "coding",
    "reasoning",
    "creative_coding",
    "creative_writing",
    "long_context_reasoning",
    "multimodal_understanding",
    "image_understanding",
    "tool_use",
    "structured_output",
    "fast_draft",
    "low_cost_execution",
    "maximum_quality_execution",
]
HybridRoutingPolicyDirection = Literal[
    "local_to_cloud",
    "cloud_to_local",
    "cloud_to_cloud",
    "local_to_local",
]
RoutingRiskBand = Literal["low", "medium", "high"]
EstimatedQualityBand = Literal["low", "medium", "high", "maximum"]
EstimatedCostBand = Literal["low", "medium", "high"]
EstimatedLatencyBand = Literal["fast", "moderate", "slow"]

PROVIDER_PROFILE_SERIALIZATION_VERSION = "routing_provider_profile.v1"
PROVIDER_PROFILE_REGISTRY_SERIALIZATION_VERSION = "routing_provider_profile_registry.v1"
PROVIDER_AVAILABILITY_SERIALIZATION_VERSION = "provider_availability_metadata.v1"
PROVIDER_AVAILABILITY_REGISTRY_SERIALIZATION_VERSION = (
    "provider_availability_registry.v1"
)
TASK_ROUTING_DECISION_SERIALIZATION_VERSION = "task_aware_routing_decision.v1"
TASK_ROUTING_REGISTRY_SERIALIZATION_VERSION = "task_aware_routing_registry.v1"
EXECUTION_MODE_PROFILE_SERIALIZATION_VERSION = "routing_execution_mode_profile.v1"
EXECUTION_MODE_REGISTRY_SERIALIZATION_VERSION = "routing_execution_mode_registry.v1"
HYBRID_ROUTING_POLICY_SERIALIZATION_VERSION = "advisory_hybrid_routing_policy.v1"
HYBRID_ROUTING_POLICY_REGISTRY_SERIALIZATION_VERSION = (
    "advisory_hybrid_routing_policy_registry.v1"
)
UNAVAILABLE_REASON_SERIALIZATION_VERSION = "routing_unavailable_reason.v1"
ROUTING_SAFETY_CONTRACT_SERIALIZATION_VERSION = "routing_safety_contract.v1"
ROUTING_SAFETY_CONTRACT_REGISTRY_SERIALIZATION_VERSION = (
    "routing_safety_contract_registry.v1"
)
MODEL_ROUTING_INTELLIGENCE_REGISTRY_SERIALIZATION_VERSION = (
    "model_routing_intelligence_registry.v1"
)

_REQUIRED_PROVIDER_IDS: tuple[ProviderId, ...] = (
    "openai",
    "anthropic",
    "gemini",
    "local",
)
_REQUIRED_TASK_TYPES: tuple[TaskRoutingType, ...] = (
    "coding",
    "reasoning",
    "creative_coding",
    "creative_writing",
    "long_context_reasoning",
    "multimodal_understanding",
    "image_understanding",
    "tool_use",
    "structured_output",
    "fast_draft",
    "low_cost_execution",
    "maximum_quality_execution",
)
_REQUIRED_EXECUTION_MODES: tuple[ExecutionModeId, ...] = (
    "manual_mode",
    "assisted_mode",
    "auto_mode",
)
_REQUIRED_HYBRID_DIRECTIONS: tuple[HybridRoutingPolicyDirection, ...] = (
    "local_to_cloud",
    "cloud_to_local",
    "cloud_to_cloud",
    "local_to_local",
)
_REQUIRED_UNAVAILABLE_REASONS: tuple[UnavailableReasonCode, ...] = (
    "missing_api_key",
    "provider_unsupported",
    "local_runtime_unavailable",
    "local_model_not_installed",
    "insufficient_local_resources",
    "missing_modality_support",
    "cost_policy_blocked",
    "latency_policy_blocked",
    "hitl_required",
)
_REQUIRED_SAFETY_BOUNDARIES = (
    "no_automatic_provider_switching",
    "no_automatic_model_download",
    "no_automatic_api_key_assumption",
    "hitl_before_unavailable_provider_or_model",
    "hitl_before_expensive_or_high_risk_auto_route",
    "provider_selection_boundary",
    "credential_boundary",
)
_BLOCKED_RUNTIME_BEHAVIORS = (
    "automatic_provider_switching",
    "automatic_model_download",
    "automatic_api_key_assumption",
    "provider_or_model_routing_application",
    "provider_execution",
    "configured_provider_switching",
    "configured_model_switching",
    "human_input_request_emission",
    "workflow_control",
    "retry_or_refinement_triggering",
    "prompt_mutation",
    "persistent_storage_write",
    "generated_output_modification",
)


class RoutingProviderProfile(BaseModel):
    """One explicit provider profile for passive V5.2 routing intelligence."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    provider_profile_id: str = Field(min_length=1, max_length=120)
    provider_id: ProviderId
    provider_family: ProviderFamily
    provider_category: ProviderCategory
    credential_requirements: tuple[CredentialRequirement, ...] = Field(
        min_length=1,
        max_length=3,
    )
    supported_model_families: tuple[ProviderModelFamily, ...] = Field(
        min_length=1,
        max_length=8,
    )
    supported_capability_families: tuple[RoutingCapabilityFamily, ...] = Field(
        min_length=1,
        max_length=12,
    )
    availability_check_policy: AvailabilityCheckPolicy
    routing_safety_notes: tuple[str, ...] = Field(min_length=1, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    provider_profile_implemented: Literal[True] = True
    provider_availability_detection_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    automatic_api_key_assumption_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["routing_provider_profile.v1"] = (
        PROVIDER_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _profile_identity_matches(self) -> Self:
        if self.provider_profile_id != f"provider_profile::{self.provider_id}":
            raise ValueError("provider_profile_id must match provider_id")
        if self.provider_family != self.provider_id:
            raise ValueError("provider_family must match provider_id")
        if self.provider_category == "local" and self.provider_id != "local":
            raise ValueError("only the local provider may use local category")
        if self.provider_category == "cloud" and self.provider_id == "local":
            raise ValueError("local provider must use local category")
        return self


class RoutingProviderProfileRegistry(BaseModel):
    """Registry for explicit V5.2 provider profiles."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["routing_provider_profile_registry"] = (
        "routing_provider_profile_registry"
    )
    serialization_version: Literal["routing_provider_profile_registry.v1"] = (
        PROVIDER_PROFILE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=(
            "V5.2 provider profiles expose advisory provider capability metadata "
            "only; they do not select providers, switch configured providers or "
            "models, execute providers, download models, assume API keys, request "
            "human input, mutate prompts, write storage, or modify generated output."
        ),
        max_length=1200,
    )
    provider_profiles: tuple[RoutingProviderProfile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    provider_ids: tuple[ProviderId, ...] = Field(min_length=4, max_length=4)
    provider_count: int = Field(ge=4, le=4)
    extension_points: tuple[str, ...] = Field(min_length=4, max_length=4)
    new_provider_requires_explicit_profile: Literal[True] = True
    runtime_adapter_required_for_execution: Literal[True] = True
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    provider_profiles_implemented: Literal[True] = True
    extensible_provider_architecture_implemented: Literal[True] = True
    provider_availability_detection_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    automatic_api_key_assumption_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_profiles(self) -> Self:
        provider_ids = tuple(profile.provider_id for profile in self.provider_profiles)
        if self.provider_ids != provider_ids:
            raise ValueError("provider_ids must match profiles")
        if self.provider_ids != _REQUIRED_PROVIDER_IDS:
            raise ValueError("provider_ids must cover required V5.2 providers")
        if self.provider_count != len(self.provider_profiles):
            raise ValueError("provider_count must match profiles")
        if self.extension_points != (
            "provider_profile_metadata",
            "provider_availability_metadata",
            "provider_capability_family_metadata",
            "routing_safety_contract_metadata",
        ):
            raise ValueError("extension_points must match V5.2 provider architecture")
        return self


class RoutingUnavailableReason(BaseModel):
    """Serializable reason that can block or degrade an advisory route."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    reason_code: UnavailableReasonCode
    reason_summary: str = Field(min_length=1, max_length=220)
    user_action_required: bool
    hitl_required: bool
    blocks_auto_mode: bool
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    unavailable_reason_model_implemented: Literal[True] = True
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["routing_unavailable_reason.v1"] = (
        UNAVAILABLE_REASON_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class ApiKeyDetectionMetadata(BaseModel):
    """Metadata-only API key detection policy for a provider."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    provider_id: ProviderId
    credential_names: tuple[str, ...] = Field(default_factory=tuple, max_length=4)
    detection_policy: AvailabilityCheckPolicy
    detection_result: Literal["not_checked_metadata_only"] = "not_checked_metadata_only"
    detection_performed: Literal[False] = False
    credential_value_exposed: Literal[False] = False
    automatic_api_key_assumption_implemented: Literal[False] = False


class CredentialBoundary(BaseModel):
    """Provider credential boundary metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    provider_id: ProviderId
    credential_required: bool
    credential_assumption_allowed: Literal[False] = False
    credential_value_storage_allowed: Literal[False] = False
    credential_value_exposed: Literal[False] = False
    provider_execution_allowed: Literal[False] = False
    boundary_notes: tuple[str, ...] = Field(min_length=1, max_length=6)


class ProviderAvailabilityMetadata(BaseModel):
    """Advisory provider availability metadata without live checks."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    provider_id: ProviderId
    provider_profile_id: str = Field(min_length=1, max_length=120)
    availability_status: ProviderAvailabilityStatus
    availability_check_policy: AvailabilityCheckPolicy
    unavailable_reason_codes: tuple[UnavailableReasonCode, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    hitl_required_before_recommendation: bool
    real_network_check_performed: Literal[False] = False
    provider_call_performed: Literal[False] = False
    provider_availability_detection_metadata_implemented: Literal[True] = True
    provider_availability_detection_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False

    @model_validator(mode="after")
    def _profile_matches_provider(self) -> Self:
        if self.provider_profile_id != f"provider_profile::{self.provider_id}":
            raise ValueError("provider_profile_id must match provider_id")
        if (
            self.availability_status == "unavailable"
            and not self.unavailable_reason_codes
        ):
            raise ValueError("unavailable providers require reason codes")
        return self


class LocalRuntimeDetectionMetadata(BaseModel):
    """Metadata-only local runtime detection policy."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    runtime_kind: LocalRuntimeKind
    runtime_name: str = Field(min_length=1, max_length=120)
    detection_policy: Literal["user_managed_runtime_metadata_only"] = (
        "user_managed_runtime_metadata_only"
    )
    detection_status: Literal["unknown_metadata_only"] = "unknown_metadata_only"
    runtime_probe_performed: Literal[False] = False
    runtime_start_attempted: Literal[False] = False
    local_runtime_detection_metadata_implemented: Literal[True] = True
    local_runtime_detection_implemented: Literal[False] = False


class LocalModelInventoryMetadata(BaseModel):
    """Static local model inventory metadata without listing or downloads."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    inventory_id: str = Field(min_length=1, max_length=140)
    runtime_kind: LocalRuntimeKind
    candidate_model_families: tuple[ProviderModelFamily, ...] = Field(
        min_length=1,
        max_length=4,
    )
    candidate_model_labels: tuple[str, ...] = Field(min_length=1, max_length=8)
    inventory_policy: Literal["static_metadata_only"] = "static_metadata_only"
    model_listing_performed: Literal[False] = False
    model_download_attempted: Literal[False] = False
    local_model_inventory_metadata_implemented: Literal[True] = True

    @model_validator(mode="after")
    def _inventory_id_matches_runtime(self) -> Self:
        if self.inventory_id != f"local_model_inventory::{self.runtime_kind}":
            raise ValueError("inventory_id must match runtime_kind")
        return self


class LocalModelAvailabilityMetadata(BaseModel):
    """Advisory local model availability metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    availability_id: str = Field(min_length=1, max_length=150)
    runtime_kind: LocalRuntimeKind
    availability_status: Literal["unknown_metadata_only"] = "unknown_metadata_only"
    unavailable_reason_codes: tuple[UnavailableReasonCode, ...] = Field(
        min_length=1,
        max_length=4,
    )
    hitl_required_before_recommendation: Literal[True] = True
    model_probe_performed: Literal[False] = False
    model_download_attempted: Literal[False] = False
    local_model_availability_metadata_implemented: Literal[True] = True
    local_model_availability_detection_implemented: Literal[False] = False

    @model_validator(mode="after")
    def _availability_id_matches_runtime(self) -> Self:
        if self.availability_id != f"local_model_availability::{self.runtime_kind}":
            raise ValueError("availability_id must match runtime_kind")
        return self


class ProviderAvailabilityRegistry(BaseModel):
    """Provider, credential, local runtime, and local model availability metadata."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["provider_availability_registry"] = "provider_availability_registry"
    serialization_version: Literal["provider_availability_registry.v1"] = (
        PROVIDER_AVAILABILITY_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=(
            "V5.2 provider availability metadata models credential boundaries, "
            "provider availability, local runtime checks, local model inventory, "
            "local model availability, and unavailable reasons for inspection only; "
            "it does not perform network checks, probe local runtimes, list or "
            "download local models, assume API keys, switch providers, execute "
            "providers, or modify generated output."
        ),
        max_length=1500,
    )
    api_key_detection: tuple[ApiKeyDetectionMetadata, ...] = Field(
        min_length=4,
        max_length=4,
    )
    credential_boundaries: tuple[CredentialBoundary, ...] = Field(
        min_length=4,
        max_length=4,
    )
    provider_availability: tuple[ProviderAvailabilityMetadata, ...] = Field(
        min_length=4,
        max_length=4,
    )
    local_runtime_detection: tuple[LocalRuntimeDetectionMetadata, ...] = Field(
        min_length=4,
        max_length=4,
    )
    local_model_inventory: tuple[LocalModelInventoryMetadata, ...] = Field(
        min_length=4,
        max_length=4,
    )
    local_model_availability: tuple[LocalModelAvailabilityMetadata, ...] = Field(
        min_length=4,
        max_length=4,
    )
    unavailable_reasons: tuple[RoutingUnavailableReason, ...] = Field(
        min_length=9,
        max_length=9,
    )
    provider_ids: tuple[ProviderId, ...] = Field(min_length=4, max_length=4)
    unavailable_reason_codes: tuple[UnavailableReasonCode, ...] = Field(
        min_length=9,
        max_length=9,
    )
    decision_count: int = Field(ge=33, le=33)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    provider_availability_metadata_implemented: Literal[True] = True
    provider_availability_detection_implemented: Literal[False] = False
    local_runtime_detection_implemented: Literal[False] = False
    local_model_discovery_implemented: Literal[False] = False
    local_model_download_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_api_key_assumption_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _availability_registry_matches_records(self) -> Self:
        if self.provider_ids != tuple(
            item.provider_id for item in self.provider_availability
        ):
            raise ValueError("provider_ids must match provider availability")
        if self.provider_ids != _REQUIRED_PROVIDER_IDS:
            raise ValueError("provider_ids must cover required providers")
        if self.unavailable_reason_codes != tuple(
            reason.reason_code for reason in self.unavailable_reasons
        ):
            raise ValueError("unavailable_reason_codes must match reasons")
        if self.unavailable_reason_codes != _REQUIRED_UNAVAILABLE_REASONS:
            raise ValueError("unavailable reasons must cover required codes")
        expected_count = (
            len(self.api_key_detection)
            + len(self.credential_boundaries)
            + len(self.provider_availability)
            + len(self.local_runtime_detection)
            + len(self.local_model_inventory)
            + len(self.local_model_availability)
            + len(self.unavailable_reasons)
        )
        if self.decision_count != expected_count:
            raise ValueError("decision_count must match availability records")
        return self


class RoutingExecutionModeProfile(BaseModel):
    """Explicit passive routing execution mode contract."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    execution_mode_id: ExecutionModeId
    mode_name: Literal["Manual Mode", "Assisted Mode", "Auto Mode"]
    routing_authority: str = Field(min_length=1, max_length=260)
    confirmation_policy: str = Field(min_length=1, max_length=260)
    hitl_required_reason_codes: tuple[UnavailableReasonCode, ...] = Field(
        min_length=1,
        max_length=9,
    )
    safe_auto_boundary: bool
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    execution_mode_profile_implemented: Literal[True] = True
    execution_mode_application_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    automatic_api_key_assumption_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["routing_execution_mode_profile.v1"] = (
        EXECUTION_MODE_PROFILE_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class RoutingExecutionModeRegistry(BaseModel):
    """Registry for Manual, Assisted, and Auto routing modes."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["routing_execution_mode_registry"] = "routing_execution_mode_registry"
    serialization_version: Literal["routing_execution_mode_registry.v1"] = (
        EXECUTION_MODE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=(
            "V5.2 execution mode metadata describes Manual, Assisted, and Auto "
            "routing posture only; it does not apply execution modes, switch "
            "providers or models, download models, assume API keys, execute "
            "providers, emit HITL requests, or modify generated output."
        ),
        max_length=1200,
    )
    execution_modes: tuple[RoutingExecutionModeProfile, ...] = Field(
        min_length=3,
        max_length=3,
    )
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    decision_count: int = Field(ge=3, le=3)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    execution_modes_implemented: Literal[True] = True
    execution_mode_application_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    automatic_api_key_assumption_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _execution_modes_match(self) -> Self:
        mode_ids = tuple(mode.execution_mode_id for mode in self.execution_modes)
        if self.execution_mode_ids != mode_ids:
            raise ValueError("execution_mode_ids must match execution_modes")
        if self.execution_mode_ids != _REQUIRED_EXECUTION_MODES:
            raise ValueError("execution modes must cover Manual, Assisted, and Auto")
        if self.decision_count != len(self.execution_modes):
            raise ValueError("decision_count must match execution_modes")
        return self


class TaskAwareRoutingDecision(BaseModel):
    """Task-aware advisory model routing path."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    decision_id: str = Field(min_length=1, max_length=180)
    task_type: TaskRoutingType
    capability_requirements: tuple[RoutingCapabilityFamily, ...] = Field(
        min_length=1,
        max_length=6,
    )
    available_model_profile_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    recommended_model_profile_id: str = Field(min_length=1, max_length=120)
    fallback_model_profile_id: str = Field(min_length=1, max_length=120)
    execution_mode_id: ExecutionModeId
    available_route_summary: str = Field(min_length=1, max_length=260)
    recommended_route_summary: str = Field(min_length=1, max_length=260)
    fallback_route_summary: str = Field(min_length=1, max_length=260)
    unavailable_route_reason_summary: str = Field(min_length=1, max_length=260)
    estimated_quality: EstimatedQualityBand
    estimated_cost: EstimatedCostBand
    estimated_latency: EstimatedLatencyBand
    confidence_score: float = Field(ge=0.0, le=1.0)
    unavailable_reason_codes: tuple[UnavailableReasonCode, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    hitl_required: bool
    risk_band: RoutingRiskBand
    routing_path: tuple[str, ...] = Field(min_length=6, max_length=6)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    task_aware_routing_implemented: Literal[True] = True
    task_routing_application_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    automatic_api_key_assumption_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["task_aware_routing_decision.v1"] = (
        TASK_ROUTING_DECISION_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _decision_is_consistent(self) -> Self:
        if self.decision_id != f"task_route::{self.task_type}":
            raise ValueError("decision_id must match task_type")
        if self.recommended_model_profile_id not in self.available_model_profile_ids:
            raise ValueError("recommended_model_profile_id must be available")
        if self.fallback_model_profile_id not in self.available_model_profile_ids:
            raise ValueError("fallback_model_profile_id must be available")
        if self.recommended_model_profile_id == self.fallback_model_profile_id:
            raise ValueError("fallback model must differ from recommendation")
        expected_path = (
            "task_type",
            "capability_requirements",
            "available_models",
            "recommended_model",
            "fallback_model",
            "execution_mode",
        )
        if self.routing_path != expected_path:
            raise ValueError("routing_path must match V5.2 task-aware routing shape")
        if self.execution_mode_id == "auto_mode":
            risky = self.risk_band == "high" or bool(self.unavailable_reason_codes)
            if risky and not self.hitl_required:
                raise ValueError("risky auto-mode decisions require HITL")
        if "hitl_required" in self.unavailable_reason_codes and not self.hitl_required:
            raise ValueError("hitl_required reason must set hitl_required")
        return self


class TaskAwareRoutingRegistry(BaseModel):
    """Registry of task-aware advisory routing paths."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["task_aware_routing_registry"] = "task_aware_routing_registry"
    serialization_version: Literal["task_aware_routing_registry.v1"] = (
        TASK_ROUTING_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=(
            "V5.2 task-aware routing maps task type to capability requirements, "
            "available model metadata, recommended model metadata, fallback model "
            "metadata, and execution mode for inspection only; it does not apply "
            "routing, switch providers or models, download models, assume API "
            "keys, execute providers, or modify generated output."
        ),
        max_length=1400,
    )
    source_model_profile_serialization_version: str = Field(min_length=1, max_length=80)
    source_execution_mode_serialization_version: str = Field(
        min_length=1,
        max_length=80,
    )
    decisions: tuple[TaskAwareRoutingDecision, ...] = Field(
        min_length=12,
        max_length=12,
    )
    decision_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    task_types: tuple[TaskRoutingType, ...] = Field(min_length=12, max_length=12)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    decision_count: int = Field(ge=12, le=12)
    hitl_required_decision_count: int = Field(ge=1, le=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    task_aware_routing_implemented: Literal[True] = True
    task_routing_application_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    automatic_api_key_assumption_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_decisions(self) -> Self:
        if self.decision_ids != tuple(
            decision.decision_id for decision in self.decisions
        ):
            raise ValueError("decision_ids must match decisions")
        if self.task_types != tuple(decision.task_type for decision in self.decisions):
            raise ValueError("task_types must match decisions")
        if self.task_types != _REQUIRED_TASK_TYPES:
            raise ValueError("task_types must cover required V5.2 taxonomy")
        if self.execution_mode_ids != _REQUIRED_EXECUTION_MODES:
            raise ValueError("execution_mode_ids must cover execution modes")
        if self.decision_count != len(self.decisions):
            raise ValueError("decision_count must match decisions")
        if self.hitl_required_decision_count != sum(
            1 for decision in self.decisions if decision.hitl_required
        ):
            raise ValueError("hitl_required_decision_count must match decisions")
        return self


class AdvisoryHybridRoutingPolicy(BaseModel):
    """Advisory hybrid routing policy for one direction."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    policy_id: str = Field(min_length=1, max_length=140)
    direction: HybridRoutingPolicyDirection
    intended_use_case: str = Field(min_length=1, max_length=260)
    fallback_logic: str = Field(min_length=1, max_length=260)
    availability_constraints: tuple[str, ...] = Field(min_length=1, max_length=8)
    cost_quality_latency_tradeoff: str = Field(min_length=1, max_length=260)
    hitl_requirements: tuple[str, ...] = Field(min_length=1, max_length=8)
    safety_constraints: tuple[str, ...] = Field(min_length=1, max_length=8)
    unavailable_reason_codes: tuple[UnavailableReasonCode, ...] = Field(
        default_factory=tuple,
        max_length=6,
    )
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    hybrid_routing_policy_implemented: Literal[True] = True
    hybrid_routing_application_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["advisory_hybrid_routing_policy.v1"] = (
        HYBRID_ROUTING_POLICY_SERIALIZATION_VERSION
    )
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _policy_id_matches_direction(self) -> Self:
        if self.policy_id != f"hybrid_policy::{self.direction}":
            raise ValueError("policy_id must match direction")
        return self


class AdvisoryHybridRoutingPolicyRegistry(BaseModel):
    """Registry for explicit advisory hybrid routing policies."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["advisory_hybrid_routing_policy_registry"] = (
        "advisory_hybrid_routing_policy_registry"
    )
    serialization_version: Literal["advisory_hybrid_routing_policy_registry.v1"] = (
        HYBRID_ROUTING_POLICY_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=(
            "V5.2 hybrid routing policy metadata describes Local to Cloud, Cloud "
            "to Local, Cloud to Cloud, and Local to Local routing postures for "
            "review only; it does not execute hybrid workflows, switch providers "
            "or models, download models, assume API keys, merge provider output, "
            "or modify generated output."
        ),
        max_length=1400,
    )
    policies: tuple[AdvisoryHybridRoutingPolicy, ...] = Field(
        min_length=4,
        max_length=4,
    )
    policy_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    directions: tuple[HybridRoutingPolicyDirection, ...] = Field(
        min_length=4,
        max_length=4,
    )
    decision_count: int = Field(ge=4, le=4)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    hybrid_routing_policies_implemented: Literal[True] = True
    hybrid_routing_application_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    advisory_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_policies(self) -> Self:
        if self.policy_ids != tuple(policy.policy_id for policy in self.policies):
            raise ValueError("policy_ids must match policies")
        if self.directions != tuple(policy.direction for policy in self.policies):
            raise ValueError("directions must match policies")
        if self.directions != _REQUIRED_HYBRID_DIRECTIONS:
            raise ValueError("directions must cover required hybrid policies")
        if self.decision_count != len(self.policies):
            raise ValueError("decision_count must match policies")
        return self


class RoutingSafetyContract(BaseModel):
    """One V5.2 routing safety contract."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    contract_id: str = Field(min_length=1, max_length=140)
    safety_boundary: str = Field(min_length=1, max_length=120)
    contract_summary: str = Field(min_length=1, max_length=280)
    hitl_required: bool
    invariant_satisfied: Literal[True] = True
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    safety_contract_implemented: Literal[True] = True
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    automatic_api_key_assumption_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["routing_safety_contract.v1"] = (
        ROUTING_SAFETY_CONTRACT_SERIALIZATION_VERSION
    )
    metadata_only: Literal[True] = True


class RoutingSafetyContractRegistry(BaseModel):
    """Registry for V5.2 routing safety contracts."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["routing_safety_contract_registry"] = (
        "routing_safety_contract_registry"
    )
    serialization_version: Literal["routing_safety_contract_registry.v1"] = (
        ROUTING_SAFETY_CONTRACT_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=(
            "V5.2 routing safety contracts describe provider selection, "
            "credential, HITL, download, and automatic-routing boundaries for "
            "inspection only; they do not emit HITL requests, execute providers, "
            "switch providers or models, download models, assume API keys, or "
            "modify generated output."
        ),
        max_length=1400,
    )
    contracts: tuple[RoutingSafetyContract, ...] = Field(min_length=7, max_length=7)
    contract_ids: tuple[str, ...] = Field(min_length=7, max_length=7)
    safety_boundaries: tuple[str, ...] = Field(min_length=7, max_length=7)
    decision_count: int = Field(ge=7, le=7)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    safety_contracts_implemented: Literal[True] = True
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    automatic_api_key_assumption_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _registry_matches_contracts(self) -> Self:
        if self.contract_ids != tuple(
            contract.contract_id for contract in self.contracts
        ):
            raise ValueError("contract_ids must match contracts")
        if self.safety_boundaries != tuple(
            contract.safety_boundary for contract in self.contracts
        ):
            raise ValueError("safety_boundaries must match contracts")
        if self.safety_boundaries != _REQUIRED_SAFETY_BOUNDARIES:
            raise ValueError("safety boundaries must cover required contracts")
        if self.decision_count != len(self.contracts):
            raise ValueError("decision_count must match contracts")
        return self


class ModelRoutingIntelligenceRegistry(BaseModel):
    """Aggregate passive V5.2 model-routing intelligence registry."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["model_routing_intelligence_registry"] = (
        "model_routing_intelligence_registry"
    )
    serialization_version: Literal["model_routing_intelligence_registry.v1"] = (
        MODEL_ROUTING_INTELLIGENCE_REGISTRY_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=(
            "V5.2 model-routing intelligence metadata aggregates provider "
            "profiles, availability and credential boundaries, task-aware "
            "routing taxonomy, execution modes, hybrid routing policies, "
            "unavailable reasons, and safety contracts for inspection only; it "
            "does not apply routing, switch providers or models, execute "
            "providers, download models, assume API keys, emit HITL requests, "
            "mutate prompts, write storage, or modify generated output."
        ),
        max_length=1600,
    )
    provider_profile_serialization_version: str = Field(min_length=1, max_length=120)
    provider_availability_serialization_version: str = Field(
        min_length=1,
        max_length=120,
    )
    task_aware_routing_serialization_version: str = Field(min_length=1, max_length=120)
    execution_mode_serialization_version: str = Field(min_length=1, max_length=120)
    hybrid_policy_serialization_version: str = Field(min_length=1, max_length=120)
    safety_contract_serialization_version: str = Field(min_length=1, max_length=120)
    provider_ids: tuple[ProviderId, ...] = Field(min_length=4, max_length=4)
    task_types: tuple[TaskRoutingType, ...] = Field(min_length=12, max_length=12)
    execution_mode_ids: tuple[ExecutionModeId, ...] = Field(min_length=3, max_length=3)
    hybrid_policy_directions: tuple[HybridRoutingPolicyDirection, ...] = Field(
        min_length=4,
        max_length=4,
    )
    unavailable_reason_codes: tuple[UnavailableReasonCode, ...] = Field(
        min_length=9,
        max_length=9,
    )
    decision_count: int = Field(ge=39, le=80)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=16,
    )
    model_routing_intelligence_implemented: Literal[True] = True
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    automatic_provider_switching_implemented: Literal[False] = False
    automatic_model_download_implemented: Literal[False] = False
    automatic_api_key_assumption_implemented: Literal[False] = False
    hitl_request_emitted: Literal[False] = False
    budget_enforcement_implemented: Literal[False] = False
    workflow_control_implemented: Literal[False] = False
    retry_triggering_implemented: Literal[False] = False
    prompt_mutation_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    metadata_only: Literal[True] = True

    @model_validator(mode="after")
    def _aggregate_matches_required_surfaces(self) -> Self:
        if self.provider_ids != _REQUIRED_PROVIDER_IDS:
            raise ValueError("provider_ids must cover required providers")
        if self.task_types != _REQUIRED_TASK_TYPES:
            raise ValueError("task_types must cover required tasks")
        if self.execution_mode_ids != _REQUIRED_EXECUTION_MODES:
            raise ValueError("execution modes must cover required modes")
        if self.hybrid_policy_directions != _REQUIRED_HYBRID_DIRECTIONS:
            raise ValueError("hybrid policies must cover required directions")
        if self.unavailable_reason_codes != _REQUIRED_UNAVAILABLE_REASONS:
            raise ValueError("unavailable reasons must cover required reasons")
        return self


def routing_provider_profile_registry() -> RoutingProviderProfileRegistry:
    """Return explicit passive provider profiles."""

    return RoutingProviderProfileRegistry(
        provider_profiles=_provider_profiles(),
        provider_ids=_REQUIRED_PROVIDER_IDS,
        provider_count=4,
        extension_points=(
            "provider_profile_metadata",
            "provider_availability_metadata",
            "provider_capability_family_metadata",
            "routing_safety_contract_metadata",
        ),
    )


def routing_provider_profile_by_id(
    provider_id: ProviderId | str,
    registry: RoutingProviderProfileRegistry | None = None,
) -> RoutingProviderProfile | None:
    """Return one provider profile without selecting a provider."""

    normalized = str(provider_id).strip()
    source_registry = registry or routing_provider_profile_registry()
    for profile in source_registry.provider_profiles:
        if profile.provider_id == normalized:
            return profile
    return None


def provider_availability_registry() -> ProviderAvailabilityRegistry:
    """Return passive provider availability and credential metadata."""

    return ProviderAvailabilityRegistry(
        api_key_detection=_api_key_detection(),
        credential_boundaries=_credential_boundaries(),
        provider_availability=_provider_availability(),
        local_runtime_detection=_local_runtime_detection(),
        local_model_inventory=_local_model_inventory(),
        local_model_availability=_local_model_availability(),
        unavailable_reasons=_unavailable_reasons(),
        provider_ids=_REQUIRED_PROVIDER_IDS,
        unavailable_reason_codes=_REQUIRED_UNAVAILABLE_REASONS,
        decision_count=33,
    )


def provider_availability_by_provider_id(
    provider_id: ProviderId | str,
    registry: ProviderAvailabilityRegistry | None = None,
) -> ProviderAvailabilityMetadata | None:
    """Return provider availability metadata without checking the provider."""

    normalized = str(provider_id).strip()
    source_registry = registry or provider_availability_registry()
    for availability in source_registry.provider_availability:
        if availability.provider_id == normalized:
            return availability
    return None


def routing_unavailable_reason_by_code(
    reason_code: UnavailableReasonCode | str,
    registry: ProviderAvailabilityRegistry | None = None,
) -> RoutingUnavailableReason | None:
    """Return one unavailable reason model."""

    normalized = str(reason_code).strip()
    source_registry = registry or provider_availability_registry()
    for reason in source_registry.unavailable_reasons:
        if reason.reason_code == normalized:
            return reason
    return None


def credential_boundary_by_provider_id(
    provider_id: ProviderId | str,
    registry: ProviderAvailabilityRegistry | None = None,
) -> CredentialBoundary | None:
    """Return provider credential boundary metadata."""

    normalized = str(provider_id).strip()
    source_registry = registry or provider_availability_registry()
    for boundary in source_registry.credential_boundaries:
        if boundary.provider_id == normalized:
            return boundary
    return None


def routing_execution_mode_registry() -> RoutingExecutionModeRegistry:
    """Return passive routing execution mode metadata."""

    return RoutingExecutionModeRegistry(
        execution_modes=_execution_modes(),
        execution_mode_ids=_REQUIRED_EXECUTION_MODES,
        decision_count=3,
    )


def routing_execution_mode_by_id(
    execution_mode_id: ExecutionModeId | str,
    registry: RoutingExecutionModeRegistry | None = None,
) -> RoutingExecutionModeProfile | None:
    """Return one routing execution mode profile."""

    normalized = str(execution_mode_id).strip()
    source_registry = registry or routing_execution_mode_registry()
    for mode in source_registry.execution_modes:
        if mode.execution_mode_id == normalized:
            return mode
    return None


def task_aware_routing_registry(
    *,
    model_profiles: ModelProfileRegistry | None = None,
    execution_modes: RoutingExecutionModeRegistry | None = None,
) -> TaskAwareRoutingRegistry:
    """Return task-aware routing metadata without applying routing."""

    profile_registry = model_profiles or model_profile_registry()
    mode_registry = execution_modes or routing_execution_mode_registry()
    decisions = _task_routing_decisions()
    return TaskAwareRoutingRegistry(
        source_model_profile_serialization_version=profile_registry.serialization_version,
        source_execution_mode_serialization_version=mode_registry.serialization_version,
        decisions=decisions,
        decision_ids=tuple(decision.decision_id for decision in decisions),
        task_types=tuple(decision.task_type for decision in decisions),
        execution_mode_ids=mode_registry.execution_mode_ids,
        decision_count=len(decisions),
        hitl_required_decision_count=sum(
            1 for decision in decisions if decision.hitl_required
        ),
    )


def task_routing_decision_by_task_type(
    task_type: TaskRoutingType | str,
    registry: TaskAwareRoutingRegistry | None = None,
) -> TaskAwareRoutingDecision | None:
    """Return one task-aware routing decision."""

    normalized = str(task_type).strip()
    source_registry = registry or task_aware_routing_registry()
    for decision in source_registry.decisions:
        if decision.task_type == normalized:
            return decision
    return None


def task_routing_decisions_requiring_hitl(
    registry: TaskAwareRoutingRegistry | None = None,
) -> tuple[TaskAwareRoutingDecision, ...]:
    """Return advisory task decisions that require HITL before application."""

    source_registry = registry or task_aware_routing_registry()
    return tuple(
        decision for decision in source_registry.decisions if decision.hitl_required
    )


def advisory_hybrid_routing_policy_registry() -> AdvisoryHybridRoutingPolicyRegistry:
    """Return explicit advisory hybrid routing policies."""

    policies = _hybrid_policies()
    return AdvisoryHybridRoutingPolicyRegistry(
        policies=policies,
        policy_ids=tuple(policy.policy_id for policy in policies),
        directions=tuple(policy.direction for policy in policies),
        decision_count=len(policies),
    )


def advisory_hybrid_routing_policy_by_direction(
    direction: HybridRoutingPolicyDirection | str,
    registry: AdvisoryHybridRoutingPolicyRegistry | None = None,
) -> AdvisoryHybridRoutingPolicy | None:
    """Return one advisory hybrid routing policy by direction."""

    normalized = str(direction).strip()
    source_registry = registry or advisory_hybrid_routing_policy_registry()
    for policy in source_registry.policies:
        if policy.direction == normalized:
            return policy
    return None


def routing_safety_contract_registry() -> RoutingSafetyContractRegistry:
    """Return V5.2 routing safety contracts."""

    contracts = _safety_contracts()
    return RoutingSafetyContractRegistry(
        contracts=contracts,
        contract_ids=tuple(contract.contract_id for contract in contracts),
        safety_boundaries=tuple(contract.safety_boundary for contract in contracts),
        decision_count=len(contracts),
    )


def routing_safety_contract_by_boundary(
    safety_boundary: str,
    registry: RoutingSafetyContractRegistry | None = None,
) -> RoutingSafetyContract | None:
    """Return one routing safety contract by boundary id."""

    normalized = str(safety_boundary).strip()
    source_registry = registry or routing_safety_contract_registry()
    for contract in source_registry.contracts:
        if contract.safety_boundary == normalized:
            return contract
    return None


def model_routing_intelligence_registry() -> ModelRoutingIntelligenceRegistry:
    """Return aggregate passive V5.2 model-routing intelligence metadata."""

    provider_profiles = routing_provider_profile_registry()
    availability = provider_availability_registry()
    execution_modes = routing_execution_mode_registry()
    tasks = task_aware_routing_registry(execution_modes=execution_modes)
    hybrid_policies = advisory_hybrid_routing_policy_registry()
    safety = routing_safety_contract_registry()
    return ModelRoutingIntelligenceRegistry(
        provider_profile_serialization_version=provider_profiles.serialization_version,
        provider_availability_serialization_version=availability.serialization_version,
        task_aware_routing_serialization_version=tasks.serialization_version,
        execution_mode_serialization_version=execution_modes.serialization_version,
        hybrid_policy_serialization_version=hybrid_policies.serialization_version,
        safety_contract_serialization_version=safety.serialization_version,
        provider_ids=provider_profiles.provider_ids,
        task_types=tasks.task_types,
        execution_mode_ids=execution_modes.execution_mode_ids,
        hybrid_policy_directions=hybrid_policies.directions,
        unavailable_reason_codes=availability.unavailable_reason_codes,
        decision_count=(
            provider_profiles.provider_count
            + availability.decision_count
            + execution_modes.decision_count
            + tasks.decision_count
            + hybrid_policies.decision_count
            + safety.decision_count
        ),
    )


def _provider_profiles() -> tuple[RoutingProviderProfile, ...]:
    common_cloud_capabilities: tuple[RoutingCapabilityFamily, ...] = (
        "coding",
        "reasoning",
        "creative_coding",
        "creative_writing",
        "long_context_reasoning",
        "tool_use",
        "structured_output",
        "fast_draft",
        "maximum_quality_execution",
    )
    return (
        RoutingProviderProfile(
            provider_profile_id="provider_profile::openai",
            provider_id="openai",
            provider_family="openai",
            provider_category="cloud",
            credential_requirements=("api_key_required",),
            supported_model_families=("gpt", "embedding"),
            supported_capability_families=common_cloud_capabilities
            + ("multimodal_understanding", "image_understanding"),
            availability_check_policy="settings_metadata_only",
            routing_safety_notes=(
                "Requires an explicit configured API key before execution.",
                "Current runtime provider factory remains OpenAI-only.",
                "V5.2 metadata must not switch configured provider or model.",
            ),
        ),
        RoutingProviderProfile(
            provider_profile_id="provider_profile::anthropic",
            provider_id="anthropic",
            provider_family="anthropic",
            provider_category="cloud",
            credential_requirements=("api_key_required",),
            supported_model_families=("claude",),
            supported_capability_families=common_cloud_capabilities,
            availability_check_policy="environment_metadata_only",
            routing_safety_notes=(
                "Profile is advisory; no Anthropic runtime adapter is enabled.",
                "Recommending this provider for use requires HITL and adapter work.",
                "No API key is assumed by V5.2 metadata.",
            ),
        ),
        RoutingProviderProfile(
            provider_profile_id="provider_profile::gemini",
            provider_id="gemini",
            provider_family="gemini",
            provider_category="cloud",
            credential_requirements=("api_key_required",),
            supported_model_families=("gemini",),
            supported_capability_families=common_cloud_capabilities
            + ("multimodal_understanding", "image_understanding"),
            availability_check_policy="environment_metadata_only",
            routing_safety_notes=(
                "Profile is advisory; no Gemini runtime adapter is enabled.",
                "Multimodal capability is catalog metadata only.",
                "No provider selection or API key assumption is performed.",
            ),
        ),
        RoutingProviderProfile(
            provider_profile_id="provider_profile::local",
            provider_id="local",
            provider_family="local",
            provider_category="local",
            credential_requirements=("none_required", "user_managed_local_runtime"),
            supported_model_families=("local_chat", "local_code", "local_multimodal"),
            supported_capability_families=(
                "coding",
                "reasoning",
                "creative_coding",
                "multimodal_understanding",
                "image_understanding",
                "fast_draft",
                "low_cost_execution",
            ),
            availability_check_policy="local_runtime_metadata_only",
            routing_safety_notes=(
                "Local runtimes are user-managed and not started by V5.2.",
                "Local model inventory is static metadata only.",
                "No model download or local probing is performed.",
            ),
        ),
    )


def _api_key_detection() -> tuple[ApiKeyDetectionMetadata, ...]:
    return (
        ApiKeyDetectionMetadata(
            provider_id="openai",
            credential_names=("OPENAI_API_KEY", "CCA_OPENAI_API_KEY"),
            detection_policy="settings_metadata_only",
        ),
        ApiKeyDetectionMetadata(
            provider_id="anthropic",
            credential_names=("ANTHROPIC_API_KEY", "CCA_ANTHROPIC_API_KEY"),
            detection_policy="environment_metadata_only",
        ),
        ApiKeyDetectionMetadata(
            provider_id="gemini",
            credential_names=("GEMINI_API_KEY", "GOOGLE_API_KEY", "CCA_GEMINI_API_KEY"),
            detection_policy="environment_metadata_only",
        ),
        ApiKeyDetectionMetadata(
            provider_id="local",
            credential_names=(),
            detection_policy="local_runtime_metadata_only",
        ),
    )


def _credential_boundaries() -> tuple[CredentialBoundary, ...]:
    return (
        CredentialBoundary(
            provider_id="openai",
            credential_required=True,
            boundary_notes=(
                "Use settings metadata only for key presence.",
                "Never expose credential values in routing metadata.",
            ),
        ),
        CredentialBoundary(
            provider_id="anthropic",
            credential_required=True,
            boundary_notes=(
                "Credential support is advisory because provider adapter is absent.",
                "HITL is required before recommending setup work.",
            ),
        ),
        CredentialBoundary(
            provider_id="gemini",
            credential_required=True,
            boundary_notes=(
                "Credential support is advisory because provider adapter is absent.",
                "HITL is required before recommending setup work.",
            ),
        ),
        CredentialBoundary(
            provider_id="local",
            credential_required=False,
            boundary_notes=(
                "Local execution depends on user-managed runtime state.",
                "No API key may be assumed for local routing metadata.",
            ),
        ),
    )


def _provider_availability() -> tuple[ProviderAvailabilityMetadata, ...]:
    return (
        ProviderAvailabilityMetadata(
            provider_id="openai",
            provider_profile_id="provider_profile::openai",
            availability_status="requires_hitl",
            availability_check_policy="settings_metadata_only",
            unavailable_reason_codes=("missing_api_key", "hitl_required"),
            hitl_required_before_recommendation=True,
        ),
        ProviderAvailabilityMetadata(
            provider_id="anthropic",
            provider_profile_id="provider_profile::anthropic",
            availability_status="unavailable",
            availability_check_policy="environment_metadata_only",
            unavailable_reason_codes=(
                "provider_unsupported",
                "missing_api_key",
                "hitl_required",
            ),
            hitl_required_before_recommendation=True,
        ),
        ProviderAvailabilityMetadata(
            provider_id="gemini",
            provider_profile_id="provider_profile::gemini",
            availability_status="unavailable",
            availability_check_policy="environment_metadata_only",
            unavailable_reason_codes=(
                "provider_unsupported",
                "missing_api_key",
                "hitl_required",
            ),
            hitl_required_before_recommendation=True,
        ),
        ProviderAvailabilityMetadata(
            provider_id="local",
            provider_profile_id="provider_profile::local",
            availability_status="unknown_metadata_only",
            availability_check_policy="local_runtime_metadata_only",
            unavailable_reason_codes=(
                "local_runtime_unavailable",
                "local_model_not_installed",
                "insufficient_local_resources",
                "hitl_required",
            ),
            hitl_required_before_recommendation=True,
        ),
    )


def _local_runtime_detection() -> tuple[LocalRuntimeDetectionMetadata, ...]:
    return (
        LocalRuntimeDetectionMetadata(
            runtime_kind="ollama",
            runtime_name="Ollama",
        ),
        LocalRuntimeDetectionMetadata(
            runtime_kind="lm_studio",
            runtime_name="LM Studio",
        ),
        LocalRuntimeDetectionMetadata(
            runtime_kind="llama_cpp",
            runtime_name="llama.cpp",
        ),
        LocalRuntimeDetectionMetadata(
            runtime_kind="local_transformers",
            runtime_name="Local Transformers",
        ),
    )


def _local_model_inventory() -> tuple[LocalModelInventoryMetadata, ...]:
    return (
        LocalModelInventoryMetadata(
            inventory_id="local_model_inventory::ollama",
            runtime_kind="ollama",
            candidate_model_families=("local_chat",),
            candidate_model_labels=("ollama chat model",),
        ),
        LocalModelInventoryMetadata(
            inventory_id="local_model_inventory::lm_studio",
            runtime_kind="lm_studio",
            candidate_model_families=("local_chat",),
            candidate_model_labels=("LM Studio chat model",),
        ),
        LocalModelInventoryMetadata(
            inventory_id="local_model_inventory::llama_cpp",
            runtime_kind="llama_cpp",
            candidate_model_families=("local_code",),
            candidate_model_labels=("llama.cpp code/completion model",),
        ),
        LocalModelInventoryMetadata(
            inventory_id="local_model_inventory::local_transformers",
            runtime_kind="local_transformers",
            candidate_model_families=("local_multimodal",),
            candidate_model_labels=("local transformers multimodal checkpoint",),
        ),
    )


def _local_model_availability() -> tuple[LocalModelAvailabilityMetadata, ...]:
    common_reasons: tuple[UnavailableReasonCode, ...] = (
        "local_runtime_unavailable",
        "local_model_not_installed",
        "insufficient_local_resources",
    )
    return (
        LocalModelAvailabilityMetadata(
            availability_id="local_model_availability::ollama",
            runtime_kind="ollama",
            unavailable_reason_codes=common_reasons,
        ),
        LocalModelAvailabilityMetadata(
            availability_id="local_model_availability::lm_studio",
            runtime_kind="lm_studio",
            unavailable_reason_codes=common_reasons,
        ),
        LocalModelAvailabilityMetadata(
            availability_id="local_model_availability::llama_cpp",
            runtime_kind="llama_cpp",
            unavailable_reason_codes=common_reasons,
        ),
        LocalModelAvailabilityMetadata(
            availability_id="local_model_availability::local_transformers",
            runtime_kind="local_transformers",
            unavailable_reason_codes=common_reasons + ("missing_modality_support",),
        ),
    )


def _unavailable_reasons() -> tuple[RoutingUnavailableReason, ...]:
    return (
        RoutingUnavailableReason(
            reason_code="missing_api_key",
            reason_summary=(
                "Required provider API key metadata is absent or unverified."
            ),
            user_action_required=True,
            hitl_required=True,
            blocks_auto_mode=True,
        ),
        RoutingUnavailableReason(
            reason_code="provider_unsupported",
            reason_summary="Provider profile exists but no runtime adapter is enabled.",
            user_action_required=True,
            hitl_required=True,
            blocks_auto_mode=True,
        ),
        RoutingUnavailableReason(
            reason_code="local_runtime_unavailable",
            reason_summary="Local runtime readiness is unknown or unavailable.",
            user_action_required=True,
            hitl_required=True,
            blocks_auto_mode=True,
        ),
        RoutingUnavailableReason(
            reason_code="local_model_not_installed",
            reason_summary="Local model inventory does not confirm installation.",
            user_action_required=True,
            hitl_required=True,
            blocks_auto_mode=True,
        ),
        RoutingUnavailableReason(
            reason_code="insufficient_local_resources",
            reason_summary="Local hardware fit has not been verified.",
            user_action_required=True,
            hitl_required=True,
            blocks_auto_mode=True,
        ),
        RoutingUnavailableReason(
            reason_code="missing_modality_support",
            reason_summary=(
                "Required input or output modality is not confirmed available."
            ),
            user_action_required=True,
            hitl_required=True,
            blocks_auto_mode=True,
        ),
        RoutingUnavailableReason(
            reason_code="cost_policy_blocked",
            reason_summary="Relative cost policy requires review before routing use.",
            user_action_required=False,
            hitl_required=True,
            blocks_auto_mode=True,
        ),
        RoutingUnavailableReason(
            reason_code="latency_policy_blocked",
            reason_summary="Latency posture requires review before routing use.",
            user_action_required=False,
            hitl_required=True,
            blocks_auto_mode=True,
        ),
        RoutingUnavailableReason(
            reason_code="hitl_required",
            reason_summary=(
                "Human review is required before the recommendation can be used."
            ),
            user_action_required=True,
            hitl_required=True,
            blocks_auto_mode=True,
        ),
    )


def _execution_modes() -> tuple[RoutingExecutionModeProfile, ...]:
    all_risky = _REQUIRED_UNAVAILABLE_REASONS
    return (
        RoutingExecutionModeProfile(
            execution_mode_id="manual_mode",
            mode_name="Manual Mode",
            routing_authority="User chooses provider/model; router only advises.",
            confirmation_policy="Every route remains user-selected before execution.",
            hitl_required_reason_codes=all_risky,
            safe_auto_boundary=False,
        ),
        RoutingExecutionModeProfile(
            execution_mode_id="assisted_mode",
            mode_name="Assisted Mode",
            routing_authority=(
                "Router recommends provider/model metadata; user confirms."
            ),
            confirmation_policy=(
                "Confirmation is required before any future application."
            ),
            hitl_required_reason_codes=all_risky,
            safe_auto_boundary=False,
        ),
        RoutingExecutionModeProfile(
            execution_mode_id="auto_mode",
            mode_name="Auto Mode",
            routing_authority=(
                "Router may select only within explicitly allowed safe metadata "
                "boundaries; V5.2 does not apply the selection."
            ),
            confirmation_policy=(
                "Unavailable, expensive, credential-requiring, download-requiring, "
                "or high-risk decisions require HITL."
            ),
            hitl_required_reason_codes=all_risky,
            safe_auto_boundary=True,
        ),
    )


def _task_routing_decisions() -> tuple[TaskAwareRoutingDecision, ...]:
    path = (
        "task_type",
        "capability_requirements",
        "available_models",
        "recommended_model",
        "fallback_model",
        "execution_mode",
    )

    def decision(
        task_type: TaskRoutingType,
        requirements: tuple[RoutingCapabilityFamily, ...],
        available: tuple[str, ...],
        recommended: str,
        fallback: str,
        mode: ExecutionModeId,
        risk: RoutingRiskBand,
        quality: EstimatedQualityBand,
        cost: EstimatedCostBand,
        latency: EstimatedLatencyBand,
        confidence: float,
        unavailable: tuple[UnavailableReasonCode, ...] = (),
    ) -> TaskAwareRoutingDecision:
        hitl_required = bool(unavailable) or risk == "high"
        return TaskAwareRoutingDecision(
            decision_id=f"task_route::{task_type}",
            task_type=task_type,
            capability_requirements=requirements,
            available_model_profile_ids=available,
            recommended_model_profile_id=recommended,
            fallback_model_profile_id=fallback,
            execution_mode_id=mode,
            available_route_summary=(
                "Available route candidates are derived from static model profile "
                "metadata."
            ),
            recommended_route_summary=f"Recommend {recommended} for {task_type}.",
            fallback_route_summary=f"Fallback to {fallback} if HITL approves.",
            unavailable_route_reason_summary=(
                "No unavailable reasons recorded."
                if not unavailable
                else "Unavailable reasons require HITL before use."
            ),
            estimated_quality=quality,
            estimated_cost=cost,
            estimated_latency=latency,
            confidence_score=confidence,
            unavailable_reason_codes=unavailable,
            hitl_required=hitl_required,
            risk_band=risk,
            routing_path=path,
            evidence=(
                f"Task type {task_type} maps to capability requirements.",
                f"Recommended model profile is {recommended}.",
                f"Fallback model profile is {fallback}.",
            ),
        )

    fast = "fast_iteration_model_profile"
    creative = "creative_reasoning_model_profile"
    code = "code_assistance_model_profile"
    review = "evaluation_review_model_profile"
    return (
        decision(
            "coding",
            ("coding", "tool_use", "structured_output"),
            (code, fast),
            code,
            fast,
            "assisted_mode",
            "medium",
            "high",
            "medium",
            "moderate",
            0.78,
        ),
        decision(
            "reasoning",
            ("reasoning", "long_context_reasoning"),
            (creative, review),
            creative,
            review,
            "assisted_mode",
            "medium",
            "maximum",
            "high",
            "moderate",
            0.76,
        ),
        decision(
            "creative_coding",
            ("creative_coding", "coding", "reasoning"),
            (creative, code),
            creative,
            code,
            "assisted_mode",
            "medium",
            "maximum",
            "medium",
            "moderate",
            0.8,
        ),
        decision(
            "creative_writing",
            ("creative_writing", "reasoning"),
            (creative, fast),
            creative,
            fast,
            "manual_mode",
            "low",
            "high",
            "low",
            "fast",
            0.72,
        ),
        decision(
            "long_context_reasoning",
            ("long_context_reasoning", "reasoning"),
            (creative, review),
            creative,
            review,
            "assisted_mode",
            "high",
            "maximum",
            "high",
            "slow",
            0.68,
            ("latency_policy_blocked", "hitl_required"),
        ),
        decision(
            "multimodal_understanding",
            ("multimodal_understanding", "image_understanding"),
            (review, creative),
            review,
            creative,
            "manual_mode",
            "high",
            "high",
            "high",
            "slow",
            0.64,
            ("missing_modality_support", "hitl_required"),
        ),
        decision(
            "image_understanding",
            ("image_understanding", "multimodal_understanding"),
            (review, creative),
            review,
            creative,
            "manual_mode",
            "high",
            "high",
            "high",
            "slow",
            0.64,
            ("missing_modality_support", "hitl_required"),
        ),
        decision(
            "tool_use",
            ("tool_use", "coding"),
            (code, fast),
            code,
            fast,
            "assisted_mode",
            "medium",
            "high",
            "medium",
            "moderate",
            0.77,
        ),
        decision(
            "structured_output",
            ("structured_output", "reasoning"),
            (review, code),
            review,
            code,
            "assisted_mode",
            "medium",
            "high",
            "medium",
            "moderate",
            0.75,
        ),
        decision(
            "fast_draft",
            ("fast_draft", "low_cost_execution"),
            (fast, code),
            fast,
            code,
            "auto_mode",
            "low",
            "medium",
            "low",
            "fast",
            0.82,
        ),
        decision(
            "low_cost_execution",
            ("low_cost_execution", "fast_draft"),
            (fast, code),
            fast,
            code,
            "assisted_mode",
            "medium",
            "medium",
            "low",
            "fast",
            0.66,
            ("local_runtime_unavailable", "hitl_required"),
        ),
        decision(
            "maximum_quality_execution",
            ("maximum_quality_execution", "reasoning", "structured_output"),
            (review, creative),
            review,
            creative,
            "assisted_mode",
            "high",
            "maximum",
            "high",
            "slow",
            0.7,
            ("cost_policy_blocked", "hitl_required"),
        ),
    )


def _hybrid_policies() -> tuple[AdvisoryHybridRoutingPolicy, ...]:
    return (
        AdvisoryHybridRoutingPolicy(
            policy_id="hybrid_policy::local_to_cloud",
            direction="local_to_cloud",
            intended_use_case=(
                "Start with local low-cost drafting, escalate to cloud quality."
            ),
            fallback_logic=(
                "If local readiness is unverified, require HITL before cloud fallback."
            ),
            availability_constraints=(
                "local runtime metadata must be user confirmed",
                "cloud credential metadata must be user confirmed",
            ),
            cost_quality_latency_tradeoff=(
                "Lower local cost and privacy first; higher cloud quality may add "
                "latency and credential requirements."
            ),
            hitl_requirements=(
                "missing local runtime requires HITL",
                "cloud credential requirement requires HITL",
            ),
            safety_constraints=(
                "no automatic provider switching",
                "no automatic model download",
                "no API key assumption",
            ),
            unavailable_reason_codes=(
                "local_runtime_unavailable",
                "missing_api_key",
                "hitl_required",
            ),
        ),
        AdvisoryHybridRoutingPolicy(
            policy_id="hybrid_policy::cloud_to_local",
            direction="cloud_to_local",
            intended_use_case=(
                "Use cloud quality metadata first, keep local fallback visible."
            ),
            fallback_logic=(
                "If cloud credentials are unavailable, require HITL before local "
                "fallback."
            ),
            availability_constraints=(
                "cloud provider must be supported and credentialed",
                "local runtime inventory must be user confirmed",
            ),
            cost_quality_latency_tradeoff=(
                "Higher cloud quality may cost more; local fallback can reduce cost "
                "but has resource uncertainty."
            ),
            hitl_requirements=(
                "missing API key requires HITL",
                "local model installation uncertainty requires HITL",
            ),
            safety_constraints=(
                "no automatic provider switching",
                "no automatic local probing",
                "no generated output mutation",
            ),
            unavailable_reason_codes=(
                "missing_api_key",
                "local_model_not_installed",
                "hitl_required",
            ),
        ),
        AdvisoryHybridRoutingPolicy(
            policy_id="hybrid_policy::cloud_to_cloud",
            direction="cloud_to_cloud",
            intended_use_case=(
                "Compare cloud profile families for quality, modality, or cost."
            ),
            fallback_logic=(
                "Unsupported cloud providers remain unavailable until HITL "
                "approves setup."
            ),
            availability_constraints=(
                "provider adapter must exist before execution",
                "provider credential metadata must be user confirmed",
            ),
            cost_quality_latency_tradeoff=(
                "Quality and modality may improve across providers; cost and "
                "latency require review."
            ),
            hitl_requirements=(
                "unsupported provider requires HITL",
                "expensive or high-risk route requires HITL",
            ),
            safety_constraints=(
                "no automatic cloud provider switching",
                "no API key assumption",
                "no provider execution",
            ),
            unavailable_reason_codes=(
                "provider_unsupported",
                "missing_api_key",
                "cost_policy_blocked",
                "hitl_required",
            ),
        ),
        AdvisoryHybridRoutingPolicy(
            policy_id="hybrid_policy::local_to_local",
            direction="local_to_local",
            intended_use_case=(
                "Compare local runtime/model families for privacy and cost."
            ),
            fallback_logic=(
                "Uninstalled or unavailable local models require HITL; no "
                "download occurs."
            ),
            availability_constraints=(
                "runtime readiness must be user confirmed",
                "model installation must be user confirmed",
                "hardware fit must be user confirmed",
            ),
            cost_quality_latency_tradeoff=(
                "Local cost is hardware-only, quality varies by installed model, "
                "latency depends on local resources."
            ),
            hitl_requirements=(
                "missing runtime requires HITL",
                "missing model requires HITL",
                "resource uncertainty requires HITL",
            ),
            safety_constraints=(
                "no automatic model download",
                "no automatic local runtime start",
                "no provider output merging",
            ),
            unavailable_reason_codes=(
                "local_runtime_unavailable",
                "local_model_not_installed",
                "insufficient_local_resources",
                "hitl_required",
            ),
        ),
    )


def _safety_contracts() -> tuple[RoutingSafetyContract, ...]:
    def contract(
        boundary: str,
        summary: str,
        hitl_required: bool,
    ) -> RoutingSafetyContract:
        return RoutingSafetyContract(
            contract_id=f"routing_safety::{boundary}",
            safety_boundary=boundary,
            contract_summary=summary,
            hitl_required=hitl_required,
        )

    return (
        contract(
            "no_automatic_provider_switching",
            "Routing metadata must not change configured provider selection.",
            False,
        ),
        contract(
            "no_automatic_model_download",
            "Routing metadata must not download or install local models.",
            False,
        ),
        contract(
            "no_automatic_api_key_assumption",
            "Routing metadata must not infer or assume provider credentials.",
            False,
        ),
        contract(
            "hitl_before_unavailable_provider_or_model",
            "Unavailable provider/model recommendations require human approval.",
            True,
        ),
        contract(
            "hitl_before_expensive_or_high_risk_auto_route",
            "Expensive or high-risk automatic routes require human approval.",
            True,
        ),
        contract(
            "provider_selection_boundary",
            (
                "Provider selection remains advisory until an explicit future "
                "runtime integration."
            ),
            False,
        ),
        contract(
            "credential_boundary",
            "Credential values remain outside routing metadata and are never exposed.",
            False,
        ),
    )
