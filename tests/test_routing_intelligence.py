import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    UnsupportedGenerationProviderError,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    TaskAwareRoutingDecision,
    advisory_hybrid_routing_policy_by_direction,
    advisory_hybrid_routing_policy_registry,
    credential_boundary_by_provider_id,
    model_routing_intelligence_registry,
    provider_availability_by_provider_id,
    provider_availability_registry,
    route_model_request,
    route_request,
    routing_execution_mode_by_id,
    routing_execution_mode_registry,
    routing_provider_profile_by_id,
    routing_provider_profile_registry,
    routing_safety_contract_by_boundary,
    routing_safety_contract_registry,
    routing_unavailable_reason_by_code,
    task_aware_routing_registry,
    task_routing_decision_by_task_type,
    task_routing_decisions_requiring_hitl,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_PROVIDER_IDS = ("openai", "anthropic", "gemini", "local")
REQUIRED_TASK_TYPES = (
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
REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_HYBRID_DIRECTIONS = (
    "local_to_cloud",
    "cloud_to_local",
    "cloud_to_cloud",
    "local_to_local",
)
REQUIRED_UNAVAILABLE_REASONS = (
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
REQUIRED_SAFETY_BOUNDARIES = (
    "no_automatic_provider_switching",
    "no_automatic_model_download",
    "no_automatic_api_key_assumption",
    "hitl_before_unavailable_provider_or_model",
    "hitl_before_expensive_or_high_risk_auto_route",
    "provider_selection_boundary",
    "credential_boundary",
)


class RoutingIntelligenceTests(unittest.TestCase):
    def test_provider_profiles_cover_required_providers_and_extensibility(self) -> None:
        registry = routing_provider_profile_registry()

        self.assertEqual(registry.provider_ids, REQUIRED_PROVIDER_IDS)
        self.assertEqual(registry.provider_count, 4)
        self.assertTrue(registry.provider_profiles_implemented)
        self.assertTrue(registry.extensible_provider_architecture_implemented)
        self.assertTrue(registry.new_provider_requires_explicit_profile)
        self.assertTrue(registry.runtime_adapter_required_for_execution)
        self.assertEqual(
            registry.extension_points,
            (
                "provider_profile_metadata",
                "provider_availability_metadata",
                "provider_capability_family_metadata",
                "routing_safety_contract_metadata",
            ),
        )
        self.assertFalse(registry.automatic_provider_switching_implemented)
        self.assertFalse(registry.automatic_model_download_implemented)
        self.assertFalse(registry.automatic_api_key_assumption_implemented)

        profile_by_id = {
            profile.provider_id: profile for profile in registry.provider_profiles
        }
        self.assertEqual(set(profile_by_id), set(REQUIRED_PROVIDER_IDS))
        self.assertEqual(profile_by_id["openai"].provider_category, "cloud")
        self.assertEqual(profile_by_id["anthropic"].provider_family, "anthropic")
        self.assertEqual(profile_by_id["gemini"].supported_model_families, ("gemini",))
        self.assertEqual(profile_by_id["local"].provider_category, "local")
        self.assertIn(
            "api_key_required",
            profile_by_id["openai"].credential_requirements,
        )
        self.assertIn(
            "user_managed_local_runtime",
            profile_by_id["local"].credential_requirements,
        )
        self.assertIn(
            "coding",
            profile_by_id["anthropic"].supported_capability_families,
        )
        self.assertIn(
            "image_understanding",
            profile_by_id["gemini"].supported_capability_families,
        )
        self.assertTrue(profile_by_id["local"].routing_safety_notes)
        self.assertIs(
            routing_provider_profile_by_id("openai", registry),
            profile_by_id["openai"],
        )

    def test_provider_availability_and_local_inventory_are_metadata_only(self) -> None:
        registry = provider_availability_registry()

        self.assertEqual(registry.provider_ids, REQUIRED_PROVIDER_IDS)
        self.assertEqual(
            registry.unavailable_reason_codes,
            REQUIRED_UNAVAILABLE_REASONS,
        )
        self.assertEqual(registry.decision_count, 33)
        self.assertTrue(registry.provider_availability_metadata_implemented)
        self.assertFalse(registry.provider_availability_detection_implemented)
        self.assertFalse(registry.local_runtime_detection_implemented)
        self.assertFalse(registry.local_model_discovery_implemented)
        self.assertFalse(registry.local_model_download_implemented)
        self.assertFalse(registry.automatic_api_key_assumption_implemented)

        for item in registry.api_key_detection:
            self.assertFalse(item.detection_performed)
            self.assertFalse(item.credential_value_exposed)
            self.assertEqual(item.detection_result, "not_checked_metadata_only")

        for availability in registry.provider_availability:
            self.assertFalse(availability.real_network_check_performed)
            self.assertFalse(availability.provider_call_performed)
            if availability.unavailable_reason_codes:
                self.assertTrue(availability.hitl_required_before_recommendation)

        for runtime in registry.local_runtime_detection:
            self.assertFalse(runtime.runtime_probe_performed)
            self.assertFalse(runtime.runtime_start_attempted)

        for inventory in registry.local_model_inventory:
            self.assertEqual(inventory.inventory_policy, "static_metadata_only")
            self.assertFalse(inventory.model_listing_performed)
            self.assertFalse(inventory.model_download_attempted)

        for availability in registry.local_model_availability:
            self.assertFalse(availability.model_probe_performed)
            self.assertFalse(availability.model_download_attempted)
            self.assertTrue(availability.hitl_required_before_recommendation)

        self.assertIsNotNone(provider_availability_by_provider_id("local", registry))
        self.assertIsNotNone(credential_boundary_by_provider_id("openai", registry))

    def test_unavailable_reasons_are_explicit_and_serializable(self) -> None:
        registry = provider_availability_registry()
        reason = routing_unavailable_reason_by_code("missing_api_key", registry)

        self.assertIsNotNone(reason)
        assert reason is not None
        self.assertTrue(reason.user_action_required)
        self.assertTrue(reason.hitl_required)
        self.assertTrue(reason.blocks_auto_mode)
        dumped = reason.model_dump(mode="json")
        self.assertEqual(dumped["reason_code"], "missing_api_key")
        self.assertIn("serialization_version", dumped)

        for reason in registry.unavailable_reasons:
            self.assertIn(reason.reason_code, REQUIRED_UNAVAILABLE_REASONS)
            self.assertTrue(reason.model_dump(mode="json"))

    def test_task_aware_routing_taxonomy_maps_required_route_shape(self) -> None:
        registry = task_aware_routing_registry()
        expected_path = (
            "task_type",
            "capability_requirements",
            "available_models",
            "recommended_model",
            "fallback_model",
            "execution_mode",
        )

        self.assertEqual(registry.task_types, REQUIRED_TASK_TYPES)
        self.assertEqual(registry.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(registry.decision_count, 12)
        self.assertGreaterEqual(registry.hitl_required_decision_count, 1)
        for decision in registry.decisions:
            self.assertEqual(decision.routing_path, expected_path)
            self.assertIn(
                decision.recommended_model_profile_id,
                decision.available_model_profile_ids,
            )
            self.assertIn(
                decision.fallback_model_profile_id,
                decision.available_model_profile_ids,
            )
            self.assertIn(decision.execution_mode_id, REQUIRED_EXECUTION_MODES)
            self.assertTrue(decision.available_route_summary)
            self.assertTrue(decision.recommended_route_summary)
            self.assertTrue(decision.fallback_route_summary)
            self.assertTrue(decision.unavailable_route_reason_summary)
            self.assertIn(
                decision.estimated_quality,
                {"low", "medium", "high", "maximum"},
            )
            self.assertIn(decision.estimated_cost, {"low", "medium", "high"})
            self.assertIn(decision.estimated_latency, {"fast", "moderate", "slow"})
            self.assertGreaterEqual(decision.confidence_score, 0.0)
            self.assertLessEqual(decision.confidence_score, 1.0)
            self.assertFalse(decision.task_routing_application_implemented)
            self.assertFalse(decision.automatic_provider_switching_implemented)

        creative = task_routing_decision_by_task_type("creative_coding", registry)
        self.assertIsNotNone(creative)
        assert creative is not None
        self.assertIn("creative_coding", creative.capability_requirements)

    def test_execution_modes_enforce_boundaries(self) -> None:
        registry = routing_execution_mode_registry()

        self.assertEqual(registry.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        manual = routing_execution_mode_by_id("manual_mode", registry)
        assisted = routing_execution_mode_by_id("assisted_mode", registry)
        auto = routing_execution_mode_by_id("auto_mode", registry)
        self.assertIsNotNone(manual)
        self.assertIsNotNone(assisted)
        self.assertIsNotNone(auto)
        assert manual is not None
        assert assisted is not None
        assert auto is not None
        self.assertIn("User chooses", manual.routing_authority)
        self.assertIn("user confirms", assisted.routing_authority)
        self.assertTrue(auto.safe_auto_boundary)
        self.assertIn("missing_api_key", auto.hitl_required_reason_codes)
        self.assertIn("local_model_not_installed", auto.hitl_required_reason_codes)
        self.assertIn("cost_policy_blocked", auto.hitl_required_reason_codes)
        self.assertFalse(registry.execution_mode_application_implemented)
        self.assertFalse(registry.automatic_provider_switching_implemented)
        self.assertFalse(registry.automatic_model_download_implemented)
        self.assertFalse(registry.automatic_api_key_assumption_implemented)

        fast_draft = task_routing_decision_by_task_type("fast_draft")
        assert fast_draft is not None
        payload = fast_draft.model_dump(mode="json")
        payload["risk_band"] = "high"
        payload["hitl_required"] = False
        with self.assertRaisesRegex(
            ValueError,
            "risky auto-mode decisions require HITL",
        ):
            TaskAwareRoutingDecision(**payload)

    def test_hybrid_routing_policies_cover_all_required_directions(self) -> None:
        registry = advisory_hybrid_routing_policy_registry()

        self.assertEqual(registry.directions, REQUIRED_HYBRID_DIRECTIONS)
        self.assertEqual(registry.decision_count, 4)
        self.assertFalse(registry.hybrid_routing_application_implemented)
        self.assertFalse(registry.automatic_provider_switching_implemented)
        self.assertFalse(registry.automatic_model_download_implemented)
        for policy in registry.policies:
            self.assertTrue(policy.intended_use_case)
            self.assertTrue(policy.fallback_logic)
            self.assertTrue(policy.availability_constraints)
            self.assertTrue(policy.cost_quality_latency_tradeoff)
            self.assertTrue(policy.hitl_requirements)
            self.assertTrue(policy.safety_constraints)
            self.assertIn("hitl_required", policy.unavailable_reason_codes)
            self.assertFalse(policy.provider_execution_implemented)
            self.assertFalse(policy.generated_output_mutation_implemented)

        self.assertIs(
            advisory_hybrid_routing_policy_by_direction("cloud_to_cloud", registry),
            registry.policies[2],
        )

    def test_safety_contracts_require_hitl_for_risky_routes(self) -> None:
        safety = routing_safety_contract_registry()
        availability = provider_availability_registry()
        tasks = task_aware_routing_registry()

        self.assertEqual(safety.safety_boundaries, REQUIRED_SAFETY_BOUNDARIES)
        self.assertFalse(safety.automatic_provider_switching_implemented)
        self.assertFalse(safety.automatic_model_download_implemented)
        self.assertFalse(safety.automatic_api_key_assumption_implemented)
        self.assertFalse(safety.hitl_request_emitted)
        unavailable_contract = routing_safety_contract_by_boundary(
            "hitl_before_unavailable_provider_or_model",
            safety,
        )
        high_risk_contract = routing_safety_contract_by_boundary(
            "hitl_before_expensive_or_high_risk_auto_route",
            safety,
        )
        self.assertIsNotNone(unavailable_contract)
        self.assertIsNotNone(high_risk_contract)
        assert unavailable_contract is not None
        assert high_risk_contract is not None
        self.assertTrue(unavailable_contract.hitl_required)
        self.assertTrue(high_risk_contract.hitl_required)

        for provider in availability.provider_availability:
            if provider.availability_status in {"unavailable", "requires_hitl"}:
                self.assertTrue(provider.hitl_required_before_recommendation)

        for decision in tasks.decisions:
            if decision.unavailable_reason_codes or decision.risk_band == "high":
                self.assertTrue(decision.hitl_required)

        hitl_tasks = task_routing_decisions_requiring_hitl(tasks)
        self.assertIn(
            "maximum_quality_execution",
            tuple(decision.task_type for decision in hitl_tasks),
        )

    def test_aggregate_registry_preserves_passive_safety_flags(self) -> None:
        registry = model_routing_intelligence_registry()

        self.assertEqual(registry.provider_ids, REQUIRED_PROVIDER_IDS)
        self.assertEqual(registry.task_types, REQUIRED_TASK_TYPES)
        self.assertEqual(registry.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(registry.hybrid_policy_directions, REQUIRED_HYBRID_DIRECTIONS)
        self.assertEqual(
            registry.unavailable_reason_codes,
            REQUIRED_UNAVAILABLE_REASONS,
        )
        self.assertEqual(registry.decision_count, 63)
        self.assertTrue(registry.model_routing_intelligence_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.provider_execution_implemented)
        self.assertFalse(registry.automatic_provider_switching_implemented)
        self.assertFalse(registry.automatic_model_download_implemented)
        self.assertFalse(registry.automatic_api_key_assumption_implemented)
        self.assertFalse(registry.hitl_request_emitted)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.prompt_mutation_implemented)
        self.assertFalse(registry.persistent_storage_write_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)

    def test_existing_provider_factory_and_workflow_routing_are_unchanged(self) -> None:
        request = AssistantRequest(
            query="Generate passive routing intelligence metadata.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )
        unsupported_settings = settings.model_copy(
            update={"default_generation_provider": "anthropic"}
        )

        model_routing_intelligence_registry()
        routing_provider_profile_registry()
        provider_availability_registry()
        task_aware_routing_registry()
        next_decision = route_request(request)
        provider = build_generation_provider(settings)
        plan = route_model_request(route=RouteName.GENERATE)

        self.assertEqual(next_decision, baseline_decision)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")
        with self.assertRaisesRegex(
            UnsupportedGenerationProviderError,
            "Unsupported generation provider: anthropic",
        ):
            build_generation_provider(unsupported_settings)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)


if __name__ == "__main__":
    unittest.main()
