import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    AgentCostTrackingFoundationRegistry,
    agent_contract_by_id,
    agent_contract_registry,
    agent_cost_tracking_foundation_registry,
    agent_cost_tracking_profile_by_agent_id,
    agent_cost_tracking_profiles_for_cost_class,
    agent_cost_tracking_profiles_for_cost_profile,
    agent_metadata_by_agent_id,
    agent_metadata_registry,
    cost_profile_registry,
    cost_threshold_routing_registry,
    creative_exploration_budget_registry,
    engine_contract_consistency_registry,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_SOURCE_REGISTRIES = (
    "agent_contract_registry",
    "agent_metadata_registry",
    "creative_exploration_budget_registry",
    "cost_threshold_routing_registry",
    "cost_profile_registry",
    "engine_contract_consistency_registry",
)

REQUIRED_PROFILE_FIELDS = {
    "agent_id",
    "role_id",
    "cost_tracking_stage",
    "contract_serialization_version",
    "metadata_serialization_version",
    "contract_cost_class",
    "metadata_cost_class",
    "contract_cost_basis",
    "metadata_cost_basis",
    "contract_cache_sensitivity",
    "external_provider_calls_declared",
    "budget_profile_ids",
    "budget_postures",
    "cost_threshold_profile_ids",
    "cost_threshold_bands",
    "cost_profile_ids",
    "studio_cost_bands",
    "consistency_family_ids",
    "cost_source_registries",
    "cost_dimensions",
    "passive_boundary_flags",
    "foundation_findings",
    "missing_coverage_items",
    "contract_blocked_runtime_behaviors",
    "metadata_blocked_runtime_behaviors",
    "budget_blocked_runtime_behaviors",
    "threshold_blocked_runtime_behaviors",
    "profile_blocked_runtime_behaviors",
    "consistency_blocked_runtime_behaviors",
    "blocked_runtime_behaviors",
    "foundation_status",
    "metadata_only_declared",
    "cost_metadata_alignment_present",
    "budget_threshold_reference_present",
    "studio_cost_profile_reference_present",
    "cost_tracking_implemented",
    "cost_metering_implemented",
    "pricing_lookup_implemented",
    "budget_enforcement_implemented",
    "cost_based_routing_implemented",
    "execution_optimization_implemented",
    "provider_model_routing_implemented",
    "external_provider_calls_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}


class AgentCostTrackingFoundationTests(unittest.TestCase):
    def test_foundation_registry_covers_cost_sources(self) -> None:
        registry = agent_cost_tracking_foundation_registry()
        contracts = agent_contract_registry()
        metadata = agent_metadata_registry()
        budget = creative_exploration_budget_registry()
        threshold = cost_threshold_routing_registry()
        studio_cost = cost_profile_registry()
        consistency = engine_contract_consistency_registry()

        self.assertEqual(registry.role, "agent_cost_tracking_foundation_registry")
        self.assertEqual(
            registry.serialization_version,
            "agent_cost_tracking_foundation_registry.v1",
        )
        self.assertEqual(
            registry.cost_tracking_stage,
            "v4_6_agent_cost_tracking_foundation",
        )
        self.assertEqual(registry.agent_ids, contracts.agent_ids)
        self.assertEqual(registry.agent_ids, metadata.agent_ids)
        self.assertEqual(registry.profile_count, 12)
        self.assertEqual(registry.cost_source_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertEqual(registry.cost_classes, ("low",))
        self.assertEqual(registry.budget_profile_ids, budget.budget_profile_ids)
        self.assertEqual(registry.budget_postures, budget.budget_postures)
        self.assertEqual(
            registry.cost_threshold_profile_ids,
            threshold.cost_threshold_profile_ids,
        )
        self.assertEqual(registry.cost_threshold_bands, threshold.cost_bands)
        self.assertEqual(registry.cost_profile_ids, studio_cost.cost_profile_ids)
        self.assertEqual(registry.studio_cost_bands, studio_cost.cost_bands)
        self.assertEqual(registry.consistency_family_ids, consistency.family_ids)
        self.assertIn("cost_metadata", consistency.shared_contract_concepts)
        self.assertTrue(registry.all_agents_covered)
        self.assertTrue(registry.no_missing_coverage)
        self.assertTrue(registry.metadata_only)
        self.assertIn("cost_metering", registry.blocked_runtime_behaviors)
        self.assertIn("pricing_lookup", registry.blocked_runtime_behaviors)
        self.assertIn("cost_based_routing", registry.blocked_runtime_behaviors)
        self.assertIn("does not meter cost", registry.authority_boundary)
        self.assertFalse(registry.cost_tracking_engine_implemented)
        self.assertFalse(registry.cost_metering_implemented)
        self.assertFalse(registry.pricing_lookup_implemented)
        self.assertFalse(registry.budget_enforcement_implemented)
        self.assertFalse(registry.cost_based_routing_implemented)
        self.assertFalse(registry.execution_optimization_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.external_provider_calls_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.prompt_mutation_implemented)
        self.assertFalse(registry.persistent_storage_write_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)

    def test_profiles_are_passive_and_source_aligned(self) -> None:
        registry = agent_cost_tracking_foundation_registry()

        for profile in registry.profiles:
            contract = agent_contract_by_id(profile.agent_id)
            metadata = agent_metadata_by_agent_id(profile.agent_id)
            self.assertIsNotNone(contract)
            self.assertIsNotNone(metadata)
            assert contract is not None
            assert metadata is not None

            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "agent_cost_tracking_foundation_profile.v1",
            )
            self.assertEqual(profile.foundation_status, "pass")
            self.assertEqual(profile.cost_tracking_stage, registry.cost_tracking_stage)
            self.assertEqual(
                profile.contract_serialization_version,
                contract.serialization_version,
            )
            self.assertEqual(
                profile.metadata_serialization_version,
                metadata.serialization_version,
            )
            self.assertEqual(
                profile.contract_cost_class,
                contract.estimated_cost_metadata.relative_cost,
            )
            self.assertEqual(profile.metadata_cost_class, metadata.estimated_cost_class)
            self.assertEqual(profile.contract_cost_class, profile.metadata_cost_class)
            self.assertEqual(profile.contract_cost_basis, metadata.estimated_cost_basis)
            self.assertEqual(profile.metadata_cost_basis, profile.contract_cost_basis)
            self.assertEqual(
                profile.contract_cache_sensitivity,
                contract.estimated_cost_metadata.cache_sensitivity,
            )
            self.assertFalse(profile.external_provider_calls_declared)
            self.assertEqual(profile.budget_profile_ids, registry.budget_profile_ids)
            self.assertEqual(profile.budget_postures, registry.budget_postures)
            self.assertEqual(
                profile.cost_threshold_profile_ids,
                registry.cost_threshold_profile_ids,
            )
            self.assertEqual(
                profile.cost_threshold_bands, registry.cost_threshold_bands
            )
            self.assertEqual(profile.cost_profile_ids, registry.cost_profile_ids)
            self.assertEqual(profile.studio_cost_bands, registry.studio_cost_bands)
            self.assertEqual(
                profile.consistency_family_ids, registry.consistency_family_ids
            )
            self.assertEqual(
                profile.cost_source_registries, registry.cost_source_registries
            )
            self.assertEqual(profile.cost_dimensions, registry.cost_dimensions)
            self.assertEqual(
                profile.passive_boundary_flags, registry.passive_boundary_flags
            )
            self.assertFalse(profile.missing_coverage_items)
            self.assertIn(
                "provider_or_model_routing", profile.contract_blocked_runtime_behaviors
            )
            self.assertIn(
                "cost_or_latency_routing", profile.metadata_blocked_runtime_behaviors
            )
            self.assertIn(
                "budget_enforcement", profile.budget_blocked_runtime_behaviors
            )
            self.assertIn(
                "cost_based_routing", profile.threshold_blocked_runtime_behaviors
            )
            self.assertIn("cost_scoring", profile.profile_blocked_runtime_behaviors)
            self.assertIn(
                "provider_or_model_routing",
                profile.consistency_blocked_runtime_behaviors,
            )
            self.assertTrue(profile.metadata_only_declared)
            self.assertTrue(profile.cost_metadata_alignment_present)
            self.assertTrue(profile.budget_threshold_reference_present)
            self.assertTrue(profile.studio_cost_profile_reference_present)
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.cost_tracking_implemented)
            self.assertFalse(profile.cost_metering_implemented)
            self.assertFalse(profile.pricing_lookup_implemented)
            self.assertFalse(profile.budget_enforcement_implemented)
            self.assertFalse(profile.cost_based_routing_implemented)
            self.assertFalse(profile.execution_optimization_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.external_provider_calls_implemented)
            self.assertFalse(profile.agent_invocation_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.prompt_mutation_implemented)
            self.assertFalse(profile.persistent_storage_write_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)

    def test_lookup_cost_class_and_cost_profile_filtering_are_stable(self) -> None:
        registry = agent_cost_tracking_foundation_registry()
        planner_profile = agent_cost_tracking_profile_by_agent_id("planner_agent")
        missing_profile = agent_cost_tracking_profile_by_agent_id("missing_agent")
        low_cost_profiles = agent_cost_tracking_profiles_for_cost_class("low")
        missing_cost_class_profiles = agent_cost_tracking_profiles_for_cost_class(
            "medium"
        )
        creative_reasoning_profiles = agent_cost_tracking_profiles_for_cost_profile(
            "creative_reasoning_cost_profile"
        )
        missing_cost_profiles = agent_cost_tracking_profiles_for_cost_profile(
            "missing_cost_profile"
        )

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(planner_profile)
        assert planner_profile is not None
        self.assertEqual(planner_profile.role_id, "planner")
        self.assertEqual(planner_profile.contract_cost_class, "low")
        self.assertIn(
            "creative_reasoning_cost_profile",
            planner_profile.cost_profile_ids,
        )
        self.assertEqual(len(low_cost_profiles), registry.profile_count)
        self.assertEqual(missing_cost_class_profiles, ())
        self.assertEqual(len(creative_reasoning_profiles), registry.profile_count)
        self.assertEqual(missing_cost_profiles, ())
        self.assertIs(
            planner_profile,
            agent_cost_tracking_profile_by_agent_id("planner_agent", registry),
        )

    def test_foundation_registry_rejects_mismatched_or_active_profiles(self) -> None:
        registry = agent_cost_tracking_foundation_registry()
        first_profile = registry.profiles[0]
        duplicate_profile = first_profile.model_copy(update={"role_id": "duplicate"})
        mismatched_cost_class_profile = first_profile.model_copy(
            update={"metadata_cost_class": "medium"}
        )
        incomplete_profile = first_profile.model_copy(
            update={"missing_coverage_items": ("budget_reference_missing",)}
        )
        active_profile = first_profile.model_copy(
            update={"cost_metering_implemented": True}
        )

        with self.assertRaisesRegex(ValueError, "agent_ids must be unique"):
            self._registry_with_profiles(
                (first_profile, duplicate_profile) + registry.profiles[2:]
            )

        with self.assertRaisesRegex(ValueError, "metadata_cost_class"):
            self._registry_with_profiles(
                (mismatched_cost_class_profile,) + registry.profiles[1:]
            )

        with self.assertRaisesRegex(ValueError, "missing coverage"):
            self._registry_with_profiles((incomplete_profile,) + registry.profiles[1:])

        with self.assertRaisesRegex(ValueError, "profiles must remain passive"):
            self._registry_with_profiles((active_profile,) + registry.profiles[1:])

    def test_agent_cost_tracking_foundation_does_not_change_request_routing(
        self,
    ) -> None:
        request = AssistantRequest(
            query="Generate cost metadata for a sketch.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        agent_cost_tracking_foundation_registry()
        agent_cost_tracking_profile_by_agent_id("planner_agent")
        agent_cost_tracking_profiles_for_cost_profile("planning_iteration_cost_profile")
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)
        self.assertNotIn(
            "agent_cost_tracking_foundation_registry",
            next_decision.model_dump_json(),
        )

    def test_foundation_metadata_does_not_declare_active_cost_terms(self) -> None:
        registry = agent_cost_tracking_foundation_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *registry.passive_boundary_flags,
                *(
                    field
                    for profile in registry.profiles
                    for field in (
                        profile.agent_id,
                        *profile.foundation_findings,
                        *profile.passive_boundary_flags,
                    )
                ),
            )
        )

        for forbidden_term in (
            "meter_runtime_cost",
            "calculate_runtime_pricing",
            "call_provider_for_cost",
            "enforce_runtime_budget",
            "route_by_runtime_cost",
            "optimize_execution_cost",
            "select_provider_for_cost",
            "invoke_agent_for_cost",
            "mutate_generated_output",
        ):
            self.assertNotIn(forbidden_term, combined_text)

    def _registry_with_profiles(
        self,
        profiles: tuple,
    ) -> AgentCostTrackingFoundationRegistry:
        registry = agent_cost_tracking_foundation_registry()
        return AgentCostTrackingFoundationRegistry(
            profiles=profiles,
            agent_ids=registry.agent_ids,
            profile_count=registry.profile_count,
            cost_source_registries=registry.cost_source_registries,
            cost_classes=registry.cost_classes,
            budget_profile_ids=registry.budget_profile_ids,
            budget_postures=registry.budget_postures,
            cost_threshold_profile_ids=registry.cost_threshold_profile_ids,
            cost_threshold_bands=registry.cost_threshold_bands,
            cost_profile_ids=registry.cost_profile_ids,
            studio_cost_bands=registry.studio_cost_bands,
            consistency_family_ids=registry.consistency_family_ids,
            cost_dimensions=registry.cost_dimensions,
            passive_boundary_flags=registry.passive_boundary_flags,
        )


if __name__ == "__main__":
    unittest.main()
