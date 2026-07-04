import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    AgentPerformanceTrackingFoundationRegistry,
    agent_contract_by_id,
    agent_contract_registry,
    agent_metadata_by_agent_id,
    agent_metadata_registry,
    agent_performance_tracking_foundation_registry,
    agent_performance_tracking_profile_by_agent_id,
    agent_performance_tracking_profiles_for_latency_class,
    agent_performance_tracking_profiles_for_latency_threshold,
    cloud_model_registry,
    engine_contract_consistency_registry,
    execution_simulator_registry,
    latency_threshold_routing_registry,
    local_model_registry,
    model_profile_registry,
    parallel_scheduling_group_for_agent,
    parallel_scheduling_registry,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_SOURCE_REGISTRIES = (
    "agent_contract_registry",
    "agent_metadata_registry",
    "latency_threshold_routing_registry",
    "model_profile_registry",
    "execution_simulator_registry",
    "local_model_registry",
    "cloud_model_registry",
    "parallel_scheduling_registry",
    "engine_contract_consistency_registry",
)

REQUIRED_PROFILE_FIELDS = {
    "agent_id",
    "role_id",
    "performance_tracking_stage",
    "contract_serialization_version",
    "metadata_serialization_version",
    "contract_latency_class",
    "metadata_latency_class",
    "contract_latency_basis",
    "metadata_latency_basis",
    "contract_blocking_inputs",
    "contract_network_required_declared",
    "metadata_parallelization_support",
    "scheduling_group_id",
    "scheduling_hint",
    "max_parallel_agents",
    "latency_threshold_profile_ids",
    "latency_bands",
    "latency_metadata_sources",
    "model_profile_ids",
    "model_profile_kinds",
    "execution_simulation_profile_ids",
    "simulation_scopes",
    "local_latency_postures",
    "cloud_latency_postures",
    "consistency_family_ids",
    "performance_source_registries",
    "performance_dimensions",
    "passive_boundary_flags",
    "foundation_findings",
    "missing_coverage_items",
    "contract_blocked_runtime_behaviors",
    "metadata_blocked_runtime_behaviors",
    "latency_threshold_blocked_runtime_behaviors",
    "model_profile_blocked_runtime_behaviors",
    "execution_simulator_blocked_runtime_behaviors",
    "local_model_blocked_runtime_behaviors",
    "cloud_model_blocked_runtime_behaviors",
    "scheduling_blocked_runtime_behaviors",
    "consistency_blocked_runtime_behaviors",
    "blocked_runtime_behaviors",
    "foundation_status",
    "metadata_only_declared",
    "latency_metadata_alignment_present",
    "scheduling_reference_present",
    "latency_threshold_reference_present",
    "execution_simulation_reference_present",
    "model_latency_posture_reference_present",
    "performance_tracking_implemented",
    "latency_measurement_implemented",
    "latency_threshold_evaluation_implemented",
    "latency_based_routing_implemented",
    "runtime_selection_implemented",
    "model_selection_implemented",
    "execution_simulation_implemented",
    "parallel_execution_implemented",
    "execution_optimization_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "external_provider_calls_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "workflow_timing_change_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}


class AgentPerformanceTrackingFoundationTests(unittest.TestCase):
    def test_foundation_registry_covers_performance_sources(self) -> None:
        registry = agent_performance_tracking_foundation_registry()
        contracts = agent_contract_registry()
        metadata = agent_metadata_registry()
        latency = latency_threshold_routing_registry()
        models = model_profile_registry()
        simulator = execution_simulator_registry()
        local_models = local_model_registry()
        cloud_models = cloud_model_registry()
        scheduling = parallel_scheduling_registry()
        consistency = engine_contract_consistency_registry()

        self.assertEqual(
            registry.role,
            "agent_performance_tracking_foundation_registry",
        )
        self.assertEqual(
            registry.serialization_version,
            "agent_performance_tracking_foundation_registry.v1",
        )
        self.assertEqual(
            registry.performance_tracking_stage,
            "v4_6_agent_performance_tracking_foundation",
        )
        self.assertEqual(registry.agent_ids, contracts.agent_ids)
        self.assertEqual(registry.agent_ids, metadata.agent_ids)
        self.assertEqual(registry.profile_count, 12)
        self.assertEqual(
            registry.performance_source_registries,
            EXPECTED_SOURCE_REGISTRIES,
        )
        self.assertEqual(registry.latency_classes, ("low",))
        self.assertEqual(registry.scheduling_group_ids, scheduling.group_ids)
        self.assertEqual(
            registry.latency_threshold_profile_ids,
            latency.latency_threshold_profile_ids,
        )
        self.assertEqual(registry.latency_bands, latency.latency_bands)
        self.assertEqual(
            registry.latency_metadata_sources,
            latency.latency_metadata_sources,
        )
        self.assertEqual(registry.model_profile_ids, models.model_profile_ids)
        self.assertEqual(registry.model_profile_kinds, models.model_profile_kinds)
        self.assertEqual(
            registry.execution_simulation_profile_ids,
            simulator.execution_simulation_profile_ids,
        )
        self.assertEqual(registry.simulation_scopes, simulator.simulation_scopes)
        self.assertEqual(
            registry.local_latency_postures,
            tuple(surface.latency_posture for surface in local_models.model_surfaces),
        )
        self.assertEqual(
            registry.cloud_latency_postures,
            tuple(surface.latency_posture for surface in cloud_models.model_surfaces),
        )
        self.assertEqual(registry.consistency_family_ids, consistency.family_ids)
        self.assertIn("latency_metadata", consistency.shared_contract_concepts)
        self.assertTrue(registry.all_agents_covered)
        self.assertTrue(registry.no_missing_coverage)
        self.assertTrue(registry.metadata_only)
        self.assertIn("latency_measurement", registry.blocked_runtime_behaviors)
        self.assertIn("latency_based_routing", registry.blocked_runtime_behaviors)
        self.assertIn("parallel_task_execution", registry.blocked_runtime_behaviors)
        self.assertIn("does not measure latency", registry.authority_boundary)
        self.assertFalse(registry.performance_tracking_engine_implemented)
        self.assertFalse(registry.latency_measurement_implemented)
        self.assertFalse(registry.latency_threshold_evaluation_implemented)
        self.assertFalse(registry.latency_based_routing_implemented)
        self.assertFalse(registry.runtime_selection_implemented)
        self.assertFalse(registry.model_selection_implemented)
        self.assertFalse(registry.execution_simulation_implemented)
        self.assertFalse(registry.parallel_execution_implemented)
        self.assertFalse(registry.execution_optimization_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.provider_execution_implemented)
        self.assertFalse(registry.external_provider_calls_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.workflow_timing_change_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.prompt_mutation_implemented)
        self.assertFalse(registry.persistent_storage_write_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)

    def test_profiles_are_passive_and_source_aligned(self) -> None:
        registry = agent_performance_tracking_foundation_registry()

        for profile in registry.profiles:
            contract = agent_contract_by_id(profile.agent_id)
            metadata = agent_metadata_by_agent_id(profile.agent_id)
            scheduling_group = parallel_scheduling_group_for_agent(profile.agent_id)
            self.assertIsNotNone(contract)
            self.assertIsNotNone(metadata)
            self.assertIsNotNone(scheduling_group)
            assert contract is not None
            assert metadata is not None
            assert scheduling_group is not None

            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "agent_performance_tracking_foundation_profile.v1",
            )
            self.assertEqual(profile.foundation_status, "pass")
            self.assertEqual(
                profile.performance_tracking_stage,
                registry.performance_tracking_stage,
            )
            self.assertEqual(
                profile.contract_serialization_version,
                contract.serialization_version,
            )
            self.assertEqual(
                profile.metadata_serialization_version,
                metadata.serialization_version,
            )
            self.assertEqual(
                profile.contract_latency_class,
                contract.estimated_latency_metadata.relative_latency,
            )
            self.assertEqual(
                profile.metadata_latency_class,
                metadata.estimated_latency_class,
            )
            self.assertEqual(
                profile.contract_latency_class,
                profile.metadata_latency_class,
            )
            self.assertEqual(
                profile.contract_latency_basis,
                metadata.estimated_latency_basis,
            )
            self.assertEqual(
                profile.metadata_latency_basis,
                profile.contract_latency_basis,
            )
            self.assertEqual(
                profile.contract_blocking_inputs,
                contract.estimated_latency_metadata.blocking_inputs,
            )
            self.assertFalse(profile.contract_network_required_declared)
            self.assertEqual(
                profile.metadata_parallelization_support,
                metadata.parallelization_support,
            )
            self.assertEqual(profile.scheduling_group_id, scheduling_group.group_id)
            self.assertEqual(profile.scheduling_hint, scheduling_group.scheduling_hint)
            self.assertEqual(
                profile.max_parallel_agents,
                scheduling_group.max_parallel_agents,
            )
            self.assertEqual(
                profile.latency_threshold_profile_ids,
                registry.latency_threshold_profile_ids,
            )
            self.assertEqual(profile.latency_bands, registry.latency_bands)
            self.assertEqual(
                profile.latency_metadata_sources,
                registry.latency_metadata_sources,
            )
            self.assertEqual(profile.model_profile_ids, registry.model_profile_ids)
            self.assertEqual(profile.model_profile_kinds, registry.model_profile_kinds)
            self.assertEqual(
                profile.execution_simulation_profile_ids,
                registry.execution_simulation_profile_ids,
            )
            self.assertEqual(profile.simulation_scopes, registry.simulation_scopes)
            self.assertEqual(
                profile.local_latency_postures,
                registry.local_latency_postures,
            )
            self.assertEqual(
                profile.cloud_latency_postures,
                registry.cloud_latency_postures,
            )
            self.assertEqual(
                profile.consistency_family_ids,
                registry.consistency_family_ids,
            )
            self.assertEqual(
                profile.performance_source_registries,
                registry.performance_source_registries,
            )
            self.assertEqual(
                profile.performance_dimensions,
                registry.performance_dimensions,
            )
            self.assertEqual(
                profile.passive_boundary_flags,
                registry.passive_boundary_flags,
            )
            self.assertFalse(profile.missing_coverage_items)
            self.assertIn(
                "runtime_selection", profile.contract_blocked_runtime_behaviors
            )
            self.assertIn(
                "cost_or_latency_routing",
                profile.metadata_blocked_runtime_behaviors,
            )
            self.assertIn(
                "latency_based_routing",
                profile.latency_threshold_blocked_runtime_behaviors,
            )
            self.assertIn(
                "execution_optimization",
                profile.model_profile_blocked_runtime_behaviors,
            )
            self.assertIn(
                "simulation_runtime_execution",
                profile.execution_simulator_blocked_runtime_behaviors,
            )
            self.assertIn(
                "local_provider_execution",
                profile.local_model_blocked_runtime_behaviors,
            )
            self.assertIn(
                "pricing_or_latency_optimization",
                profile.cloud_model_blocked_runtime_behaviors,
            )
            self.assertIn(
                "parallel_task_execution",
                profile.scheduling_blocked_runtime_behaviors,
            )
            self.assertIn(
                "runtime_selection",
                profile.consistency_blocked_runtime_behaviors,
            )
            self.assertTrue(profile.metadata_only_declared)
            self.assertTrue(profile.latency_metadata_alignment_present)
            self.assertTrue(profile.scheduling_reference_present)
            self.assertTrue(profile.latency_threshold_reference_present)
            self.assertTrue(profile.execution_simulation_reference_present)
            self.assertTrue(profile.model_latency_posture_reference_present)
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.performance_tracking_implemented)
            self.assertFalse(profile.latency_measurement_implemented)
            self.assertFalse(profile.latency_threshold_evaluation_implemented)
            self.assertFalse(profile.latency_based_routing_implemented)
            self.assertFalse(profile.runtime_selection_implemented)
            self.assertFalse(profile.model_selection_implemented)
            self.assertFalse(profile.execution_simulation_implemented)
            self.assertFalse(profile.parallel_execution_implemented)
            self.assertFalse(profile.execution_optimization_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.provider_execution_implemented)
            self.assertFalse(profile.external_provider_calls_implemented)
            self.assertFalse(profile.agent_invocation_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.workflow_timing_change_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.prompt_mutation_implemented)
            self.assertFalse(profile.persistent_storage_write_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)

    def test_lookup_latency_class_and_threshold_filtering_are_stable(self) -> None:
        registry = agent_performance_tracking_foundation_registry()
        planner_profile = agent_performance_tracking_profile_by_agent_id(
            "planner_agent"
        )
        missing_profile = agent_performance_tracking_profile_by_agent_id(
            "missing_agent"
        )
        low_latency_profiles = agent_performance_tracking_profiles_for_latency_class(
            "low"
        )
        missing_latency_profiles = (
            agent_performance_tracking_profiles_for_latency_class("medium")
        )
        planning_threshold_profiles = (
            agent_performance_tracking_profiles_for_latency_threshold(
                "latency_threshold_routing::planning_execution_fit"
            )
        )
        missing_threshold_profiles = (
            agent_performance_tracking_profiles_for_latency_threshold(
                "missing_latency_threshold"
            )
        )

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(planner_profile)
        assert planner_profile is not None
        self.assertEqual(planner_profile.role_id, "planner")
        self.assertEqual(planner_profile.contract_latency_class, "low")
        self.assertEqual(
            planner_profile.scheduling_group_id,
            "parallel_group::foundational_context",
        )
        self.assertIn(
            "latency_threshold_routing::planning_execution_fit",
            planner_profile.latency_threshold_profile_ids,
        )
        self.assertEqual(len(low_latency_profiles), registry.profile_count)
        self.assertEqual(missing_latency_profiles, ())
        self.assertEqual(len(planning_threshold_profiles), registry.profile_count)
        self.assertEqual(missing_threshold_profiles, ())
        self.assertIs(
            planner_profile,
            agent_performance_tracking_profile_by_agent_id(
                "planner_agent",
                registry,
            ),
        )

    def test_foundation_registry_rejects_mismatched_or_active_profiles(self) -> None:
        registry = agent_performance_tracking_foundation_registry()
        first_profile = registry.profiles[0]
        duplicate_profile = first_profile.model_copy(update={"role_id": "duplicate"})
        mismatched_latency_profile = first_profile.model_copy(
            update={"metadata_latency_class": "medium"}
        )
        incomplete_profile = first_profile.model_copy(
            update={"missing_coverage_items": ("latency_reference_missing",)}
        )
        active_profile = first_profile.model_copy(
            update={"latency_measurement_implemented": True}
        )

        with self.assertRaisesRegex(ValueError, "agent_ids must be unique"):
            self._registry_with_profiles(
                (first_profile, duplicate_profile) + registry.profiles[2:]
            )

        with self.assertRaisesRegex(ValueError, "metadata_latency_class"):
            self._registry_with_profiles(
                (mismatched_latency_profile,) + registry.profiles[1:]
            )

        with self.assertRaisesRegex(ValueError, "missing coverage"):
            self._registry_with_profiles((incomplete_profile,) + registry.profiles[1:])

        with self.assertRaisesRegex(ValueError, "profiles must remain passive"):
            self._registry_with_profiles((active_profile,) + registry.profiles[1:])

    def test_agent_performance_tracking_foundation_does_not_change_request_routing(
        self,
    ) -> None:
        request = AssistantRequest(
            query="Generate performance metadata for a sketch.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        agent_performance_tracking_foundation_registry()
        agent_performance_tracking_profile_by_agent_id("planner_agent")
        agent_performance_tracking_profiles_for_latency_threshold(
            "latency_threshold_routing::planning_execution_fit"
        )
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)
        self.assertNotIn(
            "agent_performance_tracking_foundation_registry",
            next_decision.model_dump_json(),
        )

    def test_foundation_metadata_does_not_declare_active_performance_terms(
        self,
    ) -> None:
        registry = agent_performance_tracking_foundation_registry()
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
            "measure_runtime_latency",
            "run_execution_simulation",
            "execute_parallel_now",
            "route_by_runtime_latency",
            "select_runtime_for_latency",
            "select_provider_for_latency",
            "call_provider_for_performance",
            "mutate_generated_output",
        ):
            self.assertNotIn(forbidden_term, combined_text)

    def _registry_with_profiles(
        self,
        profiles: tuple,
    ) -> AgentPerformanceTrackingFoundationRegistry:
        registry = agent_performance_tracking_foundation_registry()
        return AgentPerformanceTrackingFoundationRegistry(
            profiles=profiles,
            agent_ids=registry.agent_ids,
            profile_count=registry.profile_count,
            performance_source_registries=registry.performance_source_registries,
            latency_classes=registry.latency_classes,
            scheduling_group_ids=registry.scheduling_group_ids,
            latency_threshold_profile_ids=registry.latency_threshold_profile_ids,
            latency_bands=registry.latency_bands,
            latency_metadata_sources=registry.latency_metadata_sources,
            model_profile_ids=registry.model_profile_ids,
            model_profile_kinds=registry.model_profile_kinds,
            execution_simulation_profile_ids=(
                registry.execution_simulation_profile_ids
            ),
            simulation_scopes=registry.simulation_scopes,
            local_latency_postures=registry.local_latency_postures,
            cloud_latency_postures=registry.cloud_latency_postures,
            consistency_family_ids=registry.consistency_family_ids,
            performance_dimensions=registry.performance_dimensions,
            passive_boundary_flags=registry.passive_boundary_flags,
        )


if __name__ == "__main__":
    unittest.main()
