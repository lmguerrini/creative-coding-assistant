import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    PerformanceArchitectureConsistencyRegistry,
    performance_architecture_consistency_by_surface,
    performance_architecture_consistency_records_for_layer,
    performance_architecture_consistency_registry,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_SURFACE_IDS = (
    "parallel_scheduler",
    "latency_optimizer",
    "async_execution",
    "streaming_optimizer",
    "retry_policies",
    "load_balancer",
    "execution_profiling",
    "workflow_replay_engine",
    "execution_replay_engine",
    "bottleneck_detection",
    "throughput_optimizer",
    "performance_prediction",
    "performance_benchmarking",
    "reasoning_budget_optimizer",
    "performance_regression_detection",
    "resource_utilization_optimizer",
)
EXPECTED_ARCHITECTURE_LAYERS = (
    "scheduling_runtime_boundary",
    "throughput_latency_boundary",
    "profiling_replay_boundary",
    "prediction_benchmark_boundary",
    "budget_resource_boundary",
)
REQUIRED_RECORD_FIELDS = {
    "surface_id",
    "architecture_layer",
    "architecture_stage",
    "source_role",
    "source_serialization_version",
    "source_count_field",
    "source_count",
    "validated_version_rules",
    "passive_boundary_flags",
    "source_blocked_runtime_behaviors",
    "source_active_runtime_flags",
    "missing_coverage_items",
    "source_advisory_only_declared",
    "v5_architecture_consistency_confirmed",
    "v4_boundary_compatibility_confirmed",
    "version_runtime_rules_confirmed",
    "architecture_consistency_status",
    "runtime_measurement_implemented",
    "workflow_execution_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "provider_model_routing_implemented",
    "resource_allocation_implemented",
    "capacity_enforcement_implemented",
    "budget_enforcement_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class PerformanceArchitectureConsistencyTests(unittest.TestCase):
    def test_registry_covers_v5_3_performance_surfaces(self) -> None:
        registry = performance_architecture_consistency_registry()

        self.assertEqual(
            registry.role,
            "performance_architecture_consistency_registry",
        )
        self.assertEqual(
            registry.serialization_version,
            "performance_architecture_consistency_registry.v1",
        )
        self.assertEqual(
            registry.architecture_stage,
            "v5_3_architecture_consistency_pass",
        )
        self.assertEqual(registry.surface_ids, EXPECTED_SURFACE_IDS)
        self.assertEqual(registry.architecture_layers, EXPECTED_ARCHITECTURE_LAYERS)
        self.assertEqual(registry.record_count, 16)
        self.assertTrue(registry.all_surfaces_covered)
        self.assertTrue(registry.no_active_runtime_flags)
        self.assertTrue(registry.no_missing_coverage)
        self.assertTrue(registry.v4_boundaries_preserved)
        self.assertTrue(registry.runtime_evolution_not_applied)
        self.assertIn("runtime_evolution_not_applied", registry.validated_version_rules)
        self.assertIn(
            "runtime_performance_measurement",
            registry.blocked_runtime_behaviors,
        )
        self.assertIn("does not measure performance", registry.authority_boundary)
        self.assertFalse(registry.runtime_measurement_implemented)
        self.assertFalse(registry.workflow_execution_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.workflow_graph_mutation_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.resource_allocation_implemented)
        self.assertFalse(registry.capacity_enforcement_implemented)
        self.assertFalse(registry.budget_enforcement_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.prompt_mutation_implemented)
        self.assertFalse(registry.persistent_storage_write_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.runtime_evolution_implemented)
        self.assertTrue(registry.advisory_only)

    def test_records_are_passive_and_source_aligned(self) -> None:
        registry = performance_architecture_consistency_registry()

        for record in registry.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "performance_architecture_consistency_record.v1",
            )
            self.assertEqual(record.architecture_stage, registry.architecture_stage)
            self.assertTrue(record.source_serialization_version.endswith(".v1"))
            self.assertIn(
                record.source_count_field,
                {
                    "candidate_count",
                    "scenario_count",
                    "prediction_count",
                    "recommendation_count",
                    "signal_count",
                },
            )
            self.assertGreaterEqual(record.source_count, 1)
            self.assertEqual(
                record.validated_version_rules,
                registry.validated_version_rules,
            )
            self.assertEqual(
                record.passive_boundary_flags,
                registry.passive_boundary_flags,
            )
            self.assertTrue(record.source_blocked_runtime_behaviors)
            self.assertFalse(record.source_active_runtime_flags)
            self.assertFalse(record.missing_coverage_items)
            self.assertTrue(record.source_advisory_only_declared)
            self.assertTrue(record.v5_architecture_consistency_confirmed)
            self.assertTrue(record.v4_boundary_compatibility_confirmed)
            self.assertTrue(record.version_runtime_rules_confirmed)
            self.assertEqual(record.architecture_consistency_status, "pass")
            self.assertFalse(record.runtime_measurement_implemented)
            self.assertFalse(record.workflow_execution_implemented)
            self.assertFalse(record.workflow_control_implemented)
            self.assertFalse(record.workflow_graph_mutation_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.resource_allocation_implemented)
            self.assertFalse(record.capacity_enforcement_implemented)
            self.assertFalse(record.budget_enforcement_implemented)
            self.assertFalse(record.retry_triggering_implemented)
            self.assertFalse(record.prompt_mutation_implemented)
            self.assertFalse(record.persistent_storage_write_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)
            self.assertFalse(record.runtime_evolution_implemented)
            self.assertTrue(record.advisory_only)

    def test_lookup_and_layer_filters_are_stable(self) -> None:
        registry = performance_architecture_consistency_registry()
        resource = performance_architecture_consistency_by_surface(
            "resource_utilization_optimizer",
            registry,
        )
        missing = performance_architecture_consistency_by_surface("missing", registry)
        scheduling_records = performance_architecture_consistency_records_for_layer(
            "scheduling_runtime_boundary",
            registry,
        )
        prediction_records = performance_architecture_consistency_records_for_layer(
            "prediction_benchmark_boundary",
            registry,
        )
        budget_records = performance_architecture_consistency_records_for_layer(
            "budget_resource_boundary",
            registry,
        )

        self.assertIsNone(missing)
        self.assertIsNotNone(resource)
        assert resource is not None
        self.assertEqual(resource.source_role, "resource_utilization_optimizer")
        self.assertEqual(resource.source_count_field, "recommendation_count")
        self.assertEqual(
            tuple(record.surface_id for record in scheduling_records),
            (
                "parallel_scheduler",
                "async_execution",
                "streaming_optimizer",
                "retry_policies",
                "load_balancer",
            ),
        )
        self.assertEqual(
            tuple(record.surface_id for record in prediction_records),
            (
                "performance_prediction",
                "performance_benchmarking",
                "performance_regression_detection",
            ),
        )
        self.assertEqual(
            tuple(record.surface_id for record in budget_records),
            ("reasoning_budget_optimizer", "resource_utilization_optimizer"),
        )

    def test_registry_rejects_mismatched_or_active_records(self) -> None:
        registry = performance_architecture_consistency_registry()
        first_record = registry.records[0]
        duplicate_record = first_record.model_copy(
            update={"architecture_layer": "budget_resource_boundary"}
        )
        missing_record = first_record.model_copy(
            update={"missing_coverage_items": ("serialization_version_missing",)}
        )
        active_record = first_record.model_copy(
            update={"source_active_runtime_flags": ("workflow_execution_implemented",)}
        )

        with self.assertRaisesRegex(ValueError, "surface_ids must be unique"):
            self._registry_with_records(
                (first_record, duplicate_record) + registry.records[2:]
            )

        with self.assertRaisesRegex(ValueError, "missing coverage"):
            self._registry_with_records((missing_record,) + registry.records[1:])

        with self.assertRaisesRegex(ValueError, "active runtime flags"):
            self._registry_with_records((active_record,) + registry.records[1:])

    def test_architecture_pass_does_not_change_request_routing(self) -> None:
        request = AssistantRequest(
            query="Generate performance architecture consistency metadata.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        performance_architecture_consistency_registry()
        performance_architecture_consistency_by_surface("throughput_optimizer")
        performance_architecture_consistency_records_for_layer(
            "budget_resource_boundary"
        )
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)

    def test_architecture_metadata_does_not_declare_active_terms(self) -> None:
        registry = performance_architecture_consistency_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *registry.passive_boundary_flags,
                *(
                    field
                    for record in registry.records
                    for field in (
                        record.surface_id,
                        record.architecture_layer,
                        *record.validated_version_rules,
                        *record.passive_boundary_flags,
                    )
                ),
            )
        )

        for forbidden_term in (
            "measure_performance(",
            "execute_workflow(",
            "execute_benchmark(",
            "allocate_resources(",
            "enforce_capacity(",
            "enforce_budget(",
            "route_provider(",
            "control_workflow(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)

    def _registry_with_records(
        self,
        records: tuple,
    ) -> PerformanceArchitectureConsistencyRegistry:
        registry = performance_architecture_consistency_registry()
        return PerformanceArchitectureConsistencyRegistry(
            records=records,
            surface_ids=tuple(record.surface_id for record in records),
            record_count=len(records),
            architecture_layers=registry.architecture_layers,
            validated_version_rules=registry.validated_version_rules,
            passive_boundary_flags=registry.passive_boundary_flags,
        )


if __name__ == "__main__":
    unittest.main()
