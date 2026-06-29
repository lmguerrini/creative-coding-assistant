import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    ModelRoutingArchitectureConsistencyRegistry,
    model_routing_architecture_consistency_by_surface,
    model_routing_architecture_consistency_records_for_layer,
    model_routing_architecture_consistency_registry,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_SURFACE_IDS = (
    "model_router",
    "routing_intelligence",
    "local_cloud_routing",
    "hybrid_routing",
    "quality_cost_optimizer",
    "cost_estimator",
    "budget_policies",
    "hitl_budget_gate",
    "runtime_recommendation_engine",
    "execution_policy_engine",
    "model_recommendation_engine",
    "model_capability_matrix",
    "provider_capability_matrix",
    "quality_prediction_engine",
    "cost_prediction_engine",
    "creative_quality_predictor",
    "creative_diversity_predictor",
    "creative_consistency_predictor",
    "routing_explainability",
)
EXPECTED_ARCHITECTURE_LAYERS = (
    "routing_metadata_boundary",
    "optimization_budget_boundary",
    "runtime_policy_boundary",
    "capability_matrix_boundary",
    "prediction_metadata_boundary",
    "creative_prediction_boundary",
    "explainability_boundary",
)
REQUIRED_RECORD_FIELDS = {
    "surface_id",
    "architecture_layer",
    "architecture_stage",
    "source_role",
    "source_serialization_version",
    "source_route_name",
    "source_count_field",
    "source_count",
    "validated_version_rules",
    "passive_boundary_flags",
    "source_blocked_runtime_behaviors",
    "source_active_runtime_flags",
    "missing_coverage_items",
    "source_metadata_only_declared",
    "v5_architecture_consistency_confirmed",
    "v4_boundary_compatibility_confirmed",
    "version_runtime_rules_confirmed",
    "architecture_consistency_status",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "budget_enforcement_implemented",
    "hitl_emission_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class ModelRoutingArchitectureConsistencyTests(unittest.TestCase):
    def test_registry_covers_v5_2_routing_surfaces(self) -> None:
        registry = model_routing_architecture_consistency_registry()

        self.assertEqual(
            registry.role,
            "model_routing_architecture_consistency_registry",
        )
        self.assertEqual(
            registry.serialization_version,
            "model_routing_architecture_consistency_registry.v1",
        )
        self.assertEqual(
            registry.architecture_stage,
            "v5_2_architecture_consistency_pass",
        )
        self.assertEqual(registry.route_name, RouteName.GENERATE)
        self.assertEqual(registry.surface_ids, EXPECTED_SURFACE_IDS)
        self.assertEqual(registry.architecture_layers, EXPECTED_ARCHITECTURE_LAYERS)
        self.assertEqual(registry.record_count, 19)
        self.assertTrue(registry.all_surfaces_covered)
        self.assertTrue(registry.route_consistency_confirmed)
        self.assertTrue(registry.no_active_runtime_flags)
        self.assertTrue(registry.no_missing_coverage)
        self.assertTrue(registry.v4_boundaries_preserved)
        self.assertTrue(registry.runtime_evolution_not_applied)
        self.assertIn("runtime_evolution_not_applied", registry.validated_version_rules)
        self.assertIn("provider_or_model_routing", registry.blocked_runtime_behaviors)
        self.assertIn("does not apply routing", registry.authority_boundary)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.provider_execution_implemented)
        self.assertFalse(registry.budget_enforcement_implemented)
        self.assertFalse(registry.hitl_emission_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.prompt_mutation_implemented)
        self.assertFalse(registry.persistent_storage_write_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.runtime_evolution_implemented)
        self.assertTrue(registry.advisory_only)

    def test_records_are_passive_and_source_aligned(self) -> None:
        registry = model_routing_architecture_consistency_registry()

        for record in registry.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "model_routing_architecture_consistency_record.v1",
            )
            self.assertEqual(record.architecture_stage, registry.architecture_stage)
            self.assertTrue(record.source_serialization_version.endswith(".v1"))
            self.assertIn(record.source_count_field, {
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
            })
            self.assertGreaterEqual(record.source_count, 1)
            if record.source_route_name is not None:
                self.assertEqual(record.source_route_name, registry.route_name)
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
            self.assertTrue(record.source_metadata_only_declared)
            self.assertTrue(record.v5_architecture_consistency_confirmed)
            self.assertTrue(record.v4_boundary_compatibility_confirmed)
            self.assertTrue(record.version_runtime_rules_confirmed)
            self.assertEqual(record.architecture_consistency_status, "pass")
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.provider_execution_implemented)
            self.assertFalse(record.budget_enforcement_implemented)
            self.assertFalse(record.hitl_emission_implemented)
            self.assertFalse(record.workflow_control_implemented)
            self.assertFalse(record.retry_triggering_implemented)
            self.assertFalse(record.prompt_mutation_implemented)
            self.assertFalse(record.persistent_storage_write_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)
            self.assertFalse(record.runtime_evolution_implemented)
            self.assertTrue(record.advisory_only)

    def test_lookup_and_layer_filters_are_stable(self) -> None:
        registry = model_routing_architecture_consistency_registry()
        model_router = model_routing_architecture_consistency_by_surface(
            "model_router",
            registry,
        )
        missing = model_routing_architecture_consistency_by_surface("missing", registry)
        prediction_records = model_routing_architecture_consistency_records_for_layer(
            "prediction_metadata_boundary",
            registry,
        )
        creative_records = model_routing_architecture_consistency_records_for_layer(
            "creative_prediction_boundary",
            registry,
        )

        self.assertIsNone(missing)
        self.assertIsNotNone(model_router)
        assert model_router is not None
        self.assertEqual(model_router.source_role, "model_router")
        self.assertEqual(model_router.source_count_field, "candidate_count")
        self.assertEqual(
            tuple(record.surface_id for record in prediction_records),
            ("quality_prediction_engine", "cost_prediction_engine"),
        )
        self.assertEqual(
            tuple(record.surface_id for record in creative_records),
            (
                "creative_quality_predictor",
                "creative_diversity_predictor",
                "creative_consistency_predictor",
            ),
        )

    def test_registry_rejects_mismatched_or_active_records(self) -> None:
        registry = model_routing_architecture_consistency_registry()
        first_record = registry.records[0]
        duplicate_record = first_record.model_copy(
            update={"architecture_layer": "runtime_policy_boundary"}
        )
        missing_record = first_record.model_copy(
            update={"missing_coverage_items": ("serialization_version_missing",)}
        )
        active_record = first_record.model_copy(
            update={
                "source_active_runtime_flags": (
                    "provider_model_routing_implemented",
                )
            }
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
            query="Generate model routing architecture consistency metadata.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        model_routing_architecture_consistency_registry()
        model_routing_architecture_consistency_by_surface("routing_explainability")
        model_routing_architecture_consistency_records_for_layer(
            "runtime_policy_boundary"
        )
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)

    def test_architecture_metadata_does_not_declare_active_terms(self) -> None:
        registry = model_routing_architecture_consistency_registry()
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
            "apply_routing(",
            "select_model(",
            "switch_provider(",
            "execute_provider(",
            "enforce_budget(",
            "emit_hitl(",
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
    ) -> ModelRoutingArchitectureConsistencyRegistry:
        registry = model_routing_architecture_consistency_registry()
        return ModelRoutingArchitectureConsistencyRegistry(
            records=records,
            surface_ids=tuple(record.surface_id for record in records),
            record_count=len(records),
            architecture_layers=registry.architecture_layers,
            validated_version_rules=registry.validated_version_rules,
            passive_boundary_flags=registry.passive_boundary_flags,
        )


if __name__ == "__main__":
    unittest.main()
