import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    AdaptiveExecutionArchitectureConsistencyRegistry,
    adaptive_execution_architecture_consistency_by_surface,
    adaptive_execution_architecture_consistency_records_for_layer,
    adaptive_execution_architecture_consistency_registry,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_SURFACE_IDS = (
    "adaptive_hybrid_workflow_optimizer",
    "adaptive_escalation_optimizer",
    "agent_activation_optimizer",
    "adaptive_cost_quality_optimizer",
    "adaptive_latency_optimizer",
    "adaptive_execution_strategy_selection",
    "dynamic_agent_allocation",
    "dynamic_resource_allocation",
    "workflow_self_tuning_policies",
    "execution_confidence_engine",
    "workflow_risk_engine",
    "creative_exploration_optimizer",
    "emergence_optimizer",
    "agent_diversity_optimizer",
    "reflection_budget_optimizer",
    "adaptive_policy_explainability",
)
EXPECTED_ARCHITECTURE_LAYERS = (
    "hybrid_workflow_boundary",
    "escalation_policy_boundary",
    "agent_resource_boundary",
    "cost_latency_boundary",
    "execution_strategy_boundary",
    "confidence_risk_boundary",
    "creative_adaptation_boundary",
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
    "source_advisory_only_declared",
    "v5_architecture_consistency_confirmed",
    "v4_boundary_compatibility_confirmed",
    "version_runtime_rules_confirmed",
    "architecture_consistency_status",
    "policy_application_implemented",
    "strategy_application_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
    "resource_allocation_implemented",
    "budget_enforcement_implemented",
    "hitl_emission_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class AdaptiveExecutionArchitectureConsistencyTests(unittest.TestCase):
    def test_registry_covers_v5_5_adaptive_execution_surfaces(self) -> None:
        registry = adaptive_execution_architecture_consistency_registry()

        self.assertEqual(
            registry.role,
            "adaptive_execution_architecture_consistency_registry",
        )
        self.assertEqual(
            registry.serialization_version,
            "adaptive_execution_architecture_consistency_registry.v1",
        )
        self.assertEqual(
            registry.architecture_stage,
            "v5_5_architecture_consistency_pass",
        )
        self.assertEqual(registry.route_name, RouteName.GENERATE)
        self.assertEqual(registry.surface_ids, EXPECTED_SURFACE_IDS)
        self.assertEqual(registry.architecture_layers, EXPECTED_ARCHITECTURE_LAYERS)
        self.assertEqual(registry.record_count, 16)
        self.assertTrue(registry.all_surfaces_covered)
        self.assertTrue(registry.route_consistency_confirmed)
        self.assertTrue(registry.no_active_runtime_flags)
        self.assertTrue(registry.no_missing_coverage)
        self.assertTrue(registry.v4_boundaries_preserved)
        self.assertTrue(registry.runtime_evolution_not_applied)
        self.assertIn("runtime_evolution_not_applied", registry.validated_version_rules)
        self.assertIn("provider_or_model_routing", registry.blocked_runtime_behaviors)
        self.assertIn("does not apply adaptive policies", registry.authority_boundary)
        self.assertFalse(registry.policy_application_implemented)
        self.assertFalse(registry.strategy_application_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.provider_execution_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.resource_allocation_implemented)
        self.assertFalse(registry.budget_enforcement_implemented)
        self.assertFalse(registry.hitl_emission_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.workflow_graph_mutation_implemented)
        self.assertFalse(registry.workflow_execution_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.prompt_mutation_implemented)
        self.assertFalse(registry.persistent_storage_write_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.runtime_evolution_implemented)
        self.assertTrue(registry.advisory_only)

    def test_records_are_passive_and_source_aligned(self) -> None:
        registry = adaptive_execution_architecture_consistency_registry()

        for record in registry.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "adaptive_execution_architecture_consistency_record.v1",
            )
            self.assertEqual(record.architecture_stage, registry.architecture_stage)
            self.assertTrue(record.source_serialization_version.endswith(".v1"))
            self.assertIn(
                record.source_count_field,
                {
                    "candidate_count",
                    "decision_count",
                    "strategy_count",
                    "allocation_count",
                    "policy_count",
                    "signal_count",
                    "factor_count",
                    "explanation_count",
                },
            )
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
            self.assertTrue(record.source_advisory_only_declared)
            self.assertTrue(record.v5_architecture_consistency_confirmed)
            self.assertTrue(record.v4_boundary_compatibility_confirmed)
            self.assertTrue(record.version_runtime_rules_confirmed)
            self.assertEqual(record.architecture_consistency_status, "pass")
            self.assertFalse(record.policy_application_implemented)
            self.assertFalse(record.strategy_application_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.provider_execution_implemented)
            self.assertFalse(record.agent_invocation_implemented)
            self.assertFalse(record.resource_allocation_implemented)
            self.assertFalse(record.budget_enforcement_implemented)
            self.assertFalse(record.hitl_emission_implemented)
            self.assertFalse(record.workflow_control_implemented)
            self.assertFalse(record.workflow_graph_mutation_implemented)
            self.assertFalse(record.workflow_execution_implemented)
            self.assertFalse(record.retry_triggering_implemented)
            self.assertFalse(record.prompt_mutation_implemented)
            self.assertFalse(record.persistent_storage_write_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)
            self.assertFalse(record.runtime_evolution_implemented)
            self.assertTrue(record.advisory_only)

    def test_lookup_and_layer_filters_are_stable(self) -> None:
        registry = adaptive_execution_architecture_consistency_registry()
        strategy = adaptive_execution_architecture_consistency_by_surface(
            "adaptive_execution_strategy_selection",
            registry,
        )
        missing = adaptive_execution_architecture_consistency_by_surface(
            "missing",
            registry,
        )
        agent_records = adaptive_execution_architecture_consistency_records_for_layer(
            "agent_resource_boundary",
            registry,
        )
        creative_records = adaptive_execution_architecture_consistency_records_for_layer(
            "creative_adaptation_boundary",
            registry,
        )

        self.assertIsNone(missing)
        self.assertIsNotNone(strategy)
        assert strategy is not None
        self.assertEqual(strategy.source_role, "dynamic_execution_strategy_selector")
        self.assertEqual(strategy.source_count_field, "strategy_count")
        self.assertEqual(
            tuple(record.surface_id for record in agent_records),
            (
                "agent_activation_optimizer",
                "dynamic_agent_allocation",
                "dynamic_resource_allocation",
                "agent_diversity_optimizer",
            ),
        )
        self.assertEqual(
            tuple(record.surface_id for record in creative_records),
            ("creative_exploration_optimizer", "emergence_optimizer"),
        )

    def test_registry_rejects_mismatched_or_active_records(self) -> None:
        registry = adaptive_execution_architecture_consistency_registry()
        first_record = registry.records[0]
        duplicate_record = first_record.model_copy(
            update={"architecture_layer": "execution_strategy_boundary"}
        )
        missing_record = first_record.model_copy(
            update={"missing_coverage_items": ("serialization_version_missing",)}
        )
        active_record = first_record.model_copy(
            update={"source_active_runtime_flags": ("strategy_application_implemented",)}
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
            query="Generate adaptive execution architecture consistency metadata.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        adaptive_execution_architecture_consistency_registry()
        adaptive_execution_architecture_consistency_by_surface(
            "adaptive_policy_explainability"
        )
        adaptive_execution_architecture_consistency_records_for_layer(
            "confidence_risk_boundary"
        )
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)

    def test_architecture_metadata_does_not_declare_active_terms(self) -> None:
        registry = adaptive_execution_architecture_consistency_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *registry.validated_version_rules,
                *registry.passive_boundary_flags,
                *(
                    field
                    for record in registry.records
                    for field in (
                        record.surface_id,
                        record.architecture_layer,
                        record.source_role,
                        record.source_serialization_version,
                        *record.validated_version_rules,
                        *record.passive_boundary_flags,
                        *record.source_blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "apply_adaptive_policy(",
            "apply_strategy(",
            "route_provider(",
            "execute_provider(",
            "invoke_agent(",
            "allocate_resource(",
            "enforce_budget(",
            "emit_hitl_request(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "execute_workflow(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)

    def _registry_with_records(
        self,
        records: tuple[object, ...],
    ) -> AdaptiveExecutionArchitectureConsistencyRegistry:
        return AdaptiveExecutionArchitectureConsistencyRegistry(
            records=records,
            surface_ids=tuple(record.surface_id for record in records),
            record_count=len(records),
            architecture_layers=EXPECTED_ARCHITECTURE_LAYERS,
            validated_version_rules=(
                adaptive_execution_architecture_consistency_registry()
                .validated_version_rules
            ),
            passive_boundary_flags=(
                adaptive_execution_architecture_consistency_registry()
                .passive_boundary_flags
            ),
        )


if __name__ == "__main__":
    unittest.main()
