import unittest

from creative_coding_assistant.orchestration import (
    CrossSystemOptimizationLayerPlan,
    build_autonomous_optimization_suggestions,
    build_cross_system_learning_layer,
    build_cross_system_optimization_layer,
    cross_system_optimization_signal_by_id,
    cross_system_optimization_signals_for_agent,
    cross_system_optimization_signals_for_layer,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
)


class CrossSystemOptimizationLayerTests(unittest.TestCase):
    def test_cross_system_optimization_layer_projects_v6_5_suggestions(self) -> None:
        learning_layer = build_cross_system_learning_layer()
        optimization_layer = build_cross_system_optimization_layer(
            learning_layer=learning_layer,
        )

        self.assertEqual(optimization_layer.role, "cross_system_optimization_layer")
        self.assertEqual(
            optimization_layer.serialization_version,
            "cross_system_optimization_layer.v1",
        )
        self.assertEqual(optimization_layer.learning_layer_role, learning_layer.role)
        self.assertEqual(
            optimization_layer.optimization_source_role,
            "autonomous_optimization_suggestions",
        )
        self.assertEqual(optimization_layer.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(optimization_layer.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(
            optimization_layer.source_learning_signal_ids,
            learning_layer.learning_signal_ids,
        )
        self.assertEqual(optimization_layer.source_learning_signal_count, 6)
        self.assertEqual(optimization_layer.source_optimization_proposal_count, 5)
        self.assertEqual(
            optimization_layer.optimization_hitl_required_proposal_count,
            5,
        )
        self.assertEqual(len(optimization_layer.optimization_signals), 6)
        self.assertEqual(optimization_layer.optimization_signal_count, 6)
        self.assertEqual(
            optimization_layer.linked_agent_ids,
            learning_layer.linked_agent_ids,
        )
        self.assertEqual(
            optimization_layer.covered_roadmap_items,
            ("Cross-System Optimization Layer",),
        )
        self.assertEqual(optimization_layer.covered_roadmap_item_count, 1)
        self.assertEqual(
            optimization_layer.cross_cutting_contracts,
            COGNITIVE_OS_CONTRACTS,
        )
        self.assertEqual(
            optimization_layer.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(optimization_layer.cross_system_optimization_layer_implemented)
        self.assertTrue(optimization_layer.cross_system_learning_layer_integrated)
        self.assertTrue(
            optimization_layer.autonomous_optimization_suggestions_integrated,
        )
        self.assertTrue(
            optimization_layer.optimization_dependency_traceability_implemented,
        )
        self.assertTrue(optimization_layer.optimization_governance_contract_implemented)
        self.assertTrue(
            optimization_layer.optimization_explainability_contract_implemented,
        )
        self.assertFalse(optimization_layer.optimization_application_implemented)
        self.assertFalse(optimization_layer.evolution_proposal_application_implemented)
        self.assertFalse(optimization_layer.optimization_policy_mutation_implemented)
        self.assertFalse(optimization_layer.capability_activation_implemented)
        self.assertFalse(optimization_layer.agent_routing_implemented)
        self.assertFalse(optimization_layer.provider_model_routing_implemented)
        self.assertFalse(optimization_layer.provider_execution_implemented)
        self.assertFalse(optimization_layer.workflow_control_implemented)
        self.assertFalse(optimization_layer.generated_output_mutation_implemented)
        self.assertFalse(optimization_layer.runtime_evolution_implemented)
        self.assertFalse(optimization_layer.applied_optimization_signal_ids)
        self.assertFalse(optimization_layer.applied_evolution_proposal_ids)
        self.assertFalse(optimization_layer.mutated_optimization_policy_ids)
        self.assertFalse(optimization_layer.activated_capability_ids)
        self.assertFalse(optimization_layer.emitted_hitl_request_ids)
        self.assertTrue(optimization_layer.advisory_only)

    def test_cross_system_optimization_lookup_helpers_are_agent_aware(self) -> None:
        optimization_layer = build_cross_system_optimization_layer()

        core_signal = cross_system_optimization_signal_by_id(
            "cross_system_optimization::v6_6_cognitive_core",
            optimization_layer,
        )
        self.assertIsNotNone(core_signal)
        assert core_signal is not None
        self.assertEqual(core_signal.capability_name, "V6.6 Cognitive Core")
        self.assertEqual(core_signal.cognitive_layer, "cognitive_core")
        self.assertIn("planner_agent", core_signal.linked_agent_ids)
        self.assertEqual(core_signal.source_optimization_proposal_count, 5)
        self.assertIn("optimization application", core_signal.governance_contracts[0])

        research_signals = cross_system_optimization_signals_for_layer(
            "research",
            optimization_layer,
        )
        self.assertEqual(len(research_signals), 1)
        self.assertEqual(
            research_signals[0].capability_id,
            "v6_4_autonomous_research",
        )

        planner_signals = cross_system_optimization_signals_for_agent(
            "planner_agent",
            optimization_layer,
        )
        self.assertEqual(
            tuple(signal.capability_id for signal in planner_signals),
            ("v6_6_cognitive_core",),
        )
        self.assertIsNone(
            cross_system_optimization_signal_by_id("missing", optimization_layer),
        )

    def test_cross_system_optimization_layer_rejects_application_and_drift(
        self,
    ) -> None:
        optimization_layer = build_cross_system_optimization_layer()
        payload = optimization_layer.model_dump(mode="json")
        payload["optimization_signal_ids"] = ("missing",) + tuple(
            payload["optimization_signal_ids"][1:]
        )

        with self.assertRaisesRegex(ValueError, "optimization_signal_ids must match"):
            CrossSystemOptimizationLayerPlan(**payload)

        payload = optimization_layer.model_dump(mode="json")
        payload["applied_optimization_signal_ids"] = (
            "cross_system_optimization::v6_6_cognitive_core",
        )

        with self.assertRaisesRegex(
            ValueError,
            "optimization application, proposal application, mutation, and HITL ids",
        ):
            CrossSystemOptimizationLayerPlan(**payload)

    def test_cross_system_optimization_layer_reuses_supplied_sources(self) -> None:
        learning_layer = build_cross_system_learning_layer(route="generate")
        optimization_source = build_autonomous_optimization_suggestions(
            route="generate",
            task_type=learning_layer.task_type,
        )
        optimization_layer = build_cross_system_optimization_layer(
            learning_layer=learning_layer,
            optimization_source=optimization_source,
        )

        self.assertEqual(optimization_layer.route_name, learning_layer.route_name)
        self.assertEqual(optimization_layer.task_type, learning_layer.task_type)
        self.assertEqual(
            optimization_layer.source_learning_signal_ids,
            learning_layer.learning_signal_ids,
        )
        self.assertEqual(
            optimization_layer.source_optimization_proposal_ids,
            optimization_source.proposal_ids,
        )
        self.assertEqual(
            optimization_layer.optimization_hitl_required_proposal_ids,
            optimization_source.hitl_required_proposal_ids,
        )


if __name__ == "__main__":
    unittest.main()
