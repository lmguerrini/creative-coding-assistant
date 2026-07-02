import unittest

from creative_coding_assistant.orchestration import (
    CrossSystemLearningLayerPlan,
    build_cross_system_learning_layer,
    build_unified_capability_registry,
    cross_system_learning_signal_by_id,
    cross_system_learning_signals_for_agent,
    cross_system_learning_signals_for_layer,
    evaluate_adaptive_learning_engine,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
)


class CrossSystemLearningLayerTests(unittest.TestCase):
    def test_cross_system_learning_layer_projects_adaptive_signals(self) -> None:
        capability_registry = build_unified_capability_registry()
        layer = build_cross_system_learning_layer(
            capability_registry=capability_registry,
        )

        self.assertEqual(layer.role, "cross_system_learning_layer")
        self.assertEqual(
            layer.serialization_version,
            "cross_system_learning_layer.v1",
        )
        self.assertEqual(layer.capability_registry_role, capability_registry.role)
        self.assertEqual(layer.adaptive_learning_role, "adaptive_learning_engine")
        self.assertEqual(layer.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(layer.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(layer.capability_ids, capability_registry.capability_ids)
        self.assertEqual(layer.capability_count, 6)
        self.assertEqual(layer.source_adaptive_signal_count, 5)
        self.assertEqual(layer.adaptive_hitl_required_signal_count, 5)
        self.assertEqual(len(layer.learning_signals), 6)
        self.assertEqual(layer.learning_signal_count, 6)
        self.assertEqual(layer.linked_agent_ids, capability_registry.agent_ids)
        self.assertEqual(
            layer.covered_roadmap_items,
            ("Cross-System Learning Layer",),
        )
        self.assertEqual(layer.covered_roadmap_item_count, 1)
        self.assertEqual(layer.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            layer.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(layer.cross_system_learning_layer_implemented)
        self.assertTrue(layer.unified_capability_registry_integrated)
        self.assertTrue(layer.adaptive_learning_engine_integrated)
        self.assertTrue(layer.cross_capability_learning_traceability_implemented)
        self.assertTrue(layer.learning_governance_contract_implemented)
        self.assertTrue(layer.learning_explainability_contract_implemented)
        self.assertFalse(layer.learning_memory_persistence_implemented)
        self.assertFalse(layer.learning_feedback_application_implemented)
        self.assertFalse(layer.learning_policy_mutation_implemented)
        self.assertFalse(layer.capability_activation_implemented)
        self.assertFalse(layer.agent_routing_implemented)
        self.assertFalse(layer.provider_model_routing_implemented)
        self.assertFalse(layer.provider_execution_implemented)
        self.assertFalse(layer.workflow_control_implemented)
        self.assertFalse(layer.generated_output_mutation_implemented)
        self.assertFalse(layer.runtime_evolution_implemented)
        self.assertFalse(layer.persisted_learning_signal_ids)
        self.assertFalse(layer.applied_learning_signal_ids)
        self.assertFalse(layer.mutated_learning_policy_ids)
        self.assertFalse(layer.activated_capability_ids)
        self.assertFalse(layer.emitted_hitl_request_ids)
        self.assertTrue(layer.advisory_only)

    def test_cross_system_learning_lookup_helpers_are_agent_aware(self) -> None:
        layer = build_cross_system_learning_layer()

        core_signal = cross_system_learning_signal_by_id(
            "cross_system_learning::v6_6_cognitive_core",
            layer,
        )
        self.assertIsNotNone(core_signal)
        assert core_signal is not None
        self.assertEqual(core_signal.capability_name, "V6.6 Cognitive Core")
        self.assertEqual(core_signal.cognitive_layer, "cognitive_core")
        self.assertIn("planner_agent", core_signal.linked_agent_ids)
        self.assertEqual(core_signal.source_adaptive_signal_count, 5)
        self.assertIn("feedback application", core_signal.governance_contracts[0])

        research_signals = cross_system_learning_signals_for_layer(
            "research",
            layer,
        )
        self.assertEqual(len(research_signals), 1)
        self.assertEqual(
            research_signals[0].capability_id,
            "v6_4_autonomous_research",
        )

        planner_signals = cross_system_learning_signals_for_agent(
            "planner_agent",
            layer,
        )
        self.assertEqual(
            tuple(signal.capability_id for signal in planner_signals),
            ("v6_6_cognitive_core",),
        )
        self.assertIsNone(cross_system_learning_signal_by_id("missing", layer))

    def test_cross_system_learning_layer_rejects_mutation_and_drift(self) -> None:
        layer = build_cross_system_learning_layer()
        payload = layer.model_dump(mode="json")
        payload["learning_signal_ids"] = ("missing",) + tuple(
            payload["learning_signal_ids"][1:]
        )

        with self.assertRaisesRegex(ValueError, "learning_signal_ids must match"):
            CrossSystemLearningLayerPlan(**payload)

        payload = layer.model_dump(mode="json")
        payload["applied_learning_signal_ids"] = (
            "cross_system_learning::v6_6_cognitive_core",
        )

        with self.assertRaisesRegex(
            ValueError,
            "learning persistence, application, mutation, and HITL ids must be empty",
        ):
            CrossSystemLearningLayerPlan(**payload)

    def test_cross_system_learning_layer_reuses_supplied_sources(self) -> None:
        capability_registry = build_unified_capability_registry(route="generate")
        adaptive_learning = evaluate_adaptive_learning_engine(
            route="generate",
            task_type=capability_registry.task_type,
        )
        layer = build_cross_system_learning_layer(
            capability_registry=capability_registry,
            adaptive_learning=adaptive_learning,
        )

        self.assertEqual(layer.route_name, capability_registry.route_name)
        self.assertEqual(layer.task_type, capability_registry.task_type)
        self.assertEqual(layer.capability_ids, capability_registry.capability_ids)
        self.assertEqual(
            layer.source_adaptive_signal_ids,
            adaptive_learning.signal_ids,
        )
        self.assertEqual(
            layer.adaptive_hitl_required_signal_ids,
            adaptive_learning.hitl_required_signal_ids,
        )


if __name__ == "__main__":
    unittest.main()
