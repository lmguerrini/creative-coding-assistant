import unittest

from creative_coding_assistant.orchestration import (
    CreativeCognitionLayerPlan,
    build_cognitive_governance_layer,
    build_creative_cognition_layer,
    creative_cognition_signal_by_id,
    creative_cognition_signals_for_agent,
    creative_cognition_signals_for_layer,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
)


class CreativeCognitionLayerTests(unittest.TestCase):
    def test_creative_cognition_layer_builds_read_only_signals(self) -> None:
        governance_layer = build_cognitive_governance_layer()
        layer = build_creative_cognition_layer(
            cognitive_governance_layer=governance_layer,
        )

        self.assertEqual(layer.role, "creative_cognition_layer")
        self.assertEqual(layer.serialization_version, "creative_cognition_layer.v1")
        self.assertEqual(
            layer.cognitive_governance_layer_role,
            governance_layer.role,
        )
        self.assertEqual(layer.meta_planning_layer_role, "meta_planning_layer")
        self.assertEqual(layer.meta_reasoning_layer_role, "meta_reasoning_layer")
        self.assertEqual(layer.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(layer.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(layer.capability_ids, governance_layer.capability_ids)
        self.assertEqual(layer.capability_count, 6)
        self.assertEqual(layer.source_governance_ids, governance_layer.governance_ids)
        self.assertEqual(layer.source_governance_count, 6)
        self.assertEqual(
            layer.source_planning_ids,
            governance_layer.source_planning_ids,
        )
        self.assertEqual(layer.source_planning_count, 6)
        self.assertEqual(
            layer.source_reasoning_ids,
            governance_layer.source_reasoning_ids,
        )
        self.assertEqual(layer.source_reasoning_count, 6)
        self.assertEqual(layer.source_profile_ids, governance_layer.source_profile_ids)
        self.assertEqual(layer.source_profile_count, 6)
        self.assertEqual(layer.source_state_ids, governance_layer.source_state_ids)
        self.assertEqual(layer.source_state_count, 6)
        self.assertEqual(len(layer.creative_cognition_signals), 6)
        self.assertEqual(layer.cognition_count, 6)
        self.assertEqual(layer.linked_agent_ids, governance_layer.linked_agent_ids)
        self.assertEqual(
            layer.covered_roadmap_items,
            ("Creative Cognition Layer",),
        )
        self.assertEqual(layer.covered_roadmap_item_count, 1)
        self.assertEqual(layer.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            layer.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(layer.creative_cognition_layer_implemented)
        self.assertTrue(layer.cognitive_governance_layer_integrated)
        self.assertTrue(layer.creative_cognition_contract_implemented)
        self.assertTrue(layer.creative_dependency_traceability_implemented)
        self.assertTrue(layer.creative_governance_contract_implemented)
        self.assertTrue(layer.creative_explainability_contract_implemented)
        self.assertFalse(layer.creative_generation_implemented)
        self.assertFalse(layer.exploration_execution_implemented)
        self.assertFalse(layer.prompt_mutation_implemented)
        self.assertFalse(layer.memory_mutation_implemented)
        self.assertFalse(layer.retrieval_mutation_implemented)
        self.assertFalse(layer.storage_mutation_implemented)
        self.assertFalse(layer.provider_model_routing_implemented)
        self.assertFalse(layer.provider_execution_implemented)
        self.assertFalse(layer.generated_output_mutation_implemented)
        self.assertFalse(layer.runtime_evolution_implemented)
        self.assertFalse(layer.generated_creative_output_ids)
        self.assertFalse(layer.executed_exploration_ids)
        self.assertFalse(layer.mutated_creative_policy_ids)
        self.assertFalse(layer.emitted_hitl_request_ids)
        self.assertTrue(layer.advisory_only)

    def test_creative_cognition_lookup_helpers_are_layer_and_agent_aware(
        self,
    ) -> None:
        layer = build_creative_cognition_layer()

        core_signal = creative_cognition_signal_by_id(
            "creative_cognition::v6_6_cognitive_core",
            layer,
        )
        self.assertIsNotNone(core_signal)
        assert core_signal is not None
        self.assertEqual(core_signal.capability_name, "V6.6 Cognitive Core")
        self.assertEqual(core_signal.cognitive_layer, "cognitive_core")
        self.assertIn("planner_agent", core_signal.linked_agent_ids)
        self.assertEqual(
            core_signal.governance_id,
            "cognitive_governance::v6_6_cognitive_core",
        )
        self.assertFalse(core_signal.exploration_authorized)
        self.assertIn("generate creative output", core_signal.governance_contracts[0])

        research_signals = creative_cognition_signals_for_layer("research", layer)
        self.assertEqual(len(research_signals), 1)
        self.assertEqual(
            research_signals[0].capability_id,
            "v6_4_autonomous_research",
        )

        planner_signals = creative_cognition_signals_for_agent(
            "planner_agent",
            layer,
        )
        self.assertEqual(
            tuple(signal.capability_id for signal in planner_signals),
            ("v6_6_cognitive_core",),
        )
        self.assertIsNone(creative_cognition_signal_by_id("missing", layer))

    def test_creative_cognition_layer_rejects_generation_and_drift(self) -> None:
        layer = build_creative_cognition_layer()
        payload = layer.model_dump(mode="json")
        payload["cognition_ids"] = ("missing",) + tuple(payload["cognition_ids"][1:])

        with self.assertRaisesRegex(ValueError, "cognition_ids must match"):
            CreativeCognitionLayerPlan(**payload)

        payload = layer.model_dump(mode="json")
        payload["generated_creative_output_ids"] = (
            "creative_cognition::v6_6_cognitive_core",
        )

        with self.assertRaisesRegex(
            ValueError,
            "creative generation, exploration, mutation, and HITL ids must be empty",
        ):
            CreativeCognitionLayerPlan(**payload)

    def test_creative_cognition_layer_reuses_supplied_governance_layer(self) -> None:
        governance_layer = build_cognitive_governance_layer(route="generate")
        layer = build_creative_cognition_layer(
            cognitive_governance_layer=governance_layer,
        )

        self.assertEqual(layer.route_name, governance_layer.route_name)
        self.assertEqual(layer.task_type, governance_layer.task_type)
        self.assertEqual(layer.source_governance_ids, governance_layer.governance_ids)
        self.assertEqual(
            layer.source_planning_ids,
            governance_layer.source_planning_ids,
        )
        self.assertEqual(
            layer.source_reasoning_ids,
            governance_layer.source_reasoning_ids,
        )
        self.assertEqual(layer.source_profile_ids, governance_layer.source_profile_ids)
        self.assertEqual(layer.source_state_ids, governance_layer.source_state_ids)


if __name__ == "__main__":
    unittest.main()
