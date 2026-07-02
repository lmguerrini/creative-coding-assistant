import unittest

from creative_coding_assistant.orchestration import (
    EmergentCreativityLayerPlan,
    build_creative_identity_layer,
    build_emergent_creativity_layer,
    emergent_creativity_signal_by_id,
    emergent_creativity_signals_for_agent,
    emergent_creativity_signals_for_layer,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
)


class EmergentCreativityLayerTests(unittest.TestCase):
    def test_emergent_creativity_layer_builds_read_only_signals(self) -> None:
        identity_layer = build_creative_identity_layer()
        layer = build_emergent_creativity_layer(
            creative_identity_layer=identity_layer,
        )

        self.assertEqual(layer.role, "emergent_creativity_layer")
        self.assertEqual(layer.serialization_version, "emergent_creativity_layer.v1")
        self.assertEqual(layer.creative_identity_layer_role, identity_layer.role)
        self.assertEqual(
            layer.creative_cognition_layer_role,
            "creative_cognition_layer",
        )
        self.assertEqual(
            layer.cognitive_governance_layer_role,
            "cognitive_governance_layer",
        )
        self.assertEqual(layer.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(layer.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(layer.capability_ids, identity_layer.capability_ids)
        self.assertEqual(layer.capability_count, 6)
        self.assertEqual(layer.source_identity_ids, identity_layer.identity_ids)
        self.assertEqual(layer.source_identity_count, 6)
        self.assertEqual(
            layer.source_cognition_ids,
            identity_layer.source_cognition_ids,
        )
        self.assertEqual(layer.source_cognition_count, 6)
        self.assertEqual(
            layer.source_governance_ids,
            identity_layer.source_governance_ids,
        )
        self.assertEqual(layer.source_governance_count, 6)
        self.assertEqual(layer.source_planning_ids, identity_layer.source_planning_ids)
        self.assertEqual(layer.source_planning_count, 6)
        self.assertEqual(
            layer.source_reasoning_ids,
            identity_layer.source_reasoning_ids,
        )
        self.assertEqual(layer.source_reasoning_count, 6)
        self.assertEqual(layer.source_profile_ids, identity_layer.source_profile_ids)
        self.assertEqual(layer.source_profile_count, 6)
        self.assertEqual(layer.source_state_ids, identity_layer.source_state_ids)
        self.assertEqual(layer.source_state_count, 6)
        self.assertEqual(len(layer.emergent_creativity_signals), 6)
        self.assertEqual(layer.emergence_count, 6)
        self.assertEqual(layer.linked_agent_ids, identity_layer.linked_agent_ids)
        self.assertEqual(
            layer.covered_roadmap_items,
            ("Emergent Creativity Layer",),
        )
        self.assertEqual(layer.covered_roadmap_item_count, 1)
        self.assertEqual(layer.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            layer.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(layer.emergent_creativity_layer_implemented)
        self.assertTrue(layer.creative_identity_layer_integrated)
        self.assertTrue(layer.emergence_signal_contract_implemented)
        self.assertTrue(layer.emergence_dependency_traceability_implemented)
        self.assertTrue(layer.emergence_governance_contract_implemented)
        self.assertTrue(layer.emergence_explainability_contract_implemented)
        self.assertFalse(layer.emergent_generation_implemented)
        self.assertFalse(layer.autonomous_exploration_implemented)
        self.assertFalse(layer.identity_mutation_implemented)
        self.assertFalse(layer.prompt_mutation_implemented)
        self.assertFalse(layer.memory_mutation_implemented)
        self.assertFalse(layer.retrieval_mutation_implemented)
        self.assertFalse(layer.storage_mutation_implemented)
        self.assertFalse(layer.provider_model_routing_implemented)
        self.assertFalse(layer.provider_execution_implemented)
        self.assertFalse(layer.generated_output_mutation_implemented)
        self.assertFalse(layer.runtime_evolution_implemented)
        self.assertFalse(layer.generated_emergent_output_ids)
        self.assertFalse(layer.executed_emergence_ids)
        self.assertFalse(layer.mutated_emergence_policy_ids)
        self.assertFalse(layer.emitted_hitl_request_ids)
        self.assertTrue(layer.advisory_only)

    def test_emergent_creativity_lookup_helpers_are_layer_and_agent_aware(
        self,
    ) -> None:
        layer = build_emergent_creativity_layer()

        core_signal = emergent_creativity_signal_by_id(
            "emergent_creativity::v6_6_cognitive_core",
            layer,
        )
        self.assertIsNotNone(core_signal)
        assert core_signal is not None
        self.assertEqual(core_signal.capability_name, "V6.6 Cognitive Core")
        self.assertEqual(core_signal.cognitive_layer, "cognitive_core")
        self.assertIn("planner_agent", core_signal.linked_agent_ids)
        self.assertEqual(
            core_signal.identity_id,
            "creative_identity::v6_6_cognitive_core",
        )
        self.assertFalse(core_signal.emergence_execution_authorized)
        self.assertIn("generate output", core_signal.governance_contracts[0])

        research_signals = emergent_creativity_signals_for_layer("research", layer)
        self.assertEqual(len(research_signals), 1)
        self.assertEqual(
            research_signals[0].capability_id,
            "v6_4_autonomous_research",
        )

        planner_signals = emergent_creativity_signals_for_agent(
            "planner_agent",
            layer,
        )
        self.assertEqual(
            tuple(signal.capability_id for signal in planner_signals),
            ("v6_6_cognitive_core",),
        )
        self.assertIsNone(emergent_creativity_signal_by_id("missing", layer))

    def test_emergent_creativity_layer_rejects_execution_and_drift(self) -> None:
        layer = build_emergent_creativity_layer()
        payload = layer.model_dump(mode="json")
        payload["emergence_ids"] = ("missing",) + tuple(payload["emergence_ids"][1:])

        with self.assertRaisesRegex(ValueError, "emergence_ids must match"):
            EmergentCreativityLayerPlan(**payload)

        payload = layer.model_dump(mode="json")
        payload["generated_emergent_output_ids"] = (
            "emergent_creativity::v6_6_cognitive_core",
        )

        with self.assertRaisesRegex(
            ValueError,
            "emergent generation, execution, mutation, and HITL ids must be empty",
        ):
            EmergentCreativityLayerPlan(**payload)

    def test_emergent_creativity_layer_reuses_supplied_identity_layer(self) -> None:
        identity_layer = build_creative_identity_layer(route="generate")
        layer = build_emergent_creativity_layer(
            creative_identity_layer=identity_layer,
        )

        self.assertEqual(layer.route_name, identity_layer.route_name)
        self.assertEqual(layer.task_type, identity_layer.task_type)
        self.assertEqual(layer.source_identity_ids, identity_layer.identity_ids)
        self.assertEqual(
            layer.source_cognition_ids,
            identity_layer.source_cognition_ids,
        )
        self.assertEqual(
            layer.source_governance_ids,
            identity_layer.source_governance_ids,
        )
        self.assertEqual(
            layer.source_planning_ids,
            identity_layer.source_planning_ids,
        )
        self.assertEqual(
            layer.source_reasoning_ids,
            identity_layer.source_reasoning_ids,
        )
        self.assertEqual(layer.source_profile_ids, identity_layer.source_profile_ids)
        self.assertEqual(layer.source_state_ids, identity_layer.source_state_ids)


if __name__ == "__main__":
    unittest.main()
