import unittest

from creative_coding_assistant.orchestration import (
    CognitiveGovernanceLayerPlan,
    build_cognitive_governance_layer,
    build_meta_planning_layer,
    cognitive_governance_policies_for_agent,
    cognitive_governance_policies_for_layer,
    cognitive_governance_policy_by_id,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
)


class CognitiveGovernanceLayerTests(unittest.TestCase):
    def test_cognitive_governance_layer_builds_read_only_policies(self) -> None:
        planning_layer = build_meta_planning_layer()
        layer = build_cognitive_governance_layer(
            meta_planning_layer=planning_layer,
        )

        self.assertEqual(layer.role, "cognitive_governance_layer")
        self.assertEqual(
            layer.serialization_version,
            "cognitive_governance_layer.v1",
        )
        self.assertEqual(layer.meta_planning_layer_role, planning_layer.role)
        self.assertEqual(layer.meta_reasoning_layer_role, "meta_reasoning_layer")
        self.assertEqual(layer.profile_engine_role, "cognitive_profile_engine")
        self.assertEqual(layer.state_engine_role, "cognitive_state_engine")
        self.assertEqual(layer.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(layer.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(layer.capability_ids, planning_layer.capability_ids)
        self.assertEqual(layer.capability_count, 6)
        self.assertEqual(layer.source_planning_ids, planning_layer.planning_ids)
        self.assertEqual(layer.source_planning_count, 6)
        self.assertEqual(
            layer.source_reasoning_ids,
            planning_layer.source_reasoning_ids,
        )
        self.assertEqual(layer.source_reasoning_count, 6)
        self.assertEqual(layer.source_profile_ids, planning_layer.source_profile_ids)
        self.assertEqual(layer.source_profile_count, 6)
        self.assertEqual(layer.source_state_ids, planning_layer.source_state_ids)
        self.assertEqual(layer.source_state_count, 6)
        self.assertEqual(
            layer.source_optimization_signal_ids,
            planning_layer.source_optimization_signal_ids,
        )
        self.assertEqual(layer.source_optimization_signal_count, 6)
        self.assertEqual(
            layer.source_learning_signal_ids,
            planning_layer.source_learning_signal_ids,
        )
        self.assertEqual(layer.source_learning_signal_count, 6)
        self.assertEqual(len(layer.governance_policies), 6)
        self.assertEqual(layer.governance_count, 6)
        self.assertEqual(layer.linked_agent_ids, planning_layer.linked_agent_ids)
        self.assertEqual(
            layer.covered_roadmap_items,
            ("Cognitive Governance Layer",),
        )
        self.assertEqual(layer.covered_roadmap_item_count, 1)
        self.assertEqual(layer.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            layer.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(layer.cognitive_governance_layer_implemented)
        self.assertTrue(layer.meta_planning_layer_integrated)
        self.assertTrue(layer.governance_policy_contract_implemented)
        self.assertTrue(layer.governance_dependency_traceability_implemented)
        self.assertTrue(layer.governance_explainability_contract_implemented)
        self.assertTrue(layer.governance_hitl_contract_implemented)
        self.assertFalse(layer.policy_enforcement_implemented)
        self.assertFalse(layer.workflow_blocking_implemented)
        self.assertFalse(layer.autonomous_workflow_planning_implemented)
        self.assertFalse(layer.plan_mutation_implemented)
        self.assertFalse(layer.prompt_mutation_implemented)
        self.assertFalse(layer.memory_mutation_implemented)
        self.assertFalse(layer.retrieval_mutation_implemented)
        self.assertFalse(layer.storage_mutation_implemented)
        self.assertFalse(layer.provider_model_routing_implemented)
        self.assertFalse(layer.provider_execution_implemented)
        self.assertFalse(layer.generated_output_mutation_implemented)
        self.assertFalse(layer.runtime_evolution_implemented)
        self.assertFalse(layer.enforced_governance_ids)
        self.assertFalse(layer.blocked_workflow_ids)
        self.assertFalse(layer.mutated_governance_policy_ids)
        self.assertFalse(layer.emitted_hitl_request_ids)
        self.assertTrue(layer.advisory_only)

    def test_governance_lookup_helpers_are_layer_and_agent_aware(self) -> None:
        layer = build_cognitive_governance_layer()

        core_policy = cognitive_governance_policy_by_id(
            "cognitive_governance::v6_6_cognitive_core",
            layer,
        )
        self.assertIsNotNone(core_policy)
        assert core_policy is not None
        self.assertEqual(core_policy.capability_name, "V6.6 Cognitive Core")
        self.assertEqual(core_policy.cognitive_layer, "cognitive_core")
        self.assertIn("planner_agent", core_policy.linked_agent_ids)
        self.assertEqual(
            core_policy.planning_id,
            "meta_planning::v6_6_cognitive_core",
        )
        self.assertTrue(core_policy.hitl_required_before_application)
        self.assertIn(
            "does not enforce",
            core_policy.governance_contracts[0],
        )

        research_policies = cognitive_governance_policies_for_layer(
            "research",
            layer,
        )
        self.assertEqual(len(research_policies), 1)
        self.assertEqual(
            research_policies[0].capability_id,
            "v6_4_autonomous_research",
        )

        planner_policies = cognitive_governance_policies_for_agent(
            "planner_agent",
            layer,
        )
        self.assertEqual(
            tuple(policy.capability_id for policy in planner_policies),
            ("v6_6_cognitive_core",),
        )
        self.assertIsNone(cognitive_governance_policy_by_id("missing", layer))

    def test_governance_layer_rejects_enforcement_and_drift(self) -> None:
        layer = build_cognitive_governance_layer()
        payload = layer.model_dump(mode="json")
        payload["governance_ids"] = (
            "missing",
        ) + tuple(payload["governance_ids"][1:])

        with self.assertRaisesRegex(ValueError, "governance_ids must match"):
            CognitiveGovernanceLayerPlan(**payload)

        payload = layer.model_dump(mode="json")
        payload["enforced_governance_ids"] = (
            "cognitive_governance::v6_6_cognitive_core",
        )

        with self.assertRaisesRegex(
            ValueError,
            "governance enforcement, workflow blocking, mutation, and HITL ids",
        ):
            CognitiveGovernanceLayerPlan(**payload)

    def test_governance_layer_reuses_supplied_planning_layer(self) -> None:
        planning_layer = build_meta_planning_layer(route="generate")
        layer = build_cognitive_governance_layer(
            meta_planning_layer=planning_layer,
        )

        self.assertEqual(layer.route_name, planning_layer.route_name)
        self.assertEqual(layer.task_type, planning_layer.task_type)
        self.assertEqual(layer.source_planning_ids, planning_layer.planning_ids)
        self.assertEqual(
            layer.source_reasoning_ids,
            planning_layer.source_reasoning_ids,
        )
        self.assertEqual(layer.source_profile_ids, planning_layer.source_profile_ids)
        self.assertEqual(layer.source_state_ids, planning_layer.source_state_ids)
        self.assertEqual(
            layer.source_optimization_signal_ids,
            planning_layer.source_optimization_signal_ids,
        )
        self.assertEqual(
            layer.source_learning_signal_ids,
            planning_layer.source_learning_signal_ids,
        )


if __name__ == "__main__":
    unittest.main()
