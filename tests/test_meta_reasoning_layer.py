import unittest

from creative_coding_assistant.orchestration import (
    MetaReasoningLayerPlan,
    build_cognitive_profile_engine,
    build_meta_reasoning_layer,
    meta_reasoning_assessment_by_id,
    meta_reasoning_assessments_for_agent,
    meta_reasoning_assessments_for_layer,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
)


class MetaReasoningLayerTests(unittest.TestCase):
    def test_meta_reasoning_layer_builds_read_only_assessments(self) -> None:
        profile_engine = build_cognitive_profile_engine()
        layer = build_meta_reasoning_layer(profile_engine=profile_engine)

        self.assertEqual(layer.role, "meta_reasoning_layer")
        self.assertEqual(layer.serialization_version, "meta_reasoning_layer.v1")
        self.assertEqual(layer.profile_engine_role, profile_engine.role)
        self.assertEqual(layer.state_engine_role, "cognitive_state_engine")
        self.assertEqual(layer.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(layer.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(layer.capability_ids, profile_engine.capability_ids)
        self.assertEqual(layer.capability_count, 6)
        self.assertEqual(layer.source_profile_ids, profile_engine.profile_ids)
        self.assertEqual(layer.source_profile_count, 6)
        self.assertEqual(layer.source_state_ids, profile_engine.source_state_ids)
        self.assertEqual(layer.source_state_count, 6)
        self.assertEqual(
            layer.source_optimization_signal_ids,
            profile_engine.source_optimization_signal_ids,
        )
        self.assertEqual(layer.source_optimization_signal_count, 6)
        self.assertEqual(
            layer.source_learning_signal_ids,
            profile_engine.source_learning_signal_ids,
        )
        self.assertEqual(layer.source_learning_signal_count, 6)
        self.assertEqual(len(layer.reasoning_assessments), 6)
        self.assertEqual(layer.reasoning_count, 6)
        self.assertEqual(layer.linked_agent_ids, profile_engine.linked_agent_ids)
        self.assertEqual(layer.covered_roadmap_items, ("Meta-Reasoning Layer",))
        self.assertEqual(layer.covered_roadmap_item_count, 1)
        self.assertEqual(layer.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            layer.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(layer.meta_reasoning_layer_implemented)
        self.assertTrue(layer.cognitive_profile_engine_integrated)
        self.assertTrue(layer.reasoning_assessment_contract_implemented)
        self.assertTrue(layer.reasoning_dependency_traceability_implemented)
        self.assertTrue(layer.reasoning_governance_contract_implemented)
        self.assertTrue(layer.reasoning_explainability_contract_implemented)
        self.assertFalse(layer.autonomous_reasoning_execution_implemented)
        self.assertFalse(layer.reasoning_chain_mutation_implemented)
        self.assertFalse(layer.decision_authority_implemented)
        self.assertFalse(layer.agent_routing_implemented)
        self.assertFalse(layer.provider_model_routing_implemented)
        self.assertFalse(layer.provider_execution_implemented)
        self.assertFalse(layer.workflow_control_implemented)
        self.assertFalse(layer.generated_output_mutation_implemented)
        self.assertFalse(layer.runtime_evolution_implemented)
        self.assertFalse(layer.executed_reasoning_ids)
        self.assertFalse(layer.mutated_reasoning_chain_ids)
        self.assertFalse(layer.adopted_decision_ids)
        self.assertFalse(layer.emitted_hitl_request_ids)
        self.assertTrue(layer.advisory_only)

    def test_meta_reasoning_lookup_helpers_are_layer_and_agent_aware(self) -> None:
        layer = build_meta_reasoning_layer()

        core_assessment = meta_reasoning_assessment_by_id(
            "meta_reasoning::v6_6_cognitive_core",
            layer,
        )
        self.assertIsNotNone(core_assessment)
        assert core_assessment is not None
        self.assertEqual(core_assessment.capability_name, "V6.6 Cognitive Core")
        self.assertEqual(core_assessment.cognitive_layer, "cognitive_core")
        self.assertIn("planner_agent", core_assessment.linked_agent_ids)
        self.assertEqual(
            core_assessment.profile_id,
            "cognitive_profile::v6_6_cognitive_core",
        )
        self.assertIn(
            "reasoning decisions",
            core_assessment.governance_contracts[0],
        )

        research_assessments = meta_reasoning_assessments_for_layer(
            "research",
            layer,
        )
        self.assertEqual(len(research_assessments), 1)
        self.assertEqual(
            research_assessments[0].capability_id,
            "v6_4_autonomous_research",
        )

        planner_assessments = meta_reasoning_assessments_for_agent(
            "planner_agent",
            layer,
        )
        self.assertEqual(
            tuple(assessment.capability_id for assessment in planner_assessments),
            ("v6_6_cognitive_core",),
        )
        self.assertIsNone(meta_reasoning_assessment_by_id("missing", layer))

    def test_meta_reasoning_layer_rejects_execution_and_drift(self) -> None:
        layer = build_meta_reasoning_layer()
        payload = layer.model_dump(mode="json")
        payload["reasoning_ids"] = ("missing",) + tuple(payload["reasoning_ids"][1:])

        with self.assertRaisesRegex(ValueError, "reasoning_ids must match"):
            MetaReasoningLayerPlan(**payload)

        payload = layer.model_dump(mode="json")
        payload["executed_reasoning_ids"] = ("meta_reasoning::v6_6_cognitive_core",)

        with self.assertRaisesRegex(
            ValueError,
            "reasoning execution, mutation, decisions, and HITL ids must be empty",
        ):
            MetaReasoningLayerPlan(**payload)

    def test_meta_reasoning_layer_reuses_supplied_profile_engine(self) -> None:
        profile_engine = build_cognitive_profile_engine(route="generate")
        layer = build_meta_reasoning_layer(profile_engine=profile_engine)

        self.assertEqual(layer.route_name, profile_engine.route_name)
        self.assertEqual(layer.task_type, profile_engine.task_type)
        self.assertEqual(layer.source_profile_ids, profile_engine.profile_ids)
        self.assertEqual(layer.source_state_ids, profile_engine.source_state_ids)
        self.assertEqual(
            layer.source_optimization_signal_ids,
            profile_engine.source_optimization_signal_ids,
        )
        self.assertEqual(
            layer.source_learning_signal_ids,
            profile_engine.source_learning_signal_ids,
        )


if __name__ == "__main__":
    unittest.main()
