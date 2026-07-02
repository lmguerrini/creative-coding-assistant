import unittest

from creative_coding_assistant.orchestration import (
    MetaPlanningLayerPlan,
    build_meta_planning_layer,
    build_meta_reasoning_layer,
    meta_planning_projection_by_id,
    meta_planning_projections_for_agent,
    meta_planning_projections_for_layer,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
)


class MetaPlanningLayerTests(unittest.TestCase):
    def test_meta_planning_layer_builds_read_only_projections(self) -> None:
        reasoning_layer = build_meta_reasoning_layer()
        layer = build_meta_planning_layer(meta_reasoning_layer=reasoning_layer)

        self.assertEqual(layer.role, "meta_planning_layer")
        self.assertEqual(layer.serialization_version, "meta_planning_layer.v1")
        self.assertEqual(layer.meta_reasoning_layer_role, reasoning_layer.role)
        self.assertEqual(layer.profile_engine_role, "cognitive_profile_engine")
        self.assertEqual(layer.state_engine_role, "cognitive_state_engine")
        self.assertEqual(layer.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(layer.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(layer.capability_ids, reasoning_layer.capability_ids)
        self.assertEqual(layer.capability_count, 6)
        self.assertEqual(layer.source_reasoning_ids, reasoning_layer.reasoning_ids)
        self.assertEqual(layer.source_reasoning_count, 6)
        self.assertEqual(layer.source_profile_ids, reasoning_layer.source_profile_ids)
        self.assertEqual(layer.source_profile_count, 6)
        self.assertEqual(layer.source_state_ids, reasoning_layer.source_state_ids)
        self.assertEqual(layer.source_state_count, 6)
        self.assertEqual(
            layer.source_optimization_signal_ids,
            reasoning_layer.source_optimization_signal_ids,
        )
        self.assertEqual(layer.source_optimization_signal_count, 6)
        self.assertEqual(
            layer.source_learning_signal_ids,
            reasoning_layer.source_learning_signal_ids,
        )
        self.assertEqual(layer.source_learning_signal_count, 6)
        self.assertEqual(len(layer.planning_projections), 6)
        self.assertEqual(layer.planning_count, 6)
        self.assertEqual(layer.linked_agent_ids, reasoning_layer.linked_agent_ids)
        self.assertEqual(layer.covered_roadmap_items, ("Meta-Planning Layer",))
        self.assertEqual(layer.covered_roadmap_item_count, 1)
        self.assertEqual(layer.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            layer.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(layer.meta_planning_layer_implemented)
        self.assertTrue(layer.meta_reasoning_layer_integrated)
        self.assertTrue(layer.planning_projection_contract_implemented)
        self.assertTrue(layer.planning_dependency_traceability_implemented)
        self.assertTrue(layer.planning_governance_contract_implemented)
        self.assertTrue(layer.planning_explainability_contract_implemented)
        self.assertFalse(layer.autonomous_workflow_planning_implemented)
        self.assertFalse(layer.planning_execution_implemented)
        self.assertFalse(layer.plan_mutation_implemented)
        self.assertFalse(layer.prompt_mutation_implemented)
        self.assertFalse(layer.memory_mutation_implemented)
        self.assertFalse(layer.retrieval_mutation_implemented)
        self.assertFalse(layer.storage_mutation_implemented)
        self.assertFalse(layer.provider_model_routing_implemented)
        self.assertFalse(layer.provider_execution_implemented)
        self.assertFalse(layer.workflow_control_implemented)
        self.assertFalse(layer.generated_output_mutation_implemented)
        self.assertFalse(layer.runtime_evolution_implemented)
        self.assertFalse(layer.executed_planning_ids)
        self.assertFalse(layer.mutated_plan_ids)
        self.assertFalse(layer.routed_planning_ids)
        self.assertFalse(layer.emitted_hitl_request_ids)
        self.assertTrue(layer.advisory_only)

    def test_meta_planning_lookup_helpers_are_layer_and_agent_aware(self) -> None:
        layer = build_meta_planning_layer()

        core_projection = meta_planning_projection_by_id(
            "meta_planning::v6_6_cognitive_core",
            layer,
        )
        self.assertIsNotNone(core_projection)
        assert core_projection is not None
        self.assertEqual(core_projection.capability_name, "V6.6 Cognitive Core")
        self.assertEqual(core_projection.cognitive_layer, "cognitive_core")
        self.assertIn("planner_agent", core_projection.linked_agent_ids)
        self.assertEqual(
            core_projection.reasoning_id,
            "meta_reasoning::v6_6_cognitive_core",
        )
        self.assertIn(
            "autonomous workflows",
            core_projection.governance_contracts[0],
        )

        research_projections = meta_planning_projections_for_layer(
            "research",
            layer,
        )
        self.assertEqual(len(research_projections), 1)
        self.assertEqual(
            research_projections[0].capability_id,
            "v6_4_autonomous_research",
        )

        planner_projections = meta_planning_projections_for_agent(
            "planner_agent",
            layer,
        )
        self.assertEqual(
            tuple(projection.capability_id for projection in planner_projections),
            ("v6_6_cognitive_core",),
        )
        self.assertIsNone(meta_planning_projection_by_id("missing", layer))

    def test_meta_planning_layer_rejects_execution_and_drift(self) -> None:
        layer = build_meta_planning_layer()
        payload = layer.model_dump(mode="json")
        payload["planning_ids"] = ("missing",) + tuple(payload["planning_ids"][1:])

        with self.assertRaisesRegex(ValueError, "planning_ids must match"):
            MetaPlanningLayerPlan(**payload)

        payload = layer.model_dump(mode="json")
        payload["mutated_plan_ids"] = ("meta_planning::v6_6_cognitive_core",)

        with self.assertRaisesRegex(
            ValueError,
            "planning execution, mutation, routing, and HITL ids must be empty",
        ):
            MetaPlanningLayerPlan(**payload)

    def test_meta_planning_layer_reuses_supplied_reasoning_layer(self) -> None:
        reasoning_layer = build_meta_reasoning_layer(route="generate")
        layer = build_meta_planning_layer(meta_reasoning_layer=reasoning_layer)

        self.assertEqual(layer.route_name, reasoning_layer.route_name)
        self.assertEqual(layer.task_type, reasoning_layer.task_type)
        self.assertEqual(layer.source_reasoning_ids, reasoning_layer.reasoning_ids)
        self.assertEqual(layer.source_profile_ids, reasoning_layer.source_profile_ids)
        self.assertEqual(layer.source_state_ids, reasoning_layer.source_state_ids)
        self.assertEqual(
            layer.source_optimization_signal_ids,
            reasoning_layer.source_optimization_signal_ids,
        )
        self.assertEqual(
            layer.source_learning_signal_ids,
            reasoning_layer.source_learning_signal_ids,
        )


if __name__ == "__main__":
    unittest.main()
