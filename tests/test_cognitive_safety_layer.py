import unittest

from creative_coding_assistant.orchestration import (
    CognitiveSafetyLayerPlan,
    build_cognitive_explanation_engine,
    build_cognitive_safety_layer,
    cognitive_safety_boundaries_for_agent,
    cognitive_safety_boundaries_for_layer,
    cognitive_safety_boundaries_for_posture,
    cognitive_safety_boundary_by_id,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
)


class CognitiveSafetyLayerTests(unittest.TestCase):
    def test_cognitive_safety_layer_builds_read_only_boundaries(self) -> None:
        explanation = build_cognitive_explanation_engine()
        safety = build_cognitive_safety_layer(
            cognitive_explanation_engine=explanation,
        )

        self.assertEqual(safety.role, "cognitive_safety_layer")
        self.assertEqual(
            safety.serialization_version,
            "cognitive_safety_layer.v1",
        )
        self.assertEqual(
            safety.cognitive_explanation_engine_role,
            explanation.role,
        )
        self.assertEqual(
            safety.cognitive_explanation_engine_serialization_version,
            explanation.serialization_version,
        )
        self.assertEqual(
            safety.cognitive_blackboard_role,
            explanation.cognitive_blackboard_role,
        )
        self.assertEqual(
            safety.cognitive_router_role,
            explanation.cognitive_router_role,
        )
        self.assertEqual(
            safety.cognitive_planner_role,
            explanation.cognitive_planner_role,
        )
        self.assertEqual(
            safety.cognitive_scheduler_role,
            explanation.cognitive_scheduler_role,
        )
        self.assertEqual(safety.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(safety.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(safety.capability_ids, explanation.capability_ids)
        self.assertEqual(safety.capability_count, 6)
        self.assertEqual(safety.source_explanation_ids, explanation.explanation_ids)
        self.assertEqual(safety.source_explanation_count, 6)
        self.assertEqual(
            safety.source_blackboard_entry_ids,
            explanation.source_blackboard_entry_ids,
        )
        self.assertEqual(safety.source_blackboard_entry_count, 6)
        self.assertEqual(
            safety.source_route_decision_ids,
            explanation.source_route_decision_ids,
        )
        self.assertEqual(safety.source_route_decision_count, 6)
        self.assertEqual(safety.source_plan_ids, explanation.source_plan_ids)
        self.assertEqual(safety.source_plan_count, 6)
        self.assertEqual(
            safety.source_schedule_ids,
            explanation.source_schedule_ids,
        )
        self.assertEqual(safety.source_schedule_count, 6)
        self.assertEqual(
            safety.source_emergence_ids,
            explanation.source_emergence_ids,
        )
        self.assertEqual(safety.source_emergence_count, 6)
        self.assertEqual(len(safety.safety_boundaries), 6)
        self.assertEqual(safety.safety_count, 6)
        self.assertEqual(safety.candidate_safety_count, 0)
        self.assertEqual(safety.review_required_safety_count, 0)
        self.assertEqual(safety.guarded_safety_count, 6)
        self.assertEqual(safety.linked_agent_ids, explanation.linked_agent_ids)
        self.assertEqual(
            safety.covered_roadmap_items,
            ("Cognitive Safety Layer",),
        )
        self.assertEqual(safety.covered_roadmap_item_count, 1)
        self.assertEqual(safety.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            safety.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(safety.cognitive_safety_layer_implemented)
        self.assertTrue(safety.cognitive_explanation_engine_integrated)
        self.assertTrue(safety.safety_boundary_contract_implemented)
        self.assertTrue(safety.safety_dependency_traceability_implemented)
        self.assertTrue(safety.safety_explainability_contract_implemented)
        self.assertTrue(safety.safety_governance_contract_implemented)
        self.assertTrue(safety.safety_hitl_contract_implemented)
        self.assertFalse(safety.safety_enforcement_implemented)
        self.assertFalse(safety.workflow_blocking_implemented)
        self.assertFalse(safety.live_content_classification_implemented)
        self.assertFalse(safety.autonomous_workflow_planning_implemented)
        self.assertFalse(safety.routing_application_implemented)
        self.assertFalse(safety.prompt_mutation_implemented)
        self.assertFalse(safety.memory_mutation_implemented)
        self.assertFalse(safety.retrieval_mutation_implemented)
        self.assertFalse(safety.storage_mutation_implemented)
        self.assertFalse(safety.provider_model_routing_implemented)
        self.assertFalse(safety.provider_execution_implemented)
        self.assertFalse(safety.generated_output_mutation_implemented)
        self.assertFalse(safety.runtime_evolution_implemented)
        self.assertFalse(safety.enforced_safety_ids)
        self.assertFalse(safety.blocked_workflow_ids)
        self.assertFalse(safety.classified_content_ids)
        self.assertFalse(safety.mutated_safety_policy_ids)
        self.assertFalse(safety.emitted_hitl_request_ids)
        self.assertTrue(safety.advisory_only)

    def test_cognitive_safety_lookup_helpers_are_scope_aware(self) -> None:
        safety = build_cognitive_safety_layer()

        core_boundary = cognitive_safety_boundary_by_id(
            "cognitive_safety::v6_6_cognitive_core",
            safety,
        )
        self.assertIsNotNone(core_boundary)
        assert core_boundary is not None
        self.assertEqual(core_boundary.capability_name, "V6.6 Cognitive Core")
        self.assertEqual(core_boundary.cognitive_layer, "cognitive_core")
        self.assertIn("planner_agent", core_boundary.linked_agent_ids)
        self.assertEqual(core_boundary.safety_rank, 6)
        self.assertEqual(core_boundary.dependency_depth, 5)
        self.assertEqual(
            core_boundary.source_trace_ids[0],
            "cognitive_explanation::v6_6_cognitive_core",
        )
        self.assertIn(
            "cognitive_blackboard::v6_6_cognitive_core",
            core_boundary.source_trace_ids,
        )
        self.assertIn(
            "cognitive_router::v6_6_cognitive_core",
            core_boundary.source_trace_ids,
        )
        self.assertFalse(core_boundary.safety_enforcement_authorized)
        self.assertFalse(core_boundary.workflow_blocking_authorized)
        self.assertIn("does not enforce", core_boundary.governance_contracts[0])

        research_boundaries = cognitive_safety_boundaries_for_layer(
            "research",
            safety,
        )
        self.assertEqual(len(research_boundaries), 1)
        self.assertEqual(
            research_boundaries[0].capability_id,
            "v6_4_autonomous_research",
        )

        planner_boundaries = cognitive_safety_boundaries_for_agent(
            "planner_agent",
            safety,
        )
        self.assertEqual(
            tuple(boundary.capability_id for boundary in planner_boundaries),
            ("v6_6_cognitive_core",),
        )
        guarded_boundaries = cognitive_safety_boundaries_for_posture(
            "guarded",
            safety,
        )
        self.assertEqual(
            tuple(boundary.safety_id for boundary in guarded_boundaries),
            safety.guarded_safety_ids,
        )
        self.assertIsNone(cognitive_safety_boundary_by_id("missing", safety))

    def test_cognitive_safety_layer_rejects_enforcement_and_drift(self) -> None:
        safety = build_cognitive_safety_layer()
        payload = safety.model_dump(mode="json")
        payload["safety_ids"] = ("missing",) + tuple(payload["safety_ids"][1:])

        with self.assertRaisesRegex(ValueError, "safety_ids must match"):
            CognitiveSafetyLayerPlan(**payload)

        payload = safety.model_dump(mode="json")
        payload["enforced_safety_ids"] = (
            "cognitive_safety::v6_6_cognitive_core",
        )

        with self.assertRaisesRegex(
            ValueError,
            "safety enforcement, workflow blocking, classification, mutation, "
            "and HITL ids must be empty",
        ):
            CognitiveSafetyLayerPlan(**payload)

    def test_cognitive_safety_layer_reuses_supplied_explanation_engine(self) -> None:
        explanation = build_cognitive_explanation_engine(route="generate")
        safety = build_cognitive_safety_layer(
            cognitive_explanation_engine=explanation,
        )

        self.assertEqual(safety.route_name, explanation.route_name)
        self.assertEqual(safety.task_type, explanation.task_type)
        self.assertEqual(safety.source_explanation_ids, explanation.explanation_ids)
        self.assertEqual(
            safety.source_blackboard_entry_ids,
            explanation.source_blackboard_entry_ids,
        )
        self.assertEqual(
            safety.source_route_decision_ids,
            explanation.source_route_decision_ids,
        )
        self.assertEqual(safety.source_plan_ids, explanation.source_plan_ids)
        self.assertEqual(
            safety.source_schedule_ids,
            explanation.source_schedule_ids,
        )
        self.assertEqual(
            safety.source_emergence_ids,
            explanation.source_emergence_ids,
        )


if __name__ == "__main__":
    unittest.main()
