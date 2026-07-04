import unittest

from creative_coding_assistant.orchestration import (
    CognitiveHITLLayerPlan,
    build_cognitive_hitl_layer,
    build_cognitive_safety_layer,
    cognitive_hitl_checkpoint_by_id,
    cognitive_hitl_checkpoints_for_agent,
    cognitive_hitl_checkpoints_for_layer,
    cognitive_hitl_checkpoints_for_posture,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
)


class CognitiveHITLLayerTests(unittest.TestCase):
    def test_cognitive_hitl_layer_builds_read_only_checkpoints(self) -> None:
        safety = build_cognitive_safety_layer()
        hitl = build_cognitive_hitl_layer(cognitive_safety_layer=safety)

        self.assertEqual(hitl.role, "cognitive_hitl_layer")
        self.assertEqual(hitl.serialization_version, "cognitive_hitl_layer.v1")
        self.assertEqual(hitl.cognitive_safety_layer_role, safety.role)
        self.assertEqual(
            hitl.cognitive_safety_layer_serialization_version,
            safety.serialization_version,
        )
        self.assertEqual(
            hitl.cognitive_explanation_engine_role,
            safety.cognitive_explanation_engine_role,
        )
        self.assertEqual(
            hitl.cognitive_blackboard_role,
            safety.cognitive_blackboard_role,
        )
        self.assertEqual(hitl.cognitive_router_role, safety.cognitive_router_role)
        self.assertEqual(hitl.cognitive_planner_role, safety.cognitive_planner_role)
        self.assertEqual(
            hitl.cognitive_scheduler_role,
            safety.cognitive_scheduler_role,
        )
        self.assertEqual(hitl.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(hitl.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(hitl.capability_ids, safety.capability_ids)
        self.assertEqual(hitl.capability_count, 6)
        self.assertEqual(hitl.source_safety_ids, safety.safety_ids)
        self.assertEqual(hitl.source_safety_count, 6)
        self.assertEqual(hitl.source_explanation_ids, safety.source_explanation_ids)
        self.assertEqual(hitl.source_explanation_count, 6)
        self.assertEqual(
            hitl.source_blackboard_entry_ids,
            safety.source_blackboard_entry_ids,
        )
        self.assertEqual(hitl.source_blackboard_entry_count, 6)
        self.assertEqual(
            hitl.source_route_decision_ids,
            safety.source_route_decision_ids,
        )
        self.assertEqual(hitl.source_route_decision_count, 6)
        self.assertEqual(hitl.source_plan_ids, safety.source_plan_ids)
        self.assertEqual(hitl.source_plan_count, 6)
        self.assertEqual(hitl.source_schedule_ids, safety.source_schedule_ids)
        self.assertEqual(hitl.source_schedule_count, 6)
        self.assertEqual(hitl.source_emergence_ids, safety.source_emergence_ids)
        self.assertEqual(hitl.source_emergence_count, 6)
        self.assertEqual(len(hitl.hitl_checkpoints), 6)
        self.assertEqual(hitl.hitl_count, 6)
        self.assertEqual(hitl.hitl_required_ids, hitl.hitl_ids)
        self.assertEqual(hitl.hitl_required_count, 6)
        self.assertEqual(hitl.candidate_hitl_count, 0)
        self.assertEqual(hitl.review_required_hitl_count, 0)
        self.assertEqual(hitl.guarded_hitl_count, 6)
        self.assertEqual(hitl.linked_agent_ids, safety.linked_agent_ids)
        self.assertEqual(hitl.covered_roadmap_items, ("Cognitive HITL Layer",))
        self.assertEqual(hitl.covered_roadmap_item_count, 1)
        self.assertEqual(hitl.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            hitl.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(hitl.cognitive_hitl_layer_implemented)
        self.assertTrue(hitl.cognitive_safety_layer_integrated)
        self.assertTrue(hitl.hitl_checkpoint_contract_implemented)
        self.assertTrue(hitl.hitl_dependency_traceability_implemented)
        self.assertTrue(hitl.hitl_explainability_contract_implemented)
        self.assertTrue(hitl.hitl_safety_contract_implemented)
        self.assertTrue(hitl.hitl_governance_contract_implemented)
        self.assertFalse(hitl.hitl_request_emission_implemented)
        self.assertFalse(hitl.hitl_decision_application_implemented)
        self.assertFalse(hitl.safety_enforcement_implemented)
        self.assertFalse(hitl.workflow_blocking_implemented)
        self.assertFalse(hitl.autonomous_workflow_planning_implemented)
        self.assertFalse(hitl.routing_application_implemented)
        self.assertFalse(hitl.prompt_mutation_implemented)
        self.assertFalse(hitl.memory_mutation_implemented)
        self.assertFalse(hitl.retrieval_mutation_implemented)
        self.assertFalse(hitl.storage_mutation_implemented)
        self.assertFalse(hitl.provider_model_routing_implemented)
        self.assertFalse(hitl.provider_execution_implemented)
        self.assertFalse(hitl.generated_output_mutation_implemented)
        self.assertFalse(hitl.runtime_evolution_implemented)
        self.assertFalse(hitl.emitted_hitl_request_ids)
        self.assertFalse(hitl.applied_hitl_decision_ids)
        self.assertFalse(hitl.enforced_safety_ids)
        self.assertFalse(hitl.blocked_workflow_ids)
        self.assertFalse(hitl.mutated_hitl_policy_ids)
        self.assertTrue(hitl.advisory_only)

    def test_cognitive_hitl_lookup_helpers_are_scope_aware(self) -> None:
        hitl = build_cognitive_hitl_layer()

        core_checkpoint = cognitive_hitl_checkpoint_by_id(
            "cognitive_hitl::v6_6_cognitive_core",
            hitl,
        )
        self.assertIsNotNone(core_checkpoint)
        assert core_checkpoint is not None
        self.assertEqual(core_checkpoint.capability_name, "V6.6 Cognitive Core")
        self.assertEqual(core_checkpoint.cognitive_layer, "cognitive_core")
        self.assertIn("planner_agent", core_checkpoint.linked_agent_ids)
        self.assertEqual(core_checkpoint.hitl_rank, 6)
        self.assertEqual(core_checkpoint.dependency_depth, 5)
        self.assertEqual(
            core_checkpoint.source_trace_ids[0],
            "cognitive_safety::v6_6_cognitive_core",
        )
        self.assertIn(
            "cognitive_explanation::v6_6_cognitive_core",
            core_checkpoint.source_trace_ids,
        )
        self.assertIn(
            "cognitive_blackboard::v6_6_cognitive_core",
            core_checkpoint.source_trace_ids,
        )
        self.assertTrue(core_checkpoint.hitl_required_before_application)
        self.assertFalse(core_checkpoint.hitl_request_emission_authorized)
        self.assertFalse(core_checkpoint.hitl_decision_application_authorized)
        self.assertIn("does not apply", core_checkpoint.governance_contracts[0])

        research_checkpoints = cognitive_hitl_checkpoints_for_layer(
            "research",
            hitl,
        )
        self.assertEqual(len(research_checkpoints), 1)
        self.assertEqual(
            research_checkpoints[0].capability_id,
            "v6_4_autonomous_research",
        )

        planner_checkpoints = cognitive_hitl_checkpoints_for_agent(
            "planner_agent",
            hitl,
        )
        self.assertEqual(
            tuple(checkpoint.capability_id for checkpoint in planner_checkpoints),
            ("v6_6_cognitive_core",),
        )
        guarded_checkpoints = cognitive_hitl_checkpoints_for_posture(
            "guarded",
            hitl,
        )
        self.assertEqual(
            tuple(checkpoint.hitl_id for checkpoint in guarded_checkpoints),
            hitl.guarded_hitl_ids,
        )
        self.assertIsNone(cognitive_hitl_checkpoint_by_id("missing", hitl))

    def test_cognitive_hitl_layer_rejects_emission_and_drift(self) -> None:
        hitl = build_cognitive_hitl_layer()
        payload = hitl.model_dump(mode="json")
        payload["hitl_ids"] = ("missing",) + tuple(payload["hitl_ids"][1:])

        with self.assertRaisesRegex(ValueError, "hitl_ids must match"):
            CognitiveHITLLayerPlan(**payload)

        payload = hitl.model_dump(mode="json")
        payload["emitted_hitl_request_ids"] = ("cognitive_hitl::v6_6_cognitive_core",)

        with self.assertRaisesRegex(
            ValueError,
            "HITL request emission, decision application, safety enforcement, "
            "workflow blocking, and mutation ids must be empty",
        ):
            CognitiveHITLLayerPlan(**payload)

    def test_cognitive_hitl_layer_reuses_supplied_safety_layer(self) -> None:
        safety = build_cognitive_safety_layer(route="generate")
        hitl = build_cognitive_hitl_layer(cognitive_safety_layer=safety)

        self.assertEqual(hitl.route_name, safety.route_name)
        self.assertEqual(hitl.task_type, safety.task_type)
        self.assertEqual(hitl.source_safety_ids, safety.safety_ids)
        self.assertEqual(hitl.source_explanation_ids, safety.source_explanation_ids)
        self.assertEqual(
            hitl.source_blackboard_entry_ids,
            safety.source_blackboard_entry_ids,
        )
        self.assertEqual(
            hitl.source_route_decision_ids,
            safety.source_route_decision_ids,
        )
        self.assertEqual(hitl.source_plan_ids, safety.source_plan_ids)
        self.assertEqual(hitl.source_schedule_ids, safety.source_schedule_ids)
        self.assertEqual(hitl.source_emergence_ids, safety.source_emergence_ids)


if __name__ == "__main__":
    unittest.main()
