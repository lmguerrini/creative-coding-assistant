import unittest

from creative_coding_assistant.orchestration import (
    CognitivePlannerPlan,
    build_cognitive_planner,
    build_cognitive_scheduler,
    cognitive_plan_step_by_id,
    cognitive_plan_steps_for_agent,
    cognitive_plan_steps_for_layer,
    cognitive_plan_steps_for_posture,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
)


class CognitivePlannerTests(unittest.TestCase):
    def test_cognitive_planner_builds_read_only_steps(self) -> None:
        scheduler = build_cognitive_scheduler()
        planner = build_cognitive_planner(cognitive_scheduler=scheduler)

        self.assertEqual(planner.role, "cognitive_planner")
        self.assertEqual(planner.serialization_version, "cognitive_planner.v1")
        self.assertEqual(planner.cognitive_scheduler_role, scheduler.role)
        self.assertEqual(
            planner.cognitive_scheduler_serialization_version,
            scheduler.serialization_version,
        )
        self.assertEqual(
            planner.emergent_creativity_layer_role,
            "emergent_creativity_layer",
        )
        self.assertEqual(
            planner.creative_identity_layer_role,
            "creative_identity_layer",
        )
        self.assertEqual(
            planner.creative_cognition_layer_role,
            "creative_cognition_layer",
        )
        self.assertEqual(
            planner.cognitive_governance_layer_role,
            "cognitive_governance_layer",
        )
        self.assertEqual(planner.meta_planning_layer_role, "meta_planning_layer")
        self.assertEqual(planner.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(planner.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(planner.capability_ids, scheduler.capability_ids)
        self.assertEqual(planner.capability_count, 6)
        self.assertEqual(planner.source_schedule_ids, scheduler.schedule_ids)
        self.assertEqual(planner.source_schedule_count, 6)
        self.assertEqual(planner.source_emergence_ids, scheduler.source_emergence_ids)
        self.assertEqual(planner.source_emergence_count, 6)
        self.assertEqual(planner.source_identity_ids, scheduler.source_identity_ids)
        self.assertEqual(planner.source_identity_count, 6)
        self.assertEqual(planner.source_cognition_ids, scheduler.source_cognition_ids)
        self.assertEqual(planner.source_cognition_count, 6)
        self.assertEqual(
            planner.source_governance_ids,
            scheduler.source_governance_ids,
        )
        self.assertEqual(planner.source_governance_count, 6)
        self.assertEqual(planner.source_planning_ids, scheduler.source_planning_ids)
        self.assertEqual(planner.source_planning_count, 6)
        self.assertEqual(planner.source_reasoning_ids, scheduler.source_reasoning_ids)
        self.assertEqual(planner.source_reasoning_count, 6)
        self.assertEqual(planner.source_profile_ids, scheduler.source_profile_ids)
        self.assertEqual(planner.source_profile_count, 6)
        self.assertEqual(planner.source_state_ids, scheduler.source_state_ids)
        self.assertEqual(planner.source_state_count, 6)
        self.assertEqual(len(planner.plan_steps), 6)
        self.assertEqual(planner.plan_count, 6)
        self.assertEqual(planner.candidate_plan_count, 0)
        self.assertEqual(planner.review_required_plan_count, 0)
        self.assertEqual(planner.guarded_plan_count, 6)
        self.assertEqual(planner.max_dependency_depth, 5)
        self.assertEqual(planner.linked_agent_ids, scheduler.linked_agent_ids)
        self.assertEqual(planner.covered_roadmap_items, ("Cognitive Planner",))
        self.assertEqual(planner.covered_roadmap_item_count, 1)
        self.assertEqual(planner.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            planner.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(planner.cognitive_planner_implemented)
        self.assertTrue(planner.cognitive_scheduler_integrated)
        self.assertTrue(planner.plan_step_contract_implemented)
        self.assertTrue(planner.plan_dependency_traceability_implemented)
        self.assertTrue(planner.plan_governance_contract_implemented)
        self.assertTrue(planner.plan_explainability_contract_implemented)
        self.assertFalse(planner.autonomous_planning_implemented)
        self.assertFalse(planner.plan_execution_implemented)
        self.assertFalse(planner.plan_mutation_implemented)
        self.assertFalse(planner.workflow_control_implemented)
        self.assertFalse(planner.workflow_graph_mutation_implemented)
        self.assertFalse(planner.routing_mutation_implemented)
        self.assertFalse(planner.agent_invocation_implemented)
        self.assertFalse(planner.prompt_mutation_implemented)
        self.assertFalse(planner.memory_mutation_implemented)
        self.assertFalse(planner.retrieval_mutation_implemented)
        self.assertFalse(planner.storage_mutation_implemented)
        self.assertFalse(planner.provider_model_routing_implemented)
        self.assertFalse(planner.provider_execution_implemented)
        self.assertFalse(planner.generated_output_mutation_implemented)
        self.assertFalse(planner.runtime_evolution_implemented)
        self.assertFalse(planner.executed_plan_ids)
        self.assertFalse(planner.mutated_plan_ids)
        self.assertFalse(planner.routed_plan_ids)
        self.assertFalse(planner.emitted_hitl_request_ids)
        self.assertTrue(planner.advisory_only)

    def test_cognitive_planner_lookup_helpers_are_scope_aware(self) -> None:
        planner = build_cognitive_planner()

        core_step = cognitive_plan_step_by_id(
            "cognitive_planner::v6_6_cognitive_core",
            planner,
        )
        self.assertIsNotNone(core_step)
        assert core_step is not None
        self.assertEqual(core_step.capability_name, "V6.6 Cognitive Core")
        self.assertEqual(core_step.cognitive_layer, "cognitive_core")
        self.assertIn("planner_agent", core_step.linked_agent_ids)
        self.assertEqual(core_step.plan_rank, 6)
        self.assertEqual(core_step.dependency_depth, 5)
        self.assertEqual(
            core_step.upstream_plan_ids,
            ("cognitive_planner::v6_5_self_evolution",),
        )
        self.assertFalse(core_step.downstream_plan_ids)
        self.assertFalse(core_step.execution_plan_authorized)
        self.assertIn("execute plans", core_step.governance_contracts[0])

        research_steps = cognitive_plan_steps_for_layer("research", planner)
        self.assertEqual(len(research_steps), 1)
        self.assertEqual(research_steps[0].capability_id, "v6_4_autonomous_research")

        planner_steps = cognitive_plan_steps_for_agent("planner_agent", planner)
        self.assertEqual(
            tuple(step.capability_id for step in planner_steps),
            ("v6_6_cognitive_core",),
        )
        guarded_steps = cognitive_plan_steps_for_posture("guarded", planner)
        self.assertEqual(
            tuple(step.plan_id for step in guarded_steps),
            planner.guarded_plan_ids,
        )
        self.assertIsNone(cognitive_plan_step_by_id("missing", planner))

    def test_cognitive_planner_rejects_execution_and_drift(self) -> None:
        planner = build_cognitive_planner()
        payload = planner.model_dump(mode="json")
        payload["plan_ids"] = ("missing",) + tuple(payload["plan_ids"][1:])

        with self.assertRaisesRegex(ValueError, "plan_ids must match"):
            CognitivePlannerPlan(**payload)

        payload = planner.model_dump(mode="json")
        payload["executed_plan_ids"] = ("cognitive_planner::v6_6_cognitive_core",)

        with self.assertRaisesRegex(
            ValueError,
            "plan execution, mutation, routing, and HITL ids must be empty",
        ):
            CognitivePlannerPlan(**payload)

    def test_cognitive_planner_reuses_supplied_scheduler(self) -> None:
        scheduler = build_cognitive_scheduler(route="generate")
        planner = build_cognitive_planner(cognitive_scheduler=scheduler)

        self.assertEqual(planner.route_name, scheduler.route_name)
        self.assertEqual(planner.task_type, scheduler.task_type)
        self.assertEqual(planner.source_schedule_ids, scheduler.schedule_ids)
        self.assertEqual(planner.source_emergence_ids, scheduler.source_emergence_ids)
        self.assertEqual(planner.source_identity_ids, scheduler.source_identity_ids)
        self.assertEqual(planner.source_cognition_ids, scheduler.source_cognition_ids)
        self.assertEqual(
            planner.source_governance_ids,
            scheduler.source_governance_ids,
        )
        self.assertEqual(planner.source_planning_ids, scheduler.source_planning_ids)
        self.assertEqual(planner.source_reasoning_ids, scheduler.source_reasoning_ids)
        self.assertEqual(planner.source_profile_ids, scheduler.source_profile_ids)
        self.assertEqual(planner.source_state_ids, scheduler.source_state_ids)


if __name__ == "__main__":
    unittest.main()
