import unittest

from creative_coding_assistant.orchestration import (
    CognitiveRouterPlan,
    build_cognitive_planner,
    build_cognitive_router,
    cognitive_route_decision_by_id,
    cognitive_route_decisions_for_agent,
    cognitive_route_decisions_for_layer,
    cognitive_route_decisions_for_posture,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
)


class CognitiveRouterTests(unittest.TestCase):
    def test_cognitive_router_builds_read_only_decisions(self) -> None:
        planner = build_cognitive_planner()
        router = build_cognitive_router(cognitive_planner=planner)

        self.assertEqual(router.role, "cognitive_router")
        self.assertEqual(router.serialization_version, "cognitive_router.v1")
        self.assertEqual(router.cognitive_planner_role, planner.role)
        self.assertEqual(
            router.cognitive_planner_serialization_version,
            planner.serialization_version,
        )
        self.assertEqual(router.cognitive_scheduler_role, "cognitive_scheduler")
        self.assertEqual(
            router.emergent_creativity_layer_role,
            "emergent_creativity_layer",
        )
        self.assertEqual(router.creative_identity_layer_role, "creative_identity_layer")
        self.assertEqual(
            router.creative_cognition_layer_role,
            "creative_cognition_layer",
        )
        self.assertEqual(
            router.cognitive_governance_layer_role,
            "cognitive_governance_layer",
        )
        self.assertEqual(router.meta_planning_layer_role, "meta_planning_layer")
        self.assertEqual(router.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(router.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(router.capability_ids, planner.capability_ids)
        self.assertEqual(router.capability_count, 6)
        self.assertEqual(router.source_plan_ids, planner.plan_ids)
        self.assertEqual(router.source_plan_count, 6)
        self.assertEqual(router.source_schedule_ids, planner.source_schedule_ids)
        self.assertEqual(router.source_schedule_count, 6)
        self.assertEqual(router.source_emergence_ids, planner.source_emergence_ids)
        self.assertEqual(router.source_emergence_count, 6)
        self.assertEqual(router.source_identity_ids, planner.source_identity_ids)
        self.assertEqual(router.source_identity_count, 6)
        self.assertEqual(router.source_cognition_ids, planner.source_cognition_ids)
        self.assertEqual(router.source_cognition_count, 6)
        self.assertEqual(router.source_governance_ids, planner.source_governance_ids)
        self.assertEqual(router.source_governance_count, 6)
        self.assertEqual(router.source_planning_ids, planner.source_planning_ids)
        self.assertEqual(router.source_planning_count, 6)
        self.assertEqual(router.source_reasoning_ids, planner.source_reasoning_ids)
        self.assertEqual(router.source_reasoning_count, 6)
        self.assertEqual(router.source_profile_ids, planner.source_profile_ids)
        self.assertEqual(router.source_profile_count, 6)
        self.assertEqual(router.source_state_ids, planner.source_state_ids)
        self.assertEqual(router.source_state_count, 6)
        self.assertEqual(len(router.route_decisions), 6)
        self.assertEqual(router.route_decision_count, 6)
        self.assertEqual(router.candidate_route_decision_count, 0)
        self.assertEqual(router.review_required_route_decision_count, 0)
        self.assertEqual(router.guarded_route_decision_count, 6)
        self.assertEqual(router.max_dependency_depth, 5)
        self.assertEqual(router.linked_agent_ids, planner.linked_agent_ids)
        self.assertEqual(router.covered_roadmap_items, ("Cognitive Router",))
        self.assertEqual(router.covered_roadmap_item_count, 1)
        self.assertEqual(router.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            router.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(router.cognitive_router_implemented)
        self.assertTrue(router.cognitive_planner_integrated)
        self.assertTrue(router.route_decision_contract_implemented)
        self.assertTrue(router.route_dependency_traceability_implemented)
        self.assertTrue(router.route_governance_contract_implemented)
        self.assertTrue(router.route_explainability_contract_implemented)
        self.assertFalse(router.request_routing_implemented)
        self.assertFalse(router.provider_model_routing_implemented)
        self.assertFalse(router.workflow_routing_implemented)
        self.assertFalse(router.routing_mutation_implemented)
        self.assertFalse(router.plan_execution_implemented)
        self.assertFalse(router.workflow_control_implemented)
        self.assertFalse(router.workflow_graph_mutation_implemented)
        self.assertFalse(router.agent_invocation_implemented)
        self.assertFalse(router.prompt_mutation_implemented)
        self.assertFalse(router.memory_mutation_implemented)
        self.assertFalse(router.retrieval_mutation_implemented)
        self.assertFalse(router.storage_mutation_implemented)
        self.assertFalse(router.provider_execution_implemented)
        self.assertFalse(router.generated_output_mutation_implemented)
        self.assertFalse(router.runtime_evolution_implemented)
        self.assertFalse(router.applied_route_decision_ids)
        self.assertFalse(router.executed_route_decision_ids)
        self.assertFalse(router.mutated_route_decision_ids)
        self.assertFalse(router.emitted_hitl_request_ids)
        self.assertTrue(router.advisory_only)

    def test_cognitive_router_lookup_helpers_are_scope_aware(self) -> None:
        router = build_cognitive_router()

        core_decision = cognitive_route_decision_by_id(
            "cognitive_router::v6_6_cognitive_core",
            router,
        )
        self.assertIsNotNone(core_decision)
        assert core_decision is not None
        self.assertEqual(core_decision.capability_name, "V6.6 Cognitive Core")
        self.assertEqual(core_decision.cognitive_layer, "cognitive_core")
        self.assertIn("planner_agent", core_decision.linked_agent_ids)
        self.assertEqual(core_decision.route_rank, 6)
        self.assertEqual(core_decision.dependency_depth, 5)
        self.assertEqual(
            core_decision.upstream_route_decision_ids,
            ("cognitive_router::v6_5_self_evolution",),
        )
        self.assertFalse(core_decision.downstream_route_decision_ids)
        self.assertFalse(core_decision.routing_application_authorized)
        self.assertIn("route requests", core_decision.governance_contracts[0])

        research_decisions = cognitive_route_decisions_for_layer(
            "research",
            router,
        )
        self.assertEqual(len(research_decisions), 1)
        self.assertEqual(
            research_decisions[0].capability_id,
            "v6_4_autonomous_research",
        )

        planner_decisions = cognitive_route_decisions_for_agent(
            "planner_agent",
            router,
        )
        self.assertEqual(
            tuple(decision.capability_id for decision in planner_decisions),
            ("v6_6_cognitive_core",),
        )
        guarded_decisions = cognitive_route_decisions_for_posture("guarded", router)
        self.assertEqual(
            tuple(decision.route_decision_id for decision in guarded_decisions),
            router.guarded_route_decision_ids,
        )
        self.assertIsNone(cognitive_route_decision_by_id("missing", router))

    def test_cognitive_router_rejects_application_and_drift(self) -> None:
        router = build_cognitive_router()
        payload = router.model_dump(mode="json")
        payload["route_decision_ids"] = ("missing",) + tuple(
            payload["route_decision_ids"][1:]
        )

        with self.assertRaisesRegex(ValueError, "route_decision_ids must match"):
            CognitiveRouterPlan(**payload)

        payload = router.model_dump(mode="json")
        payload["applied_route_decision_ids"] = (
            "cognitive_router::v6_6_cognitive_core",
        )

        with self.assertRaisesRegex(
            ValueError,
            "route application, execution, mutation, and HITL ids must be empty",
        ):
            CognitiveRouterPlan(**payload)

    def test_cognitive_router_reuses_supplied_planner(self) -> None:
        planner = build_cognitive_planner(route="generate")
        router = build_cognitive_router(cognitive_planner=planner)

        self.assertEqual(router.route_name, planner.route_name)
        self.assertEqual(router.task_type, planner.task_type)
        self.assertEqual(router.source_plan_ids, planner.plan_ids)
        self.assertEqual(router.source_schedule_ids, planner.source_schedule_ids)
        self.assertEqual(router.source_emergence_ids, planner.source_emergence_ids)
        self.assertEqual(router.source_identity_ids, planner.source_identity_ids)
        self.assertEqual(router.source_cognition_ids, planner.source_cognition_ids)
        self.assertEqual(router.source_governance_ids, planner.source_governance_ids)
        self.assertEqual(router.source_planning_ids, planner.source_planning_ids)
        self.assertEqual(router.source_reasoning_ids, planner.source_reasoning_ids)
        self.assertEqual(router.source_profile_ids, planner.source_profile_ids)
        self.assertEqual(router.source_state_ids, planner.source_state_ids)


if __name__ == "__main__":
    unittest.main()
