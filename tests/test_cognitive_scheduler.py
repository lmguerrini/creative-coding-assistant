import unittest

from creative_coding_assistant.orchestration import (
    CognitiveSchedulerPlan,
    build_cognitive_scheduler,
    build_emergent_creativity_layer,
    cognitive_schedule_slot_by_id,
    cognitive_schedule_slots_for_agent,
    cognitive_schedule_slots_for_layer,
    cognitive_schedule_slots_for_posture,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
)


class CognitiveSchedulerTests(unittest.TestCase):
    def test_cognitive_scheduler_builds_read_only_slots(self) -> None:
        emergence_layer = build_emergent_creativity_layer()
        scheduler = build_cognitive_scheduler(
            emergent_creativity_layer=emergence_layer,
        )

        self.assertEqual(scheduler.role, "cognitive_scheduler")
        self.assertEqual(scheduler.serialization_version, "cognitive_scheduler.v1")
        self.assertEqual(
            scheduler.emergent_creativity_layer_role,
            emergence_layer.role,
        )
        self.assertEqual(
            scheduler.creative_identity_layer_role,
            "creative_identity_layer",
        )
        self.assertEqual(
            scheduler.creative_cognition_layer_role,
            "creative_cognition_layer",
        )
        self.assertEqual(
            scheduler.cognitive_governance_layer_role,
            "cognitive_governance_layer",
        )
        self.assertEqual(scheduler.meta_planning_layer_role, "meta_planning_layer")
        self.assertEqual(scheduler.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(scheduler.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(scheduler.capability_ids, emergence_layer.capability_ids)
        self.assertEqual(scheduler.capability_count, 6)
        self.assertEqual(
            scheduler.source_emergence_ids,
            emergence_layer.emergence_ids,
        )
        self.assertEqual(scheduler.source_emergence_count, 6)
        self.assertEqual(
            scheduler.source_identity_ids,
            emergence_layer.source_identity_ids,
        )
        self.assertEqual(scheduler.source_identity_count, 6)
        self.assertEqual(
            scheduler.source_cognition_ids,
            emergence_layer.source_cognition_ids,
        )
        self.assertEqual(scheduler.source_cognition_count, 6)
        self.assertEqual(
            scheduler.source_governance_ids,
            emergence_layer.source_governance_ids,
        )
        self.assertEqual(scheduler.source_governance_count, 6)
        self.assertEqual(
            scheduler.source_planning_ids,
            emergence_layer.source_planning_ids,
        )
        self.assertEqual(scheduler.source_planning_count, 6)
        self.assertEqual(
            scheduler.source_reasoning_ids,
            emergence_layer.source_reasoning_ids,
        )
        self.assertEqual(scheduler.source_reasoning_count, 6)
        self.assertEqual(
            scheduler.source_profile_ids,
            emergence_layer.source_profile_ids,
        )
        self.assertEqual(scheduler.source_profile_count, 6)
        self.assertEqual(scheduler.source_state_ids, emergence_layer.source_state_ids)
        self.assertEqual(scheduler.source_state_count, 6)
        self.assertEqual(len(scheduler.schedule_slots), 6)
        self.assertEqual(scheduler.schedule_count, 6)
        self.assertEqual(scheduler.candidate_schedule_count, 0)
        self.assertEqual(scheduler.review_required_schedule_count, 0)
        self.assertEqual(scheduler.guarded_schedule_count, 6)
        self.assertEqual(scheduler.max_dependency_depth, 5)
        self.assertEqual(scheduler.linked_agent_ids, emergence_layer.linked_agent_ids)
        self.assertEqual(scheduler.covered_roadmap_items, ("Cognitive Scheduler",))
        self.assertEqual(scheduler.covered_roadmap_item_count, 1)
        self.assertEqual(scheduler.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            scheduler.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(scheduler.cognitive_scheduler_implemented)
        self.assertTrue(scheduler.emergent_creativity_layer_integrated)
        self.assertTrue(scheduler.schedule_slot_contract_implemented)
        self.assertTrue(scheduler.schedule_dependency_traceability_implemented)
        self.assertTrue(scheduler.schedule_governance_contract_implemented)
        self.assertTrue(scheduler.schedule_explainability_contract_implemented)
        self.assertFalse(scheduler.runtime_scheduling_implemented)
        self.assertFalse(scheduler.autonomous_workflow_scheduling_implemented)
        self.assertFalse(scheduler.parallel_execution_implemented)
        self.assertFalse(scheduler.async_execution_implemented)
        self.assertFalse(scheduler.workflow_timing_mutation_implemented)
        self.assertFalse(scheduler.workflow_graph_mutation_implemented)
        self.assertFalse(scheduler.agent_invocation_implemented)
        self.assertFalse(scheduler.prompt_mutation_implemented)
        self.assertFalse(scheduler.memory_mutation_implemented)
        self.assertFalse(scheduler.retrieval_mutation_implemented)
        self.assertFalse(scheduler.storage_mutation_implemented)
        self.assertFalse(scheduler.provider_model_routing_implemented)
        self.assertFalse(scheduler.provider_execution_implemented)
        self.assertFalse(scheduler.generated_output_mutation_implemented)
        self.assertFalse(scheduler.runtime_evolution_implemented)
        self.assertFalse(scheduler.executed_schedule_ids)
        self.assertFalse(scheduler.mutated_schedule_ids)
        self.assertFalse(scheduler.routed_schedule_ids)
        self.assertFalse(scheduler.emitted_hitl_request_ids)
        self.assertTrue(scheduler.advisory_only)

    def test_cognitive_scheduler_lookup_helpers_are_scope_aware(self) -> None:
        scheduler = build_cognitive_scheduler()

        core_slot = cognitive_schedule_slot_by_id(
            "cognitive_scheduler::v6_6_cognitive_core",
            scheduler,
        )
        self.assertIsNotNone(core_slot)
        assert core_slot is not None
        self.assertEqual(core_slot.capability_name, "V6.6 Cognitive Core")
        self.assertEqual(core_slot.cognitive_layer, "cognitive_core")
        self.assertIn("planner_agent", core_slot.linked_agent_ids)
        self.assertEqual(core_slot.schedule_rank, 6)
        self.assertEqual(core_slot.dependency_depth, 5)
        self.assertEqual(
            core_slot.upstream_schedule_ids,
            ("cognitive_scheduler::v6_5_self_evolution",),
        )
        self.assertFalse(core_slot.downstream_schedule_ids)
        self.assertFalse(core_slot.execution_schedule_authorized)
        self.assertIn("runtime tasks", core_slot.governance_contracts[0])

        research_slots = cognitive_schedule_slots_for_layer("research", scheduler)
        self.assertEqual(len(research_slots), 1)
        self.assertEqual(research_slots[0].capability_id, "v6_4_autonomous_research")

        planner_slots = cognitive_schedule_slots_for_agent(
            "planner_agent",
            scheduler,
        )
        self.assertEqual(
            tuple(slot.capability_id for slot in planner_slots),
            ("v6_6_cognitive_core",),
        )
        guarded_slots = cognitive_schedule_slots_for_posture("guarded", scheduler)
        self.assertEqual(
            tuple(slot.schedule_id for slot in guarded_slots),
            scheduler.guarded_schedule_ids,
        )
        self.assertIsNone(cognitive_schedule_slot_by_id("missing", scheduler))

    def test_cognitive_scheduler_rejects_execution_and_drift(self) -> None:
        scheduler = build_cognitive_scheduler()
        payload = scheduler.model_dump(mode="json")
        payload["schedule_ids"] = ("missing",) + tuple(payload["schedule_ids"][1:])

        with self.assertRaisesRegex(ValueError, "schedule_ids must match"):
            CognitiveSchedulerPlan(**payload)

        payload = scheduler.model_dump(mode="json")
        payload["executed_schedule_ids"] = (
            "cognitive_scheduler::v6_6_cognitive_core",
        )

        with self.assertRaisesRegex(
            ValueError,
            "schedule execution, mutation, routing, and HITL ids must be empty",
        ):
            CognitiveSchedulerPlan(**payload)

    def test_cognitive_scheduler_reuses_supplied_emergence_layer(self) -> None:
        emergence_layer = build_emergent_creativity_layer(route="generate")
        scheduler = build_cognitive_scheduler(
            emergent_creativity_layer=emergence_layer,
        )

        self.assertEqual(scheduler.route_name, emergence_layer.route_name)
        self.assertEqual(scheduler.task_type, emergence_layer.task_type)
        self.assertEqual(
            scheduler.source_emergence_ids,
            emergence_layer.emergence_ids,
        )
        self.assertEqual(
            scheduler.source_identity_ids,
            emergence_layer.source_identity_ids,
        )
        self.assertEqual(
            scheduler.source_cognition_ids,
            emergence_layer.source_cognition_ids,
        )
        self.assertEqual(
            scheduler.source_governance_ids,
            emergence_layer.source_governance_ids,
        )
        self.assertEqual(
            scheduler.source_planning_ids,
            emergence_layer.source_planning_ids,
        )
        self.assertEqual(
            scheduler.source_reasoning_ids,
            emergence_layer.source_reasoning_ids,
        )
        self.assertEqual(
            scheduler.source_profile_ids,
            emergence_layer.source_profile_ids,
        )
        self.assertEqual(scheduler.source_state_ids, emergence_layer.source_state_ids)


if __name__ == "__main__":
    unittest.main()
