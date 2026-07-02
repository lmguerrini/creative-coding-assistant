import unittest

from creative_coding_assistant.orchestration import (
    CognitiveBlackboardPlan,
    blackboard_memory_registry,
    build_cognitive_blackboard,
    build_cognitive_router,
    cognitive_blackboard_entries_for_agent,
    cognitive_blackboard_entries_for_channel,
    cognitive_blackboard_entries_for_layer,
    cognitive_blackboard_entries_for_posture,
    cognitive_blackboard_entry_by_id,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
)


class CognitiveBlackboardTests(unittest.TestCase):
    def test_cognitive_blackboard_builds_read_only_entries(self) -> None:
        router = build_cognitive_router()
        registry = blackboard_memory_registry()
        blackboard = build_cognitive_blackboard(
            cognitive_router=router,
            blackboard_registry=registry,
        )

        self.assertEqual(blackboard.role, "cognitive_blackboard")
        self.assertEqual(
            blackboard.serialization_version,
            "cognitive_blackboard.v1",
        )
        self.assertEqual(blackboard.cognitive_router_role, router.role)
        self.assertEqual(
            blackboard.cognitive_router_serialization_version,
            router.serialization_version,
        )
        self.assertEqual(blackboard.cognitive_planner_role, "cognitive_planner")
        self.assertEqual(blackboard.cognitive_scheduler_role, "cognitive_scheduler")
        self.assertEqual(
            blackboard.blackboard_memory_registry_role,
            registry.role,
        )
        self.assertEqual(
            blackboard.blackboard_memory_registry_serialization_version,
            registry.serialization_version,
        )
        self.assertEqual(blackboard.source_blackboard_channel_ids, registry.channel_ids)
        self.assertEqual(blackboard.source_blackboard_channel_count, 12)
        self.assertEqual(blackboard.source_blackboard_permission_count, 12)
        self.assertEqual(blackboard.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(blackboard.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(blackboard.capability_ids, router.capability_ids)
        self.assertEqual(blackboard.capability_count, 6)
        self.assertEqual(
            blackboard.source_route_decision_ids,
            router.route_decision_ids,
        )
        self.assertEqual(blackboard.source_route_decision_count, 6)
        self.assertEqual(blackboard.source_plan_ids, router.source_plan_ids)
        self.assertEqual(blackboard.source_plan_count, 6)
        self.assertEqual(blackboard.source_schedule_ids, router.source_schedule_ids)
        self.assertEqual(blackboard.source_schedule_count, 6)
        self.assertEqual(blackboard.source_emergence_ids, router.source_emergence_ids)
        self.assertEqual(blackboard.source_emergence_count, 6)
        self.assertEqual(len(blackboard.blackboard_entries), 6)
        self.assertEqual(blackboard.blackboard_entry_count, 6)
        self.assertEqual(blackboard.candidate_blackboard_entry_count, 0)
        self.assertEqual(blackboard.review_required_blackboard_entry_count, 0)
        self.assertEqual(blackboard.guarded_blackboard_entry_count, 6)
        self.assertEqual(blackboard.linked_agent_ids, router.linked_agent_ids)
        self.assertEqual(
            blackboard.covered_roadmap_items,
            ("Cognitive Blackboard",),
        )
        self.assertEqual(blackboard.covered_roadmap_item_count, 1)
        self.assertEqual(blackboard.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            blackboard.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(blackboard.cognitive_blackboard_implemented)
        self.assertTrue(blackboard.cognitive_router_integrated)
        self.assertTrue(blackboard.blackboard_memory_registry_integrated)
        self.assertTrue(blackboard.blackboard_entry_contract_implemented)
        self.assertTrue(blackboard.blackboard_dependency_traceability_implemented)
        self.assertTrue(blackboard.blackboard_governance_contract_implemented)
        self.assertTrue(blackboard.blackboard_explainability_contract_implemented)
        self.assertFalse(blackboard.runtime_blackboard_read_implemented)
        self.assertFalse(blackboard.runtime_blackboard_write_implemented)
        self.assertFalse(blackboard.blackboard_persistence_implemented)
        self.assertFalse(blackboard.blackboard_mutation_implemented)
        self.assertFalse(blackboard.shared_context_materialization_implemented)
        self.assertFalse(blackboard.memory_mutation_implemented)
        self.assertFalse(blackboard.storage_mutation_implemented)
        self.assertFalse(blackboard.routing_application_implemented)
        self.assertFalse(blackboard.agent_invocation_implemented)
        self.assertFalse(blackboard.prompt_mutation_implemented)
        self.assertFalse(blackboard.retrieval_mutation_implemented)
        self.assertFalse(blackboard.provider_model_routing_implemented)
        self.assertFalse(blackboard.provider_execution_implemented)
        self.assertFalse(blackboard.generated_output_mutation_implemented)
        self.assertFalse(blackboard.runtime_evolution_implemented)
        self.assertFalse(blackboard.materialized_blackboard_entry_ids)
        self.assertFalse(blackboard.read_blackboard_channel_ids)
        self.assertFalse(blackboard.written_blackboard_channel_ids)
        self.assertFalse(blackboard.emitted_hitl_request_ids)
        self.assertTrue(blackboard.advisory_only)

    def test_cognitive_blackboard_lookup_helpers_are_scope_aware(self) -> None:
        blackboard = build_cognitive_blackboard()

        core_entry = cognitive_blackboard_entry_by_id(
            "cognitive_blackboard::v6_6_cognitive_core",
            blackboard,
        )
        self.assertIsNotNone(core_entry)
        assert core_entry is not None
        self.assertEqual(core_entry.capability_name, "V6.6 Cognitive Core")
        self.assertEqual(core_entry.cognitive_layer, "cognitive_core")
        self.assertIn("planner_agent", core_entry.linked_agent_ids)
        self.assertIn(
            "planner_agent_blackboard_channel",
            core_entry.visible_blackboard_channel_ids,
        )
        self.assertEqual(core_entry.visible_blackboard_channel_count, 5)
        self.assertFalse(core_entry.runtime_blackboard_access_authorized)
        self.assertIn("read runtime", core_entry.governance_contracts[0])

        research_entries = cognitive_blackboard_entries_for_layer(
            "research",
            blackboard,
        )
        self.assertEqual(len(research_entries), 1)
        self.assertEqual(
            research_entries[0].capability_id,
            "v6_4_autonomous_research",
        )

        planner_entries = cognitive_blackboard_entries_for_agent(
            "planner_agent",
            blackboard,
        )
        self.assertEqual(
            tuple(entry.capability_id for entry in planner_entries),
            ("v6_6_cognitive_core",),
        )
        planner_channel_entries = cognitive_blackboard_entries_for_channel(
            "planner_agent_blackboard_channel",
            blackboard,
        )
        self.assertEqual(
            tuple(entry.capability_id for entry in planner_channel_entries),
            ("v6_6_cognitive_core",),
        )
        guarded_entries = cognitive_blackboard_entries_for_posture(
            "guarded",
            blackboard,
        )
        self.assertEqual(
            tuple(entry.blackboard_entry_id for entry in guarded_entries),
            blackboard.guarded_blackboard_entry_ids,
        )
        self.assertIsNone(cognitive_blackboard_entry_by_id("missing", blackboard))

    def test_cognitive_blackboard_rejects_runtime_state_and_drift(self) -> None:
        blackboard = build_cognitive_blackboard()
        payload = blackboard.model_dump(mode="json")
        payload["blackboard_entry_ids"] = (
            "missing",
        ) + tuple(payload["blackboard_entry_ids"][1:])

        with self.assertRaisesRegex(ValueError, "blackboard_entry_ids must match"):
            CognitiveBlackboardPlan(**payload)

        payload = blackboard.model_dump(mode="json")
        payload["read_blackboard_channel_ids"] = (
            "planner_agent_blackboard_channel",
        )

        with self.assertRaisesRegex(
            ValueError,
            "blackboard materialization, reads, writes, and HITL ids must be empty",
        ):
            CognitiveBlackboardPlan(**payload)

    def test_cognitive_blackboard_reuses_supplied_router_and_registry(self) -> None:
        router = build_cognitive_router(route="generate")
        registry = blackboard_memory_registry()
        blackboard = build_cognitive_blackboard(
            cognitive_router=router,
            blackboard_registry=registry,
        )

        self.assertEqual(blackboard.route_name, router.route_name)
        self.assertEqual(blackboard.task_type, router.task_type)
        self.assertEqual(
            blackboard.source_route_decision_ids,
            router.route_decision_ids,
        )
        self.assertEqual(blackboard.source_plan_ids, router.source_plan_ids)
        self.assertEqual(blackboard.source_schedule_ids, router.source_schedule_ids)
        self.assertEqual(
            blackboard.source_emergence_ids,
            router.source_emergence_ids,
        )
        self.assertEqual(blackboard.source_blackboard_channel_ids, registry.channel_ids)


if __name__ == "__main__":
    unittest.main()
