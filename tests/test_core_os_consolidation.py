import unittest

from creative_coding_assistant.orchestration import (
    CoreOSConsolidationPlan,
    build_core_os_consolidation,
    build_unified_execution_graph,
    core_os_consolidation_unit_by_id,
    core_os_consolidation_units_for_agent,
    core_os_consolidation_units_for_layer,
    core_os_consolidation_units_for_posture,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
    COGNITIVE_OS_ROADMAP_ITEMS,
)


class CoreOSConsolidationTests(unittest.TestCase):
    def test_core_os_consolidation_builds_read_only_surface(self) -> None:
        execution = build_unified_execution_graph()
        consolidation = build_core_os_consolidation(
            unified_execution_graph=execution,
        )

        self.assertEqual(consolidation.role, "core_os_consolidation")
        self.assertEqual(
            consolidation.serialization_version,
            "core_os_consolidation.v1",
        )
        self.assertEqual(consolidation.unified_execution_graph_role, execution.role)
        self.assertEqual(
            consolidation.unified_execution_graph_serialization_version,
            execution.serialization_version,
        )
        self.assertEqual(
            consolidation.cognitive_hitl_layer_role,
            execution.cognitive_hitl_layer_role,
        )
        self.assertEqual(
            consolidation.cognitive_safety_layer_role,
            execution.cognitive_safety_layer_role,
        )
        self.assertEqual(
            consolidation.cognitive_explanation_engine_role,
            execution.cognitive_explanation_engine_role,
        )
        self.assertEqual(
            consolidation.cognitive_blackboard_role,
            execution.cognitive_blackboard_role,
        )
        self.assertEqual(
            consolidation.cognitive_router_role,
            execution.cognitive_router_role,
        )
        self.assertEqual(
            consolidation.cognitive_planner_role,
            execution.cognitive_planner_role,
        )
        self.assertEqual(
            consolidation.cognitive_scheduler_role,
            execution.cognitive_scheduler_role,
        )
        self.assertEqual(consolidation.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(consolidation.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(consolidation.capability_ids, execution.capability_ids)
        self.assertEqual(consolidation.capability_count, 6)
        self.assertEqual(
            consolidation.source_execution_node_ids,
            execution.execution_node_ids,
        )
        self.assertEqual(consolidation.source_execution_node_count, 6)
        self.assertEqual(
            consolidation.source_execution_edge_ids,
            execution.execution_edge_ids,
        )
        self.assertEqual(consolidation.source_execution_edge_count, 5)
        self.assertEqual(consolidation.source_hitl_ids, execution.source_hitl_ids)
        self.assertEqual(consolidation.source_hitl_count, 6)
        self.assertEqual(consolidation.source_safety_ids, execution.source_safety_ids)
        self.assertEqual(consolidation.source_safety_count, 6)
        self.assertEqual(
            consolidation.source_explanation_ids,
            execution.source_explanation_ids,
        )
        self.assertEqual(consolidation.source_explanation_count, 6)
        self.assertEqual(
            consolidation.source_blackboard_entry_ids,
            execution.source_blackboard_entry_ids,
        )
        self.assertEqual(consolidation.source_blackboard_entry_count, 6)
        self.assertEqual(
            consolidation.source_route_decision_ids,
            execution.source_route_decision_ids,
        )
        self.assertEqual(consolidation.source_route_decision_count, 6)
        self.assertEqual(consolidation.source_plan_ids, execution.source_plan_ids)
        self.assertEqual(consolidation.source_plan_count, 6)
        self.assertEqual(
            consolidation.source_schedule_ids,
            execution.source_schedule_ids,
        )
        self.assertEqual(consolidation.source_schedule_count, 6)
        self.assertEqual(
            consolidation.source_emergence_ids,
            execution.source_emergence_ids,
        )
        self.assertEqual(consolidation.source_emergence_count, 6)
        self.assertEqual(len(consolidation.consolidation_units), 6)
        self.assertEqual(consolidation.consolidation_unit_count, 6)
        self.assertEqual(
            consolidation.core_os_entry_unit_id,
            consolidation.consolidation_unit_ids[0],
        )
        self.assertEqual(
            consolidation.core_os_terminal_unit_id,
            consolidation.consolidation_unit_ids[-1],
        )
        self.assertEqual(
            consolidation.blocked_pending_hitl_unit_ids,
            consolidation.consolidation_unit_ids,
        )
        self.assertEqual(consolidation.blocked_pending_hitl_unit_count, 6)
        self.assertEqual(consolidation.linked_agent_ids, execution.linked_agent_ids)
        self.assertEqual(
            consolidation.covered_roadmap_items,
            ("Core OS Consolidation",),
        )
        self.assertEqual(consolidation.covered_roadmap_item_count, 1)
        self.assertEqual(
            consolidation.consolidated_roadmap_items,
            COGNITIVE_OS_ROADMAP_ITEMS,
        )
        self.assertEqual(consolidation.consolidated_roadmap_item_count, 24)
        self.assertEqual(
            consolidation.cross_cutting_contracts,
            COGNITIVE_OS_CONTRACTS,
        )
        self.assertEqual(
            consolidation.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(consolidation.core_os_consolidation_implemented)
        self.assertTrue(consolidation.unified_execution_graph_integrated)
        self.assertTrue(consolidation.cognitive_os_sequence_consolidated)
        self.assertTrue(consolidation.roadmap_traceability_consolidated)
        self.assertTrue(consolidation.dependency_traceability_consolidated)
        self.assertTrue(consolidation.governance_contract_consolidated)
        self.assertTrue(consolidation.explainability_contract_consolidated)
        self.assertTrue(consolidation.safety_contract_consolidated)
        self.assertTrue(consolidation.hitl_contract_consolidated)
        self.assertTrue(consolidation.future_holomind_extensibility_prepared)
        self.assertFalse(consolidation.core_os_runtime_activation_implemented)
        self.assertFalse(consolidation.execution_application_implemented)
        self.assertFalse(consolidation.workflow_execution_implemented)
        self.assertFalse(consolidation.autonomous_workflow_planning_implemented)
        self.assertFalse(consolidation.routing_application_implemented)
        self.assertFalse(consolidation.scheduler_application_implemented)
        self.assertFalse(consolidation.plan_execution_implemented)
        self.assertFalse(consolidation.hitl_request_emission_implemented)
        self.assertFalse(consolidation.hitl_decision_application_implemented)
        self.assertFalse(consolidation.safety_enforcement_implemented)
        self.assertFalse(consolidation.workflow_blocking_implemented)
        self.assertFalse(consolidation.prompt_mutation_implemented)
        self.assertFalse(consolidation.memory_mutation_implemented)
        self.assertFalse(consolidation.retrieval_mutation_implemented)
        self.assertFalse(consolidation.storage_mutation_implemented)
        self.assertFalse(consolidation.provider_model_routing_implemented)
        self.assertFalse(consolidation.provider_execution_implemented)
        self.assertFalse(consolidation.generated_output_mutation_implemented)
        self.assertFalse(consolidation.runtime_evolution_implemented)
        self.assertFalse(consolidation.activated_core_os_unit_ids)
        self.assertFalse(consolidation.executed_node_ids)
        self.assertFalse(consolidation.traversed_edge_ids)
        self.assertFalse(consolidation.applied_route_decision_ids)
        self.assertFalse(consolidation.emitted_hitl_request_ids)
        self.assertFalse(consolidation.applied_hitl_decision_ids)
        self.assertFalse(consolidation.mutated_core_os_ids)
        self.assertTrue(consolidation.advisory_only)

    def test_core_os_consolidation_lookup_helpers_are_scope_aware(self) -> None:
        consolidation = build_core_os_consolidation()

        core_unit = core_os_consolidation_unit_by_id(
            "core_os::v6_6_cognitive_core",
            consolidation,
        )
        self.assertIsNotNone(core_unit)
        assert core_unit is not None
        self.assertEqual(core_unit.capability_name, "V6.6 Cognitive Core")
        self.assertEqual(core_unit.cognitive_layer, "cognitive_core")
        self.assertIn("planner_agent", core_unit.linked_agent_ids)
        self.assertEqual(core_unit.os_sequence_position, 6)
        self.assertEqual(core_unit.dependency_depth, 5)
        self.assertEqual(
            core_unit.source_trace_ids[0],
            "unified_execution::v6_6_cognitive_core",
        )
        self.assertIn(
            "cognitive_hitl::v6_6_cognitive_core",
            core_unit.source_trace_ids,
        )
        self.assertIn(
            "cognitive_safety::v6_6_cognitive_core",
            core_unit.source_trace_ids,
        )
        self.assertFalse(core_unit.runtime_activation_authorized)
        self.assertFalse(core_unit.execution_authorized)
        self.assertEqual(core_unit.core_os_status, "consolidated_metadata_only")
        self.assertIn("does not activate", core_unit.governance_contracts[0])

        research_units = core_os_consolidation_units_for_layer(
            "research",
            consolidation,
        )
        self.assertEqual(len(research_units), 1)
        self.assertEqual(research_units[0].capability_id, "v6_4_autonomous_research")

        planner_units = core_os_consolidation_units_for_agent(
            "planner_agent",
            consolidation,
        )
        self.assertEqual(
            tuple(unit.capability_id for unit in planner_units),
            ("v6_6_cognitive_core",),
        )
        guarded_units = core_os_consolidation_units_for_posture(
            "guarded",
            consolidation,
        )
        self.assertEqual(
            tuple(unit.consolidation_unit_id for unit in guarded_units),
            consolidation.blocked_pending_hitl_unit_ids,
        )
        self.assertIsNone(core_os_consolidation_unit_by_id("missing", consolidation))

    def test_core_os_consolidation_rejects_activation_and_drift(self) -> None:
        consolidation = build_core_os_consolidation()
        payload = consolidation.model_dump(mode="json")
        payload["consolidation_unit_ids"] = (
            "missing",
        ) + tuple(payload["consolidation_unit_ids"][1:])

        with self.assertRaisesRegex(ValueError, "consolidation_unit_ids must match"):
            CoreOSConsolidationPlan(**payload)

        payload = consolidation.model_dump(mode="json")
        payload["activated_core_os_unit_ids"] = (
            "core_os::v6_6_cognitive_core",
        )

        with self.assertRaisesRegex(
            ValueError,
            "Core OS activation, execution, traversal, routing, HITL, "
            "and mutation ids must be empty",
        ):
            CoreOSConsolidationPlan(**payload)

    def test_core_os_consolidation_reuses_supplied_execution_graph(self) -> None:
        execution = build_unified_execution_graph(route="generate")
        consolidation = build_core_os_consolidation(
            unified_execution_graph=execution,
        )

        self.assertEqual(consolidation.route_name, execution.route_name)
        self.assertEqual(consolidation.task_type, execution.task_type)
        self.assertEqual(
            consolidation.source_execution_node_ids,
            execution.execution_node_ids,
        )
        self.assertEqual(
            consolidation.source_execution_edge_ids,
            execution.execution_edge_ids,
        )
        self.assertEqual(consolidation.source_hitl_ids, execution.source_hitl_ids)
        self.assertEqual(consolidation.source_safety_ids, execution.source_safety_ids)
        self.assertEqual(
            consolidation.source_explanation_ids,
            execution.source_explanation_ids,
        )
        self.assertEqual(
            consolidation.source_blackboard_entry_ids,
            execution.source_blackboard_entry_ids,
        )
        self.assertEqual(
            consolidation.source_route_decision_ids,
            execution.source_route_decision_ids,
        )
        self.assertEqual(consolidation.source_plan_ids, execution.source_plan_ids)
        self.assertEqual(
            consolidation.source_schedule_ids,
            execution.source_schedule_ids,
        )
        self.assertEqual(
            consolidation.source_emergence_ids,
            execution.source_emergence_ids,
        )


if __name__ == "__main__":
    unittest.main()
