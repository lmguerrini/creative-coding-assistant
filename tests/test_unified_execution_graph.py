import unittest

from creative_coding_assistant.orchestration import (
    UnifiedExecutionGraphPlan,
    build_cognitive_hitl_layer,
    build_unified_execution_graph,
    unified_execution_edges_from_node,
    unified_execution_edges_to_node,
    unified_execution_node_by_id,
    unified_execution_nodes_for_agent,
    unified_execution_nodes_for_layer,
    unified_execution_nodes_for_posture,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
)


class UnifiedExecutionGraphTests(unittest.TestCase):
    def test_unified_execution_graph_builds_read_only_topology(self) -> None:
        hitl = build_cognitive_hitl_layer()
        graph = build_unified_execution_graph(cognitive_hitl_layer=hitl)

        self.assertEqual(graph.role, "unified_execution_graph")
        self.assertEqual(
            graph.serialization_version,
            "unified_execution_graph.v1",
        )
        self.assertEqual(graph.cognitive_hitl_layer_role, hitl.role)
        self.assertEqual(
            graph.cognitive_hitl_layer_serialization_version,
            hitl.serialization_version,
        )
        self.assertEqual(
            graph.cognitive_safety_layer_role,
            hitl.cognitive_safety_layer_role,
        )
        self.assertEqual(
            graph.cognitive_explanation_engine_role,
            hitl.cognitive_explanation_engine_role,
        )
        self.assertEqual(
            graph.cognitive_blackboard_role,
            hitl.cognitive_blackboard_role,
        )
        self.assertEqual(graph.cognitive_router_role, hitl.cognitive_router_role)
        self.assertEqual(graph.cognitive_planner_role, hitl.cognitive_planner_role)
        self.assertEqual(graph.cognitive_scheduler_role, hitl.cognitive_scheduler_role)
        self.assertEqual(graph.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(graph.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(graph.capability_ids, hitl.capability_ids)
        self.assertEqual(graph.capability_count, 6)
        self.assertEqual(graph.source_hitl_ids, hitl.hitl_ids)
        self.assertEqual(graph.source_hitl_count, 6)
        self.assertEqual(graph.source_safety_ids, hitl.source_safety_ids)
        self.assertEqual(graph.source_safety_count, 6)
        self.assertEqual(graph.source_explanation_ids, hitl.source_explanation_ids)
        self.assertEqual(graph.source_explanation_count, 6)
        self.assertEqual(
            graph.source_blackboard_entry_ids,
            hitl.source_blackboard_entry_ids,
        )
        self.assertEqual(graph.source_blackboard_entry_count, 6)
        self.assertEqual(
            graph.source_route_decision_ids,
            hitl.source_route_decision_ids,
        )
        self.assertEqual(graph.source_route_decision_count, 6)
        self.assertEqual(graph.source_plan_ids, hitl.source_plan_ids)
        self.assertEqual(graph.source_plan_count, 6)
        self.assertEqual(graph.source_schedule_ids, hitl.source_schedule_ids)
        self.assertEqual(graph.source_schedule_count, 6)
        self.assertEqual(graph.source_emergence_ids, hitl.source_emergence_ids)
        self.assertEqual(graph.source_emergence_count, 6)
        self.assertEqual(len(graph.execution_nodes), 6)
        self.assertEqual(graph.execution_node_count, 6)
        self.assertEqual(len(graph.execution_edges), 5)
        self.assertEqual(graph.execution_edge_count, 5)
        self.assertEqual(graph.execution_entry_node_id, graph.execution_node_ids[0])
        self.assertEqual(
            graph.execution_terminal_node_id,
            graph.execution_node_ids[-1],
        )
        self.assertEqual(graph.blocked_pending_hitl_node_ids, graph.execution_node_ids)
        self.assertEqual(graph.blocked_pending_hitl_node_count, 6)
        self.assertEqual(graph.linked_agent_ids, hitl.linked_agent_ids)
        self.assertEqual(graph.covered_roadmap_items, ("Unified Execution Graph",))
        self.assertEqual(graph.covered_roadmap_item_count, 1)
        self.assertEqual(graph.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            graph.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(graph.unified_execution_graph_implemented)
        self.assertTrue(graph.cognitive_hitl_layer_integrated)
        self.assertTrue(graph.execution_node_contract_implemented)
        self.assertTrue(graph.execution_edge_contract_implemented)
        self.assertTrue(graph.execution_dependency_traceability_implemented)
        self.assertTrue(graph.execution_governance_contract_implemented)
        self.assertTrue(graph.execution_explainability_contract_implemented)
        self.assertTrue(graph.execution_safety_contract_implemented)
        self.assertTrue(graph.execution_hitl_contract_implemented)
        self.assertFalse(graph.execution_application_implemented)
        self.assertFalse(graph.workflow_execution_implemented)
        self.assertFalse(graph.autonomous_workflow_planning_implemented)
        self.assertFalse(graph.routing_application_implemented)
        self.assertFalse(graph.scheduler_application_implemented)
        self.assertFalse(graph.plan_execution_implemented)
        self.assertFalse(graph.hitl_request_emission_implemented)
        self.assertFalse(graph.hitl_decision_application_implemented)
        self.assertFalse(graph.safety_enforcement_implemented)
        self.assertFalse(graph.workflow_blocking_implemented)
        self.assertFalse(graph.prompt_mutation_implemented)
        self.assertFalse(graph.memory_mutation_implemented)
        self.assertFalse(graph.retrieval_mutation_implemented)
        self.assertFalse(graph.storage_mutation_implemented)
        self.assertFalse(graph.provider_model_routing_implemented)
        self.assertFalse(graph.provider_execution_implemented)
        self.assertFalse(graph.generated_output_mutation_implemented)
        self.assertFalse(graph.runtime_evolution_implemented)
        self.assertFalse(graph.executed_node_ids)
        self.assertFalse(graph.traversed_edge_ids)
        self.assertFalse(graph.applied_route_decision_ids)
        self.assertFalse(graph.emitted_hitl_request_ids)
        self.assertFalse(graph.applied_hitl_decision_ids)
        self.assertFalse(graph.mutated_execution_graph_ids)
        self.assertTrue(graph.advisory_only)

    def test_unified_execution_lookup_helpers_are_scope_aware(self) -> None:
        graph = build_unified_execution_graph()

        core_node = unified_execution_node_by_id(
            "unified_execution::v6_6_cognitive_core",
            graph,
        )
        self.assertIsNotNone(core_node)
        assert core_node is not None
        self.assertEqual(core_node.capability_name, "V6.6 Cognitive Core")
        self.assertEqual(core_node.cognitive_layer, "cognitive_core")
        self.assertIn("planner_agent", core_node.linked_agent_ids)
        self.assertEqual(core_node.execution_order, 6)
        self.assertEqual(core_node.dependency_depth, 5)
        self.assertEqual(
            core_node.source_trace_ids[0],
            "cognitive_hitl::v6_6_cognitive_core",
        )
        self.assertIn(
            "cognitive_safety::v6_6_cognitive_core",
            core_node.source_trace_ids,
        )
        self.assertIn(
            "cognitive_explanation::v6_6_cognitive_core",
            core_node.source_trace_ids,
        )
        self.assertFalse(core_node.execution_authorized)
        self.assertEqual(core_node.execution_state, "blocked_pending_hitl")
        self.assertIn("does not execute", core_node.governance_contracts[0])

        research_nodes = unified_execution_nodes_for_layer("research", graph)
        self.assertEqual(len(research_nodes), 1)
        self.assertEqual(research_nodes[0].capability_id, "v6_4_autonomous_research")

        planner_nodes = unified_execution_nodes_for_agent("planner_agent", graph)
        self.assertEqual(
            tuple(node.capability_id for node in planner_nodes),
            ("v6_6_cognitive_core",),
        )
        guarded_nodes = unified_execution_nodes_for_posture("guarded", graph)
        self.assertEqual(
            tuple(node.execution_node_id for node in guarded_nodes),
            graph.blocked_pending_hitl_node_ids,
        )

        learning_edges = unified_execution_edges_from_node(
            "unified_execution::v6_1_adaptive_learning",
            graph,
        )
        self.assertEqual(len(learning_edges), 1)
        self.assertEqual(
            learning_edges[0].to_execution_node_id,
            "unified_execution::v6_2_creative_memory",
        )
        self.assertFalse(learning_edges[0].execution_transition_authorized)

        core_incoming_edges = unified_execution_edges_to_node(
            "unified_execution::v6_6_cognitive_core",
            graph,
        )
        self.assertEqual(len(core_incoming_edges), 1)
        self.assertEqual(
            core_incoming_edges[0].from_execution_node_id,
            "unified_execution::v6_5_self_evolution",
        )
        self.assertIsNone(unified_execution_node_by_id("missing", graph))

    def test_unified_execution_graph_rejects_execution_and_drift(self) -> None:
        graph = build_unified_execution_graph()
        payload = graph.model_dump(mode="json")
        payload["execution_node_ids"] = (
            "missing",
        ) + tuple(payload["execution_node_ids"][1:])

        with self.assertRaisesRegex(ValueError, "execution_node_ids must match"):
            UnifiedExecutionGraphPlan(**payload)

        payload = graph.model_dump(mode="json")
        payload["executed_node_ids"] = (
            "unified_execution::v6_6_cognitive_core",
        )

        with self.assertRaisesRegex(
            ValueError,
            "execution, traversal, routing, HITL, and mutation ids must be empty",
        ):
            UnifiedExecutionGraphPlan(**payload)

    def test_unified_execution_graph_reuses_supplied_hitl_layer(self) -> None:
        hitl = build_cognitive_hitl_layer(route="generate")
        graph = build_unified_execution_graph(cognitive_hitl_layer=hitl)

        self.assertEqual(graph.route_name, hitl.route_name)
        self.assertEqual(graph.task_type, hitl.task_type)
        self.assertEqual(graph.source_hitl_ids, hitl.hitl_ids)
        self.assertEqual(graph.source_safety_ids, hitl.source_safety_ids)
        self.assertEqual(graph.source_explanation_ids, hitl.source_explanation_ids)
        self.assertEqual(
            graph.source_blackboard_entry_ids,
            hitl.source_blackboard_entry_ids,
        )
        self.assertEqual(
            graph.source_route_decision_ids,
            hitl.source_route_decision_ids,
        )
        self.assertEqual(graph.source_plan_ids, hitl.source_plan_ids)
        self.assertEqual(graph.source_schedule_ids, hitl.source_schedule_ids)
        self.assertEqual(graph.source_emergence_ids, hitl.source_emergence_ids)


if __name__ == "__main__":
    unittest.main()
