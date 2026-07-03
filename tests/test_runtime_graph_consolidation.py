import unittest

from creative_coding_assistant.orchestration import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
    ASSISTANT_WORKFLOW_RECURSION_LIMIT,
    audit_runtime_graph_determinism,
    build_runtime_graph_consolidation_plan,
    diff_runtime_graphs,
    explain_runtime_graph,
    normalize_runtime_workflow_state,
    profile_runtime_graph_cost,
    profile_runtime_graph_latency,
    profile_runtime_graph_performance,
    record_runtime_graph_trace,
    runtime_graph_node_contract_by_id,
    runtime_graph_node_contracts_for_subgraph,
    validate_runtime_graph_contracts,
    verify_runtime_graph_invariants,
    visualize_runtime_graph,
)
from creative_coding_assistant.orchestration.runtime_graph_consolidation import (
    RUNTIME_GRAPH_CONSOLIDATION_ROADMAP_ITEMS,
)
from creative_coding_assistant.orchestration.workflow_graph import (
    assistant_workflow_conditional_edge_specs,
    assistant_workflow_final_payload_keys,
    assistant_workflow_model_payload_specs,
    assistant_workflow_node_specs,
)


class RuntimeGraphConsolidationTests(unittest.TestCase):
    def test_consolidation_plan_covers_v7_1_without_behavior_changes(self) -> None:
        plan = build_runtime_graph_consolidation_plan()

        self.assertEqual(plan.role, "runtime_graph_consolidation")
        self.assertEqual(
            plan.serialization_version,
            "runtime_graph_consolidation.v1",
        )
        self.assertEqual(plan.source_node_order, ASSISTANT_WORKFLOW_NODE_ORDER)
        self.assertEqual(plan.recursion_limit, ASSISTANT_WORKFLOW_RECURSION_LIMIT)
        self.assertEqual(
            plan.covered_roadmap_items,
            RUNTIME_GRAPH_CONSOLIDATION_ROADMAP_ITEMS,
        )
        self.assertEqual(len(plan.covered_roadmap_items), 23)
        self.assertFalse(plan.behavior_change_implemented)
        self.assertFalse(plan.provider_model_routing_change_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.storage_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertFalse(plan.v7_2_scope_started)
        self.assertTrue(plan.advisory_only)

    def test_public_topology_helpers_preserve_legacy_shape(self) -> None:
        node_specs = assistant_workflow_node_specs()
        edge_specs = assistant_workflow_conditional_edge_specs()

        self.assertEqual(
            tuple(spec.name for spec in node_specs),
            ASSISTANT_WORKFLOW_NODE_ORDER,
        )
        self.assertEqual(
            assistant_workflow_final_payload_keys(),
            build_runtime_graph_consolidation_plan().final_payload_keys,
        )
        self.assertEqual(
            tuple(
                spec.payload_key
                for spec in assistant_workflow_model_payload_specs()
            ),
            build_runtime_graph_consolidation_plan().runtime_payload_keys,
        )
        self.assertIn("prompt_input", tuple(spec.source for spec in edge_specs))
        self.assertIn("review", tuple(spec.source for spec in edge_specs))
        self.assertIn("finalization", tuple(spec.source for spec in edge_specs))

    def test_node_and_subgraph_contracts_extract_runtime_boundaries(self) -> None:
        plan = build_runtime_graph_consolidation_plan()
        planning = runtime_graph_node_contract_by_id("planning", plan)
        generation = runtime_graph_node_contract_by_id("generation", plan)
        creative_nodes = runtime_graph_node_contracts_for_subgraph(
            "creative_cognition",
            plan,
        )
        artifact_nodes = runtime_graph_node_contracts_for_subgraph(
            "artifact_intelligence",
            plan,
        )

        self.assertIsNotNone(planning)
        self.assertIsNotNone(generation)
        assert planning is not None
        assert generation is not None
        self.assertEqual(planning.primary_subgraph_id, "creative_cognition")
        self.assertIn("generative_design", planning.subgraph_ids)
        self.assertIn("artifact_intelligence", planning.subgraph_ids)
        self.assertIn("creative_evaluation", planning.subgraph_ids)
        self.assertIn("creative_plan", planning.state_outputs)
        self.assertTrue(planning.handler_reference.endswith("._planning_node"))
        self.assertFalse(planning.workflow_graph_mutation_implemented)
        self.assertEqual(generation.primary_subgraph_id, "workflow_foundation")
        self.assertIn("rendered_prompt", generation.state_inputs)
        self.assertIn("final_answer", generation.state_outputs)
        self.assertEqual(
            tuple(contract.node_id for contract in creative_nodes),
            ("planning", "director", "reasoning"),
        )
        self.assertIn(
            "artifact_critique",
            tuple(contract.node_id for contract in artifact_nodes),
        )

    def test_validation_invariants_and_state_normalization_pass(self) -> None:
        plan = build_runtime_graph_consolidation_plan()
        validation = validate_runtime_graph_contracts(plan)
        invariants = verify_runtime_graph_invariants(plan)
        normalization = normalize_runtime_workflow_state(plan)

        self.assertTrue(validation.validation_passed)
        self.assertEqual(validation.checked_node_contract_count, 18)
        self.assertEqual(validation.checked_subgraph_contract_count, 6)
        self.assertEqual(validation.checked_roadmap_item_count, 23)
        self.assertFalse(validation.missing_node_contract_ids)
        self.assertFalse(validation.compatibility_failures)
        self.assertEqual(invariants.invariant_status, "pass")
        self.assertTrue(invariants.failure_path_reachable)
        self.assertTrue(invariants.bounded_retry_cycle_detected)
        self.assertTrue(invariants.terminal_nodes_stable)
        self.assertTrue(normalization.normalization_passed)
        self.assertFalse(normalization.missing_runtime_payload_keys)
        self.assertFalse(normalization.missing_final_payload_keys)

    def test_trace_explainability_visualization_diff_and_determinism(self) -> None:
        plan = build_runtime_graph_consolidation_plan()
        trace = record_runtime_graph_trace(plan)
        explanation = explain_runtime_graph(plan)
        visualization = visualize_runtime_graph()
        diff = diff_runtime_graphs(plan, plan)
        determinism = audit_runtime_graph_determinism(plan)

        self.assertEqual(trace.trace_source, "static_contract")
        self.assertEqual(trace.node_ids[0], "intake")
        self.assertEqual(trace.node_ids[-1], "finalization")
        self.assertNotIn("refinement", trace.node_ids)
        self.assertFalse(trace.workflow_execution_implemented)
        self.assertIn("failure node", explanation.failure_boundary_explanation)
        self.assertFalse(explanation.routing_change_implemented)
        self.assertEqual(visualization.format, "mermaid")
        self.assertIn("flowchart TD", visualization.diagram)
        self.assertIn(
            "generation -->|linear| artifact_extraction",
            visualization.diagram,
        )
        self.assertEqual(diff.diff_status, "no_change")
        self.assertFalse(diff.behavior_change_detected)
        self.assertTrue(determinism.deterministic)
        self.assertFalse(determinism.nondeterministic_surfaces)

    def test_static_performance_cost_and_latency_profiles_are_bounded(self) -> None:
        performance = profile_runtime_graph_performance()
        cost = profile_runtime_graph_cost(performance)
        latency = profile_runtime_graph_latency(performance)

        self.assertEqual(performance.measurement_mode, "static_relative")
        self.assertEqual(len(performance.node_profiles), 18)
        self.assertGreater(performance.total_relative_cost_units, 18)
        self.assertGreater(performance.total_relative_latency_units, 18)
        self.assertGreaterEqual(performance.branch_count, 3)
        self.assertGreaterEqual(performance.failure_edge_count, 1)
        self.assertEqual(cost.measurement_mode, "static_relative")
        self.assertIn("generation", cost.highest_relative_cost_node_ids)
        self.assertFalse(cost.pricing_lookup_implemented)
        self.assertFalse(cost.provider_model_routing_change_implemented)
        self.assertEqual(latency.measurement_mode, "static_relative")
        self.assertIn("generation", latency.highest_relative_latency_node_ids)
        self.assertFalse(latency.live_timing_implemented)
        self.assertFalse(latency.provider_model_routing_change_implemented)


if __name__ == "__main__":
    unittest.main()
