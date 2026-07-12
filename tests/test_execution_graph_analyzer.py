import unittest

from langgraph.graph import END, START

from creative_coding_assistant.orchestration import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
    ASSISTANT_WORKFLOW_RECURSION_LIMIT,
    ExecutionGraphAnalysis,
    analyze_assistant_execution_graph,
    execution_graph_edges_from,
    execution_graph_edges_to,
    execution_graph_node_by_id,
)

REQUIRED_NODE_FIELDS = {
    "node_id",
    "order_index",
    "workflow_step",
    "can_enter_failure_path",
    "analysis_flags",
    "blocked_runtime_behaviors",
    "node_handler_invocation_implemented",
    "workflow_order_mutation_implemented",
    "serialization_version",
    "analysis_only",
}

REQUIRED_EDGE_FIELDS = {
    "edge_id",
    "edge_kind",
    "source_node_id",
    "target_node_id",
    "selector_name",
    "optimization_signals",
    "blocked_runtime_behaviors",
    "node_handler_invocation_implemented",
    "workflow_order_mutation_implemented",
    "retry_triggering_implemented",
    "provider_model_routing_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "analysis_only",
}


class ExecutionGraphAnalyzerTests(unittest.TestCase):
    def test_analyzer_reports_workflow_topology_without_execution(self) -> None:
        analysis = analyze_assistant_execution_graph()

        self.assertEqual(analysis.role, "execution_graph_analyzer")
        self.assertEqual(
            analysis.serialization_version,
            "execution_graph_analysis.v1",
        )
        self.assertEqual(analysis.node_order, ASSISTANT_WORKFLOW_NODE_ORDER)
        self.assertEqual(analysis.start_node_id, "intake")
        self.assertEqual(analysis.terminal_node_ids, ("finalization", "failure"))
        self.assertEqual(
            analysis.critical_path_node_ids,
            tuple(
                node
                for node in ASSISTANT_WORKFLOW_NODE_ORDER
                if node not in {"refinement", "failure"}
            ),
        )
        self.assertEqual(analysis.recursion_limit, ASSISTANT_WORKFLOW_RECURSION_LIMIT)
        self.assertEqual(analysis.node_count, len(ASSISTANT_WORKFLOW_NODE_ORDER))
        self.assertTrue(analysis.failure_path_reachable)
        self.assertTrue(analysis.bounded_retry_cycle_detected)
        self.assertIn("does not compile or execute", analysis.authority_boundary)
        self.assertFalse(analysis.graph_compilation_implemented)
        self.assertFalse(analysis.workflow_execution_implemented)
        self.assertFalse(analysis.node_handler_invocation_implemented)
        self.assertFalse(analysis.provider_model_routing_implemented)
        self.assertFalse(analysis.generated_output_mutation_implemented)
        self.assertTrue(analysis.analysis_only)

    def test_nodes_and_edges_expose_stable_typed_contracts(self) -> None:
        analysis = analyze_assistant_execution_graph()

        for index, node in enumerate(analysis.nodes):
            self.assertEqual(set(node.model_dump(mode="json")), REQUIRED_NODE_FIELDS)
            self.assertEqual(node.order_index, index)
            self.assertEqual(node.workflow_step, node.node_id)
            self.assertEqual(node.serialization_version, "execution_graph_node.v1")
            self.assertIn("node_handler_not_invoked", node.analysis_flags)
            self.assertIn("workflow_execution", node.blocked_runtime_behaviors)
            self.assertFalse(node.node_handler_invocation_implemented)
            self.assertFalse(node.workflow_order_mutation_implemented)
            self.assertTrue(node.analysis_only)

        for edge in analysis.edges:
            self.assertEqual(set(edge.model_dump(mode="json")), REQUIRED_EDGE_FIELDS)
            self.assertEqual(edge.serialization_version, "execution_graph_edge.v1")
            self.assertIn("workflow_execution", edge.blocked_runtime_behaviors)
            self.assertFalse(edge.node_handler_invocation_implemented)
            self.assertFalse(edge.workflow_order_mutation_implemented)
            self.assertFalse(edge.retry_triggering_implemented)
            self.assertFalse(edge.provider_model_routing_implemented)
            self.assertFalse(edge.generated_output_mutation_implemented)
            self.assertTrue(edge.analysis_only)

    def test_analyzer_surfaces_entry_failure_retry_and_terminal_edges(self) -> None:
        analysis = analyze_assistant_execution_graph()

        entry_edges = execution_graph_edges_from(str(START), analysis)
        prompt_edges = execution_graph_edges_from("prompt_input", analysis)
        review_edges = execution_graph_edges_from("review", analysis)
        refinement_edges = execution_graph_edges_from("refinement", analysis)
        terminal_edges = execution_graph_edges_to(str(END), analysis)

        self.assertEqual(len(entry_edges), 1)
        self.assertEqual(entry_edges[0].edge_kind, "entry")
        self.assertEqual(entry_edges[0].target_node_id, "intake")
        self.assertEqual(
            {edge.target_node_id: edge.edge_kind for edge in prompt_edges},
            {
                "planning": "linear",
                "prompt_rendering": "conditional",
                "finalization": "short_circuit",
                "failure": "failure",
            },
        )
        self.assertEqual(
            {edge.target_node_id: edge.edge_kind for edge in review_edges},
            {
                "finalization": "conditional",
                "refinement": "retry",
                "failure": "failure",
            },
        )
        self.assertEqual(
            {edge.target_node_id: edge.edge_kind for edge in refinement_edges},
            {
                "generation": "retry",
                "failure": "failure",
            },
        )
        self.assertEqual(
            tuple(edge.source_node_id for edge in terminal_edges),
            ("finalization", "failure"),
        )
        self.assertIn("prompt_input", analysis.branch_node_ids)
        self.assertIn("review", analysis.branch_node_ids)
        self.assertIn("refinement", analysis.retry_entry_node_ids)
        self.assertIn("generation_reuse_path", refinement_edges[0].optimization_signals)

    def test_lookup_helpers_are_read_only_and_stable(self) -> None:
        analysis = analyze_assistant_execution_graph()
        generation = execution_graph_node_by_id("generation", analysis)
        missing = execution_graph_node_by_id("missing", analysis)

        self.assertIsNone(missing)
        self.assertIsNotNone(generation)
        assert generation is not None
        self.assertEqual(generation.workflow_step, "generation")
        self.assertTrue(generation.can_enter_failure_path)
        self.assertIs(
            generation,
            execution_graph_node_by_id("generation", analysis),
        )
        self.assertEqual(execution_graph_edges_from("missing", analysis), ())
        self.assertEqual(execution_graph_edges_to("missing", analysis), ())

    def test_analysis_rejects_unknown_or_unsafe_edges(self) -> None:
        analysis = analyze_assistant_execution_graph()
        payload = analysis.model_dump(mode="json")
        bad_edges = list(payload["edges"])
        bad_edges[1] = {**bad_edges[1], "target_node_id": "missing"}
        payload["edges"] = bad_edges

        with self.assertRaisesRegex(ValueError, "known target nodes"):
            ExecutionGraphAnalysis(**payload)

        payload = analysis.model_dump(mode="json")
        bad_edges = list(payload["edges"])
        bad_edges[3] = {
            **bad_edges[3],
            "edge_id": "routing->intake",
            "source_node_id": "routing",
            "target_node_id": "intake",
            "edge_kind": "linear",
        }
        payload["edges"] = bad_edges

        with self.assertRaisesRegex(ValueError, "back edges must be retry edges"):
            ExecutionGraphAnalysis(**payload)

    def test_analysis_does_not_declare_runtime_mutation_terms(self) -> None:
        analysis = analyze_assistant_execution_graph()
        combined_text = " ".join(
            (
                analysis.authority_boundary,
                *analysis.blocked_runtime_behaviors,
                *(
                    field
                    for edge in analysis.edges
                    for field in (
                        edge.edge_id,
                        edge.selector_name,
                        *edge.optimization_signals,
                        *edge.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "graph.compile(",
            "graph.stream(",
            "invoke_handler",
            "route_provider",
            "modify_output",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
