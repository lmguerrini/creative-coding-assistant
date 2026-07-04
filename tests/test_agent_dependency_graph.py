import unittest

from creative_coding_assistant.orchestration import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
    AgentDependencyGraphRegistry,
    agent_contract_registry,
    agent_dependency_downstream_nodes,
    agent_dependency_graph_registry,
    agent_dependency_node_by_id,
    agent_dependency_upstream_nodes,
    shared_context_view_registry,
)

REQUIRED_NODE_FIELDS = {
    "node_id",
    "node_type",
    "stage_id",
    "agent_id",
    "required_inputs",
    "upstream_node_ids",
    "downstream_node_ids",
    "source_registries",
    "blocked_runtime_behaviors",
    "scheduling_implemented",
    "runtime_orchestration_implemented",
    "workflow_node_order_changed",
    "serialization_version",
    "metadata_only",
}

REQUIRED_EDGE_FIELDS = {
    "edge_id",
    "edge_type",
    "from_node_id",
    "to_node_id",
    "required_input",
    "dependency_boundary",
    "blocked_runtime_behaviors",
    "scheduling_implemented",
    "runtime_orchestration_implemented",
    "serialization_version",
    "metadata_only",
}


class AgentDependencyGraphTests(unittest.TestCase):
    def test_dependency_graph_covers_agents_and_context_views(self) -> None:
        graph = agent_dependency_graph_registry()
        contract_registry = agent_contract_registry()
        view_registry = shared_context_view_registry()

        self.assertEqual(graph.role, "agent_dependency_graph_registry")
        self.assertEqual(graph.serialization_version, "agent_dependency_graph.v1")
        self.assertEqual(graph.node_count, 30)
        self.assertEqual(len(graph.edges), graph.edge_count)
        self.assertEqual(
            {node.agent_id for node in graph.nodes if node.node_type == "agent"},
            set(contract_registry.agent_ids),
        )
        self.assertEqual(
            {
                node.agent_id
                for node in graph.nodes
                if node.node_type == "shared_context_view"
            },
            set(view_registry.agent_ids),
        )
        self.assertEqual(
            graph.source_registries,
            (
                "agent_contract_registry",
                "shared_context_view_registry",
                "blackboard_memory_registry",
            ),
        )
        self.assertFalse(graph.scheduling_implemented)
        self.assertFalse(graph.runtime_orchestration_implemented)
        self.assertFalse(graph.workflow_node_order_changed)
        self.assertTrue(graph.metadata_only)

    def test_nodes_and_edges_are_static_metadata(self) -> None:
        graph = agent_dependency_graph_registry()

        for node in graph.nodes:
            dumped = node.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_NODE_FIELDS)
            self.assertEqual(node.serialization_version, "agent_dependency_node.v1")
            self.assertIn(
                "graph_scheduling_execution",
                node.blocked_runtime_behaviors,
            )
            self.assertFalse(node.scheduling_implemented)
            self.assertFalse(node.runtime_orchestration_implemented)
            self.assertFalse(node.workflow_node_order_changed)
            self.assertTrue(node.metadata_only)

        for edge in graph.edges:
            dumped = edge.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_EDGE_FIELDS)
            self.assertEqual(edge.serialization_version, "agent_dependency_edge.v1")
            self.assertIn("does not schedule", edge.dependency_boundary)
            self.assertFalse(edge.scheduling_implemented)
            self.assertFalse(edge.runtime_orchestration_implemented)
            self.assertTrue(edge.metadata_only)

    def test_graph_order_is_acyclic_and_blocks_self_edges(self) -> None:
        graph = agent_dependency_graph_registry()
        order_index = {
            node_id: index for index, node_id in enumerate(graph.topological_node_order)
        }

        for edge in graph.edges:
            self.assertNotEqual(edge.from_node_id, edge.to_node_id)
            self.assertLess(
                order_index[edge.from_node_id], order_index[edge.to_node_id]
            )

        first_edge = graph.edges[0]
        cyclic_edge = first_edge.model_copy(
            update={
                "from_node_id": first_edge.to_node_id,
                "to_node_id": first_edge.from_node_id,
            }
        )
        with self.assertRaisesRegex(ValueError, "dependency graph must be acyclic"):
            AgentDependencyGraphRegistry(
                nodes=graph.nodes,
                edges=(cyclic_edge,) + graph.edges[1:],
                node_ids=graph.node_ids,
                edge_ids=graph.edge_ids,
                stage_order=graph.stage_order,
                topological_node_order=graph.topological_node_order,
                node_count=graph.node_count,
                edge_count=graph.edge_count,
                source_registries=graph.source_registries,
                blocked_cyclic_patterns=graph.blocked_cyclic_patterns,
            )

    def test_lookup_exposes_upstream_and_downstream_metadata(self) -> None:
        graph = agent_dependency_graph_registry()
        planner_agent_node = agent_dependency_node_by_id("agent::planner_agent")
        missing_node = agent_dependency_node_by_id("missing")
        planner_upstream = agent_dependency_upstream_nodes("agent::planner_agent")
        planner_downstream = agent_dependency_downstream_nodes("agent::planner_agent")

        self.assertIsNone(missing_node)
        self.assertIsNotNone(planner_agent_node)
        assert planner_agent_node is not None
        self.assertIn(
            "context_view::planner_agent",
            planner_agent_node.upstream_node_ids,
        )
        self.assertIn(
            "stage::domain_context",
            planner_agent_node.downstream_node_ids,
        )
        self.assertEqual(planner_upstream[0].node_id, "context_view::planner_agent")
        self.assertEqual(planner_downstream[0].node_id, "stage::domain_context")
        self.assertEqual(agent_dependency_upstream_nodes("missing"), ())
        self.assertEqual(graph.nodes[0].node_id, "stage::foundational_context")

    def test_graph_serializes_without_runtime_or_workflow_changes(self) -> None:
        graph = agent_dependency_graph_registry()
        dumped = graph.model_dump(mode="json")

        self.assertEqual(dumped["role"], "agent_dependency_graph_registry")
        self.assertEqual(dumped["node_count"], 30)
        self.assertEqual(len(dumped["nodes"]), 30)
        self.assertEqual(len(dumped["edges"]), graph.edge_count)
        self.assertEqual(
            ASSISTANT_WORKFLOW_NODE_ORDER,
            (
                "intake",
                "routing",
                "memory",
                "retrieval",
                "context_assembly",
                "prompt_input",
                "planning",
                "director",
                "reasoning",
                "prompt_rendering",
                "generation",
                "artifact_extraction",
                "preview_preparation",
                "artifact_critique",
                "review",
                "refinement",
                "finalization",
                "failure",
            ),
        )
        combined_text = " ".join(
            (
                graph.authority_boundary,
                *graph.blocked_cyclic_patterns,
                *graph.blocked_runtime_behaviors,
            )
        )
        for forbidden_term in (
            "execute_schedule",
            "runtime_orchestrator",
            "workflow_reorder",
            "provider_route",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
