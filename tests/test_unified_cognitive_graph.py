import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    UnifiedCognitiveGraphPlan,
    build_unified_cognitive_graph,
    route_request,
    unified_cognitive_graph_edge_by_id,
    unified_cognitive_graph_node_by_id,
    unified_cognitive_graph_nodes_for_layer,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
)


class UnifiedCognitiveGraphTests(unittest.TestCase):
    def test_unified_cognitive_graph_links_all_v6_layers(self) -> None:
        graph = build_unified_cognitive_graph()

        self.assertEqual(graph.role, "unified_cognitive_graph")
        self.assertEqual(graph.serialization_version, "unified_cognitive_graph.v1")
        self.assertEqual(graph.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(graph.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(graph.covered_roadmap_items, ("Unified Cognitive Graph",))
        self.assertEqual(graph.covered_roadmap_item_count, 1)
        self.assertEqual(graph.source_snapshot_ids[0], "v6_1_adaptive_learning")
        self.assertEqual(graph.source_snapshot_ids[-1], "v6_6_cognitive_core")
        self.assertEqual(len(graph.nodes), 6)
        self.assertEqual(len(graph.edges), 9)
        self.assertEqual(graph.upstream_signal_id_count, 131)
        self.assertEqual(graph.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            graph.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertEqual(graph.graph_posture, "guarded")
        self.assertTrue(graph.unified_cognitive_graph_implemented)
        self.assertTrue(graph.cognitive_os_backbone_implemented)
        self.assertTrue(graph.cross_capability_dependency_awareness_implemented)
        self.assertTrue(graph.cross_capability_governance_audit_implemented)
        self.assertTrue(graph.capability_ownership_boundary_check_implemented)
        self.assertTrue(graph.unified_graph_consistency_implemented)
        self.assertTrue(graph.cognitive_explainability_contract_implemented)
        self.assertTrue(graph.cognitive_hitl_governance_contract_implemented)
        self.assertFalse(graph.runtime_evolution_implemented)
        self.assertFalse(graph.workflow_mutation_implemented)
        self.assertFalse(graph.routing_mutation_implemented)
        self.assertFalse(graph.prompt_mutation_implemented)
        self.assertFalse(graph.memory_mutation_implemented)
        self.assertFalse(graph.retrieval_mutation_implemented)
        self.assertFalse(graph.storage_mutation_implemented)
        self.assertFalse(graph.provider_execution_implemented)
        self.assertFalse(graph.generated_output_mutation_implemented)
        self.assertFalse(graph.applied_graph_mutation_ids)
        self.assertFalse(graph.mutated_node_ids)
        self.assertFalse(graph.mutated_edge_ids)
        self.assertFalse(graph.emitted_hitl_request_ids)
        self.assertTrue(graph.advisory_only)

    def test_unified_cognitive_graph_exposes_nodes_and_edges(self) -> None:
        graph = build_unified_cognitive_graph(route="generate")

        core_node = unified_cognitive_graph_node_by_id(
            "v6_6_cognitive_core_node",
            graph,
        )
        self.assertIsNotNone(core_node)
        assert core_node is not None
        self.assertEqual(core_node.layer, "cognitive_core")
        self.assertEqual(
            core_node.upstream_node_ids,
            (
                "v6_1_learning_node",
                "v6_2_memory_node",
                "v6_3_knowledge_node",
                "v6_4_research_node",
                "v6_5_self_evolution_node",
            ),
        )
        learning_nodes = unified_cognitive_graph_nodes_for_layer("learning", graph)
        self.assertEqual(len(learning_nodes), 1)
        self.assertEqual(learning_nodes[0].capability, "V6.1 Adaptive Learning")

        edge = unified_cognitive_graph_edge_by_id(
            "v6_5_self_evolution_node->v6_6_cognitive_core_node",
            graph,
        )
        self.assertIsNotNone(edge)
        assert edge is not None
        self.assertEqual(
            edge.relationship,
            "evolution_governance_feeds_cognitive_core",
        )
        self.assertIn("HITL required", edge.governance_trace[1])

    def test_unified_cognitive_graph_rejects_drift_and_mutation(self) -> None:
        graph = build_unified_cognitive_graph()
        payload = graph.model_dump(mode="json")
        payload["edge_ids"] = ("bad-edge",) + tuple(payload["edge_ids"][1:])

        with self.assertRaisesRegex(ValueError, "edge_ids must match"):
            UnifiedCognitiveGraphPlan(**payload)

        payload = graph.model_dump(mode="json")
        payload["mutated_node_ids"] = ("v6_6_cognitive_core_node",)

        with self.assertRaisesRegex(
            ValueError,
            "graph mutation and HITL emission ids must be empty",
        ):
            UnifiedCognitiveGraphPlan(**payload)

    def test_unified_cognitive_graph_preserves_routing_decisions(self) -> None:
        request = AssistantRequest(
            query="Unify cognitive graph metadata.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        graph = build_unified_cognitive_graph()
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(graph.route_name, baseline_decision.route)


if __name__ == "__main__":
    unittest.main()
