import unittest

from creative_coding_assistant.orchestration import (
    UnifiedMemoryGraphPlan,
    build_unified_cognitive_graph,
    build_unified_memory_graph,
    unified_memory_graph_edge_by_id,
    unified_memory_graph_node_by_id,
    unified_memory_graph_nodes_for_layer,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
)


class UnifiedMemoryGraphTests(unittest.TestCase):
    def test_unified_memory_graph_extends_cognitive_backbone(self) -> None:
        cognitive_graph = build_unified_cognitive_graph()
        memory_graph = build_unified_memory_graph(cognitive_graph=cognitive_graph)

        self.assertEqual(memory_graph.role, "unified_memory_graph")
        self.assertEqual(memory_graph.serialization_version, "unified_memory_graph.v1")
        self.assertEqual(memory_graph.backbone_graph_role, cognitive_graph.role)
        self.assertEqual(memory_graph.backbone_node_ids, cognitive_graph.node_ids)
        self.assertEqual(memory_graph.backbone_edge_ids, cognitive_graph.edge_ids)
        self.assertEqual(memory_graph.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(memory_graph.covered_roadmap_items, ("Unified Memory Graph",))
        self.assertEqual(memory_graph.covered_roadmap_item_count, 1)
        self.assertEqual(len(memory_graph.memory_nodes), 6)
        self.assertEqual(len(memory_graph.memory_edges), 9)
        self.assertEqual(
            memory_graph.creative_memory_node_id,
            "memory::v6_2_memory_node",
        )
        self.assertEqual(memory_graph.source_signal_id_count, 131)
        self.assertEqual(memory_graph.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            memory_graph.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(memory_graph.unified_memory_graph_implemented)
        self.assertTrue(memory_graph.unified_cognitive_graph_integrated)
        self.assertTrue(memory_graph.memory_ownership_boundary_check_implemented)
        self.assertTrue(memory_graph.memory_dependency_traceability_implemented)
        self.assertTrue(memory_graph.memory_explainability_contract_implemented)
        self.assertTrue(memory_graph.memory_hitl_governance_contract_implemented)
        self.assertFalse(memory_graph.memory_persistence_implemented)
        self.assertFalse(memory_graph.memory_mutation_implemented)
        self.assertFalse(memory_graph.retrieval_mutation_implemented)
        self.assertFalse(memory_graph.storage_mutation_implemented)
        self.assertFalse(memory_graph.provider_execution_implemented)
        self.assertFalse(memory_graph.generated_output_mutation_implemented)
        self.assertFalse(memory_graph.runtime_evolution_implemented)
        self.assertFalse(memory_graph.persisted_memory_record_ids)
        self.assertFalse(memory_graph.mutated_memory_record_ids)
        self.assertFalse(memory_graph.mutated_retrieval_record_ids)
        self.assertFalse(memory_graph.written_storage_record_ids)
        self.assertFalse(memory_graph.emitted_hitl_request_ids)
        self.assertTrue(memory_graph.advisory_only)

    def test_unified_memory_graph_exposes_memory_lookup_helpers(self) -> None:
        memory_graph = build_unified_memory_graph()

        memory_node = unified_memory_graph_node_by_id(
            "memory::v6_2_memory_node",
            memory_graph,
        )
        self.assertIsNotNone(memory_node)
        assert memory_node is not None
        self.assertEqual(memory_node.layer, "memory")
        self.assertEqual(memory_node.capability, "V6.2 Creative Memory")
        self.assertIn("creative memory", memory_node.memory_role)

        memory_nodes = unified_memory_graph_nodes_for_layer("memory", memory_graph)
        self.assertEqual(len(memory_nodes), 1)
        self.assertEqual(memory_nodes[0].memory_node_id, "memory::v6_2_memory_node")

        edge = unified_memory_graph_edge_by_id(
            "memory::v6_2_memory_node->memory::v6_3_knowledge_node",
            memory_graph,
        )
        self.assertIsNotNone(edge)
        assert edge is not None
        self.assertEqual(
            edge.cognitive_edge_id,
            "v6_2_memory_node->v6_3_knowledge_node",
        )
        self.assertIn("does not authorize mutation", edge.governance_trace[0])

    def test_unified_memory_graph_rejects_mutation_and_backbone_drift(self) -> None:
        memory_graph = build_unified_memory_graph()
        payload = memory_graph.model_dump(mode="json")
        payload["backbone_node_ids"] = ("missing",) + tuple(
            payload["backbone_node_ids"][1:]
        )

        with self.assertRaisesRegex(ValueError, "memory nodes must follow"):
            UnifiedMemoryGraphPlan(**payload)

        payload = memory_graph.model_dump(mode="json")
        payload["mutated_memory_record_ids"] = ("memory::v6_2_memory_node",)

        with self.assertRaisesRegex(
            ValueError,
            "memory mutation, storage, and HITL ids must be empty",
        ):
            UnifiedMemoryGraphPlan(**payload)

    def test_unified_memory_graph_reuses_supplied_cognitive_graph(self) -> None:
        cognitive_graph = build_unified_cognitive_graph(route="generate")
        memory_graph = build_unified_memory_graph(cognitive_graph=cognitive_graph)

        self.assertEqual(memory_graph.route_name, cognitive_graph.route_name)
        self.assertEqual(memory_graph.task_type, cognitive_graph.task_type)
        self.assertEqual(memory_graph.backbone_edge_ids, cognitive_graph.edge_ids)


if __name__ == "__main__":
    unittest.main()
