import unittest

from creative_coding_assistant.orchestration import (
    UnifiedKnowledgeGraphPlan,
    build_knowledge_evolution_core_surface,
    build_unified_cognitive_graph,
    build_unified_knowledge_graph,
    build_unified_memory_graph,
    unified_knowledge_graph_edge_by_id,
    unified_knowledge_graph_node_by_id,
    unified_knowledge_graph_nodes_for_layer,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
)


class UnifiedKnowledgeGraphTests(unittest.TestCase):
    def test_unified_knowledge_graph_extends_graph_backbones(self) -> None:
        cognitive_graph = build_unified_cognitive_graph()
        memory_graph = build_unified_memory_graph(cognitive_graph=cognitive_graph)
        knowledge_graph = build_unified_knowledge_graph(memory_graph=memory_graph)

        self.assertEqual(knowledge_graph.role, "unified_knowledge_graph")
        self.assertEqual(
            knowledge_graph.serialization_version,
            "unified_knowledge_graph.v1",
        )
        self.assertEqual(knowledge_graph.backbone_graph_role, cognitive_graph.role)
        self.assertEqual(knowledge_graph.memory_graph_role, memory_graph.role)
        self.assertEqual(knowledge_graph.backbone_node_ids, cognitive_graph.node_ids)
        self.assertEqual(knowledge_graph.backbone_edge_ids, cognitive_graph.edge_ids)
        self.assertEqual(knowledge_graph.memory_node_ids, memory_graph.memory_node_ids)
        self.assertEqual(knowledge_graph.memory_edge_ids, memory_graph.memory_edge_ids)
        self.assertEqual(knowledge_graph.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(
            knowledge_graph.covered_roadmap_items,
            ("Unified Knowledge Graph",),
        )
        self.assertEqual(knowledge_graph.covered_roadmap_item_count, 1)
        self.assertEqual(
            knowledge_graph.v6_3_knowledge_node_id,
            "knowledge::v6_3_knowledge_node",
        )
        self.assertEqual(
            knowledge_graph.v6_3_knowledge_source_role,
            "knowledge_evolution_core_surface",
        )
        self.assertEqual(knowledge_graph.v6_3_entry_count, 5)
        self.assertEqual(knowledge_graph.v6_3_source_item_count, 95)
        self.assertEqual(knowledge_graph.v6_3_roadmap_item_count, 19)
        self.assertEqual(len(knowledge_graph.knowledge_nodes), 6)
        self.assertEqual(len(knowledge_graph.knowledge_edges), 9)
        self.assertEqual(knowledge_graph.source_signal_id_count, 131)
        self.assertEqual(
            knowledge_graph.cross_cutting_contracts,
            COGNITIVE_OS_CONTRACTS,
        )
        self.assertEqual(
            knowledge_graph.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(knowledge_graph.unified_knowledge_graph_implemented)
        self.assertTrue(knowledge_graph.unified_cognitive_graph_integrated)
        self.assertTrue(knowledge_graph.unified_memory_graph_integrated)
        self.assertTrue(knowledge_graph.v6_3_knowledge_surface_integrated)
        self.assertTrue(
            knowledge_graph.knowledge_ownership_boundary_check_implemented,
        )
        self.assertTrue(knowledge_graph.knowledge_dependency_traceability_implemented)
        self.assertTrue(knowledge_graph.knowledge_explainability_contract_implemented)
        self.assertTrue(knowledge_graph.knowledge_hitl_governance_contract_implemented)
        self.assertFalse(knowledge_graph.knowledge_surface_activation_implemented)
        self.assertFalse(knowledge_graph.knowledge_retrieval_execution_implemented)
        self.assertFalse(knowledge_graph.knowledge_scoring_execution_implemented)
        self.assertFalse(knowledge_graph.knowledge_storage_write_implemented)
        self.assertFalse(knowledge_graph.knowledge_source_record_update_implemented)
        self.assertFalse(knowledge_graph.knowledge_graph_mutation_implemented)
        self.assertFalse(knowledge_graph.provider_execution_implemented)
        self.assertFalse(knowledge_graph.generated_output_mutation_implemented)
        self.assertFalse(knowledge_graph.runtime_evolution_implemented)
        self.assertFalse(knowledge_graph.activated_knowledge_surface_ids)
        self.assertFalse(knowledge_graph.executed_retrieval_ids)
        self.assertFalse(knowledge_graph.computed_knowledge_score_ids)
        self.assertFalse(knowledge_graph.written_kb_record_ids)
        self.assertFalse(knowledge_graph.updated_source_record_ids)
        self.assertFalse(knowledge_graph.mutated_knowledge_node_ids)
        self.assertFalse(knowledge_graph.emitted_hitl_request_ids)
        self.assertTrue(knowledge_graph.advisory_only)

    def test_unified_knowledge_graph_exposes_lookup_helpers(self) -> None:
        knowledge_graph = build_unified_knowledge_graph()

        knowledge_node = unified_knowledge_graph_node_by_id(
            "knowledge::v6_3_knowledge_node",
            knowledge_graph,
        )
        self.assertIsNotNone(knowledge_node)
        assert knowledge_node is not None
        self.assertEqual(knowledge_node.layer, "knowledge")
        self.assertEqual(knowledge_node.capability, "V6.3 Knowledge Evolution")
        self.assertEqual(knowledge_node.knowledge_source_item_count, 95)
        self.assertIn(
            "knowledge_health_monitoring",
            knowledge_node.knowledge_source_roles,
        )

        knowledge_nodes = unified_knowledge_graph_nodes_for_layer(
            "knowledge",
            knowledge_graph,
        )
        self.assertEqual(len(knowledge_nodes), 1)
        self.assertEqual(
            knowledge_nodes[0].knowledge_node_id,
            "knowledge::v6_3_knowledge_node",
        )

        edge = unified_knowledge_graph_edge_by_id(
            "knowledge::v6_3_knowledge_node->knowledge::v6_4_research_node",
            knowledge_graph,
        )
        self.assertIsNotNone(edge)
        assert edge is not None
        self.assertEqual(
            edge.memory_edge_id,
            "memory::v6_3_knowledge_node->memory::v6_4_research_node",
        )
        self.assertEqual(
            edge.cognitive_edge_id,
            "v6_3_knowledge_node->v6_4_research_node",
        )
        self.assertIn("KB writes", edge.governance_trace[0])

    def test_unified_knowledge_graph_rejects_mutation_and_backbone_drift(
        self,
    ) -> None:
        knowledge_graph = build_unified_knowledge_graph()
        payload = knowledge_graph.model_dump(mode="json")
        payload["memory_node_ids"] = ("missing",) + tuple(
            payload["memory_node_ids"][1:]
        )

        with self.assertRaisesRegex(ValueError, "knowledge nodes must follow"):
            UnifiedKnowledgeGraphPlan(**payload)

        payload = knowledge_graph.model_dump(mode="json")
        payload["written_kb_record_ids"] = ("knowledge::v6_3_knowledge_node",)

        with self.assertRaisesRegex(
            ValueError,
            "knowledge mutation, storage, retrieval, and HITL ids must be empty",
        ):
            UnifiedKnowledgeGraphPlan(**payload)

    def test_unified_knowledge_graph_reuses_supplied_source_plans(self) -> None:
        memory_graph = build_unified_memory_graph(route="generate")
        knowledge_core = build_knowledge_evolution_core_surface(
            route="generate",
            task_type="reasoning",
        )
        knowledge_graph = build_unified_knowledge_graph(
            memory_graph=memory_graph,
            knowledge_core=knowledge_core,
        )

        self.assertEqual(knowledge_graph.route_name, memory_graph.route_name)
        self.assertEqual(knowledge_graph.task_type, memory_graph.task_type)
        self.assertEqual(
            knowledge_graph.v6_3_source_plan_roles,
            knowledge_core.source_plan_roles,
        )
        self.assertEqual(
            knowledge_graph.v6_3_source_item_ids,
            knowledge_core.source_item_ids,
        )
        self.assertEqual(knowledge_graph.v6_3_entry_ids, knowledge_core.entry_ids)


if __name__ == "__main__":
    unittest.main()
