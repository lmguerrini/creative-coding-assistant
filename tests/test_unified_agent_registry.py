import unittest

from creative_coding_assistant.orchestration import (
    UnifiedAgentRegistryPlan,
    agent_contract_registry,
    agent_registry_audit_registry,
    build_unified_agent_registry,
    build_unified_knowledge_graph,
    unified_agent_registry_entries_for_layer,
    unified_agent_registry_entries_for_role_family,
    unified_agent_registry_entry_by_id,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
)

EXPECTED_SOURCE_REGISTRY_IDS = (
    "agent_identity_registry",
    "agent_contract_registry",
    "agent_role_registry",
    "agent_metadata_registry",
    "agent_capability_registry",
    "agent_registry_audit_registry",
)


class UnifiedAgentRegistryTests(unittest.TestCase):
    def test_unified_agent_registry_composes_audited_sources(self) -> None:
        knowledge_graph = build_unified_knowledge_graph()
        registry = build_unified_agent_registry(knowledge_graph=knowledge_graph)
        contract_registry = agent_contract_registry()
        audit_registry = agent_registry_audit_registry()

        self.assertEqual(registry.role, "unified_agent_registry")
        self.assertEqual(
            registry.serialization_version,
            "unified_agent_registry.v1",
        )
        self.assertEqual(registry.knowledge_graph_role, knowledge_graph.role)
        self.assertEqual(
            registry.knowledge_node_ids,
            knowledge_graph.knowledge_node_ids,
        )
        self.assertEqual(registry.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(registry.source_registry_ids, EXPECTED_SOURCE_REGISTRY_IDS)
        self.assertEqual(registry.source_registry_count, 6)
        self.assertEqual(registry.audited_registry_ids, audit_registry.registry_ids)
        self.assertEqual(registry.audited_registry_count, 20)
        self.assertEqual(registry.agent_ids, contract_registry.agent_ids)
        self.assertEqual(registry.agent_count, 12)
        self.assertEqual(registry.role_count, 12)
        self.assertEqual(registry.capability_profile_count, 6)
        self.assertEqual(len(registry.agent_entries), 12)
        self.assertEqual(
            registry.covered_roadmap_items,
            ("Unified Agent Registry",),
        )
        self.assertEqual(registry.covered_roadmap_item_count, 1)
        self.assertEqual(registry.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            registry.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(registry.unified_agent_registry_implemented)
        self.assertTrue(registry.unified_knowledge_graph_integrated)
        self.assertTrue(registry.agent_registry_audit_integrated)
        self.assertTrue(registry.agent_identity_alignment_implemented)
        self.assertTrue(registry.agent_contract_alignment_implemented)
        self.assertTrue(registry.agent_role_alignment_implemented)
        self.assertTrue(registry.agent_metadata_alignment_implemented)
        self.assertTrue(registry.agent_dependency_traceability_implemented)
        self.assertTrue(registry.agent_governance_contract_implemented)
        self.assertTrue(registry.agent_explainability_contract_implemented)
        self.assertFalse(registry.agent_creation_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.dynamic_agent_routing_implemented)
        self.assertFalse(registry.provider_execution_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.runtime_evolution_implemented)
        self.assertFalse(registry.created_agent_ids)
        self.assertFalse(registry.invoked_agent_ids)
        self.assertFalse(registry.routed_agent_ids)
        self.assertFalse(registry.mutated_agent_registry_ids)
        self.assertFalse(registry.emitted_hitl_request_ids)
        self.assertTrue(registry.advisory_only)

    def test_unified_agent_registry_lookup_helpers_are_layer_aware(self) -> None:
        registry = build_unified_agent_registry()

        planner = unified_agent_registry_entry_by_id("planner_agent", registry)
        self.assertIsNotNone(planner)
        assert planner is not None
        self.assertEqual(planner.role_id, "planner")
        self.assertEqual(planner.role_family, "planning")
        self.assertEqual(planner.cognitive_layer, "cognitive_core")
        self.assertEqual(
            planner.knowledge_node_id,
            "knowledge::v6_6_cognitive_core_node",
        )
        self.assertEqual(planner.source_registry_ids, EXPECTED_SOURCE_REGISTRY_IDS)
        self.assertIn("agent_instantiation", planner.agent_blocked_runtime_behaviors)
        self.assertIn("agent creation", planner.governance_contracts[0])

        research_entries = unified_agent_registry_entries_for_layer(
            "research",
            registry,
        )
        self.assertEqual(
            tuple(entry.agent_id for entry in research_entries), ("research_agent",)
        )

        critique_entries = unified_agent_registry_entries_for_role_family(
            "critique",
            registry,
        )
        self.assertEqual(
            tuple(entry.agent_id for entry in critique_entries),
            ("aesthetic_critic_agent", "critic_agent"),
        )
        self.assertIsNone(unified_agent_registry_entry_by_id("missing", registry))

    def test_unified_agent_registry_rejects_mutation_and_alignment_drift(
        self,
    ) -> None:
        registry = build_unified_agent_registry()
        payload = registry.model_dump(mode="json")
        payload["agent_ids"] = ("missing_agent",) + tuple(payload["agent_ids"][1:])

        with self.assertRaisesRegex(ValueError, "agent_ids must match"):
            UnifiedAgentRegistryPlan(**payload)

        payload = registry.model_dump(mode="json")
        payload["invoked_agent_ids"] = ("planner_agent",)

        with self.assertRaisesRegex(
            ValueError,
            "agent creation, invocation, routing, and mutation ids must be empty",
        ):
            UnifiedAgentRegistryPlan(**payload)

    def test_unified_agent_registry_reuses_supplied_knowledge_graph(self) -> None:
        knowledge_graph = build_unified_knowledge_graph(route="generate")
        registry = build_unified_agent_registry(knowledge_graph=knowledge_graph)

        self.assertEqual(registry.route_name, knowledge_graph.route_name)
        self.assertEqual(registry.task_type, knowledge_graph.task_type)
        self.assertEqual(
            registry.knowledge_node_ids,
            knowledge_graph.knowledge_node_ids,
        )
        self.assertEqual(
            registry.knowledge_graph_serialization_version,
            knowledge_graph.serialization_version,
        )


if __name__ == "__main__":
    unittest.main()
