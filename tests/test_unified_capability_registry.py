import unittest

from creative_coding_assistant.orchestration import (
    UnifiedCapabilityRegistryPlan,
    agent_capability_registry,
    build_model_capability_matrix,
    build_provider_capability_matrix,
    build_unified_agent_registry,
    build_unified_capability_registry,
    unified_capability_registry_entries_for_agent,
    unified_capability_registry_entry_by_id,
    unified_capability_registry_entry_for_layer,
)
from creative_coding_assistant.orchestration.cognitive_os_common import (
    COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
    COGNITIVE_OS_CAPABILITIES,
    COGNITIVE_OS_CONTRACTS,
    COGNITIVE_OS_LAYER_ORDER,
)

EXPECTED_SOURCE_REGISTRY_IDS = (
    "unified_agent_registry",
    "agent_capability_registry",
    "model_capability_matrix",
    "provider_capability_matrix",
)


class UnifiedCapabilityRegistryTests(unittest.TestCase):
    def test_unified_capability_registry_composes_capability_sources(self) -> None:
        agent_registry = build_unified_agent_registry()
        registry = build_unified_capability_registry(agent_registry=agent_registry)

        self.assertEqual(registry.role, "unified_capability_registry")
        self.assertEqual(
            registry.serialization_version,
            "unified_capability_registry.v1",
        )
        self.assertEqual(registry.agent_registry_role, agent_registry.role)
        self.assertEqual(registry.knowledge_node_ids, agent_registry.knowledge_node_ids)
        self.assertEqual(registry.layer_order, COGNITIVE_OS_LAYER_ORDER)
        self.assertEqual(registry.capabilities, COGNITIVE_OS_CAPABILITIES)
        self.assertEqual(registry.source_registry_ids, EXPECTED_SOURCE_REGISTRY_IDS)
        self.assertEqual(registry.source_registry_count, 4)
        self.assertEqual(registry.agent_ids, agent_registry.agent_ids)
        self.assertEqual(registry.agent_count, 12)
        self.assertEqual(registry.agent_capability_profile_count, 6)
        self.assertEqual(registry.model_capability_row_count, 4)
        self.assertEqual(registry.model_capability_dimension_count, 12)
        self.assertEqual(registry.provider_capability_row_count, 4)
        self.assertEqual(registry.provider_candidate_count, 5)
        self.assertEqual(len(registry.capability_entries), 6)
        self.assertEqual(registry.capability_count, 6)
        self.assertEqual(
            registry.covered_roadmap_items,
            ("Unified Capability Registry",),
        )
        self.assertEqual(registry.covered_roadmap_item_count, 1)
        self.assertEqual(registry.cross_cutting_contracts, COGNITIVE_OS_CONTRACTS)
        self.assertEqual(
            registry.blocked_runtime_behaviors,
            COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS,
        )
        self.assertTrue(registry.unified_capability_registry_implemented)
        self.assertTrue(registry.unified_agent_registry_integrated)
        self.assertTrue(registry.agent_capability_registry_integrated)
        self.assertTrue(registry.model_capability_matrix_integrated)
        self.assertTrue(registry.provider_capability_matrix_integrated)
        self.assertTrue(registry.capability_lookup_implemented)
        self.assertTrue(registry.capability_dependency_traceability_implemented)
        self.assertTrue(registry.capability_governance_contract_implemented)
        self.assertTrue(registry.capability_explainability_contract_implemented)
        self.assertFalse(registry.capability_scoring_implemented)
        self.assertFalse(registry.capability_activation_implemented)
        self.assertFalse(registry.agent_routing_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.provider_execution_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertFalse(registry.runtime_evolution_implemented)
        self.assertFalse(registry.activated_capability_ids)
        self.assertFalse(registry.scored_capability_ids)
        self.assertFalse(registry.routed_capability_ids)
        self.assertFalse(registry.mutated_capability_registry_ids)
        self.assertFalse(registry.emitted_hitl_request_ids)
        self.assertTrue(registry.advisory_only)

    def test_unified_capability_registry_lookup_helpers_are_agent_aware(
        self,
    ) -> None:
        registry = build_unified_capability_registry()

        core = unified_capability_registry_entry_by_id(
            "v6_6_cognitive_core",
            registry,
        )
        self.assertIsNotNone(core)
        assert core is not None
        self.assertEqual(core.capability_name, "V6.6 Cognitive Core")
        self.assertEqual(core.cognitive_layer, "cognitive_core")
        self.assertEqual(
            core.knowledge_node_id,
            "knowledge::v6_6_cognitive_core_node",
        )
        self.assertIn("planner_agent", core.linked_agent_ids)
        self.assertEqual(core.source_registry_ids, EXPECTED_SOURCE_REGISTRY_IDS)
        self.assertIn("capability activation", core.governance_contracts[0])

        research = unified_capability_registry_entry_for_layer("research", registry)
        self.assertIsNotNone(research)
        assert research is not None
        self.assertEqual(research.capability_id, "v6_4_autonomous_research")
        self.assertEqual(research.linked_agent_ids, ("research_agent",))

        planner_capabilities = unified_capability_registry_entries_for_agent(
            "planner_agent",
            registry,
        )
        self.assertEqual(
            tuple(entry.capability_id for entry in planner_capabilities),
            ("v6_6_cognitive_core",),
        )
        self.assertIsNone(
            unified_capability_registry_entry_by_id("missing", registry),
        )

    def test_unified_capability_registry_rejects_activation_and_drift(self) -> None:
        registry = build_unified_capability_registry()
        payload = registry.model_dump(mode="json")
        payload["capability_ids"] = ("missing",) + tuple(
            payload["capability_ids"][1:]
        )

        with self.assertRaisesRegex(ValueError, "capability_ids must match"):
            UnifiedCapabilityRegistryPlan(**payload)

        payload = registry.model_dump(mode="json")
        payload["scored_capability_ids"] = ("v6_6_cognitive_core",)

        with self.assertRaisesRegex(
            ValueError,
            "capability activation, scoring, routing, and mutation ids must be empty",
        ):
            UnifiedCapabilityRegistryPlan(**payload)

    def test_unified_capability_registry_reuses_supplied_sources(self) -> None:
        agent_registry = build_unified_agent_registry(route="generate")
        agent_capabilities = agent_capability_registry()
        model_capabilities = build_model_capability_matrix()
        provider_capabilities = build_provider_capability_matrix()

        registry = build_unified_capability_registry(
            agent_registry=agent_registry,
            agent_capabilities=agent_capabilities,
            model_capabilities=model_capabilities,
            provider_capabilities=provider_capabilities,
        )

        self.assertEqual(registry.route_name, agent_registry.route_name)
        self.assertEqual(registry.task_type, agent_registry.task_type)
        self.assertEqual(
            registry.agent_capability_profile_ids,
            agent_capabilities.capability_ids,
        )
        self.assertEqual(registry.model_capability_row_ids, model_capabilities.row_ids)
        self.assertEqual(
            registry.provider_capability_row_ids,
            provider_capabilities.row_ids,
        )


if __name__ == "__main__":
    unittest.main()
