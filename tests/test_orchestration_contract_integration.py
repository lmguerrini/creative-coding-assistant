import unittest

from creative_coding_assistant.orchestration import (
    OrchestrationContractIntegrationRegistry,
    agent_contract_registry,
    integrated_orchestration_registry_by_id,
    orchestration_contract_integration_registry,
)

REQUIRED_ENTRY_FIELDS = {
    "registry_id",
    "registry_role",
    "registry_kind",
    "export_symbol",
    "registry_serialization_version",
    "linked_agent_ids",
    "source_v4_1_registries",
    "integration_boundary",
    "blocked_runtime_behaviors",
    "metadata_only_declared",
    "active_orchestration_path_implemented",
    "runtime_mutation_implemented",
    "provider_model_routing_implemented",
    "serialization_version",
    "metadata_only",
}

EXPECTED_REGISTRY_IDS = (
    "agent_routing_registry",
    "blackboard_memory_registry",
    "shared_context_view_registry",
    "agent_dependency_graph_registry",
    "parallel_scheduling_registry",
    "agent_coordination_registry",
    "agent_debate_registry",
    "consensus_builder_registry",
    "agent_capability_alignment_registry",
    "agent_escalation_signal_registry",
    "agent_lifecycle_registry",
    "agent_state_synchronization_registry",
    "workflow_agent_handoff_registry",
)

EXPECTED_SOURCE_REGISTRIES = (
    "agent_contract_registry",
    "agent_role_registry",
    "agent_boundary_registry",
    "agent_metadata_registry",
)


class OrchestrationContractIntegrationTests(unittest.TestCase):
    def test_integration_registry_discovers_all_v4_2_registries(self) -> None:
        registry = orchestration_contract_integration_registry()
        contracts = agent_contract_registry()

        self.assertEqual(registry.role, "orchestration_contract_integration_registry")
        self.assertEqual(
            registry.serialization_version,
            "orchestration_contract_integration.v1",
        )
        self.assertEqual(registry.registry_ids, EXPECTED_REGISTRY_IDS)
        self.assertEqual(registry.agent_ids, contracts.agent_ids)
        self.assertEqual(registry.source_v4_1_registries, EXPECTED_SOURCE_REGISTRIES)
        self.assertEqual(registry.registry_count, 13)
        self.assertEqual(registry.contract_count, 12)
        self.assertIn("does not execute orchestration", registry.authority_boundary)
        self.assertFalse(registry.active_orchestration_path_implemented)
        self.assertFalse(registry.runtime_mutation_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertTrue(registry.metadata_only)

    def test_integrated_registry_entries_are_passive_and_agent_aligned(self) -> None:
        registry = orchestration_contract_integration_registry()
        known_agents = set(registry.agent_ids)

        for entry in registry.integrated_registries:
            dumped = entry.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_ENTRY_FIELDS)
            self.assertEqual(
                entry.serialization_version,
                "orchestration_integrated_registry.v1",
            )
            self.assertEqual(entry.registry_id, entry.registry_role)
            self.assertEqual(entry.export_symbol, entry.registry_id)
            self.assertEqual(entry.source_v4_1_registries, EXPECTED_SOURCE_REGISTRIES)
            self.assertEqual(set(entry.linked_agent_ids), known_agents)
            self.assertIn("orchestration_execution", entry.blocked_runtime_behaviors)
            self.assertTrue(entry.metadata_only_declared)
            self.assertFalse(entry.active_orchestration_path_implemented)
            self.assertFalse(entry.runtime_mutation_implemented)
            self.assertFalse(entry.provider_model_routing_implemented)
            self.assertTrue(entry.metadata_only)

    def test_integrated_registry_lookup_is_stable(self) -> None:
        registry = orchestration_contract_integration_registry()
        lifecycle = integrated_orchestration_registry_by_id("agent_lifecycle_registry")
        missing = integrated_orchestration_registry_by_id("missing_registry")

        self.assertIsNone(missing)
        self.assertIsNotNone(lifecycle)
        assert lifecycle is not None
        self.assertEqual(lifecycle.registry_kind, "per_agent")
        self.assertEqual(
            lifecycle.registry_serialization_version,
            "agent_lifecycle_registry.v1",
        )
        self.assertIs(
            lifecycle,
            integrated_orchestration_registry_by_id(
                "agent_lifecycle_registry",
                registry,
            ),
        )

    def test_registry_rejects_mismatched_or_unknown_integration(self) -> None:
        registry = orchestration_contract_integration_registry()
        mismatched_entry = registry.integrated_registries[0].model_copy(
            update={"registry_id": "other_registry"}
        )
        unknown_agent_entry = registry.integrated_registries[0].model_copy(
            update={"linked_agent_ids": ("missing_agent",) * 12}
        )

        with self.assertRaisesRegex(ValueError, "registry_ids must match"):
            OrchestrationContractIntegrationRegistry(
                integrated_registries=(mismatched_entry,)
                + registry.integrated_registries[1:],
                registry_ids=registry.registry_ids,
                agent_ids=registry.agent_ids,
                source_v4_1_registries=registry.source_v4_1_registries,
                registry_count=registry.registry_count,
                contract_count=registry.contract_count,
            )

        with self.assertRaisesRegex(ValueError, "linked_agent_ids must be known"):
            OrchestrationContractIntegrationRegistry(
                integrated_registries=(unknown_agent_entry,)
                + registry.integrated_registries[1:],
                registry_ids=("agent_routing_registry",) + registry.registry_ids[1:],
                agent_ids=registry.agent_ids,
                source_v4_1_registries=registry.source_v4_1_registries,
                registry_count=registry.registry_count,
                contract_count=registry.contract_count,
            )

    def test_integration_manifest_has_no_active_orchestration_terms(self) -> None:
        registry = orchestration_contract_integration_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for entry in registry.integrated_registries
                    for field in (
                        entry.registry_id,
                        entry.integration_boundary,
                        *entry.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "execute_agent",
            "run_orchestration",
            "mutate_runtime",
            "route_provider",
            "write_memory",
            "modify_output",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
