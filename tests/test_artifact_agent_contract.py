import unittest

from creative_coding_assistant.orchestration import (
    agent_contract_by_id,
    agent_contract_registry,
    agent_identity_by_id,
    agent_memory_contract_by_agent_id,
)


class ArtifactAgentContractTests(unittest.TestCase):
    def test_artifact_agent_contract_is_registered(self) -> None:
        registry = agent_contract_registry()
        contract = agent_contract_by_id("artifact_agent")
        identity = agent_identity_by_id("artifact_agent")
        memory_contract = agent_memory_contract_by_agent_id("artifact_agent")

        self.assertIsNotNone(contract)
        self.assertIsNotNone(identity)
        self.assertIsNotNone(memory_contract)
        assert contract is not None
        assert identity is not None
        assert memory_contract is not None
        self.assertIn("artifact_agent", registry.agent_ids)
        self.assertEqual(contract.agent_id, identity.agent_id)
        self.assertEqual(contract.agent_name, identity.agent_name)
        self.assertEqual(contract.agent_version, identity.identity_version)
        self.assertEqual(contract.role_id, "artifact")
        self.assertEqual(contract.role_name, "Artifact Agent")
        self.assertEqual(
            memory_contract.readable_memory_surfaces,
            ("session", "artifact", "provenance"),
        )

    def test_artifact_agent_maps_artifact_intelligence_boundaries(self) -> None:
        contract = agent_contract_by_id("artifact_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(
            contract.required_inputs,
            (
                "assistant_request",
                "artifact_plan",
                "artifact_engine_contract_registry",
            ),
        )
        self.assertIn("artifact_dependency_graph", contract.optional_inputs)
        self.assertIn("artifact_capability_matrix", contract.optional_inputs)
        self.assertIn("artifact_refiner_profile", contract.optional_inputs)
        self.assertIn("artifact_export_intelligence", contract.optional_inputs)
        self.assertIn("artifact_plan_metadata", contract.produced_metadata)
        self.assertIn("artifact_refinement_metadata", contract.produced_metadata)
        self.assertIn("artifact_export_metadata", contract.produced_metadata)
        self.assertIn("artifact_readiness", contract.produced_signals)
        self.assertIn("artifact_risk", contract.produced_signals)
        self.assertIn("export_readiness", contract.produced_signals)
        self.assertIn(
            "artifact_context_packet_contract",
            contract.produced_outputs,
        )

    def test_artifact_agent_contract_is_metadata_only(self) -> None:
        contract = agent_contract_by_id("artifact_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertTrue(contract.metadata_only)
        self.assertEqual(contract.cacheability, "deterministic_with_upstream_metadata")
        self.assertFalse(contract.estimated_cost_metadata.external_provider_calls)
        self.assertFalse(contract.estimated_latency_metadata.network_required)
        self.assertFalse(contract.memory_access.reads_runtime_memory)
        self.assertFalse(contract.memory_access.writes_runtime_memory)
        self.assertIn(
            "does not change artifact generation",
            contract.authority_boundary,
        )
        self.assertIn("artifact_generation_change", contract.prohibited_actions)
        self.assertIn(
            "artifact_export_behavior_change",
            contract.prohibited_actions,
        )
        self.assertIn("agent_invocation", contract.prohibited_actions)

    def test_artifact_contract_does_not_declare_generation_or_export(self) -> None:
        contract = agent_contract_by_id("artifact_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        combined_text = " ".join(
            (
                contract.authority_boundary,
                *contract.allowed_actions,
                *contract.prohibited_actions,
                *contract.capabilities,
                *contract.future_orchestration_hooks,
                *contract.blocked_runtime_behaviors,
            )
        )

        for forbidden_term in (
            "generate_artifact",
            "export_artifact",
            "execute_provider",
            "runtime_auto_selection",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
