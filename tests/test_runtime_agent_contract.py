import unittest

from creative_coding_assistant.orchestration import (
    agent_contract_by_id,
    agent_contract_registry,
    agent_identity_by_id,
    agent_memory_contract_by_agent_id,
)


class RuntimeAgentContractTests(unittest.TestCase):
    def test_runtime_agent_contract_is_registered(self) -> None:
        registry = agent_contract_registry()
        contract = agent_contract_by_id("runtime_agent")
        identity = agent_identity_by_id("runtime_agent")
        memory_contract = agent_memory_contract_by_agent_id("runtime_agent")

        self.assertIsNotNone(contract)
        self.assertIsNotNone(identity)
        self.assertIsNotNone(memory_contract)
        assert contract is not None
        assert identity is not None
        assert memory_contract is not None
        self.assertIn("runtime_agent", registry.agent_ids)
        self.assertEqual(contract.agent_id, identity.agent_id)
        self.assertEqual(contract.agent_name, identity.agent_name)
        self.assertEqual(contract.agent_version, identity.identity_version)
        self.assertEqual(contract.role_id, "runtime")
        self.assertEqual(contract.role_name, "Runtime Agent")
        self.assertEqual(
            memory_contract.readable_memory_surfaces,
            ("session", "artifact", "provenance"),
        )

    def test_runtime_agent_maps_existing_runtime_metadata(self) -> None:
        contract = agent_contract_by_id("runtime_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(
            contract.required_inputs,
            (
                "assistant_request",
                "runtime_capabilities",
                "runtime_compatibility_profile",
            ),
        )
        self.assertIn("artifact_capability_matrix", contract.optional_inputs)
        self.assertIn("creative_execution_plan", contract.optional_inputs)
        self.assertIn("workstation_engine_contract_registry", contract.optional_inputs)
        self.assertIn("runtime_capability_metadata", contract.produced_metadata)
        self.assertIn("runtime_compatibility_metadata", contract.produced_metadata)
        self.assertIn("runtime_fit_metadata", contract.produced_metadata)
        self.assertIn("runtime_confidence", contract.produced_signals)
        self.assertIn("runtime_fit_status", contract.produced_signals)
        self.assertIn("environment_risk", contract.produced_signals)
        self.assertIn(
            "runtime_context_packet_contract",
            contract.produced_outputs,
        )

    def test_runtime_agent_contract_is_metadata_only(self) -> None:
        contract = agent_contract_by_id("runtime_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertTrue(contract.metadata_only)
        self.assertEqual(contract.cacheability, "deterministic_with_upstream_metadata")
        self.assertFalse(contract.estimated_cost_metadata.external_provider_calls)
        self.assertFalse(contract.estimated_latency_metadata.network_required)
        self.assertFalse(contract.memory_access.reads_runtime_memory)
        self.assertFalse(contract.memory_access.writes_runtime_memory)
        self.assertIn(
            "does not change runtime selection",
            contract.authority_boundary,
        )
        self.assertIn("runtime_selection_change", contract.prohibited_actions)
        self.assertIn("runtime_decision_execution", contract.prohibited_actions)
        self.assertIn("provider_or_model_routing", contract.prohibited_actions)

    def test_runtime_contract_does_not_declare_runtime_decisions(self) -> None:
        contract = agent_contract_by_id("runtime_agent")

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
            "select_runtime",
            "execute_runtime",
            "runtime_auto_selection",
            "execute_provider",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
