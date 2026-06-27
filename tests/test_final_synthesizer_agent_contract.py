import unittest

from creative_coding_assistant.orchestration import (
    agent_contract_by_id,
    agent_contract_registry,
    agent_identity_by_id,
    agent_memory_contract_by_agent_id,
)


class FinalSynthesizerAgentContractTests(unittest.TestCase):
    def test_final_synthesizer_agent_contract_is_registered(self) -> None:
        registry = agent_contract_registry()
        contract = agent_contract_by_id("final_synthesizer_agent")
        identity = agent_identity_by_id("final_synthesizer_agent")
        memory_contract = agent_memory_contract_by_agent_id(
            "final_synthesizer_agent"
        )

        self.assertIsNotNone(contract)
        self.assertIsNotNone(identity)
        self.assertIsNotNone(memory_contract)
        assert contract is not None
        assert identity is not None
        assert memory_contract is not None
        self.assertIn("final_synthesizer_agent", registry.agent_ids)
        self.assertEqual(contract.agent_id, identity.agent_id)
        self.assertEqual(contract.agent_name, identity.agent_name)
        self.assertEqual(contract.agent_version, identity.identity_version)
        self.assertEqual(contract.role_id, "final_synthesizer")
        self.assertEqual(contract.role_name, "Final Synthesizer Agent")
        self.assertEqual(
            memory_contract.readable_memory_surfaces,
            ("session", "artifact", "evaluation", "provenance"),
        )

    def test_final_synthesizer_maps_finalization_boundaries(self) -> None:
        contract = agent_contract_by_id("final_synthesizer_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(
            contract.required_inputs,
            (
                "assistant_request",
                "workflow_finalization_metadata",
                "artifact_handoff_metadata",
            ),
        )
        self.assertIn("curation_context", contract.optional_inputs)
        self.assertIn("critic_context", contract.optional_inputs)
        self.assertIn("refinement_context", contract.optional_inputs)
        self.assertIn("finalization_scope_metadata", contract.produced_metadata)
        self.assertIn("synthesis_boundary_metadata", contract.produced_metadata)
        self.assertIn("final_response_guardrail_metadata", contract.produced_metadata)
        self.assertIn("finalization_readiness", contract.produced_signals)
        self.assertIn("synthesis_confidence", contract.produced_signals)
        self.assertIn("response_boundary_status", contract.produced_signals)
        self.assertIn(
            "final_synthesis_context_packet_contract",
            contract.produced_outputs,
        )

    def test_final_synthesizer_contract_is_metadata_only(self) -> None:
        contract = agent_contract_by_id("final_synthesizer_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertTrue(contract.metadata_only)
        self.assertEqual(contract.cacheability, "deterministic_with_upstream_metadata")
        self.assertFalse(contract.estimated_cost_metadata.external_provider_calls)
        self.assertFalse(contract.estimated_latency_metadata.network_required)
        self.assertFalse(contract.memory_access.reads_runtime_memory)
        self.assertFalse(contract.memory_access.writes_runtime_memory)
        self.assertIn(
            "does not change final response generation",
            contract.authority_boundary,
        )
        self.assertIn(
            "final_response_generation_change",
            contract.prohibited_actions,
        )
        self.assertIn("provider_call_alteration", contract.prohibited_actions)

    def test_final_synthesizer_does_not_declare_response_generation(self) -> None:
        contract = agent_contract_by_id("final_synthesizer_agent")

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
            "generate_final_response",
            "call_provider",
            "execute_provider",
            "runtime_auto_selection",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
