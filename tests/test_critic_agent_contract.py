import unittest

from creative_coding_assistant.orchestration import (
    agent_contract_by_id,
    agent_contract_registry,
    agent_identity_by_id,
    agent_memory_contract_by_agent_id,
)


class CriticAgentContractTests(unittest.TestCase):
    def test_critic_agent_contract_is_registered(self) -> None:
        registry = agent_contract_registry()
        contract = agent_contract_by_id("critic_agent")
        identity = agent_identity_by_id("critic_agent")
        memory_contract = agent_memory_contract_by_agent_id("critic_agent")

        self.assertIsNotNone(contract)
        self.assertIsNotNone(identity)
        self.assertIsNotNone(memory_contract)
        assert contract is not None
        assert identity is not None
        assert memory_contract is not None
        self.assertIn("critic_agent", registry.agent_ids)
        self.assertEqual(contract.agent_id, identity.agent_id)
        self.assertEqual(contract.agent_name, identity.agent_name)
        self.assertEqual(contract.agent_version, identity.identity_version)
        self.assertEqual(contract.role_id, "critic")
        self.assertEqual(contract.role_name, "Critic Agent")
        self.assertEqual(
            memory_contract.readable_memory_surfaces,
            ("session", "artifact", "evaluation", "provenance"),
        )

    def test_critic_maps_evaluation_and_consistency_boundaries(self) -> None:
        contract = agent_contract_by_id("critic_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(
            contract.required_inputs,
            (
                "assistant_request",
                "evaluation_report_profile",
                "consistency_validation_profile",
            ),
        )
        self.assertIn("self_evaluation_profile", contract.optional_inputs)
        self.assertIn("creative_confidence_profile", contract.optional_inputs)
        self.assertIn("aesthetic_review_context", contract.optional_inputs)
        self.assertIn("critique_scope_metadata", contract.produced_metadata)
        self.assertIn("consistency_review_metadata", contract.produced_metadata)
        self.assertIn("evaluation_gap_metadata", contract.produced_metadata)
        self.assertIn("critique_confidence", contract.produced_signals)
        self.assertIn("quality_risk", contract.produced_signals)
        self.assertIn("review_priority", contract.produced_signals)
        self.assertIn(
            "critic_context_packet_contract",
            contract.produced_outputs,
        )

    def test_critic_contract_is_metadata_only(self) -> None:
        contract = agent_contract_by_id("critic_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertTrue(contract.metadata_only)
        self.assertEqual(contract.cacheability, "deterministic_with_upstream_metadata")
        self.assertFalse(contract.estimated_cost_metadata.external_provider_calls)
        self.assertFalse(contract.estimated_latency_metadata.network_required)
        self.assertFalse(contract.memory_access.reads_runtime_memory)
        self.assertFalse(contract.memory_access.writes_runtime_memory)
        self.assertIn("does not change scoring", contract.authority_boundary)
        self.assertIn("scoring_change", contract.prohibited_actions)
        self.assertIn("retry_triggering", contract.prohibited_actions)
        self.assertIn("evaluation_engine_execution", contract.prohibited_actions)

    def test_critic_does_not_declare_scoring_or_retries(self) -> None:
        contract = agent_contract_by_id("critic_agent")

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
            "recalculate_score",
            "trigger_retry",
            "execute_provider",
            "runtime_auto_selection",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
