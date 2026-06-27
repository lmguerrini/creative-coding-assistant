import unittest

from creative_coding_assistant.orchestration import (
    agent_contract_by_id,
    agent_contract_registry,
    agent_identity_by_id,
    agent_memory_contract_by_agent_id,
)


class AestheticCriticAgentContractTests(unittest.TestCase):
    def test_aesthetic_critic_agent_contract_is_registered(self) -> None:
        registry = agent_contract_registry()
        contract = agent_contract_by_id("aesthetic_critic_agent")
        identity = agent_identity_by_id("aesthetic_critic_agent")
        memory_contract = agent_memory_contract_by_agent_id(
            "aesthetic_critic_agent"
        )

        self.assertIsNotNone(contract)
        self.assertIsNotNone(identity)
        self.assertIsNotNone(memory_contract)
        assert contract is not None
        assert identity is not None
        assert memory_contract is not None
        self.assertIn("aesthetic_critic_agent", registry.agent_ids)
        self.assertEqual(contract.agent_id, identity.agent_id)
        self.assertEqual(contract.agent_name, identity.agent_name)
        self.assertEqual(contract.agent_version, identity.identity_version)
        self.assertEqual(contract.role_id, "aesthetic_critic")
        self.assertEqual(contract.role_name, "Aesthetic Critic Agent")
        self.assertEqual(
            memory_contract.readable_memory_surfaces,
            ("session", "artifact", "evaluation", "provenance"),
        )

    def test_aesthetic_critic_maps_evaluation_boundaries(self) -> None:
        contract = agent_contract_by_id("aesthetic_critic_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(
            contract.required_inputs,
            (
                "assistant_request",
                "creative_quality_evaluation",
                "creative_critic_profile",
            ),
        )
        self.assertIn("quality_calibration", contract.optional_inputs)
        self.assertIn("creative_score_profile", contract.optional_inputs)
        self.assertIn("consistency_validation_profile", contract.optional_inputs)
        self.assertIn("aesthetic_quality_metadata", contract.produced_metadata)
        self.assertIn("visual_critique_metadata", contract.produced_metadata)
        self.assertIn("evaluation_boundary_metadata", contract.produced_metadata)
        self.assertIn("aesthetic_confidence", contract.produced_signals)
        self.assertIn("visual_quality_alignment", contract.produced_signals)
        self.assertIn("calibration_gap", contract.produced_signals)
        self.assertIn(
            "aesthetic_review_context_contract",
            contract.produced_outputs,
        )

    def test_aesthetic_critic_contract_is_metadata_only(self) -> None:
        contract = agent_contract_by_id("aesthetic_critic_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertTrue(contract.metadata_only)
        self.assertEqual(contract.cacheability, "deterministic_with_upstream_metadata")
        self.assertFalse(contract.estimated_cost_metadata.external_provider_calls)
        self.assertFalse(contract.estimated_latency_metadata.network_required)
        self.assertFalse(contract.memory_access.reads_runtime_memory)
        self.assertFalse(contract.memory_access.writes_runtime_memory)
        self.assertIn(
            "does not change Creative Critic scoring",
            contract.authority_boundary,
        )
        self.assertIn(
            "creative_critic_scoring_change",
            contract.prohibited_actions,
        )
        self.assertIn("critique_loop_execution", contract.prohibited_actions)
        self.assertIn("output_change", contract.prohibited_actions)

    def test_aesthetic_critic_does_not_declare_scoring_or_loops(self) -> None:
        contract = agent_contract_by_id("aesthetic_critic_agent")

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
            "run_critique_loop",
            "change_quality_score",
            "execute_provider",
            "runtime_auto_selection",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
