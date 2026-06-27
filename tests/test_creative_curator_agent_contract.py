import unittest

from creative_coding_assistant.orchestration import (
    agent_contract_by_id,
    agent_contract_registry,
    agent_identity_by_id,
    agent_memory_contract_by_agent_id,
)


class CreativeCuratorAgentContractTests(unittest.TestCase):
    def test_creative_curator_agent_contract_is_registered(self) -> None:
        registry = agent_contract_registry()
        contract = agent_contract_by_id("creative_curator_agent")
        identity = agent_identity_by_id("creative_curator_agent")
        memory_contract = agent_memory_contract_by_agent_id("creative_curator_agent")

        self.assertIsNotNone(contract)
        self.assertIsNotNone(identity)
        self.assertIsNotNone(memory_contract)
        assert contract is not None
        assert identity is not None
        assert memory_contract is not None
        self.assertIn("creative_curator_agent", registry.agent_ids)
        self.assertEqual(contract.agent_id, identity.agent_id)
        self.assertEqual(contract.agent_name, identity.agent_name)
        self.assertEqual(contract.agent_version, identity.identity_version)
        self.assertEqual(contract.role_id, "creative_curator")
        self.assertEqual(contract.role_name, "Creative Curator Agent")
        self.assertEqual(
            memory_contract.readable_memory_surfaces,
            ("session", "artifact", "evaluation", "provenance"),
        )

    def test_creative_curator_maps_selection_metadata_boundaries(self) -> None:
        contract = agent_contract_by_id("creative_curator_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(
            contract.required_inputs,
            (
                "assistant_request",
                "multi_artifact_strategy",
                "artifact_intelligence_synthesis",
            ),
        )
        self.assertIn("creative_score_profile", contract.optional_inputs)
        self.assertIn("creative_quality_evaluation", contract.optional_inputs)
        self.assertIn("artifact_merge_planner", contract.optional_inputs)
        self.assertIn("creative_selection_metadata", contract.produced_metadata)
        self.assertIn("curation_preference_metadata", contract.produced_metadata)
        self.assertIn("selection_rationale_metadata", contract.produced_metadata)
        self.assertIn("curation_confidence", contract.produced_signals)
        self.assertIn("candidate_strength", contract.produced_signals)
        self.assertIn("selection_ambiguity", contract.produced_signals)
        self.assertIn(
            "curation_context_packet_contract",
            contract.produced_outputs,
        )

    def test_creative_curator_contract_is_metadata_only(self) -> None:
        contract = agent_contract_by_id("creative_curator_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertTrue(contract.metadata_only)
        self.assertEqual(contract.cacheability, "deterministic_with_upstream_metadata")
        self.assertFalse(contract.estimated_cost_metadata.external_provider_calls)
        self.assertFalse(contract.estimated_latency_metadata.network_required)
        self.assertFalse(contract.memory_access.reads_runtime_memory)
        self.assertFalse(contract.memory_access.writes_runtime_memory)
        self.assertIn(
            "does not select final outputs autonomously",
            contract.authority_boundary,
        )
        self.assertIn(
            "autonomous_final_output_selection",
            contract.prohibited_actions,
        )
        self.assertIn("final_synthesis_alteration", contract.prohibited_actions)

    def test_creative_curator_does_not_declare_final_selection(self) -> None:
        contract = agent_contract_by_id("creative_curator_agent")

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
            "choose_final_output",
            "final_synthesis_execute",
            "execute_provider",
            "runtime_auto_selection",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
