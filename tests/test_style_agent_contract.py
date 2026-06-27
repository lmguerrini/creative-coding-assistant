import unittest

from creative_coding_assistant.orchestration import (
    agent_contract_by_id,
    agent_contract_registry,
    agent_identity_by_id,
    agent_memory_contract_by_agent_id,
)


class StyleAgentContractTests(unittest.TestCase):
    def test_style_agent_contract_is_registered(self) -> None:
        registry = agent_contract_registry()
        contract = agent_contract_by_id("style_agent")
        identity = agent_identity_by_id("style_agent")
        memory_contract = agent_memory_contract_by_agent_id("style_agent")

        self.assertIsNotNone(contract)
        self.assertIsNotNone(identity)
        self.assertIsNotNone(memory_contract)
        assert contract is not None
        assert identity is not None
        assert memory_contract is not None
        self.assertIn("style_agent", registry.agent_ids)
        self.assertEqual(contract.agent_id, identity.agent_id)
        self.assertEqual(contract.agent_name, identity.agent_name)
        self.assertEqual(contract.agent_version, identity.identity_version)
        self.assertEqual(contract.role_id, "style")
        self.assertEqual(contract.role_name, "Style Agent")
        self.assertEqual(
            memory_contract.readable_memory_surfaces,
            ("session", "artifact", "provenance"),
        )

    def test_style_agent_maps_existing_creative_style_metadata(self) -> None:
        contract = agent_contract_by_id("style_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(
            contract.required_inputs,
            (
                "assistant_request",
                "creative_translation",
                "visual_style_guidance",
            ),
        )
        self.assertIn("creative_composition", contract.optional_inputs)
        self.assertIn("semantic_motif", contract.optional_inputs)
        self.assertIn("emotional_consistency", contract.optional_inputs)
        self.assertIn("reference_fusion", contract.optional_inputs)
        self.assertIn("visual_style_metadata", contract.produced_metadata)
        self.assertIn("style_coherence_metadata", contract.produced_metadata)
        self.assertIn("style_alignment", contract.produced_signals)
        self.assertIn("visual_coherence", contract.produced_signals)
        self.assertIn(
            "style_context_packet_contract",
            contract.produced_outputs,
        )

    def test_style_agent_contract_is_metadata_only(self) -> None:
        contract = agent_contract_by_id("style_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertTrue(contract.metadata_only)
        self.assertEqual(contract.cacheability, "deterministic_with_upstream_metadata")
        self.assertFalse(contract.estimated_cost_metadata.external_provider_calls)
        self.assertEqual(contract.estimated_cost_metadata.relative_cost, "low")
        self.assertFalse(contract.estimated_latency_metadata.network_required)
        self.assertEqual(contract.estimated_latency_metadata.relative_latency, "low")
        self.assertFalse(contract.memory_access.reads_runtime_memory)
        self.assertFalse(contract.memory_access.writes_runtime_memory)
        self.assertIn(
            "does not change style output generation",
            contract.authority_boundary,
        )
        self.assertIn(
            "style_output_generation_change",
            contract.prohibited_actions,
        )
        self.assertIn(
            "style_engine_behavior_change",
            contract.prohibited_actions,
        )

    def test_style_contract_does_not_declare_generation_behavior(self) -> None:
        contract = agent_contract_by_id("style_agent")

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
            "render_style_output",
            "style_engine_execute",
            "execute_provider",
            "runtime_auto_selection",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
