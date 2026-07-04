import unittest

from creative_coding_assistant.orchestration import (
    agent_contract_by_id,
    agent_contract_registry,
    agent_identity_by_id,
    agent_memory_contract_by_agent_id,
)


class NarrativeSymbolicAgentContractTests(unittest.TestCase):
    def test_narrative_symbolic_agent_contract_is_registered(self) -> None:
        registry = agent_contract_registry()
        contract = agent_contract_by_id("narrative_symbolic_agent")
        identity = agent_identity_by_id("narrative_symbolic_agent")
        memory_contract = agent_memory_contract_by_agent_id("narrative_symbolic_agent")

        self.assertIsNotNone(contract)
        self.assertIsNotNone(identity)
        self.assertIsNotNone(memory_contract)
        assert contract is not None
        assert identity is not None
        assert memory_contract is not None
        self.assertIn("narrative_symbolic_agent", registry.agent_ids)
        self.assertEqual(contract.agent_id, identity.agent_id)
        self.assertEqual(contract.agent_name, identity.agent_name)
        self.assertEqual(contract.agent_version, identity.identity_version)
        self.assertEqual(contract.role_id, "narrative_symbolic")
        self.assertEqual(contract.role_name, "Narrative & Symbolic Agent")
        self.assertEqual(
            memory_contract.readable_memory_surfaces,
            ("session", "artifact", "provenance"),
        )

    def test_narrative_symbolic_maps_meaning_layer_boundaries(self) -> None:
        contract = agent_contract_by_id("narrative_symbolic_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(
            contract.required_inputs,
            (
                "assistant_request",
                "symbolic_narrative_plan",
                "semantic_motif",
            ),
        )
        self.assertIn("creative_intent", contract.optional_inputs)
        self.assertIn("emotional_consistency", contract.optional_inputs)
        self.assertIn("art_direction_context", contract.optional_inputs)
        self.assertIn("narrative_arc_metadata", contract.produced_metadata)
        self.assertIn("symbolic_mapping_metadata", contract.produced_metadata)
        self.assertIn("meaning_layer_metadata", contract.produced_metadata)
        self.assertIn("narrative_coherence", contract.produced_signals)
        self.assertIn("symbolic_alignment", contract.produced_signals)
        self.assertIn("conceptual_ambiguity", contract.produced_signals)
        self.assertIn(
            "narrative_context_packet_contract",
            contract.produced_outputs,
        )

    def test_narrative_symbolic_contract_is_metadata_only(self) -> None:
        contract = agent_contract_by_id("narrative_symbolic_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertTrue(contract.metadata_only)
        self.assertEqual(contract.cacheability, "deterministic_with_upstream_metadata")
        self.assertFalse(contract.estimated_cost_metadata.external_provider_calls)
        self.assertFalse(contract.estimated_latency_metadata.network_required)
        self.assertFalse(contract.memory_access.reads_runtime_memory)
        self.assertFalse(contract.memory_access.writes_runtime_memory)
        self.assertIn(
            "does not add symbolic generation behavior",
            contract.authority_boundary,
        )
        self.assertIn("symbolic_generation_behavior", contract.prohibited_actions)
        self.assertIn(
            "prompt_semantics_modification",
            contract.prohibited_actions,
        )

    def test_narrative_symbolic_does_not_declare_prompt_changes(self) -> None:
        contract = agent_contract_by_id("narrative_symbolic_agent")

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
            "rewrite_prompt",
            "generate_symbolic_content",
            "execute_provider",
            "runtime_auto_selection",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
