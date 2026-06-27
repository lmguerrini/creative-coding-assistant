import unittest

from creative_coding_assistant.orchestration import (
    agent_contract_by_id,
    agent_contract_registry,
    agent_identity_by_id,
    agent_memory_contract_by_agent_id,
)


class ArtDirectionAgentContractTests(unittest.TestCase):
    def test_art_direction_agent_contract_is_registered(self) -> None:
        registry = agent_contract_registry()
        contract = agent_contract_by_id("art_direction_agent")
        identity = agent_identity_by_id("art_direction_agent")
        memory_contract = agent_memory_contract_by_agent_id("art_direction_agent")

        self.assertIsNotNone(contract)
        self.assertIsNotNone(identity)
        self.assertIsNotNone(memory_contract)
        assert contract is not None
        assert identity is not None
        assert memory_contract is not None
        self.assertIn("art_direction_agent", registry.agent_ids)
        self.assertEqual(contract.agent_id, identity.agent_id)
        self.assertEqual(contract.agent_name, identity.agent_name)
        self.assertEqual(contract.agent_version, identity.identity_version)
        self.assertEqual(contract.role_id, "art_direction")
        self.assertEqual(contract.role_name, "Art Direction Agent")
        self.assertEqual(
            memory_contract.readable_memory_surfaces,
            ("session", "artifact", "provenance"),
        )

    def test_art_direction_maps_design_metadata_boundaries(self) -> None:
        contract = agent_contract_by_id("art_direction_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(
            contract.required_inputs,
            (
                "assistant_request",
                "creative_director_brief",
                "creative_composition",
            ),
        )
        self.assertIn("procedural_structure", contract.optional_inputs)
        self.assertIn("generative_structure", contract.optional_inputs)
        self.assertIn("audio_visual_scene", contract.optional_inputs)
        self.assertIn("composition_direction_metadata", contract.produced_metadata)
        self.assertIn("scene_intent_metadata", contract.produced_metadata)
        self.assertIn("visual_coherence_metadata", contract.produced_metadata)
        self.assertIn("direction_alignment", contract.produced_signals)
        self.assertIn("scene_coherence", contract.produced_signals)
        self.assertIn("composition_strength", contract.produced_signals)
        self.assertIn(
            "art_direction_context_packet_contract",
            contract.produced_outputs,
        )

    def test_art_direction_contract_is_metadata_only(self) -> None:
        contract = agent_contract_by_id("art_direction_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertTrue(contract.metadata_only)
        self.assertEqual(contract.cacheability, "deterministic_with_upstream_metadata")
        self.assertFalse(contract.estimated_cost_metadata.external_provider_calls)
        self.assertFalse(contract.estimated_latency_metadata.network_required)
        self.assertFalse(contract.memory_access.reads_runtime_memory)
        self.assertFalse(contract.memory_access.writes_runtime_memory)
        self.assertIn(
            "does not alter generated art direction",
            contract.authority_boundary,
        )
        self.assertIn(
            "generated_art_direction_change",
            contract.prohibited_actions,
        )
        self.assertIn(
            "autonomous_direction_behavior",
            contract.prohibited_actions,
        )

    def test_art_direction_contract_does_not_declare_autonomy(self) -> None:
        contract = agent_contract_by_id("art_direction_agent")

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
            "autonomous_direction_loop",
            "generate_art_direction",
            "execute_provider",
            "runtime_auto_selection",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
