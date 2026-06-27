import unittest

from creative_coding_assistant.orchestration import (
    agent_contract_by_id,
    agent_contract_registry,
    agent_identity_by_id,
    agent_memory_contract_by_agent_id,
)


class ResearchAgentContractTests(unittest.TestCase):
    def test_research_agent_contract_is_registered(self) -> None:
        registry = agent_contract_registry()
        contract = agent_contract_by_id("research_agent")
        identity = agent_identity_by_id("research_agent")
        memory_contract = agent_memory_contract_by_agent_id("research_agent")

        self.assertIsNotNone(contract)
        self.assertIsNotNone(identity)
        self.assertIsNotNone(memory_contract)
        assert contract is not None
        assert identity is not None
        assert memory_contract is not None
        self.assertIn("research_agent", registry.agent_ids)
        self.assertEqual(contract.agent_id, identity.agent_id)
        self.assertEqual(contract.agent_name, identity.agent_name)
        self.assertEqual(contract.agent_version, identity.identity_version)
        self.assertEqual(contract.role_id, "research")
        self.assertEqual(contract.role_name, "Research Agent")
        self.assertEqual(
            memory_contract.readable_memory_surfaces,
            ("session", "provenance"),
        )
        self.assertEqual(
            contract.memory_access.allowed_memory_sources,
            (
                "session_metadata",
                "provenance_metadata",
                "retrieval_context_metadata",
                "future_blackboard_contract",
            ),
        )

    def test_research_agent_maps_existing_source_context_boundaries(self) -> None:
        contract = agent_contract_by_id("research_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(
            contract.required_inputs,
            (
                "assistant_request",
                "retrieval_context",
                "source_metadata",
            ),
        )
        self.assertIn("assembled_context", contract.optional_inputs)
        self.assertIn("retrieval_quality_metadata", contract.optional_inputs)
        self.assertIn("kb_source_health_metadata", contract.optional_inputs)
        self.assertIn("source_context_metadata", contract.produced_metadata)
        self.assertIn("retrieval_gap_metadata", contract.produced_metadata)
        self.assertIn("source_reliability_metadata", contract.produced_metadata)
        self.assertIn("source_coverage", contract.produced_signals)
        self.assertIn("retrieval_confidence", contract.produced_signals)
        self.assertIn("missing_source_context", contract.produced_signals)
        self.assertIn(
            "research_context_packet_contract",
            contract.produced_outputs,
        )

    def test_research_agent_contract_is_metadata_only(self) -> None:
        contract = agent_contract_by_id("research_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertTrue(contract.metadata_only)
        self.assertEqual(contract.cacheability, "deterministic_with_upstream_metadata")
        self.assertFalse(contract.estimated_cost_metadata.external_provider_calls)
        self.assertEqual(contract.estimated_cost_metadata.relative_cost, "low")
        self.assertFalse(contract.estimated_latency_metadata.network_required)
        self.assertEqual(contract.estimated_latency_metadata.relative_latency, "low")
        self.assertEqual(
            contract.estimated_latency_metadata.blocking_inputs,
            contract.required_inputs,
        )
        self.assertFalse(contract.memory_access.reads_runtime_memory)
        self.assertFalse(contract.memory_access.writes_runtime_memory)
        self.assertIn(
            "does not add web research execution",
            contract.authority_boundary,
        )
        self.assertIn("web_research_execution", contract.prohibited_actions)
        self.assertIn("external_source_calling", contract.prohibited_actions)
        self.assertIn("retrieval_behavior_change", contract.prohibited_actions)

    def test_research_contract_does_not_declare_external_calls(self) -> None:
        contract = agent_contract_by_id("research_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        combined_text = " ".join(
            (
                contract.authority_boundary,
                *contract.allowed_actions,
                *contract.prohibited_actions,
                *contract.capabilities,
                *contract.future_orchestration_hooks,
                *contract.source_contract_registries,
                *contract.blocked_runtime_behaviors,
            )
        )

        for forbidden_term in (
            "perform_web_search",
            "http_request",
            "external_source_fetch",
            "retrieval_query_execution",
            "execute_provider",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
