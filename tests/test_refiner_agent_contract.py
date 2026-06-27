import unittest

from creative_coding_assistant.orchestration import (
    agent_contract_by_id,
    agent_contract_registry,
    agent_identity_by_id,
    agent_memory_contract_by_agent_id,
)


class RefinerAgentContractTests(unittest.TestCase):
    def test_refiner_agent_contract_is_registered(self) -> None:
        registry = agent_contract_registry()
        contract = agent_contract_by_id("refiner_agent")
        identity = agent_identity_by_id("refiner_agent")
        memory_contract = agent_memory_contract_by_agent_id("refiner_agent")

        self.assertIsNotNone(contract)
        self.assertIsNotNone(identity)
        self.assertIsNotNone(memory_contract)
        assert contract is not None
        assert identity is not None
        assert memory_contract is not None
        self.assertIn("refiner_agent", registry.agent_ids)
        self.assertEqual(contract.agent_id, identity.agent_id)
        self.assertEqual(contract.agent_name, identity.agent_name)
        self.assertEqual(contract.agent_version, identity.identity_version)
        self.assertEqual(contract.role_id, "refiner")
        self.assertEqual(contract.role_name, "Refiner Agent")
        self.assertEqual(
            memory_contract.readable_memory_surfaces,
            ("session", "artifact", "evaluation", "provenance"),
        )

    def test_refiner_maps_improvement_metadata_boundaries(self) -> None:
        contract = agent_contract_by_id("refiner_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(
            contract.required_inputs,
            (
                "assistant_request",
                "artifact_refiner_profile",
                "creative_improvement_planner",
            ),
        )
        self.assertIn("refinement_pass_records", contract.optional_inputs)
        self.assertIn("creative_quality_evaluation", contract.optional_inputs)
        self.assertIn("critic_context", contract.optional_inputs)
        self.assertIn("refinement_scope_metadata", contract.produced_metadata)
        self.assertIn("revision_candidate_metadata", contract.produced_metadata)
        self.assertIn("refinement_guardrail_metadata", contract.produced_metadata)
        self.assertIn("refinement_confidence", contract.produced_signals)
        self.assertIn("revision_priority", contract.produced_signals)
        self.assertIn("loop_boundary_status", contract.produced_signals)
        self.assertIn(
            "refinement_context_packet_contract",
            contract.produced_outputs,
        )

    def test_refiner_contract_is_metadata_only(self) -> None:
        contract = agent_contract_by_id("refiner_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertTrue(contract.metadata_only)
        self.assertEqual(contract.cacheability, "deterministic_with_upstream_metadata")
        self.assertFalse(contract.estimated_cost_metadata.external_provider_calls)
        self.assertFalse(contract.estimated_latency_metadata.network_required)
        self.assertFalse(contract.memory_access.reads_runtime_memory)
        self.assertFalse(contract.memory_access.writes_runtime_memory)
        self.assertIn(
            "does not change artifact refinement behavior",
            contract.authority_boundary,
        )
        self.assertIn(
            "artifact_refinement_behavior_change",
            contract.prohibited_actions,
        )
        self.assertIn("refinement_loop_execution", contract.prohibited_actions)
        self.assertIn("generation_alteration", contract.prohibited_actions)

    def test_refiner_does_not_declare_refinement_loops(self) -> None:
        contract = agent_contract_by_id("refiner_agent")

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
            "run_refinement_loop",
            "trigger_retry",
            "execute_provider",
            "runtime_auto_selection",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
