import unittest

from creative_coding_assistant.orchestration import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
    agent_contract_by_id,
    agent_contract_registry,
    agent_identity_by_id,
    agent_memory_contract_by_agent_id,
)


class PlannerAgentContractTests(unittest.TestCase):
    def test_planner_agent_contract_is_registered(self) -> None:
        registry = agent_contract_registry()
        contract = agent_contract_by_id("planner_agent")
        identity = agent_identity_by_id("planner_agent")
        memory_contract = agent_memory_contract_by_agent_id("planner_agent")

        self.assertIsNotNone(contract)
        self.assertIsNotNone(identity)
        self.assertIsNotNone(memory_contract)
        assert contract is not None
        assert identity is not None
        assert memory_contract is not None
        self.assertEqual(registry.agent_ids, ("planner_agent",))
        self.assertEqual(registry.contract_count, 1)
        self.assertEqual(contract.agent_id, identity.agent_id)
        self.assertEqual(contract.agent_name, identity.agent_name)
        self.assertEqual(contract.agent_version, identity.identity_version)
        self.assertEqual(contract.role_id, "planner")
        self.assertEqual(contract.role_name, "Planner Agent")
        self.assertEqual(
            contract.memory_access.allowed_memory_sources,
            (
                "session_metadata",
                "artifact_metadata",
                "evaluation_metadata",
                "provenance_metadata",
                "future_blackboard_contract",
            ),
        )
        self.assertEqual(
            memory_contract.writable_memory_surfaces,
            ("future_blackboard",),
        )

    def test_planner_agent_maps_existing_v3_planning_metadata(self) -> None:
        contract = agent_contract_by_id("planner_agent")

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(
            contract.required_inputs,
            (
                "assistant_request",
                "route_decision",
                "creative_execution_plan",
            ),
        )
        self.assertIn("creative_translation", contract.optional_inputs)
        self.assertIn("creative_intent", contract.optional_inputs)
        self.assertIn("creative_hierarchy", contract.optional_inputs)
        self.assertIn("creative_constraints", contract.optional_inputs)
        self.assertIn("creative_strategy", contract.optional_inputs)
        self.assertIn("creative_technique", contract.optional_inputs)
        self.assertIn("planning_scope_metadata", contract.produced_metadata)
        self.assertIn("planning_gap_metadata", contract.produced_metadata)
        self.assertIn("planning_confidence", contract.produced_signals)
        self.assertIn("missing_information", contract.produced_signals)
        self.assertIn(
            "planner_context_packet_contract",
            contract.produced_outputs,
        )

    def test_planner_agent_contract_is_metadata_only(self) -> None:
        contract = agent_contract_by_id("planner_agent")

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
            "does not execute a planner agent",
            contract.authority_boundary,
        )
        self.assertIn(
            "workflow_node_order_change",
            contract.prohibited_actions,
        )

    def test_planner_agent_does_not_change_workflow_order(self) -> None:
        self.assertEqual(
            ASSISTANT_WORKFLOW_NODE_ORDER,
            (
                "intake",
                "routing",
                "memory",
                "retrieval",
                "context_assembly",
                "prompt_input",
                "planning",
                "director",
                "reasoning",
                "prompt_rendering",
                "generation",
                "artifact_extraction",
                "preview_preparation",
                "artifact_critique",
                "review",
                "refinement",
                "finalization",
                "failure",
            ),
        )

    def test_planner_contract_does_not_declare_active_runtime_behavior(self) -> None:
        contract = agent_contract_by_id("planner_agent")

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
            "execute_provider",
            "autonomous_retry",
            "runtime_auto_selection",
            "workflow_node_insert",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
