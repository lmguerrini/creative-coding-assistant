import unittest

from creative_coding_assistant.orchestration import (
    AgentContract,
    AgentContractCostMetadata,
    AgentContractLatencyMetadata,
    AgentContractRegistry,
    AgentMemoryAccessContract,
    agent_contract_by_id,
    agent_contract_registry,
    build_agent_contract_registry,
)

REQUIRED_CONTRACT_FIELDS = {
    "agent_id",
    "agent_name",
    "agent_version",
    "agent_category",
    "contract_stage",
    "role_id",
    "role_name",
    "role_purpose",
    "authority_boundary",
    "allowed_actions",
    "prohibited_actions",
    "capabilities",
    "required_inputs",
    "optional_inputs",
    "produced_outputs",
    "produced_metadata",
    "produced_signals",
    "memory_access",
    "cacheability",
    "estimated_cost_metadata",
    "estimated_latency_metadata",
    "future_orchestration_hooks",
    "source_contract_registries",
    "blocked_runtime_behaviors",
    "serialization_version",
    "metadata_only",
}


def _sample_agent_contract() -> AgentContract:
    return AgentContract(
        agent_id="planner_agent",
        agent_name="Planner Agent",
        agent_version="v4.1",
        role_id="planner",
        role_name="Planner",
        role_purpose=(
            "Describe passive planning context requirements for future V4 "
            "orchestration."
        ),
        authority_boundary=(
            "Planner agent contract metadata describes identity, role, inputs, "
            "outputs, memory posture, and future hooks only; it does not create "
            "agents, route work, call providers, execute artifacts, or modify "
            "generated output."
        ),
        allowed_actions=(
            "describe_planning_metadata",
            "declare_required_context",
        ),
        prohibited_actions=(
            "agent_invocation",
            "provider_or_model_routing",
            "runtime_selection",
            "memory_write",
        ),
        capabilities=(
            "planning_context_contract",
            "gap_signal_contract",
        ),
        required_inputs=("assistant_request", "route_decision"),
        optional_inputs=("artifact_engine_contracts",),
        produced_outputs=("planner_context_packet_contract",),
        produced_metadata=(
            "planning_scope_metadata",
            "planning_gap_metadata",
        ),
        produced_signals=(
            "missing_information",
            "planning_confidence",
        ),
        memory_access=AgentMemoryAccessContract(
            allowed_memory_sources=(
                "conversation_summary_metadata",
                "project_memory_metadata",
            ),
        ),
        estimated_cost_metadata=AgentContractCostMetadata(
            relative_cost="none",
            cost_basis="Static contract metadata; no provider calls.",
            cache_sensitivity=(
                "Stable until passive role contract metadata changes."
            ),
        ),
        estimated_latency_metadata=AgentContractLatencyMetadata(
            relative_latency="none",
            latency_basis="Static in-process metadata access; no network.",
            blocking_inputs=("assistant_request", "route_decision"),
        ),
        future_orchestration_hooks=("v4_2_planner_orchestration_candidate",),
        source_contract_registries=("artifact_engine_contract_registry",),
    )


class AgentContractFoundationTests(unittest.TestCase):
    def test_contract_model_exposes_metadata_only_agent_surface(self) -> None:
        contract = _sample_agent_contract()
        dumped = contract.model_dump(mode="json")

        self.assertEqual(set(dumped), REQUIRED_CONTRACT_FIELDS)
        self.assertEqual(contract.agent_category, "multi_agent_core")
        self.assertEqual(contract.contract_stage, "v4_1_contract_foundation")
        self.assertEqual(contract.serialization_version, "agent_contract.v1")
        self.assertTrue(contract.metadata_only)
        self.assertIn("does not create agents", contract.authority_boundary)
        self.assertIn("planning_context_contract", contract.capabilities)
        self.assertIn("assistant_request", contract.required_inputs)
        self.assertIn("planning_gap_metadata", contract.produced_metadata)
        self.assertIn("planning_confidence", contract.produced_signals)
        self.assertIn(
            "v4_2_planner_orchestration_candidate",
            contract.future_orchestration_hooks,
        )

    def test_memory_and_performance_hints_are_passive(self) -> None:
        contract = _sample_agent_contract()

        self.assertEqual(contract.memory_access.access_mode, "metadata_reference_only")
        self.assertFalse(contract.memory_access.reads_runtime_memory)
        self.assertFalse(contract.memory_access.writes_runtime_memory)
        self.assertFalse(contract.memory_access.creates_memory_store)
        self.assertTrue(contract.memory_access.metadata_only)
        self.assertIn(
            "memory_write",
            contract.memory_access.forbidden_memory_operations,
        )
        self.assertFalse(contract.estimated_cost_metadata.external_provider_calls)
        self.assertEqual(contract.estimated_cost_metadata.relative_cost, "none")
        self.assertFalse(contract.estimated_latency_metadata.network_required)
        self.assertEqual(contract.estimated_latency_metadata.relative_latency, "none")
        self.assertEqual(
            contract.estimated_latency_metadata.blocking_inputs,
            contract.required_inputs,
        )

    def test_static_registry_starts_empty_for_later_role_tasks(self) -> None:
        registry = agent_contract_registry()

        self.assertEqual(registry.role, "agent_contract_registry")
        self.assertEqual(
            registry.serialization_version,
            "agent_contract_registry.v1",
        )
        self.assertEqual(registry.agent_ids, ())
        self.assertEqual(registry.contracts, ())
        self.assertEqual(registry.contract_count, 0)
        self.assertTrue(registry.metadata_only)
        self.assertIn("do not create agents", registry.authority_boundary)
        self.assertIsNone(agent_contract_by_id("planner_agent"))

    def test_registry_builder_and_lookup_are_stable(self) -> None:
        contract = _sample_agent_contract()
        registry = build_agent_contract_registry((contract,))
        missing_contract = agent_contract_by_id("missing", registry)
        found_contract = agent_contract_by_id("planner_agent", registry)
        dumped = registry.model_dump(mode="json")

        self.assertIsNone(missing_contract)
        self.assertEqual(found_contract, contract)
        self.assertEqual(registry.agent_ids, ("planner_agent",))
        self.assertEqual(registry.contract_count, 1)
        self.assertEqual(dumped["agent_ids"], ["planner_agent"])
        self.assertEqual(dumped["contracts"][0]["agent_id"], "planner_agent")
        self.assertTrue(dumped["metadata_only"])

    def test_registry_rejects_mismatched_or_duplicate_ids(self) -> None:
        contract = _sample_agent_contract()
        duplicate_contract = contract.model_copy(
            update={"agent_name": "Duplicate Planner Agent"}
        )

        with self.assertRaisesRegex(ValueError, "agent_ids must be unique"):
            build_agent_contract_registry((contract, duplicate_contract))

        with self.assertRaisesRegex(ValueError, "agent_ids must match contracts"):
            AgentContractRegistry(
                contracts=(contract,),
                agent_ids=("other_agent",),
                contract_count=1,
            )

    def test_contract_surface_does_not_declare_active_runtime_behaviors(
        self,
    ) -> None:
        contract = _sample_agent_contract()
        registry = build_agent_contract_registry((contract,))
        combined_text = " ".join(
            (
                registry.authority_boundary,
                contract.authority_boundary,
                *contract.allowed_actions,
                *contract.prohibited_actions,
                *contract.capabilities,
                *contract.produced_outputs,
                *contract.future_orchestration_hooks,
                *contract.blocked_runtime_behaviors,
            )
        )

        for forbidden_term in (
            "execute_provider",
            "autonomous_retry",
            "runtime_auto_selection",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
