import unittest

from creative_coding_assistant.orchestration import (
    AgentMemoryContractRegistry,
    agent_identity_registry,
    agent_memory_contract_by_agent_id,
    agent_memory_contract_registry,
)
from creative_coding_assistant.orchestration.agent_memory_contracts import (
    AGENT_MEMORY_SURFACE_ORDER,
)

REQUIRED_SURFACE_FIELDS = {
    "surface",
    "read_access",
    "write_access",
    "reference_access",
    "readable_metadata",
    "writable_metadata",
    "referenceable_metadata",
    "authority_boundary",
    "storage_implemented",
    "retrieval_side_effects",
    "blackboard_implemented",
    "serialization_version",
    "metadata_only",
}

REQUIRED_MEMORY_CONTRACT_FIELDS = {
    "agent_id",
    "memory_contract_id",
    "surfaces",
    "readable_memory_surfaces",
    "writable_memory_surfaces",
    "referenceable_memory_surfaces",
    "future_blackboard_hooks",
    "future_shared_context_hooks",
    "blocked_runtime_behaviors",
    "persistence_implemented",
    "blackboard_implemented",
    "retrieval_side_effects",
    "serialization_version",
    "metadata_only",
}


class AgentMemoryContractTests(unittest.TestCase):
    def test_registry_covers_identity_registry_agents(self) -> None:
        identity_registry = agent_identity_registry()
        memory_registry = agent_memory_contract_registry()

        self.assertEqual(memory_registry.role, "agent_memory_contract_registry")
        self.assertEqual(
            memory_registry.serialization_version,
            "agent_memory_contract_registry.v1",
        )
        self.assertEqual(memory_registry.agent_ids, identity_registry.agent_ids)
        self.assertEqual(memory_registry.contract_count, 12)
        self.assertEqual(memory_registry.memory_surfaces, AGENT_MEMORY_SURFACE_ORDER)
        self.assertEqual(
            memory_registry.source_identity_registry,
            "agent_identity_registry",
        )
        self.assertTrue(memory_registry.metadata_only)
        self.assertFalse(memory_registry.persistence_implemented)
        self.assertFalse(memory_registry.blackboard_implemented)
        self.assertFalse(memory_registry.retrieval_side_effects)
        self.assertIn(
            "do not implement persistence",
            memory_registry.authority_boundary,
        )

    def test_contracts_expose_read_write_reference_boundaries(self) -> None:
        registry = agent_memory_contract_registry()

        for contract in registry.contracts:
            dumped = contract.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_MEMORY_CONTRACT_FIELDS)
            self.assertEqual(
                tuple(surface.surface for surface in contract.surfaces),
                AGENT_MEMORY_SURFACE_ORDER,
            )
            self.assertEqual(
                contract.writable_memory_surfaces,
                ("future_blackboard",),
            )
            self.assertEqual(
                contract.referenceable_memory_surfaces,
                AGENT_MEMORY_SURFACE_ORDER,
            )
            self.assertTrue(contract.future_blackboard_hooks)
            self.assertTrue(contract.future_shared_context_hooks)
            self.assertFalse(contract.persistence_implemented)
            self.assertFalse(contract.blackboard_implemented)
            self.assertFalse(contract.retrieval_side_effects)
            self.assertEqual(
                contract.serialization_version,
                "agent_memory_contract.v1",
            )
            self.assertTrue(contract.metadata_only)

            for surface in contract.surfaces:
                surface_dumped = surface.model_dump(mode="json")
                self.assertEqual(set(surface_dumped), REQUIRED_SURFACE_FIELDS)
                self.assertTrue(surface.referenceable_metadata)
                self.assertFalse(surface.storage_implemented)
                self.assertFalse(surface.retrieval_side_effects)
                self.assertFalse(surface.blackboard_implemented)
                self.assertTrue(surface.metadata_only)

    def test_lookup_exposes_agent_specific_memory_contracts(self) -> None:
        planner = agent_memory_contract_by_agent_id("planner_agent")
        research = agent_memory_contract_by_agent_id("research_agent")
        missing = agent_memory_contract_by_agent_id("missing")

        self.assertIsNone(missing)
        self.assertIsNotNone(planner)
        self.assertIsNotNone(research)
        assert planner is not None
        assert research is not None
        self.assertEqual(
            planner.readable_memory_surfaces,
            ("session", "artifact", "evaluation", "provenance"),
        )
        self.assertEqual(research.readable_memory_surfaces, ("session", "provenance"))
        self.assertEqual(planner.writable_memory_surfaces, ("future_blackboard",))
        self.assertIn(
            "planning_context_packet",
            planner.surfaces[-1].writable_metadata,
        )
        self.assertIn(
            "source_gap_summary",
            research.surfaces[-1].writable_metadata,
        )

    def test_registry_rejects_mismatched_or_duplicate_contracts(self) -> None:
        registry = agent_memory_contract_registry()
        first_contract = registry.contracts[0]
        duplicate_contract = first_contract.model_copy(
            update={"memory_contract_id": "duplicate_memory_contract"}
        )

        with self.assertRaisesRegex(ValueError, "agent_ids must be unique"):
            AgentMemoryContractRegistry(
                contracts=(first_contract, duplicate_contract) + registry.contracts[2:],
                agent_ids=registry.agent_ids,
                contract_count=12,
            )

        with self.assertRaisesRegex(ValueError, "agent_ids must match contracts"):
            AgentMemoryContractRegistry(
                contracts=registry.contracts,
                agent_ids=("other_agent",) + registry.agent_ids[1:],
                contract_count=12,
            )

    def test_memory_contracts_do_not_implement_storage_or_retrieval(self) -> None:
        registry = agent_memory_contract_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for contract in registry.contracts
                    for field in (
                        contract.memory_contract_id,
                        *contract.future_blackboard_hooks,
                        *contract.future_shared_context_hooks,
                        *contract.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "persist_memory_record",
            "blackboard_write",
            "retrieval_query_execution",
            "storage_adapter",
            "execute_provider",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
