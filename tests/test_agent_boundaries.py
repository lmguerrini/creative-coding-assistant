import unittest

from creative_coding_assistant.orchestration import (
    AgentBoundaryRegistry,
    agent_boundary_by_agent_id,
    agent_boundary_registry,
    agent_contract_registry,
)

EXPECTED_AGENT_IDS = (
    "planner_agent",
    "research_agent",
    "style_agent",
    "runtime_agent",
    "artifact_agent",
    "art_direction_agent",
    "aesthetic_critic_agent",
    "narrative_symbolic_agent",
    "creative_curator_agent",
    "critic_agent",
    "refiner_agent",
    "final_synthesizer_agent",
)

REQUIRED_BOUNDARY_FIELDS = {
    "agent_id",
    "role_id",
    "authority_boundary",
    "boundary_rationale",
    "allowed_inputs",
    "allowed_outputs",
    "forbidden_behaviors",
    "passive_contract_refs",
    "enforcement_runtime_implemented",
    "workflow_behavior_changed",
    "autonomous_escalation_added",
    "serialization_version",
    "metadata_only",
}


class AgentBoundaryTests(unittest.TestCase):
    def test_boundary_registry_covers_all_agent_contracts(self) -> None:
        boundary_registry = agent_boundary_registry()
        contract_registry = agent_contract_registry()

        self.assertEqual(boundary_registry.role, "agent_boundary_registry")
        self.assertEqual(
            boundary_registry.serialization_version,
            "agent_boundary_registry.v1",
        )
        self.assertEqual(boundary_registry.agent_ids, EXPECTED_AGENT_IDS)
        self.assertEqual(boundary_registry.agent_ids, contract_registry.agent_ids)
        self.assertEqual(boundary_registry.boundary_count, 12)
        self.assertEqual(
            boundary_registry.source_contract_registry,
            "agent_contract_registry",
        )
        self.assertTrue(boundary_registry.metadata_only)
        self.assertFalse(boundary_registry.enforcement_runtime_implemented)
        self.assertFalse(boundary_registry.workflow_behavior_changed)
        self.assertFalse(boundary_registry.autonomous_escalation_added)
        self.assertIn(
            "does not implement enforcement runtime",
            boundary_registry.authority_boundary,
        )

    def test_boundaries_expose_inputs_outputs_and_rationale(self) -> None:
        registry = agent_boundary_registry()

        for boundary in registry.boundaries:
            dumped = boundary.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_BOUNDARY_FIELDS)
            self.assertTrue(boundary.authority_boundary)
            self.assertTrue(boundary.boundary_rationale)
            self.assertTrue(boundary.allowed_inputs)
            self.assertTrue(boundary.allowed_outputs)
            self.assertTrue(boundary.forbidden_behaviors)
            self.assertIn("agent_contract.v1", boundary.passive_contract_refs)
            self.assertIn("agent_identity.v1", boundary.passive_contract_refs)
            self.assertIn("agent_memory_contract.v1", boundary.passive_contract_refs)
            self.assertEqual(boundary.serialization_version, "agent_boundary.v1")
            self.assertTrue(boundary.metadata_only)
            self.assertFalse(boundary.enforcement_runtime_implemented)
            self.assertFalse(boundary.workflow_behavior_changed)
            self.assertFalse(boundary.autonomous_escalation_added)

    def test_boundary_lookup_is_stable(self) -> None:
        runtime_boundary = agent_boundary_by_agent_id("runtime_agent")
        missing_boundary = agent_boundary_by_agent_id("missing")

        self.assertIsNone(missing_boundary)
        self.assertIsNotNone(runtime_boundary)
        assert runtime_boundary is not None
        self.assertEqual(runtime_boundary.role_id, "runtime")
        self.assertIn("runtime_capabilities", runtime_boundary.allowed_inputs)
        self.assertIn("runtime_selection_change", runtime_boundary.forbidden_behaviors)
        self.assertIn("without selecting", runtime_boundary.boundary_rationale)

    def test_boundaries_reject_mismatched_or_duplicate_agents(self) -> None:
        registry = agent_boundary_registry()
        first_boundary = registry.boundaries[0]
        duplicate_boundary = first_boundary.model_copy(
            update={"role_id": "duplicate_planner"}
        )

        with self.assertRaisesRegex(ValueError, "agent_ids must be unique"):
            AgentBoundaryRegistry(
                boundaries=(first_boundary, duplicate_boundary)
                + registry.boundaries[2:],
                agent_ids=registry.agent_ids,
                boundary_count=12,
            )

        with self.assertRaisesRegex(ValueError, "agent_ids must match boundaries"):
            AgentBoundaryRegistry(
                boundaries=registry.boundaries,
                agent_ids=("other_agent",) + registry.agent_ids[1:],
                boundary_count=12,
            )

    def test_allowed_boundary_surface_does_not_leak_execution_language(self) -> None:
        registry = agent_boundary_registry()

        for boundary in registry.boundaries:
            allowed_surface = " ".join(
                (
                    *boundary.allowed_inputs,
                    *boundary.allowed_outputs,
                    boundary.boundary_rationale,
                )
            )
            for forbidden_term in (
                "execute_agent",
                "route_task",
                "dynamic_orchestration",
                "autonomous_escalation",
                "runtime_auto_selection",
            ):
                self.assertNotIn(forbidden_term, allowed_surface)


if __name__ == "__main__":
    unittest.main()
