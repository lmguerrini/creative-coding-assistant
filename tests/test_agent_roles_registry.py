import unittest

from creative_coding_assistant.orchestration import (
    AgentRoleRegistry,
    agent_contract_registry,
    agent_identity_registry,
    agent_role_by_id,
    agent_role_registry,
    agent_roles_by_capability_family,
    agent_roles_by_family,
)

EXPECTED_ROLE_IDS = (
    "planner",
    "research",
    "style",
    "runtime",
    "artifact",
    "art_direction",
    "aesthetic_critic",
    "narrative_symbolic",
    "creative_curator",
    "critic",
    "refiner",
    "final_synthesizer",
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

EXPECTED_ROLE_FAMILIES = (
    "planning",
    "research",
    "style",
    "runtime",
    "artifact",
    "art_direction",
    "critique",
    "narrative",
    "curation",
    "refinement",
    "synthesis",
)

EXPECTED_CAPABILITY_FAMILIES = (
    "planning_strategy",
    "source_context",
    "visual_style",
    "runtime_compatibility",
    "artifact_implementation",
    "creative_direction",
    "aesthetic_evaluation",
    "symbolic_narrative",
    "creative_selection",
    "quality_review",
    "refinement_planning",
    "final_response_synthesis",
)

REQUIRED_ROLE_FIELDS = {
    "role_id",
    "role_name",
    "agent_id",
    "agent_name",
    "role_order",
    "role_family",
    "capability_family",
    "purpose",
    "contract_serialization_version",
    "source_contract_registries",
    "produced_outputs",
    "future_orchestration_hooks",
    "blocked_runtime_behaviors",
    "serialization_version",
    "metadata_only",
}


class AgentRolesRegistryTests(unittest.TestCase):
    def test_role_registry_covers_all_agent_contracts_and_identities(self) -> None:
        role_registry = agent_role_registry()
        identity_registry = agent_identity_registry()
        contract_registry = agent_contract_registry()

        self.assertEqual(role_registry.role, "agent_role_registry")
        self.assertEqual(
            role_registry.serialization_version,
            "agent_role_registry.v1",
        )
        self.assertEqual(role_registry.role_ids, EXPECTED_ROLE_IDS)
        self.assertEqual(role_registry.agent_ids, EXPECTED_AGENT_IDS)
        self.assertEqual(role_registry.role_count, 12)
        self.assertEqual(role_registry.role_families, EXPECTED_ROLE_FAMILIES)
        self.assertEqual(
            role_registry.capability_families,
            EXPECTED_CAPABILITY_FAMILIES,
        )
        self.assertEqual(role_registry.agent_ids, identity_registry.agent_ids)
        self.assertEqual(role_registry.agent_ids, contract_registry.agent_ids)
        self.assertTrue(role_registry.metadata_only)
        self.assertIn("does not execute agents", role_registry.authority_boundary)

    def test_role_metadata_serializes_for_inspection(self) -> None:
        role_registry = agent_role_registry()

        for index, role in enumerate(role_registry.roles, start=1):
            dumped = role.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_ROLE_FIELDS)
            self.assertEqual(role.role_order, index)
            self.assertEqual(role.serialization_version, "agent_role.v1")
            self.assertEqual(
                role.contract_serialization_version,
                "agent_contract.v1",
            )
            self.assertTrue(role.source_contract_registries)
            self.assertTrue(role.produced_outputs)
            self.assertTrue(role.future_orchestration_hooks)
            self.assertTrue(role.metadata_only)

    def test_lookup_and_grouping_helpers_are_stable(self) -> None:
        planner = agent_role_by_id("planner")
        missing = agent_role_by_id("missing")
        critique_roles = agent_roles_by_family("critique")
        visual_style_roles = agent_roles_by_capability_family("visual_style")

        self.assertIsNone(missing)
        self.assertIsNotNone(planner)
        assert planner is not None
        self.assertEqual(planner.agent_id, "planner_agent")
        self.assertEqual(planner.role_order, 1)
        self.assertEqual(
            tuple(role.role_id for role in critique_roles),
            ("aesthetic_critic", "critic"),
        )
        self.assertEqual(
            tuple(role.agent_id for role in visual_style_roles),
            ("style_agent",),
        )
        self.assertEqual(agent_roles_by_family("missing"), ())
        self.assertEqual(agent_roles_by_capability_family("missing"), ())

    def test_registry_rejects_mismatched_or_duplicate_roles(self) -> None:
        registry = agent_role_registry()
        first_role = registry.roles[0]
        duplicate_role = first_role.model_copy(
            update={"agent_name": "Duplicate Planner Agent"}
        )

        with self.assertRaisesRegex(ValueError, "role_ids must be unique"):
            AgentRoleRegistry(
                roles=(first_role, duplicate_role) + registry.roles[2:],
                role_ids=registry.role_ids,
                agent_ids=registry.agent_ids,
                role_count=12,
                role_families=registry.role_families,
                capability_families=registry.capability_families,
            )

        with self.assertRaisesRegex(ValueError, "role_ids must match roles"):
            AgentRoleRegistry(
                roles=registry.roles,
                role_ids=("other_role",) + registry.role_ids[1:],
                agent_ids=registry.agent_ids,
                role_count=12,
                role_families=registry.role_families,
                capability_families=registry.capability_families,
            )

    def test_role_registry_does_not_declare_orchestration(self) -> None:
        registry = agent_role_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for role in registry.roles
                    for field in (
                        role.role_id,
                        role.agent_id,
                        *role.future_orchestration_hooks,
                        *role.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "route_task_to_agent",
            "execute_agent",
            "agent_debate",
            "consensus_vote",
            "runtime_auto_selection",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
