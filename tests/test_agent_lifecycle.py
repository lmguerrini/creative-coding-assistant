import unittest

from creative_coding_assistant.orchestration import (
    WORKFLOW_STEP_ORDER,
    AgentLifecycleRegistry,
    agent_contract_registry,
    agent_lifecycle_profile_by_agent_id,
    agent_lifecycle_registry,
    agent_lifecycle_transition_by_id,
)

REQUIRED_PROFILE_FIELDS = {
    "agent_id",
    "lifecycle_profile_id",
    "initial_state",
    "allowed_states",
    "terminal_states",
    "transition_ids",
    "source_contract_registry",
    "lifecycle_boundary",
    "blocked_runtime_behaviors",
    "transition_execution_implemented",
    "workflow_state_change_implemented",
    "runtime_lifecycle_engine_implemented",
    "serialization_version",
    "metadata_only",
}

REQUIRED_TRANSITION_FIELDS = {
    "transition_id",
    "from_state",
    "to_state",
    "transition_event",
    "transition_boundary",
    "blocked_runtime_behaviors",
    "transition_execution_implemented",
    "workflow_state_change_implemented",
    "runtime_lifecycle_engine_implemented",
    "serialization_version",
    "metadata_only",
}

EXPECTED_STATES = (
    "planned",
    "active",
    "waiting",
    "completed",
    "skipped",
    "failed",
    "blocked",
    "reviewed",
)


class AgentLifecycleTests(unittest.TestCase):
    def test_lifecycle_registry_covers_agent_contracts(self) -> None:
        lifecycle = agent_lifecycle_registry()
        contracts = agent_contract_registry()

        self.assertEqual(lifecycle.role, "agent_lifecycle_registry")
        self.assertEqual(lifecycle.serialization_version, "agent_lifecycle_registry.v1")
        self.assertEqual(lifecycle.states, EXPECTED_STATES)
        self.assertEqual(lifecycle.agent_ids, contracts.agent_ids)
        self.assertEqual(lifecycle.profile_count, 12)
        self.assertEqual(len(lifecycle.transitions), 10)
        self.assertIn(
            "does not implement a runtime lifecycle engine",
            lifecycle.authority_boundary,
        )
        self.assertFalse(lifecycle.transition_execution_implemented)
        self.assertFalse(lifecycle.workflow_state_change_implemented)
        self.assertFalse(lifecycle.runtime_lifecycle_engine_implemented)
        self.assertTrue(lifecycle.metadata_only)

    def test_profiles_and_transitions_are_passive_metadata(self) -> None:
        lifecycle = agent_lifecycle_registry()

        for profile in lifecycle.profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(profile.allowed_states, lifecycle.states)
            self.assertEqual(profile.initial_state, "planned")
            self.assertIn("reviewed", profile.terminal_states)
            self.assertFalse(profile.transition_execution_implemented)
            self.assertFalse(profile.workflow_state_change_implemented)
            self.assertFalse(profile.runtime_lifecycle_engine_implemented)
            self.assertTrue(profile.metadata_only)

        for transition in lifecycle.transitions:
            dumped = transition.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_TRANSITION_FIELDS)
            self.assertEqual(
                transition.serialization_version,
                "agent_lifecycle_transition.v1",
            )
            self.assertIn(
                "state_transition_execution", transition.blocked_runtime_behaviors
            )
            self.assertFalse(transition.transition_execution_implemented)
            self.assertFalse(transition.workflow_state_change_implemented)
            self.assertFalse(transition.runtime_lifecycle_engine_implemented)
            self.assertTrue(transition.metadata_only)

    def test_lifecycle_lookups_are_stable(self) -> None:
        profile = agent_lifecycle_profile_by_agent_id("planner_agent")
        transition = agent_lifecycle_transition_by_id("lifecycle::planned->active")

        self.assertIsNone(agent_lifecycle_profile_by_agent_id("missing_agent"))
        self.assertIsNone(agent_lifecycle_transition_by_id("missing_transition"))
        self.assertIsNotNone(profile)
        self.assertIsNotNone(transition)
        assert profile is not None
        assert transition is not None
        self.assertEqual(
            profile.lifecycle_profile_id, "planner_agent_lifecycle_profile"
        )
        self.assertEqual(transition.from_state, "planned")
        self.assertEqual(transition.to_state, "active")

    def test_registry_rejects_mismatched_states_or_profiles(self) -> None:
        lifecycle = agent_lifecycle_registry()

        with self.assertRaisesRegex(ValueError, "states must match"):
            AgentLifecycleRegistry(
                states=tuple(reversed(lifecycle.states)),
                transitions=lifecycle.transitions,
                profiles=lifecycle.profiles,
                transition_ids=lifecycle.transition_ids,
                agent_ids=lifecycle.agent_ids,
                profile_count=lifecycle.profile_count,
                source_registries=lifecycle.source_registries,
            )

        with self.assertRaisesRegex(ValueError, "agent_ids must match profiles"):
            AgentLifecycleRegistry(
                states=lifecycle.states,
                transitions=lifecycle.transitions,
                profiles=lifecycle.profiles,
                transition_ids=lifecycle.transition_ids,
                agent_ids=("other_agent",) + lifecycle.agent_ids[1:],
                profile_count=lifecycle.profile_count,
                source_registries=lifecycle.source_registries,
            )

    def test_lifecycle_metadata_does_not_change_workflow_state_order(self) -> None:
        lifecycle = agent_lifecycle_registry()
        combined_text = " ".join(
            (
                lifecycle.authority_boundary,
                *lifecycle.blocked_runtime_behaviors,
                *(
                    field
                    for transition in lifecycle.transitions
                    for field in (
                        transition.transition_id,
                        transition.transition_boundary,
                        *transition.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        self.assertEqual(WORKFLOW_STEP_ORDER[0].value, "intake")
        self.assertEqual(WORKFLOW_STEP_ORDER[-1].value, "finalization")
        for forbidden_term in (
            "run_state_transition",
            "workflow_state_mutation",
            "lifecycle_engine_execute",
            "invoke_agent",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
