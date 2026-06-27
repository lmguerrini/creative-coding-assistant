import unittest

from creative_coding_assistant.orchestration import (
    AgentCoordinationRegistry,
    agent_coordination_registry,
    coordination_event_contract_by_type,
    coordination_handoff_channel_by_id,
    coordination_responsibility_by_id,
    parallel_scheduling_registry,
)

REQUIRED_RESPONSIBILITY_FIELDS = {
    "coordinator_id",
    "stage_id",
    "group_id",
    "responsible_agent_ids",
    "responsibilities",
    "required_inputs",
    "produced_handoff_channel_ids",
    "coordination_boundary",
    "blocked_runtime_behaviors",
    "live_coordination_implemented",
    "agent_actions_triggered",
    "output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}

REQUIRED_HANDOFF_FIELDS = {
    "handoff_channel_id",
    "source_group_id",
    "target_group_id",
    "source_agent_ids",
    "target_agent_ids",
    "coordination_event_types",
    "payload_metadata_keys",
    "handoff_boundary",
    "blocked_runtime_behaviors",
    "live_coordination_implemented",
    "agent_actions_triggered",
    "output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}

REQUIRED_EVENT_FIELDS = {
    "event_type",
    "event_order",
    "payload_fields",
    "emitted_by_handoff_channel_ids",
    "event_boundary",
    "blocked_runtime_behaviors",
    "event_emission_implemented",
    "agent_actions_triggered",
    "output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}


class AgentCoordinationTests(unittest.TestCase):
    def test_coordination_registry_matches_scheduling_groups(self) -> None:
        coordination = agent_coordination_registry()
        scheduling = parallel_scheduling_registry()

        self.assertEqual(coordination.role, "agent_coordination_registry")
        self.assertEqual(coordination.serialization_version, "coordination_registry.v1")
        self.assertEqual(len(coordination.responsibilities), scheduling.group_count)
        self.assertEqual(len(coordination.handoff_channels), scheduling.group_count - 1)
        self.assertEqual(coordination.event_types, (
            "coordination_checkpoint_declared",
            "handoff_metadata_available",
            "coordination_risk_flagged",
            "human_review_signal_declared",
        ))
        self.assertEqual(
            coordination.source_registries,
            ("parallel_scheduling_registry", "agent_dependency_graph_registry"),
        )
        self.assertIn(
            "do not implement live coordination",
            coordination.authority_boundary,
        )
        self.assertFalse(coordination.live_coordination_implemented)
        self.assertFalse(coordination.agent_actions_triggered)
        self.assertFalse(coordination.output_mutation_implemented)
        self.assertTrue(coordination.metadata_only)

    def test_responsibilities_channels_and_events_are_passive_metadata(self) -> None:
        coordination = agent_coordination_registry()

        for responsibility in coordination.responsibilities:
            dumped = responsibility.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RESPONSIBILITY_FIELDS)
            self.assertEqual(
                responsibility.serialization_version,
                "coordination_responsibility.v1",
            )
            self.assertTrue(responsibility.responsible_agent_ids)
            self.assertTrue(responsibility.required_inputs)
            self.assertIn("live_coordination", responsibility.blocked_runtime_behaviors)
            self.assertFalse(responsibility.live_coordination_implemented)
            self.assertFalse(responsibility.agent_actions_triggered)
            self.assertFalse(responsibility.output_mutation_implemented)
            self.assertTrue(responsibility.metadata_only)

        for channel in coordination.handoff_channels:
            dumped = channel.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_HANDOFF_FIELDS)
            self.assertEqual(
                channel.serialization_version,
                "coordination_handoff_channel.v1",
            )
            self.assertTrue(channel.payload_metadata_keys)
            self.assertIn("does not emit events", channel.handoff_boundary)
            self.assertFalse(channel.live_coordination_implemented)
            self.assertFalse(channel.agent_actions_triggered)
            self.assertFalse(channel.output_mutation_implemented)
            self.assertTrue(channel.metadata_only)

        for event_contract in coordination.event_contracts:
            dumped = event_contract.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_EVENT_FIELDS)
            self.assertEqual(
                event_contract.serialization_version,
                "coordination_event.v1",
            )
            self.assertFalse(event_contract.event_emission_implemented)
            self.assertFalse(event_contract.agent_actions_triggered)
            self.assertFalse(event_contract.output_mutation_implemented)
            self.assertTrue(event_contract.metadata_only)

    def test_handoff_channels_move_downstream_deterministically(self) -> None:
        coordination = agent_coordination_registry()
        scheduling = parallel_scheduling_registry()
        group_index = {
            group.group_id: index for index, group in enumerate(scheduling.groups)
        }

        for channel in coordination.handoff_channels:
            self.assertLess(
                group_index[channel.source_group_id],
                group_index[channel.target_group_id],
            )

        first_channel = coordination.handoff_channels[0]
        reversed_channel = first_channel.model_copy(
            update={
                "source_group_id": first_channel.target_group_id,
                "target_group_id": first_channel.source_group_id,
            }
        )
        with self.assertRaisesRegex(ValueError, "handoff channels must move downstream"):
            AgentCoordinationRegistry(
                responsibilities=coordination.responsibilities,
                handoff_channels=(reversed_channel,) + coordination.handoff_channels[1:],
                event_contracts=coordination.event_contracts,
                coordinator_ids=coordination.coordinator_ids,
                handoff_channel_ids=coordination.handoff_channel_ids,
                event_types=coordination.event_types,
                source_registries=coordination.source_registries,
            )

    def test_coordination_lookups_are_stable(self) -> None:
        responsibility = coordination_responsibility_by_id(
            "coordinator::foundational_context"
        )
        channel = coordination_handoff_channel_by_id(
            "coordination_handoff::parallel_group::foundational_context->"
            "parallel_group::domain_context"
        )
        event_contract = coordination_event_contract_by_type(
            "handoff_metadata_available"
        )

        self.assertIsNone(coordination_responsibility_by_id("missing"))
        self.assertIsNone(coordination_handoff_channel_by_id("missing"))
        self.assertIsNone(coordination_event_contract_by_type("missing"))  # type: ignore[arg-type]
        self.assertIsNotNone(responsibility)
        self.assertIsNotNone(channel)
        self.assertIsNotNone(event_contract)
        assert responsibility is not None
        assert channel is not None
        assert event_contract is not None
        self.assertIn("planner_agent", responsibility.responsible_agent_ids)
        self.assertIn("research_agent", channel.source_agent_ids)
        self.assertEqual(event_contract.event_order, 2)

    def test_coordination_serializes_without_live_behavior(self) -> None:
        coordination = agent_coordination_registry()
        dumped = coordination.model_dump(mode="json")

        self.assertEqual(dumped["role"], "agent_coordination_registry")
        self.assertEqual(len(dumped["responsibilities"]), 6)
        self.assertEqual(len(dumped["handoff_channels"]), 5)
        self.assertEqual(len(dumped["event_contracts"]), 4)
        combined_text = " ".join(
            (
                coordination.authority_boundary,
                *coordination.blocked_runtime_behaviors,
                *(
                    field
                    for responsibility in coordination.responsibilities
                    for field in (
                        responsibility.coordinator_id,
                        responsibility.coordination_boundary,
                        *responsibility.blocked_runtime_behaviors,
                    )
                ),
                *(
                    field
                    for channel in coordination.handoff_channels
                    for field in (
                        channel.handoff_channel_id,
                        channel.handoff_boundary,
                        *channel.blocked_runtime_behaviors,
                    )
                ),
            )
        )
        for forbidden_term in (
            "live_agent_bus",
            "trigger_agent_action",
            "mutate_output",
            "execute_coordination",
            "provider_route",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
