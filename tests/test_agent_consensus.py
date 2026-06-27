import unittest

from creative_coding_assistant.orchestration import (
    ConsensusBuilderRegistry,
    agent_debate_registry,
    consensus_agreement_surface_by_topic,
    consensus_builder_registry,
    consensus_voting_input_by_topic,
)

REQUIRED_VOTING_INPUT_FIELDS = {
    "voting_input_id",
    "topic_id",
    "participant_agent_ids",
    "claim_ids",
    "voting_dimensions",
    "confidence_placeholder_key",
    "aggregation_mode",
    "input_boundary",
    "blocked_runtime_behaviors",
    "voting_execution_implemented",
    "final_answer_selection_implemented",
    "final_synthesis_mutation_implemented",
    "serialization_version",
    "metadata_only",
}

REQUIRED_AGREEMENT_FIELDS = {
    "surface_id",
    "topic_id",
    "agreement_metadata_keys",
    "disagreement_metadata_keys",
    "unresolved_risk_keys",
    "confidence_placeholder_key",
    "surface_boundary",
    "blocked_runtime_behaviors",
    "voting_execution_implemented",
    "final_answer_selection_implemented",
    "final_synthesis_mutation_implemented",
    "serialization_version",
    "metadata_only",
}


class AgentConsensusTests(unittest.TestCase):
    def test_consensus_registry_matches_debate_topics(self) -> None:
        consensus = consensus_builder_registry()
        debate = agent_debate_registry()

        self.assertEqual(consensus.role, "consensus_builder_registry")
        self.assertEqual(
            consensus.serialization_version,
            "consensus_builder_registry.v1",
        )
        self.assertEqual(consensus.topic_ids, debate.topic_ids)
        self.assertEqual(len(consensus.voting_inputs), 4)
        self.assertEqual(len(consensus.agreement_surfaces), 4)
        self.assertEqual(consensus.aggregation_mode, "placeholder_only")
        self.assertEqual(consensus.source_registries, ("agent_debate_registry",))
        self.assertIn("does not execute voting", consensus.authority_boundary)
        self.assertFalse(consensus.voting_execution_implemented)
        self.assertFalse(consensus.final_answer_selection_implemented)
        self.assertFalse(consensus.final_synthesis_mutation_implemented)
        self.assertTrue(consensus.metadata_only)

    def test_voting_inputs_and_agreement_surfaces_are_placeholders(self) -> None:
        consensus = consensus_builder_registry()

        for item in consensus.voting_inputs:
            dumped = item.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_VOTING_INPUT_FIELDS)
            self.assertEqual(
                item.serialization_version,
                "consensus_voting_input.v1",
            )
            self.assertEqual(item.aggregation_mode, "placeholder_only")
            self.assertTrue(item.voting_dimensions)
            self.assertIn("voting_execution", item.blocked_runtime_behaviors)
            self.assertFalse(item.voting_execution_implemented)
            self.assertFalse(item.final_answer_selection_implemented)
            self.assertFalse(item.final_synthesis_mutation_implemented)
            self.assertTrue(item.metadata_only)

        for surface in consensus.agreement_surfaces:
            dumped = surface.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_AGREEMENT_FIELDS)
            self.assertEqual(
                surface.serialization_version,
                "consensus_agreement_surface.v1",
            )
            self.assertTrue(surface.agreement_metadata_keys)
            self.assertTrue(surface.disagreement_metadata_keys)
            self.assertTrue(surface.unresolved_risk_keys)
            self.assertFalse(surface.voting_execution_implemented)
            self.assertFalse(surface.final_answer_selection_implemented)
            self.assertFalse(surface.final_synthesis_mutation_implemented)
            self.assertTrue(surface.metadata_only)

    def test_consensus_lookups_are_stable(self) -> None:
        voting_input = consensus_voting_input_by_topic("planning_execution_fit")
        agreement_surface = consensus_agreement_surface_by_topic(
            "final_synthesis_readiness"
        )

        self.assertIsNone(consensus_voting_input_by_topic("missing_topic"))  # type: ignore[arg-type]
        self.assertIsNone(consensus_agreement_surface_by_topic("missing_topic"))  # type: ignore[arg-type]
        self.assertIsNotNone(voting_input)
        self.assertIsNotNone(agreement_surface)
        assert voting_input is not None
        assert agreement_surface is not None
        self.assertIn("planner_agent", voting_input.participant_agent_ids)
        self.assertIn(
            "final_synthesis_readiness_agreement_points",
            agreement_surface.agreement_metadata_keys,
        )

    def test_registry_rejects_topic_and_mode_mismatches(self) -> None:
        consensus = consensus_builder_registry()
        bad_input = consensus.voting_inputs[0].model_copy(
            update={"aggregation_mode": "other"}
        )

        with self.assertRaises(ValueError):
            ConsensusBuilderRegistry(
                voting_inputs=(bad_input,) + consensus.voting_inputs[1:],
                agreement_surfaces=consensus.agreement_surfaces,
                voting_input_ids=consensus.voting_input_ids,
                agreement_surface_ids=consensus.agreement_surface_ids,
                topic_ids=consensus.topic_ids,
                source_registries=consensus.source_registries,
            )

        with self.assertRaisesRegex(ValueError, "voting_input_ids must match"):
            ConsensusBuilderRegistry(
                voting_inputs=tuple(reversed(consensus.voting_inputs)),
                agreement_surfaces=consensus.agreement_surfaces,
                voting_input_ids=consensus.voting_input_ids,
                agreement_surface_ids=consensus.agreement_surface_ids,
                topic_ids=consensus.topic_ids,
                source_registries=consensus.source_registries,
            )

    def test_consensus_metadata_does_not_select_or_mutate_outputs(self) -> None:
        consensus = consensus_builder_registry()
        dumped = consensus.model_dump(mode="json")
        combined_text = " ".join(
            (
                consensus.authority_boundary,
                *consensus.blocked_runtime_behaviors,
                *(
                    field
                    for item in consensus.voting_inputs
                    for field in (
                        item.voting_input_id,
                        item.input_boundary,
                        *item.blocked_runtime_behaviors,
                    )
                ),
                *(
                    field
                    for surface in consensus.agreement_surfaces
                    for field in (
                        surface.surface_id,
                        surface.surface_boundary,
                        *surface.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        self.assertEqual(len(dumped["voting_inputs"]), 4)
        self.assertEqual(len(dumped["agreement_surfaces"]), 4)
        for forbidden_term in (
            "execute_vote",
            "select_final_answer",
            "mutate_final_synthesis",
            "rerun_generation",
            "provider_route",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
