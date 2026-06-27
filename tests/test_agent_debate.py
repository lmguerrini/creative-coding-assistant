import unittest

from creative_coding_assistant.orchestration import (
    AgentDebateRegistry,
    agent_contract_registry,
    agent_debate_registry,
    debate_claim_by_id,
    debate_participant_by_agent_id,
    debate_round_by_topic,
)

REQUIRED_PARTICIPANT_FIELDS = {
    "participant_id",
    "agent_id",
    "debate_roles",
    "evidence_surface_ids",
    "blocked_runtime_behaviors",
    "debate_execution_implemented",
    "retry_triggering_implemented",
    "output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}

REQUIRED_CLAIM_FIELDS = {
    "claim_id",
    "topic_id",
    "claimant_agent_id",
    "counterclaim_agent_ids",
    "claim_surface",
    "counterclaim_surface",
    "evidence_surfaces",
    "advisory_output_keys",
    "debate_boundary",
    "blocked_runtime_behaviors",
    "debate_execution_implemented",
    "retry_triggering_implemented",
    "output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}

REQUIRED_ROUND_FIELDS = {
    "round_id",
    "topic_id",
    "participant_agent_ids",
    "claim_ids",
    "max_exchange_count",
    "evidence_surfaces",
    "round_boundary",
    "blocked_runtime_behaviors",
    "debate_execution_implemented",
    "retry_triggering_implemented",
    "output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}


class AgentDebateTests(unittest.TestCase):
    def test_debate_registry_covers_passive_agents(self) -> None:
        debate = agent_debate_registry()
        contracts = agent_contract_registry()

        self.assertEqual(debate.role, "agent_debate_registry")
        self.assertEqual(debate.serialization_version, "agent_debate_registry.v1")
        self.assertEqual(debate.participant_agent_ids, contracts.agent_ids)
        self.assertEqual(debate.max_rounds, 4)
        self.assertEqual(
            debate.topic_ids,
            (
                "planning_execution_fit",
                "style_aesthetic_alignment",
                "curation_refinement_need",
                "final_synthesis_readiness",
            ),
        )
        self.assertEqual(
            debate.source_registries,
            ("agent_contract_registry", "shared_context_view_registry"),
        )
        self.assertIn("does not execute debate loops", debate.authority_boundary)
        self.assertFalse(debate.debate_execution_implemented)
        self.assertFalse(debate.retry_triggering_implemented)
        self.assertFalse(debate.output_mutation_implemented)
        self.assertTrue(debate.metadata_only)

    def test_participants_claims_and_rounds_are_advisory(self) -> None:
        debate = agent_debate_registry()

        for participant in debate.participants:
            dumped = participant.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PARTICIPANT_FIELDS)
            self.assertEqual(
                participant.serialization_version,
                "agent_debate_participant.v1",
            )
            self.assertTrue(participant.debate_roles)
            self.assertIn("debate_loop_execution", participant.blocked_runtime_behaviors)
            self.assertFalse(participant.debate_execution_implemented)
            self.assertFalse(participant.retry_triggering_implemented)
            self.assertFalse(participant.output_mutation_implemented)
            self.assertTrue(participant.metadata_only)

        for claim in debate.claims:
            dumped = claim.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CLAIM_FIELDS)
            self.assertEqual(claim.serialization_version, "agent_debate_claim.v1")
            self.assertTrue(claim.counterclaim_agent_ids)
            self.assertTrue(claim.evidence_surfaces)
            self.assertIn("advisory only", claim.debate_boundary)
            self.assertFalse(claim.debate_execution_implemented)
            self.assertFalse(claim.retry_triggering_implemented)
            self.assertFalse(claim.output_mutation_implemented)
            self.assertTrue(claim.metadata_only)

        for round_contract in debate.rounds:
            dumped = round_contract.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_ROUND_FIELDS)
            self.assertEqual(round_contract.serialization_version, "agent_debate_round.v1")
            self.assertLessEqual(round_contract.max_exchange_count, 2)
            self.assertTrue(round_contract.evidence_surfaces)
            self.assertFalse(round_contract.debate_execution_implemented)
            self.assertFalse(round_contract.retry_triggering_implemented)
            self.assertFalse(round_contract.output_mutation_implemented)
            self.assertTrue(round_contract.metadata_only)

    def test_debate_lookups_are_stable(self) -> None:
        participant = debate_participant_by_agent_id("critic_agent")
        claim = debate_claim_by_id("debate_claim::style_aesthetic_alignment")
        round_contract = debate_round_by_topic("final_synthesis_readiness")

        self.assertIsNone(debate_participant_by_agent_id("missing_agent"))
        self.assertIsNone(debate_claim_by_id("missing_claim"))
        self.assertIsNone(debate_round_by_topic("missing_topic"))  # type: ignore[arg-type]
        self.assertIsNotNone(participant)
        self.assertIsNotNone(claim)
        self.assertIsNotNone(round_contract)
        assert participant is not None
        assert claim is not None
        assert round_contract is not None
        self.assertIn("counterclaimant", participant.debate_roles)
        self.assertEqual(claim.claimant_agent_id, "art_direction_agent")
        self.assertIn("critic_agent", round_contract.participant_agent_ids)

    def test_registry_rejects_unknown_participants_and_topic_order_changes(self) -> None:
        debate = agent_debate_registry()
        bad_claim = debate.claims[0].model_copy(
            update={"claimant_agent_id": "missing_agent"}
        )

        with self.assertRaisesRegex(ValueError, "claimant_agent_id"):
            AgentDebateRegistry(
                participants=debate.participants,
                claims=(bad_claim,) + debate.claims[1:],
                rounds=debate.rounds,
                participant_agent_ids=debate.participant_agent_ids,
                claim_ids=debate.claim_ids,
                round_ids=debate.round_ids,
                topic_ids=debate.topic_ids,
                max_rounds=debate.max_rounds,
                source_registries=debate.source_registries,
            )

        reversed_rounds = tuple(reversed(debate.rounds))
        with self.assertRaisesRegex(ValueError, "round_ids must match rounds"):
            AgentDebateRegistry(
                participants=debate.participants,
                claims=debate.claims,
                rounds=reversed_rounds,
                participant_agent_ids=debate.participant_agent_ids,
                claim_ids=debate.claim_ids,
                round_ids=debate.round_ids,
                topic_ids=debate.topic_ids,
                max_rounds=debate.max_rounds,
                source_registries=debate.source_registries,
            )

    def test_debate_metadata_does_not_declare_retry_or_generation_behavior(self) -> None:
        debate = agent_debate_registry()
        dumped = debate.model_dump(mode="json")
        combined_text = " ".join(
            (
                debate.authority_boundary,
                *debate.blocked_runtime_behaviors,
                *(
                    field
                    for claim in debate.claims
                    for field in (
                        claim.claim_id,
                        claim.debate_boundary,
                        *claim.blocked_runtime_behaviors,
                    )
                ),
                *(
                    field
                    for round_contract in debate.rounds
                    for field in (
                        round_contract.round_id,
                        round_contract.round_boundary,
                        *round_contract.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        self.assertEqual(len(dumped["participants"]), 12)
        self.assertEqual(len(dumped["claims"]), 4)
        self.assertEqual(len(dumped["rounds"]), 4)
        for forbidden_term in (
            "execute_debate_loop",
            "autonomous_retry",
            "rerun_generation",
            "mutate_generated_output",
            "provider_route",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
