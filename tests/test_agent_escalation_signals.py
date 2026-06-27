import unittest

from creative_coding_assistant.orchestration import (
    AgentEscalationSignalRegistry,
    agent_escalation_signal_by_id,
    agent_escalation_signal_registry,
)

REQUIRED_SIGNAL_FIELDS = {
    "signal_id",
    "category",
    "threshold_direction",
    "advisory_threshold",
    "evidence_sources",
    "policy_rule_ids",
    "signal_boundary",
    "blocked_runtime_behaviors",
    "escalation_performed",
    "provider_routing_implemented",
    "automatic_hitl_triggering_implemented",
    "serialization_version",
    "metadata_only",
}

EXPECTED_CATEGORIES = (
    "confidence",
    "risk",
    "ambiguity",
    "cost",
    "latency",
    "quality",
    "hitl",
)


class AgentEscalationSignalTests(unittest.TestCase):
    def test_signal_registry_covers_required_categories(self) -> None:
        registry = agent_escalation_signal_registry()

        self.assertEqual(registry.role, "agent_escalation_signal_registry")
        self.assertEqual(
            registry.serialization_version,
            "agent_escalation_signal_registry.v1",
        )
        self.assertEqual(registry.categories, EXPECTED_CATEGORIES)
        self.assertEqual(len(registry.signals), 7)
        self.assertEqual(
            registry.source_registries,
            (
                "escalation_policy_registry",
                "consensus_builder_registry",
                "agent_capability_alignment_registry",
            ),
        )
        self.assertIn("does not perform escalation", registry.authority_boundary)
        self.assertFalse(registry.escalation_performed)
        self.assertFalse(registry.provider_routing_implemented)
        self.assertFalse(registry.automatic_hitl_triggering_implemented)
        self.assertTrue(registry.metadata_only)

    def test_signal_thresholds_are_advisory_metadata(self) -> None:
        registry = agent_escalation_signal_registry()

        for signal in registry.signals:
            dumped = signal.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_SIGNAL_FIELDS)
            self.assertEqual(
                signal.serialization_version,
                "agent_escalation_signal.v1",
            )
            self.assertGreaterEqual(signal.advisory_threshold, 0)
            self.assertLessEqual(signal.advisory_threshold, 1)
            self.assertTrue(signal.evidence_sources)
            self.assertTrue(signal.policy_rule_ids)
            self.assertIn("escalation_execution", signal.blocked_runtime_behaviors)
            self.assertFalse(signal.escalation_performed)
            self.assertFalse(signal.provider_routing_implemented)
            self.assertFalse(signal.automatic_hitl_triggering_implemented)
            self.assertTrue(signal.metadata_only)

    def test_signal_lookup_is_stable(self) -> None:
        hitl = agent_escalation_signal_by_id("hitl_escalation_signal")
        missing = agent_escalation_signal_by_id("missing_signal")

        self.assertIsNone(missing)
        self.assertIsNotNone(hitl)
        assert hitl is not None
        self.assertEqual(hitl.category, "hitl")
        self.assertEqual(hitl.threshold_direction, "present")
        self.assertIn("hitl_questions", hitl.evidence_sources)
        self.assertFalse(hitl.automatic_hitl_triggering_implemented)

    def test_registry_rejects_duplicate_or_mismatched_categories(self) -> None:
        registry = agent_escalation_signal_registry()
        duplicate_category = registry.signals[1].model_copy(
            update={"category": "confidence"}
        )

        with self.assertRaisesRegex(ValueError, "categories must match"):
            AgentEscalationSignalRegistry(
                signals=(registry.signals[0], duplicate_category)
                + registry.signals[2:],
                signal_ids=registry.signal_ids,
                categories=registry.categories,
                source_registries=registry.source_registries,
            )

        with self.assertRaisesRegex(ValueError, "signal_ids must match"):
            AgentEscalationSignalRegistry(
                signals=registry.signals,
                signal_ids=("other_signal",) + registry.signal_ids[1:],
                categories=registry.categories,
                source_registries=registry.source_registries,
            )

    def test_signals_do_not_declare_execution_or_routing(self) -> None:
        registry = agent_escalation_signal_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for signal in registry.signals
                    for field in (
                        signal.signal_id,
                        signal.signal_boundary,
                        *signal.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "perform_escalation",
            "route_provider",
            "trigger_hitl",
            "invoke_agent",
            "execute_vote",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
