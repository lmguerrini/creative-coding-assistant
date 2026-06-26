import unittest

from creative_coding_assistant.orchestration import (
    escalation_policy_by_id,
    escalation_policy_registry,
)

EXPECTED_RULE_IDS = (
    "missing_information_review",
    "artifact_risk_review",
    "runtime_incompatibility_review",
    "evaluation_confidence_review",
    "future_agent_escalation_readiness",
)

REQUIRED_RULE_FIELDS = {
    "rule_id",
    "rule_name",
    "policy_stage",
    "authority_boundary",
    "source_contract_registries",
    "trigger_signals",
    "evidence_sources",
    "advisory_outcome",
    "blocked_runtime_behaviors",
    "serialization_version",
}


class EscalationPolicyRegistryTests(unittest.TestCase):
    def test_registry_exposes_metadata_only_policy_rules(self) -> None:
        registry = escalation_policy_registry()

        self.assertEqual(registry.role, "escalation_policy_registry")
        self.assertEqual(registry.rule_ids, EXPECTED_RULE_IDS)
        self.assertEqual(registry.rule_count, 5)
        self.assertTrue(registry.metadata_only)
        self.assertEqual(
            registry.source_contract_registries,
            (
                "agent_capability_registry",
                "artifact_engine_contract_registry",
                "evaluation_engine_contract_registry",
                "workstation_engine_contract_registry",
            ),
        )
        self.assertIn("does not evaluate policy", registry.authority_boundary)
        self.assertEqual(
            {rule.rule_id for rule in registry.rules},
            set(EXPECTED_RULE_IDS),
        )

        for rule in registry.rules:
            dumped = rule.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RULE_FIELDS)
            self.assertEqual(
                rule.serialization_version,
                "escalation_policy_rule.v1",
            )
            self.assertTrue(rule.source_contract_registries)
            self.assertTrue(rule.trigger_signals)
            self.assertTrue(rule.evidence_sources)
            self.assertTrue(rule.advisory_outcome)
            self.assertIn("agent_invocation", rule.blocked_runtime_behaviors)
            self.assertIn(
                "generated_output_modification",
                rule.blocked_runtime_behaviors,
            )
            self.assertIn("does not evaluate policy", rule.authority_boundary)

    def test_policy_lookup_is_stable(self) -> None:
        rule = escalation_policy_by_id("future_agent_escalation_readiness")
        missing = escalation_policy_by_id("missing")

        self.assertIsNone(missing)
        self.assertIsNotNone(rule)
        assert rule is not None
        self.assertEqual(rule.policy_stage, "future_agent_advisory")
        self.assertIn("agent_capability_registry", rule.source_contract_registries)
        self.assertIn("future_agent_hooks", rule.trigger_signals)

    def test_registry_serializes_for_future_policy_consumers(self) -> None:
        dumped = escalation_policy_registry().model_dump(mode="json")

        self.assertEqual(
            dumped["serialization_version"],
            "escalation_policy_registry.v1",
        )
        self.assertEqual(dumped["rule_ids"], list(EXPECTED_RULE_IDS))
        self.assertEqual(len(dumped["rules"]), 5)
        self.assertTrue(dumped["metadata_only"])
        self.assertEqual(
            dumped["rules"][0]["rule_id"],
            "missing_information_review",
        )

    def test_registry_does_not_declare_runtime_policy_behavior(self) -> None:
        registry = escalation_policy_registry()

        for rule in registry.rules:
            combined_text = " ".join(
                (
                    rule.authority_boundary,
                    rule.advisory_outcome,
                    *rule.blocked_runtime_behaviors,
                )
            )
            self.assertNotIn("execute_provider", combined_text)
            self.assertNotIn("autonomous_retry", combined_text)
            self.assertNotIn("runtime_auto_selection", combined_text)


if __name__ == "__main__":
    unittest.main()
