import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    CreativeDiversityAuditRegistry,
    agent_escalation_signal_registry,
    creative_diversity_audit_by_profile_id,
    creative_diversity_audit_registry,
    creative_diversity_audits_for_posture,
    creative_diversity_audits_for_source_registry,
    creative_exploration_budget_registry,
    decision_provenance_registry,
    escalation_trace_registry,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_AUDIT_RECORD_FIELDS = {
    "budget_profile_id",
    "topic_id",
    "audit_stage",
    "budget_serialization_version",
    "budget_posture",
    "max_advisory_variants",
    "max_advisory_refinement_passes",
    "cost_pressure_signal",
    "source_trace_profile_id",
    "source_provenance_profile_id",
    "source_escalation_signal_ids",
    "source_registries",
    "budget_dimensions",
    "advisory_outputs",
    "validated_diversity_surfaces",
    "passive_boundary_flags",
    "audit_findings",
    "missing_coverage_items",
    "profile_blocked_runtime_behaviors",
    "blocked_runtime_behaviors",
    "audit_status",
    "metadata_only_declared",
    "budget_enforcement_implemented",
    "variant_generation_implemented",
    "refinement_triggering_implemented",
    "cost_routing_implemented",
    "agent_invocation_implemented",
    "provider_model_routing_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}


class CreativeDiversityAuditTests(unittest.TestCase):
    def test_audit_registry_covers_exploration_budget_profiles(self) -> None:
        audit_registry = creative_diversity_audit_registry()
        budget_registry = creative_exploration_budget_registry()

        self.assertEqual(audit_registry.role, "creative_diversity_audit_registry")
        self.assertEqual(
            audit_registry.serialization_version,
            "creative_diversity_audit_registry.v1",
        )
        self.assertEqual(
            audit_registry.audit_stage,
            "v4_6_creative_diversity_hardening",
        )
        self.assertEqual(
            audit_registry.budget_profile_ids,
            budget_registry.budget_profile_ids,
        )
        self.assertEqual(audit_registry.topic_ids, budget_registry.topic_ids)
        self.assertEqual(
            audit_registry.budget_postures,
            ("moderate", "broad", "guarded", "narrow"),
        )
        self.assertEqual(
            audit_registry.source_creative_exploration_registry,
            "creative_exploration_budget_registry",
        )
        self.assertEqual(audit_registry.source_registries, budget_registry.source_registries)
        self.assertEqual(
            audit_registry.trace_profile_ids,
            escalation_trace_registry().trace_profile_ids,
        )
        self.assertEqual(
            audit_registry.provenance_profile_ids,
            decision_provenance_registry().provenance_profile_ids,
        )
        self.assertEqual(
            audit_registry.escalation_signal_ids,
            agent_escalation_signal_registry().signal_ids,
        )
        self.assertEqual(audit_registry.audit_count, 4)
        self.assertTrue(audit_registry.all_budget_profiles_covered)
        self.assertTrue(audit_registry.posture_sequence_confirmed)
        self.assertTrue(audit_registry.no_missing_coverage)
        self.assertTrue(audit_registry.metadata_only)
        self.assertFalse(audit_registry.active_diversity_generation_implemented)
        self.assertFalse(audit_registry.budget_enforcement_implemented)
        self.assertFalse(audit_registry.variant_generation_implemented)
        self.assertFalse(audit_registry.refinement_triggering_implemented)
        self.assertFalse(audit_registry.cost_routing_implemented)
        self.assertFalse(audit_registry.agent_invocation_implemented)
        self.assertFalse(audit_registry.generated_output_mutation_implemented)
        self.assertIn("does not enforce budgets", audit_registry.authority_boundary)

    def test_audit_records_are_passive_and_source_aligned(self) -> None:
        registry = creative_diversity_audit_registry()
        known_traces = set(registry.trace_profile_ids)
        known_provenance = set(registry.provenance_profile_ids)
        known_signals = set(registry.escalation_signal_ids)

        for record in registry.audit_records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_AUDIT_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "creative_diversity_audit_record.v1",
            )
            self.assertEqual(record.audit_status, "pass")
            self.assertEqual(record.audit_stage, registry.audit_stage)
            self.assertEqual(record.source_registries, registry.source_registries)
            self.assertEqual(
                record.validated_diversity_surfaces,
                registry.validated_diversity_surfaces,
            )
            self.assertEqual(record.passive_boundary_flags, registry.passive_boundary_flags)
            self.assertFalse(record.missing_coverage_items)
            self.assertIn(record.source_trace_profile_id, known_traces)
            self.assertIn(record.source_provenance_profile_id, known_provenance)
            self.assertTrue(set(record.source_escalation_signal_ids).issubset(known_signals))
            self.assertGreaterEqual(record.max_advisory_variants, 0)
            self.assertLessEqual(record.max_advisory_variants, 3)
            self.assertGreaterEqual(record.max_advisory_refinement_passes, 0)
            self.assertLessEqual(record.max_advisory_refinement_passes, 3)
            self.assertTrue(record.budget_dimensions)
            self.assertTrue(record.advisory_outputs)
            self.assertIn("budget_enforcement", record.profile_blocked_runtime_behaviors)
            self.assertIn("variant_generation", record.profile_blocked_runtime_behaviors)
            self.assertIn(
                "generated_output_modification",
                record.profile_blocked_runtime_behaviors,
            )
            self.assertTrue(record.metadata_only_declared)
            self.assertTrue(record.metadata_only)
            self.assertFalse(record.budget_enforcement_implemented)
            self.assertFalse(record.variant_generation_implemented)
            self.assertFalse(record.refinement_triggering_implemented)
            self.assertFalse(record.cost_routing_implemented)
            self.assertFalse(record.agent_invocation_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.workflow_control_implemented)
            self.assertFalse(record.retry_triggering_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)

    def test_audit_lookup_posture_and_source_filtering_are_stable(self) -> None:
        registry = creative_diversity_audit_registry()
        style_audit = creative_diversity_audit_by_profile_id(
            "creative_exploration_budget::style_aesthetic_alignment"
        )
        missing_audit = creative_diversity_audit_by_profile_id("missing_budget")
        broad_audits = creative_diversity_audits_for_posture("broad")
        tradeoff_source_audits = creative_diversity_audits_for_source_registry(
            "creative_tradeoff_engine"
        )
        missing_source_audits = creative_diversity_audits_for_source_registry(
            "missing_registry"
        )

        self.assertIsNone(missing_audit)
        self.assertIsNotNone(style_audit)
        assert style_audit is not None
        self.assertEqual(style_audit.budget_posture, "broad")
        self.assertEqual(style_audit.max_advisory_variants, 3)
        self.assertIn(
            "style_budget_posture_placeholder",
            style_audit.advisory_outputs,
        )
        self.assertEqual(tuple(record.budget_profile_id for record in broad_audits), (
            "creative_exploration_budget::style_aesthetic_alignment",
        ))
        self.assertEqual(len(tradeoff_source_audits), registry.audit_count)
        self.assertEqual(missing_source_audits, ())
        self.assertIs(
            style_audit,
            creative_diversity_audit_by_profile_id(
                "creative_exploration_budget::style_aesthetic_alignment",
                registry,
            ),
        )

    def test_audit_registry_rejects_mismatched_or_incomplete_records(self) -> None:
        registry = creative_diversity_audit_registry()
        first_record = registry.audit_records[0]
        duplicate_record = first_record.model_copy(update={"topic_id": "duplicate"})
        mismatched_flags_record = first_record.model_copy(
            update={
                "passive_boundary_flags": (
                    "other_flag",
                    "variant_generation_blocked",
                    "refinement_triggering_blocked",
                    "cost_routing_blocked",
                    "agent_invocation_blocked",
                    "provider_model_routing_blocked",
                    "generated_output_mutation_blocked",
                )
            }
        )
        incomplete_record = first_record.model_copy(
            update={"missing_coverage_items": ("variant_generation_enabled",)}
        )

        with self.assertRaisesRegex(ValueError, "budget_profile_ids must be unique"):
            CreativeDiversityAuditRegistry(
                audit_records=(first_record, duplicate_record)
                + registry.audit_records[2:],
                budget_profile_ids=registry.budget_profile_ids,
                topic_ids=registry.topic_ids,
                budget_postures=registry.budget_postures,
                audit_count=registry.audit_count,
                source_registries=registry.source_registries,
                trace_profile_ids=registry.trace_profile_ids,
                provenance_profile_ids=registry.provenance_profile_ids,
                escalation_signal_ids=registry.escalation_signal_ids,
                validated_diversity_surfaces=registry.validated_diversity_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "passive_boundary_flags"):
            CreativeDiversityAuditRegistry(
                audit_records=(mismatched_flags_record,) + registry.audit_records[1:],
                budget_profile_ids=registry.budget_profile_ids,
                topic_ids=registry.topic_ids,
                budget_postures=registry.budget_postures,
                audit_count=registry.audit_count,
                source_registries=registry.source_registries,
                trace_profile_ids=registry.trace_profile_ids,
                provenance_profile_ids=registry.provenance_profile_ids,
                escalation_signal_ids=registry.escalation_signal_ids,
                validated_diversity_surfaces=registry.validated_diversity_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "missing coverage"):
            CreativeDiversityAuditRegistry(
                audit_records=(incomplete_record,) + registry.audit_records[1:],
                budget_profile_ids=registry.budget_profile_ids,
                topic_ids=registry.topic_ids,
                budget_postures=registry.budget_postures,
                audit_count=registry.audit_count,
                source_registries=registry.source_registries,
                trace_profile_ids=registry.trace_profile_ids,
                provenance_profile_ids=registry.provenance_profile_ids,
                escalation_signal_ids=registry.escalation_signal_ids,
                validated_diversity_surfaces=registry.validated_diversity_surfaces,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

    def test_creative_diversity_audit_does_not_change_request_routing(self) -> None:
        request = AssistantRequest(
            query="Generate a diverse family of visual studies.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        creative_diversity_audit_registry()
        creative_diversity_audit_by_profile_id(
            "creative_exploration_budget::planning_execution_fit"
        )
        creative_diversity_audits_for_posture("moderate")
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)
        self.assertNotIn(
            "creative_diversity_audit_registry",
            next_decision.model_dump_json(),
        )

    def test_audit_metadata_does_not_declare_active_diversity_terms(self) -> None:
        registry = creative_diversity_audit_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for record in registry.audit_records
                    for field in (
                        record.budget_profile_id,
                        record.budget_posture,
                        *record.audit_findings,
                        *record.passive_boundary_flags,
                    )
                ),
            )
        )

        for forbidden_term in (
            "enforce_runtime_budget",
            "generate_variant",
            "trigger_runtime_refinement",
            "route_provider",
            "execute_agent",
            "mutate_generated_output",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
