import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    ContradictionDetectionPlan,
    build_contradiction_detection,
    build_source_credibility_engine,
    contradiction_detection_entries_for_confidence,
    contradiction_detection_entries_for_status,
    contradiction_detection_entry_by_id,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Contradiction Detection",)


class ContradictionDetectionTests(unittest.TestCase):
    def test_plan_builds_advisory_contradiction_metadata(self) -> None:
        plan = build_contradiction_detection(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "contradiction_detection")
        self.assertEqual(
            plan.serialization_version,
            "contradiction_detection_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.source_credibility_role, "source_credibility_engine")
        self.assertEqual(plan.credibility_entry_count, 5)
        self.assertEqual(plan.covered_roadmap_items, EXPECTED_ROADMAP_ITEMS)
        self.assertEqual(plan.covered_roadmap_item_count, 1)
        self.assertEqual(plan.source_count, 57)
        self.assertEqual(plan.domain_count, 43)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.entry_count, 5)
        self.assertEqual(plan.candidate_entry_count, 1)
        self.assertEqual(plan.review_required_entry_count, 1)
        self.assertEqual(plan.guarded_entry_count, 3)
        self.assertEqual(plan.high_confidence_entry_count, 4)
        self.assertEqual(plan.hitl_required_entry_count, 5)
        self.assertFalse(plan.detected_contradiction_ids)
        self.assertFalse(plan.compared_live_claim_ids)
        self.assertFalse(plan.resolved_contradiction_ids)
        self.assertFalse(plan.emitted_contradiction_escalation_ids)
        self.assertFalse(plan.written_contradiction_record_ids)
        self.assertEqual(plan.overall_contradiction_score, 820)
        self.assertEqual(plan.overall_contradiction_posture, "guarded")
        self.assertIn(
            "does not execute contradiction detection",
            plan.authority_boundary,
        )
        self.assertTrue(plan.contradiction_detection_capability_implemented)
        self.assertTrue(plan.contradiction_detection_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.source_credibility_metadata_used)
        self.assertFalse(plan.contradiction_detection_execution_implemented)
        self.assertFalse(plan.live_claim_comparison_implemented)
        self.assertFalse(plan.contradiction_resolution_execution_implemented)
        self.assertFalse(plan.contradiction_escalation_emission_implemented)
        self.assertFalse(plan.contradiction_record_write_implemented)
        self.assertFalse(plan.source_credibility_scoring_execution_implemented)
        self.assertFalse(plan.research_confidence_scoring_implemented)
        self.assertFalse(plan.external_source_fetch_implemented)
        self.assertFalse(plan.retrieval_execution_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_entries_score_contradiction_without_execution(self) -> None:
        plan = build_contradiction_detection(route="generate")
        credibility_ids = set(plan.credibility_entry_ids)

        for entry in plan.entries:
            self.assertEqual(
                entry.serialization_version,
                "contradiction_detection_entry.v1",
            )
            self.assertEqual(entry.route_name, RouteName.GENERATE)
            self.assertEqual(
                entry.entry_id,
                f"contradiction_detection::{entry.contradiction_kind}",
            )
            self.assertTrue(set(entry.credibility_entry_ids).issubset(credibility_ids))
            self.assertEqual(entry.source_count, plan.source_count)
            self.assertEqual(entry.domain_count, plan.domain_count)
            self.assertEqual(
                entry.contradiction_score,
                min(
                    1000,
                    max(
                        0,
                        entry.claim_alignment_score * 2
                        + entry.evidence_conflict_score * 3
                        + entry.source_disagreement_score * 2
                        + entry.governance_alignment_score * 2
                        + entry.mutation_risk_score
                        + entry.governance_weight,
                    ),
                ),
            )
            self.assertIn("contradiction_detection", entry.context_tags)
            self.assertIn(
                "contradiction_detection_execution",
                entry.blocked_runtime_behaviors,
            )
            self.assertIn("live_claim_comparison", entry.blocked_runtime_behaviors)
            self.assertTrue(entry.hitl_required_before_contradiction_detection)
            self.assertTrue(entry.contradiction_detection_capability_implemented)
            self.assertTrue(entry.contradiction_detection_metadata_implemented)
            self.assertTrue(entry.source_credibility_metadata_used)
            self.assertFalse(entry.contradiction_detection_execution_implemented)
            self.assertFalse(entry.live_claim_comparison_implemented)
            self.assertFalse(entry.contradiction_resolution_execution_implemented)
            self.assertFalse(entry.contradiction_record_write_implemented)
            self.assertFalse(entry.source_credibility_scoring_execution_implemented)
            self.assertFalse(entry.retrieval_execution_implemented)
            self.assertFalse(entry.provider_execution_implemented)
            self.assertFalse(entry.generated_output_mutation_implemented)
            self.assertFalse(entry.runtime_evolution_implemented)
            self.assertTrue(entry.advisory_only)

        alignment = contradiction_detection_entry_by_id(
            "contradiction_detection::claim_alignment_review",
            plan,
        )
        self.assertIsNotNone(alignment)
        assert alignment is not None
        self.assertEqual(alignment.status, "guarded")
        self.assertEqual(alignment.confidence, "guarded")
        self.assertEqual(
            len(contradiction_detection_entries_for_status("review_required", plan)),
            1,
        )
        self.assertEqual(
            len(contradiction_detection_entries_for_confidence("high", plan)),
            1,
        )

    def test_plan_rejects_mismatched_contradiction_metadata(self) -> None:
        plan = build_contradiction_detection()
        payload = plan.model_dump(mode="json")
        payload["entry_ids"] = ("missing",) + tuple(payload["entry_ids"][1:])

        with self.assertRaisesRegex(ValueError, "entry_ids must match"):
            ContradictionDetectionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_contradiction_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_contradiction_score must match",
        ):
            ContradictionDetectionPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["detected_contradiction_ids"] = (plan.entry_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "detected_contradiction_ids must remain empty",
        ):
            ContradictionDetectionPlan(**payload)

    def test_contradiction_composes_with_credibility_metadata(self) -> None:
        credibility = build_source_credibility_engine(route=RouteName.REVIEW)
        plan = build_contradiction_detection(
            route=RouteName.REVIEW,
            credibility=credibility,
        )

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(plan.checked_at, credibility.checked_at)
        self.assertEqual(plan.credibility_entry_ids, credibility.entry_ids)
        self.assertTrue(plan.source_credibility_metadata_used)
        self.assertFalse(plan.contradiction_detection_execution_implemented)

    def test_contradiction_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Plan contradiction review for a p5.js shader study.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_contradiction_detection(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_contradiction_does_not_declare_runtime_mutation_terms(self) -> None:
        plan = build_contradiction_detection(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *plan.covered_roadmap_items,
                *(
                    field
                    for entry in plan.entries
                    for field in (
                        entry.entry_id,
                        entry.contradiction_kind,
                        entry.status,
                        entry.confidence,
                        entry.contradiction_axis,
                        *entry.credibility_entry_ids,
                        entry.contradiction_summary,
                        *entry.context_tags,
                        *entry.explainability_notes,
                        *entry.advisory_actions,
                        *entry.evidence,
                        *entry.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "execute_contradiction_detection(",
            "compare_live_claims(",
            "resolve_contradiction(",
            "emit_contradiction_escalation(",
            "write_contradiction_record(",
            "score_source_credibility(",
            "score_research_confidence(",
            "fetch_external_source(",
            "browse_web(",
            "download_paper(",
            "mutate_source_registry(",
            "execute_retrieval(",
            "mutate_retrieval_config(",
            "mutate_vector_index(",
            "enrich_kb(",
            "write_kb_storage(",
            "write_storage(",
            "provision_provider(",
            "infer_api_key(",
            "route_provider(",
            "execute_provider(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
