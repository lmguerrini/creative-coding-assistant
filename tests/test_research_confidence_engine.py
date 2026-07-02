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
    ResearchConfidencePlan,
    build_contradiction_detection,
    build_research_confidence_engine,
    research_confidence_entries_for_confidence,
    research_confidence_entries_for_status,
    research_confidence_entry_by_id,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Research Confidence Engine",)


class ResearchConfidenceEngineTests(unittest.TestCase):
    def test_plan_builds_advisory_research_confidence_metadata(self) -> None:
        plan = build_research_confidence_engine(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "research_confidence_engine")
        self.assertEqual(
            plan.serialization_version,
            "research_confidence_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.contradiction_detection_role,
            "contradiction_detection",
        )
        self.assertEqual(plan.contradiction_entry_count, 5)
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
        self.assertFalse(plan.scored_research_confidence_ids)
        self.assertFalse(plan.calculated_confidence_score_ids)
        self.assertFalse(plan.mutated_confidence_label_ids)
        self.assertFalse(plan.written_confidence_record_ids)
        self.assertFalse(plan.emitted_confidence_escalation_ids)
        self.assertEqual(plan.overall_research_confidence_score, 846)
        self.assertEqual(plan.overall_research_confidence_posture, "guarded")
        self.assertIn(
            "does not execute research confidence scoring",
            plan.authority_boundary,
        )
        self.assertTrue(plan.research_confidence_engine_capability_implemented)
        self.assertTrue(plan.research_confidence_engine_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.contradiction_detection_metadata_used)
        self.assertFalse(plan.research_confidence_scoring_execution_implemented)
        self.assertFalse(plan.live_research_confidence_scoring_implemented)
        self.assertFalse(plan.confidence_score_calculation_implemented)
        self.assertFalse(plan.confidence_label_mutation_implemented)
        self.assertFalse(plan.confidence_record_write_implemented)
        self.assertFalse(plan.confidence_escalation_emission_implemented)
        self.assertFalse(plan.contradiction_detection_execution_implemented)
        self.assertFalse(plan.live_claim_comparison_implemented)
        self.assertFalse(plan.contradiction_resolution_execution_implemented)
        self.assertFalse(plan.source_credibility_scoring_execution_implemented)
        self.assertFalse(plan.external_source_fetch_implemented)
        self.assertFalse(plan.retrieval_execution_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_entries_score_confidence_posture_without_execution(self) -> None:
        plan = build_research_confidence_engine(route="generate")
        contradiction_ids = set(plan.contradiction_entry_ids)

        for entry in plan.entries:
            self.assertEqual(
                entry.serialization_version,
                "research_confidence_entry.v1",
            )
            self.assertEqual(entry.route_name, RouteName.GENERATE)
            self.assertEqual(
                entry.entry_id,
                f"research_confidence_engine::{entry.confidence_kind}",
            )
            self.assertTrue(
                set(entry.contradiction_entry_ids).issubset(contradiction_ids)
            )
            self.assertEqual(entry.source_count, plan.source_count)
            self.assertEqual(entry.domain_count, plan.domain_count)
            self.assertEqual(
                entry.research_confidence_score,
                min(
                    1000,
                    max(
                        0,
                        entry.evidence_strength_score * 2
                        + entry.source_reliability_score * 2
                        + entry.contradiction_risk_score * 3
                        + entry.coverage_completeness_score * 2
                        + entry.governance_alignment_score * 2
                        + entry.mutation_risk_score
                        + entry.governance_weight,
                    ),
                ),
            )
            self.assertIn("research_confidence_engine", entry.context_tags)
            self.assertIn(
                "research_confidence_scoring_execution",
                entry.blocked_runtime_behaviors,
            )
            self.assertIn("confidence_record_write", entry.blocked_runtime_behaviors)
            self.assertTrue(entry.hitl_required_before_confidence_scoring)
            self.assertTrue(entry.research_confidence_engine_capability_implemented)
            self.assertTrue(entry.research_confidence_engine_metadata_implemented)
            self.assertTrue(entry.contradiction_detection_metadata_used)
            self.assertFalse(entry.research_confidence_scoring_execution_implemented)
            self.assertFalse(entry.live_research_confidence_scoring_implemented)
            self.assertFalse(entry.confidence_score_calculation_implemented)
            self.assertFalse(entry.confidence_label_mutation_implemented)
            self.assertFalse(entry.confidence_record_write_implemented)
            self.assertFalse(entry.contradiction_detection_execution_implemented)
            self.assertFalse(entry.live_claim_comparison_implemented)
            self.assertFalse(entry.retrieval_execution_implemented)
            self.assertFalse(entry.provider_execution_implemented)
            self.assertFalse(entry.generated_output_mutation_implemented)
            self.assertFalse(entry.runtime_evolution_implemented)
            self.assertTrue(entry.advisory_only)

        evidence = research_confidence_entry_by_id(
            "research_confidence_engine::evidence_strength_review",
            plan,
        )
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertEqual(evidence.status, "guarded")
        self.assertEqual(evidence.confidence, "guarded")
        self.assertEqual(
            len(research_confidence_entries_for_status("review_required", plan)),
            1,
        )
        self.assertEqual(
            len(research_confidence_entries_for_confidence("high", plan)),
            1,
        )

    def test_plan_rejects_mismatched_research_confidence_metadata(self) -> None:
        plan = build_research_confidence_engine()
        payload = plan.model_dump(mode="json")
        payload["entry_ids"] = ("missing",) + tuple(payload["entry_ids"][1:])

        with self.assertRaisesRegex(ValueError, "entry_ids must match"):
            ResearchConfidencePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_research_confidence_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_research_confidence_score must match",
        ):
            ResearchConfidencePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["scored_research_confidence_ids"] = (plan.entry_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "scored_research_confidence_ids must remain empty",
        ):
            ResearchConfidencePlan(**payload)

    def test_research_confidence_composes_with_contradiction_metadata(self) -> None:
        contradiction = build_contradiction_detection(route=RouteName.REVIEW)
        plan = build_research_confidence_engine(
            route=RouteName.REVIEW,
            contradiction=contradiction,
        )

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(plan.checked_at, contradiction.checked_at)
        self.assertEqual(plan.contradiction_entry_ids, contradiction.entry_ids)
        self.assertTrue(plan.contradiction_detection_metadata_used)
        self.assertFalse(plan.research_confidence_scoring_execution_implemented)

    def test_research_confidence_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Plan confidence review for a WebGL literature summary.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.THREE_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_research_confidence_engine(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_research_confidence_does_not_declare_runtime_mutation_terms(self) -> None:
        plan = build_research_confidence_engine(route=RouteName.GENERATE)
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
                        entry.confidence_kind,
                        entry.status,
                        entry.confidence,
                        entry.confidence_axis,
                        *entry.contradiction_entry_ids,
                        entry.confidence_summary,
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
            "score_research_confidence(",
            "calculate_research_confidence(",
            "calculate_confidence_score(",
            "mutate_confidence_label(",
            "write_confidence_record(",
            "emit_confidence_escalation(",
            "execute_contradiction_detection(",
            "compare_live_claims(",
            "resolve_contradiction(",
            "write_contradiction_record(",
            "score_source_credibility(",
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
