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
    ResearchGapDiscoveryPlan,
    build_research_confidence_engine,
    build_research_gap_discovery,
    research_gap_entries_for_confidence,
    research_gap_entries_for_status,
    research_gap_entry_by_id,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Research Gap Discovery",)


class ResearchGapDiscoveryTests(unittest.TestCase):
    def test_plan_builds_advisory_gap_metadata(self) -> None:
        plan = build_research_gap_discovery(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "research_gap_discovery")
        self.assertEqual(plan.serialization_version, "research_gap_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.research_confidence_role, "research_confidence_engine")
        self.assertEqual(plan.confidence_entry_count, 5)
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
        self.assertFalse(plan.discovered_research_gap_ids)
        self.assertFalse(plan.analyzed_live_gap_ids)
        self.assertFalse(plan.written_gap_record_ids)
        self.assertFalse(plan.created_research_task_ids)
        self.assertFalse(plan.generated_recommendation_ids)
        self.assertFalse(plan.mutated_research_plan_ids)
        self.assertEqual(plan.overall_gap_discovery_score, 840)
        self.assertEqual(plan.overall_gap_discovery_posture, "guarded")
        self.assertIn(
            "does not execute research gap discovery",
            plan.authority_boundary,
        )
        self.assertTrue(plan.research_gap_discovery_capability_implemented)
        self.assertTrue(plan.research_gap_discovery_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.research_confidence_metadata_used)
        self.assertFalse(plan.research_gap_discovery_execution_implemented)
        self.assertFalse(plan.live_gap_analysis_implemented)
        self.assertFalse(plan.gap_record_write_implemented)
        self.assertFalse(plan.research_task_creation_implemented)
        self.assertFalse(plan.research_recommendation_generation_implemented)
        self.assertFalse(plan.research_plan_mutation_implemented)
        self.assertFalse(plan.research_confidence_scoring_execution_implemented)
        self.assertFalse(plan.confidence_record_write_implemented)
        self.assertFalse(plan.contradiction_detection_execution_implemented)
        self.assertFalse(plan.external_source_fetch_implemented)
        self.assertFalse(plan.retrieval_execution_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_entries_score_gap_posture_without_execution(self) -> None:
        plan = build_research_gap_discovery(route="generate")
        confidence_ids = set(plan.confidence_entry_ids)

        for entry in plan.entries:
            self.assertEqual(entry.serialization_version, "research_gap_entry.v1")
            self.assertEqual(entry.route_name, RouteName.GENERATE)
            self.assertEqual(
                entry.entry_id,
                f"research_gap_discovery::{entry.gap_kind}",
            )
            self.assertTrue(set(entry.confidence_entry_ids).issubset(confidence_ids))
            self.assertEqual(entry.source_count, plan.source_count)
            self.assertEqual(entry.domain_count, plan.domain_count)
            self.assertEqual(
                entry.gap_discovery_score,
                min(
                    1000,
                    max(
                        0,
                        entry.coverage_gap_score * 3
                        + entry.evidence_gap_score * 2
                        + entry.methodology_gap_score * 2
                        + entry.source_diversity_gap_score * 2
                        + entry.governance_alignment_score * 2
                        + entry.mutation_risk_score
                        + entry.governance_weight,
                    ),
                ),
            )
            self.assertIn("research_gap_discovery", entry.context_tags)
            self.assertIn(
                "research_gap_discovery_execution",
                entry.blocked_runtime_behaviors,
            )
            self.assertIn("research_task_creation", entry.blocked_runtime_behaviors)
            self.assertTrue(entry.hitl_required_before_gap_discovery)
            self.assertTrue(entry.research_gap_discovery_capability_implemented)
            self.assertTrue(entry.research_gap_discovery_metadata_implemented)
            self.assertTrue(entry.research_confidence_metadata_used)
            self.assertFalse(entry.research_gap_discovery_execution_implemented)
            self.assertFalse(entry.live_gap_analysis_implemented)
            self.assertFalse(entry.gap_record_write_implemented)
            self.assertFalse(entry.research_task_creation_implemented)
            self.assertFalse(entry.research_recommendation_generation_implemented)
            self.assertFalse(entry.research_plan_mutation_implemented)
            self.assertFalse(entry.research_confidence_scoring_execution_implemented)
            self.assertFalse(entry.retrieval_execution_implemented)
            self.assertFalse(entry.provider_execution_implemented)
            self.assertFalse(entry.generated_output_mutation_implemented)
            self.assertFalse(entry.runtime_evolution_implemented)
            self.assertTrue(entry.advisory_only)

        coverage = research_gap_entry_by_id(
            "research_gap_discovery::coverage_gap_review",
            plan,
        )
        self.assertIsNotNone(coverage)
        assert coverage is not None
        self.assertEqual(coverage.status, "guarded")
        self.assertEqual(coverage.confidence, "guarded")
        self.assertEqual(
            len(research_gap_entries_for_status("review_required", plan)),
            1,
        )
        self.assertEqual(
            len(research_gap_entries_for_confidence("high", plan)),
            1,
        )

    def test_plan_rejects_mismatched_gap_metadata(self) -> None:
        plan = build_research_gap_discovery()
        payload = plan.model_dump(mode="json")
        payload["entry_ids"] = ("missing",) + tuple(payload["entry_ids"][1:])

        with self.assertRaisesRegex(ValueError, "entry_ids must match"):
            ResearchGapDiscoveryPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_gap_discovery_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_gap_discovery_score must match",
        ):
            ResearchGapDiscoveryPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["discovered_research_gap_ids"] = (plan.entry_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "discovered_research_gap_ids must remain empty",
        ):
            ResearchGapDiscoveryPlan(**payload)

    def test_gap_discovery_composes_with_confidence_metadata(self) -> None:
        confidence = build_research_confidence_engine(route=RouteName.REVIEW)
        plan = build_research_gap_discovery(
            route=RouteName.REVIEW,
            confidence=confidence,
        )

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(plan.checked_at, confidence.checked_at)
        self.assertEqual(plan.confidence_entry_ids, confidence.entry_ids)
        self.assertTrue(plan.research_confidence_metadata_used)
        self.assertFalse(plan.research_gap_discovery_execution_implemented)

    def test_gap_discovery_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Plan gap discovery for generative installation research.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_research_gap_discovery(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_gap_discovery_does_not_declare_runtime_mutation_terms(self) -> None:
        plan = build_research_gap_discovery(route=RouteName.GENERATE)
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
                        entry.gap_kind,
                        entry.status,
                        entry.confidence,
                        entry.gap_axis,
                        *entry.confidence_entry_ids,
                        entry.gap_summary,
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
            "discover_research_gaps(",
            "execute_gap_discovery(",
            "analyze_live_gaps(",
            "write_gap_record(",
            "create_research_task(",
            "generate_research_recommendation(",
            "mutate_research_plan(",
            "score_research_confidence(",
            "write_confidence_record(",
            "execute_contradiction_detection(",
            "compare_live_claims(",
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
