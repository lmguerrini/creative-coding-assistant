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
    ResearchRecommendationPlan,
    build_research_gap_discovery,
    build_research_recommendation_engine,
    research_recommendation_entries_for_confidence,
    research_recommendation_entries_for_status,
    research_recommendation_entry_by_id,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Research Recommendation Engine",)


class ResearchRecommendationEngineTests(unittest.TestCase):
    def test_plan_builds_advisory_recommendation_metadata(self) -> None:
        plan = build_research_recommendation_engine(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "research_recommendation_engine")
        self.assertEqual(
            plan.serialization_version,
            "research_recommendation_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.research_gap_role, "research_gap_discovery")
        self.assertEqual(plan.gap_entry_count, 5)
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
        self.assertFalse(plan.generated_recommendation_ids)
        self.assertFalse(plan.generated_live_recommendation_ids)
        self.assertFalse(plan.executed_recommendation_ids)
        self.assertFalse(plan.written_recommendation_record_ids)
        self.assertFalse(plan.created_research_task_ids)
        self.assertFalse(plan.mutated_research_plan_ids)
        self.assertEqual(plan.overall_recommendation_score, 839)
        self.assertEqual(plan.overall_recommendation_posture, "guarded")
        self.assertIn(
            "does not generate research recommendations",
            plan.authority_boundary,
        )
        self.assertTrue(plan.research_recommendation_engine_capability_implemented)
        self.assertTrue(plan.research_recommendation_engine_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.research_gap_metadata_used)
        self.assertFalse(plan.research_recommendation_generation_implemented)
        self.assertFalse(plan.live_recommendation_generation_implemented)
        self.assertFalse(plan.recommendation_execution_implemented)
        self.assertFalse(plan.recommendation_record_write_implemented)
        self.assertFalse(plan.research_task_creation_implemented)
        self.assertFalse(plan.research_plan_mutation_implemented)
        self.assertFalse(plan.research_gap_discovery_execution_implemented)
        self.assertFalse(plan.research_confidence_scoring_execution_implemented)
        self.assertFalse(plan.external_source_fetch_implemented)
        self.assertFalse(plan.retrieval_execution_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_entries_score_recommendation_posture_without_generation(self) -> None:
        plan = build_research_recommendation_engine(route="generate")
        gap_ids = set(plan.gap_entry_ids)

        for entry in plan.entries:
            self.assertEqual(
                entry.serialization_version,
                "research_recommendation_entry.v1",
            )
            self.assertEqual(entry.route_name, RouteName.GENERATE)
            self.assertEqual(
                entry.entry_id,
                f"research_recommendation_engine::{entry.recommendation_kind}",
            )
            self.assertTrue(set(entry.gap_entry_ids).issubset(gap_ids))
            self.assertEqual(entry.source_count, plan.source_count)
            self.assertEqual(entry.domain_count, plan.domain_count)
            self.assertEqual(
                entry.recommendation_score,
                min(
                    1000,
                    max(
                        0,
                        entry.gap_priority_score * 3
                        + entry.source_followup_score * 2
                        + entry.confidence_improvement_score * 2
                        + entry.execution_readiness_score * 2
                        + entry.governance_alignment_score * 2
                        + entry.mutation_risk_score
                        + entry.governance_weight,
                    ),
                ),
            )
            self.assertIn("research_recommendation_engine", entry.context_tags)
            self.assertIn(
                "research_recommendation_generation",
                entry.blocked_runtime_behaviors,
            )
            self.assertIn("recommendation_execution", entry.blocked_runtime_behaviors)
            self.assertTrue(entry.hitl_required_before_recommendation_generation)
            self.assertTrue(
                entry.research_recommendation_engine_capability_implemented
            )
            self.assertTrue(entry.research_recommendation_engine_metadata_implemented)
            self.assertTrue(entry.research_gap_metadata_used)
            self.assertFalse(entry.research_recommendation_generation_implemented)
            self.assertFalse(entry.live_recommendation_generation_implemented)
            self.assertFalse(entry.recommendation_execution_implemented)
            self.assertFalse(entry.recommendation_record_write_implemented)
            self.assertFalse(entry.research_task_creation_implemented)
            self.assertFalse(entry.research_plan_mutation_implemented)
            self.assertFalse(entry.research_gap_discovery_execution_implemented)
            self.assertFalse(entry.retrieval_execution_implemented)
            self.assertFalse(entry.provider_execution_implemented)
            self.assertFalse(entry.generated_output_mutation_implemented)
            self.assertFalse(entry.runtime_evolution_implemented)
            self.assertTrue(entry.advisory_only)

        gap_priority = research_recommendation_entry_by_id(
            "research_recommendation_engine::gap_prioritization_review",
            plan,
        )
        self.assertIsNotNone(gap_priority)
        assert gap_priority is not None
        self.assertEqual(gap_priority.status, "guarded")
        self.assertEqual(gap_priority.confidence, "guarded")
        self.assertEqual(
            len(research_recommendation_entries_for_status("review_required", plan)),
            1,
        )
        self.assertEqual(
            len(research_recommendation_entries_for_confidence("high", plan)),
            1,
        )

    def test_plan_rejects_mismatched_recommendation_metadata(self) -> None:
        plan = build_research_recommendation_engine()
        payload = plan.model_dump(mode="json")
        payload["entry_ids"] = ("missing",) + tuple(payload["entry_ids"][1:])

        with self.assertRaisesRegex(ValueError, "entry_ids must match"):
            ResearchRecommendationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_recommendation_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_recommendation_score must match",
        ):
            ResearchRecommendationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["generated_recommendation_ids"] = (plan.entry_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "generated_recommendation_ids must remain empty",
        ):
            ResearchRecommendationPlan(**payload)

    def test_recommendation_composes_with_gap_metadata(self) -> None:
        gaps = build_research_gap_discovery(route=RouteName.REVIEW)
        plan = build_research_recommendation_engine(
            route=RouteName.REVIEW,
            gaps=gaps,
        )

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(plan.checked_at, gaps.checked_at)
        self.assertEqual(plan.gap_entry_ids, gaps.entry_ids)
        self.assertTrue(plan.research_gap_metadata_used)
        self.assertFalse(plan.research_recommendation_generation_implemented)

    def test_recommendation_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Plan recommendation posture for a shader research brief.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.THREE_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_research_recommendation_engine(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_recommendation_does_not_declare_runtime_mutation_terms(self) -> None:
        plan = build_research_recommendation_engine(route=RouteName.GENERATE)
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
                        entry.recommendation_kind,
                        entry.status,
                        entry.confidence,
                        entry.recommendation_axis,
                        *entry.gap_entry_ids,
                        entry.recommendation_summary,
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
            "generate_research_recommendation(",
            "generate_live_recommendation(",
            "execute_recommendation(",
            "write_recommendation_record(",
            "create_research_task(",
            "mutate_research_plan(",
            "discover_research_gaps(",
            "analyze_live_gaps(",
            "score_research_confidence(",
            "write_confidence_record(",
            "execute_contradiction_detection(",
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
