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
    CreativeResearchPlan,
    build_creative_research_engine,
    build_research_hitl_policies,
    creative_research_entries_for_confidence,
    creative_research_entries_for_status,
    creative_research_entry_by_id,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Creative Research Engine",)


class CreativeResearchEngineTests(unittest.TestCase):
    def test_plan_builds_advisory_creative_research_metadata(self) -> None:
        plan = build_creative_research_engine(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "creative_research_engine")
        self.assertEqual(
            plan.serialization_version,
            "creative_research_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.research_hitl_policy_role,
            "research_hitl_policies",
        )
        self.assertEqual(plan.hitl_policy_entry_count, 5)
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
        self.assertFalse(plan.created_creative_output_ids)
        self.assertFalse(plan.mutated_generated_output_ids)
        self.assertFalse(plan.discovered_inspiration_ids)
        self.assertFalse(plan.generated_recommendation_ids)
        self.assertFalse(plan.executed_research_ids)
        self.assertFalse(plan.emitted_hitl_request_ids)
        self.assertEqual(plan.highest_creative_research_score, 988)
        self.assertEqual(plan.overall_creative_research_score, 894)
        self.assertEqual(plan.overall_creative_research_posture, "guarded")
        self.assertIn("does not generate creative outputs", plan.authority_boundary)
        self.assertIn(
            "does not discover cross-domain inspiration",
            plan.authority_boundary,
        )
        self.assertTrue(plan.creative_research_engine_capability_implemented)
        self.assertTrue(plan.creative_research_engine_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.research_hitl_policy_metadata_used)
        self.assertFalse(plan.creative_output_generation_implemented)
        self.assertFalse(plan.creative_output_mutation_implemented)
        self.assertFalse(plan.cross_domain_inspiration_discovery_implemented)
        self.assertFalse(plan.inspiration_discovery_execution_implemented)
        self.assertFalse(plan.hitl_request_emission_implemented)
        self.assertFalse(plan.research_execution_implemented)
        self.assertFalse(plan.research_plan_mutation_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.external_source_fetch_implemented)
        self.assertFalse(plan.retrieval_execution_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_entries_score_creative_research_posture_without_execution(self) -> None:
        plan = build_creative_research_engine(route="generate")
        hitl_policy_ids = set(plan.hitl_policy_entry_ids)

        for entry in plan.entries:
            self.assertEqual(
                entry.serialization_version,
                "creative_research_entry.v1",
            )
            self.assertEqual(entry.route_name, RouteName.GENERATE)
            self.assertEqual(
                entry.entry_id,
                f"creative_research_engine::{entry.creative_research_kind}",
            )
            self.assertTrue(
                set(entry.hitl_policy_entry_ids).issubset(hitl_policy_ids)
            )
            self.assertEqual(entry.source_count, plan.source_count)
            self.assertEqual(entry.domain_count, plan.domain_count)
            self.assertEqual(
                entry.creative_research_score,
                min(
                    1000,
                    max(
                        0,
                        entry.question_framing_score * 2
                        + entry.source_grounding_score * 2
                        + entry.creative_novelty_score * 2
                        + entry.constraint_alignment_score * 2
                        + entry.provenance_traceability_score * 2
                        + entry.hitl_policy_alignment_score * 2
                        + entry.mutation_risk_score
                        + entry.governance_weight,
                    ),
                ),
            )
            self.assertIn("creative_research_engine", entry.context_tags)
            self.assertIn(
                "creative_output_generation",
                entry.blocked_runtime_behaviors,
            )
            self.assertIn(
                "cross_domain_inspiration_discovery",
                entry.blocked_runtime_behaviors,
            )
            self.assertTrue(entry.hitl_required_before_creative_research_execution)
            self.assertTrue(entry.creative_research_engine_capability_implemented)
            self.assertTrue(entry.creative_research_engine_metadata_implemented)
            self.assertTrue(entry.research_hitl_policy_metadata_used)
            self.assertFalse(entry.creative_output_generation_implemented)
            self.assertFalse(entry.creative_output_mutation_implemented)
            self.assertFalse(entry.cross_domain_inspiration_discovery_implemented)
            self.assertFalse(entry.inspiration_discovery_execution_implemented)
            self.assertFalse(entry.research_recommendation_generation_implemented)
            self.assertFalse(entry.hitl_request_emission_implemented)
            self.assertFalse(entry.research_execution_implemented)
            self.assertFalse(entry.workflow_control_implemented)
            self.assertFalse(entry.retrieval_execution_implemented)
            self.assertFalse(entry.provider_execution_implemented)
            self.assertFalse(entry.generated_output_mutation_implemented)
            self.assertFalse(entry.runtime_evolution_implemented)
            self.assertTrue(entry.advisory_only)

        framing = creative_research_entry_by_id(
            "creative_research_engine::creative_question_framing",
            plan,
        )
        self.assertIsNotNone(framing)
        assert framing is not None
        self.assertEqual(framing.status, "guarded")
        self.assertEqual(framing.confidence, "guarded")
        self.assertEqual(
            len(creative_research_entries_for_status("review_required", plan)),
            1,
        )
        self.assertEqual(
            len(creative_research_entries_for_confidence("high", plan)),
            1,
        )

    def test_plan_rejects_mismatched_creative_research_metadata(self) -> None:
        plan = build_creative_research_engine()
        payload = plan.model_dump(mode="json")
        payload["entry_ids"] = ("missing",) + tuple(payload["entry_ids"][1:])

        with self.assertRaisesRegex(ValueError, "entry_ids must match"):
            CreativeResearchPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_creative_research_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_creative_research_score must match",
        ):
            CreativeResearchPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["discovered_inspiration_ids"] = (plan.entry_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "discovered_inspiration_ids must remain empty",
        ):
            CreativeResearchPlan(**payload)

    def test_creative_research_composes_with_hitl_policy_metadata(self) -> None:
        hitl_policy = build_research_hitl_policies(route=RouteName.REVIEW)
        plan = build_creative_research_engine(
            route=RouteName.REVIEW,
            hitl_policy=hitl_policy,
        )

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(plan.checked_at, hitl_policy.checked_at)
        self.assertEqual(
            plan.hitl_policy_entry_ids,
            hitl_policy.entry_ids,
        )
        self.assertTrue(plan.research_hitl_policy_metadata_used)
        self.assertFalse(plan.cross_domain_inspiration_discovery_implemented)

    def test_creative_research_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Frame creative research for a shader study.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.THREE_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_creative_research_engine(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_creative_research_does_not_declare_runtime_mutation_terms(self) -> None:
        plan = build_creative_research_engine(route=RouteName.GENERATE)
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
                        entry.creative_research_kind,
                        entry.status,
                        entry.confidence,
                        entry.creative_axis,
                        *entry.hitl_policy_entry_ids,
                        entry.creative_research_summary,
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
            "generate_creative_output(",
            "mutate_generated_output(",
            "generate_creative_asset(",
            "write_creative_asset(",
            "generate_prototype(",
            "execute_prototype(",
            "discover_cross_domain_inspiration(",
            "execute_inspiration_discovery(",
            "write_inspiration_record(",
            "generate_research_recommendation(",
            "execute_recommendation(",
            "emit_hitl_request(",
            "apply_hitl_decision(",
            "execute_hitl_gate(",
            "apply_research_execution_policy(",
            "authorize_research_execution(",
            "execute_research(",
            "mutate_research_plan(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "execute_workflow(",
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
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
