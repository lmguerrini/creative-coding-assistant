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
    CrossDomainInspirationPlan,
    build_creative_research_engine,
    build_cross_domain_inspiration_discovery,
    cross_domain_inspiration_entries_for_confidence,
    cross_domain_inspiration_entries_for_status,
    cross_domain_inspiration_entry_by_id,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Cross-domain Inspiration Discovery",)


class CrossDomainInspirationDiscoveryTests(unittest.TestCase):
    def test_plan_builds_advisory_inspiration_metadata(self) -> None:
        plan = build_cross_domain_inspiration_discovery(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "cross_domain_inspiration_discovery")
        self.assertEqual(
            plan.serialization_version,
            "cross_domain_inspiration_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.creative_research_role, "creative_research_engine")
        self.assertEqual(plan.creative_research_entry_count, 5)
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
        self.assertFalse(plan.executed_inspiration_discovery_ids)
        self.assertFalse(plan.live_cross_domain_search_ids)
        self.assertFalse(plan.written_inspiration_record_ids)
        self.assertFalse(plan.created_creative_output_ids)
        self.assertFalse(plan.mutated_generated_output_ids)
        self.assertFalse(plan.executed_research_ids)
        self.assertFalse(plan.emitted_hitl_request_ids)
        # Deterministic scores come from the fixed entry inputs and the
        # inspiration formula asserted below; the domain-analogy entry is 991.
        self.assertEqual(plan.highest_inspiration_score, 991)
        self.assertEqual(plan.overall_inspiration_score, 901)
        self.assertEqual(plan.overall_inspiration_posture, "guarded")
        self.assertIn("does not execute inspiration discovery", plan.authority_boundary)
        self.assertIn("perform live cross-domain search", plan.authority_boundary)
        self.assertTrue(
            plan.cross_domain_inspiration_discovery_capability_implemented
        )
        self.assertTrue(
            plan.cross_domain_inspiration_discovery_metadata_implemented
        )
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.creative_research_metadata_used)
        self.assertFalse(plan.inspiration_discovery_execution_implemented)
        self.assertFalse(plan.live_cross_domain_search_implemented)
        self.assertFalse(plan.external_source_fetch_implemented)
        self.assertFalse(plan.creative_output_generation_implemented)
        self.assertFalse(plan.creative_output_mutation_implemented)
        self.assertFalse(plan.inspiration_record_write_implemented)
        self.assertFalse(plan.research_recommendation_generation_implemented)
        self.assertFalse(plan.hitl_request_emission_implemented)
        self.assertFalse(plan.research_execution_implemented)
        self.assertFalse(plan.research_plan_mutation_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.retrieval_execution_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_entries_score_inspiration_posture_without_execution(self) -> None:
        plan = build_cross_domain_inspiration_discovery(route="generate")
        creative_research_ids = set(plan.creative_research_entry_ids)

        for entry in plan.entries:
            self.assertEqual(
                entry.serialization_version,
                "cross_domain_inspiration_entry.v1",
            )
            self.assertEqual(entry.route_name, RouteName.GENERATE)
            self.assertEqual(
                entry.entry_id,
                f"cross_domain_inspiration::{entry.inspiration_kind}",
            )
            self.assertTrue(
                set(entry.creative_research_entry_ids).issubset(
                    creative_research_ids
                )
            )
            self.assertEqual(entry.source_count, plan.source_count)
            self.assertEqual(entry.domain_count, plan.domain_count)
            self.assertEqual(
                entry.inspiration_score,
                min(
                    1000,
                    max(
                        0,
                        entry.source_domain_distance_score * 2
                        + entry.analogy_quality_score * 2
                        + entry.transferability_score * 2
                        + entry.provenance_traceability_score * 2
                        + entry.creative_research_alignment_score * 2
                        + entry.hitl_policy_alignment_score * 2
                        + entry.mutation_risk_score
                        + entry.governance_weight,
                    ),
                ),
            )
            self.assertIn(
                "cross_domain_inspiration_discovery",
                entry.context_tags,
            )
            self.assertIn(
                "inspiration_discovery_execution",
                entry.blocked_runtime_behaviors,
            )
            self.assertIn("live_cross_domain_search", entry.blocked_runtime_behaviors)
            self.assertTrue(
                entry.hitl_required_before_inspiration_discovery_execution
            )
            self.assertTrue(
                entry.cross_domain_inspiration_discovery_capability_implemented
            )
            self.assertTrue(
                entry.cross_domain_inspiration_discovery_metadata_implemented
            )
            self.assertTrue(entry.creative_research_metadata_used)
            self.assertFalse(entry.inspiration_discovery_execution_implemented)
            self.assertFalse(entry.live_cross_domain_search_implemented)
            self.assertFalse(entry.external_source_fetch_implemented)
            self.assertFalse(entry.creative_output_generation_implemented)
            self.assertFalse(entry.creative_output_mutation_implemented)
            self.assertFalse(entry.inspiration_record_write_implemented)
            self.assertFalse(entry.research_recommendation_generation_implemented)
            self.assertFalse(entry.hitl_request_emission_implemented)
            self.assertFalse(entry.research_execution_implemented)
            self.assertFalse(entry.workflow_control_implemented)
            self.assertFalse(entry.retrieval_execution_implemented)
            self.assertFalse(entry.provider_execution_implemented)
            self.assertFalse(entry.generated_output_mutation_implemented)
            self.assertFalse(entry.runtime_evolution_implemented)
            self.assertTrue(entry.advisory_only)

        analogy = cross_domain_inspiration_entry_by_id(
            "cross_domain_inspiration::domain_analogy_mapping",
            plan,
        )
        self.assertIsNotNone(analogy)
        assert analogy is not None
        self.assertEqual(analogy.status, "guarded")
        self.assertEqual(analogy.confidence, "guarded")
        self.assertEqual(
            len(cross_domain_inspiration_entries_for_status("review_required", plan)),
            1,
        )
        self.assertEqual(
            len(cross_domain_inspiration_entries_for_confidence("high", plan)),
            1,
        )

    def test_plan_rejects_mismatched_inspiration_metadata(self) -> None:
        plan = build_cross_domain_inspiration_discovery()
        payload = plan.model_dump(mode="json")
        payload["entry_ids"] = ("missing",) + tuple(payload["entry_ids"][1:])

        with self.assertRaisesRegex(ValueError, "entry_ids must match"):
            CrossDomainInspirationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_inspiration_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_inspiration_score must match",
        ):
            CrossDomainInspirationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["executed_inspiration_discovery_ids"] = (plan.entry_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "executed_inspiration_discovery_ids must remain empty",
        ):
            CrossDomainInspirationPlan(**payload)

    def test_inspiration_composes_with_creative_research_metadata(self) -> None:
        creative_research = build_creative_research_engine(route=RouteName.REVIEW)
        plan = build_cross_domain_inspiration_discovery(
            route=RouteName.REVIEW,
            creative_research=creative_research,
        )

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(plan.checked_at, creative_research.checked_at)
        self.assertEqual(
            plan.creative_research_entry_ids,
            creative_research.entry_ids,
        )
        self.assertTrue(plan.creative_research_metadata_used)
        self.assertFalse(plan.inspiration_discovery_execution_implemented)

    def test_inspiration_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Map inspiration posture between shader and textile patterns.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.THREE_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_cross_domain_inspiration_discovery(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_inspiration_does_not_declare_runtime_mutation_terms(self) -> None:
        plan = build_cross_domain_inspiration_discovery(route=RouteName.GENERATE)
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
                        entry.inspiration_kind,
                        entry.status,
                        entry.confidence,
                        entry.inspiration_axis,
                        *entry.creative_research_entry_ids,
                        entry.inspiration_summary,
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
            "execute_inspiration_discovery(",
            "perform_live_cross_domain_search(",
            "fetch_external_source(",
            "browse_web(",
            "download_paper(",
            "generate_creative_output(",
            "mutate_generated_output(",
            "generate_creative_asset(",
            "write_creative_asset(",
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
