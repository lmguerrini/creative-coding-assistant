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
    ResearchHITLPoliciesPlan,
    build_research_execution_policy,
    build_research_hitl_policies,
    research_hitl_policy_entries_for_confidence,
    research_hitl_policy_entries_for_status,
    research_hitl_policy_entry_by_id,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Research HITL Policies",)


class ResearchHITLPoliciesTests(unittest.TestCase):
    def test_plan_builds_advisory_hitl_policy_metadata(self) -> None:
        plan = build_research_hitl_policies(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "research_hitl_policies")
        self.assertEqual(
            plan.serialization_version,
            "research_hitl_policy_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.research_execution_policy_role,
            "research_execution_policy",
        )
        self.assertEqual(plan.execution_policy_entry_count, 5)
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
        self.assertFalse(plan.emitted_hitl_request_ids)
        self.assertFalse(plan.applied_hitl_decision_ids)
        self.assertFalse(plan.executed_hitl_gate_ids)
        self.assertFalse(plan.applied_execution_policy_ids)
        self.assertFalse(plan.authorized_research_execution_ids)
        self.assertFalse(plan.mutated_workflow_ids)
        self.assertEqual(plan.overall_hitl_policy_score, 859)
        self.assertEqual(plan.overall_hitl_policy_posture, "guarded")
        self.assertIn("does not emit HITL requests", plan.authority_boundary)
        self.assertTrue(plan.research_hitl_policies_capability_implemented)
        self.assertTrue(plan.research_hitl_policies_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.research_execution_policy_metadata_used)
        self.assertFalse(plan.hitl_request_emission_implemented)
        self.assertFalse(plan.hitl_decision_application_implemented)
        self.assertFalse(plan.hitl_gate_execution_implemented)
        self.assertFalse(plan.research_execution_policy_application_implemented)
        self.assertFalse(plan.research_execution_authorization_implemented)
        self.assertFalse(plan.research_execution_implemented)
        self.assertFalse(plan.recommendation_execution_implemented)
        self.assertFalse(plan.research_task_creation_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.external_source_fetch_implemented)
        self.assertFalse(plan.retrieval_execution_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_entries_score_hitl_posture_without_emission(self) -> None:
        plan = build_research_hitl_policies(route="generate")
        execution_policy_ids = set(plan.execution_policy_entry_ids)

        for entry in plan.entries:
            self.assertEqual(
                entry.serialization_version,
                "research_hitl_policy_entry.v1",
            )
            self.assertEqual(entry.route_name, RouteName.GENERATE)
            self.assertEqual(
                entry.entry_id,
                f"research_hitl_policies::{entry.hitl_policy_kind}",
            )
            self.assertTrue(
                set(entry.execution_policy_entry_ids).issubset(
                    execution_policy_ids
                )
            )
            self.assertEqual(entry.source_count, plan.source_count)
            self.assertEqual(entry.domain_count, plan.domain_count)
            self.assertEqual(
                entry.hitl_policy_score,
                min(
                    1000,
                    max(
                        0,
                        entry.approval_threshold_score * 3
                        + entry.source_review_score * 2
                        + entry.mutation_review_score * 3
                        + entry.recommendation_review_score * 2
                        + entry.governance_alignment_score * 2
                        + entry.mutation_risk_score
                        + entry.governance_weight,
                    ),
                ),
            )
            self.assertIn("research_hitl_policies", entry.context_tags)
            self.assertIn("hitl_request_emission", entry.blocked_runtime_behaviors)
            self.assertIn("hitl_decision_application", entry.blocked_runtime_behaviors)
            self.assertTrue(entry.hitl_required_before_policy_activation)
            self.assertTrue(entry.research_hitl_policies_capability_implemented)
            self.assertTrue(entry.research_hitl_policies_metadata_implemented)
            self.assertTrue(entry.research_execution_policy_metadata_used)
            self.assertFalse(entry.hitl_request_emission_implemented)
            self.assertFalse(entry.hitl_decision_application_implemented)
            self.assertFalse(entry.hitl_gate_execution_implemented)
            self.assertFalse(entry.research_execution_policy_application_implemented)
            self.assertFalse(entry.research_execution_authorization_implemented)
            self.assertFalse(entry.recommendation_execution_implemented)
            self.assertFalse(entry.workflow_control_implemented)
            self.assertFalse(entry.retrieval_execution_implemented)
            self.assertFalse(entry.provider_execution_implemented)
            self.assertFalse(entry.generated_output_mutation_implemented)
            self.assertFalse(entry.runtime_evolution_implemented)
            self.assertTrue(entry.advisory_only)

        approval = research_hitl_policy_entry_by_id(
            "research_hitl_policies::execution_approval_policy",
            plan,
        )
        self.assertIsNotNone(approval)
        assert approval is not None
        self.assertEqual(approval.status, "guarded")
        self.assertEqual(approval.confidence, "guarded")
        self.assertEqual(
            len(research_hitl_policy_entries_for_status("review_required", plan)),
            1,
        )
        self.assertEqual(
            len(research_hitl_policy_entries_for_confidence("high", plan)),
            1,
        )

    def test_plan_rejects_mismatched_hitl_policy_metadata(self) -> None:
        plan = build_research_hitl_policies()
        payload = plan.model_dump(mode="json")
        payload["entry_ids"] = ("missing",) + tuple(payload["entry_ids"][1:])

        with self.assertRaisesRegex(ValueError, "entry_ids must match"):
            ResearchHITLPoliciesPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_hitl_policy_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_hitl_policy_score must match",
        ):
            ResearchHITLPoliciesPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["emitted_hitl_request_ids"] = (plan.entry_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "emitted_hitl_request_ids must remain empty",
        ):
            ResearchHITLPoliciesPlan(**payload)

    def test_hitl_policies_compose_with_execution_policy_metadata(self) -> None:
        execution_policy = build_research_execution_policy(route=RouteName.REVIEW)
        plan = build_research_hitl_policies(
            route=RouteName.REVIEW,
            execution_policy=execution_policy,
        )

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(plan.checked_at, execution_policy.checked_at)
        self.assertEqual(
            plan.execution_policy_entry_ids,
            execution_policy.entry_ids,
        )
        self.assertTrue(plan.research_execution_policy_metadata_used)
        self.assertFalse(plan.hitl_request_emission_implemented)

    def test_hitl_policies_preserve_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Plan HITL review posture for research execution.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.THREE_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_research_hitl_policies(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_hitl_policies_do_not_declare_runtime_mutation_terms(self) -> None:
        plan = build_research_hitl_policies(route=RouteName.GENERATE)
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
                        entry.hitl_policy_kind,
                        entry.status,
                        entry.confidence,
                        entry.hitl_axis,
                        *entry.execution_policy_entry_ids,
                        entry.hitl_summary,
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
            "emit_hitl_request(",
            "apply_hitl_decision(",
            "execute_hitl_gate(",
            "apply_research_execution_policy(",
            "authorize_research_execution(",
            "execute_research(",
            "execute_recommendation(",
            "generate_research_recommendation(",
            "create_research_task(",
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
