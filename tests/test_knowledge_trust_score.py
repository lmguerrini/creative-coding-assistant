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
    KnowledgeTrustScorePlan,
    build_knowledge_freshness_tracking,
    build_knowledge_trust_score,
    knowledge_trust_signal_by_id,
    knowledge_trust_signals_for_confidence,
    knowledge_trust_signals_for_status,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_ROADMAP_ITEMS = ("Knowledge Trust Score",)
REQUIRED_SIGNAL_FIELDS = {
    "signal_id",
    "signal_kind",
    "status",
    "confidence",
    "route_name",
    "task_type",
    "execution_mode_id",
    "trust_axis",
    "knowledge_freshness_signal_ids",
    "knowledge_freshness_signal_count",
    "source_count",
    "domain_count",
    "trust_signal_summary",
    "trust_signal_score",
    "freshness_alignment_score",
    "reliability_alignment_score",
    "governance_alignment_score",
    "mutation_risk_score",
    "governance_weight",
    "trust_score",
    "hitl_required_before_trust_action",
    "context_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "knowledge_trust_score_capability_implemented",
    "knowledge_trust_metadata_implemented",
    "knowledge_freshness_metadata_used",
    "knowledge_trust_score_execution_implemented",
    "trust_score_computation_implemented",
    "trust_score_record_write_implemented",
    "trust_threshold_enforcement_implemented",
    "source_trust_mutation_implemented",
    "source_reliability_score_mutation_implemented",
    "knowledge_freshness_tracking_execution_implemented",
    "freshness_scan_execution_implemented",
    "freshness_score_computation_implemented",
    "freshness_record_write_implemented",
    "source_timestamp_update_implemented",
    "staleness_state_mutation_implemented",
    "source_fetch_execution_implemented",
    "kb_storage_write_implemented",
    "source_record_update_implemented",
    "retrieval_query_execution_implemented",
    "retrieval_configuration_mutation_implemented",
    "ranking_mutation_implemented",
    "embedding_request_execution_implemented",
    "embedding_refresh_execution_implemented",
    "vector_indexing_implemented",
    "vector_upsert_implemented",
    "documentation_fetch_execution_implemented",
    "provider_provisioning_implemented",
    "api_key_inference_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class KnowledgeTrustScoreTests(unittest.TestCase):
    def test_plan_builds_knowledge_trust_metadata(self) -> None:
        plan = build_knowledge_trust_score(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "knowledge_trust_score")
        self.assertEqual(plan.serialization_version, "knowledge_trust_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.knowledge_freshness_role, "knowledge_freshness_tracking")
        self.assertEqual(
            plan.knowledge_freshness_serialization_version,
            "knowledge_freshness_plan.v1",
        )
        self.assertEqual(len(plan.knowledge_freshness_signal_ids), 5)
        self.assertEqual(plan.knowledge_freshness_signal_count, 5)
        self.assertEqual(plan.covered_roadmap_items, EXPECTED_ROADMAP_ITEMS)
        self.assertEqual(plan.covered_roadmap_item_count, 1)
        self.assertEqual(plan.source_count, 57)
        self.assertEqual(plan.domain_count, 43)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.signal_count, 5)
        self.assertEqual(plan.candidate_signal_count, 1)
        self.assertEqual(plan.review_required_signal_count, 3)
        self.assertEqual(plan.guarded_signal_count, 1)
        self.assertEqual(plan.high_confidence_signal_count, 3)
        self.assertEqual(plan.hitl_required_signal_count, 5)
        self.assertFalse(plan.planned_trust_score_ids)
        self.assertFalse(plan.computed_trust_score_ids)
        self.assertFalse(plan.written_trust_score_record_ids)
        self.assertFalse(plan.enforced_trust_threshold_ids)
        self.assertFalse(plan.mutated_source_trust_ids)
        self.assertFalse(plan.mutated_source_reliability_score_ids)
        self.assertFalse(plan.executed_freshness_scan_ids)
        self.assertFalse(plan.fetched_source_ids)
        self.assertFalse(plan.written_kb_record_ids)
        self.assertEqual(plan.overall_trust_posture, "guarded")
        self.assertIn("does not execute trust scoring", plan.authority_boundary)
        self.assertIn("write KB storage", plan.authority_boundary)
        self.assertTrue(plan.knowledge_trust_score_capability_implemented)
        self.assertTrue(plan.knowledge_trust_metadata_implemented)
        self.assertTrue(plan.roadmap_item_covered)
        self.assertTrue(plan.knowledge_freshness_metadata_used)
        self.assertFalse(plan.knowledge_trust_score_execution_implemented)
        self.assertFalse(plan.trust_score_computation_implemented)
        self.assertFalse(plan.trust_score_record_write_implemented)
        self.assertFalse(plan.trust_threshold_enforcement_implemented)
        self.assertFalse(plan.source_trust_mutation_implemented)
        self.assertFalse(plan.source_reliability_score_mutation_implemented)
        self.assertFalse(plan.knowledge_freshness_tracking_execution_implemented)
        self.assertFalse(plan.freshness_scan_execution_implemented)
        self.assertFalse(plan.freshness_score_computation_implemented)
        self.assertFalse(plan.freshness_record_write_implemented)
        self.assertFalse(plan.source_fetch_execution_implemented)
        self.assertFalse(plan.kb_storage_write_implemented)
        self.assertFalse(plan.source_record_update_implemented)
        self.assertFalse(plan.retrieval_query_execution_implemented)
        self.assertFalse(plan.retrieval_configuration_mutation_implemented)
        self.assertFalse(plan.ranking_mutation_implemented)
        self.assertFalse(plan.embedding_request_execution_implemented)
        self.assertFalse(plan.embedding_refresh_execution_implemented)
        self.assertFalse(plan.vector_indexing_implemented)
        self.assertFalse(plan.vector_upsert_implemented)
        self.assertFalse(plan.documentation_fetch_execution_implemented)
        self.assertFalse(plan.provider_provisioning_implemented)
        self.assertFalse(plan.api_key_inference_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_signals_score_knowledge_trust_without_execution(self) -> None:
        plan = build_knowledge_trust_score(route="generate")
        freshness_signal_ids = set(plan.knowledge_freshness_signal_ids)

        for signal in plan.signals:
            dumped = signal.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_SIGNAL_FIELDS)
            self.assertEqual(signal.serialization_version, "knowledge_trust_entry.v1")
            self.assertEqual(signal.route_name, RouteName.GENERATE)
            self.assertEqual(
                signal.signal_id,
                f"knowledge_trust_score::{signal.signal_kind}",
            )
            self.assertEqual(
                signal.knowledge_freshness_signal_count,
                len(signal.knowledge_freshness_signal_ids),
            )
            self.assertTrue(
                set(signal.knowledge_freshness_signal_ids).issubset(
                    freshness_signal_ids
                )
            )
            self.assertEqual(
                signal.trust_score,
                min(
                    1000,
                    max(
                        0,
                        signal.trust_signal_score * 3
                        + signal.freshness_alignment_score * 2
                        + signal.reliability_alignment_score * 2
                        + signal.governance_alignment_score * 2
                        + signal.mutation_risk_score
                        + signal.governance_weight,
                    ),
                ),
            )
            self.assertIn("knowledge_trust_score", signal.context_tags)
            self.assertIn(
                "knowledge_trust_score_execution",
                signal.blocked_runtime_behaviors,
            )
            self.assertIn("trust_score_computation", signal.blocked_runtime_behaviors)
            self.assertIn("source_trust_mutation", signal.blocked_runtime_behaviors)
            self.assertIn("kb_storage_write", signal.blocked_runtime_behaviors)
            self.assertTrue(signal.hitl_required_before_trust_action)
            self.assertTrue(signal.knowledge_trust_score_capability_implemented)
            self.assertTrue(signal.knowledge_trust_metadata_implemented)
            self.assertTrue(signal.knowledge_freshness_metadata_used)
            self.assertFalse(signal.knowledge_trust_score_execution_implemented)
            self.assertFalse(signal.trust_score_computation_implemented)
            self.assertFalse(signal.trust_score_record_write_implemented)
            self.assertFalse(signal.trust_threshold_enforcement_implemented)
            self.assertFalse(signal.source_trust_mutation_implemented)
            self.assertFalse(signal.source_reliability_score_mutation_implemented)
            self.assertFalse(
                signal.knowledge_freshness_tracking_execution_implemented
            )
            self.assertFalse(signal.freshness_scan_execution_implemented)
            self.assertFalse(signal.freshness_score_computation_implemented)
            self.assertFalse(signal.freshness_record_write_implemented)
            self.assertFalse(signal.source_timestamp_update_implemented)
            self.assertFalse(signal.staleness_state_mutation_implemented)
            self.assertFalse(signal.source_fetch_execution_implemented)
            self.assertFalse(signal.kb_storage_write_implemented)
            self.assertFalse(signal.source_record_update_implemented)
            self.assertFalse(signal.retrieval_query_execution_implemented)
            self.assertFalse(signal.retrieval_configuration_mutation_implemented)
            self.assertFalse(signal.ranking_mutation_implemented)
            self.assertFalse(signal.embedding_request_execution_implemented)
            self.assertFalse(signal.embedding_refresh_execution_implemented)
            self.assertFalse(signal.vector_indexing_implemented)
            self.assertFalse(signal.vector_upsert_implemented)
            self.assertFalse(signal.documentation_fetch_execution_implemented)
            self.assertFalse(signal.provider_provisioning_implemented)
            self.assertFalse(signal.api_key_inference_implemented)
            self.assertFalse(signal.provider_model_routing_implemented)
            self.assertFalse(signal.provider_execution_implemented)
            self.assertFalse(signal.workflow_control_implemented)
            self.assertFalse(signal.generated_output_mutation_implemented)
            self.assertFalse(signal.runtime_evolution_implemented)
            self.assertTrue(signal.advisory_only)

        inventory = knowledge_trust_signal_by_id(
            "knowledge_trust_score::knowledge_trust_inventory_review",
            plan,
        )
        self.assertIsNotNone(inventory)
        assert inventory is not None
        self.assertEqual(inventory.status, "guarded")
        self.assertEqual(inventory.confidence, "guarded")
        self.assertEqual(
            len(knowledge_trust_signals_for_status("review_required", plan)),
            3,
        )
        self.assertEqual(
            len(knowledge_trust_signals_for_confidence("high", plan)),
            2,
        )

    def test_plan_rejects_mismatched_knowledge_trust_metadata(self) -> None:
        plan = build_knowledge_trust_score()
        payload = plan.model_dump(mode="json")
        payload["signal_ids"] = ("missing",) + tuple(payload["signal_ids"][1:])

        with self.assertRaisesRegex(ValueError, "signal_ids must match"):
            KnowledgeTrustScorePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_trust_score"] -= 1

        with self.assertRaisesRegex(ValueError, "overall_trust_score must match"):
            KnowledgeTrustScorePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["computed_trust_score_ids"] = (plan.signal_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "computed_trust_score_ids must remain empty",
        ):
            KnowledgeTrustScorePlan(**payload)

    def test_knowledge_trust_composes_task_19_metadata(self) -> None:
        freshness_plan = build_knowledge_freshness_tracking(route=RouteName.REVIEW)
        plan = build_knowledge_trust_score(
            route=RouteName.REVIEW,
            knowledge_freshness=freshness_plan,
        )

        self.assertEqual(plan.route_name, RouteName.REVIEW)
        self.assertEqual(plan.knowledge_freshness_signal_ids, freshness_plan.signal_ids)
        self.assertEqual(plan.source_count, freshness_plan.source_count)
        self.assertEqual(plan.domain_count, freshness_plan.domain_count)
        for signal in plan.signals:
            self.assertTrue(
                set(signal.knowledge_freshness_signal_ids).issubset(
                    set(freshness_plan.signal_ids)
                )
            )
            self.assertFalse(signal.trust_score_computation_implemented)
            self.assertFalse(signal.trust_score_record_write_implemented)
            self.assertFalse(signal.source_trust_mutation_implemented)

    def test_knowledge_trust_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review knowledge trust posture for p5.js examples.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_knowledge_trust_score(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_knowledge_trust_omits_runtime_mutation_calls(self) -> None:
        plan = build_knowledge_trust_score(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *plan.covered_roadmap_items,
                *(
                    field
                    for signal in plan.signals
                    for field in (
                        signal.signal_id,
                        signal.signal_kind,
                        signal.status,
                        signal.confidence,
                        signal.trust_axis,
                        *signal.knowledge_freshness_signal_ids,
                        signal.trust_signal_summary,
                        *signal.context_tags,
                        *signal.explainability_notes,
                        *signal.advisory_actions,
                        *signal.evidence,
                        *signal.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "execute_trust_score(",
            "compute_trust_score(",
            "write_trust_record(",
            "enforce_trust_threshold(",
            "mutate_source_trust(",
            "mutate_source_reliability_score(",
            "execute_freshness_tracking(",
            "execute_freshness_scan(",
            "compute_freshness_score(",
            "write_freshness_record(",
            "update_source_timestamp(",
            "mutate_staleness_state(",
            "fetch_source(",
            "write_kb_storage(",
            "update_source_record(",
            "execute_retrieval_query(",
            "mutate_retrieval_config(",
            "mutate_ranking(",
            "request_embedding(",
            "refresh_embedding(",
            "index_vectors(",
            "upsert_vectors(",
            "fetch_documentation(",
            "provision_provider(",
            "infer_api_key(",
            "route_provider(",
            "execute_provider(",
            "invoke_agent(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "execute_workflow(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
