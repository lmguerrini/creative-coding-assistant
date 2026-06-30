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
    LongTermCreativeMemoryPlan,
    build_long_term_creative_memory,
    long_term_creative_memory_record_by_id,
    long_term_creative_memory_records_for_sensitivity,
    long_term_creative_memory_records_for_status,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_RECORD_FIELDS = {
    "record_id",
    "memory_kind",
    "status",
    "sensitivity",
    "route_name",
    "task_type",
    "execution_mode_id",
    "retention_scope",
    "source_memory_kind",
    "source_surface",
    "stability_score",
    "evidence_strength_score",
    "recency_resilience_score",
    "governance_weight",
    "memory_governance_score",
    "hitl_required_before_persistence",
    "retrieval_tags",
    "memory_summary",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "long_term_creative_memory_implemented",
    "memory_record_metadata_implemented",
    "memory_governance_metadata_implemented",
    "memory_storage_write_implemented",
    "memory_record_creation_implemented",
    "memory_record_update_implemented",
    "memory_record_deletion_implemented",
    "memory_retrieval_execution_implemented",
    "memory_consolidation_implemented",
    "preference_mutation_implemented",
    "personalization_application_implemented",
    "provider_model_routing_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_switching_implemented",
    "provider_execution_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "workflow_graph_mutation_implemented",
    "workflow_execution_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}


class LongTermCreativeMemoryTests(unittest.TestCase):
    def test_plan_builds_long_term_memory_metadata(self) -> None:
        plan = build_long_term_creative_memory(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "long_term_creative_memory")
        self.assertEqual(
            plan.serialization_version,
            "long_term_creative_memory_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.record_count, 5)
        self.assertEqual(plan.candidate_record_count, 1)
        self.assertEqual(plan.review_required_record_count, 2)
        self.assertEqual(plan.guarded_record_count, 2)
        self.assertEqual(plan.high_sensitivity_record_count, 3)
        self.assertEqual(plan.hitl_required_record_count, 5)
        self.assertFalse(plan.persisted_record_ids)
        self.assertFalse(plan.retrieved_record_ids)
        self.assertFalse(plan.personalized_record_ids)
        self.assertEqual(plan.overall_memory_posture, "guarded")
        self.assertIn("does not write memory storage", plan.authority_boundary)
        self.assertTrue(plan.long_term_creative_memory_implemented)
        self.assertTrue(plan.memory_record_metadata_implemented)
        self.assertTrue(plan.memory_governance_metadata_implemented)
        self.assertFalse(plan.memory_storage_write_implemented)
        self.assertFalse(plan.memory_record_creation_implemented)
        self.assertFalse(plan.memory_record_update_implemented)
        self.assertFalse(plan.memory_record_deletion_implemented)
        self.assertFalse(plan.memory_retrieval_execution_implemented)
        self.assertFalse(plan.memory_consolidation_implemented)
        self.assertFalse(plan.preference_mutation_implemented)
        self.assertFalse(plan.personalization_application_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.agent_invocation_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_records_score_memory_governance_without_storage_mutation(self) -> None:
        plan = build_long_term_creative_memory(route="generate")

        for record in plan.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "long_term_creative_memory_record.v1",
            )
            self.assertEqual(record.route_name, RouteName.GENERATE)
            self.assertEqual(
                record.record_id,
                f"long_term_creative_memory::{record.memory_kind}",
            )
            self.assertEqual(
                record.memory_governance_score,
                min(
                    1000,
                    max(
                        0,
                        record.stability_score * 4
                        + record.evidence_strength_score * 3
                        + record.recency_resilience_score * 2
                        + record.governance_weight,
                    ),
                ),
            )
            self.assertIn("creative_memory", record.retrieval_tags)
            self.assertIn("memory_storage_write", record.blocked_runtime_behaviors)
            self.assertTrue(record.explainability_notes)
            self.assertTrue(record.advisory_actions)
            self.assertTrue(record.evidence)
            self.assertTrue(record.hitl_required_before_persistence)
            self.assertTrue(record.long_term_creative_memory_implemented)
            self.assertTrue(record.memory_record_metadata_implemented)
            self.assertTrue(record.memory_governance_metadata_implemented)
            self.assertFalse(record.memory_storage_write_implemented)
            self.assertFalse(record.memory_record_creation_implemented)
            self.assertFalse(record.memory_record_update_implemented)
            self.assertFalse(record.memory_record_deletion_implemented)
            self.assertFalse(record.memory_retrieval_execution_implemented)
            self.assertFalse(record.memory_consolidation_implemented)
            self.assertFalse(record.preference_mutation_implemented)
            self.assertFalse(record.personalization_application_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.provider_execution_implemented)
            self.assertFalse(record.agent_invocation_implemented)
            self.assertFalse(record.workflow_control_implemented)
            self.assertFalse(record.workflow_graph_mutation_implemented)
            self.assertFalse(record.workflow_execution_implemented)
            self.assertFalse(record.persistent_storage_write_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)
            self.assertFalse(record.runtime_evolution_implemented)
            self.assertTrue(record.advisory_only)

        style = long_term_creative_memory_record_by_id(
            "long_term_creative_memory::style_pattern_memory",
            plan,
        )
        self.assertIsNotNone(style)
        assert style is not None
        self.assertEqual(style.status, "guarded")
        self.assertEqual(style.sensitivity, "guarded")
        self.assertEqual(
            len(long_term_creative_memory_records_for_status("guarded", plan)),
            2,
        )
        self.assertEqual(
            len(long_term_creative_memory_records_for_sensitivity("high", plan)),
            1,
        )

    def test_plan_rejects_mismatched_memory_metadata(self) -> None:
        plan = build_long_term_creative_memory()
        payload = plan.model_dump(mode="json")
        payload["record_ids"] = ("missing",) + tuple(payload["record_ids"][1:])

        with self.assertRaisesRegex(ValueError, "record_ids must match"):
            LongTermCreativeMemoryPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_memory_governance_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_memory_governance_score must match",
        ):
            LongTermCreativeMemoryPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["persisted_record_ids"] = (plan.record_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "persisted_record_ids must remain empty",
        ):
            LongTermCreativeMemoryPlan(**payload)

    def test_long_term_memory_does_not_change_routing_or_provider_factory(
        self,
    ) -> None:
        request = AssistantRequest(
            query="Plan long-horizon memory for a creative coding project.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_long_term_creative_memory(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_long_term_memory_does_not_declare_runtime_application_terms(
        self,
    ) -> None:
        plan = build_long_term_creative_memory(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for record in plan.records
                    for field in (
                        record.record_id,
                        record.memory_kind,
                        record.status,
                        record.sensitivity,
                        record.retention_scope,
                        record.source_memory_kind.value,
                        record.source_surface,
                        *record.retrieval_tags,
                        record.memory_summary,
                        *record.explainability_notes,
                        *record.advisory_actions,
                        *record.evidence,
                        *record.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "write_memory(",
            "create_memory_record(",
            "update_memory_record(",
            "delete_memory_record(",
            "retrieve_memory(",
            "consolidate_memory(",
            "mutate_preference(",
            "apply_personalization(",
            "route_provider(",
            "switch_provider(",
            "switch_model(",
            "execute_provider(",
            "invoke_agent(",
            "control_workflow(",
            "mutate_workflow_graph(",
            "execute_workflow(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
            "apply_runtime_evolution(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
