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
    ProjectMemoryPlan,
    build_project_memory,
    build_style_profiles,
    project_memory_signal_by_id,
    project_memory_signals_for_confidence,
    project_memory_signals_for_status,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_SIGNAL_FIELDS = {
    "project_memory_id",
    "facet_kind",
    "status",
    "confidence",
    "route_name",
    "task_type",
    "execution_mode_id",
    "memory_scope",
    "project_memory_kind",
    "source_long_term_memory_record_id",
    "source_style_profile_id",
    "project_memory_summary",
    "continuity_score",
    "specificity_score",
    "conflict_risk_score",
    "governance_weight",
    "project_memory_score",
    "hitl_required_before_storage",
    "context_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "project_memory_implemented",
    "project_memory_metadata_implemented",
    "long_term_memory_source_used",
    "style_profile_source_used",
    "project_memory_storage_write_implemented",
    "project_memory_record_creation_implemented",
    "project_memory_record_update_implemented",
    "project_memory_record_deletion_implemented",
    "memory_retrieval_execution_implemented",
    "memory_consolidation_implemented",
    "style_profile_application_implemented",
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


class ProjectMemoryTests(unittest.TestCase):
    def test_plan_builds_project_memory_metadata(self) -> None:
        styles = build_style_profiles(route=RouteName.GENERATE)
        plan = build_project_memory(route=RouteName.GENERATE, style_profiles=styles)

        self.assertEqual(plan.role, "project_memory")
        self.assertEqual(plan.serialization_version, "project_memory_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_long_term_memory_serialization_version,
            "long_term_creative_memory_plan.v1",
        )
        self.assertEqual(
            plan.source_style_profile_serialization_version,
            "style_profile_plan.v1",
        )
        self.assertEqual(plan.source_style_profile_ids, styles.profile_ids)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.signal_count, 5)
        self.assertEqual(plan.candidate_signal_count, 1)
        self.assertEqual(plan.review_required_signal_count, 2)
        self.assertEqual(plan.guarded_signal_count, 2)
        self.assertEqual(plan.high_confidence_signal_count, 3)
        self.assertEqual(plan.hitl_required_signal_count, 5)
        self.assertFalse(plan.stored_project_memory_ids)
        self.assertFalse(plan.retrieved_project_memory_ids)
        self.assertFalse(plan.consolidated_project_memory_ids)
        self.assertEqual(plan.overall_project_memory_posture, "guarded")
        self.assertIn(
            "does not write project memory storage",
            plan.authority_boundary,
        )
        self.assertTrue(plan.project_memory_implemented)
        self.assertTrue(plan.project_memory_metadata_implemented)
        self.assertTrue(plan.long_term_memory_source_used)
        self.assertTrue(plan.style_profile_source_used)
        self.assertFalse(plan.project_memory_storage_write_implemented)
        self.assertFalse(plan.project_memory_record_creation_implemented)
        self.assertFalse(plan.project_memory_record_update_implemented)
        self.assertFalse(plan.project_memory_record_deletion_implemented)
        self.assertFalse(plan.memory_retrieval_execution_implemented)
        self.assertFalse(plan.memory_consolidation_implemented)
        self.assertFalse(plan.style_profile_application_implemented)
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

    def test_signals_score_project_memory_without_storage(self) -> None:
        plan = build_project_memory(route="generate")

        for signal in plan.signals:
            dumped = signal.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_SIGNAL_FIELDS)
            self.assertEqual(
                signal.serialization_version,
                "project_memory_signal.v1",
            )
            self.assertEqual(signal.route_name, RouteName.GENERATE)
            self.assertEqual(
                signal.project_memory_id,
                f"project_memory::{signal.facet_kind}",
            )
            self.assertEqual(
                signal.project_memory_score,
                min(
                    1000,
                    max(
                        0,
                        signal.continuity_score * 3
                        + signal.specificity_score * 3
                        + signal.conflict_risk_score * 4
                        + signal.governance_weight,
                    ),
                ),
            )
            self.assertIn("project_memory", signal.context_tags)
            self.assertIn(
                "project_memory_storage_write",
                signal.blocked_runtime_behaviors,
            )
            self.assertTrue(signal.explainability_notes)
            self.assertTrue(signal.advisory_actions)
            self.assertTrue(signal.evidence)
            self.assertTrue(signal.hitl_required_before_storage)
            self.assertTrue(signal.project_memory_implemented)
            self.assertTrue(signal.project_memory_metadata_implemented)
            self.assertTrue(signal.long_term_memory_source_used)
            self.assertTrue(signal.style_profile_source_used)
            self.assertFalse(signal.project_memory_storage_write_implemented)
            self.assertFalse(signal.project_memory_record_creation_implemented)
            self.assertFalse(signal.project_memory_record_update_implemented)
            self.assertFalse(signal.project_memory_record_deletion_implemented)
            self.assertFalse(signal.memory_retrieval_execution_implemented)
            self.assertFalse(signal.memory_consolidation_implemented)
            self.assertFalse(signal.style_profile_application_implemented)
            self.assertFalse(signal.preference_mutation_implemented)
            self.assertFalse(signal.personalization_application_implemented)
            self.assertFalse(signal.provider_model_routing_implemented)
            self.assertFalse(signal.provider_execution_implemented)
            self.assertFalse(signal.agent_invocation_implemented)
            self.assertFalse(signal.workflow_control_implemented)
            self.assertFalse(signal.workflow_graph_mutation_implemented)
            self.assertFalse(signal.workflow_execution_implemented)
            self.assertFalse(signal.persistent_storage_write_implemented)
            self.assertFalse(signal.generated_output_mutation_implemented)
            self.assertFalse(signal.runtime_evolution_implemented)
            self.assertTrue(signal.advisory_only)

        goal = project_memory_signal_by_id(
            "project_memory::project_goal_memory",
            plan,
        )
        self.assertIsNotNone(goal)
        assert goal is not None
        self.assertEqual(goal.status, "guarded")
        self.assertEqual(goal.confidence, "guarded")
        self.assertEqual(len(project_memory_signals_for_status("guarded", plan)), 2)
        self.assertEqual(len(project_memory_signals_for_confidence("high", plan)), 1)

    def test_plan_rejects_mismatched_project_memory_metadata(self) -> None:
        plan = build_project_memory()
        payload = plan.model_dump(mode="json")
        payload["signal_ids"] = ("missing",) + tuple(payload["signal_ids"][1:])

        with self.assertRaisesRegex(ValueError, "signal_ids must match"):
            ProjectMemoryPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_project_memory_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_project_memory_score must match",
        ):
            ProjectMemoryPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["stored_project_memory_ids"] = (plan.signal_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "stored_project_memory_ids must remain empty",
        ):
            ProjectMemoryPlan(**payload)

    def test_project_memory_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review project memory for a creative coding project.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_project_memory(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_project_memory_does_not_declare_runtime_application_terms(self) -> None:
        plan = build_project_memory(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for signal in plan.signals
                    for field in (
                        signal.project_memory_id,
                        signal.facet_kind,
                        signal.status,
                        signal.confidence,
                        signal.memory_scope,
                        signal.project_memory_kind.value,
                        signal.source_long_term_memory_record_id,
                        signal.source_style_profile_id,
                        signal.project_memory_summary,
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
            "write_project_memory(",
            "create_project_memory(",
            "update_project_memory(",
            "delete_project_memory(",
            "retrieve_memory(",
            "consolidate_memory(",
            "apply_style_profile(",
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
