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
    CreativeDNAPlan,
    build_creative_dna,
    build_project_memory,
    creative_dna_signature_by_id,
    creative_dna_signatures_for_confidence,
    creative_dna_signatures_for_status,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_SIGNATURE_FIELDS = {
    "creative_dna_id",
    "feature_kind",
    "status",
    "confidence",
    "route_name",
    "task_type",
    "execution_mode_id",
    "expression_axis",
    "source_long_term_memory_record_id",
    "source_user_preference_id",
    "source_style_profile_id",
    "source_project_memory_signal_id",
    "dna_statement",
    "source_alignment_score",
    "style_consistency_score",
    "project_continuity_score",
    "conflict_risk_score",
    "governance_weight",
    "creative_dna_score",
    "hitl_required_before_application",
    "context_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "creative_dna_implemented",
    "creative_dna_metadata_implemented",
    "long_term_memory_source_used",
    "user_preferences_source_used",
    "style_profile_source_used",
    "project_memory_source_used",
    "creative_dna_storage_write_implemented",
    "creative_dna_signature_creation_implemented",
    "creative_dna_signature_update_implemented",
    "creative_dna_signature_deletion_implemented",
    "automatic_creative_dna_learning_implemented",
    "creative_dna_application_implemented",
    "preference_mutation_implemented",
    "personalization_application_implemented",
    "memory_retrieval_execution_implemented",
    "memory_storage_write_implemented",
    "project_memory_storage_write_implemented",
    "style_profile_application_implemented",
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


class CreativeDNATests(unittest.TestCase):
    def test_plan_builds_creative_dna_metadata(self) -> None:
        project_memory = build_project_memory(route=RouteName.GENERATE)
        plan = build_creative_dna(
            route=RouteName.GENERATE,
            project_memory=project_memory,
        )

        self.assertEqual(plan.role, "creative_dna")
        self.assertEqual(plan.serialization_version, "creative_dna_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_long_term_memory_serialization_version,
            "long_term_creative_memory_plan.v1",
        )
        self.assertEqual(
            plan.source_user_preferences_serialization_version,
            "user_preferences_plan.v1",
        )
        self.assertEqual(
            plan.source_style_profile_serialization_version,
            "style_profile_plan.v1",
        )
        self.assertEqual(
            plan.source_project_memory_serialization_version,
            "project_memory_plan.v1",
        )
        self.assertEqual(
            plan.source_project_memory_signal_ids,
            project_memory.signal_ids,
        )
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.signature_count, 5)
        self.assertEqual(plan.candidate_signature_count, 1)
        self.assertEqual(plan.review_required_signature_count, 2)
        self.assertEqual(plan.guarded_signature_count, 2)
        self.assertEqual(plan.high_confidence_signature_count, 3)
        self.assertEqual(plan.hitl_required_signature_count, 5)
        self.assertFalse(plan.persisted_creative_dna_ids)
        self.assertFalse(plan.learned_creative_dna_ids)
        self.assertFalse(plan.applied_creative_dna_ids)
        self.assertFalse(plan.personalized_creative_dna_ids)
        self.assertEqual(plan.overall_creative_dna_posture, "guarded")
        self.assertIn(
            "does not write Creative DNA storage",
            plan.authority_boundary,
        )
        self.assertIn(
            "apply Creative DNA to prompts or generated output",
            plan.authority_boundary,
        )
        self.assertTrue(plan.creative_dna_implemented)
        self.assertTrue(plan.creative_dna_metadata_implemented)
        self.assertTrue(plan.long_term_memory_source_used)
        self.assertTrue(plan.user_preferences_source_used)
        self.assertTrue(plan.style_profile_source_used)
        self.assertTrue(plan.project_memory_source_used)
        self.assertFalse(plan.creative_dna_storage_write_implemented)
        self.assertFalse(plan.creative_dna_signature_creation_implemented)
        self.assertFalse(plan.creative_dna_signature_update_implemented)
        self.assertFalse(plan.creative_dna_signature_deletion_implemented)
        self.assertFalse(plan.automatic_creative_dna_learning_implemented)
        self.assertFalse(plan.creative_dna_application_implemented)
        self.assertFalse(plan.preference_mutation_implemented)
        self.assertFalse(plan.personalization_application_implemented)
        self.assertFalse(plan.memory_retrieval_execution_implemented)
        self.assertFalse(plan.memory_storage_write_implemented)
        self.assertFalse(plan.project_memory_storage_write_implemented)
        self.assertFalse(plan.style_profile_application_implemented)
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

    def test_signatures_score_creative_dna_without_application(self) -> None:
        plan = build_creative_dna(route="generate")

        for signature in plan.signatures:
            dumped = signature.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_SIGNATURE_FIELDS)
            self.assertEqual(
                signature.serialization_version,
                "creative_dna_signature.v1",
            )
            self.assertEqual(signature.route_name, RouteName.GENERATE)
            self.assertEqual(
                signature.creative_dna_id,
                f"creative_dna::{signature.feature_kind}",
            )
            self.assertEqual(
                signature.creative_dna_score,
                min(
                    1000,
                    max(
                        0,
                        signature.source_alignment_score * 3
                        + signature.style_consistency_score * 3
                        + signature.project_continuity_score * 2
                        + signature.conflict_risk_score * 3
                        + signature.governance_weight,
                    ),
                ),
            )
            self.assertIn("creative_dna", signature.context_tags)
            self.assertIn(
                "creative_dna_application",
                signature.blocked_runtime_behaviors,
            )
            self.assertTrue(signature.explainability_notes)
            self.assertTrue(signature.advisory_actions)
            self.assertTrue(signature.evidence)
            self.assertTrue(signature.hitl_required_before_application)
            self.assertTrue(signature.creative_dna_implemented)
            self.assertTrue(signature.creative_dna_metadata_implemented)
            self.assertTrue(signature.long_term_memory_source_used)
            self.assertTrue(signature.user_preferences_source_used)
            self.assertTrue(signature.style_profile_source_used)
            self.assertTrue(signature.project_memory_source_used)
            self.assertFalse(signature.creative_dna_storage_write_implemented)
            self.assertFalse(signature.creative_dna_signature_creation_implemented)
            self.assertFalse(signature.creative_dna_signature_update_implemented)
            self.assertFalse(signature.creative_dna_signature_deletion_implemented)
            self.assertFalse(signature.automatic_creative_dna_learning_implemented)
            self.assertFalse(signature.creative_dna_application_implemented)
            self.assertFalse(signature.preference_mutation_implemented)
            self.assertFalse(signature.personalization_application_implemented)
            self.assertFalse(signature.memory_retrieval_execution_implemented)
            self.assertFalse(signature.memory_storage_write_implemented)
            self.assertFalse(signature.project_memory_storage_write_implemented)
            self.assertFalse(signature.style_profile_application_implemented)
            self.assertFalse(signature.provider_model_routing_implemented)
            self.assertFalse(signature.provider_execution_implemented)
            self.assertFalse(signature.agent_invocation_implemented)
            self.assertFalse(signature.workflow_control_implemented)
            self.assertFalse(signature.workflow_graph_mutation_implemented)
            self.assertFalse(signature.workflow_execution_implemented)
            self.assertFalse(signature.persistent_storage_write_implemented)
            self.assertFalse(signature.generated_output_mutation_implemented)
            self.assertFalse(signature.runtime_evolution_implemented)
            self.assertTrue(signature.advisory_only)

        intent = creative_dna_signature_by_id("creative_dna::intent_dna", plan)
        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.status, "guarded")
        self.assertEqual(intent.confidence, "guarded")
        self.assertEqual(len(creative_dna_signatures_for_status("guarded", plan)), 2)
        self.assertEqual(len(creative_dna_signatures_for_confidence("high", plan)), 1)

    def test_plan_rejects_mismatched_creative_dna_metadata(self) -> None:
        plan = build_creative_dna()
        payload = plan.model_dump(mode="json")
        payload["signature_ids"] = ("missing",) + tuple(payload["signature_ids"][1:])

        with self.assertRaisesRegex(ValueError, "signature_ids must match"):
            CreativeDNAPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_creative_dna_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_creative_dna_score must match",
        ):
            CreativeDNAPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["applied_creative_dna_ids"] = (plan.signature_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "applied_creative_dna_ids must remain empty",
        ):
            CreativeDNAPlan(**payload)

    def test_creative_dna_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review creative DNA for a creative coding project.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_creative_dna(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_creative_dna_does_not_declare_runtime_application_terms(self) -> None:
        plan = build_creative_dna(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for signature in plan.signatures
                    for field in (
                        signature.creative_dna_id,
                        signature.feature_kind,
                        signature.status,
                        signature.confidence,
                        signature.expression_axis,
                        signature.source_long_term_memory_record_id,
                        signature.source_user_preference_id,
                        signature.source_style_profile_id,
                        signature.source_project_memory_signal_id,
                        signature.dna_statement,
                        *signature.context_tags,
                        *signature.explainability_notes,
                        *signature.advisory_actions,
                        *signature.evidence,
                        *signature.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "write_creative_dna(",
            "create_creative_dna(",
            "update_creative_dna(",
            "delete_creative_dna(",
            "learn_creative_dna(",
            "apply_creative_dna(",
            "retrieve_memory(",
            "write_memory(",
            "write_project_memory(",
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
