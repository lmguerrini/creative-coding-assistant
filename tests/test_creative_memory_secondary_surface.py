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
    CreativeMemorySecondarySurfacePlan,
    build_creative_memory_secondary_surface,
    creative_memory_secondary_surface_entries_for_confidence,
    creative_memory_secondary_surface_entries_for_status,
    creative_memory_secondary_surface_entry_by_id,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
EXPECTED_SOURCE_ROLES = (
    "creative_memory_core_surface",
    "adaptive_learning_engine",
    "adaptive_execution_policy_engine",
)
EXPECTED_ROADMAP_ITEMS = (
    "Preference Learning",
    "User Modeling",
    "Memory Consolidation",
    "Memory Retrieval Intelligence",
    "Memory Retrieval Planner",
    "Memory Conflict Resolution",
    "Memory Explainability",
    "Memory Safety Policies",
    "Creative Taste Model",
    "Creative Preference Evolution",
)
REQUIRED_ENTRY_FIELDS = {
    "secondary_surface_id",
    "surface_kind",
    "status",
    "confidence",
    "route_name",
    "task_type",
    "execution_mode_id",
    "surface_axis",
    "roadmap_items",
    "roadmap_item_count",
    "source_plan_roles",
    "source_serialization_versions",
    "source_item_ids",
    "source_item_count",
    "surface_summary",
    "roadmap_coverage_score",
    "source_traceability_score",
    "governance_alignment_score",
    "v5_v6_composition_score",
    "activation_risk_score",
    "governance_weight",
    "secondary_surface_score",
    "hitl_required_before_secondary_surface_activation",
    "context_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "secondary_surface_implemented",
    "secondary_surface_metadata_implemented",
    "v5_policy_foundation_used",
    "v6_memory_foundation_used",
    "all_sources_metadata_only",
    "secondary_surface_activation_implemented",
    "preference_learning_execution_implemented",
    "preference_storage_write_implemented",
    "user_model_creation_implemented",
    "user_model_update_implemented",
    "user_model_application_implemented",
    "memory_consolidation_execution_implemented",
    "memory_retrieval_execution_implemented",
    "retrieval_planner_execution_implemented",
    "memory_conflict_resolution_execution_implemented",
    "memory_explainability_generation_implemented",
    "memory_safety_policy_enforcement_implemented",
    "creative_taste_model_application_implemented",
    "preference_evolution_application_implemented",
    "creative_dna_application_implemented",
    "personalization_application_implemented",
    "provider_model_routing_implemented",
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


class CreativeMemorySecondarySurfaceTests(unittest.TestCase):
    def test_plan_builds_secondary_surface_metadata(self) -> None:
        plan = build_creative_memory_secondary_surface(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "creative_memory_secondary_surface")
        self.assertEqual(
            plan.serialization_version,
            "creative_memory_secondary_surface_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.source_plan_roles, EXPECTED_SOURCE_ROLES)
        self.assertEqual(
            plan.source_plan_serialization_versions,
            (
                "creative_memory_core_surface_plan.v1",
                "adaptive_learning_plan.v1",
                "adaptive_execution_policy_plan.v1",
            ),
        )
        self.assertEqual(plan.source_item_count, 15)
        self.assertEqual(len(plan.source_item_ids), 15)
        self.assertEqual(plan.covered_roadmap_items, EXPECTED_ROADMAP_ITEMS)
        self.assertEqual(plan.covered_roadmap_item_count, 10)
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.entry_count, 5)
        self.assertEqual(plan.candidate_entry_count, 1)
        self.assertEqual(plan.review_required_entry_count, 3)
        self.assertEqual(plan.guarded_entry_count, 1)
        self.assertEqual(plan.high_confidence_entry_count, 3)
        self.assertEqual(plan.hitl_required_entry_count, 5)
        self.assertFalse(plan.activated_secondary_surface_ids)
        self.assertFalse(plan.learned_preference_ids)
        self.assertFalse(plan.created_user_model_ids)
        self.assertFalse(plan.consolidated_memory_ids)
        self.assertFalse(plan.executed_retrieval_plan_ids)
        self.assertFalse(plan.resolved_conflict_ids)
        self.assertFalse(plan.enforced_policy_ids)
        self.assertFalse(plan.applied_taste_model_ids)
        self.assertFalse(plan.mutated_output_ids)
        self.assertEqual(plan.overall_secondary_surface_posture, "guarded")
        self.assertIn("V5 controlled execution policy", plan.authority_boundary)
        self.assertIn("does not activate secondary surfaces", plan.authority_boundary)
        self.assertTrue(plan.secondary_surface_implemented)
        self.assertTrue(plan.secondary_surface_metadata_implemented)
        self.assertTrue(plan.secondary_roadmap_items_covered)
        self.assertTrue(plan.v5_policy_foundation_used)
        self.assertTrue(plan.v6_memory_foundation_used)
        self.assertTrue(plan.all_sources_metadata_only)
        self.assertFalse(plan.secondary_surface_activation_implemented)
        self.assertFalse(plan.preference_learning_execution_implemented)
        self.assertFalse(plan.preference_storage_write_implemented)
        self.assertFalse(plan.user_model_creation_implemented)
        self.assertFalse(plan.user_model_update_implemented)
        self.assertFalse(plan.user_model_application_implemented)
        self.assertFalse(plan.memory_consolidation_execution_implemented)
        self.assertFalse(plan.memory_retrieval_execution_implemented)
        self.assertFalse(plan.retrieval_planner_execution_implemented)
        self.assertFalse(plan.memory_conflict_resolution_execution_implemented)
        self.assertFalse(plan.memory_explainability_generation_implemented)
        self.assertFalse(plan.memory_safety_policy_enforcement_implemented)
        self.assertFalse(plan.creative_taste_model_application_implemented)
        self.assertFalse(plan.preference_evolution_application_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.workflow_graph_mutation_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.advisory_only)

    def test_entries_score_secondary_surface_without_activation(self) -> None:
        plan = build_creative_memory_secondary_surface(route="generate")
        source_items = set(plan.source_item_ids)
        source_roles = set(plan.source_plan_roles)

        for entry in plan.entries:
            dumped = entry.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_ENTRY_FIELDS)
            self.assertEqual(
                entry.serialization_version,
                "creative_memory_secondary_surface_entry.v1",
            )
            self.assertEqual(entry.route_name, RouteName.GENERATE)
            self.assertEqual(
                entry.secondary_surface_id,
                f"creative_memory_secondary::{entry.surface_kind}",
            )
            self.assertEqual(entry.roadmap_item_count, len(entry.roadmap_items))
            self.assertEqual(entry.source_item_count, len(entry.source_item_ids))
            self.assertTrue(set(entry.source_item_ids).issubset(source_items))
            self.assertTrue(set(entry.source_plan_roles).issubset(source_roles))
            self.assertEqual(
                entry.secondary_surface_score,
                min(
                    1000,
                    max(
                        0,
                        entry.roadmap_coverage_score * 2
                        + entry.source_traceability_score * 2
                        + entry.governance_alignment_score * 2
                        + entry.v5_v6_composition_score * 2
                        + entry.activation_risk_score
                        + entry.governance_weight,
                    ),
                ),
            )
            self.assertIn("secondary_surface", entry.context_tags)
            self.assertIn(
                "secondary_surface_activation",
                entry.blocked_runtime_behaviors,
            )
            self.assertIn(
                "memory_retrieval_execution",
                entry.blocked_runtime_behaviors,
            )
            self.assertTrue(entry.explainability_notes)
            self.assertTrue(entry.advisory_actions)
            self.assertTrue(entry.evidence)
            self.assertTrue(entry.hitl_required_before_secondary_surface_activation)
            self.assertTrue(entry.secondary_surface_implemented)
            self.assertTrue(entry.secondary_surface_metadata_implemented)
            self.assertTrue(entry.v5_policy_foundation_used)
            self.assertTrue(entry.v6_memory_foundation_used)
            self.assertTrue(entry.all_sources_metadata_only)
            self.assertFalse(entry.secondary_surface_activation_implemented)
            self.assertFalse(entry.preference_learning_execution_implemented)
            self.assertFalse(entry.preference_storage_write_implemented)
            self.assertFalse(entry.user_model_creation_implemented)
            self.assertFalse(entry.user_model_update_implemented)
            self.assertFalse(entry.user_model_application_implemented)
            self.assertFalse(entry.memory_consolidation_execution_implemented)
            self.assertFalse(entry.memory_retrieval_execution_implemented)
            self.assertFalse(entry.retrieval_planner_execution_implemented)
            self.assertFalse(entry.memory_conflict_resolution_execution_implemented)
            self.assertFalse(entry.memory_explainability_generation_implemented)
            self.assertFalse(entry.memory_safety_policy_enforcement_implemented)
            self.assertFalse(entry.creative_taste_model_application_implemented)
            self.assertFalse(entry.preference_evolution_application_implemented)
            self.assertFalse(entry.creative_dna_application_implemented)
            self.assertFalse(entry.personalization_application_implemented)
            self.assertFalse(entry.provider_model_routing_implemented)
            self.assertFalse(entry.provider_execution_implemented)
            self.assertFalse(entry.agent_invocation_implemented)
            self.assertFalse(entry.workflow_control_implemented)
            self.assertFalse(entry.workflow_graph_mutation_implemented)
            self.assertFalse(entry.workflow_execution_implemented)
            self.assertFalse(entry.persistent_storage_write_implemented)
            self.assertFalse(entry.generated_output_mutation_implemented)
            self.assertFalse(entry.runtime_evolution_implemented)
            self.assertTrue(entry.advisory_only)

        preference_learning = creative_memory_secondary_surface_entry_by_id(
            "creative_memory_secondary::preference_learning_surface",
            plan,
        )
        self.assertIsNotNone(preference_learning)
        assert preference_learning is not None
        self.assertEqual(preference_learning.status, "guarded")
        self.assertEqual(preference_learning.confidence, "guarded")
        self.assertEqual(
            len(
                creative_memory_secondary_surface_entries_for_status(
                    "review_required",
                    plan,
                )
            ),
            3,
        )
        self.assertEqual(
            len(creative_memory_secondary_surface_entries_for_confidence("high", plan)),
            2,
        )

    def test_plan_rejects_mismatched_secondary_surface_metadata(self) -> None:
        plan = build_creative_memory_secondary_surface()
        payload = plan.model_dump(mode="json")
        payload["entry_ids"] = ("missing",) + tuple(payload["entry_ids"][1:])

        with self.assertRaisesRegex(ValueError, "entry_ids must match"):
            CreativeMemorySecondarySurfacePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_secondary_surface_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_secondary_surface_score must match",
        ):
            CreativeMemorySecondarySurfacePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["activated_secondary_surface_ids"] = (plan.entry_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "activated_secondary_surface_ids must remain empty",
        ):
            CreativeMemorySecondarySurfacePlan(**payload)

    def test_secondary_surface_preserves_routing_and_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Review the creative memory secondary surface.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_creative_memory_secondary_surface(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_secondary_surface_does_not_declare_runtime_mutation_terms(self) -> None:
        plan = build_creative_memory_secondary_surface(route=RouteName.GENERATE)
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
                        entry.secondary_surface_id,
                        entry.surface_kind,
                        entry.status,
                        entry.confidence,
                        entry.surface_axis,
                        *entry.roadmap_items,
                        *entry.source_plan_roles,
                        *entry.source_serialization_versions,
                        *entry.source_item_ids,
                        entry.surface_summary,
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
            "activate_secondary_surface(",
            "learn_preference(",
            "write_preference(",
            "create_user_model(",
            "update_user_model(",
            "apply_user_model(",
            "consolidate_memory(",
            "retrieve_memory(",
            "plan_retrieval(",
            "resolve_memory_conflict(",
            "generate_memory_explainability(",
            "enforce_memory_safety(",
            "apply_creative_taste_model(",
            "evolve_creative_preference(",
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
