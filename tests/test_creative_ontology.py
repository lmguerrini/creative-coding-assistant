import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.domains import (
    DomainCategory,
    get_domain_categories,
    get_domain_category_label,
    get_domains_for_category,
)
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    CreativeOntologyPlan,
    build_creative_ontology,
    creative_ontology_concept_by_id,
    creative_ontology_concepts_for_confidence,
    creative_ontology_concepts_for_status,
    route_request,
)
from creative_coding_assistant.orchestration.creative_ontology import (
    DOMAIN_METADATA_REGISTRY_REF,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_EXECUTION_MODES = ("manual_mode", "assisted_mode", "auto_mode")
REQUIRED_CONCEPT_FIELDS = {
    "creative_ontology_id",
    "concept_kind",
    "status",
    "confidence",
    "route_name",
    "task_type",
    "execution_mode_id",
    "ontology_axis",
    "source_creative_lineage_record_id",
    "source_creative_dna_signature_id",
    "source_long_term_memory_record_id",
    "source_domain_category",
    "source_domain_category_label",
    "source_domain_values",
    "ontology_statement",
    "concept_coverage_score",
    "taxonomy_alignment_score",
    "lineage_alignment_score",
    "governance_risk_score",
    "governance_weight",
    "creative_ontology_score",
    "hitl_required_before_ontology_persistence",
    "context_tags",
    "explainability_notes",
    "advisory_actions",
    "evidence",
    "blocked_runtime_behaviors",
    "creative_ontology_implemented",
    "creative_ontology_metadata_implemented",
    "creative_lineage_source_used",
    "creative_dna_source_used",
    "long_term_memory_source_used",
    "domain_metadata_source_used",
    "ontology_storage_write_implemented",
    "ontology_node_creation_implemented",
    "ontology_edge_creation_implemented",
    "ontology_record_update_implemented",
    "ontology_record_deletion_implemented",
    "ontology_relationship_inference_implemented",
    "taxonomy_mutation_implemented",
    "semantic_graph_materialization_implemented",
    "domain_registry_mutation_implemented",
    "creative_lineage_application_implemented",
    "creative_dna_application_implemented",
    "memory_retrieval_execution_implemented",
    "memory_storage_write_implemented",
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


class CreativeOntologyTests(unittest.TestCase):
    def test_plan_builds_creative_ontology_metadata(self) -> None:
        plan = build_creative_ontology(route=RouteName.GENERATE)

        self.assertEqual(plan.role, "creative_ontology")
        self.assertEqual(plan.serialization_version, "creative_ontology_plan.v1")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(
            plan.source_creative_lineage_serialization_version,
            "creative_lineage_plan.v1",
        )
        self.assertEqual(
            plan.source_creative_dna_serialization_version,
            "creative_dna_plan.v1",
        )
        self.assertEqual(
            plan.source_long_term_memory_serialization_version,
            "long_term_creative_memory_plan.v1",
        )
        self.assertEqual(
            plan.source_domain_metadata_registry_ref,
            DOMAIN_METADATA_REGISTRY_REF,
        )
        self.assertEqual(plan.source_domain_category_ids, get_domain_categories())
        self.assertEqual(
            plan.source_domain_count,
            sum(len(get_domains_for_category(category)) for category in DomainCategory),
        )
        self.assertEqual(plan.execution_mode_ids, REQUIRED_EXECUTION_MODES)
        self.assertEqual(plan.concept_count, 5)
        self.assertEqual(plan.candidate_concept_count, 1)
        self.assertEqual(plan.review_required_concept_count, 2)
        self.assertEqual(plan.guarded_concept_count, 2)
        self.assertEqual(plan.high_confidence_concept_count, 3)
        self.assertEqual(plan.hitl_required_concept_count, 5)
        self.assertFalse(plan.persisted_ontology_ids)
        self.assertFalse(plan.inferred_ontology_ids)
        self.assertFalse(plan.materialized_semantic_graph_ids)
        self.assertFalse(plan.mutated_domain_registry_ids)
        self.assertEqual(plan.overall_creative_ontology_posture, "guarded")
        self.assertIn("does not write ontology storage", plan.authority_boundary)
        self.assertIn("infer ontology relationships", plan.authority_boundary)
        self.assertTrue(plan.creative_ontology_implemented)
        self.assertTrue(plan.creative_ontology_metadata_implemented)
        self.assertTrue(plan.creative_lineage_source_used)
        self.assertTrue(plan.creative_dna_source_used)
        self.assertTrue(plan.long_term_memory_source_used)
        self.assertTrue(plan.domain_metadata_source_used)
        self.assertFalse(plan.ontology_storage_write_implemented)
        self.assertFalse(plan.ontology_node_creation_implemented)
        self.assertFalse(plan.ontology_edge_creation_implemented)
        self.assertFalse(plan.ontology_record_update_implemented)
        self.assertFalse(plan.ontology_record_deletion_implemented)
        self.assertFalse(plan.ontology_relationship_inference_implemented)
        self.assertFalse(plan.taxonomy_mutation_implemented)
        self.assertFalse(plan.semantic_graph_materialization_implemented)
        self.assertFalse(plan.domain_registry_mutation_implemented)
        self.assertFalse(plan.creative_lineage_application_implemented)
        self.assertFalse(plan.creative_dna_application_implemented)
        self.assertFalse(plan.memory_retrieval_execution_implemented)
        self.assertFalse(plan.memory_storage_write_implemented)
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

    def test_concepts_score_creative_ontology_without_inference(self) -> None:
        plan = build_creative_ontology(route="generate")

        for concept in plan.concepts:
            dumped = concept.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CONCEPT_FIELDS)
            self.assertEqual(
                concept.serialization_version,
                "creative_ontology_concept.v1",
            )
            self.assertEqual(concept.route_name, RouteName.GENERATE)
            self.assertEqual(
                concept.creative_ontology_id,
                f"creative_ontology::{concept.concept_kind}",
            )
            self.assertEqual(
                concept.source_domain_category_label,
                get_domain_category_label(concept.source_domain_category),
            )
            self.assertEqual(
                concept.source_domain_values,
                get_domains_for_category(concept.source_domain_category),
            )
            self.assertEqual(
                concept.creative_ontology_score,
                min(
                    1000,
                    max(
                        0,
                        concept.concept_coverage_score * 3
                        + concept.taxonomy_alignment_score * 3
                        + concept.lineage_alignment_score * 2
                        + concept.governance_risk_score * 2
                        + concept.governance_weight,
                    ),
                ),
            )
            self.assertIn("creative_ontology", concept.context_tags)
            self.assertIn(
                "ontology_relationship_inference",
                concept.blocked_runtime_behaviors,
            )
            self.assertIn(
                "semantic_graph_materialization",
                concept.blocked_runtime_behaviors,
            )
            self.assertTrue(concept.explainability_notes)
            self.assertTrue(concept.advisory_actions)
            self.assertTrue(concept.evidence)
            self.assertTrue(concept.hitl_required_before_ontology_persistence)
            self.assertTrue(concept.creative_ontology_implemented)
            self.assertTrue(concept.creative_ontology_metadata_implemented)
            self.assertTrue(concept.creative_lineage_source_used)
            self.assertTrue(concept.creative_dna_source_used)
            self.assertTrue(concept.long_term_memory_source_used)
            self.assertTrue(concept.domain_metadata_source_used)
            self.assertFalse(concept.ontology_storage_write_implemented)
            self.assertFalse(concept.ontology_node_creation_implemented)
            self.assertFalse(concept.ontology_edge_creation_implemented)
            self.assertFalse(concept.ontology_record_update_implemented)
            self.assertFalse(concept.ontology_record_deletion_implemented)
            self.assertFalse(concept.ontology_relationship_inference_implemented)
            self.assertFalse(concept.taxonomy_mutation_implemented)
            self.assertFalse(concept.semantic_graph_materialization_implemented)
            self.assertFalse(concept.domain_registry_mutation_implemented)
            self.assertFalse(concept.creative_lineage_application_implemented)
            self.assertFalse(concept.creative_dna_application_implemented)
            self.assertFalse(concept.memory_retrieval_execution_implemented)
            self.assertFalse(concept.memory_storage_write_implemented)
            self.assertFalse(concept.memory_consolidation_implemented)
            self.assertFalse(concept.preference_mutation_implemented)
            self.assertFalse(concept.personalization_application_implemented)
            self.assertFalse(concept.provider_model_routing_implemented)
            self.assertFalse(concept.provider_execution_implemented)
            self.assertFalse(concept.agent_invocation_implemented)
            self.assertFalse(concept.workflow_control_implemented)
            self.assertFalse(concept.workflow_graph_mutation_implemented)
            self.assertFalse(concept.workflow_execution_implemented)
            self.assertFalse(concept.persistent_storage_write_implemented)
            self.assertFalse(concept.generated_output_mutation_implemented)
            self.assertFalse(concept.runtime_evolution_implemented)
            self.assertTrue(concept.advisory_only)

        intent = creative_ontology_concept_by_id(
            "creative_ontology::creative_intent_ontology",
            plan,
        )
        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.status, "guarded")
        self.assertEqual(intent.confidence, "guarded")
        self.assertEqual(len(creative_ontology_concepts_for_status("guarded", plan)), 2)
        self.assertEqual(
            len(creative_ontology_concepts_for_confidence("high", plan)),
            1,
        )

    def test_plan_rejects_mismatched_creative_ontology_metadata(self) -> None:
        plan = build_creative_ontology()
        payload = plan.model_dump(mode="json")
        payload["concept_ids"] = ("missing",) + tuple(payload["concept_ids"][1:])

        with self.assertRaisesRegex(ValueError, "concept_ids must match"):
            CreativeOntologyPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["overall_creative_ontology_score"] -= 1

        with self.assertRaisesRegex(
            ValueError,
            "overall_creative_ontology_score must match",
        ):
            CreativeOntologyPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["materialized_semantic_graph_ids"] = (plan.concept_ids[0],)

        with self.assertRaisesRegex(
            ValueError,
            "materialized_semantic_graph_ids must remain empty",
        ):
            CreativeOntologyPlan(**payload)

    def test_creative_ontology_does_not_change_routing_or_provider_factory(
        self,
    ) -> None:
        request = AssistantRequest(
            query="Review creative ontology for a creative coding project.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        plan = build_creative_ontology(route=RouteName.GENERATE)
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_creative_ontology_does_not_declare_runtime_mutation_terms(self) -> None:
        plan = build_creative_ontology(route=RouteName.GENERATE)
        combined_text = " ".join(
            (
                plan.authority_boundary,
                *plan.blocked_runtime_behaviors,
                *plan.advisory_actions,
                *(
                    field
                    for concept in plan.concepts
                    for field in (
                        concept.creative_ontology_id,
                        concept.concept_kind,
                        concept.status,
                        concept.confidence,
                        concept.ontology_axis,
                        concept.source_creative_lineage_record_id,
                        concept.source_creative_dna_signature_id,
                        concept.source_long_term_memory_record_id,
                        concept.source_domain_category.value,
                        concept.source_domain_category_label,
                        *(domain.value for domain in concept.source_domain_values),
                        concept.ontology_statement,
                        *concept.context_tags,
                        *concept.explainability_notes,
                        *concept.advisory_actions,
                        *concept.evidence,
                        *concept.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "write_ontology(",
            "create_ontology_node(",
            "create_ontology_edge(",
            "update_ontology(",
            "delete_ontology(",
            "infer_ontology(",
            "mutate_taxonomy(",
            "materialize_semantic_graph(",
            "mutate_domain_registry(",
            "apply_creative_lineage(",
            "apply_creative_dna(",
            "retrieve_memory(",
            "write_memory(",
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
