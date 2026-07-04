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
    ModelCapabilityMatrix,
    build_model_capability_matrix,
    model_capability_row_by_profile_id,
    model_capability_rows_for_route,
    route_request,
)
from creative_coding_assistant.orchestration.hybrid_studio import (
    model_profile_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_MODEL_CAPABILITY_ROW_FIELDS = {
    "row_id",
    "source_model_profile_id",
    "profile_name",
    "model_profile_kind",
    "route_applicability",
    "capability_dimensions",
    "provider_candidate_ids",
    "source_local_surface_ids",
    "source_cloud_surface_ids",
    "profile_inputs",
    "advisory_outputs",
    "route_count",
    "capability_dimension_count",
    "provider_candidate_count",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "model_capability_matrix_implemented",
    "capability_lookup_implemented",
    "capability_scoring_implemented",
    "quality_prediction_implemented",
    "cost_prediction_implemented",
    "model_selection_implemented",
    "automatic_model_selection_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "execution_policy_application_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class ModelCapabilityMatrixTests(unittest.TestCase):
    def test_default_matrix_projects_model_profile_registry(self) -> None:
        registry = model_profile_registry()
        matrix = build_model_capability_matrix(model_profiles=registry)

        self.assertEqual(matrix.role, "model_capability_matrix")
        self.assertEqual(matrix.serialization_version, "model_capability_matrix.v1")
        self.assertEqual(
            matrix.source_model_profile_serialization_version,
            registry.serialization_version,
        )
        self.assertEqual(matrix.row_count, 4)
        self.assertEqual(matrix.model_profile_count, 4)
        self.assertEqual(matrix.route_count, 6)
        self.assertEqual(matrix.capability_dimension_count, 12)
        self.assertEqual(matrix.provider_candidate_count, 5)
        self.assertIn("does not score capabilities", matrix.authority_boundary)
        self.assertIn(
            "creative_reasoning_metadata",
            matrix.capability_dimensions,
        )
        self.assertTrue(matrix.model_capability_matrix_implemented)
        self.assertTrue(matrix.capability_lookup_implemented)
        self.assertFalse(matrix.capability_scoring_implemented)
        self.assertFalse(matrix.quality_prediction_implemented)
        self.assertFalse(matrix.cost_prediction_implemented)
        self.assertFalse(matrix.model_selection_implemented)
        self.assertFalse(matrix.automatic_model_selection_implemented)
        self.assertFalse(matrix.provider_model_routing_implemented)
        self.assertFalse(matrix.provider_execution_implemented)
        self.assertFalse(matrix.execution_policy_application_implemented)
        self.assertFalse(matrix.workflow_control_implemented)
        self.assertFalse(matrix.retry_triggering_implemented)
        self.assertFalse(matrix.prompt_mutation_implemented)
        self.assertFalse(matrix.persistent_storage_write_implemented)
        self.assertFalse(matrix.generated_output_mutation_implemented)
        self.assertTrue(matrix.advisory_only)

    def test_matrix_rows_are_passive_capability_metadata(self) -> None:
        matrix = build_model_capability_matrix()

        for row in matrix.rows:
            dumped = row.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_MODEL_CAPABILITY_ROW_FIELDS)
            self.assertEqual(
                row.serialization_version,
                "model_capability_matrix_row.v1",
            )
            self.assertEqual(row.route_count, len(row.route_applicability))
            self.assertEqual(
                row.capability_dimension_count,
                len(row.capability_dimensions),
            )
            self.assertEqual(
                row.provider_candidate_count,
                len(row.provider_candidate_ids),
            )
            self.assertIn("capability_scoring", row.blocked_runtime_behaviors)
            self.assertTrue(row.model_capability_matrix_implemented)
            self.assertTrue(row.capability_lookup_implemented)
            self.assertFalse(row.capability_scoring_implemented)
            self.assertFalse(row.quality_prediction_implemented)
            self.assertFalse(row.cost_prediction_implemented)
            self.assertFalse(row.model_selection_implemented)
            self.assertFalse(row.automatic_model_selection_implemented)
            self.assertFalse(row.provider_model_routing_implemented)
            self.assertFalse(row.provider_execution_implemented)
            self.assertFalse(row.execution_policy_application_implemented)
            self.assertFalse(row.workflow_control_implemented)
            self.assertFalse(row.retry_triggering_implemented)
            self.assertFalse(row.prompt_mutation_implemented)
            self.assertFalse(row.generated_output_mutation_implemented)
            self.assertTrue(row.advisory_only)

    def test_lookup_helpers_return_rows_without_scoring_or_selection(self) -> None:
        matrix = build_model_capability_matrix()
        creative = model_capability_row_by_profile_id(
            "creative_reasoning_model_profile",
            matrix,
        )
        generate_rows = model_capability_rows_for_route(RouteName.GENERATE, matrix)
        missing = model_capability_row_by_profile_id("missing", matrix)

        self.assertIsNone(missing)
        self.assertIsNotNone(creative)
        assert creative is not None
        self.assertEqual(
            creative.row_id,
            "model_capability::creative_reasoning_model_profile",
        )
        self.assertIn(creative, generate_rows)

    def test_matrix_rejects_mismatched_rows_or_counts(self) -> None:
        matrix = build_model_capability_matrix()
        payload = matrix.model_dump(mode="json")
        payload["row_ids"] = ("missing",) + tuple(payload["row_ids"][1:])

        with self.assertRaisesRegex(ValueError, "row_ids must match"):
            ModelCapabilityMatrix(**payload)

        payload = matrix.model_dump(mode="json")
        payload["model_profile_ids"] = ("missing",) + tuple(
            payload["model_profile_ids"][1:]
        )

        with self.assertRaisesRegex(ValueError, "model_profile_ids must match"):
            ModelCapabilityMatrix(**payload)

        payload = matrix.model_dump(mode="json")
        payload["capability_dimension_count"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "capability_dimension_count must match",
        ):
            ModelCapabilityMatrix(**payload)

    def test_matrix_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Generate a capability matrix for model metadata.",
            mode=AssistantMode.EXPLAIN,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        matrix = build_model_capability_matrix()
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(matrix.route_names, tuple(RouteName))
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_matrix_does_not_declare_runtime_application_terms(self) -> None:
        matrix = build_model_capability_matrix()
        combined_text = " ".join(
            (
                matrix.authority_boundary,
                *matrix.blocked_runtime_behaviors,
                *matrix.advisory_actions,
                *(
                    field
                    for row in matrix.rows
                    for field in (
                        row.row_id,
                        row.source_model_profile_id,
                        row.profile_name,
                        *row.capability_dimensions,
                        *row.provider_candidate_ids,
                        *row.profile_inputs,
                        *row.advisory_outputs,
                        *row.evidence,
                        *row.advisory_actions,
                        *row.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "score_capability(",
            "predict_quality(",
            "predict_cost(",
            "select_model(",
            "switch_model(",
            "route_provider(",
            "execute_provider(",
            "apply_execution_policy(",
            "control_workflow(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
