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
    ProviderCapabilityMatrix,
    build_provider_capability_matrix,
    provider_capability_row_by_profile_id,
    provider_capability_rows_for_provider,
    provider_capability_rows_for_route,
    route_request,
)
from creative_coding_assistant.orchestration.hybrid_studio import (
    provider_selection_registry,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_PROVIDER_CAPABILITY_ROW_FIELDS = {
    "row_id",
    "source_provider_selection_profile_id",
    "profile_name",
    "provider_selection_posture",
    "provider_candidate_ids",
    "source_local_surface_ids",
    "source_cloud_surface_ids",
    "source_auto_mode_profile_ids",
    "source_hitl_decision_profile_ids",
    "route_applicability",
    "selection_inputs",
    "advisory_outputs",
    "provider_candidate_count",
    "route_count",
    "local_surface_count",
    "cloud_surface_count",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "provider_capability_matrix_implemented",
    "provider_capability_lookup_implemented",
    "provider_selection_implemented",
    "automatic_provider_selection_implemented",
    "automatic_model_selection_implemented",
    "model_switching_implemented",
    "provider_model_routing_implemented",
    "local_provider_execution_implemented",
    "cloud_provider_execution_implemented",
    "human_input_request_implemented",
    "workflow_control_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}


class ProviderCapabilityMatrixTests(unittest.TestCase):
    def test_default_matrix_projects_provider_selection_registry(self) -> None:
        registry = provider_selection_registry()
        matrix = build_provider_capability_matrix(provider_selection=registry)

        self.assertEqual(matrix.role, "provider_capability_matrix")
        self.assertEqual(matrix.serialization_version, "provider_capability_matrix.v1")
        self.assertEqual(
            matrix.source_provider_selection_serialization_version,
            registry.serialization_version,
        )
        self.assertEqual(matrix.row_count, 4)
        self.assertEqual(matrix.provider_selection_profile_count, 4)
        self.assertEqual(matrix.provider_selection_posture_count, 4)
        self.assertEqual(matrix.provider_candidate_count, 5)
        self.assertEqual(matrix.route_count, 6)
        self.assertIn("does not select providers", matrix.authority_boundary)
        self.assertIn("openai", matrix.provider_candidate_ids)
        self.assertTrue(matrix.provider_capability_matrix_implemented)
        self.assertTrue(matrix.provider_capability_lookup_implemented)
        self.assertFalse(matrix.provider_selection_implemented)
        self.assertFalse(matrix.automatic_provider_selection_implemented)
        self.assertFalse(matrix.automatic_model_selection_implemented)
        self.assertFalse(matrix.model_switching_implemented)
        self.assertFalse(matrix.provider_model_routing_implemented)
        self.assertFalse(matrix.local_provider_execution_implemented)
        self.assertFalse(matrix.cloud_provider_execution_implemented)
        self.assertFalse(matrix.human_input_request_implemented)
        self.assertFalse(matrix.workflow_control_implemented)
        self.assertFalse(matrix.retry_triggering_implemented)
        self.assertFalse(matrix.prompt_mutation_implemented)
        self.assertFalse(matrix.persistent_storage_write_implemented)
        self.assertFalse(matrix.generated_output_mutation_implemented)
        self.assertTrue(matrix.advisory_only)

    def test_provider_capability_rows_are_passive_metadata(self) -> None:
        matrix = build_provider_capability_matrix()

        for row in matrix.rows:
            dumped = row.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROVIDER_CAPABILITY_ROW_FIELDS)
            self.assertEqual(
                row.serialization_version,
                "provider_capability_matrix_row.v1",
            )
            self.assertEqual(
                row.provider_candidate_count,
                len(row.provider_candidate_ids),
            )
            self.assertEqual(row.route_count, len(row.route_applicability))
            self.assertEqual(
                row.local_surface_count,
                len(row.source_local_surface_ids),
            )
            self.assertEqual(
                row.cloud_surface_count,
                len(row.source_cloud_surface_ids),
            )
            self.assertIn(
                "automatic_provider_selection",
                row.blocked_runtime_behaviors,
            )
            self.assertTrue(row.provider_capability_matrix_implemented)
            self.assertTrue(row.provider_capability_lookup_implemented)
            self.assertFalse(row.provider_selection_implemented)
            self.assertFalse(row.automatic_provider_selection_implemented)
            self.assertFalse(row.automatic_model_selection_implemented)
            self.assertFalse(row.model_switching_implemented)
            self.assertFalse(row.provider_model_routing_implemented)
            self.assertFalse(row.local_provider_execution_implemented)
            self.assertFalse(row.cloud_provider_execution_implemented)
            self.assertFalse(row.human_input_request_implemented)
            self.assertFalse(row.workflow_control_implemented)
            self.assertFalse(row.retry_triggering_implemented)
            self.assertFalse(row.prompt_mutation_implemented)
            self.assertFalse(row.generated_output_mutation_implemented)
            self.assertTrue(row.advisory_only)

    def test_lookup_helpers_return_rows_without_selecting_providers(self) -> None:
        matrix = build_provider_capability_matrix()
        current = provider_capability_row_by_profile_id(
            "current_config_provider_visibility_profile",
            matrix,
        )
        openai_rows = provider_capability_rows_for_provider("openai", matrix)
        review_rows = provider_capability_rows_for_route(RouteName.REVIEW, matrix)
        missing = provider_capability_row_by_profile_id("missing", matrix)

        self.assertIsNone(missing)
        self.assertIsNotNone(current)
        assert current is not None
        self.assertEqual(
            current.row_id,
            "provider_capability::current_config_provider_visibility_profile",
        )
        self.assertIn(current, openai_rows)
        self.assertIn(current, review_rows)

    def test_matrix_rejects_mismatched_rows_or_counts(self) -> None:
        matrix = build_provider_capability_matrix()
        payload = matrix.model_dump(mode="json")
        payload["row_ids"] = ("missing",) + tuple(payload["row_ids"][1:])

        with self.assertRaisesRegex(ValueError, "row_ids must match"):
            ProviderCapabilityMatrix(**payload)

        payload = matrix.model_dump(mode="json")
        payload["provider_selection_profile_ids"] = (
            "missing",
        ) + tuple(payload["provider_selection_profile_ids"][1:])

        with self.assertRaisesRegex(
            ValueError,
            "provider_selection_profile_ids must match",
        ):
            ProviderCapabilityMatrix(**payload)

        payload = matrix.model_dump(mode="json")
        payload["provider_candidate_count"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "provider_candidate_count must match",
        ):
            ProviderCapabilityMatrix(**payload)

    def test_matrix_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Explain provider capability metadata.",
            mode=AssistantMode.EXPLAIN,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )

        matrix = build_provider_capability_matrix()
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(matrix.route_names, tuple(RouteName))
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")

    def test_matrix_does_not_declare_runtime_application_terms(self) -> None:
        matrix = build_provider_capability_matrix()
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
                        row.source_provider_selection_profile_id,
                        row.profile_name,
                        *row.provider_candidate_ids,
                        *row.selection_inputs,
                        *row.advisory_outputs,
                        *row.evidence,
                        *row.advisory_actions,
                        *row.blocked_runtime_behaviors,
                    )
                ),
            )
        )

        for forbidden_term in (
            "select_provider(",
            "select_model(",
            "switch_model(",
            "route_provider(",
            "execute_provider(",
            "execute_local_provider(",
            "execute_cloud_provider(",
            "request_hitl(",
            "control_workflow(",
            "trigger_retry(",
            "mutate_prompt(",
            "write_storage(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
