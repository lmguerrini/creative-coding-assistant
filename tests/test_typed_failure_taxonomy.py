import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
    StreamEventType,
)
from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import (
    OpenAIGenerationProvider,
    build_generation_provider,
)
from creative_coding_assistant.orchestration import (
    TYPED_FAILURE_TAXONOMY_ROADMAP_ITEMS,
    FailureTypeDefinition,
    TypedFailureTaxonomyRegistry,
    build_typed_failure_taxonomy_registry,
    explain_failure_type,
    failure_type_by_id,
    failure_types_for_domain,
    failure_types_for_owner,
    failure_types_for_root_cause,
    failure_types_for_severity,
    node_failure_model_by_id,
    recovery_strategy_by_id,
    regression_scenario_by_id,
    route_request,
    validate_typed_failure_taxonomy,
)
from creative_coding_assistant.orchestration.workflow_graph import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
    assistant_workflow_model_payload_specs,
)
from creative_coding_assistant.orchestration.workflow_review import (
    MAX_WORKFLOW_REFINEMENT_COUNT,
)


class TypedFailureTaxonomyTests(unittest.TestCase):
    def test_registry_covers_v7_2_roadmap_without_runtime_behavior(self) -> None:
        registry = build_typed_failure_taxonomy_registry()
        validation = validate_typed_failure_taxonomy(registry)

        self.assertEqual(registry.role, "typed_failure_taxonomy")
        self.assertEqual(registry.serialization_version, "typed_failure_taxonomy.v1")
        self.assertEqual(
            registry.covered_roadmap_items,
            TYPED_FAILURE_TAXONOMY_ROADMAP_ITEMS,
        )
        self.assertEqual(registry.roadmap_item_count, 18)
        self.assertEqual(registry.failure_type_count, 13)
        self.assertEqual(
            registry.source_workflow_node_order,
            ASSISTANT_WORKFLOW_NODE_ORDER,
        )
        self.assertEqual(
            registry.node_failure_model_count,
            len(ASSISTANT_WORKFLOW_NODE_ORDER),
        )
        self.assertEqual(
            registry.planning_sub_helper_model_count,
            len(assistant_workflow_model_payload_specs()),
        )
        self.assertTrue(validation.validation_passed)
        self.assertFalse(validation.orphaned_failure_type_ids)
        self.assertFalse(validation.orphaned_strategy_ids)
        self.assertFalse(validation.orphaned_event_contract_ids)
        self.assertIn(
            "live_failure_classification",
            registry.blocked_runtime_behaviors,
        )
        self.assertFalse(registry.live_failure_classification_implemented)
        self.assertFalse(registry.exception_interception_implemented)
        self.assertFalse(registry.failure_recovery_execution_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.provider_execution_implemented)
        self.assertFalse(registry.workflow_execution_implemented)
        self.assertFalse(registry.workflow_graph_mutation_implemented)
        self.assertFalse(registry.persistent_storage_write_implemented)
        self.assertFalse(registry.runtime_evolution_implemented)
        self.assertTrue(registry.advisory_only)

    def test_failure_type_registry_lookup_dimensions_are_stable(self) -> None:
        registry = build_typed_failure_taxonomy_registry()
        provider = failure_type_by_id(
            "typed_failure::provider_stream::provider_generation_error",
            registry,
        )

        self.assertIsNotNone(provider)
        assert provider is not None
        self.assertEqual(provider.domain, "provider_stream")
        self.assertEqual(provider.severity, "terminal")
        self.assertEqual(provider.root_cause, "provider_error")
        self.assertEqual(provider.owner, "provider_adapter")
        self.assertEqual(
            provider.recovery_strategy_id,
            "recovery_strategy::safe_terminal_answer",
        )
        self.assertTrue(provider.terminal)
        self.assertFalse(provider.retry_eligible)
        self.assertEqual(
            explain_failure_type(provider.failure_type_id, registry),
            provider.explanation,
        )
        self.assertEqual(len(failure_types_for_domain("provider_stream", registry)), 2)
        self.assertGreaterEqual(
            len(failure_types_for_severity("terminal", registry)),
            1,
        )
        self.assertEqual(
            tuple(
                failure.failure_type_id
                for failure in failure_types_for_root_cause(
                    "provider_error",
                    registry,
                )
            ),
            (
                "typed_failure::provider_stream::provider_generation_error",
                "typed_failure::provider_stream::stream_interrupted",
            ),
        )
        self.assertEqual(
            len(failure_types_for_owner("provider_adapter", registry)),
            2,
        )

    def test_node_and_planning_helper_models_cover_existing_workflow(self) -> None:
        registry = build_typed_failure_taxonomy_registry()
        planning = node_failure_model_by_id("planning", registry)
        generation = node_failure_model_by_id("generation", registry)
        missing = node_failure_model_by_id("missing", registry)

        self.assertIsNone(missing)
        self.assertIsNotNone(planning)
        self.assertIsNotNone(generation)
        assert planning is not None
        assert generation is not None
        self.assertEqual(
            tuple(model.node_id for model in registry.node_failure_models),
            ASSISTANT_WORKFLOW_NODE_ORDER,
        )
        self.assertIn(
            "typed_failure::planning_sub_helper::helper_unavailable",
            planning.supported_failure_type_ids,
        )
        self.assertIn(
            "typed_failure::provider_stream::provider_generation_error",
            generation.supported_failure_type_ids,
        )
        self.assertFalse(planning.node_handler_invocation_implemented)
        self.assertFalse(generation.workflow_execution_implemented)

        helper_ids = tuple(
            helper.helper_id for helper in registry.planning_sub_helper_models
        )
        self.assertEqual(
            helper_ids,
            tuple(
                spec.payload_key for spec in assistant_workflow_model_payload_specs()
            ),
        )
        contract_helpers = tuple(
            helper
            for helper in registry.planning_sub_helper_models
            if helper.helper_id == "artifact_engine_contracts"
        )
        self.assertEqual(len(contract_helpers), 1)
        self.assertIn(
            "typed_failure::planning_sub_helper::planning_contract_violation",
            contract_helpers[0].supported_failure_type_ids,
        )
        self.assertTrue(
            all(
                not helper.helper_invocation_implemented
                for helper in registry.planning_sub_helper_models
            )
        )

    def test_provider_serialization_client_event_boundaries_are_advisory(
        self,
    ) -> None:
        registry = build_typed_failure_taxonomy_registry()
        error_event = next(
            contract
            for contract in registry.event_contracts
            if contract.event_type is StreamEventType.ERROR
        )
        node_failed_event = next(
            contract
            for contract in registry.event_contracts
            if contract.event_type is StreamEventType.NODE_FAILED
        )

        self.assertEqual(error_event.required_payload_keys, ("code", "message"))
        self.assertEqual(
            node_failed_event.required_payload_keys,
            ("node", "error_code", "error_message"),
        )
        self.assertTrue(
            all(
                not model.provider_execution_implemented
                and not model.stream_subscription_implemented
                and not model.provider_model_routing_implemented
                for model in registry.provider_stream_models
            )
        )
        self.assertTrue(
            all(
                not model.serialization_execution_implemented
                and not model.persistent_storage_write_implemented
                for model in registry.serialization_models
            )
        )
        self.assertTrue(
            all(
                not model.client_code_execution_implemented
                and not model.server_request_mutation_implemented
                for model in registry.workstation_client_boundary_models
            )
        )

    def test_recovery_regression_ownership_fix_and_kb_links_are_complete(self) -> None:
        registry = build_typed_failure_taxonomy_registry()
        strategy = recovery_strategy_by_id(
            "recovery_strategy::retry_budget_review",
            registry,
        )

        self.assertIsNotNone(strategy)
        assert strategy is not None
        self.assertTrue(strategy.retry_allowed)
        self.assertEqual(strategy.max_retry_attempts, MAX_WORKFLOW_REFINEMENT_COUNT)
        self.assertFalse(strategy.retry_triggering_implemented)

        scenario_ids = {
            scenario.scenario_id for scenario in registry.regression_scenarios
        }
        kb_ids = {entry.kb_entry_id for entry in registry.knowledge_base_entries}
        fix_ids = {
            recommendation.recommendation_id
            for recommendation in registry.fix_recommendations
        }
        ownership_ids = {
            failure_id
            for owner in registry.ownership_records
            for failure_id in owner.failure_type_ids
        }
        for failure in registry.failure_types:
            self.assertIn(failure.regression_scenario_id, scenario_ids)
            self.assertIn(failure.knowledge_base_entry_id, kb_ids)
            self.assertIn(failure.fix_recommendation_id, fix_ids)
            self.assertIn(failure.failure_type_id, ownership_ids)
            scenario = regression_scenario_by_id(
                failure.regression_scenario_id,
                registry,
            )
            self.assertIsNotNone(scenario)
            assert scenario is not None
            self.assertEqual(scenario.expected_owner, failure.owner)
            self.assertFalse(scenario.workflow_execution_implemented)
            self.assertFalse(scenario.provider_execution_implemented)

    def test_registry_rejects_mismatched_or_unbounded_metadata(self) -> None:
        registry = build_typed_failure_taxonomy_registry()
        payload = registry.model_dump(mode="json")
        payload["failure_type_ids"] = (
            "missing",
            *tuple(payload["failure_type_ids"][1:]),
        )

        with self.assertRaisesRegex(ValueError, "failure_type_ids must match"):
            TypedFailureTaxonomyRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["covered_roadmap_items"] = (
            "missing",
            *tuple(payload["covered_roadmap_items"][1:]),
        )

        with self.assertRaisesRegex(
            ValueError,
            "covered_roadmap_items must match",
        ):
            TypedFailureTaxonomyRegistry(**payload)

        failure_payload = registry.failure_types[0].model_dump(mode="json")
        failure_payload["retry_eligible"] = True

        with self.assertRaisesRegex(
            ValueError,
            "terminal failure types cannot be retry eligible",
        ):
            FailureTypeDefinition(**failure_payload)

        failure_payload = registry.failure_types[0].model_dump(mode="json")
        failure_payload["workflow_node_ids"] = ("missing",)

        with self.assertRaisesRegex(
            ValueError,
            "workflow_node_ids must reference known nodes",
        ):
            FailureTypeDefinition(**failure_payload)

        payload = registry.model_dump(mode="json")
        payload["provider_stream_models"][0]["fallback_strategy_ids"] = ("missing",)

        with self.assertRaisesRegex(
            ValueError,
            "provider fallback_strategy_ids must be known",
        ):
            TypedFailureTaxonomyRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["event_contracts"][0]["mapped_failure_type_ids"] = ("missing",)

        with self.assertRaisesRegex(
            ValueError,
            "event mapped_failure_type_ids must be known",
        ):
            TypedFailureTaxonomyRegistry(**payload)

        payload = registry.model_dump(mode="json")
        payload["knowledge_base_entries"][0]["linked_fix_recommendation_id"] = "missing"

        with self.assertRaisesRegex(
            ValueError,
            "knowledge linked_fix_recommendation_id must be known",
        ):
            TypedFailureTaxonomyRegistry(**payload)

    def test_taxonomy_does_not_change_routing_or_provider_factory(self) -> None:
        request = AssistantRequest(
            query="Classify failure metadata for a p5.js workflow.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )
        baseline_decision = route_request(request)

        registry = build_typed_failure_taxonomy_registry()
        next_decision = route_request(request)
        provider = build_generation_provider(settings)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(registry.failure_type_count, 13)
        self.assertIsInstance(provider, OpenAIGenerationProvider)
        self.assertEqual(provider._model, "gpt-5")
        self.assertEqual(provider._settings.default_generation_provider, "openai")


if __name__ == "__main__":
    unittest.main()
