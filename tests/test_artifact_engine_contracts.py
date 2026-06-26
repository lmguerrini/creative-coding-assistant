import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
    StreamEventType,
)
from creative_coding_assistant.orchestration import (
    JinjaPromptRenderer,
    StructuredPromptInputBuilder,
    artifact_intelligence_engine_contract_by_id,
    artifact_intelligence_engine_contracts,
    build_assistant_workflow_graph,
    build_prompt_input_request,
    build_rendered_prompt_request,
    stream_assistant_workflow_events,
)
from test_langgraph_workflow_integration import (
    _first_event,
    _request,
    _route_generate,
    _runtime,
    _stream_completed_generation,
    _stream_prompt_inputs_with_builder,
)

EXPECTED_ENGINE_IDS = (
    "artifact_planner",
    "artifact_dependency_graph",
    "runtime_compatibility_engine",
    "artifact_capability_matrix",
    "multi_artifact_strategy",
    "artifact_critic",
    "artifact_refiner",
    "artifact_intelligence_synthesis",
    "artifact_merge_planner",
    "artifact_export_intelligence",
)

REQUIRED_CONTRACT_FIELDS = {
    "engine_id",
    "engine_name",
    "engine_version",
    "engine_category",
    "authority_boundary",
    "required_inputs",
    "optional_inputs",
    "produced_metadata",
    "produced_signals",
    "confidence_signals",
    "ambiguity_signals",
    "risk_signals",
    "escalation_candidates",
    "downstream_dependencies",
    "upstream_dependencies",
    "cacheability",
    "parallelization_support",
    "estimated_cost_metadata",
    "estimated_latency_metadata",
    "serialization_version",
    "future_agent_hooks",
    "future_execution_hooks",
}


class ArtifactEngineContractTests(unittest.TestCase):
    def test_registry_exposes_consistent_contract_surface(self) -> None:
        registry = artifact_intelligence_engine_contracts()

        self.assertEqual(
            registry.role,
            "artifact_intelligence_engine_contract_registry",
        )
        self.assertEqual(registry.engine_ids, EXPECTED_ENGINE_IDS)
        self.assertEqual(registry.contract_count, 10)
        self.assertEqual(
            {contract.engine_id for contract in registry.engine_contracts},
            set(EXPECTED_ENGINE_IDS),
        )
        for contract in registry.engine_contracts:
            dumped = contract.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CONTRACT_FIELDS)
            self.assertEqual(contract.engine_version, "v3.3")
            self.assertEqual(contract.engine_category, "artifact_intelligence")
            self.assertEqual(
                contract.serialization_version,
                "artifact_engine_contract.v1",
            )
            self.assertTrue(contract.authority_boundary)
            self.assertTrue(contract.produced_metadata)
            self.assertTrue(contract.produced_signals)
            self.assertTrue(contract.confidence_signals)
            self.assertLessEqual(
                set(contract.confidence_signals),
                set(contract.produced_signals),
            )
            self.assertTrue(contract.ambiguity_signals)
            self.assertLessEqual(
                set(contract.ambiguity_signals),
                set(contract.produced_signals),
            )
            self.assertIn("hitl_questions", contract.produced_signals)
            self.assertIn("hitl_questions", contract.ambiguity_signals)
            self.assertIn("hitl_questions", contract.escalation_candidates)
            self.assertTrue(contract.risk_signals)
            self.assertLessEqual(
                set(contract.risk_signals),
                set(contract.produced_signals),
            )
            self.assertTrue(contract.escalation_candidates)
            self.assertLessEqual(
                set(contract.escalation_candidates),
                set(contract.produced_signals),
            )
            if contract.upstream_dependencies:
                self.assertEqual(
                    contract.cacheability,
                    "deterministic_with_upstream_metadata",
                )
                self.assertIn(
                    "upstream",
                    contract.estimated_cost_metadata.cache_sensitivity.lower(),
                )
            else:
                self.assertEqual(contract.cacheability, "deterministic_per_request")
            cost_metadata = contract.estimated_cost_metadata
            self.assertEqual(cost_metadata.relative_cost, "low")
            self.assertFalse(cost_metadata.external_provider_calls)
            self.assertIn("no provider", cost_metadata.cost_basis.lower())

    def test_contract_lookup_and_dependencies_are_stable(self) -> None:
        export_contract = artifact_intelligence_engine_contract_by_id(
            "artifact_export_intelligence"
        )
        missing_contract = artifact_intelligence_engine_contract_by_id("missing")

        self.assertIsNone(missing_contract)
        self.assertIsNotNone(export_contract)
        assert export_contract is not None
        self.assertIn("artifact_merge_planner", export_contract.upstream_dependencies)
        self.assertEqual(export_contract.downstream_dependencies, ())
        self.assertIn(
            "v6_blueprint_export_readiness",
            export_contract.future_execution_hooks,
        )

    def test_parallelization_metadata_tracks_ordered_dependencies(self) -> None:
        registry = artifact_intelligence_engine_contracts()
        engine_order = {
            engine_id: index for index, engine_id in enumerate(registry.engine_ids)
        }

        for contract in registry.engine_contracts:
            self.assertEqual(
                contract.parallelization_support,
                "requires_ordered_upstream_metadata",
            )
            for upstream_dependency in contract.upstream_dependencies:
                upstream_index = engine_order.get(upstream_dependency)
                if upstream_index is not None:
                    self.assertLess(upstream_index, engine_order[contract.engine_id])

    def test_registry_serializes_for_workflow_metadata(self) -> None:
        dumped = artifact_intelligence_engine_contracts().model_dump(mode="json")

        self.assertEqual(
            dumped["serialization_version"],
            "artifact_engine_contract_registry.v1",
        )
        self.assertEqual(dumped["engine_ids"], list(EXPECTED_ENGINE_IDS))
        self.assertEqual(len(dumped["engine_contracts"]), 10)
        self.assertEqual(
            dumped["engine_contracts"][0]["engine_id"],
            "artifact_planner",
        )

    def test_prompt_input_can_carry_registry_without_rendering_contract_text(
        self,
    ) -> None:
        request = AssistantRequest(
            query="Generate a small p5.js sketch.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.P5_JS,
        )
        route_decision = _route_generate(request)
        prompt_input = StructuredPromptInputBuilder().build(
            build_prompt_input_request(
                assistant_request=request,
                route_decision=route_decision,
                assembled_context=None,
            )
        )
        registry = artifact_intelligence_engine_contracts()
        prompt_input = prompt_input.model_copy(
            update={"artifact_engine_contracts": registry}
        )
        rendered = JinjaPromptRenderer().render(
            build_rendered_prompt_request(
                route_decision=route_decision,
                prompt_input=prompt_input,
            )
        )

        self.assertEqual(prompt_input.artifact_engine_contracts, registry)
        self.assertNotIn(
            "Artifact Intelligence Engine Contracts",
            rendered.sections[0].content,
        )
        self.assertNotIn(
            "artifact_engine_contract_registry",
            rendered.sections[0].content,
        )

    def test_workflow_streams_contract_registry_metadata(self) -> None:
        graph = build_assistant_workflow_graph()
        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=_request(
                    query="Generate a luminous p5.js contract metadata sketch.",
                    domain=CreativeCodingDomain.P5_JS,
                ),
                runtime=_runtime(
                    stream_prompt_inputs=_stream_prompt_inputs_with_builder,
                    stream_generation=_stream_completed_generation,
                ),
            )
        )

        planning_event = _first_event(
            events,
            StreamEventType.PLANNING,
            "creative_plan_prepared",
        )
        final_event = events[-1]
        registry = planning_event.payload["artifact_engine_contracts"]

        self.assertEqual(
            registry["role"],
            "artifact_intelligence_engine_contract_registry",
        )
        self.assertEqual(registry["engine_ids"], list(EXPECTED_ENGINE_IDS))
        self.assertEqual(registry["contract_count"], 10)
        self.assertTrue(
            planning_event.payload["workflow"]["artifact_engine_contracts_available"]
        )
        self.assertEqual(
            planning_event.payload["workflow"]["artifact_engine_contracts"],
            registry,
        )
        self.assertEqual(final_event.payload["artifact_engine_contracts"], registry)
        self.assertEqual(
            final_event.payload["workflow"]["artifact_engine_contracts"],
            registry,
        )


if __name__ == "__main__":
    unittest.main()
