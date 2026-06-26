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
    build_assistant_workflow_graph,
    build_prompt_input_request,
    build_rendered_prompt_request,
    evaluation_engine_contract_by_id,
    stream_assistant_workflow_events,
)
from creative_coding_assistant.orchestration.evaluation_engine_contracts import (
    evaluation_engine_contracts,
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
    "creative_critic",
    "self_evaluation",
    "creative_improvement_planner",
    "reflection_loop",
    "creative_confidence",
    "creative_score",
    "consistency_validation",
    "evaluation_reports",
)

EXPECTED_HITL_SIGNALS = {
    "creative_critic": "hitl_questions",
    "self_evaluation": "hitl_questions",
    "creative_improvement_planner": "hitl_questions",
    "reflection_loop": "hitl_recommendation",
    "creative_confidence": "hitl_recommendation",
    "creative_score": "hitl_recommendation",
    "consistency_validation": "hitl_recommendation",
    "evaluation_reports": "hitl_recommendation",
}

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
    "downstream_dependencies",
    "upstream_dependencies",
    "evidence_contract",
    "cacheability",
    "parallelization_support",
    "estimated_cost_metadata",
    "estimated_latency_metadata",
    "serialization_version",
    "future_agent_hooks",
    "future_execution_hooks",
}


class EvaluationEngineContractTests(unittest.TestCase):
    def test_registry_exposes_consistent_contract_surface(self) -> None:
        registry = evaluation_engine_contracts()

        self.assertEqual(registry.role, "evaluation_engine_contract_registry")
        self.assertEqual(registry.engine_ids, EXPECTED_ENGINE_IDS)
        self.assertEqual(registry.contract_count, 8)
        self.assertEqual(
            {contract.engine_id for contract in registry.engine_contracts},
            set(EXPECTED_ENGINE_IDS),
        )
        for contract in registry.engine_contracts:
            dumped = contract.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_CONTRACT_FIELDS)
            self.assertEqual(contract.engine_version, "v3.4")
            self.assertEqual(contract.engine_category, "creative_evaluation")
            self.assertEqual(
                contract.serialization_version,
                "evaluation_engine_contract.v1",
            )
            self.assertTrue(contract.authority_boundary)
            self.assertTrue(contract.required_inputs)
            self.assertTrue(contract.produced_metadata)
            self.assertTrue(contract.produced_signals)
            self.assertTrue(contract.confidence_signals)
            declared_signal_sources = (
                set(contract.produced_signals)
                | set(contract.evidence_contract.evidence_payload_fields)
                | set(contract.evidence_contract.evidence_quality_signals)
            )
            self.assertLessEqual(
                set(contract.confidence_signals),
                declared_signal_sources,
            )
            self.assertTrue(contract.ambiguity_signals)
            self.assertLessEqual(
                set(contract.ambiguity_signals),
                declared_signal_sources,
            )
            hitl_signal = EXPECTED_HITL_SIGNALS[contract.engine_id]
            self.assertIn(hitl_signal, declared_signal_sources)
            self.assertIn(hitl_signal, contract.ambiguity_signals)
            self.assertTrue(contract.risk_signals)
            self.assertLessEqual(
                set(contract.risk_signals),
                declared_signal_sources,
            )
            self.assertTrue(contract.evidence_contract.evidence_sources)
            self.assertFalse(
                contract.estimated_cost_metadata.external_provider_calls
            )

    def test_contract_lookup_and_dependencies_are_stable(self) -> None:
        report_contract = evaluation_engine_contract_by_id("evaluation_reports")
        critic_contract = evaluation_engine_contract_by_id("creative_critic")
        missing_contract = evaluation_engine_contract_by_id("missing")

        self.assertIsNone(missing_contract)
        self.assertIsNotNone(report_contract)
        self.assertIsNotNone(critic_contract)
        assert report_contract is not None
        assert critic_contract is not None
        self.assertEqual(report_contract.downstream_dependencies, ())
        self.assertIn(
            "consistency_validation",
            report_contract.upstream_dependencies,
        )
        self.assertIn(
            "evaluation_reports",
            critic_contract.downstream_dependencies,
        )
        self.assertIn(
            "v3_5_inspector_report_surface",
            report_contract.future_execution_hooks,
        )

    def test_registry_serializes_for_workflow_metadata(self) -> None:
        dumped = evaluation_engine_contracts().model_dump(mode="json")

        self.assertEqual(
            dumped["serialization_version"],
            "evaluation_engine_contract_registry.v1",
        )
        self.assertEqual(dumped["engine_ids"], list(EXPECTED_ENGINE_IDS))
        self.assertEqual(len(dumped["engine_contracts"]), 8)
        self.assertEqual(
            dumped["engine_contracts"][0]["engine_id"],
            "creative_critic",
        )
        self.assertIn("evidence_contract", dumped["engine_contracts"][0])

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
        registry = evaluation_engine_contracts()
        prompt_input = prompt_input.model_copy(
            update={"evaluation_engine_contracts": registry}
        )
        rendered = JinjaPromptRenderer().render(
            build_rendered_prompt_request(
                route_decision=route_decision,
                prompt_input=prompt_input,
            )
        )

        self.assertEqual(prompt_input.evaluation_engine_contracts, registry)
        self.assertNotIn(
            "Evaluation Engine Contracts",
            rendered.sections[0].content,
        )
        self.assertNotIn(
            "evaluation_engine_contract_registry",
            rendered.sections[0].content,
        )

    def test_workflow_streams_contract_registry_metadata(self) -> None:
        graph = build_assistant_workflow_graph()
        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=_request(
                    query="Generate a luminous p5.js evaluation contract sketch.",
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
        director_event = _first_event(
            events,
            StreamEventType.PLANNING,
            "creative_director_prepared",
        )
        reasoning_event = _first_event(
            events,
            StreamEventType.PLANNING,
            "creative_reasoning_prepared",
        )
        final_event = events[-1]
        registry = planning_event.payload["evaluation_engine_contracts"]

        self.assertEqual(
            registry["role"],
            "evaluation_engine_contract_registry",
        )
        self.assertEqual(registry["engine_ids"], list(EXPECTED_ENGINE_IDS))
        self.assertEqual(registry["contract_count"], 8)
        self.assertTrue(
            planning_event.payload["workflow"][
                "evaluation_engine_contracts_available"
            ]
        )
        self.assertEqual(
            planning_event.payload["workflow"]["evaluation_engine_contracts"],
            registry,
        )
        self.assertEqual(final_event.payload["evaluation_engine_contracts"], registry)
        self.assertEqual(
            final_event.payload["workflow"]["evaluation_engine_contracts"],
            registry,
        )
        self.assertEqual(
            director_event.payload["creative_director"][
                "evaluation_engine_contracts"
            ],
            registry,
        )
        self.assertEqual(
            reasoning_event.payload["creative_reasoning"][
                "evaluation_engine_contracts"
            ],
            registry,
        )


if __name__ == "__main__":
    unittest.main()
