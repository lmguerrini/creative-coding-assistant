import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
    JinjaPromptRenderer,
    StructuredPromptInputBuilder,
    agent_capability_registry,
    build_assistant_workflow_graph,
    build_prompt_input_request,
    build_rendered_prompt_request,
    engine_contract_consistency_registry,
    escalation_policy_registry,
    hybrid_agentic_workflow_registry,
    stream_assistant_workflow_events,
)
from test_langgraph_workflow_integration import (
    _request,
    _route_generate,
    _runtime,
    _stream_completed_generation,
    _stream_prompt_inputs_with_builder,
)

FUTURE_READINESS_REGISTRY_KEYS = (
    "agent_capability_registry",
    "escalation_policy_registry",
    "hybrid_agentic_workflow_registry",
    "engine_contract_consistency_registry",
)

FUTURE_RUNTIME_MARKERS = (
    "v4_agent",
    "v5_execution",
    "v6_learning",
    "agent_invocation",
    "autonomous_retry",
    "runtime_auto_selection",
    "execute_provider",
)


class V3FinalHardeningBoundaryTests(unittest.TestCase):
    def test_future_readiness_registries_remain_export_only_metadata(self) -> None:
        registries = (
            agent_capability_registry(),
            escalation_policy_registry(),
            hybrid_agentic_workflow_registry(),
            engine_contract_consistency_registry(),
        )

        self.assertEqual(
            tuple(registry.role for registry in registries),
            FUTURE_READINESS_REGISTRY_KEYS,
        )
        for registry in registries:
            self.assertTrue(registry.metadata_only)
            self.assertIn("does not", registry.authority_boundary)

    def test_future_readiness_registries_do_not_enter_workflow_payloads(self) -> None:
        graph = build_assistant_workflow_graph()
        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=_request(
                    query="Generate a compact p5.js V3 hardening sketch.",
                    domain=CreativeCodingDomain.P5_JS,
                ),
                runtime=_runtime(
                    stream_prompt_inputs=_stream_prompt_inputs_with_builder,
                    stream_generation=_stream_completed_generation,
                ),
            )
        )

        for event in events:
            for registry_key in FUTURE_READINESS_REGISTRY_KEYS:
                self.assertNotIn(registry_key, event.payload)
                workflow_payload = event.payload.get("workflow")
                if isinstance(workflow_payload, dict):
                    self.assertNotIn(registry_key, workflow_payload)

    def test_future_readiness_registries_do_not_enter_prompt_guidance(self) -> None:
        request = AssistantRequest(
            query="Generate a small V3 hardening sketch.",
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
        rendered = JinjaPromptRenderer().render(
            build_rendered_prompt_request(
                route_decision=route_decision,
                prompt_input=prompt_input,
            )
        )
        prompt_input_dump = prompt_input.model_dump(mode="json")
        rendered_text = "\n".join(section.content for section in rendered.sections)

        for registry_key in FUTURE_READINESS_REGISTRY_KEYS:
            self.assertNotIn(registry_key, prompt_input_dump)
            self.assertNotIn(registry_key, rendered_text)

    def test_v3_workflow_node_order_has_no_future_runtime_nodes(self) -> None:
        self.assertEqual(
            ASSISTANT_WORKFLOW_NODE_ORDER,
            (
                "intake",
                "routing",
                "memory",
                "retrieval",
                "context_assembly",
                "prompt_input",
                "planning",
                "director",
                "reasoning",
                "prompt_rendering",
                "generation",
                "artifact_extraction",
                "preview_preparation",
                "artifact_critique",
                "review",
                "refinement",
                "finalization",
                "failure",
            ),
        )
        for node_id in ASSISTANT_WORKFLOW_NODE_ORDER:
            for future_marker in FUTURE_RUNTIME_MARKERS:
                self.assertNotIn(future_marker, node_id)

    def test_future_readiness_metadata_does_not_declare_runtime_actions(self) -> None:
        registry_dumps = (
            agent_capability_registry().model_dump(mode="json"),
            escalation_policy_registry().model_dump(mode="json"),
            hybrid_agentic_workflow_registry().model_dump(mode="json"),
            engine_contract_consistency_registry().model_dump(mode="json"),
        )
        combined_text = " ".join(str(registry_dump) for registry_dump in registry_dumps)

        for forbidden_runtime_action in (
            "execute_provider",
            "autonomous_retry",
            "runtime_auto_selection",
        ):
            self.assertNotIn(forbidden_runtime_action, combined_text)


if __name__ == "__main__":
    unittest.main()
