import json
import unittest

from creative_coding_assistant.orchestration import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
    JinjaPromptRenderer,
    StructuredPromptInputBuilder,
    agent_boundary_registry,
    agent_contract_registry,
    agent_metadata_registry,
    agent_routing_registry,
    agent_role_registry,
    build_assistant_workflow_graph,
    build_prompt_input_request,
    build_rendered_prompt_request,
    stream_assistant_workflow_events,
)
from test_langgraph_workflow_integration import (
    _request,
    _route_generate,
    _runtime,
    _stream_completed_generation,
    _stream_prompt_inputs_with_builder,
)

AGENT_REGISTRY_MARKERS = (
    "agent_contract_registry",
    "agent_role_registry",
    "agent_boundary_registry",
    "agent_metadata_registry",
    "agent_routing_registry",
)


class AgentContractIntegrationTests(unittest.TestCase):
    def test_passive_agent_registries_are_exported(self) -> None:
        contract_registry = agent_contract_registry()
        role_registry = agent_role_registry()
        boundary_registry = agent_boundary_registry()
        metadata_registry = agent_metadata_registry()
        routing_registry = agent_routing_registry()

        self.assertEqual(contract_registry.contract_count, 12)
        self.assertEqual(role_registry.role_count, 12)
        self.assertEqual(boundary_registry.boundary_count, 12)
        self.assertEqual(metadata_registry.metadata_count, 12)
        self.assertEqual(routing_registry.profile_count, 12)
        self.assertEqual(role_registry.agent_ids, contract_registry.agent_ids)
        self.assertEqual(boundary_registry.agent_ids, contract_registry.agent_ids)
        self.assertEqual(metadata_registry.agent_ids, contract_registry.agent_ids)
        self.assertEqual(routing_registry.agent_ids, contract_registry.agent_ids)
        self.assertTrue(contract_registry.metadata_only)
        self.assertTrue(role_registry.metadata_only)
        self.assertTrue(boundary_registry.metadata_only)
        self.assertTrue(metadata_registry.metadata_only)
        self.assertTrue(routing_registry.metadata_only)

    def test_agent_registries_do_not_change_workflow_nodes_or_payloads(self) -> None:
        graph = build_assistant_workflow_graph()
        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=_request(query="Generate a passive contract test scene."),
                runtime=_runtime(
                    stream_prompt_inputs=_stream_prompt_inputs_with_builder,
                    stream_generation=_stream_completed_generation,
                ),
            )
        )
        payload_text = json.dumps(
            [event.payload for event in events],
            sort_keys=True,
        )

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
        for marker in AGENT_REGISTRY_MARKERS:
            self.assertNotIn(marker, payload_text)

    def test_agent_registries_do_not_render_into_prompts(self) -> None:
        request = _request(query="Generate a passive prompt boundary scene.")
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
        rendered_text = "\n".join(section.content for section in rendered.sections)

        for marker in AGENT_REGISTRY_MARKERS:
            self.assertNotIn(marker, rendered_text)


if __name__ == "__main__":
    unittest.main()
