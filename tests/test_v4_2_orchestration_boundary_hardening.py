import json
import tempfile
import unittest
from pathlib import Path

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
    StreamEventType,
)
from creative_coding_assistant.core import GenerationProviderName, Settings
from creative_coding_assistant.llm import OpenAIGenerationProvider, build_generation_provider
from creative_coding_assistant.orchestration import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
    JinjaPromptRenderer,
    StructuredPromptInputBuilder,
    build_assistant_workflow_graph,
    build_prompt_input_request,
    build_rendered_prompt_request,
    orchestration_contract_integration_registry,
    route_request,
    stream_assistant_workflow_events,
)
from creative_coding_assistant.vectorstore import (
    ChromaCollection,
    ChromaRepository,
    create_chroma_client,
    ensure_project_collections,
    get_collection_definition,
)
from test_langgraph_workflow_integration import (
    _SequentialGeneration,
    _request,
    _route_generate,
    _runtime,
    _single_generation,
    _stream_prompt_inputs_with_builder,
)

EXPECTED_WORKFLOW_NODE_ORDER = (
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
)

V42_ORCHESTRATION_MARKERS = (
    "agent_routing_registry",
    "blackboard_memory_registry",
    "shared_context_view_registry",
    "agent_dependency_graph_registry",
    "parallel_scheduling_registry",
    "agent_coordination_registry",
    "agent_debate_registry",
    "consensus_builder_registry",
    "agent_capability_alignment_registry",
    "agent_escalation_signal_registry",
    "agent_lifecycle_registry",
    "agent_state_synchronization_registry",
    "workflow_agent_handoff_registry",
    "orchestration_contract_integration_registry",
    "runtime_handoff_execution",
    "orchestration_execution",
    "agent_invocation",
)


class V42OrchestrationBoundaryHardeningTests(unittest.TestCase):
    def test_metadata_does_not_change_provider_or_model_routing(self) -> None:
        request = AssistantRequest(
            query="Generate a p5.js particle field.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        settings = Settings(
            default_generation_provider=GenerationProviderName.OPENAI,
            openai_model="gpt-5",
            openai_api_key="sk-test-secret",
        )
        baseline_route = route_request(request)
        baseline_provider = build_generation_provider(settings)

        _touch_v4_2_orchestration_metadata()
        next_route = route_request(request)
        next_provider = build_generation_provider(settings)

        self.assertEqual(next_route, baseline_route)
        self.assertIsInstance(baseline_provider, OpenAIGenerationProvider)
        self.assertIsInstance(next_provider, OpenAIGenerationProvider)
        self.assertEqual(next_provider._model, "gpt-5")
        self.assertEqual(next_provider._settings.default_generation_provider, "openai")
        route_text = next_route.model_dump_json()
        for marker in V42_ORCHESTRATION_MARKERS:
            self.assertNotIn(marker, route_text)

    def test_metadata_does_not_enter_prompt_rendering(self) -> None:
        request = _request(query="Generate a passive V4.2 boundary scene.")
        route_decision = _route_generate(request)

        _touch_v4_2_orchestration_metadata()
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
        prompt_input_text = prompt_input.model_dump_json()
        rendered_text = "\n".join(section.content for section in rendered.sections)

        for marker in V42_ORCHESTRATION_MARKERS:
            self.assertNotIn(marker, prompt_input_text)
            self.assertNotIn(marker, rendered_text)

    def test_metadata_does_not_change_workflow_nodes_payloads_or_answer(self) -> None:
        answer = "\n".join(
            (
                "V4.2 boundary answer.",
                "```javascript",
                "function setup() { createCanvas(120, 120); }",
                "function draw() { background(8); circle(60, 60, 40); }",
                "```",
            )
        )
        graph = build_assistant_workflow_graph()

        _touch_v4_2_orchestration_metadata()
        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=_request(query="Write code for a p5.js boundary sketch."),
                runtime=_runtime(
                    stream_prompt_inputs=_stream_prompt_inputs_with_builder,
                    stream_generation=_single_generation(answer),
                ),
            )
        )
        payload_text = json.dumps([event.payload for event in events], sort_keys=True)

        self.assertEqual(ASSISTANT_WORKFLOW_NODE_ORDER, EXPECTED_WORKFLOW_NODE_ORDER)
        self.assertEqual(events[-1].payload["answer"], answer)
        self.assertEqual(events[-1].event_type, StreamEventType.FINAL)
        for marker in V42_ORCHESTRATION_MARKERS:
            self.assertNotIn(marker, payload_text)

    def test_metadata_does_not_change_retry_or_refinement_behavior(self) -> None:
        graph = build_assistant_workflow_graph()
        generation = _SequentialGeneration(
            "This answer has no fenced code.",
            "\n".join(
                (
                    "```javascript",
                    "const scene = new THREE.Scene();",
                    "function animate() { scene.rotation.y += 0.01; }",
                    "```",
                )
            ),
        )

        _touch_v4_2_orchestration_metadata()
        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=_request(query="Write code for a Three.js scene."),
                runtime=_runtime(stream_generation=generation.stream),
            )
        )
        event_types = tuple(event.event_type for event in events)
        payload_text = json.dumps([event.payload for event in events], sort_keys=True)

        self.assertEqual(generation.calls, 2)
        self.assertIn(StreamEventType.REVIEW_FAILED, event_types)
        self.assertIn(StreamEventType.REFINEMENT_REQUESTED, event_types)
        self.assertIn(StreamEventType.RETRY_STARTED, event_types)
        self.assertIn(StreamEventType.RETRY_COMPLETED, event_types)
        self.assertEqual(events[-1].event_type, StreamEventType.FINAL)
        self.assertIn("const scene = new THREE.Scene();", events[-1].payload["answer"])
        for marker in V42_ORCHESTRATION_MARKERS:
            self.assertNotIn(marker, payload_text)

    def test_metadata_does_not_create_or_write_storage_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            persist_dir = Path(temp_dir) / "chroma"
            client = create_chroma_client(path=persist_dir)
            ensure_project_collections(client)
            definition = get_collection_definition(ChromaCollection.PROJECT_MEMORY)
            repository = ChromaRepository(client=client, definition=definition)

            self.assertEqual(repository.count(), 0)
            _touch_v4_2_orchestration_metadata()
            self.assertEqual(repository.count(), 0)


def _touch_v4_2_orchestration_metadata() -> None:
    registry = orchestration_contract_integration_registry()
    dumped = registry.model_dump(mode="json")
    assert dumped["metadata_only"] is True
    assert dumped["registry_count"] == 13


if __name__ == "__main__":
    unittest.main()
