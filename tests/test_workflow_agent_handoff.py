import json
import unittest

from creative_coding_assistant.orchestration import (
    ASSISTANT_WORKFLOW_NODE_ORDER,
    JinjaPromptRenderer,
    StructuredPromptInputBuilder,
    WorkflowAgentHandoffRegistry,
    agent_role_registry,
    build_assistant_workflow_graph,
    build_prompt_input_request,
    build_rendered_prompt_request,
    shared_context_view_registry,
    stream_assistant_workflow_events,
    workflow_agent_handoff_by_id,
    workflow_agent_handoff_profile_by_agent_id,
    workflow_agent_handoff_registry,
    workflow_agent_handoffs_for_surface,
)
from test_langgraph_workflow_integration import (
    _request,
    _route_generate,
    _runtime,
    _stream_completed_generation,
    _stream_prompt_inputs_with_builder,
)

REQUIRED_HANDOFF_FIELDS = {
    "handoff_id",
    "surface",
    "source_workflow_steps",
    "source_state_fields",
    "target_agent_ids",
    "target_role_ids",
    "payload_exposure",
    "handoff_intent",
    "handoff_boundary",
    "blocked_runtime_behaviors",
    "workflow_graph_change_implemented",
    "prompt_alteration_implemented",
    "agent_execution_implemented",
    "runtime_handoff_implemented",
    "serialization_version",
    "metadata_only",
}

REQUIRED_PROFILE_FIELDS = {
    "agent_id",
    "role_id",
    "handoff_profile_id",
    "accepted_handoff_ids",
    "accepted_surfaces",
    "accepted_state_fields",
    "source_context_view_id",
    "profile_boundary",
    "blocked_runtime_behaviors",
    "workflow_graph_change_implemented",
    "prompt_alteration_implemented",
    "agent_execution_implemented",
    "runtime_handoff_implemented",
    "serialization_version",
    "metadata_only",
}

EXPECTED_SURFACES = (
    "planning",
    "artifact",
    "evaluation",
    "provenance",
    "finalization",
)


class WorkflowAgentHandoffTests(unittest.TestCase):
    def test_registry_maps_v3_surfaces_to_v4_agents(self) -> None:
        registry = workflow_agent_handoff_registry()
        roles = agent_role_registry()
        context_views = shared_context_view_registry()

        self.assertEqual(registry.role, "workflow_agent_handoff_registry")
        self.assertEqual(
            registry.serialization_version,
            "workflow_agent_handoff_registry.v1",
        )
        self.assertEqual(registry.surfaces, EXPECTED_SURFACES)
        self.assertEqual(registry.agent_ids, roles.agent_ids)
        self.assertEqual(registry.agent_ids, context_views.agent_ids)
        self.assertEqual(registry.workflow_step_ids, ASSISTANT_WORKFLOW_NODE_ORDER[:-1])
        self.assertEqual(registry.handoff_count, 5)
        self.assertEqual(registry.profile_count, 12)
        self.assertEqual(
            registry.source_registries,
            ("workflow_state", "agent_role_registry", "shared_context_view_registry"),
        )
        self.assertIn("does not change the workflow graph", registry.authority_boundary)
        self.assertFalse(registry.workflow_graph_change_implemented)
        self.assertFalse(registry.prompt_alteration_implemented)
        self.assertFalse(registry.agent_execution_implemented)
        self.assertFalse(registry.runtime_handoff_implemented)
        self.assertTrue(registry.metadata_only)

    def test_handoff_contracts_are_metadata_references_only(self) -> None:
        registry = workflow_agent_handoff_registry()
        known_agents = set(registry.agent_ids)
        known_steps = set(registry.workflow_step_ids)

        for handoff in registry.handoffs:
            dumped = handoff.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_HANDOFF_FIELDS)
            self.assertEqual(
                handoff.serialization_version,
                "workflow_agent_handoff_contract.v1",
            )
            self.assertEqual(handoff.payload_exposure, "metadata_reference_only")
            self.assertTrue(set(handoff.source_workflow_steps).issubset(known_steps))
            self.assertTrue(set(handoff.target_agent_ids).issubset(known_agents))
            self.assertEqual(len(handoff.target_agent_ids), len(handoff.target_role_ids))
            self.assertTrue(handoff.source_state_fields)
            self.assertIn("workflow_graph_change", handoff.blocked_runtime_behaviors)
            self.assertFalse(handoff.workflow_graph_change_implemented)
            self.assertFalse(handoff.prompt_alteration_implemented)
            self.assertFalse(handoff.agent_execution_implemented)
            self.assertFalse(handoff.runtime_handoff_implemented)
            self.assertTrue(handoff.metadata_only)

    def test_agent_profiles_cover_targeted_handoffs(self) -> None:
        registry = workflow_agent_handoff_registry()
        known_handoffs = set(registry.handoff_ids)

        for profile in registry.profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertTrue(set(profile.accepted_handoff_ids).issubset(known_handoffs))
            self.assertTrue(profile.accepted_surfaces)
            self.assertTrue(profile.accepted_state_fields)
            self.assertTrue(profile.source_context_view_id.endswith("_shared_context_view"))
            self.assertFalse(profile.workflow_graph_change_implemented)
            self.assertFalse(profile.prompt_alteration_implemented)
            self.assertFalse(profile.agent_execution_implemented)
            self.assertFalse(profile.runtime_handoff_implemented)
            self.assertTrue(profile.metadata_only)

        final_synthesizer = workflow_agent_handoff_profile_by_agent_id(
            "final_synthesizer_agent"
        )
        self.assertIsNotNone(final_synthesizer)
        assert final_synthesizer is not None
        self.assertEqual(final_synthesizer.accepted_surfaces, EXPECTED_SURFACES[1:])

    def test_handoff_lookups_are_stable(self) -> None:
        planning = workflow_agent_handoff_by_id("planning_surface_agent_handoff")
        missing_handoff = workflow_agent_handoff_by_id("missing_handoff")
        missing_profile = workflow_agent_handoff_profile_by_agent_id("missing_agent")
        evaluation_handoffs = workflow_agent_handoffs_for_surface("evaluation")

        self.assertIsNone(missing_handoff)
        self.assertIsNone(missing_profile)
        self.assertIsNotNone(planning)
        assert planning is not None
        self.assertEqual(planning.surface, "planning")
        self.assertIn("creative_plan", planning.source_state_fields)
        self.assertIn("planner_agent", planning.target_agent_ids)
        self.assertEqual(
            tuple(handoff.handoff_id for handoff in evaluation_handoffs),
            ("evaluation_surface_agent_handoff",),
        )

    def test_registry_rejects_invalid_handoff_references(self) -> None:
        registry = workflow_agent_handoff_registry()
        unknown_target = registry.handoffs[0].model_copy(
            update={"target_agent_ids": ("missing_agent",)}
        )
        unknown_profile_handoff = registry.profiles[0].model_copy(
            update={"accepted_handoff_ids": ("missing_handoff",)}
        )

        with self.assertRaisesRegex(ValueError, "target_agent_ids must be known"):
            WorkflowAgentHandoffRegistry(
                handoffs=(unknown_target,) + registry.handoffs[1:],
                profiles=registry.profiles,
                handoff_ids=registry.handoff_ids,
                surfaces=registry.surfaces,
                agent_ids=registry.agent_ids,
                profile_ids=registry.profile_ids,
                workflow_step_ids=registry.workflow_step_ids,
                handoff_count=registry.handoff_count,
                profile_count=registry.profile_count,
                source_registries=registry.source_registries,
            )

        with self.assertRaisesRegex(ValueError, "profile handoffs must be known"):
            WorkflowAgentHandoffRegistry(
                handoffs=registry.handoffs,
                profiles=(unknown_profile_handoff,) + registry.profiles[1:],
                handoff_ids=registry.handoff_ids,
                surfaces=registry.surfaces,
                agent_ids=registry.agent_ids,
                profile_ids=registry.profile_ids,
                workflow_step_ids=registry.workflow_step_ids,
                handoff_count=registry.handoff_count,
                profile_count=registry.profile_count,
                source_registries=registry.source_registries,
            )

    def test_handoff_metadata_does_not_leak_into_payloads_or_prompts(self) -> None:
        graph = build_assistant_workflow_graph()
        events = tuple(
            stream_assistant_workflow_events(
                graph=graph,
                request=_request(query="Generate a handoff boundary sketch."),
                runtime=_runtime(
                    stream_prompt_inputs=_stream_prompt_inputs_with_builder,
                    stream_generation=_stream_completed_generation,
                ),
            )
        )
        payload_text = json.dumps([event.payload for event in events], sort_keys=True)
        request = _request(query="Generate a passive handoff prompt boundary scene.")
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

        for marker in (
            "workflow_agent_handoff_registry",
            "planning_surface_agent_handoff",
            "runtime_handoff_execution",
        ):
            self.assertNotIn(marker, payload_text)
            self.assertNotIn(marker, rendered_text)


if __name__ == "__main__":
    unittest.main()
