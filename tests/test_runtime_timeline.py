import unittest

from creative_coding_assistant.orchestration import (
    RuntimeTimeline,
    build_production_telemetry,
    build_runtime_timeline,
    build_workflow_diagnostics,
    multimodal_branching_timeline_registry,
    multimodal_creative_evolution_timeline_registry,
    plan_execution_replay,
    plan_workflow_replay,
    runtime_timeline_panel_by_id,
    runtime_timeline_panels_for_status,
)

REQUIRED_RUNTIME_TIMELINE_PANEL_FIELDS = {
    "panel_id",
    "panel_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "timeline_signal_count",
    "guardrail_signal_count",
    "observed_runtime_event_count",
    "reconstructed_timeline_count",
    "emitted_timeline_event_count",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "runtime_timeline_panel_implemented",
    "timeline_reconstruction_implemented",
    "runtime_event_capture_implemented",
    "runtime_event_replay_implemented",
    "workflow_replay_execution_implemented",
    "execution_replay_execution_implemented",
    "session_recording_implemented",
    "snapshot_capture_implemented",
    "trace_capture_implemented",
    "trace_emission_implemented",
    "telemetry_emission_implemented",
    "event_export_implemented",
    "branch_creation_implemented",
    "evolution_generation_implemented",
    "replay_persistence_implemented",
    "persistent_storage_write_implemented",
    "workflow_state_mutation_implemented",
    "workflow_control_implemented",
    "workflow_execution_implemented",
    "provider_model_routing_implemented",
    "agent_invocation_implemented",
    "node_handler_invocation_implemented",
    "retry_triggering_implemented",
    "refinement_triggering_implemented",
    "prompt_mutation_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "advisory_only",
}

EXPECTED_SOURCE_SURFACES = (
    "workflow_diagnostics",
    "production_telemetry",
    "workflow_replay_plan",
    "execution_replay_plan",
    "multimodal_branching_timeline_registry",
    "multimodal_creative_evolution_timeline_registry",
)


class RuntimeTimelineTests(unittest.TestCase):
    def test_default_timeline_links_runtime_sources(self) -> None:
        workflow_replay = plan_workflow_replay()
        execution_replay = plan_execution_replay(workflow_replay=workflow_replay)
        telemetry = build_production_telemetry()
        diagnostics = build_workflow_diagnostics(
            workflow_replay=workflow_replay,
            execution_replay=execution_replay,
            production_telemetry=telemetry,
        )
        branching = multimodal_branching_timeline_registry()
        evolution = multimodal_creative_evolution_timeline_registry()
        timeline = build_runtime_timeline(
            workflow_diagnostics=diagnostics,
            production_telemetry=telemetry,
            workflow_replay=workflow_replay,
            execution_replay=execution_replay,
            branching_timeline=branching,
            creative_evolution_timeline=evolution,
        )

        self.assertEqual(timeline.role, "runtime_timeline")
        self.assertEqual(timeline.serialization_version, "runtime_timeline.v1")
        self.assertEqual(
            timeline.source_workflow_diagnostics_serialization_version,
            diagnostics.serialization_version,
        )
        self.assertEqual(
            timeline.source_production_telemetry_serialization_version,
            telemetry.serialization_version,
        )
        self.assertEqual(
            timeline.source_workflow_replay_serialization_version,
            workflow_replay.serialization_version,
        )
        self.assertEqual(
            timeline.source_execution_replay_serialization_version,
            execution_replay.serialization_version,
        )
        self.assertEqual(
            timeline.source_branching_timeline_serialization_version,
            branching.serialization_version,
        )
        self.assertEqual(
            timeline.source_creative_evolution_timeline_serialization_version,
            evolution.serialization_version,
        )
        self.assertEqual(timeline.source_surfaces, EXPECTED_SOURCE_SURFACES)
        self.assertEqual(timeline.panel_count, 6)
        self.assertEqual(
            timeline.panel_ids,
            (
                "runtime_timeline::workflow_diagnostic_timeline",
                "runtime_timeline::production_telemetry_timeline",
                "runtime_timeline::workflow_replay_timeline",
                "runtime_timeline::execution_replay_timeline",
                "runtime_timeline::branching_timeline_context",
                "runtime_timeline::creative_evolution_timeline_context",
            ),
        )
        self.assertGreater(timeline.timeline_signal_count, 0)
        self.assertGreater(timeline.guardrail_signal_count, 0)
        self.assertIsNone(timeline.observed_runtime_event_count)
        self.assertIsNone(timeline.reconstructed_timeline_count)
        self.assertIsNone(timeline.emitted_timeline_event_count)
        self.assertEqual(timeline.runtime_timeline_status, "guarded")
        self.assertIn("does not reconstruct timelines", timeline.authority_boundary)
        self.assertTrue(timeline.runtime_timeline_implemented)
        self.assertFalse(timeline.timeline_reconstruction_implemented)
        self.assertFalse(timeline.runtime_event_capture_implemented)
        self.assertFalse(timeline.runtime_event_replay_implemented)
        self.assertFalse(timeline.workflow_replay_execution_implemented)
        self.assertFalse(timeline.execution_replay_execution_implemented)
        self.assertFalse(timeline.session_recording_implemented)
        self.assertFalse(timeline.snapshot_capture_implemented)
        self.assertFalse(timeline.trace_capture_implemented)
        self.assertFalse(timeline.trace_emission_implemented)
        self.assertFalse(timeline.telemetry_emission_implemented)
        self.assertFalse(timeline.event_export_implemented)
        self.assertFalse(timeline.branch_creation_implemented)
        self.assertFalse(timeline.evolution_generation_implemented)
        self.assertFalse(timeline.replay_persistence_implemented)
        self.assertFalse(timeline.persistent_storage_write_implemented)
        self.assertFalse(timeline.workflow_state_mutation_implemented)
        self.assertFalse(timeline.workflow_control_implemented)
        self.assertFalse(timeline.workflow_execution_implemented)
        self.assertFalse(timeline.provider_model_routing_implemented)
        self.assertFalse(timeline.agent_invocation_implemented)
        self.assertFalse(timeline.node_handler_invocation_implemented)
        self.assertFalse(timeline.retry_triggering_implemented)
        self.assertFalse(timeline.refinement_triggering_implemented)
        self.assertFalse(timeline.prompt_mutation_implemented)
        self.assertFalse(timeline.generated_output_mutation_implemented)
        self.assertFalse(timeline.runtime_evolution_implemented)
        self.assertTrue(timeline.advisory_only)

    def test_panels_are_read_only_and_boundary_explicit(self) -> None:
        timeline = build_runtime_timeline()

        for panel in timeline.panels:
            dumped = panel.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RUNTIME_TIMELINE_PANEL_FIELDS)
            self.assertEqual(panel.serialization_version, "runtime_timeline_panel.v1")
            self.assertIsNone(panel.observed_runtime_event_count)
            self.assertIsNone(panel.reconstructed_timeline_count)
            self.assertIsNone(panel.emitted_timeline_event_count)
            self.assertIn("timeline_reconstruction", panel.blocked_runtime_behaviors)
            self.assertIn("runtime_event_replay", panel.blocked_runtime_behaviors)
            self.assertTrue(panel.runtime_timeline_panel_implemented)
            self.assertFalse(panel.timeline_reconstruction_implemented)
            self.assertFalse(panel.runtime_event_capture_implemented)
            self.assertFalse(panel.runtime_event_replay_implemented)
            self.assertFalse(panel.workflow_replay_execution_implemented)
            self.assertFalse(panel.execution_replay_execution_implemented)
            self.assertFalse(panel.session_recording_implemented)
            self.assertFalse(panel.snapshot_capture_implemented)
            self.assertFalse(panel.trace_capture_implemented)
            self.assertFalse(panel.trace_emission_implemented)
            self.assertFalse(panel.telemetry_emission_implemented)
            self.assertFalse(panel.event_export_implemented)
            self.assertFalse(panel.branch_creation_implemented)
            self.assertFalse(panel.evolution_generation_implemented)
            self.assertFalse(panel.replay_persistence_implemented)
            self.assertFalse(panel.persistent_storage_write_implemented)
            self.assertFalse(panel.workflow_state_mutation_implemented)
            self.assertFalse(panel.workflow_control_implemented)
            self.assertFalse(panel.workflow_execution_implemented)
            self.assertFalse(panel.provider_model_routing_implemented)
            self.assertFalse(panel.agent_invocation_implemented)
            self.assertFalse(panel.node_handler_invocation_implemented)
            self.assertFalse(panel.retry_triggering_implemented)
            self.assertFalse(panel.refinement_triggering_implemented)
            self.assertFalse(panel.prompt_mutation_implemented)
            self.assertFalse(panel.generated_output_mutation_implemented)
            self.assertFalse(panel.runtime_evolution_implemented)
            self.assertTrue(panel.advisory_only)

        branching = runtime_timeline_panel_by_id(
            "runtime_timeline::branching_timeline_context",
            timeline,
        )
        self.assertIsNotNone(branching)
        assert branching is not None
        self.assertEqual(branching.status, "guarded")
        self.assertEqual(
            branching.source_serialization_version,
            "multimodal_branching_timeline_registry.v1",
        )
        self.assertGreater(branching.timeline_signal_count, 0)

    def test_lookup_helpers_are_stable_and_non_replaying(self) -> None:
        timeline = build_runtime_timeline()
        execution = runtime_timeline_panel_by_id(
            "runtime_timeline::execution_replay_timeline",
            timeline,
        )
        guarded = runtime_timeline_panels_for_status("guarded", timeline)
        ready = runtime_timeline_panels_for_status("ready", timeline)
        missing = runtime_timeline_panel_by_id("missing", timeline)

        self.assertIsNone(missing)
        self.assertIsNotNone(execution)
        assert execution is not None
        self.assertEqual(execution.panel_kind, "execution_replay_timeline")
        self.assertEqual(len(ready), 0)
        self.assertEqual(len(guarded), timeline.panel_count)

    def test_timeline_rejects_mismatched_panel_totals(self) -> None:
        timeline = build_runtime_timeline()
        payload = timeline.model_dump(mode="json")
        payload["panel_ids"] = ("missing",) + tuple(payload["panel_ids"][1:])

        with self.assertRaisesRegex(ValueError, "panel_ids must match"):
            RuntimeTimeline(**payload)

        payload = timeline.model_dump(mode="json")
        payload["timeline_signal_count"] += 1

        with self.assertRaisesRegex(ValueError, "timeline_signal_count must match"):
            RuntimeTimeline(**payload)

        payload = timeline.model_dump(mode="json")
        payload["runtime_timeline_status"] = "ready"

        with self.assertRaisesRegex(ValueError, "runtime_timeline_status must match"):
            RuntimeTimeline(**payload)

        payload = timeline.model_dump(mode="json")
        payload["source_surfaces"] = (
            "missing",
            *tuple(payload["source_surfaces"][1:]),
        )

        with self.assertRaisesRegex(ValueError, "source_surfaces must match"):
            RuntimeTimeline(**payload)

    def test_timeline_does_not_declare_runtime_execution_terms(self) -> None:
        timeline = build_runtime_timeline()
        combined_text = " ".join(
            (
                timeline.authority_boundary,
                *timeline.blocked_runtime_behaviors,
                *timeline.advisory_actions,
                *(
                    field
                    for panel in timeline.panels
                    for field in (
                        panel.panel_id,
                        panel.source_id,
                        *(panel.source_item_ids),
                        *(panel.evidence),
                        *(panel.advisory_actions),
                        *(panel.blocked_runtime_behaviors),
                    )
                ),
            )
        )

        for forbidden_term in (
            "reconstruct_timeline(",
            "capture_runtime_event(",
            "replay_runtime_event(",
            "record_session(",
            "capture_snapshot(",
            "emit_telemetry(",
            "create_branch(",
            "mutate_workflow_state(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
