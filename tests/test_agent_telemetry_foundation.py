import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
    StreamEventType,
)
from creative_coding_assistant.orchestration import (
    AgentTelemetryFoundationRegistry,
    agent_lifecycle_registry,
    agent_metadata_registry,
    agent_telemetry_foundation_registry,
    agent_telemetry_profile_by_agent_id,
    agent_telemetry_profiles_for_dimension,
    agent_telemetry_profiles_for_event_type,
    decision_provenance_registry,
    escalation_trace_registry,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

EXPECTED_SOURCE_REGISTRIES = (
    "agent_metadata_registry",
    "agent_lifecycle_registry",
    "stream_event_contracts",
    "decision_provenance_registry",
    "escalation_trace_registry",
)

REQUIRED_PROFILE_FIELDS = {
    "agent_id",
    "role_id",
    "telemetry_stage",
    "metadata_serialization_version",
    "lifecycle_profile_id",
    "observability_surfaces",
    "auditability_surfaces",
    "telemetry_event_types",
    "provenance_profile_ids",
    "trace_profile_ids",
    "telemetry_dimensions",
    "telemetry_source_registries",
    "passive_boundary_flags",
    "foundation_findings",
    "missing_coverage_items",
    "metadata_blocked_runtime_behaviors",
    "lifecycle_blocked_runtime_behaviors",
    "provenance_blocked_runtime_behaviors",
    "trace_blocked_runtime_behaviors",
    "blocked_runtime_behaviors",
    "foundation_status",
    "metadata_only_declared",
    "observability_surface_coverage_present",
    "event_type_reference_present",
    "provenance_trace_reference_present",
    "telemetry_emission_implemented",
    "trace_capture_implemented",
    "provenance_recording_implemented",
    "event_stream_mutation_implemented",
    "external_monitoring_implemented",
    "memory_write_implemented",
    "agent_invocation_implemented",
    "workflow_control_implemented",
    "provider_model_routing_implemented",
    "retry_triggering_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "metadata_only",
}


class AgentTelemetryFoundationTests(unittest.TestCase):
    def test_foundation_registry_covers_telemetry_sources(self) -> None:
        registry = agent_telemetry_foundation_registry()
        metadata = agent_metadata_registry()
        lifecycle = agent_lifecycle_registry()
        provenance = decision_provenance_registry()
        trace = escalation_trace_registry()

        self.assertEqual(registry.role, "agent_telemetry_foundation_registry")
        self.assertEqual(
            registry.serialization_version,
            "agent_telemetry_foundation_registry.v1",
        )
        self.assertEqual(
            registry.telemetry_stage,
            "v4_6_agent_telemetry_foundation",
        )
        self.assertEqual(registry.agent_ids, metadata.agent_ids)
        self.assertEqual(
            registry.telemetry_source_registries,
            EXPECTED_SOURCE_REGISTRIES,
        )
        self.assertEqual(
            registry.observability_surfaces, metadata.observability_surfaces
        )
        self.assertEqual(registry.auditability_surfaces, metadata.auditability_surfaces)
        self.assertEqual(
            registry.telemetry_event_types,
            tuple(event_type.value for event_type in StreamEventType),
        )
        self.assertEqual(
            registry.lifecycle_profile_ids,
            tuple(profile.lifecycle_profile_id for profile in lifecycle.profiles),
        )
        self.assertEqual(
            registry.provenance_profile_ids,
            provenance.provenance_profile_ids,
        )
        self.assertEqual(registry.trace_profile_ids, trace.trace_profile_ids)
        self.assertEqual(registry.profile_count, 12)
        self.assertTrue(registry.all_agents_covered)
        self.assertTrue(registry.no_missing_coverage)
        self.assertTrue(registry.metadata_only)
        self.assertFalse(registry.telemetry_emission_implemented)
        self.assertFalse(registry.trace_capture_implemented)
        self.assertFalse(registry.provenance_recording_implemented)
        self.assertFalse(registry.event_stream_mutation_implemented)
        self.assertFalse(registry.external_monitoring_implemented)
        self.assertFalse(registry.memory_write_implemented)
        self.assertFalse(registry.agent_invocation_implemented)
        self.assertFalse(registry.workflow_control_implemented)
        self.assertFalse(registry.provider_model_routing_implemented)
        self.assertFalse(registry.retry_triggering_implemented)
        self.assertFalse(registry.generated_output_mutation_implemented)
        self.assertIn("does not emit telemetry", registry.authority_boundary)

    def test_profiles_are_passive_and_source_aligned(self) -> None:
        registry = agent_telemetry_foundation_registry()
        known_lifecycle_profiles = set(registry.lifecycle_profile_ids)

        for profile in registry.profiles:
            dumped = profile.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PROFILE_FIELDS)
            self.assertEqual(
                profile.serialization_version,
                "agent_telemetry_foundation_profile.v1",
            )
            self.assertEqual(profile.foundation_status, "pass")
            self.assertEqual(profile.telemetry_stage, registry.telemetry_stage)
            self.assertIn(profile.lifecycle_profile_id, known_lifecycle_profiles)
            self.assertEqual(
                profile.observability_surfaces, registry.observability_surfaces
            )
            self.assertEqual(
                profile.auditability_surfaces, registry.auditability_surfaces
            )
            self.assertEqual(
                profile.telemetry_event_types, registry.telemetry_event_types
            )
            self.assertEqual(
                profile.provenance_profile_ids, registry.provenance_profile_ids
            )
            self.assertEqual(profile.trace_profile_ids, registry.trace_profile_ids)
            self.assertEqual(
                profile.telemetry_dimensions, registry.telemetry_dimensions
            )
            self.assertEqual(
                profile.telemetry_source_registries,
                registry.telemetry_source_registries,
            )
            self.assertEqual(
                profile.passive_boundary_flags, registry.passive_boundary_flags
            )
            self.assertFalse(profile.missing_coverage_items)
            self.assertIn("agent_execution", profile.metadata_blocked_runtime_behaviors)
            self.assertIn(
                "runtime_lifecycle_engine",
                profile.lifecycle_blocked_runtime_behaviors,
            )
            self.assertIn(
                "provenance_recording",
                profile.provenance_blocked_runtime_behaviors,
            )
            self.assertIn("trace_capture", profile.trace_blocked_runtime_behaviors)
            self.assertIn("trace_emission", profile.trace_blocked_runtime_behaviors)
            self.assertTrue(profile.metadata_only_declared)
            self.assertTrue(profile.observability_surface_coverage_present)
            self.assertTrue(profile.event_type_reference_present)
            self.assertTrue(profile.provenance_trace_reference_present)
            self.assertTrue(profile.metadata_only)
            self.assertFalse(profile.telemetry_emission_implemented)
            self.assertFalse(profile.trace_capture_implemented)
            self.assertFalse(profile.provenance_recording_implemented)
            self.assertFalse(profile.event_stream_mutation_implemented)
            self.assertFalse(profile.external_monitoring_implemented)
            self.assertFalse(profile.memory_write_implemented)
            self.assertFalse(profile.agent_invocation_implemented)
            self.assertFalse(profile.workflow_control_implemented)
            self.assertFalse(profile.provider_model_routing_implemented)
            self.assertFalse(profile.retry_triggering_implemented)
            self.assertFalse(profile.generated_output_mutation_implemented)

    def test_lookup_event_type_and_dimension_filtering_are_stable(self) -> None:
        registry = agent_telemetry_foundation_registry()
        planner_profile = agent_telemetry_profile_by_agent_id("planner_agent")
        missing_profile = agent_telemetry_profile_by_agent_id("missing_agent")
        node_started_profiles = agent_telemetry_profiles_for_event_type(
            StreamEventType.NODE_STARTED
        )
        trace_dimension_profiles = agent_telemetry_profiles_for_dimension(
            "trace_reference"
        )
        missing_dimension_profiles = agent_telemetry_profiles_for_dimension(
            "missing_dimension"
        )

        self.assertIsNone(missing_profile)
        self.assertIsNotNone(planner_profile)
        assert planner_profile is not None
        self.assertEqual(planner_profile.role_id, "planner")
        self.assertEqual(
            planner_profile.lifecycle_profile_id,
            "planner_agent_lifecycle_profile",
        )
        self.assertIn(
            StreamEventType.FINAL.value, planner_profile.telemetry_event_types
        )
        self.assertEqual(len(node_started_profiles), registry.profile_count)
        self.assertEqual(len(trace_dimension_profiles), registry.profile_count)
        self.assertEqual(missing_dimension_profiles, ())
        self.assertIs(
            planner_profile,
            agent_telemetry_profile_by_agent_id("planner_agent", registry),
        )

    def test_foundation_registry_rejects_mismatched_or_incomplete_profiles(
        self,
    ) -> None:
        registry = agent_telemetry_foundation_registry()
        first_profile = registry.profiles[0]
        duplicate_profile = first_profile.model_copy(update={"role_id": "duplicate"})
        mismatched_flags_profile = first_profile.model_copy(
            update={
                "passive_boundary_flags": (
                    "other_flag",
                    "trace_capture_blocked",
                    "provenance_recording_blocked",
                    "event_stream_mutation_blocked",
                    "external_monitoring_blocked",
                    "memory_write_blocked",
                    "provider_model_routing_blocked",
                    "generated_output_mutation_blocked",
                )
            }
        )
        incomplete_profile = first_profile.model_copy(
            update={"missing_coverage_items": ("telemetry_emission_enabled",)}
        )

        with self.assertRaisesRegex(ValueError, "agent_ids must be unique"):
            AgentTelemetryFoundationRegistry(
                profiles=(first_profile, duplicate_profile) + registry.profiles[2:],
                agent_ids=registry.agent_ids,
                profile_count=registry.profile_count,
                telemetry_source_registries=registry.telemetry_source_registries,
                observability_surfaces=registry.observability_surfaces,
                auditability_surfaces=registry.auditability_surfaces,
                telemetry_event_types=registry.telemetry_event_types,
                lifecycle_profile_ids=registry.lifecycle_profile_ids,
                provenance_profile_ids=registry.provenance_profile_ids,
                trace_profile_ids=registry.trace_profile_ids,
                telemetry_dimensions=registry.telemetry_dimensions,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "passive_boundary_flags"):
            AgentTelemetryFoundationRegistry(
                profiles=(mismatched_flags_profile,) + registry.profiles[1:],
                agent_ids=registry.agent_ids,
                profile_count=registry.profile_count,
                telemetry_source_registries=registry.telemetry_source_registries,
                observability_surfaces=registry.observability_surfaces,
                auditability_surfaces=registry.auditability_surfaces,
                telemetry_event_types=registry.telemetry_event_types,
                lifecycle_profile_ids=registry.lifecycle_profile_ids,
                provenance_profile_ids=registry.provenance_profile_ids,
                trace_profile_ids=registry.trace_profile_ids,
                telemetry_dimensions=registry.telemetry_dimensions,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

        with self.assertRaisesRegex(ValueError, "missing coverage"):
            AgentTelemetryFoundationRegistry(
                profiles=(incomplete_profile,) + registry.profiles[1:],
                agent_ids=registry.agent_ids,
                profile_count=registry.profile_count,
                telemetry_source_registries=registry.telemetry_source_registries,
                observability_surfaces=registry.observability_surfaces,
                auditability_surfaces=registry.auditability_surfaces,
                telemetry_event_types=registry.telemetry_event_types,
                lifecycle_profile_ids=registry.lifecycle_profile_ids,
                provenance_profile_ids=registry.provenance_profile_ids,
                trace_profile_ids=registry.trace_profile_ids,
                telemetry_dimensions=registry.telemetry_dimensions,
                passive_boundary_flags=registry.passive_boundary_flags,
            )

    def test_agent_telemetry_foundation_does_not_change_request_routing(self) -> None:
        request = AssistantRequest(
            query="Generate telemetry metadata for a sketch.",
            mode=AssistantMode.GENERATE,
            domains=(CreativeCodingDomain.P5_JS,),
        )
        baseline_decision = route_request(request)

        agent_telemetry_foundation_registry()
        agent_telemetry_profile_by_agent_id("planner_agent")
        agent_telemetry_profiles_for_event_type("node_started")
        next_decision = route_request(request)

        self.assertEqual(next_decision, baseline_decision)
        self.assertEqual(next_decision.route, RouteName.GENERATE)
        self.assertNotIn(
            "agent_telemetry_foundation_registry",
            next_decision.model_dump_json(),
        )

    def test_foundation_metadata_does_not_declare_active_telemetry_terms(self) -> None:
        registry = agent_telemetry_foundation_registry()
        combined_text = " ".join(
            (
                registry.authority_boundary,
                *registry.blocked_runtime_behaviors,
                *(
                    field
                    for profile in registry.profiles
                    for field in (
                        profile.agent_id,
                        *profile.foundation_findings,
                        *profile.passive_boundary_flags,
                    )
                ),
            )
        )

        for forbidden_term in (
            "emit_runtime_telemetry",
            "capture_runtime_trace",
            "record_runtime_provenance",
            "mutate_event_stream",
            "start_external_monitoring",
            "write_memory",
            "route_provider",
            "mutate_generated_output",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
