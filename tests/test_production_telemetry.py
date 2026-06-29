import unittest

from creative_coding_assistant.orchestration import (
    ProductionTelemetrySurface,
    agent_telemetry_foundation_registry,
    build_cost_dashboard,
    build_performance_dashboard,
    build_production_telemetry,
    build_quality_dashboard,
    build_token_dashboard,
    production_telemetry_channel_by_id,
    production_telemetry_channels_for_status,
)

REQUIRED_PRODUCTION_TELEMETRY_CHANNEL_FIELDS = {
    "channel_id",
    "channel_kind",
    "status",
    "source_id",
    "source_serialization_version",
    "source_item_ids",
    "telemetry_signal_count",
    "guarded_signal_count",
    "emitted_event_count",
    "exported_event_count",
    "evidence",
    "advisory_actions",
    "blocked_runtime_behaviors",
    "production_telemetry_channel_implemented",
    "telemetry_emission_implemented",
    "live_metrics_collection_implemented",
    "trace_capture_implemented",
    "event_export_implemented",
    "external_monitoring_sink_implemented",
    "persistent_storage_write_implemented",
    "alert_emission_implemented",
    "hitl_request_implemented",
    "provider_model_routing_implemented",
    "model_selection_implemented",
    "workflow_control_implemented",
    "workflow_execution_implemented",
    "agent_invocation_implemented",
    "node_handler_invocation_implemented",
    "retry_triggering_implemented",
    "prompt_mutation_implemented",
    "generated_output_mutation_implemented",
    "serialization_version",
    "advisory_only",
}

EXPECTED_SOURCE_REGISTRIES = (
    "agent_telemetry_foundation_registry",
    "token_dashboard",
    "cost_dashboard",
    "quality_dashboard",
    "performance_dashboard",
)


class ProductionTelemetryTests(unittest.TestCase):
    def test_default_surface_links_observability_sources(self) -> None:
        agent_telemetry = agent_telemetry_foundation_registry()
        token_dashboard = build_token_dashboard()
        cost_dashboard = build_cost_dashboard()
        quality_dashboard = build_quality_dashboard()
        performance_dashboard = build_performance_dashboard()
        telemetry = build_production_telemetry(
            agent_telemetry=agent_telemetry,
            token_dashboard=token_dashboard,
            cost_dashboard=cost_dashboard,
            quality_dashboard=quality_dashboard,
            performance_dashboard=performance_dashboard,
        )

        self.assertEqual(telemetry.role, "production_telemetry")
        self.assertEqual(
            telemetry.serialization_version,
            "production_telemetry.v1",
        )
        self.assertEqual(
            telemetry.source_agent_telemetry_serialization_version,
            agent_telemetry.serialization_version,
        )
        self.assertEqual(
            telemetry.source_token_dashboard_serialization_version,
            token_dashboard.serialization_version,
        )
        self.assertEqual(
            telemetry.source_cost_dashboard_serialization_version,
            cost_dashboard.serialization_version,
        )
        self.assertEqual(
            telemetry.source_quality_dashboard_serialization_version,
            quality_dashboard.serialization_version,
        )
        self.assertEqual(
            telemetry.source_performance_dashboard_serialization_version,
            performance_dashboard.serialization_version,
        )
        self.assertEqual(
            telemetry.telemetry_source_registries,
            EXPECTED_SOURCE_REGISTRIES,
        )
        self.assertEqual(telemetry.channel_count, 6)
        self.assertEqual(
            telemetry.channel_ids,
            (
                "production_telemetry::agent_foundation",
                "production_telemetry::token_dashboard",
                "production_telemetry::cost_dashboard",
                "production_telemetry::quality_dashboard",
                "production_telemetry::performance_dashboard",
                "production_telemetry::emission_boundary",
            ),
        )
        self.assertGreater(telemetry.telemetry_signal_count, 0)
        self.assertGreater(telemetry.guarded_signal_count, 0)
        self.assertIsNone(telemetry.emitted_event_count)
        self.assertIsNone(telemetry.exported_event_count)
        self.assertEqual(telemetry.production_telemetry_status, "guarded")
        self.assertIn("does not emit telemetry", telemetry.authority_boundary)
        self.assertTrue(telemetry.production_telemetry_implemented)
        self.assertFalse(telemetry.telemetry_emission_implemented)
        self.assertFalse(telemetry.live_metrics_collection_implemented)
        self.assertFalse(telemetry.trace_capture_implemented)
        self.assertFalse(telemetry.event_export_implemented)
        self.assertFalse(telemetry.external_monitoring_sink_implemented)
        self.assertFalse(telemetry.persistent_storage_write_implemented)
        self.assertFalse(telemetry.alert_emission_implemented)
        self.assertFalse(telemetry.hitl_request_implemented)
        self.assertFalse(telemetry.provider_model_routing_implemented)
        self.assertFalse(telemetry.model_selection_implemented)
        self.assertFalse(telemetry.workflow_control_implemented)
        self.assertFalse(telemetry.workflow_execution_implemented)
        self.assertFalse(telemetry.agent_invocation_implemented)
        self.assertFalse(telemetry.node_handler_invocation_implemented)
        self.assertFalse(telemetry.retry_triggering_implemented)
        self.assertFalse(telemetry.prompt_mutation_implemented)
        self.assertFalse(telemetry.generated_output_mutation_implemented)
        self.assertTrue(telemetry.advisory_only)

    def test_channels_are_read_only_and_boundary_explicit(self) -> None:
        telemetry = build_production_telemetry()

        for channel in telemetry.channels:
            dumped = channel.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_PRODUCTION_TELEMETRY_CHANNEL_FIELDS)
            self.assertEqual(
                channel.serialization_version,
                "production_telemetry_channel.v1",
            )
            self.assertIsNone(channel.emitted_event_count)
            self.assertIsNone(channel.exported_event_count)
            self.assertIn("telemetry_emission", channel.blocked_runtime_behaviors)
            self.assertTrue(channel.production_telemetry_channel_implemented)
            self.assertFalse(channel.telemetry_emission_implemented)
            self.assertFalse(channel.live_metrics_collection_implemented)
            self.assertFalse(channel.trace_capture_implemented)
            self.assertFalse(channel.event_export_implemented)
            self.assertFalse(channel.external_monitoring_sink_implemented)
            self.assertFalse(channel.persistent_storage_write_implemented)
            self.assertFalse(channel.alert_emission_implemented)
            self.assertFalse(channel.hitl_request_implemented)
            self.assertFalse(channel.provider_model_routing_implemented)
            self.assertFalse(channel.model_selection_implemented)
            self.assertFalse(channel.workflow_control_implemented)
            self.assertFalse(channel.workflow_execution_implemented)
            self.assertFalse(channel.agent_invocation_implemented)
            self.assertFalse(channel.node_handler_invocation_implemented)
            self.assertFalse(channel.retry_triggering_implemented)
            self.assertFalse(channel.prompt_mutation_implemented)
            self.assertFalse(channel.generated_output_mutation_implemented)
            self.assertTrue(channel.advisory_only)

        boundary = production_telemetry_channel_by_id(
            "production_telemetry::emission_boundary",
            telemetry,
        )
        self.assertIsNotNone(boundary)
        assert boundary is not None
        self.assertEqual(boundary.status, "guarded")
        self.assertEqual(boundary.telemetry_signal_count, 0)
        self.assertEqual(boundary.guarded_signal_count, 0)
        self.assertEqual(
            boundary.source_serialization_version,
            "telemetry_emission_boundary.v1",
        )

    def test_lookup_helpers_are_stable_and_non_emitting(self) -> None:
        telemetry = build_production_telemetry()
        token_channel = production_telemetry_channel_by_id(
            "production_telemetry::token_dashboard",
            telemetry,
        )
        guarded = production_telemetry_channels_for_status("guarded", telemetry)
        missing = production_telemetry_channel_by_id("missing", telemetry)

        self.assertIsNone(missing)
        self.assertIsNotNone(token_channel)
        assert token_channel is not None
        self.assertEqual(token_channel.channel_kind, "token_dashboard")
        self.assertGreaterEqual(len(guarded), 1)
        self.assertIn(
            "production_telemetry::emission_boundary",
            tuple(channel.channel_id for channel in guarded),
        )

    def test_surface_rejects_mismatched_channel_totals(self) -> None:
        telemetry = build_production_telemetry()
        payload = telemetry.model_dump(mode="json")
        payload["channel_ids"] = ("missing",) + tuple(payload["channel_ids"][1:])

        with self.assertRaisesRegex(ValueError, "channel_ids must match"):
            ProductionTelemetrySurface(**payload)

        payload = telemetry.model_dump(mode="json")
        payload["telemetry_signal_count"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "telemetry_signal_count must match",
        ):
            ProductionTelemetrySurface(**payload)

        payload = telemetry.model_dump(mode="json")
        payload["production_telemetry_status"] = "ready"

        with self.assertRaisesRegex(
            ValueError,
            "production_telemetry_status must match",
        ):
            ProductionTelemetrySurface(**payload)

        payload = telemetry.model_dump(mode="json")
        payload["telemetry_source_registries"] = ("missing",) + tuple(
            payload["telemetry_source_registries"][1:]
        )

        with self.assertRaisesRegex(
            ValueError,
            "telemetry_source_registries must match",
        ):
            ProductionTelemetrySurface(**payload)

    def test_surface_does_not_declare_runtime_telemetry_application_terms(
        self,
    ) -> None:
        telemetry = build_production_telemetry()
        combined_text = " ".join(
            (
                telemetry.authority_boundary,
                *telemetry.blocked_runtime_behaviors,
                *telemetry.advisory_actions,
                *(
                    field
                    for channel in telemetry.channels
                    for field in (
                        channel.channel_id,
                        channel.source_id,
                        *(channel.source_item_ids),
                        *(channel.evidence),
                        *(channel.advisory_actions),
                        *(channel.blocked_runtime_behaviors),
                    )
                ),
            )
        )

        for forbidden_term in (
            "emit_telemetry(",
            "collect_live_metrics(",
            "capture_trace(",
            "export_events(",
            "send_alert(",
            "route_model(",
            "execute_workflow(",
            "invoke_agent(",
            "write_storage(",
            "mutate_generated_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


if __name__ == "__main__":
    unittest.main()
