import unittest

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.orchestration import (
    ProductionReleaseFinalOptimizationPlan,
    adaptive_execution_availability_context,
    build_production_release_final_optimization,
    production_optimization_record_by_domain,
    production_optimization_records_for_status,
    route_request,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_DOMAINS = (
    "provider_configuration_review",
    "execution_safety_review",
    "decision_explainability_review",
    "failure_determinism_review",
    "demo_workflow_readiness",
)
REQUIRED_EXPLANATION_FIELDS = (
    "selected_provider",
    "selected_model",
    "execution_mode",
    "execution_strategy",
    "quality_estimate",
    "cost_estimate",
    "latency_estimate",
    "fallback_strategy",
    "escalation_reason",
)
REQUIRED_DEMO_STEPS = (
    "task",
    "routing_intelligence",
    "adaptive_execution_policy",
    "execution_simulation",
    "generation",
    "artifact",
    "explanation",
    "final_output",
)
REQUIRED_RECORD_FIELDS = {
    "record_id",
    "domain",
    "status",
    "readiness_score",
    "source_surface_ids",
    "source_serialization_versions",
    "evidence",
    "guarded_reason_codes",
    "release_actions",
    "release_blocker",
    "blocked_runtime_behaviors",
    "production_optimization_record_implemented",
    "provider_configuration_mutation_implemented",
    "environment_configuration_mutation_implemented",
    "api_key_assumption_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "automatic_provider_switching_implemented",
    "automatic_model_download_implemented",
    "provider_provisioning_implemented",
    "runtime_installation_implemented",
    "hitl_request_emitted",
    "workflow_execution_implemented",
    "workflow_control_implemented",
    "persistent_storage_write_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "merge_push_tag_implemented",
    "serialization_version",
    "metadata_only",
}


class ProductionReleaseFinalOptimizationTests(unittest.TestCase):
    def test_plan_reviews_release_critical_optimization_domains(self) -> None:
        plan = build_production_release_final_optimization()

        self.assertEqual(plan.role, "production_release_final_optimization")
        self.assertEqual(
            plan.serialization_version,
            "production_release_final_optimization_plan.v1",
        )
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.provider_ids, ("openai", "anthropic", "gemini", "local"))
        self.assertEqual(plan.execution_mode_ids, ("manual_mode", "assisted_mode", "auto_mode"))
        self.assertEqual(plan.explanation_fields, REQUIRED_EXPLANATION_FIELDS)
        self.assertEqual(plan.demo_workflow_steps, REQUIRED_DEMO_STEPS)
        self.assertEqual(plan.domain_ids, REQUIRED_DOMAINS)
        self.assertEqual(plan.record_count, 5)
        self.assertGreaterEqual(plan.guarded_record_count, 1)
        self.assertEqual(plan.release_blocker_count, 0)
        self.assertEqual(plan.production_optimization_status, "guarded")
        self.assertIn("does not introduce core architecture", plan.authority_boundary)
        self.assertTrue(plan.final_optimization_implemented)
        self.assertTrue(plan.production_configuration_review_implemented)
        self.assertTrue(plan.production_safety_review_implemented)
        self.assertTrue(plan.production_explainability_review_implemented)
        self.assertTrue(plan.production_failure_review_implemented)
        self.assertTrue(plan.demo_readiness_review_implemented)
        self.assertFalse(plan.provider_configuration_mutation_implemented)
        self.assertFalse(plan.environment_configuration_mutation_implemented)
        self.assertFalse(plan.api_key_assumption_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.automatic_provider_switching_implemented)
        self.assertFalse(plan.automatic_model_download_implemented)
        self.assertFalse(plan.provider_provisioning_implemented)
        self.assertFalse(plan.runtime_installation_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertFalse(plan.merge_push_tag_implemented)
        self.assertTrue(plan.metadata_only)

    def test_records_are_metadata_only_and_guard_missing_configuration(self) -> None:
        plan = build_production_release_final_optimization()
        configuration = production_optimization_record_by_domain(
            "provider_configuration_review",
            plan,
        )
        safety = production_optimization_record_by_domain("execution_safety_review", plan)
        explainability = production_optimization_record_by_domain(
            "decision_explainability_review",
            plan,
        )

        self.assertIsNotNone(configuration)
        self.assertIsNotNone(safety)
        self.assertIsNotNone(explainability)
        assert configuration is not None
        assert safety is not None
        assert explainability is not None
        self.assertEqual(configuration.status, "guarded")
        self.assertIn("missing_api_key", configuration.guarded_reason_codes)
        self.assertIn(
            "hitl_before_provider_provisioning",
            configuration.guarded_reason_codes,
        )
        self.assertEqual(explainability.status, "ready")

        for record in plan.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "production_release_optimization_record.v1",
            )
            self.assertEqual(
                record.record_id,
                f"production_optimization::{record.domain}",
            )
            self.assertEqual(
                len(record.source_surface_ids),
                len(record.source_serialization_versions),
            )
            self.assertTrue(record.production_optimization_record_implemented)
            self.assertFalse(record.provider_configuration_mutation_implemented)
            self.assertFalse(record.environment_configuration_mutation_implemented)
            self.assertFalse(record.api_key_assumption_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.provider_execution_implemented)
            self.assertFalse(record.automatic_provider_switching_implemented)
            self.assertFalse(record.automatic_model_download_implemented)
            self.assertFalse(record.provider_provisioning_implemented)
            self.assertFalse(record.runtime_installation_implemented)
            self.assertFalse(record.hitl_request_emitted)
            self.assertFalse(record.workflow_execution_implemented)
            self.assertFalse(record.workflow_control_implemented)
            self.assertFalse(record.persistent_storage_write_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)
            self.assertFalse(record.runtime_evolution_implemented)
            self.assertFalse(record.merge_push_tag_implemented)
            self.assertTrue(record.metadata_only)

    def test_ready_context_removes_guarded_release_reasons(self) -> None:
        context = adaptive_execution_availability_context(
            configured_provider_ids=("openai",),
            safe_auto_risk_bands=("low", "medium"),
        )
        plan = build_production_release_final_optimization(
            task_type="coding",
            execution_mode_id="auto_mode",
            availability_context=context,
        )

        self.assertEqual(plan.production_optimization_status, "ready")
        self.assertEqual(plan.guarded_record_ids, ())
        self.assertEqual(plan.guarded_record_count, 0)
        self.assertEqual(plan.release_blocker_ids, ())
        self.assertEqual(plan.unavailable_reason_codes, ())
        self.assertEqual(plan.required_hitl_gates, ())
        self.assertEqual(plan.selected_execution_mode_id, "auto_mode")
        for record in production_optimization_records_for_status("ready", plan):
            self.assertEqual(record.status, "ready")
            self.assertFalse(record.guarded_reason_codes)

    def test_plan_rejects_mismatched_records_or_domains(self) -> None:
        plan = build_production_release_final_optimization()
        payload = plan.model_dump(mode="json")
        payload["record_ids"] = ("missing",) + tuple(payload["record_ids"][1:])

        with self.assertRaisesRegex(ValueError, "record_ids must match"):
            ProductionReleaseFinalOptimizationPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["domain_ids"] = tuple(payload["domain_ids"][1:]) + (
            payload["domain_ids"][0],
        )

        with self.assertRaisesRegex(ValueError, "domain_ids must match records"):
            ProductionReleaseFinalOptimizationPlan(**payload)

    def test_final_optimization_does_not_change_routing(self) -> None:
        request = AssistantRequest(
            query="Create a luminous shader study for a capstone demo.",
            mode=AssistantMode.GENERATE,
            domain=CreativeCodingDomain.THREE_JS,
        )

        before = route_request(request)
        plan = build_production_release_final_optimization()
        after = route_request(request)

        self.assertEqual(before.route, RouteName.GENERATE)
        self.assertEqual(after.route, before.route)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)


if __name__ == "__main__":
    unittest.main()
