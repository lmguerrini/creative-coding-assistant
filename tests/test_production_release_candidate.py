import unittest

from creative_coding_assistant.orchestration import (
    ProductionReleaseCandidatePlan,
    adaptive_execution_availability_context,
    build_production_release_candidate,
    release_candidate_record_by_surface,
    release_candidate_records_for_status,
)
from creative_coding_assistant.orchestration.routing import RouteName

REQUIRED_SURFACES = (
    "baseline_validation",
    "final_optimization_readiness",
    "packaging_readiness",
    "production_safety_boundaries",
    "release_operation_controls",
)
REQUIRED_CHECKS = (
    "full_validation",
    "final_optimization",
    "packaging",
    "codex_engineering_audit",
    "runtime_failure_path_audit",
    "local_app_smoke_test",
    "merge_push_tag_gate",
)
REQUIRED_RECORD_FIELDS = {
    "record_id",
    "surface_id",
    "status",
    "source_surface_ids",
    "source_serialization_versions",
    "evidence",
    "guarded_reason_codes",
    "required_followups",
    "release_blocker",
    "blocked_runtime_behaviors",
    "release_candidate_record_implemented",
    "release_artifact_creation_implemented",
    "package_build_executed",
    "deployment_execution_implemented",
    "merge_operation_implemented",
    "push_operation_implemented",
    "tag_operation_implemented",
    "provider_model_routing_implemented",
    "provider_execution_implemented",
    "runtime_installation_implemented",
    "hitl_request_emitted",
    "workflow_execution_implemented",
    "generated_output_mutation_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "metadata_only",
}


class ProductionReleaseCandidateTests(unittest.TestCase):
    def test_release_candidate_composes_optimization_and_packaging(self) -> None:
        plan = build_production_release_candidate()

        self.assertEqual(plan.role, "production_release_candidate")
        self.assertEqual(plan.serialization_version, "production_release_candidate_plan.v1")
        self.assertEqual(plan.version, "V5.6.0")
        self.assertEqual(plan.release_candidate_id, "v5.6.0-rc.1")
        self.assertEqual(plan.target_branch, "feature/production-release")
        self.assertEqual(plan.target_tag, "v5.6.0")
        self.assertEqual(plan.route_name, RouteName.GENERATE)
        self.assertEqual(plan.task_type, "creative_coding")
        self.assertEqual(plan.completed_capability_versions, ("V5.1.0", "V5.2.0", "V5.3.0", "V5.4.0", "V5.5.0"))
        self.assertEqual(plan.required_pre_release_checks, REQUIRED_CHECKS)
        self.assertEqual(plan.surface_ids, REQUIRED_SURFACES)
        self.assertEqual(plan.record_count, 5)
        self.assertEqual(plan.release_candidate_status, "guarded")
        self.assertEqual(plan.release_blocker_ids, ())
        self.assertIn("does not create release artifacts", plan.authority_boundary)
        self.assertTrue(plan.release_candidate_metadata_implemented)
        self.assertTrue(plan.validation_gate_recorded)
        self.assertTrue(plan.final_optimization_linked)
        self.assertTrue(plan.packaging_linked)
        self.assertTrue(plan.safety_boundaries_linked)
        self.assertTrue(plan.release_controls_linked)
        self.assertFalse(plan.release_artifact_creation_implemented)
        self.assertFalse(plan.package_build_executed)
        self.assertFalse(plan.deployment_execution_implemented)
        self.assertFalse(plan.merge_operation_implemented)
        self.assertFalse(plan.push_operation_implemented)
        self.assertFalse(plan.tag_operation_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.runtime_installation_implemented)
        self.assertFalse(plan.hitl_request_emitted)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.generated_output_mutation_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.metadata_only)

    def test_records_are_metadata_only_and_surface_guarded_configuration(self) -> None:
        plan = build_production_release_candidate()
        optimization = release_candidate_record_by_surface(
            "final_optimization_readiness",
            plan,
        )
        controls = release_candidate_record_by_surface("release_operation_controls", plan)
        guarded = release_candidate_records_for_status("guarded", plan)

        self.assertIsNotNone(optimization)
        self.assertIsNotNone(controls)
        assert optimization is not None
        assert controls is not None
        self.assertGreaterEqual(len(guarded), 1)
        self.assertIn("missing_api_key", optimization.guarded_reason_codes)
        self.assertEqual(controls.status, "ready")
        self.assertIn("stop_before_merge_push_tag", controls.required_followups)

        for record in plan.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "production_release_candidate_record.v1",
            )
            self.assertEqual(
                record.record_id,
                f"production_release_candidate::{record.surface_id}",
            )
            self.assertFalse(record.release_artifact_creation_implemented)
            self.assertFalse(record.package_build_executed)
            self.assertFalse(record.deployment_execution_implemented)
            self.assertFalse(record.merge_operation_implemented)
            self.assertFalse(record.push_operation_implemented)
            self.assertFalse(record.tag_operation_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.provider_execution_implemented)
            self.assertFalse(record.runtime_installation_implemented)
            self.assertFalse(record.hitl_request_emitted)
            self.assertFalse(record.workflow_execution_implemented)
            self.assertFalse(record.generated_output_mutation_implemented)
            self.assertFalse(record.runtime_evolution_implemented)
            self.assertTrue(record.metadata_only)

    def test_ready_availability_context_marks_candidate_ready(self) -> None:
        context = adaptive_execution_availability_context(
            configured_provider_ids=("openai",),
            safe_auto_risk_bands=("low", "medium"),
        )
        plan = build_production_release_candidate(
            task_type="coding",
            execution_mode_id="auto_mode",
            availability_context=context,
        )

        self.assertEqual(plan.release_candidate_status, "ready")
        self.assertEqual(plan.guarded_record_ids, ())
        self.assertEqual(plan.release_blocker_ids, ())
        self.assertEqual(plan.selected_execution_mode_id, "auto_mode")

    def test_plan_rejects_mismatched_records_or_checks(self) -> None:
        plan = build_production_release_candidate()
        payload = plan.model_dump(mode="json")
        payload["record_ids"] = ("missing",) + tuple(payload["record_ids"][1:])

        with self.assertRaisesRegex(ValueError, "record_ids must match"):
            ProductionReleaseCandidatePlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["required_pre_release_checks"] = tuple(payload["required_pre_release_checks"][1:]) + (
            payload["required_pre_release_checks"][0],
        )

        with self.assertRaisesRegex(ValueError, "required_pre_release_checks"):
            ProductionReleaseCandidatePlan(**payload)


if __name__ == "__main__":
    unittest.main()
