import unittest
from pathlib import Path

from creative_coding_assistant.orchestration import (
    ProductionDemoAssetPlan,
    build_production_demo_asset_plan,
    production_demo_asset_by_kind,
    production_demo_assets_for_status,
)

REQUIRED_ASSET_KINDS = (
    "preview_media",
    "demo_prompt",
    "retrieval_scenario_pack",
    "workflow_narrative",
    "explanation_talking_points",
)
REQUIRED_WORKFLOW_STEPS = (
    "Task",
    "Routing Intelligence",
    "Adaptive Execution Policy",
    "Execution Simulation",
    "Generation",
    "Artifact",
    "Explanation",
    "Final Output",
)
REQUIRED_TALKING_POINTS = (
    "selected provider",
    "selected model",
    "execution mode",
    "execution strategy",
    "quality estimate",
    "cost estimate",
    "latency estimate",
    "fallback strategy",
    "escalation reason",
)
REQUIRED_RECORD_FIELDS = {
    "asset_id",
    "asset_kind",
    "status",
    "source_refs",
    "required_items",
    "present_items",
    "missing_items",
    "operator_notes",
    "blocked_runtime_behaviors",
    "demo_asset_record_implemented",
    "asset_generation_implemented",
    "retrieval_execution_implemented",
    "provider_execution_implemented",
    "preview_rendering_execution_implemented",
    "project_bundle_write_implemented",
    "artifact_mutation_implemented",
    "provider_model_routing_implemented",
    "workflow_execution_implemented",
    "workflow_control_implemented",
    "deployment_execution_implemented",
    "merge_push_tag_implemented",
    "runtime_evolution_implemented",
    "serialization_version",
    "metadata_only",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProductionDemoAssetsTests(unittest.TestCase):
    def test_demo_asset_plan_indexes_existing_demo_materials(self) -> None:
        plan = build_production_demo_asset_plan()

        self.assertEqual(plan.role, "production_demo_assets")
        self.assertEqual(plan.serialization_version, "production_demo_asset_plan.v1")
        self.assertEqual(
            plan.source_retrieval_demo_pack_id,
            "creative_coding_retrieval_benchmark",
        )
        self.assertIn("luminous audio-reactive Three.js scene", plan.demo_prompt)
        self.assertIn("product demo", plan.demo_prompt)
        self.assertNotIn("capstone", plan.demo_prompt.casefold())
        self.assertEqual(plan.demo_workflow_steps, REQUIRED_WORKFLOW_STEPS)
        self.assertEqual(plan.explanation_talking_points, REQUIRED_TALKING_POINTS)
        self.assertEqual(
            plan.preview_media_paths,
            (
                "assets/screenshots-archive/preview_current.png",
                "assets/screenshots-archive/preview_v1.png",
                "assets/screenshots-archive/preview_v2.png",
            ),
        )
        for relative_path in plan.preview_media_paths:
            self.assertTrue(
                (PROJECT_ROOT / relative_path).is_file(),
                f"missing fallback preview asset: {relative_path}",
            )
        self.assertGreaterEqual(len(plan.retrieval_scenario_ids), 7)
        self.assertEqual(plan.asset_kinds, REQUIRED_ASSET_KINDS)
        self.assertEqual(plan.asset_count, 5)
        self.assertEqual(plan.guarded_asset_ids, ())
        self.assertEqual(plan.demo_asset_status, "ready")
        self.assertIn("does not generate assets", plan.authority_boundary)
        self.assertTrue(plan.demo_asset_metadata_implemented)
        self.assertTrue(plan.preview_media_inventory_implemented)
        self.assertTrue(plan.demo_prompt_implemented)
        self.assertTrue(plan.retrieval_demo_pack_linked)
        self.assertTrue(plan.workflow_narrative_implemented)
        self.assertTrue(plan.explanation_talking_points_implemented)
        self.assertFalse(plan.asset_generation_implemented)
        self.assertFalse(plan.retrieval_execution_implemented)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.preview_rendering_execution_implemented)
        self.assertFalse(plan.project_bundle_write_implemented)
        self.assertFalse(plan.artifact_mutation_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.workflow_execution_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.deployment_execution_implemented)
        self.assertFalse(plan.merge_push_tag_implemented)
        self.assertFalse(plan.runtime_evolution_implemented)
        self.assertTrue(plan.metadata_only)

    def test_demo_asset_records_are_complete_and_non_executing(self) -> None:
        plan = build_production_demo_asset_plan()
        preview = production_demo_asset_by_kind("preview_media", plan)
        retrieval = production_demo_asset_by_kind("retrieval_scenario_pack", plan)
        ready_assets = production_demo_assets_for_status("ready", plan)

        self.assertIsNotNone(preview)
        self.assertIsNotNone(retrieval)
        assert preview is not None
        assert retrieval is not None
        prompt = production_demo_asset_by_kind("demo_prompt", plan)
        assert prompt is not None
        self.assertEqual(len(ready_assets), 5)
        self.assertIn(
            "assets/screenshots-archive/preview_current.png",
            preview.present_items,
        )
        self.assertIn("shader_post_fx_pipeline", retrieval.present_items)
        self.assertIn("creative_coding_prompt", prompt.present_items)

        for record in plan.records:
            dumped = record.model_dump(mode="json")
            self.assertEqual(set(dumped), REQUIRED_RECORD_FIELDS)
            self.assertEqual(
                record.serialization_version,
                "production_demo_asset_record.v1",
            )
            self.assertEqual(
                record.asset_id,
                f"production_demo_asset::{record.asset_kind}",
            )
            self.assertEqual(record.status, "ready")
            self.assertFalse(record.missing_items)
            self.assertTrue(record.demo_asset_record_implemented)
            self.assertFalse(record.asset_generation_implemented)
            self.assertFalse(record.retrieval_execution_implemented)
            self.assertFalse(record.provider_execution_implemented)
            self.assertFalse(record.preview_rendering_execution_implemented)
            self.assertFalse(record.project_bundle_write_implemented)
            self.assertFalse(record.artifact_mutation_implemented)
            self.assertFalse(record.provider_model_routing_implemented)
            self.assertFalse(record.workflow_execution_implemented)
            self.assertFalse(record.workflow_control_implemented)
            self.assertFalse(record.deployment_execution_implemented)
            self.assertFalse(record.merge_push_tag_implemented)
            self.assertFalse(record.runtime_evolution_implemented)
            self.assertTrue(record.metadata_only)

    def test_plan_rejects_mismatched_asset_ids_or_flow(self) -> None:
        plan = build_production_demo_asset_plan()
        payload = plan.model_dump(mode="json")
        payload["asset_ids"] = ("missing",) + tuple(payload["asset_ids"][1:])

        with self.assertRaisesRegex(ValueError, "asset_ids must match"):
            ProductionDemoAssetPlan(**payload)

        payload = plan.model_dump(mode="json")
        payload["demo_workflow_steps"] = tuple(payload["demo_workflow_steps"][1:]) + (
            payload["demo_workflow_steps"][0],
        )

        with self.assertRaisesRegex(ValueError, "demo_workflow_steps"):
            ProductionDemoAssetPlan(**payload)


if __name__ == "__main__":
    unittest.main()
