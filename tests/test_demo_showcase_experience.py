import json
import unittest
from pathlib import Path

from creative_coding_assistant.orchestration import (
    DemoShowcasePlan,
    build_demo_showcase_plan,
    demo_showcase_coverage_by_item,
    demo_showcase_fallback_by_trigger,
    demo_showcase_flow_by_id,
    demo_showcase_prompt_by_id,
)

REQUIRED_COVERAGE_ITEMS = (
    "Demo Mode",
    "Primary Product Flows",
    "Scenario Coverage",
    "Preview Fallbacks",
    "Demo Prompt Library",
    "Evaluation Summary",
    "Product Boundary Summary",
    "Artifact Availability Check",
    "Workflow Context",
    "Validation Criteria",
    "Ethical AI Summary",
    "Fallback Guidance",
    "Product Explanation",
    "Manual Validation Checklist",
    "Demo Reliability Validation",
    "Deterministic Demo Dataset",
    "Offline Fallback",
    "Provider Failure Recovery",
    "Evaluation Metrics Summary",
)
REQUIRED_CASE_IDS = {
    "case_1_rag_knowledge_assistant",
    "case_2_bounded_agent_automation",
    "case_3_source_grounded_search",
    "case_5_ai_coding_assistant",
    "case_6_advanced_llm_tools",
}
REQUIRED_FALLBACK_TRIGGERS = (
    "provider_failure",
    "retrieval_unavailable",
    "preview_unavailable",
    "network_unavailable",
    "time_overrun",
)


class DemoShowcaseExperienceTests(unittest.TestCase):
    def test_plan_covers_product_surfaces_without_runtime_execution_claims(self) -> None:
        plan = build_demo_showcase_plan()

        self.assertEqual(plan.role, "demo_showcase_experience")
        self.assertEqual(plan.serialization_version, "demo_showcase_plan.v1")
        self.assertEqual(plan.coverage_items, REQUIRED_COVERAGE_ITEMS)
        self.assertEqual(len(plan.coverage_records), 19)
        self.assertEqual({case.case_id for case in plan.capstone_case_alignments}, REQUIRED_CASE_IDS)
        self.assertTrue(plan.demo_mode_prepared)
        self.assertTrue(plan.golden_demo_flows_prepared)
        self.assertTrue(plan.capstone_case_alignment_prepared)
        self.assertTrue(plan.demo_prompt_library_prepared)
        self.assertTrue(plan.evaluation_summary_prepared)
        self.assertTrue(plan.ethical_ai_summary_prepared)
        self.assertTrue(plan.fallback_mode_prepared)
        self.assertTrue(plan.demo_reliability_validation_prepared)
        self.assertFalse(plan.showcase_upload_prepared)
        self.assertTrue(plan.metadata_only)
        self.assertFalse(plan.provider_execution_implemented)
        self.assertFalse(plan.retrieval_execution_implemented)
        self.assertFalse(plan.preview_rendering_execution_implemented)
        self.assertFalse(plan.provider_model_routing_implemented)
        self.assertFalse(plan.workflow_control_implemented)
        self.assertFalse(plan.artifact_mutation_implemented)
        self.assertFalse(plan.persistent_storage_write_implemented)
        self.assertFalse(plan.external_dcc_execution_implemented)
        self.assertFalse(plan.mcp_tool_execution_implemented)
        self.assertFalse(plan.holomind_implemented)
        self.assertFalse(plan.holoiverse_implemented)
        self.assertFalse(plan.merge_push_tag_implemented)
        self.assertFalse(plan.version_freeze_implemented)
        self.assertFalse(plan.grand_review_started)
        self.assertIn("does not execute providers", plan.authority_boundary)
        self.assertIn("does not", plan.authority_boundary)

    def test_flows_prompts_metrics_and_fallbacks_are_queryable(self) -> None:
        plan = build_demo_showcase_plan()
        primary_flow = demo_showcase_flow_by_id("primary_creative_coding_flow", plan)
        primary_prompt = demo_showcase_prompt_by_id("luminous_audio_reactive_three_scene", plan)
        provider_fallback = demo_showcase_fallback_by_trigger("provider_failure", plan)
        demo_mode = demo_showcase_coverage_by_item("Demo Mode", plan)
        metrics = {metric.metric_id: metric for metric in plan.demo_metrics}

        self.assertIsNotNone(primary_flow)
        self.assertIsNotNone(primary_prompt)
        self.assertIsNotNone(provider_fallback)
        self.assertIsNotNone(demo_mode)
        assert primary_flow is not None
        assert primary_prompt is not None
        assert provider_fallback is not None
        assert demo_mode is not None
        self.assertEqual(primary_flow.duration_seconds, 420)
        self.assertIn("luminous audio-reactive Three.js scene", primary_prompt.prompt_text)
        self.assertEqual(provider_fallback.trigger, "provider_failure")
        self.assertIn("without implying", provider_fallback.audience_framing)
        self.assertEqual(demo_mode.status, "prepared")
        self.assertEqual(metrics["ragas_context_precision_workflow"].status, "manual")
        self.assertFalse(metrics["ragas_context_precision_workflow"].live_metric_collection_implemented)

    def test_case_alignment_keeps_secondary_cases_guarded(self) -> None:
        plan = build_demo_showcase_plan()
        alignments = {case.case_id: case for case in plan.capstone_case_alignments}

        self.assertEqual(alignments["case_5_ai_coding_assistant"].alignment_status, "primary")
        self.assertEqual(alignments["case_1_rag_knowledge_assistant"].alignment_status, "primary")
        self.assertEqual(alignments["case_6_advanced_llm_tools"].alignment_status, "primary")
        self.assertEqual(alignments["case_2_bounded_agent_automation"].alignment_status, "guarded_support")
        self.assertEqual(alignments["case_3_source_grounded_search"].alignment_status, "guarded_support")
        self.assertIn("Do not claim autonomous agent swarms", alignments["case_2_bounded_agent_automation"].boundary)
        self.assertIn("registered KB", alignments["case_3_source_grounded_search"].boundary)

    def test_compatibility_segments_cover_product_validation_sections(self) -> None:
        plan = build_demo_showcase_plan()

        self.assertEqual(
            {segment.phase for segment in plan.presentation_segments},
            {"product_walkthrough", "validation_notes"},
        )
        self.assertEqual(tuple(fallback.trigger for fallback in plan.fallback_plans), REQUIRED_FALLBACK_TRIGGERS)

    def test_dataset_matches_registered_plan_surfaces(self) -> None:
        plan = build_demo_showcase_plan()
        dataset_path = Path("demo/golden_demo_dataset.json")
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], "v8.8-demo-dataset.v1")
        self.assertEqual(payload["source_plan_serialization_version"], plan.serialization_version)
        self.assertEqual(
            {flow["flow_id"] for flow in payload["golden_flows"]},
            {flow.flow_id for flow in plan.golden_demo_flows},
        )
        self.assertEqual(
            {prompt["prompt_id"] for prompt in payload["prompt_library"]},
            {prompt.prompt_id for prompt in plan.demo_prompt_library},
        )
        self.assertEqual(payload["fallback_triggers"], list(REQUIRED_FALLBACK_TRIGGERS))

    def test_serialized_plan_does_not_expose_private_runtime_pack_paths(self) -> None:
        plan = build_demo_showcase_plan()

        serialized = json.dumps(plan.model_dump(mode="json"), sort_keys=True)

        self.assertNotIn(".runtime_pack/", serialized)
        for private_phrase in (
            "10-minute",
            "5-minute",
            "Q&A",
            "Grand Review",
            "Showcase Upload Preparation",
            "SCR Presentation",
            "SMART Presentation",
            "docs/CAPSTONE_",
        ):
            self.assertNotIn(private_phrase, serialized)
        hitl_item = next(item for item in plan.checklist_items if item.item_id == "confirm_manual_boundaries")
        self.assertEqual(
            hitl_item.evidence_refs,
            ("README.md", "demo/README.md"),
        )

    def test_plan_rejects_mismatched_coverage_or_missing_section(self) -> None:
        plan = build_demo_showcase_plan()
        payload = plan.model_dump(mode="json")
        payload["coverage_items"] = tuple(payload["coverage_items"][1:]) + (payload["coverage_items"][0],)

        with self.assertRaisesRegex(ValueError, "coverage_items"):
            DemoShowcasePlan(**payload)

        payload = plan.model_dump(mode="json")
        for segment in payload["presentation_segments"]:
            segment["phase"] = "product_walkthrough"

        with self.assertRaisesRegex(ValueError, "product walkthrough and validation notes"):
            DemoShowcasePlan(**payload)


if __name__ == "__main__":
    unittest.main()
