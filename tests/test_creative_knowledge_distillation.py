import unittest
from pathlib import Path

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.knowledge import (
    CreativeKnowledgeRecordKind,
    DemoKBSyncFeasibility,
    KnowledgeDomainCoverageStatus,
    build_demo_kb_hardening_manifest,
    build_kb_reality_snapshot,
    build_repository_knowledge_graph,
    build_v8_1_creative_knowledge_distillation,
    inventory_local_chroma_kb,
)

PARTIAL_INDEXED_COUNTS = {
    "three_manual_effects": 5,
    "three_manual": 4,
    "glsl_mdn_webgl_examples": 3,
    "p5_reference": 2,
}


class CreativeKnowledgeDistillationTests(unittest.TestCase):
    def test_kb_reality_snapshot_separates_registry_from_indexed_sources(self) -> None:
        snapshot = build_kb_reality_snapshot(
            indexed_chunk_counts_by_source=PARTIAL_INDEXED_COUNTS,
        )

        self.assertEqual(snapshot.registry_source_count, 57)
        self.assertEqual(snapshot.registry_domain_count, 43)
        self.assertEqual(snapshot.indexed_source_count, 4)
        self.assertEqual(snapshot.indexed_chunk_count, 14)
        self.assertEqual(snapshot.demo_required_source_count, 12)
        self.assertEqual(snapshot.demo_indexed_source_count, 4)
        self.assertIn("hydra_docs", snapshot.unindexed_demo_source_ids)
        self.assertIn("tone_js_docs", snapshot.unindexed_demo_source_ids)

        tone_coverage = next(
            item
            for item in snapshot.domain_coverage
            if item.domain is CreativeCodingDomain.TONE_JS
        )
        self.assertEqual(tone_coverage.status, KnowledgeDomainCoverageStatus.REGISTERED_ONLY)
        self.assertEqual(
            tone_coverage.missing_demo_source_ids,
            ("tone_js_analysis_reference", "tone_js_docs"),
        )

    def test_distillation_builds_provenance_confidence_and_hardening_actions(self) -> None:
        report = build_v8_1_creative_knowledge_distillation(
            indexed_chunk_counts_by_source=PARTIAL_INDEXED_COUNTS,
        )

        self.assertEqual(report.capability_id, "v8_1_creative_knowledge_distillation")
        self.assertFalse(report.chroma_write_implemented)
        self.assertFalse(report.source_registry_mutation_implemented)
        self.assertFalse(report.holomind_implemented)
        self.assertGreaterEqual(len(report.records), 7)
        self.assertGreater(len(report.relationships), 0)
        self.assertGreater(len(report.taxonomy_nodes), 0)
        self.assertIn("Creative Technique Extraction", report.implemented_roadmap_items)
        self.assertIn("PDF/Paper Distillation", report.deferred_roadmap_items)

        kinds = {record.kind for record in report.records}
        self.assertIn(CreativeKnowledgeRecordKind.TECHNIQUE, kinds)
        self.assertIn(CreativeKnowledgeRecordKind.WORKFLOW, kinds)
        self.assertIn(CreativeKnowledgeRecordKind.PATTERN, kinds)
        self.assertIn(CreativeKnowledgeRecordKind.BEST_PRACTICE, kinds)

        audio_record = next(
            record
            for record in report.records
            if record.record_id == "creative_knowledge::audio_reactive_browser_mapping"
        )
        self.assertEqual(audio_record.kind, CreativeKnowledgeRecordKind.TECHNIQUE)
        self.assertIn("audio_reactive_mappings", audio_record.technique_tags)
        self.assertTrue(audio_record.confidence.caveats)
        self.assertIn("p5_sound_analysis_reference", audio_record.confidence.caveats[0])

        hardening_ids = tuple(action.action_id for action in report.hardening_actions)
        self.assertIn("v8_1_hardening::index_demo_required_sources", hardening_ids)
        self.assertIn("v8_1_hardening::separate_registry_from_index_claims", hardening_ids)

    def test_repository_graph_links_sources_retrieval_demo_eval_and_architecture(self) -> None:
        graph = build_repository_knowledge_graph()

        node_ids = tuple(node.node_id for node in graph.nodes)
        edge_ids = tuple(edge.edge_id for edge in graph.edges)
        self.assertIn("repo::source_registry", node_ids)
        self.assertIn("repo::retrieval_runtime", node_ids)
        self.assertIn("doc::eval_pipeline", node_ids)
        self.assertIn("architecture::engine_matrix", node_ids)
        self.assertIn("repo::source_registry->repo::demo_pack", edge_ids)
        self.assertFalse(graph.mutation_implemented)

    def test_demo_kb_hardening_manifest_reports_focused_sync_blocker(self) -> None:
        manifest = build_demo_kb_hardening_manifest(
            indexed_chunk_counts_by_source=PARTIAL_INDEXED_COUNTS,
            sync_probe_error="KeyError: '_type'",
        )

        self.assertEqual(manifest.manifest_id, "v8_1_demo_kb_hardening_manifest")
        self.assertIsNone(manifest.kb_manifest_path)
        self.assertFalse(manifest.kb_manifest_path_standardized)
        self.assertEqual(manifest.sync_feasibility, DemoKBSyncFeasibility.BLOCKED_BY_SYNC_PROBE)
        self.assertEqual(manifest.sync_probe_error, "KeyError: '_type'")
        self.assertEqual(manifest.focused_sync_source_ids, manifest.missing_demo_source_ids)
        self.assertIn("hydra_docs", manifest.focused_sync_source_ids)
        self.assertIn("--source-id hydra_docs", manifest.focused_sync_command)
        self.assertIn("--continue-on-error", manifest.focused_sync_command)
        self.assertFalse(manifest.coverage_improved)
        self.assertTrue(manifest.report_only)
        self.assertFalse(manifest.external_fetch_implemented)
        self.assertFalse(manifest.chroma_write_implemented)
        self.assertFalse(manifest.source_registry_mutation_implemented)

    def test_missing_chroma_inventory_is_read_only_and_empty(self) -> None:
        inventory = inventory_local_chroma_kb(Path("missing/chroma.sqlite3"))

        self.assertFalse(inventory.chroma_exists)
        self.assertEqual(inventory.source_chunk_counts, {})
        self.assertEqual(inventory.collection_counts, {})


if __name__ == "__main__":
    unittest.main()
