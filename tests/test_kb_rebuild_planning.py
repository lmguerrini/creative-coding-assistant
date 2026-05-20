import unittest
from datetime import UTC, datetime, timedelta

from creative_coding_assistant.app import (
    build_official_kb_rebuild_plan,
    resolve_rebuild_source_ids,
    select_stale_rebuild_source_ids,
)
from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.rag import (
    OfficialSourceSyncMetadata,
    SourceHealthStatus,
    SourceSyncStatus,
    get_official_source,
)


class KnowledgeBaseRebuildPlanningTests(unittest.TestCase):
    def test_rebuild_plan_targets_explicit_sources_in_registry_order(self) -> None:
        plan = build_official_kb_rebuild_plan(
            source_ids=("p5_reference", "three_docs", "p5_reference"),
            checked_at=_time(),
        )

        self.assertEqual(plan.source_ids, ("three_docs", "p5_reference"))
        self.assertEqual(plan.explicit_source_ids, ("p5_reference", "three_docs"))
        self.assertEqual(
            [candidate.reasons[0].value for candidate in plan.candidates],
            ["explicit_source", "explicit_source"],
        )

    def test_rebuild_plan_targets_all_sources_for_selected_domain(self) -> None:
        plan = build_official_kb_rebuild_plan(
            domains=(CreativeCodingDomain.THREE_JS,),
            checked_at=_time(),
        )

        self.assertEqual(
            plan.source_ids,
            ("three_docs", "three_manual", "three_examples"),
        )
        self.assertEqual(plan.affected_domains, (CreativeCodingDomain.THREE_JS,))

    def test_rebuild_plan_selects_only_stale_and_failed_sources(self) -> None:
        plan = build_official_kb_rebuild_plan(
            sync_metadata_by_source={
                "three_docs": _success_metadata("three_docs", age_days=130),
                "p5_reference": _success_metadata("p5_reference", age_days=7),
                "wgsl_spec": _failed_metadata("wgsl_spec"),
            },
            checked_at=_time(),
            stale_only=True,
        )

        self.assertEqual(plan.source_ids, ("three_docs", "wgsl_spec"))
        self.assertEqual(
            plan.summary_payload()["stale_source_ids"],
            ["three_docs"],
        )
        self.assertEqual(
            plan.summary_payload()["sync_failed_source_ids"],
            ["wgsl_spec"],
        )

    def test_rebuild_plan_recommends_warned_but_not_stale_sources(self) -> None:
        plan = build_official_kb_rebuild_plan(
            sync_metadata_by_source={
                "three_docs": _success_metadata("three_docs", age_days=50),
            },
            checked_at=_time(),
            include_refresh_recommended=True,
            include_sync_failed=False,
        )

        self.assertEqual(plan.source_ids, ("three_docs",))
        self.assertEqual(plan.candidates[0].health_status, SourceHealthStatus.HEALTHY)
        self.assertTrue(plan.candidates[0].refresh_recommended)
        self.assertEqual(
            plan.summary_payload()["refresh_recommended_source_ids"],
            ["three_docs"],
        )

    def test_resolve_rebuild_source_ids_filters_stale_domain_targets(self) -> None:
        source_ids = resolve_rebuild_source_ids(
            domains=(CreativeCodingDomain.THREE_JS,),
            sync_metadata_by_source={
                "three_docs": _success_metadata("three_docs", age_days=130),
                "three_manual": _success_metadata("three_manual", age_days=10),
                "three_examples": _failed_metadata("three_examples"),
            },
            checked_at=_time(),
            stale_only=True,
        )

        self.assertEqual(source_ids, ("three_docs", "three_examples"))

    def test_select_stale_rebuild_source_ids_is_a_lightweight_helper(self) -> None:
        source_ids = select_stale_rebuild_source_ids(
            domains=(CreativeCodingDomain.THREE_JS,),
            sync_metadata_by_source={
                "three_docs": _success_metadata("three_docs", age_days=130),
                "three_manual": _success_metadata("three_manual", age_days=10),
            },
            checked_at=_time(),
        )

        self.assertEqual(source_ids, ("three_docs",))


def _success_metadata(source_id: str, *, age_days: int) -> OfficialSourceSyncMetadata:
    source = get_official_source(source_id)
    last_synced_at = _time() - timedelta(days=age_days)
    return OfficialSourceSyncMetadata.from_success(
        source_id=source.source_id,
        requested_at=last_synced_at,
        last_synced_at=last_synced_at,
        source_url=source.url,
        resolved_url=source.url,
        domain=source.domain,
        source_type=source.source_type,
        content_hash="a" * 64,
        chunk_count=1,
        record_count=1,
    )


def _failed_metadata(source_id: str) -> OfficialSourceSyncMetadata:
    source = get_official_source(source_id)
    return OfficialSourceSyncMetadata(
        source_id=source.source_id,
        domain=source.domain,
        source_type=source.source_type,
        source_url=source.url,
        sync_status=SourceSyncStatus.FAILED,
        requested_at=_time(),
        completed_at=_time(),
    )


def _time() -> datetime:
    return datetime(2026, 5, 20, 12, 0, tzinfo=UTC)


if __name__ == "__main__":
    unittest.main()
