import unittest
from datetime import UTC, datetime, timedelta

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.rag import (
    OfficialSourceSyncMetadata,
    OfficialSourceType,
    SourceFreshnessPolicy,
    SourceFreshnessStatus,
    SourceHealthStatus,
    SourceSyncStatus,
    build_official_source_health_metadata,
    evaluate_official_source_health,
    get_official_source,
)


class SourceHealthTests(unittest.TestCase):
    def test_freshness_policy_rejects_inverted_thresholds(self) -> None:
        with self.assertRaisesRegex(ValueError, "warning threshold"):
            SourceFreshnessPolicy(warn_after_hours=48, stale_after_hours=24)

    def test_health_metadata_uses_source_type_defaults(self) -> None:
        metadata = build_official_source_health_metadata("three_docs")

        self.assertEqual(metadata.source_id, "three_docs")
        self.assertEqual(metadata.domain, CreativeCodingDomain.THREE_JS)
        self.assertEqual(metadata.source_type, OfficialSourceType.API_REFERENCE)
        self.assertEqual(metadata.freshness_policy.warn_after_hours, 24 * 45)
        self.assertEqual(metadata.freshness_policy.stale_after_hours, 24 * 120)
        self.assertEqual(metadata.approved_url_count, 1)

    def test_sync_metadata_requires_success_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "last_synced_at"):
            OfficialSourceSyncMetadata(
                source_id="three_docs",
                domain=CreativeCodingDomain.THREE_JS,
                source_type=OfficialSourceType.API_REFERENCE,
                source_url="https://threejs.org/docs/",
                sync_status=SourceSyncStatus.SUCCEEDED,
                requested_at=_time(),
            )

    def test_health_evaluation_marks_sources_fresh_before_stale_threshold(self) -> None:
        source = get_official_source("three_docs")
        sync_metadata = OfficialSourceSyncMetadata.from_success(
            source_id=source.source_id,
            requested_at=_time(),
            last_synced_at=_time(),
            source_url=source.url,
            resolved_url=source.url,
            domain=source.domain,
            source_type=source.source_type,
            content_hash="a" * 64,
            chunk_count=3,
            record_count=3,
        )

        snapshot = evaluate_official_source_health(
            source,
            sync_metadata=sync_metadata,
            checked_at=_time() + timedelta(days=30),
        )

        self.assertEqual(snapshot.freshness_status, SourceFreshnessStatus.FRESH)
        self.assertEqual(snapshot.health_status, SourceHealthStatus.HEALTHY)
        self.assertFalse(snapshot.is_stale)
        self.assertFalse(snapshot.refresh_recommended)

    def test_health_evaluation_marks_sources_stale_after_threshold(self) -> None:
        source = get_official_source("p5_examples")
        sync_metadata = OfficialSourceSyncMetadata.from_success(
            source_id=source.source_id,
            requested_at=_time(),
            last_synced_at=_time(),
            source_url=source.url,
            resolved_url=source.url,
            domain=source.domain,
            source_type=source.source_type,
            content_hash="b" * 64,
            chunk_count=2,
            record_count=2,
        )

        snapshot = evaluate_official_source_health(
            source.source_id,
            sync_metadata=sync_metadata,
            checked_at=_time() + timedelta(days=60),
        )

        self.assertEqual(snapshot.freshness_status, SourceFreshnessStatus.STALE)
        self.assertEqual(snapshot.health_status, SourceHealthStatus.STALE)
        self.assertTrue(snapshot.refresh_recommended)
        self.assertTrue(snapshot.is_stale)
        self.assertGreater(snapshot.age_hours or 0.0, 0.0)

    def test_health_evaluation_returns_unknown_without_sync_metadata(self) -> None:
        snapshot = evaluate_official_source_health(
            "wgsl_spec",
            sync_metadata=None,
            checked_at=_time(),
        )

        self.assertEqual(snapshot.health_status, SourceHealthStatus.UNKNOWN)
        self.assertEqual(snapshot.freshness_status, SourceFreshnessStatus.UNKNOWN)
        self.assertIsNone(snapshot.sync)


def _time() -> datetime:
    return datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


if __name__ == "__main__":
    unittest.main()
