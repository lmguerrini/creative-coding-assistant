import unittest
from datetime import UTC, datetime, timedelta

from creative_coding_assistant.orchestration import (
    ExecutionCacheEntry,
    ExecutionCacheLookup,
    InMemoryExecutionCache,
    build_execution_cache_key,
    execution_cache_entry_is_fresh,
)


class CacheLayerTests(unittest.TestCase):
    def test_cache_key_is_stable_and_order_independent(self) -> None:
        first = build_execution_cache_key(
            namespace="prompt",
            components={"route": "generate", "version": 1},
        )
        second = build_execution_cache_key(
            namespace="prompt",
            components={"version": 1, "route": "generate"},
        )

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("cache::prompt::"))

    def test_cache_put_and_hit_preserve_in_memory_boundaries(self) -> None:
        cache = InMemoryExecutionCache()
        now = _now()
        entry = cache.put(
            namespace="context",
            components={"conversation_id": "conversation-1"},
            payload={"summary": "cached"},
            ttl_seconds=60,
            tags=("context",),
            now=now,
        )
        lookup = cache.get(
            namespace="context",
            components={"conversation_id": "conversation-1"},
            now=now + timedelta(seconds=10),
        )

        self.assertEqual(lookup.role, "execution_cache_layer")
        self.assertEqual(lookup.serialization_version, "execution_cache_lookup.v1")
        self.assertEqual(lookup.status, "hit")
        self.assertEqual(lookup.entry, entry)
        self.assertIsNone(lookup.stale_entry)
        self.assertTrue(execution_cache_entry_is_fresh(entry, now=now))
        self.assertTrue(lookup.cache_layer_implemented)
        self.assertFalse(lookup.persistent_storage_write_implemented)
        self.assertFalse(lookup.network_cache_access_implemented)
        self.assertFalse(lookup.provider_model_routing_implemented)
        self.assertFalse(lookup.workflow_control_implemented)
        self.assertFalse(lookup.memory_write_implemented)
        self.assertFalse(lookup.generated_output_mutation_implemented)
        self.assertTrue(lookup.in_memory_only)
        self.assertIn("persistent_storage_write", entry.blocked_runtime_behaviors)
        self.assertFalse(entry.persistent_storage_write_implemented)
        self.assertTrue(entry.in_memory_only)

    def test_cache_reports_miss_stale_and_invalidation(self) -> None:
        cache = InMemoryExecutionCache()
        now = _now()
        miss = cache.get(namespace="retrieval", components={"query": "p5"}, now=now)
        entry = cache.put(
            namespace="retrieval",
            components={"query": "p5"},
            payload={"chunks": 2},
            ttl_seconds=5,
            now=now,
        )
        stale = cache.get(
            namespace="retrieval",
            components={"query": "p5"},
            now=now + timedelta(seconds=6),
        )

        self.assertEqual(miss.status, "miss")
        self.assertIsNone(miss.entry)
        self.assertEqual(stale.status, "stale")
        self.assertIsNone(stale.entry)
        self.assertEqual(stale.stale_entry, entry)
        self.assertFalse(
            execution_cache_entry_is_fresh(entry, now=now + timedelta(seconds=6))
        )
        self.assertTrue(
            cache.invalidate(namespace="retrieval", components={"query": "p5"})
        )
        self.assertFalse(
            cache.invalidate(namespace="retrieval", components={"query": "p5"})
        )
        self.assertEqual(cache.snapshot(), ())

    def test_models_reject_mismatched_payloads_and_statuses(self) -> None:
        now = _now()
        cache = InMemoryExecutionCache()
        entry = cache.put(
            namespace="generic",
            components={"value": "key"},
            payload={"value": "cached"},
            now=now,
        )
        payload = entry.model_dump(mode="json")
        payload["payload"] = {"value": "changed"}

        with self.assertRaisesRegex(ValueError, "payload_fingerprint must match"):
            ExecutionCacheEntry(**payload)

        with self.assertRaisesRegex(ValueError, "hit lookup requires entry"):
            ExecutionCacheLookup(
                namespace="generic",
                cache_key="cache::generic::abc",
                status="hit",
                looked_up_at=now,
                advisory_actions=("Use cache.",),
            )

    def test_lookup_does_not_declare_persistent_or_routing_terms(self) -> None:
        cache = InMemoryExecutionCache()
        lookup = cache.get(
            namespace="workflow", components={"node": "planning"}, now=_now()
        )
        combined_text = " ".join(
            (
                lookup.authority_boundary,
                *lookup.blocked_runtime_behaviors,
                *lookup.advisory_actions,
            )
        )

        for forbidden_term in (
            "write_persistent_storage(",
            "network_cache_get(",
            "route_provider(",
            "control_workflow(",
            "write_memory(",
            "modify_output(",
        ):
            self.assertNotIn(forbidden_term, combined_text)


def _now() -> datetime:
    return datetime(2026, 6, 28, 12, 0, tzinfo=UTC)


if __name__ == "__main__":
    unittest.main()
