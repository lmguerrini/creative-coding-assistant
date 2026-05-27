from __future__ import annotations

import json
import os
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from creative_coding_assistant.analytics import (
    build_langsmith_observability,
    build_langsmith_runtime_config,
)
from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)
from creative_coding_assistant.core import Settings, load_settings
from creative_coding_assistant.eval.live_session import (
    LiveSessionEvalSample,
    LiveSessionRetrievedContext,
    LiveSessionRouteMetadata,
)
from creative_coding_assistant.eval.ragas_runner import (
    ragas_run_manifest_path,
    run_ragas_live_eval,
)
from creative_coding_assistant.orchestration import (
    AssistantService,
    RetrievalContextRequest,
    RetrievalContextResponse,
    RetrievalContextSource,
    RetrievedKnowledgeChunk,
    RouteCapability,
    RouteDecision,
    RouteName,
)
from creative_coding_assistant.rag.sources import OfficialSourceType


class LangSmithObservabilityTests(unittest.TestCase):
    def test_langsmith_runtime_is_disabled_by_default(self) -> None:
        settings = Settings(_env_file=None)
        config = build_langsmith_runtime_config(settings)
        observability = build_langsmith_observability(settings)
        run = observability.assistant_run_context(
            AssistantRequest(query="Create a calm p5 sketch.")
        )

        self.assertFalse(config.requested)
        self.assertFalse(config.enabled)
        self.assertEqual(config.reason, "tracing_disabled")
        self.assertIsNone(observability.event_payload(run))

        with observability.trace(run):
            pass

    def test_langsmith_env_aliases_enable_safe_runtime_config(self) -> None:
        with _temporary_langsmith_env(
            LANGSMITH_TRACING="true",
            LANGSMITH_API_KEY="lsv2-test-secret",
            LANGSMITH_PROJECT="creative-prod",
            LANGSMITH_ENDPOINT="https://api.smith.langchain.com",
        ):
            settings = load_settings()

        config = build_langsmith_runtime_config(settings)
        payload = config.summary_payload()

        self.assertTrue(config.requested)
        self.assertTrue(config.enabled)
        self.assertTrue(settings.has_langsmith_api_key)
        self.assertEqual(config.project_name, "creative-prod")
        self.assertEqual(config.endpoint, "https://api.smith.langchain.com")
        self.assertNotIn("lsv2-test-secret", json.dumps(payload))

    def test_missing_langsmith_key_is_non_blocking_and_visible(self) -> None:
        settings = Settings(langsmith_tracing=True, langsmith_api_key=None)
        service = AssistantService(settings=settings)

        events = tuple(
            service.stream(
                AssistantRequest(
                    query="Generate a Three.js particle field.",
                    domain=CreativeCodingDomain.THREE_JS,
                    mode=AssistantMode.GENERATE,
                )
            )
        )

        self.assertEqual(len(events), 3)
        self.assertEqual(
            events[0].payload["observability"]["reason"],
            "missing_api_key",
        )
        self.assertEqual(
            events[-1].payload["observability"]["trace_kind"],
            "assistant_workflow",
        )
        self.assertEqual(
            events[-1].payload["observability"]["lineage"]["stage"],
            "finalization",
        )

    def test_retrieval_event_carries_langsmith_lineage_when_requested(self) -> None:
        service = AssistantService(
            settings=Settings(langsmith_tracing=True, langsmith_api_key=None),
            route_fn=_route_with_retrieval,
            retrieval_gateway=_FakeRetrievalGateway(),
        )

        events = tuple(service.stream(_request()))
        retrieval_event = next(
            event
            for event in events
            if event.payload.get("code") == "retrieval_completed"
        )
        observability = retrieval_event.payload["observability"]

        self.assertEqual(observability["reason"], "missing_api_key")
        self.assertEqual(observability["lineage"]["stage"], "retrieval")
        self.assertEqual(observability["lineage"]["source"], "official_kb")
        self.assertEqual(observability["lineage"]["chunk_count"], 1)
        self.assertEqual(observability["lineage"]["source_ids"], ["three_docs"])

    def test_ragas_manifest_records_langsmith_eval_lineage_when_requested(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "live_sessions.jsonl"
            output_path = Path(temp_dir) / "ragas_results.jsonl"
            input_path.write_text(
                _sample(sample_id="sample-1").model_dump_json(),
                encoding="utf-8",
            )
            observability = build_langsmith_observability(
                Settings(
                    langsmith_tracing=True,
                    langsmith_api_key=None,
                    langsmith_project="eval-project",
                )
            )

            result = run_ragas_live_eval(
                input_path=input_path,
                output_path=output_path,
                dry_run=True,
                run_id="dry-run",
                evaluated_at=datetime(2026, 4, 29, 12, 0, tzinfo=UTC),
                langsmith_observability=observability,
            )

            manifest = json.loads(
                ragas_run_manifest_path(output_path).read_text(encoding="utf-8")
            )

        self.assertIsNotNone(result.manifest.langsmith)
        self.assertEqual(manifest["langsmith"]["reason"], "missing_api_key")
        self.assertEqual(manifest["langsmith"]["project_name"], "eval-project")
        self.assertEqual(
            manifest["langsmith"]["metadata"]["eval_run_id"],
            "dry-run",
        )
        self.assertEqual(
            manifest["langsmith"]["metadata"]["metrics"],
            ["context_precision"],
        )


def _request() -> AssistantRequest:
    return AssistantRequest(
        query="Explain this Three.js camera setup.",
        domain=CreativeCodingDomain.THREE_JS,
        mode=AssistantMode.EXPLAIN,
    )


def _route_with_retrieval(request: AssistantRequest) -> RouteDecision:
    return RouteDecision(
        route=RouteName.EXPLAIN,
        mode=request.mode,
        domain=request.domain,
        domains=request.domains,
        capabilities=(RouteCapability.OFFICIAL_DOCS,),
    )


class _FakeRetrievalGateway:
    def retrieve_context(
        self,
        request: RetrievalContextRequest,
    ) -> RetrievalContextResponse:
        return RetrievalContextResponse(
            request=request,
            source=RetrievalContextSource.OFFICIAL_KB,
            chunks=(
                RetrievedKnowledgeChunk(
                    source_id="three_docs",
                    domain=CreativeCodingDomain.THREE_JS,
                    source_type=OfficialSourceType.API_REFERENCE,
                    publisher="three.js",
                    registry_title="three.js Documentation",
                    document_title="PerspectiveCamera",
                    source_url="https://threejs.org/docs/",
                    resolved_url="https://threejs.org/docs/",
                    chunk_index=0,
                    excerpt="PerspectiveCamera controls field of view.",
                    score=0.83,
                ),
            ),
        )


def _sample(*, sample_id: str) -> LiveSessionEvalSample:
    return LiveSessionEvalSample(
        sample_id=sample_id,
        question="How does a Three.js camera work?",
        answer="PerspectiveCamera controls field of view and projection.",
        conversation_id="conversation-1",
        route=LiveSessionRouteMetadata(
            route=RouteName.EXPLAIN,
            mode=AssistantMode.EXPLAIN,
            domains=(CreativeCodingDomain.THREE_JS,),
            capabilities=(RouteCapability.OFFICIAL_DOCS,),
        ),
        retrieved_contexts=(
            LiveSessionRetrievedContext(
                source_id="three_docs",
                domain=CreativeCodingDomain.THREE_JS,
                source_type=OfficialSourceType.API_REFERENCE,
                publisher="three.js",
                registry_title="three.js Documentation",
                document_title="PerspectiveCamera",
                source_url="https://threejs.org/docs/",
                chunk_index=0,
                excerpt="PerspectiveCamera controls field of view.",
                score=0.83,
            ),
        ),
        started_at=datetime(2026, 4, 29, 11, 0, tzinfo=UTC),
        completed_at=datetime(2026, 4, 29, 11, 0, 2, tzinfo=UTC),
        recorded_at=datetime(2026, 4, 29, 11, 0, 3, tzinfo=UTC),
    )


@contextmanager
def _temporary_langsmith_env(**updates: str) -> Iterator[None]:
    keys = {
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
        "CCA_LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
        "CCA_LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGCHAIN_PROJECT",
        "CCA_LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
        "LANGCHAIN_ENDPOINT",
        "CCA_LANGSMITH_ENDPOINT",
    }
    original_values = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, original in original_values.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


if __name__ == "__main__":
    unittest.main()
