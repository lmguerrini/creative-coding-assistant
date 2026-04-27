from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
    StreamEvent,
    StreamEventType,
)
from creative_coding_assistant.eval import (
    JsonlLiveSessionRecorder,
    build_live_session_sample,
)
from creative_coding_assistant.orchestration import (
    RouteCapability,
    RouteDecision,
    RouteName,
)
from creative_coding_assistant.orchestration.service import AssistantService


class SessionEvalFoundationTests(unittest.TestCase):
    def test_build_live_session_sample_serializes_real_trace_fields(self) -> None:
        request = AssistantRequest(
            query="Explain how fog works in Three.js.",
            conversation_id="conversation-123",
            project_id="project-abc",
            domains=(
                CreativeCodingDomain.THREE_JS,
                CreativeCodingDomain.GLSL,
            ),
            mode=AssistantMode.EXPLAIN,
        )
        started_at = datetime(2026, 4, 27, 9, 0, tzinfo=UTC)
        completed_at = datetime(2026, 4, 27, 9, 0, 2, tzinfo=UTC)
        recorded_at = datetime(2026, 4, 27, 9, 0, 3, tzinfo=UTC)

        sample = build_live_session_sample(
            request=request,
            events=_trace_events(answer="Fog softens distant objects."),
            started_at=started_at,
            completed_at=completed_at,
            recorded_at=recorded_at,
            sample_id="sample-001",
        )

        assert sample is not None
        self.assertEqual(sample.sample_id, "sample-001")
        self.assertEqual(sample.question, "Explain how fog works in Three.js.")
        self.assertEqual(sample.answer, "Fog softens distant objects.")
        self.assertEqual(sample.conversation_id, "conversation-123")
        self.assertEqual(sample.project_id, "project-abc")
        self.assertEqual(sample.route.route, RouteName.EXPLAIN)
        self.assertEqual(sample.route.mode, AssistantMode.EXPLAIN)
        self.assertEqual(len(sample.route.domains), 2)
        self.assertEqual(len(sample.retrieved_contexts), 1)
        self.assertEqual(sample.retrieved_contexts[0].source_id, "three_docs")
        self.assertEqual(sample.started_at, started_at)
        self.assertEqual(sample.completed_at, completed_at)
        self.assertEqual(sample.recorded_at, recorded_at)

        serialized = sample.model_dump(mode="json")
        self.assertEqual(serialized["route"]["route"], "explain")
        self.assertEqual(
            serialized["retrieved_contexts"][0]["document_title"],
            "Fog",
        )

    def test_build_live_session_sample_handles_missing_retrieval(self) -> None:
        request = AssistantRequest(
            query="Summarize the scene setup.",
            mode=AssistantMode.GENERATE,
        )

        sample = build_live_session_sample(
            request=request,
            events=(
                StreamEvent(
                    event_type=StreamEventType.STATUS,
                    sequence=0,
                    payload={
                        "code": "route_selected",
                        "message": "Route selected.",
                        "route": RouteDecision(
                            route=RouteName.GENERATE,
                            mode=AssistantMode.GENERATE,
                        ).model_dump(mode="json"),
                    },
                ),
                StreamEvent(
                    event_type=StreamEventType.FINAL,
                    sequence=1,
                    payload={"answer": "Scene setup summary."},
                ),
            ),
            started_at=datetime(2026, 4, 27, 10, 0, tzinfo=UTC),
            completed_at=datetime(2026, 4, 27, 10, 0, 1, tzinfo=UTC),
            sample_id="sample-002",
        )

        assert sample is not None
        self.assertEqual(sample.retrieved_contexts, ())
        self.assertEqual(sample.route.route, RouteName.GENERATE)

    def test_jsonl_live_session_recorder_appends_records(self) -> None:
        request = AssistantRequest(
            query="Debug the fragment shader.",
            conversation_id="conversation-xyz",
            domains=(CreativeCodingDomain.GLSL,),
            mode=AssistantMode.DEBUG,
        )
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "live_sessions.jsonl"
            recorder = JsonlLiveSessionRecorder(output_path=output_path)

            sample = recorder.record(
                request=request,
                events=_trace_events(answer="Check the varying precision."),
                started_at=datetime(2026, 4, 27, 11, 0, tzinfo=UTC),
                completed_at=datetime(2026, 4, 27, 11, 0, 1, tzinfo=UTC),
            )

            assert sample is not None
            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["question"], "Debug the fragment shader.")
            self.assertEqual(payload["answer"], "Check the varying precision.")
            self.assertEqual(payload["route"]["mode"], "explain")
            self.assertEqual(payload["retrieved_contexts"][0]["domain"], "three_js")

            second_sample = recorder.record(
                request=request,
                events=_trace_events(answer="Clamp the uv coordinates."),
                started_at=datetime(2026, 4, 27, 11, 1, tzinfo=UTC),
                completed_at=datetime(2026, 4, 27, 11, 1, 1, tzinfo=UTC),
            )

            assert second_sample is not None
            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)

    def test_recorder_skips_incomplete_traces(self) -> None:
        request = AssistantRequest(
            query="What does fog do?",
            mode=AssistantMode.EXPLAIN,
        )
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "live_sessions.jsonl"
            recorder = JsonlLiveSessionRecorder(output_path=output_path)

            sample = recorder.record(
                request=request,
                events=(
                    StreamEvent(
                        event_type=StreamEventType.STATUS,
                        sequence=0,
                        payload={"message": "Request accepted."},
                    ),
                ),
                started_at=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
                completed_at=datetime(2026, 4, 27, 12, 0, 1, tzinfo=UTC),
            )

            self.assertIsNone(sample)
            self.assertFalse(output_path.exists())

    def test_assistant_service_records_completed_live_turns(self) -> None:
        request = AssistantRequest(
            query="Explain orbit controls.",
            conversation_id="conversation-live",
            mode=AssistantMode.EXPLAIN,
        )
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "live_sessions.jsonl"
            recorder = JsonlLiveSessionRecorder(output_path=output_path)
            service = AssistantService(
                route_fn=lambda assistant_request: RouteDecision(
                    route=RouteName.EXPLAIN,
                    mode=assistant_request.mode,
                    capabilities=(RouteCapability.LIVE_EVALUATION,),
                ),
                eval_recorder=recorder,
            )

            events = tuple(service.stream(request))

            self.assertEqual(events[-1].event_type, StreamEventType.FINAL)
            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["question"], "Explain orbit controls.")
            self.assertIn("selected the explain route", payload["answer"])


def _trace_events(*, answer: str) -> tuple[StreamEvent, ...]:
    return (
        StreamEvent(
            event_type=StreamEventType.STATUS,
            sequence=0,
            payload={"message": "Request accepted."},
        ),
        StreamEvent(
            event_type=StreamEventType.STATUS,
            sequence=1,
            payload={
                "code": "route_selected",
                "message": "Route selected.",
                "route": RouteDecision(
                    route=RouteName.EXPLAIN,
                    mode=AssistantMode.EXPLAIN,
                    domains=(
                        CreativeCodingDomain.THREE_JS,
                        CreativeCodingDomain.GLSL,
                    ),
                    capabilities=(
                        RouteCapability.MEMORY_CONTEXT,
                        RouteCapability.OFFICIAL_DOCS,
                        RouteCapability.LIVE_EVALUATION,
                    ),
                ).model_dump(mode="json"),
            },
        ),
        StreamEvent(
            event_type=StreamEventType.RETRIEVAL,
            sequence=2,
            payload={
                "code": "retrieval_completed",
                "message": "Retrieval context prepared.",
                "context": {
                    "request": {
                        "query": "Explain how fog works in Three.js.",
                        "route": "explain",
                        "limit": 5,
                        "filters": {
                            "domain": None,
                            "domains": ["three_js", "glsl"],
                            "source_id": None,
                            "source_type": None,
                            "publisher": None,
                        },
                    },
                    "source": "official_kb",
                    "chunks": [
                        {
                            "source_id": "three_docs",
                            "domain": "three_js",
                            "source_type": "api_reference",
                            "publisher": "three.js",
                            "registry_title": "three.js Documentation",
                            "document_title": "Fog",
                            "source_url": "https://threejs.org/docs/",
                            "resolved_url": "https://threejs.org/docs/#api/en/scenes/Fog",
                            "chunk_index": 0,
                            "excerpt": "Fog defines linear fog for scene depth.",
                            "score": 0.92,
                        }
                    ],
                },
            },
        ),
        StreamEvent(
            event_type=StreamEventType.FINAL,
            sequence=3,
            payload={"answer": answer},
        ),
    )
