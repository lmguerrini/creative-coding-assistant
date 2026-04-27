"""Live-session evaluation recorders built from existing assistant traces."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from loguru import logger

from creative_coding_assistant.contracts import AssistantRequest, StreamEvent
from creative_coding_assistant.core import Settings
from creative_coding_assistant.eval.live_session import (
    LiveSessionEvalSample,
    LiveSessionRetrievedContext,
    LiveSessionRouteMetadata,
)
from creative_coding_assistant.orchestration.retrieval import RetrievalContextResponse
from creative_coding_assistant.orchestration.routing import RouteDecision


class LiveSessionRecorder(Protocol):
    def record(
        self,
        *,
        request: AssistantRequest,
        events: Sequence[StreamEvent],
        started_at: datetime,
        completed_at: datetime,
    ) -> LiveSessionEvalSample | None:
        """Record one completed live session sample from existing stream events."""


class JsonlLiveSessionRecorder:
    """Append real live-session evaluation samples to a local JSONL file."""

    def __init__(self, *, output_path: Path) -> None:
        self._output_path = output_path

    def record(
        self,
        *,
        request: AssistantRequest,
        events: Sequence[StreamEvent],
        started_at: datetime,
        completed_at: datetime,
    ) -> LiveSessionEvalSample | None:
        sample = build_live_session_sample(
            request=request,
            events=events,
            started_at=started_at,
            completed_at=completed_at,
        )
        if sample is None:
            return None

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._output_path.open("a", encoding="utf-8") as handle:
            handle.write(sample.model_dump_json())
            handle.write("\n")
        logger.info(
            "Recorded live session eval sample '{}' to '{}'",
            sample.sample_id,
            self._output_path,
        )
        return sample


def build_live_session_eval_recorder(
    settings: Settings,
) -> JsonlLiveSessionRecorder:
    """Build the default local-first recorder for live session evaluation."""

    return JsonlLiveSessionRecorder(output_path=settings.eval_data_path)


def build_live_session_sample(
    *,
    request: AssistantRequest,
    events: Sequence[StreamEvent],
    started_at: datetime,
    completed_at: datetime,
    recorded_at: datetime | None = None,
    sample_id: str | None = None,
) -> LiveSessionEvalSample | None:
    final_event = _last_event(events, event_type="final", code=None)
    if final_event is None:
        return None

    answer = str(final_event.payload.get("answer", "")).strip()
    if not answer:
        return None

    route_decision = _extract_route_decision(events)
    retrieval_context = _extract_retrieval_context(events)
    resolved_recorded_at = recorded_at or datetime.now(UTC)
    resolved_route = LiveSessionRouteMetadata(
        route=(route_decision.route if route_decision is not None else None),
        mode=(route_decision.mode if route_decision is not None else request.mode),
        domain=(
            route_decision.domain if route_decision is not None else request.domain
        ),
        domains=(
            route_decision.domains if route_decision is not None else request.domains
        ),
        domain_selection=(
            route_decision.domain_selection if route_decision is not None else None
        ),
        capabilities=(
            route_decision.capabilities if route_decision is not None else ()
        ),
    )
    retrieved_contexts = (
        tuple(
            LiveSessionRetrievedContext.from_chunk(chunk)
            for chunk in retrieval_context.chunks
        )
        if retrieval_context is not None
        else ()
    )
    return LiveSessionEvalSample(
        sample_id=sample_id or uuid4().hex,
        question=request.query,
        answer=answer,
        conversation_id=request.conversation_id,
        project_id=request.project_id,
        route=resolved_route,
        retrieved_contexts=retrieved_contexts,
        started_at=started_at,
        completed_at=completed_at,
        recorded_at=resolved_recorded_at,
    )


def _extract_route_decision(
    events: Sequence[StreamEvent],
) -> RouteDecision | None:
    route_event = _last_event(events, event_type="status", code="route_selected")
    if route_event is None:
        return None

    raw_route = route_event.payload.get("route")
    if not isinstance(raw_route, dict):
        return None
    return RouteDecision.model_validate(raw_route)


def _extract_retrieval_context(
    events: Sequence[StreamEvent],
) -> RetrievalContextResponse | None:
    retrieval_event = _last_event(
        events,
        event_type="retrieval",
        code="retrieval_completed",
    )
    if retrieval_event is None:
        return None

    raw_context = retrieval_event.payload.get("context")
    if not isinstance(raw_context, dict):
        return None
    return RetrievalContextResponse.model_validate(raw_context)


def _last_event(
    events: Sequence[StreamEvent],
    *,
    event_type: str,
    code: str | None,
) -> StreamEvent | None:
    for event in reversed(events):
        if event.event_type.value != event_type:
            continue
        if code is None or event.payload.get("code") == code:
            return event
    return None
