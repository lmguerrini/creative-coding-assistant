"""Streaming event construction for backend services."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from creative_coding_assistant.contracts import StreamEvent, StreamEventType
from creative_coding_assistant.preview import PreviewResult
from creative_coding_assistant.tools import ToolRequest, ToolResult, ToolStatus


class StreamEventBuilder:
    """Build monotonically sequenced events for one assistant turn."""

    def __init__(self) -> None:
        self._sequence = 0

    def status(self, *, code: str, message: str, **details: Any) -> StreamEvent:
        return self._event(
            StreamEventType.STATUS,
            {"code": code, "message": message, **details},
        )

    def memory(self, *, code: str, message: str, **details: Any) -> StreamEvent:
        return self._event(
            StreamEventType.MEMORY,
            {"code": code, "message": message, **details},
        )

    def token_delta(self, text: str, **details: Any) -> StreamEvent:
        return self._event(
            StreamEventType.TOKEN_DELTA,
            {"text": text, **details},
        )

    def retrieval(self, *, code: str, message: str, **details: Any) -> StreamEvent:
        return self._event(
            StreamEventType.RETRIEVAL,
            {"code": code, "message": message, **details},
        )

    def context(self, *, code: str, message: str, **details: Any) -> StreamEvent:
        return self._event(
            StreamEventType.CONTEXT,
            {"code": code, "message": message, **details},
        )

    def prompt_input(
        self,
        *,
        code: str,
        message: str,
        **details: Any,
    ) -> StreamEvent:
        return self._event(
            StreamEventType.PROMPT_INPUT,
            {"code": code, "message": message, **details},
        )

    def prompt_rendered(
        self,
        *,
        code: str,
        message: str,
        **details: Any,
    ) -> StreamEvent:
        return self._event(
            StreamEventType.PROMPT_RENDERED,
            {"code": code, "message": message, **details},
        )

    def generation_input(
        self,
        *,
        code: str,
        message: str,
        **details: Any,
    ) -> StreamEvent:
        return self._event(
            StreamEventType.GENERATION_INPUT,
            {"code": code, "message": message, **details},
        )

    def tool_start(self, request: ToolRequest, **details: Any) -> StreamEvent:
        return self._event(
            StreamEventType.TOOL_START,
            {
                "status": ToolStatus.RUNNING.value,
                "tool_name": request.tool_name,
                "request": request.model_dump(mode="json"),
                **details,
            },
        )

    def tool_result(self, result: ToolResult, **details: Any) -> StreamEvent:
        return self._event(
            StreamEventType.TOOL_RESULT,
            {
                "status": result.status.value,
                "tool_name": result.tool_name,
                "result": result.model_dump(mode="json"),
                **details,
            },
        )

    def node_started(
        self,
        *,
        node: str,
        message: str,
        **details: Any,
    ) -> StreamEvent:
        return self._workflow_truth_event(
            StreamEventType.NODE_STARTED,
            code="node_started",
            message=message,
            node=node,
            **details,
        )

    def node_completed(
        self,
        *,
        node: str,
        message: str,
        **details: Any,
    ) -> StreamEvent:
        return self._workflow_truth_event(
            StreamEventType.NODE_COMPLETED,
            code="node_completed",
            message=message,
            node=node,
            **details,
        )

    def node_failed(
        self,
        *,
        node: str,
        message: str,
        **details: Any,
    ) -> StreamEvent:
        return self._workflow_truth_event(
            StreamEventType.NODE_FAILED,
            code="node_failed",
            message=message,
            node=node,
            **details,
        )

    def review_passed(
        self,
        *,
        message: str,
        **details: Any,
    ) -> StreamEvent:
        return self._workflow_truth_event(
            StreamEventType.REVIEW_PASSED,
            code="review_passed",
            message=message,
            **details,
        )

    def review_failed(
        self,
        *,
        message: str,
        **details: Any,
    ) -> StreamEvent:
        return self._workflow_truth_event(
            StreamEventType.REVIEW_FAILED,
            code="review_failed",
            message=message,
            **details,
        )

    def refinement_requested(
        self,
        *,
        message: str,
        **details: Any,
    ) -> StreamEvent:
        return self._workflow_truth_event(
            StreamEventType.REFINEMENT_REQUESTED,
            code="refinement_requested",
            message=message,
            **details,
        )

    def refinement_completed(
        self,
        *,
        message: str,
        **details: Any,
    ) -> StreamEvent:
        return self._workflow_truth_event(
            StreamEventType.REFINEMENT_COMPLETED,
            code="refinement_completed",
            message=message,
            **details,
        )

    def retry_started(
        self,
        *,
        message: str,
        **details: Any,
    ) -> StreamEvent:
        return self._workflow_truth_event(
            StreamEventType.RETRY_STARTED,
            code="retry_started",
            message=message,
            **details,
        )

    def retry_completed(
        self,
        *,
        message: str,
        **details: Any,
    ) -> StreamEvent:
        return self._workflow_truth_event(
            StreamEventType.RETRY_COMPLETED,
            code="retry_completed",
            message=message,
            **details,
        )

    def artifact_extracted(
        self,
        *,
        artifacts: tuple[object, ...],
        code: str,
        message: str,
        **details: Any,
    ) -> StreamEvent:
        return self._event(
            StreamEventType.ARTIFACT_EXTRACTED,
            {
                "code": code,
                "message": message,
                "artifacts": [_dump_event_payload(artifact) for artifact in artifacts],
                "artifact_count": len(artifacts),
                **details,
            },
        )

    def preview_artifact(self, result: PreviewResult, **details: Any) -> StreamEvent:
        return self._event(
            StreamEventType.PREVIEW_ARTIFACT,
            {
                "status": result.status.value,
                "preview_id": result.preview_id,
                "artifact_id": result.artifact_id,
                "result": result.model_dump(mode="json"),
                **details,
            },
        )

    def final(self, *, answer: str, **details: Any) -> StreamEvent:
        return self._event(
            StreamEventType.FINAL,
            {"answer": answer, **details},
        )

    def error(self, *, code: str, message: str, **details: Any) -> StreamEvent:
        return self._event(
            StreamEventType.ERROR,
            {"code": code, "message": message, **details},
        )

    def _event(
        self,
        event_type: StreamEventType,
        payload: dict[str, Any],
    ) -> StreamEvent:
        if "emitted_at" not in payload:
            payload = {
                **payload,
                "emitted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            }
        event = StreamEvent(
            event_type=event_type,
            sequence=self._sequence,
            payload=payload,
        )
        self._sequence += 1
        return event

    def _workflow_truth_event(
        self,
        event_type: StreamEventType,
        *,
        code: str,
        message: str,
        **details: Any,
    ) -> StreamEvent:
        return self._event(
            event_type,
            {
                "code": code,
                "message": message,
                **details,
            },
        )


def _dump_event_payload(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value
