"""Streaming event construction for backend services."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

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

    def token_delta(self, text: str) -> StreamEvent:
        return self._event(StreamEventType.TOKEN_DELTA, {"text": text})

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
