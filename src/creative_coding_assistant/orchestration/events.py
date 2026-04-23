"""Streaming event construction for backend services."""

from __future__ import annotations

from typing import Any

from creative_coding_assistant.contracts import StreamEvent, StreamEventType


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

    def final(self, *, answer: str, **details: Any) -> StreamEvent:
        return self._event(
            StreamEventType.FINAL,
            {"answer": answer, **details},
        )

    def error(self, *, code: str, message: str) -> StreamEvent:
        return self._event(
            StreamEventType.ERROR,
            {"code": code, "message": message},
        )

    def _event(
        self,
        event_type: StreamEventType,
        payload: dict[str, Any],
    ) -> StreamEvent:
        event = StreamEvent(
            event_type=event_type,
            sequence=self._sequence,
            payload=payload,
        )
        self._sequence += 1
        return event
