from collections.abc import Iterable

from creative_coding_assistant.contracts import StreamEvent, StreamEventType


WORKFLOW_TRUTH_EVENT_TYPES = {
    StreamEventType.NODE_STARTED,
    StreamEventType.NODE_COMPLETED,
    StreamEventType.NODE_FAILED,
    StreamEventType.REVIEW_PASSED,
    StreamEventType.REVIEW_FAILED,
    StreamEventType.REFINEMENT_REQUESTED,
    StreamEventType.REFINEMENT_COMPLETED,
    StreamEventType.RETRY_STARTED,
    StreamEventType.RETRY_COMPLETED,
}


def legacy_events(events: Iterable[StreamEvent]) -> list[StreamEvent]:
    return [
        event
        for event in events
        if event.event_type not in WORKFLOW_TRUTH_EVENT_TYPES
    ]


def event_types(events: Iterable[StreamEvent]) -> list[StreamEventType]:
    return [event.event_type for event in events]


def first_event(
    events: Iterable[StreamEvent],
    event_type: StreamEventType,
    code: str | None = None,
) -> StreamEvent:
    for event in events:
        if event.event_type is not event_type:
            continue
        if code is not None and event.payload.get("code") != code:
            continue
        return event
    raise AssertionError(f"Missing event {event_type} with code {code!r}.")
