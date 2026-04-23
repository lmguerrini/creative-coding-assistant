"""Frontend-agnostic assistant service shell."""

from __future__ import annotations

from collections.abc import Callable, Iterator

from loguru import logger

from creative_coding_assistant.contracts import (
    AssistantRequest,
    AssistantResponse,
    StreamEvent,
    StreamEventType,
)
from creative_coding_assistant.core import Settings, load_settings
from creative_coding_assistant.orchestration.events import StreamEventBuilder
from creative_coding_assistant.orchestration.retrieval import (
    RetrievalGateway,
    build_retrieval_context_request,
)
from creative_coding_assistant.orchestration.routing import RouteDecision, route_request

RouteFn = Callable[[AssistantRequest], RouteDecision]


class AssistantService:
    """Service boundary for clients that need streamed assistant events."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        route_fn: RouteFn = route_request,
        retrieval_gateway: RetrievalGateway | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self._route_fn = route_fn
        self._retrieval_gateway = retrieval_gateway

    def stream(self, request: AssistantRequest) -> Iterator[StreamEvent]:
        """Yield the current backend event flow for one assistant request."""

        builder = StreamEventBuilder()
        logger.info("assistant_request_received")
        yield builder.status(code="request_received", message="Request accepted.")

        decision = self._route_fn(request)
        route_payload = decision.model_dump(mode="json")
        logger.bind(route=decision.route.value).info("assistant_route_selected")
        yield builder.status(
            code="route_selected",
            message="Route selected.",
            route=route_payload,
        )

        retrieval_request = None
        retrieval_context = None
        if self._retrieval_gateway is not None:
            retrieval_request = build_retrieval_context_request(request, decision)

        if retrieval_request is not None and self._retrieval_gateway is not None:
            retrieval_payload = retrieval_request.model_dump(mode="json")
            logger.bind(route=decision.route.value).info("assistant_retrieval_requested")
            yield builder.retrieval(
                code="retrieval_requested",
                message="Retrieval context requested.",
                request=retrieval_payload,
            )
            retrieval_context = self._retrieval_gateway.retrieve_context(
                retrieval_request
            )
            yield builder.retrieval(
                code="retrieval_completed",
                message="Retrieval context prepared.",
                context=retrieval_context.model_dump(mode="json"),
            )

        answer = _build_shell_answer(decision)
        yield builder.final(answer=answer, route=route_payload)

    def respond(self, request: AssistantRequest) -> AssistantResponse:
        """Collect streamed events into a final response object."""

        events = tuple(self.stream(request))
        final_event = _last_final_event(events)
        return AssistantResponse(
            answer=str(final_event.payload["answer"]),
            events=events,
        )


def _last_final_event(events: tuple[StreamEvent, ...]) -> StreamEvent:
    for event in reversed(events):
        if event.event_type == StreamEventType.FINAL:
            return event
    raise RuntimeError("Assistant stream completed without a final event.")


def _build_shell_answer(decision: RouteDecision) -> str:
    return (
        "The backend service accepted the request and selected the "
        f"{decision.route.value} route. Downstream handlers are not connected yet."
    )
