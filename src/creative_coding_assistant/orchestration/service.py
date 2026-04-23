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
from creative_coding_assistant.orchestration.context import (
    ContextAssembler,
    build_assembled_context_request,
)
from creative_coding_assistant.orchestration.events import StreamEventBuilder
from creative_coding_assistant.orchestration.memory import (
    MemoryGateway,
    build_memory_context_request,
)
from creative_coding_assistant.orchestration.prompt_inputs import (
    PromptInputBuilder,
    build_prompt_input_request,
)
from creative_coding_assistant.orchestration.prompt_templates import (
    PromptRenderer,
    build_rendered_prompt_request,
)
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
        memory_gateway: MemoryGateway | None = None,
        retrieval_gateway: RetrievalGateway | None = None,
        context_assembler: ContextAssembler | None = None,
        prompt_input_builder: PromptInputBuilder | None = None,
        prompt_renderer: PromptRenderer | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self._route_fn = route_fn
        self._memory_gateway = memory_gateway
        self._retrieval_gateway = retrieval_gateway
        self._context_assembler = context_assembler
        self._prompt_input_builder = prompt_input_builder
        self._prompt_renderer = prompt_renderer

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

        memory_request = None
        memory_context = None
        if self._memory_gateway is not None:
            memory_request = build_memory_context_request(request, decision)

        if memory_request is not None and self._memory_gateway is not None:
            memory_payload = memory_request.model_dump(mode="json")
            logger.bind(route=decision.route.value).info("assistant_memory_requested")
            yield builder.memory(
                code="memory_requested",
                message="Memory context requested.",
                request=memory_payload,
            )
            memory_context = self._memory_gateway.retrieve_context(memory_request)
            yield builder.memory(
                code="memory_completed",
                message="Memory context prepared.",
                context=memory_context.model_dump(mode="json"),
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

        assembled_context = None
        context_request = None
        if self._context_assembler is not None:
            context_request = build_assembled_context_request(
                route_decision=decision,
                memory_context=memory_context,
                retrieval_context=retrieval_context,
            )

        if context_request is not None and self._context_assembler is not None:
            assembled_context = self._context_assembler.assemble(context_request)
            yield builder.context(
                code="context_assembled",
                message="Combined context prepared.",
                context=assembled_context.model_dump(mode="json"),
            )

        prompt_inputs = None
        if self._prompt_input_builder is not None:
            prompt_input_request = build_prompt_input_request(
                assistant_request=request,
                route_decision=decision,
                assembled_context=assembled_context,
            )
            prompt_inputs = self._prompt_input_builder.build(prompt_input_request)
            yield builder.prompt_input(
                code="prompt_inputs_prepared",
                message="Prompt inputs prepared.",
                prompt_input=prompt_inputs.model_dump(mode="json"),
            )

        if prompt_inputs is not None and self._prompt_renderer is not None:
            rendered_prompt_request = build_rendered_prompt_request(
                route_decision=decision,
                prompt_input=prompt_inputs,
            )
            rendered_prompt = self._prompt_renderer.render(rendered_prompt_request)
            yield builder.prompt_rendered(
                code="prompt_rendered",
                message="Rendered prompt prepared.",
                rendered_prompt=rendered_prompt.model_dump(mode="json"),
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
