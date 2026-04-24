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
from creative_coding_assistant.llm import GenerationEventType, GenerationProvider
from creative_coding_assistant.orchestration.context import (
    ContextAssembler,
    build_assembled_context_request,
)
from creative_coding_assistant.orchestration.events import StreamEventBuilder
from creative_coding_assistant.orchestration.generation import (
    ProviderGenerationGateway,
    build_provider_generation_request,
)
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
        generation_gateway: ProviderGenerationGateway | None = None,
        generation_provider: GenerationProvider | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self._route_fn = route_fn
        self._memory_gateway = memory_gateway
        self._retrieval_gateway = retrieval_gateway
        self._context_assembler = context_assembler
        self._prompt_input_builder = prompt_input_builder
        self._prompt_renderer = prompt_renderer
        self._generation_gateway = generation_gateway
        self._generation_provider = generation_provider

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

            if self._generation_gateway is not None:
                generation_request = build_provider_generation_request(
                    route_decision=decision,
                    rendered_prompt=rendered_prompt,
                )
                generation_input = self._generation_gateway.prepare_generation(
                    generation_request
                )
                yield builder.generation_input(
                    code="generation_input_prepared",
                    message="Provider generation input prepared.",
                    generation_input=generation_input.model_dump(mode="json"),
                )

                generation_result = _stream_provider_generation(
                    builder=builder,
                    generation_provider=self._generation_provider,
                    generation_input=generation_input,
                )
                if generation_result is not None:
                    yield from generation_result.events
                    yield builder.final(
                        answer=generation_result.answer,
                        route=route_payload,
                    )
                    return

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


class _GenerationResult:
    def __init__(self, *, answer: str, events: tuple[StreamEvent, ...]) -> None:
        self.answer = answer
        self.events = events


def _stream_provider_generation(
    *,
    builder: StreamEventBuilder,
    generation_provider: GenerationProvider | None,
    generation_input: object,
) -> _GenerationResult | None:
    if generation_provider is None:
        return None

    delta_text: list[str] = []
    streamed_events: list[StreamEvent] = []
    completed_answer: str | None = None
    generation_error: tuple[str, str] | None = None

    for generation_event in generation_provider.stream(generation_input):
        if generation_event.event_type is GenerationEventType.DELTA:
            assert generation_event.delta is not None
            delta_text.append(generation_event.delta.content)
            streamed_events.append(
                builder.token_delta(generation_event.delta.content)
            )
            continue

        if generation_event.event_type is GenerationEventType.COMPLETED:
            assert generation_event.response is not None
            completed_answer = generation_event.response.output.content
            continue

        if generation_event.event_type is GenerationEventType.ERROR:
            assert generation_event.error is not None
            generation_error = (
                generation_event.error.code,
                generation_event.error.message,
            )
            streamed_events.append(
                builder.error(
                    code=generation_event.error.code,
                    message=generation_event.error.message,
                )
            )
            break

    if completed_answer is not None:
        return _GenerationResult(
            answer=completed_answer,
            events=tuple(streamed_events),
        )

    if delta_text:
        return _GenerationResult(
            answer="".join(delta_text),
            events=tuple(streamed_events),
        )

    if generation_error is not None:
        code, message = generation_error
        return _GenerationResult(
            answer=f"Generation failed ({code}): {message}",
            events=tuple(streamed_events),
        )

    return None
