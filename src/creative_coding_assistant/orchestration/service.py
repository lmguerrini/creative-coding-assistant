"""Frontend-agnostic assistant service shell."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime

from loguru import logger

from creative_coding_assistant.contracts import (
    AssistantRequest,
    AssistantResponse,
    StreamEvent,
    StreamEventType,
)
from creative_coding_assistant.core import Settings, load_settings
from creative_coding_assistant.eval import LiveSessionRecorder
from creative_coding_assistant.llm.factory import build_generation_provider
from creative_coding_assistant.llm.generation import (
    GenerationEventType,
    GenerationProvider,
)
from creative_coding_assistant.orchestration.context import (
    AssembledContextResponse,
    ContextAssembler,
    build_assembled_context_request,
)
from creative_coding_assistant.orchestration.events import StreamEventBuilder
from creative_coding_assistant.orchestration.generation import (
    ProviderGenerationGateway,
    build_provider_generation_request,
)
from creative_coding_assistant.orchestration.memory import (
    MemoryContextResponse,
    MemoryGateway,
    build_memory_context_request,
)
from creative_coding_assistant.orchestration.memory_recording import (
    ConversationMemoryRecorder,
)
from creative_coding_assistant.orchestration.prompt_inputs import (
    PromptInputBuilder,
    PromptInputResponse,
    build_prompt_input_request,
)
from creative_coding_assistant.orchestration.prompt_templates import (
    PromptRenderer,
    RenderedPromptResponse,
    build_rendered_prompt_request,
)
from creative_coding_assistant.orchestration.retrieval import (
    RetrievalContextResponse,
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
        eval_recorder: LiveSessionRecorder | None = None,
        memory_recorder: ConversationMemoryRecorder | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self._route_fn = route_fn
        self._memory_gateway = memory_gateway
        self._retrieval_gateway = retrieval_gateway
        self._context_assembler = context_assembler
        self._prompt_input_builder = prompt_input_builder
        self._prompt_renderer = prompt_renderer
        self._generation_gateway = generation_gateway
        self._generation_provider = _resolve_generation_provider(
            settings=self.settings,
            generation_gateway=generation_gateway,
            generation_provider=generation_provider,
        )
        self._eval_recorder = eval_recorder
        self._memory_recorder = memory_recorder

    def stream(self, request: AssistantRequest) -> Iterator[StreamEvent]:
        """Yield the current backend event flow for one assistant request."""

        started_at = _utcnow()
        streamed_events: list[StreamEvent] = []

        try:
            for event in self._stream_events(request):
                streamed_events.append(event)
                yield event
        except Exception as exc:
            logger.bind(
                mode=request.mode.value,
                domain=request.domain.value if request.domain is not None else None,
                domains=[domain.value for domain in request.domains],
                conversation_id=request.conversation_id,
                project_id=request.project_id,
                error_type=type(exc).__name__,
            ).exception(
                "assistant_request_failed_unexpectedly: {}: {}",
                type(exc).__name__,
                exc,
            )
            raise
        finally:
            completed_at = _utcnow()
            _record_stream_completion(
                eval_recorder=self._eval_recorder,
                memory_recorder=self._memory_recorder,
                request=request,
                events=tuple(streamed_events),
                started_at=started_at,
                completed_at=completed_at,
        )

    def _stream_events(self, request: AssistantRequest) -> Iterator[StreamEvent]:
        builder = StreamEventBuilder()
        yield from _stream_request_received(builder)

        decision = self._route_fn(request)
        route_payload = decision.model_dump(mode="json")
        yield from _stream_route_selected(
            builder=builder,
            decision=decision,
            route_payload=route_payload,
        )

        memory_context = yield from self._stream_memory_context(
            builder=builder,
            request=request,
            decision=decision,
        )

        retrieval_context = yield from self._stream_retrieval_context(
            builder=builder,
            request=request,
            decision=decision,
        )

        assembled_context = yield from self._stream_assembled_context(
            builder=builder,
            decision=decision,
            memory_context=memory_context,
            retrieval_context=retrieval_context,
        )

        prompt_inputs = yield from self._stream_prompt_inputs(
            builder=builder,
            request=request,
            decision=decision,
            assembled_context=assembled_context,
        )

        rendered_prompt = yield from self._stream_rendered_prompt(
            builder=builder,
            decision=decision,
            prompt_inputs=prompt_inputs,
        )
        generation_result = yield from self._stream_generation(
            builder=builder,
            decision=decision,
            rendered_prompt=rendered_prompt,
        )
        if generation_result is not None:
            yield builder.final(
                answer=generation_result.answer,
                route=route_payload,
            )
            return

        answer = _build_shell_answer(decision)
        yield builder.final(answer=answer, route=route_payload)

    def _stream_memory_context(
        self,
        *,
        builder: StreamEventBuilder,
        request: AssistantRequest,
        decision: RouteDecision,
    ) -> Iterator[StreamEvent | MemoryContextResponse | None]:
        if self._memory_gateway is None:
            return None

        memory_request = build_memory_context_request(request, decision)
        if memory_request is None:
            return None

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
        return memory_context

    def _stream_retrieval_context(
        self,
        *,
        builder: StreamEventBuilder,
        request: AssistantRequest,
        decision: RouteDecision,
    ) -> Iterator[StreamEvent | RetrievalContextResponse | None]:
        if self._retrieval_gateway is None:
            return None

        retrieval_request = build_retrieval_context_request(request, decision)
        if retrieval_request is None:
            return None

        retrieval_payload = retrieval_request.model_dump(mode="json")
        logger.bind(route=decision.route.value).info("assistant_retrieval_requested")
        yield builder.retrieval(
            code="retrieval_requested",
            message="Retrieval context requested.",
            request=retrieval_payload,
        )
        try:
            retrieval_context = self._retrieval_gateway.retrieve_context(
                retrieval_request
            )
        except Exception as exc:
            logger.bind(
                route=decision.route.value,
                error_type=type(exc).__name__,
            ).exception(
                "assistant_retrieval_failed_using_empty_context: {}: {}",
                type(exc).__name__,
                exc,
            )
            retrieval_context = RetrievalContextResponse(
                request=retrieval_request,
                chunks=(),
            )
        yield builder.retrieval(
            code="retrieval_completed",
            message="Retrieval context prepared.",
            context=retrieval_context.model_dump(mode="json"),
        )
        return retrieval_context

    def _stream_assembled_context(
        self,
        *,
        builder: StreamEventBuilder,
        decision: RouteDecision,
        memory_context: MemoryContextResponse | None,
        retrieval_context: RetrievalContextResponse | None,
    ) -> Iterator[StreamEvent | AssembledContextResponse | None]:
        if self._context_assembler is None:
            return None

        context_request = build_assembled_context_request(
            route_decision=decision,
            memory_context=memory_context,
            retrieval_context=retrieval_context,
        )
        assembled_context = self._context_assembler.assemble(context_request)
        yield builder.context(
            code="context_assembled",
            message="Combined context prepared.",
            context=assembled_context.model_dump(mode="json"),
        )
        return assembled_context

    def _stream_prompt_inputs(
        self,
        *,
        builder: StreamEventBuilder,
        request: AssistantRequest,
        decision: RouteDecision,
        assembled_context: AssembledContextResponse | None,
    ) -> Iterator[StreamEvent | PromptInputResponse | None]:
        if self._prompt_input_builder is None:
            return None

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
        return prompt_inputs

    def _stream_rendered_prompt(
        self,
        *,
        builder: StreamEventBuilder,
        decision: RouteDecision,
        prompt_inputs: PromptInputResponse | None,
    ) -> Iterator[StreamEvent | RenderedPromptResponse | None]:
        if prompt_inputs is None or self._prompt_renderer is None:
            return None

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
        return rendered_prompt

    def _stream_generation(
        self,
        *,
        builder: StreamEventBuilder,
        decision: RouteDecision,
        rendered_prompt: RenderedPromptResponse | None,
    ) -> Iterator[StreamEvent | _GenerationResult | None]:
        if rendered_prompt is None or self._generation_gateway is None:
            return None

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
        if generation_result is None:
            return None
        yield from generation_result.events
        return generation_result

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


def _stream_request_received(builder: StreamEventBuilder) -> Iterator[StreamEvent]:
    logger.info("assistant_request_received")
    yield builder.status(code="request_received", message="Request accepted.")


def _stream_route_selected(
    *,
    builder: StreamEventBuilder,
    decision: RouteDecision,
    route_payload: dict[str, object],
) -> Iterator[StreamEvent]:
    logger.bind(route=decision.route.value).info("assistant_route_selected")
    yield builder.status(
        code="route_selected",
        message="Route selected.",
        route=route_payload,
    )


def _record_stream_completion(
    *,
    eval_recorder: LiveSessionRecorder | None,
    memory_recorder: ConversationMemoryRecorder | None,
    request: AssistantRequest,
    events: tuple[StreamEvent, ...],
    started_at: datetime,
    completed_at: datetime,
) -> None:
    _record_live_session(
        recorder=eval_recorder,
        request=request,
        events=events,
        started_at=started_at,
        completed_at=completed_at,
    )
    _record_conversation_memory(
        recorder=memory_recorder,
        request=request,
        events=events,
        started_at=started_at,
        completed_at=completed_at,
    )


def _record_live_session(
    *,
    recorder: LiveSessionRecorder | None,
    request: AssistantRequest,
    events: tuple[StreamEvent, ...],
    started_at: datetime,
    completed_at: datetime,
) -> None:
    if recorder is None or not events:
        return

    try:
        recorder.record(
            request=request,
            events=events,
            started_at=started_at,
            completed_at=completed_at,
        )
    except Exception:
        logger.exception("live_session_eval_recording_failed")


def _record_conversation_memory(
    *,
    recorder: ConversationMemoryRecorder | None,
    request: AssistantRequest,
    events: tuple[StreamEvent, ...],
    started_at: datetime,
    completed_at: datetime,
) -> None:
    if recorder is None or not events or request.conversation_id is None:
        return

    if any(event.event_type is StreamEventType.ERROR for event in events):
        return

    try:
        final_event = _last_final_event(events)
        answer = str(final_event.payload.get("answer", "")).strip()
        if not answer:
            return
        recorder.record_turns(
            request=request,
            answer=answer,
            started_at=started_at,
            completed_at=completed_at,
        )
    except Exception:
        logger.exception("conversation_memory_recording_failed")


def _utcnow() -> datetime:
    return datetime.now(UTC)


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


def _resolve_generation_provider(
    *,
    settings: Settings,
    generation_gateway: ProviderGenerationGateway | None,
    generation_provider: GenerationProvider | None,
) -> GenerationProvider | None:
    if generation_provider is not None:
        return generation_provider

    if generation_gateway is None:
        return None

    return build_generation_provider(settings)
