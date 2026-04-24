"""OpenAI-backed generation provider adapter."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from loguru import logger

from creative_coding_assistant.core import Settings, load_settings
from creative_coding_assistant.llm.generation import (
    GeneratedOutput,
    GenerationDelta,
    GenerationError,
    GenerationEventType,
    GenerationFinishReason,
    GenerationInput,
    GenerationMessage,
    GenerationMessageName,
    GenerationMessageRole,
    GenerationProvider,
    GenerationResponse,
    GenerationStreamEvent,
)


class OpenAIGenerationProvider(GenerationProvider):
    """Translate provider-neutral generation input into OpenAI Responses calls."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        model: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._settings = settings or load_settings()
        self._model = model or self._settings.openai_model
        self._api_key = api_key
        self._client = client

    def stream(
        self,
        request: GenerationInput,
    ) -> Iterable[GenerationStreamEvent]:
        try:
            client = self._client or _build_openai_client(api_key=self._api_key)
            payload = _build_openai_payload(
                request=request,
                model=self._model,
            )
            logger.info(
                "Dispatching generation request to OpenAI with {} message(s)",
                len(request.messages),
            )
            if request.request.stream:
                response_stream = client.responses.create(**payload, stream=True)
                return self._stream_response_events(
                    request=request,
                    response_stream=response_stream,
                )
            response = client.responses.create(**payload, stream=False)
            return iter((self._completed_event(request=request, response=response),))
        except Exception as exc:  # pragma: no cover - validated via tests
            logger.exception("OpenAI generation request failed")
            error = _map_provider_error(exc)
            return iter(
                (
                    GenerationStreamEvent(
                        event_type=GenerationEventType.ERROR,
                        error=error,
                    ),
                )
            )

    def _stream_response_events(
        self,
        *,
        request: GenerationInput,
        response_stream: Iterable[Any],
    ) -> Iterator[GenerationStreamEvent]:
        accumulated_text = ""
        completed_emitted = False

        for event in response_stream:
            event_type = _read_field(event, "type", "")

            if event_type == "response.output_text.delta":
                delta_text = str(_read_field(event, "delta", "") or "")
                if delta_text:
                    accumulated_text += delta_text
                    yield GenerationStreamEvent(
                        event_type=GenerationEventType.DELTA,
                        delta=GenerationDelta(index=0, content=delta_text),
                    )
                continue

            if event_type in {"response.completed", "response.incomplete"}:
                response = _read_field(event, "response")
                yield self._completed_event(
                    request=request,
                    response=response,
                    fallback_text=accumulated_text,
                )
                completed_emitted = True
                continue

            if event_type in {"response.failed", "error"}:
                yield GenerationStreamEvent(
                    event_type=GenerationEventType.ERROR,
                    error=_map_provider_error(event),
                )
                return

        if not completed_emitted and accumulated_text:
            yield GenerationStreamEvent(
                event_type=GenerationEventType.COMPLETED,
                response=GenerationResponse(
                    request=request,
                    output=GeneratedOutput(
                        content=accumulated_text,
                        finish_reason=GenerationFinishReason.STOP,
                    ),
                ),
            )

    def _completed_event(
        self,
        *,
        request: GenerationInput,
        response: Any,
        fallback_text: str = "",
    ) -> GenerationStreamEvent:
        text = _extract_output_text(response) or fallback_text
        finish_reason = _extract_finish_reason(response)
        return GenerationStreamEvent(
            event_type=GenerationEventType.COMPLETED,
            response=GenerationResponse(
                request=request,
                output=GeneratedOutput(
                    content=text,
                    finish_reason=finish_reason,
                ),
            ),
        )


def _build_openai_client(*, api_key: str | None) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("The OpenAI SDK is not installed.") from exc

    settings = load_settings()
    resolved_api_key = api_key or settings.get_openai_api_key()
    if not resolved_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=resolved_api_key)


def _build_openai_payload(
    *,
    request: GenerationInput,
    model: str,
) -> dict[str, Any]:
    instructions: list[str] = []
    input_messages: list[dict[str, Any]] = []

    for message in request.messages:
        if message.role is GenerationMessageRole.SYSTEM:
            instructions.append(message.content)
            continue
        input_messages.append(_build_openai_message(message))

    payload: dict[str, Any] = {
        "model": model,
        "input": input_messages,
    }
    if instructions:
        payload["instructions"] = "\n\n".join(instructions)
    return payload


def _build_openai_message(message: GenerationMessage) -> dict[str, Any]:
    role = (
        "developer"
        if message.role is GenerationMessageRole.CONTEXT
        else message.role.value
    )
    text = _format_message_text(message)
    return {
        "type": "message",
        "role": role,
        "content": [
            {
                "type": "input_text",
                "text": text,
            }
        ],
    }


def _format_message_text(message: GenerationMessage) -> str:
    if message.name is GenerationMessageName.MEMORY:
        return f"Memory Context:\n{message.content}"
    if message.name is GenerationMessageName.RETRIEVAL:
        return f"Retrieval Context:\n{message.content}"
    return message.content


def _extract_output_text(response: Any) -> str:
    if response is None:
        return ""

    output_text = _read_field(response, "output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = _read_field(response, "output", ())
    texts: list[str] = []
    for item in output or ():
        content = _read_field(item, "content", ())
        for part in content or ():
            part_type = _read_field(part, "type", "")
            if part_type in {"output_text", "text"}:
                text = str(_read_field(part, "text", "") or "").strip()
                if text:
                    texts.append(text)
    return "\n".join(texts).strip()


def _extract_finish_reason(response: Any) -> GenerationFinishReason:
    status = str(_read_field(response, "status", "") or "").lower()
    if status == "cancelled":
        return GenerationFinishReason.CANCELLED
    if status in {"failed", "error"}:
        return GenerationFinishReason.ERROR

    incomplete_details = _read_field(response, "incomplete_details", {}) or {}
    reason = str(_read_field(incomplete_details, "reason", "") or "").lower()
    if reason in {"max_output_tokens", "length"}:
        return GenerationFinishReason.LENGTH
    if reason == "cancelled":
        return GenerationFinishReason.CANCELLED
    if reason in {"failed", "error"}:
        return GenerationFinishReason.ERROR
    return GenerationFinishReason.STOP


def _map_provider_error(error_like: Any) -> GenerationError:
    error_obj = _read_field(error_like, "error", error_like)
    code = str(_read_field(error_obj, "code", "") or "").strip() or "openai_error"
    raw_message = (
        _read_field(error_obj, "message", "")
        or str(error_like)
        or "OpenAI request failed."
    )
    message = str(raw_message).strip()
    return GenerationError(code=code, message=message)


def _read_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)
