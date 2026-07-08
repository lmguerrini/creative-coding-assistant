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
    GenerationTokenUsage,
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
            client = self._client or _build_openai_client(
                settings=self._settings,
                api_key=self._api_key,
            )
            payload = _build_openai_payload(
                request=request,
                model=self._model,
                max_output_tokens=self._settings.openai_max_output_tokens,
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
                        delta=GenerationDelta(
                            index=0,
                            content=delta_text,
                            provider="openai",
                            model=self._model,
                        ),
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
                        provider="openai",
                        model=self._model,
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
        if not text.strip():
            text = _fallback_empty_output_text(finish_reason)
        return GenerationStreamEvent(
            event_type=GenerationEventType.COMPLETED,
            response=GenerationResponse(
                request=request,
                output=GeneratedOutput(
                    content=text,
                    finish_reason=finish_reason,
                    provider="openai",
                    model=_extract_model(response) or self._model,
                    response_id=_extract_response_id(response),
                    usage=_extract_token_usage(response),
                ),
            ),
        )


def _build_openai_client(
    *,
    settings: Settings | None = None,
    api_key: str | None = None,
) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("The OpenAI SDK is not installed.") from exc

    resolved_settings = settings or load_settings()
    resolved_api_key = api_key or resolved_settings.get_openai_api_key()
    if not resolved_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=resolved_api_key)


def _build_openai_payload(
    *,
    request: GenerationInput,
    model: str,
    max_output_tokens: int | None = None,
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
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens
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


def _fallback_empty_output_text(finish_reason: GenerationFinishReason) -> str:
    if finish_reason is GenerationFinishReason.LENGTH:
        return (
            "The provider stopped before returning visible text because the "
            "configured output limit was reached. Narrow the prompt, increase "
            "CCA_OPENAI_MAX_OUTPUT_TOKENS, or use the documented demo artifact "
            "fallback for this scenario."
        )
    if finish_reason is GenerationFinishReason.ERROR:
        return (
            "The provider completed without visible text after reporting an error. "
            "Review provider status and retry the request."
        )
    return (
        "The provider completed without visible text. Retry the request or use the "
        "documented demo fallback for this scenario."
    )


def _extract_model(response: Any) -> str | None:
    model = _read_field(response, "model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return None


def _extract_response_id(response: Any) -> str | None:
    response_id = _read_field(response, "id")
    if isinstance(response_id, str) and response_id.strip():
        return response_id.strip()
    return None


def _extract_token_usage(response: Any) -> GenerationTokenUsage | None:
    usage = _read_field(response, "usage")
    if usage is None:
        return None

    input_tokens = _read_token_count(
        usage,
        "input_tokens",
        "prompt_tokens",
    )
    output_tokens = _read_token_count(
        usage,
        "output_tokens",
        "completion_tokens",
    )
    total_tokens = _read_token_count(usage, "total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    input_details = _read_field(usage, "input_tokens_details", {}) or {}
    output_details = _read_field(usage, "output_tokens_details", {}) or {}
    cached_input_tokens = _read_token_count(input_details, "cached_tokens")
    reasoning_tokens = _read_token_count(output_details, "reasoning_tokens")

    if all(
        value is None
        for value in (
            input_tokens,
            output_tokens,
            total_tokens,
            cached_input_tokens,
            reasoning_tokens,
        )
    ):
        return None

    return GenerationTokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _read_token_count(value: Any, *field_names: str) -> int | None:
    for field_name in field_names:
        raw_value = _read_field(value, field_name)
        if isinstance(raw_value, bool):
            continue
        if isinstance(raw_value, int) and raw_value >= 0:
            return raw_value
        if isinstance(raw_value, float) and raw_value >= 0 and raw_value.is_integer():
            return int(raw_value)
    return None


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
