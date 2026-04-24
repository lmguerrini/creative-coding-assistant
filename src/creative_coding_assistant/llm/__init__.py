"""Provider-agnostic generation contracts and future LLM adapters."""

from __future__ import annotations

from importlib import import_module

_EXPORT_MAP = {
    "GeneratedOutput": "creative_coding_assistant.llm.generation",
    "GenerationDelta": "creative_coding_assistant.llm.generation",
    "GenerationError": "creative_coding_assistant.llm.generation",
    "GenerationEventType": "creative_coding_assistant.llm.generation",
    "GenerationFinishReason": "creative_coding_assistant.llm.generation",
    "GenerationInput": "creative_coding_assistant.llm.generation",
    "GenerationInputBuilder": "creative_coding_assistant.llm.generation",
    "GenerationMessage": "creative_coding_assistant.llm.generation",
    "GenerationMessageName": "creative_coding_assistant.llm.generation",
    "GenerationMessageRole": "creative_coding_assistant.llm.generation",
    "GenerationProvider": "creative_coding_assistant.llm.generation",
    "GenerationRequest": "creative_coding_assistant.llm.generation",
    "GenerationResponse": "creative_coding_assistant.llm.generation",
    "GenerationStreamEvent": "creative_coding_assistant.llm.generation",
    "OpenAIGenerationProvider": "creative_coding_assistant.llm.openai_adapter",
    "RenderedPromptGenerationBuilder": "creative_coding_assistant.llm.generation",
    "UnsupportedGenerationProviderError": "creative_coding_assistant.llm.factory",
    "build_generation_provider": "creative_coding_assistant.llm.factory",
    "build_generation_request": "creative_coding_assistant.llm.generation",
}

__all__ = list(_EXPORT_MAP)


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
