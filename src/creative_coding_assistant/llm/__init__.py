"""Provider-agnostic generation contracts and future LLM adapters."""

from creative_coding_assistant.llm.generation import (
    GeneratedOutput,
    GenerationDelta,
    GenerationError,
    GenerationEventType,
    GenerationFinishReason,
    GenerationInput,
    GenerationInputBuilder,
    GenerationMessage,
    GenerationMessageName,
    GenerationMessageRole,
    GenerationProvider,
    GenerationRequest,
    GenerationResponse,
    GenerationStreamEvent,
    RenderedPromptGenerationBuilder,
    build_generation_request,
)

__all__ = [
    "GeneratedOutput",
    "GenerationDelta",
    "GenerationError",
    "GenerationEventType",
    "GenerationFinishReason",
    "GenerationInput",
    "GenerationInputBuilder",
    "GenerationMessage",
    "GenerationMessageName",
    "GenerationMessageRole",
    "GenerationProvider",
    "GenerationRequest",
    "GenerationResponse",
    "GenerationStreamEvent",
    "RenderedPromptGenerationBuilder",
    "build_generation_request",
]
