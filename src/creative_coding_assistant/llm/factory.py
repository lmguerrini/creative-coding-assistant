"""Provider selection and construction for generation adapters."""

from __future__ import annotations

from loguru import logger

from creative_coding_assistant.core import (
    GenerationProviderName,
    Settings,
    load_settings,
)
from creative_coding_assistant.llm.generation import GenerationProvider
from creative_coding_assistant.llm.openai_adapter import OpenAIGenerationProvider


class UnsupportedGenerationProviderError(RuntimeError):
    """Raised when generation provider selection is not supported."""


def build_generation_provider(
    settings: Settings | None = None,
) -> GenerationProvider:
    resolved_settings = settings or load_settings()
    provider_name = resolved_settings.default_generation_provider

    if provider_name is GenerationProviderName.OPENAI:
        logger.info("Configured generation provider: {}", provider_name.value)
        return OpenAIGenerationProvider(settings=resolved_settings)

    raise UnsupportedGenerationProviderError(
        f"Unsupported generation provider: {provider_name.value}"
    )
