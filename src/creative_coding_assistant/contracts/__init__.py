"""Stable request, response, and streaming event contracts."""

from creative_coding_assistant.contracts.events import StreamEvent, StreamEventType
from creative_coding_assistant.contracts.requests import (
    MAX_IMAGE_REFERENCE_BYTES,
    MAX_IMAGE_REFERENCE_COUNT,
    SUPPORTED_IMAGE_REFERENCE_MIME_TYPES,
    AssistantImageReference,
    AssistantMode,
    AssistantRequest,
    AssistantResponse,
    CreativeCodingDomain,
)

__all__ = [
    "AssistantImageReference",
    "AssistantMode",
    "AssistantRequest",
    "AssistantResponse",
    "CreativeCodingDomain",
    "MAX_IMAGE_REFERENCE_BYTES",
    "MAX_IMAGE_REFERENCE_COUNT",
    "SUPPORTED_IMAGE_REFERENCE_MIME_TYPES",
    "StreamEvent",
    "StreamEventType",
]
