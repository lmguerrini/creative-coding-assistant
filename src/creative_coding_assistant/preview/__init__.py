"""Typed preview contracts for future artifact renderers."""

from __future__ import annotations

from importlib import import_module

_CONTRACTS = "creative_coding_assistant.preview.contracts"

_EXPORT_MAP = {
    "NoOpPreviewRenderer": _CONTRACTS,
    "PreviewArtifactLink": _CONTRACTS,
    "PreviewError": _CONTRACTS,
    "PreviewIdentity": _CONTRACTS,
    "PreviewProvenance": _CONTRACTS,
    "PreviewRenderer": _CONTRACTS,
    "PreviewRendererCapability": _CONTRACTS,
    "PreviewRequest": _CONTRACTS,
    "PreviewResult": _CONTRACTS,
    "PreviewStatus": _CONTRACTS,
    "PreviewTarget": _CONTRACTS,
    "TERMINAL_PREVIEW_STATUSES": _CONTRACTS,
    "can_transition_preview_status": _CONTRACTS,
    "get_allowed_preview_status_transitions": _CONTRACTS,
    "is_previewable_artifact": _CONTRACTS,
    "resolve_preview_target_for_artifact": _CONTRACTS,
    "validate_preview_status_transition": _CONTRACTS,
}

__all__ = list(_EXPORT_MAP)


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
