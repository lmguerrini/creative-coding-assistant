"""Typed artifact contracts for persistent creative outputs."""

from __future__ import annotations

from importlib import import_module

_CONTRACTS = "creative_coding_assistant.artifacts.contracts"

_EXPORT_MAP = {
    "ArtifactCategory": _CONTRACTS,
    "ArtifactContentLocator": _CONTRACTS,
    "ArtifactContentReference": _CONTRACTS,
    "ArtifactIdentity": _CONTRACTS,
    "ArtifactLineage": _CONTRACTS,
    "ArtifactMetadata": _CONTRACTS,
    "ArtifactOrigin": _CONTRACTS,
    "ArtifactProvenance": _CONTRACTS,
    "ArtifactRecord": _CONTRACTS,
    "ArtifactStatus": _CONTRACTS,
    "ArtifactTimestamps": _CONTRACTS,
    "ArtifactType": _CONTRACTS,
    "ArtifactWorkflowLink": _CONTRACTS,
    "ArtifactWorkspace": _CONTRACTS,
    "attach_artifact_to_workspace": _CONTRACTS,
    "can_transition_artifact_status": _CONTRACTS,
    "get_allowed_artifact_status_transitions": _CONTRACTS,
    "transition_artifact_status": _CONTRACTS,
    "validate_artifact_status_transition": _CONTRACTS,
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
