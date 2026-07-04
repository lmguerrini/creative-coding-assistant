"""Typed tool contracts and future execution primitives."""

from __future__ import annotations

from importlib import import_module

_CONTRACTS = "creative_coding_assistant.tools.contracts"

_EXPORT_MAP = {
    "AssistantTool": _CONTRACTS,
    "DuplicateToolRegistrationError": _CONTRACTS,
    "TERMINAL_TOOL_STATUSES": _CONTRACTS,
    "ToolError": _CONTRACTS,
    "ToolIdentity": _CONTRACTS,
    "ToolMetadata": _CONTRACTS,
    "ToolNotRegisteredError": _CONTRACTS,
    "ToolRegistry": _CONTRACTS,
    "ToolRequest": _CONTRACTS,
    "ToolResult": _CONTRACTS,
    "ToolStatus": _CONTRACTS,
    "can_transition_tool_status": _CONTRACTS,
    "get_allowed_tool_status_transitions": _CONTRACTS,
    "validate_tool_status_transition": _CONTRACTS,
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
