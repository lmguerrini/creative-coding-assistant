"""Backward-compatible orchestration module shim."""
from __future__ import annotations

import sys as _sys
from importlib import import_module as _import_module

_CANONICAL_MODULE = (
    "creative_coding_assistant.orchestration."
    "metadata.agent_parallel_scheduling"
)

_sys.modules[__name__] = _import_module(_CANONICAL_MODULE)
