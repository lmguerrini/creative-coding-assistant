"""Compatibility shim for the decomposed assistant workflow graph."""
from __future__ import annotations

import sys as _sys
from importlib import import_module as _import_module

_CANONICAL_MODULE = (
    "creative_coding_assistant.orchestration.runtime."
    "nodes.handlers"
)

_sys.modules[__name__] = _import_module(_CANONICAL_MODULE)
