"""Shared metadata helper utilities for orchestration engines."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Protocol


class PlanningMetadataItem(Protocol):
    """Minimal profile shape accepted by evaluation planning metadata helpers."""

    role: str


PlanningMetadata = Sequence[PlanningMetadataItem]

_TOKEN_PATTERN = re.compile(r"[a-z0-9_.+#-]+")


def _clip(value: str, limit: int = 360) -> str:
    normalized = " ".join(value.strip().split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "."


def _dedupe(
    values: Iterable[str],
    *,
    clip_limit: int | None = 360,
) -> tuple[str, ...]:
    deduped: list[str] = []
    for value in values:
        if clip_limit is None:
            cleaned = " ".join(value.split())
        else:
            cleaned = _clip(value, clip_limit)
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return tuple(deduped)


def _contains_any(value: str, needles: Iterable[str]) -> bool:
    lowered = value.lower()
    return any(needle in lowered for needle in needles)


def _token_set(value: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(value.lower()))


def _metadata_values(
    item: object,
    attribute: str,
    *,
    stringify_before_filter: bool = False,
) -> tuple[str, ...]:
    value = getattr(item, attribute, ())
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        if stringify_before_filter:
            return tuple(str(entry) for entry in value if str(entry))
        return tuple(str(entry) for entry in value if entry)
    return ()


def _metadata_label(item: object) -> str:
    role = getattr(item, "role", None)
    if isinstance(role, str) and role:
        return role
    return item.__class__.__name__


def _score(
    base: float,
    *,
    positives: Sequence[object | None],
    bonus: float = 0,
    penalties: float = 0,
) -> float:
    present = sum(item is not None for item in positives)
    return _clamp_score(base + present * 0.055 + bonus - penalties)


def _clamp_score(value: float) -> float:
    return round(max(0.05, min(0.98, value)), 2)
