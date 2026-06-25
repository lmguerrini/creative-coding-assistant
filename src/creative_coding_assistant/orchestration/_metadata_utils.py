"""Shared metadata helper utilities for orchestration engines."""

from __future__ import annotations

from collections.abc import Iterable


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
