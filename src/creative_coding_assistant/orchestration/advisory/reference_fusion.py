"""Truthful metadata-only image-reference guidance for creative generation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.title() for part in tail)


class ReferenceImageMetadata(Protocol):
    id: str
    name: str
    mime_type: str
    size_bytes: int


class ReferenceFusionGuidance(BaseModel):
    """Non-identifying, metadata-only guidance from uploaded references."""

    model_config = ConfigDict(
        alias_generator=_to_camel,
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    source_count: int = Field(ge=1, le=4)
    source_names: tuple[str, ...] = Field(min_length=1, max_length=4)
    palette_direction: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    composition: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    lighting_contrast: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    texture_material_cues: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    geometric_structure: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    mood_atmosphere: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    motion_implications: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    runtime_style_implications: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    safety_constraints: tuple[str, ...] = Field(min_length=2, max_length=5)
    summary: str = Field(min_length=1, max_length=360)


_SAFETY_CONSTRAINTS = (
    "Image references are metadata-only in this path; do not infer palette, "
    "composition, people, or visual content from filenames.",
    "Do not identify people, infer identity, or describe facial/person attributes.",
    "Do not claim exact copying or replication of any uploaded reference.",
)


def derive_reference_fusion_guidance(
    image_references: Sequence[ReferenceImageMetadata],
) -> ReferenceFusionGuidance | None:
    """Preserve image provenance without fabricating pixel-level analysis."""

    if not image_references:
        return None

    names = tuple(reference.name for reference in image_references[:4])
    return ReferenceFusionGuidance(
        source_count=len(names),
        source_names=names,
        safety_constraints=_SAFETY_CONSTRAINTS,
        summary=(
            f"{len(names)} image reference(s) are attached as metadata only; "
            "request written visual direction before deriving palette, composition, "
            "or material choices."
        ),
    )


def reference_fusion_prompt_lines(
    guidance: ReferenceFusionGuidance,
) -> tuple[str, ...]:
    """Render compact prompt lines for reference fusion metadata."""

    lines = [
        f"Reference fusion sources: {guidance.source_count}",
        f"Reference fusion summary: {guidance.summary}",
    ]
    _append_list_line(lines, "Reference palette direction", guidance.palette_direction)
    _append_list_line(lines, "Reference composition", guidance.composition)
    _append_list_line(lines, "Reference lighting", guidance.lighting_contrast)
    _append_list_line(
        lines,
        "Reference texture/material",
        guidance.texture_material_cues,
    )
    _append_list_line(lines, "Reference geometry", guidance.geometric_structure)
    _append_list_line(lines, "Reference mood", guidance.mood_atmosphere)
    _append_list_line(lines, "Reference motion", guidance.motion_implications)
    _append_list_line(
        lines,
        "Reference runtime/style implications",
        guidance.runtime_style_implications,
    )
    _append_list_line(lines, "Reference safety", guidance.safety_constraints)
    return tuple(lines)


def _append_list_line(lines: list[str], label: str, values: tuple[str, ...]) -> None:
    if values:
        lines.append(f"{label}: " + " / ".join(values))
