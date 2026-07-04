"""V5.1 prompt compression contracts and deterministic section compression."""

from __future__ import annotations

import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.prompt_templates import (
    RenderedPromptResponse,
)

PromptCompressionRole = Literal["system", "user", "context"]
PromptCompressionSectionName = Literal[
    "system",
    "user",
    "memory",
    "retrieval",
    "custom",
]
PromptCompressionStatus = Literal["unchanged", "compressed"]
PromptCompressionPressure = Literal["low", "medium", "high"]

PROMPT_COMPRESSION_SECTION_SERIALIZATION_VERSION = "prompt_compression_section.v1"
PROMPT_COMPRESSION_RESULT_SERIALIZATION_VERSION = "prompt_compression_result.v1"
PROMPT_COMPRESSION_AUTHORITY_BOUNDARY = (
    "Prompt compression produces a separate compressed prompt artifact from "
    "explicit prompt text or rendered prompt sections only; it preserves the "
    "source prompt response, does not route context, compress retrieval, "
    "summarize memory, select providers or models, write storage, execute "
    "providers, or modify generated output."
)

_BLOCKED_RUNTIME_BEHAVIORS = (
    "source_prompt_mutation",
    "context_routing",
    "retrieval_compression",
    "memory_summarization",
    "provider_or_model_routing",
    "provider_execution",
    "persistent_storage_write",
    "generated_output_modification",
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


class PromptCompressionInputSection(BaseModel):
    """Explicit prompt section input for compression."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    section_id: str = Field(min_length=1, max_length=180)
    role: PromptCompressionRole
    name: PromptCompressionSectionName
    content: str = Field(min_length=1, max_length=120_000)


class PromptCompressionSection(BaseModel):
    """One compressed prompt section with the original section preserved."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    section_id: str = Field(min_length=1, max_length=180)
    role: PromptCompressionRole
    name: PromptCompressionSectionName
    original_text: str = Field(min_length=1, max_length=120_000)
    compressed_text: str = Field(min_length=1, max_length=120_000)
    original_token_estimate: int = Field(ge=1, le=240_000)
    compressed_token_estimate: int = Field(ge=1, le=240_000)
    saved_tokens: int = Field(ge=0, le=240_000)
    compression_status: PromptCompressionStatus
    compression_pressure: PromptCompressionPressure
    evidence: tuple[str, ...] = Field(min_length=1, max_length=8)
    advisory_actions: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    prompt_compression_implemented: Literal[True] = True
    source_prompt_mutation_implemented: Literal[False] = False
    context_routing_implemented: Literal[False] = False
    retrieval_compression_implemented: Literal[False] = False
    memory_summarization_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    serialization_version: Literal["prompt_compression_section.v1"] = (
        PROMPT_COMPRESSION_SECTION_SERIALIZATION_VERSION
    )
    compression_only: Literal[True] = True

    @model_validator(mode="after")
    def _section_matches_compression(self) -> Self:
        if self.compressed_token_estimate > self.original_token_estimate:
            raise ValueError(
                "compressed_token_estimate must not exceed original_token_estimate"
            )
        if self.saved_tokens != (
            self.original_token_estimate - self.compressed_token_estimate
        ):
            raise ValueError("saved_tokens must match token estimate delta")
        expected_status = "compressed" if self.saved_tokens > 0 else "unchanged"
        if self.compression_status != expected_status:
            raise ValueError("compression_status must match saved tokens")
        if self.compression_status == "unchanged" and (
            self.compressed_text != self.original_text
        ):
            raise ValueError("unchanged sections must preserve original text")
        return self


class PromptCompressionResult(BaseModel):
    """Bounded V5.1 prompt compression result for explicit prompt text."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: Literal["prompt_compressor"] = "prompt_compressor"
    serialization_version: Literal["prompt_compression_result.v1"] = (
        PROMPT_COMPRESSION_RESULT_SERIALIZATION_VERSION
    )
    authority_boundary: str = Field(
        default=PROMPT_COMPRESSION_AUTHORITY_BOUNDARY,
        max_length=1200,
    )
    sections: tuple[PromptCompressionSection, ...] = Field(
        min_length=1,
        max_length=16,
    )
    section_ids: tuple[str, ...] = Field(min_length=1, max_length=16)
    target_token_budget: int = Field(ge=1, le=240_000)
    original_total_tokens: int = Field(ge=1, le=240_000)
    compressed_total_tokens: int = Field(ge=1, le=240_000)
    saved_total_tokens: int = Field(ge=0, le=240_000)
    within_budget: bool
    compression_pressure: PromptCompressionPressure
    compressed_prompt_text: str = Field(min_length=1, max_length=240_000)
    advisory_actions: tuple[str, ...] = Field(min_length=1, max_length=12)
    blocked_runtime_behaviors: tuple[str, ...] = Field(
        default=_BLOCKED_RUNTIME_BEHAVIORS,
        min_length=1,
        max_length=12,
    )
    prompt_compression_implemented: Literal[True] = True
    source_prompt_mutation_implemented: Literal[False] = False
    context_routing_implemented: Literal[False] = False
    retrieval_compression_implemented: Literal[False] = False
    memory_summarization_implemented: Literal[False] = False
    provider_model_routing_implemented: Literal[False] = False
    provider_execution_implemented: Literal[False] = False
    persistent_storage_write_implemented: Literal[False] = False
    generated_output_mutation_implemented: Literal[False] = False
    compression_only: Literal[True] = True

    @model_validator(mode="after")
    def _result_matches_sections(self) -> Self:
        derived_section_ids = tuple(section.section_id for section in self.sections)
        if len(set(derived_section_ids)) != len(derived_section_ids):
            raise ValueError("section_ids must be unique")
        if self.section_ids != derived_section_ids:
            raise ValueError("section_ids must match sections")
        original_total = sum(
            section.original_token_estimate for section in self.sections
        )
        compressed_total = sum(
            section.compressed_token_estimate for section in self.sections
        )
        saved_total = sum(section.saved_tokens for section in self.sections)
        if self.original_total_tokens != original_total:
            raise ValueError("original_total_tokens must match sections")
        if self.compressed_total_tokens != compressed_total:
            raise ValueError("compressed_total_tokens must match sections")
        if self.saved_total_tokens != saved_total:
            raise ValueError("saved_total_tokens must match sections")
        if self.saved_total_tokens != (
            self.original_total_tokens - self.compressed_total_tokens
        ):
            raise ValueError("saved_total_tokens must match token delta")
        if self.within_budget != (
            self.compressed_total_tokens <= self.target_token_budget
        ):
            raise ValueError("within_budget must match compressed token total")
        if self.compressed_prompt_text != _join_compressed_text(self.sections):
            raise ValueError("compressed_prompt_text must match sections")
        return self


def compress_prompt_text(
    prompt_text: str,
    *,
    target_token_budget: int = 4_000,
) -> PromptCompressionResult:
    """Compress one prompt text string into a separate compressed artifact."""

    return compress_prompt_sections(
        (
            PromptCompressionInputSection(
                section_id="prompt::text",
                role="user",
                name="custom",
                content=prompt_text,
            ),
        ),
        target_token_budget=target_token_budget,
    )


def compress_rendered_prompt(
    rendered_prompt: RenderedPromptResponse,
    *,
    target_token_budget: int = 4_000,
) -> PromptCompressionResult:
    """Compress rendered prompt sections without mutating the rendered prompt."""

    return compress_prompt_sections(
        tuple(
            PromptCompressionInputSection(
                section_id=f"rendered::{index}:{section.name.value}",
                role=section.role.value,
                name=section.name.value,
                content=section.content,
            )
            for index, section in enumerate(rendered_prompt.sections)
        ),
        target_token_budget=target_token_budget,
    )


def compress_prompt_sections(
    sections: tuple[PromptCompressionInputSection, ...],
    *,
    target_token_budget: int = 4_000,
) -> PromptCompressionResult:
    """Compress prompt sections with deterministic local text reduction."""

    if not sections:
        raise ValueError("prompt compression requires at least one section")
    if target_token_budget <= 0:
        raise ValueError("target_token_budget must be positive")

    original_total = sum(_estimate_tokens(section.content) for section in sections)
    section_budgets = _section_token_budgets(
        sections,
        target_token_budget=target_token_budget,
        original_total=original_total,
    )
    compressed_sections = tuple(
        _compress_section(section, section_budgets[section.section_id])
        for section in sections
    )
    compressed_total = sum(
        section.compressed_token_estimate for section in compressed_sections
    )
    saved_total = sum(section.saved_tokens for section in compressed_sections)

    return PromptCompressionResult(
        sections=compressed_sections,
        section_ids=tuple(section.section_id for section in compressed_sections),
        target_token_budget=target_token_budget,
        original_total_tokens=original_total,
        compressed_total_tokens=compressed_total,
        saved_total_tokens=saved_total,
        within_budget=compressed_total <= target_token_budget,
        compression_pressure=_compression_pressure(
            original_total=original_total,
            compressed_total=compressed_total,
            target_token_budget=target_token_budget,
        ),
        compressed_prompt_text=_join_compressed_text(compressed_sections),
        advisory_actions=_result_actions(saved_total),
    )


def prompt_compression_section_by_id(
    section_id: str,
    result: PromptCompressionResult | None = None,
) -> PromptCompressionSection | None:
    """Return one compressed section without mutating prompt text."""

    source_result = result or compress_prompt_text("Prompt compression placeholder.")
    for section in source_result.sections:
        if section.section_id == section_id:
            return section
    return None


def prompt_compression_sections_for_status(
    status: PromptCompressionStatus,
    result: PromptCompressionResult | None = None,
) -> tuple[PromptCompressionSection, ...]:
    """Return compressed sections by status without provider execution."""

    source_result = result or compress_prompt_text("Prompt compression placeholder.")
    return tuple(
        section
        for section in source_result.sections
        if section.compression_status == status
    )


def _compress_section(
    section: PromptCompressionInputSection,
    target_tokens: int,
) -> PromptCompressionSection:
    original = section.content
    original_tokens = _estimate_tokens(original)
    compressed = (
        original
        if original_tokens <= target_tokens
        else _compress_text_to_budget(original, target_tokens)
    )
    compressed_tokens = _estimate_tokens(compressed)
    if compressed_tokens >= original_tokens:
        compressed = original
        compressed_tokens = original_tokens
    saved_tokens = original_tokens - compressed_tokens

    return PromptCompressionSection(
        section_id=section.section_id,
        role=section.role,
        name=section.name,
        original_text=original,
        compressed_text=compressed,
        original_token_estimate=original_tokens,
        compressed_token_estimate=compressed_tokens,
        saved_tokens=saved_tokens,
        compression_status="compressed" if saved_tokens else "unchanged",
        compression_pressure=_section_pressure(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            target_tokens=target_tokens,
        ),
        evidence=(
            f"section:{section.section_id}",
            f"original_tokens:{original_tokens}",
            f"target_tokens:{target_tokens}",
            f"compressed_tokens:{compressed_tokens}",
        ),
        advisory_actions=_section_actions(saved_tokens),
    )


def _compress_text_to_budget(text: str, target_tokens: int) -> str:
    normalized_lines = tuple(
        " ".join(line.strip().split()) for line in text.splitlines() if line.strip()
    )
    normalized = (
        "\n".join(normalized_lines) if normalized_lines else " ".join(text.split())
    )
    if _estimate_tokens(normalized) <= target_tokens:
        return normalized

    marker = "[compressed: prompt detail omitted]"
    character_budget = max(24, target_tokens * 4 - len(marker) - 1)
    sentences = tuple(
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY.split(normalized)
        if sentence.strip()
    )
    selected: list[str] = []
    used = 0
    for sentence in sentences:
        separator = "\n" if selected else ""
        next_length = used + len(separator) + len(sentence)
        if next_length > character_budget:
            break
        selected.append(sentence)
        used = next_length

    if not selected:
        selected_text = normalized[:character_budget].rstrip()
    else:
        selected_text = "\n".join(selected).rstrip()
    return f"{selected_text}\n{marker}".strip()


def _section_token_budgets(
    sections: tuple[PromptCompressionInputSection, ...],
    *,
    target_token_budget: int,
    original_total: int,
) -> dict[str, int]:
    if original_total <= target_token_budget:
        return {
            section.section_id: _estimate_tokens(section.content)
            for section in sections
        }

    remaining = target_token_budget
    budgets: dict[str, int] = {}
    for index, section in enumerate(sections):
        original_tokens = _estimate_tokens(section.content)
        if index == len(sections) - 1:
            budget = max(1, remaining)
        else:
            proportional = max(
                1, target_token_budget * original_tokens // original_total
            )
            budget = min(original_tokens, proportional)
        budgets[section.section_id] = max(1, budget)
        remaining = max(0, remaining - budgets[section.section_id])
    return budgets


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _section_pressure(
    *,
    original_tokens: int,
    compressed_tokens: int,
    target_tokens: int,
) -> PromptCompressionPressure:
    if compressed_tokens > target_tokens:
        return "high"
    if compressed_tokens < original_tokens:
        return "medium"
    return "low"


def _compression_pressure(
    *,
    original_total: int,
    compressed_total: int,
    target_token_budget: int,
) -> PromptCompressionPressure:
    if compressed_total > target_token_budget:
        return "high"
    if compressed_total < original_total:
        return "medium"
    return "low"


def _section_actions(saved_tokens: int) -> tuple[str, ...]:
    actions = [
        "Produce compressed prompt text as a separate artifact.",
        "Preserve source prompt text for auditability.",
    ]
    if saved_tokens:
        actions.append("Record token savings without changing provider routing.")
    return tuple(actions)


def _result_actions(saved_total: int) -> tuple[str, ...]:
    actions = [
        "Expose compressed prompt text only through the compression result.",
        "Preserve source prompt, provider routing, and output boundaries.",
    ]
    if saved_total:
        actions.append("Use compressed prompt artifact only when explicitly selected.")
    else:
        actions.append("Keep original prompt text because it already fits the budget.")
    return tuple(actions)


def _join_compressed_text(
    sections: tuple[PromptCompressionSection, ...],
) -> str:
    return "\n\n".join(
        f"[{section.role}:{section.name}]\n{section.compressed_text}"
        for section in sections
    )
