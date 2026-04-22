"""Domain metadata used by routing, RAG, and UI layers."""

from __future__ import annotations

from dataclasses import dataclass

from creative_coding_assistant.contracts.requests import CreativeCodingDomain


@dataclass(frozen=True)
class DomainInfo:
    value: CreativeCodingDomain
    label: str
    official_source_key: str


SUPPORTED_DOMAINS: tuple[DomainInfo, ...] = (
    DomainInfo(CreativeCodingDomain.THREE_JS, "Three.js", "three_js"),
    DomainInfo(CreativeCodingDomain.REACT_THREE_FIBER, "React Three Fiber", "r3f"),
    DomainInfo(CreativeCodingDomain.P5_JS, "p5.js", "p5_js"),
    DomainInfo(CreativeCodingDomain.GLSL, "GLSL", "glsl"),
)


def get_supported_domain_values() -> tuple[str, ...]:
    """Return stable domain values for clients and validators."""

    return tuple(domain.value.value for domain in SUPPORTED_DOMAINS)
