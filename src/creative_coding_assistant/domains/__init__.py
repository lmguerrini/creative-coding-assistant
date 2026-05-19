"""Creative coding domain registry."""

from creative_coding_assistant.domains.registry import (
    SUPPORTED_DOMAINS,
    DomainCategory,
    DomainInfo,
    get_domain_category,
    get_domain_default_topic,
    get_domain_info,
    get_domain_label,
    get_domain_memory_label,
    get_domain_prompt_guidance,
    get_domain_slug,
    get_supported_domain_values,
)

__all__ = [
    "DomainCategory",
    "DomainInfo",
    "SUPPORTED_DOMAINS",
    "get_domain_category",
    "get_domain_default_topic",
    "get_domain_info",
    "get_domain_label",
    "get_domain_memory_label",
    "get_domain_prompt_guidance",
    "get_domain_slug",
    "get_supported_domain_values",
]
