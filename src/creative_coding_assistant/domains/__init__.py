"""Creative coding domain registry."""

from creative_coding_assistant.domains.experience import (
    DomainDeliveryKind,
    DomainExperienceRecord,
    DomainValidationStatus,
    domain_experience_records,
    get_domain_experience,
)
from creative_coding_assistant.domains.registry import (
    SUPPORTED_DOMAINS,
    DomainCategory,
    DomainInfo,
    get_domain_categories,
    get_domain_category,
    get_domain_category_label,
    get_domain_default_topic,
    get_domain_info,
    get_domain_label,
    get_domain_memory_label,
    get_domain_prompt_guidance,
    get_domain_slug,
    get_domains_for_category,
    get_supported_domain_values,
)

__all__ = [
    "DomainCategory",
    "DomainDeliveryKind",
    "DomainExperienceRecord",
    "DomainInfo",
    "DomainValidationStatus",
    "SUPPORTED_DOMAINS",
    "domain_experience_records",
    "get_domain_categories",
    "get_domain_category",
    "get_domain_category_label",
    "get_domain_default_topic",
    "get_domain_experience",
    "get_domain_info",
    "get_domain_label",
    "get_domain_memory_label",
    "get_domain_prompt_guidance",
    "get_domain_slug",
    "get_domains_for_category",
    "get_supported_domain_values",
]
