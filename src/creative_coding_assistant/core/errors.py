"""Shared exception types."""


class CreativeCodingAssistantError(Exception):
    """Base exception for application-level failures."""


class ConfigurationError(CreativeCodingAssistantError):
    """Raised when runtime configuration is invalid."""


class RoutingError(CreativeCodingAssistantError):
    """Raised when a request cannot be routed safely."""
