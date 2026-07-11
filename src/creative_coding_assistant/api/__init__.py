"""HTTP API helpers for browser-facing assistant clients."""

from creative_coding_assistant.api.contracts import (
    API_CONTRACT_VERSION,
    ERROR_CONTRACT_VERSION,
    HEALTH_CONTRACT_VERSION,
    STREAM_CONTRACT_VERSION,
    WORKSPACE_SESSION_CONTRACT_VERSION,
    ApiErrorResponse,
)
from creative_coding_assistant.api.cors import (
    CorsPolicyReport,
    build_cors_policy_report,
    resolve_cors_allow_origin,
)
from creative_coding_assistant.api.dev_server import (
    BackendDevApplication,
    MountedWsgiApp,
    create_backend_dev_app,
    run_backend_dev_server,
)
from creative_coding_assistant.api.domain_experience import (
    DOMAIN_EXPERIENCE_CONTRACT_VERSION,
    DomainExperienceApplication,
    build_domain_experience_payload,
    create_domain_experience_app,
)
from creative_coding_assistant.api.health import (
    HealthCheckApplication,
    build_health_payload,
    create_health_check_app,
)
from creative_coding_assistant.api.production import (
    CHROMA_REQUIREMENT,
    CONFIGURATION_MIGRATIONS,
    V7_5_ROADMAP_ITEMS,
    ApiTelemetryEvent,
    DependencyHealthReport,
    DeploymentReadinessChecklist,
    ProductionConfigurationReport,
    ReleaseChecklist,
    build_api_telemetry_event,
    build_dependency_health_report,
    build_deployment_readiness_checklist,
    build_release_checklist,
    log_api_telemetry_event,
    validate_production_configuration,
)
from creative_coding_assistant.api.streaming import (
    AssistantStreamingApplication,
    AssistantStreamRequest,
    create_assistant_streaming_app,
    iter_assistant_stream_ndjson,
    serialize_stream_event,
)
from creative_coding_assistant.api.workspace_sessions import (
    WorkspaceSessionApplication,
    create_workspace_session_app,
)

__all__ = [
    "API_CONTRACT_VERSION",
    "CHROMA_REQUIREMENT",
    "CONFIGURATION_MIGRATIONS",
    "ERROR_CONTRACT_VERSION",
    "DOMAIN_EXPERIENCE_CONTRACT_VERSION",
    "HEALTH_CONTRACT_VERSION",
    "STREAM_CONTRACT_VERSION",
    "V7_5_ROADMAP_ITEMS",
    "WORKSPACE_SESSION_CONTRACT_VERSION",
    "ApiErrorResponse",
    "ApiTelemetryEvent",
    "AssistantStreamRequest",
    "AssistantStreamingApplication",
    "BackendDevApplication",
    "CorsPolicyReport",
    "DependencyHealthReport",
    "DeploymentReadinessChecklist",
    "DomainExperienceApplication",
    "HealthCheckApplication",
    "MountedWsgiApp",
    "ProductionConfigurationReport",
    "ReleaseChecklist",
    "build_api_telemetry_event",
    "build_cors_policy_report",
    "build_dependency_health_report",
    "build_domain_experience_payload",
    "build_deployment_readiness_checklist",
    "build_health_payload",
    "build_release_checklist",
    "WorkspaceSessionApplication",
    "create_backend_dev_app",
    "create_domain_experience_app",
    "create_assistant_streaming_app",
    "create_health_check_app",
    "create_workspace_session_app",
    "iter_assistant_stream_ndjson",
    "log_api_telemetry_event",
    "run_backend_dev_server",
    "serialize_stream_event",
    "resolve_cors_allow_origin",
    "validate_production_configuration",
]
