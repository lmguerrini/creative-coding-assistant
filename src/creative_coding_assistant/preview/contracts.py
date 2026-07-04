"""Typed preview contracts for future artifact renderers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.artifacts import (
    ArtifactOrigin,
    ArtifactRecord,
    ArtifactType,
    ArtifactWorkflowLink,
)
from creative_coding_assistant.contracts import CreativeCodingDomain

_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,63}$")

_BROWSER_SANDBOX_DOMAINS: frozenset[CreativeCodingDomain] = frozenset(
    {
        CreativeCodingDomain.THREE_JS,
        CreativeCodingDomain.REACT_THREE_FIBER,
        CreativeCodingDomain.P5_JS,
        CreativeCodingDomain.GLSL,
        CreativeCodingDomain.CANVAS_2D,
        CreativeCodingDomain.WEBGPU_WGSL,
        CreativeCodingDomain.GSAP,
        CreativeCodingDomain.TONE_JS,
        CreativeCodingDomain.PIXI_JS,
        CreativeCodingDomain.MATTER_JS,
        CreativeCodingDomain.RAPIER,
        CreativeCodingDomain.HYDRA,
        CreativeCodingDomain.SHADERTOY,
        CreativeCodingDomain.WEB_AUDIO_API,
        CreativeCodingDomain.P5_SOUND,
        CreativeCodingDomain.ML5_JS,
        CreativeCodingDomain.TENSORFLOW_JS,
        CreativeCodingDomain.CABLES_GL,
    }
)

_BROWSER_SANDBOX_LANGUAGES = frozenset(
    {"javascript", "typescript", "html", "css", "wgsl", "glsl"}
)


class PreviewStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


TERMINAL_PREVIEW_STATUSES: tuple[PreviewStatus, ...] = (
    PreviewStatus.SUCCEEDED,
    PreviewStatus.FAILED,
    PreviewStatus.SKIPPED,
)

_ALLOWED_PREVIEW_STATUS_TRANSITIONS: dict[
    PreviewStatus,
    tuple[PreviewStatus, ...],
] = {
    PreviewStatus.PENDING: (
        PreviewStatus.RUNNING,
        PreviewStatus.SUCCEEDED,
        PreviewStatus.FAILED,
        PreviewStatus.SKIPPED,
    ),
    PreviewStatus.RUNNING: (
        PreviewStatus.SUCCEEDED,
        PreviewStatus.FAILED,
    ),
    PreviewStatus.SUCCEEDED: (),
    PreviewStatus.FAILED: (),
    PreviewStatus.SKIPPED: (),
}


class PreviewTarget(StrEnum):
    BROWSER_SANDBOX = "browser_sandbox"
    IMAGE_ASSET = "image_asset"
    AUDIO_ASSET = "audio_asset"
    VIDEO_ASSET = "video_asset"
    TEXT_PANEL = "text_panel"
    JSON_PANEL = "json_panel"


class PreviewIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    preview_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    attempt: int = Field(default=1, ge=1)

    @field_validator("preview_id", "workspace_id", mode="before")
    @classmethod
    def normalize_identifiers(cls, value: object) -> str:
        return _normalize_identifier(value)


class PreviewArtifactLink(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    artifact_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    artifact_version: int = Field(ge=1)
    content_reference_id: str | None = None

    @field_validator(
        "artifact_id",
        "workspace_id",
        "content_reference_id",
        mode="before",
    )
    @classmethod
    def normalize_optional_identifiers(
        cls,
        value: object,
    ) -> str | None:
        if value is None:
            return None
        return _normalize_identifier(value)

    @classmethod
    def from_artifact(cls, artifact: ArtifactRecord) -> PreviewArtifactLink:
        primary_reference = artifact.primary_content_reference
        return cls(
            artifact_id=artifact.artifact_id,
            workspace_id=artifact.workspace_id,
            artifact_version=artifact.identity.version,
            content_reference_id=(
                primary_reference.reference_id
                if primary_reference is not None
                else None
            ),
        )


class PreviewRendererCapability(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    renderer_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    description: str | None = None
    supported_targets: tuple[PreviewTarget, ...]
    supported_artifact_types: tuple[ArtifactType, ...] = ()
    supported_domains: tuple[CreativeCodingDomain, ...] = ()
    supports_live_updates: bool = False
    produces_preview_artifact: bool = False
    tags: tuple[str, ...] = ()

    @field_validator("renderer_id", mode="before")
    @classmethod
    def normalize_renderer_id(cls, value: object) -> str:
        return _normalize_identifier(value)

    @field_validator("supported_targets", mode="before")
    @classmethod
    def normalize_supported_targets(
        cls,
        value: Sequence[PreviewTarget | str] | PreviewTarget | str,
    ) -> tuple[PreviewTarget, ...]:
        return _normalize_enum_sequence(value, PreviewTarget)

    @field_validator("supported_artifact_types", mode="before")
    @classmethod
    def normalize_supported_artifact_types(
        cls,
        value: Sequence[ArtifactType | str] | ArtifactType | str | None,
    ) -> tuple[ArtifactType, ...]:
        return _normalize_enum_sequence(value, ArtifactType)

    @field_validator("supported_domains", mode="before")
    @classmethod
    def normalize_supported_domains(
        cls,
        value: (
            Sequence[CreativeCodingDomain | str] | CreativeCodingDomain | str | None
        ),
    ) -> tuple[CreativeCodingDomain, ...]:
        return _normalize_enum_sequence(value, CreativeCodingDomain)

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(
        cls,
        value: Sequence[str] | str | None,
    ) -> tuple[str, ...]:
        return _normalize_tags(value)

    def supports_request(self, request: PreviewRequest) -> bool:
        if (
            request.preferred_renderer_id is not None
            and request.preferred_renderer_id != self.renderer_id
        ):
            return False
        if request.target not in self.supported_targets:
            return False
        if (
            self.supported_artifact_types
            and request.artifact_type not in self.supported_artifact_types
        ):
            return False
        if self.supported_domains and request.domain not in self.supported_domains:
            return False
        return True


class PreviewRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str = Field(min_length=1)
    identity: PreviewIdentity
    artifact_link: PreviewArtifactLink
    target: PreviewTarget
    artifact_type: ArtifactType
    domain: CreativeCodingDomain | None = None
    requested_at: datetime
    preferred_renderer_id: str | None = None
    workflow_link: ArtifactWorkflowLink | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "preferred_renderer_id", mode="before")
    @classmethod
    def normalize_optional_identifier(
        cls,
        value: object,
    ) -> str | None:
        if value is None:
            return None
        return _normalize_identifier(value)

    @field_validator("requested_at")
    @classmethod
    def require_requested_at_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value)

    @field_validator("options", mode="before")
    @classmethod
    def normalize_options(cls, value: dict[str, Any] | None) -> dict[str, Any]:
        return {} if value is None else dict(value)

    @property
    def preview_id(self) -> str:
        return self.identity.preview_id

    @property
    def artifact_id(self) -> str:
        return self.artifact_link.artifact_id

    @model_validator(mode="after")
    def validate_request_alignment(self) -> Self:
        if self.identity.workspace_id != self.artifact_link.workspace_id:
            raise ValueError(
                "Preview identity workspace_id must match the linked artifact."
            )
        if (
            self.workflow_link is not None
            and self.domain is not None
            and self.workflow_link.domains
            and self.domain not in self.workflow_link.domains
        ):
            raise ValueError(
                "Preview request domain must align with workflow-linked domains."
            )
        return self

    @classmethod
    def from_artifact(
        cls,
        artifact: ArtifactRecord,
        *,
        request_id: str,
        preview_id: str,
        requested_at: datetime,
        target: PreviewTarget | None = None,
        preferred_renderer_id: str | None = None,
        workflow_link: ArtifactWorkflowLink | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreviewRequest:
        resolved_target = target or resolve_preview_target_for_artifact(artifact)
        if resolved_target is None or not is_previewable_artifact(artifact):
            raise ValueError(
                "Artifact is not previewable with the current preview foundation."
            )
        return cls(
            request_id=request_id,
            identity=PreviewIdentity(
                preview_id=preview_id,
                workspace_id=artifact.workspace_id,
            ),
            artifact_link=PreviewArtifactLink.from_artifact(artifact),
            target=resolved_target,
            artifact_type=artifact.artifact_type,
            domain=artifact.metadata.domain,
            requested_at=requested_at,
            preferred_renderer_id=preferred_renderer_id,
            workflow_link=workflow_link or artifact.workflow_link,
            options=options or {},
        )


class PreviewError(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("code", mode="before")
    @classmethod
    def normalize_code(cls, value: object) -> str:
        return _normalize_identifier(value)


class PreviewProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    origin: ArtifactOrigin = ArtifactOrigin.ASSISTANT_WORKFLOW
    renderer_id: str | None = None
    renderer_version: str | None = None
    workflow_link: ArtifactWorkflowLink | None = None

    @field_validator("renderer_id", mode="before")
    @classmethod
    def normalize_renderer_id(
        cls,
        value: object,
    ) -> str | None:
        if value is None:
            return None
        return _normalize_identifier(value)


class PreviewResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: PreviewRequest
    status: PreviewStatus
    preview_artifact_id: str | None = None
    summary: str | None = None
    completed_at: datetime | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    provenance: PreviewProvenance | None = None
    error: PreviewError | None = None

    @field_validator("preview_artifact_id", mode="before")
    @classmethod
    def normalize_optional_preview_artifact_id(
        cls,
        value: object,
    ) -> str | None:
        if value is None:
            return None
        return _normalize_identifier(value)

    @field_validator("completed_at")
    @classmethod
    def validate_completed_at(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        if value is None:
            return None
        return _require_timezone(value)

    @field_validator("details", mode="before")
    @classmethod
    def normalize_details(cls, value: dict[str, Any] | None) -> dict[str, Any]:
        return {} if value is None else dict(value)

    @property
    def preview_id(self) -> str:
        return self.request.preview_id

    @property
    def artifact_id(self) -> str:
        return self.request.artifact_id

    @model_validator(mode="after")
    def validate_terminal_result(self) -> Self:
        if self.status not in TERMINAL_PREVIEW_STATUSES:
            raise ValueError("Preview results must use a terminal preview status.")

        if (
            self.completed_at is not None
            and self.completed_at < self.request.requested_at
        ):
            raise ValueError(
                "Preview completed_at cannot be earlier than requested_at."
            )

        if self.status is PreviewStatus.FAILED:
            if self.error is None:
                raise ValueError("Failed preview results require an error payload.")
            if self.preview_artifact_id is not None:
                raise ValueError(
                    "Failed preview results cannot include a preview artifact id."
                )
            return self

        if self.error is not None:
            raise ValueError(
                "Successful or skipped preview results cannot include an error payload."
            )

        if (
            self.status is PreviewStatus.SKIPPED
            and self.preview_artifact_id is not None
        ):
            raise ValueError(
                "Skipped preview results cannot include a preview artifact id."
            )

        return self

    @classmethod
    def succeeded(
        cls,
        *,
        request: PreviewRequest,
        preview_artifact_id: str | None = None,
        summary: str | None = None,
        completed_at: datetime | None = None,
        details: dict[str, Any] | None = None,
        provenance: PreviewProvenance | None = None,
    ) -> PreviewResult:
        return cls(
            request=request,
            status=PreviewStatus.SUCCEEDED,
            preview_artifact_id=preview_artifact_id,
            summary=summary,
            completed_at=completed_at,
            details=details or {},
            provenance=provenance,
        )

    @classmethod
    def failed(
        cls,
        *,
        request: PreviewRequest,
        code: str,
        message: str,
        retryable: bool = False,
        completed_at: datetime | None = None,
        details: dict[str, Any] | None = None,
        provenance: PreviewProvenance | None = None,
    ) -> PreviewResult:
        return cls(
            request=request,
            status=PreviewStatus.FAILED,
            completed_at=completed_at,
            provenance=provenance,
            error=PreviewError(
                code=code,
                message=message,
                retryable=retryable,
                details=details or {},
            ),
        )

    @classmethod
    def skipped(
        cls,
        *,
        request: PreviewRequest,
        summary: str | None = None,
        completed_at: datetime | None = None,
        details: dict[str, Any] | None = None,
        provenance: PreviewProvenance | None = None,
    ) -> PreviewResult:
        return cls(
            request=request,
            status=PreviewStatus.SKIPPED,
            summary=summary,
            completed_at=completed_at,
            details=details or {},
            provenance=provenance,
        )


class PreviewRenderer(Protocol):
    @property
    def capability(self) -> PreviewRendererCapability:
        """Expose stable metadata for one preview renderer."""

    def render(self, request: PreviewRequest) -> PreviewResult:
        """Process one typed preview request."""


class NoOpPreviewRenderer:
    """Foundation renderer that advertises capability without real rendering."""

    def __init__(
        self,
        capability: PreviewRendererCapability | None = None,
    ) -> None:
        self._capability = capability or PreviewRendererCapability(
            renderer_id="preview.noop",
            display_name="No-op Preview Renderer",
            description="Foundation-only renderer placeholder.",
            supported_targets=tuple(PreviewTarget),
            tags=("preview", "noop"),
        )

    @property
    def capability(self) -> PreviewRendererCapability:
        return self._capability

    def render(self, request: PreviewRequest) -> PreviewResult:
        if not self.capability.supports_request(request):
            return PreviewResult.skipped(
                request=request,
                summary="Preview request is unsupported by this renderer.",
                completed_at=request.requested_at,
                provenance=PreviewProvenance(
                    renderer_id=self.capability.renderer_id,
                    workflow_link=request.workflow_link,
                ),
                details={"renderer_id": self.capability.renderer_id},
            )

        return PreviewResult.skipped(
            request=request,
            summary="Preview pipeline foundation only; renderer execution is deferred.",
            completed_at=request.requested_at,
            provenance=PreviewProvenance(
                renderer_id=self.capability.renderer_id,
                workflow_link=request.workflow_link,
            ),
            details={"renderer_id": self.capability.renderer_id},
        )


def get_allowed_preview_status_transitions(
    status: PreviewStatus,
) -> tuple[PreviewStatus, ...]:
    return _ALLOWED_PREVIEW_STATUS_TRANSITIONS[status]


def can_transition_preview_status(
    current: PreviewStatus,
    next_status: PreviewStatus,
) -> bool:
    return next_status in get_allowed_preview_status_transitions(current)


def validate_preview_status_transition(
    current: PreviewStatus,
    next_status: PreviewStatus,
) -> None:
    if not can_transition_preview_status(current, next_status):
        raise ValueError(
            f"Invalid preview status transition: {current.value} -> "
            f"{next_status.value}."
        )


def is_previewable_artifact(artifact: ArtifactRecord) -> bool:
    return (
        artifact.primary_content_reference is not None
        and resolve_preview_target_for_artifact(artifact) is not None
    )


def resolve_preview_target_for_artifact(
    artifact: ArtifactRecord,
) -> PreviewTarget | None:
    if artifact.artifact_type is ArtifactType.IMAGE:
        return PreviewTarget.IMAGE_ASSET
    if artifact.artifact_type is ArtifactType.AUDIO:
        return PreviewTarget.AUDIO_ASSET
    if artifact.artifact_type is ArtifactType.VIDEO:
        return PreviewTarget.VIDEO_ASSET
    if artifact.artifact_type is ArtifactType.TEXT:
        return PreviewTarget.TEXT_PANEL
    if artifact.artifact_type is ArtifactType.JSON:
        return PreviewTarget.JSON_PANEL
    if artifact.artifact_type is not ArtifactType.CODE:
        return None

    domain = artifact.metadata.domain
    language = artifact.metadata.language
    if domain in _BROWSER_SANDBOX_DOMAINS:
        return PreviewTarget.BROWSER_SANDBOX
    if language is not None and language.strip().lower() in _BROWSER_SANDBOX_LANGUAGES:
        return PreviewTarget.BROWSER_SANDBOX
    return None


def _normalize_identifier(value: object) -> str:
    normalized = str(value).strip().lower()
    if not _IDENTIFIER_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Preview identifiers must be lowercase values using letters, "
            "numbers, '.', '-', or '_' characters."
        )
    return normalized


def _normalize_enum_sequence(
    value: Sequence[Any] | Any | None,
    enum_type: type[StrEnum],
) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, enum_type):
        return (value,)
    if isinstance(value, str):
        return (enum_type(value.strip()),)

    normalized: list[Any] = []
    for item in value:
        normalized_item = (
            item if isinstance(item, enum_type) else enum_type(str(item).strip())
        )
        if normalized_item not in normalized:
            normalized.append(normalized_item)
    return tuple(normalized)


def _normalize_tags(value: Sequence[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)

    normalized: list[str] = []
    for item in values:
        tag = "-".join(str(item).strip().lower().split())
        if not tag:
            continue
        if not _TAG_PATTERN.fullmatch(tag):
            raise ValueError(
                "Preview tags must be lowercase values using letters, numbers, "
                "'.', ':', '-', or '_' characters."
            )
        if tag not in normalized:
            normalized.append(tag)
    return tuple(normalized)


def _require_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Preview timestamps must be timezone-aware.")
    return value
