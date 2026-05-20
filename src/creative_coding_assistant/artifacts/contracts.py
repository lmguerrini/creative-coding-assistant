"""Typed artifact contracts for workflow-linked creative outputs."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.contracts import (
    AssistantMode,
    AssistantRequest,
    CreativeCodingDomain,
)

_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,63}$")


class ArtifactCategory(StrEnum):
    GENERATED = "generated"
    PREVIEW = "preview"
    SOURCE = "source"
    EXPORT = "export"
    SUPPORT = "support"


class ArtifactType(StrEnum):
    CODE = "code"
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    JSON = "json"
    ARCHIVE = "archive"
    BINARY = "binary"


class ArtifactStatus(StrEnum):
    DRAFT = "draft"
    READY = "ready"
    SUPERSEDED = "superseded"
    FAILED = "failed"
    ARCHIVED = "archived"


_ALLOWED_ARTIFACT_STATUS_TRANSITIONS: dict[
    ArtifactStatus,
    tuple[ArtifactStatus, ...],
] = {
    ArtifactStatus.DRAFT: (
        ArtifactStatus.READY,
        ArtifactStatus.FAILED,
        ArtifactStatus.ARCHIVED,
    ),
    ArtifactStatus.READY: (
        ArtifactStatus.SUPERSEDED,
        ArtifactStatus.FAILED,
        ArtifactStatus.ARCHIVED,
    ),
    ArtifactStatus.SUPERSEDED: (ArtifactStatus.ARCHIVED,),
    ArtifactStatus.FAILED: (
        ArtifactStatus.DRAFT,
        ArtifactStatus.ARCHIVED,
    ),
    ArtifactStatus.ARCHIVED: (),
}


class ArtifactContentLocator(StrEnum):
    WORKSPACE_FILE = "workspace_file"
    URI = "uri"


class ArtifactOrigin(StrEnum):
    ASSISTANT_WORKFLOW = "assistant_workflow"
    USER = "user"
    IMPORT = "import"
    SYSTEM = "system"


class ArtifactIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    artifact_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)

    @field_validator("artifact_id", "workspace_id", mode="before")
    @classmethod
    def normalize_identifiers(cls, value: object) -> str:
        return _normalize_identifier(value)


class ArtifactMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    title: str | None = None
    summary: str | None = None
    tags: tuple[str, ...] = ()
    domain: CreativeCodingDomain | None = None
    language: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(
        cls,
        value: Sequence[str] | str | None,
    ) -> tuple[str, ...]:
        return _normalize_tags(value)

    @field_validator("extra", mode="before")
    @classmethod
    def normalize_extra(cls, value: dict[str, Any] | None) -> dict[str, Any]:
        return {} if value is None else dict(value)


class ArtifactContentReference(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    reference_id: str = Field(min_length=1)
    locator: ArtifactContentLocator
    label: str | None = None
    workspace_path: str | None = None
    uri: str | None = None
    mime_type: str | None = None
    content_hash: str | None = None
    byte_size: int | None = Field(default=None, ge=0)
    is_primary: bool = False

    @field_validator("reference_id", mode="before")
    @classmethod
    def normalize_reference_id(cls, value: object) -> str:
        return _normalize_identifier(value)

    @field_validator("workspace_path")
    @classmethod
    def validate_workspace_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                "Workspace content references must use safe relative paths."
            )
        return path.as_posix()

    @field_validator("content_hash")
    @classmethod
    def normalize_content_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        return normalized or None

    @model_validator(mode="after")
    def validate_locator_fields(self) -> Self:
        if self.locator is ArtifactContentLocator.WORKSPACE_FILE:
            if self.workspace_path is None:
                raise ValueError(
                    "Workspace-file artifact content references require a "
                    "workspace_path."
                )
            if self.uri is not None:
                raise ValueError(
                    "Workspace-file artifact content references cannot declare a uri."
                )
            return self

        if self.uri is None:
            raise ValueError("URI artifact content references require a uri.")
        if self.workspace_path is not None:
            raise ValueError(
                "URI artifact content references cannot declare a workspace_path."
            )
        return self


class ArtifactLineage(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    parent_artifact_ids: tuple[str, ...] = ()
    supersedes_artifact_id: str | None = None
    superseded_by_artifact_id: str | None = None

    @field_validator("parent_artifact_ids", mode="before")
    @classmethod
    def normalize_parent_artifact_ids(
        cls,
        value: Sequence[str] | str | None,
    ) -> tuple[str, ...]:
        return _normalize_identifier_sequence(value)

    @field_validator(
        "supersedes_artifact_id",
        "superseded_by_artifact_id",
        mode="before",
    )
    @classmethod
    def normalize_optional_identifier(cls, value: object) -> str | None:
        if value is None:
            return None
        return _normalize_identifier(value)

    @model_validator(mode="after")
    def validate_supersession_links(self) -> Self:
        if (
            self.supersedes_artifact_id is not None
            and self.supersedes_artifact_id == self.superseded_by_artifact_id
        ):
            raise ValueError(
                "Artifact lineage cannot supersede and be superseded by "
                "the same artifact."
            )
        return self


class ArtifactProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    origin: ArtifactOrigin = ArtifactOrigin.ASSISTANT_WORKFLOW
    generator: str | None = None
    source_uri: str | None = None
    source_artifact_ids: tuple[str, ...] = ()

    @field_validator("source_artifact_ids", mode="before")
    @classmethod
    def normalize_source_artifact_ids(
        cls,
        value: Sequence[str] | str | None,
    ) -> tuple[str, ...]:
        return _normalize_identifier_sequence(value)


class ArtifactWorkflowLink(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    workflow_run_id: str = Field(min_length=1)
    assistant_mode: AssistantMode
    conversation_id: str | None = None
    project_id: str | None = None
    domains: tuple[CreativeCodingDomain, ...] = ()
    step: str | None = None

    @field_validator("workflow_run_id", mode="before")
    @classmethod
    def normalize_workflow_run_id(cls, value: object) -> str:
        return _normalize_identifier(value)

    @field_validator("domains", mode="before")
    @classmethod
    def normalize_domains(
        cls,
        value: Sequence[CreativeCodingDomain | str] | CreativeCodingDomain | str | None,
    ) -> tuple[CreativeCodingDomain, ...]:
        if value is None:
            return ()
        if isinstance(value, CreativeCodingDomain):
            return (value,)
        if isinstance(value, str):
            return (CreativeCodingDomain(value.strip()),)

        normalized: list[CreativeCodingDomain] = []
        for item in value:
            domain = (
                item
                if isinstance(item, CreativeCodingDomain)
                else CreativeCodingDomain(str(item).strip())
            )
            if domain not in normalized:
                normalized.append(domain)
        return tuple(normalized)

    @classmethod
    def from_request(
        cls,
        request: AssistantRequest,
        *,
        workflow_run_id: str,
        step: str | None = None,
    ) -> ArtifactWorkflowLink:
        return cls(
            workflow_run_id=workflow_run_id,
            assistant_mode=request.mode,
            conversation_id=request.conversation_id,
            project_id=request.project_id,
            domains=request.domains,
            step=step,
        )


class ArtifactTimestamps(BaseModel):
    model_config = ConfigDict(frozen=True)

    created_at: datetime
    updated_at: datetime
    status_changed_at: datetime | None = None

    @field_validator("created_at", "updated_at", "status_changed_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _require_timezone(value)

    @model_validator(mode="after")
    def validate_ordering(self) -> Self:
        if self.updated_at < self.created_at:
            raise ValueError("Artifact updated_at cannot be earlier than created_at.")
        if self.status_changed_at is None:
            object.__setattr__(self, "status_changed_at", self.updated_at)
            return self
        if self.status_changed_at < self.created_at:
            raise ValueError(
                "Artifact status_changed_at cannot be earlier than created_at."
            )
        if self.status_changed_at > self.updated_at:
            raise ValueError(
                "Artifact status_changed_at cannot be later than updated_at."
            )
        return self


class ArtifactRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    identity: ArtifactIdentity
    category: ArtifactCategory
    artifact_type: ArtifactType
    status: ArtifactStatus = ArtifactStatus.DRAFT
    metadata: ArtifactMetadata = Field(default_factory=ArtifactMetadata)
    content_references: tuple[ArtifactContentReference, ...] = ()
    timestamps: ArtifactTimestamps
    provenance: ArtifactProvenance | None = None
    lineage: ArtifactLineage = Field(default_factory=ArtifactLineage)
    workflow_link: ArtifactWorkflowLink | None = None

    @field_validator("content_references", mode="before")
    @classmethod
    def normalize_content_references(
        cls,
        value: (
            Sequence[ArtifactContentReference | dict[str, Any]]
            | ArtifactContentReference
            | dict[str, Any]
            | None
        ),
    ) -> tuple[ArtifactContentReference, ...]:
        if value is None:
            return ()
        if isinstance(value, ArtifactContentReference):
            return (value,)
        if isinstance(value, dict):
            return (ArtifactContentReference.model_validate(value),)

        references: list[ArtifactContentReference] = []
        seen: set[str] = set()
        for item in value:
            reference = (
                item
                if isinstance(item, ArtifactContentReference)
                else ArtifactContentReference.model_validate(item)
            )
            if reference.reference_id in seen:
                continue
            seen.add(reference.reference_id)
            references.append(reference)
        return tuple(references)

    @property
    def artifact_id(self) -> str:
        return self.identity.artifact_id

    @property
    def workspace_id(self) -> str:
        return self.identity.workspace_id

    @property
    def primary_content_reference(self) -> ArtifactContentReference | None:
        if not self.content_references:
            return None
        for reference in self.content_references:
            if reference.is_primary:
                return reference
        return self.content_references[0]

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        primary_reference_count = sum(
            1 for reference in self.content_references if reference.is_primary
        )
        if primary_reference_count > 1:
            raise ValueError(
                "Artifacts can declare at most one primary content reference."
            )
        if (
            self.status is ArtifactStatus.SUPERSEDED
            and self.lineage.superseded_by_artifact_id is None
        ):
            raise ValueError(
                "Superseded artifacts require a "
                "superseded_by_artifact_id lineage reference."
            )
        if (
            self.workflow_link is not None
            and self.metadata.domain is not None
            and self.workflow_link.domains
            and self.metadata.domain not in self.workflow_link.domains
        ):
            raise ValueError(
                "Artifact metadata domain must align with workflow-linked domains."
            )
        return self


class ArtifactWorkspace(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    workspace_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    conversation_id: str | None = None
    project_id: str | None = None
    artifact_ids: tuple[str, ...] = ()
    workflow_run_ids: tuple[str, ...] = ()
    created_at: datetime
    updated_at: datetime

    @field_validator("workspace_id", mode="before")
    @classmethod
    def normalize_workspace_id(cls, value: object) -> str:
        return _normalize_identifier(value)

    @field_validator("artifact_ids", "workflow_run_ids", mode="before")
    @classmethod
    def normalize_identifier_fields(
        cls,
        value: Sequence[str] | str | None,
    ) -> tuple[str, ...]:
        return _normalize_identifier_sequence(value)

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_workspace_timestamps(cls, value: datetime) -> datetime:
        return _require_timezone(value)

    @model_validator(mode="after")
    def validate_timestamp_ordering(self) -> Self:
        if self.updated_at < self.created_at:
            raise ValueError(
                "Artifact workspace updated_at cannot be earlier than created_at."
            )
        return self

    @property
    def artifact_count(self) -> int:
        return len(self.artifact_ids)

    @classmethod
    def from_request(
        cls,
        request: AssistantRequest,
        *,
        workspace_id: str,
        session_id: str,
        created_at: datetime,
    ) -> ArtifactWorkspace:
        return cls(
            workspace_id=workspace_id,
            session_id=session_id,
            conversation_id=request.conversation_id,
            project_id=request.project_id,
            created_at=created_at,
            updated_at=created_at,
        )


def get_allowed_artifact_status_transitions(
    status: ArtifactStatus,
) -> tuple[ArtifactStatus, ...]:
    return _ALLOWED_ARTIFACT_STATUS_TRANSITIONS[status]


def can_transition_artifact_status(
    current: ArtifactStatus,
    next_status: ArtifactStatus,
) -> bool:
    return next_status in get_allowed_artifact_status_transitions(current)


def validate_artifact_status_transition(
    current: ArtifactStatus,
    next_status: ArtifactStatus,
) -> None:
    if not can_transition_artifact_status(current, next_status):
        raise ValueError(
            f"Invalid artifact status transition: {current.value} -> "
            f"{next_status.value}."
        )


def transition_artifact_status(
    artifact: ArtifactRecord,
    next_status: ArtifactStatus,
    *,
    changed_at: datetime | None = None,
) -> ArtifactRecord:
    validate_artifact_status_transition(artifact.status, next_status)

    effective_changed_at = artifact.timestamps.updated_at
    if changed_at is not None:
        effective_changed_at = _require_timezone(changed_at)
        if effective_changed_at < artifact.timestamps.updated_at:
            raise ValueError(
                "Artifact status transitions cannot move timestamps backwards."
            )

    return artifact.model_copy(
        update={
            "status": next_status,
            "timestamps": artifact.timestamps.model_copy(
                update={
                    "updated_at": effective_changed_at,
                    "status_changed_at": effective_changed_at,
                }
            ),
        }
    )


def attach_artifact_to_workspace(
    workspace: ArtifactWorkspace,
    artifact: ArtifactRecord,
) -> ArtifactWorkspace:
    if artifact.workspace_id != workspace.workspace_id:
        raise ValueError("Artifact workspace_id must match the target workspace.")

    artifact_ids = _append_unique_identifier(
        workspace.artifact_ids,
        artifact.artifact_id,
    )
    workflow_run_ids = workspace.workflow_run_ids
    if artifact.workflow_link is not None:
        workflow_run_ids = _append_unique_identifier(
            workflow_run_ids,
            artifact.workflow_link.workflow_run_id,
        )

    updated_at = max(workspace.updated_at, artifact.timestamps.updated_at)
    return workspace.model_copy(
        update={
            "artifact_ids": artifact_ids,
            "workflow_run_ids": workflow_run_ids,
            "updated_at": updated_at,
        }
    )


def _append_unique_identifier(
    identifiers: tuple[str, ...],
    identifier: str,
) -> tuple[str, ...]:
    if identifier in identifiers:
        return identifiers
    return (*identifiers, identifier)


def _normalize_identifier(value: object) -> str:
    normalized = str(value).strip().lower()
    if not _IDENTIFIER_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Artifact identifiers must be lowercase values using letters, numbers, "
            "'.', '-', or '_' characters."
        )
    return normalized


def _normalize_identifier_sequence(
    value: Sequence[str] | str | None,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)

    normalized: list[str] = []
    for item in values:
        identifier = _normalize_identifier(item)
        if identifier not in normalized:
            normalized.append(identifier)
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
                "Artifact tags must be lowercase values using letters, numbers, "
                "'.', ':', '-', or '_' characters."
            )
        if tag not in normalized:
            normalized.append(tag)
    return tuple(normalized)


def _require_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Artifact timestamps must be timezone-aware.")
    return value
