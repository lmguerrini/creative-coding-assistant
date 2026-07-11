"""Explicit official Knowledge Base inventory and update operations."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from creative_coding_assistant.api.contracts import (
    ApiRequestBodyError,
    StartResponse,
    empty_response,
    error_response,
    json_response,
    read_json_body,
    request_id_from_environ,
)
from creative_coding_assistant.api.cors import resolve_cors_allow_origin
from creative_coding_assistant.api.domain_experience import (
    build_domain_experience_payload,
)
from creative_coding_assistant.app.sync import SyncFailureMode
from creative_coding_assistant.app.sync_bootstrap import build_official_kb_sync_service
from creative_coding_assistant.core.config import Settings, load_settings
from creative_coding_assistant.rag.sources import get_official_source
from creative_coding_assistant.rag.sync import (
    OfficialSourceFetcher,
    OfficialSourceNormalizer,
    OfficialSourceSyncRequest,
    UrllibSourceTransport,
)

KNOWLEDGE_BASE_CONTRACT_VERSION = "knowledge-base.v1"
KNOWLEDGE_BASE_CONTRACT_HEADER = "X-CCA-Knowledge-Base-Contract-Version"
DEFAULT_KNOWLEDGE_BASE_PATH = "/api/knowledge-base"
KNOWLEDGE_BASE_METHODS = "GET, POST, OPTIONS"
MAX_KNOWLEDGE_BASE_REQUEST_BYTES = 32 * 1024


class KnowledgeBaseOperationRequest(BaseModel):
    """A user-triggered operation; mutating operations require confirmation."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    action: Literal["check", "validate", "update", "rebuild"]
    source_ids: tuple[str, ...] = Field(default_factory=tuple, alias="sourceIds", max_length=57)
    confirmed: bool = False

    @field_validator("source_ids")
    @classmethod
    def validate_source_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        resolved: list[str] = []
        for source_id in value:
            source = get_official_source(source_id)
            if source.source_id not in resolved:
                resolved.append(source.source_id)
        return tuple(resolved)


class KnowledgeBaseApplication:
    """Expose controlled official-source operations without background mutation."""

    def __init__(
        self,
        *,
        settings_factory: Callable[[], Settings] = load_settings,
        sync_service_factory: Callable[..., Any] = build_official_kb_sync_service,
        source_check_fn: Callable[
            [Sequence[str], dict[str, str]], list[dict[str, object]]
        ]
        | None = None,
        path: str = DEFAULT_KNOWLEDGE_BASE_PATH,
    ) -> None:
        self._settings_factory = settings_factory
        self._sync_service_factory = sync_service_factory
        self._source_check_fn = source_check_fn or _check_official_sources
        self._path = path

    def __call__(
        self,
        environ: dict[str, Any],
        start_response: StartResponse,
    ) -> Iterable[bytes]:
        request_id = request_id_from_environ(environ)
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        settings = self._settings_factory()
        allow_origin = resolve_cors_allow_origin(environ, settings=settings)

        if str(environ.get("PATH_INFO", "")) != self._path:
            return _error(
                start_response,
                HTTPStatus.NOT_FOUND,
                "not_found",
                "Knowledge Base route was not found.",
                request_id,
                allow_origin,
            )
        if method == "OPTIONS":
            return empty_response(
                start_response,
                HTTPStatus.NO_CONTENT,
                request_id=request_id,
                allow_methods=KNOWLEDGE_BASE_METHODS,
                allow_origin=allow_origin,
                extra_headers=[(KNOWLEDGE_BASE_CONTRACT_HEADER, KNOWLEDGE_BASE_CONTRACT_VERSION)],
            )
        if method == "GET":
            return json_response(
                start_response,
                HTTPStatus.OK,
                _inventory_payload(settings),
                request_id=request_id,
                allow_methods=KNOWLEDGE_BASE_METHODS,
                allow_origin=allow_origin,
                extra_headers=[(KNOWLEDGE_BASE_CONTRACT_HEADER, KNOWLEDGE_BASE_CONTRACT_VERSION)],
            )
        if method != "POST":
            return _error(
                start_response,
                HTTPStatus.METHOD_NOT_ALLOWED,
                "method_not_allowed",
                "Knowledge Base accepts GET, POST, and OPTIONS.",
                request_id,
                allow_origin,
            )
        try:
            request = KnowledgeBaseOperationRequest.model_validate(
                read_json_body(environ, max_bytes=MAX_KNOWLEDGE_BASE_REQUEST_BYTES)
            )
        except ApiRequestBodyError as exc:
            return _error(
                start_response,
                exc.status,
                exc.code,
                exc.message,
                request_id,
                allow_origin,
            )
        except (ValidationError, ValueError):
            return _error(
                start_response,
                HTTPStatus.BAD_REQUEST,
                "invalid_knowledge_base_operation",
                "Knowledge Base operation options were invalid.",
                request_id,
                allow_origin,
            )

        if request.action in {"check", "update", "rebuild"} and not request.source_ids:
            return _error(
                start_response,
                HTTPStatus.BAD_REQUEST,
                "knowledge_base_source_selection_required",
                "Select one or more official sources before checking or changing the local index.",
                request_id,
                allow_origin,
            )

        if request.action in {"update", "rebuild"} and not request.confirmed:
            return _error(
                start_response,
                HTTPStatus.BAD_REQUEST,
                "knowledge_base_confirmation_required",
                "Updating the local index requires explicit confirmation.",
                request_id,
                allow_origin,
            )

        if request.action == "check":
            try:
                inventory = _inventory_payload(settings)
                local_fingerprints = {
                    str(source.get("id")): str(source["fingerprint"])
                    for source in inventory.get("sources", [])
                    if isinstance(source, dict)
                    and isinstance(source.get("fingerprint"), str)
                }
                source_changes = self._source_check_fn(
                    request.source_ids, local_fingerprints
                )
            except Exception:
                return _error(
                    start_response,
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "knowledge_base_check_unavailable",
                    "Official sources could not be checked. The local index was not changed.",
                    request_id,
                    allow_origin,
                )
            payload = {
                "action": "check",
                "status": "review_ready",
                "detail": (
                    "Official source content was checked after your request. Review the "
                    "change summary before updating the local index."
                ),
                "sourceIds": list(request.source_ids),
                "sourceChanges": source_changes,
                "cancellation": "Check completed before an update or rebuild started.",
                "inventory": inventory,
            }
        elif request.action == "validate":
            payload = _validation_payload(settings, request.source_ids)
        else:
            payload, operation_error = self._run_update(settings, request)
            if operation_error is not None:
                return _error(
                    start_response,
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "knowledge_base_update_failed",
                    operation_error,
                    request_id,
                    allow_origin,
                )

        return json_response(
            start_response,
            HTTPStatus.OK,
            payload,
            request_id=request_id,
            allow_methods=KNOWLEDGE_BASE_METHODS,
            allow_origin=allow_origin,
            extra_headers=[(KNOWLEDGE_BASE_CONTRACT_HEADER, KNOWLEDGE_BASE_CONTRACT_VERSION)],
        )

    def _run_update(
        self,
        settings: Settings,
        request: KnowledgeBaseOperationRequest,
    ) -> tuple[dict[str, object], str | None]:
        source_ids = request.source_ids
        before = _inventory_payload(settings)
        with _IndexBackup(settings.chroma_persist_dir) as backup:
            try:
                service = self._sync_service_factory(
                    settings=settings,
                    failure_mode=SyncFailureMode.CONTINUE,
                )
                result = service.sync_selected_sources(source_ids)
                if result.failed_count:
                    backup.restore()
                    return {}, "Some selected sources failed to update; the prior local index was restored."
            except Exception:
                backup.restore()
                return {}, "The Knowledge Base update failed; the prior local index was restored."

        after = _inventory_payload(settings)
        return (
            {
                "action": request.action,
                "status": "completed",
                "detail": "Selected official sources were updated after explicit confirmation.",
                "sourceIds": list(result.source_ids),
                "succeededSourceIds": [
                    item.request.source_id for item in result.results
                ],
                "failedSourceIds": list(result.failed_source_ids),
                "chunkCount": result.total_chunks,
                "recordCount": result.total_records,
                "sourceSummary": _source_count_delta(before, after, result.source_ids),
                "validation": _validation_payload(settings, result.source_ids),
                "cancellation": "Not available after an in-process update begins.",
                "inventory": after,
            },
            None,
        )


def create_knowledge_base_app(
    *, settings_factory: Callable[[], Settings] = load_settings
) -> KnowledgeBaseApplication:
    return KnowledgeBaseApplication(settings_factory=settings_factory)


def _inventory_payload(settings: Settings) -> dict[str, object]:
    return build_domain_experience_payload(
        chroma_sqlite_path=settings.chroma_persist_dir / "chroma.sqlite3"
    )["knowledgeBase"]


def _check_official_sources(
    source_ids: Sequence[str], local_fingerprints: dict[str, str]
) -> list[dict[str, object]]:
    """Fetch selected approved sources for a review-only content fingerprint."""

    fetcher = OfficialSourceFetcher(transport=UrllibSourceTransport())
    normalizer = OfficialSourceNormalizer()
    checked_at = datetime.now(tz=UTC)
    changes: list[dict[str, object]] = []
    for source_id in source_ids:
        fetched = fetcher.fetch(
            OfficialSourceSyncRequest(source_id=source_id, requested_at=checked_at)
        )
        normalized = normalizer.normalize(fetched)
        local_fingerprint = local_fingerprints.get(source_id)
        changes.append(
            {
                "sourceId": source_id,
                "fetchedAt": checked_at.isoformat(),
                "fingerprint": normalized.content_hash,
                "localFingerprint": local_fingerprint,
                "changeStatus": (
                    "new" if local_fingerprint is None else "unchanged"
                    if local_fingerprint == normalized.content_hash
                    else "changed"
                ),
                "resolvedUrl": normalized.resolved_url,
            }
        )
    return changes


def _validation_payload(
    settings: Settings, source_ids: Sequence[str]
) -> dict[str, object]:
    inventory = _inventory_payload(settings)
    selected = set(source_ids)
    sources = [
        source
        for source in inventory.get("sources", [])
        if not selected or source.get("id") in selected
    ]
    missing = [source["id"] for source in sources if not source.get("indexed")]
    return {
        "action": "validate",
        "status": "passed" if not missing else "needs_indexing",
        "detail": (
            "Selected sources have local indexed records. Retrieval quality is evaluated "
            "separately by the explicit RAGAs action."
            if not missing
            else "Some selected official sources have no local index records."
        ),
        "sourceIds": list(source_ids),
        "missingSourceIds": missing,
        "localIndexValidation": "completed",
        "postBuildRetrievalValidation": "deferred_to_explicit_evaluation",
        "inventory": inventory,
    }


def _source_count_delta(
    before: dict[str, object], after: dict[str, object], source_ids: Sequence[str]
) -> list[dict[str, object]]:
    before_counts = {
        str(source.get("id")): int(source.get("chunkCount", 0))
        for source in before.get("sources", [])
        if isinstance(source, dict)
    }
    after_counts = {
        str(source.get("id")): int(source.get("chunkCount", 0))
        for source in after.get("sources", [])
        if isinstance(source, dict)
    }
    return [
        {
            "sourceId": source_id,
            "beforeChunkCount": before_counts.get(source_id, 0),
            "afterChunkCount": after_counts.get(source_id, 0),
        }
        for source_id in source_ids
    ]


class _IndexBackup:
    """Restore the prior Chroma directory if an explicit rebuild fails."""

    def __init__(self, index_dir: Path) -> None:
        self._index_dir = index_dir
        self._original_exists = index_dir.exists()
        self._temporary_dir: tempfile.TemporaryDirectory[str] | None = None
        self._backup_path: Path | None = None

    def __enter__(self) -> _IndexBackup:
        self._temporary_dir = tempfile.TemporaryDirectory(prefix="cca-kb-backup-")
        self._backup_path = Path(self._temporary_dir.name) / "chroma"
        if self._original_exists:
            shutil.copytree(self._index_dir, self._backup_path)
        return self

    def restore(self) -> None:
        if self._index_dir.exists():
            shutil.rmtree(self._index_dir)
        if self._original_exists:
            assert self._backup_path is not None
            shutil.copytree(self._backup_path, self._index_dir)

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        if self._temporary_dir is not None:
            self._temporary_dir.cleanup()


def _error(
    start_response: StartResponse,
    status: HTTPStatus,
    code: str,
    message: str,
    request_id: str,
    allow_origin: str | None,
) -> Iterable[bytes]:
    return error_response(
        start_response,
        status,
        error=code,
        message=message,
        request_id=request_id,
        allow_methods=KNOWLEDGE_BASE_METHODS,
        allow_origin=allow_origin,
        extra_headers=[
            ("Allow", KNOWLEDGE_BASE_METHODS),
            (KNOWLEDGE_BASE_CONTRACT_HEADER, KNOWLEDGE_BASE_CONTRACT_VERSION),
        ],
    )
