"""Read-only domain contracts and persistent knowledge-base inventory API."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from http import HTTPStatus
from pathlib import Path
from typing import Any

from creative_coding_assistant.api.contracts import (
    StartResponse,
    empty_response,
    error_response,
    json_response,
    request_id_from_environ,
)
from creative_coding_assistant.api.cors import resolve_cors_allow_origin
from creative_coding_assistant.core.config import Settings, load_settings
from creative_coding_assistant.domains import domain_experience_records
from creative_coding_assistant.knowledge.creative_distillation import (
    build_kb_reality_snapshot,
    inventory_local_chroma_kb,
)
from creative_coding_assistant.rag.sources import approved_official_sources

DOMAIN_EXPERIENCE_CONTRACT_VERSION = "domain-experience.v1"
DOMAIN_EXPERIENCE_CONTRACT_HEADER = "X-CCA-Domain-Experience-Contract-Version"
DEFAULT_DOMAIN_EXPERIENCE_PATH = "/api/domain-experience"
DOMAIN_EXPERIENCE_METHODS = "GET, OPTIONS"


class DomainExperienceApplication:
    """Expose only public domain claims and read-only local KB inventory."""

    def __init__(
        self,
        *,
        settings_factory: Callable[[], Settings] = load_settings,
        path: str = DEFAULT_DOMAIN_EXPERIENCE_PATH,
    ) -> None:
        self._settings_factory = settings_factory
        self._path = path

    def __call__(
        self,
        environ: dict[str, Any],
        start_response: StartResponse,
    ) -> Iterable[bytes]:
        request_id = request_id_from_environ(environ)
        path = str(environ.get("PATH_INFO", ""))
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        settings = self._settings_factory()
        allow_origin = resolve_cors_allow_origin(environ, settings=settings)

        if path != self._path:
            return error_response(
                start_response,
                HTTPStatus.NOT_FOUND,
                error="not_found",
                message="Domain experience route was not found.",
                request_id=request_id,
                allow_methods=DOMAIN_EXPERIENCE_METHODS,
                allow_origin=allow_origin,
                details={"available_paths": [self._path]},
            )

        if method == "OPTIONS":
            return empty_response(
                start_response,
                HTTPStatus.NO_CONTENT,
                request_id=request_id,
                allow_methods=DOMAIN_EXPERIENCE_METHODS,
                allow_origin=allow_origin,
                extra_headers=[
                    (DOMAIN_EXPERIENCE_CONTRACT_HEADER, DOMAIN_EXPERIENCE_CONTRACT_VERSION)
                ],
            )

        if method != "GET":
            return error_response(
                start_response,
                HTTPStatus.METHOD_NOT_ALLOWED,
                error="method_not_allowed",
                message="Domain experience accepts GET and OPTIONS.",
                request_id=request_id,
                allow_methods=DOMAIN_EXPERIENCE_METHODS,
                allow_origin=allow_origin,
                details={"allowed_methods": ["GET", "OPTIONS"]},
                extra_headers=[
                    ("Allow", DOMAIN_EXPERIENCE_METHODS),
                    (DOMAIN_EXPERIENCE_CONTRACT_HEADER, DOMAIN_EXPERIENCE_CONTRACT_VERSION),
                ],
            )

        payload = build_domain_experience_payload(
            chroma_sqlite_path=settings.chroma_persist_dir / "chroma.sqlite3"
        )
        return json_response(
            start_response,
            HTTPStatus.OK,
            payload,
            request_id=request_id,
            allow_methods=DOMAIN_EXPERIENCE_METHODS,
            allow_origin=allow_origin,
            extra_headers=[
                (DOMAIN_EXPERIENCE_CONTRACT_HEADER, DOMAIN_EXPERIENCE_CONTRACT_VERSION),
            ],
        )


def create_domain_experience_app(
    *,
    settings_factory: Callable[[], Settings] = load_settings,
) -> DomainExperienceApplication:
    """Create the public-safe domain and KB inventory endpoint."""

    return DomainExperienceApplication(settings_factory=settings_factory)


def build_domain_experience_payload(
    *,
    chroma_sqlite_path: Path | str,
) -> dict[str, object]:
    """Build a read-only catalog payload without exposing source content or prompts."""

    records = domain_experience_records()
    try:
        inventory = inventory_local_chroma_kb(chroma_sqlite_path)
        inventory_error: str | None = None
    except (OSError, sqlite3.Error):
        inventory = None
        inventory_error = "Local KB inventory could not be read."

    indexed_source_counts = inventory.source_chunk_counts if inventory else {}
    kb_reality = build_kb_reality_snapshot(
        indexed_chunk_counts_by_source=indexed_source_counts
    )
    registered_source_count = kb_reality.registry_source_count
    registered_domain_count = kb_reality.registry_domain_count
    indexed_source_count = kb_reality.indexed_source_count
    indexed_chunk_count = kb_reality.indexed_chunk_count
    indexed_domain_count = kb_reality.indexed_domain_count
    knowledge_status = _knowledge_status(
        inventory_exists=bool(inventory and inventory.chroma_exists),
        indexed_chunk_count=indexed_chunk_count,
        inventory_error=inventory_error,
    )

    return {
        "schemaVersion": DOMAIN_EXPERIENCE_CONTRACT_VERSION,
        "domains": [
            {
                **record.model_dump(mode="json"),
                "knowledge": {
                    "registeredSourceCount": len(record.knowledge_source_ids),
                    "indexedSourceCount": sum(
                        1
                        for source_id in record.knowledge_source_ids
                        if indexed_source_counts.get(source_id, 0) > 0
                    ),
                    "indexedChunkCount": sum(
                        indexed_source_counts.get(source_id, 0)
                        for source_id in record.knowledge_source_ids
                    ),
                },
            }
            for record in records
        ],
        "knowledgeBase": {
            "status": knowledge_status,
            "detail": _knowledge_detail(
                knowledge_status=knowledge_status,
                indexed_chunk_count=indexed_chunk_count,
                inventory_error=inventory_error,
            ),
            "registeredSourceCount": registered_source_count,
            "registeredDomainCount": registered_domain_count,
            "indexedSourceCount": indexed_source_count,
            "indexedDomainCount": indexed_domain_count,
            "indexedChunkCount": indexed_chunk_count,
            "lastIndexedAt": inventory.fetched_at_max if inventory else None,
            "freshnessStatus": (
                "local_index_timestamp"
                if inventory and inventory.fetched_at_max
                else "not_reported"
            ),
            "freshnessDetail": (
                "Last indexed is the newest local Chroma record timestamp. It does "
                "not verify that an upstream official page is currently unchanged."
                if inventory and inventory.fetched_at_max
                else "No local index timestamp is available, so upstream-source "
                "freshness is not reported."
            ),
            "updateStatus": "explicit_selected_source_actions",
            "updateHint": (
                "Select official sources in the dashboard to check, validate, update, "
                "or rebuild the local index. Updates require explicit confirmation and "
                "never send private workspace data."
            ),
            "provenanceBoundary": (
                "Counts come from registered official-source metadata and the local "
                "Chroma index only. Retrieved source text, private prompts, and "
                "memory are not returned here."
            ),
            "sources": [
                {
                    "id": source.source_id,
                    "title": source.title,
                    "publisher": source.publisher,
                    "url": source.url,
                    "domain": source.domain.value,
                    "sourceType": source.source_type.value,
                    "priority": source.priority,
                    "tags": list(source.tags),
                    "indexed": indexed_source_counts.get(source.source_id, 0) > 0,
                    "chunkCount": indexed_source_counts.get(source.source_id, 0),
                    "lastIndexedAt": (
                        inventory.source_last_indexed_at.get(source.source_id)
                        if inventory
                        else None
                    ),
                    "fingerprint": (
                        inventory.source_content_hashes.get(source.source_id)
                        if inventory
                        else None
                    ),
                    "health": _source_health(
                        indexed=indexed_source_counts.get(source.source_id, 0) > 0,
                        last_indexed_at=(
                            inventory.source_last_indexed_at.get(source.source_id)
                            if inventory
                            else None
                        ),
                    ),
                    "freshnessLimitation": (
                        "Local index timestamps show when this source was last stored; "
                        "they do not confirm the current upstream document version."
                    ),
                    "provenance": "Approved official-source registry and local Chroma index.",
                }
                for source in approved_official_sources()
            ],
        },
    }


def _knowledge_status(
    *,
    inventory_exists: bool,
    indexed_chunk_count: int,
    inventory_error: str | None,
) -> str:
    if inventory_error:
        return "unavailable"
    if indexed_chunk_count > 0:
        return "available"
    if inventory_exists:
        return "empty"
    return "not_initialized"


def _source_health(*, indexed: bool, last_indexed_at: str | None) -> str:
    if not indexed:
        return "registered_only"
    if not last_indexed_at:
        return "indexed_without_timestamp"
    return "locally_indexed"


def _knowledge_detail(
    *,
    knowledge_status: str,
    indexed_chunk_count: int,
    inventory_error: str | None,
) -> str:
    if inventory_error:
        return inventory_error
    if knowledge_status == "available":
        return f"The local Chroma index currently reports {indexed_chunk_count} official knowledge chunks."
    if knowledge_status == "empty":
        return "Local Chroma is present but no official-source chunks are indexed yet."
    return "No local Chroma inventory is initialized yet; registered sources are not the same as indexed knowledge."
