"""Fetch boundaries for approved official knowledge-base sources."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.rag.sources import OfficialSource, get_official_source
from creative_coding_assistant.rag.sync.models import (
    FetchedSourceDocument,
    OfficialSourceSyncRequest,
    SourceContentFormat,
)


class TransportResponse(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    resolved_url: str = Field(min_length=1)
    status_code: int = Field(ge=100, le=599)
    content_type: str = Field(min_length=1)
    content: str = Field(min_length=1)


class SourceTransport(Protocol):
    def fetch(self, url: str) -> TransportResponse:
        """Fetch source content for a single approved URL."""


class UrllibSourceTransport:
    """Simple HTTP transport kept behind a narrow local interface."""

    _REQUEST_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "text/plain;q=0.8,*/*;q=0.5"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    def fetch(self, url: str) -> TransportResponse:
        try:
            request = Request(url, headers=self._REQUEST_HEADERS)
            with urlopen(request) as response:  # noqa: S310
                charset = response.headers.get_content_charset() or "utf-8"
                content = response.read().decode(charset, errors="replace")
                return TransportResponse(
                    resolved_url=response.geturl(),
                    status_code=response.status,
                    content_type=response.headers.get_content_type(),
                    content=content,
                )
        except HTTPError as error:
            raise RuntimeError(
                f"Official source fetch failed with status {error.code}: {url}"
            ) from error
        except URLError as error:
            raise RuntimeError(f"Official source fetch failed: {url}") from error


class OfficialSourceFetcher:
    """Fetch approved official sources without letting callers bypass the registry."""

    def __init__(self, *, transport: SourceTransport) -> None:
        self._transport = transport

    def fetch(
        self,
        request: OfficialSourceSyncRequest,
    ) -> FetchedSourceDocument:
        source = get_official_source(request.source_id)
        response = self._transport.fetch(source.url)
        self._validate_response(source, response)
        content_format = self._resolve_content_format(response.content_type)

        document = FetchedSourceDocument.from_content(
            source_id=source.source_id,
            domain=source.domain,
            source_type=source.source_type,
            registry_title=source.title,
            publisher=source.publisher,
            source_url=source.url,
            resolved_url=response.resolved_url,
            fetched_at=request.requested_at,
            content_format=content_format,
            raw_content=response.content,
        )
        logger.info(
            "Fetched official KB source '{}' from {}",
            source.source_id,
            response.resolved_url,
        )
        return document

    def _validate_response(
        self,
        source: OfficialSource,
        response: TransportResponse,
    ) -> None:
        if response.status_code < 200 or response.status_code >= 300:
            raise ValueError("Official source fetch must return a successful status.")

        parsed = urlparse(response.resolved_url)
        source_host = urlparse(source.url).hostname
        path_is_allowed = any(
            parsed.path.startswith(prefix) for prefix in source.allowed_path_prefixes
        )
        if parsed.scheme != "https" or parsed.hostname != source_host:
            raise ValueError("Fetched source resolved outside the approved host scope.")
        if not path_is_allowed:
            raise ValueError("Fetched source resolved outside the approved path scope.")

    def _resolve_content_format(self, content_type: str) -> SourceContentFormat:
        normalized = content_type.split(";", maxsplit=1)[0].strip().lower()
        if normalized in {"text/html", "application/xhtml+xml"}:
            return SourceContentFormat.HTML
        if normalized == "text/plain":
            return SourceContentFormat.TEXT
        raise ValueError(f"Unsupported official source content type: {content_type}")


def default_sync_request(source_id: str) -> OfficialSourceSyncRequest:
    return OfficialSourceSyncRequest(
        source_id=source_id,
        requested_at=datetime.now(tz=UTC),
    )
