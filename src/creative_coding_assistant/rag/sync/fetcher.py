"""Fetch boundaries for approved official knowledge-base sources."""

from __future__ import annotations

import html
import re
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
        responses = tuple(
            self._fetch_response(source, url) for url in source_urls(source)
        )
        content_format = self._resolve_content_format(
            tuple(response.content_type for response in responses)
        )
        response = responses[0]
        raw_content = self._build_raw_content(source=source, responses=responses)

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
            raw_content=raw_content,
        )
        logger.info(
            "Fetched official KB source '{}' from {} page(s)",
            source.source_id,
            len(responses),
        )
        return document

    def _fetch_response(
        self,
        source: OfficialSource,
        url: str,
    ) -> TransportResponse:
        response = self._transport.fetch(url)
        self._validate_response(source, response)
        return response

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

    def _resolve_content_format(
        self,
        content_types: tuple[str, ...],
    ) -> SourceContentFormat:
        normalized_content_types = tuple(
            content_type.split(";", maxsplit=1)[0].strip().lower()
            for content_type in content_types
        )
        if all(
            content_type in {"text/html", "application/xhtml+xml"}
            for content_type in normalized_content_types
        ):
            return SourceContentFormat.HTML
        if all(
            content_type == "text/plain"
            for content_type in normalized_content_types
        ):
            return SourceContentFormat.TEXT
        values = ", ".join(content_types)
        raise ValueError(f"Unsupported official source content type(s): {values}")

    def _build_raw_content(
        self,
        *,
        source: OfficialSource,
        responses: tuple[TransportResponse, ...],
    ) -> str:
        if len(responses) == 1:
            return responses[0].content

        sections = []
        for response in responses:
            title = _extract_html_title(response.content) or response.resolved_url
            body = _extract_html_body(response.content)
            sections.append(
                "\n".join(
                    (
                        "<section>",
                        f"<h1>Page: {html.escape(title)}</h1>",
                        body,
                        "</section>",
                    )
                )
            )

        return "\n".join(
            (
                "<html>",
                "<head>",
                f"<title>{html.escape(source.title)}</title>",
                "</head>",
                "<body>",
                *sections,
                "</body>",
                "</html>",
            )
        )


def source_urls(source: OfficialSource) -> tuple[str, ...]:
    return (source.url, *source.additional_urls)


def _extract_html_title(content: str) -> str:
    match = re.search(
        r"<title[^>]*>(.*?)</title>",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""
    return " ".join(match.group(1).split())


def _extract_html_body(content: str) -> str:
    match = re.search(
        r"<body[^>]*>(.*)</body>",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1)
    return content


def default_sync_request(source_id: str) -> OfficialSourceSyncRequest:
    return OfficialSourceSyncRequest(
        source_id=source_id,
        requested_at=datetime.now(tz=UTC),
    )
