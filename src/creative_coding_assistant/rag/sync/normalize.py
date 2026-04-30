"""Normalization boundaries for fetched official source content."""

from __future__ import annotations

from html.parser import HTMLParser

from loguru import logger

from creative_coding_assistant.rag.sync.models import (
    FetchedSourceDocument,
    NormalizedSourceDocument,
    SourceContentFormat,
)

_BLOCK_TAGS = {
    "article",
    "aside",
    "blockquote",
    "br",
    "div",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}
_CODE_TAGS = {"pre"}
_SKIPPED_TAGS = {"aside", "footer", "nav", "noscript", "script", "style"}


class _HtmlTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._text_parts: list[str] = []
        self._title_parts: list[str] = []
        self._skip_depth = 0
        self._code_depth = 0
        self._in_title = False

    @property
    def document_title(self) -> str:
        return " ".join(part.strip() for part in self._title_parts if part.strip())

    @property
    def text_content(self) -> str:
        return "".join(self._text_parts)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in _SKIPPED_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag in _CODE_TAGS:
            self._code_depth += 1
            self._text_parts.append("\n```text\n")
        if tag in _BLOCK_TAGS:
            self._text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
        if tag in _CODE_TAGS and self._code_depth > 0:
            self._code_depth -= 1
            self._text_parts.append("\n```\n")
        if tag in _BLOCK_TAGS:
            self._text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title:
            self._title_parts.append(data)
        self._text_parts.append(data)


class OfficialSourceNormalizer:
    """Turn fetched source content into normalized plain-text documents."""

    def normalize(
        self,
        document: FetchedSourceDocument,
    ) -> NormalizedSourceDocument:
        if document.content_format == SourceContentFormat.HTML:
            title, text = self._normalize_html(document.raw_content)
        else:
            title = document.registry_title
            text = self._normalize_text(document.raw_content)

        if not text:
            raise ValueError("Official source normalization produced empty content.")

        normalized = NormalizedSourceDocument.from_text(
            fetched_document=document,
            document_title=title or document.registry_title,
            normalized_text=text,
        )
        logger.info(
            "Normalized official KB source '{}' into {} characters",
            document.source_id,
            len(normalized.normalized_text),
        )
        return normalized

    def _normalize_html(self, raw_content: str) -> tuple[str, str]:
        extractor = _HtmlTextExtractor()
        extractor.feed(raw_content)
        title = self._normalize_text(extractor.document_title)
        text = self._normalize_text(extractor.text_content)
        return title, text

    def _normalize_text(self, raw_content: str) -> str:
        paragraphs: list[str] = []
        current_lines: list[str] = []
        for raw_line in raw_content.replace("\xa0", " ").splitlines():
            if raw_line.strip().startswith("```"):
                normalized_line = raw_line.strip()
            else:
                normalized_line = " ".join(raw_line.split())
            if normalized_line:
                current_lines.append(normalized_line)
                continue
            if current_lines:
                paragraphs.append(" ".join(current_lines))
                current_lines = []

        if current_lines:
            paragraphs.append(" ".join(current_lines))
        return "\n\n".join(paragraphs).strip()
