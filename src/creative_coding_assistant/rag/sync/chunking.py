"""Chunking boundaries for normalized official knowledge-base documents."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.rag.sync.models import (
    NormalizedSourceDocument,
    OfficialSourceChunk,
)

_DROPPED_SECTION_PREFIXES = (
    "bibliography",
    "external links",
    "image:",
    "next step",
    "previous step",
    "related examples",
    "related references",
    "resources",
    "retrieved from",
    "see also",
)
_NAVIGATION_PHRASES = (
    "edited and maintained by",
    "licensed under cc by",
    "menu reference tutorials",
    "skip to main content",
    "you can find the code history",
)
_P5_CODE_MARKERS = (
    "function draw",
    "function setup",
    "background(",
    "circle(",
    "createcanvas(",
    "draw()",
    "ellipse(",
    "fill(",
    "framecount",
    "mousex",
    "mousey",
    "random(",
    "setup()",
    "stroke(",
)
_P5_TECHNICAL_MARKERS = _P5_CODE_MARKERS + (
    "animation",
    "canvas",
    "interactive",
    "sketch",
)
_GLSL_CODE_MARKERS = (
    "#version",
    "fragcolor",
    "gl_fragcolor",
    "gl_position",
    "precision ",
    "sampler2d",
    "uniform ",
    "varying ",
    "vec2",
    "vec3",
    "vec4",
    "void main",
)
_GLSL_TECHNICAL_MARKERS = _GLSL_CODE_MARKERS + (
    "fragment shader",
    "shader stage",
    "type qualifier",
    "uniform",
    "vertex shader",
)


class ChunkingPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_chars: int = Field(default=1200, ge=200)
    min_chunk_chars: int = Field(default=300, ge=50)


class OfficialSourceChunker:
    """Build deterministic text chunks without introducing retrieval logic."""

    def __init__(self, *, policy: ChunkingPolicy | None = None) -> None:
        self._policy = policy or ChunkingPolicy()

    def chunk(
        self,
        document: NormalizedSourceDocument,
    ) -> tuple[OfficialSourceChunk, ...]:
        content_parts = self._prepare_content_parts(document)
        if not content_parts:
            return ()

        chunk_texts: list[str] = []
        current_parts: list[str] = []
        current_size = 0

        for paragraph in content_parts:
            for piece in self._split_paragraph(paragraph):
                if self._should_break_before(piece) and current_parts:
                    chunk_texts.append("\n\n".join(current_parts))
                    current_parts = []
                    current_size = 0

                if not current_parts:
                    current_parts = [piece]
                    current_size = len(piece)
                    continue

                projected_size = current_size + 2 + len(piece)
                if projected_size <= self._policy.max_chars:
                    current_parts.append(piece)
                    current_size = projected_size
                    continue

                chunk_texts.append("\n\n".join(current_parts))
                current_parts = [piece]
                current_size = len(piece)

        if current_parts:
            chunk_texts.append("\n\n".join(current_parts))

        merged_texts = tuple(
            text
            for text in self._merge_small_tail(chunk_texts)
            if self._should_keep_chunk(document, text)
        )
        return tuple(
            OfficialSourceChunk.from_text(
                normalized_document=document,
                chunk_index=index,
                text=text,
            )
            for index, text in enumerate(merged_texts)
        )

    def _prepare_content_parts(
        self,
        document: NormalizedSourceDocument,
    ) -> tuple[str, ...]:
        paragraphs = [part.strip() for part in document.normalized_text.split("\n\n")]
        return tuple(
            paragraph
            for paragraph in paragraphs
            if paragraph and not self._should_drop_paragraph(document, paragraph)
        )

    def _split_paragraph(self, paragraph: str) -> tuple[str, ...]:
        if len(paragraph) <= self._policy.max_chars:
            return (paragraph,)

        chunks: list[str] = []
        current_words: list[str] = []
        current_length = 0

        for word in paragraph.split():
            if len(word) > self._policy.max_chars:
                if current_words:
                    chunks.append(" ".join(current_words))
                    current_words = []
                    current_length = 0
                chunks.extend(self._split_long_word(word))
                continue

            projected_length = current_length + (1 if current_words else 0) + len(word)
            if projected_length <= self._policy.max_chars:
                current_words.append(word)
                current_length = projected_length
                continue

            chunks.append(" ".join(current_words))
            current_words = [word]
            current_length = len(word)

        if current_words:
            chunks.append(" ".join(current_words))

        return tuple(chunks)

    def _split_long_word(self, word: str) -> tuple[str, ...]:
        step = self._policy.max_chars
        return tuple(word[index : index + step] for index in range(0, len(word), step))

    def _merge_small_tail(self, chunk_texts: list[str]) -> tuple[str, ...]:
        if len(chunk_texts) < 2:
            return tuple(chunk_texts)

        tail = chunk_texts[-1]
        if len(tail) >= self._policy.min_chunk_chars:
            return tuple(chunk_texts)

        merged = f"{chunk_texts[-2]}\n\n{tail}"
        if len(merged) <= self._policy.max_chars:
            return tuple([*chunk_texts[:-2], merged])
        return tuple(chunk_texts)

    def _should_break_before(self, piece: str) -> bool:
        return piece.startswith("Page: ")

    def _should_drop_paragraph(
        self,
        document: NormalizedSourceDocument,
        paragraph: str,
    ) -> bool:
        normalized = " ".join(paragraph.lower().split())
        if any(normalized.startswith(prefix) for prefix in _DROPPED_SECTION_PREFIXES):
            return True
        if any(phrase in normalized for phrase in _NAVIGATION_PHRASES):
            return True
        if (
            document.domain == CreativeCodingDomain.GLSL
            and len(normalized) > 900
            and not self._contains_any(normalized, _GLSL_TECHNICAL_MARKERS)
        ):
            return True
        return False

    def _should_keep_chunk(
        self,
        document: NormalizedSourceDocument,
        text: str,
    ) -> bool:
        normalized = " ".join(text.lower().split())
        if not normalized or any(
            phrase in normalized for phrase in _NAVIGATION_PHRASES
        ):
            return False
        if document.domain == CreativeCodingDomain.P5_JS:
            return self._contains_any(normalized, _P5_TECHNICAL_MARKERS)
        if document.domain == CreativeCodingDomain.GLSL:
            return self._contains_any(normalized, _GLSL_TECHNICAL_MARKERS)
        return True

    def _contains_any(self, text: str, markers: tuple[str, ...]) -> bool:
        return any(marker in text for marker in markers)
