"""Pure helpers for rendering assistant answers in Streamlit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AnswerSegment:
    kind: Literal["prose", "code"]
    content: str
    language: str | None = None


def split_answer_segments(
    text: str,
    *,
    allow_unclosed_code_block: bool = False,
) -> tuple[AnswerSegment, ...]:
    """Split assistant markdown into prose and fenced code segments."""

    if not text.strip():
        return ()

    segments: list[AnswerSegment] = []
    buffer: list[str] = []
    in_code = False
    code_language: str | None = None

    for raw_line in text.splitlines():
        if raw_line.startswith("```"):
            if in_code:
                code_content = "\n".join(buffer).rstrip()
                if code_content:
                    segments.append(
                        AnswerSegment(
                            kind="code",
                            content=code_content,
                            language=code_language,
                        )
                    )
                buffer = []
                in_code = False
                code_language = None
                continue

            prose_content = "\n".join(buffer).strip()
            if prose_content:
                segments.append(AnswerSegment(kind="prose", content=prose_content))
            buffer = []
            in_code = True
            code_language = raw_line[3:].strip() or None
            continue

        buffer.append(raw_line)

    trailing_content = "\n".join(buffer).strip()
    if in_code:
        if allow_unclosed_code_block and trailing_content:
            segments.append(
                AnswerSegment(
                    kind="code",
                    content=trailing_content,
                    language=code_language,
                )
            )
        elif trailing_content:
            segments.append(AnswerSegment(kind="prose", content=trailing_content))
    elif trailing_content:
        segments.append(AnswerSegment(kind="prose", content=trailing_content))

    return tuple(segments)


def answer_working_message(
    *,
    status_message: str | None,
    has_content: bool,
) -> str | None:
    """Return a UI-safe answer-area hint while generation is in flight."""

    if has_content:
        return None
    if status_message == "Generating response...":
        return "Waiting for model output..."
    if status_message == "Streaming response...":
        return "Receiving output..."
    return None
