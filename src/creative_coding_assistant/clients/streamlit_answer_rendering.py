"""Pure helpers for rendering assistant answers in Streamlit."""

from __future__ import annotations

import re
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
    """Split assistant markdown into prose and code segments."""

    if not text.strip():
        return ()

    segments: list[AnswerSegment] = []
    for segment in _split_fenced_answer_segments(
        text,
        allow_unclosed_code_block=allow_unclosed_code_block,
    ):
        if segment.kind == "code":
            segments.append(segment)
            continue
        segments.extend(_split_unfenced_code_segments(segment.content))
    return tuple(segments)


def _split_fenced_answer_segments(
    text: str,
    *,
    allow_unclosed_code_block: bool,
) -> tuple[AnswerSegment, ...]:
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


def _split_unfenced_code_segments(text: str) -> tuple[AnswerSegment, ...]:
    if not text.strip():
        return ()

    lines = text.splitlines()
    prose_buffer: list[str] = []
    segments: list[AnswerSegment] = []
    index = 0

    while index < len(lines):
        detected = _detect_unfenced_code_block(lines, index)
        if detected is None:
            prose_buffer.append(lines[index])
            index += 1
            continue

        end_index, language = detected
        prose_content = _normalize_prose_lines(prose_buffer)
        if prose_content:
            segments.append(AnswerSegment(kind="prose", content=prose_content))
        prose_buffer = []

        code_content = "\n".join(lines[index:end_index]).strip()
        if code_content:
            segments.append(
                AnswerSegment(
                    kind="code",
                    content=code_content,
                    language=language,
                )
            )
        index = end_index

    prose_content = _normalize_prose_lines(prose_buffer)
    if prose_content:
        segments.append(AnswerSegment(kind="prose", content=prose_content))
    return tuple(segments)


def _detect_unfenced_code_block(
    lines: list[str],
    start_index: int,
) -> tuple[int, str] | None:
    first_line = lines[start_index].strip()
    if not first_line or _looks_like_list_or_heading(first_line):
        return None
    if not _looks_like_codeish_line(first_line):
        return None

    end_index = start_index
    while end_index < len(lines):
        current_line = lines[end_index].strip()
        if not current_line:
            next_non_empty = _next_non_empty_line(lines, end_index + 1)
            if next_non_empty is not None and _looks_like_codeish_line(next_non_empty):
                end_index += 1
                continue
            break

        if not _looks_like_codeish_line(current_line):
            break
        end_index += 1

    candidate_lines = lines[start_index:end_index]
    language = _detect_unfenced_code_language(candidate_lines)
    if language is None:
        return None
    return end_index, language


def _detect_unfenced_code_language(lines: list[str]) -> str | None:
    content = "\n".join(line.strip() for line in lines if line.strip())
    if not content:
        return None

    significant_lines = [line for line in lines if line.strip()]
    if len(significant_lines) < 3:
        return None

    if _looks_like_html_document(content):
        return "html"
    if _looks_like_glsl_block(content):
        return "glsl"
    if _looks_like_jsx_block(content):
        return "jsx"
    if _looks_like_python_block(significant_lines):
        return "python"
    if _looks_like_javascript_block(content, significant_lines):
        return "javascript"
    return None


def _looks_like_codeish_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or _looks_like_list_or_heading(stripped):
        return False
    if re.match(r"^</?[A-Za-z][^>]*>$", stripped) or stripped.startswith("<!DOCTYPE"):
        return True
    if stripped.startswith(
        ("const ", "let ", "var ", "function ", "class ", "import ")
    ):
        return True
    if stripped.startswith(("export ", "return ", "renderer.", "scene.", "camera.")):
        return True
    if stripped.startswith(("precision ", "uniform ", "varying ", "void main")):
        return True
    if stripped.startswith(("def ", "from ", "print(", "if __name__ == ")):
        return True
    if any(
        marker in stripped
        for marker in (
            "THREE.",
            "requestAnimationFrame(",
            "document.",
            "window.",
            "new THREE.",
            "addEventListener(",
            "gl_FragColor",
            "gl_Position",
            "from '@react-three/fiber'",
            "from \"@react-three/fiber\"",
            "useFrame(",
            "useRef(",
            "<Canvas",
        )
    ):
        return True
    return stripped.endswith(("{", "}", ";", ">", ");"))


def _looks_like_list_or_heading(line: str) -> bool:
    return bool(
        re.match(r"^([-*]|\d+\.)\s", line)
        or line.startswith("#")
    )


def _next_non_empty_line(lines: list[str], start_index: int) -> str | None:
    for index in range(start_index, len(lines)):
        if lines[index].strip():
            return lines[index].strip()
    return None


def _looks_like_html_document(content: str) -> bool:
    lowered = content.lower()
    return (
        "<!doctype html" in lowered
        or "<html" in lowered
        or (
            sum(tag in lowered for tag in ("<head", "<body", "<script", "</html>")) >= 2
            and "<" in lowered
            and ">" in lowered
        )
    )


def _looks_like_glsl_block(content: str) -> bool:
    return (
        "void main" in content
        and any(
            marker in content
            for marker in ("gl_FragColor", "gl_Position", "precision ", "uniform ")
        )
    )


def _looks_like_jsx_block(content: str) -> bool:
    return any(
        marker in content
        for marker in (
            "from '@react-three/fiber'",
            'from "@react-three/fiber"',
            "<Canvas",
            "useFrame(",
            "useRef(",
        )
    )


def _looks_like_python_block(lines: list[str]) -> bool:
    return any(line.lstrip().startswith("def ") for line in lines) and any(
        line.startswith(("    ", "\t")) for line in lines[1:]
    )


def _looks_like_javascript_block(content: str, lines: list[str]) -> bool:
    if len(lines) < 3:
        return False
    strong_signal_count = sum(
        1
        for line in lines
        if any(
            marker in line
            for marker in (
                "THREE.",
                "requestAnimationFrame(",
                "document.",
                "window.",
                "new THREE.",
                "addEventListener(",
                "import ",
                "const ",
                "let ",
                "var ",
                "function ",
                "export ",
                "=>",
            )
        )
    )
    return strong_signal_count >= 2 or (
        strong_signal_count >= 1 and content.count(";") >= 2
    )


def _normalize_prose_lines(lines: list[str]) -> str:
    return "\n".join(lines).strip()


def answer_working_message(
    *,
    status_message: str | None,
    has_content: bool,
) -> str | None:
    """Return a UI-safe answer-area hint while generation is in flight."""

    if has_content:
        return None
    if status_message in {"Preparing response...", "Receiving response..."}:
        return status_message
    return None
