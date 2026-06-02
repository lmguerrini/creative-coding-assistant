"""Workflow-owned artifact extraction and preview preparation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field

from creative_coding_assistant.artifacts import (
    ArtifactCategory,
    ArtifactContentLocator,
    ArtifactContentReference,
    ArtifactIdentity,
    ArtifactMetadata,
    ArtifactOrigin,
    ArtifactProvenance,
    ArtifactRecord,
    ArtifactStatus,
    ArtifactTimestamps,
    ArtifactType,
    ArtifactWorkflowLink,
)
from creative_coding_assistant.contracts import AssistantRequest, CreativeCodingDomain
from creative_coding_assistant.orchestration.routing import RouteDecision
from creative_coding_assistant.preview import (
    PreviewProvenance,
    PreviewRequest,
    PreviewResult,
    PreviewTarget,
)


class ArtifactCritiqueDimension(BaseModel):
    """One bounded, deterministic artifact critique dimension."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    score: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)


class WorkflowArtifactCritique(BaseModel):
    """Per-artifact critique metadata used for ranking and refinement guidance."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    artifact_id: str = Field(min_length=1)
    artifact_title: str = Field(min_length=1)
    source_order: int = Field(ge=1)
    overall_score: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)
    passed: bool
    recommended: bool = False
    prompt_alignment: ArtifactCritiqueDimension
    creative_quality: ArtifactCritiqueDimension
    runtime_suitability: ArtifactCritiqueDimension
    code_quality: ArtifactCritiqueDimension
    preview_readiness: ArtifactCritiqueDimension
    domain_appropriateness: ArtifactCritiqueDimension
    reasons: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)
    refinement_guidance: str | None = None


class WorkflowArtifact(BaseModel):
    """Structured artifact extracted from a generation result."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    name: str = Field(min_length=1)
    type: str = Field(default=ArtifactType.CODE.value, min_length=1)
    language: str = Field(min_length=1)
    source_language: str = Field(min_length=1)
    content: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    status: str = "Generated"
    source: str = "generation"
    source_order: int = Field(ge=1)
    domain: str | None = None
    is_creative: bool = False
    is_default: bool = False
    preview_eligible: bool = False
    runtime: str | None = None
    renderer_id: str | None = None
    preview_target: str | None = None
    content_hash: str = Field(min_length=1)
    critique: WorkflowArtifactCritique | None = None
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    quality_rank: int | None = Field(default=None, ge=1)
    is_recommended: bool = False
    refinement_reason: str | None = None


@dataclass(frozen=True)
class _CodeBlock:
    content: str
    language: str
    title: str | None


_FENCE_PATTERN = re.compile(r"```([^\n`]*)\n([\s\S]*?)```")
_RUNTIME_RENDERERS = {
    "glsl": "surface.glsl",
    "hydra": "surface.hydra",
    "p5": "surface.p5",
    "three": "surface.three",
}
_RUNTIME_LABELS = {
    "glsl": "GLSL",
    "hydra": "Hydra",
    "p5": "p5.js",
    "three": "Three.js",
}
_BROWSER_SOURCE_LANGUAGES = frozenset(
    {"javascript", "typescript", "html", "css", "glsl", "wgsl"}
)


def extract_workflow_artifacts(
    answer: str,
    *,
    request: AssistantRequest,
    route_decision: RouteDecision,
) -> tuple[WorkflowArtifact, ...]:
    """Extract code artifacts from generated assistant output."""

    blocks = _parse_markdown_code_blocks(answer)
    artifacts: list[WorkflowArtifact] = []
    for index, block in enumerate(blocks, start=1):
        artifact = _build_workflow_artifact(
            block,
            index=index,
            request=request,
            route_decision=route_decision,
        )
        if artifact is not None:
            artifacts.append(artifact)
    return _finalize_workflow_artifacts(tuple(artifacts))


def prepare_workflow_preview_results(
    artifacts: tuple[WorkflowArtifact, ...],
    *,
    request: AssistantRequest,
    route_decision: RouteDecision,
) -> tuple[PreviewResult, ...]:
    """Prepare preview-ready results for creative workflow artifacts."""

    completed_at = datetime.now(UTC)
    results: list[PreviewResult] = []
    for artifact in artifacts:
        if artifact.preview_target != PreviewTarget.BROWSER_SANDBOX.value:
            continue
        artifact_record = _to_artifact_record(
            artifact,
            request=request,
            route_decision=route_decision,
            timestamp=completed_at,
        )
        workflow_link = artifact_record.workflow_link
        preview_request = PreviewRequest.from_artifact(
            artifact_record,
            request_id=_prefixed_identifier("preview-request", artifact.id),
            preview_id=_prefixed_identifier("preview", artifact.id),
            requested_at=completed_at,
            target=PreviewTarget.BROWSER_SANDBOX,
            preferred_renderer_id=artifact.renderer_id,
            workflow_link=workflow_link,
        )
        renderer_label = _RUNTIME_LABELS.get(artifact.runtime or "", "browser")
        results.append(
            PreviewResult.succeeded(
                request=preview_request,
                preview_artifact_id=artifact.id,
                summary=(
                    f"{renderer_label} runtime metadata prepared for "
                    f"{artifact.title}."
                ),
                completed_at=completed_at,
                details={
                    "artifact": artifact.model_dump(mode="json"),
                    "runtime": {
                        "kind": artifact.runtime,
                        "renderer_id": artifact.renderer_id,
                        "target": artifact.preview_target,
                    },
                },
                provenance=PreviewProvenance(
                    renderer_id=artifact.renderer_id,
                    workflow_link=workflow_link,
                ),
            )
        )
    return tuple(results)


def _parse_markdown_code_blocks(answer: str) -> tuple[_CodeBlock, ...]:
    blocks: list[_CodeBlock] = []
    for match in _FENCE_PATTERN.finditer(answer):
        info = _parse_fence_info(match.group(1) or "")
        content = _trim_code_block(match.group(2) or "")
        if not content.strip():
            continue
        blocks.append(
            _CodeBlock(
                content=content,
                language=info["language"],
                title=info["title"],
            )
        )
    return tuple(blocks)


def _parse_fence_info(info: str) -> dict[str, str | None]:
    tokens = [token for token in info.strip().split() if token]
    language = _normalize_language(tokens[0] if tokens else "")
    named_title = next(
        (
            token.split("=", 1)[1]
            for token in tokens
            if re.match(r"^(?:file|filename|name)=", token, flags=re.I)
        ),
        None,
    )
    title_token = next((token for token in tokens if "." in token), None)
    return {
        "language": language,
        "title": _sanitize_filename(named_title or title_token or "") or None,
    }


def _build_workflow_artifact(
    block: _CodeBlock,
    *,
    index: int,
    request: AssistantRequest,
    route_decision: RouteDecision,
) -> WorkflowArtifact | None:
    language = block.language or _infer_language(block.content, block.title)
    if not language:
        return None
    runtime = _infer_runtime(block.content, language, block.title)
    title = block.title or _default_artifact_title(language, runtime, index)
    artifact_id = _sanitize_identifier(title) or f"generated-artifact-{index}"
    content_hash = sha256(block.content.encode("utf-8")).hexdigest()
    renderer_id = _RUNTIME_RENDERERS.get(runtime or "")
    preview_target = (
        PreviewTarget.BROWSER_SANDBOX.value
        if runtime is not None or language in _BROWSER_SOURCE_LANGUAGES
        else None
    )
    domain_label = _domain_label(route_decision.domain or request.domain)
    domain_value = _domain_value(route_decision.domain or request.domain)
    runtime_label = _RUNTIME_LABELS.get(runtime or "")
    summary_parts = ["Extracted from the generation result"]
    if runtime_label:
        summary_parts.append(f"matched {runtime_label} creative runtime")
    if domain_label:
        summary_parts.append(f"for {domain_label}")

    return WorkflowArtifact(
        id=artifact_id,
        title=title,
        name=title,
        language=_format_language_label(language, runtime, title),
        source_language=language,
        content=block.content,
        summary="; ".join(summary_parts) + ".",
        source_order=index,
        domain=domain_value,
        is_creative=runtime is not None,
        preview_eligible=preview_target is not None,
        runtime=runtime,
        renderer_id=renderer_id,
        preview_target=preview_target,
        content_hash=content_hash,
    )


def _finalize_workflow_artifacts(
    artifacts: tuple[WorkflowArtifact, ...],
) -> tuple[WorkflowArtifact, ...]:
    if not artifacts:
        return ()

    default_index = next(
        (
            index
            for index, artifact in enumerate(artifacts)
            if artifact.preview_eligible
        ),
        0,
    )
    used_ids: set[str] = set()
    finalized: list[WorkflowArtifact] = []
    for index, artifact in enumerate(artifacts, start=1):
        artifact_id = _unique_artifact_id(
            artifact.id or f"generated-artifact-{index}",
            used_ids=used_ids,
            source_order=index,
        )
        used_ids.add(artifact_id)
        finalized.append(
            artifact.model_copy(
                update={
                    "id": artifact_id,
                    "source_order": index,
                    "is_default": index - 1 == default_index,
                }
            )
        )
    return tuple(finalized)


def _unique_artifact_id(
    artifact_id: str,
    *,
    used_ids: set[str],
    source_order: int,
) -> str:
    base_id = _sanitize_identifier(artifact_id) or f"generated-artifact-{source_order}"
    candidate = base_id
    suffix = 2
    while candidate in used_ids:
        candidate = (
            _sanitize_identifier(f"{base_id}-{suffix}")
            or f"generated-artifact-{source_order}-{suffix}"
        )
        suffix += 1
    return candidate


def _to_artifact_record(
    artifact: WorkflowArtifact,
    *,
    request: AssistantRequest,
    route_decision: RouteDecision,
    timestamp: datetime,
) -> ArtifactRecord:
    workspace_id = _sanitize_identifier(request.project_id or "local-workspace")
    workflow_link = ArtifactWorkflowLink.from_request(
        request,
        workflow_run_id=_workflow_run_id(request),
        step="artifact_extraction",
    )
    return ArtifactRecord(
        identity=ArtifactIdentity(
            artifact_id=artifact.id,
            workspace_id=workspace_id,
        ),
        category=ArtifactCategory.GENERATED,
        artifact_type=ArtifactType.CODE,
        status=ArtifactStatus.READY,
        metadata=ArtifactMetadata(
            title=artifact.title,
            summary=artifact.summary,
            tags=_artifact_tags(artifact),
            domain=route_decision.domain or request.domain,
            language=artifact.source_language,
            extra={
                "content_hash": artifact.content_hash,
                "domain": artifact.domain,
                "is_default": artifact.is_default,
                "preview_eligible": artifact.preview_eligible,
                "preview_target": artifact.preview_target,
                "renderer_id": artifact.renderer_id,
                "runtime": artifact.runtime,
                "source_order": artifact.source_order,
            },
        ),
        content_references=(
            ArtifactContentReference(
                reference_id=_prefixed_identifier(artifact.id, "source"),
                locator=ArtifactContentLocator.WORKSPACE_FILE,
                label=artifact.title,
                workspace_path=artifact.title,
                mime_type=_mime_type_for_language(artifact.source_language),
                content_hash=artifact.content_hash,
                byte_size=len(artifact.content.encode("utf-8")),
                is_primary=True,
            ),
        ),
        timestamps=ArtifactTimestamps(created_at=timestamp, updated_at=timestamp),
        provenance=ArtifactProvenance(
            origin=ArtifactOrigin.ASSISTANT_WORKFLOW,
            generator="workflow.artifact_extraction",
        ),
        workflow_link=workflow_link,
    )


def _infer_runtime(
    content: str,
    language: str,
    title: str | None,
) -> str | None:
    haystack = f"{content} {language} {title or ''}".lower()
    normalized_title = (title or "").lower()
    if (
        normalized_title.endswith((".frag", ".glsl", ".fs"))
        or language == "glsl"
        or "gl_fragcolor" in haystack
        or "fragment shader" in haystack
        or "mainimage" in haystack
    ):
        return "glsl"
    if (
        normalized_title.endswith((".hydra.js", ".hydra.ts"))
        or "hydra" in haystack
        or ("osc(" in haystack and "out(" in haystack)
    ):
        return "hydra"
    if (
        normalized_title.endswith((".three.js", ".three.ts", ".r3f.tsx"))
        or "webglrenderer" in haystack
        or "perspectivecamera" in haystack
        or "@react-three/fiber" in haystack
        or "new three." in haystack
    ):
        return "three"
    if (
        normalized_title.endswith((".p5.js", ".p5.ts"))
        or "createcanvas" in haystack
        or ("function setup" in haystack and "function draw" in haystack)
        or "new p5" in haystack
    ):
        return "p5"
    return None


def _infer_language(content: str, title: str | None) -> str:
    normalized_title = (title or "").lower()
    if normalized_title.endswith((".frag", ".glsl", ".fs", ".vs")):
        return "glsl"
    if normalized_title.endswith((".ts", ".tsx")):
        return "typescript"
    if normalized_title.endswith((".js", ".jsx", ".mjs")):
        return "javascript"
    if normalized_title.endswith(".json"):
        return "json"
    if "export " in content or ": " in content.split("\n", 1)[0]:
        return "typescript"
    return "javascript"


def _normalize_language(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "frag": "glsl",
        "fragment": "glsl",
        "fragment-shader": "glsl",
        "js": "javascript",
        "jsx": "javascript",
        "mjs": "javascript",
        "ts": "typescript",
        "tsx": "typescript",
    }
    return aliases.get(normalized, normalized)


def _format_language_label(language: str, runtime: str | None, title: str) -> str:
    if runtime == "three":
        return (
            "JavaScript + Three.js"
            if language == "javascript"
            else "TypeScript + Three.js"
        )
    if runtime == "p5":
        return (
            "JavaScript + p5.js"
            if language == "javascript"
            else "TypeScript + p5.js"
        )
    if runtime == "glsl":
        return "GLSL"
    if runtime == "hydra":
        return "JavaScript + Hydra"
    if language == "typescript":
        return "TypeScript + React" if title.endswith(".tsx") else "TypeScript"
    return {
        "css": "CSS",
        "glsl": "GLSL",
        "html": "HTML",
        "javascript": "JavaScript",
        "json": "JSON",
        "python": "Python",
    }.get(language, language.title())


def _default_artifact_title(
    language: str,
    runtime: str | None,
    index: int,
) -> str:
    if runtime == "three":
        extension = "js" if language == "javascript" else "ts"
        return f"generated-scene-{index}.three.{extension}"
    if runtime == "p5":
        extension = "js" if language == "javascript" else "ts"
        return f"generated-sketch-{index}.p5.{extension}"
    if runtime == "glsl":
        return f"generated-shader-{index}.frag"
    if runtime == "hydra":
        return f"generated-patch-{index}.hydra.js"
    extensions = {
        "css": "css",
        "glsl": "glsl",
        "html": "html",
        "javascript": "js",
        "json": "json",
        "python": "py",
        "typescript": "ts",
    }
    return f"generated-artifact-{index}.{extensions.get(language, 'txt')}"


def _workflow_run_id(request: AssistantRequest) -> str:
    raw = request.conversation_id or request.project_id or "assistant-workflow"
    digest = sha256(f"{raw}:{request.query}".encode("utf-8")).hexdigest()[:12]
    return f"workflow-{digest}"


def _mime_type_for_language(language: str) -> str:
    return {
        "css": "text/css",
        "glsl": "text/plain",
        "html": "text/html",
        "javascript": "text/javascript",
        "json": "application/json",
        "typescript": "text/typescript",
    }.get(language, "text/plain")


def _artifact_tags(artifact: WorkflowArtifact) -> tuple[str, ...]:
    tags = ["workflow"]
    if artifact.is_creative:
        tags.append("creative")
    if artifact.runtime is not None:
        tags.append(artifact.runtime)
    return tuple(tags)


def _domain_label(domain: CreativeCodingDomain | None) -> str | None:
    if domain is None:
        return None
    return domain.value.replace("_", " ")


def _domain_value(domain: CreativeCodingDomain | None) -> str | None:
    return domain.value if domain is not None else None


def _sanitize_filename(value: str) -> str:
    normalized = value.strip().strip("\"'")
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", normalized)
    return normalized.strip(".-")


def _sanitize_identifier(value: str) -> str:
    normalized = _sanitize_filename(value).lower()
    normalized = re.sub(r"[^a-z0-9._-]+", "-", normalized)
    normalized = normalized.strip(".-_")
    if not normalized or not re.match(r"^[a-z0-9]", normalized):
        return ""
    return normalized[:128]


def _prefixed_identifier(prefix: str, value: str) -> str:
    suffix_limit = max(1, 127 - len(prefix))
    return _sanitize_identifier(f"{prefix}-{value[:suffix_limit]}")


def _trim_code_block(content: str) -> str:
    return content.replace("\r\n", "\n").strip("\n")
