"""Approved official source definitions for the knowledge base."""

from __future__ import annotations

from enum import StrEnum
from typing import Self
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.contracts import CreativeCodingDomain


class OfficialSourceType(StrEnum):
    API_REFERENCE = "api_reference"
    GUIDE = "guide"
    EXAMPLES = "examples"
    SPECIFICATION = "specification"


class SourceApprovalStatus(StrEnum):
    APPROVED = "approved"


OFFICIAL_HOSTS_BY_DOMAIN: dict[CreativeCodingDomain, tuple[str, ...]] = {
    CreativeCodingDomain.THREE_JS: ("threejs.org",),
    CreativeCodingDomain.REACT_THREE_FIBER: ("r3f.docs.pmnd.rs",),
    CreativeCodingDomain.P5_JS: ("p5js.org",),
    CreativeCodingDomain.GLSL: ("registry.khronos.org", "developer.mozilla.org"),
}


class OfficialSource(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    domain: CreativeCodingDomain
    title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    url: str = Field(min_length=1)
    source_type: OfficialSourceType
    approval_status: SourceApprovalStatus = SourceApprovalStatus.APPROVED
    priority: int = Field(ge=1)
    allowed_path_prefixes: tuple[str, ...] = Field(min_length=1)
    additional_urls: tuple[str, ...] = Field(default_factory=tuple)
    tags: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("url")
    @classmethod
    def require_https_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Official source URLs must be absolute HTTPS URLs.")
        return value

    @field_validator("allowed_path_prefixes")
    @classmethod
    def validate_path_prefixes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for prefix in value:
            if not prefix.startswith("/"):
                raise ValueError("Allowed path prefixes must start with '/'.")
        return value

    @field_validator("additional_urls")
    @classmethod
    def validate_additional_urls(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for url in value:
            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError(
                    "Additional official source URLs must be absolute HTTPS URLs."
                )
        return value

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for tag in value:
            if not tag or tag != tag.lower() or " " in tag:
                raise ValueError("Source tags must be lowercase tokens.")
        return value

    @model_validator(mode="after")
    def validate_official_scope(self) -> Self:
        allowed_hosts = OFFICIAL_HOSTS_BY_DOMAIN[self.domain]
        for url in (self.url, *self.additional_urls):
            parsed = urlparse(url)
            if parsed.hostname not in allowed_hosts:
                raise ValueError(
                    "Official source host is not approved for this domain."
                )

            path_is_allowed = any(
                parsed.path.startswith(prefix) for prefix in self.allowed_path_prefixes
            )
            if not path_is_allowed:
                raise ValueError(
                    "Official source URL is outside its approved path scope."
                )

        return self


def _validate_source_registry(
    sources: tuple[OfficialSource, ...],
) -> tuple[OfficialSource, ...]:
    source_ids = [source.source_id for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Official source IDs must be unique.")

    registered_domains = {source.domain for source in sources}
    missing_domains = tuple(
        domain for domain in CreativeCodingDomain if domain not in registered_domains
    )
    if missing_domains:
        values = ", ".join(domain.value for domain in missing_domains)
        raise ValueError(f"Official sources missing domains: {values}")

    return sources


APPROVED_OFFICIAL_SOURCES: tuple[OfficialSource, ...] = _validate_source_registry((
    OfficialSource(
        source_id="three_docs",
        domain=CreativeCodingDomain.THREE_JS,
        title="three.js Documentation",
        publisher="three.js",
        url="https://threejs.org/docs/",
        source_type=OfficialSourceType.API_REFERENCE,
        priority=10,
        allowed_path_prefixes=("/docs/",),
        tags=("reference", "api", "webgl"),
    ),
    OfficialSource(
        source_id="three_manual",
        domain=CreativeCodingDomain.THREE_JS,
        title="three.js Manual",
        publisher="three.js",
        url="https://threejs.org/manual/",
        source_type=OfficialSourceType.GUIDE,
        priority=20,
        allowed_path_prefixes=("/manual/",),
        additional_urls=(
            "https://threejs.org/manual/en/fundamentals.html",
            "https://threejs.org/manual/en/responsive.html",
            "https://threejs.org/manual/en/cameras.html",
            "https://threejs.org/manual/en/lights.html",
        ),
        tags=("guide", "fundamentals", "webgl"),
    ),
    OfficialSource(
        source_id="three_examples",
        domain=CreativeCodingDomain.THREE_JS,
        title="three.js Examples",
        publisher="three.js",
        url="https://threejs.org/examples/",
        source_type=OfficialSourceType.EXAMPLES,
        priority=30,
        allowed_path_prefixes=("/examples/",),
        tags=("examples", "patterns", "webgl"),
    ),
    OfficialSource(
        source_id="r3f_introduction",
        domain=CreativeCodingDomain.REACT_THREE_FIBER,
        title="React Three Fiber Introduction",
        publisher="pmndrs",
        url="https://r3f.docs.pmnd.rs/getting-started/introduction",
        source_type=OfficialSourceType.GUIDE,
        priority=10,
        allowed_path_prefixes=("/getting-started/",),
        tags=("guide", "react", "three"),
    ),
    OfficialSource(
        source_id="r3f_canvas_api",
        domain=CreativeCodingDomain.REACT_THREE_FIBER,
        title="React Three Fiber Canvas API",
        publisher="pmndrs",
        url="https://r3f.docs.pmnd.rs/api/canvas",
        source_type=OfficialSourceType.API_REFERENCE,
        priority=20,
        allowed_path_prefixes=("/api/",),
        tags=("reference", "canvas", "react"),
    ),
    OfficialSource(
        source_id="r3f_hooks_api",
        domain=CreativeCodingDomain.REACT_THREE_FIBER,
        title="React Three Fiber Hooks API",
        publisher="pmndrs",
        url="https://r3f.docs.pmnd.rs/api/hooks",
        source_type=OfficialSourceType.API_REFERENCE,
        priority=30,
        allowed_path_prefixes=("/api/",),
        tags=("reference", "hooks", "react"),
    ),
    OfficialSource(
        source_id="p5_reference",
        domain=CreativeCodingDomain.P5_JS,
        title="p5.js Core Sketch Reference",
        publisher="p5.js",
        url="https://p5js.org/reference/p5/setup/",
        source_type=OfficialSourceType.API_REFERENCE,
        priority=10,
        allowed_path_prefixes=("/reference/p5/",),
        additional_urls=(
            "https://p5js.org/reference/p5/draw/",
            "https://p5js.org/reference/p5/createCanvas/",
            "https://p5js.org/reference/p5/background/",
            "https://p5js.org/reference/p5/circle/",
            "https://p5js.org/reference/p5/ellipse/",
            "https://p5js.org/reference/p5/fill/",
            "https://p5js.org/reference/p5/frameCount/",
            "https://p5js.org/reference/p5/mouseX/",
            "https://p5js.org/reference/p5/mouseY/",
            "https://p5js.org/reference/p5/random/",
        ),
        tags=("reference", "api", "sketch"),
    ),
    OfficialSource(
        source_id="p5_tutorials",
        domain=CreativeCodingDomain.P5_JS,
        title="p5.js Motion Tutorials",
        publisher="p5.js",
        url="https://p5js.org/tutorials/get-started/",
        source_type=OfficialSourceType.GUIDE,
        priority=20,
        allowed_path_prefixes=("/tutorials/",),
        additional_urls=(
            "https://p5js.org/tutorials/variables-and-change",
            "https://p5js.org/tutorials/conditionals-and-interactivity/",
        ),
        tags=("guide", "animation", "creative-coding"),
    ),
    OfficialSource(
        source_id="p5_examples",
        domain=CreativeCodingDomain.P5_JS,
        title="p5.js Runnable Examples",
        publisher="p5.js",
        url="https://p5js.org/examples/calculating-values-constrain/",
        source_type=OfficialSourceType.EXAMPLES,
        priority=30,
        allowed_path_prefixes=("/examples/",),
        additional_urls=(
            "https://p5js.org/examples/animation-and-variables-drawing-lines/",
            "https://p5js.org/examples/angles-and-motion-sine-cosine/",
            "https://p5js.org/examples/games-circle-clicker/",
        ),
        tags=("examples", "patterns", "animation"),
    ),
    OfficialSource(
        source_id="glsl_language_spec_460",
        domain=CreativeCodingDomain.GLSL,
        title="OpenGL Shading Language 4.60 Specification",
        publisher="Khronos Group",
        url="https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html",
        source_type=OfficialSourceType.SPECIFICATION,
        priority=10,
        allowed_path_prefixes=("/OpenGL/specs/gl/",),
        tags=("specification", "opengl", "glsl"),
    ),
    OfficialSource(
        source_id="glsl_mdn_webgl_examples",
        domain=CreativeCodingDomain.GLSL,
        title="MDN WebGL GLSL Examples",
        publisher="MDN",
        url="https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/By_example/Hello_GLSL",
        source_type=OfficialSourceType.EXAMPLES,
        priority=15,
        allowed_path_prefixes=(
            "/en-US/docs/Web/API/WebGL_API/By_example/",
            "/en-US/docs/Web/API/WebGL_API/Tutorial/",
        ),
        additional_urls=(
            "https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/By_example/Textures_from_code",
            "https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Tutorial/Using_shaders_to_apply_color_in_WebGL",
        ),
        tags=("examples", "fragment-shader", "webgl"),
    ),
    OfficialSource(
        source_id="glsl_es_language_spec_320",
        domain=CreativeCodingDomain.GLSL,
        title="OpenGL ES Shading Language 3.20 Specification",
        publisher="Khronos Group",
        url=(
            "https://registry.khronos.org/OpenGL/specs/es/3.2/"
            "GLSL_ES_Specification_3.20.html"
        ),
        source_type=OfficialSourceType.SPECIFICATION,
        priority=20,
        allowed_path_prefixes=("/OpenGL/specs/es/3.2/",),
        tags=("specification", "opengl-es", "glsl-es"),
    ),
))


def approved_official_sources() -> tuple[OfficialSource, ...]:
    return APPROVED_OFFICIAL_SOURCES


def approved_sources_for_domain(
    domain: CreativeCodingDomain,
) -> tuple[OfficialSource, ...]:
    sources = (
        source for source in APPROVED_OFFICIAL_SOURCES if source.domain == domain
    )
    return tuple(
        sorted(sources, key=lambda source: (source.priority, source.source_id))
    )


def official_source_domains() -> tuple[CreativeCodingDomain, ...]:
    domains = {source.domain for source in APPROVED_OFFICIAL_SOURCES}
    return tuple(domain for domain in CreativeCodingDomain if domain in domains)


def get_official_source(source_id: str) -> OfficialSource:
    for source in APPROVED_OFFICIAL_SOURCES:
        if source.source_id == source_id:
            return source
    raise ValueError(f"Unknown official source: {source_id}")
