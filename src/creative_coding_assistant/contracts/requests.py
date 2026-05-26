"""Assistant request and response contracts."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.contracts.events import StreamEvent


class CreativeCodingDomain(StrEnum):
    THREE_JS = "three_js"
    REACT_THREE_FIBER = "react_three_fiber"
    P5_JS = "p5_js"
    GLSL = "glsl"
    PROCESSING = "processing"
    CANVAS_2D = "canvas_2d"
    WEBGPU_WGSL = "webgpu_wgsl"
    GSAP = "gsap"
    TONE_JS = "tone_js"
    PIXI_JS = "pixi_js"
    MATTER_JS = "matter_js"
    RAPIER = "rapier"
    HYDRA = "hydra"
    SHADERTOY = "shadertoy"
    TOUCHDESIGNER = "touchdesigner"
    HOUDINI = "houdini"
    BLENDER_GEOMETRY_NODES = "blender_geometry_nodes"
    UNITY = "unity"
    UNREAL = "unreal"
    MAX_MSP = "max_msp"
    NOTCH = "notch"
    VVVV = "vvvv"
    OPENFRAMEWORKS = "openframeworks"
    OPENRNDR = "openrndr"
    SUPERCOLLIDER = "supercollider"
    SONIC_PI = "sonic_pi"
    TIDALCYCLES = "tidalcycles"
    WEB_AUDIO_API = "web_audio_api"
    P5_SOUND = "p5_sound"
    ML5_JS = "ml5_js"
    TENSORFLOW_JS = "tensorflow_js"
    COMFYUI = "comfyui"
    STABLE_DIFFUSION_WORKFLOWS = "stable_diffusion_workflows"
    RUNWAY = "runway"
    BLENDER_PYTHON_API = "blender_python_api"
    UNREAL_BLUEPRINTS = "unreal_blueprints"
    ABLETON_LIVE = "ableton_live"
    VCV_RACK = "vcv_rack"
    GODOT = "godot"
    RESOLUME = "resolume"
    MADMAPPER = "madmapper"
    CABLES_GL = "cables_gl"
    PURE_DATA = "pure_data"


class AssistantMode(StrEnum):
    GENERATE = "generate"
    EXPLAIN = "explain"
    DEBUG = "debug"
    DESIGN = "design"
    REVIEW = "review"
    PREVIEW = "preview"


SUPPORTED_IMAGE_REFERENCE_MIME_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/gif"}
)
MAX_IMAGE_REFERENCE_BYTES = 1024 * 1024
MAX_IMAGE_REFERENCE_COUNT = 4


class AssistantImageReference(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    type: Literal["image"] = "image"
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    mime_type: str = Field(alias="mimeType", min_length=1)
    size_bytes: int = Field(alias="sizeBytes", gt=0, le=MAX_IMAGE_REFERENCE_BYTES)
    data_url: str | None = Field(default=None, alias="dataUrl")

    @field_validator("mime_type")
    @classmethod
    def validate_mime_type(cls, value: str) -> str:
        if value not in SUPPORTED_IMAGE_REFERENCE_MIME_TYPES:
            raise ValueError("Unsupported assistant image reference MIME type.")
        return value

    @field_validator("data_url")
    @classmethod
    def validate_data_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.startswith("data:image/"):
            raise ValueError("Assistant image reference data URL must be an image.")
        return value


class AssistantRequest(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    query: str = Field(min_length=1)
    conversation_id: str | None = None
    project_id: str | None = None
    domain: CreativeCodingDomain | None = None
    domains: tuple[CreativeCodingDomain, ...] = Field(default_factory=tuple)
    mode: AssistantMode = AssistantMode.GENERATE
    attachments: tuple[AssistantImageReference, ...] = Field(default_factory=tuple)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Assistant request query must not be empty.")
        return value

    @field_validator("domains", mode="before")
    @classmethod
    def normalize_domains(
        cls,
        value: Sequence[CreativeCodingDomain | str] | CreativeCodingDomain | str | None,
    ) -> tuple[CreativeCodingDomain, ...]:
        if value is None:
            return ()
        if isinstance(value, CreativeCodingDomain):
            return (value,)
        if isinstance(value, str):
            return (CreativeCodingDomain(value.strip()),)

        normalized: list[CreativeCodingDomain] = []
        for item in value:
            domain = (
                item
                if isinstance(item, CreativeCodingDomain)
                else CreativeCodingDomain(str(item).strip())
            )
            if domain not in normalized:
                normalized.append(domain)
        return tuple(normalized)

    @model_validator(mode="before")
    @classmethod
    def populate_legacy_domain_fields(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        domain = normalized.get("domain")
        domains = normalized.get("domains")

        if domain is not None and not domains:
            normalized["domains"] = (domain,)

        return normalized

    @model_validator(mode="after")
    def validate_domain_alignment(self) -> AssistantRequest:
        if self.domain is None and len(self.domains) == 1:
            object.__setattr__(self, "domain", self.domains[0])

        if self.domain is not None and self.domain not in self.domains:
            raise ValueError(
                "Assistant request domain must be included in domains "
                "when both are provided."
            )

        return self

    @field_validator("attachments", mode="before")
    @classmethod
    def normalize_attachments(
        cls,
        value: (
            Sequence[AssistantImageReference | dict[str, object]]
            | AssistantImageReference
            | dict[str, object]
            | None
        ),
    ) -> tuple[AssistantImageReference, ...]:
        if value is None:
            return ()
        if isinstance(value, AssistantImageReference):
            attachments = (value,)
        elif isinstance(value, dict):
            attachments = (AssistantImageReference.model_validate(value),)
        else:
            attachments = tuple(
                item
                if isinstance(item, AssistantImageReference)
                else AssistantImageReference.model_validate(item)
                for item in value
            )

        if len(attachments) > MAX_IMAGE_REFERENCE_COUNT:
            raise ValueError(
                f"Attach up to {MAX_IMAGE_REFERENCE_COUNT} image references."
            )
        return attachments


class AssistantResponse(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    answer: str = Field(min_length=1)
    events: tuple[StreamEvent, ...] = Field(default_factory=tuple)
