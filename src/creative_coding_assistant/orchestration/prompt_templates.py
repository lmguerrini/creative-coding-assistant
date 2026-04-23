"""Prompt template contracts and rendering boundaries."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, Self

from jinja2 import Environment, StrictUndefined
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from creative_coding_assistant.orchestration.prompt_inputs import PromptInputResponse
from creative_coding_assistant.orchestration.routing import RouteDecision, RouteName

_SYSTEM_TEMPLATE = """
Route: {{ route.value }}
Mode: {{ prompt_input.user_input.mode.value }}
{% if prompt_input.user_input.domain is not none -%}
Domain: {{ prompt_input.user_input.domain.value }}
{% else -%}
Domain: unspecified
{% endif %}
Use the provided context sections as working context. Keep responses grounded in
the structured inputs that follow.
""".strip()

_USER_TEMPLATE = """
User Request:
{{ prompt_input.user_input.query }}
""".strip()

_MEMORY_TEMPLATE = """
{% if prompt_input.memory_input.running_summary is not none -%}
Running Summary:
{{ prompt_input.memory_input.running_summary.content }}

{% endif -%}
{% if prompt_input.memory_input.recent_turns -%}
Recent Turns:
{% for turn in prompt_input.memory_input.recent_turns -%}
- {{ turn.role.value }}[{{ turn.turn_index }}]: {{ turn.content }}
{% endfor %}

{% endif -%}
{% if prompt_input.memory_input.project_memories -%}
Project Memory:
{% for memory in prompt_input.memory_input.project_memories -%}
- {{ memory.memory_kind.value }} ({{ memory.source }}): {{ memory.content }}
{% endfor %}
{% endif -%}
""".strip()

_RETRIEVAL_TEMPLATE = """
Official Knowledge Base:
{% for chunk in prompt_input.retrieval_input.chunks -%}
- {{ chunk.registry_title }} / {{ chunk.document_title }} ({{ chunk.source_id }})
  Source: {{ chunk.source_url }}
  Score: {{ '%.4f'|format(chunk.score) }}
  Excerpt: {{ chunk.excerpt }}
{% endfor %}
""".strip()


class RenderedPromptRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    CONTEXT = "context"


class RenderedPromptSectionName(StrEnum):
    SYSTEM = "system"
    USER = "user"
    MEMORY = "memory"
    RETRIEVAL = "retrieval"


class RenderedPromptSection(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    role: RenderedPromptRole
    name: RenderedPromptSectionName
    content: str = Field(min_length=1)


class RenderedPromptRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    route: RouteName
    prompt_input: PromptInputResponse

    @model_validator(mode="after")
    def validate_route_alignment(self) -> Self:
        if self.prompt_input.request.route != self.route:
            raise ValueError("Prompt input route must match the rendered route.")
        return self


class RenderedPromptResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: RenderedPromptRequest
    sections: tuple[RenderedPromptSection, ...] = Field(default_factory=tuple)


class PromptRenderer(Protocol):
    def render(
        self,
        request: RenderedPromptRequest,
    ) -> RenderedPromptResponse:
        """Render provider-independent prompt sections from structured inputs."""


class JinjaPromptRenderer:
    """Render prompt-ready sections with Jinja2 and no provider assumptions."""

    def __init__(self) -> None:
        self._environment = Environment(
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        )

    def render(
        self,
        request: RenderedPromptRequest,
    ) -> RenderedPromptResponse:
        sections = [
            self._render_section(
                role=RenderedPromptRole.SYSTEM,
                name=RenderedPromptSectionName.SYSTEM,
                template=_SYSTEM_TEMPLATE,
                request=request,
            ),
            self._render_section(
                role=RenderedPromptRole.USER,
                name=RenderedPromptSectionName.USER,
                template=_USER_TEMPLATE,
                request=request,
            ),
        ]

        if request.prompt_input.memory_input is not None:
            sections.append(
                self._render_section(
                    role=RenderedPromptRole.CONTEXT,
                    name=RenderedPromptSectionName.MEMORY,
                    template=_MEMORY_TEMPLATE,
                    request=request,
                )
            )

        if request.prompt_input.retrieval_input is not None:
            sections.append(
                self._render_section(
                    role=RenderedPromptRole.CONTEXT,
                    name=RenderedPromptSectionName.RETRIEVAL,
                    template=_RETRIEVAL_TEMPLATE,
                    request=request,
                )
            )

        rendered = RenderedPromptResponse(
            request=request,
            sections=tuple(section for section in sections if section.content),
        )
        logger.info(
            "Rendered prompt with {} section(s) for route '{}'",
            len(rendered.sections),
            request.route.value,
        )
        return rendered

    def _render_section(
        self,
        *,
        role: RenderedPromptRole,
        name: RenderedPromptSectionName,
        template: str,
        request: RenderedPromptRequest,
    ) -> RenderedPromptSection:
        content = self._environment.from_string(template).render(
            route=request.route,
            prompt_input=request.prompt_input,
        )
        normalized = "\n".join(
            line.rstrip() for line in content.splitlines() if line.strip()
        ).strip()
        return RenderedPromptSection(role=role, name=name, content=normalized)


def build_rendered_prompt_request(
    *,
    route_decision: RouteDecision | RouteName,
    prompt_input: PromptInputResponse,
) -> RenderedPromptRequest:
    route = (
        route_decision
        if isinstance(route_decision, RouteName)
        else route_decision.route
    )
    return RenderedPromptRequest(route=route, prompt_input=prompt_input)
