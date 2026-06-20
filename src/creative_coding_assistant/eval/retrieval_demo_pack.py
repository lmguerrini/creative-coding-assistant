"""Operational retrieval demo pack for capstone-oriented KB validation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from creative_coding_assistant.contracts import CreativeCodingDomain
from creative_coding_assistant.rag import get_official_source
from creative_coding_assistant.rag.retrieval import (
    KnowledgeBaseRetrievalFilter,
    KnowledgeBaseRetrievalRequest,
)

CCA_OPERATIONAL_KB_SCOPE = (
    "CCA's KB is for practical creative-production guidance: runtime choice, "
    "generative structure, audio mapping, shader patterns, and debugging."
)
FUTURE_HOLOMIND_BOUNDARY = (
    "Future HoloMind should own deeper symbolic, semantic, and theoretical "
    "interpretation; these demos stay bounded to operational translation."
)


class RetrievalDemoScenario(BaseModel):
    """Small typed retrieval scenario for capstone demo preparation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    demo_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    title: str = Field(min_length=1)
    query: str = Field(min_length=1)
    operational_goal: str = Field(min_length=1)
    domains: tuple[CreativeCodingDomain, ...] = Field(min_length=1)
    expected_source_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("domains", mode="before")
    @classmethod
    def normalize_domains(
        cls,
        value: tuple[CreativeCodingDomain | str, ...]
        | list[CreativeCodingDomain | str]
        | CreativeCodingDomain
        | str,
    ) -> tuple[CreativeCodingDomain, ...]:
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

    @field_validator("expected_source_ids", mode="before")
    @classmethod
    def normalize_expected_source_ids(
        cls,
        value: tuple[str, ...] | list[str] | str,
    ) -> tuple[str, ...]:
        if isinstance(value, str):
            return (value.strip(),)

        normalized: list[str] = []
        for item in value:
            source_id = str(item).strip()
            if source_id and source_id not in normalized:
                normalized.append(source_id)
        return tuple(normalized)

    @model_validator(mode="after")
    def validate_source_alignment(self) -> RetrievalDemoScenario:
        for source_id in self.expected_source_ids:
            source = get_official_source(source_id)
            if source.domain not in self.domains:
                raise ValueError(
                    "Retrieval demo sources must stay within the declared scenario "
                    "domains."
                )
        return self

    def build_request(self, *, limit: int = 5) -> KnowledgeBaseRetrievalRequest:
        return KnowledgeBaseRetrievalRequest(
            query=self.query,
            limit=limit,
            filters=KnowledgeBaseRetrievalFilter(domains=self.domains),
        )


class RetrievalDemoPack(BaseModel):
    """Standalone pack of operational retrieval scenarios."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    pack_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    title: str = Field(min_length=1)
    operational_scope: str = Field(min_length=1)
    holomind_boundary: str = Field(min_length=1)
    scenarios: tuple[RetrievalDemoScenario, ...] = Field(min_length=1)


_CAPSTONE_RETRIEVAL_DEMO_SCENARIOS = (
    RetrievalDemoScenario(
        demo_id="runtime_selection_hydra_vs_p5",
        title="Runtime selection for fast live visuals",
        query=(
            "I want feedback-heavy live visuals with fast iteration and layered "
            "oscillator textures. Should I use Hydra or p5.js first?"
        ),
        operational_goal=(
            "Compare runtimes for immediate live-coding feedback versus browser "
            "sketch control."
        ),
        domains=(CreativeCodingDomain.HYDRA, CreativeCodingDomain.P5_JS),
        expected_source_ids=("hydra_docs", "p5_reference"),
    ),
    RetrievalDemoScenario(
        demo_id="audio_reactive_browser_mapping",
        title="Audio-reactive mapping foundations",
        query=(
            "How should I map bass, mids, amplitude, and FFT data into a browser "
            "visual system?"
        ),
        operational_goal=(
            "Ground audio-reactive mappings in browser-safe analysis patterns and "
            "practical signal choices."
        ),
        domains=(
            CreativeCodingDomain.P5_SOUND,
            CreativeCodingDomain.WEB_AUDIO_API,
            CreativeCodingDomain.TONE_JS,
        ),
        expected_source_ids=(
            "p5_sound_analysis_reference",
            "web_audio_analyser_node",
            "web_audio_visualization_guide",
            "tone_js_analysis_reference",
        ),
    ),
    RetrievalDemoScenario(
        demo_id="shader_post_fx_pipeline",
        title="Shader and post-processing pipeline choices",
        query=(
            "How do I build a glow-heavy kaleidoscopic browser visual with shaders "
            "or post-processing passes?"
        ),
        operational_goal=(
            "Connect shader references with concrete browser render-pipeline "
            "guidance."
        ),
        domains=(
            CreativeCodingDomain.THREE_JS,
            CreativeCodingDomain.GLSL,
            CreativeCodingDomain.SHADERTOY,
        ),
        expected_source_ids=(
            "three_manual_effects",
            "glsl_mdn_webgl_examples",
            "shadertoy_howto",
        ),
    ),
    RetrievalDemoScenario(
        demo_id="audiovisual_composition_browser_set",
        title="Audiovisual composition timing",
        query=(
            "How should I coordinate loop timing, sample playback, and visual "
            "accents across Tone.js and a browser visual runtime?"
        ),
        operational_goal=(
            "Retrieve timing, playback, and browser rendering references for a "
            "small audiovisual composition workflow."
        ),
        domains=(
            CreativeCodingDomain.TONE_JS,
            CreativeCodingDomain.P5_JS,
            CreativeCodingDomain.THREE_JS,
        ),
        expected_source_ids=(
            "tone_js_analysis_reference",
            "tone_js_docs",
            "p5_reference",
            "three_manual",
        ),
    ),
    RetrievalDemoScenario(
        demo_id="creative_debugging_silent_audio",
        title="Creative debugging for silent browser audio",
        query=(
            "My browser audio-reactive sketch stays silent until a click and the "
            "analyser looks flat. What should I check first?"
        ),
        operational_goal=(
            "Surface bounded debugging guidance for browser audio startup, "
            "analysis nodes, and p5 sound input wiring."
        ),
        domains=(
            CreativeCodingDomain.WEB_AUDIO_API,
            CreativeCodingDomain.P5_JS,
            CreativeCodingDomain.P5_SOUND,
        ),
        expected_source_ids=(
            "web_audio_visualization_guide",
            "web_audio_analyser_node",
            "p5_reference",
            "p5_sound_reference",
        ),
    ),
    RetrievalDemoScenario(
        demo_id="creative_debugging_three_effects",
        title="Creative debugging for Three.js effects",
        query=(
            "My Three.js post-processing scene is slow and the shadows look wrong. "
            "Which render-pipeline parts should I inspect?"
        ),
        operational_goal=(
            "Retrieve render-target, post-processing, and shadow tradeoff guidance "
            "for browser debugging."
        ),
        domains=(CreativeCodingDomain.THREE_JS,),
        expected_source_ids=("three_manual_effects", "three_manual"),
    ),
    RetrievalDemoScenario(
        demo_id="symbol_to_art_operational_translation",
        title="Operational symbol-to-art translation",
        query=(
            "Turn a concentric mandala motif into a practical browser visual system "
            "with motion, rhythm, and runtime choices."
        ),
        operational_goal=(
            "Keep symbolic prompts grounded in concrete runtime, composition, and "
            "mapping guidance instead of theoretical interpretation."
        ),
        domains=(
            CreativeCodingDomain.P5_JS,
            CreativeCodingDomain.GLSL,
            CreativeCodingDomain.TONE_JS,
            CreativeCodingDomain.THREE_JS,
        ),
        expected_source_ids=(
            "p5_reference",
            "glsl_mdn_webgl_examples",
            "tone_js_analysis_reference",
            "three_manual_effects",
        ),
    ),
)


def build_capstone_retrieval_demo_pack() -> RetrievalDemoPack:
    return RetrievalDemoPack(
        pack_id="capstone_kb_expansion_retrieval_demo_pack",
        title="Capstone KB Expansion Retrieval Demo Pack",
        operational_scope=CCA_OPERATIONAL_KB_SCOPE,
        holomind_boundary=FUTURE_HOLOMIND_BOUNDARY,
        scenarios=_CAPSTONE_RETRIEVAL_DEMO_SCENARIOS,
    )


def capstone_retrieval_demo_source_ids() -> tuple[str, ...]:
    ordered_source_ids: list[str] = []
    for scenario in _CAPSTONE_RETRIEVAL_DEMO_SCENARIOS:
        for source_id in scenario.expected_source_ids:
            if source_id not in ordered_source_ids:
                ordered_source_ids.append(source_id)
    return tuple(ordered_source_ids)


def build_capstone_retrieval_demo_requests(
    *,
    limit: int = 5,
) -> tuple[KnowledgeBaseRetrievalRequest, ...]:
    return tuple(
        scenario.build_request(limit=limit)
        for scenario in _CAPSTONE_RETRIEVAL_DEMO_SCENARIOS
    )
