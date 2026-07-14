"""Operational retrieval demo pack for capstone-oriented KB validation."""

from __future__ import annotations

import hashlib
import json

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
CURRENT_PRODUCT_RETRIEVAL_BENCHMARK_VERSION = "current-product-retrieval.v1"
CURRENT_PRODUCT_RETRIEVAL_METRICS = (
    "context_precision",
    "faithfulness",
    "answer_relevancy",
    "context_relevancy",
    "context_recall",
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
    reference_source_ids: tuple[str, ...] = Field(min_length=1)
    reference_context: tuple[str, ...] = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    applicable_metrics: tuple[str, ...] = Field(
        default=CURRENT_PRODUCT_RETRIEVAL_METRICS,
        min_length=1,
    )

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

    @field_validator("expected_source_ids", "reference_source_ids", mode="before")
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

    @field_validator("reference_context", mode="before")
    @classmethod
    def normalize_reference_context(
        cls,
        value: tuple[str, ...] | list[str] | str,
    ) -> tuple[str, ...]:
        if isinstance(value, str):
            value = (value,)
        normalized = tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
        if not normalized:
            raise ValueError("Reference context must contain at least one source-bounded claim.")
        return normalized

    @model_validator(mode="after")
    def validate_source_alignment(self) -> RetrievalDemoScenario:
        if not set(self.reference_source_ids).issubset(self.expected_source_ids):
            raise ValueError(
                "Reference sources must be an explicit subset of expected sources."
            )
        for source_id in (*self.expected_source_ids, *self.reference_source_ids):
            source = get_official_source(source_id)
            if source.domain not in self.domains:
                raise ValueError(
                    "Retrieval demo sources must stay within the declared scenario "
                    "domains."
                )
        return self

    @field_validator("applicable_metrics")
    @classmethod
    def validate_applicable_metrics(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(metric.strip() for metric in value if metric.strip()))
        unsupported = tuple(
            metric
            for metric in normalized
            if metric not in CURRENT_PRODUCT_RETRIEVAL_METRICS
        )
        if unsupported:
            raise ValueError(
                "Unsupported current-product retrieval metric(s): "
                + ", ".join(unsupported)
            )
        if not normalized:
            raise ValueError("At least one current-product retrieval metric is required.")
        return normalized

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
        reference_source_ids=("hydra_docs", "p5_reference"),
        reference_context=(
            "Hydra documents a browser-based video synthesizer designed for live "
            "coding and composable visual transformations.",
            "The p5.js core reference organizes sketches around JavaScript setup and "
            "draw functions and exposes drawing and interaction APIs.",
        ),
        reference_answer=(
            "Start with Hydra when the priority is live-coded video synthesis through "
            "composable visual transformations. Start with p5.js when the work needs an "
            "explicit JavaScript setup/draw sketch, drawing APIs, or interaction state."
        ),
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
        reference_source_ids=(
            "p5_sound_analysis_reference",
            "web_audio_analyser_node",
            "web_audio_visualization_guide",
            "tone_js_analysis_reference",
        ),
        reference_context=(
            "Web Audio AnalyserNode exposes real-time time-domain and "
            "frequency-domain analysis data for visualization.",
            "p5.sound documents p5.FFT for spectrum analysis and p5.Amplitude for amplitude measurement.",
            "Tone.js documents FFT and Meter analysis nodes, while Web Audio best "
            "practices cover browser audio startup from a user gesture.",
        ),
        reference_answer=(
            "Start browser audio from a user gesture, connect the active source to an "
            "analysis node, and read its time-domain or frequency-domain data. Use p5.FFT or "
            "Tone.FFT for spectrum values and p5.Amplitude or Tone.Meter for overall level, "
            "then map normalized analysis values to bounded visual parameters."
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
            "Connect shader references with concrete browser render-pipeline guidance."
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
        reference_source_ids=(
            "three_manual_effects",
            "glsl_mdn_webgl_examples",
            "shadertoy_howto",
        ),
        reference_context=(
            "The Three.js post-processing guide composes a RenderPass and subsequent "
            "effect passes with EffectComposer.",
            "MDN WebGL examples and the Shadertoy guide document fragment-shader inputs and pixel-level shader output.",
        ),
        reference_answer=(
            "Build the base scene first, then render it through a Three.js EffectComposer "
            "with a RenderPass before bloom or custom shader passes. Implement the repeated "
            "kaleidoscopic pixel treatment in a fragment shader, and keep render-target "
            "resolution and the number of effect passes bounded."
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
        reference_source_ids=(
            "tone_js_analysis_reference",
            "tone_js_docs",
            "p5_reference",
            "three_manual",
        ),
        reference_context=(
            "Tone.js documents Loop and Player for scheduled repeated callbacks and sample playback.",
            "p5.js draw and the Three.js render-loop manual describe browser animation "
            "loops that update and render visual state.",
        ),
        reference_answer=(
            "Use Tone.js scheduling and playback primitives such as Loop and Player for the "
            "musical events. Keep p5.js draw or the Three.js render loop as the visual update "
            "loop, and have it read the current schedule phase or analysis values to trigger "
            "visual accents."
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
        reference_source_ids=(
            "web_audio_visualization_guide",
            "web_audio_analyser_node",
            "p5_reference",
            "p5_sound_reference",
        ),
        reference_context=(
            "Browser audio commonly requires explicit start or resume from a user "
            "gesture, and p5.js exposes userStartAudio for that purpose.",
            "AnalyserNode must receive an audio source before its time-domain or "
            "frequency-domain buffers can represent that signal.",
        ),
        reference_answer=(
            "First require a click or other user gesture and explicitly resume or start the "
            "audio context. Verify that the intended media or microphone source is connected "
            "to the analyser and is actually playing. Then inspect the analyser's FFT size, "
            "smoothing setting, and returned time-domain or frequency-domain buffers."
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
        reference_source_ids=("three_manual_effects", "three_manual"),
        reference_context=(
            "The Three.js manuals document post-processing passes, render targets, "
            "shadow-map configuration, responsive resolution, lights, materials, and "
            "geometry.",
            "Each post-processing pass adds render work, while shadow cost depends "
            "on shadow-casting lights and shadow-map resources.",
        ),
        reference_answer=(
            "Inspect render resolution and pixel ratio, shadow-map configuration and "
            "shadow-casting lights, material and geometry cost, and each post-processing "
            "pass or render target. Reduce or disable one stage at a time to isolate the "
            "expensive or incorrectly configured part of the pipeline."
        ),
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
        reference_source_ids=(
            "p5_reference",
            "glsl_mdn_webgl_examples",
            "tone_js_analysis_reference",
            "three_manual_effects",
        ),
        reference_context=(
            "p5.js exposes browser drawing and animation-loop APIs for explicit 2D composition and interaction.",
            "GLSL fragment shaders produce per-pixel output, Tone.js analysis nodes "
            "expose audio values, and Three.js supports spatial scenes and "
            "post-processing passes.",
        ),
        reference_answer=(
            "Express the motif as visual parameters such as center, rings, angular repetition, "
            "symmetry count, line weight, and palette. Use p5.js for explicit 2D drawing and "
            "interaction, a GLSL fragment shader for dense per-pixel repetition, Three.js for "
            "a spatial scene or post-processing pipeline, and Tone.js analysis when audio "
            "values should drive motion."
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


def fingerprint_capstone_retrieval_demo_pack(
    pack: RetrievalDemoPack | None = None,
) -> str:
    """Bind immutable benchmark inputs and expectations to a stable SHA-256 digest."""

    resolved = pack or build_capstone_retrieval_demo_pack()
    payload = {
        "benchmarkVersion": CURRENT_PRODUCT_RETRIEVAL_BENCHMARK_VERSION,
        "packId": resolved.pack_id,
        "scenarios": [
            {
                "caseId": scenario.demo_id,
                "label": scenario.title,
                "question": scenario.query,
                "expectedDomains": [domain.value for domain in scenario.domains],
                "expectedSourceIds": list(scenario.expected_source_ids),
                "referenceSourceIds": list(scenario.reference_source_ids),
                "referenceContext": list(scenario.reference_context),
                "referenceAnswer": scenario.reference_answer,
                "applicableMetrics": list(scenario.applicable_metrics),
            }
            for scenario in resolved.scenarios
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


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
