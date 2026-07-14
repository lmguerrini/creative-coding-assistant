"""Lightweight query-domain intent detection for retrieval postprocessing."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from creative_coding_assistant.contracts import CreativeCodingDomain

_WHITESPACE_PATTERN = re.compile(r"\s+")
_DOMAIN_BRIDGE_ACTION_PATTERN = re.compile(
    r"\b(?:across|between|bridge|combine|coordinate|connect|integrate|"
    r"synchroni[sz]e)\b"
)
_DOMAIN_BRIDGE_COUNTERPART_PATTERN = re.compile(
    r"\b(?:browser|canvas|graphics|runtime|scene|shader|sketch|system|visual)\w*\b"
)
_THREE_JS_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bthree(?:\.js|js|\s+js)\b"), 3),
)
_REACT_THREE_FIBER_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\breact\s+three\s+fiber\b"), 3),
    (re.compile(r"@react-three/fiber"), 3),
    (re.compile(r"\br3f\b"), 3),
    (re.compile(r"\buseframe\b"), 2),
)
_P5_JS_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bp5(?:\.js|js)?\b(?!\.sound)"), 3),
)
_GLSL_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bglsl\b"), 3),
    (re.compile(r"\bfragment\s+shader\b"), 2),
    (re.compile(r"\bvertex\s+shader\b"), 2),
)
_PROCESSING_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bprocessing\.org\b"), 3),
    (re.compile(r"\bprocessing\s+(?:sketch|code|java|reference|api)\b"), 3),
    (re.compile(r"\bpde\s+(?:sketch|file|code)\b"), 2),
    (re.compile(r"\.pde\b"), 2),
)
_CANVAS_2D_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bcanvasrenderingcontext2d\b"), 3),
    (re.compile(r"\bcanvas\s*2d\b"), 3),
    (re.compile(r"\b2d\s+canvas\b"), 3),
    (re.compile(r"\bgetcontext\([\"']2d[\"']\)"), 2),
)
_WEBGPU_WGSL_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bwebgpu\b"), 3),
    (re.compile(r"\bwgsl\b"), 3),
    (re.compile(r"\bnavigator\.gpu\b"), 3),
    (re.compile(r"\bgpucanvascontext\b"), 2),
    (re.compile(r"\bgpudevice\b"), 2),
)
_GSAP_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bgsap\b"), 3),
    (re.compile(r"\bgreensock\b"), 3),
    (re.compile(r"\bgsap\.(?:to|from|fromto|timeline|set)\b"), 3),
)
_TONE_JS_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\btone\.js\b"), 3),
    (re.compile(r"\btonejs\b"), 3),
    (re.compile(r"\btone\.(?:synth|transport|sequence|player|start)\b"), 3),
    (re.compile(r"\bnew\s+tone\."), 3),
)
_PIXI_JS_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bpixi\.js\b"), 3),
    (re.compile(r"\bpixijs\b"), 3),
    (re.compile(r"\bpixi\.(?:application|graphics|sprite|container|assets)\b"), 3),
)
_MATTER_JS_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bmatter\.js\b"), 3),
    (re.compile(r"\bmatterjs\b"), 3),
    (
        re.compile(
            r"\bmatter\.(?:engine|world|bodies|body|runner|composite|constraint)\b"
        ),
        3,
    ),
)
_RAPIER_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"@dimforge/rapier(?:2d|3d)?"), 3),
    (re.compile(r"\brapier\.rs\b"), 3),
    (
        re.compile(
            r"\brapier\s+(?:physics|rigid\s+bodies|colliders?|world|js|"
            r"javascript|2d|3d)\b"
        ),
        3,
    ),
    (re.compile(r"\brapier(?:2d|3d)\b"), 3),
)
_HYDRA_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bhydra(?:-synth|\s+synth|\s+video\s+synth)\b"), 3),
    (re.compile(r"\bhydra\.ojack\b"), 3),
    (re.compile(r"\bhydra\s+(?:osc|src|modulate|live\s+coding|sketch)\b"), 3),
    (
        re.compile(
            r"(?:"
            r"\bhydra\b(?=[^.!?\n]{0,120}\b(?:p5(?:\.js|js)?|runtime|"
            r"visuals?|oscillators?|textures?|live\s+coding)\b)"
            r"|"
            r"\b(?:p5(?:\.js|js)?|runtime|visuals?|oscillators?|textures?|"
            r"live\s+coding)\b[^.!?\n]{0,120}\bhydra\b"
            r")"
        ),
        3,
    ),
)
_SHADERTOY_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bshadertoy\b"), 3),
    (re.compile(r"\bmainimage\s*\("), 3),
    (re.compile(r"\bfragcoord\b"), 2),
    (re.compile(r"\bi(?:time|resolution|mouse|channel0)\b"), 2),
)
_TOUCHDESIGNER_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\btouchdesigner\b"), 3),
    (re.compile(r"\btouch\s+designer\b"), 3),
    (re.compile(r"\bderivative\.ca\b"), 3),
    (re.compile(r"\b(?:chop|top|dat|comp|pop)\s+(?:operator|network|family)\b"), 2),
)
_HOUDINI_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bhoudini\b"), 3),
    (re.compile(r"\bsidefx\b"), 3),
    (re.compile(r"\bvex\b"), 2),
    (re.compile(r"\bhda\b"), 2),
    (re.compile(r"\b(?:sop|dop|vop|lop)\s+(?:network|node|solver)\b"), 2),
)
_BLENDER_GEOMETRY_NODES_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bgeometry\s+nodes?\b"), 3),
    (re.compile(r"\bgeo\s+nodes?\b"), 3),
    (re.compile(r"\bblender\s+(?:scene|mesh|modifier|shader|node\s+tree)\b"), 3),
)
_UNITY_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (
        re.compile(
            r"\bunity\s+(?:engine|editor|scene|script|c#|shader|urp|hdrp|"
            r"gameobject|prefab)\b"
        ),
        3,
    ),
    (re.compile(r"\bunityengine\b"), 3),
    (re.compile(r"\bgameobject\b"), 2),
    (re.compile(r"\bmonobehaviou?r\b"), 2),
    (re.compile(r"\bscriptableobject\b"), 2),
)
_UNREAL_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bunreal\s+engine\b"), 3),
    (re.compile(r"\bue\s*5\b"), 3),
    (re.compile(r"\bue5\b"), 3),
    (re.compile(r"\b(?:nanite|lumen|niagara)\b"), 2),
)
_MAX_MSP_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bmax/msp\b"), 3),
    (re.compile(r"\bmax\s+msp\b"), 3),
    (re.compile(r"\bcycling\s*'?74\b"), 3),
    (re.compile(r"\bmsp\s+(?:patch|object|signal|tutorial)\b"), 2),
    (re.compile(r"\bmax\s+(?:patcher|object|external|gen~|jitter)\b"), 2),
    (re.compile(r"\bgen~\b"), 2),
    (re.compile(r"\bjitter\s+matrix\b"), 2),
)
_NOTCH_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bnotch\s+(?:builder|vfx|rfx|block|blocks|node|manual)\b"), 3),
    (re.compile(r"\bnotchvfx\b"), 3),
    (re.compile(r"\bvfx\s+blocks?\b"), 2),
    (re.compile(r"\bnotchlc\b"), 2),
)
_VVVV_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bvvvv\b"), 3),
    (re.compile(r"\bvvvv\s+gamma\b"), 3),
    (re.compile(r"\bthegraybook\.vvvv\.org\b"), 3),
    (re.compile(r"\bvl\.(?:skia|stride|corelib)\b"), 2),
)
_OPENFRAMEWORKS_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bopenframeworks\b"), 3),
    (re.compile(r"\bopenframeworks\.cc\b"), 3),
    (re.compile(r"\bofx[a-z0-9_]+\b"), 2),
)
_OPENRNDR_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bopenrndr\b"), 3),
    (re.compile(r"\bguide\.openrndr\.org\b"), 3),
    (re.compile(r"\borg\.openrndr\b"), 3),
)
_SUPERCOLLIDER_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bsupercollider\b"), 3),
    (re.compile(r"\bsclang\b"), 3),
    (re.compile(r"\bscsynth\b"), 3),
    (re.compile(r"\bsynthdef\b"), 2),
)
_SONIC_PI_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bsonic(?:\s+pi|-pi|_pi)\b"), 3),
    (re.compile(r"\bsonic-pi\.net\b"), 3),
    (re.compile(r"\blive_loop\b"), 2),
)
_TIDALCYCLES_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\btidalcycles\b"), 3),
    (re.compile(r"\btidalcycles\.org\b"), 3),
    (
        re.compile(
            r"\btidal\s+cycles?\s+(?:pattern|live\s+coding|haskell|"
            r"mini-notation|superdirt)\b"
        ),
        3,
    ),
    (re.compile(r"\bsuperdirt\b"), 2),
)
_WEB_AUDIO_API_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bweb\s+audio\s+api\b"), 3),
    (re.compile(r"\baudiocontext\b"), 3),
    (re.compile(r"\bofflineaudiocontext\b"), 3),
    (re.compile(r"\baudioworklet\b"), 3),
    (re.compile(r"\b(?:oscillator|gain|analyser|convolver)node\b"), 2),
)
_P5_SOUND_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bp5\.sound\b"), 3),
    (re.compile(r"\bp5sound\b"), 3),
    (re.compile(r"\bp5\.soundfile\b"), 3),
    (re.compile(r"\bloadsound\s*\("), 2),
)
_ML5_JS_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bml5\.js\b"), 3),
    (re.compile(r"\bml5js\b"), 3),
    (
        re.compile(
            r"\bml5\.(?:bodypose|handpose|facemesh|imageclassifier|"
            r"soundclassifier|neuralnetwork)\b"
        ),
        3,
    ),
)
_TENSORFLOW_JS_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\btensorflow\.js\b"), 3),
    (re.compile(r"\btensorflowjs\b"), 3),
    (re.compile(r"\btfjs\b"), 3),
    (
        re.compile(
            r"\btf\.(?:tensor|sequential|loadlayersmodel|browser|layers|train)\b"
        ),
        3,
    ),
)
_COMFYUI_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bcomfyui\b"), 3),
    (re.compile(r"\bcomfy\s+ui\b"), 3),
    (re.compile(r"\bcomfy\s+(?:workflow|node|nodes|graph)\b"), 3),
)
_STABLE_DIFFUSION_WORKFLOWS_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bstable\s+diffusion\b"), 3),
    (re.compile(r"\bsdxl\b"), 3),
    (re.compile(r"\bsd\s*1\.5\b"), 2),
    (
        re.compile(r"\bdiffusers\s+(?:stable\s+diffusion|pipeline|workflow)\b"),
        3,
    ),
    (re.compile(r"\bcontrolnet\b"), 2),
)
_RUNWAY_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\brunwayml\b"), 3),
    (re.compile(r"\bdocs\.dev\.runwayml\.com\b"), 3),
    (
        re.compile(
            r"\brunway\s+(?:api|gen-?\d|video|generation|text-to-video|"
            r"image-to-video)\b"
        ),
        3,
    ),
)
_BLENDER_PYTHON_API_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bblender\s+python\b"), 3),
    (re.compile(r"\bblender\s+api\b"), 3),
    (re.compile(r"\bbpy\.(?:data|ops|types|context|props)\b"), 3),
    (re.compile(r"\bpython\s+script(?:ing)?\s+in\s+blender\b"), 3),
)
_UNREAL_BLUEPRINTS_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bunreal\s+blueprints?\b"), 3),
    (re.compile(r"\bblueprints?\s+in\s+unreal\b"), 3),
    (re.compile(r"\bunreal\s+.*\bblueprint\s+(?:class|node|graph|api)\b"), 3),
    (re.compile(r"\bblueprint\s+visual\s+scripting\b"), 3),
)
_ABLETON_LIVE_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bableton\s+live\b"), 3),
    (re.compile(r"\bableton\b"), 3),
    (re.compile(r"\bmax\s+for\s+live\b"), 3),
    (re.compile(r"\blive\s+(?:set|clip|session\s+view|arrangement\s+view)\b"), 2),
)
_VCV_RACK_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bvcv\s+rack\b"), 3),
    (re.compile(r"\bvcvrack\b"), 3),
    (re.compile(r"\beurorack\s+(?:patch|module|modular|synth|vcv)\b"), 2),
    (re.compile(r"\bcv\s*/\s*gate\b"), 2),
    (re.compile(r"\bcv\s+gate\b"), 2),
)
_GODOT_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bgodot\b"), 3),
    (re.compile(r"\bgodot\s+engine\b"), 3),
    (re.compile(r"\bgdscript\b"), 3),
    (re.compile(r"\bnode2d\b"), 2),
    (re.compile(r"\bscene\s+tree\b"), 2),
)
_RESOLUME_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bresolume\b"), 3),
    (re.compile(r"\bresolume\s+(?:arena|avenue|wire)\b"), 3),
    (re.compile(r"\bdxv3?\s+codec\b"), 2),
)
_MADMAPPER_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bmadmapper\b"), 3),
    (re.compile(r"\bmad\s+mapper\b"), 3),
    (re.compile(r"\bmadmapper\s+(?:surface|surfaces|quad|mapping)\b"), 3),
    (re.compile(r"\bprojection\s+mapping\s+in\s+madmapper\b"), 3),
)
_CABLES_GL_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bcables\.gl\b"), 3),
    (re.compile(r"\bcables\s+gl\b"), 3),
    (re.compile(r"\bcables\s+(?:patch|operator|operators|ops|op|graph)\b"), 3),
)
_PURE_DATA_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (
        re.compile(
            r"\bpure\s+data\s+(?:patch|patching|object|message|external|"
            r"abstraction|dsp|audio)\b"
        ),
        3,
    ),
    (re.compile(r"\bpuredata\b"), 3),
    (re.compile(r"\bpd\s+(?:patch|object|message|external|abstraction)\b"), 3),
    (re.compile(r"\.pd\b"), 2),
)
_INTENT_PATTERNS: tuple[
    tuple[CreativeCodingDomain, tuple[tuple[re.Pattern[str], int], ...]],
    ...,
] = (
    (CreativeCodingDomain.THREE_JS, _THREE_JS_PATTERNS),
    (CreativeCodingDomain.REACT_THREE_FIBER, _REACT_THREE_FIBER_PATTERNS),
    (CreativeCodingDomain.P5_JS, _P5_JS_PATTERNS),
    (CreativeCodingDomain.GLSL, _GLSL_PATTERNS),
    (CreativeCodingDomain.PROCESSING, _PROCESSING_PATTERNS),
    (CreativeCodingDomain.CANVAS_2D, _CANVAS_2D_PATTERNS),
    (CreativeCodingDomain.WEBGPU_WGSL, _WEBGPU_WGSL_PATTERNS),
    (CreativeCodingDomain.GSAP, _GSAP_PATTERNS),
    (CreativeCodingDomain.TONE_JS, _TONE_JS_PATTERNS),
    (CreativeCodingDomain.PIXI_JS, _PIXI_JS_PATTERNS),
    (CreativeCodingDomain.MATTER_JS, _MATTER_JS_PATTERNS),
    (CreativeCodingDomain.RAPIER, _RAPIER_PATTERNS),
    (CreativeCodingDomain.HYDRA, _HYDRA_PATTERNS),
    (CreativeCodingDomain.SHADERTOY, _SHADERTOY_PATTERNS),
    (CreativeCodingDomain.TOUCHDESIGNER, _TOUCHDESIGNER_PATTERNS),
    (CreativeCodingDomain.HOUDINI, _HOUDINI_PATTERNS),
    (
        CreativeCodingDomain.BLENDER_GEOMETRY_NODES,
        _BLENDER_GEOMETRY_NODES_PATTERNS,
    ),
    (CreativeCodingDomain.UNITY, _UNITY_PATTERNS),
    (CreativeCodingDomain.UNREAL, _UNREAL_PATTERNS),
    (CreativeCodingDomain.MAX_MSP, _MAX_MSP_PATTERNS),
    (CreativeCodingDomain.NOTCH, _NOTCH_PATTERNS),
    (CreativeCodingDomain.VVVV, _VVVV_PATTERNS),
    (CreativeCodingDomain.OPENFRAMEWORKS, _OPENFRAMEWORKS_PATTERNS),
    (CreativeCodingDomain.OPENRNDR, _OPENRNDR_PATTERNS),
    (CreativeCodingDomain.SUPERCOLLIDER, _SUPERCOLLIDER_PATTERNS),
    (CreativeCodingDomain.SONIC_PI, _SONIC_PI_PATTERNS),
    (CreativeCodingDomain.TIDALCYCLES, _TIDALCYCLES_PATTERNS),
    (CreativeCodingDomain.WEB_AUDIO_API, _WEB_AUDIO_API_PATTERNS),
    (CreativeCodingDomain.P5_SOUND, _P5_SOUND_PATTERNS),
    (CreativeCodingDomain.ML5_JS, _ML5_JS_PATTERNS),
    (CreativeCodingDomain.TENSORFLOW_JS, _TENSORFLOW_JS_PATTERNS),
    (CreativeCodingDomain.COMFYUI, _COMFYUI_PATTERNS),
    (
        CreativeCodingDomain.STABLE_DIFFUSION_WORKFLOWS,
        _STABLE_DIFFUSION_WORKFLOWS_PATTERNS,
    ),
    (CreativeCodingDomain.RUNWAY, _RUNWAY_PATTERNS),
    (CreativeCodingDomain.BLENDER_PYTHON_API, _BLENDER_PYTHON_API_PATTERNS),
    (CreativeCodingDomain.UNREAL_BLUEPRINTS, _UNREAL_BLUEPRINTS_PATTERNS),
    (CreativeCodingDomain.ABLETON_LIVE, _ABLETON_LIVE_PATTERNS),
    (CreativeCodingDomain.VCV_RACK, _VCV_RACK_PATTERNS),
    (CreativeCodingDomain.GODOT, _GODOT_PATTERNS),
    (CreativeCodingDomain.RESOLUME, _RESOLUME_PATTERNS),
    (CreativeCodingDomain.MADMAPPER, _MADMAPPER_PATTERNS),
    (CreativeCodingDomain.CABLES_GL, _CABLES_GL_PATTERNS),
    (CreativeCodingDomain.PURE_DATA, _PURE_DATA_PATTERNS),
)
_RELATED_DOMAIN_FALLBACKS: dict[
    CreativeCodingDomain,
    tuple[CreativeCodingDomain, ...],
] = {
    CreativeCodingDomain.THREE_JS: (CreativeCodingDomain.REACT_THREE_FIBER,),
    CreativeCodingDomain.REACT_THREE_FIBER: (CreativeCodingDomain.THREE_JS,),
}


@dataclass(frozen=True)
class DomainIntent:
    primary_domain: CreativeCodingDomain
    allowed_domains: tuple[CreativeCodingDomain, ...]


def detect_domain_intent(query: str) -> DomainIntent | None:
    """Return a conservative dominant-domain intent when one clearly exists."""

    normalized_query = _normalize_query(query)
    if not normalized_query:
        return None

    scores = {
        domain: _score_domain(normalized_query, patterns)
        for domain, patterns in _INTENT_PATTERNS
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary_domain, primary_score = ranked[0]
    secondary_score = ranked[1][1]

    if primary_score <= 0 or primary_score <= secondary_score:
        return None

    related_domains = _RELATED_DOMAIN_FALLBACKS.get(primary_domain, ())
    return DomainIntent(
        primary_domain=primary_domain,
        allowed_domains=(primary_domain, *related_domains),
    )


def detect_explicit_query_domains(query: str) -> tuple[CreativeCodingDomain, ...]:
    """Return all explicitly named query domains in stable enum order."""

    normalized_query = _normalize_query(query)
    if not normalized_query:
        return ()

    scores = [
        (domain, _score_domain(normalized_query, patterns))
        for domain, patterns in _INTENT_PATTERNS
    ]
    return tuple(domain for domain, score in scores if score > 0)


def resolve_effective_query_domains(
    *,
    query: str,
    selected_domains: Sequence[CreativeCodingDomain],
) -> tuple[CreativeCodingDomain, ...]:
    """Resolve explicit domains without erasing an intentional bridge scope."""

    explicit_domains = detect_explicit_query_domains(query)
    normalized: list[CreativeCodingDomain] = []
    for domain in selected_domains:
        if domain not in normalized:
            normalized.append(domain)
    selected = tuple(normalized)
    if not explicit_domains:
        return selected

    normalized_query = _normalize_query(query)
    explicit_domain = explicit_domains[0] if len(explicit_domains) == 1 else None
    bridge_scope = (
        explicit_domain is not None
        and len(selected) > 1
        and explicit_domain in selected
        and _DOMAIN_BRIDGE_ACTION_PATTERN.search(normalized_query) is not None
        and _DOMAIN_BRIDGE_COUNTERPART_PATTERN.search(normalized_query) is not None
    )
    if bridge_scope:
        return (
            explicit_domain,
            *(domain for domain in selected if domain != explicit_domain),
        )
    return explicit_domains


def _normalize_query(value: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", value.strip().lower())


def _score_domain(
    query: str,
    patterns: tuple[tuple[re.Pattern[str], int], ...],
) -> int:
    return sum(weight for pattern, weight in patterns if pattern.search(query))
