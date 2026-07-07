"""Catalog rows for the V8.6 immersive audiovisual composer."""

from __future__ import annotations

from collections.abc import Mapping

ROADMAP_CLASSIFICATION_ROWS: Mapping[str, tuple[str, str, bool, bool]] = {
    "Scene Graph Composer": (
        "implemented_runtime_behavior",
        "Implemented as typed scene nodes and transitions composed from V8.2-V8.5 guidance.",
        False,
        False,
    ),
    "Visual Language Engine": (
        "implemented_runtime_behavior",
        "Implemented as a composed visual identity plan using translation, style, shader, and symbolic cues.",
        False,
        False,
    ),
    "Sacred Lighting Engine": (
        "implemented_runtime_behavior",
        "Implemented as bounded lighting guidance from geometry, shader presets, and architecture cues.",
        False,
        False,
    ),
    "Symbolic Color Engine": (
        "implemented_runtime_behavior",
        "Implemented as symbolic color guidance from V8.2 motifs and request-visible palette cues.",
        False,
        False,
    ),
    "Geometry Animation Engine": (
        "implemented_runtime_behavior",
        "Implemented as geometry-to-motion guidance reusing V8.3 geometry patterns.",
        False,
        False,
    ),
    "Particle Symbolism Engine": (
        "implemented_runtime_behavior",
        "Implemented as particle-behavior guidance bound to symbolic, geometric, and narrative roles.",
        False,
        False,
    ),
    "Spatial Audio Planner": (
        "implemented_runtime_behavior",
        "Implemented as explicit-activation spatial audio planning without audio playback execution.",
        False,
        False,
    ),
    "Sacred Music Mapping": (
        "implemented_runtime_behavior",
        "Implemented as harmonic and ritual-timing mapping guidance without musicological authority claims.",
        False,
        False,
    ),
    "Quadrivium Engine": (
        "implemented_runtime_behavior",
        "Implemented as bounded number/geometry/music/astronomy correspondence guidance.",
        False,
        False,
    ),
    "Planetary Motion Engine": (
        "implemented_runtime_behavior",
        "Implemented as orbital and phase-based motion guidance, not astronomical simulation.",
        False,
        False,
    ),
    "Ritual Timing Engine": (
        "implemented_runtime_behavior",
        "Implemented as temporal cueing and transition timing without ritual efficacy claims.",
        False,
        False,
    ),
    "Transition Composer": (
        "implemented_runtime_behavior",
        "Implemented as transitions between composed scene graph nodes with continuity guidance.",
        False,
        False,
    ),
    "Multi-layer Scene Composition": (
        "implemented_runtime_behavior",
        "Implemented across symbolic, geometric, architectural, narrative, visual, audio, and audience layers.",
        False,
        False,
    ),
    "Installation Flow Engine": (
        "implemented_runtime_behavior",
        "Implemented as bounded installation flow guidance reusing V8.4 architecture and V8.5 narrative.",
        False,
        False,
    ),
    "Audience Journey Planner": (
        "implemented_runtime_behavior",
        "Implemented as audience path and attention guidance without measurement or psychology claims.",
        False,
        False,
    ),
    "Explainable Artistic Decisions": (
        "implemented_runtime_behavior",
        "Implemented as typed artistic decisions with rationale, evidence, and reused surface IDs.",
        False,
        False,
    ),
    "Creative Style Profiles": (
        "reused_existing_runtime",
        "Reuses the existing V6.2 style profile metadata as advisory style references.",
        False,
        False,
    ),
    "Embodied Experience Engine": (
        "implemented_runtime_behavior",
        "Implemented as embodied attention, movement, scale, and sensory-load guidance.",
        False,
        False,
    ),
    "Spatial Dramaturgy Engine": (
        "implemented_runtime_behavior",
        "Implemented as spatial dramaturgy using V8.4 topology and V8.5 installation narrative.",
        False,
        False,
    ),
    "Temporal Dramaturgy Engine": (
        "implemented_runtime_behavior",
        "Implemented as ordered temporal dramaturgy from V8.5 scenes and V3 audio-visual phases.",
        False,
        False,
    ),
    "Emotional Resonance Engine": (
        "implemented_runtime_behavior",
        "Implemented as creative emotional resonance guidance, not psychological assessment.",
        False,
        False,
    ),
    "Audience Flow Simulation": (
        "implemented_runtime_behavior",
        "Implemented as qualitative audience-flow simulation and bottleneck/attention guidance.",
        False,
        False,
    ),
    "Internal Three.js Preview": (
        "reused_existing_runtime",
        "Reuses the existing bounded Three.js-compatible browser preview foundation.",
        False,
        False,
    ),
    "Internal p5.js Preview": (
        "reused_existing_runtime",
        "Reuses the existing p5-compatible browser preview foundation.",
        False,
        False,
    ),
    "Internal GLSL Preview": (
        "reused_existing_runtime",
        "Reuses the existing WebGL fragment shader preview foundation.",
        False,
        False,
    ),
    "Internal Hydra Preview (se supportato)": (
        "reused_existing_runtime",
        "Reuses the existing bounded Hydra-compatible browser preview foundation.",
        False,
        False,
    ),
    "Artifact Preview Loop": (
        "partial_reusable",
        "Existing preview metadata and stream updates are reusable; V8.6 does not add workflow loops.",
        False,
        False,
    ),
    "Live Preview": (
        "reused_existing_runtime",
        "Reuses existing live preview runtime status, frame, controller, and sandbox behavior.",
        False,
        False,
    ),
    "Critique \u2192 Refinement \u2192 Preview Loop": (
        "partial_reusable",
        "Existing refinement and preview metadata are reusable; automatic loop control remains out of scope.",
        False,
        False,
    ),
    "Browser Preview Sandbox": (
        "reused_existing_runtime",
        "Reuses the existing browser sandbox iframe/document runtime foundation.",
        False,
        False,
    ),
    "Interactive Iteration": (
        "partial_reusable",
        "Existing reload/reset controller supports iteration; V8.6 does not add interactive canvas input.",
        False,
        False,
    ),
    "Demo Preview Flow": (
        "partial_reusable",
        "Existing preview surfaces support demo use; V8.8 owns dedicated showcase/demo flow.",
        False,
        False,
    ),
    "Internal Export Preview": (
        "partial_reusable",
        "Existing export and panel metadata are reusable; V8.6 does not implement export rendering.",
        False,
        False,
    ),
    "Preview Runtime Validation": (
        "reused_existing_runtime",
        "Reuses existing renderer routing, source mismatch, and sandbox validation behavior.",
        False,
        False,
    ),
    "Preview Error Recovery": (
        "reused_existing_runtime",
        "Reuses existing recoverable workstation errors, reload, reset, and controller behavior.",
        False,
        False,
    ),
    "Multi Preview Comparison": (
        "reused_existing_runtime",
        "Reuses the existing multi-preview comparison workspace and model.",
        False,
        False,
    ),
}

PREVIEW_AUDIT_ROWS: Mapping[str, tuple[str, bool, str, str, tuple[str, ...]]] = {
    "Internal Three.js Preview": (
        "partially_implemented",
        True,
        "A controlled Three.js-compatible sandbox/runtime exists for bounded scene primitives and signals.",
        "Reuse for V8.6 preview planning; do not add or duplicate Three.js runtime code.",
        (
            "clients/nextjs/src/lib/preview-renderers.ts",
            "clients/nextjs/src/lib/preview-runtime-adapters.ts",
            "clients/nextjs/src/lib/preview-sandbox-runtime.ts",
            "clients/nextjs/public/preview-sandbox.html",
        ),
    ),
    "Internal p5.js Preview": (
        "already_implemented",
        True,
        "A controlled p5-compatible browser runtime and sandbox mount path already exist.",
        "Reuse for V8.6 preview planning; no new p5 runtime behavior is needed.",
        (
            "clients/nextjs/src/lib/preview-renderers.ts",
            "clients/nextjs/src/lib/preview-runtime-adapters.ts",
            "clients/nextjs/src/lib/preview-sandbox-runtime.ts",
        ),
    ),
    "Internal GLSL Preview": (
        "already_implemented",
        True,
        "A WebGL fragment shader runtime and sandbox preparation path already exist.",
        "Reuse for V8.6 preview planning; no new GLSL runtime behavior is needed.",
        (
            "clients/nextjs/src/lib/preview-renderers.ts",
            "clients/nextjs/src/lib/preview-runtime-adapters.ts",
            "clients/nextjs/src/lib/preview-sandbox-runtime.ts",
        ),
    ),
    "Internal Hydra Preview": (
        "partially_implemented",
        True,
        "A bounded Hydra-compatible parser and runtime plan exist for supported chains.",
        "Reuse supported Hydra subsets; keep unsupported Hydra behavior classified as bounded.",
        (
            "clients/nextjs/src/lib/hydra-runtime.ts",
            "clients/nextjs/src/lib/preview-runtime-adapters.ts",
            "clients/nextjs/src/lib/preview-sandbox-runtime.ts",
        ),
    ),
    "Browser Preview Sandbox": (
        "already_implemented",
        True,
        "The same-origin preview sandbox iframe/document isolates supported browser runtimes.",
        "Reuse the sandbox; V8.6 only references its availability in planning metadata.",
        (
            "clients/nextjs/src/lib/preview-sandbox-runtime.ts",
            "clients/nextjs/public/preview-sandbox.html",
        ),
    ),
    "Live Preview Loop": (
        "already_implemented",
        True,
        "Runtime status, frame samples, session controls, and stream preview updates already exist.",
        "Reuse live loop foundations without changing workflow execution or preview streaming.",
        (
            "clients/nextjs/src/lib/preview-runtime.ts",
            "clients/nextjs/src/lib/preview-controller.ts",
            "clients/nextjs/src/lib/preview-runtime-diagnostics.ts",
        ),
    ),
    "Preview Runtime Validation": (
        "already_implemented",
        True,
        "Renderer routing, source classification, sandbox rejection, and runtime mismatch checks exist.",
        "Reuse validation evidence in V8.6 reports; do not duplicate source validators.",
        (
            "clients/nextjs/src/lib/preview-renderers.ts",
            "clients/nextjs/src/lib/preview-source-classification.ts",
            "clients/nextjs/src/lib/preview-sandbox-runtime.ts",
        ),
    ),
    "Preview Error Recovery": (
        "already_implemented",
        True,
        "Recoverable workstation errors plus reload, reset, restart, and clear controls already exist.",
        "Reuse recovery posture; V8.6 does not add runtime repair behavior.",
        (
            "clients/nextjs/src/lib/preview-controller.ts",
            "clients/nextjs/src/lib/workstation-errors.ts",
            "clients/nextjs/src/lib/preview-sandbox-runtime.ts",
        ),
    ),
    "Multi Preview Comparison": (
        "already_implemented",
        True,
        "Multi-preview candidate modeling and comparison workspace already exist.",
        "Reuse comparison support for planning; V8.6 does not create a new comparison UI.",
        (
            "clients/nextjs/src/lib/multi-preview-comparison.ts",
            "clients/nextjs/src/components/multi-preview-comparison-workspace.tsx",
        ),
    ),
}

UNSUPPORTED_COMPOSER_CLAIM_TOKENS = frozenset(
    {
        "ableton",
        "blender",
        "cad",
        "dcc",
        "houdini",
        "hologenesis os",
        "holomind",
        "holoiverse",
        "mcp",
        "touchdesigner",
        "unity",
        "unreal",
    }
)

__all__ = [
    "PREVIEW_AUDIT_ROWS",
    "ROADMAP_CLASSIFICATION_ROWS",
    "UNSUPPORTED_COMPOSER_CLAIM_TOKENS",
]
