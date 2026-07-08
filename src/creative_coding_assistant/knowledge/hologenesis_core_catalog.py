"""Catalog rows for the V8.7 HoloGenesis Creative Operating System."""

from __future__ import annotations

from collections.abc import Mapping

ROADMAP_CLASSIFICATION_ROWS: Mapping[str, tuple[str, str, bool, bool]] = {
    "Unified Symbolic Graph": (
        "implemented_report_behavior",
        "Implemented as a unified graph projection over V8.2 motif mappings and V8.5 symbol nodes.",
        False,
        False,
    ),
    "Unified Sacred Knowledge Graph": (
        "implemented_report_behavior",
        "Implemented as a report graph over V8.1 distilled knowledge and bounded sacred-knowledge claims.",
        False,
        False,
    ),
    "Unified Geometry Graph": (
        "implemented_report_behavior",
        "Implemented as a graph projection over V8.3 geometry patterns and V8.6 animation guidance.",
        False,
        False,
    ),
    "Unified Narrative Graph": (
        "implemented_report_behavior",
        "Implemented as a graph projection over V8.5 narrative scenes and V8.6 scene composition.",
        False,
        False,
    ),
    "Unified Installation Graph": (
        "implemented_report_behavior",
        "Implemented as a graph projection over V8.4 installation guidance and V8.6 audience journey planning.",
        False,
        False,
    ),
    "Creative Blackboard": (
        "implemented_report_behavior",
        "Implemented as report-only blackboard entries with no persistence or runtime synchronization.",
        False,
        False,
    ),
    "Symbolic Scheduler": (
        "implemented_report_behavior",
        "Implemented as deterministic symbolic planning order, not as task execution or workflow control.",
        False,
        False,
    ),
    "Creative Planner": (
        "implemented_report_behavior",
        "Implemented as staged creative planning derived from unified graphs and existing V3 planning metadata.",
        False,
        False,
    ),
    "Creative Router": (
        "implemented_report_behavior",
        "Implemented as advisory creative path recommendations, explicitly not provider/model routing.",
        False,
        False,
    ),
    "Artistic Decision Engine": (
        "implemented_report_behavior",
        "Implemented as explainable artistic decisions with graph provenance and confidence.",
        False,
        False,
    ),
    "Curatorial Intelligence Engine": (
        "implemented_report_behavior",
        "Implemented as bounded curatorial assessment over theme, audience, and installation coherence.",
        False,
        False,
    ),
    "Curatorial Reasoning Engine": (
        "implemented_report_behavior",
        "Implemented as deterministic rationale linking symbolic, geometric, narrative, and audiovisual evidence.",
        False,
        False,
    ),
    "Curatorial Validation Engine": (
        "implemented_report_behavior",
        "Implemented as validation findings and readiness scoring, not museum certification.",
        False,
        False,
    ),
    "Curatorial Explainability": (
        "implemented_report_behavior",
        "Implemented as explainability notes attached to assessments, decisions, and project bundle outputs.",
        False,
        False,
    ),
    "Mystical Consistency Engine": (
        "implemented_report_behavior",
        "Implemented only as bounded/non-authoritative consistency framing with explicit claim-risk handling.",
        False,
        False,
    ),
    "Symbolic Explainability Engine": (
        "implemented_report_behavior",
        "Implemented by carrying V8.2 symbolic evidence into graph nodes, decisions, and prompt lines.",
        False,
        False,
    ),
    "Aesthetic Evaluation Engine": (
        "implemented_report_behavior",
        "Implemented as bounded aesthetic review using V8.6 visual, audio, and audience plans.",
        False,
        False,
    ),
    "Installation Simulation Engine": (
        "implemented_report_behavior",
        "Implemented as qualitative report/planning behavior only; no physical, crowd, or DCC simulation runs.",
        False,
        False,
    ),
    "Multi-Domain Creative Synthesis": (
        "implemented_report_behavior",
        "Implemented by composing V8.1-V8.6 reports across symbolic, geometry, "
        "architecture, narrative, audiovisual, and delivery layers.",
        False,
        False,
    ),
    "Unity Integration Layer": (
        "export_planning_only",
        "Audited as export/recommendation planning only; no Unity process, API, scene, or asset execution exists.",
        False,
        False,
    ),
    "Unreal Integration Layer": (
        "export_planning_only",
        "Audited as export/recommendation planning only; no Unreal editor, Blueprint, or project execution exists.",
        False,
        False,
    ),
    "TouchDesigner Integration Layer": (
        "export_planning_only",
        "Audited as export/recommendation planning only; no TouchDesigner network or operator execution exists.",
        False,
        False,
    ),
    "Blender Integration Layer": (
        "export_planning_only",
        "Audited as export/recommendation planning only; no Blender Python, geometry-node, or file execution exists.",
        False,
        False,
    ),
    "Houdini Integration Layer": (
        "export_planning_only",
        "Audited as export/recommendation planning only; no Houdini session, HDA, "
        "or procedural graph execution exists.",
        False,
        False,
    ),
    "MCP Creative Tool Layer": (
        "export_planning_only",
        "Audited as tool-layer recommendation planning only; no MCP creative tool invocation is implemented.",
        False,
        False,
    ),
    "Creative Project Generator": (
        "implemented_report_behavior",
        "Implemented as a typed project bundle outline, not filesystem generation.",
        False,
        False,
    ),
    "Installation Quality Scoring": (
        "implemented_report_behavior",
        "Implemented as bounded readiness scoring for review, not objective quality certification.",
        False,
        False,
    ),
    "Museum Readiness Evaluation": (
        "implemented_report_behavior",
        "Implemented as advisory museum-readiness review notes, not institutional approval.",
        False,
        False,
    ),
    "International Exhibition Readiness": (
        "implemented_report_behavior",
        "Implemented as advisory international-exhibition readiness notes, not legal/logistics certification.",
        False,
        False,
    ),
    "Creative Pipeline Planner": (
        "implemented_report_behavior",
        "Implemented as project-bundle pipeline steps and symbolic schedule outputs.",
        False,
        False,
    ),
    "Workflow Recommendation": (
        "implemented_report_behavior",
        "Implemented as recommendations without changing or controlling runtime workflow graphs.",
        False,
        False,
    ),
    "Tool Recommendation": (
        "implemented_report_behavior",
        "Implemented as advisory tool choices, including external tools only as non-executing recommendations.",
        False,
        False,
    ),
    "Tech Stack Recommendation": (
        "implemented_report_behavior",
        "Implemented as bounded stack recommendations derived from requested domains and existing preview support.",
        False,
        False,
    ),
    "Creative Tradeoff Analysis": (
        "implemented_report_behavior",
        "Implemented as bundle tradeoff notes and decision rationales, not automatic selection or enforcement.",
        False,
        False,
    ),
    "Creative Director Surface": (
        "implemented_report_behavior",
        "Implemented as director-facing report sections and HITL questions, not a new frontend surface.",
        False,
        False,
    ),
    "Project Architecture Generator": (
        "implemented_report_behavior",
        "Implemented as architecture outline generation inside the typed project bundle.",
        False,
        False,
    ),
    "Portfolio Generator": (
        "implemented_report_behavior",
        "Implemented as portfolio outline generation inside the typed project bundle.",
        False,
        False,
    ),
    "README Generator": (
        "implemented_report_behavior",
        "Implemented as README outline generation inside the typed project bundle.",
        False,
        False,
    ),
    "Capstone Output Generator": (
        "implemented_report_behavior",
        "Implemented as capstone output outline generation inside the typed project bundle.",
        False,
        False,
    ),
    "Future HoloMind Hooks": (
        "future_hook_only",
        "Represented only as explicit future hook posture; no HoloMind behavior is implemented.",
        False,
        False,
    ),
    "Creative Project Bundle Generator": (
        "implemented_report_behavior",
        "Implemented as a typed project bundle with architecture, portfolio, "
        "README, capstone, pipeline, and research sections.",
        False,
        False,
    ),
    "Creative Research Mode": (
        "implemented_report_behavior",
        "Implemented as bounded research-mode planning and follow-up prompts "
        "without browsing, fetching, or storage writes.",
        False,
        False,
    ),
    "Creative Reference Discovery": (
        "implemented_report_behavior",
        "Implemented as deterministic reference discovery query planning, not live source discovery.",
        False,
        False,
    ),
}

EXTERNAL_INTEGRATION_ROWS: Mapping[str, tuple[str, str, str, tuple[str, ...]]] = {
    "unity": (
        "Unity",
        "Export planning for scene hierarchy, interaction intents, shader/material "
        "notes, and asset handoff checklists.",
        "No Unity editor automation, package export, asset import, runtime "
        "playback, or C# script generation is executed.",
        (
            "Prepare scene hierarchy notes from installation graph nodes.",
            "Keep browser-internal preview as the implemented validation path.",
        ),
    ),
    "unreal": (
        "Unreal",
        "Export planning for world composition, Blueprint candidates, lighting "
        "notes, and interactive installation handoff.",
        "No Unreal editor automation, Blueprint creation, project packaging, or runtime playback is executed.",
        (
            "Map narrative phases to level/sequence planning notes.",
            "Treat Unreal as downstream manual production planning only.",
        ),
    ),
    "touchdesigner": (
        "TouchDesigner",
        "Export planning for node-network ideas, audio-reactive mappings, OSC/MIDI "
        "notes, and installation control surfaces.",
        "No TouchDesigner network, operator graph, tox file, or live control process is executed.",
        (
            "Translate V8.6 spatial audio and visual language into node-network notes.",
            "Keep all TouchDesigner work as manual follow-up.",
        ),
    ),
    "blender": (
        "Blender",
        "Export planning for geometry, camera, material, lighting, and render references.",
        "No Blender Python call, geometry-node graph, file creation, or render execution is performed.",
        (
            "Use V8.3 geometry and V8.4 architecture patterns as modeling notes.",
            "Keep actual asset production outside V8.7.",
        ),
    ),
    "houdini": (
        "Houdini",
        "Export planning for procedural geometry, growth systems, simulation "
        "notes, and installation-scale asset strategy.",
        "No Houdini session, HDA creation, procedural cook, simulation, or file export is executed.",
        (
            "Translate recursive geometry and architecture topology into procedural notes.",
            "Keep simulation claims qualitative unless a future integration exists.",
        ),
    ),
    "mcp_creative_tool_layer": (
        "MCP Creative Tool Layer",
        "Recommendation planning for future creative tool connectors and handoff boundaries.",
        "No MCP creative tool invocation, connector installation, tool call, or external mutation is implemented.",
        (
            "List tool-layer candidates as review notes only.",
            "Do not claim live integration without a callable tool path.",
        ),
    ),
}

UNSUPPORTED_CLAIM_TOKENS = frozenset(
    {
        "ableton",
        "api",
        "automate",
        "blender",
        "dcc",
        "execute",
        "export",
        "external",
        "holomind",
        "holoiverse",
        "houdini",
        "live integration",
        "mcp",
        "run unity",
        "touchdesigner",
        "unity",
        "unreal",
    }
)
