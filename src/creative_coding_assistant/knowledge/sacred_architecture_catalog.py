"""Static V8.4 sacred architecture catalog rows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ArchitecturePatternRow:
    label: str
    family: str
    taxonomy_path: tuple[str, ...]
    spatial_intent: str
    proportions: tuple[str, ...]
    plan: tuple[str, ...]
    axes: tuple[str, ...]
    thresholds: tuple[str, ...]
    center_periphery: tuple[str, ...]
    topology: tuple[str, ...]
    geometry_mappings: tuple[str, ...]
    symbolic_mappings: tuple[str, ...]
    installation: tuple[str, ...]
    reverse_engineering_cues: tuple[str, ...]
    parameters: tuple[str, ...]
    runtimes: tuple[str, ...]
    notes: tuple[str, ...]
    boundary: str


UNSUPPORTED_ARCHITECTURE_CLAIM_TOKENS = frozenset(
    {
        "scan",
        "scanned",
        "lidar",
        "photogrammetry",
        "photogrammetric",
        "survey",
        "surveyed",
        "reconstruct",
        "reconstruction",
        "measure",
        "measured",
        "accurate",
        "exact",
        "image",
        "images",
        "as-built",
        "cad",
        "bim",
        "photo",
        "photos",
        "photograph",
        "photographs",
        "picture",
        "pictures",
        "blueprint",
        "blueprints",
    }
)


ARCHITECTURE_ALIASES: Mapping[str, tuple[str, ...]] = {
    "architecture": ("axis_threshold_procession", "modular_hypostyle_grid"),
    "architectural": ("axis_threshold_procession", "modular_hypostyle_grid"),
    "altar": ("axis_threshold_procession", "compressed_threshold_chamber_sequence"),
    "antechamber": ("compressed_threshold_chamber_sequence",),
    "axis": ("axis_threshold_procession",),
    "axial": ("axis_threshold_procession",),
    "basilica": ("axis_threshold_procession", "gothic_vertical_light_axis"),
    "bay": ("modular_hypostyle_grid", "gothic_vertical_light_axis"),
    "cathedral": ("gothic_vertical_light_axis", "axis_threshold_procession"),
    "center": ("central_periphery_mandala_plan",),
    "central": ("central_periphery_mandala_plan",),
    "chamber": ("compressed_threshold_chamber_sequence",),
    "cloister": ("courtyard_cloister_perimeter",),
    "column": ("modular_hypostyle_grid",),
    "columns": ("modular_hypostyle_grid",),
    "courtyard": ("courtyard_cloister_perimeter", "tessellated_courtyard_field"),
    "crossing": ("axis_threshold_procession", "gothic_vertical_light_axis"),
    "dome": ("central_periphery_mandala_plan",),
    "environment": ("pavilion_field_environment", "installation_gallery_sequence"),
    "exhibit": ("installation_gallery_sequence",),
    "exhibition": ("installation_gallery_sequence",),
    "floor": ("modular_hypostyle_grid", "axis_threshold_procession"),
    "floor-plan": ("modular_hypostyle_grid", "axis_threshold_procession"),
    "floorplan": ("modular_hypostyle_grid", "axis_threshold_procession"),
    "gallery": ("installation_gallery_sequence",),
    "gate": ("compressed_threshold_chamber_sequence", "axis_threshold_procession"),
    "gothic": ("gothic_vertical_light_axis",),
    "grid": ("modular_hypostyle_grid",),
    "hall": ("modular_hypostyle_grid", "axis_threshold_procession"),
    "hypostyle": ("modular_hypostyle_grid",),
    "installation": ("installation_gallery_sequence", "pavilion_field_environment"),
    "islamic": ("tessellated_courtyard_field", "courtyard_cloister_perimeter"),
    "labyrinth": ("labyrinthine_procession_path",),
    "layout": ("axis_threshold_procession", "modular_hypostyle_grid"),
    "light": ("gothic_vertical_light_axis", "pavilion_field_environment"),
    "mandala": ("central_periphery_mandala_plan",),
    "maze": ("labyrinthine_procession_path",),
    "mosque": ("tessellated_courtyard_field", "courtyard_cloister_perimeter"),
    "museum": ("installation_gallery_sequence",),
    "nave": ("axis_threshold_procession", "gothic_vertical_light_axis"),
    "path": ("labyrinthine_procession_path", "installation_gallery_sequence"),
    "pavilion": ("pavilion_field_environment",),
    "perimeter": ("courtyard_cloister_perimeter",),
    "periphery": ("central_periphery_mandala_plan", "courtyard_cloister_perimeter"),
    "plan": ("modular_hypostyle_grid", "axis_threshold_procession"),
    "planimetry": ("modular_hypostyle_grid", "axis_threshold_procession"),
    "portal": ("compressed_threshold_chamber_sequence",),
    "procession": ("axis_threshold_procession", "labyrinthine_procession_path"),
    "processional": ("axis_threshold_procession", "labyrinthine_procession_path"),
    "quadrangle": ("courtyard_cloister_perimeter",),
    "radial": ("central_periphery_mandala_plan",),
    "rose": ("gothic_vertical_light_axis",),
    "sanctuary": ("compressed_threshold_chamber_sequence", "axis_threshold_procession"),
    "sequence": ("installation_gallery_sequence", "compressed_threshold_chamber_sequence"),
    "site": ("pavilion_field_environment", "installation_gallery_sequence"),
    "spatial": ("axis_threshold_procession", "installation_gallery_sequence"),
    "symmetry": ("axis_threshold_procession", "central_periphery_mandala_plan"),
    "temple": ("axis_threshold_procession", "central_periphery_mandala_plan"),
    "tessellation": ("tessellated_courtyard_field",),
    "threshold": ("compressed_threshold_chamber_sequence", "axis_threshold_procession"),
    "viewer": ("installation_gallery_sequence",),
    "window": ("gothic_vertical_light_axis",),
}


GEOMETRY_PATTERN_TO_ARCHITECTURE: Mapping[str, tuple[str, ...]] = {
    "golden_ratio": ("harmonic_proportion_plan", "central_periphery_mandala_plan"),
    "harmonic_proportion": ("harmonic_proportion_plan", "axis_threshold_procession"),
    "sacred_polygon_circle_grid": ("modular_hypostyle_grid", "central_periphery_mandala_plan"),
    "mandala_generator": ("central_periphery_mandala_plan",),
    "yantra_generator": ("central_periphery_mandala_plan", "compressed_threshold_chamber_sequence"),
    "sacred_tessellation": ("tessellated_courtyard_field", "modular_hypostyle_grid"),
    "islamic_tessellation": ("tessellated_courtyard_field",),
    "recursive_spiral": ("labyrinthine_procession_path", "central_periphery_mandala_plan"),
    "fractal_structure": ("pavilion_field_environment",),
    "l_system_growth": ("pavilion_field_environment",),
    "flow_field": ("pavilion_field_environment", "installation_gallery_sequence"),
    "reaction_diffusion": ("pavilion_field_environment",),
    "cellular_automata": ("modular_hypostyle_grid", "pavilion_field_environment"),
    "particle_geometry": ("installation_gallery_sequence", "pavilion_field_environment"),
    "platonic_solid": ("pavilion_field_environment", "central_periphery_mandala_plan"),
    "temple_axis": ("axis_threshold_procession", "compressed_threshold_chamber_sequence"),
    "gothic_geometry": ("gothic_vertical_light_axis",),
    "egyptian_grid": ("modular_hypostyle_grid", "axis_threshold_procession"),
    "biological_growth": ("pavilion_field_environment",),
}


SYMBOL_TO_ARCHITECTURE: Mapping[str, tuple[str, ...]] = {
    "ascent": ("gothic_vertical_light_axis",),
    "axis": ("axis_threshold_procession",),
    "center": ("central_periphery_mandala_plan",),
    "circumference": ("central_periphery_mandala_plan", "courtyard_cloister_perimeter"),
    "gate": ("compressed_threshold_chamber_sequence", "axis_threshold_procession"),
    "grid": ("modular_hypostyle_grid",),
    "labyrinth": ("labyrinthine_procession_path",),
    "lattice": ("tessellated_courtyard_field", "modular_hypostyle_grid"),
    "mandala": ("central_periphery_mandala_plan",),
    "mirror": ("axis_threshold_procession", "central_periphery_mandala_plan"),
    "network": ("pavilion_field_environment",),
    "orbit": ("central_periphery_mandala_plan",),
    "pulse": ("installation_gallery_sequence", "compressed_threshold_chamber_sequence"),
    "threshold": ("compressed_threshold_chamber_sequence", "axis_threshold_procession"),
    "void": ("courtyard_cloister_perimeter", "pavilion_field_environment"),
}


PATTERN_ROWS: Mapping[str, ArchitecturePatternRow] = {
    "harmonic_proportion_plan": ArchitecturePatternRow(
        label="Harmonic Proportion Plan",
        family="proportion",
        taxonomy_path=("sacred architecture", "proportion", "harmonic plan"),
        spatial_intent="Use harmonic ratios as explicit spacing, bay, room, and path-length guidance.",
        proportions=("Coordinate bay, court, chamber, and path modules by 1:1, 2:1, 3:2, 4:3, 5:4, or phi.",),
        plan=("Derive plan zones from a base module before adding secondary detail.",),
        axes=("Let primary and cross axes inherit the same module so symmetry remains readable.",),
        thresholds=("Place thresholds at ratio-derived intervals rather than arbitrary positions.",),
        center_periphery=("Scale center, ring, and boundary zones by declared ratio relationships.",),
        topology=(
            "Represent ratios as named spatial relationships between entry, center, "
            "periphery, and service zones.",
        ),
        geometry_mappings=("harmonic_proportion", "golden_ratio"),
        symbolic_mappings=("axis", "center", "grid"),
        installation=("Use ratio-derived distances for viewer pause points, projection zones, and object spacing.",),
        reverse_engineering_cues=("List repeated bays, room widths, path lengths, and visible proportional families.",),
        parameters=("base_module", "ratio_set", "bay_count", "threshold_interval"),
        runtimes=("p5.js", "Canvas 2D", "Three.js"),
        notes=("Expose chosen ratios as creative parameters; do not claim discovered measurements.",),
        boundary="Harmonic proportion guidance is compositional, not measured architectural proof.",
    ),
    "axis_threshold_procession": ArchitecturePatternRow(
        label="Axis, Threshold, and Procession",
        family="processional",
        taxonomy_path=("sacred architecture", "axis", "procession"),
        spatial_intent="Organize spatial experience as entry, alignment, crossings, reveal, center, and return.",
        proportions=("Use path length, bay rhythm, and threshold count as proportional scaffolds.",),
        plan=("Arrange major zones along one legible primary axis with optional secondary cross-axis.",),
        axes=("Make the primary axis visible through alignment, sightline, lighting, or repeated bays.",),
        thresholds=("Model gates, portals, curtains, courts, or color shifts as discrete threshold events.",),
        center_periphery=("Treat the destination or gathering node as a center framed by quieter periphery.",),
        topology=("Entry -> threshold -> procession -> center is the default semantic edge chain.",),
        geometry_mappings=("temple_axis", "harmonic_proportion"),
        symbolic_mappings=("axis", "threshold", "gate", "center"),
        installation=("Stage sound, light, density, or interaction changes at each threshold crossing.",),
        reverse_engineering_cues=(
            "Identify entrance, axis line, repeated bays, threshold events, and terminal focus.",
        ),
        parameters=("axis_angle", "threshold_count", "bay_count", "procession_length"),
        runtimes=("p5.js", "Canvas 2D", "Three.js"),
        notes=("Use procession as spatial pacing, not V8.5 narrative generation.",),
        boundary="Procession guidance is spatial sequencing only and does not simulate ritual efficacy.",
    ),
    "central_periphery_mandala_plan": ArchitecturePatternRow(
        label="Center and Periphery Mandala Plan",
        family="radial",
        taxonomy_path=("sacred architecture", "center periphery", "radial plan"),
        spatial_intent="Use center, rings, radial sectors, and perimeter zones as a spatial hierarchy.",
        proportions=("Set ring radii, sector count, and perimeter thickness from bounded ratio sets.",),
        plan=("Build a central focal node with nested rings, radial spokes, and perimeter circulation.",),
        axes=("Use radial axes, mirror axes, or cardinal lines to anchor the center.",),
        thresholds=("Mark transitions between rings as soft thresholds or controlled apertures.",),
        center_periphery=("Protect the center as a focal anchor while giving periphery a clear circulation role.",),
        topology=("Center connects to rings, rings connect to perimeter, perimeter returns to entry.",),
        geometry_mappings=("mandala_generator", "yantra_generator", "sacred_polygon_circle_grid"),
        symbolic_mappings=("center", "circumference", "mandala", "orbit"),
        installation=("Place the strongest sensory event at center and use outer rings for approach or reflection.",),
        reverse_engineering_cues=(
            "Look for central anchors, nested boundaries, radial symmetry, and perimeter paths.",
        ),
        parameters=("ring_count", "radial_symmetry", "center_weight", "perimeter_width"),
        runtimes=("p5.js", "Canvas 2D", "Three.js"),
        notes=("Keep radial hierarchy inspectable before adding ornamental layers.",),
        boundary="Mandala-plan guidance is bounded spatial composition, not doctrine.",
    ),
    "courtyard_cloister_perimeter": ArchitecturePatternRow(
        label="Courtyard and Cloister Perimeter",
        family="courtyard",
        taxonomy_path=("sacred architecture", "void", "courtyard perimeter"),
        spatial_intent="Use an open void, perimeter walk, and inward-facing rooms to structure attention.",
        proportions=("Balance void size, perimeter depth, and opening cadence with a declared module.",),
        plan=("Reserve a central void and wrap circulation, frames, or chambers around it.",),
        axes=("Use corner-to-corner, entry-to-void, or perimeter axes rather than a single terminal axis.",),
        thresholds=("Let entry, arcade, court edge, and chamber door become layered thresholds.",),
        center_periphery=("Treat the empty court as active center and the perimeter as contemplative edge.",),
        topology=("Entry connects to perimeter; perimeter contains court; chambers branch from perimeter.",),
        geometry_mappings=("sacred_polygon_circle_grid", "harmonic_proportion"),
        symbolic_mappings=("circumference", "void", "gate"),
        installation=("Use the void for audience dwell time, projection spill, or a central quiet zone.",),
        reverse_engineering_cues=(
            "Identify central void, surrounding ring, repeated openings, and chamber adjacency.",
        ),
        parameters=("court_ratio", "perimeter_depth", "opening_count", "chamber_count"),
        runtimes=("Canvas 2D", "p5.js", "Three.js"),
        notes=("Void is a spatial organizer, not an inferred historical or religious claim.",),
        boundary="Courtyard guidance is planimetric and experiential, not real-site survey.",
    ),
    "modular_hypostyle_grid": ArchitecturePatternRow(
        label="Modular Grid and Hypostyle Field",
        family="grid",
        taxonomy_path=("sacred architecture", "grid", "hypostyle field"),
        spatial_intent="Use repeated columns, cells, or bays as a field that can compress, open, or direct movement.",
        proportions=("Choose grid columns, row spacing, bay size, and density from a bounded module.",),
        plan=("Represent floor-plan reasoning as cells, aisles, columns, voids, and focus zones.",),
        axes=("Let aisles, row alignment, and symmetry lines reveal navigable directions.",),
        thresholds=("Use density changes, column spacing, or aisle breaks as threshold cues.",),
        center_periphery=("Create local centers through grid clearing, light pools, or widened bays.",),
        topology=("Cells connect by adjacency; aisles connect entry to focus; clearings become gathering nodes.",),
        geometry_mappings=("sacred_polygon_circle_grid", "egyptian_grid", "harmonic_proportion"),
        symbolic_mappings=("grid", "lattice"),
        installation=("Map objects, speakers, projection surfaces, or sensors to grid cells and clear aisles.",),
        reverse_engineering_cues=("Count rows, columns, bay rhythm, clearings, and axis-aligned paths.",),
        parameters=("grid_columns", "grid_rows", "bay_size", "clearance_cells"),
        runtimes=("p5.js", "Canvas 2D", "Three.js"),
        notes=("Use a stable module before expressive distortion.",),
        boundary="Grid guidance is layout reasoning, not CAD extraction.",
    ),
    "gothic_vertical_light_axis": ArchitecturePatternRow(
        label="Gothic-Inspired Vertical Light Axis",
        family="light_orientation",
        taxonomy_path=("sacred architecture", "light", "vertical axis"),
        spatial_intent="Translate verticality, rib rhythm, rose geometry, and filtered light into spatial guidance.",
        proportions=("Coordinate vertical bands, arch height, bay rhythm, and light aperture size.",),
        plan=("Use nave-like axis, crossing focus, side zones, and luminous end or rose-window anchor.",),
        axes=("Combine horizontal procession with a strong vertical light or elevation cue.",),
        thresholds=("Let light intensity, window color, or rib density mark progress through the space.",),
        center_periphery=("Use crossing, apse-like end, or rose-window center as focal hierarchy.",),
        topology=("Entry axis connects bays; bays connect to side zones; light source anchors focus.",),
        geometry_mappings=("gothic_geometry", "harmonic_proportion"),
        symbolic_mappings=("ascent", "axis", "center"),
        installation=("Use projection, haze, or light shafts as bounded spatial cues if the runtime supports them.",),
        reverse_engineering_cues=(
            "Look for nave axis, repeated bays, pointed arches, vertical emphasis, and light focus.",
        ),
        parameters=("arch_height", "bay_count", "window_segments", "light_axis_strength"),
        runtimes=("Three.js", "p5.js", "Canvas 2D"),
        notes=("Use inspired-by language unless a scoped architectural source is supplied.",),
        boundary="Gothic guidance is visual-spatial inspiration, not historical scholarship.",
    ),
    "tessellated_courtyard_field": ArchitecturePatternRow(
        label="Tessellated Courtyard Field",
        family="courtyard",
        taxonomy_path=("sacred architecture", "tessellation", "courtyard field"),
        spatial_intent="Use tiling, rosette grids, court edges, and perimeter rhythm as navigable structure.",
        proportions=("Set tile scale, rosette spacing, court ratio, and perimeter band depth explicitly.",),
        plan=("Combine a readable court or field with repeated tessellated bands and transition edges.",),
        axes=("Use tile axes, diagonal alignments, and court centerlines as soft orientation cues.",),
        thresholds=("Turn pattern density changes into entry, court edge, and gathering thresholds.",),
        center_periphery=("Let the court or rosette cluster act as center while tiling carries periphery.",),
        topology=("Entry links to tiled field; field surrounds court; court links to gathering or focus node.",),
        geometry_mappings=("islamic_tessellation", "sacred_tessellation", "sacred_polygon_circle_grid"),
        symbolic_mappings=("lattice", "grid", "center"),
        installation=("Use repeated floor marks, light cells, or projection tiles to guide audience flow.",),
        reverse_engineering_cues=(
            "Identify tiling basis, rosette centers, court edge, pattern density, and symmetry order.",
        ),
        parameters=("tile_size", "symmetry_order", "court_ratio", "pattern_density"),
        runtimes=("Canvas 2D", "p5.js", "GLSL"),
        notes=("Keep cultural references bounded to user-authored or source-backed cues.",),
        boundary="Tessellated architecture guidance does not claim religious or historical authority.",
    ),
    "labyrinthine_procession_path": ArchitecturePatternRow(
        label="Labyrinthine Procession Path",
        family="labyrinthine",
        taxonomy_path=("sacred architecture", "path", "labyrinth"),
        spatial_intent="Use winding path, delay, return, and reveal as spatial composition guidance.",
        proportions=("Bound path complexity, turn density, chamber spacing, and center distance.",),
        plan=("Draw a legible traversal path with nested turns, pauses, and a visible destination.",),
        axes=("Use local axes at turns even when the global path is non-linear.",),
        thresholds=("Treat each turn, narrowing, widening, or view reveal as a threshold.",),
        center_periphery=("Let the center or destination remain discoverable through path logic.",),
        topology=("Path nodes form ordered traversal; pause nodes and reveal nodes annotate the route.",),
        geometry_mappings=("recursive_spiral", "sacred_polygon_circle_grid"),
        symbolic_mappings=("labyrinth", "threshold", "gate"),
        installation=("Use sound, light, or object placement to confirm progress without confusing navigation.",),
        reverse_engineering_cues=("Trace entry, turns, decision points, pauses, center, and return path.",),
        parameters=("path_complexity", "turn_density", "pause_count", "center_distance"),
        runtimes=("p5.js", "Canvas 2D", "Three.js"),
        notes=("Keep route readable and avoid inaccessible maze-like confusion unless explicitly desired.",),
        boundary="Labyrinth guidance is spatial traversal, not an esoteric map.",
    ),
    "compressed_threshold_chamber_sequence": ArchitecturePatternRow(
        label="Compressed Threshold Chamber Sequence",
        family="threshold",
        taxonomy_path=("sacred architecture", "threshold", "chamber sequence"),
        spatial_intent="Use nested portals, antechambers, compression, release, and focus as a spatial grammar.",
        proportions=("Control chamber depth, portal scale, aperture ratio, and compression/release contrast.",),
        plan=("Sequence entry, antechamber, narrow threshold, chamber, and focus without requiring a real building.",),
        axes=("Keep the chamber sequence aligned unless deliberate offset creates reveal.",),
        thresholds=("Make each doorway, aperture, curtain, lighting change, or sound boundary explicit.",),
        center_periphery=("Use the final chamber or object as center after compression releases.",),
        topology=("Entry contains threshold; threshold opens to chamber; chamber contains focus.",),
        geometry_mappings=("temple_axis", "yantra_generator", "harmonic_proportion"),
        symbolic_mappings=("gate", "threshold", "center", "pulse"),
        installation=("Use this for immersive room-by-room installation planning without preview mutation.",),
        reverse_engineering_cues=(
            "List nested frames, aperture sizes, chamber order, and compression-release changes.",
        ),
        parameters=("chamber_count", "portal_scale", "compression_ratio", "release_depth"),
        runtimes=("Three.js", "p5.js", "Canvas 2D"),
        notes=("Treat ritual language as spatial pacing and threshold structure.",),
        boundary="Threshold sequencing does not implement V8.5 narrative or ritual simulation.",
    ),
    "installation_gallery_sequence": ArchitecturePatternRow(
        label="Installation and Gallery Sequence",
        family="installation",
        taxonomy_path=("sacred architecture", "installation", "exhibition sequence"),
        spatial_intent=(
            "Translate exhibition or installation intent into circulation, sightline, "
            "dwell, and reveal guidance."
        ),
        proportions=("Balance room zones, object spacing, viewing distance, and dwell intervals.",),
        plan=("Map entry, orientation zone, primary work, secondary works, pause, and exit path.",),
        axes=("Use sightlines and circulation axes instead of assuming a sacred central axis.",),
        thresholds=("Mark changes in sound, light, object density, or interaction mode as exhibit thresholds.",),
        center_periphery=(
            "Place hero work, interaction core, or projection anchor as center with supporting periphery.",
        ),
        topology=("Audience path connects exhibit nodes; sightline edges connect viewer positions to works.",),
        geometry_mappings=("temple_axis", "flow_field", "particle_geometry"),
        symbolic_mappings=("threshold", "center", "pulse"),
        installation=("Specify audience flow, clearance, sightlines, sensory zones, and accessibility constraints.",),
        reverse_engineering_cues=("Extract path, sightlines, dwell nodes, occlusion risks, exits, and sensory zones.",),
        parameters=("viewer_count", "dwell_nodes", "sightline_count", "clearance_width"),
        runtimes=("Three.js", "p5.js", "Canvas 2D"),
        notes=("Keep guidance product-facing and do not claim venue-specific analysis without venue data.",),
        boundary="Installation guidance is planning support, not physical safety certification.",
    ),
    "pavilion_field_environment": ArchitecturePatternRow(
        label="Pavilion Field and Creative Environment",
        family="installation",
        taxonomy_path=("sacred architecture", "environment", "pavilion field"),
        spatial_intent="Use pavilions, anchors, fields, paths, and atmosphere to plan a creative environment.",
        proportions=("Set pavilion spacing, field density, path width, and anchor scale from bounded modules.",),
        plan=("Arrange anchors in a field with clear approaches, rests, and environmental gradients.",),
        axes=("Use local axes between pavilions or anchors instead of a single central hierarchy.",),
        thresholds=("Let terrain, light, sound, density, or material shifts define zones.",),
        center_periphery=("Allow multiple local centers while preserving an overall orientation strategy.",),
        topology=("Anchor nodes connect by paths; field regions connect by gradients and boundaries.",),
        geometry_mappings=("flow_field", "l_system_growth", "biological_growth", "platonic_solid"),
        symbolic_mappings=("network", "void", "center"),
        installation=("Plan distributed audience movement, rest areas, sightlines, and sensory boundaries.",),
        reverse_engineering_cues=("Identify anchors, paths, field gradients, boundaries, and local centers.",),
        parameters=("anchor_count", "path_width", "field_density", "local_center_count"),
        runtimes=("Three.js", "p5.js", "Canvas 2D"),
        notes=(
            "Keep environmental planning conceptual unless real site dimensions are supplied and verified elsewhere.",
        ),
        boundary="Creative environment guidance is not real-site engineering, LIDAR, or safety analysis.",
    ),
}


ROADMAP_CLASSIFICATION_ROWS: Mapping[str, tuple[str, str, bool, bool]] = {
    "Architectural Reverse Engineering": (
        "implemented_runtime_behavior",
        (
            "Implemented as bounded textual/spatial description parsing and pattern "
            "reconstruction guidance; no image or survey reconstruction."
        ),
        False,
        False,
    ),
    "Planimetry Analysis": (
        "implemented_runtime_behavior",
        "Implemented as floor-plan and spatial-layout reasoning from user-authored text and catalog patterns.",
        False,
        False,
    ),
    "Image-based Reconstruction": (
        "out_of_scope_unsupported",
        "Explicitly not implemented because current runtime has no safe architecture image reconstruction path.",
        False,
        False,
    ),
    "LIDAR Interpretation Layer": (
        "out_of_scope_unsupported",
        "Explicitly not implemented because LIDAR and photogrammetry are outside the product boundary.",
        False,
        False,
    ),
    "Sacred Axis Detection": (
        "implemented_runtime_behavior",
        (
            "Implemented through deterministic axis and alignment cue extraction from "
            "text, V8.2 motifs, and V8.3 geometry."
        ),
        False,
        False,
    ),
    "Symmetry Analysis": (
        "implemented_runtime_behavior",
        "Implemented as symmetry and center/periphery guidance, not measured geometric analysis from images.",
        False,
        False,
    ),
    "Harmonic Ratio Extraction": (
        "implemented_runtime_behavior",
        "Implemented as architecture-specific proportion guidance by reusing V8.3 harmonic and golden-ratio signals.",
        False,
        False,
    ),
    "Sacred Topology Engine": (
        "implemented_runtime_behavior",
        (
            "Implemented as semantic nodes and edges for entry, threshold, axis, center, "
            "periphery, path, exhibit, and light roles."
        ),
        False,
        False,
    ),
    "Temple Semantic Graph": (
        "implemented_runtime_behavior",
        "Implemented as a bounded architecture semantic graph without claiming real temple analysis.",
        False,
        False,
    ),
    "Architectural Symbol Extraction": (
        "implemented_runtime_behavior",
        "Implemented by mapping V8.2 symbolic motifs into spatial roles, thresholds, centers, axes, and paths.",
        False,
        False,
    ),
    "Multi-scale Geometry Decomposition": (
        "implemented_runtime_behavior",
        "Implemented as site/building/room/detail guidance across plan, topology, proportion, and installation layers.",
        False,
        False,
    ),
    "Structural Pattern Discovery": (
        "implemented_runtime_behavior",
        "Implemented as architecture pattern recommendations ranked by request, V8.2, V8.3, and V8.1 evidence.",
        False,
        False,
    ),
    "Architectural Evolution Engine": (
        "implemented_runtime_behavior",
        (
            "Implemented as bounded variation and adaptation guidance, not autonomous "
            "mutation or historical evolution claims."
        ),
        False,
        False,
    ),
    "Geometry Reconstruction Validation": (
        "implemented_runtime_behavior",
        (
            "Implemented as validation findings and unsupported-reconstruction boundaries, "
            "not real reconstruction validation."
        ),
        False,
        False,
    ),
    "Architecture Explainability": (
        "implemented_runtime_behavior",
        "Implemented through typed pattern, operation, semantic graph, evidence, boundary, and validation fields.",
        False,
        False,
    ),
    "Architecture Provenance": (
        "implemented_runtime_behavior",
        "Implemented by linking request, V8.1 knowledge, V8.2 symbolic, V8.3 geometry, and bounded catalog sources.",
        False,
        False,
    ),
    "Architecture Generator": (
        "implemented_runtime_behavior",
        "Implemented as generation guidance contracts, not asset generation, CAD, or preview mutation.",
        False,
        False,
    ),
    "Architectural DNA Extraction": (
        "implemented_runtime_behavior",
        "Implemented as pattern taxonomy, proportion, topology, symbolic mapping, and reverse-cue signatures.",
        False,
        False,
    ),
    "Architectural Grammar Reconstruction": (
        "implemented_runtime_behavior",
        "Implemented as grammar-like plan, threshold, topology, and reverse-engineering cue guidance from text.",
        False,
        False,
    ),
    "Architectural Ritual Simulation": (
        "implemented_runtime_behavior",
        (
            "Implemented only as spatial procession and threshold pacing guidance without "
            "V8.5 narrative or ritual efficacy claims."
        ),
        False,
        False,
    ),
    "Creative Installation Planning": (
        "implemented_runtime_behavior",
        (
            "Implemented through installation pattern guidance, audience path, sightline, "
            "dwell, and sensory-zone recommendations."
        ),
        False,
        False,
    ),
    "Exhibition Space Analysis": (
        "implemented_runtime_behavior",
        "Implemented as textual exhibition-space reasoning; no venue survey, safety certification, or image analysis.",
        False,
        False,
    ),
    "Generative Installation Design": (
        "implemented_runtime_behavior",
        "Implemented as generative spatial guidance for installation layouts, paths, anchors, and sensory thresholds.",
        False,
        False,
    ),
    "Creative Environment Planning": (
        "implemented_runtime_behavior",
        "Implemented as pavilion, field, environment, path, and local-center planning guidance.",
        False,
        False,
    ),
    "Interactive Architecture Preview": (
        "later_v8_boundary",
        (
            "Deferred because broad preview behavior belongs to later preview/composer "
            "scope and requires product/HITL selection."
        ),
        True,
        True,
    ),
    "Installation Explainability": (
        "implemented_runtime_behavior",
        "Implemented through explainable operations, semantic edges, evidence, validation, and safe boundaries.",
        False,
        False,
    ),
}


__all__ = [
    "ARCHITECTURE_ALIASES",
    "GEOMETRY_PATTERN_TO_ARCHITECTURE",
    "PATTERN_ROWS",
    "ROADMAP_CLASSIFICATION_ROWS",
    "SYMBOL_TO_ARCHITECTURE",
    "UNSUPPORTED_ARCHITECTURE_CLAIM_TOKENS",
    "ArchitecturePatternRow",
]
