import { describe, expect, it, vi } from "vitest";
import {
  AssistantStreamError,
  decodeAssistantStream,
  parseAssistantStreamLine,
  readArtifactCapabilityMatrixSummary,
  readArtifactCriticSummary,
  readCreativeCriticSummary,
  readArtifactEngineContractRegistrySummary,
  readArtifactExportIntelligenceSummary,
  readArtifactIntelligenceSynthesisSummary,
  readArtifactMergePlannerSummary,
  readArtifactRefinerSummary,
  readArtifactDependencyGraphSummary,
  readArtifactPlanSummary,
  readMultiArtifactStrategySummary,
  readAudioVisualSceneProfileSummary,
  readClarificationSummary,
  readCrossModalityCompositionProfileSummary,
  readCreativeCompositionPlanSummary,
  readCreativeConstraintSolverSummary,
  readCreativeExecutionPlanSummary,
  readCreativeQualityPredictionSummary,
  readCreativeReasoningSummary,
  readCreativeStrategySummary,
  readCreativeTechniqueSummary,
  readCreativeTradeoffExplorerSummary,
  readEmotionalConsistencyProfileSummary,
  readGenerativeStructureBlueprintSummary,
  readProceduralStructurePlanSummary,
  readSemanticMotifSystemSummary,
  readSymbolicNarrativePlanSummary,
  readEventTimestamp,
  readPreviewArtifactUpdate,
  readRuntimeCapabilityReasonerSummary,
  readRuntimeCompatibilityProfileSummary,
  readStreamEventError,
  readWorkflowMetadata,
  streamAssistantEvents,
  workflowNodeFromAssistantStreamEvent,
  type AssistantStreamEvent
} from "./assistant-stream";

function symbolicNarrativeFixture() {
  return {
    role: "symbolic_narrative_planner",
    narrative_archetype: "threshold_crossing",
    symbolic_arc:
      "Move from invitation through boundary crossing into integrated form.",
    opening_phase: symbolicNarrativePhaseFixture(
      "opening",
      "Invitation",
      "Present the symbolic gate.",
      "anticipation",
      "threshold frame",
      "slow approach"
    ),
    development_phase: symbolicNarrativePhaseFixture(
      "development",
      "Pressure",
      "Complicate the motif.",
      "tension",
      "veiled geometry",
      "testing oscillation"
    ),
    threshold_phase: symbolicNarrativePhaseFixture(
      "threshold",
      "Crossing",
      "Mark the liminal boundary.",
      "focus",
      "gate line",
      "single decisive pass"
    ),
    climax_phase: symbolicNarrativePhaseFixture(
      "climax",
      "Revelation",
      "Reveal the reorganized order.",
      "wonder",
      "bright geometry",
      "expansion from center"
    ),
    resolution_phase: symbolicNarrativePhaseFixture(
      "resolution",
      "Integration",
      "Stabilize the revealed order.",
      "settled meaning",
      "balanced field",
      "slow cycle"
    ),
    symbolic_transitions: [
      "Invitation -> Pressure: Complicate the motif.",
      "Pressure -> Crossing: Mark the liminal boundary."
    ],
    emotional_progression: ["opening: anticipation", "climax: wonder"],
    visual_progression: ["opening: threshold frame", "climax: bright geometry"],
    motion_progression: ["opening: slow approach", "climax: expansion"],
    audio_progression: ["opening: low tone", "resolution: soft cadence"],
    experiential_goal: "Guide the audience through a threshold crossing.",
    unresolved_narrative_gaps: ["Interaction state is ambiguous."],
    hitl_questions: ["What narrative state should interaction change?"],
    prompt_guidance: [
      "Use the symbolic narrative as an ordering spine, not as doctrine."
    ],
    authority_boundary:
      "The Symbolic Narrative Planner structures a symbolic artwork journey.",
    evidence: ["Narrative archetype: threshold_crossing."]
  };
}

function symbolicNarrativePhaseFixture(
  phase: string,
  title: string,
  symbolicFunction: string,
  emotionalState: string,
  visualState: string,
  motionState: string
) {
  return {
    phase,
    title,
    symbolic_function: symbolicFunction,
    emotional_state: emotionalState,
    visual_state: visualState,
    motion_state: motionState,
    audio_state: "supporting tone",
    guidance: [`Make the ${phase} phase visible.`],
    evidence: ["threshold"]
  };
}

function creativeCompositionFixture() {
  return {
    role: "creative_composition_planner",
    composition_pattern: "threshold_composition",
    primary_focal_point: "A gate-like focal boundary at center stage.",
    secondary_focal_elements: [
      "threshold cue: Crossing",
      "resolution cue: Integration"
    ],
    spatial_organization:
      "Organize before/after zones separated by a readable threshold.",
    foreground_background_relationship:
      "Foreground marks the crossing; background reveals what lies beyond.",
    visual_hierarchy: [
      "Lead with threshold composition as the layout spine.",
      "Protect symbolism before secondary composition details."
    ],
    density_plan:
      "Keep density lower at the threshold so the boundary reads clearly.",
    rhythm_plan: "Use approach, pause, crossing, and release as rhythm.",
    balance_plan: "Balance both sides while making the crossing decisive.",
    symmetry_asymmetry_guidance:
      "Use bilateral tension or asymmetry to show transition.",
    depth_layering_guidance:
      "Layer near-side, threshold plane, and far-side space distinctly.",
    transition_guidance: [
      "Compose transition through narrative phase: Invitation -> Crossing."
    ],
    camera_viewpoint_guidance:
      "Use a stable viewpoint looking into or across the threshold.",
    audiovisual_composition_notes: [
      "Use audio as a compositional timing cue, not as a new feature layer."
    ],
    composition_risks: ["Composition may lose focal clarity."],
    unresolved_composition_gaps: ["Primary visible focal motif is unclear."],
    hitl_questions: ["What should be the primary visible focal motif?"],
    prompt_guidance: [
      "Use the composition plan as layout guidance, not code structure."
    ],
    authority_boundary:
      "The Creative Composition Planner structures artwork organization.",
    evidence: ["Composition pattern: threshold_composition."]
  };
}

function proceduralStructureFixture() {
  return {
    role: "procedural_structure_planner",
    recommended_families: [
      "recursive_geometry",
      "polar_radial_systems",
      "particle_systems"
    ],
    primary_structure: {
      family: "recursive_geometry",
      label: "Recursive Geometry",
      rationale:
        "Recursive geometry best matches the spiral transformation request.",
      evidence: ["Matched request terms: recursive, spiral."]
    },
    secondary_structures: [
      {
        family: "polar_radial_systems",
        label: "Polar/Radial Systems",
        rationale:
          "Use polar/radial systems as a supporting layer for orbiting rings.",
        evidence: ["composition signal favors polar_radial_systems."]
      },
      {
        family: "particle_systems",
        label: "Particle Systems",
        rationale:
          "Use particle systems as a supporting layer for dissolution.",
        evidence: ["narrative signal favors particle_systems."]
      }
    ],
    combination_strategy:
      "Lead with recursive geometry as the structural spine and use polar/radial systems as a bounded secondary system.",
    spatial_structure_plan:
      "Build geometry by repeatedly transforming a simple form with explicit depth caps.",
    temporal_structure_plan:
      "Animate recursive depth, rotation, scale, or reveal order over time.",
    interaction_structure_plan:
      "Map direct interaction to parameters of recursive geometry.",
    audiovisual_structure_plan:
      "Map audio or rhythm to procedural parameters of recursive geometry.",
    complexity_level: "medium",
    runtime_suitability_notes: [
      "Use inspected runtime candidates as non-binding feasibility notes: p5_js."
    ],
    performance_risks: [
      "Nested drawing can become expensive if each level adds many children."
    ],
    implementation_risks: [
      "Recursive transforms need clear stopping rules and readable parameters."
    ],
    fallback_structure_options: [
      {
        family: "polar_radial_systems",
        label: "Polar/Radial Systems",
        rationale:
          "Use polar/radial systems as a lower-risk procedural fallback.",
        evidence: ["Fallback from procedural structure planner."]
      }
    ],
    unresolved_procedural_gaps: [
      "Interaction is relevant but the controlling gesture is unclear."
    ],
    hitl_questions: ["What user gesture should control the structure?"],
    prompt_guidance: [
      "Use procedural structure guidance as an implementation spine, not as code generation."
    ],
    authority_boundary:
      "The Procedural Structure Planner recommends inspectable procedural families.",
    evidence: ["Primary procedural family: recursive_geometry."]
  };
}

function generativeStructureFixture() {
  return {
    role: "generative_structure_engine",
    blueprint_name: "Recursive Geometry Blueprint for spiral threshold",
    generative_architecture: "recursive_modular_blueprint",
    procedural_modules: [
      {
        module_id: "seed_system",
        kind: "seed_system",
        label: "Seed System",
        source_family: "recursive_geometry",
        purpose: "Define deterministic origin values and shared state.",
        inputs: ["user intent", "procedural structure plan"],
        outputs: ["seeded coordinate state", "global timing state"],
        parameters: ["random_seed", "global_scale", "time_phase"],
        evolution_role:
          "Initializes the blueprint before procedural modules evolve.",
        implementation_notes: ["Keep seeding deterministic and inspectable."],
        safeguards: ["Avoid hidden random loops."],
        evidence: ["Required root module."]
      },
      {
        module_id: "recursive_module_0",
        kind: "recursive_module",
        label: "Recursive Module",
        source_family: "recursive_geometry",
        purpose: "Build recursive geometry through bounded depth.",
        inputs: ["seed_system", "time_phase"],
        outputs: ["recursive geometry state"],
        parameters: ["recursion_depth", "spiral_tightness"],
        evolution_role: "Controls recursive build-up.",
        implementation_notes: ["Prefer iterative depth counters."],
        safeguards: ["Clamp recursion_depth."],
        evidence: ["Source procedural family: recursive_geometry."]
      },
      {
        module_id: "particle_emitter_1",
        kind: "particle_emitter",
        label: "Particle Emitter",
        source_family: "particle_systems",
        purpose: "Emit bounded particles with lifecycle state.",
        inputs: ["seed_system", "time_phase"],
        outputs: ["particle state"],
        parameters: ["particle_count", "max_particle_count"],
        evolution_role: "Controls dissolution and reassembly material.",
        implementation_notes: ["Represent particle lifecycle explicitly."],
        safeguards: ["Clamp particle_count and max_particle_count."],
        evidence: ["Secondary procedural family: particle_systems."]
      }
    ],
    module_relationships: [
      {
        source_module_id: "seed_system",
        target_module_id: "recursive_module_0",
        relationship_type: "feeds",
        description: "Seeded coordinates initialize the recursive module.",
        parameters: ["random_seed", "global_scale"],
        evidence: ["All modules depend on deterministic seed state."]
      },
      {
        source_module_id: "recursive_module_0",
        target_module_id: "particle_emitter_1",
        relationship_type: "attracts",
        description: "Particles inherit the recursive attractor path.",
        parameters: ["spiral_tightness"],
        evidence: ["Recursive and particle modules coexist."]
      }
    ],
    parameter_schema: [
      {
        name: "random_seed",
        label: "Random Seed",
        value_type: "integer",
        role: "control",
        default_value: "1",
        bounds: "0..99999",
        controlled_by: null,
        target_modules: ["seed_system"],
        rationale: "Keeps stochastic variation deterministic."
      },
      {
        name: "recursion_depth",
        label: "Recursion Depth",
        value_type: "integer",
        role: "control",
        default_value: "5",
        bounds: "1..9",
        controlled_by: null,
        target_modules: ["recursive_module_0"],
        rationale: "Caps recursive expansion."
      },
      {
        name: "max_particle_count",
        label: "Max Particle Count",
        value_type: "integer",
        role: "constraint",
        default_value: "800",
        bounds: "50..2000",
        controlled_by: null,
        target_modules: ["particle_emitter_1"],
        rationale: "Caps particle memory and update cost."
      }
    ],
    control_parameters: ["random_seed", "recursion_depth"],
    evolution_rules: [
      {
        phase: "seed",
        trigger: "time",
        rule: "Initialize seed, scale, coordinate frame, and phase clock.",
        affected_modules: ["seed_system"],
        parameter_changes: ["random_seed fixed"],
        safeguards: ["No hidden randomness after initialization."]
      },
      {
        phase: "growth",
        trigger: "narrative_phase",
        rule: "Grow modules according to symbolic phase order.",
        affected_modules: ["seed_system", "recursive_module_0"],
        parameter_changes: ["recursion_depth increases within bounds"],
        safeguards: ["Clamp depth before adding detail."]
      },
      {
        phase: "stabilization",
        trigger: "time",
        rule: "Resolve toward a stable readable final hierarchy.",
        affected_modules: ["recursive_module_0"],
        parameter_changes: ["global_scale stabilizes"],
        safeguards: ["Avoid autonomous loops or runtime repair behavior."]
      }
    ],
    spatial_evolution:
      "Anchor recursive geometry to a central focal threshold and preserve spiral readability.",
    temporal_evolution:
      "Advance through seed, growth, reassembly, and stabilization phases.",
    interaction_hooks: [
      {
        hook_id: "interaction_control_hook",
        hook_type: "interaction",
        signal: "unspecified interaction gesture",
        target_modules: ["recursive_module_0"],
        parameter_mapping: [
          "pointer or gesture -> interaction_strength",
          "interaction_strength -> density, radius, or force"
        ],
        fallback_behavior: "Use time-based evolution if interaction is unresolved."
      }
    ],
    audiovisual_hooks: [
      {
        hook_id: "audiovisual_modulation_hook",
        hook_type: "audiovisual",
        signal: "unspecified audio envelope",
        target_modules: ["particle_emitter_1"],
        parameter_mapping: [
          "audio envelope -> audio_gain",
          "audio_gain -> phase, amplitude, density, or color shift"
        ],
        fallback_behavior: "Use a slow oscillator if no audio signal exists."
      }
    ],
    runtime_implementation_guidance: [
      "Treat inspected runtime candidates as feasibility guidance only: p5_js."
    ],
    performance_safeguards: [
      "Expose frame_budget_ms as a constraint parameter.",
      "Clamp particle_count and max_particle_count."
    ],
    fallback_blueprint: {
      name: "Bounded Polar/Radial System Fallback",
      architecture: "radial_pattern_blueprint",
      module_kinds: ["seed_system", "symmetry_transform"],
      parameter_reductions: [
        "Reduce active module count to seed plus one primary structure."
      ],
      reason:
        "Use when performance pressure or ambiguity makes the primary blueprint too expensive.",
      prompt_guidance: [
        "Keep the same creative intent while reducing module count."
      ]
    },
    unresolved_implementation_gaps: [
      "Interaction hook exists but the controlling gesture is unclear."
    ],
    hitl_questions: ["Which gesture should control the generative structure?"],
    prompt_guidance: [
      "Use the generative blueprint as structure guidance, not executable code."
    ],
    authority_boundary:
      "The Generative Structure Engine turns procedural metadata into an inspectable generative blueprint only.",
    evidence: ["Primary procedural family: recursive_geometry."]
  };
}

function semanticMotifFixture() {
  return {
    role: "semantic_motif_engine",
    motif_system_name: "Fragmentation / Reintegration Motif System",
    primary_motifs: [
      semanticMotifEntry(
        "fragmentation",
        "Fragmentation",
        "transformation",
        "primary"
      ),
      semanticMotifEntry(
        "reintegration",
        "Reintegration",
        "transformation",
        "primary"
      )
    ],
    secondary_motifs: [
      semanticMotifEntry("spiral", "Spiral", "rhythm", "secondary"),
      semanticMotifEntry("flame", "Flame", "material_signal", "secondary")
    ],
    motif_hierarchy: [
      "Primary motif 1: fragmentation acts as transformation.",
      "Primary motif 2: reintegration acts as transformation.",
      "Secondary motif: spiral supports fragmentation."
    ],
    motif_recurrence_plan: [
      "Repeat fragmentation at each major visual or narrative phase.",
      "Repeat reintegration at each major visual or narrative phase."
    ],
    motif_transformation_plan: [
      "Transform fragmentation through bounded parameter changes.",
      "Align motif transformation with generative phases: seed, growth, reassembly."
    ],
    motif_to_structure_mapping: [
      {
        motif_id: "fragmentation",
        procedural_families: ["particle_systems"],
        generative_module_ids: ["particle_emitter_1"],
        generative_module_kinds: ["particle_emitter"],
        structural_behavior: "Break form into bounded fragments or particles.",
        evidence: ["Generative modules: particle_emitter."]
      },
      {
        motif_id: "reintegration",
        procedural_families: ["particle_systems"],
        generative_module_ids: ["geometry_reassembly_layer"],
        generative_module_kinds: ["geometry_reassembly_layer"],
        structural_behavior: "Reassemble fragments into a readable structure.",
        evidence: ["Generative modules: geometry_reassembly_layer."]
      }
    ],
    motif_to_composition_mapping: [
      {
        motif_id: "fragmentation",
        composition_role: "Use fragmentation inside fragmented recomposition.",
        spatial_anchor: "A gate-like focal boundary at center stage.",
        rhythm_or_density_guidance:
          "Use approach, pause, crossing, and release as rhythm.",
        evidence: ["Composition pattern: fragmented_recomposition."]
      }
    ],
    motif_to_narrative_mapping: [
      {
        motif_id: "reintegration",
        narrative_function:
          "Use reintegration to clarify rebirth without asserting doctrine.",
        phase_alignment: ["climax", "resolution"],
        evidence: ["Narrative archetype: death_and_rebirth."]
      }
    ],
    motif_to_parameter_mapping: [
      {
        motif_id: "fragmentation",
        parameter_names: ["fragmentation_amount", "particle_count"],
        parameter_guidance:
          "Use fragmentation_amount to make fragmentation recur.",
        evidence: ["Motif-to-parameter mapping for fragmentation."]
      },
      {
        motif_id: "reintegration",
        parameter_names: ["reassembly_speed"],
        parameter_guidance:
          "Use reassembly_speed to make reintegration transform.",
        evidence: ["Motif-to-parameter mapping for reintegration."]
      }
    ],
    coherence_risks: [
      "Keep fragmentation visibly dominant over decorative motifs."
    ],
    overuse_risks: [
      "Overusing fragmentation literally may flatten symbolic ambiguity."
    ],
    underuse_risks: [
      "Underusing spiral may make the motif system feel disconnected."
    ],
    unsupported_symbolic_claims: [
      "Treat symbolic terms as user-supplied design language, not factual doctrine."
    ],
    motif_fallback_plan: {
      fallback_primary_motif: "fragmentation",
      fallback_secondary_motifs: ["spiral", "flame"],
      simplification_strategy:
        "Preserve fragmentation and remove weak secondary motifs first.",
      preserved_meaning:
        "Preserve transformation through fragmentation and reintegration.",
      prompt_guidance: [
        "Use fewer motifs before adding new symbolic material."
      ]
    },
    unresolved_motif_gaps: [
      "Motif language is abstract; confirm the primary motif emphasis."
    ],
    hitl_questions: ["Which motif should remain visually dominant?"],
    prompt_guidance: [
      "Use the motif system as semantic guidance, not executable code."
    ],
    authority_boundary:
      "The Semantic Motif Engine organizes recurring symbolic motifs as inspectable design metadata only.",
    evidence: ["Primary motifs: fragmentation, reintegration."]
  };
}

function semanticMotifEntry(
  motifId: string,
  label: string,
  role: string,
  hierarchyLevel: string
) {
  return {
    motif_id: motifId,
    label,
    role,
    hierarchy_level: hierarchyLevel,
    rationale: `Use ${motifId} as a motif because it is supported by signals.`,
    recurrence_guidance: [`Introduce ${motifId} early.`],
    transformation_guidance: [`Let ${motifId} transform through parameters.`],
    evidence: [`Motif evidence: ${motifId}.`]
  };
}

function emotionalConsistencyFixture() {
  return {
    role: "emotional_consistency_engine",
    primary_emotional_tone: "transformation",
    secondary_emotional_tones: [
      "rupture",
      "suspension",
      "release",
      "integration"
    ],
    emotional_arc: [
      "contraction",
      "fragmentation or destabilization",
      "threshold stillness",
      "emergence",
      "integration"
    ],
    emotional_phase_mapping: [
      {
        phase: "opening",
        tone: "tension",
        intensity: "medium",
        guidance: "Use tension as the opening emotional state.",
        evidence: ["Narrative phase: opening."]
      },
      {
        phase: "threshold",
        tone: "suspension",
        intensity: "high",
        guidance: "Slow motion near the threshold.",
        evidence: ["Narrative phase: threshold."]
      }
    ],
    emotional_to_narrative_mapping: [
      {
        tone: "transformation",
        narrative_phase: "climax",
        narrative_function:
          "Use transformation to support death_and_rebirth without universal claims.",
        evidence: ["Narrative archetype: death_and_rebirth."]
      }
    ],
    emotional_to_motif_mapping: [
      {
        tone: "rupture",
        motif_id: "fragmentation",
        emotional_function:
          "Let fragmentation carry rupture as a recurring design cue.",
        evidence: ["Semantic Motif Engine primary motifs: fragmentation."]
      },
      {
        tone: "integration",
        motif_id: "reintegration",
        emotional_function:
          "Let reintegration carry integration as a recurring design cue.",
        evidence: ["Semantic Motif Engine primary motifs: reintegration."]
      }
    ],
    emotional_to_composition_mapping: [
      {
        tone: "transformation",
        composition_pattern: "fragmented_recomposition",
        composition_guidance:
          "Use transformation inside fragmented recomposition.",
        spatial_or_density_guidance:
          "Keep density shifts tied to the emotional arc.",
        evidence: ["Composition pattern: fragmented_recomposition."]
      }
    ],
    emotional_to_structure_mapping: [
      {
        tone: "rupture",
        procedural_families: ["particle_systems"],
        generative_module_kinds: ["particle_emitter"],
        structural_guidance:
          "Preserve rupture through existing particle structure.",
        evidence: ["Generative modules: particle_emitter."]
      }
    ],
    emotional_to_parameter_mapping: [
      {
        tone: "rupture",
        parameter_names: ["fragmentation_amount", "particle_count"],
        parameter_guidance:
          "Use fragmentation_amount to tune rupture without new runtime behavior.",
        evidence: ["Emotional-to-parameter mapping for rupture."]
      },
      {
        tone: "integration",
        parameter_names: ["reassembly_speed"],
        parameter_guidance:
          "Use reassembly_speed to tune integration without new runtime behavior.",
        evidence: ["Emotional-to-parameter mapping for integration."]
      }
    ],
    color_light_guidance: [
      "Begin muted, pass through low-contrast threshold light, then use luminous reintegration for transformation."
    ],
    motion_rhythm_guidance: [
      "Stage motion as contraction, scatter, pause, acceleration, then calm expansion."
    ],
    audiovisual_guidance: [
      "Use audio or audiovisual changes to reinforce transformation."
    ],
    emotional_coherence_score: 86,
    emotional_tensions: [
      "Calm or resolving tones must be staged separately from rupture."
    ],
    mismatch_risks: [
      "Playful motion may weaken solemn transformation or ritual weight."
    ],
    flattening_risks: [
      "Treating transformation as a single constant mood may flatten the emotional arc."
    ],
    over_intensity_risks: [
      "Over-intensifying rupture may exhaust the viewer before resolution."
    ],
    under_intensity_risks: [
      "Underplaying transformation may make the arc feel unresolved."
    ],
    fallback_emotional_strategy: {
      fallback_primary_tone: "transformation",
      fallback_secondary_tones: ["release", "integration"],
      simplification_strategy:
        "Preserve transformation and reduce secondary emotional cues first.",
      preserved_feeling: "Preserve the death_and_rebirth arc through transformation.",
      prompt_guidance: [
        "Use fewer emotional tones before adding new visual systems."
      ]
    },
    unresolved_emotional_gaps: [
      "Emotional language is abstract; confirm the dominant tone."
    ],
    hitl_questions: ["Which emotional tone should remain dominant?"],
    prompt_guidance: [
      "Use emotional consistency as design guidance, not executable code."
    ],
    authority_boundary:
      "The Emotional Consistency Engine organizes emotional direction as inspectable design metadata only.",
    evidence: ["Primary emotional tone: transformation."]
  };
}

function crossModalityFixture() {
  return {
    role: "cross_modality_composer",
    modality_pattern: "fragmentation_reassembly_visual_motion_layers",
    primary_modality: "visual_structure",
    supporting_modalities: ["motion", "audio", "rhythm", "structure", "motif", "emotion"],
    modality_hierarchy: [
      {
        modality: "visual_structure",
        role: "Carry fragmentation and reassembly as the visible anchor.",
        priority: "primary",
        evidence: ["Primary modality selected from composer pattern."]
      },
      {
        modality: "audio",
        role: "Use audio as pulse and restraint guidance.",
        priority: "secondary",
        evidence: ["Audio relevance detected."]
      }
    ],
    visual_role: "Show rupture, particle dispersal, and reassembly.",
    motion_role: "Stage motion as contraction, scatter, pause, acceleration, then calm expansion.",
    audio_role: "Use audio as a design cue for pulse and thresholds.",
    rhythm_role: "Coordinate visual rhythm with requested audio pacing.",
    camera_viewpoint_role: null,
    structure_role:
      "Bind modalities to particle and geometry reassembly modules.",
    motif_role:
      "Let recurring motifs bridge visual, motion, rhythm, and emotional cues.",
    emotion_role:
      "Use transformation as the modality modulation target.",
    modality_synchronization_plan: [
      "Synchronize phase changes before adding decorative effects."
    ],
    visual_to_audio_mapping: [
      {
        source_modality: "visual_structure",
        target_modality: "audio",
        mapping:
          "Treat density and brightness changes as cues for stronger pulse.",
        cues: ["visual density", "brightness", "phase threshold"],
        motif_id: null,
        emotional_tone: null,
        evidence: ["Audio relevance detected."]
      }
    ],
    audio_to_motion_mapping: [
      {
        source_modality: "audio",
        target_modality: "motion",
        mapping:
          "Use pulse and silence as advisory timing for particle speed.",
        cues: ["pulse", "silence"],
        motif_id: null,
        emotional_tone: null,
        evidence: ["Audio-to-motion mapping remains design guidance."]
      }
    ],
    motion_to_structure_mapping: [
      {
        source_modality: "motion",
        target_modality: "structure",
        mapping:
          "Let motion phases expose growth, rupture, threshold, and reassembly.",
        cues: ["particle_emitter", "geometry_reassembly_layer"],
        motif_id: null,
        emotional_tone: null,
        evidence: ["Motion must remain attached to structure."]
      }
    ],
    motif_to_modality_mapping: [
      {
        source_modality: "motif",
        target_modality: "motion",
        mapping: "Use fragmentation as a recurring motion cue.",
        cues: ["fragmentation", "motion", "recurrence"],
        motif_id: "fragmentation",
        emotional_tone: null,
        evidence: ["Semantic motif metadata mapped to modality role."]
      }
    ],
    emotional_to_modality_mapping: [
      {
        source_modality: "emotion",
        target_modality: "visual_structure",
        mapping: "Use transformation to shape color, density, and contrast.",
        cues: ["transformation", "color", "density"],
        motif_id: null,
        emotional_tone: "transformation",
        evidence: ["Emotional consistency mapped to visual structure."]
      }
    ],
    temporal_cue_plan: [
      {
        phase: "threshold",
        cue: "Threshold stillness",
        modalities: ["visual_structure", "motion", "audio"],
        timing_guidance:
          "Coordinate visual state, motion state, and rhythm before adding extra layers.",
        evidence: ["Narrative threshold."]
      }
    ],
    contrast_balance_plan: [
      "Keep one leading modality per phase; use the others as reinforcement."
    ],
    modality_conflicts: [
      "Dense visuals and loud/intense audio may compete for attention."
    ],
    overload_risks: [
      "Dense visual systems can overload motion and motif readability."
    ],
    underuse_risks: [
      "Broad multimodal phrasing can underuse one modality."
    ],
    fallback_multimodal_strategy: {
      fallback_pattern: "visual_led_composition",
      preserved_modalities: ["visual_structure", "motion", "rhythm"],
      reduced_modalities: ["audio"],
      simplification_strategy:
        "Preserve visual structure and motion timing; reduce audio to optional prompt cues.",
      prompt_guidance: [
        "When multimodal scope is broad, keep visual-motion structure first."
      ]
    },
    unresolved_modality_gaps: [
      "Multimodal intent is broad; confirm which modality should lead."
    ],
    hitl_questions: [
      "Which modality should lead if visual, motion, audio, and emotion compete?"
    ],
    prompt_guidance: [
      "Treat cross-modality mappings as design guidance, not generated media."
    ],
    authority_boundary:
      "The Cross-Modality Composer organizes modality signals as inspectable design metadata only.",
    evidence: ["Pattern: fragmentation_reassembly_visual_motion_layers."]
  };
}

function audioVisualSceneFixture() {
  const opening = audioVisualScenePhaseFixture("opening", "Whole Form");
  const development = audioVisualScenePhaseFixture("development", "Fragmentation");
  const threshold = audioVisualScenePhaseFixture("threshold", "Sparse Stillness");
  const climax = audioVisualScenePhaseFixture("climax", "Reassembly");
  const resolution = audioVisualScenePhaseFixture("resolution", "Integrated Geometry");

  return {
    role: "audio_visual_scene_system",
    scene_pattern: "fragmentation_to_reintegration",
    scene_arc:
      "Open with a coherent form, fragment into turbulent pieces, hold sparse threshold stillness, reassemble at climax, and resolve as integrated geometry.",
    scene_phases: [opening, development, threshold, climax, resolution],
    opening_scene: opening,
    development_scene: development,
    threshold_scene: threshold,
    climax_scene: climax,
    resolution_scene: resolution,
    cue_plan: [
      {
        cue_id: "opening_visual",
        phase: "opening",
        cue_type: "visual",
        description: "Show the coherent form before rupture.",
        timing: "Lead opening with visual state before secondary cues.",
        modalities: ["visual_structure"],
        evidence: ["Scene phase: opening."]
      },
      {
        cue_id: "threshold_sync",
        phase: "threshold",
        cue_type: "synchronization",
        description: "Align visual, motion, rhythm, motif, and emotional cues.",
        timing: "Checkpoint after threshold before entering the next phase.",
        modalities: ["visual_structure", "motion", "audio", "rhythm"],
        evidence: ["Scene synchronization checkpoint."]
      }
    ],
    transition_plan: [
      {
        from_phase: "opening",
        to_phase: "development",
        transition: "Let coherent form contract or fracture.",
        visual_motion_guidance:
          "Carry Whole Form into Fragmentation by changing visual density.",
        audio_rhythm_guidance:
          "Use rhythm or audio density to mark opening -> development.",
        continuity_guidance:
          "Keep fragmentation visible enough to connect opening and development.",
        evidence: ["opening to development."]
      }
    ],
    climax_strategy:
      "Make Reassembly the only peak-density scene with synchronized convergence.",
    resolution_strategy:
      "After climax, reduce density and stabilize motif, motion, and procedural behavior.",
    visual_timing_plan: ["opening: coherent luminous form"],
    motion_timing_plan: ["opening: slow contraction or orbit"],
    audio_timing_plan: ["threshold: silence or thin pulse"],
    rhythm_timing_plan: ["opening: measured pulse"],
    camera_timing_plan: ["threshold: reserve active viewpoint emphasis"],
    motif_timing_plan: ["opening: introduce fragmentation"],
    emotional_timing_plan: ["opening: transformation at medium intensity"],
    procedural_timing_plan: ["opening: use particle emitter as scene basis"],
    synchronization_checkpoints: [
      "threshold: align visual_structure, motion, audio, rhythm before transition."
    ],
    scene_contrast_plan: [
      "Make climax the only maximum-density scene."
    ],
    scene_continuity_plan: [
      "Carry one visual anchor, one motion rule, and one rhythm rule through all scenes."
    ],
    scene_risks: [
      "Scene structure is broad; the lead phase emphasis may need HITL confirmation."
    ],
    pacing_risks: [
      "Dense cues can collapse development, threshold, and climax into one flat peak."
    ],
    overload_risks: [
      "Audio, camera, and dense motion should not peak in every scene."
    ],
    fallback_scene_strategy: {
      fallback_pattern: "seed_to_expansion",
      preserved_phases: ["opening", "threshold", "climax", "resolution"],
      reduced_elements: ["audio timing", "camera/viewpoint timing"],
      simplification_strategy:
        "Preserve the five-phase scene arc, but reduce optional audio and camera cues first.",
      prompt_guidance: [
        "If scene scope is too broad, keep opening, threshold, climax, and resolution legible."
      ]
    },
    unresolved_scene_gaps: [
      "Scene language is broad; confirm desired pacing if precision matters."
    ],
    hitl_questions: [
      "Should audio timing drive scene transitions, or only support visual rhythm?"
    ],
    prompt_guidance: [
      "Use fragmentation_to_reintegration as the bounded audio-visual scene arc."
    ],
    authority_boundary:
      "The Audio-Visual Scene System organizes scene phases, cues, transitions, climax, resolution, and timing guidance as inspectable design metadata only.",
    evidence: ["Scene pattern: fragmentation_to_reintegration."]
  };
}

function artifactPlanFixture() {
  return {
    role: "artifact_planner",
    primary_artifact_intent:
      "Generate a luminous p5.js mandala that preserves the symbolic scene arc.",
    artifact_type: "runnable_code",
    artifact_family: "p5_sketch",
    required_components: [
      "One clearly labeled primary artifact.",
      "A fenced code block with an explicit language tag.",
      "p5.js setup/draw lifecycle."
    ],
    runtime_requirements: [
      "Respect existing runtime hint: p5.",
      "Keep renderer compatibility with surface.p5."
    ],
    creative_dependencies: [
      "Strategy: sacred_geometry.",
      "Technique: recursive_geometry."
    ],
    generative_dependencies: [
      "Procedural structure: recursive_geometry.",
      "Audio-visual scene: fragmentation_to_reintegration."
    ],
    expected_output_structure: [
      "Lead with the primary runnable artifact.",
      "Use a fenced code block with an explicit filename or language tag."
    ],
    implementation_risks: [
      "Recursive transforms need clear stopping rules and readable parameters."
    ],
    missing_information: [
      "Target runtime/domain is inferred rather than explicit."
    ],
    hitl_questions: [
      "Should we resolve this artifact planning gap before generation: target runtime?"
    ],
    prompt_guidance: [
      "Use the Artifact Planner as artifact-shape guidance only.",
      "Satisfy required artifact components before adding secondary effects."
    ],
    authority_boundary:
      "The Artifact Planner structures intended artifact shape as inspectable metadata only.",
    evidence: ["Artifact family: p5_sketch."]
  };
}

function artifactDependencyGraphFixture() {
  return {
    role: "artifact_dependency_graph",
    primary_artifact_node_id: "primary_artifact",
    artifact_nodes: [
      {
        node_id: "primary_artifact",
        label: "Primary planned artifact",
        node_type: "planned_artifact",
        status: "available",
        summary: "runnable_code/p5_sketch: luminous mandala",
        evidence: ["Artifact plan: runnable_code; p5_sketch."]
      },
      {
        node_id: "runtime_requirements",
        label: "Runtime-facing requirements",
        node_type: "runtime_requirement",
        status: "available",
        summary: "Respect existing runtime hint: p5.",
        evidence: ["Respect existing runtime hint: p5."]
      }
    ],
    dependency_edges: [
      {
        source_node_id: "runtime_requirements",
        target_node_id: "primary_artifact",
        relationship: "requires",
        strength: "required",
        rationale:
          "Runtime-facing requirements constrain artifact implementation notes."
      }
    ],
    required_upstream_metadata: [
      "assistant_request:available",
      "artifact_plan:available"
    ],
    optional_upstream_metadata: [
      "creative_plan:available",
      "runtime_capabilities:available"
    ],
    blocking_dependencies: [],
    soft_dependencies: [
      "Use creative_plan as non-blocking context."
    ],
    runtime_facing_dependencies: [
      "Respect existing runtime hint: p5."
    ],
    prompt_facing_dependencies: [
      "Use the Artifact Dependency Graph as dependency metadata only."
    ],
    downstream_consumers: [
      "prompt_renderer",
      "creative_assistant_director",
      "creative_reasoning_engine",
      "workflow_serialization",
      "final_payload",
      "nextjs_stream_hydration"
    ],
    missing_dependency_risks: [
      "Artifact Plan missing information: target runtime is inferred."
    ],
    dependency_conflicts: [
      "intent vs performance: keep recursive depth bounded."
    ],
    hitl_questions: [
      "Should we resolve this missing artifact dependency risk?"
    ],
    prompt_guidance: [
      "Use the Artifact Dependency Graph as dependency metadata only."
    ],
    authority_boundary:
      "The Artifact Dependency Graph structures inspectable metadata only.",
    evidence: ["Artifact dependency graph: 2 nodes; 1 edges."]
  };
}

function runtimeCompatibilityFixture() {
  return {
    role: "runtime_compatibility_engine",
    compatible_runtimes: ["p5_js", "canvas"],
    unsupported_runtimes: ["glsl"],
    preferred_runtimes: ["p5_js"],
    runtime_confidence: [
      {
        runtime: "p5_js",
        label: "p5.js",
        confidence: 0.93
      },
      {
        runtime: "glsl",
        label: "GLSL",
        confidence: 0.42
      }
    ],
    compatibility_assessments: [
      {
        runtime: "p5_js",
        label: "p5.js",
        compatibility: "compatible",
        confidence: 0.93,
        compatibility_reasons: [
          "p5.js directly supports p5_sketch."
        ],
        runtime_requirements: [
          "Respect existing runtime hint: p5."
        ],
        runtime_limitations: [
          "Compatibility metadata must not change runtime execution."
        ],
        dependency_compatibility: [
          "p5.js satisfies: Respect existing runtime hint: p5."
        ],
        expected_implementation_complexity: "medium",
        portability: "high",
        interoperability: "high",
        implementation_risks: [
          "Large particle counts can pressure frame rate."
        ],
        prompt_guidance: [
          "Treat p5.js compatibility as metadata only."
        ],
        evidence: ["Runtime evaluated: p5_js."]
      }
    ],
    runtime_requirements: [
      "Respect existing runtime hint: p5."
    ],
    runtime_limitations: [
      "GLSL: Unsupported runtime should not be used as an output target."
    ],
    dependency_compatibility: [
      "Top runtime dependency fit: p5.js compatible."
    ],
    expected_implementation_complexity: "medium",
    portability: "high",
    interoperability: "high",
    missing_runtime_information: [
      "Route/domain metadata is inferred or unavailable."
    ],
    implementation_risks: [
      "Do not use compatibility metadata to auto-select runtimes."
    ],
    hitl_questions: [
      "Should unsupported runtimes be explicitly excluded from the response: GLSL?"
    ],
    prompt_guidance: [
      "Use Runtime Compatibility Engine output as compatibility metadata only."
    ],
    authority_boundary:
      "The Runtime Compatibility Engine evaluates runtime compatibility as inspectable metadata only.",
    evidence: ["Compatibility order: p5_js:compatible:0.93."]
  };
}

function artifactCapabilityMatrixFixture() {
  return {
    role: "artifact_capability_matrix",
    capability_profiles: [
      {
        target: "p5_js",
        label: "p5.js",
        capability_confidence: 0.91,
        capability_reasons: [
          "p5.js artifact capability fit is strong."
        ],
        strengths: [
          "Fast iteration for sketches, particles, geometry, and interaction."
        ],
        weaknesses: [
          "Large particle counts can pressure frame rate."
        ],
        unsupported_capabilities: [
          "Native shader pipelines require additional scaffolding."
        ],
        risky_capabilities: [
          "Dense animation can become CPU-bound without caps."
        ],
        artifact_fit: "strong",
        creative_fit: "strong",
        generative_fit: "strong",
        interaction_fit: "strong",
        audiovisual_fit: "moderate",
        export_fit: "moderate",
        interoperability_fit: "strong",
        portability_fit: "strong",
        capability_risks: [
          "Do not use capability metadata to auto-select targets."
        ],
        prompt_guidance: [
          "Use p5.js capability notes as planning metadata only."
        ],
        evidence: ["Target evaluated: p5_js."]
      }
    ],
    strongest_targets: ["p5_js", "canvas"],
    weakest_targets: ["glsl"],
    target_strengths: [
      "p5.js: Fast iteration for sketches, particles, geometry, and interaction."
    ],
    target_weaknesses: [
      "p5.js: Large particle counts can pressure frame rate."
    ],
    unsupported_or_risky_capabilities: [
      "p5.js: Native shader pipelines require additional scaffolding."
    ],
    capability_confidence: [
      {
        target: "p5_js",
        label: "p5.js",
        confidence: 0.91
      }
    ],
    artifact_fit: "strong",
    creative_fit: "strong",
    generative_fit: "strong",
    interaction_fit: "strong",
    audiovisual_fit: "moderate",
    export_fit: "moderate",
    interoperability_fit: "strong",
    portability_fit: "strong",
    missing_capability_information: [
      "Creative plan does not provide an explicit runtime hint."
    ],
    capability_risks: [
      "Do not use capability metadata to auto-select targets."
    ],
    hitl_questions: [
      "Should weak or unsupported targets be explicitly de-emphasized: GLSL?"
    ],
    prompt_guidance: [
      "Use Artifact Capability Matrix output as target capability metadata only."
    ],
    authority_boundary:
      "The Artifact Capability Matrix describes runtime and artifact target capabilities as inspectable planning metadata only.",
    evidence: ["Capability order: p5_js:strong:0.91."]
  };
}

function multiArtifactStrategyFixture() {
  return {
    role: "multi_artifact_strategy",
    artifact_strategy_summary:
      "Lead with primary p5.js code, then separate dependency, runtime, and capability notes.",
    primary_artifact: {
      artifact_id: "primary_artifact",
      title: "Primary p5.js artifact",
      role: "primary",
      artifact_type: "runnable_code",
      artifact_family: "p5_sketch",
      priority: "critical",
      purpose: "Deliver the requested creative-coding output first.",
      runtime_targets: ["p5_js"],
      capability_targets: ["p5_js", "canvas"],
      depends_on: [],
      handoff_points: [
        "Primary output hands off to supporting notes after code shape is clear."
      ],
      evidence: ["Artifact family: p5_sketch."]
    },
    supporting_artifacts: [
      {
        artifact_id: "runtime_notes",
        title: "Runtime compatibility notes",
        role: "supporting",
        artifact_type: "explanation",
        artifact_family: "p5_sketch",
        priority: "medium",
        purpose: "Surface runtime metadata without changing execution.",
        runtime_targets: ["p5_js"],
        capability_targets: ["p5_js"],
        depends_on: ["primary_artifact"],
        handoff_points: [
          "Hand off to runtime notes after the primary artifact."
        ],
        evidence: ["Use Runtime Compatibility Engine output as metadata only."]
      }
    ],
    artifact_sequence: [
      {
        step_id: "step_1_primary_artifact",
        order: 1,
        artifact_id: "primary_artifact",
        action: "produce",
        rationale: "Primary artifact leads the response.",
        depends_on: [],
        prompt_guidance: ["Produce the primary artifact before support notes."]
      },
      {
        step_id: "step_2_runtime_notes",
        order: 2,
        artifact_id: "runtime_notes",
        action: "document",
        rationale: "Runtime notes support the primary artifact.",
        depends_on: ["primary_artifact"],
        prompt_guidance: ["Document runtime notes as metadata only."]
      }
    ],
    artifact_priority: [
      {
        artifact_id: "primary_artifact",
        priority: "critical",
        rationale: "Primary artifact leads the response."
      },
      {
        artifact_id: "runtime_notes",
        priority: "medium",
        rationale: "Runtime notes support the primary artifact."
      }
    ],
    artifact_grouping: [
      {
        group_id: "primary_output_group",
        label: "Primary output",
        artifact_ids: ["primary_artifact"],
        grouping_rationale: "Keep the main artifact isolated and first.",
        separation_rationale: "Do not bury the primary artifact in metadata."
      }
    ],
    artifact_separation_strategy: [
      "Lead with Primary p5.js artifact as the only primary artifact."
    ],
    artifact_combination_strategy: [
      "Combine artifacts in one response only as separated sections."
    ],
    artifact_dependency_order: [
      "1. primary_artifact",
      "2. runtime_notes after primary_artifact"
    ],
    artifact_handoff_points: [
      "primary_artifact -> runtime_notes: Runtime notes support the primary artifact."
    ],
    runtime_aware_artifact_strategy: [
      "Preferred compatible runtimes are metadata only: p5_js."
    ],
    capability_aware_artifact_strategy: [
      "Strongest target capabilities are metadata only: p5_js, canvas."
    ],
    combination_mode: "primary_with_supporting_sections",
    risk_areas: [
      "Do not let supporting artifacts expand implementation scope."
    ],
    missing_information: [
      "Artifact downstream consumers are unavailable."
    ],
    hitl_questions: [
      "Should this multi-artifact risk constrain response structure?"
    ],
    prompt_guidance: [
      "Use Multi-Artifact Strategy output as response-structure metadata only."
    ],
    authority_boundary:
      "The Multi-Artifact Strategy plans ordering as inspectable metadata only; it does not generate artifacts.",
    evidence: ["Supporting artifact count: 1."]
  };
}

function artifactCriticFixture() {
  return {
    role: "artifact_critic",
    critique_confidence: 0.82,
    critique_summary:
      "Artifact planning critique risk is medium with visible capability and runtime concerns.",
    strengths: [
      "Artifact plan declares runnable_code / p5_sketch with required components.",
      "Critique remains metadata-only and non-executing."
    ],
    weaknesses: [
      "Unsupported runtimes should not be treated as viable targets: glsl."
    ],
    capability_gaps: [
      "p5.js: Native shader pipelines require additional scaffolding."
    ],
    dependency_concerns: [
      "Runtime-facing dependency conflicts with output structure."
    ],
    runtime_concerns: [
      "Unsupported runtimes should not be treated as viable targets: glsl."
    ],
    scalability_concerns: [
      "Dense particle counts can pressure frame rate."
    ],
    maintainability_concerns: [
      "Multiple supporting artifacts may require strict section labels."
    ],
    complexity_concerns: [
      "Blocking dependencies raise planning complexity."
    ],
    risk_assessment: "medium",
    unsupported_assumptions: [
      "Artifact Critic findings are advisory and must not reject or refine strategy."
    ],
    missing_information: [
      "Artifact downstream consumers are unavailable."
    ],
    open_questions: [
      "Should this missing planning metadata be resolved?"
    ],
    hitl_questions: [
      "Should generation wait because Artifact Critic risk is medium?"
    ],
    improvement_opportunities: [
      "Use critic findings as visible caveats in prompt guidance, not as edits."
    ],
    prompt_guidance: [
      "Use Artifact Critic output as metadata-only critique of planning signals."
    ],
    authority_boundary:
      "The Artifact Critic evaluates planning metadata only; it does not modify artifacts.",
    evidence: ["Runtime compatibility: 1 compatible; 1 unsupported."]
  };
}

function creativeCriticFixture() {
  return {
    role: "creative_critic_engine",
    critic_confidence: 0.86,
    critique_summary:
      "Creative critique risk is medium with strong concept quality and visible runtime caveats.",
    creative_strengths: [
      "Strategy and technique are coherent.",
      "Artifact-aware critique can inspect planning metadata."
    ],
    creative_weaknesses: [
      "Runtime fit quality needs an explicit caveat."
    ],
    concept_quality: 0.82,
    execution_quality: 0.74,
    artifact_quality: 0.71,
    coherence_quality: 0.78,
    runtime_fit_quality: 0.63,
    originality_quality: 0.79,
    clarity_quality: 0.76,
    feasibility_quality: 0.68,
    risk_assessment: "medium",
    missing_information: [
      "Generated response and artifacts are not available; critique is pre-generation."
    ],
    unsupported_assumptions: [
      "Creative Critic findings are advisory metadata only."
    ],
    improvement_opportunities: [
      "Improve runtime fit clarity before expanding scope."
    ],
    hitl_questions: [
      "Should generation proceed with Creative Critic risk medium?"
    ],
    prompt_guidance: [
      "Use Creative Critic output as metadata-only critique, not as artifact modification or rejection."
    ],
    authority_boundary:
      "The Creative Critic Engine evaluates creative and artifact metadata only; it does not modify artifacts.",
    evidence: ["Authority boundary verified: metadata-only critique."]
  };
}

function artifactRefinerFixture() {
  return {
    role: "artifact_refiner",
    refinement_confidence: 0.79,
    refinement_summary:
      "Artifact refinement intelligence is advisory only with priority improvements.",
    recommended_improvements: [
      "Address critic capability gap: Native shader pipelines require scaffolding.",
      "Preserve refinement advice as metadata-only guidance, not an edit."
    ],
    priority_improvements: [
      "Resolve dependency risks before expanding artifact scope."
    ],
    capability_improvements: [
      "Clarify capability limitation: Native shader pipelines require scaffolding."
    ],
    dependency_improvements: [
      "Separate conflicting dependency assumption: Runtime-facing dependency conflict."
    ],
    runtime_improvements: [
      "Caveat unsupported runtimes without selecting alternatives: glsl."
    ],
    scalability_improvements: [
      "Add bounded-scope caveat for scalability signal: Dense particle counts."
    ],
    maintainability_improvements: [
      "Keep supporting artifacts separated with strict labels."
    ],
    complexity_reductions: [
      "Reduce implementation scope before adding optional details."
    ],
    risk_reductions: [
      "Preserve refinement advice as metadata-only guidance, not an edit."
    ],
    refinement_candidates: [
      "Primary candidate: tighten p5_sketch response structure without modifying the artifact."
    ],
    implementation_suggestions: [
      "Label every refinement as advisory metadata, not an artifact edit."
    ],
    alternative_refinement_paths: [
      "Capability-first path: clarify target limits first.",
      "Dependency-first path: resolve handoffs and conflicts first."
    ],
    hitl_questions: [
      "Which advisory refinement should be prioritized first?"
    ],
    prompt_guidance: [
      "Use Artifact Refiner output as metadata-only refinement intelligence."
    ],
    authority_boundary:
      "The Artifact Refiner derives refinement intelligence from planning metadata only; it does not modify artifacts.",
    evidence: ["Artifact critic: medium risk; 1 weakness signals."]
  };
}

function artifactIntelligenceSynthesisFixture() {
  return {
    role: "artifact_intelligence_synthesis",
    synthesis_confidence: 0.83,
    synthesis_summary:
      "Artifact intelligence synthesis reports needs_caveats readiness with medium implementation risk.",
    recommended_artifact_path:
      "Lead with primary_artifact with supporting artifacts runtime_notes; keep sections separated and advisory.",
    recommended_strategy_summary:
      "Use primary_with_supporting_sections with 2 ordered steps; treat critic risk as medium.",
    recommended_runtime_direction:
      "Document preferred runtime metadata as advisory only: p5_js; use capability matrix caveats for p5_js.",
    major_strengths: [
      "Artifact shape is declared as runnable_code / p5_sketch.",
      "Synthesis remains metadata-only and does not execute."
    ],
    major_weaknesses: [
      "Priority improvement remains unresolved: Resolve dependency risks."
    ],
    major_risks: [
      "Unsupported runtime remains advisory only: glsl."
    ],
    dependency_overview:
      "8 nodes, 7 edges, 0 blocking dependencies, 1 conflicts, and downstream consumers prompt_renderer.",
    capability_overview:
      "Strongest targets: p5_js; weakest targets: glsl; 1 unsupported/risky capabilities; artifact fit strong.",
    refinement_overview:
      "Refiner confidence 0.79; 1 priority improvements; 1 candidates; top priority: Resolve dependency risks.",
    critique_overview:
      "Critic risk medium; confidence 0.82; 1 weaknesses; 1 open questions.",
    implementation_readiness: "needs_caveats",
    implementation_complexity: "medium",
    implementation_risk: "medium",
    implementation_priority: "medium",
    hitl_questions: [
      "Should synthesis risks be resolved before generation?"
    ],
    prompt_guidance: [
      "Use Artifact Intelligence Synthesis as metadata-only prompt guidance."
    ],
    authority_boundary:
      "The Artifact Intelligence Synthesis capability summarizes metadata only; it does not modify artifacts.",
    evidence: ["Artifact critic: medium risk; 0.82 confidence."]
  };
}

function artifactMergePlannerFixture() {
  return {
    role: "artifact_merge_planner",
    merge_confidence: 0.81,
    merge_summary:
      "Artifact merge planning recommends primary_with_supporting_sections with visible boundaries.",
    merge_strategy: "primary_with_supporting_sections",
    composition_strategy:
      "Compose primary artifact first, then attach supporting sections with explicit labels.",
    artifact_boundaries: [
      "Primary boundary: primary_artifact remains the lead runnable_code artifact."
    ],
    artifact_join_points: [
      "primary_artifact -> runtime_notes: Runtime notes support the primary artifact."
    ],
    artifact_separation_points: [
      "Lead with Primary p5.js artifact as the only primary artifact."
    ],
    integration_order: [
      "1. primary_artifact: produce",
      "2. runtime_notes: document"
    ],
    composition_risks: [
      "Do not let supporting artifacts expand implementation scope."
    ],
    dependency_merge_risks: [
      "Runtime notes conflict with primary output structure."
    ],
    runtime_merge_risks: [
      "Unsupported runtime must not be merged into path: glsl."
    ],
    capability_merge_risks: [
      "Native shader pipelines require additional scaffolding."
    ],
    recommended_merge_path:
      "Follow synthesis path as advisory merge guidance: Lead with primary_artifact.",
    alternative_merge_paths: [
      "Alternative: preserve all artifacts as separate sections."
    ],
    rejected_merge_paths: [
      "Reject automatic artifact merging because this planner is metadata-only."
    ],
    hitl_questions: [
      "Should merge planning preserve separation until risks resolve?"
    ],
    prompt_guidance: [
      "Use Artifact Merge Planner output as metadata-only merge guidance."
    ],
    authority_boundary:
      "The Artifact Merge Planner recommends merge strategy only; it does not merge artifacts.",
    evidence: ["Multi-artifact strategy: 1 supporting."]
  };
}

function artifactExportIntelligenceFixture() {
  return {
    role: "artifact_export_intelligence",
    export_confidence: 0.78,
    export_summary:
      "Artifact export intelligence reports ready_with_caveats readiness.",
    export_targets: [
      "inline_response",
      "single_source_artifact",
      "multi_artifact_package"
    ],
    preferred_export_target: "multi_artifact_package",
    export_format_recommendations: [
      "Represent p5_sketch as advisory runnable_code export metadata."
    ],
    export_readiness: "ready_with_caveats",
    export_requirements: [
      "1. primary_artifact: produce",
      "Runtime requires p5.js lifecycle."
    ],
    export_constraints: [
      "Export Intelligence is metadata-only and cannot write files or packages."
    ],
    export_risks: [
      "Unsupported runtime cannot be exported directly: glsl."
    ],
    runtime_export_notes: [
      "Preferred runtimes are advisory export metadata: p5_js."
    ],
    artifact_package_notes: [
      "Package primary artifact first: primary_artifact."
    ],
    portability_notes: ["Runtime portability is medium."],
    interoperability_notes: ["Runtime interoperability is medium."],
    documentation_requirements: [
      "Document that export intelligence is advisory metadata only."
    ],
    downstream_tool_handoffs: [
      "Future export workflows must consume this metadata explicitly; this workflow does not trigger export."
    ],
    rejected_export_paths: [
      "Reject direct file export because this engine is metadata-only."
    ],
    hitl_questions: [
      "Should export remain deferred until risks are resolved?"
    ],
    prompt_guidance: [
      "Use Artifact Export Intelligence as metadata-only export guidance."
    ],
    authority_boundary:
      "The Artifact Export Intelligence engine recommends export paths only; it does not export files.",
    evidence: ["Capability export fit: moderate."]
  };
}

function artifactEngineContractRegistryFixture() {
  const engineContracts = [
    "artifact_planner",
    "artifact_dependency_graph",
    "runtime_compatibility_engine",
    "artifact_capability_matrix",
    "multi_artifact_strategy",
    "artifact_critic",
    "artifact_refiner",
    "artifact_intelligence_synthesis",
    "artifact_merge_planner",
    "artifact_export_intelligence"
  ].map((engineId) => ({
    engine_id: engineId,
    engine_name: engineId.replaceAll("_", " "),
    engine_version: "v3.3",
    engine_category: "artifact_intelligence",
    authority_boundary: "Metadata-only contract; no behavior changes.",
    required_inputs: ["assistant_request"],
    optional_inputs: ["route_decision"],
    produced_metadata: [`${engineId}_metadata`],
    produced_signals: ["prompt_guidance"],
    confidence_signals: ["evidence"],
    ambiguity_signals: ["hitl_questions"],
    risk_signals: ["risks"],
    escalation_candidates: ["hitl_questions"],
    downstream_dependencies: [],
    upstream_dependencies:
      engineId === "artifact_export_intelligence"
        ? ["artifact_merge_planner"]
        : [],
    cacheability: "deterministic_with_upstream_metadata",
    parallelization_support: "requires_ordered_upstream_metadata",
    estimated_cost_metadata: {
      relative_cost: "low",
      external_provider_calls: false,
      cost_basis: "Local metadata derivation.",
      cache_sensitivity: "Request and upstream metadata sensitive."
    },
    estimated_latency_metadata: {
      relative_latency: "low",
      latency_basis: "Bounded local metadata construction.",
      blocking_inputs: ["assistant_request"]
    },
    serialization_version: "artifact_engine_contract.v1",
    future_agent_hooks: ["v4_planner_agent_contract"],
    future_execution_hooks: ["v5_execution_optimization_readiness"]
  }));

  return {
    role: "artifact_intelligence_engine_contract_registry",
    engine_category: "artifact_intelligence",
    serialization_version: "artifact_engine_contract_registry.v1",
    authority_boundary:
      "Contracts describe metadata surfaces only; they do not change behavior.",
    engine_contracts: engineContracts,
    engine_ids: engineContracts.map((contract) => contract.engine_id),
    contract_count: engineContracts.length,
    future_agent_consumers: [
      "v4_planner_agent",
      "v4_artifact_agent",
      "v4_runtime_agent"
    ]
  };
}

function audioVisualScenePhaseFixture(phase: string, title: string) {
  return {
    phase,
    title,
    scene_function: `Scene function for ${title}.`,
    visual_state: `Visual state for ${title}.`,
    motion_state: `Motion state for ${title}.`,
    audio_state: `Audio state for ${title}.`,
    rhythm_state: `Rhythm state for ${title}.`,
    camera_state: `Camera state for ${title}.`,
    motif_state: `Motif state for ${title}.`,
    emotional_state: `Emotional state for ${title}.`,
    procedural_state: `Procedural state for ${title}.`,
    cue_ids: [`${phase}_visual`, `${phase}_sync`],
    transition_out: `Transition out from ${title}.`,
    evidence: [`Scene phase: ${phase}.`]
  };
}

describe("assistant stream client", () => {
  it("parses backend NDJSON lines into typed events", () => {
    const event = parseAssistantStreamLine(
      '{"event_type":"status","sequence":0,"payload":{"code":"request_received"}}'
    );

    expect(event).toEqual({
      event_type: "status",
      sequence: 0,
      payload: {
        code: "request_received"
      }
    });
  });

  it("rejects invalid stream event shapes", () => {
    expect(() =>
      parseAssistantStreamLine('{"event_type":"unknown","sequence":0,"payload":{}}')
    ).toThrow(AssistantStreamError);
  });

  it("decodes chunked response bodies", async () => {
    const response = new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(
            new TextEncoder().encode(
              '{"event_type":"status","sequence":0,"payload":{"code":"request'
            )
          );
          controller.enqueue(
            new TextEncoder().encode(
              '_received"}}\n{"event_type":"final","sequence":1,"payload":{"answer":"Done."}}\n'
            )
          );
          controller.close();
        }
      })
    );

    const events = [];
    for await (const event of decodeAssistantStream(response)) {
      events.push(event);
    }

    expect(events.map((event) => event.event_type)).toEqual(["status", "final"]);
    expect(events[1].payload.answer).toBe("Done.");
  });

  it("posts frontend stream requests to the configured endpoint", async () => {
    const fetchImpl = vi.fn(
      async (_url: Parameters<typeof fetch>[0], _init?: Parameters<typeof fetch>[1]) => {
        return new Response(
          '{"event_type":"final","sequence":0,"payload":{"answer":"Done."}}\n'
        );
      }
    );

    const events = [];
    for await (const event of streamAssistantEvents(
      {
        query: "Generate particles.",
        attachments: [
          {
            type: "image",
            id: "image-reference-1",
            name: "palette.png",
            mimeType: "image/png",
            sizeBytes: 128,
            dataUrl: "data:image/png;base64,cGFsZXR0ZQ=="
          }
        ],
        conversationId: "browser-session",
        domain: "webgpu_wgsl",
        mode: "generate",
        artifactRefinement: {
          artifactId: "source-sketch",
          title: "aurora-field.p5.js",
          language: "p5.js",
          content: "function draw() { background(0); }",
          instruction: "Make this more organic.",
          domain: "p5_js",
          runtime: "p5",
          rendererId: "surface.p5",
          previewEligible: true,
          qualityScore: 0.91,
          qualityRank: 1,
          critiqueRationale: "Strong visual candidate.",
          refinementGuidance: "Soften particle motion.",
          creativeTranslation: {
            outputModality: "visual",
            creativeIntent: "Create a calm particle field.",
            symbolicReferences: [],
            geometricReferences: [],
            musicalReferences: [],
            moodAtmosphere: ["calm"],
            movementLanguage: ["drift"],
            colorMaterialDirection: [],
            runtimeRecommendations: ["p5.js"],
            structureDirection: [],
            generationConstraints: [],
            refinementTargets: ["Preserve atmosphere: calm"],
            sacredGeometry: {
              concepts: ["mandala"],
              geometricStructure: ["Build nested rings."],
              symmetryType: ["Use radial symmetry."],
              movementBehavior: [],
              visualComposition: [],
              colorMaterialDirection: [],
              runtimeRecommendations: ["p5.js"],
              audioImplications: [],
              generationConstraints: [
                "Do not add unsupported symbolic claims."
              ]
            },
            shaderPresets: {
              presets: ["glow"],
              colorBehavior: ["Use a bright core color."],
              lightMaterialBehavior: ["Use bounded emission layers."],
              motionBehavior: ["Pulse intensity slowly."],
              shaderStructure: ["Separate an emission mask."],
              runtimeSuitability: [
                "Use the selected compatible runtime: p5.js."
              ],
              performanceConstraints: [
                "Use a bounded number of glow layers."
              ]
            },
            visualStyle: {
              styles: ["minimal"],
              paletteBehavior: ["Use one dominant tone."],
              contrastBehavior: ["Use clear value hierarchy."],
              compositionTendencies: ["Use deliberate negative space."],
              motionTendencies: ["Use slow readable transitions."],
              textureTendencies: ["Keep surfaces clean."],
              spatialOrganization: ["Favor a stable focal point."],
              runtimeSuitability: [
                "Use the selected compatible runtime: p5.js."
              ]
            }
          }
        }
      },
      {
        endpoint: "http://backend.test/api/assistant/stream",
        fetchImpl
      }
    )) {
      events.push(event);
    }

    expect(fetchImpl).toHaveBeenCalledWith(
      "http://backend.test/api/assistant/stream",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          Accept: "application/x-ndjson",
          "Content-Type": "application/json"
        })
      })
    );
    expect(JSON.parse(String(fetchImpl.mock.calls[0][1]?.body))).toMatchObject({
      conversationId: "browser-session",
      domain: "webgpu_wgsl",
      mode: "generate",
      query: "Generate particles.",
      artifactRefinement: expect.objectContaining({
        artifactId: "source-sketch",
        title: "aurora-field.p5.js",
        content: "function draw() { background(0); }",
        instruction: "Make this more organic.",
        domain: "p5_js",
        runtime: "p5",
        rendererId: "surface.p5",
        previewEligible: true,
        creativeTranslation: expect.objectContaining({
          creativeIntent: "Create a calm particle field.",
          moodAtmosphere: ["calm"],
          sacredGeometry: expect.objectContaining({
            concepts: ["mandala"],
            runtimeRecommendations: ["p5.js"]
          }),
          shaderPresets: expect.objectContaining({
            presets: ["glow"],
            runtimeSuitability: [
              "Use the selected compatible runtime: p5.js."
            ]
          }),
          visualStyle: expect.objectContaining({
            styles: ["minimal"],
            runtimeSuitability: [
              "Use the selected compatible runtime: p5.js."
            ]
          })
        })
      }),
      attachments: [
        expect.objectContaining({
          type: "image",
          name: "palette.png",
          mimeType: "image/png",
          dataUrl: "data:image/png;base64,cGFsZXR0ZQ=="
        })
      ]
    });
    expect(events).toHaveLength(1);
  });

  it("throws a stream error for failed HTTP responses", async () => {
    const events = decodeAssistantStream(new Response("Nope.", { status: 503 }));

    await expect(events.next()).rejects.toThrow(AssistantStreamError);
  });

  it("builds a structured stream error from error events", () => {
    const event: AssistantStreamEvent = {
      event_type: "error",
      sequence: 3,
      payload: {
        code: "provider_unavailable",
        message: "Provider unavailable."
      }
    };

    expect(readStreamEventError(event)).toMatchObject({
      category: "stream",
      subsystem: "generation_provider",
      type: "provider_unavailable",
      userMessage: "The model provider is unavailable for this live response.",
      retryLabel: "Send prompt again",
      resetLabel: "Clear workspace session"
    });
  });

  it("maps backend events to workflow nodes", () => {
    const events: AssistantStreamEvent[] = [
      {
        event_type: "status",
        sequence: 0,
        payload: { code: "request_received" }
      },
      {
        event_type: "status",
        sequence: 1,
        payload: { code: "route_selected" }
      },
      {
        event_type: "planning",
        sequence: 2,
        payload: { code: "creative_plan_prepared" }
      },
      {
        event_type: "planning",
        sequence: 3,
        payload: { code: "creative_director_prepared" }
      },
      {
        event_type: "planning",
        sequence: 4,
        payload: { code: "creative_reasoning_prepared" }
      },
      {
        event_type: "prompt_rendered",
        sequence: 5,
        payload: { code: "prompt_rendered" }
      },
      {
        event_type: "token_delta",
        sequence: 6,
        payload: { text: "Hello" }
      },
      {
        event_type: "node_started",
        sequence: 7,
        payload: { code: "node_started", node: "review" }
      },
      {
        event_type: "review_failed",
        sequence: 8,
        payload: { code: "review_failed" }
      },
      {
        event_type: "refinement_requested",
        sequence: 9,
        payload: { code: "refinement_requested" }
      },
      {
        event_type: "retry_started",
        sequence: 10,
        payload: { code: "retry_started" }
      },
      {
        event_type: "refinement_completed",
        sequence: 11,
        payload: { code: "refinement_completed" }
      },
      {
        event_type: "node_completed",
        sequence: 12,
        payload: { code: "node_completed", node: "refinement" }
      },
      {
        event_type: "retry_completed",
        sequence: 13,
        payload: { code: "retry_completed" }
      },
      {
        event_type: "review_passed",
        sequence: 14,
        payload: { code: "review_passed" }
      },
      {
        event_type: "node_failed",
        sequence: 15,
        payload: { code: "node_failed", node: "generation" }
      },
      {
        event_type: "artifact_extracted",
        sequence: 16,
        payload: { code: "artifact_extracted" }
      },
      {
        event_type: "preview_artifact",
        sequence: 17,
        payload: { code: "preview_artifact_prepared", status: "succeeded" }
      },
      {
        event_type: "final",
        sequence: 18,
        payload: { answer: "Done." }
      }
    ];

    expect(events.map(workflowNodeFromAssistantStreamEvent)).toEqual([
      "intake",
      "routing",
      "planning",
      "director",
      "reasoning",
      "prompt_rendering",
      "generation",
      "review",
      "review",
      "review",
      "review",
      "refinement",
      "refinement",
      "review",
      "review",
      "generation",
      "artifact_extraction",
      "preview_preparation",
      "finalization"
    ]);
  });

  it("reads workflow runtime metadata from enriched stream events", () => {
    const creativePlan = {
      output_modality: "visual",
      generation_strategy: "Generate one p5 candidate.",
      recommended_runtime: "p5",
      recommended_renderer_id: "surface.p5",
      recommended_preview_target: "browser_sandbox",
      recommended_shader_style: "glow",
      candidate_count: 1,
      refinement_budget: 1,
      expected_complexity: "medium",
      estimated_token_cost: 2800,
      export_readiness: "ready",
      runtime_available: true,
      runtime_support_summary: "p5.js browser preview is available.",
      plan_steps: ["Target p5 output."],
      constraints: ["Keep code browser-safe."],
      evidence: ["Route selected: generate."]
    };
    const intentDimension = (
      name: string,
      explicitness: string,
      summary: string,
      signals: string[]
    ) => ({
      name,
      explicitness,
      summary,
      signals,
      guidance: [`Preserve ${name} intent.`]
    });
    const creativeIntent = {
      role: "creative_intent_decomposer",
      normalized_intent: "Generate a luminous particle field.",
      primary_expression: "emotional=awe; motion=drift; light color=glow",
      narrative_intent: intentDimension(
        "narrative",
        "absent",
        "No explicit narrative intent was detected.",
        []
      ),
      symbolic_intent: intentDimension(
        "symbolic",
        "inferred",
        "Use symbolic cues around constellation as an atomic design dimension.",
        ["constellation"]
      ),
      emotional_intent: intentDimension(
        "emotional",
        "explicit",
        "Use emotional cues around awe as an atomic design dimension.",
        ["awe"]
      ),
      geometric_intent: intentDimension(
        "geometric",
        "absent",
        "No explicit geometric intent was detected.",
        []
      ),
      motion_intent: intentDimension(
        "motion",
        "explicit",
        "Use motion cues around drift as an atomic design dimension.",
        ["drift"]
      ),
      rhythm_intent: intentDimension(
        "rhythm",
        "absent",
        "No explicit rhythm intent was detected.",
        []
      ),
      light_color_intent: intentDimension(
        "light_color",
        "explicit",
        "Use light color cues around glow as an atomic design dimension.",
        ["glow"]
      ),
      audio_intent: intentDimension(
        "audio",
        "absent",
        "No explicit audio intent was detected.",
        []
      ),
      interaction_intent: intentDimension(
        "interaction",
        "absent",
        "No explicit interaction intent was detected.",
        []
      ),
      climax_transformation_intent: intentDimension(
        "climax_transformation",
        "absent",
        "No explicit climax transformation intent was detected.",
        []
      ),
      abstraction_level: "abstract",
      experiential_goal:
        "Create an abstract experience that balances emotional, motion, and light color cues.",
      unresolved_intent_gaps: ["Interaction is not explicit."],
      hitl_questions: ["What should change when the user interacts?"],
      atomic_dimensions: [
        intentDimension(
          "narrative",
          "absent",
          "No explicit narrative intent was detected.",
          []
        ),
        intentDimension(
          "symbolic",
          "inferred",
          "Use symbolic cues around constellation as an atomic design dimension.",
          ["constellation"]
        ),
        intentDimension(
          "emotional",
          "explicit",
          "Use emotional cues around awe as an atomic design dimension.",
          ["awe"]
        ),
        intentDimension(
          "geometric",
          "absent",
          "No explicit geometric intent was detected.",
          []
        ),
        intentDimension(
          "motion",
          "explicit",
          "Use motion cues around drift as an atomic design dimension.",
          ["drift"]
        ),
        intentDimension(
          "rhythm",
          "absent",
          "No explicit rhythm intent was detected.",
          []
        ),
        intentDimension(
          "light_color",
          "explicit",
          "Use light color cues around glow as an atomic design dimension.",
          ["glow"]
        ),
        intentDimension(
          "audio",
          "absent",
          "No explicit audio intent was detected.",
          []
        ),
        intentDimension(
          "interaction",
          "absent",
          "No explicit interaction intent was detected.",
          []
        ),
        intentDimension(
          "climax_transformation",
          "absent",
          "No explicit climax transformation intent was detected.",
          []
        )
      ],
      prompt_guidance: [
        "Use decomposed intent dimensions as design constraints."
      ],
      authority_boundary:
        "The Creative Intent Decomposer structures user intent for inspection only.",
      evidence: ["Active intent dimensions: symbolic, emotional, motion."]
    };
    const expectedCreativeIntent = {
      role: creativeIntent.role,
      normalizedIntent: creativeIntent.normalized_intent,
      primaryExpression: creativeIntent.primary_expression,
      narrativeIntent: creativeIntent.narrative_intent,
      symbolicIntent: creativeIntent.symbolic_intent,
      emotionalIntent: creativeIntent.emotional_intent,
      geometricIntent: creativeIntent.geometric_intent,
      motionIntent: creativeIntent.motion_intent,
      rhythmIntent: creativeIntent.rhythm_intent,
      lightColorIntent: creativeIntent.light_color_intent,
      audioIntent: creativeIntent.audio_intent,
      interactionIntent: creativeIntent.interaction_intent,
      climaxTransformationIntent:
        creativeIntent.climax_transformation_intent,
      abstractionLevel: creativeIntent.abstraction_level,
      experientialGoal: creativeIntent.experiential_goal,
      unresolvedIntentGaps: creativeIntent.unresolved_intent_gaps,
      hitlQuestions: creativeIntent.hitl_questions,
      atomicDimensions: creativeIntent.atomic_dimensions,
      promptGuidance: creativeIntent.prompt_guidance,
      authorityBoundary: creativeIntent.authority_boundary,
      evidence: creativeIntent.evidence
    };
    const creativeHierarchy = {
      role: "creative_hierarchy_planner",
      primary_creative_priorities: [
        {
          dimension: "visual_impact",
          tier: "primary",
          rank: 1,
          priority_score: 9,
          source: "explicit",
          rationale: "visual impact should dominate because score 9 is strongest.",
          evidence: ["keyword:visual_impact"],
          sacrifice_guidance:
            "Do not sacrifice visual_impact unless the user explicitly redirects."
        }
      ],
      secondary_creative_priorities: [
        {
          dimension: "performance",
          tier: "secondary",
          rank: 1,
          priority_score: 4,
          source: "constraint",
          rationale:
            "performance supports coherence but should not override primary intent.",
          evidence: ["constraint:performance"],
          sacrifice_guidance:
            "Keep performance visible when reducing creative scope."
        }
      ],
      non_negotiable_dimensions: ["visual_impact"],
      flexible_dimensions: ["audio", "rhythm"],
      priority_rationale: ["Primary hierarchy: visual_impact."],
      priority_conflicts: [
        "Visual impact may compete with performance priority."
      ],
      hierarchy_confidence: 0.78,
      hitl_questions: ["Should visual richness or performance win first?"],
      prompt_guidance: [
        "Use hierarchy priorities as ordering guidance, not as new features."
      ],
      authority_boundary:
        "The Creative Hierarchy Planner ranks creative priorities for inspection only.",
      evidence: ["Top hierarchy scores: visual_impact=9."]
    };
    const expectedCreativeHierarchy = {
      role: creativeHierarchy.role,
      primaryCreativePriorities:
        creativeHierarchy.primary_creative_priorities.map((item) => ({
          dimension: item.dimension,
          tier: item.tier,
          rank: item.rank,
          priorityScore: item.priority_score,
          source: item.source,
          rationale: item.rationale,
          evidence: item.evidence,
          sacrificeGuidance: item.sacrifice_guidance
        })),
      secondaryCreativePriorities:
        creativeHierarchy.secondary_creative_priorities.map((item) => ({
          dimension: item.dimension,
          tier: item.tier,
          rank: item.rank,
          priorityScore: item.priority_score,
          source: item.source,
          rationale: item.rationale,
          evidence: item.evidence,
          sacrificeGuidance: item.sacrifice_guidance
        })),
      nonNegotiableDimensions: creativeHierarchy.non_negotiable_dimensions,
      flexibleDimensions: creativeHierarchy.flexible_dimensions,
      priorityRationale: creativeHierarchy.priority_rationale,
      priorityConflicts: creativeHierarchy.priority_conflicts,
      hierarchyConfidence: creativeHierarchy.hierarchy_confidence,
      hitlQuestions: creativeHierarchy.hitl_questions,
      promptGuidance: creativeHierarchy.prompt_guidance,
      authorityBoundary: creativeHierarchy.authority_boundary,
      evidence: creativeHierarchy.evidence
    };
    const creativeStrategy = {
      role: "creative_strategy_engine",
      primary_strategy: "particle_cosmology",
      confidence: 0.75,
      rationale: "Particle Cosmology best matches detected signals: particle.",
      creative_goals: ["Evoke a coherent world through small moving elements."],
      symbolic_alignment: ["Particle Cosmology", "constellation"],
      alternative_strategies: [
        {
          strategy: "field_dynamics",
          confidence: 0.51,
          rationale: "Also relevant because of drift."
        }
      ],
      strategy_directives: [
        "Frame the concept around collective motion and spatial density."
      ],
      implementation_boundary:
        "The Creative Strategy Engine selects high-level artistic strategy only.",
      evidence: ["Primary signals: particle."]
    };
    const creativeTechniques = {
      role: "creative_technique_selector",
      primary_technique: "particle_systems",
      confidence: 0.79,
      rationale: "Particle Systems best matches detected signals: particle.",
      strategy_alignment: "particle_cosmology",
      compatibility: "strong",
      complexity_pressure: "medium",
      performance_pressure: "high",
      artistic_suitability: ["Supports strategy: particle_cosmology."],
      implementation_notes: ["Keep counts and lifetimes bounded."],
      alternative_techniques: [
        {
          technique: "noise_fields",
          confidence: 0.51,
          rationale: "Also relevant because of drift."
        }
      ],
      technique_constraints: [
        "Do not treat technique selection as runtime or renderer selection."
      ],
      selection_boundary:
        "The Creative Technique Selector recommends creative implementation techniques only.",
      evidence: ["Primary technique signals: particle."]
    };
    const creativeConstraints = {
      role: "creative_constraint_solver",
      intent_summary: "Generate a luminous field.",
      output_goal: "Generate one p5 candidate.",
      modality: "visual",
      runtime_fit: "supported",
      recommended_runtime: "p5",
      complexity_pressure: "medium",
      safety_pressure: "low",
      performance_pressure: "medium",
      cost_pressure: "low",
      hitl_advisable: false,
      hitl_reason: null,
      active_constraints: [
        {
          axis: "runtime",
          severity: "info",
          summary: "p5.js browser preview is available.",
          recommendation: "Use p5 through surface.p5.",
          evidence: ["recommended_runtime: p5"]
        }
      ],
      tradeoffs: [
        {
          source_axis: "complexity",
          target_axis: "performance",
          severity: "watch",
          summary: "Keep effects bounded.",
          recommendation: "Reduce particle count before adding candidates."
        }
      ],
      conflicts: [],
      prompt_guidance: ["Target p5 output."],
      authority_boundary: "The solver structures trade-offs for inspection.",
      evidence: ["Route selected: generate."]
    };
    const creativeConstraintPriorities = {
      role: "creative_constraint_prioritizer",
      non_negotiable_constraints: [
        {
          category: "symbolic_fidelity",
          priority_level: "non_negotiable",
          rank: 1,
          priority_score: 12,
          source: "hierarchy",
          rationale:
            "symbolic fidelity is non-negotiable because hierarchy evidence dominates.",
          negotiation_guidance:
            "Do not relax symbolic fidelity without explicit user confirmation.",
          evidence: ["hierarchy primary:symbolism"]
        }
      ],
      high_priority_constraints: [
        {
          category: "performance",
          priority_level: "high_priority",
          rank: 2,
          priority_score: 8,
          source: "solver",
          rationale:
            "performance is high priority because solver pressure is visible.",
          negotiation_guidance:
            "Protect performance unless a non-negotiable constraint conflicts.",
          evidence: ["solver performance medium"]
        }
      ],
      flexible_constraints: [],
      relaxable_constraints: [
        {
          category: "interaction_complexity",
          priority_level: "relaxable",
          rank: 3,
          priority_score: 4,
          source: "explicit",
          rationale:
            "interaction complexity is relaxable because optional wording is present.",
          negotiation_guidance:
            "Relax interaction complexity before weakening protected constraints.",
          evidence: ["keyword:interaction_complexity"]
        }
      ],
      sacrificial_constraints: [
        {
          category: "previewability",
          priority_level: "sacrificial",
          rank: 4,
          priority_score: 2,
          source: "explicit",
          rationale:
            "previewability is sacrificial because the user allowed sacrifice.",
          negotiation_guidance:
            "Sacrifice previewability first if feasibility requires reduction.",
          evidence: ["keyword:previewability"]
        }
      ],
      priority_rationale: [
        "Constraint priorities are ranked from user emphasis and solver pressure."
      ],
      negotiation_notes: [
        "Do not relax symbolic fidelity without explicit user confirmation."
      ],
      conflict_relationships: [
        {
          protected_category: "symbolic_fidelity",
          competing_category: "implementation_simplicity",
          severity: "watch",
          summary:
            "symbolic_fidelity has stronger priority than implementation_simplicity.",
          negotiation_note:
            "Protect symbolic_fidelity first; relax implementation_simplicity explicitly.",
          hitl_recommended: true
        }
      ],
      hitl_questions: [
        "May implementation_simplicity be relaxed to protect symbolic_fidelity?"
      ],
      prompt_guidance: [
        "Protect non-negotiable constraints before optimizing flexible ones."
      ],
      authority_boundary:
        "The Creative Constraint Prioritizer ranks constraint importance only.",
      evidence: ["Solver pressures: performance medium."]
    };
    const expectedCreativeConstraintPriorities = {
      role: creativeConstraintPriorities.role,
      nonNegotiableConstraints:
        creativeConstraintPriorities.non_negotiable_constraints.map((item) => ({
          category: item.category,
          priorityLevel: item.priority_level,
          rank: item.rank,
          priorityScore: item.priority_score,
          source: item.source,
          rationale: item.rationale,
          negotiationGuidance: item.negotiation_guidance,
          evidence: item.evidence
        })),
      highPriorityConstraints:
        creativeConstraintPriorities.high_priority_constraints.map((item) => ({
          category: item.category,
          priorityLevel: item.priority_level,
          rank: item.rank,
          priorityScore: item.priority_score,
          source: item.source,
          rationale: item.rationale,
          negotiationGuidance: item.negotiation_guidance,
          evidence: item.evidence
        })),
      flexibleConstraints: [],
      relaxableConstraints:
        creativeConstraintPriorities.relaxable_constraints.map((item) => ({
          category: item.category,
          priorityLevel: item.priority_level,
          rank: item.rank,
          priorityScore: item.priority_score,
          source: item.source,
          rationale: item.rationale,
          negotiationGuidance: item.negotiation_guidance,
          evidence: item.evidence
        })),
      sacrificialConstraints:
        creativeConstraintPriorities.sacrificial_constraints.map((item) => ({
          category: item.category,
          priorityLevel: item.priority_level,
          rank: item.rank,
          priorityScore: item.priority_score,
          source: item.source,
          rationale: item.rationale,
          negotiationGuidance: item.negotiation_guidance,
          evidence: item.evidence
        })),
      priorityRationale: creativeConstraintPriorities.priority_rationale,
      negotiationNotes: creativeConstraintPriorities.negotiation_notes,
      conflictRelationships:
        creativeConstraintPriorities.conflict_relationships.map((item) => ({
          protectedCategory: item.protected_category,
          competingCategory: item.competing_category,
          severity: item.severity,
          summary: item.summary,
          negotiationNote: item.negotiation_note,
          hitlRecommended: item.hitl_recommended
        })),
      hitlQuestions: creativeConstraintPriorities.hitl_questions,
      promptGuidance: creativeConstraintPriorities.prompt_guidance,
      authorityBoundary: creativeConstraintPriorities.authority_boundary,
      evidence: creativeConstraintPriorities.evidence
    };
    const runtimeCapabilities = {
      role: "runtime_capability_reasoner",
      output_goal: "Generate one p5 candidate.",
      likely_candidates: ["p5_js", "canvas", "svg"],
      candidate_runtimes: [
        {
          runtime: "p5_js",
          label: "p5.js",
          suitability: "strong",
          confidence: 0.86,
          strategy_alignment: "strong",
          technique_compatibility: "strong",
          output_goal_fit: "strong",
          implementation_complexity: "medium",
          performance_pressure: "high",
          preview_support: "backend_preview_supported",
          strengths: ["Fits the selected creative technique."],
          limitations: ["Less natural for deep 3D scenes."],
          risks: ["High performance pressure requires bounded effect scope."],
          prompt_guidance: ["Use p5.js capability for sketch output."],
          evidence: ["Capability score: 12."]
        }
      ],
      strategy_context: "particle_cosmology with confidence 0.75.",
      technique_context: "particle_systems with high performance pressure.",
      constraint_context:
        "Runtime fit supported; complexity medium; performance medium; HITL false.",
      hitl_advisable: false,
      hitl_reason: null,
      prompt_guidance: [
        "Use runtime capability metadata to explain trade-offs."
      ],
      authority_boundary:
        "The Runtime Capability Reasoner evaluates runtime fit for inspection only.",
      evidence: ["Top runtime scores: p5_js=12."]
    };
    const creativeTradeoffs = {
      role: "creative_tradeoff_explorer",
      output_goal: "Generate one p5 candidate.",
      primary_tradeoffs: [
        {
          source_axis: "creative_expressiveness",
          target_axis: "implementation_complexity",
          severity: "risk",
          summary:
            "Expressive strategy and technique choices can increase implementation scope.",
          creative_benefit:
            "Particle cosmology with particle systems preserves a distinct direction.",
          technical_cost:
            "Requires managing plan complexity medium and technique complexity medium.",
          runtime_implication:
            "p5.js has strong suitability and backend preview support.",
          mitigation:
            "Keep the selected strategy visible while bounding the number of systems.",
          director_discussion_point:
            "Should the output prioritize richness or a simpler implementation?",
          hitl_recommended: false,
          evidence: ["Strategy: particle_cosmology."]
        }
      ],
      creative_benefits: ["Fits the selected creative technique."],
      technical_costs: ["Expected complexity: medium."],
      runtime_risks: ["High performance pressure requires bounded effect scope."],
      performance_concerns: ["Technique performance pressure: high."],
      complexity_risks: ["Plan complexity is medium."],
      fidelity_risks: [],
      cost_sensitivity: "low",
      safety_concerns: [],
      maintainability_concerns: ["Keep particle_systems behavior readable."],
      hitl_advisable: false,
      hitl_reason: null,
      director_discussion_points: [
        "Should the output prioritize richness or a simpler implementation?"
      ],
      prompt_guidance: [
        "Use trade-off metadata to explain consequences, not to select an outcome."
      ],
      authority_boundary:
        "The Creative Trade-off Explorer structures consequences and discussion points only.",
      evidence: ["Runtime candidates: p5_js, canvas, svg."]
    };
    const creativeDirector = {
      role: "creative_assistant_director",
      creative_brief: "Generate a luminous field.",
      ambiguity_level: "low",
      ambiguity_signals: [],
      retrieval_posture: "available",
      modality_direction: "visual",
      runtime_direction: "p5",
      planning_focus: ["Generate one p5 candidate."],
      critique_focus: ["Check output against runtime support."],
      refinement_focus: ["Use bounded refinement only for concrete gaps."],
      next_actions: ["Render the prompt and continue."],
      hitl_required: false,
      hitl_reason: null,
      authority_boundary: "The user remains the Creative Director.",
      evidence: ["Route selected: generate."]
    };
    const creativeReasoning = {
      role: "creative_reasoning_engine",
      recommended_creative_direction:
        "Recommend a particle_cosmology direction expressed through particle_systems because it best preserves the luminous field.",
      reasoning_path: [
        {
          stage: "strategy",
          claim: "Use particle_cosmology as the conceptual spine.",
          because:
            "Particle Cosmology best matches detected signals: particle.",
          implications: ["Keep one visible creative idea in focus."]
        },
        {
          stage: "technique",
          claim: "Translate that strategy through particle_systems.",
          because:
            "Particle Systems best matches detected signals: particle.",
          implications: ["Technique choices should serve the strategy."]
        },
        {
          stage: "runtime",
          claim: "Shape implementation around inspected capability: p5.js.",
          because: "p5.js shows strong suitability.",
          implications: ["Runtime guidance remains non-binding."]
        },
        {
          stage: "tradeoff",
          claim:
            "Manage the main consequence: creative_expressiveness vs implementation_complexity.",
          because:
            "The creative benefit is distinct direction while the technical cost is complexity.",
          implications: ["Prefer bounded implementation over feature growth."]
        },
        {
          stage: "recommendation",
          claim:
            "Recommend a particle_cosmology direction expressed through particle_systems because it best preserves the luminous field.",
          because:
            "Strategy, technique, runtime capability, and trade-off signals converge.",
          implications: ["Use this as the prompt spine before generation."]
        }
      ],
      evidence_chain: [
        {
          source: "creative_strategy",
          signal: "particle_cosmology confidence 0.75.",
          interpretation:
            "Particle Cosmology best matches detected signals: particle."
        },
        {
          source: "creative_technique",
          signal: "particle_systems compatibility strong.",
          interpretation: "Technique shows how strategy becomes behavior."
        },
        {
          source: "runtime_capability",
          signal: "p5_js, canvas, svg",
          interpretation:
            "Runtime evidence informs feasibility without selecting runtime."
        }
      ],
      strongest_supporting_signals: [
        "Strategy particle_cosmology confidence 0.75."
      ],
      rejected_alternatives: [
        {
          alternative: "Unbounded feature expansion",
          reason:
            "Rejected because inspected capability and the primary trade-off favor a bounded execution path.",
          evidence: ["High performance pressure requires bounded effect scope."]
        }
      ],
      unresolved_decisions: [
        "No blocking creative decision remains unresolved in current metadata."
      ],
      implementation_guidance: [
        "Make particle_systems visibly serve the selected strategy."
      ],
      prompt_guidance: [
        "Use the Creative Reasoning Engine recommendation as the creative spine."
      ],
      hitl_questions: [],
      future_knowledge_context: {
        status: "not_attached"
      },
      authority_boundary:
        "The Creative Reasoning Engine synthesizes inspectable guidance only."
    };
    const event: AssistantStreamEvent = {
      event_type: "generation_input",
      sequence: 2,
      payload: {
        code: "generation_input_prepared",
        emitted_at: "2026-05-22T10:20:30Z",
        workflow: {
          step: "generation",
          phase: "running",
          status: "running",
          current_step: "generation",
          completed_steps: ["intake", "routing"],
          skipped_steps: ["memory"],
          refinement_count: 0,
          review_outcome: null,
          review_reasons: [],
          artifact_count: 1,
          preview_artifact_count: 0,
          image_reference_count: 1,
          intent_decomposer_available: true,
          creative_intent: creativeIntent,
          hierarchy_planner_available: true,
          creative_hierarchy: creativeHierarchy,
          strategy_available: true,
          creative_strategy: creativeStrategy,
          technique_selector_available: true,
          creative_techniques: creativeTechniques,
          planning_available: true,
          creative_plan: creativePlan,
          constraint_solver_available: true,
          creative_constraints: creativeConstraints,
          constraint_prioritizer_available: true,
          creative_constraint_priorities: creativeConstraintPriorities,
          runtime_capability_reasoner_available: true,
          runtime_capabilities: runtimeCapabilities,
          tradeoff_explorer_available: true,
          creative_tradeoffs: creativeTradeoffs,
          director_available: true,
          creative_director: creativeDirector,
          creative_reasoning_available: true,
          creative_reasoning: creativeReasoning,
          image_references: [
            {
              id: "image-reference-1",
              name: "palette.png",
              mime_type: "image/png",
              size_bytes: 128
            }
          ]
        }
      }
    };

    expect(readEventTimestamp(event)).toBe("2026-05-22T10:20:30Z");
    expect(readWorkflowMetadata(event)).toEqual({
      step: "generation",
      phase: "running",
      status: "running",
      current_step: "generation",
      completed_steps: ["intake", "routing"],
      skipped_steps: ["memory"],
      refinement_count: 0,
      review_outcome: null,
      review_reasons: [],
      artifact_count: 1,
      artifact_critique_count: 0,
      recommended_artifact_id: null,
      preview_artifact_count: 0,
      creative_intent: expectedCreativeIntent,
      intent_decomposer_available: true,
      creative_hierarchy: expectedCreativeHierarchy,
      hierarchy_planner_available: true,
      creative_strategy: {
        role: "creative_strategy_engine",
        primaryStrategy: "particle_cosmology",
        confidence: 0.75,
        rationale: "Particle Cosmology best matches detected signals: particle.",
        creativeGoals: [
          "Evoke a coherent world through small moving elements."
        ],
        symbolicAlignment: ["Particle Cosmology", "constellation"],
        alternativeStrategies: [
          {
            strategy: "field_dynamics",
            confidence: 0.51,
            rationale: "Also relevant because of drift."
          }
        ],
        strategyDirectives: [
          "Frame the concept around collective motion and spatial density."
        ],
        implementationBoundary:
          "The Creative Strategy Engine selects high-level artistic strategy only.",
        evidence: ["Primary signals: particle."]
      },
      strategy_available: true,
      creative_techniques: {
        role: "creative_technique_selector",
        primaryTechnique: "particle_systems",
        confidence: 0.79,
        rationale: "Particle Systems best matches detected signals: particle.",
        strategyAlignment: "particle_cosmology",
        compatibility: "strong",
        complexityPressure: "medium",
        performancePressure: "high",
        artisticSuitability: ["Supports strategy: particle_cosmology."],
        implementationNotes: ["Keep counts and lifetimes bounded."],
        alternativeTechniques: [
          {
            technique: "noise_fields",
            confidence: 0.51,
            rationale: "Also relevant because of drift."
          }
        ],
        techniqueConstraints: [
          "Do not treat technique selection as runtime or renderer selection."
        ],
        selectionBoundary:
          "The Creative Technique Selector recommends creative implementation techniques only.",
        evidence: ["Primary technique signals: particle."]
      },
      technique_selector_available: true,
      creative_plan: {
        outputModality: "visual",
        generationStrategy: "Generate one p5 candidate.",
        recommendedRuntime: "p5",
        recommendedRendererId: "surface.p5",
        recommendedPreviewTarget: "browser_sandbox",
        recommendedShaderStyle: "glow",
        candidateCount: 1,
        refinementBudget: 1,
        expectedComplexity: "medium",
        estimatedTokenCost: 2800,
        exportReadiness: "ready",
        runtimeAvailable: true,
        runtimeSupportSummary: "p5.js browser preview is available.",
        planSteps: ["Target p5 output."],
        constraints: ["Keep code browser-safe."],
        evidence: ["Route selected: generate."]
      },
      planning_available: true,
      creative_constraints: {
        role: "creative_constraint_solver",
        intentSummary: "Generate a luminous field.",
        outputGoal: "Generate one p5 candidate.",
        modality: "visual",
        runtimeFit: "supported",
        recommendedRuntime: "p5",
        complexityPressure: "medium",
        safetyPressure: "low",
        performancePressure: "medium",
        costPressure: "low",
        hitlAdvisable: false,
        hitlReason: null,
        activeConstraints: [
          {
            axis: "runtime",
            severity: "info",
            summary: "p5.js browser preview is available.",
            recommendation: "Use p5 through surface.p5.",
            evidence: ["recommended_runtime: p5"]
          }
        ],
        tradeoffs: [
          {
            sourceAxis: "complexity",
            targetAxis: "performance",
            severity: "watch",
            summary: "Keep effects bounded.",
            recommendation: "Reduce particle count before adding candidates."
          }
        ],
        conflicts: [],
        promptGuidance: ["Target p5 output."],
        authorityBoundary: "The solver structures trade-offs for inspection.",
        evidence: ["Route selected: generate."]
      },
      constraint_solver_available: true,
      creative_constraint_priorities: expectedCreativeConstraintPriorities,
      constraint_prioritizer_available: true,
      runtime_capabilities: {
        role: "runtime_capability_reasoner",
        outputGoal: "Generate one p5 candidate.",
        likelyCandidates: ["p5_js", "canvas", "svg"],
        candidateRuntimes: [
          {
            runtime: "p5_js",
            label: "p5.js",
            suitability: "strong",
            confidence: 0.86,
            strategyAlignment: "strong",
            techniqueCompatibility: "strong",
            outputGoalFit: "strong",
            implementationComplexity: "medium",
            performancePressure: "high",
            previewSupport: "backend_preview_supported",
            strengths: ["Fits the selected creative technique."],
            limitations: ["Less natural for deep 3D scenes."],
            risks: ["High performance pressure requires bounded effect scope."],
            promptGuidance: ["Use p5.js capability for sketch output."],
            evidence: ["Capability score: 12."]
          }
        ],
        strategyContext: "particle_cosmology with confidence 0.75.",
        techniqueContext: "particle_systems with high performance pressure.",
        constraintContext:
          "Runtime fit supported; complexity medium; performance medium; HITL false.",
        hitlAdvisable: false,
        hitlReason: null,
        promptGuidance: [
          "Use runtime capability metadata to explain trade-offs."
        ],
        authorityBoundary:
          "The Runtime Capability Reasoner evaluates runtime fit for inspection only.",
        evidence: ["Top runtime scores: p5_js=12."]
      },
      runtime_capability_reasoner_available: true,
      creative_tradeoffs: {
        role: "creative_tradeoff_explorer",
        outputGoal: "Generate one p5 candidate.",
        primaryTradeoffs: [
          {
            sourceAxis: "creative_expressiveness",
            targetAxis: "implementation_complexity",
            severity: "risk",
            summary:
              "Expressive strategy and technique choices can increase implementation scope.",
            creativeBenefit:
              "Particle cosmology with particle systems preserves a distinct direction.",
            technicalCost:
              "Requires managing plan complexity medium and technique complexity medium.",
            runtimeImplication:
              "p5.js has strong suitability and backend preview support.",
            mitigation:
              "Keep the selected strategy visible while bounding the number of systems.",
            directorDiscussionPoint:
              "Should the output prioritize richness or a simpler implementation?",
            hitlRecommended: false,
            evidence: ["Strategy: particle_cosmology."]
          }
        ],
        creativeBenefits: ["Fits the selected creative technique."],
        technicalCosts: ["Expected complexity: medium."],
        runtimeRisks: [
          "High performance pressure requires bounded effect scope."
        ],
        performanceConcerns: ["Technique performance pressure: high."],
        complexityRisks: ["Plan complexity is medium."],
        fidelityRisks: [],
        costSensitivity: "low",
        safetyConcerns: [],
        maintainabilityConcerns: [
          "Keep particle_systems behavior readable."
        ],
        hitlAdvisable: false,
        hitlReason: null,
        directorDiscussionPoints: [
          "Should the output prioritize richness or a simpler implementation?"
        ],
        promptGuidance: [
          "Use trade-off metadata to explain consequences, not to select an outcome."
        ],
        authorityBoundary:
          "The Creative Trade-off Explorer structures consequences and discussion points only.",
        evidence: ["Runtime candidates: p5_js, canvas, svg."]
      },
      tradeoff_explorer_available: true,
      creative_director: {
        role: "creative_assistant_director",
        creativeBrief: "Generate a luminous field.",
        ambiguityLevel: "low",
        ambiguitySignals: [],
        retrievalPosture: "available",
        modalityDirection: "visual",
        runtimeDirection: "p5",
        planningFocus: ["Generate one p5 candidate."],
        critiqueFocus: ["Check output against runtime support."],
        refinementFocus: ["Use bounded refinement only for concrete gaps."],
        nextActions: ["Render the prompt and continue."],
        hitlRequired: false,
        hitlReason: null,
        authorityBoundary: "The user remains the Creative Director.",
        evidence: ["Route selected: generate."]
      },
      director_available: true,
      creative_reasoning: {
        role: "creative_reasoning_engine",
        recommendedCreativeDirection:
          "Recommend a particle_cosmology direction expressed through particle_systems because it best preserves the luminous field.",
        reasoningPath: [
          {
            stage: "strategy",
            claim: "Use particle_cosmology as the conceptual spine.",
            because:
              "Particle Cosmology best matches detected signals: particle.",
            implications: ["Keep one visible creative idea in focus."]
          },
          {
            stage: "technique",
            claim: "Translate that strategy through particle_systems.",
            because:
              "Particle Systems best matches detected signals: particle.",
            implications: ["Technique choices should serve the strategy."]
          },
          {
            stage: "runtime",
            claim: "Shape implementation around inspected capability: p5.js.",
            because: "p5.js shows strong suitability.",
            implications: ["Runtime guidance remains non-binding."]
          },
          {
            stage: "tradeoff",
            claim:
              "Manage the main consequence: creative_expressiveness vs implementation_complexity.",
            because:
              "The creative benefit is distinct direction while the technical cost is complexity.",
            implications: ["Prefer bounded implementation over feature growth."]
          },
          {
            stage: "recommendation",
            claim:
              "Recommend a particle_cosmology direction expressed through particle_systems because it best preserves the luminous field.",
            because:
              "Strategy, technique, runtime capability, and trade-off signals converge.",
            implications: ["Use this as the prompt spine before generation."]
          }
        ],
        evidenceChain: [
          {
            source: "creative_strategy",
            signal: "particle_cosmology confidence 0.75.",
            interpretation:
              "Particle Cosmology best matches detected signals: particle."
          },
          {
            source: "creative_technique",
            signal: "particle_systems compatibility strong.",
            interpretation: "Technique shows how strategy becomes behavior."
          },
          {
            source: "runtime_capability",
            signal: "p5_js, canvas, svg",
            interpretation:
              "Runtime evidence informs feasibility without selecting runtime."
          }
        ],
        strongestSupportingSignals: [
          "Strategy particle_cosmology confidence 0.75."
        ],
        rejectedAlternatives: [
          {
            alternative: "Unbounded feature expansion",
            reason:
              "Rejected because inspected capability and the primary trade-off favor a bounded execution path.",
            evidence: ["High performance pressure requires bounded effect scope."]
          }
        ],
        unresolvedDecisions: [
          "No blocking creative decision remains unresolved in current metadata."
        ],
        implementationGuidance: [
          "Make particle_systems visibly serve the selected strategy."
        ],
        promptGuidance: [
          "Use the Creative Reasoning Engine recommendation as the creative spine."
        ],
        hitlQuestions: [],
        futureKnowledgeContext: {
          status: "not_attached"
        },
        authorityBoundary:
          "The Creative Reasoning Engine synthesizes inspectable guidance only."
      },
      creative_reasoning_available: true,
      image_reference_count: 1,
      image_references: [
        {
          id: "image-reference-1",
          name: "palette.png",
          mime_type: "image/png",
          size_bytes: 128
        }
      ]
    });
    expect(workflowNodeFromAssistantStreamEvent(event)).toBe("generation");
  });

  it("reads creative strategy metadata", () => {
    const strategy = readCreativeStrategySummary({
      role: "creative_strategy_engine",
      primaryStrategy: "sacred_geometry",
      confidence: 0.83,
      rationale: "Sacred Geometry best matches detected signals: mandala.",
      creativeGoals: ["Align form, rhythm, and symmetry."],
      symbolicAlignment: ["Sacred Geometry", "mandala"],
      alternativeStrategies: [],
      strategyDirectives: ["Preserve symbolic structure."],
      implementationBoundary:
        "The Creative Strategy Engine selects high-level artistic strategy only.",
      evidence: ["Primary signals: mandala."]
    });

    expect(strategy?.role).toBe("creative_strategy_engine");
    expect(strategy?.primaryStrategy).toBe("sacred_geometry");
    expect(strategy?.strategyDirectives).toEqual([
      "Preserve symbolic structure."
    ]);
  });

  it("reads creative technique selector metadata", () => {
    const techniques = readCreativeTechniqueSummary({
      role: "creative_technique_selector",
      primaryTechnique: "recursive_geometry",
      confidence: 0.83,
      rationale: "Recursive Geometry best matches detected signals: mandala.",
      strategyAlignment: "sacred_geometry",
      compatibility: "strong",
      complexityPressure: "medium",
      performancePressure: "medium",
      artisticSuitability: ["Supports strategy: sacred_geometry."],
      implementationNotes: ["Preserve the symbolic hierarchy of shapes."],
      alternativeTechniques: [],
      techniqueConstraints: [
        "Do not treat technique selection as runtime or renderer selection."
      ],
      selectionBoundary:
        "The Creative Technique Selector recommends creative implementation techniques only.",
      evidence: ["Primary technique signals: mandala."]
    });

    expect(techniques?.role).toBe("creative_technique_selector");
    expect(techniques?.primaryTechnique).toBe("recursive_geometry");
    expect(techniques?.compatibility).toBe("strong");
  });

  it("reads runtime capability reasoner metadata", () => {
    const profile = readRuntimeCapabilityReasonerSummary({
      role: "runtime_capability_reasoner",
      outputGoal: "Generate one p5 candidate.",
      likelyCandidates: ["p5_js", "canvas"],
      candidateRuntimes: [
        {
          runtime: "p5_js",
          label: "p5.js",
          suitability: "strong",
          confidence: 0.86,
          strategyAlignment: "strong",
          techniqueCompatibility: "strong",
          outputGoalFit: "strong",
          implementationComplexity: "medium",
          performancePressure: "high",
          previewSupport: "backend_preview_supported",
          strengths: ["Fits the selected creative technique."],
          limitations: ["Less natural for deep 3D scenes."],
          risks: ["High performance pressure requires bounded effect scope."],
          promptGuidance: ["Use p5.js capability for sketch output."],
          evidence: ["Capability score: 12."]
        }
      ],
      strategyContext: "particle_cosmology with confidence 0.75.",
      techniqueContext: "particle_systems with high performance pressure.",
      constraintContext:
        "Runtime fit supported; complexity medium; performance medium; HITL false.",
      hitlAdvisable: false,
      promptGuidance: [
        "Use runtime capability metadata to explain trade-offs."
      ],
      authorityBoundary:
        "The Runtime Capability Reasoner evaluates runtime fit for inspection only.",
      evidence: ["Top runtime scores: p5_js=12."]
    });

    expect(profile?.role).toBe("runtime_capability_reasoner");
    expect(profile?.likelyCandidates).toEqual(["p5_js", "canvas"]);
    expect(profile?.candidateRuntimes[0]?.previewSupport).toBe(
      "backend_preview_supported"
    );
  });

  it("reads runtime compatibility metadata", () => {
    const profile = readRuntimeCompatibilityProfileSummary(
      runtimeCompatibilityFixture()
    );

    expect(profile?.role).toBe("runtime_compatibility_engine");
    expect(profile?.compatibleRuntimes).toEqual(["p5_js", "canvas"]);
    expect(profile?.preferredRuntimes).toEqual(["p5_js"]);
    expect(profile?.runtimeConfidence[0]).toMatchObject({
      runtime: "p5_js",
      confidence: 0.93
    });
    expect(profile?.compatibilityAssessments[0]).toMatchObject({
      runtime: "p5_js",
      compatibility: "compatible",
      expectedImplementationComplexity: "medium",
      portability: "high",
      interoperability: "high"
    });
  });

  it("reads artifact capability matrix metadata", () => {
    const matrix = readArtifactCapabilityMatrixSummary(
      artifactCapabilityMatrixFixture()
    );

    expect(matrix?.role).toBe("artifact_capability_matrix");
    expect(matrix?.strongestTargets).toEqual(["p5_js", "canvas"]);
    expect(matrix?.weakestTargets).toEqual(["glsl"]);
    expect(matrix?.capabilityConfidence[0]).toMatchObject({
      target: "p5_js",
      confidence: 0.91
    });
    expect(matrix?.capabilityProfiles[0]).toMatchObject({
      target: "p5_js",
      artifactFit: "strong",
      creativeFit: "strong",
      generativeFit: "strong",
      interoperabilityFit: "strong",
      portabilityFit: "strong"
    });
  });

  it("reads multi-artifact strategy metadata", () => {
    const strategy = readMultiArtifactStrategySummary(
      multiArtifactStrategyFixture()
    );

    expect(strategy?.role).toBe("multi_artifact_strategy");
    expect(strategy?.primaryArtifact).toMatchObject({
      artifactId: "primary_artifact",
      role: "primary",
      artifactType: "runnable_code",
      artifactFamily: "p5_sketch",
      priority: "critical"
    });
    expect(strategy?.supportingArtifacts[0]).toMatchObject({
      artifactId: "runtime_notes",
      role: "supporting",
      runtimeTargets: ["p5_js"]
    });
    expect(strategy?.artifactSequence[0]).toMatchObject({
      stepId: "step_1_primary_artifact",
      action: "produce",
      promptGuidance: ["Produce the primary artifact before support notes."]
    });
    expect(strategy?.artifactPriority[1]).toMatchObject({
      artifactId: "runtime_notes",
      priority: "medium"
    });
    expect(strategy?.combinationMode).toBe("primary_with_supporting_sections");
    expect(strategy?.artifactHandoffPoints).toEqual([
      "primary_artifact -> runtime_notes: Runtime notes support the primary artifact."
    ]);
  });

  it("reads artifact critic metadata", () => {
    const critic = readArtifactCriticSummary(artifactCriticFixture());

    expect(critic?.role).toBe("artifact_critic");
    expect(critic?.critiqueConfidence).toBe(0.82);
    expect(critic?.riskAssessment).toBe("medium");
    expect(critic?.strengths).toContain(
      "Critique remains metadata-only and non-executing."
    );
    expect(critic?.capabilityGaps[0]).toContain("Native shader pipelines");
    expect(critic?.dependencyConcerns[0]).toContain("conflicts");
    expect(critic?.runtimeConcerns[0]).toContain("Unsupported runtimes");
    expect(critic?.promptGuidance[0]).toContain("metadata-only critique");
  });

  it("reads creative critic metadata", () => {
    const critic = readCreativeCriticSummary(creativeCriticFixture());

    expect(critic?.role).toBe("creative_critic_engine");
    expect(critic?.criticConfidence).toBe(0.86);
    expect(critic?.riskAssessment).toBe("medium");
    expect(critic?.conceptQuality).toBe(0.82);
    expect(critic?.runtimeFitQuality).toBe(0.63);
    expect(critic?.creativeStrengths[0]).toContain("Strategy");
    expect(critic?.creativeWeaknesses[0]).toContain("Runtime fit");
    expect(critic?.promptGuidance[0]).toContain("metadata-only critique");
  });

  it("reads artifact refiner metadata", () => {
    const refiner = readArtifactRefinerSummary(artifactRefinerFixture());

    expect(refiner?.role).toBe("artifact_refiner");
    expect(refiner?.refinementConfidence).toBe(0.79);
    expect(refiner?.recommendedImprovements[0]).toContain("critic capability");
    expect(refiner?.priorityImprovements[0]).toContain("dependency risks");
    expect(refiner?.capabilityImprovements[0]).toContain("capability limitation");
    expect(refiner?.dependencyImprovements[0]).toContain("conflicting");
    expect(refiner?.runtimeImprovements[0]).toContain("unsupported runtimes");
    expect(refiner?.refinementCandidates[0]).toContain("Primary candidate");
    expect(refiner?.promptGuidance[0]).toContain("metadata-only refinement");
  });

  it("reads artifact intelligence synthesis metadata", () => {
    const synthesis = readArtifactIntelligenceSynthesisSummary(
      artifactIntelligenceSynthesisFixture()
    );

    expect(synthesis?.role).toBe("artifact_intelligence_synthesis");
    expect(synthesis?.synthesisConfidence).toBe(0.83);
    expect(synthesis?.implementationReadiness).toBe("needs_caveats");
    expect(synthesis?.implementationRisk).toBe("medium");
    expect(synthesis?.recommendedArtifactPath).toContain("primary_artifact");
    expect(synthesis?.recommendedRuntimeDirection).toContain("advisory only");
    expect(synthesis?.majorStrengths[0]).toContain("runnable_code");
    expect(synthesis?.promptGuidance[0]).toContain("metadata-only");
  });

  it("reads artifact merge planner metadata", () => {
    const mergePlanner = readArtifactMergePlannerSummary(
      artifactMergePlannerFixture()
    );

    expect(mergePlanner?.role).toBe("artifact_merge_planner");
    expect(mergePlanner?.mergeConfidence).toBe(0.81);
    expect(mergePlanner?.mergeStrategy).toBe("primary_with_supporting_sections");
    expect(mergePlanner?.artifactBoundaries[0]).toContain("primary_artifact");
    expect(mergePlanner?.artifactJoinPoints[0]).toContain("runtime_notes");
    expect(mergePlanner?.runtimeMergeRisks[0]).toContain("glsl");
    expect(mergePlanner?.rejectedMergePaths[0]).toContain("metadata-only");
    expect(mergePlanner?.promptGuidance[0]).toContain("metadata-only");
  });

  it("reads artifact export intelligence metadata", () => {
    const exportIntelligence = readArtifactExportIntelligenceSummary(
      artifactExportIntelligenceFixture()
    );

    expect(exportIntelligence?.role).toBe("artifact_export_intelligence");
    expect(exportIntelligence?.exportConfidence).toBe(0.78);
    expect(exportIntelligence?.exportReadiness).toBe("ready_with_caveats");
    expect(exportIntelligence?.preferredExportTarget).toBe(
      "multi_artifact_package"
    );
    expect(exportIntelligence?.exportTargets).toContain("single_source_artifact");
    expect(exportIntelligence?.runtimeExportNotes[0]).toContain("p5_js");
    expect(exportIntelligence?.rejectedExportPaths[0]).toContain("metadata-only");
    expect(exportIntelligence?.promptGuidance[0]).toContain("metadata-only");
  });

  it("reads artifact engine contract registry metadata", () => {
    const registry = readArtifactEngineContractRegistrySummary(
      artifactEngineContractRegistryFixture()
    );

    expect(registry?.role).toBe(
      "artifact_intelligence_engine_contract_registry"
    );
    expect(registry?.serializationVersion).toBe(
      "artifact_engine_contract_registry.v1"
    );
    expect(registry?.contractCount).toBe(10);
    expect(registry?.engineIds).toContain("artifact_export_intelligence");
    expect(registry?.engineContracts).toHaveLength(10);
    expect(registry?.engineContracts[0]).toMatchObject({
      engineId: "artifact_planner",
      engineVersion: "v3.3",
      engineCategory: "artifact_intelligence",
      cacheability: "deterministic_with_upstream_metadata",
      parallelizationSupport: "requires_ordered_upstream_metadata",
      serializationVersion: "artifact_engine_contract.v1",
      estimatedCostMetadata: {
        relativeCost: "low",
        externalProviderCalls: false
      },
      estimatedLatencyMetadata: {
        relativeLatency: "low",
        blockingInputs: ["assistant_request"]
      }
    });
  });

  it("reads creative trade-off explorer metadata", () => {
    const profile = readCreativeTradeoffExplorerSummary({
      role: "creative_tradeoff_explorer",
      outputGoal: "Generate one p5 candidate.",
      primaryTradeoffs: [
        {
          sourceAxis: "runtime_support",
          targetAxis: "concept_fidelity",
          severity: "watch",
          summary: "Supported runtime may preserve only part of the concept.",
          creativeBenefit: "Keeps the result aligned with the concept.",
          technicalCost: "Requires explaining runtime limits.",
          runtimeImplication:
            "p5.js has strong suitability and backend preview support.",
          mitigation: "Do not switch runtimes automatically.",
          directorDiscussionPoint:
            "Should fidelity or the lowest-risk runtime path lead?",
          hitlRecommended: false,
          evidence: ["Runtime candidates: p5_js."]
        }
      ],
      creativeBenefits: ["Fits the selected creative technique."],
      technicalCosts: ["Expected complexity: medium."],
      runtimeRisks: ["High performance pressure requires bounded effect scope."],
      performanceConcerns: ["Technique performance pressure: high."],
      complexityRisks: ["Plan complexity is medium."],
      fidelityRisks: [],
      costSensitivity: "low",
      safetyConcerns: [],
      maintainabilityConcerns: ["Keep particle_systems behavior readable."],
      hitlAdvisable: false,
      directorDiscussionPoints: [
        "Should fidelity or the lowest-risk runtime path lead?"
      ],
      promptGuidance: [
        "Use trade-off metadata to explain consequences, not to select an outcome."
      ],
      authorityBoundary:
        "The Creative Trade-off Explorer structures consequences and discussion points only.",
      evidence: ["Runtime candidates: p5_js."]
    });

    expect(profile?.role).toBe("creative_tradeoff_explorer");
    expect(profile?.primaryTradeoffs[0]?.sourceAxis).toBe("runtime_support");
    expect(profile?.costSensitivity).toBe("low");
  });

  it("reads creative quality predictor metadata", () => {
    const profile = readCreativeQualityPredictionSummary({
      role: "creative_quality_predictor",
      predictedQualityLevel: "promising",
      confidence: 0.82,
      readinessScore: 78,
      strongestQualitySignals: [
        {
          dimension: "geometric_formal_clarity",
          score: 9,
          summary: "Formal structure is specific.",
          evidence: ["mandala"]
        }
      ],
      weakestQualitySignals: [
        {
          dimension: "performance_risk",
          score: 5,
          summary: "Performance pressure should stay bounded.",
          evidence: ["particle density"]
        }
      ],
      qualityRisks: ["Performance may force simplification."],
      missingInformation: [],
      likelyFailureModes: [
        "No high-likelihood failure mode is visible before generation."
      ],
      suggestedImprovements: ["Cap density and animation cost."],
      hitlQuestions: [],
      promptGuidance: [
        "Treat this as pre-generation readiness guidance."
      ],
      authorityBoundary:
        "The Creative Quality Predictor estimates pre-generation readiness only.",
      evidence: ["Readiness score: 78/100."]
    });

    expect(profile?.role).toBe("creative_quality_predictor");
    expect(profile?.predictedQualityLevel).toBe("promising");
    expect(profile?.readinessScore).toBe(78);
    expect(profile?.strongestQualitySignals[0]?.dimension).toBe(
      "geometric_formal_clarity"
    );
  });

  it("reads symbolic narrative planner metadata", () => {
    const profile = readSymbolicNarrativePlanSummary(symbolicNarrativeFixture());

    expect(profile?.role).toBe("symbolic_narrative_planner");
    expect(profile?.narrativeArchetype).toBe("threshold_crossing");
    expect(profile?.openingPhase.title).toBe("Invitation");
    expect(profile?.thresholdPhase.symbolicFunction).toBe(
      "Mark the liminal boundary."
    );
    expect(profile?.symbolicTransitions).toContain(
      "Invitation -> Pressure: Complicate the motif."
    );
    expect(profile?.hitlQuestions).toContain(
      "What narrative state should interaction change?"
    );
  });

  it("reads creative composition planner metadata", () => {
    const profile = readCreativeCompositionPlanSummary(
      creativeCompositionFixture()
    );

    expect(profile?.role).toBe("creative_composition_planner");
    expect(profile?.compositionPattern).toBe("threshold_composition");
    expect(profile?.primaryFocalPoint).toBe(
      "A gate-like focal boundary at center stage."
    );
    expect(profile?.visualHierarchy).toContain(
      "Lead with threshold composition as the layout spine."
    );
    expect(profile?.audiovisualCompositionNotes).toContain(
      "Use audio as a compositional timing cue, not as a new feature layer."
    );
    expect(profile?.hitlQuestions).toContain(
      "What should be the primary visible focal motif?"
    );
  });

  it("reads procedural structure planner metadata", () => {
    const profile = readProceduralStructurePlanSummary(
      proceduralStructureFixture()
    );

    expect(profile?.role).toBe("procedural_structure_planner");
    expect(profile?.recommendedFamilies).toContain("recursive_geometry");
    expect(profile?.primaryStructure.family).toBe("recursive_geometry");
    expect(profile?.secondaryStructures[0]?.family).toBe(
      "polar_radial_systems"
    );
    expect(profile?.fallbackStructureOptions[0]?.family).toBe(
      "polar_radial_systems"
    );
    expect(profile?.runtimeSuitabilityNotes[0]).toContain("p5_js");
    expect(profile?.hitlQuestions).toContain(
      "What user gesture should control the structure?"
    );
  });

  it("reads generative structure engine metadata", () => {
    const profile = readGenerativeStructureBlueprintSummary(
      generativeStructureFixture()
    );

    expect(profile?.role).toBe("generative_structure_engine");
    expect(profile?.blueprintName).toBe(
      "Recursive Geometry Blueprint for spiral threshold"
    );
    expect(profile?.generativeArchitecture).toBe(
      "recursive_modular_blueprint"
    );
    expect(profile?.proceduralModules.map((module) => module.kind)).toContain(
      "particle_emitter"
    );
    expect(profile?.moduleRelationships[1]?.relationshipType).toBe("attracts");
    expect(profile?.parameterSchema.map((parameter) => parameter.name)).toContain(
      "max_particle_count"
    );
    expect(profile?.evolutionRules.map((rule) => rule.phase)).toContain(
      "stabilization"
    );
    expect(profile?.interactionHooks[0]?.hookType).toBe("interaction");
    expect(profile?.audiovisualHooks[0]?.hookType).toBe("audiovisual");
    expect(profile?.fallbackBlueprint.moduleKinds).toContain(
      "symmetry_transform"
    );
  });

  it("reads semantic motif engine metadata", () => {
    const profile = readSemanticMotifSystemSummary(semanticMotifFixture());

    expect(profile?.role).toBe("semantic_motif_engine");
    expect(profile?.motifSystemName).toBe(
      "Fragmentation / Reintegration Motif System"
    );
    expect(profile?.primaryMotifs.map((motif) => motif.motifId)).toEqual([
      "fragmentation",
      "reintegration"
    ]);
    expect(profile?.primaryMotifs[0]?.role).toBe("transformation");
    expect(profile?.secondaryMotifs.map((motif) => motif.motifId)).toContain(
      "spiral"
    );
    expect(profile?.motifToStructureMapping[0]?.generativeModuleKinds).toContain(
      "particle_emitter"
    );
    expect(profile?.motifToParameterMapping[1]?.parameterNames).toContain(
      "reassembly_speed"
    );
    expect(profile?.unsupportedSymbolicClaims[0]).toContain("doctrine");
    expect(profile?.motifFallbackPlan.fallbackPrimaryMotif).toBe("fragmentation");
    expect(profile?.hitlQuestions).toContain(
      "Which motif should remain visually dominant?"
    );
  });

  it("reads emotional consistency engine metadata", () => {
    const profile = readEmotionalConsistencyProfileSummary(
      emotionalConsistencyFixture()
    );

    expect(profile?.role).toBe("emotional_consistency_engine");
    expect(profile?.primaryEmotionalTone).toBe("transformation");
    expect(profile?.secondaryEmotionalTones).toContain("rupture");
    expect(profile?.emotionalArc).toContain("threshold stillness");
    expect(profile?.emotionalPhaseMapping[1]).toMatchObject({
      phase: "threshold",
      tone: "suspension",
      intensity: "high"
    });
    expect(profile?.emotionalToMotifMapping[0]).toMatchObject({
      tone: "rupture",
      motifId: "fragmentation"
    });
    expect(profile?.emotionalToStructureMapping[0]?.generativeModuleKinds).toContain(
      "particle_emitter"
    );
    expect(profile?.emotionalToParameterMapping[1]?.parameterNames).toContain(
      "reassembly_speed"
    );
    expect(profile?.emotionalCoherenceScore).toBe(86);
    expect(profile?.fallbackEmotionalStrategy.fallbackPrimaryTone).toBe(
      "transformation"
    );
    expect(profile?.hitlQuestions).toContain(
      "Which emotional tone should remain dominant?"
    );
  });

  it("reads cross-modality composer metadata", () => {
    const profile = readCrossModalityCompositionProfileSummary(
      crossModalityFixture()
    );

    expect(profile?.role).toBe("cross_modality_composer");
    expect(profile?.modalityPattern).toBe(
      "fragmentation_reassembly_visual_motion_layers"
    );
    expect(profile?.primaryModality).toBe("visual_structure");
    expect(profile?.supportingModalities).toContain("audio");
    expect(profile?.modalityHierarchy[0]).toMatchObject({
      modality: "visual_structure",
      priority: "primary"
    });
    expect(profile?.visualToAudioMapping[0]).toMatchObject({
      sourceModality: "visual_structure",
      targetModality: "audio"
    });
    expect(profile?.audioToMotionMapping[0]?.cues).toContain("pulse");
    expect(profile?.motifToModalityMapping[0]).toMatchObject({
      motifId: "fragmentation",
      targetModality: "motion"
    });
    expect(profile?.emotionalToModalityMapping[0]).toMatchObject({
      emotionalTone: "transformation",
      targetModality: "visual_structure"
    });
    expect(profile?.temporalCuePlan[0]).toMatchObject({
      phase: "threshold",
      modalities: ["visual_structure", "motion", "audio"]
    });
    expect(profile?.fallbackMultimodalStrategy.reducedModalities).toContain(
      "audio"
    );
    expect(profile?.hitlQuestions[0]).toContain("Which modality should lead");
  });

  it("reads audio-visual scene system metadata", () => {
    const profile = readAudioVisualSceneProfileSummary(audioVisualSceneFixture());

    expect(profile?.role).toBe("audio_visual_scene_system");
    expect(profile?.scenePattern).toBe("fragmentation_to_reintegration");
    expect(profile?.scenePhases.map((phase) => phase.phase)).toEqual([
      "opening",
      "development",
      "threshold",
      "climax",
      "resolution"
    ]);
    expect(profile?.openingScene.title).toBe("Whole Form");
    expect(profile?.climaxScene.phase).toBe("climax");
    expect(profile?.cuePlan[1]).toMatchObject({
      cueId: "threshold_sync",
      cueType: "synchronization",
      modalities: ["visual_structure", "motion", "audio", "rhythm"]
    });
    expect(profile?.transitionPlan[0]).toMatchObject({
      fromPhase: "opening",
      toPhase: "development"
    });
    expect(profile?.audioTimingPlan[0]).toContain("silence");
    expect(profile?.cameraTimingPlan[0]).toContain("viewpoint");
    expect(profile?.fallbackSceneStrategy.reducedElements).toContain(
      "audio timing"
    );
    expect(profile?.hitlQuestions[0]).toContain("audio timing");
  });

  it("reads creative reasoning engine metadata", () => {
    const profile = readCreativeReasoningSummary({
      role: "creative_reasoning_engine",
      recommendedCreativeDirection:
        "Recommend sacred_geometry through recursive_geometry because the signals converge.",
      reasoningPath: [
        {
          stage: "strategy",
          claim: "Use sacred_geometry as the conceptual spine.",
          because: "Sacred Geometry best matches mandala signals.",
          implications: ["Keep symbolic structure visible."]
        },
        {
          stage: "technique",
          claim: "Translate that strategy through recursive_geometry.",
          because: "Recursive geometry matches the selected strategy.",
          implications: ["Technique choices should serve the strategy."]
        },
        {
          stage: "runtime",
          claim: "Shape implementation around inspected p5.js capability.",
          because: "p5.js has strong suitability.",
          implications: ["Runtime guidance remains non-binding."]
        },
        {
          stage: "tradeoff",
          claim: "Manage expressiveness versus implementation complexity.",
          because: "Complexity increases with recursive detail.",
          implications: ["Prefer bounded implementation over feature growth."]
        },
        {
          stage: "recommendation",
          claim:
            "Recommend sacred_geometry through recursive_geometry because the signals converge.",
          because:
            "Strategy, technique, runtime, and trade-offs point to the same direction.",
          implications: ["Use this as the prompt spine."]
        }
      ],
      evidenceChain: [
        {
          source: "creative_strategy",
          signal: "sacred_geometry confidence 0.83.",
          interpretation: "Strategy preserves symbolic structure."
        },
        {
          source: "creative_technique",
          signal: "recursive_geometry compatibility strong.",
          interpretation: "Technique makes the strategy concrete."
        },
        {
          source: "tradeoff_explorer",
          signal: "creative_expressiveness vs implementation_complexity.",
          interpretation: "Trade-off evidence bounds scope."
        },
        {
          source: "quality_predictor",
          signal: "promising readiness 78/100.",
          interpretation:
            "Quality prediction estimates pre-generation readiness."
        },
        {
          source: "symbolic_narrative",
          signal: "threshold_crossing arc.",
          interpretation: "Narrative evidence orders the experiential arc."
        },
        {
          source: "creative_composition",
          signal: "threshold_composition focal boundary.",
          interpretation: "Composition evidence defines focal structure."
        },
        {
          source: "procedural_structure",
          signal: "recursive_geometry procedural spine.",
          interpretation: "Procedural evidence defines structure."
        },
        {
          source: "generative_structure",
          signal: "recursive_modular_blueprint.",
          interpretation: "Generative evidence defines blueprint modules."
        },
        {
          source: "semantic_motif",
          signal: "fragmentation, reintegration.",
          interpretation: "Motif evidence defines recurrence and risk guidance."
        },
        {
          source: "emotional_consistency",
          signal: "transformation 86/100.",
          interpretation:
            "Emotional evidence defines tone hierarchy and mismatch guidance."
        },
        {
          source: "cross_modality",
          signal:
            "fragmentation_reassembly_visual_motion_layers: visual_structure -> motion, audio.",
          interpretation:
            "Cross-modality evidence coordinates modalities as design metadata."
        },
        {
          source: "audio_visual_scene",
          signal:
            "fragmentation_to_reintegration: Whole Form -> Reassembly -> Integrated Geometry",
          interpretation:
            "Audio-visual scene evidence orders phases and cues as design metadata."
        }
      ],
      strongestSupportingSignals: ["Strategy sacred_geometry confidence 0.83."],
      rejectedAlternatives: [
        {
          alternative: "Technique: particle_systems",
          reason:
            "Deferred because recursive_geometry more directly carries the selected strategy.",
          evidence: ["Alternative confidence 0.42."]
        }
      ],
      unresolvedDecisions: [
        "No blocking creative decision remains unresolved in current metadata."
      ],
      implementationGuidance: ["Preserve the symbolic hierarchy of shapes."],
      promptGuidance: [
        "Use the Creative Reasoning Engine recommendation as the creative spine."
      ],
      hitlQuestions: [],
      futureKnowledgeContext: { status: "not_attached" },
      authorityBoundary:
        "The Creative Reasoning Engine synthesizes inspectable guidance only."
    });

    expect(profile?.role).toBe("creative_reasoning_engine");
    expect(profile?.reasoningPath.map((step) => step.stage)).toEqual([
      "strategy",
      "technique",
      "runtime",
      "tradeoff",
      "recommendation"
    ]);
    expect(profile?.evidenceChain[0]?.source).toBe("creative_strategy");
    expect(profile?.evidenceChain.map((item) => item.source)).toContain(
      "quality_predictor"
    );
    expect(profile?.evidenceChain.map((item) => item.source)).toContain(
      "symbolic_narrative"
    );
    expect(profile?.evidenceChain.map((item) => item.source)).toContain(
      "creative_composition"
    );
    expect(profile?.evidenceChain.map((item) => item.source)).toContain(
      "procedural_structure"
    );
    expect(profile?.evidenceChain.map((item) => item.source)).toContain(
      "generative_structure"
    );
    expect(profile?.evidenceChain.map((item) => item.source)).toContain(
      "semantic_motif"
    );
    expect(profile?.evidenceChain.map((item) => item.source)).toContain(
      "emotional_consistency"
    );
    expect(profile?.evidenceChain.map((item) => item.source)).toContain(
      "cross_modality"
    );
    expect(profile?.evidenceChain.map((item) => item.source)).toContain(
      "audio_visual_scene"
    );
    expect(profile?.futureKnowledgeContext.status).toBe("not_attached");
  });

  it("reads creative constraint solver metadata", () => {
    const solution = readCreativeConstraintSolverSummary({
      role: "creative_constraint_solver",
      intentSummary: "Generate a luminous field.",
      outputGoal: "Generate one p5 candidate.",
      runtimeFit: "supported",
      recommendedRuntime: "p5",
      complexityPressure: "medium",
      safetyPressure: "low",
      performancePressure: "medium",
      costPressure: "low",
      hitlAdvisable: false,
      activeConstraints: [
        {
          axis: "runtime",
          severity: "info",
          summary: "p5.js browser preview is available.",
          recommendation: "Use p5 through surface.p5.",
          evidence: ["recommendedRuntime: p5"]
        }
      ],
      tradeoffs: [],
      conflicts: [],
      promptGuidance: ["Target p5 output."],
      authorityBoundary: "The solver structures trade-offs for inspection.",
      evidence: ["Route selected: generate."]
    });

    expect(solution?.role).toBe("creative_constraint_solver");
    expect(solution?.runtimeFit).toBe("supported");
    expect(solution?.activeConstraints[0]?.axis).toBe("runtime");
    expect(solution?.promptGuidance).toEqual(["Target p5 output."]);
  });

  it("reads creative execution plans from planning events", () => {
    const plan = readCreativeExecutionPlanSummary({
      outputModality: "visual",
      generationStrategy: "Generate one p5 candidate.",
      recommendedRuntime: "p5",
      recommendedRendererId: "surface.p5",
      recommendedPreviewTarget: "browser_sandbox",
      candidateCount: 1,
      refinementBudget: 1,
      expectedComplexity: "medium",
      estimatedTokenCost: 2800,
      exportReadiness: "ready",
      runtimeAvailable: true,
      runtimeSupportSummary: "p5.js browser preview is available.",
      planSteps: ["Target p5 output."],
      constraints: ["Keep code browser-safe."],
      evidence: ["Route selected: generate."]
    });

    expect(plan).toMatchObject({
      outputModality: "visual",
      recommendedRuntime: "p5",
      candidateCount: 1,
      exportReadiness: "ready"
    });
  });

  it("hydrates creative quality prediction workflow metadata", () => {
    const creativeQualityPrediction = {
      role: "creative_quality_predictor",
      predicted_quality_level: "ambiguous",
      confidence: 0.61,
      readiness_score: 57,
      strongest_quality_signals: [
        {
          dimension: "runtime_suitability",
          score: 8,
          summary: "Runtime support is suitable.",
          evidence: ["p5_js strong fit"]
        }
      ],
      weakest_quality_signals: [
        {
          dimension: "intent_clarity",
          score: 4,
          summary: "Intent needs sharper subject detail.",
          evidence: ["ambiguous wording"]
        }
      ],
      quality_risks: ["Intent may remain generic."],
      missing_information: ["Palette direction is missing."],
      likely_failure_modes: ["Generation may invent missing details."],
      suggested_improvements: ["Clarify palette before generation."],
      hitl_questions: ["What palette should lead the piece?"],
      prompt_guidance: [
        "Treat this as pre-generation readiness guidance."
      ],
      authority_boundary:
        "The Creative Quality Predictor estimates pre-generation readiness only.",
      evidence: ["Readiness score: 57/100."]
    };
    const event: AssistantStreamEvent = {
      event_type: "planning",
      sequence: 7,
      payload: {
        workflow: {
          step: "planning",
          phase: "running",
          status: "running",
          current_step: "planning",
          completed_steps: ["intake", "routing"],
          skipped_steps: [],
          refinement_count: 0,
          review_reasons: [],
          artifact_count: 0,
          artifact_critique_count: 0,
          preview_artifact_count: 0,
          image_reference_count: 0,
          image_references: [],
          creative_quality_prediction: creativeQualityPrediction,
          quality_predictor_available: true
        }
      }
    };

    expect(readWorkflowMetadata(event)).toMatchObject({
      step: "planning",
      phase: "running",
      status: "running",
      creative_quality_prediction: {
        role: "creative_quality_predictor",
        predictedQualityLevel: "ambiguous",
        confidence: 0.61,
        readinessScore: 57,
        strongestQualitySignals: [
          {
            dimension: "runtime_suitability",
            score: 8,
            summary: "Runtime support is suitable.",
            evidence: ["p5_js strong fit"]
          }
        ],
        weakestQualitySignals: [
          {
            dimension: "intent_clarity",
            score: 4,
            summary: "Intent needs sharper subject detail.",
            evidence: ["ambiguous wording"]
          }
        ],
        qualityRisks: ["Intent may remain generic."],
        missingInformation: ["Palette direction is missing."],
        likelyFailureModes: ["Generation may invent missing details."],
        suggestedImprovements: ["Clarify palette before generation."],
        hitlQuestions: ["What palette should lead the piece?"],
        promptGuidance: [
          "Treat this as pre-generation readiness guidance."
        ],
        authorityBoundary:
          "The Creative Quality Predictor estimates pre-generation readiness only.",
        evidence: ["Readiness score: 57/100."]
      },
      quality_predictor_available: true
    });
  });

  it("hydrates symbolic narrative workflow metadata", () => {
    const symbolicNarrative = symbolicNarrativeFixture();
    const event: AssistantStreamEvent = {
      event_type: "planning",
      sequence: 8,
      payload: {
        workflow: {
          step: "planning",
          phase: "running",
          status: "running",
          current_step: "planning",
          completed_steps: ["intake", "routing"],
          skipped_steps: [],
          refinement_count: 0,
          review_reasons: [],
          artifact_count: 0,
          artifact_critique_count: 0,
          preview_artifact_count: 0,
          image_reference_count: 0,
          image_references: [],
          symbolic_narrative: symbolicNarrative,
          symbolic_narrative_available: true
        }
      }
    };

    expect(readWorkflowMetadata(event)).toMatchObject({
      step: "planning",
      phase: "running",
      status: "running",
      symbolic_narrative: {
        role: "symbolic_narrative_planner",
        narrativeArchetype: "threshold_crossing",
        openingPhase: {
          phase: "opening",
          title: "Invitation",
          symbolicFunction: "Present the symbolic gate."
        },
        climaxPhase: {
          phase: "climax",
          title: "Revelation",
          visualState: "bright geometry"
        },
        unresolvedNarrativeGaps: ["Interaction state is ambiguous."],
        hitlQuestions: ["What narrative state should interaction change?"]
      },
      symbolic_narrative_available: true
    });
  });

  it("hydrates creative composition workflow metadata", () => {
    const creativeComposition = creativeCompositionFixture();
    const event: AssistantStreamEvent = {
      event_type: "planning",
      sequence: 9,
      payload: {
        workflow: {
          step: "planning",
          phase: "running",
          status: "running",
          current_step: "planning",
          completed_steps: ["intake", "routing"],
          skipped_steps: [],
          refinement_count: 0,
          review_reasons: [],
          artifact_count: 0,
          artifact_critique_count: 0,
          preview_artifact_count: 0,
          image_reference_count: 0,
          image_references: [],
          creative_composition: creativeComposition,
          creative_composition_available: true
        }
      }
    };

    expect(readWorkflowMetadata(event)).toMatchObject({
      step: "planning",
      phase: "running",
      status: "running",
      creative_composition: {
        role: "creative_composition_planner",
        compositionPattern: "threshold_composition",
        primaryFocalPoint: "A gate-like focal boundary at center stage.",
        spatialOrganization:
          "Organize before/after zones separated by a readable threshold.",
        visualHierarchy: [
          "Lead with threshold composition as the layout spine.",
          "Protect symbolism before secondary composition details."
        ],
        densityPlan:
          "Keep density lower at the threshold so the boundary reads clearly.",
        unresolvedCompositionGaps: [
          "Primary visible focal motif is unclear."
        ],
        hitlQuestions: ["What should be the primary visible focal motif?"]
      },
      creative_composition_available: true
    });
  });

  it("hydrates procedural structure workflow metadata", () => {
    const proceduralStructure = proceduralStructureFixture();
    const event: AssistantStreamEvent = {
      event_type: "planning",
      sequence: 10,
      payload: {
        workflow: {
          step: "planning",
          phase: "running",
          status: "running",
          current_step: "planning",
          completed_steps: ["intake", "routing"],
          skipped_steps: [],
          refinement_count: 0,
          review_reasons: [],
          artifact_count: 0,
          artifact_critique_count: 0,
          preview_artifact_count: 0,
          image_reference_count: 0,
          image_references: [],
          procedural_structure: proceduralStructure,
          procedural_structure_available: true
        }
      }
    };

    expect(readWorkflowMetadata(event)).toMatchObject({
      step: "planning",
      phase: "running",
      status: "running",
      procedural_structure: {
        role: "procedural_structure_planner",
        recommendedFamilies: [
          "recursive_geometry",
          "polar_radial_systems",
          "particle_systems"
        ],
        primaryStructure: {
          family: "recursive_geometry",
          label: "Recursive Geometry"
        },
        secondaryStructures: [
          {
            family: "polar_radial_systems"
          },
          {
            family: "particle_systems"
          }
        ],
        complexityLevel: "medium",
        runtimeSuitabilityNotes: [
          "Use inspected runtime candidates as non-binding feasibility notes: p5_js."
        ],
        unresolvedProceduralGaps: [
          "Interaction is relevant but the controlling gesture is unclear."
        ],
        hitlQuestions: ["What user gesture should control the structure?"]
      },
      procedural_structure_available: true
    });
  });

  it("hydrates generative structure workflow metadata", () => {
    const generativeStructure = generativeStructureFixture();
    const event: AssistantStreamEvent = {
      event_type: "planning",
      sequence: 11,
      payload: {
        workflow: {
          step: "planning",
          phase: "running",
          status: "running",
          current_step: "planning",
          completed_steps: ["intake", "routing"],
          skipped_steps: [],
          refinement_count: 0,
          review_reasons: [],
          artifact_count: 0,
          artifact_critique_count: 0,
          preview_artifact_count: 0,
          image_reference_count: 0,
          image_references: [],
          generative_structure: generativeStructure,
          generative_structure_available: true
        }
      }
    };

    expect(readWorkflowMetadata(event)).toMatchObject({
      step: "planning",
      phase: "running",
      status: "running",
      generative_structure: {
        role: "generative_structure_engine",
        blueprintName: "Recursive Geometry Blueprint for spiral threshold",
        generativeArchitecture: "recursive_modular_blueprint",
        proceduralModules: [
          {
            moduleId: "seed_system",
            kind: "seed_system"
          },
          {
            moduleId: "recursive_module_0",
            kind: "recursive_module"
          },
          {
            moduleId: "particle_emitter_1",
            kind: "particle_emitter"
          }
        ],
        moduleRelationships: [
          {
            sourceModuleId: "seed_system",
            targetModuleId: "recursive_module_0",
            relationshipType: "feeds"
          },
          {
            sourceModuleId: "recursive_module_0",
            targetModuleId: "particle_emitter_1",
            relationshipType: "attracts"
          }
        ],
        controlParameters: ["random_seed", "recursion_depth"],
        fallbackBlueprint: {
          architecture: "radial_pattern_blueprint",
          moduleKinds: ["seed_system", "symmetry_transform"]
        },
        hitlQuestions: ["Which gesture should control the generative structure?"]
      },
      generative_structure_available: true
    });
  });

  it("hydrates semantic motif workflow metadata", () => {
    const semanticMotif = semanticMotifFixture();
    const event: AssistantStreamEvent = {
      event_type: "planning",
      sequence: 12,
      payload: {
        workflow: {
          step: "planning",
          phase: "running",
          status: "running",
          current_step: "planning",
          completed_steps: ["intake", "routing"],
          skipped_steps: [],
          refinement_count: 0,
          review_reasons: [],
          artifact_count: 0,
          artifact_critique_count: 0,
          preview_artifact_count: 0,
          image_reference_count: 0,
          image_references: [],
          semantic_motif: semanticMotif,
          semantic_motif_available: true
        }
      }
    };

    expect(readWorkflowMetadata(event)).toMatchObject({
      step: "planning",
      phase: "running",
      status: "running",
      semantic_motif: {
        role: "semantic_motif_engine",
        motifSystemName: "Fragmentation / Reintegration Motif System",
        primaryMotifs: [
          {
            motifId: "fragmentation",
            role: "transformation",
            hierarchyLevel: "primary"
          },
          {
            motifId: "reintegration",
            role: "transformation",
            hierarchyLevel: "primary"
          }
        ],
        motifToStructureMapping: [
          {
            motifId: "fragmentation",
            generativeModuleKinds: ["particle_emitter"]
          },
          {
            motifId: "reintegration",
            generativeModuleKinds: ["geometry_reassembly_layer"]
          }
        ],
        motifToParameterMapping: [
          {
            motifId: "fragmentation",
            parameterNames: ["fragmentation_amount", "particle_count"]
          },
          {
            motifId: "reintegration",
            parameterNames: ["reassembly_speed"]
          }
        ],
        motifFallbackPlan: {
          fallbackPrimaryMotif: "fragmentation",
          fallbackSecondaryMotifs: ["spiral", "flame"]
        },
        hitlQuestions: ["Which motif should remain visually dominant?"]
      },
      semantic_motif_available: true
    });
  });

  it("hydrates emotional consistency workflow metadata", () => {
    const emotionalConsistency = emotionalConsistencyFixture();
    const event: AssistantStreamEvent = {
      event_type: "planning",
      sequence: 13,
      payload: {
        workflow: {
          step: "planning",
          phase: "running",
          status: "running",
          current_step: "planning",
          completed_steps: ["intake", "routing"],
          skipped_steps: [],
          refinement_count: 0,
          review_reasons: [],
          artifact_count: 0,
          artifact_critique_count: 0,
          preview_artifact_count: 0,
          image_reference_count: 0,
          image_references: [],
          emotional_consistency: emotionalConsistency,
          emotional_consistency_available: true
        }
      }
    };

    expect(readWorkflowMetadata(event)).toMatchObject({
      step: "planning",
      phase: "running",
      status: "running",
      emotional_consistency: {
        role: "emotional_consistency_engine",
        primaryEmotionalTone: "transformation",
        secondaryEmotionalTones: [
          "rupture",
          "suspension",
          "release",
          "integration"
        ],
        emotionalPhaseMapping: [
          {
            phase: "opening",
            tone: "tension",
            intensity: "medium"
          },
          {
            phase: "threshold",
            tone: "suspension",
            intensity: "high"
          }
        ],
        emotionalToMotifMapping: [
          {
            tone: "rupture",
            motifId: "fragmentation"
          },
          {
            tone: "integration",
            motifId: "reintegration"
          }
        ],
        emotionalToParameterMapping: [
          {
            tone: "rupture",
            parameterNames: ["fragmentation_amount", "particle_count"]
          },
          {
            tone: "integration",
            parameterNames: ["reassembly_speed"]
          }
        ],
        fallbackEmotionalStrategy: {
          fallbackPrimaryTone: "transformation",
          fallbackSecondaryTones: ["release", "integration"]
        },
        hitlQuestions: ["Which emotional tone should remain dominant?"]
      },
      emotional_consistency_available: true
    });
  });

  it("hydrates cross-modality workflow metadata", () => {
    const crossModality = crossModalityFixture();
    const event: AssistantStreamEvent = {
      event_type: "planning",
      sequence: 14,
      payload: {
        workflow: {
          step: "planning",
          phase: "running",
          status: "running",
          current_step: "planning",
          completed_steps: ["intake", "routing"],
          skipped_steps: [],
          refinement_count: 0,
          review_reasons: [],
          artifact_count: 0,
          artifact_critique_count: 0,
          preview_artifact_count: 0,
          image_reference_count: 0,
          image_references: [],
          cross_modality: crossModality,
          cross_modality_available: true
        }
      }
    };

    expect(readWorkflowMetadata(event)).toMatchObject({
      step: "planning",
      phase: "running",
      status: "running",
      cross_modality: {
        role: "cross_modality_composer",
        modalityPattern: "fragmentation_reassembly_visual_motion_layers",
        primaryModality: "visual_structure",
        supportingModalities: [
          "motion",
          "audio",
          "rhythm",
          "structure",
          "motif",
          "emotion"
        ],
        visualToAudioMapping: [
          {
            sourceModality: "visual_structure",
            targetModality: "audio",
            cues: ["visual density", "brightness", "phase threshold"]
          }
        ],
        audioToMotionMapping: [
          {
            sourceModality: "audio",
            targetModality: "motion",
            cues: ["pulse", "silence"]
          }
        ],
        temporalCuePlan: [
          {
            phase: "threshold",
            modalities: ["visual_structure", "motion", "audio"]
          }
        ],
        fallbackMultimodalStrategy: {
          fallbackPattern: "visual_led_composition",
          reducedModalities: ["audio"]
        },
        hitlQuestions: [
          "Which modality should lead if visual, motion, audio, and emotion compete?"
        ]
      },
      cross_modality_available: true
    });
  });

  it("hydrates audio-visual scene workflow metadata", () => {
    const audioVisualScene = audioVisualSceneFixture();
    const event: AssistantStreamEvent = {
      event_type: "planning",
      sequence: 15,
      payload: {
        workflow: {
          step: "planning",
          phase: "running",
          status: "running",
          current_step: "planning",
          completed_steps: ["intake", "routing"],
          skipped_steps: [],
          refinement_count: 0,
          review_reasons: [],
          artifact_count: 0,
          artifact_critique_count: 0,
          preview_artifact_count: 0,
          image_reference_count: 0,
          image_references: [],
          audio_visual_scene: audioVisualScene,
          audio_visual_scene_available: true
        }
      }
    };

    expect(readWorkflowMetadata(event)).toMatchObject({
      step: "planning",
      phase: "running",
      status: "running",
      audio_visual_scene: {
        role: "audio_visual_scene_system",
        scenePattern: "fragmentation_to_reintegration",
        openingScene: {
          phase: "opening",
          title: "Whole Form"
        },
        climaxScene: {
          phase: "climax",
          title: "Reassembly"
        },
        cuePlan: [
          {
            cueId: "opening_visual",
            cueType: "visual"
          },
          {
            cueId: "threshold_sync",
            cueType: "synchronization"
          }
        ],
        transitionPlan: [
          {
            fromPhase: "opening",
            toPhase: "development"
          }
        ],
        fallbackSceneStrategy: {
          fallbackPattern: "seed_to_expansion",
          reducedElements: ["audio timing", "camera/viewpoint timing"]
        },
        hitlQuestions: [
          "Should audio timing drive scene transitions, or only support visual rhythm?"
        ]
      },
      audio_visual_scene_available: true
    });
  });

  it("reads artifact planner summaries", () => {
    const plan = readArtifactPlanSummary(artifactPlanFixture());

    expect(plan).toMatchObject({
      role: "artifact_planner",
      primaryArtifactIntent:
        "Generate a luminous p5.js mandala that preserves the symbolic scene arc.",
      artifactType: "runnable_code",
      artifactFamily: "p5_sketch",
      requiredComponents: [
        "One clearly labeled primary artifact.",
        "A fenced code block with an explicit language tag.",
        "p5.js setup/draw lifecycle."
      ],
      runtimeRequirements: [
        "Respect existing runtime hint: p5.",
        "Keep renderer compatibility with surface.p5."
      ],
      expectedOutputStructure: [
        "Lead with the primary runnable artifact.",
        "Use a fenced code block with an explicit filename or language tag."
      ]
    });
  });

  it("hydrates artifact planner workflow metadata", () => {
    const artifactPlan = artifactPlanFixture();
    const event: AssistantStreamEvent = {
      event_type: "planning",
      sequence: 16,
      payload: {
        workflow: {
          step: "planning",
          phase: "running",
          status: "running",
          current_step: "planning",
          completed_steps: ["intake", "routing"],
          skipped_steps: [],
          refinement_count: 0,
          review_reasons: [],
          artifact_count: 0,
          artifact_critique_count: 0,
          preview_artifact_count: 0,
          image_reference_count: 0,
          image_references: [],
          artifact_plan: artifactPlan,
          artifact_planner_available: true
        }
      }
    };

    expect(readWorkflowMetadata(event)).toMatchObject({
      step: "planning",
      phase: "running",
      status: "running",
      artifact_plan: {
        role: "artifact_planner",
        artifactType: "runnable_code",
        artifactFamily: "p5_sketch",
        requiredComponents: [
          "One clearly labeled primary artifact.",
          "A fenced code block with an explicit language tag.",
          "p5.js setup/draw lifecycle."
        ]
      },
      artifact_planner_available: true
    });
  });

  it("reads artifact dependency graph summaries", () => {
    const graph = readArtifactDependencyGraphSummary(
      artifactDependencyGraphFixture()
    );

    expect(graph).toMatchObject({
      role: "artifact_dependency_graph",
      primaryArtifactNodeId: "primary_artifact",
      artifactNodes: [
        {
          nodeId: "primary_artifact",
          nodeType: "planned_artifact",
          status: "available"
        },
        {
          nodeId: "runtime_requirements",
          nodeType: "runtime_requirement",
          status: "available"
        }
      ],
      dependencyEdges: [
        {
          sourceNodeId: "runtime_requirements",
          targetNodeId: "primary_artifact",
          relationship: "requires",
          strength: "required"
        }
      ],
      requiredUpstreamMetadata: [
        "assistant_request:available",
        "artifact_plan:available"
      ],
      downstreamConsumers: [
        "prompt_renderer",
        "creative_assistant_director",
        "creative_reasoning_engine",
        "workflow_serialization",
        "final_payload",
        "nextjs_stream_hydration"
      ]
    });
  });

  it("hydrates artifact dependency graph workflow metadata", () => {
    const graph = artifactDependencyGraphFixture();
    const event: AssistantStreamEvent = {
      event_type: "planning",
      sequence: 17,
      payload: {
        workflow: {
          step: "planning",
          phase: "running",
          status: "running",
          current_step: "planning",
          completed_steps: ["intake", "routing"],
          skipped_steps: [],
          refinement_count: 0,
          review_reasons: [],
          artifact_count: 0,
          artifact_critique_count: 0,
          preview_artifact_count: 0,
          image_reference_count: 0,
          image_references: [],
          artifact_dependency_graph: graph,
          artifact_dependency_graph_available: true
        }
      }
    };

    expect(readWorkflowMetadata(event)).toMatchObject({
      step: "planning",
      phase: "running",
      status: "running",
      artifact_dependency_graph: {
        role: "artifact_dependency_graph",
        primaryArtifactNodeId: "primary_artifact",
        artifactNodes: [
          {
            nodeId: "primary_artifact",
            nodeType: "planned_artifact",
            status: "available"
          },
          {
            nodeId: "runtime_requirements",
            nodeType: "runtime_requirement",
            status: "available"
          }
        ],
        dependencyEdges: [
          {
            sourceNodeId: "runtime_requirements",
            targetNodeId: "primary_artifact",
            relationship: "requires",
            strength: "required"
          }
        ]
      },
      artifact_dependency_graph_available: true
    });
  });

  it("hydrates runtime compatibility workflow metadata", () => {
    const runtimeCompatibility = runtimeCompatibilityFixture();
    const event: AssistantStreamEvent = {
      event_type: "planning",
      sequence: 18,
      payload: {
        workflow: {
          step: "planning",
          phase: "running",
          status: "running",
          current_step: "planning",
          completed_steps: ["intake", "routing"],
          skipped_steps: [],
          refinement_count: 0,
          review_reasons: [],
          artifact_count: 0,
          artifact_critique_count: 0,
          preview_artifact_count: 0,
          image_reference_count: 0,
          image_references: [],
          runtime_compatibility: runtimeCompatibility,
          runtime_compatibility_available: true
        }
      }
    };

    expect(readWorkflowMetadata(event)).toMatchObject({
      step: "planning",
      phase: "running",
      status: "running",
      runtime_compatibility: {
        role: "runtime_compatibility_engine",
        compatibleRuntimes: ["p5_js", "canvas"],
        unsupportedRuntimes: ["glsl"],
        preferredRuntimes: ["p5_js"],
        compatibilityAssessments: [
          {
            runtime: "p5_js",
            compatibility: "compatible",
            portability: "high",
            interoperability: "high"
          }
        ]
      },
      runtime_compatibility_available: true
    });
  });

  it("hydrates artifact capability matrix workflow metadata", () => {
    const artifactCapabilityMatrix = artifactCapabilityMatrixFixture();
    const event: AssistantStreamEvent = {
      event_type: "planning",
      sequence: 19,
      payload: {
        workflow: {
          step: "planning",
          phase: "running",
          status: "running",
          current_step: "planning",
          completed_steps: ["intake", "routing"],
          skipped_steps: [],
          refinement_count: 0,
          review_reasons: [],
          artifact_count: 0,
          artifact_critique_count: 0,
          preview_artifact_count: 0,
          image_reference_count: 0,
          image_references: [],
          artifact_capability_matrix: artifactCapabilityMatrix,
          artifact_capability_matrix_available: true
        }
      }
    };

    expect(readWorkflowMetadata(event)).toMatchObject({
      step: "planning",
      phase: "running",
      status: "running",
      artifact_capability_matrix: {
        role: "artifact_capability_matrix",
        strongestTargets: ["p5_js", "canvas"],
        weakestTargets: ["glsl"],
        capabilityProfiles: [
          {
            target: "p5_js",
            artifactFit: "strong",
            generativeFit: "strong",
            portabilityFit: "strong"
          }
        ]
      },
      artifact_capability_matrix_available: true
    });
  });

  it("hydrates multi-artifact strategy workflow metadata", () => {
    const multiArtifactStrategy = multiArtifactStrategyFixture();
    const event: AssistantStreamEvent = {
      event_type: "planning",
      sequence: 20,
      payload: {
        workflow: {
          step: "planning",
          phase: "running",
          status: "running",
          current_step: "planning",
          completed_steps: ["intake", "routing"],
          skipped_steps: [],
          refinement_count: 0,
          review_reasons: [],
          artifact_count: 0,
          artifact_critique_count: 0,
          preview_artifact_count: 0,
          image_reference_count: 0,
          image_references: [],
          multi_artifact_strategy: multiArtifactStrategy,
          multi_artifact_strategy_available: true
        }
      }
    };

    expect(readWorkflowMetadata(event)).toMatchObject({
      step: "planning",
      phase: "running",
      status: "running",
      multi_artifact_strategy: {
        role: "multi_artifact_strategy",
        primaryArtifact: {
          artifactId: "primary_artifact",
          artifactType: "runnable_code"
        },
        supportingArtifacts: [
          {
            artifactId: "runtime_notes",
            priority: "medium"
          }
        ],
        artifactSequence: [
          {
            stepId: "step_1_primary_artifact",
            action: "produce"
          },
          {
            stepId: "step_2_runtime_notes",
            action: "document"
          }
        ],
        combinationMode: "primary_with_supporting_sections"
      },
      multi_artifact_strategy_available: true
    });
  });

  it("hydrates artifact critic workflow metadata", () => {
    const artifactCritic = artifactCriticFixture();
    const event: AssistantStreamEvent = {
      event_type: "planning",
      sequence: 21,
      payload: {
        workflow: {
          step: "planning",
          phase: "running",
          status: "running",
          current_step: "planning",
          completed_steps: ["intake", "routing"],
          skipped_steps: [],
          refinement_count: 0,
          review_reasons: [],
          artifact_count: 0,
          artifact_critique_count: 0,
          preview_artifact_count: 0,
          image_reference_count: 0,
          image_references: [],
          artifact_critic: artifactCritic,
          artifact_critic_available: true
        }
      }
    };

    expect(readWorkflowMetadata(event)).toMatchObject({
      step: "planning",
      phase: "running",
      status: "running",
      artifact_critic: {
        role: "artifact_critic",
        critiqueConfidence: 0.82,
        riskAssessment: "medium",
        dependencyConcerns: [
          "Runtime-facing dependency conflicts with output structure."
        ],
        runtimeConcerns: [
          "Unsupported runtimes should not be treated as viable targets: glsl."
        ]
      },
      artifact_critic_available: true
    });
  });

  it("hydrates creative critic workflow metadata", () => {
    const creativeCritic = creativeCriticFixture();
    const event: AssistantStreamEvent = {
      event_type: "planning",
      sequence: 22,
      payload: {
        workflow: {
          step: "planning",
          phase: "running",
          status: "running",
          current_step: "planning",
          completed_steps: ["intake", "routing"],
          skipped_steps: [],
          refinement_count: 0,
          review_reasons: [],
          artifact_count: 0,
          artifact_critique_count: 0,
          preview_artifact_count: 0,
          image_reference_count: 0,
          image_references: [],
          creative_critic: creativeCritic,
          creative_critic_available: true
        }
      }
    };

    expect(readWorkflowMetadata(event)).toMatchObject({
      step: "planning",
      phase: "running",
      status: "running",
      creative_critic: {
        role: "creative_critic_engine",
        criticConfidence: 0.86,
        riskAssessment: "medium",
        conceptQuality: 0.82,
        runtimeFitQuality: 0.63,
        creativeWeaknesses: [
          "Runtime fit quality needs an explicit caveat."
        ]
      },
      creative_critic_available: true
    });
  });

  it("hydrates artifact refiner workflow metadata", () => {
    const artifactRefiner = artifactRefinerFixture();
    const event: AssistantStreamEvent = {
      event_type: "planning",
      sequence: 22,
      payload: {
        workflow: {
          step: "planning",
          phase: "running",
          status: "running",
          current_step: "planning",
          completed_steps: ["intake", "routing"],
          skipped_steps: [],
          refinement_count: 0,
          review_reasons: [],
          artifact_count: 0,
          artifact_critique_count: 0,
          preview_artifact_count: 0,
          image_reference_count: 0,
          image_references: [],
          artifact_refiner: artifactRefiner,
          artifact_refiner_available: true
        }
      }
    };

    expect(readWorkflowMetadata(event)).toMatchObject({
      step: "planning",
      phase: "running",
      status: "running",
      artifact_refiner: {
        role: "artifact_refiner",
        refinementConfidence: 0.79,
        priorityImprovements: [
          "Resolve dependency risks before expanding artifact scope."
        ],
        alternativeRefinementPaths: [
          "Capability-first path: clarify target limits first.",
          "Dependency-first path: resolve handoffs and conflicts first."
        ]
      },
      artifact_refiner_available: true
    });
  });

  it("hydrates artifact intelligence synthesis workflow metadata", () => {
    const artifactIntelligenceSynthesis = artifactIntelligenceSynthesisFixture();
    const event: AssistantStreamEvent = {
      event_type: "planning",
      sequence: 23,
      payload: {
        workflow: {
          step: "planning",
          phase: "running",
          status: "running",
          current_step: "planning",
          completed_steps: ["intake", "routing"],
          skipped_steps: [],
          refinement_count: 0,
          review_reasons: [],
          artifact_count: 0,
          artifact_critique_count: 0,
          preview_artifact_count: 0,
          image_reference_count: 0,
          image_references: [],
          artifact_intelligence_synthesis: artifactIntelligenceSynthesis,
          artifact_intelligence_synthesis_available: true
        }
      }
    };

    expect(readWorkflowMetadata(event)).toMatchObject({
      step: "planning",
      phase: "running",
      status: "running",
      artifact_intelligence_synthesis: {
        role: "artifact_intelligence_synthesis",
        synthesisConfidence: 0.83,
        implementationReadiness: "needs_caveats",
        recommendedRuntimeDirection:
          "Document preferred runtime metadata as advisory only: p5_js; use capability matrix caveats for p5_js.",
        majorRisks: [
          "Unsupported runtime remains advisory only: glsl."
        ]
      },
      artifact_intelligence_synthesis_available: true
    });
  });

  it("hydrates artifact merge planner workflow metadata", () => {
    const artifactMergePlanner = artifactMergePlannerFixture();
    const event: AssistantStreamEvent = {
      event_type: "planning",
      sequence: 24,
      payload: {
        workflow: {
          step: "planning",
          phase: "running",
          status: "running",
          current_step: "planning",
          completed_steps: ["intake", "routing"],
          skipped_steps: [],
          refinement_count: 0,
          review_reasons: [],
          artifact_count: 0,
          artifact_critique_count: 0,
          preview_artifact_count: 0,
          image_reference_count: 0,
          image_references: [],
          artifact_merge_planner: artifactMergePlanner,
          artifact_merge_planner_available: true
        }
      }
    };

    expect(readWorkflowMetadata(event)).toMatchObject({
      step: "planning",
      phase: "running",
      status: "running",
      artifact_merge_planner: {
        role: "artifact_merge_planner",
        mergeConfidence: 0.81,
        mergeStrategy: "primary_with_supporting_sections",
        recommendedMergePath:
          "Follow synthesis path as advisory merge guidance: Lead with primary_artifact.",
        runtimeMergeRisks: [
          "Unsupported runtime must not be merged into path: glsl."
        ]
      },
      artifact_merge_planner_available: true
    });
  });

  it("hydrates artifact export intelligence workflow metadata", () => {
    const artifactExportIntelligence = artifactExportIntelligenceFixture();
    const event: AssistantStreamEvent = {
      event_type: "planning",
      sequence: 25,
      payload: {
        workflow: {
          step: "planning",
          phase: "running",
          status: "running",
          current_step: "planning",
          completed_steps: ["intake", "routing"],
          skipped_steps: [],
          refinement_count: 0,
          review_reasons: [],
          artifact_count: 0,
          artifact_critique_count: 0,
          preview_artifact_count: 0,
          image_reference_count: 0,
          image_references: [],
          artifact_export_intelligence: artifactExportIntelligence,
          artifact_export_intelligence_available: true
        }
      }
    };

    expect(readWorkflowMetadata(event)).toMatchObject({
      step: "planning",
      phase: "running",
      status: "running",
      artifact_export_intelligence: {
        role: "artifact_export_intelligence",
        exportConfidence: 0.78,
        exportReadiness: "ready_with_caveats",
        preferredExportTarget: "multi_artifact_package",
        runtimeExportNotes: [
          "Preferred runtimes are advisory export metadata: p5_js."
        ],
        rejectedExportPaths: [
          "Reject direct file export because this engine is metadata-only."
        ]
      },
      artifact_export_intelligence_available: true
    });
  });

  it("hydrates artifact engine contract registry workflow metadata", () => {
    const artifactEngineContracts = artifactEngineContractRegistryFixture();
    const event: AssistantStreamEvent = {
      event_type: "planning",
      sequence: 26,
      payload: {
        workflow: {
          step: "planning",
          phase: "running",
          status: "running",
          current_step: "planning",
          completed_steps: ["intake", "routing"],
          skipped_steps: [],
          refinement_count: 0,
          review_reasons: [],
          artifact_count: 0,
          artifact_critique_count: 0,
          preview_artifact_count: 0,
          image_reference_count: 0,
          image_references: [],
          artifact_engine_contracts: artifactEngineContracts,
          artifact_engine_contracts_available: true
        }
      }
    };

    expect(readWorkflowMetadata(event)).toMatchObject({
      step: "planning",
      phase: "running",
      status: "running",
      artifact_engine_contracts: {
        role: "artifact_intelligence_engine_contract_registry",
        contractCount: 10,
        engineIds: expect.arrayContaining([
          "artifact_planner",
          "artifact_export_intelligence"
        ]),
        engineContracts: expect.arrayContaining([
          expect.objectContaining({
            engineId: "artifact_export_intelligence",
            upstreamDependencies: ["artifact_merge_planner"]
          })
        ])
      },
      artifact_engine_contracts_available: true
    });
  });

  it("reads clarification metadata from prompt input workflow events", () => {
    const clarification = {
      reason: "ambiguous_modality",
      confidence: 0.44,
      summary: "The output modality is unclear.",
      original_query: "Make something evocative about rain.",
      suggested_options: ["Visual sketch", "Audio piece", "Audiovisual piece"],
      default_recommendation: "Visual sketch",
      signal_summary: ["route=generate", "modality=unspecified"],
      questions: [
        {
          id: "output_modality",
          prompt: "What should the assistant generate first?",
          kind: "single_choice",
          suggested_options: ["Visual sketch", "Audio piece", "Audiovisual piece"],
          default_recommendation: "Visual sketch"
        }
      ]
    };
    const event: AssistantStreamEvent = {
      event_type: "prompt_input",
      sequence: 4,
      payload: {
        code: "clarification_required",
        clarification,
        workflow: {
          step: "prompt_input",
          phase: "running",
          status: "running",
          current_step: "prompt_input",
          completed_steps: ["intake", "routing"],
          skipped_steps: [],
          refinement_count: 0,
          review_outcome: null,
          review_reasons: [],
          artifact_count: 0,
          preview_artifact_count: 0,
          image_reference_count: 0,
          image_references: [],
          clarification_required: true,
          clarification_reason: "ambiguous_modality",
          clarification_question_count: 1,
          clarification
        }
      }
    };

    expect(workflowNodeFromAssistantStreamEvent(event)).toBe("prompt_input");
    expect(readClarificationSummary(event.payload.clarification)).toEqual({
      reason: "ambiguous_modality",
      confidence: 0.44,
      summary: "The output modality is unclear.",
      originalQuery: "Make something evocative about rain.",
      suggestedOptions: ["Visual sketch", "Audio piece", "Audiovisual piece"],
      defaultRecommendation: "Visual sketch",
      signalSummary: ["route=generate", "modality=unspecified"],
      questions: [
        {
          id: "output_modality",
          prompt: "What should the assistant generate first?",
          kind: "single_choice",
          suggestedOptions: ["Visual sketch", "Audio piece", "Audiovisual piece"],
          defaultRecommendation: "Visual sketch"
        }
      ]
    });
    expect(readWorkflowMetadata(event)).toEqual(
      expect.objectContaining({
        clarification: expect.objectContaining({
          reason: "ambiguous_modality",
          questions: expect.arrayContaining([
            expect.objectContaining({ id: "output_modality" })
          ])
        }),
        clarification_question_count: 1,
        clarification_reason: "ambiguous_modality",
        clarification_required: true
      })
    );
  });

  it("reads preview runtime updates from preview artifact events", () => {
    const event: AssistantStreamEvent = {
      event_type: "preview_artifact",
      sequence: 4,
      payload: {
        artifact_id: "source-sketch",
        status: "succeeded",
        emitted_at: "2026-05-22T10:25:00Z",
        result: {
          summary: "p5.js runtime ready for sandbox execution.",
          completed_at: "2026-05-22T10:25:00Z",
          preview_artifact_id: "source-sketch",
          request: {
            target: "browser_sandbox"
          },
          provenance: {
            renderer_id: "surface.p5"
          }
        }
      }
    };

    expect(readPreviewArtifactUpdate(event)).toEqual({
      status: "succeeded",
      artifactId: "source-sketch",
      previewArtifactId: "source-sketch",
      rendererId: "surface.p5",
      target: "browser_sandbox",
      summary: "p5.js runtime ready for sandbox execution.",
      errorMessage: null,
      error: null,
      emittedAt: "2026-05-22T10:25:00Z",
      completedAt: "2026-05-22T10:25:00Z"
    });
  });
});
