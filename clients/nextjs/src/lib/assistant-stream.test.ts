import { describe, expect, it, vi } from "vitest";
import {
  AssistantStreamError,
  decodeAssistantStream,
  parseAssistantStreamLine,
  readClarificationSummary,
  readCreativeCompositionPlanSummary,
  readCreativeConstraintSolverSummary,
  readCreativeExecutionPlanSummary,
  readCreativeQualityPredictionSummary,
  readCreativeReasoningSummary,
  readCreativeStrategySummary,
  readCreativeTechniqueSummary,
  readCreativeTradeoffExplorerSummary,
  readProceduralStructurePlanSummary,
  readSymbolicNarrativePlanSummary,
  readEventTimestamp,
  readPreviewArtifactUpdate,
  readRuntimeCapabilityReasonerSummary,
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
