import { describe, expect, it } from "vitest";
import type { ArtifactSummary } from "./assistant-client";
import {
  appendRefinementPassRecord,
  canRunRefinementPass,
  collectRefinementOpportunities,
  enrichArtifactRefinementRequest,
  nextRefinementPassNumber
} from "./refinement-passes";

describe("refinement pass helpers", () => {
  it("enriches one-pass refinement requests with bounded pass metadata", () => {
    const artifact = artifactWithCritique();

    const request = enrichArtifactRefinementRequest(
      {
        artifactId: artifact.id,
        title: artifact.title,
        language: artifact.language,
        content: artifact.content ?? "",
        instruction: "Make the motion more legible."
      },
      artifact
    );

    expect(request.passNumber).toBe(1);
    expect(request.maxPasses).toBe(2);
    expect(request.qualityBefore).toBe(0.55);
    expect(request.refinementObjective).toContain("Make the motion more legible");
    expect(request.refinementObjective).toContain("Clarify focal hierarchy");
    expect(request.refinementPasses).toEqual([]);
  });

  it("records a completed pass when calibrated quality improves", () => {
    const source = artifactWithCritique();
    const request = enrichArtifactRefinementRequest(
      {
        artifactId: source.id,
        title: source.title,
        language: source.language,
        content: source.content ?? "",
        instruction: "Improve the palette."
      },
      source
    );
    const result = artifactWithCritique({
      id: "field-refined",
      qualityScore: 0.72
    });

    const history = appendRefinementPassRecord({
      refinement: request,
      resultArtifact: result,
      sourceArtifact: source
    });

    expect(history).toHaveLength(1);
    expect(history[0]).toMatchObject({
      passNumber: 1,
      sourceArtifactId: source.id,
      resultArtifactId: "field-refined",
      qualityBefore: 0.55,
      qualityAfter: 0.72,
      stopReason: "quality_improved"
    });
  });

  it("stops additional selected-artifact passes at the default limit", () => {
    const artifact = artifactWithCritique({
      refinementPasses: [
        passRecord(1, "continue_available"),
        passRecord(2, "max_passes_reached")
      ]
    });

    expect(nextRefinementPassNumber(artifact)).toBe(3);
    expect(canRunRefinementPass(artifact)).toBe(false);
  });

  it("falls back to no useful opportunities for legacy artifacts", () => {
    const artifact: ArtifactSummary = {
      actions: ["Open"],
      id: "legacy",
      language: "Text",
      status: "Ready",
      summary: "Legacy artifact.",
      title: "legacy.txt",
      type: "code"
    };

    expect(collectRefinementOpportunities(artifact)).toEqual([]);
  });
});

function artifactWithCritique(
  overrides: Partial<ArtifactSummary> = {}
): ArtifactSummary {
  return {
    actions: ["Open", "Preview"],
    content: "function draw() { circle(width / 2, height / 2, 120); }",
    creativeTranslation: {
      audioReactive: {
        activation: "explicit_user_gesture",
        audioRuntime: "web_audio",
        mappings: [
          {
            behavior: "Pulse the halo from amplitude.",
            evidence: ["audio-reactive prompt"],
            intensity: "balanced",
            source: "amplitude",
            targets: ["pulse", "glow"]
          }
        ],
        summary: "Amplitude drives glow.",
        visualRuntime: "p5"
      },
      colorMaterialDirection: [],
      creativeIntent: "Create a calm field.",
      generationConstraints: [],
      geometricReferences: [],
      moodAtmosphere: ["calm"],
      movementLanguage: [],
      musicalReferences: [],
      outputModality: "audiovisual",
      refinementTargets: ["Preserve atmosphere: calm"],
      runtimeRecommendations: ["p5.js"],
      structureDirection: [],
      symbolicReferences: []
    },
    critique: {
      artifactId: "field",
      artifactTitle: "field.p5.js",
      calibratedQuality: {
        adjustments: ["Reduce unsupported symbolic claims."],
        confidence: "medium",
        decisionBand: "needs_refinement",
        legacyScore: 0.62,
        rationale: "Needs refinement.",
        score: overrides.qualityScore ?? 0.55,
        signals: [],
        summary: "Needs refinement at 0.55."
      },
      codeQuality: { rationale: "Code parses.", score: 0.8 },
      creativeEvaluation: {
        aestheticConsistency: observation(),
        coherence: observation(),
        composition: observation(),
        expressiveness: observation(),
        originality: observation(),
        overallScore: 0.58,
        refinementOpportunities: ["Clarify focal hierarchy."],
        strengths: [],
        summary: "Creative quality needs focus."
      },
      creativeQuality: { rationale: "Needs focus.", score: 0.58 },
      domainAppropriateness: { rationale: "p5.js domain.", score: 1 },
      overallScore: overrides.qualityScore ?? 0.55,
      passed: false,
      previewReadiness: { rationale: "Preview ready.", score: 0.8 },
      promptAlignment: { rationale: "Aligned.", score: 0.8 },
      rank: 1,
      rationale: "Needs refinement.",
      reasons: ["sacred_consistency"],
      recommended: true,
      refinementGuidance: "Clarify focal hierarchy.",
      runtimeSuitability: { rationale: "Runtime ready.", score: 1 },
      sourceOrder: 1
    },
    domain: "p5_js",
    id: "field",
    isRecommended: true,
    language: "p5.js",
    previewEligible: true,
    qualityScore: overrides.qualityScore ?? 0.55,
    runtime: "p5",
    status: "Generated",
    summary: "A calm p5.js field.",
    title: "field.p5.js",
    type: "code",
    ...overrides
  };
}

function observation() {
  return {
    evidence: [],
    level: "developing" as const,
    observation: "Developing.",
    score: 0.58
  };
}

function passRecord(
  passNumber: number,
  stopReason: "continue_available" | "max_passes_reached"
) {
  return {
    passNumber,
    sourceArtifactId: "field",
    refinementObjective: "Refine the field.",
    stopReason,
    summary: `Pass ${passNumber}.`
  };
}
