import { describe, expect, it } from "vitest";
import type { ArtifactSummary } from "./assistant-client";
import {
  buildArtifactComparisonModel,
  classifyArtifactRuntimeSupport
} from "./artifact-comparison";

const baseArtifact: ArtifactSummary = {
  actions: ["Open", "Copy", "Download"],
  id: "base",
  language: "JavaScript",
  status: "Generated",
  summary: "Generated artifact.",
  title: "base.js",
  type: "code"
};

function artifact(overrides: Partial<ArtifactSummary>): ArtifactSummary {
  return {
    ...baseArtifact,
    ...overrides
  };
}

function critique(overrides: Partial<NonNullable<ArtifactSummary["critique"]>>) {
  return {
    artifactId: "artifact",
    artifactTitle: "artifact.p5.js",
    codeQuality: { rationale: "Complete source.", score: 0.9 },
    creativeQuality: { rationale: "Strong composition.", score: 0.92 },
    domainAppropriateness: { rationale: "Domain matches.", score: 0.9 },
    overallScore: 0.94,
    passed: true,
    previewReadiness: { rationale: "Preview ready.", score: 1 },
    promptAlignment: { rationale: "Matches prompt.", score: 0.94 },
    rank: 1,
    rationale: "Best artifact candidate.",
    reasons: [],
    recommended: true,
    refinementGuidance: null,
    runtimeSuitability: { rationale: "Runtime supported.", score: 0.95 },
    sourceOrder: 1,
    ...overrides
  };
}

describe("artifact comparison", () => {
  it("highlights the recommended candidate and preserves critique rationale", () => {
    const recommended = artifact({
      actions: ["Open", "Preview", "Copy", "Download"],
      critique: critique({ artifactId: "p5", artifactTitle: "field.p5.js" }),
      domain: "p5_js",
      id: "p5",
      isRecommended: true,
      language: "p5.js",
      qualityRank: 1,
      qualityScore: 0.94,
      rendererId: "surface.p5",
      runtime: "p5",
      title: "field.p5.js"
    });
    const alternate = artifact({
      id: "notes",
      language: "Markdown",
      title: "notes.md",
      type: "export"
    });

    const model = buildArtifactComparisonModel({
      activeArtifactId: "notes",
      artifacts: [alternate, recommended]
    });

    expect(model.recommendedRow?.artifactId).toBe("p5");
    expect(model.recommendedReason).toBe("Best artifact candidate.");
    expect(model.rows.find((row) => row.artifactId === "notes")?.isActive).toBe(
      true
    );
    expect(model.rows.find((row) => row.artifactId === "p5")?.scoreLabel).toBe(
      "94%"
    );
  });

  it("distinguishes previewable, code-only, and unsupported runtime artifacts", () => {
    const previewable = artifact({
      actions: ["Open", "Preview", "Copy"],
      domain: "glsl",
      id: "shader",
      language: "GLSL",
      rendererId: "surface.glsl",
      runtime: "glsl",
      title: "shader.frag"
    });
    const codeOnly = artifact({
      id: "notes",
      language: "Markdown",
      title: "notes.md",
      type: "export"
    });
    const unsupported = artifact({
      domain: "hydra",
      id: "hydra",
      previewEligible: false,
      previewTarget: "",
      rendererId: null,
      runtime: null,
      title: "feedback-lattice.hydra.js"
    });

    expect(classifyArtifactRuntimeSupport(previewable).state).toBe("previewable");
    expect(classifyArtifactRuntimeSupport(codeOnly).state).toBe("code_only");
    expect(classifyArtifactRuntimeSupport(unsupported).state).toBe("unsupported");
  });
});
