import { describe, expect, it } from "vitest";
import type { ArtifactSummary } from "./assistant-client";
import { resolveArtifactFollowUp } from "./artifact-follow-up";

const artifact: ArtifactSummary = {
  actions: ["Open", "Preview"],
  content: "function draw() { background(20); }",
  id: "latest-sketch",
  language: "p5.js",
  status: "Ready",
  summary: "Latest generated sketch.",
  title: "latest-sketch.p5.js",
  type: "code"
};

describe("resolveArtifactFollowUp", () => {
  it.each([
    "make it brighter",
    "Could you make this calmer?",
    "Improve performance",
    "add audio-reactive behavior",
    "slightly darker",
    "more organic"
  ])("targets the latest active artifact for %s", (prompt) => {
    expect(
      resolveArtifactFollowUp({
        activeArtifact: artifact,
        artifacts: [artifact],
        prompt
      })
    ).toMatchObject({
      artifact,
      confidence: "high",
      kind: "refinement"
    });
  });

  it.each([
    "Make a new brighter particle system",
    "Create another shader",
    "What do you think of it?",
    "it",
    "Explain how brightness works"
  ])("keeps ambiguous or new-generation prompt in normal chat for %s", (prompt) => {
    expect(
      resolveArtifactFollowUp({
        activeArtifact: artifact,
        artifacts: [artifact],
        prompt
      }).kind
    ).toBe("none");
  });

  it("does not bypass the refinement pass limit", () => {
    const exhaustedArtifact: ArtifactSummary = {
      ...artifact,
      refinementPasses: [
        {
          passNumber: 2,
          refinementObjective: "Complete the final bounded pass.",
          sourceArtifactId: artifact.id,
          stopReason: "max_passes_reached",
          summary: "Pass limit reached."
        }
      ]
    };

    expect(
      resolveArtifactFollowUp({
        activeArtifact: exhaustedArtifact,
        artifacts: [exhaustedArtifact],
        prompt: "make it brighter"
      }).kind
    ).toBe("none");
  });
});
