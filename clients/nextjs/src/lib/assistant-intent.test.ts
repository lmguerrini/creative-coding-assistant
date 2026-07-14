import { describe, expect, it } from "vitest";
import { resolveAssistantRequestMode } from "./assistant-intent";

describe("assistant request intent", () => {
  it.each([
    "what's creative coding?",
    "How does p5.js work?",
    "Explain creative coding",
    "Compare p5.js and Three.js",
    "How do I create generative art?",
    "Che cos'è il creative coding?",
    "Spiega come funziona p5.js"
  ])("routes an informational request to explain: %s", (prompt) => {
    expect(resolveAssistantRequestMode({ prompt })).toBe("explain");
  });

  it.each([
    "Create a generative visual",
    "Can you create a generative visual?",
    "Make it brighter",
    "Refine the current shader",
    "Crea un visual generativo"
  ])("keeps a delegated creative request on generate: %s", (prompt) => {
    expect(resolveAssistantRequestMode({ prompt })).toBe("generate");
  });

  it("keeps refinement and clarification continuations on generate", () => {
    expect(
      resolveAssistantRequestMode({
        hasArtifactRefinement: true,
        prompt: "Why is it so dark?"
      })
    ).toBe("generate");
    expect(
      resolveAssistantRequestMode({
        hasClarificationResponse: true,
        prompt: "Visual sketch"
      })
    ).toBe("generate");
  });
});
