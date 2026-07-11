import { describe, expect, it } from "vitest";
import {
  domainStarterPromptLibrary,
  homepagePromptLibrary,
  morphogenesisPromptLibrary,
  rhythmicLineStudyPrompt
} from "./curated-prompt-library";

describe("curated prompt library", () => {
  it("keeps the homepage concise and limited to controlled runtime paths", () => {
    expect(homepagePromptLibrary.map((prompt) => prompt.id)).toEqual([
      "physarum-drift",
      "kinetic-orbit-sculpture",
      "chladni-light-field",
      "cymatic-audio-study"
    ]);
    expect(homepagePromptLibrary).toHaveLength(4);
    expect(homepagePromptLibrary.every((prompt) => prompt.prompt.includes("Return only"))).toBe(
      true
    );
  });

  it("records all canonical morphogenesis inspirations without claiming external execution", () => {
    expect(morphogenesisPromptLibrary).toHaveLength(12);
    expect(morphogenesisPromptLibrary.map((prompt) => prompt.concept)).toEqual([
      "Physarum-inspired collective motion",
      "Cellular automata",
      "Chladni-style audio-visual mapping",
      "Fibonacci sequence",
      "Fractals",
      "Golden ratio",
      "Phyllotaxis",
      "Strange attractors",
      "Superformula",
      "Belousov–Zhabotinsky reaction",
      "Hele-Shaw flow",
      "Fluid simulation"
    ]);
    expect(morphogenesisPromptLibrary.every((prompt) => prompt.fallback)).toBe(true);
    expect(rhythmicLineStudyPrompt.previewBoundary).toContain("p5.js");
  });

  it("offers one starter for every supported runtime boundary without promoting Hydra to Demo Mode", () => {
    expect(domainStarterPromptLibrary.map((prompt) => prompt.id)).toEqual([
      "physarum-drift",
      "kinetic-orbit-sculpture",
      "chladni-light-field",
      "cymatic-audio-study",
      "feedback-lattice-hydra",
      "signal-bloom-gsap",
      "signal-markup-svg",
      "signal-grid-canvas"
    ]);
    expect(
      domainStarterPromptLibrary.find(
        (prompt) => prompt.id === "feedback-lattice-hydra"
      )?.previewBoundary
    ).toContain("code/export-only in Demo Mode");
  });
});
