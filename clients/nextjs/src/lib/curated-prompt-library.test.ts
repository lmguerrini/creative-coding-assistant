import { describe, expect, it } from "vitest";
import {
  demoShowcasePromptLibrary,
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

  it("keeps Demo Mode distinct from Homepage while covering every active runtime", () => {
    expect(demoShowcasePromptLibrary.map((prompt) => prompt.id)).toEqual([
      "polyrhythmic-constellation",
      "recursive-aurora-garden",
      "kinetic-orbit-capstone",
      "fractal-solar-bloom"
    ]);
    expect(new Set(demoShowcasePromptLibrary.map((prompt) => prompt.runtime)).size).toBe(4);
    expect(
      demoShowcasePromptLibrary.filter((showcase) =>
        homepagePromptLibrary.some(
          (homepage) => String(homepage.prompt) === String(showcase.prompt)
        )
      )
    ).toEqual([]);
    expect(
      demoShowcasePromptLibrary.filter((showcase) =>
        homepagePromptLibrary.some(
          (homepage) =>
            String(homepage.expectedArtifact) === String(showcase.expectedArtifact)
        )
      )
    ).toEqual([]);
  });

  it("authors the Three.js demo hero around geometry, camera, and parent transforms", () => {
    const prompt = demoShowcasePromptLibrary.find(
      (candidate) => candidate.id === "kinetic-orbit-capstone"
    );

    expect(prompt?.prompt).toContain("TorusKnotGeometry");
    expect(prompt?.prompt).toContain("sculptureRig");
    expect(prompt?.prompt).toContain("orbitRig");
    expect(prompt?.prompt).toContain("cameraRig");
    expect(prompt?.prompt).toContain("camera.lookAt()");
    expect(prompt?.previewBoundary).toContain("Three.js r176");
  });

  it("keeps the Three.js hero visually specific and inside the controlled color API", () => {
    const prompt = homepagePromptLibrary.find(
      (candidate) => candidate.id === "kinetic-orbit-sculpture"
    );

    expect(prompt?.prompt).toContain("TorusKnotGeometry");
    expect(prompt?.prompt).toContain("warm-gold sculpture");
    expect(prompt?.prompt).toContain("never call setHSL");
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

  it("offers one starter for every supported runtime boundary", () => {
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
