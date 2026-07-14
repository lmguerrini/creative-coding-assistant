import { describe, expect, it } from "vitest";
import {
  boundedP5DemoSurfaceContract,
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

  it("keeps generated p5 demos inside the preview allowlist", () => {
    const aurora = demoShowcasePromptLibrary.find(
      (candidate) => candidate.id === "recursive-aurora-garden"
    );

    expect(boundedP5DemoSurfaceContract).toContain("Use frameCount for time");
    expect(boundedP5DemoSurfaceContract).toContain("Do not use createGraphics");
    expect(aurora?.prompt).toContain(boundedP5DemoSurfaceContract);
    expect(rhythmicLineStudyPrompt.prompt).toContain(boundedP5DemoSurfaceContract);
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

  it("keeps the GLSL demo close to the blue-gold spiral reference", () => {
    const prompt = demoShowcasePromptLibrary.find(
      (candidate) => candidate.id === "fractal-solar-bloom"
    );

    expect(prompt?.prompt).toContain("compile-ready WebGL 1 source under 80 lines");
    expect(prompt?.prompt).toContain("uniform float u_time; uniform vec2 u_resolution;");
    expect(prompt?.prompt).toContain("float spiral(vec2 p,float scale,float twist)");
    expect(prompt?.prompt).toContain("asymmetric logarithmic nautilus curls");
    expect(prompt?.prompt).toContain("cyan-white rims and amber-gold cores");
    expect(prompt?.prompt).toContain("avoid a centered flower");
    expect(prompt?.prompt).toContain("balance braces/parentheses");
    expect(prompt?.prompt.length).toBeLessThanOrEqual(800);
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
