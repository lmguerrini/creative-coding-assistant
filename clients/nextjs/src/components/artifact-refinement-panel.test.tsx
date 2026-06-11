import {
  act,
  fireEvent,
  render,
  screen,
  within
} from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ArtifactSummary } from "@/lib/assistant-client";
import { ArtifactRefinementPanel } from "./artifact-refinement-panel";

describe("ArtifactRefinementPanel", () => {
  it("renders artifact-specific controls and serializes changed values", async () => {
    const artifact = visualArtifact();
    const onArtifactRefine = vi.fn(
      async (_artifact: ArtifactSummary, _instruction: string) => undefined
    );

    render(
      <ArtifactRefinementPanel
        artifact={artifact}
        disabled={false}
        onArtifactRefine={onArtifactRefine}
      />
    );

    const parameterPanel = screen.getByRole("region", {
      name: "Artifact parameter controls"
    });

    expect(within(parameterPanel).getByText("GLSL")).toBeVisible();
    expect(within(parameterPanel).getByText("Refinement guidance")).toBeVisible();
    expect(within(parameterPanel).getByLabelText("Movement complexity parameter")).toHaveValue(
      "3"
    );
    expect(within(parameterPanel).getByLabelText("Accent color parameter")).toHaveValue(
      "#45d8c8"
    );

    fireEvent.change(
      within(parameterPanel).getByLabelText("Movement complexity parameter"),
      { target: { value: "8" } }
    );
    fireEvent.change(
      within(parameterPanel).getByLabelText("Accent color parameter"),
      { target: { value: "#ff7668" } }
    );

    expect(within(parameterPanel).getByText("2 local changes ready for refinement")).toBeVisible();

    await act(async () => {
      fireEvent.click(
        screen.getByRole("button", {
          name: "Refine with parameter changes"
        })
      );
    });

    expect(onArtifactRefine).toHaveBeenCalledWith(
      artifact,
      expect.stringContaining("Apply the selected artifact parameter changes.")
    );
    expect(onArtifactRefine.mock.calls[0][1]).toContain(
      "Accent color: #ff7668"
    );
    expect(onArtifactRefine.mock.calls[0][1]).toContain(
      "Movement complexity: 8"
    );
    expect(artifact.content).toBe("void main() { vec3 glow = vec3(0.0); }");
  });

  it("resets local parameter changes without submitting refinement", () => {
    const onArtifactRefine = vi.fn(
      async (_artifact: ArtifactSummary, _instruction: string) => undefined
    );

    render(
      <ArtifactRefinementPanel
        artifact={visualArtifact()}
        disabled={false}
        onArtifactRefine={onArtifactRefine}
      />
    );

    fireEvent.change(screen.getByLabelText("Movement complexity parameter"), {
      target: { value: "9" }
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Reset parameters" })
    );

    expect(screen.getByLabelText("Movement complexity parameter")).toHaveValue(
      "3"
    );
    expect(
      screen.getByRole("button", { name: "Refine selected artifact" })
    ).toBeDisabled();
    expect(onArtifactRefine).not.toHaveBeenCalled();
  });

  it("keeps manual refinement available for unsupported legacy artifacts", async () => {
    const artifact: ArtifactSummary = {
      actions: ["Open"],
      id: "legacy",
      language: "Text",
      status: "Ready",
      summary: "Legacy artifact without parameter metadata.",
      title: "legacy.txt",
      type: "code"
    };
    const onArtifactRefine = vi.fn(
      async (_artifact: ArtifactSummary, _instruction: string) => undefined
    );

    render(
      <ArtifactRefinementPanel
        artifact={artifact}
        disabled={false}
        onArtifactRefine={onArtifactRefine}
      />
    );

    expect(screen.getByText("No safe parameters derived")).toBeVisible();
    fireEvent.change(screen.getByLabelText("Refinement instruction"), {
      target: { value: "Make the output more concise." }
    });
    await act(async () => {
      fireEvent.click(
        screen.getByRole("button", { name: "Refine selected artifact" })
      );
    });

    expect(onArtifactRefine).toHaveBeenCalledWith(
      artifact,
      "Make the output more concise."
    );
  });
});

function visualArtifact(): ArtifactSummary {
  return {
    actions: ["Open", "Preview"],
    content: "void main() { vec3 glow = vec3(0.0); }",
    creativeTranslation: {
      colorMaterialDirection: ["cyan"],
      creativeIntent: "Create a calm sacred shader.",
      generationConstraints: [],
      geometricReferences: ["mandala"],
      moodAtmosphere: ["calm", "minimal"],
      movementLanguage: ["slow pulse"],
      musicalReferences: [],
      outputModality: "visual",
      refinementTargets: [],
      runtimeRecommendations: ["GLSL"],
      sacredGeometry: {
        audioImplications: [],
        colorMaterialDirection: [],
        concepts: ["mandala"],
        generationConstraints: [],
        geometricStructure: [],
        movementBehavior: [],
        runtimeRecommendations: ["GLSL"],
        symmetryType: ["radial"],
        visualComposition: []
      },
      shaderPresets: {
        colorBehavior: [],
        lightMaterialBehavior: [],
        motionBehavior: [],
        performanceConstraints: [],
        presets: ["glow"],
        runtimeSuitability: ["GLSL"],
        shaderStructure: []
      },
      structureDirection: [],
      symbolicReferences: [],
      visualStyle: {
        compositionTendencies: [],
        contrastBehavior: [],
        motionTendencies: [],
        paletteBehavior: ["cyan monochrome"],
        runtimeSuitability: ["GLSL"],
        spatialOrganization: [],
        styles: ["minimal"],
        textureTendencies: []
      }
    },
    domain: "glsl",
    id: "shader-field",
    language: "GLSL",
    previewEligible: true,
    runtime: "glsl",
    status: "Generated",
    summary: "Calm cyan shader.",
    title: "field.frag",
    type: "code"
  };
}
