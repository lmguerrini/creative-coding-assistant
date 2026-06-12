import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { CreativeTranslationSummary } from "@/lib/assistant-client";
import { AudioReactiveMappingSummaryCard } from "./audio-reactive-mapping-summary";

describe("AudioReactiveMappingSummaryCard", () => {
  it("renders compact bounded mappings and explicit activation safety", () => {
    render(
      <AudioReactiveMappingSummaryCard translation={audiovisualTranslation()} />
    );

    const card = screen.getByRole("region", {
      name: "Audio-reactive mapping summary"
    });
    expect(within(card).getByText("2 bounded links")).toBeVisible();
    expect(within(card).getByText("Amplitude")).toBeVisible();
    expect(within(card).getByText("Scale / Glow")).toBeVisible();
    expect(within(card).getByText("Rhythm")).toBeVisible();
    expect(within(card).getByText("Rotation / Pattern phase")).toBeVisible();
    expect(within(card).getByText("Tone.js")).toBeVisible();
    expect(within(card).getByText("Hydra")).toBeVisible();
    expect(
      within(card).getByText(/Audio remains silent until explicit start/)
    ).toBeVisible();
  });

  it("does not imply mappings for visual-only artifacts", () => {
    render(
      <AudioReactiveMappingSummaryCard
        translation={{
          ...audiovisualTranslation(),
          audioReactive: null,
          outputModality: "visual"
        }}
      />
    );

    expect(screen.getByText("Not active")).toBeVisible();
    expect(
      screen.getByText(
        "Audio-reactive mappings are only derived for explicit audiovisual intent."
      )
    ).toBeVisible();
  });

  it("renders a safe fallback for legacy audiovisual metadata", () => {
    render(
      <AudioReactiveMappingSummaryCard
        translation={{
          ...audiovisualTranslation(),
          audioReactive: undefined
        }}
      />
    );

    expect(screen.getByText("Mapping unavailable")).toBeVisible();
    expect(
      screen.getByText(/Existing source and preview behavior remain unchanged/)
    ).toBeVisible();
  });
});

function audiovisualTranslation(): CreativeTranslationSummary {
  return {
    audioReactive: {
      activation: "explicit_user_gesture",
      audioRuntime: "Tone.js",
      mappings: [
        {
          behavior: "Smooth short peaks so scale and light remain readable.",
          evidence: ["user prompt", "shader presets"],
          intensity: "balanced",
          source: "amplitude",
          targets: ["scale", "glow"]
        },
        {
          behavior: "Quantize structural changes to the requested pulse or BPM.",
          evidence: ["user prompt", "Tone.js metadata"],
          intensity: "subtle",
          source: "rhythm",
          targets: ["rotation", "pattern_phase"]
        }
      ],
      summary: "amplitude -> scale / glow; rhythm -> rotation / pattern phase",
      visualRuntime: "Hydra"
    },
    colorMaterialDirection: ["cyan"],
    creativeIntent: "Create a calm audio-reactive field.",
    generationConstraints: [
      "Require explicit user interaction before audio playback"
    ],
    geometricReferences: [],
    moodAtmosphere: ["calm"],
    movementLanguage: ["pulse"],
    musicalReferences: ["rhythm"],
    outputModality: "audiovisual",
    refinementTargets: [],
    runtimeRecommendations: ["Hydra", "Tone.js"],
    structureDirection: [],
    symbolicReferences: []
  };
}
