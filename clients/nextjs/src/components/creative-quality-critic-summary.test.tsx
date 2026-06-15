import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type {
  CreativeQualityEvaluation,
  CreativeQualityObservation
} from "@/lib/assistant-client";
import { CreativeQualityCriticSummary } from "./creative-quality-critic-summary";

describe("CreativeQualityCriticSummary", () => {
  it("renders strong structured observations without inventing refinements", () => {
    render(<CreativeQualityCriticSummary evaluation={strongEvaluation()} />);

    const critic = screen.getByRole("region", {
      name: "Creative quality critic"
    });
    expect(within(critic).getByText("86%")).toBeVisible();
    expect(
      within(critic).getByRole("list", {
        name: "Creative quality dimensions"
      })
    ).toBeVisible();
    expect(within(critic).getByText("Aesthetic consistency")).toBeVisible();
    expect(within(critic).getByText("Strengths")).toBeVisible();
    expect(
      within(critic).queryByText("Refinement opportunities")
    ).not.toBeInTheDocument();
  });

  it("surfaces bounded actionable feedback for weak outputs", () => {
    render(
      <CreativeQualityCriticSummary
        evaluation={{
          ...strongEvaluation(),
          overallScore: 0.34,
          composition: observation(0.32, "weak", "No focal hierarchy detected."),
          originality: observation(0.28, "weak", "No generative variation detected."),
          strengths: [],
          refinementOpportunities: [
            "Clarify focal hierarchy and spatial balance.",
            "Add a distinctive generative transformation."
          ],
          summary: "0 of 5 creative dimensions are strong."
        }}
      />
    );

    const critic = screen.getByRole("region", {
      name: "Creative quality critic"
    });
    expect(within(critic).getByText("34%")).toBeVisible();
    expect(within(critic).getByText("Refinement opportunities")).toBeVisible();
    expect(
      within(critic).getByText("Clarify focal hierarchy and spatial balance.")
    ).toBeVisible();
  });

  it("preserves the legacy inspector when evaluation metadata is absent", () => {
    const { container } = render(
      <CreativeQualityCriticSummary evaluation={null} />
    );

    expect(container).toBeEmptyDOMElement();
    expect(
      screen.queryByRole("region", { name: "Creative quality critic" })
    ).not.toBeInTheDocument();
  });
});

function strongEvaluation(): CreativeQualityEvaluation {
  return {
    overallScore: 0.86,
    composition: observation(0.88, "strong", "Clear focal hierarchy detected."),
    originality: observation(0.82, "strong", "Generative variation is present."),
    coherence: observation(0.91, "strong", "Runtime structure is coherent."),
    aestheticConsistency: observation(
      0.84,
      "strong",
      "Palette and material signals are consistent."
    ),
    expressiveness: observation(
      0.85,
      "strong",
      "Motion and variation develop over time."
    ),
    strengths: [
      "Composition: Clear focal hierarchy detected.",
      "Coherence: Runtime structure is coherent."
    ],
    refinementOpportunities: [],
    summary: "5 of 5 creative dimensions are strong."
  };
}

function observation(
  score: number,
  level: CreativeQualityObservation["level"],
  text: string
): CreativeQualityObservation {
  return {
    score,
    level,
    observation: text,
    evidence: []
  };
}
