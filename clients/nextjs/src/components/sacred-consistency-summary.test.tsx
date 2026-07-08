import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type {
  SacredConsistencyEvaluation,
  SacredConsistencyObservation
} from "@/lib/assistant-client";
import { SacredConsistencySummary } from "./sacred-consistency-summary";

describe("SacredConsistencySummary", () => {
  it("renders bounded geometry consistency observations when metadata exists", () => {
    render(<SacredConsistencySummary evaluation={alignedEvaluation()} />);

    const evaluator = screen.getByRole("region", {
      name: "Geometry consistency evaluator"
    });
    expect(within(evaluator).getByText("86%")).toBeVisible();
    expect(
      within(evaluator).getByRole("list", {
        name: "Geometry consistency dimensions"
      })
    ).toBeVisible();
    expect(within(evaluator).getByText("Guidance alignment")).toBeVisible();
    expect(within(evaluator).getByText("Claim safety")).toBeVisible();
    expect(within(evaluator).getByText("Strengths")).toBeVisible();
    expect(
      within(evaluator).queryByText("Refinement opportunities")
    ).not.toBeInTheDocument();
  });

  it("surfaces unsupported symbolic claim refinements without new claims", () => {
    render(
      <SacredConsistencySummary
        evaluation={{
          ...alignedEvaluation(),
          overallScore: 0.58,
          claimSafety: observation(
            0.24,
            "unsupported",
            "Detected unsupported symbolic authority language."
          ),
          strengths: [],
          refinementOpportunities: [
            "Replace symbolic authority claims with bounded visual design language."
          ],
          summary:
            "Checked 3 symbolic/geometric metadata cues; 1 dimension needs refinement."
        }}
      />
    );

    const evaluator = screen.getByRole("region", {
      name: "Geometry consistency evaluator"
    });
    expect(within(evaluator).getByText("58%")).toBeVisible();
    expect(within(evaluator).getByText("unsupported")).toBeVisible();
    expect(within(evaluator).getByText("Refinement opportunities")).toBeVisible();
    expect(
      within(evaluator).getByText(
        "Replace conceptual authority claims with bounded visual design language."
      )
    ).toBeVisible();
  });

  it("preserves legacy inspector behavior when metadata is absent", () => {
    const { container } = render(<SacredConsistencySummary evaluation={null} />);

    expect(container).toBeEmptyDOMElement();
    expect(
      screen.queryByRole("region", { name: "Geometry consistency evaluator" })
    ).not.toBeInTheDocument();
  });
});

function alignedEvaluation(): SacredConsistencyEvaluation {
  return {
    overallScore: 0.86,
    alignment: observation(
      0.88,
      "aligned",
      "Matched requested mandala and radial symmetry cues."
    ),
    motifConsistency: observation(
      0.84,
      "aligned",
      "Detected geometric construction signals."
    ),
    modalityCoherence: observation(
      0.83,
      "aligned",
      "Detected visual and motion signals."
    ),
    claimSafety: observation(
      0.9,
      "aligned",
      "No unsupported symbolic authority markers were detected."
    ),
    strengths: ["Claim safety: No unsupported symbolic authority markers."],
    refinementOpportunities: [],
    summary:
      "Checked 3 symbolic/geometric metadata cues; 4 dimensions are aligned."
  };
}

function observation(
  score: number,
  level: SacredConsistencyObservation["level"],
  text: string
): SacredConsistencyObservation {
  return {
    score,
    level,
    observation: text,
    evidence: []
  };
}
