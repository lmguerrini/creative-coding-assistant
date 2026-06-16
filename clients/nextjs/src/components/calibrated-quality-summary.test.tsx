import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { CalibratedQualityEvaluation } from "@/lib/assistant-client";
import { CalibratedQualitySummary } from "./calibrated-quality-summary";

describe("CalibratedQualitySummary", () => {
  it("renders bounded decision support without replacing legacy score", () => {
    render(<CalibratedQualitySummary evaluation={strongEvaluation()} />);

    const summary = screen.getByRole("region", {
      name: "Calibrated quality summary"
    });
    expect(within(summary).getByText("86%")).toBeVisible();
    expect(within(summary).getByText("Strong candidate")).toBeVisible();
    expect(within(summary).getByText("Legacy score")).toBeVisible();
    expect(within(summary).getByText("91%")).toBeVisible();
    expect(
      within(summary).getByRole("list", {
        name: "Calibrated quality signals"
      })
    ).toBeVisible();
    expect(
      within(summary).getByText(/not an objective measure of artistic quality/)
    ).toBeVisible();
  });

  it("surfaces conservative adjustments for risky artifacts", () => {
    render(
      <CalibratedQualitySummary
        evaluation={{
          ...strongEvaluation(),
          score: 0.58,
          decisionBand: "needs_refinement",
          adjustments: [
            "Capped because generated artifact signals contain unsupported symbolic claims."
          ],
          rationale:
            "needs refinement at 0.58; legacy score 0.88. 1 conservative adjustment applied."
        }}
      />
    );

    const summary = screen.getByRole("region", {
      name: "Calibrated quality summary"
    });
    expect(within(summary).getByText("Needs refinement")).toBeVisible();
    expect(within(summary).getByText("Conservative adjustments")).toBeVisible();
    expect(
      within(summary).getByText(
        "Capped because generated artifact signals contain unsupported symbolic claims."
      )
    ).toBeVisible();
  });

  it("preserves legacy inspector behavior when calibration metadata is absent", () => {
    const { container } = render(<CalibratedQualitySummary evaluation={null} />);

    expect(container).toBeEmptyDOMElement();
    expect(
      screen.queryByRole("region", { name: "Calibrated quality summary" })
    ).not.toBeInTheDocument();
  });
});

function strongEvaluation(): CalibratedQualityEvaluation {
  return {
    score: 0.86,
    legacyScore: 0.91,
    decisionBand: "strong_candidate",
    confidence: "medium",
    signals: [
      {
        key: "legacy_critique",
        label: "Legacy critique",
        score: 0.91,
        weight: 0.34,
        rationale: "Existing weighted artifact critique score is preserved."
      },
      {
        key: "runtime_preview",
        label: "Runtime and preview",
        score: 1,
        weight: 0.18,
        rationale: "Runtime suitability and preview readiness are aligned."
      }
    ],
    adjustments: [],
    rationale:
      "strong candidate at 0.86; legacy score 0.91. No conservative caps were required.",
    summary:
      "Calibrated decision-support score 0.86 from 2 available signal(s). This is bounded guidance, not an objective measure of artistic quality."
  };
}
