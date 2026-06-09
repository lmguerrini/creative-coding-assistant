import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  getInitialWorkspaceSnapshot,
  getLocalWorkspaceSnapshot
} from "@/lib/assistant-client";
import { buildRetrievalQualityModel } from "@/lib/retrieval-quality";
import { buildRetrievalRuntimeModel } from "@/lib/retrieval-runtime";
import { RetrievalQualityDeepDive } from "./retrieval-quality-deep-dive";

describe("RetrievalQualityDeepDive", () => {
  it("renders the quality explanation, metric evidence, balance, and weaknesses", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const model = buildRetrievalQualityModel(
      buildRetrievalRuntimeModel(snapshot.retrieval, [])
    );

    render(<RetrievalQualityDeepDive model={model} />);

    const deepDive = screen.getByLabelText("Retrieval quality deep dive");
    expect(deepDive).toHaveAttribute("data-open", "true");
    expect(deepDive).toHaveAttribute("data-quality", "medium");
    expect(deepDive).toHaveTextContent("Medium retrieval quality");
    expect(deepDive).toHaveTextContent(
      "Quality is medium because precision remains the limiting signal."
    );

    const metrics = within(deepDive).getByRole("list", {
      name: "Retrieval quality metrics"
    });
    expect(
      within(metrics).getByRole("listitem", { name: "Retrieval precision" })
    ).toHaveTextContent("83% average");
    expect(
      within(metrics).getByRole("listitem", { name: "Retrieval diversity" })
    ).toHaveTextContent("2 sources · 2 domains");
    expect(
      within(metrics).getByRole("listitem", { name: "Retrieval coverage" })
    ).toHaveTextContent("2/2 requested domains");
    expect(
      within(metrics).getByRole("listitem", {
        name: "Retrieval context sufficiency"
      })
    ).toHaveTextContent("3 chunks selected");

    const balance = within(deepDive).getByRole("region", {
      name: "Retrieval domain balance"
    });
    expect(balance).toHaveTextContent("Balanced across 2 domains");
    expect(balance).toHaveTextContent("WebGPU / WGSL");
    expect(balance).toHaveTextContent("67% of context");

    const weaknesses = within(deepDive).getByRole("region", {
      name: "Retrieval weaknesses"
    });
    expect(weaknesses).toHaveTextContent(
      "Average selected-chunk relevance is moderate rather than high."
    );
    expect(weaknesses).toHaveTextContent("1 selected source may be stale.");
  });

  it("allows evidence-backed analysis to be collapsed and expanded", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const model = buildRetrievalQualityModel(
      buildRetrievalRuntimeModel(snapshot.retrieval, [])
    );

    render(<RetrievalQualityDeepDive model={model} />);

    const deepDive = screen.getByLabelText("Retrieval quality deep dive");
    const toggle = screen.getByLabelText("Toggle retrieval quality deep dive");

    expect(deepDive).toHaveAttribute("data-open", "true");
    expect(toggle).toHaveAttribute("aria-expanded", "true");

    fireEvent.click(toggle);
    expect(deepDive).toHaveAttribute("data-open", "false");
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(
      screen.queryByRole("list", { name: "Retrieval quality metrics" })
    ).not.toBeInTheDocument();

    fireEvent.click(toggle);
    expect(deepDive).toHaveAttribute("data-open", "true");
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(
      screen.getByRole("list", { name: "Retrieval quality metrics" })
    ).toBeVisible();
  });

  it("renders legacy quality safely without fabricated score percentages", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const retrieval = {
      ...snapshot.retrieval,
      sources: snapshot.retrieval.sources.map((source) => ({
        ...source,
        chunks: source.chunks.map((chunk) => ({
          ...chunk,
          score: null
        }))
      }))
    };
    const model = buildRetrievalQualityModel(
      buildRetrievalRuntimeModel(retrieval, [])
    );

    render(<RetrievalQualityDeepDive model={model} />);

    const precision = screen.getByRole("listitem", {
      name: "Retrieval precision"
    });
    expect(precision).toHaveTextContent("Not scored");
    expect(precision).toHaveTextContent(
      "score-based precision cannot be verified"
    );
    expect(precision).not.toHaveTextContent(/% average/);
  });

  it("keeps first-run analysis collapsed with an honest empty explanation", () => {
    const snapshot = getInitialWorkspaceSnapshot();
    const model = buildRetrievalQualityModel(
      buildRetrievalRuntimeModel(snapshot.retrieval, [])
    );

    render(<RetrievalQualityDeepDive model={model} />);

    const deepDive = screen.getByLabelText("Retrieval quality deep dive");
    expect(deepDive).toHaveAttribute("data-open", "false");
    expect(deepDive).toHaveTextContent("Retrieval quality unknown");
    expect(deepDive).toHaveTextContent(
      "No selected retrieval evidence is available for a quality assessment."
    );

    const toggle = screen.getByLabelText("Toggle retrieval quality deep dive");
    expect(toggle).toHaveAttribute("aria-expanded", "false");

    fireEvent.click(toggle);
    expect(deepDive).toHaveAttribute("data-open", "true");
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(
      screen.getByRole("list", { name: "Retrieval quality metrics" })
    ).toBeVisible();

    fireEvent.click(toggle);
    expect(deepDive).toHaveAttribute("data-open", "false");
    expect(toggle).toHaveAttribute("aria-expanded", "false");
  });
});
