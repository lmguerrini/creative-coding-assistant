import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { getLocalWorkspaceSnapshot } from "@/lib/assistant-client";
import { buildRetrievalRuntimeModel } from "@/lib/retrieval-runtime";
import { RetrievalInspector } from "./retrieval-inspector";

describe("RetrievalInspector", () => {
  it("keeps the workstation inspector focused on current-run signals", () => {
    const runtime = createRuntime();

    render(<RetrievalInspector runtime={runtime} showDebugPanels />);

    const panel = screen.getByRole("tabpanel", {
      name: "Retrieval inspector"
    });

    expect(
      within(panel).getByRole("group", { name: "Retrieval status" })
    ).toHaveTextContent("Retrieved context available");
    expect(
      within(panel).getByRole("group", { name: "Retrieval confidence" })
    ).toHaveTextContent("Medium confidence");
    expect(
      within(panel).getByRole("group", { name: "Retrieval coverage" })
    ).toHaveTextContent("2/2 domains covered");
    expect(
      within(panel).getByRole("group", { name: "Knowledge Base status" })
    ).toBeVisible();
    expect(
      within(panel).queryByLabelText("Retrieval quality deep dive")
    ).not.toBeInTheDocument();
    expect(
      within(panel).queryByRole("region", {
        name: "Retrieval source explorer"
      })
    ).not.toBeInTheDocument();
  });

  it("preserves deep retrieval evidence in the dashboard presentation", () => {
    render(<RetrievalInspector runtime={createRuntime()} />);

    expect(screen.getByLabelText("Retrieval quality deep dive")).toBeVisible();
    expect(
      screen.getByRole("region", { name: "Retrieval source explorer" })
    ).toBeVisible();
  });
});

function createRuntime() {
  const snapshot = getLocalWorkspaceSnapshot();

  return buildRetrievalRuntimeModel(snapshot.retrieval, []);
}
