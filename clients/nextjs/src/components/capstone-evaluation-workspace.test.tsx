import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ProductIntelligenceModel } from "@/lib/product-intelligence";
import { CapstoneEvaluationWorkspace } from "./capstone-evaluation-workspace";

const model = { artifactRegistry: [], details: null } as unknown as ProductIntelligenceModel;

describe("Capstone Evaluation workspace", () => {
  it("presents four separate systems and a provider-safe local preflight", async () => {
    const onRun = vi.fn().mockResolvedValue(undefined);
    render(<CapstoneEvaluationWorkspace history={[]} model={model} onRun={onRun} running={false} />);

    expect(screen.getByRole("heading", { name: "Evidence you can inspect, rerun, and defend." })).toBeVisible();
    expect(screen.getByText(/unique cases/)).toHaveTextContent("31 unique cases");
    expect(screen.getAllByText("RAG / Retrieval").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Creative Artifacts").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Agents & Workflow").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Product Reliability").length).toBeGreaterThan(0);
    expect(screen.queryByText(/overall quality score/i)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Run Evaluation" }));
    expect(screen.getByLabelText("Evaluation preflight")).toHaveTextContent(/no provider call/i);
    const authorization = screen.getByRole("checkbox", { name: /explicitly authorize evaluator provider calls/i });
    expect(authorization).not.toBeChecked();
    expect(screen.getByLabelText("Approved RAGAS dataset")).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Run 31 cases" }));
    await waitFor(() => expect(onRun).toHaveBeenCalledWith({
      scope: "full",
      caseIds: [],
      allowProviderCalls: false,
      approvedRagasDataset: "sanitized_public"
    }));
  });

  it("requires explicit authorization before sending an approved provider request", async () => {
    const onRun = vi.fn().mockResolvedValue(undefined);
    render(<CapstoneEvaluationWorkspace history={[]} model={model} onRun={onRun} running={false} />);

    fireEvent.click(screen.getByRole("button", { name: "Run Evaluation" }));
    fireEvent.click(screen.getByRole("checkbox", { name: /explicitly authorize evaluator provider calls/i }));
    fireEvent.change(screen.getByLabelText("Approved RAGAS dataset"), { target: { value: "redacted_public" } });
    fireEvent.click(screen.getByRole("button", { name: "Run 31 cases" }));

    await waitFor(() => expect(onRun).toHaveBeenCalledWith(expect.objectContaining({
      allowProviderCalls: true,
      approvedRagasDataset: "redacted_public"
    })));
  });
});
