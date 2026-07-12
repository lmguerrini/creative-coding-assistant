import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  productIntelligenceCategories,
  type ProductIntelligenceModel
} from "@/lib/product-intelligence";
import { loadingDomainExperienceCatalog } from "@/lib/domain-experience";
import {
  ProductIntelligenceDashboard,
  ProductIntelligenceInspector
} from "./product-intelligence-dashboard";

function buildModel(): ProductIntelligenceModel {
  return {
    activeDomainId: null,
    domainExperience: loadingDomainExperienceCatalog,
    artifactRegistry: [],
    session: { id: "session-1", title: "Creative workspace" },
    sections: productIntelligenceCategories.map((category) => ({
      category,
      tone: category === "Workflow" ? "active" : "ready",
      summary: `${category} summary`,
      detail: `${category} detailed product information`,
      metrics: [
        { label: "Primary", value: `${category} value` },
        { label: "Secondary", value: "Available" }
      ],
      notes: [`${category} is derived from the shared product model.`]
    })),
    summary: { activeCount: 1, attentionCount: 0, readyCount: 18 }
  };
}

describe("Product Intelligence surfaces", () => {
  it("separates Dashboard categories into focused decision pages", () => {
    const onCategoryChange = vi.fn();
    const onClose = vi.fn();

    render(
      <ProductIntelligenceDashboard
        activeCategory="Overview"
        model={buildModel()}
        onCategoryChange={onCategoryChange}
        onClose={onClose}
      />
    );

    const navigation = screen.getByRole("navigation", { name: "Dashboard categories" });
    expect(navigation)
      .toHaveTextContent("Manual guide");
    expect(navigation)
      .toHaveTextContent("Knowledge Base");
    expect(navigation).not.toHaveTextContent("Current workspace outcome and selected artifact.");
    expect(screen.getByRole("heading", { name: "Overview" })).toBeVisible();
    expect(screen.getByLabelText("Overview live signal board")).toBeVisible();
    expect(screen.getByLabelText("Help with Overview live signal board")).toBeVisible();

    fireEvent.click(screen.getAllByLabelText("Help with Overview")[0]!);
    expect(screen.getAllByRole("note")[0]).toHaveTextContent(
      "Review the metric cards for the current values"
    );

    fireEvent.click(
      screen.getByRole("button", { name: /Architecture/ })
    );
    expect(onCategoryChange).toHaveBeenCalledWith("Architecture");

    fireEvent.click(screen.getByRole("button", { name: "Close dashboard" }));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("keeps output feedback in the AI & agents Dashboard group", () => {
    const onSubmitFeedback = vi.fn();
    const onCategoryChange = vi.fn();

    const { rerender } = render(
      <ProductIntelligenceDashboard
        activeCategory="Overview"
        feedback={{ artifactTitle: "aurora-field.p5.js", onSubmit: onSubmitFeedback }}
        model={buildModel()}
        onCategoryChange={onCategoryChange}
        onClose={vi.fn()}
      />
    );

    expect(screen.queryByLabelText("Output feedback")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /AI & agents/ }));
    expect(onCategoryChange).toHaveBeenCalledWith("Agents");
    rerender(
      <ProductIntelligenceDashboard
        activeCategory="Agents"
        feedback={{ artifactTitle: "aurora-field.p5.js", onSubmit: onSubmitFeedback }}
        model={buildModel()}
        onCategoryChange={onCategoryChange}
        onClose={vi.fn()}
      />
    );

    expect(screen.getByLabelText("Output feedback")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Mark output helpful" }));
    expect(onSubmitFeedback).toHaveBeenCalledWith("positive", "");
  });

  it("uses the same model for a compact Inspector category", () => {
    render(<ProductIntelligenceInspector category="Providers" model={buildModel()} />);

    expect(screen.getByRole("tabpanel", { name: "Providers inspector" }))
      .toHaveTextContent("Providers detailed product information");
    expect(screen.getByText("Providers value")).toBeVisible();
    expect(screen.getByLabelText("Help with Providers")).toBeVisible();
  });
});
