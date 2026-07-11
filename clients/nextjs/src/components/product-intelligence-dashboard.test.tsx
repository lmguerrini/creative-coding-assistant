import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  productIntelligenceCategories,
  type ProductIntelligenceModel
} from "@/lib/product-intelligence";
import {
  ProductIntelligenceDashboard,
  ProductIntelligenceInspector
} from "./product-intelligence-dashboard";

function buildModel(): ProductIntelligenceModel {
  return {
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
  it("renders every flat Dashboard category and switches the detailed surface", () => {
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

    expect(screen.getByRole("navigation", { name: "Dashboard categories" }))
      .toHaveTextContent("Product Bugs");
    expect(screen.getByRole("heading", { name: "Overview" })).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: /Workflow/ }));
    expect(onCategoryChange).toHaveBeenCalledWith("Workflow");

    fireEvent.click(screen.getByRole("button", { name: "Return to workspace" }));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("uses the same model for a compact Inspector category", () => {
    render(<ProductIntelligenceInspector category="Providers" model={buildModel()} />);

    expect(screen.getByRole("tabpanel", { name: "Providers inspector" }))
      .toHaveTextContent("Providers detailed product information");
    expect(screen.getByText("Providers value")).toBeVisible();
  });
});
