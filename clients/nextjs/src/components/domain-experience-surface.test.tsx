import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { KnowledgeBaseInventory } from "@/lib/domain-experience";
import { KnowledgeBaseInventorySurface } from "./domain-experience-surface";

const inventory: KnowledgeBaseInventory = {
  status: "available",
  detail: "The local Chroma index reports one official knowledge chunk.",
  registeredSourceCount: 1,
  registeredDomainCount: 1,
  indexedSourceCount: 1,
  indexedDomainCount: 1,
  indexedChunkCount: 1,
  lastIndexedAt: "2026-07-11T10:00:00Z",
  freshnessStatus: "local_index_timestamp",
  freshnessDetail: "The timestamp is local, not an upstream freshness claim.",
  updateStatus: "explicit_selected_source_actions",
  updateHint: "Explicit selected-source actions are available.",
  provenanceBoundary: "No source content is returned.",
  sources: [
    {
      id: "three_docs",
      title: "three.js Documentation",
      publisher: "three.js",
      url: "https://threejs.org/docs/",
      domain: "three_js",
      sourceType: "api_reference",
      priority: 1,
      tags: ["three"],
      indexed: true,
      chunkCount: 1,
      lastIndexedAt: "2026-07-11T10:00:00Z",
      fingerprint: "abcdef0123456789",
      health: "locally_indexed",
      freshnessLimitation: "The local timestamp does not establish upstream freshness.",
      provenance: "Approved official-source registry and local Chroma index."
    }
  ]
};

describe("KnowledgeBaseInventorySurface", () => {
  it("selects all official sources before an explicit check", async () => {
    const fetcher = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        status: "review_ready",
        detail: "Review the changed selected source before updating.",
        sourceChanges: [{ sourceId: "three_docs", changeStatus: "changed" }]
      })
    });
    vi.stubGlobal("fetch", fetcher);

    render(<KnowledgeBaseInventorySurface detailed inventory={inventory} />);

    const check = screen.getByRole("button", { name: "Check for updates" });
    expect(check).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Select all" }));
    expect(
      screen.getByRole("checkbox", {
        name: "Select three.js Documentation for a Knowledge Base operation"
      })
    ).toBeChecked();
    expect(screen.getByRole("button", { name: "Clear selection" })).toBeVisible();
    expect(check).toBeEnabled();
    fireEvent.click(check);

    expect(await screen.findByRole("list", { name: "Source change summary" }))
      .toHaveTextContent("changedthree_docs");
    expect(fetcher).toHaveBeenCalledWith(
      "http://localhost:8000/api/knowledge-base",
      expect.objectContaining({
        body: JSON.stringify({
          action: "check",
          confirmed: false,
          sourceIds: ["three_docs"]
        }),
        method: "POST"
      })
    );
  });
});
