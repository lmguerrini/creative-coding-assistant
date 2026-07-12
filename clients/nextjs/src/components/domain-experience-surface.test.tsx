import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
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

const inventoryWithTwoSources: KnowledgeBaseInventory = {
  ...inventory,
  registeredSourceCount: 2,
  sources: [
    ...inventory.sources,
    {
      ...inventory.sources[0],
      id: "p5_reference",
      title: "p5.js Core Sketch Reference",
      url: "https://p5js.org/reference/"
    }
  ]
};

const syncedInventory: KnowledgeBaseInventory = {
  ...inventoryWithTwoSources,
  indexedChunkCount: 24,
  indexedSourceCount: 1,
  lastIndexedAt: "2026-07-12T02:21:00Z",
  sources: inventoryWithTwoSources.sources.map((source) => source.id === "three_docs"
    ? { ...source, chunkCount: 24, fingerprint: "updated-fingerprint", indexed: true, lastIndexedAt: "2026-07-12T02:21:00Z" }
    : { ...source, chunkCount: 0, fingerprint: null, indexed: false, lastIndexedAt: null }
  )
};

function jsonResponse(payload: Record<string, unknown>, ok = true) {
  return { json: async () => payload, ok };
}

describe("KnowledgeBaseInventorySurface", () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.spyOn(window, "confirm").mockReturnValue(true);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

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
      .toHaveTextContent("changedthree.js Documentation");
    expect(screen.getByRole("list", { name: "Knowledge Base action guide" }))
      .toHaveTextContent("Check for updatesCompares official content with local fingerprints. Read-only.");
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

  it("removes unavailable sources from the update selection after checking", async () => {
    const fetcher = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        status: "review_ready_with_unavailable_sources",
        detail: "The source could not be reached and was removed from the update selection.",
        sourceChanges: [
          {
            sourceId: "three_docs",
            changeStatus: "unavailable",
            detail: "The official source could not be reached."
          }
        ]
      })
    });
    vi.stubGlobal("fetch", fetcher);

    render(<KnowledgeBaseInventorySurface detailed inventory={inventory} />);

    fireEvent.click(screen.getByRole("button", { name: "Select all" }));
    fireEvent.click(screen.getByRole("button", { name: "Check for updates" }));

    expect(await screen.findByText("The official source could not be reached.")).toBeVisible();
    expect(
      screen.getByRole("checkbox", {
        name: "Select three.js Documentation for a Knowledge Base operation"
      })
    ).not.toBeChecked();
    expect(screen.getByRole("button", { name: "Check for updates" })).toBeDisabled();
  });

  it("shows animated, accessible progress while checking selected sources", async () => {
    let resolveFetch: (value: unknown) => void = () => undefined;
    const fetcher = vi.fn().mockImplementation(
      () => new Promise((resolve) => {
        resolveFetch = resolve;
      })
    );
    vi.stubGlobal("fetch", fetcher);

    render(<KnowledgeBaseInventorySurface detailed inventory={inventory} />);

    fireEvent.click(screen.getByRole("button", { name: "Select all" }));
    fireEvent.click(screen.getByRole("button", { name: "Check for updates" }));

    expect(screen.getByRole("button", { name: "Checking selected official sources" }))
      .toHaveTextContent("Checking sources");
    expect(screen.getByText("Checking 1 selected official source without changing the local index."))
      .toBeVisible();
    expect(screen.getByRole("region", { name: "Official Knowledge Base sources" }))
      .toHaveAttribute("aria-busy", "true");

    resolveFetch({
      ok: true,
      json: async () => ({
        status: "review_ready",
        detail: "The source check is ready for review.",
        sourceChanges: [{ sourceId: "three_docs", changeStatus: "new" }]
      })
    });

    expect(await screen.findByText("The source check is ready for review.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Check for updates" })).toBeEnabled();
  });

  it("runs Smart Update in the ordered check, update, rebuild, validate workflow", async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(jsonResponse({
        detail: "One changed source is ready for the selected update.",
        sourceChanges: [
          { sourceId: "three_docs", changeStatus: "changed" },
          { sourceId: "p5_reference", changeStatus: "unavailable" }
        ],
        status: "review_ready_with_unavailable_sources"
      }))
      .mockResolvedValueOnce(jsonResponse({ inventory: syncedInventory, status: "completed" }))
      .mockResolvedValueOnce(jsonResponse({ status: "completed" }))
      .mockResolvedValueOnce(jsonResponse({ detail: "The selected local index is valid.", status: "passed" }));
    vi.stubGlobal("fetch", fetcher);

    render(<KnowledgeBaseInventorySurface detailed inventory={inventoryWithTwoSources} />);
    fireEvent.click(screen.getByRole("button", { name: "Smart Update" }));

    expect(await screen.findByText("Smart Update complete")).toBeVisible();
    expect(screen.getByLabelText("Smart Update progress")).toHaveTextContent(
      /Check for updates.*Update affected.*Rebuild affected.*Validate index/s
    );
    expect(screen.getByText(/1 unavailable source was skipped; validation passed/i)).toBeVisible();
    expect(screen.getByText(/Last successful Smart Update:/)).toBeVisible();
    expect(
      screen.getByRole("checkbox", {
        name: "Select three.js Documentation for a Knowledge Base operation"
      }).closest("article")
    ).toHaveTextContent("24 chunks");
    expect(fetcher).toHaveBeenCalledTimes(4);

    expect(JSON.parse(fetcher.mock.calls[0][1].body)).toEqual({
      action: "check",
      confirmed: false,
      sourceIds: ["three_docs", "p5_reference"]
    });
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual({
      action: "update",
      confirmed: true,
      sourceIds: ["three_docs"]
    });
    expect(JSON.parse(fetcher.mock.calls[2][1].body)).toEqual({
      action: "rebuild",
      confirmed: true,
      sourceIds: ["three_docs"]
    });
    expect(JSON.parse(fetcher.mock.calls[3][1].body)).toEqual({
      action: "validate",
      confirmed: false,
      sourceIds: ["three_docs"]
    });
  });

  it("stops on a failed update, keeps the recovery guidance visible, and does not continue", async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(jsonResponse({
        sourceChanges: [{ sourceId: "three_docs", changeStatus: "changed" }],
        status: "review_ready"
      }))
      .mockResolvedValueOnce(jsonResponse({
        message: "The Knowledge Base update failed; the prior local index was restored."
      }, false));
    vi.stubGlobal("fetch", fetcher);

    render(<KnowledgeBaseInventorySurface detailed inventory={inventory} />);
    fireEvent.click(screen.getByRole("button", { name: "Smart Update" }));

    expect(await screen.findByText(
      "The Knowledge Base update failed; the prior local index was restored. Recovery: the later steps were not run; review the failed source and retry Smart Update."
    )).toBeVisible();
    expect(screen.getByText(/Recovery: the later steps were not run/i)).toBeVisible();
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("prevents repeated Smart Update clicks while the initial check is still running", async () => {
    let resolveCheck: (value: ReturnType<typeof jsonResponse>) => void = () => undefined;
    const fetcher = vi.fn()
      .mockImplementationOnce(() => new Promise<ReturnType<typeof jsonResponse>>((resolve) => {
        resolveCheck = resolve;
      }))
      .mockResolvedValueOnce(jsonResponse({ status: "passed" }));
    vi.stubGlobal("fetch", fetcher);

    render(<KnowledgeBaseInventorySurface detailed inventory={inventory} />);
    const smartUpdate = screen.getByRole("button", { name: "Smart Update" });
    fireEvent.click(smartUpdate);
    fireEvent.click(smartUpdate);

    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("button", { name: "Smart Update in progress" })).toBeDisabled();

    resolveCheck(jsonResponse({
      sourceChanges: [{ sourceId: "three_docs", changeStatus: "unchanged" }],
      status: "review_ready"
    }));

    expect(await screen.findByText("Smart Update complete")).toBeVisible();
    expect(fetcher).toHaveBeenCalledTimes(2);
  });
});
