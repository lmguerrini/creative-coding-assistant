import { describe, expect, it } from "vitest";
import { getInitialWorkspaceSnapshot, getLocalWorkspaceSnapshot } from "./assistant-client";
import { buildRetrievalRuntimeModel } from "./retrieval-runtime";
import { buildRetrievalSourceExplorerModel } from "./retrieval-source-explorer";

describe("retrieval source explorer", () => {
  it("groups globally ranked chunks and identifies the top contributor", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const explorer = buildRetrievalSourceExplorerModel(
      buildRetrievalRuntimeModel(snapshot.retrieval, [])
    );

    expect(explorer).toMatchObject({
      selectedSourceCount: 2,
      ignoredSourceCount: 0,
      overviewLabel: "2 selected sources · No ignored sources reported",
      contributionLabel:
        "WebGPU API contributed most with 2/3 context chunks."
    });
    expect(explorer.sources[0]).toMatchObject({
      sourceId: "webgpu_mdn_api",
      contextStatusLabel: "Selected for context",
      chunkCountLabel: "2 retrieved chunks",
      rankRangeLabel: "Ranks #1–#2",
      coverageLabel: "2/3 context chunks",
      isTopContributor: true
    });
    expect(explorer.sources[0]?.chunks).toEqual([
      expect.objectContaining({
        rank: 1,
        confidenceLabel: "High confidence",
        contextStatusLabel: "Used in context"
      }),
      expect.objectContaining({
        rank: 2,
        confidenceLabel: "Medium confidence",
        contextStatusLabel: "Used in context"
      })
    ]);
  });

  it("represents explicitly ignored sources and chunks", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const retrieval = {
      ...snapshot.retrieval,
      sources: snapshot.retrieval.sources.map((source, index) =>
        index === 1
          ? {
              ...source,
              selectedForContext: false,
              chunks: source.chunks.map((chunk) => ({
                ...chunk,
                usedInContext: false
              }))
            }
          : source
      )
    };
    const runtime = buildRetrievalRuntimeModel(retrieval, []);
    const explorer = buildRetrievalSourceExplorerModel(runtime);

    expect(runtime.summary.usedChunkLabel).toBe("2 chunks used");
    expect(explorer).toMatchObject({
      selectedSourceCount: 1,
      ignoredSourceCount: 1,
      overviewLabel: "1 selected source · 1 ignored source"
    });
    expect(explorer.sources[1]).toMatchObject({
      contextStatusLabel: "Not selected",
      rankRangeLabel: "Rank #3",
      coverageLabel: "0/2 context chunks",
      coverageDetail:
        "This source did not contribute a chunk to the final retrieval context.",
      contextReason:
        "Retrieved as a candidate source but not included in the final context; no exclusion reason was recorded."
    });
    expect(explorer.sources[1]?.chunks[0]).toMatchObject({
      contextStatusLabel: "Not used",
      usedInContext: false
    });
  });

  it("uses safe fallback labels for legacy source metadata", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const legacyRetrieval = {
      ...snapshot.retrieval,
      sources: snapshot.retrieval.sources.map((source) => ({
        ...source,
        selectedForContext: undefined,
        chunks: source.chunks.map((chunk) => ({
          ...chunk,
          rank: undefined,
          score: null,
          selectionReason: undefined,
          usedInContext: undefined
        }))
      }))
    };
    const explorer = buildRetrievalSourceExplorerModel(
      buildRetrievalRuntimeModel(legacyRetrieval, [])
    );

    expect(explorer.selectedSourceCount).toBe(2);
    expect(explorer.sources[0]?.rankRangeLabel).toBe("Ranks #1–#3");
    expect(explorer.sources[0]?.chunks[0]).toMatchObject({
      confidenceLabel: "Confidence unknown",
      contextStatusLabel: "Used in context"
    });
    expect(explorer.sources[0]?.chunks[0]?.selectionReason).toContain(
      "Ranked #1"
    );
  });

  it("keeps the first-run explorer model empty", () => {
    const snapshot = getInitialWorkspaceSnapshot();
    const explorer = buildRetrievalSourceExplorerModel(
      buildRetrievalRuntimeModel(snapshot.retrieval, [])
    );

    expect(explorer.sources).toEqual([]);
    expect(explorer.overviewLabel).toBe(
      "0 selected sources · No ignored sources reported"
    );
    expect(explorer.contributionLabel).toBe(
      "No source contribution is available yet."
    );
  });
});
