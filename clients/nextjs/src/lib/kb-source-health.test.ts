import { describe, expect, it } from "vitest";
import {
  getLocalWorkspaceSnapshot,
  type RetrievalSourceHealthMetadata
} from "./assistant-client";
import { buildKbSourceHealthDashboardModel } from "./kb-source-health";
import { buildRetrievalRuntimeModel } from "./retrieval-runtime";

describe("knowledge base source health", () => {
  it("summarizes reported source health, indexed chunks, ownership, and sync recency", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const model = buildKbSourceHealthDashboardModel(
      buildRetrievalRuntimeModel(snapshot.retrieval, [])
    );

    expect(model).toMatchObject({
      status: "stale",
      statusLabel: "Stale",
      sourceCount: 2,
      healthySourceCount: 1,
      attentionSourceCount: 1,
      availableSourceCount: 2,
      availabilityLabel: "2/2 sources available",
      indexedChunkCount: 280,
      indexedChunkLabel: "280 indexed chunks",
      domainOwnerCount: 2,
      domainOwnerLabel: "2 domain owners"
    });
    expect(model.latestSyncAttemptLabel).toContain("20d ago");
    expect(model.sources[0]).toMatchObject({
      status: "healthy",
      statusLabel: "Healthy",
      availabilityLabel: "Available",
      domainOwner: "Web platform / MDN",
      indexedChunkLabel: "184 indexed chunks",
      syncOutcomeLabel: "Succeeded",
      coverageLabel: "2/3 context chunks"
    });
    expect(model.sources[1]).toMatchObject({
      status: "stale",
      freshnessLabel: "Stale",
      indexedChunkLabel: "96 indexed chunks"
    });
  });

  it.each([
    [
      "healthy",
      {
        availability: "available",
        freshnessStatus: "fresh",
        status: "healthy",
        syncOutcome: "succeeded"
      }
    ],
    [
      "warning",
      {
        availability: "available",
        freshnessStatus: "fresh",
        status: "warning",
        syncOutcome: "succeeded"
      }
    ],
    [
      "stale",
      {
        availability: "available",
        freshnessStatus: "stale",
        status: "stale",
        syncOutcome: "succeeded"
      }
    ],
    [
      "failed",
      {
        availability: "unavailable",
        freshnessStatus: "unknown",
        status: "failed",
        syncOutcome: "failed"
      }
    ],
    [
      "unknown",
      {
        availability: "unknown",
        freshnessStatus: "unknown",
        status: "unknown",
        syncOutcome: "unknown"
      }
    ]
  ] as const)("supports the %s source health state", (expectedStatus, health) => {
    const model = buildDashboardWithHealth(health);

    expect(model.status).toBe(expectedStatus);
    expect(model.sources[0]?.status).toBe(expectedStatus);
  });

  it("degrades legacy sources to sensible defaults without inventing sync metrics", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const legacyRetrieval = {
      ...snapshot.retrieval,
      sources: snapshot.retrieval.sources.map(({ health: _health, ...source }) => source)
    };
    const model = buildKbSourceHealthDashboardModel(
      buildRetrievalRuntimeModel(legacyRetrieval, [])
    );

    expect(model.status).toBe("stale");
    expect(model.indexedChunkCount).toBeNull();
    expect(model.indexedChunkLabel).toBe("Indexed total unavailable");
    expect(model.latestSyncAttemptLabel).toBe("No sync attempt reported");
    expect(model.sources[0]).toMatchObject({
      status: "warning",
      availabilityLabel: "Available",
      domainOwner: "MDN",
      indexedChunkLabel: "Not reported",
      syncOutcomeLabel: "Unknown",
      lastSuccessfulSyncLabel: "Not reported",
      metadataAvailable: false
    });
    expect(model.sources[0]?.statusDetail).toContain("legacy session");
  });

  it("keeps first-run source health empty and unknown", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const runtime = buildRetrievalRuntimeModel(
      {
        ...snapshot.retrieval,
        sources: [],
        state: "empty"
      },
      []
    );
    const model = buildKbSourceHealthDashboardModel(runtime);

    expect(model).toMatchObject({
      status: "unknown",
      sourceCount: 0,
      availabilityLabel: "No source availability reported",
      indexedChunkLabel: "Indexed total unavailable"
    });
    expect(model.sources).toEqual([]);
  });
});

function buildDashboardWithHealth(
  health: RetrievalSourceHealthMetadata
) {
  const snapshot = getLocalWorkspaceSnapshot();
  const source = snapshot.retrieval.sources[0]!;
  const runtime = buildRetrievalRuntimeModel(
    {
      ...snapshot.retrieval,
      sources: [
        {
          ...source,
          freshness: health.freshnessStatus ?? "unknown",
          health: {
            ...health,
            checkedAt: "2026-06-09T08:30:00Z"
          }
        }
      ]
    },
    []
  );

  return buildKbSourceHealthDashboardModel(runtime);
}
