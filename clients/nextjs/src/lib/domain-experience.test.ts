import { describe, expect, it, vi } from "vitest";
import {
  fetchDomainExperienceCatalog,
  formatDomainDeliveryKind,
  getDomainExperienceRecord
} from "./domain-experience";

describe("domain experience catalog", () => {
  it("normalizes the public domain registry and separate KB inventory", async () => {
    const fetcher = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        schemaVersion: "domain-experience.v1",
        domains: [
          {
            domain: "p5_js",
            display_name: "p5.js",
            aliases: ["p5.js"],
            intent_triggers: ["p5", "sketch"],
            knowledge_source_ids: ["p5_reference"],
            workflow_compatibility: ["retrieve", "generate", "preview", "export"],
            artifact_types: ["global-mode JavaScript sketch"],
            filename_extensions: [".p5.js"],
            delivery_kind: "browser_preview",
            live_preview: true,
            runtime_requirements: ["Bounded global-mode source."],
            validation_status: "validated_browser_contract",
            fallback: "Open Code.",
            public_claim_boundary: "Bounded browser preview only.",
            demo_eligible: true,
            knowledge: {
              registeredSourceCount: 1,
              indexedSourceCount: 1,
              indexedChunkCount: 12
            }
          }
        ],
        knowledgeBase: {
          status: "available",
          detail: "The local index reports official chunks.",
          registeredSourceCount: 57,
          registeredDomainCount: 43,
          indexedSourceCount: 12,
          indexedDomainCount: 5,
          indexedChunkCount: 280,
          lastIndexedAt: "2026-07-11T10:00:00Z",
          freshnessStatus: "local_index_timestamp",
          freshnessDetail: "The local timestamp is not upstream freshness.",
          updateStatus: "manual_sync_only",
          updateHint: "Use local sync.",
          provenanceBoundary: "No source text is returned."
        }
      })
    });

    const catalog = await fetchDomainExperienceCatalog(fetcher as typeof fetch);

    expect(catalog.state).toBe("available");
    expect(catalog.knowledgeBase.indexedChunkCount).toBe(280);
    expect(catalog.knowledgeBase.freshnessStatus).toBe("local_index_timestamp");
    expect(catalog.domains).toHaveLength(1);
    expect(getDomainExperienceRecord(catalog, "p5_js")).toMatchObject({
      displayName: "p5.js",
      deliveryKind: "browser_preview",
      knowledge: { indexedChunkCount: 12 }
    });
    expect(formatDomainDeliveryKind("external_handoff")).toBe("External-tool handoff");
  });

  it("returns a bounded unavailable model when the inventory endpoint fails", async () => {
    const catalog = await fetchDomainExperienceCatalog(
      vi.fn().mockResolvedValue({ ok: false, status: 503 }) as unknown as typeof fetch
    );

    expect(catalog).toMatchObject({
      state: "unavailable",
      domains: [],
      knowledgeBase: { status: "unavailable" }
    });
  });
});
