import { describe, expect, it } from "vitest";
import type { CreativeCostRunRecord } from "./creative-cost-intelligence";
import {
  deleteSessionUsage,
  readSessionUsageSummaries,
  recordSessionUsageRun,
  renameSessionUsage
} from "./session-usage-ledger";

describe("session usage ledger", () => {
  it("keeps local token and cost totals separate by session and deduplicates runs", () => {
    const storage = new MemoryStorage();
    const first = run({ id: "run-a", totalTokens: 120, cost: 0.004 });
    const second = run({ id: "run-b", totalTokens: 80, cost: 0.002 });

    recordSessionUsageRun({ run: first, sessionId: "session-a", title: "A", userId: "user", storage });
    recordSessionUsageRun({ run: first, sessionId: "session-a", title: "A", userId: "user", storage });
    recordSessionUsageRun({ run: second, sessionId: "session-b", title: "B", userId: "user", storage });

    expect(readSessionUsageSummaries("user", storage)).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ sessionId: "session-a", runCount: 1, latestTokens: 120, latestCost: 0.004, totalTokens: 120, totalCost: 0.004 }),
        expect.objectContaining({ sessionId: "session-b", runCount: 1, totalTokens: 80, totalCost: 0.002 })
      ])
    );
  });

  it("renames and deletes a session together with its retained usage", () => {
    const storage = new MemoryStorage();
    recordSessionUsageRun({ run: run({ id: "run-a" }), sessionId: "session-a", title: "A", userId: "user", storage });
    recordSessionUsageRun({ run: run({ id: "run-b" }), sessionId: "session-b", title: "B", userId: "user", storage });

    expect(renameSessionUsage("user", "session-a", "Aurora", storage)).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ sessionId: "session-a", title: "Aurora" })
      ])
    );
    expect(deleteSessionUsage("user", "session-a", storage)).toEqual([
      expect.objectContaining({ sessionId: "session-b", totalTokens: 120 })
    ]);
    expect(readSessionUsageSummaries("user", storage)).toHaveLength(1);
  });

  it("reports the latest request separately from cumulative session totals", () => {
    const storage = new MemoryStorage();
    recordSessionUsageRun({ run: run({ id: "run-a", totalTokens: 120, cost: 0.004 }), sessionId: "session-a", title: "A", userId: "user", storage });
    const summaries = recordSessionUsageRun({ run: run({ id: "run-b", totalTokens: 80, cost: 0.002 }), sessionId: "session-a", title: "A", userId: "user", storage });

    expect(summaries).toEqual([
      expect.objectContaining({
        sessionId: "session-a",
        runCount: 2,
        latestTokens: 80,
        latestCost: 0.002,
        totalTokens: 200,
        totalCost: 0.006
      })
    ]);
  });
});

function run({ id, ...overrides }: Partial<CreativeCostRunRecord> & { id: string }): CreativeCostRunRecord {
  return {
    id,
    status: "complete",
    completedAt: "2026-07-12T00:00:00.000Z",
    kind: "generation",
    providerName: "openai",
    modelName: "gpt-5-mini",
    generationMode: "streaming",
    pricing: null,
    inputTokens: 50,
    outputTokens: 70,
    totalTokens: 120,
    cost: 0.004,
    currency: "USD",
    costSource: "pricing_metadata",
    durationMs: 500,
    retryCount: 0,
    fallbackCount: 0,
    retryCost: 0,
    fallbackCost: 0,
    artifactCount: 1,
    refinementCount: 0,
    critiqueCount: 0,
    reviewCount: 0,
    ...overrides
  };
}

class MemoryStorage implements Storage {
  #items = new Map<string, string>();
  get length() { return this.#items.size; }
  clear() { this.#items.clear(); }
  getItem(key: string) { return this.#items.get(key) ?? null; }
  key(index: number) { return [...this.#items.keys()][index] ?? null; }
  removeItem(key: string) { this.#items.delete(key); }
  setItem(key: string, value: string) { this.#items.set(key, value); }
}
