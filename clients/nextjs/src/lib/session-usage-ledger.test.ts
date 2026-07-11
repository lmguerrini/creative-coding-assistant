import { describe, expect, it } from "vitest";
import type { CreativeCostRunRecord } from "./creative-cost-intelligence";
import {
  readSessionUsageSummaries,
  recordSessionUsageRun
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
        expect.objectContaining({ sessionId: "session-a", runCount: 1, totalTokens: 120, totalCost: 0.004 }),
        expect.objectContaining({ sessionId: "session-b", runCount: 1, totalTokens: 80, totalCost: 0.002 })
      ])
    );
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
