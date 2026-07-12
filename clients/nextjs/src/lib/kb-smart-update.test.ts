import { describe, expect, it } from "vitest";
import {
  readKnowledgeBaseSmartUpdateSnapshot,
  writeKnowledgeBaseSmartUpdateSnapshot
} from "./kb-smart-update";

function createStorage(): Storage {
  const values = new Map<string, string>();
  return {
    clear: () => values.clear(),
    getItem: (key) => values.get(key) ?? null,
    key: (index) => [...values.keys()][index] ?? null,
    get length() {
      return values.size;
    },
    removeItem: (key) => values.delete(key),
    setItem: (key, value) => values.set(key, value)
  };
}

describe("Knowledge Base Smart Update persistence", () => {
  it("persists and restores the latest successful update, rebuild, and validation state", () => {
    const storage = createStorage();
    const snapshot = {
      affectedSourceCount: 2,
      completedAt: "2026-07-12T10:30:00.000Z",
      rebuiltSourceCount: 2,
      scopeSourceCount: 5,
      unavailableSourceCount: 1,
      updatedSourceCount: 2,
      validationStatus: "passed"
    };

    expect(writeKnowledgeBaseSmartUpdateSnapshot(snapshot, storage)).toBe(true);
    expect(readKnowledgeBaseSmartUpdateSnapshot(storage)).toEqual(snapshot);
  });

  it("does not restore malformed local data as a successful Smart Update", () => {
    const storage = createStorage();
    storage.setItem("cca.knowledge-base.smart-update.v1", "{not valid json");

    expect(readKnowledgeBaseSmartUpdateSnapshot(storage)).toBeNull();
  });
});
