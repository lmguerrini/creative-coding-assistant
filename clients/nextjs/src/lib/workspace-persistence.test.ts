import { describe, expect, it, vi } from "vitest";
import { getLocalWorkspaceSnapshot } from "./assistant-client";
import {
  createWorkspacePersistenceClient,
  createWorkspaceSessionRecord,
  defaultWorkspacePreferences,
  defaultWorkspaceLayoutState,
  fingerprintWorkspaceSessionRecord,
  normalizeWorkspacePreferences,
  normalizeWorkspaceLayoutState,
  snapshotFromWorkspaceSessionRecord,
  type WorkspaceSessionRecord
} from "./workspace-persistence";

describe("workspace persistence client", () => {
  it("builds a typed session record from the workspace snapshot", () => {
    const snapshot = getLocalWorkspaceSnapshot();

    const record = createWorkspaceSessionRecord({
      activeArtifactId: "source-sketch",
      activeInspectorTab: "Code",
      previewArtifactId: "preview-manifest",
      previewOpen: true,
      snapshot
    });

    expect(record).toMatchObject({
      schemaVersion: 4,
      userId: "local-user",
      sessionId: "local-nextjs-session",
      projectId: "local-nextjs-workspace",
      activeArtifactId: "source-sketch",
      activeInspectorTab: "Code",
      previewOpen: true,
      previewArtifactId: "preview-manifest",
      layout: defaultWorkspaceLayoutState,
      preferences: defaultWorkspacePreferences,
      multimodal: {
        state: "empty",
        imageAttachments: []
      }
    });
    expect(record.messages).toEqual(snapshot.messages);
    expect(record.artifacts).toHaveLength(3);
    expect(record.updatedAt).toBeDefined();
  });

  it("normalizes layout preferences into safe persisted values", () => {
    expect(
      normalizeWorkspaceLayoutState({
        density: "compact",
        inspectorCollapsed: true,
        inspectorWidth: 999,
        previewHeight: 120
      })
    ).toEqual({
      density: "compact",
      inspectorCollapsed: true,
      inspectorWidth: 560,
      previewHeight: 220
    });
    expect(normalizeWorkspaceLayoutState()).toEqual(defaultWorkspaceLayoutState);
  });

  it("normalizes workspace preferences into safe persisted values", () => {
    expect(
      normalizeWorkspacePreferences({
        theme: "matrix",
        autoOpenPreview: false,
        showDebugPanels: false
      })
    ).toEqual({
      theme: "matrix",
      autoOpenPreview: false,
      showDebugPanels: false
    });
    expect(
      normalizeWorkspacePreferences({
        theme: "invalid" as never
      })
    ).toEqual(defaultWorkspacePreferences);
  });

  it("restores messages, active tab, artifact, and preview state", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const record = {
      ...createWorkspaceSessionRecord({
        activeArtifactId: "session-notes",
        activeInspectorTab: "Artifacts",
        previewArtifactId: "preview-manifest",
        previewOpen: true,
        snapshot
      }),
      messages: [
        {
          role: "user",
          time: "11:00",
          content: "Restore this."
        }
      ]
    } satisfies WorkspaceSessionRecord;

    const restored = snapshotFromWorkspaceSessionRecord(snapshot, record);

    expect(restored.messages).toEqual(record.messages);
    expect(
      restored.inspectorTabs.find((tab) => tab.label === "Artifacts")?.active
    ).toBe(true);
    expect(restored.preview.active).toBe(true);
    expect(restored.preview.collapsed).toBe(false);
  });

  it("loads a remote session and mirrors it to local storage", async () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const storage = new MemoryStorage();
    const record = createWorkspaceSessionRecord({
      activeArtifactId: "source-sketch",
      activeInspectorTab: "Overview",
      previewArtifactId: "source-sketch",
      previewOpen: false,
      snapshot
    });
    const fetchImpl = vi.fn(async () => new Response(JSON.stringify(record)));
    const client = createWorkspacePersistenceClient({ fetchImpl, storage });

    await expect(client.load()).resolves.toEqual({
      error: null,
      record,
      source: "remote"
    });

    expect(fetchImpl).toHaveBeenCalledWith(
      expect.stringContaining("/api/workspace/session?userId=local-user"),
      expect.objectContaining({ method: "GET" })
    );
    expect(storage.length).toBe(1);
  });

  it("saves remotely while keeping a local fallback copy", async () => {
    const storage = new MemoryStorage();
    const record = createWorkspaceSessionRecord({
      activeArtifactId: "source-sketch",
      activeInspectorTab: "Overview",
      previewArtifactId: "source-sketch",
      previewOpen: false,
      snapshot: getLocalWorkspaceSnapshot()
    });
    const fetchImpl = vi.fn(async () => new Response("{}"));
    const client = createWorkspacePersistenceClient({ fetchImpl, storage });

    await expect(client.save(record)).resolves.toEqual({
      error: null,
      target: "remote"
    });

    expect(fetchImpl).toHaveBeenCalledWith(
      "http://localhost:8000/api/workspace/session",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({ "Content-Type": "application/json" })
      })
    );
    expect(storage.length).toBe(1);
  });

  it("falls back to local session storage when the endpoint is unavailable", async () => {
    const storage = new MemoryStorage();
    const record = createWorkspaceSessionRecord({
      activeArtifactId: "source-sketch",
      activeInspectorTab: "Workflow",
      previewArtifactId: "preview-manifest",
      previewOpen: true,
      snapshot: getLocalWorkspaceSnapshot()
    });
    const offlineFetch = vi.fn(async () => {
      throw new Error("offline");
    });
    const client = createWorkspacePersistenceClient({
      fetchImpl: offlineFetch,
      storage
    });

    await expect(client.save(record)).resolves.toMatchObject({
      error: expect.objectContaining({
        category: "persistence",
        type: "session_save_unavailable"
      }),
      target: "local"
    });
    await expect(client.load()).resolves.toMatchObject({
      error: expect.objectContaining({
        category: "persistence",
        type: "session_restore_unavailable"
      }),
      record: expect.objectContaining({
        activeInspectorTab: "Workflow",
        previewOpen: true,
        layout: defaultWorkspaceLayoutState
      }),
      source: "local"
    });
  });

  it("fingerprints records without timestamp churn", () => {
    const record = createWorkspaceSessionRecord({
      activeArtifactId: "source-sketch",
      activeInspectorTab: "Overview",
      previewArtifactId: "source-sketch",
      previewOpen: false,
      snapshot: getLocalWorkspaceSnapshot()
    });

    expect(
      fingerprintWorkspaceSessionRecord({
        ...record,
        updatedAt: "later",
        snapshot: {
          ...record.snapshot,
          session: {
            ...record.snapshot.session,
            updatedAt: "later"
          }
        }
      })
    ).toBe(fingerprintWorkspaceSessionRecord(record));
  });
});

class MemoryStorage implements Storage {
  private readonly items = new Map<string, string>();

  get length() {
    return this.items.size;
  }

  clear() {
    this.items.clear();
  }

  getItem(key: string) {
    return this.items.get(key) ?? null;
  }

  key(index: number) {
    return Array.from(this.items.keys())[index] ?? null;
  }

  removeItem(key: string) {
    this.items.delete(key);
  }

  setItem(key: string, value: string) {
    this.items.set(key, value);
  }
}
