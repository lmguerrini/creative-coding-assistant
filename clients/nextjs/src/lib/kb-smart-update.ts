export type KnowledgeBaseSmartUpdateSnapshot = {
  affectedSourceCount: number;
  completedAt: string;
  rebuiltSourceCount: number;
  scopeSourceCount: number;
  unavailableSourceCount: number;
  updatedSourceCount: number;
  validationStatus: string;
};

const SMART_UPDATE_STORAGE_KEY = "cca.knowledge-base.smart-update.v1";

export function readKnowledgeBaseSmartUpdateSnapshot(
  storage: Storage | null | undefined = browserStorage()
): KnowledgeBaseSmartUpdateSnapshot | null {
  if (!storage) {
    return null;
  }

  try {
    const value: unknown = JSON.parse(storage.getItem(SMART_UPDATE_STORAGE_KEY) ?? "null");
    return isKnowledgeBaseSmartUpdateSnapshot(value) ? value : null;
  } catch {
    return null;
  }
}

export function writeKnowledgeBaseSmartUpdateSnapshot(
  snapshot: KnowledgeBaseSmartUpdateSnapshot,
  storage: Storage | null | undefined = browserStorage()
): boolean {
  if (!storage || !isKnowledgeBaseSmartUpdateSnapshot(snapshot)) {
    return false;
  }

  try {
    storage.setItem(SMART_UPDATE_STORAGE_KEY, JSON.stringify(snapshot));
    return true;
  } catch {
    return false;
  }
}

function browserStorage(): Storage | null {
  if (typeof window === "undefined") {
    return null;
  }

  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function isKnowledgeBaseSmartUpdateSnapshot(
  value: unknown
): value is KnowledgeBaseSmartUpdateSnapshot {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }

  const record = value as Record<string, unknown>;
  return (
    typeof record.completedAt === "string" &&
    typeof record.validationStatus === "string" &&
    isCount(record.scopeSourceCount) &&
    isCount(record.affectedSourceCount) &&
    isCount(record.updatedSourceCount) &&
    isCount(record.rebuiltSourceCount) &&
    isCount(record.unavailableSourceCount)
  );
}

function isCount(value: unknown) {
  return typeof value === "number" && Number.isInteger(value) && value >= 0;
}
