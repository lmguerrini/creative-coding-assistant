import type { CreativeCostRunRecord } from "./creative-cost-intelligence";

const storageKey = "cca.session-usage-ledger.v1";

type StoredUsageRun = Pick<
  CreativeCostRunRecord,
  | "id"
  | "completedAt"
  | "cost"
  | "currency"
  | "inputTokens"
  | "outputTokens"
  | "totalTokens"
>;

type StoredSessionUsage = {
  sessionId: string;
  title: string;
  runs: StoredUsageRun[];
  updatedAt: string | null;
};

type UsageLedger = {
  version: 1;
  users: Record<string, StoredSessionUsage[]>;
};

export type SessionUsageSummary = {
  sessionId: string;
  runCount: number;
  knownTokenRunCount: number;
  totalTokens: number | null;
  knownCostRunCount: number;
  totalCost: number | null;
  currency: string;
  updatedAt: string | null;
};

export function readSessionUsageSummaries(
  userId: string,
  storage: Storage | null = resolveStorage()
): SessionUsageSummary[] {
  return readLedger(storage).users[userId]?.map(summarizeSessionUsage) ?? [];
}

export function recordSessionUsageRun({
  run,
  sessionId,
  title,
  userId,
  storage = resolveStorage()
}: {
  run: CreativeCostRunRecord;
  sessionId: string;
  title: string;
  userId: string;
  storage?: Storage | null;
}): SessionUsageSummary[] {
  const ledger = readLedger(storage);
  const current = ledger.users[userId] ?? [];
  const existing = current.find((entry) => entry.sessionId === sessionId);
  const nextRun: StoredUsageRun = {
    id: run.id,
    completedAt: run.completedAt,
    cost: run.cost,
    currency: run.currency,
    inputTokens: run.inputTokens,
    outputTokens: run.outputTokens,
    totalTokens: run.totalTokens
  };
  const runs = existing?.runs.some((entry) => entry.id === run.id)
    ? existing.runs
    : [...(existing?.runs ?? []), nextRun].slice(-80);
  const nextEntry: StoredSessionUsage = {
    sessionId,
    title,
    runs,
    updatedAt: run.completedAt ?? new Date().toISOString()
  };

  ledger.users[userId] = existing
    ? current.map((entry) => (entry.sessionId === sessionId ? nextEntry : entry))
    : [nextEntry, ...current];
  writeLedger(ledger, storage);
  return ledger.users[userId].map(summarizeSessionUsage);
}

function summarizeSessionUsage(entry: StoredSessionUsage): SessionUsageSummary {
  const knownTokenRuns = entry.runs.filter((run) => run.totalTokens != null);
  const knownCostRuns = entry.runs.filter((run) => run.cost != null);
  const currencies = new Set(knownCostRuns.map((run) => run.currency));

  return {
    sessionId: entry.sessionId,
    runCount: entry.runs.length,
    knownTokenRunCount: knownTokenRuns.length,
    totalTokens:
      knownTokenRuns.length > 0
        ? knownTokenRuns.reduce((total, run) => total + (run.totalTokens ?? 0), 0)
        : null,
    knownCostRunCount: knownCostRuns.length,
    totalCost:
      knownCostRuns.length > 0 && currencies.size <= 1
        ? knownCostRuns.reduce((total, run) => total + (run.cost ?? 0), 0)
        : null,
    currency: knownCostRuns[0]?.currency ?? "USD",
    updatedAt: entry.updatedAt
  };
}

function readLedger(storage: Storage | null): UsageLedger {
  if (!storage) return { version: 1, users: {} };
  try {
    const raw = storage.getItem(storageKey);
    if (!raw) return { version: 1, users: {} };
    const value: unknown = JSON.parse(raw);
    if (!isLedger(value)) return { version: 1, users: {} };
    return value;
  } catch {
    return { version: 1, users: {} };
  }
}

function writeLedger(value: UsageLedger, storage: Storage | null) {
  if (!storage) return;
  try {
    storage.setItem(storageKey, JSON.stringify(value));
  } catch {
    // Local history is optional; the current run still remains visible in telemetry.
  }
}

function resolveStorage() {
  try {
    return typeof window === "undefined" ? null : window.localStorage;
  } catch {
    return null;
  }
}

function isLedger(value: unknown): value is UsageLedger {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const record = value as Record<string, unknown>;
  return record.version === 1 && record.users != null && typeof record.users === "object";
}
