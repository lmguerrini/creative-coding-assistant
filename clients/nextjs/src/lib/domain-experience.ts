export type DomainDeliveryKind =
  | "browser_preview"
  | "code_export"
  | "external_handoff";

export type DomainKnowledgeCoverage = {
  registeredSourceCount: number;
  indexedSourceCount: number;
  indexedChunkCount: number;
};

export type DomainExperienceRecord = {
  id: string;
  displayName: string;
  aliases: string[];
  intentTriggers: string[];
  knowledgeSourceIds: string[];
  workflowCompatibility: string[];
  artifactTypes: string[];
  filenameExtensions: string[];
  deliveryKind: DomainDeliveryKind;
  livePreview: boolean;
  runtimeRequirements: string[];
  validationStatus: string;
  fallback: string;
  publicClaimBoundary: string;
  demoEligible: boolean;
  knowledge: DomainKnowledgeCoverage;
};

export type KnowledgeBaseInventory = {
  status: "available" | "empty" | "not_initialized" | "unavailable";
  detail: string;
  registeredSourceCount: number;
  registeredDomainCount: number;
  indexedSourceCount: number;
  indexedDomainCount: number;
  indexedChunkCount: number;
  lastIndexedAt: string | null;
  freshnessStatus: "local_index_timestamp" | "not_reported";
  freshnessDetail: string;
  updateStatus: string;
  updateHint: string;
  provenanceBoundary: string;
  sources: KnowledgeBaseSource[];
};

export type KnowledgeBaseSource = {
  id: string;
  title: string;
  publisher: string;
  url: string;
  domain: string;
  sourceType: string;
  priority: number;
  tags: string[];
  indexed: boolean;
  chunkCount: number;
  lastIndexedAt: string | null;
  fingerprint: string | null;
  health: string;
  freshnessLimitation: string;
  provenance: string;
};

export type DomainExperienceCatalog = {
  state: "loading" | "available" | "unavailable";
  schemaVersion: string | null;
  domains: DomainExperienceRecord[];
  knowledgeBase: KnowledgeBaseInventory;
  error: string | null;
};

type DomainExperiencePayload = {
  schemaVersion?: unknown;
  domains?: unknown;
  knowledgeBase?: unknown;
};

const defaultKnowledgeBase: KnowledgeBaseInventory = {
  status: "unavailable",
  detail: "Knowledge Base inventory has not been loaded from the local backend.",
  registeredSourceCount: 0,
  registeredDomainCount: 0,
  indexedSourceCount: 0,
  indexedDomainCount: 0,
  indexedChunkCount: 0,
  lastIndexedAt: null,
  freshnessStatus: "not_reported",
  freshnessDetail:
    "No local index timestamp is loaded, so upstream-source freshness is not reported.",
  updateStatus: "not_loaded",
  updateHint: "Start the local backend to inspect registered and indexed KB coverage.",
  provenanceBoundary:
    "This screen does not infer inventory from the current retrieval run.",
  sources: []
};

export const loadingDomainExperienceCatalog: DomainExperienceCatalog = {
  state: "loading",
  schemaVersion: null,
  domains: [],
  knowledgeBase: defaultKnowledgeBase,
  error: null
};

export async function fetchDomainExperienceCatalog(
  fetcher: typeof fetch = fetch
): Promise<DomainExperienceCatalog> {
  try {
    const response = await fetcher(getDomainExperienceEndpoint(), {
      headers: { Accept: "application/json" },
      method: "GET"
    });
    if (!response.ok) {
      throw new Error(`The domain inventory request returned ${response.status}.`);
    }
    return normalizeDomainExperiencePayload(await response.json());
  } catch (error) {
    return {
      ...loadingDomainExperienceCatalog,
      state: "unavailable",
      error:
        error instanceof Error
          ? error.message
          : "The domain inventory could not be loaded."
    };
  }
}

export function getDomainExperienceEndpoint() {
  return (
    process.env.NEXT_PUBLIC_DOMAIN_EXPERIENCE_URL ??
    "http://localhost:8000/api/domain-experience"
  );
}

export function getDomainExperienceRecord(
  catalog: DomainExperienceCatalog,
  domain: string | null | undefined
) {
  if (!domain) {
    return null;
  }
  return catalog.domains.find((record) => record.id === domain) ?? null;
}

export function formatDomainDeliveryKind(kind: DomainDeliveryKind) {
  switch (kind) {
    case "browser_preview":
      return "Live browser preview";
    case "external_handoff":
      return "External-tool handoff";
    default:
      return "Code/export";
  }
}

function normalizeDomainExperiencePayload(value: unknown): DomainExperienceCatalog {
  const payload = asRecord(value) as DomainExperiencePayload | null;
  const domains = Array.isArray(payload?.domains)
    ? payload.domains.map(normalizeDomainRecord).filter(isDomainRecord)
    : [];
  const knowledgeBase = normalizeKnowledgeBase(payload?.knowledgeBase);

  return {
    state: "available",
    schemaVersion: typeof payload?.schemaVersion === "string" ? payload.schemaVersion : null,
    domains,
    knowledgeBase,
    error: null
  };
}

function normalizeDomainRecord(value: unknown): DomainExperienceRecord | null {
  const record = asRecord(value);
  const id = readString(record?.domain);
  const displayName = readString(record?.display_name);
  const deliveryKind = readDeliveryKind(record?.delivery_kind);
  if (!id || !displayName || !deliveryKind) {
    return null;
  }

  return {
    id,
    displayName,
    aliases: readStrings(record?.aliases),
    intentTriggers: readStrings(record?.intent_triggers),
    knowledgeSourceIds: readStrings(record?.knowledge_source_ids),
    workflowCompatibility: readStrings(record?.workflow_compatibility),
    artifactTypes: readStrings(record?.artifact_types),
    filenameExtensions: readStrings(record?.filename_extensions),
    deliveryKind,
    livePreview: record?.live_preview === true,
    runtimeRequirements: readStrings(record?.runtime_requirements),
    validationStatus: readString(record?.validation_status) ?? "not_published",
    fallback: readString(record?.fallback) ?? "No fallback was published.",
    publicClaimBoundary:
      readString(record?.public_claim_boundary) ?? "No public claim boundary was published.",
    demoEligible: record?.demo_eligible === true,
    knowledge: normalizeDomainKnowledge(record?.knowledge)
  };
}

function normalizeKnowledgeBase(value: unknown): KnowledgeBaseInventory {
  const knowledge = asRecord(value);
  const status = readKnowledgeStatus(knowledge?.status);
  return {
    status,
    detail: readString(knowledge?.detail) ?? defaultKnowledgeBase.detail,
    registeredSourceCount: readCount(knowledge?.registeredSourceCount),
    registeredDomainCount: readCount(knowledge?.registeredDomainCount),
    indexedSourceCount: readCount(knowledge?.indexedSourceCount),
    indexedDomainCount: readCount(knowledge?.indexedDomainCount),
    indexedChunkCount: readCount(knowledge?.indexedChunkCount),
    lastIndexedAt: readString(knowledge?.lastIndexedAt),
    freshnessStatus: readFreshnessStatus(knowledge?.freshnessStatus),
    freshnessDetail: readString(knowledge?.freshnessDetail) ?? defaultKnowledgeBase.freshnessDetail,
    updateStatus: readString(knowledge?.updateStatus) ?? defaultKnowledgeBase.updateStatus,
    updateHint: readString(knowledge?.updateHint) ?? defaultKnowledgeBase.updateHint,
    provenanceBoundary:
      readString(knowledge?.provenanceBoundary) ?? defaultKnowledgeBase.provenanceBoundary,
    sources: readKnowledgeBaseSources(knowledge?.sources)
  };
}

function readKnowledgeBaseSources(value: unknown): KnowledgeBaseSource[] {
  return Array.isArray(value)
    ? value.map(normalizeKnowledgeBaseSource).filter(isKnowledgeBaseSource)
    : [];
}

function normalizeKnowledgeBaseSource(value: unknown): KnowledgeBaseSource | null {
  const source = asRecord(value);
  const id = readString(source?.id);
  const title = readString(source?.title);
  const publisher = readString(source?.publisher);
  const url = readString(source?.url);
  const domain = readString(source?.domain);
  const sourceType = readString(source?.sourceType);
  const health = readString(source?.health);
  const freshnessLimitation = readString(source?.freshnessLimitation);
  const provenance = readString(source?.provenance);
  if (
    !id ||
    !title ||
    !publisher ||
    !url ||
    !domain ||
    !sourceType ||
    !health ||
    !freshnessLimitation ||
    !provenance
  ) {
    return null;
  }
  return {
    id,
    title,
    publisher,
    url,
    domain,
    sourceType,
    priority: readCount(source?.priority),
    tags: readStrings(source?.tags),
    indexed: source?.indexed === true,
    chunkCount: readCount(source?.chunkCount),
    lastIndexedAt: readString(source?.lastIndexedAt),
    fingerprint: readString(source?.fingerprint),
    health,
    freshnessLimitation,
    provenance
  };
}

function isKnowledgeBaseSource(
  value: KnowledgeBaseSource | null
): value is KnowledgeBaseSource {
  return value !== null;
}

function normalizeDomainKnowledge(value: unknown): DomainKnowledgeCoverage {
  const knowledge = asRecord(value);
  return {
    registeredSourceCount: readCount(knowledge?.registeredSourceCount),
    indexedSourceCount: readCount(knowledge?.indexedSourceCount),
    indexedChunkCount: readCount(knowledge?.indexedChunkCount)
  };
}

function isDomainRecord(value: DomainExperienceRecord | null): value is DomainExperienceRecord {
  return value !== null;
}

function readDeliveryKind(value: unknown): DomainDeliveryKind | null {
  return value === "browser_preview" ||
    value === "code_export" ||
    value === "external_handoff"
    ? value
    : null;
}

function readKnowledgeStatus(value: unknown): KnowledgeBaseInventory["status"] {
  return value === "available" ||
    value === "empty" ||
    value === "not_initialized" ||
    value === "unavailable"
    ? value
    : "unavailable";
}

function readFreshnessStatus(value: unknown): KnowledgeBaseInventory["freshnessStatus"] {
  return value === "local_index_timestamp" || value === "not_reported"
    ? value
    : "not_reported";
}

function readStrings(value: unknown) {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string" && Boolean(item))
    : [];
}

function readString(value: unknown) {
  return typeof value === "string" && value.trim() ? value : null;
}

function readCount(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : 0;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}
