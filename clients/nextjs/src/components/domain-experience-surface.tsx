"use client";

import { useState } from "react";
import {
  formatDomainDeliveryKind,
  getDomainExperienceRecord,
  type DomainExperienceCatalog,
  type DomainExperienceRecord,
  type KnowledgeBaseInventory
} from "@/lib/domain-experience";

export function DomainExperienceSurface({
  activeDomainId,
  catalog,
  detailed = false,
  includeKnowledgeBase = true
}: {
  activeDomainId?: string | null;
  catalog: DomainExperienceCatalog;
  detailed?: boolean;
  includeKnowledgeBase?: boolean;
}) {
  if (catalog.state === "loading") {
    return (
      <section aria-label="Domain experience" className="domainExperienceSurface" role="status">
        <strong>Loading domain contracts</strong>
        <p>Reading the local product registry and Knowledge Base inventory.</p>
      </section>
    );
  }

  if (catalog.state === "unavailable") {
    return (
      <section aria-label="Domain experience" className="domainExperienceSurface" role="status">
        <strong>Domain inventory unavailable</strong>
        <p>{catalog.error ?? catalog.knowledgeBase.detail}</p>
        <p>{catalog.knowledgeBase.provenanceBoundary}</p>
      </section>
    );
  }

  const activeDomain = getDomainExperienceRecord(catalog, activeDomainId);
  const records = detailed
    ? catalog.domains
    : activeDomain
      ? [activeDomain]
      : catalog.domains.filter((record) => record.livePreview).slice(0, 3);
  const groups = detailed
    ? [
        {
          id: "live",
          label: "Live in this browser",
          detail: "Active browser-preview domains. These are the only cards that claim a live in-workspace runtime.",
          records: records.filter((record) => record.livePreview)
        },
        {
          id: "export",
          label: "Source export",
          detail: "Code can be generated or exported; an interactive browser runtime is not claimed.",
          records: records.filter((record) => record.deliveryKind === "code_export")
        },
        {
          id: "handoff",
          label: "External-tool handoff",
          detail: "A documented handoff for the named tool, separate from browser execution.",
          records: records.filter((record) => record.deliveryKind === "external_handoff")
        }
      ].filter((group) => group.records.length > 0)
    : [];

  return (
    <section aria-label="Domain experience" className="domainExperienceSurface">
      {includeKnowledgeBase ? (
        <KnowledgeBaseInventorySurface inventory={catalog.knowledgeBase} detailed={detailed} />
      ) : null}
      <header className="domainExperienceHeader">
        <div>
          <span>{detailed ? "Domain registry" : "Domain contract"}</span>
          <strong>
            {activeDomain
              ? activeDomain.displayName
              : `${catalog.domains.length} registered creative domains`}
          </strong>
          <p>
            {detailed
              ? "Each card distinguishes browser preview, source export, and external-tool handoff."
              : "The current artifact’s delivery boundary is kept separate from retrieval for this run."}
          </p>
        </div>
      </header>
      {detailed ? (
        <div className="domainExperienceGroups">
          {groups.map((group) => (
            <section aria-label={group.label} key={group.id}>
              <header>
                <div>
                  <strong>{group.label}</strong>
                  <p>{group.detail}</p>
                </div>
                <span>{group.records.length}</span>
              </header>
              <div className="domainExperienceGrid" role="list">
                {group.records.map((record) => (
                  <DomainCapabilityCard
                    active={record.id === activeDomain?.id}
                    detailed
                    key={record.id}
                    record={record}
                  />
                ))}
              </div>
            </section>
          ))}
        </div>
      ) : (
        <div className="domainExperienceList" role="list">
          {records.map((record) => (
            <DomainCapabilityCard detailed={false} key={record.id} record={record} />
          ))}
        </div>
      )}
    </section>
  );
}

export function KnowledgeBaseInventorySurface({
  detailed = false,
  inventory
}: {
  detailed?: boolean;
  inventory: KnowledgeBaseInventory;
}) {
  return (
    <article
      aria-label="Persistent Knowledge Base inventory"
      className="kbInventorySurface"
      data-state={inventory.status}
      role="group"
    >
      <header>
        <div>
          <span>Knowledge Base inventory</span>
          <strong>{formatKnowledgeBaseStatus(inventory.status)}</strong>
          <p>{inventory.detail}</p>
        </div>
      </header>
      <div aria-label="Knowledge Base inventory metrics" className="kbInventoryMetrics" role="list">
        <Metric label="Registered" value={`${inventory.registeredSourceCount} sources`} />
        <Metric label="Indexed" value={`${inventory.indexedSourceCount} sources`} />
        <Metric label="Chunks" value={`${inventory.indexedChunkCount}`} />
        {detailed ? (
          <>
            <Metric label="Registered domains" value={`${inventory.registeredDomainCount}`} />
            <Metric label="Indexed domains" value={`${inventory.indexedDomainCount}`} />
            <Metric label="Last indexed" value={formatIndexedAt(inventory.lastIndexedAt)} />
          </>
        ) : null}
      </div>
      {detailed ? (
        <>
          <dl aria-label="Knowledge Base metric guide" className="kbInventoryLegend">
            <div><dt>Registered</dt><dd>Official sources listed in the product registry.</dd></div>
            <div><dt>Indexed</dt><dd>Registered sources available in the local search index.</dd></div>
            <div><dt>Chunks</dt><dd>Searchable passages created from indexed sources.</dd></div>
            <div><dt>Domains</dt><dd>Creative domains with at least one registered or indexed source.</dd></div>
            <div><dt>Last indexed</dt><dd>The local index timestamp, not a claim that an upstream site is unchanged.</dd></div>
          </dl>
          <footer>
            <p>{inventory.freshnessDetail}</p>
            <p>{inventory.updateHint}</p>
            <p>{inventory.provenanceBoundary}</p>
          </footer>
          <KnowledgeBaseSourceExplorer inventory={inventory} />
        </>
      ) : null}
    </article>
  );
}

function KnowledgeBaseSourceExplorer({ inventory }: { inventory: KnowledgeBaseInventory }) {
  const [selectedSourceIds, setSelectedSourceIds] = useState<string[]>([]);
  const [operationState, setOperationState] = useState<{
    detail: string;
    status: string;
    sourceChanges: KnowledgeBaseSourceChange[];
  } | null>(null);
  const [runningAction, setRunningAction] = useState<KnowledgeBaseAction | null>(null);
  const sourceIds = inventory.sources.map((source) => source.id);
  const sourceTitleById = new Map(inventory.sources.map((source) => [source.id, source.title]));
  const operationRunning = runningAction !== null;
  const areAllSourcesSelected =
    sourceIds.length > 0 && sourceIds.every((sourceId) => selectedSourceIds.includes(sourceId));

  function toggleSource(sourceId: string) {
    setSelectedSourceIds((current) =>
      current.includes(sourceId)
        ? current.filter((candidate) => candidate !== sourceId)
        : [...current, sourceId]
    );
  }

  function toggleAllSources() {
    setSelectedSourceIds(areAllSourcesSelected ? [] : sourceIds);
  }

  async function runOperation(action: KnowledgeBaseAction) {
    if (
      (action === "update" || action === "rebuild") &&
      !window.confirm(
        "Update the selected official sources? If the operation fails, the prior local index will be restored."
      )
    ) {
      return;
    }
    setRunningAction(action);
    setOperationState({
      detail: operationDetail(action, selectedSourceIds.length),
      status: "running",
      sourceChanges: []
    });
    try {
      const response = await fetch(getKnowledgeBaseEndpoint(), {
        body: JSON.stringify({
          action,
          confirmed: action === "update" || action === "rebuild",
          sourceIds: selectedSourceIds
        }),
        headers: { "Content-Type": "application/json" },
        method: "POST"
      });
      const payload: unknown = await response.json();
      const record = asRecord(payload);
      if (!response.ok || !record) {
        throw new Error(readText(record?.message) ?? "Knowledge Base operation is unavailable.");
      }
      const sourceChanges = readKnowledgeBaseSourceChanges(record.sourceChanges);
      if (action === "check") {
        const unavailableSourceIds = new Set(
          sourceChanges
            .filter((change) => change.changeStatus === "unavailable")
            .map((change) => change.sourceId)
        );
        if (unavailableSourceIds.size > 0) {
          setSelectedSourceIds((current) =>
            current.filter((sourceId) => !unavailableSourceIds.has(sourceId))
          );
        }
      }
      setOperationState({
        detail: readText(record.detail) ?? "Knowledge Base operation completed.",
        status: readText(record.status) ?? "completed",
        sourceChanges
      });
    } catch (error) {
      setOperationState({
        detail:
          error instanceof Error
            ? error.message
            : "Knowledge Base operation could not be completed.",
        status: "failed",
        sourceChanges: []
      });
    } finally {
      setRunningAction(null);
    }
  }

  return (
    <section
      aria-busy={operationRunning}
      aria-label="Official Knowledge Base sources"
      className="kbSourceExplorer"
      data-running={operationRunning ? "true" : "false"}
    >
      <header>
        <div>
          <strong>Official sources</strong>
          <p>Registry inventory is separate from the references used in the current run.</p>
        </div>
        <div className="kbSourceActions" role="group" aria-label="Knowledge Base update actions">
          <button
            aria-pressed={areAllSourcesSelected}
            disabled={operationRunning || sourceIds.length === 0}
            onClick={toggleAllSources}
            type="button"
          >
            {areAllSourcesSelected ? "Clear selection" : "Select all"}
          </button>
          <button
            aria-label={runningAction === "check" ? "Checking selected official sources" : undefined}
            disabled={operationRunning || selectedSourceIds.length === 0}
            onClick={() => void runOperation("check")}
            type="button"
          >
            {runningAction === "check" ? (
              <><span aria-hidden="true" className="kbActionSpinner" />Checking sources</>
            ) : "Check for updates"}
          </button>
          <button disabled={operationRunning} onClick={() => void runOperation("validate")} type="button">
            Validate index
          </button>
          <button disabled={operationRunning || selectedSourceIds.length === 0} onClick={() => void runOperation("update")} type="button">
            Update selected
          </button>
          <button disabled={operationRunning || selectedSourceIds.length === 0} onClick={() => void runOperation("rebuild")} type="button">
            Rebuild selected
          </button>
        </div>
      </header>
      <p className="kbSourceBoundary">
        Select sources to check their official content fingerprints. Updating is an explicit, confirmed operation; a failed rebuild restores the prior local index.
      </p>
      <ul aria-label="Knowledge Base action guide" className="kbActionGuide">
        <li><strong>Check for updates</strong><span>Compares official content with local fingerprints. Read-only.</span></li>
        <li><strong>Validate index</strong><span>Inspects the local index only. No official pages are fetched.</span></li>
        <li><strong>Update selected</strong><span>Synchronizes selected official content after confirmation.</span></li>
        <li><strong>Rebuild selected</strong><span>Re-runs the selected source build; a failure restores the prior index.</span></li>
      </ul>
      {operationState ? (
        <>
          <p aria-live="polite" className="kbOperationStatus" data-status={operationState.status}>
            {operationState.status === "running" ? <span aria-hidden="true" className="kbOperationSpinner" /> : null}
            {operationState.detail}
          </p>
          {operationState.sourceChanges.length > 0 ? (
            <ul aria-label="Source change summary" className="kbSourceChangeSummary">
              {operationState.sourceChanges.map((change) => (
                <li data-status={change.changeStatus} key={change.sourceId}>
                  <strong className="kbSourceChangeBadge">{formatSourceChangeStatus(change.changeStatus)}</strong>
                  <span className="kbSourceChangeTitle">{sourceTitleById.get(change.sourceId) ?? change.sourceId}</span>
                  {change.detail ? <small>{change.detail}</small> : null}
                </li>
              ))}
            </ul>
          ) : null}
        </>
      ) : null}
      <div role="list">
        {inventory.sources.map((source) => (
          <article data-indexed={source.indexed ? "true" : "false"} key={source.id} role="listitem">
            <header>
              <div>
                <strong>{source.title}</strong>
                <span>{source.publisher} · {source.domain} · {source.sourceType}</span>
              </div>
              <label className="kbSourceSelect">
                <input
                  aria-label={`Select ${source.title} for a Knowledge Base operation`}
                  checked={selectedSourceIds.includes(source.id)}
                  onChange={() => toggleSource(source.id)}
                  type="checkbox"
                />
                <span>{source.indexed ? `${source.chunkCount} chunks` : "Not indexed"}</span>
              </label>
            </header>
            <a href={source.url} rel="noreferrer" target="_blank">Open official documentation</a>
            <dl>
              <div><dt>Health</dt><dd>{formatSourceHealth(source.health)}</dd></div>
              <div><dt>Last indexed</dt><dd>{formatIndexedAt(source.lastIndexedAt)}</dd></div>
              <div><dt>Fingerprint</dt><dd>{source.fingerprint ? source.fingerprint.slice(0, 12) : "Not recorded"}</dd></div>
              <div><dt>Provenance</dt><dd>{source.provenance}</dd></div>
            </dl>
            <p>{source.freshnessLimitation}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function getKnowledgeBaseEndpoint() {
  return process.env.NEXT_PUBLIC_KNOWLEDGE_BASE_URL ?? "http://localhost:8000/api/knowledge-base";
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function readText(value: unknown) {
  return typeof value === "string" && value.trim() ? value : null;
}

type KnowledgeBaseSourceChange = {
  sourceId: string;
  changeStatus: string;
  detail?: string;
};

type KnowledgeBaseAction = "check" | "validate" | "update" | "rebuild";

function readKnowledgeBaseSourceChanges(value: unknown): KnowledgeBaseSourceChange[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.flatMap((candidate) => {
    const record = asRecord(candidate);
    const sourceId = readText(record?.sourceId);
    const changeStatus = readText(record?.changeStatus);
    const detail = readText(record?.detail);
    return sourceId && changeStatus ? [{ sourceId, changeStatus, ...(detail ? { detail } : {}) }] : [];
  });
}

function operationDetail(action: KnowledgeBaseAction, selectedSourceCount: number) {
  switch (action) {
    case "check":
      return `Checking ${selectedSourceCount} selected official ${selectedSourceCount === 1 ? "source" : "sources"} without changing the local index.`;
    case "validate":
      return "Validating the current local index without fetching official sources.";
    case "update":
      return "Updating the selected official sources after your confirmation.";
    case "rebuild":
      return "Rebuilding the selected official sources after your confirmation.";
  }
}

function formatSourceChangeStatus(status: string) {
  return status.replace(/_/g, " ");
}

function DomainCapabilityCard({
  active = false,
  detailed,
  record
}: {
  active?: boolean;
  detailed: boolean;
  record: DomainExperienceRecord;
}) {
  return (
    <article
      aria-label={`${record.displayName} capability contract`}
      className="domainCapabilityCard"
      data-active={active ? "true" : "false"}
      data-delivery={record.deliveryKind}
      role="listitem"
    >
      <header>
        <div>
          <span>{formatDomainDeliveryKind(record.deliveryKind)}</span>
          <strong>{record.displayName}</strong>
          <p>{record.publicClaimBoundary}</p>
        </div>
        <span className="domainDeliveryBadge">{record.livePreview ? "Live" : "Export"}</span>
      </header>
      <dl className="domainCapabilityMetrics">
        <div>
          <dt>Artifacts</dt>
          <dd>{record.filenameExtensions.join(", ")}</dd>
        </div>
        <div>
          <dt>Knowledge</dt>
          <dd>
            {record.knowledge.indexedSourceCount}/{record.knowledge.registeredSourceCount} indexed
          </dd>
        </div>
        <div>
          <dt>Validation</dt>
          <dd>{formatValidationStatus(record.validationStatus)}</dd>
        </div>
      </dl>
      {detailed ? (
        <details className="domainCapabilityDetails">
          <summary>Technical contract</summary>
          <div>
            <Detail label="Use for" values={record.intentTriggers} />
            <Detail label="Runtime" values={record.runtimeRequirements} />
            <Detail label="Fallback" values={[record.fallback]} />
            <Detail label="Sources" values={record.knowledgeSourceIds} />
          </div>
        </details>
      ) : null}
    </article>
  );
}

function Detail({ label, values }: { label: string; values: string[] }) {
  return (
    <div>
      <strong>{label}</strong>
      <ul>
        {values.map((value) => (
          <li key={value}>{value}</li>
        ))}
      </ul>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <span role="listitem">
      <small>{label}</small>
      <strong>{value}</strong>
    </span>
  );
}

function formatKnowledgeBaseStatus(status: KnowledgeBaseInventory["status"]) {
  switch (status) {
    case "available":
      return "Indexed inventory available";
    case "empty":
      return "Index is empty";
    case "not_initialized":
      return "Index not initialized";
    default:
      return "Inventory unavailable";
  }
}

function formatIndexedAt(value: string | null) {
  if (!value) {
    return "Not reported";
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? value
    : parsed.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function formatSourceHealth(value: string) {
  switch (value) {
    case "locally_indexed":
      return "Locally indexed";
    case "indexed_without_timestamp":
      return "Indexed; timestamp unavailable";
    default:
      return "Registered only";
  }
}

function formatValidationStatus(status: string) {
  switch (status) {
    case "validated_browser_contract":
      return "Browser contract";
    case "validated_code_export":
      return "Code/export contract";
    case "handoff_package":
      return "Handoff package";
    default:
      return "Not published";
  }
}
