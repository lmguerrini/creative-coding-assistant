"use client";

import { useEffect, useRef, useState } from "react";
import {
  formatDomainDeliveryKind,
  getDomainExperienceRecord,
  type DomainExperienceCatalog,
  type DomainExperienceRecord,
  type KnowledgeBaseInventory
} from "@/lib/domain-experience";
import {
  readKnowledgeBaseSmartUpdateSnapshot,
  writeKnowledgeBaseSmartUpdateSnapshot,
  type KnowledgeBaseSmartUpdateSnapshot
} from "@/lib/kb-smart-update";
import {
  DashboardCardGrid,
  DashboardDisclosure,
  DashboardMetricGrid
} from "./dashboard-page-primitives";

export function DomainExperienceSurface({
  activeDomainId,
  catalog,
  collapseSecondary = false,
  detailed = false,
  embedded = false,
  includeKnowledgeBase = true
}: {
  activeDomainId?: string | null;
  catalog: DomainExperienceCatalog;
  collapseSecondary?: boolean;
  detailed?: boolean;
  embedded?: boolean;
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
      {!embedded ? (
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
      ) : null}
      {detailed ? (
        <div className="domainExperienceGroups">
          {groups.map((group) => collapseSecondary && group.id !== "live" ? (
            <DashboardDisclosure
              className="domainExperienceGroupDisclosure"
              key={group.id}
              summary={(
                <span className="domainExperienceGroupSummary">
                  <span>
                    <strong>{group.label}</strong>
                    <small>{group.detail}</small>
                  </span>
                  <em>{group.records.length}</em>
                </span>
              )}
            >
              <DomainExperienceGroup
                activeDomainId={activeDomain?.id}
                group={group}
                showHeader={false}
              />
            </DashboardDisclosure>
          ) : (
            <DomainExperienceGroup
              activeDomainId={activeDomain?.id}
              group={group}
              key={group.id}
              showHeader
            />
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

function DomainExperienceGroup({
  activeDomainId,
  group,
  showHeader
}: {
  activeDomainId?: string;
  group: {
    detail: string;
    id: string;
    label: string;
    records: DomainExperienceRecord[];
  };
  showHeader: boolean;
}) {
  return (
    <section aria-label={group.label} className="domainExperienceGroup">
      {showHeader ? (
        <header>
          <div>
            <strong>{group.label}</strong>
            <p>{group.detail}</p>
          </div>
          <span>{group.records.length}</span>
        </header>
      ) : null}
      <DashboardCardGrid className="domainExperienceGrid" label={`${group.label} domain contracts`} role="list">
        {group.records.map((record) => (
          <DomainCapabilityCard
            active={record.id === activeDomainId}
            detailed
            key={record.id}
            record={record}
          />
        ))}
      </DashboardCardGrid>
    </section>
  );
}

export function KnowledgeBaseInventorySurface({
  detailed = false,
  headerMode = "default",
  inventory,
  progressive = false
}: {
  detailed?: boolean;
  headerMode?: "default" | "embedded";
  inventory: KnowledgeBaseInventory;
  progressive?: boolean;
}) {
  const [currentInventory, setCurrentInventory] = useState(inventory);

  useEffect(() => {
    setCurrentInventory(inventory);
  }, [inventory]);

  const inventoryEvidence = <KnowledgeBaseInventoryEvidence inventory={currentInventory} />;

  return (
    <article
      aria-label="Persistent Knowledge Base inventory"
      className="kbInventorySurface"
      data-state={currentInventory.status}
      data-header-mode={headerMode}
      role="group"
    >
      {headerMode === "default" ? (
        <header>
          <div>
            <span>Knowledge Base inventory</span>
            <strong>{formatKnowledgeBaseStatus(currentInventory.status)}</strong>
            <p>{currentInventory.detail}</p>
          </div>
        </header>
      ) : null}
      <DashboardMetricGrid
        className="kbInventoryMetrics"
        label="Knowledge Base inventory metrics"
        metrics={[
          { label: "Registered", value: `${currentInventory.registeredSourceCount} sources` },
          { label: "Indexed", value: `${currentInventory.indexedSourceCount} sources` },
          { label: "Chunks", value: `${currentInventory.indexedChunkCount}` },
          ...(detailed ? [
            { label: "Registered domains", value: `${currentInventory.registeredDomainCount}` },
            { label: "Indexed domains", value: `${currentInventory.indexedDomainCount}` },
            { label: "Last indexed", value: formatIndexedAt(currentInventory.lastIndexedAt) }
          ] : [])
        ]}
      />
      {detailed ? (
        <>
          {progressive ? (
            <DashboardDisclosure
              className="kbInventoryEvidenceDisclosure"
              summary="Metric definitions, freshness, and provenance"
            >
              {inventoryEvidence}
            </DashboardDisclosure>
          ) : inventoryEvidence}
          <KnowledgeBaseSourceExplorer inventory={currentInventory} onInventoryChange={setCurrentInventory} />
        </>
      ) : null}
    </article>
  );
}

function KnowledgeBaseInventoryEvidence({ inventory }: { inventory: KnowledgeBaseInventory }) {
  return (
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
    </>
  );
}

function KnowledgeBaseSourceExplorer({
  inventory,
  onInventoryChange
}: {
  inventory: KnowledgeBaseInventory;
  onInventoryChange: (inventory: KnowledgeBaseInventory) => void;
}) {
  const [selectedSourceIds, setSelectedSourceIds] = useState<string[]>([]);
  const [sourcesExpanded, setSourcesExpanded] = useState(false);
  const [operationState, setOperationState] = useState<{
    detail: string;
    status: string;
    sourceChanges: KnowledgeBaseSourceChange[];
  } | null>(null);
  const [runningAction, setRunningAction] = useState<KnowledgeBaseAction | null>(null);
  const [smartUpdateRunning, setSmartUpdateRunning] = useState(false);
  const [smartUpdateState, setSmartUpdateState] = useState<SmartUpdateState | null>(null);
  const [latestSmartUpdate, setLatestSmartUpdate] = useState<KnowledgeBaseSmartUpdateSnapshot | null>(null);
  const smartUpdateInFlight = useRef(false);
  const sourceIds = inventory.sources.map((source) => source.id);
  const sourceTitleById = new Map(inventory.sources.map((source) => [source.id, source.title]));
  const operationRunning = runningAction !== null || smartUpdateRunning;
  const areAllSourcesSelected =
    sourceIds.length > 0 && sourceIds.every((sourceId) => selectedSourceIds.includes(sourceId));

  useEffect(() => {
    setLatestSmartUpdate(readKnowledgeBaseSmartUpdateSnapshot());
  }, []);

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

  function setSmartStep(
    stepId: SmartUpdateStepId,
    status: SmartUpdateStepStatus,
    detail?: string
  ) {
    setSmartUpdateState((current) => current
      ? {
          ...current,
          steps: current.steps.map((step) =>
            step.id === stepId ? { ...step, status, ...(detail ? { detail } : {}) } : step
          )
        }
      : current);
  }

  function failSmartUpdate(stepId: SmartUpdateStepId, message: string) {
    setSmartStep(stepId, "failed", message);
    setSmartUpdateState((current) => current
      ? {
          ...current,
          detail: `${message} Recovery: the later steps were not run; review the failed source and retry Smart Update.`,
          status: "failed"
        }
      : current);
  }

  function applyReturnedInventory(record: Record<string, unknown>) {
    const returnedInventory = readKnowledgeBaseInventory(record.inventory);
    if (returnedInventory) {
      onInventoryChange(returnedInventory);
    }
  }

  async function runOperation(action: KnowledgeBaseAction) {
    if (operationRunning) {
      return;
    }
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
      const record = await requestKnowledgeBaseOperation(
        action,
        selectedSourceIds,
        action === "update" || action === "rebuild"
      );
      applyReturnedInventory(record);
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

  async function runSmartUpdate() {
    if (smartUpdateInFlight.current || operationRunning || sourceIds.length === 0) {
      return;
    }

    const sourceScope = selectedSourceIds.length > 0 ? selectedSourceIds : sourceIds;
    const scopeDescription = selectedSourceIds.length > 0 ? "the selected" : "all registered";
    if (!window.confirm(
      `Smart Update will check ${scopeDescription} official sources, then update and rebuild only reachable changed sources before validating the local index. Continue?`
    )) {
      return;
    }

    smartUpdateInFlight.current = true;
    setSmartUpdateRunning(true);
    setOperationState(null);
    setSmartUpdateState({
      affectedSourceCount: 0,
      detail: `Checking ${sourceScope.length} official ${sourceScope.length === 1 ? "source" : "sources"}. No local index changes have started.`,
      scopeSourceCount: sourceScope.length,
      status: "running",
      steps: SMART_UPDATE_STEPS.map((step) => ({
        ...step,
        status: step.id === "check" ? "running" : "pending"
      })),
      unavailableSourceCount: 0
    });

    let activeStep: SmartUpdateStepId = "check";
    try {
      const checkRecord = await requestKnowledgeBaseOperation("check", sourceScope, false);
      const sourceChanges = readKnowledgeBaseSourceChanges(checkRecord.sourceChanges);
      const unavailableSourceIds = sourceChanges
        .filter((change) => change.changeStatus === "unavailable")
        .map((change) => change.sourceId);
      const affectedSourceIds = sourceChanges
        .filter((change) => change.changeStatus === "new" || change.changeStatus === "changed")
        .map((change) => change.sourceId);

      setSelectedSourceIds(affectedSourceIds);
      setSmartStep("check", "completed", `${sourceScope.length} checked; ${affectedSourceIds.length} changed and reachable.`);
      setSmartUpdateState((current) => current
        ? {
            ...current,
            affectedSourceCount: affectedSourceIds.length,
            detail: smartCheckDetail(affectedSourceIds.length, unavailableSourceIds.length),
            unavailableSourceCount: unavailableSourceIds.length
          }
        : current);

      if (affectedSourceIds.length === 0) {
        setSmartStep("update", "skipped", "No changed reachable sources.");
        setSmartStep("rebuild", "skipped", "No affected index needs rebuilding.");
      } else {
        activeStep = "update";
        setSmartStep("update", "running", `Updating ${affectedSourceIds.length} affected source${affectedSourceIds.length === 1 ? "" : "s"}.`);
        const updateRecord = await requestKnowledgeBaseOperation("update", affectedSourceIds, true);
        applyReturnedInventory(updateRecord);
        setSmartStep("update", "completed", `${affectedSourceIds.length} affected source${affectedSourceIds.length === 1 ? " was" : "s were"} updated.`);

        activeStep = "rebuild";
        setSmartStep("rebuild", "running", `Rebuilding ${affectedSourceIds.length} affected index${affectedSourceIds.length === 1 ? "" : "es"}.`);
        const rebuildRecord = await requestKnowledgeBaseOperation("rebuild", affectedSourceIds, true);
        applyReturnedInventory(rebuildRecord);
        setSmartStep("rebuild", "completed", `${affectedSourceIds.length} affected index${affectedSourceIds.length === 1 ? " was" : "es were"} rebuilt.`);
      }

      const validationSourceIds = affectedSourceIds.length > 0
        ? affectedSourceIds
        : sourceChanges
          .filter((change) => change.changeStatus === "unchanged")
          .map((change) => change.sourceId);
      let validationStatus = "not_required";
      if (validationSourceIds.length === 0) {
        setSmartStep("validate", "skipped", "No reachable local index changed or required validation.");
      } else {
        activeStep = "validate";
        setSmartStep("validate", "running", "Validating only the affected reachable local indexes.");
        const validationRecord = await requestKnowledgeBaseOperation("validate", validationSourceIds, false);
        applyReturnedInventory(validationRecord);
        validationStatus = readText(validationRecord.status) ?? "unknown";
        if (validationStatus !== "passed") {
          failSmartUpdate("validate", readText(validationRecord.detail) ?? "The affected local index needs attention after Smart Update.");
          return;
        }
        setSmartStep("validate", "completed", "Affected local index validation passed.");
      }

      const completedAt = new Date().toISOString();
      const snapshot: KnowledgeBaseSmartUpdateSnapshot = {
        affectedSourceCount: affectedSourceIds.length,
        completedAt,
        rebuiltSourceCount: affectedSourceIds.length,
        scopeSourceCount: sourceScope.length,
        unavailableSourceCount: unavailableSourceIds.length,
        updatedSourceCount: affectedSourceIds.length,
        validationStatus
      };
      writeKnowledgeBaseSmartUpdateSnapshot(snapshot);
      setLatestSmartUpdate(snapshot);
      setSmartUpdateState((current) => current
        ? {
            ...current,
            detail: smartCompletionDetail(
              affectedSourceIds.length,
              unavailableSourceIds.length,
              validationStatus
            ),
            status: "completed"
          }
        : current);
    } catch (error) {
      const message = error instanceof Error
        ? error.message
        : "Smart Update could not complete.";
      failSmartUpdate(activeStep, message);
    } finally {
      smartUpdateInFlight.current = false;
      setSmartUpdateRunning(false);
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
        <div className="kbSourceExplorerIntro">
          <strong>Official sources</strong>
          <p>Registry inventory is separate from the references used in the current run.</p>
          <p className="kbSourceBoundary">
            Smart Update uses all registered sources when none are selected. It checks first, writes only changed reachable sources after your confirmation, and keeps the prior valid local index when a write fails.
          </p>
        </div>
        <div className="kbSourceExplorerControls">
          <div className="kbSmartUpdateActions" role="group" aria-label="Knowledge Base Smart Update action">
            <button
              aria-label={smartUpdateRunning ? "Smart Update in progress" : undefined}
              className="kbSmartUpdateButton"
              disabled={operationRunning || sourceIds.length === 0}
              onClick={() => void runSmartUpdate()}
              type="button"
            >
              {smartUpdateRunning ? <><span aria-hidden="true" className="kbActionSpinner" />Smart Update running</> : "Smart Update"}
            </button>
            <span>Checks, updates changed reachable sources, rebuilds, then validates after one confirmation.</span>
          </div>
          <button
            aria-controls="official-source-management"
            aria-expanded={sourcesExpanded}
            aria-label={sourcesExpanded ? "Hide official sources" : `Show all ${sourceIds.length} official ${sourceIds.length === 1 ? "source" : "sources"}`}
            className="kbSourceListToggle"
            onClick={() => setSourcesExpanded((current) => !current)}
            type="button"
          >
            <span>{sourcesExpanded ? "Hide official sources" : `Show all ${sourceIds.length} official ${sourceIds.length === 1 ? "source" : "sources"}`}</span>
            <small>{sourcesExpanded ? "Advanced controls and the source registry are open." : "Browse, select, and manage individual sources."}</small>
          </button>
        </div>
      </header>
      {smartUpdateState ? (
        <section aria-live="polite" className="kbSmartUpdateStatus" data-status={smartUpdateState.status}>
          <header>
            <div>
              <strong>{smartUpdateState.status === "completed" ? "Smart Update complete" : smartUpdateState.status === "failed" ? "Smart Update needs attention" : "Smart Update in progress"}</strong>
              <p>{smartUpdateState.detail}</p>
            </div>
            <span>{smartUpdateState.scopeSourceCount} checked · {smartUpdateState.affectedSourceCount} affected · {smartUpdateState.unavailableSourceCount} unavailable</span>
          </header>
          <ol aria-label="Smart Update progress">
            {smartUpdateState.steps.map((step) => (
              <li aria-current={step.status === "running" ? "step" : undefined} data-status={step.status} key={step.id}>
                <span aria-hidden="true" className={step.status === "running" ? "kbSmartStepSpinner" : "kbSmartStepMarker"} />
                <div><strong>{step.label}</strong><small>{step.detail}</small></div>
              </li>
            ))}
          </ol>
        </section>
      ) : null}
      {latestSmartUpdate ? (
        <p className="kbSmartUpdateHistory">
          Last successful Smart Update: {formatIndexedAt(latestSmartUpdate.completedAt)} · {latestSmartUpdate.affectedSourceCount} affected · {latestSmartUpdate.validationStatus === "passed" ? "validation passed." : "validation was not required."}
        </p>
      ) : null}
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
      {sourcesExpanded ? (
        <div className="kbSourceExplorerDetails" id="official-source-management">
          <div className="kbSourceActions" role="group" aria-label="Advanced Knowledge Base update actions">
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
            <button disabled={operationRunning || selectedSourceIds.length === 0} onClick={() => void runOperation("update")} type="button">
              Update selected
            </button>
            <button disabled={operationRunning || selectedSourceIds.length === 0} onClick={() => void runOperation("rebuild")} type="button">
              Rebuild selected
            </button>
            <button disabled={operationRunning} onClick={() => void runOperation("validate")} type="button">
              Validate index
            </button>
          </div>
          <ul aria-label="Knowledge Base action guide" className="kbActionGuide">
            <li><strong>Smart Update</strong><span>Checks first, then updates and rebuilds only changed reachable sources before validation.</span></li>
            <li><strong>Check for updates</strong><span>Compares official content with local fingerprints. Read-only.</span></li>
            <li><strong>Update selected</strong><span>Synchronizes selected official content after confirmation.</span></li>
            <li><strong>Rebuild selected</strong><span>Re-runs the selected source build; a failure restores the prior index.</span></li>
            <li><strong>Validate index</strong><span>Inspects the local index only. No official pages are fetched.</span></li>
          </ul>
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
        </div>
      ) : (
        <p className="kbSourceExplorerCollapsedHint">
          Individual source selection and advanced controls are hidden. Expand the registry to browse all {sourceIds.length} official {sourceIds.length === 1 ? "source" : "sources"}.
        </p>
      )}
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

function readKnowledgeBaseInventory(value: unknown): KnowledgeBaseInventory | null {
  const record = asRecord(value);
  if (
    !record ||
    typeof record.status !== "string" ||
    typeof record.detail !== "string" ||
    !Array.isArray(record.sources)
  ) {
    return null;
  }
  return record as unknown as KnowledgeBaseInventory;
}

type KnowledgeBaseSourceChange = {
  sourceId: string;
  changeStatus: string;
  detail?: string;
};

type KnowledgeBaseAction = "check" | "validate" | "update" | "rebuild";

type SmartUpdateStepId = "check" | "update" | "rebuild" | "validate";
type SmartUpdateStepStatus = "pending" | "running" | "completed" | "skipped" | "failed";
type SmartUpdateState = {
  affectedSourceCount: number;
  detail: string;
  scopeSourceCount: number;
  status: "running" | "completed" | "failed";
  steps: Array<{ detail: string; id: SmartUpdateStepId; label: string; status: SmartUpdateStepStatus }>;
  unavailableSourceCount: number;
};

const SMART_UPDATE_STEPS: Array<{ detail: string; id: SmartUpdateStepId; label: string }> = [
  { detail: "Compare official content with local fingerprints.", id: "check", label: "Check for updates" },
  { detail: "Write only changed reachable official sources.", id: "update", label: "Update affected" },
  { detail: "Rebuild only the affected local source indexes.", id: "rebuild", label: "Rebuild affected" },
  { detail: "Verify the resulting local index.", id: "validate", label: "Validate index" }
];

async function requestKnowledgeBaseOperation(
  action: KnowledgeBaseAction,
  sourceIds: string[],
  confirmed: boolean
) {
  const response = await fetch(getKnowledgeBaseEndpoint(), {
    body: JSON.stringify({ action, confirmed, sourceIds }),
    headers: { "Content-Type": "application/json" },
    method: "POST"
  });
  const payload: unknown = await response.json();
  const record = asRecord(payload);
  if (!response.ok || !record) {
    throw new Error(readText(record?.message) ?? "Knowledge Base operation is unavailable.");
  }
  return record;
}

function smartCheckDetail(affectedSourceCount: number, unavailableSourceCount: number) {
  if (affectedSourceCount === 0) {
    return unavailableSourceCount > 0
      ? `No changed reachable sources were found. ${unavailableSourceCount} unavailable ${unavailableSourceCount === 1 ? "source was" : "sources were"} skipped; their prior local index remains valid.`
      : "No changed reachable sources were found. Update and rebuild will be skipped.";
  }
  const affectedDetail = `${affectedSourceCount} changed reachable ${affectedSourceCount === 1 ? "source" : "sources"}`;
  return unavailableSourceCount > 0
    ? `${affectedDetail} will be updated. ${unavailableSourceCount} unavailable ${unavailableSourceCount === 1 ? "source was" : "sources were"} skipped; their prior local index remains valid.`
    : `${affectedDetail} will be updated and rebuilt.`;
}

function smartCompletionDetail(
  affectedSourceCount: number,
  unavailableSourceCount: number,
  validationStatus: string
) {
  const sourceDetail = affectedSourceCount === 0
    ? "No changed reachable sources required a write."
    : `${affectedSourceCount} affected ${affectedSourceCount === 1 ? "source was" : "sources were"} updated and rebuilt.`;
  const validationDetail = validationStatus === "passed"
    ? "Validation passed."
    : "No eligible local index required validation.";
  return unavailableSourceCount > 0
    ? `${sourceDetail} ${unavailableSourceCount} unavailable ${unavailableSourceCount === 1 ? "source was" : "sources were"} skipped; ${validationDetail}`
    : `${sourceDetail} ${validationDetail}`;
}

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
      className="dashboardInnerCard domainCapabilityCard"
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
        <DashboardDisclosure className="domainCapabilityDetails" summary="Technical contract">
          <div>
            <Detail label="Use for" values={record.intentTriggers} />
            <Detail label="Runtime" values={record.runtimeRequirements} />
            <Detail label="Fallback" values={[record.fallback]} />
            <Detail label="Sources" values={record.knowledgeSourceIds} />
          </div>
        </DashboardDisclosure>
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
