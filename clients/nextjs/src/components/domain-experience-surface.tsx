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
  detailed = false
}: {
  activeDomainId?: string | null;
  catalog: DomainExperienceCatalog;
  detailed?: boolean;
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

  return (
    <section aria-label="Domain experience" className="domainExperienceSurface">
      <KnowledgeBaseInventorySurface inventory={catalog.knowledgeBase} detailed={detailed} />
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
      <div className={detailed ? "domainExperienceGrid" : "domainExperienceList"} role="list">
        {records.map((record) => (
          <DomainCapabilityCard detailed={detailed} key={record.id} record={record} />
        ))}
      </div>
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
        <footer>
          <p>{inventory.freshnessDetail}</p>
          <p>{inventory.updateHint}</p>
          <p>{inventory.provenanceBoundary}</p>
        </footer>
      ) : null}
    </article>
  );
}

function DomainCapabilityCard({
  detailed,
  record
}: {
  detailed: boolean;
  record: DomainExperienceRecord;
}) {
  return (
    <article
      aria-label={`${record.displayName} capability contract`}
      className="domainCapabilityCard"
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
        <div className="domainCapabilityDetails">
          <Detail label="Use for" values={record.intentTriggers} />
          <Detail label="Runtime" values={record.runtimeRequirements} />
          <Detail label="Fallback" values={[record.fallback]} />
          <Detail label="Sources" values={record.knowledgeSourceIds} />
        </div>
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
