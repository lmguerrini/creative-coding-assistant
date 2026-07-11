import type {
  WorkflowExplorerModel,
  WorkflowExplorerStage
} from "@/lib/workflow-explorer";

type WorkflowExplorerSurfaceProps = {
  model: WorkflowExplorerModel;
};

export function WorkflowExplorerSurface({ model }: WorkflowExplorerSurfaceProps) {
  return (
    <article
      aria-label="Workflow explorer"
      className="workflowExplorerSurface"
      data-state={model.state}
      role="group"
    >
      <header className="workflowExplorerHeader">
        <div>
          <span>Execution path</span>
          <strong>Live workflow route</strong>
          <p>
            {`${model.summary.availableMetadataGroupCount}/${model.summary.metadataGroupCount} metadata groups available`}
          </p>
        </div>
        <div aria-label="Workflow explorer summary" className="workflowExplorerStats">
          <span>{`${model.summary.availableStageCount} available`}</span>
          <span>{`${model.summary.partialStageCount} partial`}</span>
          <span>{`${model.summary.missingStageCount} missing`}</span>
          {model.summary.errorStageCount > 0 ? (
            <span>{`${model.summary.errorStageCount} errors`}</span>
          ) : null}
        </div>
      </header>

      <div aria-label="Workflow explorer stages" className="workflowExplorerStages">
        {model.stages.map((stage) => (
          <WorkflowExplorerStageCard key={stage.id} stage={stage} />
        ))}
      </div>
    </article>
  );
}

function WorkflowExplorerStageCard({ stage }: { stage: WorkflowExplorerStage }) {
  return (
    <section
      aria-label={`${stage.label} workflow explorer stage`}
      className="workflowExplorerStage"
      data-state={stage.status}
      role="group"
    >
      <header>
        <div>
          <span>{formatStatus(stage.status)}</span>
          <strong>{stage.label}</strong>
          <p>{stage.summary}</p>
        </div>
        <small>{`${stage.eventCount} events`}</small>
      </header>
      <div className="workflowExplorerNodeList" aria-label={`${stage.label} nodes`}>
        {stage.nodeIds.map((nodeId) => (
          <code key={nodeId}>{nodeId}</code>
        ))}
      </div>
      {stage.latestEventLabel ? (
        <p className="workflowExplorerLatest">{stage.latestEventLabel}</p>
      ) : null}
      <div
        aria-label={`${stage.label} metadata groups`}
        className="workflowExplorerMetadataGroups"
      >
        {stage.metadataGroups.map((group) => (
          <article
            aria-label={`${group.label} metadata group`}
            className="workflowExplorerMetadataGroup"
            data-state={group.status}
            key={group.id}
            role="group"
          >
            <span>{formatStatus(group.status)}</span>
            <strong>{group.label}</strong>
            <p>{group.summary}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function formatStatus(status: WorkflowExplorerStage["status"]) {
  switch (status) {
    case "available":
      return "Available";
    case "partial":
      return "Partial";
    case "running":
      return "Running";
    case "error":
      return "Error";
    default:
      return "Missing";
  }
}
