import type {
  WorkflowExecutionMode,
  WorkflowExecutionModel
} from "@/lib/workflow-execution";

export function WorkflowExecutionInspector({
  execution
}: {
  execution: WorkflowExecutionModel;
}) {
  const resolvedLabel = execution.resolvedMode
    ? formatMode(execution.resolvedMode)
    : "Awaiting run";

  return (
    <article
      aria-label="Workflow execution decision"
      className="workflowExecutionInspector"
      data-state={execution.state}
      role="group"
    >
      <header>
        <div>
          <span>Execution decision</span>
          <strong>{resolvedLabel}</strong>
          <p>{execution.rationale}</p>
        </div>
        <small>{`Requested ${formatMode(execution.requestedMode)}`}</small>
      </header>
      {execution.agentRoles.length > 0 ? (
        <div aria-label="Executed agent roles" className="workflowExecutionRoles">
          {execution.agentRoles.map((role) => (
            <span key={role}>{formatRole(role)}</span>
          ))}
        </div>
      ) : null}
      {execution.researcherRequired !== null ? (
        <p className="workflowExecutionResearch">
          <strong>{execution.researcherRequired ? "Researcher active" : "Researcher skipped"}</strong>
          {execution.researcherReason ? ` — ${execution.researcherReason}` : ""}
        </p>
      ) : null}
      {execution.maxRefinementLoops !== null ? (
        <small>{`Refinement budget: ${execution.maxRefinementLoops} bounded loop${
          execution.maxRefinementLoops === 1 ? "" : "s"
        }.`}</small>
      ) : null}
    </article>
  );
}

function formatMode(mode: WorkflowExecutionMode) {
  return mode === "single_agent"
    ? "Single Agent"
    : mode === "multi_agent"
      ? "Multi Agent"
      : "Auto";
}

function formatRole(role: string) {
  return role.replace(/[_-]+/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}
