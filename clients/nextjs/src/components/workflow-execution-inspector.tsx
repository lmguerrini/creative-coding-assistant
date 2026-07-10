import type {
  WorkflowExecutionMode,
  WorkflowExecutionModel
} from "@/lib/workflow-execution";

const modeOptions: Array<{
  value: WorkflowExecutionMode;
  label: string;
  detail: string;
}> = [
  {
    value: "auto",
    label: "Auto",
    detail: "Choose the bounded route from the request."
  },
  {
    value: "single_agent",
    label: "Single Agent",
    detail: "Generate without separate researcher, critic, or reviewer stages."
  },
  {
    value: "multi_agent",
    label: "Multi Agent",
    detail: "Run the bounded planner, researcher, generator, critic, and reviewer route."
  }
];

export function WorkflowExecutionSelector({
  disabled = false,
  mode,
  onChange
}: {
  disabled?: boolean;
  mode: WorkflowExecutionMode;
  onChange: (mode: WorkflowExecutionMode) => void;
}) {
  return (
    <fieldset aria-label="Workflow selector" className="workflowExecutionSelector">
      <legend>Workflow</legend>
      <div>
        {modeOptions.map((option) => (
          <label key={option.value} title={option.detail}>
            <input
              checked={mode === option.value}
              disabled={disabled}
              name="workflow-execution-mode"
              onChange={() => onChange(option.value)}
              type="radio"
              value={option.value}
            />
            <span>{option.label}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

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
