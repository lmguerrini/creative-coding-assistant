"use client";

import type { WorkstationError } from "@/lib/workstation-errors";

type SubsystemErrorCalloutProps = {
  error: WorkstationError;
  className?: string;
  role?: "alert" | "status";
  title?: string;
};

export function SubsystemErrorCallout({
  error,
  className = "",
  role = "status",
  title
}: SubsystemErrorCalloutProps) {
  const classes = ["subsystemErrorCallout", className].filter(Boolean).join(" ");

  return (
    <article
      className={classes}
      data-category={error.category}
      data-recoverable={error.recoverable ? "true" : "false"}
      role={role}
    >
      <header className="subsystemErrorHeader">
        <div>
          <span className="eyebrow">Subsystem error</span>
          <strong>{title ?? formatErrorType(error.type)}</strong>
        </div>
        <small>{error.recoverable ? "Recoverable" : "Action required"}</small>
      </header>
      <p className="subsystemErrorMessage">{error.userMessage}</p>
      <p className="subsystemErrorAction">{error.suggestedAction}</p>
      <dl className="subsystemErrorMeta">
        <div>
          <dt>Type</dt>
          <dd>{formatErrorType(error.type)}</dd>
        </div>
        <div>
          <dt>Subsystem</dt>
          <dd>{formatErrorType(error.subsystem)}</dd>
        </div>
        <div>
          <dt>Recovery</dt>
          <dd>{error.recoverable ? "Retry available" : "Reset required"}</dd>
        </div>
        {error.retryLabel ? (
          <div>
            <dt>Retry</dt>
            <dd>{error.retryLabel}</dd>
          </div>
        ) : null}
        {error.resetLabel ? (
          <div>
            <dt>Reset</dt>
            <dd>{error.resetLabel}</dd>
          </div>
        ) : null}
      </dl>
      {error.debugMessage ? (
        <p className="subsystemErrorDebug">
          <code>{error.debugMessage}</code>
        </p>
      ) : null}
    </article>
  );
}

function formatErrorType(value: string) {
  return value
    .replace(/[_-]/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}
