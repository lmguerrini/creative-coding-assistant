"use client";

import { Code2, RefreshCw, TriangleAlert } from "lucide-react";
import type { WorkstationError } from "@/lib/workstation-errors";

type PreviewRuntimeErrorCardProps = {
  error: WorkstationError;
  onOpenCode?: (() => void) | undefined;
  onReload?: (() => void) | undefined;
};

export function PreviewRuntimeErrorCard({
  error,
  onOpenCode,
  onReload
}: PreviewRuntimeErrorCardProps) {
  const technicalDetail = error.debugMessage ?? error.userMessage;
  const shaderLine = readShaderLine(technicalDetail);
  const title = previewErrorTitle(error.type);

  return (
    <article
      className="previewRuntimeErrorCard"
      data-error-type={error.type}
      role="alert"
    >
      <header className="previewRuntimeErrorCardHeader">
        <span className="previewRuntimeErrorIcon" aria-hidden="true">
          <TriangleAlert size={17} strokeWidth={2.2} />
        </span>
        <div>
          <span>Preview needs attention</span>
          <h3>{title}</h3>
        </div>
        <small>{error.recoverable ? "Can recover" : "Action required"}</small>
      </header>
      <p className="previewRuntimeErrorMessage">{error.userMessage}</p>
      {shaderLine ? (
        <div className="previewRuntimeErrorLocation">
          <Code2 size={13} aria-hidden="true" />
          <span>Shader line {shaderLine}</span>
        </div>
      ) : null}
      <p className="previewRuntimeErrorGuidance">{error.suggestedAction}</p>
      <div className="previewRuntimeErrorActions">
        {onOpenCode ? (
          <button
            aria-label="Open generated code"
            className="previewRuntimeActionButton previewRuntimeActionButton--primary"
            data-action="open-code"
            onClick={onOpenCode}
            type="button"
          >
            <Code2 size={14} aria-hidden="true" />
            Review code
          </button>
        ) : null}
        {onReload ? (
          <button
            aria-label="Reload preview runtime"
            className="previewRuntimeActionButton"
            data-action="reload"
            onClick={onReload}
            type="button"
          >
            <RefreshCw size={14} aria-hidden="true" />
            Try reload
          </button>
        ) : null}
      </div>
      <details className="previewRuntimeErrorDetails">
        <summary>Technical details</summary>
        <code>{technicalDetail}</code>
      </details>
    </article>
  );
}

function previewErrorTitle(type: string) {
  if (type === "shader_compile_failed") {
    return "Shader needs a quick repair";
  }
  if (type === "shader_source_invalid") {
    return "Shader entry point is missing";
  }
  if (type === "webgl_unavailable") {
    return "WebGL is unavailable";
  }
  return "Preview could not start";
}

function readShaderLine(value: string) {
  return value.match(/ERROR:\s*\d+:(\d+)/i)?.[1] ?? null;
}
