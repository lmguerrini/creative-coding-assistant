const htmlSourcePatterns: readonly RegExp[] = [
  /^\s*<!doctype\s+html\b/i,
  /^\s*<html(?:\s|>)/i,
  /^\s*<head(?:\s|>)/i,
  /^\s*<body(?:\s|>)/i,
  /^\s*<(?:script|style|canvas|main|section|div|meta|link)(?:\s|>)/i
];

export const p5HtmlSourceMismatchMessage =
  "HTML documents cannot run in the p5 JavaScript preview runtime. Use JavaScript p5 source with setup() or draw(), or route the artifact to a compatible preview surface.";

export function getP5RuntimeSourceSupportIssue(
  source: string | null | undefined
) {
  return looksLikeHtmlSource(source) ? p5HtmlSourceMismatchMessage : null;
}

export function looksLikeHtmlSource(source: string | null | undefined) {
  const text = source?.trim();

  if (!text) {
    return false;
  }

  return htmlSourcePatterns.some((pattern) => pattern.test(text));
}
