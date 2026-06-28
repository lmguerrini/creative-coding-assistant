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
  const text = stripLeadingHtmlComments(source?.trimStart() ?? "");

  if (!text) {
    return false;
  }

  return htmlSourcePatterns.some((pattern) => pattern.test(text));
}

function stripLeadingHtmlComments(source: string) {
  let text = source;
  let match = text.match(/^<!--[\s\S]*?-->\s*/);

  while (match) {
    text = text.slice(match[0].length).trimStart();
    match = text.match(/^<!--[\s\S]*?-->\s*/);
  }

  return text.trim();
}
