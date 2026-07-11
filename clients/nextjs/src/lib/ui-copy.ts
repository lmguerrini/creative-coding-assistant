const technicalPrefixes = /^(?:gpt(?:-|\b)|p5(?:\.|\b)|three\.js\b|glsl\b|openai\b|anthropic\b|gemini\b|langsmith\b)/i;

/**
 * Shared UI-copy rule for human-readable state labels. It normalizes only the
 * first ordinary word and deliberately leaves model, runtime, and brand names
 * untouched.
 */
export function formatUiStatusLabel(value: string) {
  if (!value || technicalPrefixes.test(value.trim())) {
    return value;
  }

  return value.replace(/^(\s*)(\p{Ll})/u, (_match, spacing: string, letter: string) =>
    `${spacing}${letter.toLocaleUpperCase()}`
  );
}
