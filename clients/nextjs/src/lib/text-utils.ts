export type FormatCodeOptions = {
  separatorPattern?: RegExp;
  splitCamelCase?: boolean;
};

export type UniqueStringsOptions = {
  dropEmpty?: boolean;
  trim?: boolean;
};

export function truncate(value: string, maxLength = 140): string {
  return value.length > maxLength
    ? `${value.slice(0, Math.max(0, maxLength - 3))}...`
    : value;
}

export function formatCode(
  value: string,
  {
    separatorPattern = /[_-]+/g,
    splitCamelCase = false
  }: FormatCodeOptions = {}
): string {
  const separated = value.replace(separatorPattern, " ");
  const spaced = splitCamelCase
    ? separated.replace(/([a-z])([A-Z])/g, "$1 $2")
    : separated;

  return spaced.replace(/\b\w/g, (character) => character.toUpperCase());
}

export function uniqueStrings(
  values: string[],
  { dropEmpty = false, trim = false }: UniqueStringsOptions = {}
): string[] {
  const normalized = trim ? values.map((value) => value.trim()) : values;
  const filtered = dropEmpty ? normalized.filter(Boolean) : normalized;

  return [...new Set(filtered)];
}

export function trimToSentence(value: string, maxLength = 140): string {
  return truncate(value.trim().replace(/\s+/g, " "), maxLength);
}
