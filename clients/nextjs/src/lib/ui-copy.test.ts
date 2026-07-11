import { describe, expect, it } from "vitest";
import { formatUiStatusLabel } from "./ui-copy";

describe("formatUiStatusLabel", () => {
  it("capitalizes ordinary user-visible status labels", () => {
    expect(formatUiStatusLabel("available")).toBe("Available");
    expect(formatUiStatusLabel("  needs attention")).toBe("  Needs attention");
  });

  it("preserves technical identifiers", () => {
    expect(formatUiStatusLabel("gpt-5-mini / streaming")).toBe("gpt-5-mini / streaming");
    expect(formatUiStatusLabel("three.js ready")).toBe("three.js ready");
  });
});
