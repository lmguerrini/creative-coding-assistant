const { test, expect } = require("@playwright/test");
const { mountSandboxRuntime } = require("./support/preview-sandbox");

test("executes a bounded GLSL fragment shader without compile errors", async ({ page }) => {
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));

  const source = [
    "precision mediump float;",
    "uniform vec2 u_resolution;",
    "uniform float u_time;",
    "void main() {",
    "  vec2 uv = gl_FragCoord.xy / u_resolution.xy;",
    "  float pulse = 0.5 + 0.5 * sin(u_time + uv.x * 6.28318);",
    "  gl_FragColor = vec4(uv.x, uv.y, pulse, 1.0);",
    "}"
  ].join("\n");

  const sandbox = await mountSandboxRuntime(page, {
    kind: "glsl",
    runtimeId: "glsl-fragment-contract",
    source: {
      fingerprint: "glsl-fragment-contract",
      lineCount: source.split("\n").length,
      source,
      title: "concentric-glow.frag"
    }
  });

  await expect(sandbox.locator("body")).toHaveAttribute("data-runtime-state", "running");
  await expect(sandbox.locator("#preview-canvas")).toBeVisible();
  expect(pageErrors).toEqual([]);
});
