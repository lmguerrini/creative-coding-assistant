const { test, expect } = require("@playwright/test");

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

  await page.goto(
    `/preview-sandbox.html#${encodeURIComponent(
      JSON.stringify({
        kind: "glsl",
        runtimeId: "glsl-fragment-contract",
        source: {
          fingerprint: "glsl-fragment-contract",
          lineCount: source.split("\n").length,
          source,
          title: "concentric-glow.frag"
        }
      })
    )}`
  );

  await expect(page.locator("body")).toHaveAttribute("data-runtime-state", "running");
  await expect(page.locator("#preview-canvas")).toBeVisible();
  expect(pageErrors).toEqual([]);
});
