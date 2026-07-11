const { test, expect } = require("@playwright/test");
const {
  expectGeneratedPreview,
  expectLoadedWorkstation,
  expectRetrievalRegressionSurface,
  expectStableVisualLayout,
  expectWorkspacePersistence,
  installApiMocks,
  installConsoleGate,
  submitCreativePrompt
} = require("./support/quality-gates");

test.describe("V7.4 workstation E2E smoke", () => {
  test("loads localhost and preserves first-run shell reliability", async ({ page }) => {
    const consoleGate = installConsoleGate(page);
    await installApiMocks(page);

    const response = await page.request.get("/");
    expect(response.status()).toBe(200);

    await expectLoadedWorkstation(page);
    await expectStableVisualLayout(page);
    consoleGate.assertClean();
  });

  test("keeps inspector controls readable without wrapped or overlapping labels", async ({
    page
  }) => {
    const consoleGate = installConsoleGate(page);
    await installApiMocks(page);
    await expectLoadedWorkstation(page);

    const inspectorTabs = page.getByRole("tablist", { name: "Inspector tabs" });
    await expect(inspectorTabs).toBeVisible();
    const labelLayout = await inspectorTabs.locator("button span").evaluateAll((labels) =>
      labels.map((label) => {
        const style = window.getComputedStyle(label);
        return {
          isSingleLine: style.whiteSpace === "nowrap",
          fits: label.scrollWidth <= label.clientWidth
        };
      })
    );

    expect(labelLayout).not.toHaveLength(0);
    expect(labelLayout.every(({ isSingleLine, fits }) => isSingleLine && fits)).toBe(true);
    consoleGate.assertClean();
  });

  test("completes a creative user journey with preview, code, artifacts, retrieval, and persistence", async ({
    page
  }) => {
    const consoleGate = installConsoleGate(page);
    await installApiMocks(page, "success");
    await expectLoadedWorkstation(page);

    await submitCreativePrompt(
      page,
      "Create a teal p5 orbit field with a quiet background for E2E smoke."
    );

    await expectGeneratedPreview(page);
    await expectRetrievalRegressionSurface(page);
    await expectWorkspacePersistence(page);
    consoleGate.assertClean();
  });

  test("runs the homepage p5 suggestion through a previewable p5 route", async ({
    page
  }) => {
    const consoleGate = installConsoleGate(page);
    await installApiMocks(page, "success");
    await expectLoadedWorkstation(page);

    const p5Suggestion =
      "Create a single .p5.js JavaScript sketch for a flow-field particle system with setup(), draw(), soft trails, and interaction controls. Optimize for browser preview at 60 fps. Use strokeCap(ROUND) for rounded paths. Return only one runnable p5.js artifact.";
    await page.getByRole("button", { name: p5Suggestion }).click();
    await expect(page.getByRole("textbox", { name: "Assistant prompt" })).toHaveValue(
      p5Suggestion
    );
    await expect(page.getByRole("textbox", { name: "Assistant prompt" })).toHaveValue(
      /strokeCap\(ROUND\)/
    );
    await page.getByRole("button", { name: "Send prompt" }).click();

    await expectGeneratedPreview(page);
    await page.getByRole("tab", { name: "Preview" }).click();
    await expect(page.getByRole("tabpanel", { name: "Preview inspector" })).not.toContainText(
      "No matching live renderer"
    );
    const runtime = page.getByRole("group", { name: "p5.js live runtime" });
    await expect(runtime).toHaveAttribute("data-runtime-state", "running");
    await expect(runtime).not.toContainText("colorMode is not defined");
    await expect(
      page
        .frameLocator('iframe[title="p5.js preview runtime"]')
        .locator("canvas")
    ).toBeVisible();
    consoleGate.assertClean();
  });

  test("runs the original p5 flow-field prompt through a previewable p5 route", async ({
    page
  }) => {
    const consoleGate = installConsoleGate(page);
    await installApiMocks(page, "success");
    await expectLoadedWorkstation(page);

    await submitCreativePrompt(
      page,
      "Create a p5.js flow-field particle system with soft trails and interaction controls."
    );

    await expectGeneratedPreview(page);
    await page.getByRole("tab", { name: "Preview" }).click();
    await expect(page.getByRole("tabpanel", { name: "Preview inspector" })).not.toContainText(
      "No matching live renderer"
    );
    consoleGate.assertClean();
  });

  test("opens integrated Demo Mode and preloads a curated scenario", async ({
    page
  }) => {
    const consoleGate = installConsoleGate(page);
    await installApiMocks(page, "success");
    await expectLoadedWorkstation(page);

    await page.getByRole("button", { name: "Demo Mode" }).click();
    const demoMode = page.getByRole("region", { name: "Demo Mode" });
    await expect(demoMode).toBeVisible();
    await expect(demoMode).toContainText("Capstone scenarios");

    await demoMode
      .getByRole("button", { name: /p5\.js Browser Preview Flow Field/ })
      .click();

    await expect(demoMode).toContainText("Prompt loaded");
    await expect(page.getByRole("textbox", { name: "Assistant prompt" })).toHaveValue(
      /flow-field particle system/
    );
    await expect(page.getByRole("textbox", { name: "Assistant prompt" })).toHaveValue(
      /strokeCap\(ROUND\)/
    );
    await expect(demoMode).not.toContainText(/HoloGenesis/i);
    await expect(demoMode).not.toContainText(/\bsacred\b/i);
    consoleGate.assertClean();
  });

  test("runs the verified Demo Mode p5 prompt through the normal preview flow", async ({
    page
  }) => {
    const consoleGate = installConsoleGate(page);
    await installApiMocks(page, "success");
    await expectLoadedWorkstation(page);

    await page.getByRole("button", { name: "Demo Mode" }).click();
    const demoMode = page.getByRole("region", { name: "Demo Mode" });
    await expect(demoMode).toBeVisible();
    await expect(page.getByRole("complementary", { name: "Right inspector" })).toHaveAttribute(
      "data-state",
      "collapsed"
    );

    await expect(demoMode).toContainText("1 flows");
    await demoMode
      .getByRole("button", { name: /p5\.js Browser Preview Flow Field/ })
      .click();
    await expect(page.getByRole("textbox", { name: "Assistant prompt" })).toHaveValue(
      /flow-field particle system/
    );
    await page.getByRole("button", { name: "Send prompt" }).click();
    await expectGeneratedPreview(page);

    await expect(demoMode).not.toContainText(/HoloGenesis/i);
    await expect(demoMode).not.toContainText(/\bsacred\b/i);
    consoleGate.assertClean();
  });

  test("keeps User and Developer Mode controls usable while clearing session state", async ({
    page
  }) => {
    const consoleGate = installConsoleGate(page);
    await installApiMocks(page, "success");
    await expectLoadedWorkstation(page);

    await page.getByRole("button", { name: "Demo Mode" }).click();
    await page
      .getByRole("region", { name: "Demo Mode" })
      .getByRole("button", { name: /p5\.js Browser Preview Flow Field/ })
      .click();
    await expect(page.getByRole("button", { name: "Display mode" })).toContainText(
      "Developer"
    );
    await page.getByRole("button", { name: "Display mode" }).click();
    await expect(page.getByRole("button", { name: "Display mode" })).toContainText(
      "User"
    );
    await expect(page.getByRole("complementary", { name: "Right inspector" })).toHaveAttribute(
      "data-state",
      "collapsed"
    );

    await page.getByRole("button", { name: "Expand inspector" }).click();
    await expect(page.getByRole("tab", { name: "Preview" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "Code" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "Saved" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "Runtime" })).toHaveCount(0);

    await page.getByRole("button", { name: "Display mode" }).click();
    await expect(page.getByRole("button", { name: "Display mode" })).toContainText(
      "Developer"
    );
    await expect(page.getByRole("tab", { name: "Workflow" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "Telemetry" })).toBeVisible();

    await page.getByRole("button", { name: "Theme" }).click();
    await page.getByRole("button", { name: "Use Codex theme" }).click();
    await expect(page.locator(".workstation")).toHaveAttribute("data-theme", "codex");

    await page.getByRole("button", { name: "Command menu" }).click();
    await page.getByRole("button", { name: "Clear workspace session" }).click();
    await page.getByRole("button", { name: "Clear workspace" }).click();

    await expect(page.getByRole("region", { name: "Demo Mode" })).toHaveCount(0);
    await expect(page.getByRole("textbox", { name: "Assistant prompt" })).toHaveValue("");
    await expect(page.getByRole("region", { name: "Preview workspace" })).toHaveCount(0);
    await expect(page.getByRole("group", { name: "Empty creative workspace" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Display mode" })).toContainText(
      "Developer"
    );
    await expect(page.getByRole("complementary", { name: "Right inspector" })).toHaveAttribute(
      "data-state",
      "open"
    );
    consoleGate.assertClean();
  });

  test("shows provider fallback without claiming preview or trace success", async ({
    page
  }) => {
    const consoleGate = installConsoleGate(page);
    await installApiMocks(page, "provider-fallback");
    await expectLoadedWorkstation(page);

    await submitCreativePrompt(
      page,
      "Create a compact p5 fallback sketch if the provider is unavailable."
    );

    await expect(
      page.getByLabel("Product outcome summary").getByText("Provider fallback completed")
    ).toBeVisible();
    await expect(page.getByRole("region", { name: "Preview workspace" })).toContainText(
      "Preview unavailable"
    );
    await expect(page.getByRole("region", { name: "Preview workspace" })).not.toContainText(
      "Preview ready"
    );
    consoleGate.assertClean();
  });

  for (const prompt of [
    "Create a space-colonization branching growth sketch with browser-safe p5 controls.",
    "Create a DLA or differential-growth visual study with simple p5 preview routing."
  ]) {
    test(`runs non-demo creative smoke: ${prompt.slice(0, 34)}`, async ({ page }) => {
      const consoleGate = installConsoleGate(page);
      await installApiMocks(page, "success");
      await expectLoadedWorkstation(page);

      await submitCreativePrompt(page, prompt);
      await expectGeneratedPreview(page);
      consoleGate.assertClean();
    });
  }
});
