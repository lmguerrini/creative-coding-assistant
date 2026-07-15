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

test.describe("Workstation E2E smoke", () => {
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

    const expandInspector = page.getByRole("button", { name: "Expand inspector" });
    if (await expandInspector.isVisible().catch(() => false)) {
      await expandInspector.click();
    }
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

    await page.getByRole("button", { name: "Physarum drift" }).click();
    await expect(page.getByRole("textbox", { name: "Assistant prompt" })).toHaveValue(
      /physarum-drift\.p5\.js/
    );
    await expect(page.getByRole("textbox", { name: "Assistant prompt" })).toHaveValue(
      /pointer attraction/
    );
    await page.getByRole("button", { name: "Send prompt" }).click();

    await expectGeneratedPreview(page, {
      artifactTitle: "physarum-drift-2.p5.js"
    });
    await page.getByRole("tab", { name: "Preview" }).click();
    await expect(page.getByRole("tabpanel", { exact: true, name: "Preview" })).not.toContainText(
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
    await expect(page.getByRole("tabpanel", { exact: true, name: "Preview" })).not.toContainText(
      "No matching live renderer"
    );
    consoleGate.assertClean();
  });

  test("mounts the Cymatics demo muted with an explicit Tone.js start control", async ({
    page
  }) => {
    const consoleGate = installConsoleGate(page);
    await installApiMocks(page, "cymatics");
    await expectLoadedWorkstation(page);

    await submitCreativePrompt(
      page,
      "Create exactly one executable .tone.js artifact named cymatic-chladni.tone.js."
    );

    await expect(page.getByRole("region", { name: "Preview workspace" })).toBeVisible();
    const expandInspector = page.getByRole("button", { name: "Expand inspector" });
    if (await expandInspector.isVisible().catch(() => false)) {
      await expandInspector.click();
    }
    await page.getByRole("tab", { name: "Preview" }).click();
    await expect(page.getByRole("tabpanel", { exact: true, name: "Preview" })).toContainText(
      "Tone.js audio surface"
    );
    const runtime = page.getByRole("group", { name: "Tone.js live runtime" });
    await expect(runtime).toHaveAttribute("data-runtime-state", "ready");
    await expect(runtime).toContainText("Audio remains silent until Start audio is selected.");
    const frame = page.frameLocator('iframe[title="Tone.js preview runtime"]');
    await expect(frame.locator("canvas")).toBeVisible();
    await expect(frame.getByRole("button", { name: "Start audio" })).toBeVisible();
    consoleGate.assertClean();
  });

  test("inspects a curated Demo Mode prompt before explicitly loading and running it", async ({
    page
  }) => {
    const consoleGate = installConsoleGate(page);
    await installApiMocks(page, "success");
    await expectLoadedWorkstation(page);

    await page.getByRole("button", { name: "Demo Mode" }).click();
    const demoMode = page.getByRole("region", { name: "Demo Mode" });
    const demoScenarios = demoMode.getByRole("list", { name: "Demo Mode scenarios" });
    await expect(demoMode).toBeVisible();
    await expect(demoMode).toContainText("Creative scenarios");

    await demoScenarios
      .getByRole("button", { name: /Recursive aurora garden/ })
      .click();

    await expect(demoMode).toBeVisible();
    const selectedScenario = demoMode.getByRole("article", {
      name: "Selected demo scenario"
    });
    await expect(selectedScenario).toContainText("Recursive aurora garden");
    await selectedScenario.getByRole("button", { name: "Load prompt & run" }).click();

    await expect(demoMode).toHaveCount(0);
    await expect(page.getByRole("log", { name: "Conversation" })).toContainText(
      /recursive-aurora-garden\.p5\.js/
    );
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
    const demoScenarios = demoMode.getByRole("list", { name: "Demo Mode scenarios" });
    await expect(demoMode).toBeVisible();
    await expect(page.getByRole("complementary", { name: "Right inspector" })).toHaveAttribute(
      "data-state",
      "open"
    );
    await expect(page.getByRole("tab", { name: "Overview" })).toHaveAttribute(
      "aria-selected",
      "true"
    );

    await expect(demoMode).toContainText("10 flows");
    await demoScenarios
      .getByRole("button", { name: /Recursive aurora garden/ })
      .click();
    await demoMode
      .getByRole("article", { name: "Selected demo scenario" })
      .getByRole("button", { name: "Load prompt & run" })
      .click();
    await expect(demoMode).toHaveCount(0);
    await expectGeneratedPreview(page, {
      artifactTitle: "recursive-aurora-garden-2.p5.js"
    });

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
      .getByRole("list", { name: "Demo Mode scenarios" })
      .getByRole("button", { name: /Recursive aurora garden/ })
      .click();
    await page.getByRole("button", { exact: true, name: "Settings" }).click();
    const displayMode = page.getByRole("button", { name: "Display mode" });
    await expect(displayMode).toContainText(
      "Developer"
    );
    await displayMode.click();
    await expect(displayMode).toContainText("User");
    await page.getByRole("button", { exact: true, name: "Settings" }).click();
    await expect(page.getByRole("complementary", { name: "Right inspector" })).toHaveAttribute(
      "data-state",
      "collapsed"
    );

    await page.getByRole("button", { name: "Expand inspector" }).click();
    await expect(page.getByRole("tab", { name: "Preview" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "Code" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "Saved" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "Runtime" })).toHaveCount(0);

    await page.getByRole("button", { exact: true, name: "Settings" }).click();
    await displayMode.click();
    await expect(displayMode).toContainText(
      "Developer"
    );
    await page.getByRole("button", { exact: true, name: "Settings" }).click();
    await expect(page.getByRole("tab", { name: "Workflow" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "Telemetry" })).toBeVisible();

    await page.getByRole("button", { name: "Theme" }).click();
    await page.getByRole("button", { name: "Use Deep Blue theme" }).click();
    await expect(page.locator(".workstation")).toHaveAttribute("data-theme", "codex");

    await page.getByRole("button", { exact: true, name: "Settings" }).click();
    await page.getByRole("button", { name: "Clear workspace session" }).click();
    await page.getByRole("button", { name: "Clear workspace" }).click();

    await expect(page.getByRole("region", { name: "Demo Mode" })).toHaveCount(0);
    await expect(page.getByRole("textbox", { name: "Assistant prompt" })).toHaveValue("");
    await expect(page.getByRole("region", { name: "Preview workspace" })).toHaveCount(0);
    await expect(page.getByRole("group", { name: "Empty creative workspace" })).toBeVisible();
    await page.getByRole("button", { exact: true, name: "Settings" }).click();
    await expect(page.getByRole("button", { name: "Display mode" })).toContainText(
      "Developer"
    );
    await expect(page.getByRole("complementary", { name: "Right inspector" })).toHaveAttribute(
      "data-state",
      "open"
    );
    await expect(page.getByRole("tab", { name: "Overview" })).toHaveAttribute(
      "aria-selected",
      "true"
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

    await expect(page.getByLabel("Current session", { exact: true })).toContainText(
      "Provider fallback completed"
    );
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
