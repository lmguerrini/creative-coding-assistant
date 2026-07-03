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
});
