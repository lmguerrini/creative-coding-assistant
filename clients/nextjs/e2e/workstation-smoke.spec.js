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
      .getByRole("button", { name: /p5\.js Generative Morphogenesis Sketch/ })
      .click();

    await expect(demoMode).toContainText("Prompt loaded");
    await expect(page.getByRole("textbox", { name: "Assistant prompt" })).toHaveValue(
      /reaction diffusion/
    );
    await expect(demoMode).not.toContainText(/HoloGenesis/i);
    await expect(demoMode).not.toContainText(/\bsacred\b/i);
    consoleGate.assertClean();
  });
});
