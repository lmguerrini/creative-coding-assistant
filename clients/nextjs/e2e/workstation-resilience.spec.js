const { test, expect } = require("@playwright/test");
const {
  expectLoadedWorkstation,
  installApiMocks,
  installConsoleGate,
  submitCreativePrompt
} = require("./support/quality-gates");

test.describe("V7.4 workstation resilience", () => {
  test("shows a local draft when the assistant stream fails without browser console errors", async ({
    page
  }) => {
    const consoleGate = installConsoleGate(page);
    await installApiMocks(page, "failure");
    await expectLoadedWorkstation(page);

    await submitCreativePrompt(page, "Force a recoverable stream failure.");

    await expect(page.getByRole("log", { name: "Conversation" })).toContainText(
      "The live response could not complete"
    );
    await expect(page.getByRole("log", { name: "Conversation" })).toContainText(
      "Retry from the composer"
    );
    await page.getByRole("textbox", { name: "Assistant prompt" }).fill("Retry after failure.");
    await expect(page.getByRole("button", { name: "Send prompt" })).toBeEnabled();
    consoleGate.assertClean();
  });

  test("keeps provider fallback recoverable and avoids failure-node regression", async ({
    page
  }) => {
    const consoleGate = installConsoleGate(page);
    await installApiMocks(page, "provider-fallback");
    await expectLoadedWorkstation(page);

    await submitCreativePrompt(
      page,
      "Use provider fallback if the primary model is unavailable."
    );

    await expect(page.getByRole("log", { name: "Conversation" })).toContainText(
      "Provider fallback completed"
    );
    await expect(page.getByLabel("Current session")).not.toContainText("Failure");
    consoleGate.assertClean();
  });

  test("handles a longer creative session without unbounded local storage growth", async ({
    page
  }) => {
    const consoleGate = installConsoleGate(page);
    await installApiMocks(page, "long");
    await expectLoadedWorkstation(page);

    for (const prompt of [
      "Start a long-session orbit field.",
      "Refine the color field while preserving the p5 runtime.",
      "Add a calmer motion layer without changing preview routing."
    ]) {
      await submitCreativePrompt(page, prompt);
      await expect(page.getByRole("log", { name: "Conversation" })).toContainText(
        "Generated the E2E p5 orbit sketch"
      );
    }

    const storageSize = await page.evaluate(() =>
      JSON.stringify(window.localStorage).length
    );
    expect(storageSize).toBeLessThan(200_000);
    consoleGate.assertClean();
  });
});
