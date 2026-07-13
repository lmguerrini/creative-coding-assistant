const { test, expect } = require("@playwright/test");
const {
  expectLoadedWorkstation,
  installApiMocks,
  installConsoleGate
} = require("./support/quality-gates");

const dashboardPages = [
  "Overview",
  "Architecture",
  "Workflow",
  "Workspace",
  "Runtime",
  "Preview",
  "Artifacts",
  "Domains",
  "Knowledge Base",
  "AI & agents",
  "Memory",
  "Sessions",
  "Telemetry",
  "Evaluation",
  "User Guide",
  "Settings"
];

const runnableDemoScenarios = [
  ["Polyrhythmic constellation", "single_agent"],
  ["Recursive aurora garden", "single_agent"],
  ["Kinetic orbit sculpture", "single_agent"],
  ["Fractal solar bloom", "single_agent"],
  ["Source-grounded design brief", "auto"],
  ["Multi-agent production plan", "multi_agent"],
  ["Single-agent line study", "single_agent"],
  ["Export handoff package", "auto"],
  ["Failure-recovery rehearsal", "auto"]
];

const homepagePromptTitles = [
  "Physarum drift",
  "Kinetic orbit sculpture",
  "Chladni light field",
  "Cymatic audio study"
];

test.describe("V9.7 Phase 3 product exploration", () => {
  test("reads every live Dashboard page and exposes the local Creative Knowledge inventory", async ({
    page
  }) => {
    const consoleGate = installConsoleGate(page);
    await installApiMocks(page, "success");

    await page.goto("/");
    await expectLoadedWorkstation(page);
    await page.getByRole("button", { name: "Open Product Intelligence Dashboard" }).click();

    const dashboard = page.getByRole("region", { name: "Advanced Dashboard" });
    const navigation = dashboard.getByRole("navigation", {
      name: "Dashboard categories"
    });
    await expect(dashboard).toBeVisible();

    for (const label of dashboardPages) {
      const pageButton = navigation.getByRole("button", { exact: true, name: label });
      await expect(pageButton).toHaveCount(1);
      await pageButton.click();
      await expect(dashboard.getByRole("heading", { exact: true, name: label })).toBeVisible();
    }

    await navigation.getByRole("button", { name: "Knowledge Base" }).click();
    const creativeKnowledge = dashboard.getByRole("region", {
      name: "Creative Knowledge Base"
    });
    await expect(creativeKnowledge).toBeVisible();
    await expect(
      creativeKnowledge
        .getByRole("list", { exact: true, name: "Featured creative knowledge records" })
        .locator(':scope > [role="listitem"]')
    ).toHaveCount(3);
    await creativeKnowledge.getByText("4 more creative knowledge records", { exact: true }).click();
    await expect(
      creativeKnowledge
        .getByRole("list", { exact: true, name: "Additional creative knowledge records" })
        .locator(':scope > [role="listitem"]')
    ).toHaveCount(4);
    await expect(creativeKnowledge).toContainText("Live visual runtime triage");
    await page.screenshot({
      fullPage: true,
      path: "../../.local/test-results/phase3-dashboard-knowledge.png"
    });

    consoleGate.assertClean();
  });

  test("runs every runnable Demo only after its selected detail is reviewed", async ({ page }) => {
    const consoleGate = installConsoleGate(page);
    await installApiMocks(page, "success");
    await expectLoadedWorkstation(page);

    for (const [title, workflowMode] of runnableDemoScenarios) {
      await page.getByRole("button", { name: "Demo Mode" }).click();
      const demoMode = page.getByRole("region", { name: "Demo Mode" });
      const scenarioList = demoMode.getByRole("list", { name: "Demo Mode scenarios" });
      const scenario = scenarioList.getByRole("button").filter({ hasText: title });
      await expect(scenario).toHaveCount(1);
      await scenario.click();

      const detail = demoMode.getByRole("article", { name: "Selected demo scenario" });
      await expect(detail).toContainText(title);
      const streamRequest = page.waitForRequest("**/api/assistant/stream");
      await detail.getByRole("button", { name: "Load prompt & run" }).click();
      const request = await streamRequest;

      expect(request.postDataJSON()).toMatchObject({ workflowMode });
      await expect(demoMode).toHaveCount(0);
      await expect(page.getByRole("log", { name: "Conversation" })).toHaveAttribute(
        "aria-busy",
        "false"
      );
    }

    await page.getByRole("button", { name: "Demo Mode" }).click();
    const demoMode = page.getByRole("region", { name: "Demo Mode" });
    const scenarioList = demoMode.getByRole("list", { name: "Demo Mode scenarios" });
    const referenceScenario = scenarioList
      .getByRole("button")
      .filter({ hasText: "Reference-guided palette study" });
    await expect(referenceScenario).toHaveCount(1);
    await referenceScenario.click();

    const referenceDetail = demoMode.getByRole("article", {
      name: "Selected demo scenario"
    });
    await expect(referenceDetail.getByRole("button", { name: "Attach image to run" })).toBeDisabled();
    await expect(referenceDetail).toContainText(
      "Add one image reference through the composer before running this demo."
    );

    await page.getByRole("button", { name: "Add attachment" }).click();
    await page.getByLabel("Upload image attachment").setInputFiles({
      buffer: Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
        "base64"
      ),
      mimeType: "image/png",
      name: "phase3-reference.png"
    });
    await expect(page.getByRole("region", { name: "Image references" })).toContainText(
      "phase3-reference.png"
    );

    const streamRequest = page.waitForRequest("**/api/assistant/stream");
    await referenceDetail.getByRole("button", { name: "Load prompt & run" }).click();
    const request = await streamRequest;
    expect(request.postDataJSON()).toMatchObject({
      attachments: [expect.objectContaining({ name: "phase3-reference.png" })],
      workflowMode: "single_agent"
    });
    await expect(demoMode).toHaveCount(0);

    consoleGate.assertClean();
  });

  test("loads and submits every homepage recommendation through the normal composer", async ({
    page
  }) => {
    const consoleGate = installConsoleGate(page);
    await installApiMocks(page, "success");

    for (const title of homepagePromptTitles) {
      await page.goto("/");
      await page.evaluate(() => window.localStorage.clear());
      await page.reload();
      await expectLoadedWorkstation(page);
      const composer = page.getByRole("textbox", { name: "Assistant prompt" });
      await page.getByRole("button", { name: title }).click();
      await expect(composer).not.toHaveValue("");

      const streamRequest = page.waitForRequest("**/api/assistant/stream");
      await page.getByRole("button", { name: "Send prompt" }).click();
      const request = await streamRequest;
      expect(request.postDataJSON()).toMatchObject({ workflowMode: "auto" });
      await expect(page.getByRole("log", { name: "Conversation" })).toHaveAttribute(
        "aria-busy",
        "false"
      );
    }

    consoleGate.assertClean();
  });

  for (const viewport of [
    { height: 980, name: "desktop", width: 1440 },
    { height: 900, name: "laptop", width: 1024 },
    { height: 900, name: "compact", width: 720 }
  ]) {
    test(`keeps the ordinary ${viewport.name} workspace free of horizontal overflow`, async ({
      page
    }) => {
      const consoleGate = installConsoleGate(page);
      await installApiMocks(page, "success");
      await page.setViewportSize({ height: viewport.height, width: viewport.width });
      await expectLoadedWorkstation(page);

      const layout = await page.evaluate(() => ({
        clientWidth: document.documentElement.clientWidth,
        scrollWidth: document.documentElement.scrollWidth,
        workspacePaddingBottom: Number.parseFloat(
          window.getComputedStyle(
            document.querySelector('[aria-label="Creative workspace"]')
          ).paddingBottom
        )
      }));
      expect(layout.scrollWidth).toBeLessThanOrEqual(layout.clientWidth);

      if (viewport.width >= 1024) {
        const workspace = page.getByRole("region", { name: "Creative workspace" });
        const session = page.getByRole("region", { name: "Creative session" });
        const sessions = page.getByRole("complementary", { name: "Sessions" });
        const inspector = page.getByRole("complementary", { name: "Right inspector" });
        const [workspaceBox, sessionBox, sessionsBox, inspectorBox] = await Promise.all([
          workspace.boundingBox(),
          session.boundingBox(),
          sessions.boundingBox(),
          inspector.boundingBox()
        ]);

        expect(sessionBox?.height).toBeGreaterThan(240);
        expect(sessionsBox?.height).toBeGreaterThan(240);
        expect(inspectorBox?.height).toBeGreaterThan(240);
        const workspaceContentBottom =
          (workspaceBox?.y ?? 0) +
          (workspaceBox?.height ?? 0) -
          layout.workspacePaddingBottom;
        expect(
          Math.abs((sessionBox?.y ?? 0) + (sessionBox?.height ?? 0) - workspaceContentBottom)
        ).toBeLessThanOrEqual(2);
        expect(
          Math.abs((sessionsBox?.y ?? 0) + (sessionsBox?.height ?? 0) - workspaceContentBottom)
        ).toBeLessThanOrEqual(2);
        expect(
          Math.abs((inspectorBox?.y ?? 0) + (inspectorBox?.height ?? 0) - workspaceContentBottom)
        ).toBeLessThanOrEqual(2);

        await page.getByRole("button", { name: "Collapse session sidebar" }).click();
        await expect(inspector).toHaveAttribute("data-state", "open");
        await expect(
          page.getByRole("button", { name: "Expand session sidebar" })
        ).toBeFocused();
        await page.getByRole("button", { name: "Expand session sidebar" }).click();
        await expect(
          page.getByRole("button", { name: "Collapse session sidebar" })
        ).toBeFocused();
      } else {
        const sessions = page.getByRole("complementary", { name: "Sessions" });
        const sessionList = sessions.getByRole("list", { name: "Saved sessions" });

        await expect(sessions).toBeVisible();
        await expect(sessionList).toBeVisible();
        await expect(
          page.getByRole("button", { name: "Collapse session sidebar" })
        ).toBeVisible();

        const compactRail = await sessions.evaluate((element) => {
          const list = element.querySelector('[aria-label="Saved sessions"]');
          const elementRect = element.getBoundingClientRect();
          return {
            listOverflowX: list ? window.getComputedStyle(list).overflowX : "missing",
            right: elementRect.right,
            viewportWidth: window.innerWidth
          };
        });
        expect(compactRail.listOverflowX).toBe("auto");
        expect(compactRail.right).toBeLessThanOrEqual(compactRail.viewportWidth);

        await page.getByRole("button", { name: "Collapse session sidebar" }).click();
        await expect(
          page.getByRole("button", { name: "Expand session sidebar" })
        ).toBeFocused();
        await expect(sessions).toContainText("Sessions");
        await page.getByRole("button", { name: "Expand session sidebar" }).click();
        await expect(
          page.getByRole("button", { name: "Collapse session sidebar" })
        ).toBeFocused();
      }

      await expectResponsiveRightInspector(page, viewport);

      consoleGate.assertClean();
    });
  }

  for (const viewport of [
    { height: 980, name: "desktop", width: 1440 },
    { height: 900, name: "laptop", width: 1024 },
    { height: 900, name: "compact", width: 720 }
  ]) {
    test(`keeps the canonical ${viewport.name} Dashboard responsive`, async ({ page }) => {
      const consoleGate = installConsoleGate(page);
      await installApiMocks(page, "success");
      await page.setViewportSize({ height: viewport.height, width: viewport.width });
      await expectLoadedWorkstation(page);
      await page.getByRole("button", { name: "Open Product Intelligence Dashboard" }).click();

      const dashboard = page.getByRole("region", { name: "Advanced Dashboard" });
      const navigation = dashboard.getByRole("navigation", { name: "Dashboard categories" });

      for (const label of ["User Guide", "Settings"]) {
        await navigation.getByRole("button", { exact: true, name: label }).click();
        await expect(dashboard.getByRole("heading", { exact: true, name: label })).toBeVisible();
        const layout = await page.evaluate(() => {
          const dashboardElement = document.querySelector('[aria-label="Advanced Dashboard"]');
          const content = dashboardElement?.querySelector(".productDashboardContent");
          return {
            contentClientWidth: content?.clientWidth ?? 0,
            contentScrollWidth: content?.scrollWidth ?? 0,
            documentClientWidth: document.documentElement.clientWidth,
            documentScrollWidth: document.documentElement.scrollWidth
          };
        });

        expect(layout.documentScrollWidth).toBeLessThanOrEqual(layout.documentClientWidth);
        expect(layout.contentScrollWidth).toBeLessThanOrEqual(layout.contentClientWidth);
      }

      consoleGate.assertClean();
    });
  }
});

async function expectResponsiveRightInspector(page, viewport) {
  const inspector = page.getByRole("complementary", { name: "Right inspector" });
  await expect(inspector).toHaveAttribute("data-state", "open");

  const openGeometry = await inspector.evaluate((element) => {
    const rect = element.getBoundingClientRect();
    return {
      clientWidth: element.clientWidth,
      height: rect.height,
      left: rect.left,
      right: rect.right,
      scrollWidth: element.scrollWidth,
      viewportWidth: window.innerWidth,
      width: rect.width
    };
  });
  expect(openGeometry.width).toBeGreaterThan(240);
  expect(openGeometry.height).toBeGreaterThan(240);
  expect(openGeometry.left).toBeGreaterThanOrEqual(0);
  expect(openGeometry.right).toBeLessThanOrEqual(openGeometry.viewportWidth + 1);
  expect(openGeometry.scrollWidth).toBeLessThanOrEqual(openGeometry.clientWidth);
  if (viewport.width < 1024) {
    expect(openGeometry.height).toBeLessThanOrEqual(
      Math.min(640, viewport.height * 0.7) + 1
    );
  } else {
    expect(openGeometry.height).toBeLessThanOrEqual(viewport.height - 84 + 1);
  }

  const tablist = inspector.getByRole("tablist", { name: "Inspector tabs" });
  const tabs = tablist.getByRole("tab");
  const firstTab = tabs.first();
  const lastTab = tabs.last();
  expect(await tabs.count()).toBeGreaterThan(2);

  await firstTab.focus();
  await firstTab.press("End");
  await expect(lastTab).toBeFocused();
  await expect(lastTab).toHaveAttribute("aria-selected", "true");
  await lastTab.press("Home");
  await expect(firstTab).toBeFocused();
  await firstTab.press("ArrowLeft");
  await expect(lastTab).toBeFocused();
  await lastTab.press("ArrowRight");
  await expect(firstTab).toBeFocused();
  await expect(firstTab).toHaveAttribute("aria-selected", "true");

  const addButton = inspector.getByRole("button", { name: "Add Inspector tab" });
  await addButton.click();
  const addMenu = inspector.getByRole("menu", { name: "Available Inspector tabs" });
  await expect(addMenu).toBeVisible();
  await expect(addMenu.getByRole("menuitem").first()).toBeFocused();

  const menuGeometry = await addMenu.evaluate((element) => {
    const rect = element.getBoundingClientRect();
    return {
      bottom: rect.bottom,
      left: rect.left,
      right: rect.right,
      top: rect.top,
      viewportHeight: window.innerHeight,
      viewportWidth: window.innerWidth
    };
  });
  expect(menuGeometry.left).toBeGreaterThanOrEqual(0);
  expect(menuGeometry.right).toBeLessThanOrEqual(menuGeometry.viewportWidth + 1);
  expect(menuGeometry.top).toBeGreaterThanOrEqual(0);
  expect(menuGeometry.bottom).toBeLessThanOrEqual(menuGeometry.viewportHeight + 1);

  await addMenu.getByRole("menuitem").first().press("Escape");
  await expect(addMenu).toHaveCount(0);
  await expect(addButton).toBeFocused();

  await inspector.getByRole("button", { name: "Collapse inspector" }).click();
  await expect(inspector).toHaveAttribute("data-state", "collapsed");
  const expandButton = inspector.getByRole("button", { name: "Expand inspector" });
  await expect(expandButton).toBeFocused();

  const collapsedGeometry = await inspector.evaluate((element) => {
    const rect = element.getBoundingClientRect();
    return {
      height: rect.height,
      left: rect.left,
      right: rect.right,
      viewportWidth: window.innerWidth,
      width: rect.width
    };
  });
  expect(collapsedGeometry.left).toBeGreaterThanOrEqual(0);
  expect(collapsedGeometry.right).toBeLessThanOrEqual(
    collapsedGeometry.viewportWidth + 1
  );
  if (viewport.width < 1024) {
    expect(collapsedGeometry.height).toBeLessThanOrEqual(59);
  } else {
    expect(collapsedGeometry.width).toBeLessThanOrEqual(72);
  }

  await expandButton.click();
  await expect(inspector).toHaveAttribute("data-state", "open");
  await expect(
    inspector.getByRole("button", { name: "Collapse inspector" })
  ).toBeFocused();
}
