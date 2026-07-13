const { test, expect } = require("@playwright/test");
const {
  expectLoadedWorkstation,
  installApiMocks,
  installConsoleGate
} = require("./support/quality-gates");

const floatingSurfaceViewports = [
  { height: 900, label: "desktop", width: 1440 },
  { height: 768, label: "laptop", width: 1024 },
  { height: 900, label: "compact", width: 720 }
];

const visualStates = [
  { mode: "Developer", themeLabel: "Deep Blue", themeValue: "codex" },
  { mode: "User", themeLabel: "Deep Blue", themeValue: "codex" },
  { mode: "User", themeLabel: "Light", themeValue: "light" },
  { mode: "Developer", themeLabel: "Light", themeValue: "light" }
];

test.describe("Application floating-surface design-system boundary", () => {
  for (const viewport of floatingSurfaceViewports) {
    test(`${viewport.label} keeps shared panels and the operator checkpoint coherent`, async ({
      page
    }) => {
      test.setTimeout(90_000);
      await page.setViewportSize({ height: viewport.height, width: viewport.width });
      const consoleGate = installConsoleGate(page);
      await installApiMocks(page, "success");
      await expectLoadedWorkstation(page);

      for (const state of visualStates) {
        await ensureVisualState(page, state);
        await expectThemePanelContract(page, viewport, state);
        await expectSettingsPanelContract(page, viewport, state);
      }

      if (viewport.label === "desktop") {
        await expectDashboardSettingsBoundary(page);
      }
      await expectPristineWorkspaceCheckpoint(page, viewport);

      consoleGate.assertClean();
    });
  }

  test("the shared confirmation modal cancels and confirms across dark and light themes", async ({
    page
  }) => {
    await page.setViewportSize({ height: 900, width: 1440 });
    const consoleGate = installConsoleGate(page);
    await installApiMocks(page, "success");
    await installWorkspaceDeleteMock(page);
    await expectLoadedWorkstation(page);

    await ensureVisualState(page, visualStates[0]);
    const darkSurface = await openSessionDeleteConfirmation(page);
    await expectConfirmationFocusTrap(page, darkSurface.dialog);
    await page.keyboard.press("Escape");
    await expect(darkSurface.dialog).toHaveCount(0);
    await expect(darkSurface.origin).toBeFocused();
    await expect.poll(() => page.evaluate(() => document.body.style.overflow)).toBe("");

    const cancelSurface = await openSessionDeleteConfirmation(page);
    await cancelSurface.dialog.getByRole("button", { name: "Keep session" }).click();
    await expect(cancelSurface.dialog).toHaveCount(0);
    await expect(cancelSurface.origin).toBeFocused();

    await ensureVisualState(page, visualStates[2]);
    const lightSurface = await openSessionDeleteConfirmation(page);
    expect(lightSurface.colors.backgroundColor).not.toBe(
      darkSurface.colors.backgroundColor
    );
    expect(lightSurface.colors.color).not.toBe(darkSurface.colors.color);
    await lightSurface.dialog.getByRole("button", { name: "Delete session" }).click();
    await expect(lightSurface.dialog).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Settings" })).toBeFocused();
    await expect(page.locator(".sessionSidebarActions button[aria-label^='Delete ']"))
      .toHaveCount(1);

    consoleGate.assertClean();
  });
});

async function ensureVisualState(page, state) {
  const workstation = page.locator(".workstation");
  const composer = page.getByRole("form", { name: "Creative request composer" });

  if ((await workstation.getAttribute("data-theme")) !== state.themeValue) {
    const trigger = page.getByRole("button", { name: "Theme" });
    await trigger.click();
    const panel = page.getByRole("dialog", { name: "Theme presets" });
    await expect(panel).toBeVisible();
    await panel
      .getByRole("button", { name: `Use ${state.themeLabel} theme` })
      .click();
    await expect(panel).toHaveCount(0);
    await expect(trigger).toBeFocused();
  }

  if ((await composer.getAttribute("data-mode")) !== state.mode.toLowerCase()) {
    const trigger = page.getByRole("button", { name: "Settings" });
    await trigger.click();
    const panel = page.getByRole("dialog", { name: "Workspace settings" });
    await expect(panel).toBeVisible();
    const displayMode = panel.getByRole("button", { name: "Display mode" });
    await displayMode.click();
    await expect(displayMode).toHaveText(state.mode);
    await panel.press("Escape");
    await expect(panel).toHaveCount(0);
    await expect(trigger).toBeFocused();
  }

  await expect(workstation).toHaveAttribute("data-theme", state.themeValue);
  await expect(page.locator("html")).toHaveAttribute("data-cca-theme", state.themeValue);
  await expect(composer).toHaveAttribute("data-mode", state.mode.toLowerCase());
}

async function installWorkspaceDeleteMock(page) {
  await page.route("**/api/workspace/session**", async (route, request) => {
    if (request.method() !== "DELETE") {
      await route.fallback();
      return;
    }
    await route.fulfill({
      body: JSON.stringify({ ok: true, target: "e2e-memory" }),
      contentType: "application/json",
      headers: {
        "Access-Control-Allow-Headers": "Accept, Content-Type",
        "Access-Control-Allow-Methods": "DELETE, GET, OPTIONS, POST",
        "Access-Control-Allow-Origin": "*"
      },
      status: 200
    });
  });
}

async function expectThemePanelContract(page, viewport, state) {
  const trigger = page.getByRole("button", { name: "Theme" });
  await trigger.focus();
  await trigger.click();

  const panel = page.getByRole("dialog", { name: "Theme presets" });
  await expect(panel).toBeVisible();
  await expect(panel).toBeFocused();
  await expect(panel).toContainText(
    "Switch the workspace accent and shell tone without changing the layout."
  );
  await expect(
    panel.getByRole("button", { name: `Use ${state.themeLabel} theme` })
  ).toHaveAttribute("aria-pressed", "true");
  await expectFloatingPanelGeometry(panel, viewport, { mustScroll: false });

  await panel.press("Escape");
  await expect(panel).toHaveCount(0);
  await expect(trigger).toHaveAttribute("aria-expanded", "false");
  await expect(trigger).toBeFocused();
}

async function expectSettingsPanelContract(page, viewport, state) {
  const trigger = page.getByRole("button", { name: "Settings" });
  await trigger.focus();
  await trigger.click();

  const panel = page.getByRole("dialog", { name: "Workspace settings" });
  await expect(panel).toBeVisible();
  await expect(panel).toBeFocused();
  await expect(panel).toContainText("Compact controls for the active creative session.");
  await expect(panel).toContainText(
    "Adjust the compact cockpit without duplicating the complete Dashboard reference."
  );
  await expect(
    panel.getByRole("button", { name: /Open Dashboard Settings/ })
  ).toBeVisible();
  await expect(panel).toContainText(
    "Complete appearance, typography, privacy, and prompt defaults."
  );
  await expect(panel.getByRole("button", { name: /^Use .* theme$/ })).toHaveCount(0);
  await expect(panel).not.toContainText("Comfortable reading");
  await expect(panel).not.toContainText("Colour themes");

  const displayMode = panel.getByRole("button", { name: "Display mode" });
  await expect(displayMode).toHaveText(state.mode);
  await expect(displayMode).toHaveAttribute(
    "aria-pressed",
    state.mode === "Developer" ? "true" : "false"
  );
  await expectFloatingPanelGeometry(panel, viewport, { mustScroll: true });

  await panel.press("Escape");
  await expect(panel).toHaveCount(0);
  await expect(trigger).toHaveAttribute("aria-expanded", "false");
  await expect(trigger).toBeFocused();
}

async function expectFloatingPanelGeometry(panel, viewport, { mustScroll }) {
  const geometry = await panel.evaluate((element) => {
    const rect = element.getBoundingClientRect();
    const style = window.getComputedStyle(element);
    const clippingAncestors = [];
    let ancestor = element.parentElement;

    while (ancestor) {
      const ancestorStyle = window.getComputedStyle(ancestor);
      const clipsX = ancestorStyle.overflowX !== "visible";
      const clipsY = ancestorStyle.overflowY !== "visible";
      if (clipsX || clipsY) {
        const ancestorRect = ancestor.getBoundingClientRect();
        clippingAncestors.push({
          bottom: ancestorRect.bottom,
          clipsX,
          clipsY,
          left: ancestorRect.left,
          right: ancestorRect.right,
          top: ancestorRect.top
        });
      }
      ancestor = ancestor.parentElement;
    }

    return {
      backgroundColor: style.backgroundColor,
      bottom: rect.bottom,
      clientHeight: element.clientHeight,
      clientWidth: element.clientWidth,
      clippingAncestors,
      display: style.display,
      left: rect.left,
      overflowY: style.overflowY,
      right: rect.right,
      scrollHeight: element.scrollHeight,
      scrollWidth: element.scrollWidth,
      top: rect.top,
      visibility: style.visibility,
      width: rect.width
    };
  });

  expect(geometry.display).not.toBe("none");
  expect(geometry.visibility).toBe("visible");
  expect(geometry.backgroundColor).not.toBe("rgba(0, 0, 0, 0)");
  expect(geometry.backgroundColor).not.toBe("transparent");
  expect(geometry.width).toBeGreaterThan(260);
  expect(geometry.left).toBeGreaterThanOrEqual(0);
  expect(geometry.right).toBeLessThanOrEqual(viewport.width + 1);
  expect(geometry.top).toBeGreaterThanOrEqual(0);
  expect(geometry.bottom).toBeLessThanOrEqual(viewport.height + 1);
  expect(geometry.overflowY).toBe("auto");
  expect(geometry.scrollWidth).toBeLessThanOrEqual(geometry.clientWidth + 1);
  if (mustScroll) {
    expect(geometry.scrollHeight).toBeGreaterThan(geometry.clientHeight + 1);
    await panel.evaluate((element) => {
      element.scrollTop = element.scrollHeight;
    });
    await expect.poll(() => panel.evaluate((element) => element.scrollTop)).toBeGreaterThan(0);
    await panel.evaluate((element) => {
      element.scrollTop = 0;
    });
  }

  for (const ancestor of geometry.clippingAncestors) {
    if (ancestor.clipsX) {
      expect.soft(geometry.left).toBeGreaterThanOrEqual(ancestor.left - 1);
      expect.soft(geometry.right).toBeLessThanOrEqual(ancestor.right + 1);
    }
    if (ancestor.clipsY) {
      expect.soft(geometry.top).toBeGreaterThanOrEqual(ancestor.top - 1);
      expect.soft(geometry.bottom).toBeLessThanOrEqual(ancestor.bottom + 1);
    }
  }
}

async function expectDashboardSettingsBoundary(page) {
  const settingsTrigger = page.getByRole("button", { name: "Settings" });
  await settingsTrigger.click();
  const panel = page.getByRole("dialog", { name: "Workspace settings" });
  await panel.getByRole("button", { name: /Open Dashboard Settings/ }).click();

  await expect(panel).toHaveCount(0);
  const dashboard = page.getByRole("region", { name: "Advanced Dashboard" });
  await expect(dashboard).toBeVisible();
  await expect(dashboard.getByRole("heading", { level: 1, name: "Settings" })).toBeVisible();
  await expect(dashboard.getByRole("region", { name: "Dashboard settings" })).toBeVisible();

  await dashboard.getByRole("button", { name: "Close dashboard" }).click();
  await expect(dashboard).toHaveCount(0);
  await expect(page.getByRole("region", { name: "Creative workspace" })).toBeVisible();
}

async function expectPristineWorkspaceCheckpoint(page, viewport) {
  const session = page.getByRole("region", { name: "Creative session" });
  await expect(session).toHaveAttribute("data-homepage", "true");

  const settingsTrigger = page.getByRole("button", { name: "Settings" });
  await settingsTrigger.click();
  const settings = page.getByRole("dialog", { name: "Workspace settings" });
  await settings.getByRole("button", { name: "Clear workspace session" }).click();
  await expect(settings).toHaveCount(0);

  const checkpoint = page.locator(
    ".workspaceConversationFrame > .operatorCheckpoint"
  );
  await expect(checkpoint).toBeVisible();
  await expect(checkpoint).toBeFocused();
  await expect(checkpoint).toHaveAttribute("aria-label", "Operator checkpoint");
  await expect(checkpoint).toHaveAttribute("data-kind", "destructive");
  await expect(checkpoint).toHaveAttribute("data-state", "pending_approval");
  await expect(checkpoint).toContainText("Clear workspace session");
  await expect(checkpoint).toContainText("while keeping layout and theme preferences");
  await expectCheckpointGeometry(page, checkpoint, viewport);

  await checkpoint.getByRole("button", { name: "Keep session" }).click();
  await expect(checkpoint).toHaveAttribute("data-state", "rejected");
  await expect(checkpoint).toBeFocused();
  await checkpoint
    .getByRole("button", { name: "Dismiss operator checkpoint" })
    .click();
  await expect(checkpoint).toHaveCount(0);
  await expect(settingsTrigger).toBeFocused();
  await expect(session).toHaveAttribute("data-homepage", "true");
}

async function expectCheckpointGeometry(page, checkpoint, viewport) {
  const geometry = await checkpoint.evaluate((element) => {
    const rect = element.getBoundingClientRect();
    const style = window.getComputedStyle(element);
    return {
      backgroundColor: style.backgroundColor,
      bottom: rect.bottom,
      clientWidth: element.clientWidth,
      display: style.display,
      left: rect.left,
      right: rect.right,
      scrollWidth: element.scrollWidth,
      top: rect.top,
      visibility: style.visibility,
      width: rect.width
    };
  });

  expect(geometry.display).not.toBe("none");
  expect(geometry.visibility).toBe("visible");
  expect(geometry.backgroundColor).not.toBe("rgba(0, 0, 0, 0)");
  expect(geometry.width).toBeGreaterThan(240);
  expect(geometry.left).toBeGreaterThanOrEqual(0);
  expect(geometry.right).toBeLessThanOrEqual(viewport.width + 1);
  expect(geometry.top).toBeGreaterThanOrEqual(0);
  const layoutGeometry = await page.evaluate(() => {
    const read = (selector) => {
      const element = document.querySelector(selector);
      if (!(element instanceof HTMLElement)) {
        return null;
      }
      const rect = element.getBoundingClientRect();
      const style = window.getComputedStyle(element);
      return {
        alignItems: style.alignItems,
        alignSelf: style.alignSelf,
        bottom: rect.bottom,
        clientHeight: element.clientHeight,
        display: style.display,
        gridAutoRows: style.gridAutoRows,
        gridRowEnd: style.gridRowEnd,
        gridRowStart: style.gridRowStart,
        gridTemplateRows: style.gridTemplateRows,
        height: rect.height,
        maxHeight: style.maxHeight,
        minHeight: style.minHeight,
        order: style.order,
        overflowY: style.overflowY,
        position: style.position,
        scrollHeight: element.scrollHeight,
        top: rect.top
      };
    };
    return {
      composer: read(".workspaceComposer"),
      conversation: read(".workspaceConversation[data-empty='true']"),
      emptyState: read(".emptyWorkspace"),
      frame: read(".workspaceConversationFrame"),
      intro: read(".sessionIntro"),
      panel: read(".sessionPanel")
    };
  });
  expect(
    geometry.bottom,
    `Checkpoint must remain inside the viewport: ${JSON.stringify({
      checkpoint: geometry,
      layout: layoutGeometry,
      viewport
    })}`
  ).toBeLessThanOrEqual(viewport.height + 1);
  expect(geometry.scrollWidth).toBeLessThanOrEqual(geometry.clientWidth + 1);

  const documentGeometry = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth
  }));
  expect(documentGeometry.scrollWidth).toBeLessThanOrEqual(documentGeometry.clientWidth);
}

async function openSessionDeleteConfirmation(page) {
  const origin = page.locator(
    ".sessionSidebarActions button[aria-label^='Delete ']"
  );
  await expect(origin).toHaveCount(1);
  await origin.focus();
  await origin.click();

  const dialog = page.getByRole("alertdialog");
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText("This permanently removes the browser-local session");
  await expect(dialog.getByRole("button", { name: "Keep session" })).toBeFocused();
  await expect(page.locator(".applicationModalBackdrop")).toHaveAttribute(
    "data-tone",
    "danger"
  );
  await expect.poll(() => page.evaluate(() => document.body.style.overflow)).toBe("hidden");

  const colors = await expectConfirmationGeometry(page, dialog);
  return { colors, dialog, origin };
}

async function expectConfirmationFocusTrap(page, dialog) {
  const cancel = dialog.getByRole("button", { name: "Keep session" });
  const confirm = dialog.getByRole("button", { name: "Delete session" });

  await expect(cancel).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(confirm).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(cancel).toBeFocused();
}

async function expectConfirmationGeometry(page, dialog) {
  const geometry = await dialog.evaluate((element) => {
    const rect = element.getBoundingClientRect();
    const style = window.getComputedStyle(element);
    return {
      backgroundColor: style.backgroundColor,
      bottom: rect.bottom,
      clientWidth: element.clientWidth,
      color: style.color,
      left: rect.left,
      right: rect.right,
      scrollWidth: element.scrollWidth,
      top: rect.top,
      width: rect.width
    };
  });
  const viewport = await page.evaluate(() => ({
    height: window.innerHeight,
    width: window.innerWidth
  }));

  expect(geometry.backgroundColor).not.toBe("rgba(0, 0, 0, 0)");
  expect(geometry.backgroundColor).not.toBe("transparent");
  expect(geometry.width).toBeGreaterThan(320);
  expect(geometry.width).toBeLessThanOrEqual(480);
  expect(geometry.left).toBeGreaterThanOrEqual(0);
  expect(geometry.right).toBeLessThanOrEqual(viewport.width + 1);
  expect(geometry.top).toBeGreaterThanOrEqual(0);
  expect(geometry.bottom).toBeLessThanOrEqual(viewport.height + 1);
  expect(geometry.scrollWidth).toBeLessThanOrEqual(geometry.clientWidth + 1);

  return {
    backgroundColor: geometry.backgroundColor,
    color: geometry.color
  };
}
