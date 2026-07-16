const { test, expect } = require("@playwright/test");
const {
  expectLoadedWorkstation,
  installApiMocks,
  installConsoleGate,
  submitCreativePrompt
} = require("./support/quality-gates");

test.describe("Workstation resilience", () => {
  test("keeps a delayed Session A completion out of a newly created Session B", async ({
    page
  }) => {
    const consoleGate = installConsoleGate(page);
    await installApiMocks(page, "success");
    const persistenceSaves = await observeWorkspaceSessionSaves(page);
    const delayedStream = await installDelayedAssistantStream(page);
    const assistantCancellation = page.waitForEvent("requestfailed", {
      predicate: (request) =>
        request.method() === "POST" &&
        request.url().includes("/api/assistant/stream")
    });
    await expectLoadedWorkstation(page);

    await submitCreativePrompt(page, delayedSessionPrompt);
    const sessionARequest = await delayedStream.waitUntilBlocked;
    const sessionAId = sessionARequest.conversationId;

    await expect(
      page.getByRole("form", { name: "Creative request composer" })
    ).toHaveAttribute("aria-busy", "true");
    await page.getByRole("button", { name: "New session" }).click();

    await expect(
      page.getByRole("group", { name: "Empty creative workspace" })
    ).toBeVisible();
    await expect(page.getByLabel("Current session", { exact: true })).toContainText(
      "Ready"
    );
    const sessionBSave = await expectSessionBSave(persistenceSaves, sessionAId);
    const sessionBId = sessionBSave.sessionId;
    const sessionBCard = page
      .getByRole("list", { name: "Saved sessions" })
      .getByRole("button")
      .filter({ hasText: "New creative session" });
    await expect(sessionBCard).toHaveAttribute("aria-current", "true");

    const beforeRelease = await readSessionIsolationState(
      page,
      sessionBSave.userId,
      sessionBId
    );
    const sessionBSavesBeforeRelease = persistenceSaves.filter(
      (record) => record.sessionId === sessionBId
    ).length;

    delayedStream.release();
    await delayedStream.waitUntilReleased;
    const canceledRequest = await assistantCancellation;
    expect(canceledRequest.failure()?.errorText).toMatch(/ERR_ABORTED|aborted/i);

    await expect(page.getByText(delayedSessionPrompt)).toHaveCount(0);
    await expect(page.getByText(delayedSessionAnswer)).toHaveCount(0);
    await expect(page.getByText(delayedSessionArtifact.title)).toHaveCount(0);
    await expect(page.locator(".workstation")).toHaveAttribute(
      "data-stream-state",
      "idle"
    );
    await expect(page.getByLabel("Current session", { exact: true })).toContainText(
      "Ready"
    );
    await expect(page.getByLabel("Current session", { exact: true })).toContainText(
      "Tokens and estimated cost pending"
    );

    const afterRelease = await readSessionIsolationState(
      page,
      sessionBSave.userId,
      sessionBId
    );
    expect(afterRelease).toEqual(beforeRelease);
    expect(afterRelease.record.messages).toEqual([]);
    expect(afterRelease.record.artifacts).toEqual([]);
    expect(afterRelease.record.workflow.status).toBe("Idle");
    expect(afterRelease.record.snapshot.messages).toEqual([]);
    expect(afterRelease.record.snapshot.artifacts).toEqual([]);
    expect(afterRelease.record.snapshot.workflow.status).toBe("Idle");
    expect(afterRelease.usage).toBeNull();
    expect(afterRelease.serializedRecord).not.toContain(delayedSessionAnswer);
    expect(afterRelease.serializedRecord).not.toContain(delayedSessionArtifact.title);
    expect(
      persistenceSaves.filter((record) => record.sessionId === sessionBId)
    ).toHaveLength(sessionBSavesBeforeRelease);
    consoleGate.assertClean();
  });

  test("shows a local draft when the assistant stream fails without browser console errors", async ({
    page
  }) => {
    const consoleGate = installConsoleGate(page);
    await installApiMocks(page, "failure");
    await expectLoadedWorkstation(page);

    await submitCreativePrompt(page, "Force a recoverable stream failure.");

    await expect(
      page.getByRole("form", { name: "Creative request composer" })
    ).toHaveAttribute("data-mode", "developer");
    await expect(page.getByRole("log", { name: "Conversation" })).toContainText(
      "Live response error: The live response stopped before completion."
    );
    await expect(page.getByRole("log", { name: "Conversation" })).toContainText(
      "The workflow stopped before the requested output was ready."
    );
    const streamFailure = page.locator(".chatErrorCallout");
    await expect(streamFailure).toHaveAttribute("data-recoverable", "true");
    await expect(streamFailure).toContainText("Retry available");
    await expect(streamFailure).toContainText("Send prompt again");

    await setDisplayMode(page, "User");
    await expect(page.getByRole("log", { name: "Conversation" })).toContainText(
      "The live response could not complete."
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
    await expect(page.getByLabel("Current session", { exact: true })).not.toContainText(
      "Failure"
    );
    consoleGate.assertClean();
  });

  test("presents an unsupported requested runtime as a partial product outcome", async ({
    page
  }) => {
    const consoleGate = installConsoleGate(page);
    await installApiMocks(page, "partial-outcome");
    await expectLoadedWorkstation(page);

    await submitCreativePrompt(
      page,
      "Create a browser-ready React Three Fiber installation study."
    );

    await expect(page.getByLabel("Current session", { exact: true })).toContainText(
      "Partial"
    );
    await expect(page.getByRole("log", { name: "Conversation" })).toContainText(
      "A usable artifact was produced, but live preview is unavailable."
    );

    const expandInspector = page.getByRole("button", { name: "Expand inspector" });
    if (await expandInspector.isVisible().catch(() => false)) {
      await expandInspector.click();
    }
    await page.getByRole("tab", { name: "Preview" }).click();
    const previewPanel = page.getByRole("tabpanel", { exact: true, name: "Preview" });
    await expect(previewPanel).toContainText("PARTIAL");
    await expect(previewPanel).toContainText("USABLE");
    await expect(previewPanel).toContainText(
      "Open Code to inspect, copy, or download the component."
    );
    await expect(previewPanel).toContainText(
      "Use a React project with its own React Three Fiber bundle to run it."
    );
    await expect(previewPanel).toContainText("React Three Fiber export");
    await expect(previewPanel).toContainText(
      "No internal R3F iframe or preview control is presented"
    );
    await expect(previewPanel.locator("iframe")).toHaveCount(0);
    await expect(
      page.getByRole("button", { name: "Enter preview fullscreen" })
    ).toHaveCount(0);
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
        "Generated the E2E p5 artifact with preview routing"
      );
    }

    const storageSize = await page.evaluate(() =>
      JSON.stringify(window.localStorage).length
    );
    expect(storageSize).toBeLessThan(200_000);
    consoleGate.assertClean();
  });
});

async function setDisplayMode(page, targetMode) {
  await page.getByRole("button", { exact: true, name: "Settings" }).click();
  const settings = page.getByRole("dialog", { name: "Workspace settings" });
  const displayMode = settings.getByRole("button", { name: "Display mode" });
  const currentMode = (await displayMode.textContent())?.trim();
  if (currentMode !== targetMode) {
    await displayMode.click();
  }
  await expect(displayMode).toHaveText(targetMode);
  await page.getByRole("button", { exact: true, name: "Settings" }).click();
  await expect(settings).toHaveCount(0);
}

const delayedSessionPrompt = "Hold Session A until the session switch completes.";
const delayedSessionAnswer = "Delayed Session A completion must remain isolated.";
const delayedSessionArtifact = {
  id: "delayed-session-a-artifact",
  title: "delayed-session-a.p5.js",
  type: "code",
  language: "javascript",
  runtime: "p5",
  renderer_id: "surface.p5",
  preview_eligible: true,
  preview_target: "browser_sandbox",
  status: "Generated",
  summary: "A delayed artifact that belongs only to Session A.",
  content: "function setup() { createCanvas(320, 180); }"
};

async function observeWorkspaceSessionSaves(page) {
  const completedSaves = [];

  await page.route("**/api/workspace/session**", async (route, request) => {
    if (request.method() !== "POST") {
      await route.fallback();
      return;
    }

    const record = request.postDataJSON();
    await route.fallback();
    completedSaves.push(record);
  });

  return completedSaves;
}

async function installDelayedAssistantStream(page) {
  let blockedResolve;
  let releaseResolve;
  let releasedResolve;
  let used = false;
  const waitUntilBlocked = new Promise((resolve) => {
    blockedResolve = resolve;
  });
  const waitForRelease = new Promise((resolve) => {
    releaseResolve = resolve;
  });
  const waitUntilReleased = new Promise((resolve) => {
    releasedResolve = resolve;
  });

  await page.route("**/api/assistant/stream", async (route, request) => {
    if (request.method() !== "POST" || used) {
      await route.fallback();
      return;
    }

    used = true;
    blockedResolve(request.postDataJSON());
    await waitForRelease;
    try {
      await route.fulfill({
        body: buildDelayedAssistantNdjson(),
        contentType: "application/x-ndjson",
        headers: {
          "Access-Control-Allow-Headers": "Accept, Content-Type",
          "Access-Control-Allow-Methods": "DELETE, GET, OPTIONS, POST",
          "Access-Control-Allow-Origin": "*"
        },
        status: 200
      });
    } catch {
      // Chromium can close an aborted route before the delayed fixture is released.
    } finally {
      releasedResolve();
    }
  });

  return {
    release() {
      releaseResolve();
    },
    waitUntilBlocked,
    waitUntilReleased
  };
}

async function expectSessionBSave(persistenceSaves, sessionAId) {
  await expect
    .poll(
      () =>
        persistenceSaves.find((record) => record.sessionId !== sessionAId) ?? null
    )
    .not.toBeNull();

  return persistenceSaves.find((record) => record.sessionId !== sessionAId);
}

async function readSessionIsolationState(page, userId, sessionId) {
  return page.evaluate(
    ({ activeSessionId, activeUserId }) => {
      const rawRecord = window.localStorage.getItem(
        `cca.workspace.${activeUserId}.${activeSessionId}`
      );
      if (!rawRecord) {
        throw new Error("Session B was not persisted in browser-local storage.");
      }
      const record = JSON.parse(rawRecord);
      const ledger = JSON.parse(
        window.localStorage.getItem("cca.session-usage-ledger.v1") ??
          '{"version":1,"users":{}}'
      );
      const usage =
        ledger.users?.[activeUserId]?.find(
          (entry) => entry.sessionId === activeSessionId
        ) ?? null;

      return {
        record,
        serializedRecord: rawRecord,
        usage
      };
    },
    { activeSessionId: sessionId, activeUserId: userId }
  );
}

function buildDelayedAssistantNdjson() {
  const events = [
    {
      event_type: "status",
      sequence: 0,
      payload: {
        code: "request_received",
        message: "Delayed Session A request released.",
        workflow: {
          current_step: "generation",
          phase: "running",
          status: "running"
        }
      }
    },
    {
      event_type: "artifact_extracted",
      sequence: 1,
      payload: {
        artifacts: [delayedSessionArtifact],
        workflow: {
          current_step: "artifact_extraction",
          phase: "running",
          status: "running"
        }
      }
    },
    {
      event_type: "final",
      sequence: 2,
      payload: {
        answer: delayedSessionAnswer,
        artifacts: [delayedSessionArtifact],
        telemetry: {
          execution: {
            generation_mode: "streaming",
            request_duration_ms: 900,
            retry_count: 0,
            streaming: true,
            streaming_status: "completed"
          },
          provider: {
            name: "openai",
            model: "gpt-5-mini",
            response_id: "delayed-session-a-response"
          },
          token_usage: {
            input_tokens: 800,
            output_tokens: 200,
            total_tokens: 1000
          }
        },
        workflow: {
          current_step: null,
          phase: "completed",
          status: "completed"
        }
      }
    }
  ];

  return `${events.map((event) => JSON.stringify(event)).join("\n")}\n`;
}
