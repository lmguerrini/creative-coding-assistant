const { test, expect } = require("@playwright/test");
const { mountSandboxRuntime } = require("./support/preview-sandbox");

test("keeps direct top-level preview navigation inert", async ({ page }) => {
  const source = [
    'document.body.dataset.directRuntimeExecuted = "true";',
    "function setup() { createCanvas(160, 90); }",
    "function draw() { circle(40, 40, 20); }"
  ].join("\n");
  const runtime = {
    kind: "p5",
    runtimeId: "direct-runtime-sentinel",
    source: {
      fingerprint: "direct-runtime-sentinel",
      lineCount: source.split("\n").length,
      source,
      title: "direct-runtime-sentinel.p5.js"
    }
  };

  await page.goto(
    `/preview-sandbox.html#${encodeURIComponent(JSON.stringify(runtime))}`
  );
  await page.waitForTimeout(200);

  await expect(page.locator("body")).not.toHaveAttribute(
    "data-direct-runtime-executed",
    "true"
  );
  await expect(page.locator("body")).not.toHaveAttribute(
    "data-runtime-state",
    /.+/
  );
  await expect(page.locator("#runtime-status")).toHaveText(
    "Runtime host waiting"
  );

  await page.evaluate((runtimePayload) => {
    window.postMessage(
      {
        handshakeId: "preview-handshake-00000000000000000000000000000000",
        runtime: runtimePayload,
        runtimeId: runtimePayload.runtimeId,
        source: "cca-preview-runtime",
        type: "mount"
      },
      window.location.origin
    );
  }, runtime);
  await page.waitForTimeout(100);

  await expect(page.locator("body")).not.toHaveAttribute(
    "data-direct-runtime-executed",
    "true"
  );
  await expect(page.locator("#runtime-status")).toHaveText(
    "Runtime host waiting"
  );
});

test("rejects unsolicited and malformed iframe mount messages", async ({
  page
}) => {
  await page.goto("/");
  await page.evaluate(
    () =>
      new Promise((resolve, reject) => {
        const iframe = document.createElement("iframe");
        const challengeId =
          "preview-challenge-22222222222222222222222222222222";
        const timeout = window.setTimeout(
          () => reject(new Error("Sandbox readiness message was not received.")),
          8_000
        );
        const runtime = {
          kind: "p5",
          runtimeId: "rejected-runtime-sentinel",
          source: {
            fingerprint: "rejected-runtime-sentinel",
            lineCount: "1",
            source:
              'document.body.dataset.malformedRuntimeExecuted = "true"; function draw() {}',
            title: "rejected-runtime-sentinel.p5.js"
          }
        };

        function postMount(handshakeId) {
          iframe.contentWindow.postMessage(
            {
              handshakeId,
              runtime,
              runtimeId: runtime.runtimeId,
              source: "cca-preview-runtime",
              type: "mount"
            },
            "*"
          );
        }

        function handleMessage(event) {
          if (
            event.source !== iframe.contentWindow ||
            event.origin !== "null" ||
            !event.data ||
            event.data.challengeId !== challengeId ||
            event.data.type !== "ready"
          ) {
            return;
          }

          postMount(event.data.handshakeId);
          window.setTimeout(() => {
            window.clearTimeout(timeout);
            window.removeEventListener("message", handleMessage);
            resolve(undefined);
          }, 200);
        }

        iframe.id = "rejected-preview-sandbox-frame";
        iframe.setAttribute("sandbox", "allow-scripts");
        iframe.addEventListener("load", () => {
          postMount("preview-handshake-00000000000000000000000000000000");
        });
        window.addEventListener("message", handleMessage);
        iframe.src =
          `/preview-sandbox.html?challenge=${challengeId}#${challengeId}`;
        document.body.appendChild(iframe);
      })
  );

  const sandbox = page.frameLocator("#rejected-preview-sandbox-frame");
  await expect(sandbox.locator("body")).not.toHaveAttribute(
    "data-malformed-runtime-executed",
    "true"
  );
  await expect(sandbox.locator("#runtime-status")).toHaveText(
    "Runtime host waiting"
  );
});

test("refuses runtime mounting outside an opaque sandbox", async ({ page }) => {
  await page.goto("/");
  const announcedReady = await page.evaluate(
    () =>
      new Promise((resolve) => {
        const iframe = document.createElement("iframe");
        const challengeId =
          "preview-challenge-33333333333333333333333333333333";
        let receivedReady = false;

        function handleMessage(event) {
          if (
            event.source === iframe.contentWindow &&
            event.data &&
            event.data.type === "ready"
          ) {
            receivedReady = true;
          }
        }

        iframe.id = "non-sandbox-preview-frame";
        iframe.addEventListener("load", () => {
          iframe.contentWindow.postMessage(
            {
              handshakeId:
                "preview-handshake-00000000000000000000000000000000",
              runtime: {
                kind: "p5",
                runtimeId: "non-sandbox-runtime",
                source: {
                  fingerprint: "non-sandbox-runtime",
                  lineCount: 1,
                  source:
                    'document.body.dataset.nonSandboxRuntimeExecuted = "true"; function draw() {}',
                  title: "non-sandbox-runtime.p5.js"
                }
              },
              runtimeId: "non-sandbox-runtime",
              source: "cca-preview-runtime",
              type: "mount"
            },
            window.location.origin
          );
          window.setTimeout(() => {
            window.removeEventListener("message", handleMessage);
            resolve(receivedReady);
          }, 200);
        });
        window.addEventListener("message", handleMessage);
        iframe.src =
          `/preview-sandbox.html?challenge=${challengeId}#${challengeId}`;
        document.body.appendChild(iframe);
      })
  );

  expect(announcedReady).toBe(false);
  const frame = page.frameLocator("#non-sandbox-preview-frame");
  await expect(frame.locator("body")).not.toHaveAttribute(
    "data-non-sandbox-runtime-executed",
    "true"
  );
  await expect(frame.locator("#runtime-status")).toHaveText(
    "Runtime host waiting"
  );
});

test("renders p5 through the framed preview handshake", async ({ page }) => {
  const source = [
    "function setup() { createCanvas(160, 90); }",
    "function draw() { background(8, 12, 18); circle(80, 45, 28); }"
  ].join("\n");
  const sandbox = await mountSandboxRuntime(page, {
    kind: "p5",
    runtimeId: "p5-handshake-runtime",
    source: {
      fingerprint: "p5-handshake-runtime",
      lineCount: source.split("\n").length,
      source,
      title: "p5-handshake-runtime.p5.js"
    }
  });

  await expect(sandbox.locator("body")).toHaveAttribute(
    "data-runtime-state",
    "running"
  );
  await expect(sandbox.locator("#preview-canvas")).toBeVisible();
});

test("renders a muted Tone.js plan through the framed preview handshake", async ({
  page
}) => {
  const program = JSON.stringify({
    effects: [],
    patterns: [{ notes: ["C4", "E4", "G4"], subdivision: "8n" }],
    tempo: 96,
    version: 1,
    visualization: "spectrum",
    voices: [
      {
        envelope: { attack: 0.02, decay: 0.18, release: 0.6, sustain: 0.42 },
        frequency: null,
        kind: "synth",
        waveform: "sine"
      }
    ],
    volumeDb: -12
  });
  const sandbox = await mountSandboxRuntime(
    page,
    {
      kind: "tone",
      runtimeId: "tone-handshake-runtime",
      source: {
        fingerprint: "tone-handshake-runtime",
        lineCount: 1,
        source: program,
        title: "tone-handshake-runtime.tone.js"
      }
    },
    ["ready"]
  );

  await expect(sandbox.locator("body")).toHaveAttribute(
    "data-runtime-state",
    "ready"
  );
  await expect(sandbox.locator("#audio-controls")).toBeVisible();
  await expect(sandbox.locator("#audio-start")).toBeEnabled();
});
