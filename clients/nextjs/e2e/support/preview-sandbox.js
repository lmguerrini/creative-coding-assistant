async function mountSandboxRuntime(page, runtime, expectedStates = ["running"]) {
  await page.goto("/");

  await page.evaluate(
    ({ expectedStates: targetStates, runtime: runtimePayload }) =>
      new Promise((resolve, reject) => {
        const iframe = document.createElement("iframe");
        const challengeBytes = new Uint8Array(16);
        crypto.getRandomValues(challengeBytes);
        const challengeId = `preview-challenge-${Array.from(
          challengeBytes,
          (value) => value.toString(16).padStart(2, "0")
        ).join("")}`;
        const timeout = window.setTimeout(() => {
          window.removeEventListener("message", handleMessage);
          reject(new Error("Timed out waiting for the preview sandbox runtime."));
        }, 8_000);
        let handshakeId = "";

        function finish(value) {
          window.clearTimeout(timeout);
          window.removeEventListener("message", handleMessage);
          resolve(value);
        }

        function handleMessage(event) {
          if (event.source !== iframe.contentWindow || event.origin !== "null") {
            return;
          }

          const message = event.data;
          if (
            message &&
            message.source === "cca-preview-runtime" &&
            message.type === "ready" &&
            message.challengeId === challengeId &&
            typeof message.handshakeId === "string" &&
            /^preview-handshake-[a-f0-9]{32}$/.test(message.handshakeId)
          ) {
            handshakeId = message.handshakeId;
            iframe.contentWindow.postMessage(
              {
                handshakeId,
                runtime: runtimePayload,
                runtimeId: runtimePayload.runtimeId,
                source: "cca-preview-runtime",
                type: "mount"
              },
              "*"
            );
            return;
          }

          if (
            !message ||
            !handshakeId ||
            message.handshakeId !== handshakeId ||
            message.runtimeId !== runtimePayload.runtimeId ||
            message.source !== "cca-preview-runtime" ||
            message.type !== "status" ||
            !message.status
          ) {
            return;
          }

          if (message.status.state === "error") {
            window.clearTimeout(timeout);
            window.removeEventListener("message", handleMessage);
            reject(new Error(message.status.detail || "Preview runtime failed."));
            return;
          }

          if (targetStates.includes(message.status.state)) {
            finish(message.status.state);
          }
        }

        iframe.id = "preview-sandbox-test-frame";
        iframe.setAttribute("sandbox", "allow-scripts");
        Object.assign(iframe.style, {
          border: "0",
          height: "100vh",
          inset: "0",
          position: "fixed",
          width: "100vw",
          zIndex: "2147483647"
        });
        iframe.src =
          `/preview-sandbox.html?challenge=${challengeId}#${challengeId}`;
        window.addEventListener("message", handleMessage);
        document.body.appendChild(iframe);
      }),
    { expectedStates, runtime }
  );

  return page.frameLocator("#preview-sandbox-test-frame");
}

module.exports = { mountSandboxRuntime };
