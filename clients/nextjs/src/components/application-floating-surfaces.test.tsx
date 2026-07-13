import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor
} from "@testing-library/react";
import { useRef, useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ApplicationConfirmDialog,
  ApplicationFloatingPanel,
  type ApplicationConfirmationRequest
} from "./application-floating-surfaces";

function confirmationRequest(
  overrides: Partial<ApplicationConfirmationRequest> = {}
): ApplicationConfirmationRequest {
  return {
    cancelLabel: "Keep session",
    confirmLabel: "Delete session",
    detail: "This removes the saved session and its local workspace history.",
    id: "delete-session",
    onConfirm: vi.fn(),
    title: "Delete this session?",
    tone: "danger",
    ...overrides
  };
}

function ConfirmationHarness({
  onClosed = vi.fn(),
  request
}: {
  onClosed?: () => void;
  request: ApplicationConfirmationRequest;
}) {
  const [open, setOpen] = useState(true);

  return open ? (
    <ApplicationConfirmDialog
      onClose={() => {
        onClosed();
        setOpen(false);
      }}
      request={request}
    />
  ) : null;
}

beforeEach(() => {
  vi.stubGlobal(
    "requestAnimationFrame",
    (callback: FrameRequestCallback) =>
      window.setTimeout(() => callback(performance.now()), 0)
  );
  vi.stubGlobal("cancelAnimationFrame", (frameId: number) => {
    window.clearTimeout(frameId);
  });
});

afterEach(async () => {
  cleanup();
  await new Promise((resolve) => window.setTimeout(resolve, 0));
  document.body.style.overflow = "";
  vi.unstubAllGlobals();
});

describe("ApplicationFloatingPanel", () => {
  it("takes initial focus and requests close when Escape is pressed", async () => {
    const onRequestClose = vi.fn();

    render(
      <ApplicationFloatingPanel
        description="Choose the active workspace theme."
        id="theme-presets"
        label="Theme presets"
        onRequestClose={onRequestClose}
        title="Theme and colour"
      >
        <button type="button">Deep Blue</button>
      </ApplicationFloatingPanel>
    );

    const panel = screen.getByRole("dialog", { name: "Theme and colour" });
    await waitFor(() => expect(panel).toHaveFocus());

    fireEvent.keyDown(panel, { key: "Escape" });

    expect(onRequestClose).toHaveBeenCalledTimes(1);
  });
});

describe("ApplicationConfirmDialog", () => {
  it("focuses Cancel first and traps forward and reverse Tab navigation", async () => {
    render(<ConfirmationHarness request={confirmationRequest()} />);

    const cancel = screen.getByRole("button", { name: "Keep session" });
    const confirm = screen.getByRole("button", { name: "Delete session" });
    await waitFor(() => expect(cancel).toHaveFocus());

    fireEvent.keyDown(cancel, { key: "Tab", shiftKey: true });
    expect(confirm).toHaveFocus();

    fireEvent.keyDown(confirm, { key: "Tab" });
    expect(cancel).toHaveFocus();
  });

  it("cancels with Escape and restores focus to the invoking control", async () => {
    const onClosed = vi.fn();

    function RestoreFocusHarness() {
      const triggerRef = useRef<HTMLButtonElement>(null);
      const [open, setOpen] = useState(false);

      return (
        <>
          <button
            onClick={() => setOpen(true)}
            ref={triggerRef}
            type="button"
          >
            Delete saved session
          </button>
          {open ? (
            <ApplicationConfirmDialog
              onClose={() => {
                onClosed();
                setOpen(false);
              }}
              request={confirmationRequest({ returnFocus: triggerRef.current })}
            />
          ) : null}
        </>
      );
    }

    render(<RestoreFocusHarness />);
    const trigger = screen.getByRole("button", { name: "Delete saved session" });
    fireEvent.click(trigger);
    const dialog = screen.getByRole("alertdialog", {
      name: "Delete this session?"
    });
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Keep session" })).toHaveFocus()
    );

    fireEvent.keyDown(dialog, { key: "Escape" });

    expect(onClosed).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(trigger).toHaveFocus());
  });

  it("cancels when the backdrop itself is pressed", async () => {
    const onClosed = vi.fn();
    render(
      <ConfirmationHarness
        onClosed={onClosed}
        request={confirmationRequest()}
      />
    );
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Keep session" })).toHaveFocus()
    );

    const backdrop = document.querySelector<HTMLElement>(
      ".applicationModalBackdrop"
    );
    expect(backdrop).not.toBeNull();
    fireEvent.pointerDown(backdrop!);

    expect(onClosed).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
  });

  it("runs an asynchronous confirmation only once while it is pending", async () => {
    let resolveConfirmation!: () => void;
    const pendingConfirmation = new Promise<void>((resolve) => {
      resolveConfirmation = resolve;
    });
    const onConfirm = vi.fn(() => pendingConfirmation);
    const onClosed = vi.fn();
    render(
      <ConfirmationHarness
        onClosed={onClosed}
        request={confirmationRequest({ onConfirm })}
      />
    );

    const confirm = screen.getByRole("button", { name: "Delete session" });
    fireEvent.click(confirm);
    fireEvent.click(confirm);

    expect(onConfirm).toHaveBeenCalledTimes(1);
    expect(confirm).toBeDisabled();
    expect(screen.getByRole("button", { name: "Working…" })).toBeDisabled();

    await act(async () => {
      resolveConfirmation();
      await pendingConfirmation;
    });

    await waitFor(() => expect(onClosed).toHaveBeenCalledTimes(1));
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
  });

  it("keeps the dialog open, reports an async failure, and enables retry", async () => {
    const onClosed = vi.fn();
    const onConfirm = vi
      .fn()
      .mockRejectedValue(new Error("Session service unavailable"));
    render(
      <ConfirmationHarness
        onClosed={onClosed}
        request={confirmationRequest({ onConfirm })}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Delete session" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Session service unavailable"
    );
    expect(screen.getByRole("alertdialog")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Keep session" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Delete session" })).toBeEnabled();
    expect(onClosed).not.toHaveBeenCalled();
  });
});
