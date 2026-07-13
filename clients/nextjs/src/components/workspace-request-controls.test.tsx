import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState, type ComponentProps } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AssistantWorkspaceSnapshot } from "@/lib/assistant-client";
import {
  ProviderRouteControl,
  WorkspaceAttachmentControl,
  WorkspaceGenerationControls,
  WorkspaceImageReferences
} from "./workspace-request-controls";

type MultimodalSummary = AssistantWorkspaceSnapshot["multimodal"];

function AttachmentHarness({
  disabled = false,
  isProcessing = false,
  onFilesSelected = vi.fn()
}: Partial<Pick<
  ComponentProps<typeof WorkspaceAttachmentControl>,
  "disabled" | "isProcessing" | "onFilesSelected"
>>) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <WorkspaceAttachmentControl
        disabled={disabled}
        isOpen={isOpen}
        isProcessing={isProcessing}
        onFilesSelected={onFilesSelected}
        onOpenChange={setIsOpen}
      />
      <button type="button">Outside attachment control</button>
    </>
  );
}

function imageAttachment(
  id: string,
  name: string
): MultimodalSummary["imageAttachments"][number] {
  return {
    createdAt: "2026-07-13T08:00:00Z",
    dataUrl: "data:image/png;base64,iVBORw0KGgo=",
    id,
    kind: "image",
    mimeType: "image/png",
    name,
    sizeBytes: 1024
  };
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

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("WorkspaceAttachmentControl", () => {
  it("opens from either Arrow key and moves focus to the first menu item", async () => {
    render(<AttachmentHarness />);
    const trigger = screen.getByRole("button", { name: "Add attachment" });

    fireEvent.keyDown(trigger, { key: "ArrowDown" });

    const menu = screen.getByRole("menu", { name: "Attachment options" });
    const upload = within(menu).getByRole("menuitem", {
      name: /Upload image reference/
    });
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    await waitFor(() => expect(upload).toHaveFocus());

    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() => expect(trigger).toHaveFocus());

    fireEvent.keyDown(trigger, { key: "ArrowUp" });
    const reopenedUpload = within(
      screen.getByRole("menu", { name: "Attachment options" })
    ).getByRole("menuitem", { name: /Upload image reference/ });
    await waitFor(() => expect(reopenedUpload).toHaveFocus());
  });

  it("uses native Enter activation to open the attachment menu", async () => {
    render(<AttachmentHarness />);
    const trigger = screen.getByRole("button", { name: "Add attachment" });

    expect(trigger.tagName).toBe("BUTTON");
    expect(fireEvent.keyDown(trigger, { key: "Enter" })).toBe(true);
    fireEvent.click(trigger);

    expect(
      screen.getByRole("menu", { name: "Attachment options" })
    ).toBeVisible();
    await waitFor(() =>
      expect(
        screen.getByRole("menuitem", { name: /Upload image reference/ })
      ).toHaveFocus()
    );
  });

  it("includes the disabled menu item in roving keyboard focus", async () => {
    render(<AttachmentHarness />);
    fireEvent.click(screen.getByRole("button", { name: "Add attachment" }));

    const menu = screen.getByRole("menu", { name: "Attachment options" });
    const upload = within(menu).getByRole("menuitem", {
      name: /Upload image reference/
    });
    const unavailableAudio = within(menu).getByRole("menuitem", {
      name: /Audio input unavailable/
    });
    expect(unavailableAudio).toHaveAttribute("aria-disabled", "true");
    await waitFor(() => expect(upload).toHaveFocus());

    fireEvent.keyDown(menu, { key: "ArrowDown" });
    expect(unavailableAudio).toHaveFocus();
    fireEvent.keyDown(menu, { key: "ArrowDown" });
    expect(upload).toHaveFocus();
    fireEvent.keyDown(menu, { key: "ArrowUp" });
    expect(unavailableAudio).toHaveFocus();
    fireEvent.keyDown(menu, { key: "Home" });
    expect(upload).toHaveFocus();
    fireEvent.keyDown(menu, { key: "End" });
    expect(unavailableAudio).toHaveFocus();
  });

  it("closes outside and restores trigger focus when Escape closes it", async () => {
    render(<AttachmentHarness />);
    const trigger = screen.getByRole("button", { name: "Add attachment" });
    const outside = screen.getByRole("button", {
      name: "Outside attachment control"
    });

    fireEvent.click(trigger);
    await waitFor(() =>
      expect(
        screen.getByRole("menuitem", { name: /Upload image reference/ })
      ).toHaveFocus()
    );
    fireEvent.keyDown(window, { key: "Escape" });
    expect(
      screen.queryByRole("menu", { name: "Attachment options" })
    ).not.toBeInTheDocument();
    await waitFor(() => expect(trigger).toHaveFocus());

    fireEvent.click(trigger);
    await screen.findByRole("menu", { name: "Attachment options" });
    outside.focus();
    fireEvent.pointerDown(outside);
    expect(
      screen.queryByRole("menu", { name: "Attachment options" })
    ).not.toBeInTheDocument();
    expect(outside).toHaveFocus();
  });

  it("forwards selected files, resets the input, and closes the menu", async () => {
    const onFilesSelected = vi.fn();
    render(<AttachmentHarness onFilesSelected={onFilesSelected} />);
    const trigger = screen.getByRole("button", { name: "Add attachment" });

    fireEvent.click(trigger);
    fireEvent.click(
      screen.getByRole("menuitem", { name: /Upload image reference/ })
    );

    const input = screen.getByLabelText("Upload image attachment");
    const file = new File(["image pixels"], "palette.png", {
      type: "image/png"
    });
    Object.defineProperty(input, "value", {
      configurable: true,
      value: "C:\\fakepath\\palette.png",
      writable: true
    });

    fireEvent.change(input, { target: { files: [file] } });

    expect(onFilesSelected).toHaveBeenCalledOnce();
    expect(onFilesSelected).toHaveBeenCalledWith([file]);
    expect(input).toHaveValue("");
    expect(
      screen.queryByRole("menu", { name: "Attachment options" })
    ).not.toBeInTheDocument();
    await waitFor(() => expect(trigger).toHaveFocus());
  });

  it("disables attachment input and describes the processing state", () => {
    render(<AttachmentHarness isProcessing />);

    const trigger = screen.getByRole("button", { name: "Add attachment" });
    expect(trigger).toBeDisabled();
    expect(trigger).toHaveAccessibleDescription(
      "Preparing image reference. Send is paused."
    );
    expect(screen.getByRole("status")).toHaveTextContent(
      "Preparing image reference. Send is paused."
    );
    expect(screen.getByLabelText("Upload image attachment")).toBeDisabled();
    expect(
      screen.queryByRole("menu", { name: "Attachment options" })
    ).not.toBeInTheDocument();
  });
});

describe("ProviderRouteControl", () => {
  it("opens its provider disclosure, closes outside, and restores focus on Escape", async () => {
    render(
      <>
        <ProviderRouteControl />
        <button type="button">Outside provider control</button>
      </>
    );
    const trigger = screen.getByRole("button", {
      name: "Selected AI provider: OpenAI"
    });

    fireEvent.click(trigger);
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(
      screen.getByRole("region", { name: "AI provider configuration" })
    ).toBeVisible();

    fireEvent.keyDown(window, { key: "Escape" });
    expect(
      screen.queryByRole("region", { name: "AI provider configuration" })
    ).not.toBeInTheDocument();
    await waitFor(() => expect(trigger).toHaveFocus());

    fireEvent.click(trigger);
    const outside = screen.getByRole("button", {
      name: "Outside provider control"
    });
    outside.focus();
    fireEvent.pointerDown(outside);
    expect(
      screen.queryByRole("region", { name: "AI provider configuration" })
    ).not.toBeInTheDocument();
    expect(outside).toHaveFocus();
  });

  it("closes an open provider disclosure when the control becomes disabled", () => {
    const { rerender } = render(<ProviderRouteControl />);
    const trigger = screen.getByRole("button", {
      name: "Selected AI provider: OpenAI"
    });
    fireEvent.click(trigger);
    expect(
      screen.getByRole("region", { name: "AI provider configuration" })
    ).toBeVisible();

    rerender(<ProviderRouteControl disabled />);

    expect(trigger).toBeDisabled();
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(
      screen.queryByRole("region", { name: "AI provider configuration" })
    ).not.toBeInTheDocument();
    fireEvent.click(trigger);
    expect(
      screen.queryByRole("region", { name: "AI provider configuration" })
    ).not.toBeInTheDocument();
  });
});

describe("WorkspaceGenerationControls", () => {
  it("describes and reports workflow and creativity changes", () => {
    const onCreativityChange = vi.fn();
    const onWorkflowChange = vi.fn();
    const { rerender } = render(
      <WorkspaceGenerationControls
        creativity="balanced"
        onCreativityChange={onCreativityChange}
        onWorkflowChange={onWorkflowChange}
        workflowMode="auto"
      />
    );
    const workflow = screen.getByRole("combobox", { name: "Workflow" });
    const creativity = screen.getByRole("combobox", { name: "Creativity" });

    expect(workflow).toHaveAccessibleDescription(
      "Choose the bounded route from the request."
    );
    expect(creativity).toHaveAccessibleDescription(
      "Balance distinct ideas with a reliable implementation path."
    );

    fireEvent.change(workflow, { target: { value: "multi_agent" } });
    fireEvent.change(creativity, { target: { value: "exploratory" } });
    expect(onWorkflowChange).toHaveBeenCalledWith("multi_agent");
    expect(onCreativityChange).toHaveBeenCalledWith("exploratory");

    rerender(
      <WorkspaceGenerationControls
        creativity="exploratory"
        onCreativityChange={onCreativityChange}
        onWorkflowChange={onWorkflowChange}
        workflowMode="multi_agent"
      />
    );
    expect(workflow).toHaveValue("multi_agent");
    expect(workflow).toHaveAccessibleDescription(
      "Use the bounded planner, researcher, generator, critic, and reviewer route."
    );
    expect(creativity).toHaveValue("exploratory");
    expect(creativity).toHaveAccessibleDescription(
      "Favor broader visual and interaction variations within runtime boundaries."
    );
  });
});

describe("WorkspaceImageReferences", () => {
  it("moves focus to the next removal action, then back to attachment control", async () => {
    const onRemove = vi.fn();

    function ImageShelfHarness() {
      const [multimodal, setMultimodal] = useState<MultimodalSummary>({
        detail: "Ready for the next request.",
        error: null,
        imageAttachments: [
          imageAttachment("first-image", "first.png"),
          imageAttachment("second-image", "second.png")
        ],
        state: "ready",
        status: "2 image references"
      });

      return (
        <>
          <button className="workspaceAttachmentTrigger" type="button">
            Add attachment
          </button>
          <WorkspaceImageReferences
            multimodal={multimodal}
            onDismissError={vi.fn()}
            onRemove={(attachmentId) => {
              onRemove(attachmentId);
              setMultimodal((current) => ({
                ...current,
                imageAttachments: current.imageAttachments.filter(
                  (attachment) => attachment.id !== attachmentId
                )
              }));
            }}
          />
        </>
      );
    }

    render(<ImageShelfHarness />);
    const firstRemove = screen.getByRole("button", {
      name: "Remove image reference first.png"
    });
    firstRemove.focus();
    fireEvent.click(firstRemove);

    expect(onRemove).toHaveBeenLastCalledWith("first-image");
    const secondRemove = screen.getByRole("button", {
      name: "Remove image reference second.png"
    });
    await waitFor(() => expect(secondRemove).toHaveFocus());

    fireEvent.click(secondRemove);
    expect(onRemove).toHaveBeenLastCalledWith("second-image");
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Add attachment" })).toHaveFocus()
    );
  });

  it("returns focus to attachment control after dismissing an upload issue", async () => {
    const onDismissError = vi.fn();

    function ErrorShelfHarness() {
      const [multimodal, setMultimodal] = useState<MultimodalSummary>({
        detail: "Only supported images can be attached.",
        error: {
          category: "multimodal",
          debugMessage: null,
          id: "multimodal:image-upload:unsupported-type",
          recoverable: true,
          resetLabel: null,
          retryLabel: null,
          subsystem: "image_upload",
          suggestedAction: "Choose a PNG, JPEG, WebP, or GIF file.",
          type: "unsupported_type",
          userMessage: "This file type is not supported."
        },
        imageAttachments: [],
        state: "error",
        status: "Image upload issue"
      });

      return (
        <>
          <button className="workspaceAttachmentTrigger" type="button">
            Add attachment
          </button>
          <WorkspaceImageReferences
            multimodal={multimodal}
            onDismissError={() => {
              onDismissError();
              setMultimodal((current) => ({
                ...current,
                error: null
              }));
            }}
            onRemove={vi.fn()}
          />
        </>
      );
    }

    render(<ErrorShelfHarness />);
    const dismiss = screen.getByRole("button", {
      name: "Dismiss image upload issue"
    });
    dismiss.focus();
    fireEvent.click(dismiss);

    expect(onDismissError).toHaveBeenCalledOnce();
    expect(
      screen.queryByRole("button", { name: "Dismiss image upload issue" })
    ).not.toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Add attachment" })).toHaveFocus()
    );
  });
});
