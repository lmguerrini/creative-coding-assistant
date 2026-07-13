import { fireEvent, render, screen } from "@testing-library/react";
import {
  createRef,
  type ComponentProps,
  type FormEvent
} from "react";
import { describe, expect, it, vi } from "vitest";
import type { ConversationEntry } from "@/lib/streaming-conversation";
import {
  WorkspaceComposer,
  WorkspaceConversation
} from "./workspace-conversation";

function createEntry(
  overrides: Partial<ConversationEntry> = {}
): ConversationEntry {
  return {
    activity: null,
    content: "A generated response.",
    id: "assistant-message",
    pending: false,
    phase: "complete",
    role: "assistant",
    time: "09:41",
    ...overrides
  };
}

function conversationProps(
  overrides: Partial<ComponentProps<typeof WorkspaceConversation>> = {}
): ComponentProps<typeof WorkspaceConversation> {
  return {
    entries: [],
    getDisplayContent: (entry) => entry.content,
    initialEntryCount: 0,
    isPristine: false,
    isStreaming: false,
    sessionKey: "session-a",
    ...overrides
  };
}

function composerProps(
  overrides: Partial<ComponentProps<typeof WorkspaceComposer>> = {}
): ComponentProps<typeof WorkspaceComposer> {
  return {
    attachmentSlot: <span>Attachment slot</span>,
    controlsSlot: <span>Controls slot</span>,
    hasImages: false,
    isReady: false,
    isStreaming: false,
    mode: "user",
    onChange: vi.fn(),
    onSubmit: vi.fn((event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
    }),
    value: "",
    ...overrides
  };
}

describe("WorkspaceConversation", () => {
  it("exposes an accessible transcript while keeping streaming status separate", () => {
    const entries = [
      createEntry({
        content: "Build a soft particle field.",
        id: "user-message",
        role: "user",
        time: "09:40"
      }),
      createEntry({
        activity: "Rendering the field.",
        content: "The first pass is in progress.",
        id: "streaming-assistant-message",
        phase: "streaming"
      })
    ];
    const { rerender } = render(
      <WorkspaceConversation
        {...conversationProps({
          entries,
          initialEntryCount: 1,
          isStreaming: true
        })}
      />
    );

    const log = screen.getByRole("log", { name: "Conversation" });
    expect(log).toHaveAttribute("aria-atomic", "false");
    expect(log).toHaveAttribute("aria-live", "polite");
    expect(log).toHaveAttribute("aria-relevant", "additions");
    expect(log).not.toHaveAttribute("aria-busy");
    expect(log).toHaveAttribute("tabindex", "0");

    const userMessage = screen.getByRole("article", {
      name: "You message at 09:40"
    });
    const assistantMessage = screen.getByRole("article", {
      name: "Assistant message at 09:41"
    });
    expect(userMessage).not.toHaveAttribute("data-fresh");
    expect(assistantMessage).toHaveAttribute("data-fresh", "true");
    expect(assistantMessage.querySelector("time")).toHaveTextContent("09:41");

    const status = screen.getByRole("status");
    expect(status).toHaveTextContent(
      "Assistant Generating. Rendering the field."
    );
    expect(log).not.toContainElement(status);

    rerender(
      <WorkspaceConversation
        {...conversationProps({ entries, initialEntryCount: 1 })}
      />
    );
    expect(screen.getByRole("status")).toBeEmptyDOMElement();
  });

  it("keeps the pristine overview out of live-log semantics", () => {
    const { container } = render(
      <WorkspaceConversation
        {...conversationProps({
          emptyState: <section>Choose a starting prompt.</section>,
          isPristine: true
        })}
      />
    );

    expect(screen.queryByRole("log")).not.toBeInTheDocument();
    expect(screen.getByText("Choose a starting prompt.")).toBeVisible();

    const overview = container.querySelector(".workspaceConversation");
    expect(overview).toHaveAttribute("aria-label", "Creative session overview");
    expect(overview).not.toHaveAttribute("aria-live");
    expect(overview).not.toHaveAttribute("aria-relevant");
    expect(overview).not.toHaveAttribute("tabindex");
  });

  it("preserves manual reading position, jumps on request, and resets per session", () => {
    const initialEntries = [createEntry()];
    const { rerender } = render(
      <WorkspaceConversation
        {...conversationProps({ entries: initialEntries })}
      />
    );
    const log = screen.getByRole("log", { name: "Conversation" });
    let scrollTop = 100;

    Object.defineProperties(log, {
      clientHeight: {
        configurable: true,
        get: () => 300
      },
      scrollHeight: {
        configurable: true,
        get: () => 1000
      },
      scrollTop: {
        configurable: true,
        get: () => scrollTop,
        set: (value: number) => {
          scrollTop = value;
        }
      }
    });

    fireEvent.scroll(log);
    expect(screen.getByRole("button", { name: "Jump to latest" })).toBeVisible();

    const appendedEntries = [
      ...initialEntries,
      createEntry({ id: "new-assistant-message", time: "09:42" })
    ];
    rerender(
      <WorkspaceConversation
        {...conversationProps({ entries: appendedEntries })}
      />
    );
    expect(scrollTop).toBe(100);
    expect(screen.getByRole("button", { name: "Jump to latest" })).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Jump to latest" }));
    expect(scrollTop).toBe(1000);
    expect(log).toHaveFocus();
    expect(
      screen.queryByRole("button", { name: "Jump to latest" })
    ).not.toBeInTheDocument();

    scrollTop = 120;
    fireEvent.scroll(log);
    expect(screen.getByRole("button", { name: "Jump to latest" })).toBeVisible();

    rerender(
      <WorkspaceConversation
        {...conversationProps({
          entries: appendedEntries,
          sessionKey: "session-b"
        })}
      />
    );
    expect(scrollTop).toBe(1000);
    expect(
      screen.queryByRole("button", { name: "Jump to latest" })
    ).not.toBeInTheDocument();
  });
});

describe("WorkspaceComposer", () => {
  it("names the form and prompt while exposing the keyboard contract", () => {
    const textareaRef = createRef<HTMLTextAreaElement>();
    const onChange = vi.fn();
    render(
      <WorkspaceComposer
        {...composerProps({
          hasImages: true,
          isReady: true,
          isStreaming: true,
          mode: "developer",
          onChange,
          value: "Keep intentional spacing."
        })}
        ref={textareaRef}
      />
    );

    const form = screen.getByRole("form", {
      name: "Creative request composer"
    });
    const textarea = screen.getByRole("textbox", { name: "Assistant prompt" });
    const send = screen.getByRole("button", { name: "Send prompt" });

    expect(form).toHaveAttribute("aria-busy", "true");
    expect(form).toHaveAttribute("data-has-images", "true");
    expect(form).toHaveAttribute("data-mode", "developer");
    expect(textarea).toHaveAccessibleDescription(
      "Press Enter to send. Press Shift and Enter for a new line."
    );
    expect(textareaRef.current).toBe(textarea);
    expect(send).toHaveAttribute("aria-keyshortcuts", "Enter");
    expect(send).toBeEnabled();

    fireEvent.change(textarea, { target: { value: "  preserve this spacing  " } });
    expect(onChange).toHaveBeenCalledWith("  preserve this spacing  ");
  });

  it("does not submit whitespace when readiness is false", () => {
    const onSubmit = vi.fn((event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
    });
    render(
      <WorkspaceComposer
        {...composerProps({
          isReady: false,
          onSubmit,
          value: "   \n  "
        })}
      />
    );

    const textarea = screen.getByRole("textbox", { name: "Assistant prompt" });
    expect(screen.getByRole("button", { name: "Send prompt" })).toBeDisabled();
    expect(fireEvent.keyDown(textarea, { key: "Enter" })).toBe(true);
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("submits with Enter but preserves Shift+Enter and composition", () => {
    const onSubmit = vi.fn((event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
    });
    render(
      <WorkspaceComposer
        {...composerProps({
          isReady: true,
          onSubmit,
          value: "Generate a field."
        })}
      />
    );
    const textarea = screen.getByRole("textbox", { name: "Assistant prompt" });

    expect(fireEvent.keyDown(textarea, { key: "Enter" })).toBe(false);
    expect(onSubmit).toHaveBeenCalledTimes(1);

    onSubmit.mockClear();
    expect(
      fireEvent.keyDown(textarea, { key: "Enter", shiftKey: true })
    ).toBe(true);
    expect(onSubmit).not.toHaveBeenCalled();

    expect(
      fireEvent.keyDown(textarea, { key: "Enter", isComposing: true })
    ).toBe(true);
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("autosizes, clamps overflow, and resets after submission", () => {
    const props = composerProps({ isReady: true, value: "First line" });
    const { rerender } = render(<WorkspaceComposer {...props} />);
    const textarea = screen.getByRole("textbox", {
      name: "Assistant prompt"
    }) as HTMLTextAreaElement;
    let scrollHeight = 112;

    Object.defineProperty(textarea, "scrollHeight", {
      configurable: true,
      get: () => scrollHeight
    });

    rerender(<WorkspaceComposer {...props} value="First line\nSecond line" />);
    expect(textarea.style.height).toBe("112px");
    expect(textarea.style.overflowY).toBe("hidden");

    scrollHeight = 240;
    rerender(
      <WorkspaceComposer
        {...props}
        value="A prompt long enough to exceed the composer height clamp."
      />
    );
    expect(textarea.style.height).toBe("168px");
    expect(textarea.style.overflowY).toBe("auto");

    scrollHeight = 0;
    rerender(<WorkspaceComposer {...props} isReady={false} value="" />);
    expect(textarea.style.height).toBe("38px");
    expect(textarea.style.overflowY).toBe("hidden");
  });
});
