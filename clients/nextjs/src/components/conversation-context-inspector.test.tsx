import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ConversationContextInspector } from "./conversation-context-inspector";

describe("ConversationContextInspector", () => {
  it("stays available while a development refresh replaces its context prop", () => {
    render(<ConversationContextInspector />);

    expect(
      screen.getByRole("group", { name: "Conversation context diagnostics" })
    ).toBeVisible();
    expect(screen.getByText("Awaiting evidence")).toBeVisible();
    expect(screen.getByText("Not reported")).toBeVisible();
  });
});
