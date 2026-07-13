"use client";

import {
  forwardRef,
  useEffect,
  useId,
  useRef,
  useState,
  type FormEventHandler,
  type ReactNode
} from "react";
import { ArrowDown, SendHorizontal } from "lucide-react";
import {
  getConversationPhaseBadge,
  getConversationPhasePlaceholder,
  type ConversationEntry
} from "@/lib/streaming-conversation";

type WorkspaceConversationProps = {
  emptyState?: ReactNode;
  entries: readonly ConversationEntry[];
  getDisplayContent: (entry: ConversationEntry) => string;
  initialEntryCount: number;
  isPristine: boolean;
  isStreaming: boolean;
  leadingSlot?: ReactNode;
  sessionKey: string;
};

export function WorkspaceConversation({
  emptyState,
  entries,
  getDisplayContent,
  initialEntryCount,
  isPristine,
  isStreaming,
  leadingSlot,
  sessionKey
}: WorkspaceConversationProps) {
  const logRef = useRef<HTMLDivElement>(null);
  const shouldAutoScrollRef = useRef(true);
  const [isAtLatest, setIsAtLatest] = useState(true);
  const activeAssistantEntry = isStreaming
    ? [...entries].reverse().find((entry) => entry.role === "assistant")
    : undefined;

  useEffect(() => {
    const log = logRef.current;
    if (!log) {
      return undefined;
    }

    const syncScrollState = () => {
      const distanceFromBottom = log.scrollHeight - log.scrollTop - log.clientHeight;
      const nextIsAtLatest = distanceFromBottom <= 88;
      shouldAutoScrollRef.current = nextIsAtLatest;
      setIsAtLatest(nextIsAtLatest);
    };

    syncScrollState();
    log.addEventListener("scroll", syncScrollState, { passive: true });

    return () => log.removeEventListener("scroll", syncScrollState);
  }, []);

  useEffect(() => {
    const log = logRef.current;
    shouldAutoScrollRef.current = true;
    setIsAtLatest(true);
    if (log) {
      log.scrollTop = log.scrollHeight;
    }
  }, [sessionKey]);

  useEffect(() => {
    const log = logRef.current;
    if (!log) {
      return;
    }

    if (entries.length === 0 && !isStreaming) {
      log.scrollTop = 0;
      shouldAutoScrollRef.current = true;
      setIsAtLatest(true);
      return;
    }

    if (shouldAutoScrollRef.current) {
      log.scrollTop = log.scrollHeight;
      setIsAtLatest(true);
    }
  }, [entries, isStreaming]);

  function jumpToLatest() {
    const log = logRef.current;
    if (!log) {
      return;
    }
    log.scrollTop = log.scrollHeight;
    shouldAutoScrollRef.current = true;
    setIsAtLatest(true);
    log.focus({ preventScroll: true });
  }

  return (
    <div
      className="workspaceConversationFrame"
      data-at-latest={isAtLatest}
      data-has-leading={leadingSlot ? "true" : undefined}
    >
      {leadingSlot}
      <div
        aria-atomic="false"
        aria-label={isPristine ? "Creative session overview" : "Conversation"}
        aria-live={isPristine ? undefined : "polite"}
        aria-relevant={isPristine ? undefined : "additions"}
        className="workspaceConversation"
        data-empty={isPristine ? "true" : undefined}
        ref={logRef}
        role={isPristine ? undefined : "log"}
        tabIndex={isPristine ? undefined : 0}
      >
        {isPristine ? emptyState : null}
        {entries.map((message, index) => {
          const displayContent = getDisplayContent(message);

          return (
            <article
              aria-label={`${formatConversationRole(message.role)} message at ${message.time}`}
              className="workspaceMessage"
              data-fresh={index >= initialEntryCount ? "true" : undefined}
              data-role={message.role}
              data-stream-phase={message.phase}
              key={message.id}
            >
              <div className="workspaceMessageMeta">
                <span>{formatConversationRole(message.role)}</span>
                <div className="workspaceMessageMetaDetail">
                  {message.role === "assistant" ? (
                    <small data-phase={message.phase}>
                      {getConversationPhaseBadge(message.phase)}
                    </small>
                  ) : null}
                  <time>{message.time}</time>
                </div>
              </div>
              <p>
                {displayContent || getConversationPhasePlaceholder(message.phase)}
                {message.phase === "streaming" ? (
                  <span className="workspaceStreamingCaret" aria-hidden="true" />
                ) : null}
              </p>
              {message.activity && message.phase !== "complete" ? (
                <div
                  className="workspaceMessageActivity"
                  data-activity={message.phase}
                >
                  <span aria-hidden="true" />
                  <small>{message.activity}</small>
                </div>
              ) : null}
            </article>
          );
        })}
      </div>
      <div
        aria-atomic="true"
        aria-live="polite"
        className="workspaceVisuallyHidden"
        role="status"
      >
        {activeAssistantEntry
          ? `Assistant ${getConversationPhaseBadge(activeAssistantEntry.phase)}. ${
              activeAssistantEntry.activity ||
              getConversationPhasePlaceholder(activeAssistantEntry.phase)
            }`
          : ""}
      </div>
      {entries.length > 0 && !isPristine && !isAtLatest ? (
        <button
          className="workspaceConversationJump"
          onClick={jumpToLatest}
          type="button"
        >
          <ArrowDown aria-hidden="true" size={14} />
          <span>Jump to latest</span>
        </button>
      ) : null}
    </div>
  );
}

type WorkspaceComposerProps = {
  attachmentSlot: ReactNode;
  controlsSlot: ReactNode;
  hasImages: boolean;
  isPreparingAttachments?: boolean;
  isReady: boolean;
  isStreaming: boolean;
  mode: "developer" | "user";
  onChange: (value: string) => void;
  onSubmit: FormEventHandler<HTMLFormElement>;
  value: string;
};

export const WorkspaceComposer = forwardRef<
  HTMLTextAreaElement,
  WorkspaceComposerProps
>(function WorkspaceComposer(
  {
    attachmentSlot,
    controlsSlot,
    hasImages,
    isPreparingAttachments = false,
    isReady,
    isStreaming,
    mode,
    onChange,
    onSubmit,
    value
  },
  forwardedRef
) {
  const keyboardHintId = useId();
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }

    const maxHeight = 168;
    textarea.style.height = "auto";
    const nextHeight = Math.min(textarea.scrollHeight || 38, maxHeight);
    textarea.style.height = `${nextHeight}px`;
    textarea.style.overflowY =
      textarea.scrollHeight > maxHeight ? "auto" : "hidden";
  }, [value]);

  function setTextareaRef(node: HTMLTextAreaElement | null) {
    textareaRef.current = node;
    if (typeof forwardedRef === "function") {
      forwardedRef(node);
    } else if (forwardedRef) {
      forwardedRef.current = node;
    }
  }

  return (
    <form
      aria-busy={isStreaming || isPreparingAttachments}
      aria-label="Creative request composer"
      className="workspaceComposer"
      data-has-images={hasImages ? "true" : "false"}
      data-upload-state={isPreparingAttachments ? "processing" : "idle"}
      data-mode={mode}
      data-ready={isReady}
      onSubmit={onSubmit}
    >
      <div className="workspaceComposerInputFrame">
        <textarea
          aria-describedby={keyboardHintId}
          aria-label="Assistant prompt"
          onChange={(event) => onChange(event.currentTarget.value)}
          onKeyDown={(event) => {
            if (
              event.key !== "Enter" ||
              event.shiftKey ||
              event.nativeEvent.isComposing ||
              !isReady
            ) {
              return;
            }
            event.preventDefault();
            event.currentTarget.form?.requestSubmit();
          }}
          placeholder="Describe the visual, audio, or interactive experience you want to create."
          ref={setTextareaRef}
          value={value}
        />
        <span className="workspaceVisuallyHidden" id={keyboardHintId}>
          Press Enter to send. Press Shift and Enter for a new line.
        </span>
      </div>
      <div className="workspaceComposerActions">
        <div className="workspaceComposerAttachmentSlot">{attachmentSlot}</div>
        <div className="workspaceComposerControlsSlot">{controlsSlot}</div>
        <button
          aria-keyshortcuts="Enter"
          aria-label="Send prompt"
          className="workspaceComposerSend"
          data-ready={isReady}
          disabled={!isReady}
          title={isReady ? "Send prompt" : "Type a prompt to send"}
          type="submit"
        >
          <SendHorizontal aria-hidden="true" size={17} />
          <span>Send</span>
        </button>
      </div>
    </form>
  );
});

function formatConversationRole(role: ConversationEntry["role"]) {
  return role === "user" ? "You" : "Assistant";
}
