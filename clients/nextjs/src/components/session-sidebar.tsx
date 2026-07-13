"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import {
  History,
  PanelLeft,
  Pencil,
  Plus,
  ShieldCheck,
  Trash2
} from "lucide-react";
import type { WorkspaceSessionSummary } from "@/lib/workspace-persistence";
import {
  DashboardCallout,
  DashboardChoiceCard,
  DashboardSidebarHeader
} from "./dashboard-page-primitives";

type SessionSidebarProps = {
  activeSessionId: string;
  collapsed: boolean;
  onCreate: () => void;
  onDelete: (sessionId: string) => void;
  onRename: (sessionId: string, title: string) => void;
  onSelect: (sessionId: string) => void;
  onToggle: () => void;
  sessions: WorkspaceSessionSummary[];
};

export function SessionSidebar({
  activeSessionId,
  collapsed,
  onCreate,
  onDelete,
  onRename,
  onSelect,
  onToggle,
  sessions
}: SessionSidebarProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [title, setTitle] = useState("");
  const previousCollapsedRef = useRef(collapsed);
  const renameActionRef = useRef<HTMLButtonElement>(null);
  const restoreRenameFocusRef = useRef(false);
  const toggleRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (previousCollapsedRef.current !== collapsed) {
      toggleRef.current?.focus();
      previousCollapsedRef.current = collapsed;
    }
  }, [collapsed]);

  useEffect(() => {
    if (editingId === null && restoreRenameFocusRef.current) {
      renameActionRef.current?.focus();
      restoreRenameFocusRef.current = false;
    }
  }, [editingId]);

  if (collapsed) {
    return (
      <aside
        aria-label="Sessions"
        className="sessionSidebar sessionSidebar--collapsed"
        data-state="collapsed"
        id="session-sidebar"
      >
        <div className="sessionSidebarRail">
          <button
            aria-expanded="false"
            aria-label="Expand session sidebar"
            className="iconButton"
            onClick={onToggle}
            ref={toggleRef}
            title="Expand session sidebar"
            type="button"
          >
            <PanelLeft aria-hidden="true" size={18} />
          </button>
          <strong>Sessions</strong>
          <span>{sessions.length} sessions</span>
          <small>Browser local</small>
        </div>
      </aside>
    );
  }

  function startRename(session: WorkspaceSessionSummary) {
    setEditingId(session.sessionId);
    setTitle(session.title);
  }

  function cancelRename() {
    restoreRenameFocusRef.current = true;
    setEditingId(null);
  }

  function submitRename(event: FormEvent<HTMLFormElement>, sessionId: string) {
    event.preventDefault();
    const nextTitle = title.trim();
    if (!nextTitle) {
      return;
    }
    onRename(sessionId, nextTitle);
    restoreRenameFocusRef.current = true;
    setEditingId(null);
  }

  return (
    <aside
      aria-label="Sessions"
      className="sessionSidebar"
      data-state="open"
      id="session-sidebar"
    >
      <DashboardSidebarHeader
        action={(
          <button
            aria-expanded="true"
            aria-label="Collapse session sidebar"
            className="iconButton"
            onClick={onToggle}
            ref={toggleRef}
            title="Collapse session sidebar"
            type="button"
          >
            <PanelLeft aria-hidden="true" size={18} />
          </button>
        )}
        className="sessionSidebarHeader"
        detail="Open or start browser-local creative sessions."
        eyebrow="Workspace"
        icon={History}
        title="Sessions"
      />

      <button
        className="dashboardPrimaryAction sessionSidebarCreate"
        onClick={onCreate}
        type="button"
      >
        <Plus aria-hidden="true" size={15} />
        <span>New session</span>
      </button>

      <div aria-label="Saved sessions" className="sessionSidebarList" role="list">
        {sessions.map((session) => {
          const active = session.sessionId === activeSessionId;
          const artifactLabel = `${session.artifactCount} artifact${
            session.artifactCount === 1 ? "" : "s"
          }`;

          return (
            <article
              className="sessionSidebarItem"
              data-active={active ? "true" : "false"}
              key={session.sessionId}
              role="listitem"
            >
              {editingId === session.sessionId ? (
                <form
                  aria-label={`Rename ${session.title}`}
                  onSubmit={(event) => submitRename(event, session.sessionId)}
                >
                  <input
                    aria-label="Session name"
                    autoFocus
                    onChange={(event) => setTitle(event.currentTarget.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Escape") {
                        event.preventDefault();
                        cancelRename();
                      }
                    }}
                    required
                    value={title}
                  />
                  <button type="submit">Save</button>
                  <button onClick={cancelRename} type="button">
                    Cancel
                  </button>
                </form>
              ) : (
                <>
                  <DashboardChoiceCard
                    ariaCurrent={active ? "true" : undefined}
                    ariaPressed={false}
                    className="sessionSidebarSelect"
                    detail={formatSessionUpdatedAt(session.updatedAt)}
                    eyebrow={artifactLabel}
                    icon={History}
                    idleLabel="Open"
                    onClick={() => onSelect(session.sessionId)}
                    selected={active}
                    selectedLabel="Current"
                    title={session.title}
                  />
                  {active ? (
                    <div aria-label="Current session actions" className="sessionSidebarActions">
                      <button
                        aria-label={`Rename ${session.title}`}
                        onClick={() => startRename(session)}
                        ref={renameActionRef}
                        title={`Rename ${session.title}`}
                        type="button"
                      >
                        <Pencil aria-hidden="true" size={13} />
                        <span>Rename</span>
                      </button>
                      <button
                        aria-label={`Delete ${session.title}`}
                        onClick={() => onDelete(session.sessionId)}
                        title={`Delete ${session.title}`}
                        type="button"
                      >
                        <Trash2 aria-hidden="true" size={13} />
                        <span>Delete</span>
                      </button>
                    </div>
                  ) : null}
                </>
              )}
            </article>
          );
        })}
      </div>

      <DashboardCallout
        as="footer"
        className="sessionSidebarBoundary"
        detail="Stored only in this browser profile; damaged records are skipped."
        icon={ShieldCheck}
        title="Browser-local history"
        tone="info"
      />
    </aside>
  );
}

function formatSessionUpdatedAt(value: string | null) {
  if (!value) {
    return "Current session";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Saved locally";
  }

  return `Saved ${date.toLocaleString(undefined, {
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    month: "short"
  })}`;
}
