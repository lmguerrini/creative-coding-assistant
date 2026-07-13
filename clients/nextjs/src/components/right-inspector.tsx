"use client";

import {
  useEffect,
  useRef,
  type KeyboardEvent,
  type ReactNode
} from "react";
import {
  LayoutDashboard,
  PanelRight,
  Plus,
  X,
  type LucideIcon
} from "lucide-react";
import { DashboardSidebarHeader } from "./dashboard-page-primitives";

export type RightInspectorTab<TabId extends string> = {
  badge?: string;
  closeable?: boolean;
  icon: LucideIcon;
  id: TabId;
  label: string;
  panelId: string;
};

type RightInspectorProps<TabId extends string> = {
  activeTab: TabId;
  addMenuOpen: boolean;
  availableTabs: readonly RightInspectorTab<TabId>[];
  children: ReactNode;
  collapsed: boolean;
  detail: string;
  onAddMenuOpenChange: (open: boolean) => void;
  onAddTab: (tab: TabId) => void;
  onCloseTab: (tab: TabId) => void;
  onOpenDashboard: (tab: TabId) => void;
  onSelectTab: (tab: TabId) => void;
  onToggle: () => void;
  tabs: readonly RightInspectorTab<TabId>[];
};

export function RightInspector<TabId extends string>({
  activeTab,
  addMenuOpen,
  availableTabs,
  children,
  collapsed,
  detail,
  onAddMenuOpenChange,
  onAddTab,
  onCloseTab,
  onOpenDashboard,
  onSelectTab,
  onToggle,
  tabs
}: RightInspectorProps<TabId>) {
  const addButtonRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const restoreAddedTabFocusRef = useRef<string | null>(null);
  const restoreClosedTabFocusRef = useRef(false);
  const restoreToggleFocusRef = useRef(false);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const active = tabs.find((tab) => tab.id === activeTab) ?? tabs[0];
  const activeLabel = active?.label ?? "Inspector";

  useEffect(() => {
    if (!restoreToggleFocusRef.current) {
      return;
    }

    toggleRef.current?.focus();
    restoreToggleFocusRef.current = false;
  }, [collapsed]);

  useEffect(() => {
    if (!addMenuOpen) {
      return;
    }

    menuRef.current
      ?.querySelector<HTMLElement>("[role='menuitem']")
      ?.focus();
  }, [addMenuOpen]);

  useEffect(() => {
    if (!addMenuOpen) {
      return;
    }

    function handlePointerDown(event: PointerEvent) {
      const target = event.target;
      if (
        target instanceof Node &&
        !menuRef.current?.contains(target) &&
        !addButtonRef.current?.contains(target)
      ) {
        onAddMenuOpenChange(false);
      }
    }

    document.addEventListener("pointerdown", handlePointerDown);
    return () => document.removeEventListener("pointerdown", handlePointerDown);
  }, [addMenuOpen, onAddMenuOpenChange]);

  useEffect(() => {
    const addedPanelId = restoreAddedTabFocusRef.current;
    if (addedPanelId) {
      document.getElementById(`${addedPanelId}-tab`)?.focus();
      restoreAddedTabFocusRef.current = null;
      return;
    }

    if (restoreClosedTabFocusRef.current) {
      document.getElementById(`${active?.panelId}-tab`)?.focus();
      restoreClosedTabFocusRef.current = false;
    }
  }, [active?.panelId, tabs.length]);

  function toggleInspector() {
    restoreToggleFocusRef.current = true;
    onToggle();
  }

  function focusTab(index: number) {
    const nextTab = tabs[index];
    if (!nextTab) {
      return;
    }

    onSelectTab(nextTab.id);
    document.getElementById(`${nextTab.panelId}-tab`)?.focus();
  }

  function handleTabKeyDown(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    let nextIndex: number | null = null;

    if (event.key === "ArrowRight") {
      nextIndex = (index + 1) % tabs.length;
    } else if (event.key === "ArrowLeft") {
      nextIndex = (index - 1 + tabs.length) % tabs.length;
    } else if (event.key === "Home") {
      nextIndex = 0;
    } else if (event.key === "End") {
      nextIndex = tabs.length - 1;
    }

    if (nextIndex === null) {
      return;
    }

    event.preventDefault();
    focusTab(nextIndex);
  }

  function closeAddMenu() {
    onAddMenuOpenChange(false);
    addButtonRef.current?.focus();
  }

  function handleCloseTab(tab: RightInspectorTab<TabId>) {
    restoreClosedTabFocusRef.current = true;
    onCloseTab(tab.id);
  }

  function handleAddTab(tab: RightInspectorTab<TabId>) {
    restoreAddedTabFocusRef.current = tab.panelId;
    onAddTab(tab.id);
  }

  function handleMenuKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Escape") {
      event.preventDefault();
      closeAddMenu();
      return;
    }

    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
      return;
    }

    const items = Array.from(
      menuRef.current?.querySelectorAll<HTMLElement>("[role='menuitem']") ?? []
    );
    if (items.length === 0) {
      return;
    }

    const currentIndex = items.indexOf(document.activeElement as HTMLElement);
    let nextIndex = currentIndex;
    if (event.key === "ArrowDown") {
      nextIndex = (currentIndex + 1 + items.length) % items.length;
    } else if (event.key === "ArrowUp") {
      nextIndex = (currentIndex - 1 + items.length) % items.length;
    } else if (event.key === "Home") {
      nextIndex = 0;
    } else if (event.key === "End") {
      nextIndex = items.length - 1;
    }

    event.preventDefault();
    items[nextIndex]?.focus();
  }

  if (collapsed) {
    return (
      <aside
        aria-label="Right inspector"
        className="inspector inspector--dashboardSystem"
        data-state="collapsed"
      >
        <div className="inspectorRail">
          <button
            aria-expanded="false"
            aria-label="Expand inspector"
            className="iconButton"
            onClick={toggleInspector}
            ref={toggleRef}
            title="Expand inspector"
            type="button"
          >
            <PanelRight aria-hidden="true" size={18} />
          </button>
          <strong>Inspector</strong>
          <span>{activeLabel}</span>
          <small>Live cockpit</small>
        </div>
      </aside>
    );
  }

  return (
    <aside
      aria-label="Right inspector"
      className="inspector inspector--dashboardSystem"
      data-state="open"
    >
      <DashboardSidebarHeader
        action={(
          <div className="inspectorHeaderActions">
            <button
              aria-label={`Open ${activeLabel} in Dashboard`}
              className="dashboardSecondaryAction inspectorDashboardButton"
              onClick={() => onOpenDashboard(activeTab)}
              title={`Open ${activeLabel} in Dashboard`}
              type="button"
            >
              <LayoutDashboard aria-hidden="true" size={15} />
              <span>Dashboard</span>
            </button>
            <button
              aria-expanded="true"
              aria-label="Collapse inspector"
              className="iconButton"
              onClick={toggleInspector}
              ref={toggleRef}
              title="Collapse inspector"
              type="button"
            >
              <PanelRight aria-hidden="true" size={18} />
            </button>
          </div>
        )}
        className="inspectorHeader"
        detail={detail}
        eyebrow="Inspector"
        icon={active?.icon ?? PanelRight}
        title={activeLabel}
        titleAs="h2"
      />

      <div className="inspectorTabsShell">
        <div className="inspectorTabs" role="tablist" aria-label="Inspector tabs">
          {tabs.map((tab, index) => {
            const Icon = tab.icon;
            const selected = tab.id === activeTab;

            return (
              <div className="inspectorTabItem" key={tab.id}>
                <button
                  aria-controls={tab.panelId}
                  aria-label={tab.label}
                  aria-selected={selected}
                  className="dashboardInspectorTab"
                  data-active={selected}
                  id={`${tab.panelId}-tab`}
                  onClick={() => onSelectTab(tab.id)}
                  onKeyDown={(event) => handleTabKeyDown(event, index)}
                  role="tab"
                  tabIndex={selected ? 0 : -1}
                  title={tab.label}
                  type="button"
                >
                  <Icon aria-hidden="true" size={15} />
                  <span>{tab.label}</span>
                  {tab.badge ? <small>{tab.badge}</small> : null}
                </button>
                {tab.closeable ? (
                  <button
                    aria-label={`Close ${tab.label} Inspector tab`}
                    className="inspectorTabClose"
                    onClick={() => handleCloseTab(tab)}
                    title={`Close ${tab.label}`}
                    type="button"
                  >
                    <X aria-hidden="true" size={12} />
                  </button>
                ) : null}
              </div>
            );
          })}
        </div>

        {availableTabs.length > 0 ? (
          <div className="inspectorTabAdd">
            <button
              aria-controls="available-inspector-tabs"
              aria-expanded={addMenuOpen}
              aria-haspopup="menu"
              aria-label="Add Inspector tab"
              className="inspectorAddButton"
              onClick={() => onAddMenuOpenChange(!addMenuOpen)}
              ref={addButtonRef}
              title="Add Inspector tab"
              type="button"
            >
              <Plus aria-hidden="true" size={15} />
            </button>
            {addMenuOpen ? (
              <div
                aria-label="Available Inspector tabs"
                className="inspectorAddMenu"
                id="available-inspector-tabs"
                onKeyDown={handleMenuKeyDown}
                ref={menuRef}
                role="menu"
              >
                {availableTabs.map((tab) => {
                  const Icon = tab.icon;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => handleAddTab(tab)}
                      role="menuitem"
                      type="button"
                    >
                      <Icon aria-hidden="true" size={15} />
                      <span>{tab.label}</span>
                    </button>
                  );
                })}
              </div>
            ) : null}
          </div>
        ) : null}
      </div>

      {children}
    </aside>
  );
}
