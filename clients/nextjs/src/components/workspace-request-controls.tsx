"use client";

import {
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent
} from "react";
import {
  ChevronDown,
  ImagePlus,
  LoaderCircle,
  Plus,
  Server,
  ShieldCheck,
  VolumeX,
  X
} from "lucide-react";
import type { AssistantWorkspaceSnapshot } from "@/lib/assistant-client";
import type { WorkflowExecutionMode } from "@/lib/workflow-execution";
import {
  formatImageAttachmentSize,
  supportedImageUploadAccept
} from "@/lib/multimodal-attachments";
import {
  buildGenerationControls
} from "@/lib/product-controls";
import type { WorkspacePreferences } from "@/lib/workspace-persistence";
import { SubsystemErrorCallout } from "./subsystem-error-callout";

const workflowModeOptions: Array<{
  value: WorkflowExecutionMode;
  label: string;
  detail: string;
}> = [
  {
    value: "auto",
    label: "Auto",
    detail: "Choose the bounded route from the request."
  },
  {
    value: "single_agent",
    label: "Single Agent",
    detail: "Generate without separate researcher, critic, or reviewer stages."
  },
  {
    value: "multi_agent",
    label: "Multi Agent",
    detail: "Use the bounded planner, researcher, generator, critic, and reviewer route."
  }
];

type WorkspaceAttachmentControlProps = {
  disabled?: boolean;
  isOpen: boolean;
  isProcessing: boolean;
  onFilesSelected: (files: File[]) => void | Promise<void>;
  onOpenChange: (isOpen: boolean) => void;
};

export function WorkspaceAttachmentControl({
  disabled = false,
  isOpen,
  isProcessing,
  onFilesSelected,
  onOpenChange
}: WorkspaceAttachmentControlProps) {
  const menuId = useId();
  const processingStatusId = `${menuId}-processing`;
  const isUnavailable = disabled || isProcessing;
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const uploadActionRef = useRef<HTMLButtonElement>(null);
  const unavailableAudioRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isOpen) {
      return undefined;
    }

    const frameId = window.requestAnimationFrame(() => {
      uploadActionRef.current?.focus({ preventScroll: true });
    });

    const handlePointerDown = (event: PointerEvent) => {
      if (
        event.target instanceof Node &&
        rootRef.current &&
        !rootRef.current.contains(event.target)
      ) {
        onOpenChange(false);
      }
    };

    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key !== "Escape") {
        return;
      }
      event.preventDefault();
      onOpenChange(false);
      window.requestAnimationFrame(() => {
        triggerRef.current?.focus({ preventScroll: true });
      });
    };

    document.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.cancelAnimationFrame(frameId);
      document.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen, onOpenChange]);

  useEffect(() => {
    if (isUnavailable && isOpen) {
      onOpenChange(false);
    }
  }, [isOpen, isUnavailable, onOpenChange]);

  function openFilePicker() {
    inputRef.current?.click();
    onOpenChange(false);
    window.requestAnimationFrame(() => {
      triggerRef.current?.focus({ preventScroll: true });
    });
  }

  function handleMenuKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Tab") {
      onOpenChange(false);
      return;
    }
    if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      const items = [uploadActionRef.current, unavailableAudioRef.current].filter(
        (item): item is HTMLButtonElement | HTMLDivElement => item !== null
      );
      const currentIndex = items.findIndex((item) => item === document.activeElement);
      const nextIndex =
        event.key === "Home"
          ? 0
          : event.key === "End"
            ? items.length - 1
            : event.key === "ArrowDown"
              ? (Math.max(currentIndex, 0) + 1) % items.length
              : (currentIndex <= 0 ? items.length : currentIndex) - 1;
      items[nextIndex]?.focus({ preventScroll: true });
    }
  }

  return (
    <div className="workspaceAttachmentControl" ref={rootRef}>
      <button
        aria-controls={menuId}
        aria-describedby={isProcessing ? processingStatusId : undefined}
        aria-expanded={isOpen}
        aria-haspopup="menu"
        aria-label="Add attachment"
        className="workspaceAttachmentTrigger"
        data-processing={isProcessing ? "true" : undefined}
        disabled={isUnavailable}
        onClick={() => onOpenChange(!isOpen)}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault();
            onOpenChange(true);
          }
        }}
        ref={triggerRef}
        title={isProcessing ? "Preparing image reference" : "Add attachment"}
        type="button"
      >
        {isProcessing ? (
          <LoaderCircle aria-hidden="true" className="workspaceControlSpinner" size={17} />
        ) : (
          <Plus aria-hidden="true" size={18} />
        )}
      </button>
      {isProcessing ? (
        <span
          className="workspaceVisuallyHidden"
          id={processingStatusId}
          role="status"
        >
          Preparing image reference. Send is paused.
        </span>
      ) : null}
      <input
        accept={supportedImageUploadAccept}
        aria-label="Upload image attachment"
        className="workspaceVisuallyHidden"
        disabled={isUnavailable}
        multiple
        onChange={(event) => {
          const files = Array.from(event.currentTarget.files ?? []);
          event.currentTarget.value = "";
          onOpenChange(false);
          triggerRef.current?.focus({ preventScroll: true });
          if (files.length > 0) {
            void onFilesSelected(files);
          }
        }}
        ref={inputRef}
        tabIndex={-1}
        type="file"
      />
      {isOpen ? (
        <div
          aria-label="Attachment options"
          className="workspaceAttachmentMenu"
          id={menuId}
          onKeyDown={handleMenuKeyDown}
          role="menu"
        >
          <span className="workspaceRequestMenuEyebrow">Add visual context</span>
          <button
            className="workspaceAttachmentOption"
            onClick={openFilePicker}
            ref={uploadActionRef}
            role="menuitem"
            tabIndex={-1}
            type="button"
          >
            <ImagePlus aria-hidden="true" size={17} />
            <span>
              <strong>Upload image reference</strong>
              <small>PNG, JPEG, WebP, or GIF · next request only</small>
            </span>
          </button>
          <div
            aria-disabled="true"
            className="workspaceAttachmentOption workspaceAttachmentOption--disabled"
            ref={unavailableAudioRef}
            role="menuitem"
            tabIndex={-1}
          >
            <VolumeX aria-hidden="true" size={17} />
            <span>
              <strong>Audio input unavailable</strong>
              <small>Tone.js playback accepts compatible source artifacts only.</small>
            </span>
          </div>
          <p className="workspaceRequestMenuBoundary">
            <ShieldCheck aria-hidden="true" size={13} />
            Files remain queued until you send an explicit request.
          </p>
        </div>
      ) : null}
    </div>
  );
}

type WorkspaceGenerationControlsProps = {
  creativity: WorkspacePreferences["creativity"];
  disabled?: boolean;
  onCreativityChange: (profile: WorkspacePreferences["creativity"]) => void;
  onWorkflowChange: (mode: WorkflowExecutionMode) => void;
  workflowMode: WorkflowExecutionMode;
};

export function WorkspaceGenerationControls({
  creativity,
  disabled = false,
  onCreativityChange,
  onWorkflowChange,
  workflowMode
}: WorkspaceGenerationControlsProps) {
  const workflowHintId = useId();
  const workflowDetail =
    workflowModeOptions.find((option) => option.value === workflowMode)?.detail ??
    workflowModeOptions[0].detail;

  return (
    <div
      aria-label="Generation controls"
      className="workspaceGenerationControls"
      role="group"
    >
      <label
        className="workspaceGenerationField"
        data-kind="workflow"
        title="Choose the bounded route for this request."
      >
        <span>Workflow</span>
        <select
          aria-describedby={workflowHintId}
          aria-label="Workflow"
          disabled={disabled}
          onChange={(event) =>
            onWorkflowChange(event.currentTarget.value as WorkflowExecutionMode)
          }
          value={workflowMode}
        >
          {workflowModeOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <small className="workspaceVisuallyHidden" id={workflowHintId}>
          {workflowDetail}
        </small>
      </label>
      <ProviderRouteControl disabled={disabled} />
      <CreativityControl
        disabled={disabled}
        onChange={onCreativityChange}
        profile={creativity}
      />
    </div>
  );
}

export function ProviderRouteControl({
  disabled = false
}: {
  disabled?: boolean;
}) {
  const popoverId = useId();
  const [isOpen, setIsOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!isOpen) {
      return undefined;
    }

    const handlePointerDown = (event: PointerEvent) => {
      if (
        event.target instanceof Node &&
        rootRef.current &&
        !rootRef.current.contains(event.target)
      ) {
        setIsOpen(false);
      }
    };
    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key !== "Escape") {
        return;
      }
      event.preventDefault();
      setIsOpen(false);
      window.requestAnimationFrame(() => {
        triggerRef.current?.focus({ preventScroll: true });
      });
    };

    document.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen]);

  useEffect(() => {
    if (disabled) {
      setIsOpen(false);
    }
  }, [disabled]);

  return (
    <div className="workspaceGenerationField" data-kind="provider">
      <span>AI Providers</span>
      <div className="workspaceProviderControl" ref={rootRef}>
        <button
          aria-controls={popoverId}
          aria-expanded={isOpen}
          aria-label="Selected AI provider: OpenAI"
          className="workspaceProviderTrigger"
          disabled={disabled}
          onClick={() => setIsOpen((current) => !current)}
          ref={triggerRef}
          type="button"
        >
          <Server aria-hidden="true" size={13} />
          <strong>OpenAI</strong>
          <ChevronDown aria-hidden="true" size={12} />
        </button>
        {isOpen ? (
          <div
            aria-label="AI provider configuration"
            className="workspaceProviderPopover"
            id={popoverId}
            role="region"
          >
            <section
              aria-label="Selected AI provider"
              className="workspaceProviderSelected"
            >
              <span>Selected provider</span>
              <strong>OpenAI</strong>
              <small>Configured server-side</small>
            </section>
            <details className="workspaceProviderGroup">
              <summary>Local models</summary>
              <p>No local models are available in this workspace.</p>
            </details>
            <p className="workspaceRequestMenuBoundary">
              <ShieldCheck aria-hidden="true" size={13} />
              Credentials and live routing remain server-owned.
            </p>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function CreativityControl({
  disabled,
  onChange,
  profile
}: {
  disabled: boolean;
  onChange: (profile: WorkspacePreferences["creativity"]) => void;
  profile: WorkspacePreferences["creativity"];
}) {
  const hintId = useId();
  const controls = buildGenerationControls(profile);
  return (
    <label
      className="workspaceGenerationField"
      data-kind="creativity"
      title={controls.detail}
    >
      <span>Creativity</span>
      <select
        aria-describedby={hintId}
        aria-label="Creativity"
        disabled={disabled}
        onChange={(event) =>
          onChange(event.currentTarget.value as WorkspacePreferences["creativity"])
        }
        value={profile}
      >
        <option value="controlled">Controlled</option>
        <option value="balanced">Balanced</option>
        <option value="exploratory">Exploratory</option>
      </select>
      <small className="workspaceVisuallyHidden" id={hintId}>
        {controls.detail}
      </small>
    </label>
  );
}

export function WorkspaceImageReferences({
  multimodal,
  onDismissError,
  onRemove
}: {
  multimodal: AssistantWorkspaceSnapshot["multimodal"];
  onDismissError: () => void;
  onRemove: (attachmentId: string) => void;
}) {
  function restoreFocusAfterShelfChange(preferredIndex?: number) {
    window.requestAnimationFrame(() => {
      const remainingRemoveButtons = Array.from(
        document.querySelectorAll<HTMLButtonElement>(
          ".workspaceImageReferenceCard button"
        )
      );
      const nextRemoveButton =
        typeof preferredIndex === "number" && remainingRemoveButtons.length > 0
          ? remainingRemoveButtons[
              Math.min(preferredIndex, remainingRemoveButtons.length - 1)
            ]
          : undefined;
      (
        nextRemoveButton ??
        document.querySelector<HTMLButtonElement>(".workspaceAttachmentTrigger")
      )?.focus({ preventScroll: true });
    });
  }

  return (
    <section
      aria-label="Image references"
      className="workspaceImageReferences"
      data-state={multimodal.state}
    >
      <header className="workspaceImageReferencesHeader">
        <div>
          <span className="workspaceRequestMenuEyebrow">Visual context</span>
          <strong>{multimodal.error ? "Reference not added" : multimodal.status}</strong>
          <p>
            {multimodal.error
              ? "The queued request is unchanged. Review the issue and try again."
              : multimodal.detail}
          </p>
        </div>
        {multimodal.error ? (
          <button
            aria-label="Dismiss image upload issue"
            className="workspaceImageReferencesDismiss"
            onClick={() => {
              onDismissError();
              restoreFocusAfterShelfChange();
            }}
            type="button"
          >
            <X aria-hidden="true" size={14} />
          </button>
        ) : null}
      </header>
      {multimodal.error ? (
        <SubsystemErrorCallout
          className="workspaceImageUploadError"
          error={multimodal.error}
          title="Image upload issue"
        />
      ) : null}
      {multimodal.imageAttachments.length > 0 ? (
        <div className="workspaceImageReferencesList" role="list">
          {multimodal.imageAttachments.map((attachment, index) => (
            <article
              aria-label={`${attachment.name} image reference`}
              className="workspaceImageReferenceCard"
              key={attachment.id}
              role="listitem"
            >
              <div
                aria-hidden="true"
                className="workspaceImageReferenceThumb"
                style={{ backgroundImage: `url(${attachment.dataUrl})` }}
              />
              <div>
                <strong>{attachment.name}</strong>
                <span>
                  {attachment.mimeType.replace("image/", "").toUpperCase()} /{" "}
                  {formatImageAttachmentSize(attachment.sizeBytes)}
                </span>
              </div>
              <button
                aria-label={`Remove image reference ${attachment.name}`}
                onClick={() => {
                  onRemove(attachment.id);
                  restoreFocusAfterShelfChange(index);
                }}
                type="button"
              >
                <X aria-hidden="true" size={13} />
              </button>
            </article>
          ))}
        </div>
      ) : null}
      <p className="workspaceImageReferencesBoundary">
        Image pixels are included only in the next explicit request, then removed
        from the composer and not persisted. Provider receipt, use, and influence
        require separate live evidence.
      </p>
    </section>
  );
}
