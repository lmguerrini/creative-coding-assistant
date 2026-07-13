"use client";

import {
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode
} from "react";
import { createPortal } from "react-dom";
import { AlertTriangle, LoaderCircle, X } from "lucide-react";

export function ApplicationFloatingPanel({
  children,
  className = "",
  description,
  id,
  label,
  onRequestClose,
  title
}: {
  children: ReactNode;
  className?: string;
  description: string;
  id: string;
  label: string;
  onRequestClose: () => void;
  title: string;
}) {
  const panelRef = useRef<HTMLElement>(null);
  const headingId = `${id}-heading`;
  const descriptionId = `${id}-description`;

  useEffect(() => {
    const frameId = window.requestAnimationFrame(() => {
      panelRef.current?.focus({ preventScroll: true });
    });
    return () => window.cancelAnimationFrame(frameId);
  }, []);

  return (
    <section
      aria-describedby={descriptionId}
      aria-label={label}
      aria-labelledby={headingId}
      className={`applicationFloatingPanel ${className}`.trim()}
      id={id}
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          event.preventDefault();
          event.stopPropagation();
          onRequestClose();
        }
      }}
      ref={panelRef}
      role="dialog"
      tabIndex={-1}
    >
      <header className="applicationFloatingPanelHeader">
        <div>
          <strong id={headingId}>{title}</strong>
          <p id={descriptionId}>{description}</p>
        </div>
        <button
          aria-label={`Close ${label}`}
          className="applicationFloatingPanelClose"
          onClick={onRequestClose}
          type="button"
        >
          <X aria-hidden="true" size={15} />
        </button>
      </header>
      {children}
    </section>
  );
}

export type ApplicationConfirmationRequest = {
  cancelLabel: string;
  confirmLabel: string;
  detail: string;
  eyebrow?: string;
  id: string;
  onConfirm: () => void | Promise<void>;
  returnFocus?: HTMLElement | null;
  title: string;
  tone?: "danger" | "warning";
};

export function ApplicationConfirmDialog({
  onClose,
  request
}: {
  onClose: () => void;
  request: ApplicationConfirmationRequest;
}) {
  const reactId = useId();
  const cancelButtonRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const requestRef = useRef(request);
  const onCloseRef = useRef(onClose);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  requestRef.current = request;
  onCloseRef.current = onClose;

  const headingId = `${reactId}-heading`;
  const detailId = `${reactId}-detail`;

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    const returnFocus = requestRef.current.returnFocus ?? null;
    document.body.style.overflow = "hidden";
    const frameId = window.requestAnimationFrame(() => {
      cancelButtonRef.current?.focus({ preventScroll: true });
    });

    return () => {
      window.cancelAnimationFrame(frameId);
      document.body.style.overflow = previousOverflow;
      window.requestAnimationFrame(() => {
        const fallback = document.querySelector<HTMLElement>(
          '[aria-label="Settings"], [aria-label="Add attachment"], .workspaceComposer textarea'
        );
        (returnFocus?.isConnected ? returnFocus : fallback)?.focus({
          preventScroll: true
        });
      });
    };
  }, [request.id]);

  function cancel() {
    if (!isSubmitting) {
      onCloseRef.current();
    }
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Escape") {
      event.preventDefault();
      cancel();
      return;
    }
    if (event.key !== "Tab") {
      return;
    }

    const focusable = Array.from(
      dialogRef.current?.querySelectorAll<HTMLElement>(
        'button:not(:disabled), [href], input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex="-1"])'
      ) ?? []
    );
    if (focusable.length === 0) {
      event.preventDefault();
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  async function confirm() {
    if (isSubmitting) {
      return;
    }
    setError(null);
    setIsSubmitting(true);
    try {
      await Promise.resolve(requestRef.current.onConfirm());
      onCloseRef.current();
    } catch (confirmationError) {
      setError(
        confirmationError instanceof Error
          ? confirmationError.message
          : "The requested action could not be completed."
      );
      setIsSubmitting(false);
    }
  }

  if (typeof document === "undefined") {
    return null;
  }

  return createPortal(
    <div
      className="applicationModalBackdrop"
      data-tone={request.tone ?? "warning"}
      onPointerDown={(event) => {
        if (event.target === event.currentTarget) {
          cancel();
        }
      }}
    >
      <div
        aria-busy={isSubmitting}
        aria-describedby={detailId}
        aria-labelledby={headingId}
        aria-modal="true"
        className="applicationConfirmDialog"
        onKeyDown={handleKeyDown}
        ref={dialogRef}
        role="alertdialog"
      >
        <header className="applicationConfirmDialogHeader">
          <span className="applicationConfirmDialogIcon" aria-hidden="true">
            <AlertTriangle size={18} />
          </span>
          <div>
            <span>{request.eyebrow ?? "Confirm action"}</span>
            <h2 id={headingId}>{request.title}</h2>
          </div>
        </header>
        <p id={detailId}>{request.detail}</p>
        {error ? <p className="applicationConfirmDialogError" role="alert">{error}</p> : null}
        <footer className="applicationConfirmDialogActions">
          <button
            className="applicationConfirmDialogCancel"
            disabled={isSubmitting}
            onClick={cancel}
            ref={cancelButtonRef}
            type="button"
          >
            {request.cancelLabel}
          </button>
          <button
            className="applicationConfirmDialogSubmit"
            disabled={isSubmitting}
            onClick={() => void confirm()}
            type="button"
          >
            {isSubmitting ? <LoaderCircle aria-hidden="true" size={15} /> : null}
            {isSubmitting ? "Working…" : request.confirmLabel}
          </button>
        </footer>
      </div>
    </div>,
    document.body
  );
}
