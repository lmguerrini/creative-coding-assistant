import { describe, expect, it } from "vitest";
import type { PreviewRuntimeLifecycleState } from "./preview-runtime-adapters";
import type { PreviewRuntimeSessionOverrideMode } from "./preview-controller";
import {
  canArmPreviewAutoRecovery,
  isPreviewRuntimeAwaitingFirstFrame,
  previewAutoRecoveryReadinessBudgetMs
} from "./preview-controller";

/**
 * Faithful, dependency-free model of the bounded automatic preview-recovery
 * wired into workstation-shell. It mirrors the real control flow exactly:
 *
 *   1. `canArmPreviewAutoRecovery` gates whether a readiness timer is scheduled;
 *   2. when that timer fires, `isPreviewRuntimeAwaitingFirstFrame` confirms the
 *      runtime never advanced past Starting/Warming before the manual Reload
 *      path is reused; and
 *   3. exactly one recovery is consumed per artifact/version, reset when the
 *      previewed artifact changes.
 *
 * Reload requests are recorded so the tests can assert *how many* automatic
 * reloads a given lifecycle produced.
 */
class PreviewAutoRecoveryHarness {
  reloadCount = 0;

  private consumedIdentity: string | null = null;
  private runtimeState: PreviewRuntimeLifecycleState | null = null;
  private overrideMode: PreviewRuntimeSessionOverrideMode | null = null;
  private artifactId: string;
  private version: number;

  constructor(options: { artifactId: string; version?: number } = { artifactId: "sketch" }) {
    this.artifactId = options.artifactId;
    this.version = options.version ?? 1;
  }

  private get recoveryIdentity() {
    return `${this.artifactId}:${this.version}`;
  }

  /** The runtime reports a new lifecycle state (starting → running / error …). */
  observeRuntimeState(state: PreviewRuntimeLifecycleState | null) {
    this.runtimeState = state;
  }

  /** An override is created/settled by a manual or automatic reload/restart. */
  setOverrideMode(mode: PreviewRuntimeSessionOverrideMode | null) {
    this.overrideMode = mode;
  }

  /** The operator selects a different artifact to preview. */
  changeArtifact(artifactId: string, version = 1) {
    this.artifactId = artifactId;
    this.version = version;
    // workstation-shell resets the allowance when the previewed artifact changes.
    this.consumedIdentity = null;
    this.overrideMode = null;
    this.runtimeState = null;
  }

  /**
   * Attempt to arm and fire the bounded recovery for the current open preview.
   * Returns true when a reload was actually issued.
   */
  tick({ isOpen = true, isPreviewable = true }: { isOpen?: boolean; isPreviewable?: boolean } = {}) {
    const armed = canArmPreviewAutoRecovery({
      consumedIdentity: this.consumedIdentity,
      isPreviewable,
      isOpen,
      recoveryIdentity: this.recoveryIdentity,
      sessionOverrideMode: this.overrideMode
    });

    if (!armed) {
      return false;
    }

    // The readiness timer has elapsed — only recover a runtime still awaiting
    // its first runnable frame.
    if (!isPreviewRuntimeAwaitingFirstFrame(this.runtimeState)) {
      return false;
    }

    this.consumedIdentity = this.recoveryIdentity;
    this.reloadCount += 1;
    // The reload path installs a "reloading" override until the runtime runs.
    this.overrideMode = "reloading";
    return true;
  }
}

describe("preview auto recovery decision", () => {
  it("keeps a short, bounded readiness budget rather than an arbitrary long delay", () => {
    expect(previewAutoRecoveryReadinessBudgetMs).toBeGreaterThan(0);
    expect(previewAutoRecoveryReadinessBudgetMs).toBeLessThanOrEqual(5000);
  });

  it("only treats never-runnable states as awaiting the first frame", () => {
    expect(isPreviewRuntimeAwaitingFirstFrame(null)).toBe(true);
    expect(isPreviewRuntimeAwaitingFirstFrame("idle")).toBe(true);
    expect(isPreviewRuntimeAwaitingFirstFrame("starting")).toBe(true);

    // Explicit, successful, or terminal outcomes must never be auto-reloaded.
    expect(isPreviewRuntimeAwaitingFirstFrame("running")).toBe(false);
    expect(isPreviewRuntimeAwaitingFirstFrame("ready")).toBe(false);
    expect(isPreviewRuntimeAwaitingFirstFrame("stopped")).toBe(false);
    expect(isPreviewRuntimeAwaitingFirstFrame("error")).toBe(false);
  });

  it("1. reloads a newly opened preview stuck at Starting/Warming exactly once", () => {
    const harness = new PreviewAutoRecoveryHarness({ artifactId: "sketch" });
    harness.observeRuntimeState("starting");

    expect(harness.tick()).toBe(true);
    expect(harness.reloadCount).toBe(1);

    // Still stuck after the automatic reload — it must not fire again.
    harness.setOverrideMode("settled");
    harness.observeRuntimeState("starting");
    expect(harness.tick()).toBe(false);
    expect(harness.reloadCount).toBe(1);
  });

  it("2. never reloads a preview that reaches Running", () => {
    const harness = new PreviewAutoRecoveryHarness({ artifactId: "sketch" });
    harness.observeRuntimeState("running");

    expect(harness.tick()).toBe(false);
    expect(harness.reloadCount).toBe(0);
  });

  it("3. cannot enter a reload loop on the same artifact/version", () => {
    const harness = new PreviewAutoRecoveryHarness({ artifactId: "sketch" });
    harness.observeRuntimeState("starting");

    // Ten readiness windows on a permanently stuck runtime yield one reload.
    for (let attempt = 0; attempt < 10; attempt += 1) {
      harness.tick();
      // The reload override never settles because the runtime never runs.
      harness.observeRuntimeState("starting");
    }

    expect(harness.reloadCount).toBe(1);
  });

  it("4. resets the one-recovery allowance when the previewed artifact changes", () => {
    const harness = new PreviewAutoRecoveryHarness({ artifactId: "sketch-a" });
    harness.observeRuntimeState("starting");
    expect(harness.tick()).toBe(true);
    expect(harness.reloadCount).toBe(1);

    // A brand-new previewed artifact is entitled to its own single recovery.
    harness.changeArtifact("sketch-b");
    harness.observeRuntimeState("starting");
    expect(harness.tick()).toBe(true);
    expect(harness.reloadCount).toBe(2);
  });

  it("5. does not hide or endlessly retry an explicit runtime failure", () => {
    const harness = new PreviewAutoRecoveryHarness({ artifactId: "broken" });
    harness.observeRuntimeState("error");

    for (let attempt = 0; attempt < 5; attempt += 1) {
      expect(harness.tick()).toBe(false);
    }

    expect(harness.reloadCount).toBe(0);
  });

  it("6. leaves the manual Reload path unblocked (an in-flight override just defers auto-recovery)", () => {
    const harness = new PreviewAutoRecoveryHarness({ artifactId: "sketch" });
    // Operator clicked Reload: a non-settled override exists and the runtime is
    // remounting. Automatic recovery must stand down while that plays out.
    harness.setOverrideMode("reloading");
    harness.observeRuntimeState("starting");
    expect(harness.tick()).toBe(false);
    expect(harness.reloadCount).toBe(0);

    // Once the manual reload succeeds the override settles on a running runtime,
    // and there is nothing left to auto-recover.
    harness.setOverrideMode("settled");
    harness.observeRuntimeState("running");
    expect(harness.tick()).toBe(false);
    expect(harness.reloadCount).toBe(0);
  });

  it("7. matches each runtime's healthy terminal state (three/glsl/p5 run, Tone.js arms)", () => {
    // Three.js, GLSL and p5.js resolve to "running"; Tone.js deliberately stops
    // at "ready" (armed, awaiting explicit playback). None are stalls.
    for (const successfulState of ["running", "running", "running", "ready"] as const) {
      const harness = new PreviewAutoRecoveryHarness({ artifactId: "route" });
      harness.observeRuntimeState(successfulState);
      expect(harness.tick()).toBe(false);
      expect(harness.reloadCount).toBe(0);
    }

    // A stalled mount on any of those routes still earns one recovery.
    const stalled = new PreviewAutoRecoveryHarness({ artifactId: "route" });
    stalled.observeRuntimeState("starting");
    expect(stalled.tick()).toBe(true);
    expect(stalled.reloadCount).toBe(1);
  });

  it("does not arm while the preview is closed or non-previewable", () => {
    const harness = new PreviewAutoRecoveryHarness({ artifactId: "sketch" });
    harness.observeRuntimeState("starting");

    expect(harness.tick({ isOpen: false })).toBe(false);
    expect(harness.tick({ isPreviewable: false })).toBe(false);
    expect(harness.reloadCount).toBe(0);
  });
});
