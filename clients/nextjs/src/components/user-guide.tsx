import {
  Activity,
  AlertTriangle,
  BookOpen,
  Boxes,
  Braces,
  CheckCircle2,
  ChevronRight,
  Database,
  Download,
  Gauge,
  LayoutDashboard,
  MessageSquare,
  MonitorPlay,
  PanelRight,
  Play,
  Settings,
  Sparkles,
  TerminalSquare,
  type LucideIcon
} from "lucide-react";
import { demoModeScenarioCatalog } from "@/lib/demo-mode";
import {
  creativePreviewRendererRegistry,
  type CreativePreviewRendererKind
} from "@/lib/preview-renderers";

type GuideCard = {
  detail: string;
  icon: LucideIcon;
  title: string;
};

const guideChapters = [
  ["guide-start", "Start"],
  ["guide-workspace", "Workspace"],
  ["guide-workflows", "Workflows"],
  ["guide-results", "Results"],
  ["guide-knowledge", "Knowledge"],
  ["guide-dashboard", "Dashboard"],
  ["guide-settings", "Settings"],
  ["guide-help", "Help"]
] as const;

const quickStartSteps: GuideCard[] = [
  {
    detail: "Describe the visual, audio, or interactive system, plus the runtime and constraints that matter.",
    icon: MessageSquare,
    title: "Write the brief"
  },
  {
    detail: "Keep Auto for most work, or deliberately choose the direct Single-Agent or specialist Multi-Agent route.",
    icon: Sparkles,
    title: "Choose a route"
  },
  {
    detail: "Send the request and follow the streamed answer. A run can return guidance, code, or several saved outputs.",
    icon: Activity,
    title: "Run the workflow"
  },
  {
    detail: "Read Code and Saved first, then use Preview only when the selected artifact has a supported runtime.",
    icon: MonitorPlay,
    title: "Inspect the result"
  },
  {
    detail: "Refine a code artifact, rename it, copy or download it, or approve a project-bundle export when an Export action is present.",
    icon: Download,
    title: "Keep or hand off"
  }
];

const workspaceAreas: GuideCard[] = [
  {
    detail: "Create, open, rename, and delete browser-profile sessions. Each session retains its own conversation, artifacts, layout, and preferences.",
    icon: Boxes,
    title: "Sessions rail"
  },
  {
    detail: "The conversation shows streamed status and recovery messages. The composer holds the prompt, attachment, workflow, provider, creativity, and Send controls.",
    icon: MessageSquare,
    title: "Conversation + composer"
  },
  {
    detail: "User Mode keeps Preview, Code, Saved, Domains, and Settings close. Developer Mode adds workflow, retrieval, runtime, telemetry, and other evidence views.",
    icon: PanelRight,
    title: "Right inspector"
  },
  {
    detail: "A ready runnable artifact can open in the resizable shelf below the conversation, with fullscreen and lifecycle controls.",
    icon: MonitorPlay,
    title: "Preview shelf"
  },
  {
    detail: "Demo Mode starts curated flows. The Dashboard opens the complete run, knowledge, session, evaluation, and configuration reference.",
    icon: LayoutDashboard,
    title: "Top bar"
  }
];

const workflowModes = [
  {
    label: "Auto",
    use: "Best default",
    detail: "The application resolves the request to a bounded single- or multi-agent route from its scope and evidence needs."
  },
  {
    label: "Single-Agent",
    use: "Fast, contained requests",
    detail: "One direct generation path handles a focused deliverable with the smallest orchestration footprint."
  },
  {
    label: "Multi-Agent",
    use: "Research or critique-heavy work",
    detail: "Published specialist roles can add research, creative direction, generation, critique, and review. It is not an autonomous external-tool swarm."
  }
] as const;

const runtimeBoundaries: Record<CreativePreviewRendererKind, string> = {
  p5: "Global p5-compatible sketch source in an isolated frame.",
  three: "Controlled Three.js JavaScript; React Three Fiber and standalone HTML are export-only.",
  glsl: "Bounded WebGL fragment shaders; unsupported shader features fail visibly.",
  hydra: "Supported Hydra chains are parsed into a bounded plan inside the preview sandbox.",
  tone: "Audio starts muted and requires an explicit Start action; no microphone or audio upload.",
  gsap: "Motion is limited to the sandbox stage; plugins, remote assets, and unrestricted DOM access are rejected.",
  svg: "Inline SVG is sanitized; scripts, event handlers, and remote assets are rejected.",
  canvas: "Deterministic Canvas 2D drawing only; unrestricted DOM and input handlers are rejected."
};

const dashboardPages = [
  ["Overview", "Outcome, active artifact, and the current workspace signal board."],
  ["Architecture", "Requested and resolved route, agent responsibilities, research need, and refinement limit."],
  ["Workflow", "Live stages, transitions, retries, recovery state, context, and timelines."],
  ["Workspace", "The active creative document, generated source, and its current metadata."],
  ["Runtime", "Renderer lifecycle, diagnostics, errors, frames, reloads, and recovery evidence."],
  ["Preview", "Selected renderer route, support boundary, readiness, lifecycle, and visible-output evidence."],
  ["Artifacts", "Read-only retained artifact metadata, Preview status, session links, and bounded source excerpts."],
  ["Domains", "Supported browser runtimes and honest code/export or external-handoff boundaries."],
  ["Knowledge Base", "Registered technical sources, local index coverage, creative guidance, freshness, and current-run Retrieval."],
  ["AI & agents", "Provider route, model context, published responsibilities, observability, and explicit local output feedback."],
  ["Memory", "Published context counts and privacy-safe session-memory signals; never hidden provider reasoning."],
  ["Sessions", "Create, select, rename, and delete saved sessions, with artifact and reported-usage summaries."],
  ["Telemetry", "Published stream, provider, retrieval, Preview, Runtime, usage, and event evidence for the current workspace."],
  ["Evaluation", "A versioned 31-case console for separate RAG, Creative Artifact, Workflow, and Product Reliability evidence, with filters, history, trends, candidates, and Capstone mapping."],
  ["User Guide", "This canonical end-user reference and its implementation boundaries."],
  ["Settings", "Theme, typography, layout, display mode, preview behavior, workflow, provider, and creativity defaults."]
] as const;

const troubleshooting = [
  {
    title: "Send is disabled or a run will not start",
    checks: [
      "Enter a non-empty prompt and wait for any active stream to finish.",
      "For Reference-guided palette study, attach a supported image before loading the demo.",
      "The provider disclosure identifies the server-configured OpenAI route; it does not test availability or accept provider credentials.",
      "The Ready badge means the session is idle and can accept a prompt. It is not proof that OpenAI, Chroma, or upstream documentation is reachable."
    ]
  },
  {
    title: "The answer arrived without runnable code",
    checks: [
      "Ask explicitly for one runnable artifact and name the intended runtime and file type.",
      "Open Saved and Code: a valid guidance or handoff request may intentionally produce a document instead of a live visual.",
      "Use Domains to confirm whether the target is browser-previewable or code/export-only."
    ]
  },
  {
    title: "The live stream or provider was interrupted",
    checks: [
      "Treat a labeled fallback or local draft as recovery state, not as proof of a successful provider response.",
      "Edit or keep the prompt in the composer and press Send again after the provider or backend is ready.",
      "Retry and reset wording inside the generic error card is informational; use the composer, Preview controls, artifact actions, Settings clear action, or Knowledge Base Smart Update as appropriate.",
      "Provider keys and configuration are server-side and cannot be repaired from the browser interface."
    ]
  },
  {
    title: "Preview is unavailable, blank, or failed",
    checks: [
      "Select the intended artifact in Saved, then choose Preview. Preview and artifact selection are separate state.",
      "Check Preview for the support reason and Runtime for the reported renderer error.",
      "Try Reload for the same source, Restart for a fresh runtime session, or Clear before selecting a different artifact.",
      "If the domain is export-only, inspect Code or export the handoff instead of expecting an internal preview."
    ]
  },
  {
    title: "Retrieval or Knowledge Base evidence is missing",
    checks: [
      "Knowledge Base reports the registered and locally indexed inventory; Retrieval reports only evidence selected for the current request.",
      "A run can truthfully use no retrieved source. Do not treat an empty Retrieval view as proof that the inventory is empty.",
      "Use the Knowledge Base source-health and freshness controls to inspect the published local state."
    ]
  },
  {
    title: "A session, copy, download, or export did not persist",
    checks: [
      "Sessions are scoped to this browser profile; a damaged record is skipped rather than replacing the active workspace.",
      "Saved history is compact: up to 12 messages and artifacts, five Retrieval sources with three chunks each, 40 debug events, and 96,000 restored code characters. Reopen or rerun an oversized source when a recovery placeholder appears.",
      "Allow the browser clipboard or download action when prompted and retry the explicit action.",
      "Downloads and project-bundle exports require an operator checkpoint. Approve it to continue, or cancel without changing the artifact.",
      "Deleted artifacts offer Undo/Redo only while the current session remains open."
    ]
  },
  {
    title: "The interface feels crowded or important diagnostics are hidden",
    checks: [
      "Use Focus Mode, collapse the sessions rail or inspector, resize the inspector/preview, or switch to Compact density.",
      "Choose User Mode for the quiet creative path; choose Developer Mode when you need traces, Retrieval, Runtime, or Telemetry.",
      "Restore surfaces from the top-bar Settings quick actions or the Dashboard Settings page."
    ]
  }
] as const;

export function UserGuide() {
  return (
    <section aria-label="User Guide" className="productDashboardManual userGuide">
      <header className="userGuideHero">
        <div className="userGuideHeroIcon" aria-hidden="true">
          <BookOpen size={24} />
        </div>
        <div>
          <span>Canonical product documentation</span>
          <h2>From idea to inspected, previewed, and saved output</h2>
          <p>
            Follow the five-step path for a first run, then open the focused
            reference cards only when you need deeper product detail.
          </p>
        </div>
        <div className="userGuideHeroBadges" aria-label="Guide reading summary">
          <span>2-minute start</span>
          <span>Complete reference</span>
          <span>Current product only</span>
        </div>
      </header>

      <nav aria-label="User Guide contents" className="userGuideContents">
        {guideChapters.map(([id, label]) => (
          <a href={`#${id}`} key={id}>{label}</a>
        ))}
      </nav>

      <section aria-labelledby="guide-start-title" className="userGuideSection" id="guide-start">
        <GuideSectionHeader
          detail="The shortest reliable path through the product."
          icon={Play}
          title="Your first run"
          titleId="guide-start-title"
        />
        <ol aria-label="First run workflow" className="userGuideJourney">
          {quickStartSteps.map((step, index) => (
            <li key={step.title}>
              <div className="userGuideStepNumber">{index + 1}</div>
              <step.icon aria-hidden="true" size={18} />
              <strong>{step.title}</strong>
              <p>{step.detail}</p>
              {index < quickStartSteps.length - 1 ? (
                <ChevronRight aria-hidden="true" className="userGuideJourneyArrow" size={16} />
              ) : null}
            </li>
          ))}
        </ol>
        <aside className="userGuideCallout" data-tone="success">
          <CheckCircle2 aria-hidden="true" size={19} />
          <div>
            <strong>A result has three separate truths</strong>
            <p>
              The saved artifact, its visible Preview, and Runtime health are
              related but independent. Check all three before calling a visual ready.
            </p>
          </div>
        </aside>
      </section>

      <section aria-labelledby="guide-workspace-title" className="userGuideSection" id="guide-workspace">
        <GuideSectionHeader
          detail="Five areas carry the entire end-user workflow."
          icon={LayoutDashboard}
          title="Workspace map"
          titleId="guide-workspace-title"
        />
        <div className="userGuideCardGrid userGuideCardGrid--areas">
          {workspaceAreas.map((area) => (
            <article key={area.title}>
              <area.icon aria-hidden="true" size={19} />
              <strong>{area.title}</strong>
              <p>{area.detail}</p>
            </article>
          ))}
        </div>
        <details className="userGuideDisclosure">
          <summary>Composer controls and attachments</summary>
          <div className="userGuideDisclosureBody">
            <dl className="userGuideDefinitionGrid">
              <div><dt>Workflow</dt><dd>Auto, Single-Agent, or Multi-Agent for the next request.</dd></div>
              <div><dt>AI provider</dt><dd>Shows the active server-configured OpenAI route and availability; credentials are not edited here.</dd></div>
              <div><dt>Creativity</dt><dd>Controlled, Balanced, or Exploratory changes the requested generation profile; provider application is reported only when published.</dd></div>
              <div><dt>Image reference</dt><dd>Attach up to four PNG, JPEG, WebP, or GIF files of 1 MB each. References are retained with the session and bundle when available; this path exposes metadata, not pixel analysis.</dd></div>
              <div><dt>Audio input</dt><dd>Audio upload and audio analysis are not implemented. Compatible Tone.js artifacts can play only after an explicit start.</dd></div>
              <div><dt>Send</dt><dd>Runs the normal streamed assistant workflow; Demo Mode also enters through this same path.</dd></div>
            </dl>
          </div>
        </details>
        <details className="userGuideDisclosure">
          <summary>Inspector tabs, layout, and Focus Mode</summary>
          <div className="userGuideDisclosureBody">
            <ul className="userGuideBullets">
              <li><strong>Open a tab</strong> from the Inspector add menu or Settings quick actions. Close tabs you do not need; they can be re-added later.</li>
              <li><strong>Open in Dashboard</strong> carries the current Inspector category into the larger evidence view.</li>
              <li><strong>Resize</strong> the Inspector and a visual Preview shelf with their separators; the saved layout returns with the session.</li>
              <li><strong>Focus Mode</strong> hides surrounding workspace chrome and restores the previous Inspector and Preview state when you exit.</li>
              <li><strong>Preview auto-open</strong> can open the shelf when a ready supported artifact arrives, or remain Manual in Settings.</li>
            </ul>
          </div>
        </details>
      </section>

      <section aria-labelledby="guide-workflows-title" className="userGuideSection" id="guide-workflows">
        <GuideSectionHeader
          detail="Route choice changes orchestration, not the workspace safety boundary."
          icon={Sparkles}
          title="Workflows and Demo Mode"
          titleId="guide-workflows-title"
        />
        <div className="userGuideModeGrid">
          {workflowModes.map((mode) => (
            <article key={mode.label}>
              <span>{mode.use}</span>
              <strong>{mode.label}</strong>
              <p>{mode.detail}</p>
            </article>
          ))}
        </div>
        <aside className="userGuideCallout">
          <MessageSquare aria-hidden="true" size={19} />
          <div>
            <strong>Ambiguous briefs can pause for clarification</strong>
            <p>
              Choose a suggested option or type its number in the composer. The
              answer continues as a new, explicit run rather than silently guessing.
            </p>
          </div>
        </aside>
        <details className="userGuideDisclosure">
          <summary>How to use Demo Mode · {demoModeScenarioCatalog.length} curated flows</summary>
          <div className="userGuideDisclosureBody">
            <ol className="userGuideCompactSteps">
              <li><strong>Open Demo Mode</strong><span>Use the Play control in the top bar.</span></li>
              <li><strong>Select a scenario</strong><span>Read its runtime, input, expected artifact, preview, interaction, validation, and fallback before running.</span></li>
              <li><strong>Satisfy the input</strong><span>The reference-guided flow stays disabled until an image is attached.</span></li>
              <li><strong>Load prompt &amp; run</strong><span>The scenario sets its workflow mode and uses the normal composer and streaming path.</span></li>
              <li><strong>Verify the contract</strong><span>Compare the result with the scenario’s expected artifact, Preview, and fallback boundary.</span></li>
            </ol>
            <div aria-label="Demo Mode scenario catalog" className="userGuideScenarioGrid" role="list">
              {demoModeScenarioCatalog.map((scenario) => (
                <div key={scenario.id} role="listitem">
                  <Play aria-hidden="true" size={14} />
                  <strong>{scenario.title}</strong>
                  <span>{scenario.runtime}</span>
                </div>
              ))}
            </div>
            <p className="userGuideFinePrint">
              Developer Mode reveals featured demo paths and extra source,
              workflow, validation, and fallback evidence. Prepared fallbacks are
              evidence, not a new live provider, retrieval, or preview result.
            </p>
          </div>
        </details>
      </section>

      <section aria-labelledby="guide-results-title" className="userGuideSection" id="guide-results">
        <GuideSectionHeader
          detail="Keep source, visible output, and runtime evidence distinct."
          icon={Braces}
          title="Code, Saved outputs, Preview, and Runtime"
          titleId="guide-results-title"
        />
        <div className="userGuideCardGrid">
          <article>
            <Braces aria-hidden="true" size={19} />
            <strong>Code</strong>
            <p>Reads the selected artifact with language, type, status, line count, source, copy, and supported transfer actions. Provenance belongs to Workflow evidence.</p>
          </article>
          <article>
            <Boxes aria-hidden="true" size={19} />
            <strong>Saved / Artifacts</strong>
            <p>Select among outputs; inspect metadata, creative translation, critique, and plan; refine code; rename; delete; or use available actions.</p>
          </article>
          <article>
            <MonitorPlay aria-hidden="true" size={19} />
            <strong>Preview</strong>
            <p>Shows the selected previewable artifact. Fullscreen and Reload act on the view; checkpointed Restart and Clear act on the preview session, not the saved source.</p>
          </article>
          <article>
            <TerminalSquare aria-hidden="true" size={19} />
            <strong>Runtime</strong>
            <p>Reports published lifecycle, renderer, frame, reload, and error signals. It does not infer hidden runtime state.</p>
          </article>
          <article>
            <Download aria-hidden="true" size={19} />
            <strong>Copy, download, and export</strong>
            <p>Actions appear only when supported. File download and ZIP project-bundle export require an explicit operator checkpoint.</p>
          </article>
        </div>
        <details className="userGuideDisclosure">
          <summary>Supported live Preview runtimes and their boundaries</summary>
          <div className="userGuideDisclosureBody">
            <div className="userGuideTableWrap">
              <table>
                <thead><tr><th>Runtime</th><th>Preview surface</th><th>Important boundary</th></tr></thead>
                <tbody>
                  {creativePreviewRendererRegistry.map((runtime) => (
                    <tr key={runtime.id}>
                      <th scope="row">{runtime.displayName}</th>
                      <td>{runtime.description}</td>
                      <td>{runtimeBoundaries[runtime.kind]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="userGuideFinePrint">
              The canonical generation-domain catalog advertises internal
              preview delivery for p5.js, Three.js, GLSL, and Tone.js. Hydra,
              GSAP, SVG, and Canvas are additional bounded preview foundations
              when a compatible artifact is present; they are not unrestricted
              generation-domain guarantees. Runtime FPS is a short-window
              browser-frame health sample, not a display or production benchmark.
            </p>
            <aside className="userGuideCallout" data-tone="warning">
              <AlertTriangle aria-hidden="true" size={19} />
              <div>
                <strong>Preview support belongs to the artifact, not the idea</strong>
                <p>
                  Images, audio/video assets, JSON, and text may route to surface
                  foundations or placeholders with metadata; this does not mean
                  their bytes or playback are rendered. React applications,
                  standalone HTML, and external tools remain code/export or handoff
                  outputs unless a supported runtime artifact is also present.
                </p>
              </div>
            </aside>
          </div>
        </details>
        <details className="userGuideDisclosure">
          <summary>Artifact lifecycle and project-bundle export</summary>
          <div className="userGuideDisclosureBody">
            <ul className="userGuideBullets">
              <li><strong>Open</strong> selects the artifact in Code; <strong>Preview</strong> selects its supported renderer route.</li>
              <li><strong>Copy</strong> writes the current document text to the browser clipboard. <strong>Download</strong> saves that document.</li>
              <li><strong>Export</strong> appears only on an export artifact that supplies that action; there is no universal “Export workspace” button.</li>
              <li><strong>Project bundle</strong> ZIPs can include all artifact files, manifest and README, the saved conversation/preferences/layout snapshot, Workflow, Retrieval, Preview, and operator-checkpoint summaries, plus includable image files and external-tool handoffs.</li>
              <li><strong>Review before sharing</strong>: a bundle is a local browser download, not a deployment, and can contain prompts, session data, diagnostics, and original image bytes.</li>
              <li><strong>Refine</strong> sends an explicit instruction through a new workflow pass and creates a new version. Parameter controls and quick suggestions are guidance only: they do not mutate saved source or the running Preview until submitted. The targeted flow defaults to a two-pass limit.</li>
              <li><strong>Delete</strong> asks for confirmation and offers Undo/Redo while the session remains open. Rename changes the saved display/file title after validation.</li>
            </ul>
          </div>
        </details>
      </section>

      <section aria-labelledby="guide-knowledge-title" className="userGuideSection" id="guide-knowledge">
        <GuideSectionHeader
          detail="Three evidence layers answer different questions."
          icon={Database}
          title="Knowledge Base, Retrieval, and Memory"
          titleId="guide-knowledge-title"
        />
        <div className="userGuideKnowledgeFlow" aria-label="Knowledge evidence flow">
          <article>
            <span>1 · Inventory</span>
            <strong>Technical Knowledge</strong>
            <p>Registered official sources, locally indexed coverage, source health, and freshness controls.</p>
          </article>
          <ChevronRight aria-hidden="true" size={18} />
          <article>
            <span>2 · Direction</span>
            <strong>Creative Knowledge Base</strong>
            <p>Inspectable techniques, workflows, patterns, best practices, and artifact creative translation.</p>
          </article>
          <ChevronRight aria-hidden="true" size={18} />
          <article>
            <span>3 · Current run</span>
            <strong>Retrieval</strong>
            <p>The request, selected sources and chunks, quality, freshness, and boundaries published for this run only.</p>
          </article>
        </div>
        <aside className="userGuideCallout">
          <Database aria-hidden="true" size={19} />
          <div>
            <strong>Memory is not provider reasoning</strong>
            <p>
              Memory reports published, privacy-safe context and session-history
              signals. The product never presents private model reasoning as memory,
              retrieval evidence, or provenance.
            </p>
          </div>
        </aside>
        <details className="userGuideDisclosure">
          <summary>Maintain the local Technical Knowledge index</summary>
          <div className="userGuideDisclosureBody">
            <ol className="userGuideCompactSteps userGuideCompactSteps--four">
              <li><strong>Check</strong><span>Compare selected registered sources with local fingerprints and reachability.</span></li>
              <li><strong>Update</strong><span>Fetch changed reachable sources after the explicit write confirmation.</span></li>
              <li><strong>Rebuild</strong><span>Rebuild affected local indexes for the selected sources.</span></li>
              <li><strong>Validate</strong><span>Confirm index state and retain the last successful Smart Update summary.</span></li>
            </ol>
            <p className="userGuideFinePrint">
              Smart Update runs the sequence for you. Advanced controls expose
              Check for updates, Update selected, Rebuild selected, and Validate
              index separately. If a selected-source write fails, the prior valid
              local index is restored. Index validation does not itself prove
              Retrieval quality for a future prompt.
            </p>
          </div>
        </details>
      </section>

      <section aria-labelledby="guide-dashboard-title" className="userGuideSection" id="guide-dashboard">
        <GuideSectionHeader
          detail="Use the Dashboard when you need evidence or configuration, not to run the creative conversation."
          icon={LayoutDashboard}
          title="Every Dashboard page"
          titleId="guide-dashboard-title"
        />
        <div aria-label="Dashboard page reference" className="userGuideDashboardGrid" role="list">
          {dashboardPages.map(([title, detail], index) => (
            <article key={title} role="listitem">
              <span>{String(index + 1).padStart(2, "0")}</span>
              <div><strong>{title}</strong><p>{detail}</p></div>
            </article>
          ))}
        </div>
        <p className="userGuideFinePrint">
          Dashboard metrics and status cards read the shared workspace model. A
          missing value stays “Not published” or equivalent; the UI does not guess
          provider usage, cost, retrieval, execution, or runtime state.
        </p>
      </section>

      <section aria-labelledby="guide-settings-title" className="userGuideSection" id="guide-settings">
        <GuideSectionHeader
          detail="Preferences are local to the current browser workspace."
          icon={Settings}
          title="Settings, sessions, and privacy"
          titleId="guide-settings-title"
        />
        <div className="userGuideCardGrid">
          <article><Settings aria-hidden="true" size={19} /><strong>Appearance</strong><p>Choose Aqua, Deep Blue, Dark, Light, Matrix, Terminal, Horizon, Zen, or Blueprint themes and separate heading, body, label/control, and code scales.</p></article>
          <article><LayoutDashboard aria-hidden="true" size={19} /><strong>Layout</strong><p>Use Cozy or Compact density; collapse the sessions rail or inspector; open the Preview shelf; resize panels; or enter Focus Mode.</p></article>
          <article><Gauge aria-hidden="true" size={19} /><strong>Display mode</strong><p>User Mode presents the quiet creative path. Developer Mode exposes detailed workflow, runtime, retrieval, telemetry, and evaluation evidence.</p></article>
          <article><Sparkles aria-hidden="true" size={19} /><strong>Generation defaults</strong><p>Set workflow and creativity for the next prompt. Provider selection is shown for clarity and remains server-configured.</p></article>
          <article><Database aria-hidden="true" size={19} /><strong>Personalization</strong><p>Only explicit local helpful/needs-work signals are retained. Enable, remove, or clear them from top-bar Settings.</p></article>
          <article><Boxes aria-hidden="true" size={19} /><strong>Sessions</strong><p>Create, open, rename, or delete sessions from the rail or Dashboard. Token and cost totals include only provider-published values.</p></article>
        </div>
        <details className="userGuideDisclosure">
          <summary>Local storage, privacy, evaluation, and safety boundaries</summary>
          <div className="userGuideDisclosureBody">
            <ul className="userGuideBullets">
              <li>Sessions and preferences are isolated to this browser profile. “Local” does not mean a claim of absolute security.</li>
              <li>Image references are retained with the session. Do not attach private material to a public demonstration.</li>
              <li>Raw output-feedback comments and titles stay local. When personalization is enabled, up to three relevant derived preference categories and a signal count can shape a later request.</li>
              <li>Evaluation always supports deterministic local evidence. Provider-assisted RAGAS is optional, requires explicit authorization, and can use only the committed sanitized or redacted fixture—never raw local sessions.</li>
              <li>Provider calls depend on server configuration. The interface does not expose or edit provider credentials.</li>
              <li>Clear workspace uses an operator checkpoint and resets the conversation, workflow, Preview, and artifacts while keeping saved appearance and layout preferences.</li>
              <li>Deleting a session asks for confirmation and is not the same as session-scoped artifact Undo/Redo.</li>
              <li>Potentially destructive reset, transfer, export, or other guarded actions use an operator checkpoint where implemented.</li>
            </ul>
          </div>
        </details>
      </section>

      <section aria-labelledby="guide-help-title" className="userGuideSection" id="guide-help">
        <GuideSectionHeader
          detail="Start with the visible product signal before repeating a run."
          icon={AlertTriangle}
          title="Troubleshooting"
          titleId="guide-help-title"
        />
        <div className="userGuideTroubleshooting">
          {troubleshooting.map((topic) => (
            <details className="userGuideDisclosure" key={topic.title}>
              <summary>{topic.title}</summary>
              <div className="userGuideDisclosureBody">
                <ol>
                  {topic.checks.map((check) => <li key={check}>{check}</li>)}
                </ol>
              </div>
            </details>
          ))}
        </div>
        <aside className="userGuideCallout" data-tone="warning">
          <AlertTriangle aria-hidden="true" size={19} />
          <div>
            <strong>Know the product boundary</strong>
            <p>
              The application generates, inspects, previews, evaluates, and
              exports bounded creative-coding artifacts. It does not install or
              run Blender, Houdini, TouchDesigner, Unity, Unreal, remote DCC tools,
              venue scanning, deployment infrastructure, or autonomous external agents.
            </p>
          </div>
        </aside>
      </section>
    </section>
  );
}

function GuideSectionHeader({
  detail,
  icon: Icon,
  title,
  titleId
}: {
  detail: string;
  icon: LucideIcon;
  title: string;
  titleId: string;
}) {
  return (
    <header className="userGuideSectionHeader">
      <div aria-hidden="true"><Icon size={19} /></div>
      <div>
        <h2 id={titleId}>{title}</h2>
        <p>{detail}</p>
      </div>
    </header>
  );
}
