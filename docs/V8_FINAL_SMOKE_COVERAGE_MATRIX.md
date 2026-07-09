# V8 Final Smoke Coverage Matrix

Date: 2026-07-09
Branch: `version-review/v8`

This matrix records the final reviewer-journey evidence for Creative Coding
Assistant V8. It separates live provider validation, mocked browser smoke,
artifact/browser QA, and documented accepted boundaries.

## Evidence Legend

- **Live provider**: exercised the normal assistant service with configured
  provider credentials and local retrieval where available.
- **LangSmith active**: verified trace visibility in project
  `creative-coding-assistant`.
- **Mocked browser smoke**: exercised the real Next.js workstation shell in
  Playwright with API mocks, without provider cost.
- **Artifact QA**: validated committed demo artifacts through syntax checks,
  browser runtime checks, or WebGL nonblank pixel checks.
- **Accepted boundary**: a truthful limitation that is documented and not
  presented as broader runtime support.

## Live Provider And LangSmith Evidence

| Workflow | Evidence | Result | Boundary |
|---|---|---|---|
| Retrieval-backed assistant workflow | `CCA_LANGSMITH_SMOKE_20260709_015` | **PASS**: 31.42s, 269 stream events, 5 retrieved contexts, final output present, 0 workflow errors. | Streaming telemetry did not expose provider token usage for this run. |
| LangSmith trace visibility | Trace id `f5420da971d0475d8b87951ce34d2bd0` in project `creative-coding-assistant` | **ACTIVE TESTED**: 34 matching successful spans, including retrieval, planning, generation, preview preparation, review, and finalization. | Deterministic verification required LangChain compatibility env vars and explicit LangSmith client flush. |
| RAG/retrieval evidence | Redacted latest-live RAGAs plus sanitized public fixture | **PASS WITH BOUNDARY**: provider-backed evaluator fixtures completed with 0 skipped rows and 0 metric failures. | Raw private live-session rows remain local-only. |

## Demo Mode Scenarios

| Scenario | Evidence mode | Duration / tokens | Status | Boundary |
|---|---|---:|---|---|
| Three.js audio-reactive visual system | Live provider smoke plus artifact QA | 68.8s / 41,518 total | **FULL END-TO-END PASS** | Browser audio remains user-gesture controlled; QA is not display-FPS benchmarking. |
| p5.js generative morphogenesis sketch | Live provider smoke plus artifact QA | 33.9s / 39,645 total | **FULL END-TO-END PASS** | Digital Morphogenesis/Jason Webb material is inspiration only; no KB source claim. |
| GLSL shader/post-processing visual | Live provider smoke plus WebGL QA | 32.8s / 39,400 total | **FULL END-TO-END PASS** | WebGL nonblank validation, not load/soak/performance testing. |
| Hydra feedback-pattern demo | Bounded route plus `hydra-synth` artifact QA | 0.4s / no provider tokens | **PASS WITH ACCEPTED BOUNDARY** | Support is limited to the validated local `hydra-synth` browser artifact path. |
| Retrieval-grounded creative-coding answer | Live provider smoke, retrieval, RAGAs | 19.2s / 38,778 total | **FULL END-TO-END PASS** | Public evidence uses redacted/sanitized fixtures. |
| Concept-to-visual translation | Live provider smoke plus UI preload smoke | 26.3s / 38,919 total | **FULL END-TO-END PASS** | Advisory creative translation, not objective interpretation. |
| Geometry/morphogenesis visual system | Live provider smoke plus p5/GLSL artifact QA | 21.5s / 38,556 total | **FULL END-TO-END PASS** | Multi-domain routing is validated; live multi-agent execution is not claimed. |
| Installation/immersive scene planning | Live provider smoke plus demo docs | 52.2s / 37,699 total | **FULL END-TO-END PASS** | Planning and handoff only; no venue scan, deployment, certification, or DCC/MCP execution. |

## App Capability Coverage

| Capability | Validation path | Result | Boundary |
|---|---|---|---|
| Normal chat workflow | Playwright `workstation-smoke.spec.js` creative journey with API mocks | **PASS** | Mocked browser smoke proves UI routing, not provider quality. |
| Demo Mode preload | Playwright scenario preload checks for all 8 scenarios | **PASS** | Demo Mode curates prompts; normal assistant flow still runs the scenario. |
| p5 generation | Live p5 scenario smoke, non-demo p5 prompts, p5 artifact browser QA | **PASS** | Runtime QA is local browser evidence. |
| Three.js generation | Live Three.js scenario smoke and Three.js artifact browser QA | **PASS** | Audio-reactive behavior remains browser-permission bounded. |
| GLSL generation | Live GLSL scenario smoke and direct WebGL QA | **PASS** | Nonblank shader render only. |
| Hydra artifact path | Hydra Demo Mode boundary test and `hydra-synth` browser QA | **PASS WITH BOUNDARY** | No full Hydra editor or broad runtime claim. |
| Retrieval/RAG answer | Live LangSmith retrieval workflow, retrieval smoke, RAGAs fixtures | **PASS** | Raw private rows remain unscored externally. |
| Concept translation | Demo Mode live smoke and frontend metadata tests | **PASS** | Public language avoids unsupported authority claims. |
| Visual planning | Geometry/morphogenesis and prompt-library evidence | **PASS** | Planning/advisory surface only. |
| Installation planning | Live planning smoke and Capstone demo guide | **PASS** | Local demo target only; no public cloud deployment. |
| Code panel | Frontend unit tests and Playwright smoke | **PASS** | User Mode routes long code away from chat. |
| Preview panel | Frontend unit tests, Playwright smoke, visual screenshots | **PASS** | User Mode hides diagnostics; Developer Mode preserves them. |
| Saved/Artifacts panel | Frontend unit tests and Playwright smoke | **PASS** | Raw filenames stay Developer Mode-only where practical. |
| User Mode | Playwright mode tests and visual QA | **PASS** | Right inspector collapsed by default. |
| Developer Mode | Playwright mode tests | **PASS** | Full diagnostics available after mode switch. |
| Theme switching | Playwright mode/theme smoke | **PASS** | Themes are Aqua, Codex, and Matrix only. |
| Clear workspace | Playwright clear-session smoke | **PASS** | Resets composer, Demo Mode, active preview/artifact, and inspector state. |
| Provider fallback | Playwright provider-fallback smoke | **PASS** | Does not claim preview or trace success. |
| Retrieval empty state | Frontend tests and workstation UI checks | **PASS** | Distinguishes current-run empty retrieval from KB unavailable. |
| KB status surface | Frontend tests and workstation UI checks | **PASS** | UI exposes status/check guidance; no fake refresh claim. |
| Public claim boundary | README/docs/UI claim scan | **PASS** | Internal backend names may remain; public UI/docs avoid future/internal naming. |

## Extra Non-Demo Morphogenesis Smoke

| Prompt | Validation path | Result | Boundary |
|---|---|---|---|
| Space-colonization / branching growth prompt | Playwright mocked full app smoke outside Demo Mode | **PASS** | Uses Digital Morphogenesis ideas as general inspiration only. |
| DLA or differential-growth prompt | Playwright mocked full app smoke outside Demo Mode | **PASS** | No Jason Webb/morphogenesis-resources KB coverage is claimed. |

## Remaining Accepted Boundaries

- Live provider token usage is available for optimized Demo Mode smoke rows,
  but not for the final LangSmith traced stream because streaming telemetry did
  not expose provider usage metadata.
- Hydra is validated as a local `hydra-synth` artifact path, not as a full
  live-code editor or broad Hydra production runtime.
- Chroma/Pydantic warnings remain third-party dependency warnings with a
  documented upgrade-validation path.
- Multi-agent live execution is not claimed; current evidence covers
  single-domain, retrieval/hybrid, planning, and multi-domain workflows.
- Public cloud deployment, external DCC/MCP execution, merge, push, tag, and
  final freeze remain HITL-gated.
